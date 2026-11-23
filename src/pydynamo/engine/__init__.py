"""Dynamo 1.1.514's alignment engine (dynamo_compute_iteration_spp) on the GPU, through Dynamo's own kernels.

Engine runs Dynamo's PTX kernels, cuFFT 8 and legacy cuBLAS 8 in Dynamo's order (pydynamo.engine.kernels.DynamoExact):
one particle at a time, as Dynamo streams them (spp.cu:2127-2146), and its orientations in batches of `batch` (BATCH
for "auto", fewer when the free GPU memory holds fewer; 1 = Dynamo's own loop) with the same bits, so that the refined
tables equal the official engine's bit for bit on the same GPU model. `sms` reproduces the cuBLAS 8 reduction order of
a GPU model with that many streaming multiprocessors (e.g. 64 for an RTX A5000; default $PYDYNAMO_SMS, else the GPU in
use).
Arrays are [z, y, x] (x fastest, EM order).
"""
import os
import time

try:                                   # CPU-only environments (CI) can still import this module
    import cupy as cp
except ImportError:
    cp = None

BATCH = 128                     # orientations per step when batch="auto" (1 = Dynamo's own loop), at most what the
                                # free GPU memory holds (kernels.batch_for_memory)


def preflight():
    """Check, before any project, set-up or engine work, that the engine can run: Dynamo 1.1.514's engine binary,
    the CUDA 8 cuFFT / cuBLAS libraries and the PTX extraction (RuntimeError naming the missing piece)."""
    from pydynamo.engine import kernels
    kernels.preflight()


class Engine:
    def __init__(self, template, mask, card, batch="auto", sms=None):
        self.card = card
        self.n = n = template.shape[0]
        assert template.shape == (n, n, n) and mask.shape == (n, n, n), "cubic template/mask required"
        assert card.dim == card.sidelength, "binning (dim < sidelength) is not implemented"
        if getattr(card, "dynamo_version", "1.1.514") != "1.1.514":
            raise NotImplementedError("the engine executes Dynamo 1.1.514's code only")
        self.timings = {}
        self.orientation_counts = []
        self.trace = None             # set to [] to record every orientation (harness comparisons)
        from pydynamo.engine.kernels import DynamoExact
        b = BATCH if batch == "auto" else int(batch)
        sms = sms if sms is not None else int(os.environ.get("PYDYNAMO_SMS", "0") or 0)
        self._ex = DynamoExact(template, mask, card, powf=True, batch=b, sdot_sms=sms or None)

    def close(self):
        """Release what CuPy's memory pool cannot reclaim: the cuFFT 8 plans (DynamoExact.close). Idempotent."""
        ex = getattr(self, "_ex", None)
        if ex is not None:
            ex.close()
            self._ex = None

    def prepare_particles(self, particles, rows):
        """The particles to align ([P, n, n, n] float32, host). Dynamo streams them one at a time (spp.cu:2127-2146)."""
        t0 = time.perf_counter()
        self.particles_host = particles
        self.residency = {"particles": len(particles), "resident_per_chunk": 1}
        self.timings["prepare_particles"] = self.timings.get("prepare_particles", 0.0) + time.perf_counter() - t0

    def set_template(self, template, mask=None):
        """Replace the reference (and optionally the mask) between iterations."""
        self._ex.set_template(template, mask)

    def refine(self, rows, progress=None):
        """All refinement levels of the prepared particles, particle by particle as dynamo_compute_iteration_spp
        (spp.cu:2127-2146). progress(done, out): called after every particle, when out[:done] holds the refined rows
        (as spp.cu:2160-2174 writes its counter and intermediate table)."""
        ex = self._ex
        ex.card = self.card
        out = rows.copy()
        counts = None
        t0 = time.perf_counter()
        with cp.cuda.Stream.null:           # cuFFT 8 / cuBLAS 8 run on the legacy default stream, so do the kernels
            for p in range(len(rows)):
                ex.record = [] if self.trace is not None else None
                _tl = time.perf_counter()
                ex.load_particle(self.particles_host[p])
                if ex.htime is not None:
                    ex.htime["load"] = ex.htime.get("load", 0.0) + time.perf_counter() - _tl
                out[p] = ex.refine_particle(rows[p])
                if progress:
                    progress(p + 1, out)
                if self.trace is not None:
                    self.trace += [(p, r) for r in ex.record]
                counts = list(ex.counts) if counts is None else [a + b for a, b in zip(counts, ex.counts)]
        ex.record = None
        self.orientation_counts = counts or []
        self.timings["align"] = self.timings.get("align", 0.0) + time.perf_counter() - t0
        if ex.htime is not None:
            self.timings["host"] = {k: round(v, 3) for k, v in ex.htime.items()}
        return out
