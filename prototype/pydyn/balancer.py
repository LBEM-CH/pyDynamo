"""GPU memory balancer: pick the largest batch that fits in free GPU memory, and back off instead of crashing.

Usage:
    bal = MemoryBalancer()
    B = bal.calibrate(run_batch, n_items)     # run_batch(b0, b1) processes items [b0, b1)
    bal.run_all(run_batch, n_items)           # processes everything with batch B, halving on out-of-memory

Calibration runs two small probe batches (their results are discarded, as run_all reprocesses all items).
Between probes the memory pool is emptied, so `pool.total_bytes()` after a probe is that probe's peak transient
footprint: arrays, cuFFT work areas and cached plans, on top of the resident data. The linear fit
peak(b) = fixed + per_item * b gives
    B = floor((available * safety - fixed) / per_item),   available = device free memory with the pool emptied.
"""
import math

try:
    import cupy as cp
except ImportError:
    cp = None


def _oom_types():
    if cp is None:
        return (MemoryError,)
    types = [cp.cuda.memory.OutOfMemoryError]
    try:
        from cupy.cuda import cufft
        types.append(cufft.CuFFTError)            # CUFFT_ALLOC_FAILED when a plan's work area does not fit
    except Exception:
        pass
    return tuple(types)


OOM = _oom_types()


class MemoryBalancer:
    def __init__(self, safety=0.90, probes=(8, 24), max_batch=None, plan_cache_size=8, log=None):
        self.safety = safety
        self.probes = probes
        self.max_batch = max_batch
        self.log = log if log is not None else []
        self.batch = None
        self.per_item = self.fixed = None
        cp.fft.config.get_plan_cache().set_size(plan_cache_size)

    @staticmethod
    def _release():
        cp.cuda.Device().synchronize()
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()

    def _peak(self, run, b):
        pool = cp.get_default_memory_pool()
        self._release()
        resident = pool.used_bytes()
        run(0, b)
        cp.cuda.Device().synchronize()
        return pool.total_bytes() - resident

    def calibrate(self, run, n_items):
        b1, b2 = (min(p, n_items) for p in self.probes)
        if n_items <= b1 or b1 == b2:
            self.batch = max(1, n_items)
            return self.batch
        p1, p2 = self._peak(run, b1), self._peak(run, b2)
        self.per_item = max(1.0, (p2 - p1) / (b2 - b1))
        self.fixed = max(0.0, p1 - self.per_item * b1)
        self._release()
        free, total = cp.cuda.runtime.memGetInfo()
        avail = free * self.safety - self.fixed
        b = int(avail // self.per_item)
        if self.max_batch:
            b = min(b, self.max_batch)
        self.batch = max(1, min(b, n_items))
        self.log.append({"event": "calibrated", "per_item_MB": round(self.per_item / 2**20, 2),
                         "fixed_MB": round(self.fixed / 2**20, 1), "free_MB": round(free / 2**20),
                         "total_MB": round(total / 2**20), "batch": self.batch})
        return self.batch

    def run_all(self, run, n_items):
        """Process items [0, n_items) in batches; on out-of-memory, release memory and halve the batch."""
        b = self.batch or n_items
        b0 = 0
        while b0 < n_items:
            b1 = min(n_items, b0 + b)
            try:
                run(b0, b1)
                b0 = b1
            except OOM as e:
                self._release()
                if b == 1:
                    raise
                b = max(1, b // 2)
                self.batch = b
                self.log.append({"event": "oom_backoff", "new_batch": b, "error": type(e).__name__})
        return self.batch

    def peak_used_fraction(self):
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        return (total - free - pool.free_bytes()) / total if total else math.nan
