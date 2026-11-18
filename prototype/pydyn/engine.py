"""Batched GPU re-implementation of Dynamo's alignment engine (dynamo_compute_iteration_spp, localnc=1 path).

The math follows docs/analysis/04_cuda_engine.md section 4 (rotate the template and mask with Dynamo's kernel,
apply the particle wedge and band-pass to the rotated template, Roseman local normalization, |cc+1| argmax peak,
spline sub-pixel, first-best orientation).

What differs from Dynamo is HOW it is computed:
* many (particle, orientation) pairs per batch instead of one at a time;
* real-to-complex FFTs instead of C2C;
* particle spectra computed once and kept on the GPU;
* one device-to-host transfer per batch instead of ~12 blocking copies per orientation.
Arrays are [z, y, x] (x fastest, EM order).

Engine(..., exact=True) selects the EXACT mode instead (pydyn/dynamo_exact.py): Dynamo 1.1.514's own kernels (its
PTX), cuFFT 8 and legacy cuBLAS 8 in Dynamo's order, one particle at a time and its orientations in batches of
`batch` (EXACT_BATCH for "auto"; 1 = Dynamo's own loop) with the same bits, so that the refined tables equal the
official engine's bit for bit on the same GPU model. The mode is chosen by `exact`, or, when `exact` is None, by the
environment variable PYDYN_ENGINE=exact|fast (default: DEFAULT_ENGINE).
"""
import os
import time

import numpy as np
try:                                   # CPU-only environments (CI) can still import the pure-NumPy helpers
    import cupy as cp
    import cupyx.scipy.fft as cpfft
except ImportError:
    cp = cpfft = None

from . import angles as A
from .balancer import MemoryBalancer

_ROTATE_SRC = r"""
extern "C" __global__
void rotate_batch(const float* t, const float* m, const float* roma, float* rt, float* rm, const int n, const long total)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    long N = (long)n * n * n;
    long b = gid / N;
    long v = gid - b * N;
    int z = (int)(v / ((long)n * n));
    int y = (int)((v - (long)z * n * n) / n);
    int x = (int)(v - (long)z * n * n - (long)y * n);
    const float* R = roma + 9 * b;
    int c = n / 2;
    float rx = float(x - c), ry = float(y - c), rz = float(z - c);
    float d = sqrtf(rx * rx + ry * ry + rz * rz);
    if (d >= n / 2 - 1) { rt[gid] = 0.f; rm[gid] = 0.f; return; }
    float orx = R[0] * rx + R[1] * ry + R[2] * rz;
    float ory = R[3] * rx + R[4] * ry + R[5] * rz;
    float orz = R[6] * rx + R[7] * ry + R[8] * rz;
    int fx = (int)floorf(orx), fy = (int)floorf(ory), fz = (int)floorf(orz);
    long px = fx + c, PX = px + 1;
    long py = (long)(fy + c) * n, PY = py + n;
    long pz = (long)(fz + c) * n * n, PZ = pz + (long)n * n;
    float cond = m[px + py + pz] * m[px + py + PZ] * m[px + PY + pz] * m[px + PY + PZ]
               * m[PX + py + pz] * m[PX + py + PZ] * m[PX + PY + pz] * m[PX + PY + PZ];
    if ((int)cond == 0) { rt[gid] = 0.f; rm[gid] = 0.f; return; }
    rm[gid] = 1.f;
    float xf = orx - fx, yf = ory - fy, zf = orz - fz;
    float vxyz = t[px + py + pz] * (1 - xf) * (1 - yf) * (1 - zf);
    float vxyZ = t[px + py + PZ] * (1 - xf) * (1 - yf) * zf;
    float vxYz = t[px + PY + pz] * (1 - xf) * yf * (1 - zf);
    float vxYZ = t[px + PY + PZ] * (1 - xf) * yf * zf;
    float vXyz = t[PX + py + pz] * xf * (1 - yf) * (1 - zf);
    float vXyZ = t[PX + py + PZ] * xf * (1 - yf) * zf;
    float vXYz = t[PX + PY + pz] * xf * yf * (1 - zf);
    float vXYZ = t[PX + PY + PZ] * xf * yf * zf;
    rt[gid] = vxyz + vXyz + vxYz + vxyZ + vXyZ + vxYZ + vXYz + vXYZ;
}
"""
_FUSED_SRC = r"""
#include <cupy/complex.cuh>
extern "C" __global__
void apply_hw(complex<float>* RT, const float* HW, const long long* widx, const long nh, const long total)
{   // RT[b] *= HW[widx[b]]  (wedge * band-pass on the half spectrum)
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long b = i / nh, k = i - b * nh;
    RT[i] *= HW[widx[b] * nh + k];
}
extern "C" __global__
void center_masked(float* t, const float* rm, const float* mu, const long N, const long total)
{   // t = (f - mu_b) * [rm == 1]   (normalize_float2cufft)
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long b = i / N;
    t[i] = (rm[i] == 1.f) ? (t[i] - mu[b]) : 0.f;
}
extern "C" __global__
void products(const complex<float>* Th, const complex<float>* Mh, const complex<float>* D, const complex<float>* D2,
              const long long* pid, complex<float>* out, const long nh, const long B)
{   // out[0]=T^ conj(D), out[1]=M^ conj(D), out[2]=M^ conj(D2); TM = [T^ (B), M^ (B)]
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = B * nh;
    if (i >= total) return;
    long b = i / nh, k = i - b * nh;
    long dofs = pid[b] * nh + k;
    complex<float> th = Th[i], mh = Mh[i];
    complex<float> dc = conj(D[dofs]), d2c = conj(D2[dofs]);
    out[i] = th * dc;
    out[total + i] = mh * dc;
    out[2 * total + i] = mh * d2c;
}
extern "C" __global__
void products_idx(const complex<float>* Th, const complex<float>* Mh, const complex<float>* D, const complex<float>* D2,
                  const long long* tidx, const long long* pid, complex<float>* out, const long nh, const long B)
{   // shared-grid variant: pair b uses template tidx[b] and particle pid[b]
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = B * nh;
    if (i >= total) return;
    long b = i / nh, k = i - b * nh;
    long tofs = tidx[b] * nh + k, dofs = pid[b] * nh + k;
    complex<float> th = Th[tofs], mh = Mh[tofs];
    complex<float> dc = conj(D[dofs]), d2c = conj(D2[dofs]);
    out[i] = th * dc;
    out[total + i] = mh * dc;
    out[2 * total + i] = mh * d2c;
}
extern "C" __global__
void roseman(float* res, const float* nnz, const float* sig, const long N, const long B)
{   // res[0] <- cc+1, res[1] <- |cc+1|   (assemble_roseman + copy_cufft2float_plus_one)
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = B * N;
    if (i >= total) return;
    long b = i / N;
    float rd = res[i], dm = res[total + i], d2m = res[2 * total + i];
    float v = d2m - (dm * dm) / nnz[b];
    float cc = (v > 0) ? rd / (sig[b] * sqrtf(v)) : -1.f;
    float f1 = cc + 1.f;
    res[i] = f1;
    res[total + i] = fabsf(f1);
}
"""
_kernels = {}
ENGINE_MODES = ("exact", "fast")
DEFAULT_ENGINE = "exact"        # the user requires Dynamo's results (benchmarks/09_engine_parity/exact: bit-identical)
EXACT_BATCH = 128             # orientations per step of the exact engine when batch="auto" (1 = Dynamo's own loop)


def engine_mode(mode=None):
    """'exact' or 'fast': the given mode, else $PYDYN_ENGINE, else DEFAULT_ENGINE."""
    m = (mode or os.environ.get("PYDYN_ENGINE") or DEFAULT_ENGINE).strip().lower()
    if m not in ENGINE_MODES:
        raise ValueError(f"engine mode {m!r}: expected one of {ENGINE_MODES}")
    return m


def preflight(mode=None):
    """Check, before any project/setup/engine work, that the chosen engine can run. The exact engine needs Dynamo
    1.1.514's engine binary, the CUDA 8 cuFFT/cuBLAS .so files and a working PTX extraction (RuntimeError naming the
    missing piece and suggesting --engine fast); the fast engine needs nothing from the installation."""
    m = engine_mode(mode)
    if m == "exact":
        from . import dynamo_exact
        dynamo_exact.preflight()
    return m


def _k(name):
    if name not in _kernels:
        src = _ROTATE_SRC if name == "rotate_batch" else _FUSED_SRC
        _kernels[name] = cp.RawKernel(src, name)
    return _kernels[name]


def _launch(name, total, args):
    _k(name)(((int(total) + 255) // 256,), (256,), args)


def folded_indices_half(n):
    """Folded |k| per axis for an rfftn half spectrum [z, y, x(0..n/2)], as Dynamo folds (x >= n/2 -> n - x)."""
    k = np.arange(n)
    fold = np.where(k >= n // 2, n - k, k).astype(np.float32)
    kx = np.arange(n // 2 + 1)
    foldx = np.where(kx >= n // 2, n - kx, kx).astype(np.float32)
    return fold[:, None, None], fold[None, :, None], foldx[None, None, :]


def bandpass_half(n, high, low, window):
    """aplyfourierfilter_fft as a real multiplier on the half spectrum."""
    kz, ky, kx = folded_indices_half(n)
    r = np.sqrt((kz * kz + ky * ky + kx * kx).astype(np.float32)).astype(np.float32)
    h, l, w = np.float32(high), np.float32(low), np.float32(window)
    H = np.ones(r.shape, dtype=np.float32)
    H[r < h - w] = 0
    sel = (r >= h - w) & (r < h)
    H[sel] = np.exp(-((h - r[sel]) / w) ** 2)
    sel = (r > l) & (r <= l + w)
    H[sel] = np.exp(-((r[sel] - l) / w) ** 2)
    H[r > l + w] = 0
    return H


def wedge_half(n, ftype, row):
    """Binary Fourier-sampling mask applied to the rotated template (spp.cu wedge kernels), on the half spectrum."""
    kz, ky, kx = folded_indices_half(n)
    kz, ky, kx = np.broadcast_arrays(kz, ky, kx)
    maxy, maxx, maxz = row[14], row[16], row[18]
    keep = np.ones(kz.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        def zap(num, den, maxang):
            ratio = np.abs(num / den)
            return ratio > np.tan(np.float32(maxang) * A.DEG2RAD_F)  # NaN (0/0) compares False -> kept
        if ftype == 1:
            keep &= ~zap(kz, kx, maxy)
        elif ftype == 2:
            keep &= ~zap(kz, ky, maxx)
        elif ftype == 4:
            keep &= ~(zap(kz, kx, maxy) & zap(kz, ky, maxx))
        elif ftype == 6:
            keep &= ~zap(ky, kx, maxz)
        elif ftype == 7:
            keep &= ~zap(ky, kz, maxz)
        elif ftype in (0, 3):
            pass
        else:
            raise NotImplementedError(f"ftype {ftype} not supported by the prototype")
    return keep.astype(np.float32)


def subpixel_spline(y1, y2, y3, powf=True):
    """Exact port of Dynamo's subpixel_spline (cuda/src/dynamo_compute_iteration_spp.cu, identical in 1.1.514/1.1.558).

    The C code mixes float and double. Sub-expressions made only of the float inputs and integer constants are
    evaluated in float (float32 here); they become double when combined with sqrt/pow results. IEEE semantics
    are kept: x/0 -> inf, sqrt(<0) and pow(<0, 1.5) -> NaN. inf/NaN candidate values are replaced by -10000. The
    running maximum is a float, and so is the returned position. NaN positions pass the range test, as in C
    (not(NaN < 0) && not(NaN > 1) is true). Verified against the compiled C function:
    benchmarks/03_bin8_align_1_rerun/checks/subpixel/.

    powf: how `pow(2*y2 - 3*y1 + y3, 2)` was compiled. Dynamo 1.1.514/1.1.532 (GCC 5.4, C++98) call
    std::pow(float, int), so the square is a float. Dynamo 1.1.557+ (GCC 11.4) promote it to double.
    Default True = 1.1.514, the version the user runs.
    """
    F, D = np.float32, np.float64
    y1, y2, y3 = F(y1), F(y2), F(y3)
    if y1 == y3:
        return 0.0
    with np.errstate(all="ignore"):
        z1 = D(-(y1 * y1) / F(4) + (y1 * y3) / F(2) + (y2 * y2) - F(2) * y2 * y3 + (F(3) * (y3 * y3)) / F(4))
        z2 = D((F(3) * (y1 * y1)) / F(4) - F(2) * y1 * y2 + (y1 * y3) / F(2) + (y2 * y2) - (y3 * y3) / F(4))
        s1, s2 = np.sqrt(z1), np.sqrt(z2)
        den1 = D(y1 + F(2) * y2 - F(3) * y3)                 # float expression
        den2 = D(F(2) * y2 - F(3) * y1 + y3)                 # float expression
        pv1 = -(D(2) * s1) / den1
        pv2 = (D(2) * s1) / den1
        pv3 = (D(F(2) * y2 - F(3) * y1 + y3) + D(2) * s2) / den2
        pv4 = -(D(F(3) * y1 - F(2) * y2 - y3) + D(2) * s2) / den2
        a = D(F(2) * y1 * y2 - F(3) * y1 * y3)               # float part of v1/v2 numerators
        yy = D(y1 * y1)
        v1 = ((((a + D(y1) * s1) - D(F(2) * y2) * s1) + D(y3) * s1) + yy) / den1
        v2 = ((((a - D(y1) * s1) + D(F(2) * y2) * s1) - D(y3) * s1) + yy) / den1
        p15 = z2 ** D(1.5)
        sq = D(F(F(2) * y2 - F(3) * y1 + y3) * F(F(2) * y2 - F(3) * y1 + y3)) if powf else den2 ** D(2)
        e = D(F(2) * y2 - F(2) * y3)
        v3 = ((D(y3) - s2 / D(2)) + (D(2) * p15) / sq) - (e * s2) / den2
        v4 = ((D(y3) + s2 / D(2)) - (D(2) * p15) / sq) + (e * s2) / den2
    my_min = F(-10000)
    vals = [D(my_min) if (np.isinf(v) or np.isnan(v)) else v for v in (v1, v2, v3, v4)]
    my_value, pos = my_min, F(0)
    for v, pv, off in ((vals[0], pv1, 1.0), (vals[1], pv2, 1.0), (vals[2], pv3, 0.0), (vals[3], pv4, 0.0)):
        if v > D(my_value) and not (pv < 0) and not (pv > 1):
            my_value = F(v)
            pos = F(pv - off)
    return float(pos)


class Engine:
    def __init__(self, template, mask, card, batch="auto", exact=None, exact_sms=None):
        self.card = card
        self.n = n = template.shape[0]
        assert template.shape == (n, n, n) and mask.shape == (n, n, n), "cubic template/mask required"
        assert card.dim == card.sidelength, "binning (dim < sidelength) is not implemented in the prototype"
        self.batch = batch            # int, or "auto" to let the memory balancer choose
        # exact compiled behaviour of the Dynamo build to reproduce (see subpixel_spline); the user runs 1.1.514
        self.dynamo_version = getattr(card, "dynamo_version", "1.1.514")
        self.powf = tuple(int(x) for x in self.dynamo_version.split(".")) < (1, 1, 557)
        self.balancer = None
        self.balance_log = []
        self.mode = engine_mode(None if exact is None else ("exact" if exact else "fast"))
        self.exact = self.mode == "exact"
        self.timings = {}
        self.profile = False
        self._events = []
        self.trace = None             # exact mode: set to [] to record every orientation (harness comparisons)
        if self.exact:
            if self.dynamo_version != "1.1.514":
                raise NotImplementedError("the exact engine executes Dynamo 1.1.514's code only")
            from .dynamo_exact import DynamoExact
            b = EXACT_BATCH if batch == "auto" else int(batch)
            # exact_sms: reproduce the cuBLAS 8 reduction order of a GPU with this many SMs (e.g. 64 = RTX A5000);
            # default ($PYDYN_EXACT_SMS unset) = the GPU in use
            sms = exact_sms if exact_sms is not None else int(os.environ.get("PYDYN_EXACT_SMS", "0") or 0)
            self._ex = DynamoExact(template, mask, card, powf=self.powf, batch=b, sdot_sms=sms or None)
            return
        s = np.float32(card.shrink)
        self.H = cp.asarray(bandpass_half(n, s * card.high, s * card.low, 3 * s))
        self.T = cp.asarray(np.ascontiguousarray(template, dtype=np.float32))
        self.M = cp.asarray(np.ascontiguousarray(mask, dtype=np.float32))

    def close(self):
        """Release what CuPy's memory pool cannot reclaim: the exact engine's cuFFT 8 plans (DynamoExact.close).
        Idempotent; the fast engine holds only CuPy arrays, freed with the object."""
        ex = getattr(self, "_ex", None)
        if ex is not None:
            ex.close()
            self._ex = None

    def _mark(self, name):
        if not self.profile:
            return
        ev = cp.cuda.Event()
        ev.record()
        self._events.append((name, ev))

    def profile_report(self):
        """Seconds per stage (GPU time between consecutive marks), summed over batches."""
        cp.cuda.Device().synchronize()
        out, prev = {}, None
        for name, ev in self._events:
            if name == "start":
                prev = ev
                continue
            out[name] = out.get(name, 0.0) + cp.cuda.get_elapsed_time(prev, ev) / 1000.0
            prev = ev
        return out

    def _tic(self, key, t0):
        cp.cuda.Device().synchronize()
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - t0

    # bytes of working memory one (particle, orientation) pair needs during alignment, per voxel of the box
    # (measured by the balancer: 15.4 MB per pair at 64^3 -> about 59 B/voxel; rounded up)
    PAIR_BYTES_PER_VOXEL = 64

    def prepare_particles(self, particles, rows, target_pairs=128, max_resident=None, safety=0.90):
        """Set up the particles for alignment.

        Keeps D^ = FFT(d) and D2^ = FFT(d^2) of the band-passed particles on the GPU when they fit; otherwise
        works in chunks (Dynamo itself streams one particle at a time from disk).
        * Working memory is reserved for `target_pairs` (particle, orientation) pairs.
        * The rest of the free GPU memory (x safety) holds resident spectra.
        * If not all particles fit, refine() processes them in chunks of that size and recomputes each chunk's
          spectra on the fly. That costs 4 FFTs per particle per refinement level, small next to the orientations.
        `max_resident` forces a smaller chunk (for testing).
        """
        t0 = time.perf_counter()
        n, P = self.n, len(particles)
        self.particles_host = particles            # [P, n, n, n] float32 (host)
        if self.exact:                             # Dynamo streams one particle at a time (spp.cu:2127-2146)
            self.chunk = P
            self.residency = {"particles": P, "engine": "exact", "resident_per_chunk": 1}
            self._tic("prepare_particles", t0)
            return
        # wedge * band-pass multiplier per distinct wedge description
        keys, self.widx_all = {}, np.zeros(P, dtype=np.int64)
        for p, row in enumerate(rows):
            ftype = int(row[12]) if self.card.use_CC > 0 else 0
            key = (ftype, float(row[14]), float(row[16]), float(row[18]))
            if key not in keys:
                keys[key] = len(keys)
            self.widx_all[p] = keys[key]
        hw = []
        for (ftype, my, mx, mz) in keys:
            r = np.zeros(34, dtype=np.float32)
            r[14], r[16], r[18] = my, mx, mz
            hw.append(cp.asarray(wedge_half(n, ftype, r)) * self.H)
        self.HW = cp.ascontiguousarray(cp.stack(hw))
        # residency plan
        spec_bytes = 2 * n * n * (n // 2 + 1) * 8
        cp.get_default_memory_pool().free_all_blocks()
        free, total = cp.cuda.runtime.memGetInfo()
        reserve = target_pairs * self.PAIR_BYTES_PER_VOXEL * n ** 3
        fit = int(max(1, (free * safety - reserve) // spec_bytes))
        self.chunk = min(P, fit if max_resident is None else min(fit, int(max_resident)))
        self.residency = {"particles": P, "resident_per_chunk": self.chunk, "chunks": -(-P // self.chunk),
                          "spectra_MB_per_particle": round(spec_bytes / 2**20, 2), "free_MB": round(free / 2**20),
                          "reserved_for_batches_MB": round(reserve / 2**20)}
        if self.chunk >= P:
            self._load_spectra(np.arange(P))
        self._tic("prepare_particles", t0)

    def _load_spectra(self, idx):
        """Compute and keep on the GPU the band-passed particle spectra of particles idx (host indices)."""
        n = self.n
        self.D = self.D2 = None
        cp.get_default_memory_pool().free_all_blocks()
        self.D = cp.empty((len(idx), n, n, n // 2 + 1), dtype=cp.complex64)
        self.D2 = cp.empty_like(self.D)
        step = 16 if n > 96 else 64
        for i in range(0, len(idx), step):
            sel = idx[i:i + step]
            d = cp.asarray(np.asarray(self.particles_host[sel], dtype=np.float32))
            D = cp.fft.rfftn(d, axes=(1, 2, 3)) * self.H
            d = cp.fft.irfftn(D, s=(n, n, n), axes=(1, 2, 3)).astype(cp.float32)
            self.D[i:i + len(sel)] = cp.fft.rfftn(d, axes=(1, 2, 3))
            self.D2[i:i + len(sel)] = cp.fft.rfftn(d * d, axes=(1, 2, 3))
            del d, D
        self.widx = self.widx_all[idx]
        self.loaded = idx

    def set_template(self, template, mask=None):
        """Replace the reference (and optionally the mask) between iterations; particle spectra stay resident."""
        if self.exact:
            self._ex.set_template(template, mask)
            return
        self.T = cp.asarray(np.ascontiguousarray(template, dtype=np.float32))
        if mask is not None:
            self.M = cp.asarray(np.ascontiguousarray(mask, dtype=np.float32))

    def _space_coords(self):
        n = self.n
        r = cp.arange(n)
        return (n // 2 - r) % n  # rawpeak2space for ns == nl

    def align_pairs(self, pidx, angles, lim_origin=None, lim_matrix=None):
        """Score every (particle, orientation) pair. Returns host arrays: cc, space[3], neighbours[6] per pair."""
        n, N = self.n, self.n ** 3
        nh = n * n * (n // 2 + 1)
        npairs = len(pidx)
        out_cc = np.empty(npairs, dtype=np.float32)
        out_sp = np.empty((npairs, 3), dtype=np.int32)
        out_nb = np.empty((npairs, 6), dtype=np.float32)
        mats_all = A.rotation_matrices(angles)
        sp = self._space_coords()
        modus = self.card.area_search_modus
        def run(b0, b1):
            t0 = time.perf_counter()
            B = b1 - b0
            pid_np = np.asarray(pidx[b0:b1], dtype=np.int64)
            pid = cp.asarray(pid_np)
            widx = cp.asarray(self.widx[pid_np].astype(np.int64))
            roma = cp.asarray(mats_all[b0:b1].ravel())
            rt = cp.empty((B, n, n, n), dtype=cp.float32)
            rm = cp.empty_like(rt)
            total = B * N
            self._mark("start")
            _launch("rotate_batch", total, (self.T, self.M, roma, rt, rm, np.int32(n), np.int64(total)))
            self._mark("rotate")
            RT = cpfft.rfftn(rt, axes=(1, 2, 3), overwrite_x=True)
            self._mark("fft_rt")
            _launch("apply_hw", B * nh, (RT, self.HW, widx, np.int64(nh), np.int64(B * nh)))
            self._mark("wedge")
            del rt
            t = cpfft.irfftn(RT, s=(n, n, n), axes=(1, 2, 3), overwrite_x=True)
            del RT
            self._mark("ifft_rt")
            t2d, m2d = t.reshape(B, N), rm.reshape(B, N)
            nnz = m2d.sum(axis=1)
            mu = cp.matmul(t2d[:, None, :], m2d[:, :, None]).reshape(B) / nnz
            _launch("center_masked", total, (t, rm, mu.astype(cp.float32), np.int64(N), np.int64(total)))
            sig = cp.sqrt(cp.matmul(t2d[:, None, :], t2d[:, :, None]).reshape(B))
            self._mark("normalize")
            Th = cpfft.rfftn(t, axes=(1, 2, 3), overwrite_x=True)
            Mh = cpfft.rfftn(rm, axes=(1, 2, 3), overwrite_x=True)
            del t, rm, t2d, m2d
            self._mark("fft_tm")
            prods = cp.empty((3, B, n, n, n // 2 + 1), dtype=cp.complex64)
            _launch("products", B * nh, (Th, Mh, self.D, self.D2, pid, prods, np.int64(nh), np.int64(B)))
            del Th, Mh
            self._mark("products")
            res = cpfft.irfftn(prods, s=(n, n, n), axes=(2, 3, 4), overwrite_x=True)
            del prods
            self._mark("ifft_3")
            _launch("roseman", total, (res, nnz.astype(cp.float32), sig.astype(cp.float32), np.int64(N), np.int64(B)))
            self._mark("roseman")
            f1, af = res[0].reshape(B, N), res[1].reshape(B, N)
            if modus != 0 and lim_origin is not None:
                o = cp.asarray(lim_origin[pid_np], dtype=cp.float32)
                Ml = cp.asarray(lim_matrix[pid_np], dtype=cp.float32)
                a = cp.asarray(np.array(self.card.area_search, dtype=np.float32) * np.float32(self.card.shrink))
                dx = sp[None, None, None, :].astype(cp.float32) - o[:, 0, None, None, None]
                dy = sp[None, None, :, None].astype(cp.float32) - o[:, 1, None, None, None]
                dz = sp[None, :, None, None].astype(cp.float32) - o[:, 2, None, None, None]
                q0 = Ml[:, 0, None, None, None] * dx + Ml[:, 1, None, None, None] * dy + Ml[:, 2, None, None, None] * dz
                q1 = Ml[:, 3, None, None, None] * dx + Ml[:, 4, None, None, None] * dy + Ml[:, 5, None, None, None] * dz
                q2 = Ml[:, 6, None, None, None] * dx + Ml[:, 7, None, None, None] * dy + Ml[:, 8, None, None, None] * dz
                outside = (cp.sqrt((q0 / a[0]) ** 2 + (q1 / a[1]) ** 2 + (q2 / a[2]) ** 2) > 1).reshape(B, N)
                f1[outside] = 0
                af[outside] = 0
            self._mark("limits")
            p = cp.argmax(af, axis=1)                                     # cublasIsamax: first max of |cc+1|
            self._mark("argmax")
            rz, ry, rx = p // (n * n), (p // n) % n, p % n
            sx, sy, sz = sp[rx], sp[ry], sp[rz]
            ar = cp.arange(B)
            ccv = f1[ar, p] - 1

            def val(ix, iy, iz):
                return f1[ar, iz * n * n + iy * n + ix] - 1

            def raw(s):
                return (n - s + n // 2) % n                                # spacepeak2raw (wraps periodically)

            nb = cp.stack([val(raw(sx - 1), ry, rz), val(raw(sx + 1), ry, rz),
                           val(rx, raw(sy - 1), rz), val(rx, raw(sy + 1), rz),
                           val(rx, ry, raw(sz - 1)), val(rx, ry, raw(sz + 1))], axis=1)
            self._mark("gather")
            out_cc[b0:b1] = ccv.get()
            out_sp[b0:b1] = cp.stack([sx, sy, sz], axis=1).get()
            out_nb[b0:b1] = nb.get()
            self._mark("to_host")
            del res, f1, af
            self.timings["align"] = self.timings.get("align", 0.0) + time.perf_counter() - t0

        if self.batch == "auto":
            if self.balancer is None:
                self.balancer = MemoryBalancer(log=self.balance_log)
                self.balancer.calibrate(run, npairs)
            self.balancer.run_all(run, npairs)
        else:
            for b0 in range(0, npairs, int(self.batch)):
                run(b0, min(npairs, b0 + int(self.batch)))
        return out_cc, out_sp, out_nb

    def template_terms(self, mats, w):
        """Rotated, wedge-w-filtered, masked template terms for a tile of orientations (shared by all particles)."""
        n, N = self.n, self.n ** 3
        nh = n * n * (n // 2 + 1)
        O = len(mats)
        rt = cp.empty((O, n, n, n), dtype=cp.float32)
        rm = cp.empty_like(rt)
        _launch("rotate_batch", O * N, (self.T, self.M, cp.asarray(mats.ravel()), rt, rm, np.int32(n), np.int64(O * N)))
        RT = cpfft.rfftn(rt, axes=(1, 2, 3), overwrite_x=True)
        del rt
        widx = cp.full(O, w, dtype=cp.int64)
        _launch("apply_hw", O * nh, (RT, self.HW, widx, np.int64(nh), np.int64(O * nh)))
        t = cpfft.irfftn(RT, s=(n, n, n), axes=(1, 2, 3), overwrite_x=True)
        del RT
        t2d, m2d = t.reshape(O, N), rm.reshape(O, N)
        nnz = m2d.sum(axis=1)
        mu = cp.matmul(t2d[:, None, :], m2d[:, :, None]).reshape(O) / nnz
        _launch("center_masked", O * N, (t, rm, mu.astype(cp.float32), np.int64(N), np.int64(O * N)))
        sig = cp.sqrt(cp.matmul(t2d[:, None, :], t2d[:, :, None]).reshape(O))
        Th = cpfft.rfftn(t, axes=(1, 2, 3), overwrite_x=True)
        Mh = cpfft.rfftn(rm, axes=(1, 2, 3), overwrite_x=True)
        return Th, Mh, nnz.astype(cp.float32), sig.astype(cp.float32)

    def align_shared(self, angles, otile=64):
        """Score every prepared particle against one common orientation set (reference-major batching).

        Keeps, per particle on the GPU, the first best orientation (Dynamo's strict '<' rule) and its peak data.
        Returns host arrays: best orientation index, cc, space[3], neighbours[6] per particle (-1 index if none).
        """
        n, N = self.n, self.n ** 3
        nh = n * n * (n // 2 + 1)
        P, O = self.D.shape[0], len(angles)
        mats = A.rotation_matrices(angles)
        sp = self._space_coords()
        best_cc = cp.full(P, -1.0, dtype=cp.float32)
        best_o = cp.full(P, -1, dtype=cp.int64)
        best_sp = cp.zeros((P, 3), dtype=cp.int64)
        best_nb = cp.zeros((P, 6), dtype=cp.float32)
        groups = [np.nonzero(self.widx == w)[0] for w in range(self.HW.shape[0])]
        ptile = max(1, self.batch // otile)
        for o0 in range(0, O, otile):
            t0 = time.perf_counter()
            o1 = min(O, o0 + otile)
            Ot = o1 - o0
            for w, members in enumerate(groups):
                if len(members) == 0:
                    continue
                Th, Mh, nnz, sig = self.template_terms(mats[o0:o1], w)
                for q0 in range(0, len(members), ptile):
                    parts = members[q0:q0 + ptile]
                    Pt = len(parts)
                    B = Ot * Pt
                    tidx = cp.repeat(cp.arange(Ot, dtype=cp.int64), Pt)          # pair order: orientation-major
                    pid = cp.tile(cp.asarray(parts, dtype=cp.int64), Ot)
                    prods = cp.empty((3, B, n, n, n // 2 + 1), dtype=cp.complex64)
                    _launch("products_idx", B * nh, (Th, Mh, self.D, self.D2, tidx, pid, prods, np.int64(nh), np.int64(B)))
                    res = cpfft.irfftn(prods, s=(n, n, n), axes=(2, 3, 4), overwrite_x=True)
                    del prods
                    _launch("roseman", B * N, (res, cp.ascontiguousarray(nnz[tidx]), cp.ascontiguousarray(sig[tidx]),
                                               np.int64(N), np.int64(B)))
                    f1, af = res[0].reshape(B, N), res[1].reshape(B, N)
                    p = cp.argmax(af, axis=1)
                    ar = cp.arange(B)
                    ccv = (f1[ar, p] - 1).reshape(Ot, Pt)
                    k = cp.argmax(ccv, axis=0)                                   # first best orientation in tile
                    kb = k * Pt + cp.arange(Pt)
                    cbest = ccv[k, cp.arange(Pt)]
                    pt = cp.asarray(parts)
                    better = cbest > best_cc[pt]                                 # strict: earlier tile wins ties
                    if bool(better.any()):
                        sel = cp.nonzero(better)[0]
                        pb = kb[sel]
                        pp = p[pb]
                        rz, ry, rx = pp // (n * n), (pp // n) % n, pp % n
                        sx, sy, sz = sp[rx], sp[ry], sp[rz]

                        def val(ix, iy, iz):
                            return f1[pb, iz * n * n + iy * n + ix] - 1

                        def raw(s):
                            return (n - s + n // 2) % n

                        nb = cp.stack([val(raw(sx - 1), ry, rz), val(raw(sx + 1), ry, rz),
                                       val(rx, raw(sy - 1), rz), val(rx, raw(sy + 1), rz),
                                       val(rx, ry, raw(sz - 1)), val(rx, ry, raw(sz + 1))], axis=1)
                        tgt = pt[sel]
                        best_cc[tgt] = cbest[sel]
                        best_o[tgt] = o0 + k[sel]
                        best_sp[tgt] = cp.stack([sx, sy, sz], axis=1)
                        best_nb[tgt] = nb
                    del res, f1, af
                del Th, Mh
            cp.cuda.Device().synchronize()
            self.timings["align"] = self.timings.get("align", 0.0) + time.perf_counter() - t0
        return best_o.get(), best_cc.get(), best_sp.get(), best_nb.get()

    def refine(self, rows, progress=None):
        """Full refinement (all levels) of the prepared particles; chunked when not all spectra fit on the GPU.
        progress(done, out): called when particles [0, done) are finished and out[:done] holds their refined rows:
        after every particle in the exact engine (as dynamo_compute_iteration_spp.cu:2160-2174 writes its counter
        and intermediate table), after every spectra chunk / at the end in the fast engine (level-wise batches)."""
        P = len(rows)
        if self.exact:
            return self._refine_exact(rows, progress)
        if self.chunk >= P:
            out = self._refine(rows)
            if progress:
                progress(P, out)
            return out
        out = np.empty_like(rows)
        counts = []
        for c0 in range(0, P, self.chunk):
            idx = np.arange(c0, min(P, c0 + self.chunk))
            t0 = time.perf_counter()
            self._load_spectra(idx)
            self._tic("stream_spectra", t0)
            out[idx] = self._refine(rows[idx])
            counts.append(self.orientation_counts)
            if progress:
                progress(int(idx[-1]) + 1, out)
        self.orientation_counts = [sum(c[k] for c in counts) for k in range(len(counts[0]))]
        self.D = self.D2 = None
        return out

    def _refine_exact(self, rows, progress=None):
        """Exact mode: particle by particle, as dynamo_compute_iteration_spp (spp.cu:2127-2146)."""
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
            self.timings["exact_host"] = {k: round(v, 3) for k, v in ex.htime.items()}
        return out

    def _refine(self, rows):
        """Refinement of the particles whose spectra are loaded (rows correspond to self.loaded)."""
        card = self.card
        P, n = len(rows), self.n
        s = np.float32(card.shrink)
        seeds = rows[:, 6:9].astype(np.float32).copy()
        shifts = (rows[:, 3:6] * s).astype(np.float32)
        best_cc = np.full(P, -1.0, dtype=np.float32)
        lim_matrix = A.rotation_matrices(rows[:, 6:9])            # M_lim: original table angles, all levels
        rng = A.ScanRange(card.cone_range, card.cone_sampling, card.inplane_range, card.inplane_sampling,
                          card.cone_flip, card.inplane_flip)
        self.orientation_counts = []
        for level in range(card.refine + 1):
            if level == 0 and getattr(self, "shared_grid", False):
                # one common orientation set for every particle (seed 0,0,0): reference-major batching
                common = A.orientation_set((0.0, 0.0, 0.0), rng)
                self.orientation_counts.append(int(len(common) * P))
                bo, bc, bs, bn = self.align_shared(common)
                new_seeds, new_shifts = seeds.copy(), shifts.copy()
                for p in range(P):
                    if bo[p] < 0:
                        continue
                    pos = [float(bs[p, a]) + subpixel_spline(bn[p, 2 * a], bc[p], bn[p, 2 * a + 1], self.powf) for a in range(3)]
                    new_seeds[p] = common[bo[p]]
                    new_shifts[p] = np.array(pos, dtype=np.float32) - np.float32(n / 2)
                    best_cc[p] = bc[p]
                seeds, shifts = new_seeds, new_shifts
                rng.shrink(card.refine_factor)
                rng.cone_flip = int(card.cone_flip > 1)
                rng.inplane_flip = int(card.inplane_flip > 1)
                continue
            t0 = time.perf_counter()
            sets = [A.orientation_set(tuple(seeds[p]), rng) for p in range(P)]
            self.timings["angles"] = self.timings.get("angles", 0.0) + time.perf_counter() - t0
            counts = np.array([len(x) for x in sets])
            self.orientation_counts.append(int(counts.sum()))
            pidx = np.repeat(np.arange(P), counts)
            allang = np.concatenate(sets, axis=0)
            if card.area_search_modus in (1, 11, 5, 51):
                origin = np.full((P, 3), n // 2, dtype=np.float32)
            else:
                origin = np.trunc(n / 2 + shifts).astype(np.float32)
            cc, spc, nb = self.align_pairs(pidx, allang, origin, lim_matrix)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            new_seeds, new_shifts = seeds.copy(), shifts.copy()
            for p in range(P):
                c = cc[starts[p]:starts[p] + counts[p]]
                if counts[p] == 0:
                    continue
                k = int(np.argmax(c))
                if not (c[k] > np.float32(-1)):                   # Dynamo: best starts at -1, strict '<'
                    continue
                j = starts[p] + k
                pos = [float(spc[j, a]) + subpixel_spline(nb[j, 2 * a], c[k], nb[j, 2 * a + 1], self.powf) for a in range(3)]
                new_seeds[p] = allang[j]
                new_shifts[p] = np.array(pos, dtype=np.float32) - np.float32(n / 2)
                best_cc[p] = c[k]
            seeds, shifts = new_seeds, new_shifts
            rng.shrink(card.refine_factor)
            rng.cone_flip = int(card.cone_flip > 1)
            rng.inplane_flip = int(card.inplane_flip > 1)
        out = rows.copy()
        out[:, 3:6] = shifts / s
        out[:, 6:9] = seeds
        out[:, 9] = best_cc
        return out
