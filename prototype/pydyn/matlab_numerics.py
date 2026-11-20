"""MATLAB numerics as Dynamo 1.1.514's executables compute them, bit for bit.

MATLAB's thread count matters for sums. Dynamo's kernel executables (dynamo_iteration_setup / _compute / _assemble /
_check) are compiled with `-R -singleCompThread` (matlab/src/dynamo_compile.m:377-387 for targets with graphics = 0,
matlab/src/+dpkdev/+compile/convention.m read_group3), but a standalone_gpu project does not run them: its .exe runs
dynamo_execute_project (graphics = 1, no such flag), which calls set-up, assembly (the averaging) and the adaptive
bandpass (its ref2 -> ref1 realignment, dynamo_align) as functions in its own process, with MATLAB's default thread
count, one per available core (measured: a run in a 16-core job is reproduced by maxNumCompThreads(16) and by no
other count; a 64-thread average of 9,965 particles of 64^3 reproduced bit for bit in double). With
matlab_workers_average > 0, dynamo_average runs its particle and weight loops as parfor loops on a pool of
single-threaded workers (parfor_subranges).
Measured in the Dynamo 1.1.514 MATLAB runtime (R2019a):
* sum(x) of a double vector with T threads: n < 89,000 or T = 1 -> strictly sequential, left to right from 0;
  otherwise T chunks of floor(n/T) summed sequentially, the chunk sums added in order, then the n mod T last
  elements added one by one (msum(x, threads=T));
* mean(x) = sum(x) / n; std(x) = sqrt(sum(abs(x - mean(x)).^2) / (n - 1)) with those sums;
* fftn / ifftn are MATLAB's own FFTW calls, reproduced in pydyn.masks._FFTW (read from MATLAB with an FFTW call
  logger): 1-D transforms one dimension at a time, 4 lines per plan execution plus single left-over lines, ifft
  divided by n after every dimension; a real fftn's redundant half is the conjugate mirror of the computed one;
  ifftn of an exactly conjugate-symmetric array takes a complex-to-real last step along z. Measured bit-identical
  for boxes 16 to 320 incl. odd, non-power-of-two and non-cubic sizes (MATLAB plans FFTW single-threaded at any
  thread count); the system FFTW 3.3.8 gives the same bits as MATLAB's own FFTW 3.3.3, so no MATLAB library is
  needed.
* MATLAB's sum rule (n >= 89,000 chunked by threads) is verified for n up to 33,000,000 and T = 1, 2, 3, 4, 7, 16, 64.
* parfor on a pool of W workers splits 1:N into fixed subranges (parfor_subranges; measured for W = 8, 16, 50, 64 and
  49 values of N from 1 to 30,000, all identical to the rule); a worker runs a subrange's iterations in descending
  order, and a sum reduction adds each subrange's partial sum to the client's variable. The client usually adds
  the partial sums in subrange order; a subrange that finishes out of turn can be merged with its neighbour first,
  so the last bits of a pool's sum change from run to run (MATLAB itself does not repeat them).
NumPy's sum / mean / std use pairwise orders and NumPy's FFT is pocketfft; both differ in the last bits, which is
enough to change float32 voxels of Dynamo's averages and the noise-level shells of an FSC. MATLAB's linear index
order of an [x, y, z] array is the C order of pyDynamo's [z, y, x] arrays, so flat order = MATLAB order.
"""
import numpy as np


THREADED_SUM_MIN = 89000


def _mround(x):
    """MATLAB round: halves away from zero."""
    import math
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def parfor_subranges(N, W):
    """Sizes of the subranges into which MATLAB R2019a's parfor splits 1:N on a pool of W workers, in loop order:
    a first group of W subranges of c0 = ceil(N / (20 W)); then groups of W subranges of max(c0, round(R / (2 W)))
    for the R iterations left; when a group would leave fewer than W iterations, the R left are split into W
    subranges of floor(R / W) and floor(R / W) + 1 (the smaller first; empty ones dropped). Measured
    (benchmarks/19_parfor, worker id and time of every iteration): 196 of 196 partitions identical."""
    import math
    N, W = int(N), int(W)
    if N <= 0:
        return []
    c0 = max(1, math.ceil(N / (20 * W)))
    sizes, R, s = [], N, c0
    while R > 0:
        if R - W * s < W:
            q, r = divmod(R, W)
            sizes += [z for z in [q] * (W - r) + [q + 1] * r if z > 0]
            break
        sizes += [s] * W
        R -= W * s
        s = max(c0, _mround(R / (2 * W)))
    return sizes


def matlab_threads():
    """MATLAB's default computational thread count here, as dynamo_execute_project would get it: one per physical
    core the process may run on (os.sched_getaffinity; logical CPUs that are SMT siblings of one core count once).
    $PYDYN_MATLAB_THREADS overrides it (e.g. 64 to reproduce a Dynamo run that had a whole 64-core node)."""
    import os
    env = os.environ.get("PYDYN_MATLAB_THREADS", "").strip()
    if env:
        t = int(env)
        if t < 1:
            raise ValueError("PYDYN_MATLAB_THREADS must be >= 1, got %r" % env)
        return t
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)
    cores = set()
    for c in cpus:
        base = "/sys/devices/system/cpu/cpu%d/topology/" % c
        try:
            with open(base + "physical_package_id") as f1, open(base + "core_id") as f2:
                cores.add((f1.read().strip(), f2.read().strip()))
        except OSError:
            cores.add(("cpu", str(c)))
    return max(1, len(cores))


def msum(x, threads=1):
    """MATLAB sum of a vector with `threads` computational threads (see the module doc)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        return 0.0
    T = int(threads)
    if T <= 1 or n < THREADED_SUM_MIN:
        return float(np.cumsum(x)[-1])
    q = n // T
    if q:
        parts = np.cumsum(x[:T * q].reshape(T, q), axis=1)[:, -1]
        r = parts[0]
        for c in range(1, T):
            r = r + parts[c]
    else:
        r = 0.0
    for k in range(T * q, n):
        r = r + x[k]
    return float(r)


def mmean(x, threads=1):
    x = np.asarray(x, dtype=np.float64).ravel()
    return msum(x, threads) / x.size


def mstd(x, threads=1):
    """MATLAB std(x) (normalised by n - 1): sqrt(sum(abs(x - mean(x)).^2) / (n - 1)); 0 for a scalar."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0
    d = x - mmean(x, threads)
    return float(np.sqrt(msum(np.abs(d) ** 2, threads) / (n - 1)))


def normalize_roi(v, mask, threads=1):
    """dynamo_normalize_roi_one (dynamo_normalize_roi.m): iroi = v(setdiff(find(mask), find(isnan(v)))) in index
    order; miroi = mean(iroi); stdiroi = std(iroi - miroi); (v - mean(iroi)) ./ stdiroi. In double, with MATLAB's
    sums for `threads` computational threads."""
    v = np.asarray(v, dtype=np.float64)
    sel = (np.asarray(mask).ravel() != 0) & ~np.isnan(v.ravel())
    iroi = v.ravel()[sel]
    if iroi.size == 0:
        raise ValueError("normalize_roi: no valid points found in this particle")
    mu = msum(iroi, threads) / iroi.size
    sd = mstd(iroi - mu, threads)
    return (v - mu) / sd


# ------------------------------------------------------------------------------------------------------------ FFT
_NOTED = []


def _fftw():
    from . import masks
    masks.fft_backend()
    b = masks._BACKEND["b"]
    if b is None and not _NOTED:                       # once per process: say that the FFTs are not MATLAB's
        import sys
        _NOTED.append(1)
        print("pyDynamo: no FFTW 3 library found (PYDYN_FFTW_LIB or the system libfftw3.so.3): "
              "numpy FFTs are used, which are not bit-identical to MATLAB's", file=sys.stderr)
    return b


def fftn(v):
    """MATLAB fftn of a real 3-D array (FFTW when available, else numpy, which is not bit-identical)."""
    b = _fftw()
    v = np.asarray(v, dtype=np.float64)
    return b.fftn_real(v) if b is not None else np.fft.fftn(v)


def ifftn_real(V):
    """real(ifftn(V)) of a 3-D complex array as MATLAB computes it (pydyn.masks._FFTW.ifftn_real): an exactly
    conjugate-symmetric V takes MATLAB's real path (ifft along x / nx, along y / ny, complex-to-real along z / nz);
    any other V MATLAB's complex ifftn (ifft along x, y, z, each part / n), real part."""
    b = _fftw()
    if b is None:
        return np.real(np.fft.ifftn(V))
    return b.ifftn_real(V)


def ifftn(V):
    """MATLAB ifftn of a 3-D complex array that is not conjugate-symmetric (complex result; FFTW when available,
    else numpy)."""
    b = _fftw()
    if b is None:
        return np.fft.ifftn(V)
    return b.ifftn(V)


def exact_fft():
    """True when fftn / ifftn_real use FFTW (bit-identical to MATLAB)."""
    return _fftw() is not None


def fweight_divide(volume, fweight, threshold):
    """dynamo_fweight_divide (dynamo_fweight_divide.m): NaN -> mean (dynamo_nan 'mean'); weights below the threshold
    or NaN -> 1 and their components -> 0; V_shifted = fftshift(fftn(volume)) ./ fweight;
    real(ifftn(ifftshift(V)))."""
    v = np.array(volume, dtype=np.float64)
    nan = np.isnan(v)
    if nan.any():
        v[nan] = mmean(v[~nan])
    F = np.array(fweight, dtype=np.float64)
    false = (F < threshold) | np.isnan(F)
    F[false] = 1
    V = np.fft.fftshift(fftn(v))
    V[false] = 0
    V = cdiv_real(V, F)
    V[false] = 0
    return ifftn_real(np.fft.ifftshift(V))


def cdiv_real(V, F):
    """MATLAB complex ./ real: each part divided by the real divisor (NumPy promotes the divisor to complex and
    multiplies by its reciprocal, which differs in the last bit)."""
    out = np.empty(np.broadcast(V, F).shape, dtype=np.complex128)
    out.real = V.real / F
    out.imag = V.imag / F
    return out


def conj(V):
    """MATLAB conj of a complex array: (re, 0 - im); a zero imaginary part becomes +0 (measured: with NumPy's
    -im, A .* conj(B) differs from MATLAB in the sign of 3 zeros of a 64^3 product)."""
    out = np.empty(np.shape(V), dtype=np.complex128)
    out.real = np.real(V)
    out.imag = 0.0 - np.imag(V)
    return out


def cmul(A, B):
    """MATLAB A .* B of complex arrays: (ar*br - ai*bi, ar*bi + ai*br), every product rounded on its own.
    NumPy's complex multiply may use fused multiply-adds (measured: 57,690 real and 80,329 imaginary parts of a
    64^3 product differ from MATLAB)."""
    ar, ai, br, bi = np.real(A), np.imag(A), np.real(B), np.imag(B)
    out = np.empty(np.broadcast(A, B).shape, dtype=np.complex128)
    out.real = ar * br - ai * bi
    out.imag = ar * bi + ai * br
    return out


def bandpass_filter(vol, mask_shifted):
    """real(ifftn(fftn(vol) .* mask_shifted)) (dynamo_bandpass with a precomputed, already fftshifted shell; MATLAB
    complex .* real multiplies each part, without NumPy's promotion of the mask to complex)."""
    V = fftn(vol)
    m = np.asarray(mask_shifted, dtype=np.float64)
    out = np.empty(V.shape, dtype=np.complex128)
    out.real = V.real * m
    out.imag = V.imag * m
    return ifftn_real(out)


# ------------------------------------------------------------------------------------------------------------ FSC
def _cmul_conj(ar, ai, br, bi):
    """MATLAB A .* conj(B) on split parts: (ar*br - ai*(-bi), ar*(-bi) + ai*br), one rounding per product."""
    return ar * br - ai * (-bi), ar * (-bi) + ai * br


def shell_fsc(a, b, shells):
    """real(sum A conj B) / (sqrt(sum A conj A) .* sqrt(sum B conj B)) over each shell (boolean [z,y,x] masks, in
    index order), A = fftshift(fftn(a)), sums sequential (dynamo_fsc_fouriermask.m radial_profile_mask)."""
    A = np.fft.fftshift(fftn(a)).ravel()
    B = np.fft.fftshift(fftn(b)).ravel()
    ar, ai, br, bi = A.real, A.imag, B.real, B.imag
    nr, _ = _cmul_conj(ar, ai, br, bi)
    a2, _ = _cmul_conj(ar, ai, ar, ai)
    b2, _ = _cmul_conj(br, bi, br, bi)
    out = np.empty(len(shells))
    for k, s in enumerate(shells):
        s = s.ravel()
        out[k] = msum(nr[s]) / (np.sqrt(msum(a2[s])) * np.sqrt(msum(b2[s])))
    return out


# ---------------------------------------------------------------------------------------------- grids and interp1
def linspace(a, b, n):
    """MATLAB linspace(a, b, n): a + (k*(b - a))/(n - 1) for k = 0..n-2, then b (bit for bit against Dynamo's
    eo_fsc grid; NumPy's linspace multiplies k by the step and differs in the last bit)."""
    n = int(n)
    if n < 2:
        return np.array([float(b)] if n == 1 else [], dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    v = a + (k * (b - a)) / (n - 1)
    v[0], v[-1] = a, b                                  # endpoints exactly (a -0.0 start stays -0.0)
    return v


def interp1(xs, ys, xq):
    """MATLAB interp1(xs, ys, xq) ('linear', increasing xs, xq inside [xs(1), xs(end)]): t = (x - x0)/(x1 - x0),
    y = t*y1 + (1 - t)*y0 (bit for bit against Dynamo's eo_fsc); NaN outside, as interp1."""
    xs, ys, xq = (np.asarray(a, dtype=np.float64) for a in (xs, ys, xq))
    i = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, len(xs) - 2)
    x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
    t = (xq - x0) / (x1 - x0)
    out = t * y1 + (1 - t) * y0
    out[xq == xs[-1]] = ys[-1]
    out[(xq < xs[0]) | (xq > xs[-1])] = np.nan
    return out
