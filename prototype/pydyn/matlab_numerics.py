"""MATLAB numerics as Dynamo 1.1.514's executables compute them, bit for bit.

Dynamo compiles its non-graphical executables (dynamo_execute_project, which runs a dcp project) with
`-R -singleCompThread` (matlab/src/dynamo_compile.m:377-387), so MATLAB runs single-threaded there. Measured in the
Dynamo 1.1.514 MATLAB runtime (R2019a):
* sum(x) of a double vector is then strictly sequential, left to right, starting from 0 (with more threads and
  n >= 89,000 MATLAB splits x into T chunks of floor(n/T), which changes the last bits; the project path never does);
* mean(x) = sum(x) / n; std(x) = sqrt(sum(abs(x - mean(x)).^2) / (n - 1)) with those sums;
* fftn / ifftn are FFTW (pydyn.masks: r2c + Hermitian completion for a real input; for ifftn MATLAB tests the
  input for exact conjugate symmetry and then computes ifft(ifft(ifft(V,[],1),[],2),[],3) with a complex-to-real
  last step, otherwise the complex 3-D backward transform; / N); the system FFTW 3.3.8 gives the same bits as
  MATLAB's own FFTW 3.3.3, so no MATLAB library is needed. Verified on 64^3 boxes; MATLAB's FFTs of
  non-power-of-two boxes (48, 50, 45 tested) differ from these FFTW plans.
NumPy's sum / mean / std use pairwise orders and NumPy's FFT is pocketfft; both differ in the last bits, which is
enough to change float32 voxels of Dynamo's averages and the noise-level shells of an FSC. MATLAB's linear index
order of an [x, y, z] array is the C order of pyDynamo's [z, y, x] arrays, so flat order = MATLAB order.
"""
import numpy as np


def msum(x):
    """MATLAB sum of a vector (single computational thread): sequential, left to right."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.cumsum(x)[-1]) if x.size else 0.0


def mmean(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return msum(x) / x.size


def mstd(x):
    """MATLAB std(x) (normalised by n - 1): sqrt(sum(abs(x - mean(x)).^2) / (n - 1)); 0 for a scalar."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0
    d = x - mmean(x)
    return float(np.sqrt(msum(np.abs(d) ** 2) / (n - 1)))


def normalize_roi(v, mask):
    """dynamo_normalize_roi_one (dynamo_normalize_roi.m): iroi = v(setdiff(find(mask), find(isnan(v)))) in index
    order; miroi = mean(iroi); stdiroi = std(iroi - miroi); (v - mean(iroi)) ./ stdiroi. In double."""
    v = np.asarray(v, dtype=np.float64)
    sel = (np.asarray(mask).ravel() != 0) & ~np.isnan(v.ravel())
    iroi = v.ravel()[sel]
    if iroi.size == 0:
        raise ValueError("normalize_roi: no valid points found in this particle")
    mu = msum(iroi) / iroi.size
    sd = mstd(iroi - mu)
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
    return b.rfftn_full(v) if b is not None else np.fft.fftn(v)


def ifftn_real(V):
    """real(ifftn(V)) of a 3-D complex array as MATLAB computes it (pydyn.masks._FFTW.ifftn_real: an exactly
    conjugate-symmetric V takes MATLAB's real path, x and y complex then z complex-to-real; any other V the
    complex 3-D backward transform; / N)."""
    b = _fftw()
    if b is None:
        return np.real(np.fft.ifftn(V))
    return b.ifftn_real(V)


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
