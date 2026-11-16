"""Reference preparation between iterations, as Dynamo's setup does it (docs/analysis/05 section 7, section 5).

starting_reference_transformed = norm( band( sym(average, fmask) ) ), applied in that fixed order. dim is 1 here
(no binning). The mask used for alignment (mask_transformed) is the project mask, unchanged.

Symmetry operators: pydyn.symmetry.dynamo_sym (cN, dN, helical; docs/analysis/10a_symmetry_1.1.514.md).
"""
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from .average import _kernel, fweight_divide, mex_rotation_matrix, normalize_roi


def binary_mask(mask, dim=None):
    """Dynamo's treatment of map type 'mask' / 'fmask_template' (binary=true) in dynamo_database_volume_process
    (Dynamo 1.1.514 = 1.1.558): dynamo_pipeline_resize(..., 'binary', true) -> dynamo_resize_mask, i.e. resize to `dim`,
    then value > 0.5 -> 1 and <= 0.5 -> 0. Used for the alignment mask (mask_transformed) and for normalising the
    template. The averaging uses the RAW project mask instead (pipeline/average.m:93; normalize_roi takes mask != 0).
    Only dim == box is supported, so the resize is the identity."""
    m = np.asarray(mask, dtype=np.float64)
    if dim is not None and dim != m.shape[0]:
        raise NotImplementedError("mask resize (dim != box) is not implemented")
    return (m > 0.5).astype(np.float32)


def rot_sphere0(vol, eulers):
    """dynamo_rot(vol, eulers) on the GPU: dynamo_shift_rot_mex masking option 0 with outscope 0 (dynamo_rot.m,
    dynamo_shift_rot.m; mex-exact kernel average.rot_mex0: double matrix of float32(eulers), implicit sphere
    |r| >= n/2 -> 0, pivots outside the box -> 0). The volume enters as float32, as the mex converts it."""
    n = vol.shape[0]
    N = n ** 3
    E = np.atleast_2d(np.asarray(eulers, dtype=np.float64))
    mats = np.array([mex_rotation_matrix(*e) for e in E], dtype=np.float64)
    out = cp.empty((len(E), n, n, n), dtype=cp.float32)
    _kernel("rot_mex0")(((len(E) * N + 255) // 256,), (256,),
                        (cp.asarray(vol, dtype=cp.float32), cp.zeros(len(E), dtype=cp.int64), cp.asarray(mats.ravel()),
                         out, np.float32(0.0), np.int32(n), np.int64(len(E) * N)))
    return out.get().astype(np.float64)


def symmetrize(vol, sym, fmask=None):
    """dynamo_sym(vol, sym, 'fmask', fmask, 'foutput', 'binary') (pydyn.symmetry; 10a): c1/none (identity, the fmask
    returned as given), cN, dN and helical h[dphi,dz(,reps)]."""
    from .symmetry import dynamo_sym
    return dynamo_sym(vol, sym or "c1", fmask, "binary")


def isotropic_shell(n, high, low, smoothing=2.0):
    """dpkgeom.isotropicShell / sphere: centred band-pass mask (values <= 0.13 set to 0)."""
    r = np.arange(n) - n // 2
    rz, ry, rx = np.meshgrid(r, r, r, indexing="ij")
    R = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    if high > 0:
        m = np.ones_like(R)
        m[R < high] = np.exp(-((high - R[R < high]) ** 2) / 4)
        m[R > low] = np.exp(-((low - R[R > low]) ** 2) / 4)
    else:
        rho = R / low
        m = np.where(rho <= 1, 1.0, np.exp(-((rho - 1) / (2.0 / low)) ** 2))
    m[m <= 0.13] = 0
    return m


def bandpass(vol, high, low):
    """dynamo_bandpass(vol, [high, low]) for low >= 1: real(ifftn(fftn(V) .* fftshift(shell)))."""
    n = vol.shape[0]
    mask = isotropic_shell(n, high, low)
    return np.real(np.fft.ifftn(np.fft.fftn(vol) * np.fft.fftshift(mask)))


def prepare_template(average, mask, sym="c1", high=2, low=20, fmask=None, symmetrize_with_fmask=True):
    """starting_reference_transformed for the next iteration (dim -> sym -> band -> norm).

    The sym step is dynamo_database_volume_process.m:432-433 (dim == box, 'fmask' given): with
    vpr.symmetrize_with_fmask (convention default 1, dynamo_vpr_convention.m:523-526) dynamo_sym(map, sym, 'fmask',
    <the starting fmask, binarised as map type 'fmask_template'>), else dynamo_sym(map, sym) (:435); without an fmask
    :453-456. This is the same step as DynamoProject.preprocess_references; the "full fmask -> ones" replacement
    of preprocessReferences applies to the written fmask_transformed only, not to the symmetrization fmask."""
    v = np.asarray(average, dtype=np.float64)
    mask = binary_mask(mask)                                # normalisation mask: binarised (volume_process 'norm')
    fm = None
    if fmask is not None and symmetrize_with_fmask:
        fm = binary_mask(fmask)                             # map type 'fmask_template': binary=true
    v, _ = symmetrize(v, sym, fm)
    if low >= 1:
        v = bandpass(v, high, low)
    else:
        raise NotImplementedError("Nyquist-fraction low-pass (dynamo_bandpass_nyquist) not implemented")
    return normalize_roi(v, mask).astype(np.float32)
