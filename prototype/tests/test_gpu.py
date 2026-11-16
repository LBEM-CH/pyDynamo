"""GPU tests (skipped without CuPy and a CUDA device). Run them on a GPU machine:
python -I -m pytest -m gpu prototype/tests
"""
import numpy as np
import pytest

from pydyn import emio
from pydyn.angles import rotation_matrices
from pydyn.project import default_mask

pytestmark = pytest.mark.gpu


class _Card:
    dim = sidelength = 32
    shrink = 1.0
    high, low = 2.0, 12.0
    use_CC, localnc = 1, 1
    area_search, area_search_modus = [4, 4, 4], 0
    cone_range, cone_sampling, cone_flip = 0.0, 0.0, 0
    inplane_range, inplane_sampling, inplane_flip = 0.0, 0.0, 0
    refine, refine_factor = 0, 2.0


def _blob(n=32):
    rs = np.random.RandomState(0)
    v = np.zeros((n, n, n), dtype=np.float32)
    for _ in range(6):
        c = rs.randint(10, 22, size=3)
        z, y, x = np.ogrid[:n, :n, :n]
        v += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / 6.0).astype(np.float32)
    return v


def test_engine_recovers_known_shift():
    from pydyn.engine import Engine
    n = 32
    t = _blob(n)
    shift = (2, -1, 3)                                     # (x, y, z) voxels
    p = np.roll(t, shift=(shift[2], shift[1], shift[0]), axis=(0, 1, 2))
    rows = np.zeros((1, 35), dtype=np.float32)
    rows[0, 0], rows[0, 12] = 1, 0                         # no wedge
    eng = Engine(t, default_mask(n), _Card(), batch=8, exact=False)    # exact mode: tests/test_exact.py
    eng.prepare_particles(p[None], rows)
    out = eng.refine(rows)
    assert np.allclose(out[0, 3:6], shift, atol=0.3)
    assert out[0, 9] > 0.95


def test_rotation_kernel_identity_and_gpu_average_matches_cpu_reference():
    import cupy as cp
    from pydyn.average import finalize, halves_for, partial_sums
    n = 32
    t = _blob(n)
    rows = np.zeros((4, 35), dtype=np.float32)
    rows[:, 0], rows[:, 2], rows[:, 12] = [1, 2, 3, 4], 1, 0
    parts = np.stack([t] * 4)
    ps = partial_sums(parts, rows, halves_for(rows), default_mask(n), batch=2)
    out = finalize([ps], default_mask(n), n)
    assert np.corrcoef(out["average_unweighted"][:-1, :-1, :-1].ravel(), t[:-1, :-1, :-1].ravel())[0, 1] > 0.999
    assert ps["n"].tolist() == [2, 2] and cp.cuda.runtime.getDeviceCount() > 0


def _serial_dynamo_sums(parts, rows, half, mask):
    """CPU reference of dynamo_average's project loop (alignParticle / extractAlignedFMask): option-4 rotation,
    NaN -> 0, dynamo_normalize_roi in double (MATLAB mean / var), NaN -> 0, option-0 wedge rotation; double sums in
    row order per half."""
    from pydyn.average import mex_matrices, shift_rot_mex_cpu, wedge_mask
    n = parts.shape[1]
    M, Sh = mex_matrices(rows)
    I = np.asarray(mask).ravel() != 0
    S, C, F = np.zeros((2, n ** 3)), np.zeros((2, n ** 3)), np.zeros((2, n ** 3))
    for i in range(len(rows)):
        r = shift_rot_mex_cpu(parts[i], M[i], Sh[i]).astype(np.float64).ravel()
        nan = np.isnan(r)
        r[nan] = 0
        iroi = r[I]
        mu = iroi.sum() / iroi.size
        x = iroi - mu
        x = x - x.sum() / x.size
        r = (r - mu) / np.sqrt((x * x).sum() / (x.size - 1))
        r[nan] = 0
        w = shift_rot_mex_cpu(wedge_mask(n, rows[i]), M[i], np.zeros(3, np.float32), implicit_sphere=True,
                              outscope=0.0).astype(np.float64).ravel()
        k = int(half[i])
        S[k] += r
        C[k] += nan
        F[k] += w
    return S, C, F


def test_project_average_kernels_are_mex_exact():
    """partial_sums (GPU) == the serial CPU port of Dynamo's loop: identical NaN counts and rotated wedges, and S
    equal to summation-order rounding; option-0 GPU rotation bit-equal to the CPU port."""
    import cupy as cp
    from pydyn.average import _kernel, halves_for, mex_rotation_matrix, partial_sums, shift_rot_mex_cpu
    n = 32
    rs = np.random.RandomState(3)
    t = _blob(n)
    P = 6
    rows = np.zeros((P, 35), dtype=np.float32)
    rows[:, 0], rows[:, 2] = np.arange(1, P + 1), 1
    rows[:, 3:6] = rs.uniform(-3, 3, size=(P, 3))
    rows[:, 6:9] = rs.uniform(-180, 180, size=(P, 3))
    rows[:, 12], rows[:, 15], rows[:, 16] = 2, -60, 60
    parts = np.stack([np.roll(t, k, axis=2) + 0.01 * rs.standard_normal((n, n, n)).astype(np.float32)
                      for k in range(P)])
    half = halves_for(rows)
    ps = partial_sums(parts, rows, half, default_mask(n), batch=4)
    S, C, F = _serial_dynamo_sums(parts, rows, half, default_mask(n))
    assert np.array_equal(ps["C"], C)
    assert np.array_equal(ps["F"], F)
    assert np.abs(ps["S"] - S).max() < 1e-11 * max(1.0, np.abs(S).max())
    e = rows[0, 6:9].astype(np.float64)
    m = np.array(mex_rotation_matrix(*e))
    out = cp.empty((1, n, n, n), dtype=cp.float32)
    _kernel("rot_mex0")((((n ** 3) + 255) // 256,), (256,),
                        (cp.asarray(t), cp.zeros(1, dtype=cp.int64), cp.asarray(m), out, np.float32(0.0),
                         np.int32(n), np.int64(n ** 3)))
    cpu = shift_rot_mex_cpu(t, m, np.zeros(3, np.float32), implicit_sphere=True, outscope=0.0)
    assert np.array_equal(out.get()[0].view(np.uint32), cpu.view(np.uint32))
