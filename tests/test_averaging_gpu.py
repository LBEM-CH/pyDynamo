"""GPU tests, opt-in (tests/conftest.py): inside a GPU job, e.g. the Slurm sandbox,
PYDYNAMO_GPU_TESTS=1 python -I -m pytest -m gpu tests
"""
import numpy as np
import pytest

from pydynamo.masks import default_mask

pytestmark = pytest.mark.gpu


def _blob(n=32):
    rs = np.random.RandomState(0)
    v = np.zeros((n, n, n), dtype=np.float32)
    for _ in range(6):
        c = rs.randint(10, 22, size=3)
        z, y, x = np.ogrid[:n, :n, :n]
        v += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / 6.0).astype(np.float32)
    return v


def test_rotation_kernel_identity_and_gpu_average_matches_cpu_reference():
    import cupy as cp
    from pydynamo.averaging.average import finalize, halves_for, new_state, ordered_sums
    n = 32
    t = _blob(n)
    rows = np.zeros((4, 35), dtype=np.float32)
    rows[:, 0], rows[:, 2], rows[:, 12] = [1, 2, 3, 4], 1, 0
    parts = np.stack([t] * 4)
    half = halves_for(rows)
    nh = np.bincount(half, minlength=2).astype(np.float64)
    st = ordered_sums(parts, rows, half, nh, default_mask(n), new_state(n), batch=2)
    out = finalize(st, nh, default_mask(n), n)
    assert np.corrcoef(out["average_unweighted"][:-1, :-1, :-1].ravel(), t[:-1, :-1, :-1].ravel())[0, 1] > 0.999
    assert out["n"] == [2, 2] and cp.cuda.runtime.getDeviceCount() > 0


def _serial_dynamo_sums(parts, rows, half, mask, nh):
    """CPU reference of dynamo_average's single-thread project loop (alignParticle / extractAlignedFMask):
    option-4 rotation, NaN -> 0, dynamo_normalize_roi with MATLAB's sequential sums, NaN -> 0, option-0 wedge
    rotation / Ntags; double sums in row (ascending tag) order per half."""
    from pydynamo.averaging.average import mex_matrices, shift_rot_mex_cpu, wedge_mask
    from pydynamo.matlab.numerics import normalize_roi
    n = parts.shape[1]
    M, Sh = mex_matrices(rows)
    S, C, F = np.zeros((2, n ** 3)), np.zeros((2, n ** 3)), np.zeros((2, n ** 3))
    for i in range(len(rows)):
        r = shift_rot_mex_cpu(parts[i], M[i], Sh[i]).astype(np.float64).ravel()
        nan = np.isnan(r)
        r[nan] = 0
        r = normalize_roi(r, np.asarray(mask).ravel())
        r[nan] = 0
        w = shift_rot_mex_cpu(wedge_mask(n, rows[i]), M[i], np.zeros(3, np.float32), implicit_sphere=True,
                              outscope=0.0).astype(np.float64).ravel()
        k = int(half[i])
        S[k] = S[k] + r
        C[k] = C[k] + nan
        F[k] = F[k] + w / nh[k]
    return S, C, F


def test_project_average_kernels_are_mex_exact():
    """ordered_sums (GPU) == the serial CPU port of Dynamo's single-thread loop, bit for bit (S, C, F), also when
    the state is chained over two calls; option-0 GPU rotation bit-equal to the CPU port."""
    import cupy as cp
    from pydynamo.averaging.average import _mex_kernel, halves_for, mex_rotation_matrix, new_state, ordered_sums, shift_rot_mex_cpu
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
    nh = np.bincount(half, minlength=2).astype(np.float64)
    st = ordered_sums(parts[:4], rows[:4], half[:4], nh, default_mask(n), new_state(n), batch=3)
    st = ordered_sums(parts[4:], rows[4:], half[4:], nh, default_mask(n), st, batch=3)
    S, C, F = _serial_dynamo_sums(parts, rows, half, default_mask(n), nh)
    for got, want in ((st["S"], S), (st["C"], C), (st["F"], F)):
        assert np.array_equal(got.view(np.int64), want.view(np.int64))
    e = rows[0, 6:9].astype(np.float64)
    m = np.array(mex_rotation_matrix(*e))
    out = cp.empty((1, n, n, n), dtype=cp.float32)
    _mex_kernel("rot_mex0")((((n ** 3) + 255) // 256,), (256,),
                        (cp.asarray(t), cp.zeros(1, dtype=cp.int64), cp.asarray(m), out, np.float32(0.0),
                         np.int32(n), np.int64(n ** 3)))
    cpu = shift_rot_mex_cpu(t, m, np.zeros(3, np.float32), implicit_sphere=True, outscope=0.0)
    assert np.array_equal(out.get()[0].view(np.uint32), cpu.view(np.uint32))


@pytest.mark.parametrize("T", [1, 2, 16, 64])
def test_roi_moments_kernel_matlab_threads(T):
    """The per-particle dynamo_normalize_roi moments on the GPU (roi_moments) == numerics.normalize_roi with
    the same MATLAB thread count, bit for bit, for a ROI above the 89,000-element threshold where the thread count
    changes the sums (and below it, where it does not)."""
    import cupy as cp
    from pydynamo.matlab import numerics as MN
    from pydynamo.averaging.average import _mex_kernel
    rs = np.random.RandomState(11)
    for nvox in (50000, 100003):
        X = rs.standard_normal((nvox, 3)) * 2.5 + 0.7                   # [nvox, B], one column per particle
        mu, sd = cp.empty(3, cp.float64), cp.empty(3, cp.float64)
        _mex_kernel("roi_moments")((1,), (128,), (cp.asarray(X), np.int32(3), np.int64(nvox), mu, sd, np.int32(T)))
        for b in range(3):
            v = X[:, b]
            m = MN.msum(v, T) / nvox
            assert mu.get()[b] == m
            assert sd.get()[b] == MN.mstd(v - m, T)
            assert np.array_equal(MN.normalize_roi(v, np.ones(nvox), T), (v - mu.get()[b]) / sd.get()[b])


def test_pool_sums_equal_subrange_by_subrange_sums():
    """average.pool_sums (dynamo_average on a MATLAB pool) == the parfor emulation with ordered_sums: per half,
    MATLAB's subranges, each summed from 0 in descending order, the partials added in subrange order; also when a
    subrange is split into two pieces (two GPU workers)."""
    from pydynamo.matlab import numerics as MN
    from pydynamo.averaging.average import halves_for, new_pool_state, new_state, ordered_sums, pool_sums
    n, P, W = 16, 23, 3
    rs = np.random.RandomState(5)
    t = _blob(n)
    rows = np.zeros((P, 35), dtype=np.float32)
    rows[:, 0], rows[:, 2] = np.arange(1, P + 1), 1
    rows[:, 3:6] = rs.uniform(-2, 2, size=(P, 3))
    rows[:, 6:9] = rs.uniform(-180, 180, size=(P, 3))
    rows[:, 12], rows[:, 15], rows[:, 16] = 2, -60, 60
    parts = np.stack([np.roll(t, k, axis=1) + 0.05 * rs.standard_normal((n, n, n)).astype(np.float32)
                      for k in range(P)])
    half = halves_for(rows)
    nh = np.bincount(half, minlength=2).astype(np.float64)
    mask = default_mask(n)
    want = new_state(n)
    order, pieces, mid = [], [], None
    for k in (0, 1):
        idx = np.flatnonzero(half == k)
        a = 0
        for size in MN.parfor_subranges(len(idx), W):
            sub = idx[a:a + size][::-1]
            a += size
            st = ordered_sums(parts[sub], rows[sub], half[sub], nh, mask, new_state(n), batch=2)
            for key in ("S", "C", "F"):
                want[key][k] = want[key][k] + st[key][k]
            cut = 1 if (mid is None and size > 1) else size        # split the first subrange of 2+ in two pieces
            for u, v in ((0, cut), (cut, size)):
                if v > u:
                    if u > 0:
                        mid = len(order)                             # the second call starts inside that subrange
                    pieces.append((len(order), len(order) + v - u, k, u == 0, v == size))
                    order.extend(sub[u:v].tolist())
    assert mid is not None
    order = np.asarray(order)
    got = new_pool_state(n)
    p1 = [q for q in pieces if q[0] < mid]
    p2 = [(q[0] - mid, q[1] - mid) + tuple(q[2:]) for q in pieces if q[0] >= mid]
    got = pool_sums(parts[order[:mid]], rows[order[:mid]], half[order[:mid]], nh, mask, got, p1, batch=2)
    got = pool_sums(parts[order[mid:]], rows[order[mid:]], half[order[mid:]], nh, mask, got, p2, batch=3)
    for key in ("S", "C", "F"):
        assert np.array_equal(got[key].view(np.int64), want[key].view(np.int64)), key
