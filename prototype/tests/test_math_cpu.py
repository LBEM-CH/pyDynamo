"""CPU checks of the NumPy parts of the engine and the averaging (no GPU needed)."""
import math
import os

import numpy as np

from pydyn.average import finalize, fsc_fouriermask, fweight_divide, halves_for, normalize_roi, wedge_mask
from pydyn.engine import bandpass_half, folded_indices_half, subpixel_spline, wedge_half


def test_folded_indices_like_dynamo():
    kz, ky, kx = folded_indices_half(8)
    assert list(kz.ravel()) == [0, 1, 2, 3, 4, 3, 2, 1]               # x >= n/2 -> n - x
    assert list(kx.ravel()) == [0, 1, 2, 3, 4]


def test_bandpass_half_shape():
    H = bandpass_half(64, 2, 20, 3)
    assert H[0, 0, 0] == np.float32(np.exp(-(2 / 3) ** 2))             # DC kept at exp(-(2/3)^2)
    assert H[0, 0, 10] == 1 and H[0, 0, 24] == 0                        # plateau, and beyond low + window
    assert np.isclose(H[0, 0, 22], np.exp(-((22 - 20) / 3) ** 2), atol=1e-6)


def test_wedge_half_type1():
    row = np.zeros(34, dtype=np.float32)
    row[14] = 60
    W = wedge_half(16, 1, row)
    assert W[0, 0, 0] == 1                                              # DC (0/0) kept
    assert W[3, 0, 0] == 0                                              # kx = 0, kz != 0: missing
    assert W[1, 0, 5] == 1 and W[7, 0, 1] == 0                          # small vs steep kz/kx


def test_subpixel_spline():
    assert subpixel_spline(0.5, 1.0, 0.5) == 0.0                        # symmetric peak
    right = subpixel_spline(0.2, 1.0, 0.8)
    left = subpixel_spline(0.8, 1.0, 0.2)
    assert 0 < right < 0.5 and np.isclose(left, -right, atol=1e-6)


def test_normalize_roi_and_fweight_divide():
    v = np.random.RandomState(0).rand(16, 16, 16) * 5 + 3
    m = np.zeros_like(v)
    m[4:12, 4:12, 4:12] = 1
    n = normalize_roi(v, m)
    assert np.isclose(n[m > 0].mean(), 0, atol=1e-12) and np.isclose(n[m > 0].std(ddof=1), 1)
    assert np.allclose(fweight_divide(v, np.ones_like(v), 0.1), v)


def test_fsc_identical_volumes_is_one():
    v = np.random.RandomState(2).rand(16, 16, 16)
    assert np.allclose(fsc_fouriermask(v, v, 4), 1)


def test_halves_by_position_in_sorted_tags():
    rows = np.zeros((5, 35), dtype=np.float32)
    rows[:, 0] = [44, 5, 51, 48, 65]
    rows[:, 2] = [1, 1, 1, 1, 0]                                        # last one not selected
    assert list(halves_for(rows)) == [1, 0, 1, 0, -1]                   # sorted: 5(o) 44(e) 48(o) 51(e)


def test_finalize_on_constructed_sums():
    n = 16
    rs = np.random.RandomState(3)
    vol = rs.rand(n, n, n)
    mask = np.ones((n, n, n))
    state = {"S": np.stack([vol.ravel() * 4, vol.ravel() * 4]), "C": np.zeros((2, n ** 3)),
             "F": np.ones((2, n ** 3))}
    out = finalize(state, np.array([4.0, 4.0]), mask, n)
    assert out["n"] == [4, 4] and np.allclose(out["eo_fsc"][:4], 1)
    assert np.isclose(np.corrcoef(out["average"].ravel(), vol.ravel())[0, 1], 1)


def test_wedge_mask_type1_centred():
    row = np.zeros(35, dtype=np.float32)
    row[12], row[13], row[14] = 1, -60, 60
    W = wedge_mask(16, row)
    c = 8
    assert W[c, c, c] == 1 and W[c + 3, c, c] == 0 and W[c, c, c + 3] == 1


def test_subpixel_spline_degenerate_denominators_do_not_raise():
    # 2*y2 - 3*y1 + y3 == 0 and y1 + 2*y2 - 3*y3 == 0 cases: Dynamo's C code yields inf/NaN, which are discarded
    for y in ((0.5, 0.25, -0.25), (0.5, 0.5, 0.5000001), (1.0, 1.0, 0.0), (-0.3, 0.1, -0.9)):
        p = subpixel_spline(*y)
        assert -1.0 <= p <= 1.0


def test_matlab_numerics_sums_are_sequential_like_single_thread_matlab():
    """Single-thread MATLAB sum/mean/std (Dynamo's executables run -singleCompThread): sequential left-to-right sums,
    mean = sum/n, std = sqrt(sum(abs(x - mean(x)).^2)/(n-1)); measured bit for bit in the Dynamo 1.1.514 runtime."""
    from pydyn import matlab_numerics as MN
    rs = np.random.RandomState(4)
    x = rs.random_sample(100003) * 10.0 ** rs.uniform(-3, 3, 100003)
    acc = 0.0
    for v in x.tolist():
        acc = acc + v
    assert MN.msum(x) == acc
    m = MN.mmean(x)
    assert m == acc / x.size
    d = [(v - m) for v in x.tolist()]
    q = 0.0
    for v in d:
        q = q + abs(v) * abs(v)
    assert MN.mstd(x) == math.sqrt(q / (x.size - 1))
    mask = (rs.random_sample(x.size) > 0.2).astype(np.float64)
    v = x.copy()
    v[::11] = np.nan
    roi = v[(mask != 0) & ~np.isnan(v)]
    mu = MN.msum(roi) / roi.size
    out = MN.normalize_roi(v, mask)
    assert np.array_equal(out[~np.isnan(v)], ((v - mu) / MN.mstd(roi - mu))[~np.isnan(v)])
    assert np.isnan(out[np.isnan(v)]).all()


def test_default_mask_is_dynamos():
    from pydyn.masks import default_mask
    m = default_mask(64)
    assert int(m.sum()) == 124471                       # voxel count of dcp 'masks default' at l = 64
    # Dynamo's double-precision test loses 16 of the 198 lattice points lying exactly on the sphere |r|^2 = 31^2
    # (its rounding depends on the x+y+z summation order, so the mask is NOT exactly axis-symmetric).
    g = np.arange(64) - 32
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    r2 = x ** 2 + y ** 2 + z ** 2
    ideal = (r2 <= 961).astype(np.float32)
    diff = np.argwhere(ideal != m)
    assert len(diff) == 16 and all(r2[tuple(d)] == 961 for d in diff)


def test_matlab_linspace_and_interp1_formulas():
    """MATLAB linspace = a + (k*(b - a))/(n - 1) with exact endpoints, interp1 'linear' = t*y1 + (1 - t)*y0 with
    t = (x - x0)/(x1 - x0) (both matched bit for bit against Dynamo's eo_fsc in the Dynamo 1.1.514 runtime)."""
    from pydyn import matlab_numerics as MN
    a, b, n = 1.0, 32.0, 16
    v = MN.linspace(a, b, n)
    want = [a + (k * (b - a)) / (n - 1) for k in range(n)]
    want[0], want[-1] = a, b
    assert v.tolist() == want
    assert np.array_equal(MN.linspace(-0.0, 1.0, 3)[:1].view(np.int64), np.array([-0.0]).view(np.int64))
    rs = np.random.RandomState(8)
    ys = rs.standard_normal(n)
    xq = MN.linspace(1.0, 32.0, 32)
    got = MN.interp1(v, ys, xq)
    for x, g in zip(xq.tolist(), got.tolist()):
        if x == v[-1]:
            assert g == ys[-1]
            continue
        i = max(k for k in range(n - 1) if v[k] <= x)
        t = (x - v[i]) / (v[i + 1] - v[i])
        assert g == t * ys[i + 1] + (1 - t) * ys[i]
    assert np.isnan(MN.interp1(v, ys, np.array([0.5, 40.0]))).all()


def test_fdlibm_trig_equals_matlab_runtime_values():
    """pydyn.fdlibm sin/cos/atan2/acos against values recorded from Dynamo 1.1.514's MATLAB runtime (bit patterns,
    tests/fixtures/matlab_trig_reference.json); measured on 1e6 sin/cos, 3e5 atan2 and 3.4e5 acos arguments: 0 differ."""
    import json
    from pydyn import fdlibm as F
    fx = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "matlab_trig_reference.json")))
    h = float.fromhex
    assert all(F.sin(h(x)).hex() == y for x, y in fx["sin"]) and all(F.cos(h(x)).hex() == y for x, y in fx["cos"])
    assert all(F.atan2(h(y), h(x)).hex() == z for y, x, z in fx["atan2"])
    assert all(F.acos(h(x)).hex() == y for x, y in fx["acos"])
    assert math.copysign(1.0, F.sin(-0.0)) == -1.0 and F.acos(1.0) == 0.0 and F.atan2(0.0, -1.0) == math.pi
