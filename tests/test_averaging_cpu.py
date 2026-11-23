"""CPU checks of the host arithmetic of the engine and of the averaging (no GPU needed)."""
import math
import os

import numpy as np
import pytest

from pydynamo.averaging.average import finalize, fsc_fouriermask, fweight_divide, halves_for, normalize_roi, wedge_mask
from pydynamo.engine.kernels import subpixel_spline


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
    from pydynamo.matlab import numerics as MN
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
    from pydynamo.masks import default_mask
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
    from pydynamo.matlab import numerics as MN
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
    """pydynamo.matlab.fdlibm sin/cos/atan2/acos against values recorded from Dynamo 1.1.514's MATLAB runtime (bit patterns,
    tests/fixtures/matlab_trig_reference.json); measured on 1e6 sin/cos, 3e5 atan2 and 3.4e5 acos arguments: 0 differ."""
    import json
    from pydynamo.matlab import fdlibm as F
    fx = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "matlab_trig_reference.json")))
    h = float.fromhex
    assert all(F.sin(h(x)).hex() == y for x, y in fx["sin"]) and all(F.cos(h(x)).hex() == y for x, y in fx["cos"])
    assert all(F.atan2(h(y), h(x)).hex() == z for y, x, z in fx["atan2"])
    assert all(F.acos(h(x)).hex() == y for x, y in fx["acos"])
    assert all(F.pow(h(x), h(y)).hex() == z for x, y, z in fx["pow"])           # MATLAB scalar ^ (e_pow.c)
    assert math.copysign(1.0, F.sin(-0.0)) == -1.0 and F.acos(1.0) == 0.0 and F.atan2(0.0, -1.0) == math.pi


def test_fdlibm_rem_pio2_quick_check_of_e_rem_pio2():
    """e_rem_pio2.c, medium size: one round with 33+53 bits of pi/2 when n < 32 and the high word of |x| is not
    npio2_hw[n-1] (the high word of n*pi/2), whatever the exponent loss; the 2nd/3rd iterations otherwise."""
    from fractions import Fraction
    from pydynamo.matlab import fdlibm as F
    pi = Fraction("3.14159265358979323846264338327950288419716939937510582097494459")
    assert [F._hi(float(pi * (k + 1) / 2)) for k in range(32)] == list(F._NPIO2_HW)
    x = math.pi * (1 + 2.0 ** -17)                                 # about 180.0014 degrees
    n = int(x * F._INVPIO2 + 0.5)
    r = x - float(n) * F._PIO2_1
    w = float(n) * F._PIO2_1T
    y0 = r - w
    ix = F._hi(x) & 0x7fffffff
    # the case: not n*pi/2's high word, yet an exponent loss > 16 (where the 2nd iteration would start)
    assert n == 2 and ix != F._NPIO2_HW[1] and (ix >> 20) - ((F._hi(y0) >> 20) & 0x7ff) > 16
    assert F._rem_pio2(x) == (2, y0, (r - y0) - w)
    assert F._rem_pio2(-x) == (-2, -y0, -((r - y0) - w))
    # high word of pi itself (npio2_hw[1]): iterated to 151 bits, sin(pi) = pi - fl(pi) rounded
    assert F.sin(math.pi) == 1.2246467991473532e-16 and F.cos(math.pi) == -1.0


# ------------------------------------------------------------------------------------------------ masks
def test_matlab_exp_takes_e_exp_overflow_underflow_branch():
    """fdlibm e_exp.c for |x| >= 709.78: NaN -> x+x, exp(+-inf) = {inf, 0}, inf above o_threshold, 0 below
    u_threshold, and the normal reduction in between (subnormal results, one rounding as e_exp.c's
    (y*2^(k+1000))*2^-1000); dynamo_blob reaches it for a small 'gaussian' in a large box."""
    from pydynamo.masks import dynamo_blob, matlab_exp
    sp = matlab_exp(np.array([np.nan, np.inf, -np.inf, 709.79, 1e308, -745.2, -1e308]))
    assert np.isnan(sp[0]) and sp[1] == np.inf and sp[2] == 0.0 and sp[3] == np.inf and sp[4] == np.inf
    assert sp[5] == 0.0 and sp[6] == 0.0
    x = np.concatenate([np.linspace(-745.13, -709.79, 2001), [-720.0, -745.0]])
    got = matlab_exp(x)
    want = np.array([math.exp(v) for v in x.tolist()])
    assert np.all(got > 0) and np.all(got < 2.2250738585072014e-308)             # subnormal, not flushed
    assert np.all(np.abs(got - want) <= 2.0 ** -1074)                            # within one subnormal step
    top = matlab_exp(np.array([709.7825]))[0]                                    # >= 709.78, <= o_threshold
    assert np.isfinite(top) and abs(top - math.exp(709.7825)) <= 2.3e-16 * top
    b = dynamo_blob(64, 3)                                     # 'gaussian' 1.5 in a 64 box: corner argument -946
    assert b[32, 32, 32] == 1.0 and b[0, 0, 0] == 0.0 and b.min() >= 0.0


def test_dynamo_mask_cylinder_with_a_small_gaussian():
    """dynamo_mask(..., 'gaussian', g) with g < box/37 (dynamo_blob arguments below -709.78) gives a mask."""
    from pydynamo.masks import dynamo_mask
    m = dynamo_mask(size=32, type="cylinder", r=8, height=16, gaussian=0.5)
    assert m.shape == (32, 32, 32) and m.min() == 0.0 and m.max() == 1.0 and np.isfinite(m).all()


def test_dynamo_mask_cylinder_ignores_what_dynamo_mask_does_not_read(tmp_path):
    """dynamo_mask.m for a cylinder (:483-485): smooth, fwhm, tilt, sz, r2, thick, nfig change nothing; r1 stands in
    for r only when r and sx are empty (:350-360); 'radius' and 'h' override 'r' and 'height' (:262-263); an empty
    value is not given. What changes a cylinder and is not ported is still refused."""
    from pydynamo.masks import MaskEditor, dynamo_mask
    base = dynamo_mask(size=16, type="cylinder", r=4, height=8)
    same = [dict(tilt=[-45, 45]), dict(fwhm=3), dict(sz=2), dict(r2=7), dict(thick=3), dict(smooth=2), dict(nfig=1),
            dict(band=[]), dict(eulers=np.zeros(0)), dict(position=""), dict(shifts=None), dict(invert=np.zeros(0)),
            dict(r1=6), dict(r1=[1, 2])]
    for kw in same:
        assert np.array_equal(dynamo_mask(size=16, type="cylinder", r=4, height=8, **kw), base), kw
    for kw in (dict(r1=4), dict(sx=4, r1=6), dict(r=6, radius=4), dict(r=np.zeros(0), radius=4),
               dict(r=4, height=12, h=8)):
        kw.setdefault("height", 8)
        assert np.array_equal(dynamo_mask(size=16, type="cylinder", **kw), base), kw
    with pytest.raises(ValueError):                            # q.r1+q.thick of unequal lengths (:395)
        dynamo_mask(size=16, type="cylinder", r=4, height=8, r1=[1, 2], thick=[1, 2, 3])
    with pytest.raises(ValueError):                            # q.r2-q.r1 of unequal lengths (:405)
        dynamo_mask(size=16, type="cylinder", r=4, height=8, r1=[1, 2], r2=[1, 2, 3])
    for kw in (dict(band=[0.1, 0.4]), dict(shifts=[1, 0, 0]), dict(position=[5, 9, 9]), dict(ws="m")):
        with pytest.raises(NotImplementedError):
            dynamo_mask(size=16, type="cylinder", r=4, height=8, **kw)
    # the editor: a radio left on from another type, and ticked radios with empty boxes ([] = not given)
    ed = MaskEditor(size=[16, 16, 16], mask=str(tmp_path / "m.em"))
    ed.set_type("cylinder")
    ed.select("r", "4")
    ed.select("height", "8")
    ed.select("tilt", "-60 60")
    for p in ("fwhm", "eulers", "band", "gaussian"):
        ed.select(p)
    assert np.array_equal(ed.save(), base) and ed.messages[-1] == "Mask created and saved"


def test_dynamo_mollify_takes_one_or_three_widths(monkeypatch):
    """dynamo_mollify.m:107 dynamo_blob(size(volume,3), hwhm*2) of the whole 'gaussian' (dynamo_mask.m:565):
    3 values are a width per axis [x, y, z], 2 values are dynamo_blob's error (dynamo_blob.m:60-62)."""
    from pydynamo import masks
    v = np.zeros((16, 16, 16))
    v[8, 8, 8] = 1.0
    f = masks.dynamo_mollify(v, [1, 2, 3])
    assert f[8, 8, 9] < f[8, 9, 8] < f[9, 8, 8]                # x (last axis) narrowest, z widest
    assert np.array_equal(masks.dynamo_mollify(v, [1, 1, 1]), masks.dynamo_mollify(v, 1))
    with pytest.raises(ValueError, match="one value or three"):
        masks.dynamo_mollify(v, [1, 2])
    seen = []
    blob = masks.dynamo_blob
    monkeypatch.setattr(masks, "dynamo_blob", lambda l, fw: seen.append(np.ravel(fw).tolist()) or blob(l, fw))
    masks.dynamo_mask(size=16, type="cylinder", r=4, height=8, gaussian=[1, 2, 3])
    masks.dynamo_mask(size=16, type="cylinder", r=4, height=8, gaussian="1 2 3")
    assert seen == [[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]
    with pytest.raises(ValueError, match="one value or three"):
        masks.dynamo_mask(size=16, type="cylinder", r=4, height=8, gaussian=[1, 2])
