"""CPU checks of the NumPy parts of the engine and the averaging (no GPU needed)."""
import numpy as np

from pydyn.average import finalize, fsc_fouriermask, fweight_divide, halves_for, normalize_roi, wedge_mask
from pydyn.engine import bandpass_half, folded_indices_half, subpixel_spline, wedge_half
from pydyn.reference import bandpass, isotropic_shell


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
    parts = [{"S": np.stack([vol.ravel() * 4, vol.ravel() * 4]), "C": np.zeros((2, n ** 3)),
              "F": np.ones((2, n ** 3)) * 4, "n": np.array([4, 4])}]
    out = finalize(parts, mask, n)
    assert out["n"] == [4, 4] and np.allclose(out["eo_fsc"][:4], 1)
    assert np.isclose(np.corrcoef(out["average"].ravel(), vol.ravel())[0, 1], 1)


def test_wedge_mask_type1_centred():
    row = np.zeros(35, dtype=np.float32)
    row[12], row[13], row[14] = 1, -60, 60
    W = wedge_mask(16, row)
    c = 8
    assert W[c, c, c] == 1 and W[c + 3, c, c] == 0 and W[c, c, c + 3] == 1


def test_template_bandpass_shell():
    m = isotropic_shell(32, 2, 10)
    assert m[16, 16, 16 + 5] == 1 and m[16, 16, 16 + 15] == 0
    v = np.random.RandomState(4).rand(32, 32, 32)
    assert bandpass(v, 2, 10).shape == v.shape


def test_subpixel_spline_degenerate_denominators_do_not_raise():
    # 2*y2 - 3*y1 + y3 == 0 and y1 + 2*y2 - 3*y3 == 0 cases: Dynamo's C code yields inf/NaN, which are discarded
    for y in ((0.5, 0.25, -0.25), (0.5, 0.5, 0.5000001), (1.0, 1.0, 0.0), (-0.3, 0.1, -0.9)):
        p = subpixel_spline(*y)
        assert -1.0 <= p <= 1.0


def test_binary_mask_is_dynamo_resize_mask_threshold():
    # dynamo_resize_mask: > 0.5 -> 1, <= 0.5 -> 0 (alignment mask and template normalisation; averaging uses the raw mask)
    from pydyn.reference import binary_mask
    m = np.array([0.0, 0.2, 0.5, 0.5000001, 0.9, 1.0]).reshape(1, 1, 6) * np.ones((6, 6, 1))
    b = binary_mask(m)
    assert b[0, 0].tolist() == [0, 0, 0, 1, 1, 1]
