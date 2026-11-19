"""Bit-for-bit checks of MATLAB R2019a numerics (the runtime of Dynamo 1.1.514) used by pydyn.dalign / pydyn.abp,
against values computed by MATLAB itself (headless Dynamo 1.1.514, one computational thread) on synthetic inputs;
fixture tests/fixtures/matlab_exact_reference.json (float64 as hex, large outputs as SHA-256 of their little-endian
bytes in C order). No user data is involved: every input is either stored in the fixture or generated below with
+, -, *, / only, which IEEE 754 makes identical on every machine.

* fdlibm.atan2d: MATLAB's atan2d builtin (t = atan2(y, x); t * lo + t * hi);
* dalign.matlab_colon: a:d:b, incl. the even-n middle element (a + (c-a)/2 for same-sign ends);
* matlab_numerics.cmul / conj: A .* conj(B) without fused multiply-adds, conj giving +0;
* matlab_numerics.ifftn_real: MATLAB's ifftn of a conjugate-symmetric array (needs FFTW);
* dalign.dynamo_align on a synthetic 32^3 pair (needs FFTW);
* matlab_numerics.msum: MATLAB's sum with T computational threads (n >= 89,000 chunked), T = 1 .. 64;
* masks._FFTW fftn / ifftn on 18^3, 20^3, 21^3, 30^3, 80^3, 96^3, 128^3 and a non-cubic MATLAB size 20x18x21,
  through the self-checked fast path and through the exact (grouped 1-D) path (needs FFTW).
"""
import hashlib
import json
import os
import struct

import numpy as np
import pytest

from pydyn import dalign as D, fdlibm as FD, matlab_numerics as MN

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "matlab_exact_reference.json")
HAVE_FIX = os.path.isfile(FIX)
REF = json.load(open(FIX)) if HAVE_FIX else {}
needs_fftw = pytest.mark.skipif(not MN.exact_fft(), reason="no FFTW 3 library (numpy FFTs are not MATLAB's)")
needs_fix = pytest.mark.skipif(not HAVE_FIX, reason="fixture absent")

h2d = lambda s: struct.unpack(">d", bytes.fromhex(s))[0]
d2h = lambda v: struct.pack(">d", float(v)).hex()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


# --------------------------------------------------------------------------------------- synthetic inputs (IEEE)
def blobs(n, centres):
    """Sum of w / (1 + r^2 / s) on the [z, y, x] grid -n/2 .. n/2-1 (only +, -, *, /); n or (nz, ny, nx)."""
    shape = (n,) * 3 if np.isscalar(n) else tuple(n)
    z, y, x = np.meshgrid(*[np.arange(k, dtype=np.float64) - k / 2 for k in shape], indexing="ij")
    v = np.zeros(shape)
    for cx, cy, cz, s, w in centres:
        r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)
        v = v + w / (1.0 + r2 / s)
    return v


FFT_SHAPES = [(18, 18, 18), (20, 20, 20), (21, 21, 21), (30, 30, 30), (21, 18, 20),     # [z, y, x]
              (80, 80, 80), (96, 96, 96), (128, 128, 128)]


def fft_inputs(shape):
    a = blobs(shape, PARTICLE)
    b = blobs(shape, TEMPLATE)
    return a, a + 1j * b


PARTICLE = [(1.5, -2.0, 0.5, 6.0, 1.0), (-4.0, 3.0, -1.0, 4.0, 0.7), (3.0, 4.5, 2.5, 3.0, 0.5),
            (-2.0, -5.0, 3.0, 5.0, 0.6)]
TEMPLATE = [(1.0, -2.25, 0.0, 6.0, 1.0), (-4.5, 2.5, -1.5, 4.0, 0.7), (2.5, 4.0, 2.0, 3.0, 0.5),
            (-2.5, -5.5, 2.5, 5.0, 0.6)]


def synthetic_align_inputs(n=32):
    g = np.arange(n, dtype=np.float64) - n / 2
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    mask = ((x * x + y * y + z * z) <= 144.0).astype(np.float64)
    return blobs(n, PARTICLE), blobs(n, TEMPLATE), mask


ALIGN = {"cone_range": 30.0, "cone_sampling": 10.0, "inplane_range": 60.0, "inplane_sampling": 20.0, "refine": 2.0,
         "refine_factor": 2.0, "use_CC": 1.0, "area_search": [4.0, 4.0, 2.0], "area_search_modus": 4.0,
         "localnc": 1.0}


def complex_pairs(m=2000):
    """A, B with every fourth imaginary part exactly 0 (MATLAB's conj gives +0 there)."""
    t = np.arange(m, dtype=np.float64)
    ar = (t * 7.0 - 3001.0) / 37.0
    ai = (t * 13.0 - 9000.0) / 41.0
    br = (t * 11.0 - 7777.0) / 43.0
    bi = (t * 17.0 - 5000.0) / 47.0
    ai[::4] = 0.0
    bi[1::4] = 0.0
    return ar + 1j * ai, br + 1j * bi


# ------------------------------------------------------------------------------------------------------- tests
@needs_fix
def test_atan2d_is_matlab_builtin():
    y = np.array([h2d(s) for s in REF["atan2d"]["y"]])
    x = np.array([h2d(s) for s in REF["atan2d"]["x"]])
    got = np.array([FD.atan2d(a, b) for a, b in zip(y, x)])
    assert sha(got) == REF["atan2d"]["sha"]
    plain = (180 / np.pi) * np.array([FD.atan2(a, b) for a, b in zip(y, x)])
    assert sha(plain) != REF["atan2d"]["sha"]                          # 180/pi * atan2 is not MATLAB's atan2d


@needs_fix
def test_matlab_colon_even_middle():
    vals = []
    for a, d, b in REF["colon"]["ranges"]:
        v = D.matlab_colon(h2d(a), h2d(d), h2d(b))
        vals.append(float(v.size))
        vals.extend(v.tolist())
    assert sha(np.array(vals)) == REF["colon"]["sha"]


@needs_fix
def test_cmul_conj_as_matlab():
    A, B = complex_pairs()
    Q = MN.cmul(A, MN.conj(B))
    assert sha(np.concatenate([Q.real, Q.imag])) == REF["cmul_conj"]["sha"]


@needs_fix
@needs_fftw
def test_ifftn_conjugate_symmetric_path():
    a, b = blobs(16, PARTICLE[:2]), blobs(16, TEMPLATE[2:])
    F = MN.fftn(a)
    assert sha(MN.ifftn_real(F)) == REF["ifftn"]["roundtrip_sha"]
    assert sha(MN.ifftn_real(MN.cmul(F, MN.conj(MN.fftn(b))))) == REF["ifftn"]["product_sha"]


@needs_fix
@needs_fftw
def test_dynamo_align_synthetic_bit_identical():
    p, t, m = synthetic_align_inputs()
    vpr = {"%s_r1" % k: v for k, v in ALIGN.items()}
    s = D.dynamo_align(p, t, vpr, 1, mask=m, verbose=False)
    R = REF["dynamo_align"]
    assert [d2h(v) for v in s["ptable"]] == R["ptable"]
    assert [d2h(v) for v in s["ccmax"]] == R["ccmax"]
    assert sha(np.concatenate([np.ravel(a) for a in s["angles"]])) == R["angles_sha"]
    assert sha(np.concatenate([np.ravel(c) for c in s["cc_angles"]])) == R["cc_angles_sha"]


@needs_fix
@needs_fftw
@pytest.mark.parametrize("fast", [True, False], ids=["checked-fast-path", "exact-path"])
@pytest.mark.parametrize("shape", FFT_SHAPES, ids=lambda sh: "x".join(map(str, sh)))
def test_fft_sizes_as_matlab(shape, fast):
    """MATLAB's grouped 1-D FFTs (pydyn.masks._FFTW) on sizes that are not powers of two (and one non-cubic
    shape): fftn of a real array, ifftn of a symmetric spectrum and of a product spectrum, ifftn of a complex
    array. With fast=True the whole-volume plans are used only where the self-check proves them identical."""
    from pydyn import masks as M
    b = M._BACKEND["b"]
    keep = b.FAST
    b.FAST = fast
    try:
        a, q = fft_inputs(shape)
        F = MN.fftn(a)
        R = REF["fft_shapes"]["x".join(map(str, shape))]
        assert sha(np.concatenate([F.real.ravel(), F.imag.ravel()])) == R["fftn"]
        assert sha(MN.ifftn_real(F)) == R["ifftn_sym"]
        Y = MN.ifftn(q)
        assert sha(np.concatenate([Y.real.ravel(), Y.imag.ravel()])) == R["ifftn"]
        assert sha(MN.ifftn_real(MN.cmul(F, MN.conj(MN.fftn(a * a))))) == R["product_sym"]
    finally:
        b.FAST = keep


def sum_input(n):
    """The vector of benchmarks/15_fft_sizes/sums (only *, /, -, + and fmod, exact in IEEE)."""
    t = np.arange(n, dtype=np.float64)
    return ((t * 7919.0) % 10007.0) / 10007.0 - 0.37 + ((t * 104729.0) % 1009.0) * 1e-7


@needs_fix
def test_msum_threads_as_matlab():
    R = REF["msum_threads"]
    x = sum_input(max(R["n"]))
    for n, row in zip(R["n"], R["sum"]):
        for T, h in zip(R["threads"], row):
            assert MN.msum(x[:n], T) == h2d(h), (n, T)
    assert MN.msum(x[:100000], 16) != MN.msum(x[:100000], 1)               # the thread count does change the bits


def test_matlab_threads_override(monkeypatch):
    monkeypatch.setenv("PYDYN_MATLAB_THREADS", "64")
    assert MN.matlab_threads() == 64
    monkeypatch.delenv("PYDYN_MATLAB_THREADS")
    assert 1 <= MN.matlab_threads() <= (os.cpu_count() or 1)
    monkeypatch.setenv("PYDYN_MATLAB_THREADS", "0")
    with pytest.raises(ValueError):
        MN.matlab_threads()
