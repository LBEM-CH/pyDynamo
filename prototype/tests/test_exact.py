"""Exact engine mode (pydyn/dynamo_exact.py): Dynamo 1.1.514's own kernels, cuFFT 8 and cuBLAS 8.

CPU tests: host arithmetic transcribed from spp.cu (float rotation matrices, peak index helpers, engine selection)
and the PTX batching rewrite (needs the installed Dynamo 1.1.514 binary, skipped without it).
GPU tests (marker gpu): batched == one-orientation-at-a-time bit for bit, the batching primitives (cufftPlanMany,
cuBLAS v2 device pointer mode, batched kernels) against Dynamo's per-volume calls, and the refined table of the
official engine on a harness case (benchmarks/09_engine_parity/harness, A40 ground truth).
"""
import json
import os
import re

import numpy as np
import pytest

from pydyn import dynamo_exact as X
from pydyn import engine as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(ROOT, "benchmarks", "09_engine_parity", "harness")
HAVE_DYNAMO = os.path.exists(X.ENGINE_BINARY) and os.path.exists(os.path.join(X.CUDA8_LIB, "libcufft.so.8.0"))
needs_dynamo = pytest.mark.skipif(not HAVE_DYNAMO, reason="needs Dynamo 1.1.514 and the CUDA 8 libraries")

# (angles, h_rotmatrix) of Dynamo's own engine (dyntrace H2D copies, harness/runs/bin8_i1_trace_g0, particle 0)
DYNAMO_MATRICES = [
    ((90.0, 163.74000549316406, -66.70001220703125),
     ("-0x1.c36f7p-1 -0x1.9509d4p-2 -0x1.07556cp-2 -0x1.84d658p-2 0x1.d63e9ep-1 -0x1.c5a326p-4 0x1.1eb766p-2 "
      "0x1.3c4438p-56 -0x1.eb8542p-1")),
    ((90.0, 163.74000549316406, -51.70001220703125),
     ("-0x1.81bbd8p-1 -0x1.3d53a8p-1 -0x1.c2044p-3 -0x1.30a25p-1 0x1.91ce3ap-1 -0x1.6366cap-3 0x1.1eb766p-2 "
      "0x1.3c4438p-56 -0x1.eb8542p-1")),
    ((110.68787384033203, 171.7333526611328, -45.81358337402344),
     ("-0x1.d1f8ap-1 -0x1.9afd9p-2 -0x1.a64d88p-4 -0x1.915b88p-2 0x1.d43ae4p-1 -0x1.9a7a12p-4 0x1.13797ap-3 "
      "-0x1.a01b28p-5 -0x1.faae1ep-1")),
    ((107.28690338134766, 340.79779052734375, 101.53959655761719),
     ("-0x1.a5e7d8p-1 0x1.dd24bcp-2 -0x1.49fd2cp-2 -0x1.e2d944p-2 -0x1.c23f8cp-1 -0x1.0d7f86p-4 -0x1.41954cp-2 "
      "0x1.905356p-4 0x1.e383a4p-1")),
]


# ------------------------------------------------------------------------------------------------- CPU
def test_float_rotation_matrices_equal_dynamo_bits():
    ang = np.array([a for a, _ in DYNAMO_MATRICES], np.float32)
    want = np.array([[float.fromhex(h) for h in m.split()] for _, m in DYNAMO_MATRICES], np.float32)
    got = X.matrices_float(ang)
    assert np.array_equal(got.view(np.int32), want.view(np.int32))
    # the double-precision matrices of the fast engine differ in some entries (docs/analysis/12d section 7 item 3)
    assert not np.array_equal(E.A.rotation_matrices(ang).view(np.int32), want.view(np.int32))


@pytest.mark.parametrize("n", [32, 64, 96, 128])
def test_vectorised_peak_index_helpers_equal_spp_transcription(n):
    r = np.arange(n)
    s = np.arange(-2, n + 2)
    assert [X.rawpeak2space(int(v), n, n) for v in r] == X.rawpeak2space_v(r, n).tolist()
    assert [X.spacepeak2raw(int(v), n, n) for v in s] == X.spacepeak2raw_v(s, n).tolist()


def test_engine_mode_selection(monkeypatch):
    monkeypatch.delenv("PYDYN_ENGINE", raising=False)
    assert E.engine_mode() == E.DEFAULT_ENGINE
    assert E.engine_mode("fast") == "fast" and E.engine_mode("EXACT") == "exact"
    monkeypatch.setenv("PYDYN_ENGINE", "fast")
    assert E.engine_mode() == "fast"
    with pytest.raises(ValueError):
        E.engine_mode("quick")


_EXTRA = re.compile(r"^(mov\.u32 %bzr0, %ctaid\.z|cvt\.u64\.u32 %bzd0, %bzr0|mad\.lo\.s64 %rd\d+, %bzd0, %bzd\d+, %rd\d+"
                    r"|cvta\.to\.global\.u64 %bzd\d+, %bzd\d+|shl\.b64 %bzd\d+, %bzd0, 2|add\.s64 %bzd\d+, %bzd\d+, %bzd\d+"
                    r"|ld\.global\.f32 %f\d+, \[%bzd\d+\]|ld\.param\.u64 %bzd\d+, \[\w+\]);$")


@needs_dynamo
def test_batched_ptx_keeps_every_instruction_of_dynamo_kernels():
    """The batched copy of each kernel = Dynamo's instructions in Dynamo's order, plus only integer address
    arithmetic at the parameter loads (and, for per-volume scalars, a global load instead of a parameter load);
    threadIdx.x and blockIdx.x are exchanged (coalesced voxel order, dynamo_exact._swap_xz)."""
    txt = open(X.ptx_file()).read()
    assert X.hashlib.md5(txt.encode()).hexdigest() == X.PTX_MD5
    for key in X.BATCHED:
        name = X.KERNELS[key]
        pv = X.PER_VOLUME_FLOATS.get(key, ())
        orig = X._entry_text(txt, name).split("\n")
        bat, _ = X._batched_entry(txt, name, X.POINTER_PARAMS[key], pv)
        bat = [ln.replace(name + "_bz", name) for ln in bat.split("\n")]
        body = lambda ls: [ln for ln in ls if not ln.startswith((".param", ".reg", ".visible .entry"))]
        kept = [ln for ln in body(bat) if not _EXTRA.match(ln.strip())]
        pv_loads = [ln for ln in body(orig) if ln.startswith("ld.param.f32") and
                    any(f"_param_{k}]" in ln for k in pv)]
        assert len(pv_loads) == len(pv), key
        assert kept == [X._swap_xz(ln) for ln in body(orig) if ln not in pv_loads], key
        assert sum("%tid.x" in ln for ln in kept) == sum("%ctaid.x" in ln for ln in body(orig)), key


# ------------------------------------------------------------------------------------------------- GPU
class _Card:
    dim = sidelength = 32
    shrink = 1.0
    high, low = 2.0, 10.0
    use_CC, localnc = 1, 1
    area_search, area_search_modus = [4, 4, 2], 4
    cone_range, cone_sampling, cone_flip = 20.0, 10.0, 2
    inplane_range, inplane_sampling, inplane_flip = 40.0, 20.0, 2
    refine, refine_factor = 1, 2.0


def _blob(n=32, seed=0):
    rs = np.random.RandomState(seed)
    v = np.zeros((n, n, n), dtype=np.float32)
    z, y, x = np.ogrid[:n, :n, :n]
    for _ in range(6):
        c = rs.randint(10, 22, size=3)
        v += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / 6.0).astype(np.float32)
    return v


def _case(n=32, P=3):
    from pydyn.project import default_mask
    t = _blob(n)
    rs = np.random.RandomState(1)
    parts, rows = [], np.zeros((P, 35), dtype=np.float32)
    for p in range(P):
        sh = rs.randint(-2, 3, size=3)
        parts.append((np.roll(t, shift=(sh[2], sh[1], sh[0]), axis=(0, 1, 2))
                      + 0.3 * rs.standard_normal((n, n, n))).astype(np.float32))
        rows[p, 0] = p + 1
        rows[p, 3:6] = rs.uniform(-1, 1, 3)
        rows[p, 6:9] = rs.uniform(-30, 30, 3)
        rows[p, 12], rows[p, 15], rows[p, 16] = 2, -50, 50          # x-axis wedge (fsampling type 2)
    return t, default_mask(n), np.stack(parts), rows


@pytest.mark.gpu
@needs_dynamo
def test_exact_recovers_known_shift():
    from pydyn.project import default_mask
    n = 32
    t = _blob(n)
    shift = (2, -1, 3)
    p = np.roll(t, shift=(shift[2], shift[1], shift[0]), axis=(0, 1, 2))
    rows = np.zeros((1, 35), dtype=np.float32)
    rows[0, 0] = 1
    card = _Card()
    card.area_search_modus, card.cone_range, card.inplane_range, card.refine = 0, 0.0, 0.0, 0
    eng = E.Engine(t, default_mask(n), card, batch=4, exact=True)
    eng.prepare_particles(p[None], rows)
    out = eng.refine(rows)
    assert np.allclose(out[0, 3:6], shift, atol=0.3)
    assert out[0, 9] > 0.95


@pytest.mark.gpu
@needs_dynamo
@pytest.mark.parametrize("batch", [7, 64])
def test_exact_batched_is_bitwise_the_one_orientation_loop(batch):
    t, m, parts, rows = _case()
    res = {}
    for b in (1, batch):
        eng = E.Engine(t, m, _Card(), batch=b, exact=True)
        eng.trace = []
        eng.prepare_particles(parts, rows)
        res[b] = (eng.refine(rows), eng.trace, eng.orientation_counts)
    (o1, tr1, c1), (ob, trb, cb) = res[1], res[batch]
    assert c1 == cb and sum(c1) == len(tr1) == len(trb)
    assert np.array_equal(o1.view(np.int32), ob.view(np.int32))
    for (p1, r1), (p2, r2) in zip(tr1, trb):
        assert p1 == p2 and r1[0] == r2[0]
        assert np.float32(r1[3]).view(np.int32) == np.float32(r2[3]).view(np.int32) and r1[4] == r2[4]
        a = np.array(r1[5][:4] + tuple(r1[5][4]), np.float32)
        b_ = np.array(r2[5][:4] + tuple(r2[5][4]), np.float32)
        assert np.array_equal(a.view(np.int32), b_.view(np.int32))


@pytest.mark.gpu
@needs_dynamo
def test_batched_kernels_equal_dynamo_kernels_per_volume():
    import cupy as cp
    n, B = 32, 3
    N = n ** 3
    L = X._Libs.get()
    kb = L.batched()
    rs = np.random.RandomState(2)
    c = lambda: cp.asarray((rs.standard_normal((B, N)) + 1j * rs.standard_normal((B, N))).astype(np.complex64))
    f = lambda: cp.asarray(rs.standard_normal((B, N)).astype(np.float32))
    msk = cp.asarray((rs.random_sample((B, N)) < 0.4).astype(np.float32))
    C, R, Z, n32 = np.uint64(8 * N), np.uint64(4 * N), np.uint64(0), np.int32(n)
    g1, g, blk = (n, n), (n, n, B), (n,)
    shared = c()[0]
    mats = cp.asarray(X.matrices_float([(10, 20, 30), (-40, 100, 5), (170, 3, -60)]).ravel())
    vals = cp.asarray(np.array([0.3, -1.2, 2.5], np.float32))
    nnz = cp.asarray(np.array([5000.0, 7000.0, 9000.0], np.float32))
    cases = {
        "rotate": (lambda a, k: ((shared, a[0][k], shared, a[1][k], n32, mats[9 * k:9 * k + 9]),
                                 (shared, a[0], shared, a[1], n32, mats, Z, C, Z, C, np.uint64(36))), [c(), c()]),
        "bandpass": (lambda a, k: ((a[0][k], n32, np.float32(2), np.float32(10), np.float32(3)),
                                   (a[0], n32, np.float32(2), np.float32(10), np.float32(3), C)), [c()]),
        "wedge2": (lambda a, k: ((a[0][k], n32, np.float32(-50), np.float32(50)),
                                 (a[0], n32, np.float32(-50), np.float32(50), C)), [c()]),
        "normalize_fft": (lambda a, k: ((a[0][k], np.int64(n)), (a[0], np.int64(n), C)), [c()]),
        "c2f": (lambda a, k: ((a[0][k], a[1][k], n32), (a[0], a[1], n32, C, R)), [c(), f()]),
        "nf2c": (lambda a, k: ((a[0][k], a[1][k], msk[k], vals[k:k + 1].get()[0], n32),
                               (a[0], a[1], msk, vals, n32, C, R, R)), [c(), f()]),
        "mbc": (lambda a, k: ((a[0][k], shared, a[1][k], n32), (a[0], shared, a[1], n32, C, Z, C)), [c(), c()]),
        "roseman": (lambda a, k: ((a[0][k], a[1][k], a[2][k], n32, nnz[k:k + 1].get()[0], vals[k:k + 1].get()[0] + 3),
                                  (a[0], a[1], a[2], n32, nnz, vals + 3, C, C, C)), [c(), c(), c()]),
        "limits": (lambda a, k: ((a[0][k], np.float32(4), np.float32(4), np.float32(2), n32 // 2, n32 // 2 + 1,
                                  n32 // 2 - 1, n32, n32, mats[:9]),
                                 (a[0], np.float32(4), np.float32(4), np.float32(2), n32 // 2, n32 // 2 + 1,
                                  n32 // 2 - 1, n32, n32, mats[:9], C, Z)), [c()]),
        "plus_one": (lambda a, k: ((a[0][k], a[1][k], n32), (a[0], a[1], n32, C, R)), [c(), f()]),
    }
    for key, (args, bufs) in cases.items():
        ref = [b.copy() for b in bufs]
        for k in range(B):
            L.k[key](g1, blk, args(ref, k)[0])
        kb[key](g, blk, args(bufs, 0)[1])
        for a, r in zip(bufs, ref):
            assert bool((a.view(cp.int32) == r.view(cp.int32)).all()), key


@pytest.mark.gpu
@needs_dynamo
def test_cufft8_planmany_and_cublas8_v2_equal_dynamo_calls():
    import cupy as cp
    n, B = 32, 5
    N = n ** 3
    L = X._Libs.get()
    L.batched()
    rs = np.random.RandomState(3)
    h = (rs.standard_normal((B, N)) + 1j * rs.standard_normal((B, N))).astype(np.complex64)
    p1, pm = L.plan3d(n), L.plan_many(n, B)
    for d in (X.CUFFT_FORWARD, X.CUFFT_INVERSE):
        a, r = cp.asarray(h), cp.asarray(h)
        L.fft_ptr(pm, a.data.ptr, d)
        for k in range(B):
            L.fft_ptr(p1, r[k].data.ptr, d)
        assert bool((a.view(cp.int32) == r.view(cp.int32)).all())
    for p in (p1, pm):
        L.cufft.cufftDestroy(p)
    x = cp.asarray(rs.standard_normal((B, N)).astype(np.float32))
    y = cp.asarray((rs.random_sample((B, N)) < 0.3).astype(np.float32))
    out = cp.zeros(3 * B, cp.float32)
    pos = cp.zeros(B, cp.int32)
    for k in range(B):
        L.dot_dev(N, x[k].data.ptr, y[k].data.ptr, out.data.ptr + 4 * k)
        L.asum_dev(N, y[k].data.ptr, out.data.ptr + 4 * (B + k))
        L.dot_dev(N, x[k].data.ptr, x[k].data.ptr, out.data.ptr + 4 * (2 * B + k))
        L.iamax_dev(N, x[k].data.ptr, pos.data.ptr + 4 * k)
    o = out.get()
    for k in range(B):
        assert o[k] == L.dot(x[k], y[k]) and o[B + k] == L.asum(y[k]) and o[2 * B + k] == L.dot(x[k], x[k])
        assert int(pos[k].get()) == L.iamax(x[k])


def _harness_case(name):
    case_dir = os.path.join(HARNESS, "cases", "work", name)
    gt = os.path.join(HARNESS, "runs", f"{name}_plain_g0")
    gpu = os.path.join(gt, "gpu.txt")
    if not (os.path.exists(os.path.join(case_dir, "case.json")) and os.path.isdir(gt)):
        pytest.skip("harness ground truth not present")
    return case_dir, gt, open(gpu).read() if os.path.exists(gpu) else ""


@pytest.mark.gpu
@needs_dynamo
@pytest.mark.parametrize("batch", [1, 128])
def test_exact_engine_table_equals_official_engine_bitwise(batch):
    """bin8_i1 harness case (4 particles, 8,331 orientations): the refined rows equal the official binary's own
    refined table (A40, driver JIT) bit for bit. Valid only on the GPU model of the ground truth. The one-at-a-time
    loop (batch 1) runs on the first two particles only, to keep the test short."""
    import cupy as cp
    from pydyn import emio
    from pydyn.card import Card
    case_dir, gt, gpu = _harness_case("bin8_i1")
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()
    if dev not in gpu:
        pytest.skip(f"ground truth was produced on another GPU model ({dev} here)")
    case = json.load(open(os.path.join(case_dir, "case.json")))
    cwd = os.getcwd()
    os.chdir(case["cwd"])
    try:
        card = Card(case["card"])
        with open(card.particle_list(0)) as fh:
            paths = [ln.strip() for ln in fh if len(ln.strip()) > 4]
        P = len(paths) if batch > 1 else 2
        rows = emio.read_btbl(card.starting_table(0))[:P]
        parts = np.stack([emio.read_em(p).astype(np.float32) for p in paths[:P]])
        eng = E.Engine(emio.read_em(card.file_template).astype(np.float32),
                       emio.read_em(card.file_mask).astype(np.float32), card, batch=batch, exact=True)
        eng.prepare_particles(parts, rows)
        out = eng.refine(rows).astype(np.float32)
        ref = [f for f in os.listdir(gt) if f.startswith("refined_") and f.endswith(".btbl")][0]
        want = emio.read_btbl(os.path.join(gt, ref)).astype(np.float32)[:P]
    finally:
        os.chdir(cwd)
    assert np.array_equal(out.view(np.int32), want.view(np.int32))


@pytest.mark.gpu
@needs_dynamo
def test_sdot_order_reproduces_cublas8_on_this_gpu():
    import cupy as cp
    L = X._Libs.get()
    sms = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["multiProcessorCount"]
    so = X.SdotOrder(sms, L, n_check=32 ** 3 * 8)       # checks 4 vectors at construction
    rs = np.random.RandomState(5)
    for n in (64 ** 3, 96 ** 3):
        x = cp.asarray(rs.standard_normal(n).astype(np.float32))
        y = cp.asarray(rs.standard_normal(n).astype(np.float32))
        out = cp.zeros(1, cp.float32)
        so.dot_dev(n, x.data.ptr, y.data.ptr, out.data.ptr)
        assert out.get()[0].view(np.int32) == L.dot(x, y).view(np.int32) == so.dot(x, y).view(np.int32)


def _run_harness_case(name, batch, **kw):
    import cupy as cp  # noqa: F401
    from pydyn import emio
    from pydyn.card import Card
    case_dir = os.path.join(HARNESS, "cases", "work", name)
    if not os.path.exists(os.path.join(case_dir, "case.json")):
        pytest.skip("harness case not present")
    case = json.load(open(os.path.join(case_dir, "case.json")))
    cwd = os.getcwd()
    os.chdir(case["cwd"])
    try:
        card = Card(case["card"])
        with open(card.particle_list(0)) as fh:
            paths = [ln.strip() for ln in fh if len(ln.strip()) > 4]
        rows = emio.read_btbl(card.starting_table(0))[:len(paths)]
        parts = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
        eng = E.Engine(emio.read_em(card.file_template).astype(np.float32),
                       emio.read_em(card.file_mask).astype(np.float32), card, batch=batch, exact=True, **kw)
        eng.prepare_particles(parts, rows)
        return case, eng.refine(rows).astype(np.float32)
    finally:
        os.chdir(cwd)


@pytest.mark.gpu
@needs_dynamo
def test_exact_with_a5000_reduction_order_reproduces_the_users_a5000_table():
    """The user's bin8_align_1 iteration 1 ran on RTX A5000s (64 SMs). With cuBLAS 8's reduction order of a 64-SM GPU
    the exact engine gives the user's refined rows at the precision of the text table (5 significant digits) for
    all four bin8_i1 tags, including 6573, where Dynamo itself on an A40 chooses another orientation."""
    from pydyn import dynio
    case, out = _run_harness_case("bin8_i1", 128, exact_sms=64)
    if not os.path.exists(case["user_refined_table"]):
        pytest.skip("user's table not present")
    u = np.real(dynio.read_tbl(case["user_refined_table"]))
    idx = {int(t): i for i, t in enumerate(u[:, 0])}
    txt = lambda v: [dynio.dlm_field(float(a)) for a in v]
    for r in out:
        assert txt(r[3:10]) == txt(u[idx[int(r[0])], 3:10]), int(r[0])
