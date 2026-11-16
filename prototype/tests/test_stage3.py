"""Stage 3: console functions and the dcp project life cycle (docs/analysis/08a, 08b).

CPU tests run on small synthetic data and compare with Dynamo 1.1.514's own output kept as small text fixtures in
tests/fixtures/stage3 (vpr field dumps, cards, .exe, .sel of benchmarks/04_journey_bin8_align_1/dynamo/dcpnew, runs A and B).
Tests marked 'reference' compare with the full headless-Dynamo ground truth and skip when it is absent (CI).
"""
import filecmp
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import CHECK_CLEAN

from pydyn import dynio
from pydyn import vpr as V

P_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "stage3")
P = os.path.abspath(os.path.join(HERE, "..", ".."))
REF = os.path.join(P, "benchmarks", "04_journey_bin8_align_1", "dynamo")
TD = os.path.join(P, "test_data", "bin8_filament")
SRC = "/programs/x86_64-linux/dynamo/1.1.514/matlab/src"


# ------------------------------------------------------------------------------------------------ synthetic data
def make_data(root, n=64, N=3, pad=6, seed=1):
    """A Dynamo data folder (particle_<tag>.em), a 42-column table and a template, named like the user's files."""
    rs = np.random.RandomState(seed)
    d = os.path.join(root, "filamentsData_ctf")
    os.makedirs(d)
    for t in range(1, N + 1):
        dynio.write_em(os.path.join(d, "particle_%0*d.em" % (pad, t)), rs.randn(n, n, n).astype(np.float32))
    T = np.zeros((N, 42))
    T[:, 0] = np.arange(1, N + 1)
    T[:, 1] = 1
    T[:, 12] = 1
    T[:, 13], T[:, 14] = -60, 60
    T[:, 6:9] = rs.uniform(0, 360, (N, 3))
    dynio.write_tbl(os.path.join(root, "particles_edit_mod.tbl"), T)
    dynio.write_em(os.path.join(root, "raw_template_ctf.em"), rs.randn(n, n, n).astype(np.float32))
    return T


def _json_form(path):
    """The vpr2json.py form (name, mclass, dims, value) of a virtual_project.mat written by pydyn."""
    out = []
    for name, cls, dims, v in V.dump_vpr(path):
        if isinstance(v, V.MatCell):
            val = None
        elif isinstance(v, str):
            val = v
        else:
            a = np.atleast_1d(np.asarray(v, dtype=float))
            val = [None if np.isnan(x) else (int(x) if float(x).is_integer() else float(x)) for x in a]
            if len(val) == 1:
                val = val[0]
        out.append(dict(name=name, mclass=cls, dims=list(dims), value=val))
    return out


def _compare_json(mine, ref):
    assert [x["name"] for x in mine] == [x["name"] for x in ref]
    for a, b in zip(mine, ref):
        if a["name"] == "path":
            assert a["mclass"] == b["mclass"] == "char"
            continue
        assert (a["name"], a["mclass"], a["dims"]) == (b["name"], b["mclass"], b["dims"])
        if a["mclass"] != "cell":
            assert a["value"] == b["value"], a["name"]


# ------------------------------------------------------------------------------------------------ convention
def test_convention_table_counts():
    from pydyn import convention as c
    assert len(c.NC) == 44 and len(c.RC) == 53
    assert [p["name"] for p in c.NC[:2]] == ["name_project", "path"]
    assert V.HIDDEN_ROUND_NAMES[:2] == ["cone_check_peak", "cone_freeze_reference"]


@pytest.mark.skipif(not os.path.isfile(os.path.join(SRC, "dynamo_vpr_convention.m")), reason="Dynamo source absent")
def test_convention_table_matches_source():
    from pydyn import convention as c
    from pydyn.convention_parse import parse_convention
    nc, rc = parse_convention(os.path.join(SRC, "dynamo_vpr_convention.m"))
    assert nc == c.NC and rc == c.RC


@pytest.mark.skipif(not os.path.isfile(os.path.join(REF, "dcpnew", "convention_1.1.514.mat")),
                    reason="headless Dynamo convention dump absent")
def test_convention_defaults_match_matlab_dump():
    import scipy.io as sio
    from pydyn import convention as c
    m = sio.loadmat(os.path.join(REF, "dcpnew", "convention_1.1.514.mat"), squeeze_me=True)
    assert list(m["ncn"]) == [p["name"] for p in c.NC]
    assert list(m["rcn"]) == [p["name"] for p in c.RC]
    for p, v in zip(c.NC + c.RC, list(m["ncv"]) + list(m["rcv"])):
        mine = V.default_value(p)
        if isinstance(mine, str):
            assert (str(v) if np.size(v) else "") == mine, p["name"]
        elif isinstance(mine, bool):
            assert bool(v) == mine, p["name"]
        else:
            assert V.values_equal(as_float(v), mine), p["name"]


def as_float(v):
    a = np.atleast_1d(np.asarray(v, dtype=float))
    return float(a[0]) if a.size == 1 else a


# ------------------------------------------------------------------------------------------------ vpr file
def test_vpr_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    v = V.blank("p")
    v = V.modify(v, "ite_r1", 2, "gpu_identifier_set", [0, 1, 2, 3], "area_search_r2", [4, 4, 2])
    V.save(v)
    f = V.vpr_location("p")
    w = V.read_vpr_file(f)
    assert list(w) == list(v)
    assert all(V.values_equal(v[k], w[k]) for k in v)
    meta = dict(V.mat_fields_meta(f))
    assert meta["options"]["class"] == "char" and meta["options"]["dims"] == [0, 0]
    assert meta["useSmoothingMaskForFourierCompensation"]["class"] == "uint8 logical"
    assert meta["gpu_identifier_set"]["dims"] == [1, 4]
    assert meta["area_search_r2"]["dims"] == [1, 3]
    assert meta["convergence_all"]["class"] == "cell" and meta["convergence_all"]["dims"] == [1, 8]
    assert len(v) == 468 + 20


def test_modify_errors_and_round_general():
    v = V.blank("p")
    with pytest.raises(V.DynamoError):
        V.modify(v, "ite_r1", "2")                   # type 1 must be numeric (dynamo_vpr_modify.m:419-446)
    with pytest.raises(V.DynamoError):
        V.modify(v, "no_such_parameter", 1)
    with pytest.raises(V.DynamoError):
        V.modify(v, "ite", 2)                        # generic round names need a cell
    w = V.modify(v, "ite", V.MatCell([2, 3]))
    assert (w["ite_r1"], w["ite_r2"], w["ite_r3"]) == (2.0, 3.0, 0.0)


# ------------------------------------------------------------------------------------------------ dcp.new
def _dcp_new(name="bin8_align_1"):
    from pydyn import dcp
    return dcp.new(name, "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default",
                   "t", "particles_edit_mod.tbl", "gui", 0)


def test_dcp_new_matches_dynamo_run_a(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _dcp_new()
    f = os.path.join("bin8_align_1", "settings", "virtual_project.mat")
    _compare_json(_json_form(f), json.load(open(os.path.join(FIX, "dynamo_vpr_after_dcpnew.json"))))
    s = os.path.join("bin8_align_1", "settings")
    from pydyn.project import default_mask
    assert open(os.path.join(s, "mask.em"), "rb").read() == dynio.em_bytes(default_mask(64))
    assert open(os.path.join(s, "mask_classification.em"), "rb").read() == dynio.em_bytes(default_mask(64))
    assert open(os.path.join(s, "fmask_initial_ref_001.em"), "rb").read() == dynio.em_bytes(np.full((64,) * 3, 64.))
    assert open(os.path.join(s, "mask_smoothing.em"), "rb").read() == dynio.em_bytes(np.ones((64,) * 3))
    assert open(os.path.join(s, "multisettings_fmask_initial.sel")).read() == \
        "./bin8_align_1/settings/fmask_initial_ref_001.em 1\n"
    assert sorted(os.listdir(".")) == ["bin8_align_1", "filamentsData_ctf", "particles_edit_mod.tbl",
                                       "raw_template_ctf.em"]
    # existing project: message and return, nothing changed
    before = open(f, "rb").read()
    assert _dcp_new() is None
    assert open(f, "rb").read() == before


def test_dvcheck_and_dvunfold_fresh_project(tmp_path, monkeypatch):
    from pydyn import dvcheck, dvunfold
    monkeypatch.chdir(tmp_path)
    T = make_data(str(tmp_path))
    _dcp_new()
    r = dvcheck("bin8_align_1")
    assert r.report == ['Apparently all the rounds are empty: no iterations are executed. Check the "ite" '
                        'parameter']
    assert (r.fields_warning, r.fields_error, r.warning_state) == (["ite_r1"], [], "warning")
    name_script, status = dvunfold("bin8_align_1")      # run A: stops after settings2first_references
    assert (name_script, status) == (None, "error")
    assert open("bin8_align_1/next_iteration.txt").read() == "1\n"
    T2 = dynio.read_tbl("bin8_align_1/settings/table_initial_ref_001.tbl")
    assert np.all(T2[:, 33] == 1) and np.array_equal(np.delete(T2, 33, 1), np.delete(dynio.read_tbl(
        "particles_edit_mod.tbl"), 33, 1))
    assert np.all(dynio.read_tbl("particles_edit_mod.tbl")[:, 33] == 0)
    assert not os.path.exists("bin8_align_1/cards")
    v = V.load("bin8_align_1")
    assert list(v)[-8:] == V.FEATURE_FIELDS and v["feature_data_padding"] == 6 and v["feature_data_sidelength"] == 64
    assert np.isnan(v["feature_data_batch"]) and v["feature_data_type"] == "dynamo_folder"


def _journey(name="bin4_align_1"):
    """08a section 8 (run B), with a synthetic editorMask."""
    from pydyn import dvput
    _dcp_new(name)
    dynio.write_em("%s/settings/editorMask.em" % name, np.ones((64,) * 3) * 0.5)
    dynio.write_em("%s/settings/fmask_initial_ref_001.em" % name, np.ones((64,) * 3))
    em = "%s/settings/editorMask.em" % name
    dvput(name, "d", "file_mask", em, "file_mask_classification", em, "file_mask_smoothing", em,
          "destination", "standalone_gpu", "gpu_identifier_set", [0, 1, 2, 3], "matlab_workers_average", 50)
    pairs = []
    for r in (1, 2, 3, 4):
        pairs += ["cone_range_r%d" % r, 20, "cone_sampling_r%d" % r, 3, "cone_flip_r%d" % r, 1, "refine_r%d" % r, 2,
                  "low_r%d" % r, 25, "dim_r%d" % r, 64, "area_search_r%d" % r, [4, 4, 2],
                  "area_search_modus_r%d" % r, 4]
    dvput(name, "d", "ite", [2, 2], *pairs)          # 'ite' [2 2] expands to ite_r1, ite_r2 (dynamo_vpr_put.m:433)
    dvput(name, "d", "inplane_range_r2", 45, "inplane_range_r3", 90, "inplane_range_r4", 45,
          "inplane_sampling_r1", 15, "inplane_sampling_r2", 5, "inplane_sampling_r3", 15, "inplane_sampling_r4", 5)
    v = V.load(name)
    V.save({k: v[k] for k in V.blank(name) if k in v})      # GUI-style save (@dProject/getVpr.m)


def test_journey_unfold_matches_dynamo_run_b(tmp_path, monkeypatch):
    from pydyn import dvcheck, dvunfold
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _journey()
    r = dvcheck("bin4_align_1")
    assert (r.report, r.fields_warning, r.fields_error, r.warning_state) == CHECK_CLEAN
    script, status = dvunfold("bin4_align_1")
    assert (script, status) == ("bin4_align_1.exe", "ok")
    _compare_json(_json_form("bin4_align_1/settings/virtual_project.mat"),
                  json.load(open(os.path.join(FIX, "dynamo_vpr_journey_after_unfold.json"))))
    for ite in range(1, 5):
        mine = open("bin4_align_1/cards/ite_%04d/card_ite_ite_%04d.card" % (ite, ite)).read().splitlines()
        ref = open(os.path.join(FIX, "dynamo_journey_card_ite_ite_%04d.card" % ite)).read().splitlines()
        assert len(mine) == len(ref)
        assert [a for a, b in zip(mine, ref) if a != b] == [x for x in mine if x.startswith("path=")]
    assert open("bin4_align_1/settings/multisettings_fmask_initial.sel").read() == \
        open(os.path.join(FIX, "dynamo_journey_multisettings_fmask_initial.sel")).read()
    exe = open(script).read().splitlines()
    ref = open(os.path.join(FIX, "dynamo_bin4_align_1.exe")).read().splitlines()
    assert ref[5] == "dynamo_execute_project bin4_align_1"
    assert exe == ref and open(script, "rb").read() == open(os.path.join(FIX, "dynamo_bin4_align_1.exe"), "rb").read()
    assert os.stat(script).st_mode & 0o700 == 0o700
    assert os.path.isdir("bin4_align_1/rescaled_data/data_64")
    sv = "bin4_align_1/results/ite_0001/starting_values/"
    assert filecmp.cmp(sv + "starting_reference_ref_001_ite_0001.em", "raw_template_ctf.em", shallow=False)
    assert filecmp.cmp(sv + "starting_table_ref_001_ite_0001.tbl", "bin4_align_1/settings/table_initial_ref_001.tbl",
                       shallow=False)
    # mtime kept by the copy (MATLAB copyfile), 08a 6.1
    assert os.stat("bin4_align_1/settings/mask.em").st_mtime == os.stat("bin4_align_1/settings/editorMask.em").st_mtime
    # pydynamo info / check through the command line
    env = dict(os.environ, PYTHONPATH=os.path.join(P, "prototype"))
    out = subprocess.run([sys.executable, "-m", "pydyn", "info", "bin4_align_1"], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 0 and "[1, 1, 2, 2]" in out.stdout


def test_dvput_rejects_unknown_and_saves_into_path(tmp_path, monkeypatch):
    from pydyn import dvput
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _dcp_new("p")
    with pytest.raises(V.DynamoError):
        dvput("p", "d", "cone_rangeX", 3)
    v = V.load("p")
    v["path"] = str(tmp_path / "elsewhere")
    V.save(v)
    with pytest.raises(V.DynamoError):              # Dynamo would create elsewhere/p/settings (08a 5.2)
        dvput("p", "d", "cr", 30)
    assert not os.path.exists(tmp_path / "elsewhere")


def test_unfold_copies_symlink_content(tmp_path, monkeypatch):
    """Deviation 08a 6.1: a symlinked seed is copied by content, so column 34 is never written through the link."""
    from pydyn import dvput, dvunfold
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    os.rename("particles_edit_mod.tbl", "real.tbl")
    os.symlink("real.tbl", "particles_edit_mod.tbl")
    before = open("real.tbl", "rb").read()
    _dcp_new("p")
    dvput("p", "d", "ite_r1", 1, "destination", "standalone_gpu", "dim_r1", 64)
    assert dvunfold("p")[1] == "ok"
    assert open("real.tbl", "rb").read() == before
    assert not os.path.islink("p/settings/table_initial_ref_001.tbl")


def test_inplace_dynamoproject_initialize_writes_nothing(tmp_path, monkeypatch):
    from pydyn import dvput, dvunfold
    from pydyn.dynproject import DynamoProject
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _dcp_new("p")
    dvput("p", "d", "ite_r1", 1, "destination", "standalone_gpu", "dim_r1", 64)
    dvunfold("p")
    root = str(tmp_path / "p")
    listing = sorted(os.listdir(root))
    pr = DynamoProject(root, str(tmp_path / "filamentsData_ctf"), root, echo=False)
    pr.initialize()
    assert sorted(os.listdir(root)) == listing      # no log.txt, no temp/, nothing copied
    os.remove(os.path.join(root, "next_iteration.txt"))
    with pytest.raises(FileNotFoundError):
        pr.initialize()


# ------------------------------------------------------------------------------------------------ console
def test_dtrandomize_azimuth_stream(tmp_path):
    from pydyn import dtrandomize_azimuth, rng
    rng("default")
    T = np.arange(5 * 42, dtype=float).reshape(5, 42)
    T2 = dtrandomize_azimuth(T)
    T3 = dtrandomize_azimuth(T, "o", str(tmp_path / "x.tbl"))
    rs = np.random.RandomState(5489)
    assert np.array_equal(T2[:, 8], 360 * rs.random_sample(5))
    assert np.array_equal(T3[:, 8], 360 * rs.random_sample(5))       # the stream continues
    assert np.array_equal(np.delete(T2, 8, 1), np.delete(T, 8, 1))
    assert open(tmp_path / "x.tbl", "rb").read() == dynio.tbl_bytes(T3)
    with pytest.raises(NotImplementedError):
        rng(7)


@pytest.mark.skipif(not os.path.isfile(os.path.join(REF, "console", "work", "gt_particles_edit_mod.tbl"))
                    or not os.path.isfile(os.path.join(TD, "particles_edit.tbl")), reason="ground truth absent")
def test_dtrandomize_azimuth_reference(tmp_path):
    from pydyn import dread, dwrite, dtrandomize_azimuth, rng
    rng("default")
    dwrite(dtrandomize_azimuth(dread(os.path.join(TD, "particles_edit.tbl"))), str(tmp_path / "m.tbl"))
    assert filecmp.cmp(str(tmp_path / "m.tbl"), os.path.join(REF, "console", "work", "gt_particles_edit_mod.tbl"),
                       shallow=False)


def test_dread_dwrite_roundtrips(tmp_path):
    from pydyn import dread, dwrite
    from pydyn.console import mrc_bytes
    T = np.array([[1, 2.5, 163.73979529168298], [99999, 0, -63.99]])
    dwrite(T, str(tmp_path / "a" / "t.tbl"))                         # creates the parent folder
    assert np.array_equal(dread(str(tmp_path / "a" / "t.tbl")), np.array([[1, 2.5, 163.74], [99999, 0, -63.99]]))
    v = np.random.RandomState(0).randn(4, 5, 6).astype(np.float32)
    dwrite(v, str(tmp_path / "v.em"))
    assert dread(str(tmp_path / "v.em")).dtype == np.float64 and np.array_equal(dread(str(tmp_path / "v.em")), v)
    u = np.arange(16, dtype=np.uint8).reshape(4, 4)
    dwrite(u, str(tmp_path / "u.em"))
    assert open(tmp_path / "u.em", "rb").read()[3] == 1 and dread(str(tmp_path / "u.em")).dtype == np.uint8
    dwrite(v, str(tmp_path / "v.mrc"))
    b = open(tmp_path / "v.mrc", "rb").read()
    h = np.frombuffer(b[:1024], "<i4")
    assert {i + 1: int(x) for i, x in enumerate(h) if x} == {1: 6, 2: 5, 3: 4, 4: 2, 8: 1, 9: 1, 10: 1, 17: 1, 18: 2,
                                                              19: 3}
    assert b == mrc_bytes(v) and np.array_equal(dread(str(tmp_path / "v.mrc")), v)
    dwrite(np.array([[1.0, 2.123456789123]]), str(tmp_path / "x.txt"))
    assert open(tmp_path / "x.txt").read() == "1 2.12345679\n"
    dwrite(["a", "b"], str(tmp_path / "l.sh"))
    assert dread(str(tmp_path / "l.sh")) == ["a", "b"]
    with pytest.raises(V.DynamoError):
        dread(str(tmp_path / "missing.em"))
    assert dread(str(tmp_path / "missing.em"), 1) is None
    with pytest.raises(NotImplementedError):
        dwrite(v, str(tmp_path / "v.spi"))


@pytest.mark.skipif(not os.path.isfile(os.path.join(REF, "console", "work", "gt_p1.mrc")), reason="ground truth absent")
def test_dwrite_reference_files(tmp_path):
    from pydyn import dread, dwrite
    W = os.path.join(REF, "console", "work")
    p = dread(os.path.join(W, "gt_p1.em"))
    dwrite(p, str(tmp_path / "p1.mrc"))
    assert filecmp.cmp(str(tmp_path / "p1.mrc"), os.path.join(W, "gt_p1.mrc"), shallow=False)
    dwrite(dread(os.path.join(W, "gt_rand_double.mrc")), str(tmp_path / "r.mrc"))
    assert filecmp.cmp(str(tmp_path / "r.mrc"), os.path.join(W, "gt_rand_double.mrc"), shallow=False)
    if os.path.isfile(os.path.join(TD, "particles_edit.tbl")):       # a_randomize_io.m lines 6, 13, 20-27
        T = dread(os.path.join(TD, "particles_edit.tbl"))
        dwrite(T, str(tmp_path / "rw.tbl"))
        assert filecmp.cmp(str(tmp_path / "rw.tbl"), os.path.join(W, "gt_particles_edit_rewrite.tbl"), shallow=False)
        B = T[:3].copy()
        B[:, 0] = [100000, 100001, 100002]
        B[0, 3], B[1, 4] = 1.25, -0.5
        dwrite(B, str(tmp_path / "big.tbl"))
        assert filecmp.cmp(str(tmp_path / "big.tbl"), os.path.join(W, "gt_bigtags.tbl"), shallow=False)


def test_daverage_cpu_identity(tmp_path, monkeypatch):
    """Zero angles and shifts: the mex rotation is the identity inside [0, n-2] and NaN on the last planes."""
    from pydyn import daverage
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), n=16, N=4)
    T = dynio.read_tbl("particles_edit_mod.tbl")
    T[:, 6:9] = 0
    dynio.write_tbl("t0.tbl", T)
    oa = daverage("filamentsData_ctf", "t", "t0.tbl", "mw", 50, device="cpu")
    ps = [dynio.read_em("filamentsData_ctf/particle_%06d.em" % t) for t in range(1, 5)]
    s = ps[0] + ps[1] + ps[2] + ps[3]
    inner = (slice(0, 15),) * 3
    assert np.array_equal(oa.average[inner], (s / 4)[inner])
    assert np.all(oa.nanContributions[15, :, :] == 4) and np.all(oa.average[15] == 0)
    assert oa.average is not oa.average_raw and np.array_equal(oa.average, oa.average_raw)
    assert oa.A.shape == (3,) and oa.dtype == "average" and np.all(oa.fmask == 1)
    with pytest.raises(NotImplementedError):
        daverage("filamentsData_ctf", "t", "t0.tbl", "fsc", 1, device="cpu")


@pytest.mark.gpu
def test_daverage_gpu_equals_cpu(tmp_path, monkeypatch):
    from pydyn import daverage
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), n=32, N=5)
    T = dynio.read_tbl("particles_edit_mod.tbl")
    T[:, 3:6] = np.random.RandomState(3).uniform(-2, 2, (5, 3))
    dynio.write_tbl("t.tbl", T)
    a = daverage("filamentsData_ctf", "t", "t.tbl", device="cpu")
    b = daverage("filamentsData_ctf", "t", "t.tbl", device="gpu")
    assert np.array_equal(a.average, b.average) and np.array_equal(a.nanContributions, b.nanContributions)


@pytest.mark.gpu
@pytest.mark.skipif(not os.path.isfile(os.path.join(REF, "console", "work", "gt_raw_template_ctf_mw0.em")),
                    reason="ground truth absent")
def test_daverage_reference_bytes(tmp_path):
    from pydyn import daverage, dwrite
    W = os.path.join(REF, "console", "work")
    oa = daverage(os.path.join(TD, "filamentsData_ctf"), "t", os.path.join(W, "orig_particles_edit_mod.tbl"), "mw", 50)
    dwrite(oa.average, str(tmp_path / "raw_template_ctf.em"))
    assert filecmp.cmp(str(tmp_path / "raw_template_ctf.em"), os.path.join(TD, "raw_template_ctf.em"), shallow=False)
    assert filecmp.cmp(str(tmp_path / "raw_template_ctf.em"), os.path.join(W, "gt_raw_template_ctf_mw0.em"),
                       shallow=False)


@pytest.mark.gpu
def test_exe_runs_one_iteration_in_place(tmp_path, monkeypatch):
    """dcp.new -> dvput -> dvunfold -> ./name.exe: one iteration in pure Python on GPU 0, in place."""
    from pydyn import dvput, dvunfold
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), n=32, N=8)
    _dcp_new("p")
    dvput("p", "d", "ite_r1", 1, "destination", "standalone_gpu", "gpu_identifier_set", 0, "dim_r1", 32,
          "cone_range_r1", 30, "cone_sampling_r1", 15, "inplane_range_r1", 30, "inplane_sampling_r1", 15,
          "refine_r1", 1, "low_r1", 12)
    script, status = dvunfold("p")
    assert status == "ok"
    # the .exe is Dynamo's file (`dynamo_execute_project p`); with pyDynamo's environment active the console script
    # pydyn.cli:execute_main answers. Stand-in for it (same entry point), so the test does not need the pip install:
    shim = tmp_path / "shimbin"
    shim.mkdir()
    (shim / "dynamo_execute_project").write_text(
        "#!/bin/sh\nexec %s -I -c \"import sys; sys.path.insert(0, %r); from pydyn.cli import execute_main; "
        "sys.exit(execute_main())\" \"$@\"\n" % (sys.executable, os.path.join(P_ROOT, "prototype")))
    (shim / "dynamo_execute_project").chmod(0o755)
    out = subprocess.run(["bash", script], capture_output=True, text=True, timeout=900,
                         env={**os.environ, "PATH": "%s:%s" % (shim, os.environ["PATH"])})
    assert out.returncode == 0, out.stdout + out.stderr
    assert open("p/next_iteration.txt").read() == "2\n"
    for k in ("average", "fmask_average", "refined_table"):
        assert os.path.isfile("p/results/ite_0001/averages/%s_ref_001_ite_0001.%s" % (k, "tbl" if k[0] == "r"
                                                                                       else "em"))


# ------------------------------------------------------------------------------------------------ Stage 3 review
def _touch_particles(d, names):
    os.makedirs(d)
    for x in names:
        dynio.write_em(os.path.join(d, "particle_%s.em" % x), np.ones((4, 4, 4), np.float32))


def test_data_info_padding_is_fixed_by_the_first_match(tmp_path):
    """dynamo_info_data.m:385-418; expected values from headless Dynamo 1.1.514
    (benchmarks/04_journey_bin8_align_1/checks/review/reference/logs/e_review_checks.log)."""
    cases = {"mixed": (["000001", "000002", "00003", "0000004", "abc"], 7, [4]),
             "stray": (["000001", "000002", "zz"], 6, [1, 2]),
             "five_first": (["00009", "000010"], 6, [10])}
    for name, (files, pad, tags) in cases.items():
        d = str(tmp_path / name)
        _touch_particles(d, files)
        f = V.data_info(d)
        assert (f["padding"], f["tags"], f["N"]) == (pad, tags, len(tags)), name


def test_dread_missing_and_folder_as_dynamo(tmp_path, capsys):
    """dynamo_read.m:212-266 [RUN e_review_checks.log]: the existence check comes before the extension dispatch."""
    from pydyn import dread
    with pytest.raises(V.DynamoError, match=r'cannot find requested file "%s"' % (tmp_path / "zz")):
        dread(str(tmp_path / "zz"))
    assert "Missing file!!!" in capsys.readouterr().out
    assert dread(str(tmp_path / "zz"), 1) is None
    os.makedirs(tmp_path / "folder")
    with pytest.raises(V.DynamoError, match="not assigned during call"):
        dread(str(tmp_path / "folder"))
    assert "no policy defined for this type of folder" in capsys.readouterr().out


def test_daverage_rejects_repeated_tags(tmp_path, monkeypatch):
    from pydyn import daverage
    monkeypatch.chdir(tmp_path)
    T = make_data(str(tmp_path), n=8, N=3)
    dynio.write_tbl("rep.tbl", np.concatenate([T, T[:1]]))
    with pytest.raises(NotImplementedError, match="repeated"):
        daverage("filamentsData_ctf", "t", "rep.tbl", device="cpu")


def test_links2settings_runs_both_blocks(tmp_path, monkeypatch, capsys):
    """dynamo_pipeline_links2settings.m:93-112: a missing mask does not stop the template/table/fmask copies."""
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _dcp_new("p")
    v = V.load("p")
    v["file_mask"] = "missing_mask.em"
    v["file_template_initial"] = "missing_template.em"
    err = V._links2settings(v)
    out = capsys.readouterr().out
    assert err == sorted(["file_mask", "mask", "file_template_initial", "template_initial"])
    assert "missing_mask.em" in out and "missing_template.em" in out


def test_dvput_string_values_strict(tmp_path, monkeypatch):
    from pydyn import dvput
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    _dcp_new("p")
    assert V._str2num_ifnum("4 4 2").tolist() == [4, 4, 2] and V._str2num_ifnum("[]") == "[]"
    assert V._str2num_ifnum("standalone_gpu") == "standalone_gpu" and V._str2num_ifnum("./a/b.em") == "./a/b.em"
    assert V._str2num_ifnum("1e3") == 1000.0 and V._str2num_ifnum("true") is True
    for s in ("4;4;2", "pi", "2*i", "a|b", "1.5:3"):
        with pytest.raises(NotImplementedError):
            V._str2num_ifnum(s)
    with pytest.raises(V.DynamoError, match="Invalid parameter file"):
        dvput("p", "missing.card", "cr", 3)


def test_setfromvpr_missing_fields_keep_class_value(capsys):
    """@dProject/setFromVpr.m: a missing field keeps the parameter class value, with Dynamo's warning texts."""
    from pydyn.dcpcontrol import DcpController, class_value, parameter
    v = V.blank("p")
    del v["cone_flip_r3"], v["how_many_processors"]
    rt = DcpController._runtime_from_vpr(v)
    out = capsys.readouterr().out
    assert '[warning] : vpr misses a parameter of type"cone_flip" in round 3' in out
    assert '[warning] : the vpr being read misses parameter of type "how_many_processors"' in out
    assert rt["cone_flip_r3"] == class_value(parameter("cone_flip")) == 0.0
    assert rt["how_many_processors"] == class_value(parameter("how_many_processors"))
    assert class_value(parameter("MCR_CACHE_ROOT")) == "[]"


def test_console_steps_as_shell_commands(tmp_path, monkeypatch):
    """`dtrandomize_azimuth --i IN --o OUT` and `daverage --d DATA --t TABLE --o AVG`: the console functions on files,
    from the shell. Each command is a fresh process, so the random stream starts as in a fresh MATLAB session."""
    from pydyn import cli
    from pydyn.console import daverage, dread
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=3)
    assert cli.dtrandomize_azimuth_main(["--i", "particles_edit_mod.tbl", "--o", "rand.tbl"]) == 0
    a, b = dread("particles_edit_mod.tbl"), dread("rand.tbl")
    assert np.array_equal(np.delete(a, 8, 1), np.delete(b, 8, 1))           # only column 9 (narot) changes
    assert ((b[:, 8] >= 0) & (b[:, 8] < 360)).all() and not np.array_equal(a[:, 8], b[:, 8])
    assert cli.daverage_main(["--d", "filamentsData_ctf", "--t", "rand.tbl", "--o", "avg.em", "--mw", "50",
                              "--device", "cpu"]) == 0
    oa = daverage("filamentsData_ctf", "t", "rand.tbl", device="cpu")
    assert np.array_equal(dynio.read_em("avg.em"), oa.average.astype(np.float32))
    assert not os.path.exists("raw_avg.em")                                  # --o: the average only
    assert cli.daverage_main(["-d", "filamentsData_ctf", "-t", "rand.tbl", "-o", "avg2.em", "--device", "cpu",
                              "--all-outputs"]) == 0
    assert os.path.exists("raw_avg2.em") and os.path.exists("fweight_avg2.em")  # Dynamo's 'o' option
    assert cli.daverage_main(["--d", "nofolder", "--t", "rand.tbl", "--o", "x.em"]) == 1
    exe = os.path.join(P, ".envs", "pydynamo", "bin", "dtrandomize_azimuth")
    if os.path.isfile(exe):                                                  # two fresh processes: the same angles
        for out in ("r1.tbl", "r2.tbl"):
            r = subprocess.run([exe, "--i", "particles_edit_mod.tbl", "--o", out], capture_output=True, text=True,
                               timeout=120)
            assert r.returncode == 0, r.stdout + r.stderr
        assert filecmp.cmp("r1.tbl", "r2.tbl", shallow=False)
