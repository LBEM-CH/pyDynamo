"""CPU tests of the Step-1 work: review fixes M1/M2/L1-L7, the output layout (log.txt, no pyDynamo files in the
project, engine files during the run), the byte-identical .exe with the `dynamo_execute_project` command, and the
viewer hooks of the dcp windows. No GPU and no display are needed."""
import hashlib
import os
import sys

import numpy as np
import pytest

from pydyn import dynio

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", ".."))
TD = os.path.join(P, "test_data", "bin8_filament")


# ------------------------------------------------------------------------------------------------ M1 preflight, L7 PTX cache
def test_preflight_names_the_missing_piece_and_the_fast_engine(monkeypatch):
    from pydyn import dynamo_exact as E
    from pydyn.engine import preflight
    monkeypatch.setattr(E, "ENGINE_BINARY", "/nonexistent/dynamo_compute_iteration_spp")
    monkeypatch.setattr(E, "CUDA8_LIB", "/nonexistent/lib64")
    with pytest.raises(RuntimeError) as e:
        preflight("exact")
    msg = str(e.value)
    assert "engine binary not found" in msg and "libcufft.so.8.0" in msg and "libcublas.so.8.0" in msg
    assert "--engine fast" in msg
    assert preflight("fast") == "fast"                       # the fast engine needs nothing from the installation


def test_preflight_runs_before_project_work(tmp_path, monkeypatch):
    """execute_project, DynamoProject and run_engine all fail at once, before reading any project."""
    from pydyn import dynamo_exact as E
    from pydyn.dynproject import DynamoProject
    from pydyn.execute import execute_project
    monkeypatch.setattr(E, "ENGINE_BINARY", "/nonexistent/spp")
    with pytest.raises(RuntimeError, match="--engine fast"):
        execute_project("no_such_project", engine="exact")
    with pytest.raises(RuntimeError, match="--engine fast"):
        DynamoProject(str(tmp_path / "src"), str(tmp_path / "data"), str(tmp_path / "out"), engine="exact")
    import subprocess
    r = subprocess.run([sys.executable, "-I", os.path.join(P, "prototype", "run_engine.py"), "x.card", "--engine", "exact"],
                       env={**os.environ, "PYDYN_DYNAMO_ROOT": "/nonexistent"}, capture_output=True, text=True,
                       cwd=str(tmp_path))
    assert r.returncode != 0 and "--engine fast" in r.stderr


@pytest.mark.skipif(not os.path.isfile("/programs/x86_64-linux/dynamo/1.1.514/cuda/bin/dynamo_compute_iteration_spp"),
                    reason="needs Dynamo 1.1.514's engine binary")
def test_ptx_cache_bad_md5_is_re_extracted_and_unwritable_cache_falls_back(tmp_path, monkeypatch):
    from pydyn import dynamo_exact as E
    monkeypatch.delenv("PYDYN_DYNAMO_PTX", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("PYDYN_CACHE_DIR", str(cache))
    bad = cache / ("dynamo_spp_1.1.514_%s.ptx" % E.PTX_MD5)
    bad.write_bytes(b"corrupt")
    p = E.ptx_file()
    assert p == str(bad) and hashlib.md5(open(p, "rb").read()).hexdigest() == E.PTX_MD5
    # batched PTX: a corrupt cached copy is rewritten, the md5 in the name is that of the content
    bp, ptrs = E.batched_ptx_file()
    assert hashlib.md5(open(bp, "rb").read()).hexdigest() in bp
    open(bp, "wb").write(b"corrupt")
    bp2, _ = E.batched_ptx_file()
    assert bp2 == bp and hashlib.md5(open(bp2, "rb").read()).hexdigest() in bp2
    # ~/.cache not writable (HOME is a file): the per-user folder under the temp dir is used
    monkeypatch.delenv("PYDYN_CACHE_DIR")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home_is_a_file"
    home.write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    p = E.ptx_file()
    assert p.startswith(str(tmp_path)) and "pydyn-cache" in p
    assert hashlib.md5(open(p, "rb").read()).hexdigest() == E.PTX_MD5


# ------------------------------------------------------------------------------------------------ M2 symmetry device
def test_symmetry_gpu_work_runs_on_the_given_device(monkeypatch):
    from pydyn import symmetry as S
    seen = []

    class Dev:
        def __init__(self, i):
            self.i = i

        def __enter__(self):
            seen.append(("enter", self.i))

        def __exit__(self, *a):
            seen.append(("exit", self.i))

    class FakeCp:
        class cuda:
            Device = Dev
    monkeypatch.setattr(S, "cp", FakeCp)
    S.set_device(None)
    with S._device_ctx():
        pass
    assert seen == []                                          # default: CuPy's current device, untouched
    S.set_device(2)
    try:
        with S._device_ctx():
            pass
    finally:
        S.set_device(None)
    assert seen == [("enter", 2), ("exit", 2)]


def test_dynproject_sets_the_symmetry_device_to_the_first_gpu():
    import inspect
    from pydyn import dynproject
    assert "symmetry.set_device(self.gpus[0])" in inspect.getsource(dynproject.DynamoProject.__init__)


# ------------------------------------------------------------------------------------------------ L1, L2: table comparison
def _rows(tilt_cur, tilt_prev, az_cur=0.0, az_prev=0.0):
    cur = np.zeros((1, 12))
    prev = np.zeros((1, 12))
    cur[0, 6], cur[0, 7] = az_cur, tilt_cur
    prev[0, 6], prev[0, 7] = az_prev, tilt_prev
    return cur, prev


def test_column28_d_symmetry_fold():
    """dynamo_rigid_compare.m:117-127: for kind 'd' an axis difference above 90 degrees is folded by 180."""
    from pydyn.dynproject import FdlibmTrig, compare_columns
    cur, prev = _rows(0.0, 150.0)
    _, c28, _, _ = compare_columns(cur, prev, FdlibmTrig, 1.0)
    assert abs(c28[0].real - 150.0) < 1e-9
    _, c28d, _, _ = compare_columns(cur, prev, FdlibmTrig, 2.0, sym_kind="d")
    assert abs(c28d[0].real - (-30.0)) < 1e-9
    _, c28c, _, _ = compare_columns(cur, prev, FdlibmTrig, 3.0, sym_kind="c")
    assert abs(c28c[0].real - 150.0) < 1e-9                       # only kind 'd' folds


def test_helical_dphi_zero_gives_abs_dnarot():
    """rigid_compare.m:87/113: 360/0 = Inf, mod(x, 0) = x, column 29 = |dnarot| (no ZeroDivisionError)."""
    from pydyn.dynproject import FdlibmTrig, compare_columns, matlab_mod
    from pydyn.symmetry import symorder
    so = symorder("h[0,0.3]")
    assert so == float("inf")
    cur, prev = _rows(10.0, 10.0, az_cur=-37.0, az_prev=5.0)       # dnarot = (-(-37)) - (-5) = 42
    _, _, c29, _ = compare_columns(cur, prev, FdlibmTrig, so)
    assert abs(c29[0] - 42.0) < 1e-12
    cur[0, 6] = 37.0                                               # dnarot = -37 - (-5) = -32
    _, _, c29, _ = compare_columns(cur, prev, FdlibmTrig, so)
    assert abs(c29[0] - 32.0) < 1e-12
    assert np.array_equal(matlab_mod(np.array([-3.0, 4.5]), 0.0), [-3.0, 4.5])


# ------------------------------------------------------------------------------------------------ L3 masks
def test_mollify_has_no_default_hwhm():
    from pydyn.masks import dynamo_mollify
    with pytest.raises(TypeError, match="hwhm"):
        dynamo_mollify(np.ones((4, 4, 4)))


def test_mask_size_from_a_file(tmp_path):
    """dynamo_mask.m:274-292: a size that is a file name is read and its dimensions [nx ny nz] are the size."""
    from pydyn.masks import dynamo_mask
    f = str(tmp_path / "vol.em")
    dynio.write_em(f, np.zeros((6, 6, 6), np.float32))
    m = dynamo_mask("size", f, "type", "cylinder", "r", 2, "height", 4)
    assert m.shape == (6, 6, 6) and m.sum() > 0
    vol = np.zeros((5, 6, 7))                                      # [z, y, x]: size [7 6 5]
    from pydyn.masks import _size_argument
    assert list(_size_argument(vol)) == [7, 6, 5]
    with pytest.raises(FileNotFoundError, match="cannot find requested file"):
        dynamo_mask("size", str(tmp_path / "missing.em"), "type", "cylinder")


# ------------------------------------------------------------------------------------------------ L6
def test_last_completed_iteration_uses_vpr_path(tmp_path, monkeypatch):
    import pydyn.viewers as V
    run = tmp_path / "elsewhere"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    vpr = {"intertwin": "1/2/3/4/5/6/7/8", "path": str(run), "name_project": "proj"}
    for r in range(1, 9):
        vpr["nref_r%d" % r] = 1
        vpr["ite_r%d" % r] = 2 if r == 1 else 0
    for ite in (1, 2):
        f = run / "proj" / "results" / ("ite_%04d" % ite) / "averages" / ("average_ref_001_ite_%04d.em" % ite)
        f.parent.mkdir(parents=True)
        f.write_bytes(b"")
    assert V.last_completed_iteration("proj", vpr) == 2             # under vpr.path, not under the current folder
    vpr["path"] = str(tmp_path / "gone")                             # a vpr.path that is not a folder: pwd
    assert V.last_completed_iteration("proj", vpr) == 0
    assert V.project_folder_of("proj", vpr) == os.path.join(str(cwd), "proj")


# ------------------------------------------------------------------------------------------------ output layout
def test_report_path_refused_inside_the_project(tmp_path):
    from pydyn.dynproject import check_report_path
    proj = tmp_path / "proj"
    assert check_report_path(None, str(proj)) is None
    assert check_report_path(str(tmp_path / "r.json"), str(proj)) == str(tmp_path / "r.json")
    assert check_report_path(str(tmp_path / "proj.report.json"), str(proj)) == str(tmp_path / "proj.report.json")
    for bad in (proj / "r.json", proj, proj / "a" / "b.json"):
        with pytest.raises(ValueError, match="inside the project folder"):
            check_report_path(str(bad), str(proj))


def test_no_pydyn_files_are_written_by_the_project_code():
    import inspect
    from pydyn import dynproject
    src = inspect.getsource(dynproject)
    assert 'p("pydyn_log.txt")' not in src and 'p("pydyn_report.json")' not in src
    assert "def dynamo_log_setup_banner" not in src


def test_engine_progress_files(tmp_path):
    """The engine's per-particle files: completedTasks (one float, after every particle) and the refined table every
    1000 particles (rows done so far, the rest zero)."""
    from pydyn.dynproject import _Progress
    inter = tmp_path / "temp" / "ite_0001" / "intermediate_results"
    inter.mkdir(parents=True)
    names = ["refined_gpuProcRefTable_ite_0001_ref_001_proc_00000_gpu_%02d.btbl" % g for g in (0, 1)]
    chunks = [{"refined_btbl": str(inter / n)} for n in names]
    out = np.zeros((1200, 35), np.float32)
    out[:, 0] = np.arange(1, 1201)
    pg = _Progress(chunks, [700, 500], 35)
    done = lambda g: float(dynio.read_btbl(str(inter / names[g]).replace("refined_gpuProcRefTable",
                                                                       "completedTasks_gpuProcRef")).ravel()[0])
    pg(1, out)
    assert done(0) == 1 and not os.path.exists(str(inter / names[0]))
    pg(700, out)
    assert done(0) == 700 and not os.path.exists(str(inter / names[0]))      # no multiple of 1000 in chunk 0
    pg(1200, out)
    assert done(0) == 700 and done(1) == 500
    assert not os.path.exists(str(inter / names[1]))                          # 500 < 1000: only the final write
    big = [{"refined_btbl": str(inter / "refined_gpuProcRefTable_ite_0001_ref_001_proc_00000_gpu_09.btbl")}]
    out2 = np.zeros((2500, 35), np.float32)
    out2[:, 0] = np.arange(1, 2501)
    pg = _Progress(big, [2500], 35)
    pg(999, out2)
    assert not os.path.exists(big[0]["refined_btbl"])
    pg(1000, out2)
    t = dynio.read_btbl(big[0]["refined_btbl"])
    assert t.shape == (2500, 35) and t[999, 0] == 1000 and t[1000, 0] == 0
    pg(2100, out2)
    t = dynio.read_btbl(big[0]["refined_btbl"])
    assert t[1999, 0] == 2000 and t[2000, 0] == 0                              # rewritten at 2000, not at 2100


# ------------------------------------------------------------------------------------------------ .exe
@pytest.mark.skipif(not os.path.isfile(os.path.join(TD, "abp_align_eo.exe")), reason="needs test_data")
def test_exe_is_byte_identical_to_dynamos():
    from pydyn.vpr import execution_script_lines
    for name, f in (("abp_align_eo", "abp_align_eo.exe"), ("bin4_align_1", "bin8_align_1.exe")):   # the second file was
        mine = "".join(x + "\n" for x in execution_script_lines({"name_project": name})).encode()    # written by the
        assert mine == open(os.path.join(TD, f), "rb").read()                                         # project bin4_align_1


def test_dynamo_execute_project_console_script_and_function(capsys, tmp_path, monkeypatch):
    import re
    from pydyn import cli
    text = open(os.path.join(P, "pyproject.toml")).read()
    assert re.search(r'^dynamo_execute_project = "pydyn\.cli:execute_main"$', text, re.M)
    assert 'packages = ["pydyn", "pydyn.viewers", "pydyn.webui"]' in text and "dmapview" in text
    assert "PySide6-Essentials==6.11.3" in text and "PySide6-Addons==6.11.3" in text
    monkeypatch.chdir(tmp_path)
    assert cli.execute_main(["no_such_project", "--engine", "fast"]) == 1       # called directly, as the script does
    assert "Sorry, the project has not" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.execute_main(["--help"])
    ns = []
    monkeypatch.setattr("pydyn.execute.execute_project", lambda name, **kw: ns.append((name, kw)) or True)
    assert cli.execute_main(["p", "--gpus", "1,2", "--engine", "fast", "--exact-sms", "64", "--iterations", "3",
                             "--report", "/x/r.json"]) == 0
    assert cli.execute_main(["q", "--matlab-threads", "64"]) == 0
    assert ns == [("p", dict(gpus=[1, 2], keep_temp=False, iterations=3, engine="fast",
                             exact_sms=64, report="/x/r.json", matlab_threads=None)),
                  ("q", dict(gpus=None, keep_temp=False, iterations=None, engine=None,
                             exact_sms=None, report=None, matlab_threads=64))]


# ------------------------------------------------------------------------------------------------ viewers in the dcp windows
class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, arrays=(), wait=False):
        self.calls.append((list(argv), [np.asarray(a).shape for a in arrays]))
        return object()


def _controller(tmp_path, monkeypatch, **runtime):
    import pydyn.viewers as VW
    from pydyn.dcpcontrol import DcpController
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(VW, "_launch", _Rec())
    c = DcpController()
    c.name = "p"
    c.runtime = {"nref_r1": 1.0, "file_template_initial": "t.em", "file_table_initial": "x.tbl", "file_mask": "m.em",
                 "file_mask_classification": "cm.em", **runtime}
    return c, VW._launch


def test_template_table_and_mask_menus_launch_the_viewers(tmp_path, monkeypatch):
    c, rec = _controller(tmp_path, monkeypatch)
    dynio.write_em("t.em", np.zeros((6, 6, 6), np.float32))
    dynio.write_em("m.em", np.ones((6, 6, 6), np.float32))
    dynio.write_em("cm.em", np.ones((4, 4, 4), np.float32))
    c.view_template()
    assert rec.calls[-1][0][:2] == ["view", "t.em"] and ("Showing file in dview: t.em", "k") in c.messages
    c.plot_orientations()
    assert rec.calls[-1][0] == ["tableplot", "x.tbl"]
    c.view_mask()
    assert rec.calls[-1][0][:2] == ["view", "m.em"]
    c.overlay_mask("y")
    argv = rec.calls[-1][0]
    assert argv[0] == "slices" and "t.em" in argv and "--overlay" in argv and "m.em" in argv and "y" in argv
    n = len(rec.calls)
    c.overlay_mask("x", "cmask")                                   # 6^3 template, 4^3 mask: Dynamo's message, no viewer
    assert len(rec.calls) == n
    assert any(m.startswith("Dimension mistmatch") for m, _ in c.messages)
    assert any("Dimension :  4 4 4" in m for m, _ in c.messages)
    c.runtime["file_mask_classification"] = ""
    c.view_mask("cmask")
    assert ("Cannot retrieve current mask", "k") in c.messages and len(rec.calls) == n


def test_last_averages_and_fsc_menus(tmp_path, monkeypatch):
    c, rec = _controller(tmp_path, monkeypatch)
    import pydyn.viewers as VW
    monkeypatch.setattr("pydyn.vpr.load", lambda name, **kw: {"intertwin": "1/2/3/4/5/6/7/8", "path": str(tmp_path),
                                                              "name_project": "p", "ite_r1": 1,
                                                              **{"nref_r%d" % r: 1 for r in range(1, 9)},
                                                              **{"ite_r%d" % r: 0 for r in range(2, 9)}})
    for f in ("average_ref_001_ite_0001.em",):
        p = tmp_path / "p" / "results" / "ite_0001" / "averages" / f
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    c.view_last_averages()
    assert rec.calls[-1][0][:2] == ["view", str(tmp_path / "p" / "results" / "ite_0001" / "averages" /
                                                 "average_ref_001_ite_0001.em")]
    c.plot_adaptive_filtering_fsc()
    assert rec.calls[-1][0] == ["fsc", "p", "--last", "1"]
    assert VW.last_completed_iteration("p") == 1


def test_mask_editor_show_and_view(tmp_path, monkeypatch):
    import pydyn.viewers as VW
    from pydyn.masks import MaskEditor
    monkeypatch.chdir(tmp_path)
    rec = _Rec()
    monkeypatch.setattr(VW, "_launch", rec)
    dynio.write_em("t.em", np.zeros((12, 12, 12), np.float32))
    ed = MaskEditor(size=[12, 12, 12], mask="m.em", volume="t.em")
    ed.select("r", "4")
    ed.select("height", "6")
    ed.set_type("cylinder")
    ed.show("mask", "x")
    argv, arrays = rec.calls[-1]
    assert argv[0] == "slices" and "--kind" in argv and "x" in argv and "Mask: x view" in argv and arrays == [(12, 12, 12)]
    ed.show("overlay", "z", alpha=0.3, color="g")
    argv, arrays = rec.calls[-1]
    assert "--alpha" in argv and "0.3" in argv and "g" in argv and len(arrays) == 2
    ed.view_file("nonexistent.em")
    assert ed.messages[-1] == "file not found"
    ed.view_file("t.em")
    assert rec.calls[-1][0][:2] == ["view", "t.em"]


def test_viewer_functions_are_exported():
    import pydyn
    for n in ("dmapview", "dview", "dslices", "dtplot", "dynamo_mapview", "dynamo_view", "dynamo_slices",
              "dynamo_tableplot"):
        assert n in pydyn.__all__

