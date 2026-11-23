"""CPU tests of the Step-1 work: review fixes M1/M2/L1-L7, the output layout (log.txt, no pyDynamo files in the
project, engine files during the run), the byte-identical .exe with the `dynamo_execute_project` command, and the
viewer hooks of the dcp windows. No GPU and no display are needed."""
import hashlib
import os
import sys

import numpy as np
import pytest

from pydynamo.files import formats

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, ".."))
TD = os.path.join(P, "test_data", "bin8_filament")


# ------------------------------------------------------------------------------------------------ M1 preflight, L7 PTX cache
def test_preflight_names_the_missing_pieces(monkeypatch):
    from pydynamo.engine import kernels as E
    from pydynamo.engine import preflight
    monkeypatch.setattr(E, "ENGINE_BINARY", "/nonexistent/dynamo_compute_iteration_spp")
    monkeypatch.setattr(E, "CUDA8_LIB", "/nonexistent/lib64")
    with pytest.raises(RuntimeError) as e:
        preflight()
    msg = str(e.value)
    assert "engine binary not found" in msg and "libcufft.so.8.0" in msg and "libcublas.so.8.0" in msg
    assert "PYDYNAMO_DYNAMO_ROOT" in msg


def test_preflight_runs_before_project_work(tmp_path, monkeypatch):
    """execute_project, DynamoProject and run_engine all fail at once, before reading any project."""
    from pydynamo.engine import kernels as E
    from pydynamo.project.runner import DynamoProject
    from pydynamo.project.execute import execute_project
    monkeypatch.setattr(E, "ENGINE_BINARY", "/nonexistent/spp")
    with pytest.raises(RuntimeError, match="engine binary not found"):
        execute_project("no_such_project")
    with pytest.raises(RuntimeError, match="engine binary not found"):
        DynamoProject(str(tmp_path / "src"), str(tmp_path / "data"), str(tmp_path / "out"))
    import subprocess
    r = subprocess.run([sys.executable, "-I", os.path.join(P, "scripts", "align_card.py"), "x.card"],
                       env={**os.environ, "PYDYNAMO_DYNAMO_ROOT": "/nonexistent"}, capture_output=True, text=True,
                       cwd=str(tmp_path))
    assert r.returncode != 0 and "engine binary not found" in r.stderr


@pytest.mark.skipif(not os.path.isfile("/programs/x86_64-linux/dynamo/1.1.514/cuda/bin/dynamo_compute_iteration_spp"),
                    reason="needs Dynamo 1.1.514's engine binary")
def test_ptx_cache_bad_md5_is_re_extracted_and_unwritable_cache_falls_back(tmp_path, monkeypatch):
    from pydynamo.engine import kernels as E
    monkeypatch.delenv("PYDYNAMO_DYNAMO_PTX", raising=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("PYDYNAMO_CACHE_DIR", str(cache))
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
    monkeypatch.delenv("PYDYNAMO_CACHE_DIR")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    home = tmp_path / "home_is_a_file"
    home.write_text("x")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    p = E.ptx_file()
    assert p.startswith(str(tmp_path)) and "pydynamo-cache" in p
    assert hashlib.md5(open(p, "rb").read()).hexdigest() == E.PTX_MD5


# ------------------------------------------------------------------------------------------------ M2 symmetry device
def test_symmetry_gpu_work_runs_on_the_given_device(monkeypatch):
    from pydynamo.averaging import symmetry as S
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
    from pydynamo.project import runner
    assert "symmetry.set_device(self.gpus[0])" in inspect.getsource(runner.DynamoProject.__init__)


# ------------------------------------------------------------------------------------------------ L1, L2: table comparison
def _rows(tilt_cur, tilt_prev, az_cur=0.0, az_prev=0.0):
    cur = np.zeros((1, 12))
    prev = np.zeros((1, 12))
    cur[0, 6], cur[0, 7] = az_cur, tilt_cur
    prev[0, 6], prev[0, 7] = az_prev, tilt_prev
    return cur, prev


def test_column28_d_symmetry_fold():
    """dynamo_rigid_compare.m:117-127: for kind 'd' an axis difference above 90 degrees is folded by 180."""
    from pydynamo.project.runner import FdlibmTrig, compare_columns
    cur, prev = _rows(0.0, 150.0)
    _, c28, _, _ = compare_columns(cur, prev, FdlibmTrig, 1.0)
    assert abs(c28[0].real - 150.0) < 1e-9
    _, c28d, _, _ = compare_columns(cur, prev, FdlibmTrig, 2.0, sym_kind="d")
    assert abs(c28d[0].real - (-30.0)) < 1e-9
    _, c28c, _, _ = compare_columns(cur, prev, FdlibmTrig, 3.0, sym_kind="c")
    assert abs(c28c[0].real - 150.0) < 1e-9                       # only kind 'd' folds


def test_helical_dphi_zero_gives_abs_dnarot():
    """rigid_compare.m:87/113: 360/0 = Inf, mod(x, 0) = x, column 29 = |dnarot| (no ZeroDivisionError)."""
    from pydynamo.project.runner import FdlibmTrig, compare_columns, matlab_mod
    from pydynamo.averaging.symmetry import symorder
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
    from pydynamo.masks import dynamo_mollify
    with pytest.raises(TypeError, match="hwhm"):
        dynamo_mollify(np.ones((4, 4, 4)))


def test_mask_size_from_a_file(tmp_path):
    """dynamo_mask.m:274-292: a size that is a file name is read and its dimensions [nx ny nz] are the size."""
    from pydynamo.masks import dynamo_mask
    f = str(tmp_path / "vol.em")
    formats.write_em(f, np.zeros((6, 6, 6), np.float32))
    m = dynamo_mask("size", f, "type", "cylinder", "r", 2, "height", 4)
    assert m.shape == (6, 6, 6) and m.sum() > 0
    vol = np.zeros((5, 6, 7))                                      # [z, y, x]: size [7 6 5]
    from pydynamo.masks import _size_argument
    assert list(_size_argument(vol)) == [7, 6, 5]
    with pytest.raises(FileNotFoundError, match="cannot find requested file"):
        dynamo_mask("size", str(tmp_path / "missing.em"), "type", "cylinder")


# ------------------------------------------------------------------------------------------------ L6
def test_last_completed_iteration_uses_vpr_path(tmp_path, monkeypatch):
    from pydynamo import viewers as V
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
    from pydynamo.project.runner import check_report_path
    proj = tmp_path / "proj"
    assert check_report_path(None, str(proj)) is None
    assert check_report_path(str(tmp_path / "r.json"), str(proj)) == str(tmp_path / "r.json")
    assert check_report_path(str(tmp_path / "proj.report.json"), str(proj)) == str(tmp_path / "proj.report.json")
    for bad in (proj / "r.json", proj, proj / "a" / "b.json"):
        with pytest.raises(ValueError, match="inside the project folder"):
            check_report_path(str(bad), str(proj))


def test_no_pydynamo_files_are_written_by_the_project_code():
    import inspect
    from pydynamo.project import runner
    src = inspect.getsource(runner)
    assert 'p("pydynamo_log.txt")' not in src and 'p("pydynamo_report.json")' not in src
    assert "def dynamo_log_setup_banner" not in src


def test_engine_progress_files(tmp_path):
    """The engine's per-particle files: completedTasks (one float, after every particle) and the refined table every
    1000 particles (rows done so far, the rest zero)."""
    from pydynamo.project.runner import _Progress
    inter = tmp_path / "temp" / "ite_0001" / "intermediate_results"
    inter.mkdir(parents=True)
    names = ["refined_gpuProcRefTable_ite_0001_ref_001_proc_00000_gpu_%02d.btbl" % g for g in (0, 1)]
    chunks = [{"refined_btbl": str(inter / n)} for n in names]
    out = np.zeros((1200, 35), np.float32)
    out[:, 0] = np.arange(1, 1201)
    pg = _Progress(chunks, [700, 500], 35)
    done = lambda g: float(formats.read_btbl(str(inter / names[g]).replace("refined_gpuProcRefTable",
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
    t = formats.read_btbl(big[0]["refined_btbl"])
    assert t.shape == (2500, 35) and t[999, 0] == 1000 and t[1000, 0] == 0
    pg(2100, out2)
    t = formats.read_btbl(big[0]["refined_btbl"])
    assert t[1999, 0] == 2000 and t[2000, 0] == 0                              # rewritten at 2000, not at 2100


# ------------------------------------------------------------------------------------------------ .exe
@pytest.mark.skipif(not os.path.isfile(os.path.join(TD, "abp_align_eo.exe")), reason="needs test_data")
def test_exe_is_byte_identical_to_dynamos():
    from pydynamo.vpr import execution_script_lines
    for name, f in (("abp_align_eo", "abp_align_eo.exe"), ("bin4_align_1", "bin8_align_1.exe")):   # the second file was
        mine = "".join(x + "\n" for x in execution_script_lines({"name_project": name})).encode()    # written by the
        assert mine == open(os.path.join(TD, f), "rb").read()                                         # project bin4_align_1


def test_dynamo_execute_project_console_script_and_function(capsys, tmp_path, monkeypatch):
    import re
    from pydynamo import cli
    text = open(os.path.join(P, "pyproject.toml")).read()
    assert re.search(r'^dynamo_execute_project = "pydynamo\.cli:execute_main"$', text, re.M)
    assert '[tool.setuptools.packages.find]\nwhere = ["src"]' in text and "dmapview" in text
    assert "PySide6-Essentials==6.11.3" in text and "PySide6-Addons==6.11.3" in text
    monkeypatch.chdir(tmp_path)
    assert cli.execute_main(["no_such_project"]) == 1                          # called directly, as the script does
    assert "Sorry, the project has not" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.execute_main(["--help"])
    ns = []
    monkeypatch.setattr("pydynamo.project.execute.execute_project", lambda name, **kw: ns.append((name, kw)) or True)
    assert cli.execute_main(["p", "--gpus", "1,2", "--sms", "64", "--iterations", "3", "--report", "/x/r.json"]) == 0
    assert cli.execute_main(["q", "--matlab-threads", "64"]) == 0
    assert ns == [("p", dict(gpus=[1, 2], keep_temp=False, iterations=3, sms=64, report="/x/r.json",
                             matlab_threads=None)),
                  ("q", dict(gpus=None, keep_temp=False, iterations=None, sms=None, report=None, matlab_threads=64))]


# ------------------------------------------------------------------------------------------------ viewers in the dcp windows
class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, arrays=(), wait=False):
        self.calls.append((list(argv), [np.asarray(a).shape for a in arrays]))
        return object()


def _controller(tmp_path, monkeypatch, **runtime):
    from pydynamo import viewers as VW
    from pydynamo.manager.controller import DcpController
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(VW, "_launch", _Rec())
    c = DcpController()
    c.name = "p"
    c.runtime = {"nref_r1": 1.0, "file_template_initial": "t.em", "file_table_initial": "x.tbl", "file_mask": "m.em",
                 "file_mask_classification": "cm.em", **runtime}
    return c, VW._launch


def test_template_table_and_mask_menus_launch_the_viewers(tmp_path, monkeypatch):
    c, rec = _controller(tmp_path, monkeypatch)
    formats.write_em("t.em", np.zeros((6, 6, 6), np.float32))
    formats.write_em("m.em", np.ones((6, 6, 6), np.float32))
    formats.write_em("cm.em", np.ones((4, 4, 4), np.float32))
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
    from pydynamo import viewers as VW
    monkeypatch.setattr("pydynamo.vpr.load", lambda name, **kw: {"intertwin": "1/2/3/4/5/6/7/8", "path": str(tmp_path),
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
    from pydynamo import viewers as VW
    from pydynamo.masks import MaskEditor
    monkeypatch.chdir(tmp_path)
    rec = _Rec()
    monkeypatch.setattr(VW, "_launch", rec)
    formats.write_em("t.em", np.zeros((12, 12, 12), np.float32))
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
    import pydynamo
    for n in ("dmapview", "dview", "dslices", "dtplot", "dynamo_mapview", "dynamo_view", "dynamo_slices",
              "dynamo_tableplot"):
        assert n in pydynamo.__all__


# ------------------------------------------------------------------------------------------------ dvcheck sizes, unfold copy
_ROUNDS_EMPTY = 'Apparently all the rounds are empty: no iterations are executed. Check the "ite" parameter'


def _fresh_project(tmp_path, monkeypatch, name="p"):
    """dcp.new with default masks on synthetic 64^3 data (test_console_and_lifecycle run A): its dvcheck reports only
    _ROUNDS_EMPTY (warning ite_r1)."""
    from pydynamo import dcp
    from pydynamo import vpr as V
    from test_console_and_lifecycle import make_data
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    dcp.new(name, "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default",
            "t", "particles_edit_mod.tbl", "gui", 0)
    return V.load(name, verbose=False)


def test_dvcheck_reports_a_size_left_nan(tmp_path, monkeypatch):
    """dynamo_vpr_check.m:632/647/697/910 test sizes with ~=, and NaN ~= x is true in MATLAB: a file that is missing
    keeps its NaN size (dynamo_vpr_statistics.m:68-80, :169-174) and is reported, sprintf('%d', NaN) = 'NaN'."""
    from pydynamo.vpr import checks as vpr_check
    v = _fresh_project(tmp_path, monkeypatch)
    r = vpr_check.check(dict(v, file_mask_smoothing="my_smask.em"), 0)          # check 1 does not test this field
    assert r.report == [_ROUNDS_EMPTY, "Smoothing FSC Mask has size NaN, reference has size 64. Please check "]
    assert (r.fields_warning, r.fields_error, r.warning_state) == (
        ["ite_r1", "file_mask_smoothing", "file_template_initial"], [], "warning")
    r = vpr_check.check(dict(v, file_fmask_initial="missing_fmask.em"), 0)
    assert r.report == ['ATTENTION on parameter "file_fmask_initial"',
                        "   file/folder that defines the initial count of fourier components does not seem to exist.",
                        '   Apparent current value in project: "missing_fmask.em".', _ROUNDS_EMPTY,
                        "fmask has size NaN, reference has size 64. Please check "]
    assert r.fields_warning == ["ite_r1"]
    assert r.fields_error == ["file_fmask_initial", "folder_data", "file_fmask_initial"]
    r = vpr_check.check(dict(v, file_template_initial="missing_template.em", localnc_r1=0.0, localnc_r2=0.0), 0)
    localnc = ('reference size (NaN) differs from particle size (64). This is not allowed when "localnc" is set to '
               "zero (as in round %d). Please check. ")
    assert r.report == ['ATTENTION on parameter "file_template_initial"',
                        "   file/folder with the initial template does not seem to exist.",
                        '   Apparent current value in project: "missing_template.em".', _ROUNDS_EMPTY,
                        "Mask has size 64, reference has size NaN. Please check ",
                        "Smoothing FSC Mask has size 64, reference has size NaN. Please check ",
                        localnc % 1, localnc % 2,
                        "fmask has size 64, reference has size NaN. Please check "]
    assert r.fields_warning == ["ite_r1", "file_mask", "file_template_initial", "file_mask_smoothing",
                                "file_template_initial"]
    assert r.fields_error == ["file_template_initial", "folder_data", "file_template_initial",   # check 14: errors{1},
                              "folder_data", "file_fmask_initial"]                                # {2} once per call


def test_copyfile_of_a_file_onto_itself(tmp_path, monkeypatch, capsys):
    """dynamo_copyfile.m:117-123 skips equal strings only; the same file under another spelling reaches copyfile,
    which returns success 0, and Dynamo prints its lines and goes on (:170-177) instead of failing."""
    from pydynamo import vpr as V
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("p", "settings"))
    formats.write_em("p/settings/mask.em", np.ones((4, 4, 4), np.float32))
    before = open("p/settings/mask.em", "rb").read()
    V._copyfile("./p/settings/mask.em", "./p/settings/mask.em", True)          # strcmp: nothing to do
    assert capsys.readouterr().out == ""
    absolute = os.path.join(str(tmp_path), "p", "settings", "mask.em")
    for src in ("p/settings/mask.em", absolute):
        V._copyfile(src, "./p/settings/mask.em", True)
        assert capsys.readouterr().out.splitlines() == [
            "[dcopyfile] Error copying %s into ./p/settings/mask.em" % src,
            "Cannot copy or move a file or directory onto itself.", " "]
        assert open("p/settings/mask.em", "rb").read() == before
    V._copyfile("p/settings/mask.em", "./p/settings/other.em", True)           # another file: copied, silent
    assert open("p/settings/other.em", "rb").read() == before and capsys.readouterr().out == ""


def test_unfold_with_a_seed_spelled_as_its_database_file(tmp_path, monkeypatch, capsys):
    """file_mask 'p/settings/mask.em' (as the dcp file chooser gives it) is ./p/settings/mask.em: links2settings
    goes on, and unfold ends as for the fresh project (all rounds empty, test_console_and_lifecycle run A)."""
    from pydynamo import vpr as V
    v = _fresh_project(tmp_path, monkeypatch)
    before = open("p/settings/mask.em", "rb").read()
    capsys.readouterr()
    assert V.unfold(dict(v, file_mask="p/settings/mask.em"), verbose=False) == (None, "error")
    out = capsys.readouterr().out.splitlines()
    i = out.index("[dcopyfile] Error copying p/settings/mask.em into ./p/settings/mask.em")
    assert out[i + 1:i + 3] == ["Cannot copy or move a file or directory onto itself.", " "]
    assert open("p/settings/mask.em", "rb").read() == before
    assert os.path.isfile("p/settings/template_initial_ref_001.em")          # the later copies were made
    assert open("p/next_iteration.txt").read() == "1\n"

