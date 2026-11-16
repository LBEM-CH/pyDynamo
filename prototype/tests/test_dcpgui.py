"""The dcp GUI: controller (pydyn.dcpcontrol, no display needed) and the Tk view's entry points (pydyn.gui).

The controller is compared with Dynamo 1.1.514's own dcp windows, driven headless in run G
(benchmarks/04_journey_bin8_align_1/dynamo/gui, script scripts/g_gui_journey.m). Window states, the rounds tables, the mask files
(md5) and the final vpr of that run are kept as small fixtures in tests/fixtures/stage3/gui, so these tests run in CI
on synthetic data of the same size (64^3, 6-digit particle names, the user's file names). The full replay on the
user's data is benchmarks/04_journey_bin8_align_1/checks/gui/run_check_gui.py (parity_gui.md).
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import CHECK_CLEAN

from pydyn import dynio
from pydyn import vpr as V
from pydyn.dcpcontrol import DcpController, ellipsoid, format_str, str2num

from test_stage3 import _json_form, make_data

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "stage3", "gui")
P = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = "/programs/x86_64-linux/dynamo/1.1.514/matlab/src"
GT = os.path.join(P, "benchmarks", "04_journey_bin8_align_1", "dynamo", "gui")
BUTTONS = [("pushbutton_data", "data"), ("pushbutton_table", "table"), ("pushbutton_template", "template"),
           ("pushbutton_masks", "masks"), ("pushbutton_settings", "settings"), ("pushbutton_hardware", "destination"),
           ("pushbutton_check", "check"), ("pushbutton_unfold", "unfold"), ("pushbutton_run", "run")]


def fix(name):
    return open(os.path.join(FIX, name)).read().splitlines()


def buttons(c):
    return ["%s %s" % (b, "on" if (s == "check" or c.enabled[s]) else "off") for b, s in BUTTONS]


def table(c, **kw):
    rows, _ = c.rounds_table(**kw)
    return ["%s|%s|%s" % (n, lab, "|".join(cells)) for n, lab, cells in rows]


def md5(f):
    return hashlib.md5(open(f, "rb").read()).hexdigest()


# ---------------------------------------------------------------------------------------------------- helpers
def test_str2num_restricted():
    assert np.array_equal(str2num("4 4 2"), [4, 4, 2])
    assert np.array_equal(str2num("[4,4,2]"), [4, 4, 2])
    assert np.array_equal(str2num(" 31 31 31"), [31, 31, 31])
    assert np.array_equal(str2num("1:3"), [1, 2, 3])
    assert np.array_equal(str2num("0:2:6"), [0, 2, 4, 6])
    assert np.array_equal(str2num("-1.5e1"), [-15])
    assert np.isinf(str2num("Inf")[0]) and np.isnan(str2num("NaN")[0])
    assert str2num("true") is True
    assert str2num("").size == 0 and str2num("[]").size == 0
    for bad in ("rand(3)", "abc", "1;2", "0.5:1", "'x'"):
        with pytest.raises(NotImplementedError):
            str2num(bad)


def test_format_str():
    """dynamo__format_str.m: leading blank per number, %d for integers, %5.2f otherwise."""
    assert format_str(64.0) == " 64"
    assert format_str(np.array([4.0, 4, 2])) == " 4 4 2"
    assert format_str(0.2) == "  0.20"
    assert format_str("c1") == "c1"
    assert format_str("") == "[]"
    assert format_str(False) == "0"
    assert format_str(np.zeros(0)) == "[]"


def test_ellipsoid_default_is_dynamo_default_mask():
    from pydyn.project import default_mask
    assert np.array_equal(ellipsoid(31, [64, 64, 64]), default_mask(64))
    assert np.array_equal(ellipsoid([31, 31, 31], 64), default_mask(64))
    assert not ellipsoid(0, 8).any()


def test_ellipsoid_bytes_match_dynamo(tmp_path):
    """dcp 'ellipsoid' button outputs of run G: semiaxes '20 15 10' and '12.5' on a 64^3 template."""
    ref = json.load(open(os.path.join(FIX, "dynamo_mask_md5.json")))
    for semi, f in (([20, 15, 10], "ell_20_15_10.em"), (12.5, "ell_12p5.em")):
        out = str(tmp_path / f)
        dynio.write_em(out, ellipsoid(semi, [64, 64, 64]))
        assert md5(out) == ref[f]


@pytest.mark.skipif(not os.path.isdir(SRC), reason="Dynamo 1.1.514 source not available")
def test_parameter_classes_match_source():
    sys.path.insert(0, os.path.join(P, "prototype", "tools"))
    from gen_dcp_parameters import parse_all
    from pydyn.dcp_parameters import PARAMETERS
    assert parse_all(SRC) == PARAMETERS


def test_parameter_help_matches_source():
    """The rounds window's detailed help is Dynamo's dynamo_vpr_help text (S/dynamo__help_parameters.m), verbatim."""
    from pydyn.dcp_help import HELP
    lines = DcpController.parameter_help("area_search_modus")
    assert "4:  limis are understood from the estimation provided for the first iteration" in lines
    assert DcpController.parameter_help("no_such_parameter") == []
    if not os.path.isdir(SRC):
        pytest.skip("Dynamo 1.1.514 source not present")
    sys.path.insert(0, os.path.join(P, "prototype", "tools"))
    from gen_dcp_help import parse_help
    assert parse_help(SRC) == HELP and sum(map(len, HELP.values())) == 912


def test_round_area_defaults_match_dynamo_fig():
    from pydyn.dcpcontrol import ROUND_AREAS, ROUND_AREAS_ON
    st = dict(ln.split(" ", 1) for ln in fix("dynamo_rounds_window_state.txt"))
    for a in ROUND_AREAS:
        assert st["checkbox_functionality_" + a] == ("1" if a in ROUND_AREAS_ON else "0")
    assert st["checkbox_advanced"] == "0" and st["togglebutton_round"] == "1"


# ---------------------------------------------------------------------------------------------------- journey
def _journey(root, monkeypatch, make=DcpController):
    """Run G's inputs through the controller on synthetic data (same names and size as the user's). `make` builds
    the controller: DcpController itself, or the HTML view's API seen as a controller (test_webui.py)."""
    monkeypatch.chdir(root)
    make_data(str(root))
    dynio.write_em("small32.em", np.ones((32, 32, 32)))
    c = make()
    out = {}
    c.create("g1")
    out["buttons_after_create"] = buttons(c)
    assert c.data_enter("filamentsData_ctf").ok
    v = V.load("g1")
    assert (v["dim_r1"], v["low_r1"], v["ite_r1"]) == (64.0, 21.0, 1.0)      # dcp.dataEnter defaults [RUN G]
    assert c.table_enter("particles_edit_mod.tbl").ok
    assert c.template_enter("raw_template_ctf.em").ok
    out["masks_window_state"] = ["edit_semiaxis [%s]" % c.masks_window_opened(),
                                 "text_size [%s]" % format_str(c.template_size())]
    assert not c.mask_enter("small32.em").ok                                  # size mismatch: rejected
    assert c.runtime["file_mask"] == "my_mask.em" and V.load("g1")["file_mask"] == "my_mask.em"
    assert c.masks_use_defaults().ok
    out["masks_after_defaults"] = ["edit_semiaxis [%s]" % c.semiaxis] + ["%s [%s]" % (e, c.runtime[f]) for e, f in (
        ("edit_mask", "file_mask"), ("edit_cmask", "file_mask_classification"),
        ("edit_fmask", "file_fmask_initial"), ("edit_smask", "file_mask_smoothing"))]
    out["masks_md5"] = {f: md5("g1/settings/" + f) for f in ("automatic_mask.em", "automatic_cmask.em",
                                                             "fmask_initial_ref_001.em", "smoothingMaskOnes.em")}
    dynio.write_em("g1/settings/editorMask.em", np.asarray(ellipsoid([25, 25, 12], 64)))
    for enter in (c.mask_enter, c.cmask_enter, c.smask_enter):
        assert enter("g1/settings/editorMask.em").ok
    out["buttons_after_masks"] = buttons(c)
    out["rounds_table_initial"] = table(c)
    E = c.set_value
    E("ite", "2", 1)
    E("ite", "2", 2)
    for r in range(1, 5):
        for p, s in (("cone_range", "20"), ("cone_sampling", "3"), ("refine", "2"), ("low", "25"),
                     ("area_search", "4 4 2"), ("area_search_modus", "4")):
            assert E(p, s, r).ok
    for r in range(2, 5):
        E("dim", "64", r)
    for r, s in ((2, "45"), (3, "90"), (4, "45")):
        E("inplane_range", s, r)
    for r, s in ((1, "15"), (2, "5"), (3, "15"), (4, "5")):
        E("inplane_sampling", s, r)
    rej = E("cone_range", "[]", 5)
    assert not rej.ok and "cannot be converted" in rej.report[0]
    out["rounds_table_advanced"] = table(c, include_advanced=True)
    for r in range(1, 5):
        E("cone_flip", "1", r)
    out["rounds_table_general"] = table(c, include_advanced=True, round_parameters=False)
    E("apix", "1")
    out["vpr_after_rounds"] = _json_form("g1/settings/virtual_project.mat")
    out["environment_window_state"] = ["%s [%s]" % (p, s) for p, (s, ok) in c.environment_values().items()]
    c.set_destination("standalone_gpu")
    assert c.environment_edit("gpu_identifier_set", "0 1 2 3").ok
    assert c.environment_edit("matlab_workers_average", "50").ok
    assert not c.environment_edit("gpu_motor", "xyz").ok
    out["buttons_before_check"] = buttons(c)
    out["check"] = c.check()
    out["buttons_after_check"] = buttons(c)
    out["unfold"] = c.unfold()
    out["vpr_after_unfold"] = _json_form("g1/settings/virtual_project.mat")
    return c, out


def _same_except_path(mine, ref):
    assert len(mine) == len(ref)
    for a, b in zip(mine, ref):
        if a.startswith("path|path|") and b.startswith("path|path|"):
            continue
        assert a == b


def _vpr_same_except_path(mine, ref):
    assert [x["name"] for x in mine] == [x["name"] for x in ref]
    for a, b in zip(mine, ref):
        if a["name"] == "path":
            continue
        assert a == b, (a, b)


def test_gui_journey_matches_dynamo_run_g(tmp_path, monkeypatch):
    c, out = _journey(tmp_path, monkeypatch)
    for k in ("buttons_after_create", "masks_window_state", "masks_after_defaults", "buttons_after_masks",
              "rounds_table_initial", "rounds_table_advanced", "environment_window_state", "buttons_before_check",
              "buttons_after_check"):
        assert out[k] == fix("dynamo_%s.txt" % k), k
    _same_except_path(out["rounds_table_general"], fix("dynamo_rounds_table_general.txt"))
    ref = json.load(open(os.path.join(FIX, "dynamo_mask_md5.json")))
    for f, h in out["masks_md5"].items():
        assert h == ref["snapG5b_defaults/settings/" + f], f
    # GUI saves rebuild the vpr from the runtime: 468 convention fields, no *_all (08a 5.1) [RUN G]
    _vpr_same_except_path(out["vpr_after_rounds"],
                          json.load(open(os.path.join(FIX, "dynamo_vpr_g1_after_rounds.json"))))
    r = out["check"]
    assert (r.report, r.fields_warning, r.fields_error, r.warning_state) == CHECK_CLEAN
    assert out["unfold"] == ("g1.exe", "ok")
    _vpr_same_except_path(out["vpr_after_unfold"], json.load(open(os.path.join(FIX, "dynamo_vpr_g1_after_unfold.json"))))
    mine = open("g1/cards/ite_0001/card_ite_ite_0001.card").read().splitlines()
    ref_card = fix("dynamo_g1_card_ite_ite_0001.card")
    assert [x for x in mine if not x.startswith("path=")] == [x for x in ref_card if not x.startswith("path=")]
    assert sorted(os.listdir("g1/cards")) == ["ite_0001", "ite_0002", "ite_0003", "ite_0004"]
    assert c.iteration_rounds() == [1, 1, 2, 2]
    assert not os.path.exists("decoy")                                      # documented deviation
    assert c.execution_order() == "./g1.exe"


def test_rounds_add_remove_and_rejections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path))
    c = DcpController()
    c.new("p1", d="filamentsData_ctf", template="raw_template_ctf.em", t="particles_edit_mod.tbl")
    assert any(k.endswith("_all") for k in V.load("p1"))                    # dcp.new: 488 fields with *_all
    assert c.active_rounds() == []
    r, chk = c.add_round()
    assert (r, chk.ok, c.active_rounds()) == (1, True, [1])
    assert len(V.load("p1")) == 468 and not any(k.endswith("_all") for k in V.load("p1"))
    c.add_round("3")
    assert c.iteration_rounds() == [1, 2, 2, 2]
    assert c.remove_round()[0] == 2 and c.iteration_rounds() == [1]
    flags = c.set_value("flags", "x", 1)                                    # type 3: no conversion in 1.1.514
    assert not flags.ok and "valueRightClass" in flags.report[0]
    assert c.set_value("localnc", "true", 1).ok                             # type 2: double(str2num) = 1
    assert V.load("p1")["localnc_r1"] == 1.0
    with pytest.raises(NotImplementedError):
        c.set_value("cone_range", "rand(1)", 1)
    assert c.set_value("sym", "c4", 2).ok and V.load("p1")["sym_r2"] == "c4"
    assert not c.environment_edit("cluster_header", "nofile.sh").ok
    ed, box, f = c.mask_editor()                     # ported (pydyn.masks): only the cylinder type
    assert (box, f) == ("mask", "p1/settings/editorMask.em")
    with pytest.raises(NotImplementedError):
        ed.save()                                    # default type 'sphere' is not ported
    with pytest.raises(NotImplementedError):
        c.template_average(fcompensate=True)
    assert c.enter_project_name("nope") == "not_a_project"
    assert c.enter_project_name("a b") == "invalid"


def test_template_average_cpu(tmp_path, monkeypatch):
    """'average all': the project data and table through daverage -> <name>/settings/computedAverage.em."""
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=3)
    c = DcpController()
    c.create("g1")
    c.data_enter("filamentsData_ctf")
    c.table_enter("particles_edit_mod.tbl")
    assert c.template_average(device="cpu").ok
    from pydyn.console import daverage
    oa = daverage("filamentsData_ctf", "t", "particles_edit_mod.tbl", device="cpu")
    assert np.array_equal(dynio.read_em("g1/settings/computedAverage.em"), oa.average.astype(np.float32))
    assert V.load("g1")["file_template_initial"] == "g1/settings/computedAverage.em"


# ---------------------------------------------------------------------------------------------------- Tk view
def test_gui_without_display(monkeypatch, capsys, tmp_path):
    import pydyn.gui as gui
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert gui.display_problem() is not None
    assert gui.main([]) == 2
    assert "no X display" in capsys.readouterr().out
    assert gui.open_project("x") is None
    monkeypatch.chdir(tmp_path)
    assert gui.main(["missing_project"]) == 1
    assert "Project NOT in disk" in capsys.readouterr().out


def test_dcp_command_without_display(tmp_path):
    """prototype/bin/dcp (and the installed `dcp`) start, restart in a clean environment and stop cleanly."""
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    py = os.path.join(P, ".envs", "pydynamo", "bin", "python")
    if not os.path.isfile(py):
        pytest.skip("project environment not present")
    r = subprocess.run([os.path.join(P, "prototype", "bin", "dcp")], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "no X display" in r.stdout, (r.stdout, r.stderr)


def test_tk_mask_editor_and_multireference_menu(tmp_path, monkeypatch):
    """The Tk view on a virtual X display (Xvnc/Xvfb; skipped without DISPLAY): masks window -> mask editor ->
    save -> Enter in the mask box; Multireference > Adaptive filtering > Derive a project; project box p1_eo; Edit
    parameters for Adaptive Filtering run (0.143 / 25 / 0)."""
    if not os.environ.get("DISPLAY"):
        pytest.skip("no X display")
    tk = pytest.importorskip("tkinter")
    from pydyn import gui, masks
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    from pydyn.dcptool import dcp
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    root = tk.Tk()
    try:
        app = gui.DcpApp(root, "p1")
        assert app.c.name == "p1" and app.v_nref.get() == "1"
        app.win_masks()
        mw = app.windows["masks"]
        editor_buttons = [w for w in mw.winfo_children()[0].winfo_children()
                          if isinstance(w, tk.ttk.LabelFrame) and w.cget("text") == "generate"][0]
        b = [w for w in editor_buttons.winfo_children() if isinstance(w, tk.ttk.Button)
             and w.cget("text") == "mask editor"][0]
        b.invoke()
        v = app.mask_editor_vars
        assert v["size"].get() == "64  64  64" and v["save"].get() == "p1/settings/editorMask.em"
        assert app.mask_rows["mask"][0].get() == "p1/settings/editorMask.em"
        v["type"].set("cylinder")
        for p, s in (("r", "15"), ("height", "30"), ("gaussian", "3")):
            v["radio"][p].set(True)
            v["edit"][p].set(s)
        app.windows["mask_editor"].mask_editor_save()
        assert os.path.isfile("p1/settings/editorMask.em")
        if masks.fft_backend() != "scipy":
            assert md5("p1/settings/editorMask.em") == "90f4bb01a9f2fc670bd758ea3aeb58dd"
        ent = app.mask_rows["mask"][1]
        ent.master.grid_slaves(row=0, column=3)[0].invoke()        # the row's "enter" button (= Return)
        assert V.load("p1")["file_mask"] == "p1/settings/editorMask.em"
        gs = app.menus["golden_standard"]
        assert [gs.entrycget(i, "label") for i in range(gs.index("end") + 1)] == [
            "Derive a project.", "Edit parameters for Adaptive Filtering run"]
        gs.invoke(1)                                     # single reference: refused with Dynamo's warning
        assert "abp" not in app.windows
        gs.invoke(0)
        assert V.isproject("p1_eo")
        app.v_project.set("p1_eo")
        app.enter_project()
        assert app.c.name == "p1_eo" and app.v_nref.get() == "2"
        gs.invoke(1)
        for n, s in (("adaptive_bandpass_threshold", "0.143"), ("adaptive_bandpass_initial", "25"),
                     ("adaptive_bandpass_pushback", "0")):
            var, commit = app.abp_vars[n]
            var.set(s)
            assert commit().ok
        vp = V.load("p1_eo")
        assert (vp["adaptive_bandpass_threshold"], vp["adaptive_bandpass_initial"],
                vp["adaptive_bandpass_pushback"]) == (0.143, 25.0, 0.0)
    finally:
        root.destroy()


def test_fmask_use_ones_multireference(tmp_path, monkeypatch):
    """masks window, fmask "use ones" on an even/odd (2-reference) project (dynamo_fourier_count_full.m): ones of the
    template size for every reference, and the fmask box gets their folder (fileparts), as Dynamo does."""
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    from pydyn.dcptool import dcp
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    c = DcpController()
    c.open("p1")
    eo = c.golden_standard_derive()
    assert eo == "p1_eo" and c.enter_project_name(eo) == "opened"
    r = c.masks_use_ones()
    assert r.ok and c.correct["fmask"] is True
    assert V.load("p1_eo")["file_fmask_initial"] == "./p1_eo/settings"
    for ref in (1, 2):
        f = dynio.read_em("p1_eo/settings/fmask_initial_ref_%03d.em" % ref)
        assert f.shape == (64, 64, 64) and np.all(f == 1)


def test_data_size_is_read_from_the_particles(tmp_path, monkeypatch):
    """The data window reads the box size from a particle's EM header (dynamo_info_data.m), nothing defaults to 64:
    40^3 particles give l = 40, and dcp.dataEnter sets dim_r1 = 40, low_r1 = round(40/3) = 13."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    for t in (1, 2, 3):
        dynio.write_em("data40/particle_%06d.em" % t, rng.normal(size=(40, 40, 40)).astype(np.float32))
    c = DcpController()
    c.create("p40")
    assert c.data_enter("data40").ok
    assert (c.data_facts["N"], c.data_facts["l"], c.data_facts["size"]) == (3, 40, [40, 40, 40])
    v = V.load("p40")
    assert (v["dim_r1"], v["low_r1"], v["ite_r1"]) == (40.0, 13.0, 1.0)
