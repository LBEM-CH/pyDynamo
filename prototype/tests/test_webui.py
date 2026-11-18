"""The dcp window of pydyn.webui: its API replays Dynamo's own dcp journey (run G), its server refuses strangers, the
`dcp` command and the `pydynamo` console start, and the HTML view runs in Qt WebEngine without script errors.

The journey test is test_dcpgui's run-G replay with every controller call sent through the HTTP API (the calls the
HTML view makes), compared with the same fixtures of Dynamo 1.1.514's own windows.
"""
import json
import os
import subprocess
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from types import SimpleNamespace

import pytest

from conftest import CHECK_CLEAN

from pydyn.webui.server import serve

from test_dcpgui import FIX, _journey, _same_except_path, _vpr_same_except_path, fix
from test_stage3 import make_data

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = os.path.join(P, ".envs", "pydynamo", "bin", "python")


TUPLE_RESULTS = {"unfold", "rounds_table", "add_round", "remove_round", "unfold_and_hint"}   # tuples in DcpController


class _Obj(SimpleNamespace):
    """A Check / CheckResult as returned through JSON: attributes, and truth = ok (Check.__bool__)."""

    def __bool__(self):
        return bool(getattr(self, "ok", True))


def _obj(x):
    if isinstance(x, dict) and "__type__" in x:
        return _Obj(**{k: v for k, v in x.items() if k != "__type__"})
    return x


class ApiController:
    """The HTML view's API seen as a DcpController: every method is a POST /api/call, state comes from /api/state."""

    def __init__(self, srv):
        self.srv = srv
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        self.op.open(srv.start_url).read()

    def _state(self):
        return json.loads(self.op.open(self.srv.url + "api/state").read())

    @property
    def enabled(self):
        return self._state()["enabled"]

    @property
    def runtime(self):
        return self._state()["fields"]

    @property
    def semiaxis(self):
        return self._state()["semiaxis"]

    def call(self, method, *args, **kwargs):
        req = urllib.request.Request(self.srv.url + "api/call",
                                     json.dumps({"method": method, "args": args, "kwargs": kwargs}).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(self.op.open(req).read())
        assert r["ok"], (method, r)
        res = r["result"]
        if method in TUPLE_RESULTS and isinstance(res, list):
            return tuple(_obj(v) for v in res)
        return _obj(res)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **kw: self.call(name, *a, **kw)


@pytest.fixture
def server():
    srv = serve()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_api_journey_matches_dynamo_run_g(tmp_path, monkeypatch, server):
    """test_dcpgui.test_gui_journey_matches_dynamo_run_g with every call made through the HTML view's API."""
    c, out = _journey(tmp_path, monkeypatch, make=lambda: ApiController(server))
    for k in ("buttons_after_create", "masks_window_state", "masks_after_defaults", "buttons_after_masks",
              "rounds_table_initial", "rounds_table_advanced", "environment_window_state", "buttons_before_check",
              "buttons_after_check"):
        assert out[k] == fix("dynamo_%s.txt" % k), k
    _same_except_path(out["rounds_table_general"], fix("dynamo_rounds_table_general.txt"))
    ref = json.load(open(os.path.join(FIX, "dynamo_mask_md5.json")))
    for f, h in out["masks_md5"].items():
        assert h == ref["snapG5b_defaults/settings/" + f], f
    _vpr_same_except_path(out["vpr_after_rounds"],
                          json.load(open(os.path.join(FIX, "dynamo_vpr_g1_after_rounds.json"))))
    r = out["check"]
    assert (r.report, r.fields_warning, r.fields_error, r.warning_state) == CHECK_CLEAN
    assert out["unfold"] == ("g1.exe", "ok")
    _vpr_same_except_path(out["vpr_after_unfold"],
                          json.load(open(os.path.join(FIX, "dynamo_vpr_g1_after_unfold.json"))))
    mine = open("g1/cards/ite_0001/card_ite_ite_0001.card").read().splitlines()
    ref_card = fix("dynamo_g1_card_ite_ite_0001.card")
    assert [x for x in mine if not x.startswith("path=")] == [x for x in ref_card if not x.startswith("path=")]
    assert c.iteration_rounds() == [1, 1, 2, 2]
    assert c.execution_order() == "./g1.exe"
    texts = [m["text"] for m in json.loads(c.op.open(server.url + "api/messages?after=0").read())]
    said = 'No errors found in project : "g1"' if not CHECK_CLEAN[0] else CHECK_CLEAN[0][0]   # no CuPy: the warning
    assert any(said in t for t in texts)                                     # the main window's message list


def test_api_view_actions_and_previews(tmp_path, monkeypatch, server):
    """masks window closing, mask editor (cylinder, saved, then entered), preview PNG, run hint, file listing."""
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    from pydyn.dcptool import dcp
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    c = ApiController(server)
    assert c.enter_project_name("p1") == "opened"
    assert c.masks_closed() is True
    ed = c.editor_open("align")
    assert (ed["box"], ed["file"]) == ("mask", "p1/settings/editorMask.em")
    form = {"size": ed["size"], "type": "cylinder", "invert": False, "save_file": ed["file"],
            "radio": {"r": True, "height": True, "gaussian": True}, "edit": {"r": "15", "height": "30", "gaussian": "3"}}
    live = c.editor_slices(form, "y", None)                     # the live view, before saving: nothing written
    assert live["live"] and live["k"] == 33 and "m" in live and "t" in live and not os.path.exists("p1/settings/editorMask.em")
    c.editor_save(form)
    assert os.path.isfile("p1/settings/editorMask.em")
    import base64
    import numpy as np
    from pydyn.console import dread
    from pydyn.viewers.logic import view_tile
    shown = np.frombuffer(base64.b64decode(live["m"]), np.float32).reshape(live["h"], live["w"])
    saved = np.asarray(dread("p1/settings/editorMask.em"), np.float32)
    assert np.array_equal(shown, view_tile(saved, "y", 33).astype(np.float32))   # the view shows what Save writes
    bad = c.editor_slices(dict(form, edit={"r": "x", "height": "30", "gaussian": "3"}), "z", None)
    assert "t" in bad and ("error" in bad or "do not make a mask" in bad.get("note", ""))   # reported, no crash
    assert c.mask_enter("p1/settings/editorMask.em").ok
    d = json.loads(c.op.open(server.url + "api/slices?which=mask&axis=y").read())
    assert d["k"] == 33 and d["n"] == 64 and d["mask_file"] == "p1/settings/editorMask.em"
    assert 0 < d["mstats"]["n_soft"] and d["mstats"]["max"] <= 1.0          # a soft (Gaussian-edged) cylinder
    gpus = json.loads(c.op.open(server.url + "api/gpus").read())
    assert isinstance(gpus, list) and all({"index", "name", "mem_total"} <= set(g) for g in gpus)
    assert c.data_enter("filamentsData_ctf").ok
    df = c._state()["data_facts"]                                 # summary only: no per-particle lists in the view
    assert df == {"N": 4, "l": 64, "size": [64, 64, 64], "pattern": "particle_######.em",
                  "first": "particle_000001.em", "last": "particle_000004.em"}, df
    listing = json.loads(c.op.open(server.url + "api/fs?path=.").read())
    assert listing["dirs"] == ["filamentsData_ctf", "p1"] and "raw_template_ctf.em" in listing["files"]
    assert c.run_hint()["order"] == "run p1"                               # Dynamo's order for destination matlab
    assert c.set_destination("standalone_gpu").ok
    assert c.run_hint()["order"] == "./p1.exe"


def test_server_refuses_strangers(server):
    def code(url, headers=None, data=None):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data, headers or {})).status
        except urllib.error.HTTPError as e:
            return e.code
    assert code(server.url + "api/state") == 403                          # no token cookie
    assert code(server.url + "?t=wrong") == 403
    tok = {"Cookie": "pydyn_t=" + server.token}
    assert code(server.url + "api/state", tok) == 200
    assert code(server.url + "api/state", dict(tok, Host="evil.example:80")) == 403   # DNS rebinding
    assert code(server.url + "static/../server.py", tok) == 404
    body = json.dumps({"method": "_reset", "args": []}).encode()
    assert code(server.url + "api/call", dict(tok, **{"Content-Type": "application/json"}), body) == 400
    body = json.dumps({"method": "__init__", "args": []}).encode()
    assert code(server.url + "api/call", dict(tok, **{"Content-Type": "application/json"}), body) == 400


def test_dcp_shell_one_line_start(tmp_path, monkeypatch, capsys):
    """dcp --p NAME --d DATA --template REF --masks default --t TABLE: dcp.new with Dynamo's option names and checks;
    Dynamo's single-dash spellings (-template, -masks, -gui 0) and the bare name (dcp NAME) still work."""
    from pydyn import vpr as V
    from pydyn.webui import main
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    rc = main(["--p", "p2", "--d", "filamentsData_ctf", "--template", "raw_template_ctf.em", "--masks", "default",
               "--t", "particles_edit_mod.tbl", "--no-gui"])
    assert rc == 0 and V.isproject("p2")
    v = V.load("p2")
    assert (v["folder_data"], v["file_table_initial"], v["file_template_initial"]) == (
        "filamentsData_ctf", "particles_edit_mod.tbl", "raw_template_ctf.em")
    assert main(["--project", "p2", "--data", "filamentsData_ctf", "--no-gui"]) == 1   # exists: Dynamo's message
    assert "cannot use" in capsys.readouterr().out
    assert main(["--p", "p2", "--d", "filamentsData_ctf", "--force", "--no-gui"]) == 0  # forceOverwrite
    rc = main(["p3", "-d", "filamentsData_ctf", "-template", "raw_template_ctf.em", "-masks", "default",
               "-t", "particles_edit_mod.tbl", "-gui", "0"])                           # Dynamo's spellings
    assert rc == 0 and V.load("p3")["file_template_initial"] == "raw_template_ctf.em"
    assert main(["--p", "p4", "--d", "nofolder", "--no-gui"]) == 1
    assert "invalid data" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--p", "p5", "--bogus", "x"])                                     # unknown option: argparse error
    assert main(["--d", "filamentsData_ctf"]) == 2                               # no project name
    assert main(["--p", "missing"]) == 1 and "Project NOT in disk" in capsys.readouterr().out
    assert main(["--p", "a", "b"]) == 2                                          # two names


def test_pydynamo_console(tmp_path):
    """`pydynamo` without arguments: a Python console with Dynamo's commands, where dcp.new(...) can be typed."""
    exe = os.path.join(P, ".envs", "pydynamo", "bin", "pydynamo")
    if not os.path.isfile(exe):
        pytest.skip("project environment not present")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    r = subprocess.run([exe], input="print(type(dcp).__name__, callable(daverage))\n", cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "pyDynamo console" in r.stderr and "_Dcp True" in r.stdout


def test_html_view_runs_in_qt_webengine(tmp_path):
    """The real window (offscreen): open a project, visit every step, run check; no JavaScript error."""
    import importlib.util
    if importlib.util.find_spec("PySide6") is None or importlib.util.find_spec("PySide6.QtWebEngineWidgets") is None:
        pytest.skip("Qt WebEngine (PySide6-Addons) not installed")
    if not os.path.isfile(PY):
        pytest.skip("project environment not present")
    make_data(str(tmp_path), N=4)
    js = ("(async () => { for (const s of ['data','table','template','masks','settings','destination','check',"
          "'unfold','run']) { const el = document.querySelector('[data-step='+s+']'); if (el) { el.click(); "
          "await new Promise(r => setTimeout(r, 350)); } } document.querySelector('[data-step=check]').click(); "
          "await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=check]').click(); })()")
    code = ("import sys; sys.path[:0] = [%r, %r]\n"
            "from pydyn.dcptool import dcp\n"
            "dcp.new('p1', 'd', 'filamentsData_ctf', 'template', 'raw_template_ctf.em', 'masks', 'default', 't', "
            "'particles_edit_mod.tbl', 'gui', 0)\n"
            "from pydyn.webui.server import serve\nfrom pydyn.webui.shell import run_window\n"
            "srv = serve('p1')\nrun_window(srv, snap='view.png', script=%r, settle_ms=8000)\n"
            "print('MESSAGES', [m['text'] for m in srv.session.messages()])\n"
            % (os.path.join(P, "prototype"), HERE, js))
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH")}
    env["QT_QPA_PLATFORM"] = "offscreen"
    r = subprocess.run([PY, "-I", "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                       timeout=240)
    assert r.returncode == 0, r.stderr[-3000:]
    assert os.path.getsize(tmp_path / "view.png") > 20000
    errors = [ln for ln in r.stderr.splitlines() if ln.startswith("[dcp view]")]
    assert errors == [], errors
    assert "Checking project consistency" in r.stdout


def test_mask_inspector_tiles_are_dynamo_mapview_tiles(tmp_path, monkeypatch, server):
    """/api/slices gives the template and mask tiles exactly as Dynamo's mapview draws them (viewers.logic.view_tile),
    for the project's mask or for a file given by the view (the mask editor's output before it is entered)."""
    import base64
    import numpy as np
    from pydyn import dynio
    from pydyn.console import dread
    from pydyn.dcptool import dcp
    from pydyn.viewers.logic import view_tile
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    c = ApiController(server)
    assert c.enter_project_name("p1") == "opened"
    t = np.asarray(dread("raw_template_ctf.em"), np.float32)
    m = np.asarray(dread(c.runtime["file_mask"]), np.float32)
    get = lambda q: json.loads(c.op.open(server.url + "api/slices?" + q).read())
    tile = lambda d, key: np.frombuffer(base64.b64decode(d[key]), np.float32).reshape(d["h"], d["w"])
    for axis in "zyx":
        d = get("which=mask&axis=%s&k=5" % axis)
        assert d["k"] == 5 and "error" not in d
        assert np.array_equal(tile(d, "t"), view_tile(t, axis, 5).astype(np.float32))
        assert np.array_equal(tile(d, "m"), view_tile(m, axis, 5).astype(np.float32))
    assert get("which=mask&axis=z")["k"] == 33                          # default: the central frame
    other = np.zeros((64, 64, 64), np.float32)
    other[20:40, 10:30, 30:50] = 0.75
    dynio.write_em("other_mask.em", other)
    d = get("which=mask&axis=x&k=40&mfile=other_mask.em")
    assert d["mask_file"] == "other_mask.em" and np.array_equal(tile(d, "m"), view_tile(other, "x", 40).astype(np.float32))
    dynio.write_em("small.em", np.ones((32, 32, 32), np.float32))
    d = get("which=mask&axis=z&mfile=small.em")
    assert "size mismatch" in d["error"] and "t" in d and "m" not in d   # the template is still shown


def test_file_chooser_lists_every_entry_fast(tmp_path):
    """A data folder can hold 100,000 particles: the listing returns every entry (no cap), folders and files apart,
    and reads sizes only for small folders (one stat per file is slow on NFS)."""
    from pydyn.webui.api import Session
    big = tmp_path / "particles"
    big.mkdir()
    (big / "sub").mkdir()
    for t in range(1, 6001):
        (big / ("particle_%06d.em" % t)).touch()
    d = Session.listdir(str(big))
    assert d["dirs"] == ["sub"] and len(d["files"]) == 6000 and d["sizes"] is None
    assert d["files"][0] == "particle_000001.em" and d["files"][-1] == "particle_006000.em"
    small = Session.listdir(str(tmp_path))
    assert small["dirs"] == ["particles"] and small["files"] == [] and small["sizes"] == []


def test_mask_editor_keeps_its_mask_and_settings(tmp_path, monkeypatch):
    """Create mask writes (and overwrites) a log next to the mask, <mask>.log, with the settings that made it.
    Reopening the editor brings back this session's settings, or in a new session the settings of the log; Load of a
    mask with a log fills in its settings; a mask changed after its log keeps the settings with a warning; a mask
    without a log is shown as it is. The page knows the editor's mask for its 'from editor' buttons."""
    import shutil
    import numpy as np
    from pydyn import dynio
    from pydyn.dcptool import dcp
    from pydyn.webui.api import mask_log_path, read_mask_log
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    cyl = {"size": "64  64  64", "type": "cylinder", "invert": False, "save_file": "p1/settings/editorMask.em",
           "radio": {"r": True, "height": True, "gaussian": True}, "edit": {"r": "15", "height": "50", "gaussian": "3"}}
    log = "p1/settings/editorMask.em.log"
    srv = serve()
    try:
        c = ApiController(srv)
        assert c.enter_project_name("p1") == "opened"
        e = c.editor_open("align")
        assert e["source"] == "default" and not e["exists"] and e["log"] is None and c._state()["editor_file"] is None
        c.editor_slices(cyl, "z", None)                              # typed, not created: no log yet
        assert not os.path.exists(log)
        e = c.editor_open("align")                                   # reopened in the same session
        assert e["source"] == "session" and (e["type"], e["edit"]["r"], e["edit"]["height"]) == ("cylinder", "15", "50")
        e = c.editor_save(cyl)                                       # Create mask: the mask and its log
        assert e["exists"] and e["log"] == log and mask_log_path(cyl["save_file"]) == log
        text = open(log).read()
        assert "type = cylinder" in text and "r = 15" in text and "height = 50" in text and "gaussian = 3" in text
        assert "sx =" not in text                                    # only the ticked settings
        e = c.editor_save(dict(cyl, edit=dict(cyl["edit"], r="12")))   # edited and created again: overwritten
        assert "r = 12" in open(log).read() and "r = 15" not in open(log).read()
        e = c.editor_save(cyl)
        assert c.mask_enter(c._state()["editor_file"]).ok            # what the 'from editor' button does
        info = c._state()["editor_info"]                             # shown on the masks page, editor closed
        assert info["file"] == "p1/settings/editorMask.em" and info["used_as"] == ["alignment"] and info["log_matches"]
        assert info["settings"] == "cylinder \u00b7 r 15 \u00b7 height 50 \u00b7 gaussian 3"
    finally:
        srv.shutdown(); srv.server_close()
    srv = serve()                                                    # a new dcp window: the log fills the editor
    try:
        c = ApiController(srv)
        assert c.enter_project_name("p1") == "opened"
        e = c.editor_open("align")
        assert e["source"] == "log" and e["exists"] and e["type"] == "cylinder" and e["log_matches"] is True
        assert (e["edit"]["r"], e["edit"]["height"], e["edit"]["gaussian"], e["radio"]["r"]) == ("15", "50", "3", True)
        shutil.copy("p1/settings/editorMask.em", "copied_mask.em")    # a mask carried with its log
        shutil.copy(log, "copied_mask.em.log")
        e = c.editor_load("copied_mask.em")
        assert e["source"] == "log" and e["edit"]["height"] == "50" and e["log_matches"] is True
        other = np.zeros((64, 64, 64), np.float32)
        other[20:44, 20:44, 20:44] = 1
        dynio.write_em("foreign_mask.em", other)                     # no log: shown as it is
        e = c.editor_load("foreign_mask.em")
        assert e["source"] == "file" and e["log"] is None and not any(e["radio"].values())
        live = c.editor_slices({k: e[k] for k in ("size", "type", "radio", "edit", "invert", "save_file")}, "z", None)
        assert "showing the file foreign_mask.em" in live["note"] and "m" in live
        st = c._state()
        assert st["editor_file"] == "foreign_mask.em" and st["editor_info"]["settings"] is None
    finally:
        srv.shutdown(); srv.server_close()
    dynio.write_em("p1/settings/editorMask.em", other)               # the mask changed after its log
    rec = read_mask_log("p1/settings/editorMask.em")
    assert rec["matches"] is False and rec["edit"]["r"] == "15"     # settings kept, flagged
    srv = serve()
    try:
        c = ApiController(srv)
        c.enter_project_name("p1")
        e = c.editor_open("align")
        assert e["source"] == "log" and e["log_matches"] is False
    finally:
        srv.shutdown(); srv.server_close()


def test_window_runs_projects_on_gpus(tmp_path, monkeypatch, server):
    """The window has no CPU / MATLAB / cluster choice: unfold sets Dynamo's destination standalone_gpu (what
    pyDynamo runs) when the project has another one, e.g. Dynamo's default 'matlab' of a new project."""
    from pydyn import vpr as V
    from pydyn.dcptool import dcp
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    assert V.load("p1")["destination"] == "matlab"                       # Dynamo's default
    c = ApiController(server)
    assert c.enter_project_name("p1") == "opened"
    assert c.set_value("ite", "1", 1).ok                                   # dcp.new leaves 0 iterations
    script, status = c.unfold_and_hint()
    assert (script, status) == ("p1.exe", "ok") and V.load("p1")["destination"] == "standalone_gpu"
    assert c.execution_order() == "./p1.exe" and c.use_gpus() == "standalone_gpu"
    texts = [m["text"] for m in json.loads(c.op.open(server.url + "api/messages?after=0").read())]
    assert any("destination set to standalone_gpu" in t for t in texts)


def test_unfolded_project_stays_unfolded_when_reopened(tmp_path, monkeypatch):
    """The unfold state is read from the project's files (the .exe is written after the parameters): reopened, an
    unfolded project keeps its unfold and run steps open; a parameter changed afterwards makes it stale."""
    import time
    from pydyn.dcptool import dcp
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    srv = serve()
    try:
        c = ApiController(srv)
        assert c.enter_project_name("p1") == "opened"
        assert c._state()["unfold_info"] is None
        assert c.set_value("ite", "1", 1).ok
        assert c.environment_edit("gpu_identifier_set", "0 1").ok
        assert c.unfold_and_hint()[1] == "ok"
    finally:
        srv.shutdown(); srv.server_close()
    srv = serve()                                                    # the dcp window opened again
    try:
        c = ApiController(srv)
        assert c.enter_project_name("p1") == "opened"
        st = c._state()
        assert st["unfold_info"]["exe"] == "p1.exe" and st["unfold_info"]["current"] is True
        assert st["enabled"]["unfold"] and st["enabled"]["run"]
        assert c.check().fields_error == [] and c._state()["unfold_info"]["current"] is True   # the check on opening
        vpr = os.path.join("p1", "settings", "virtual_project.mat")
        t = os.stat(vpr).st_mtime_ns
        assert c.environment_edit("gpu_identifier_set", "0 1").ok     # the value it has: the file is left as it is
        assert c.enter_project_name("p1") == "opened"                  # opened again in the same window
        assert os.stat(vpr).st_mtime_ns == t and c._state()["unfold_info"]["current"] is True
        time.sleep(0.05)
        assert c.set_value("cone_range", "20", 1).ok                 # changed after the unfold
        assert c._state()["unfold_info"]["current"] is False
    finally:
        srv.shutdown(); srv.server_close()


SINFO_CLUSTER = """\
gpu-a.cluster.example|app-one|up|gpu:4|Ampere
gpu-a.cluster.example|app-two|up|gpu:4|Ampere
gpu-a.cluster.example|gpus|up|gpu:4|Ampere
gpu-a.cluster.example|gpu-a|up|gpu:4|Ampere
gpu-b.cluster.example|app-one|up|gpu:4|Ampere
gpu-b.cluster.example|app-two|up|gpu:4|Ampere
gpu-b.cluster.example|gpus|up|gpu:4|Ampere
gpu-b.cluster.example|gpu-b|up|gpu:4|Ampere
gpu-old.cluster.example|gpu-old|up|gpu:4|Turing
gpu-x.cluster.example|gpu-x|up|gpu:4|Ampere
pc-1.cluster.example|pc*|up|gpu:2|(null)
pc-2.cluster.example|pc*|up|gpu:1|Ada
"""


def test_slurm_plan_sends_jobs_to_the_shared_machines():
    """The Slurm line names only GPUs, CPUs and memory: the partitions are the shared ones (several machines) whose
    machines have that many GPUs, one per set of machines, so Slurm picks the machine (earliest start)."""
    from pydyn.webui.api import parse_sinfo, slurm_plan
    parts = parse_sinfo(SINFO_CLUSTER)
    assert {p["name"]: p["default"] for p in parts}["pc"] is True
    four = slurm_plan(parts, 4)
    assert four["partitions"] == ["gpus"]                       # not its twins, not one machine's partition
    assert four["machines"] == ["gpu-a", "gpu-b"] and four["features"] == ["Ampere"]
    two = slurm_plan(parts, 2)
    assert two["partitions"] == ["gpus", "pc"] and "pc-2" not in two["machines"]
    assert "pc-1" in two["machines"]
    five = slurm_plan(parts, 5)
    assert five["partitions"] == [] and five["machines"] == [] and five["most_gpus"] == 4
    assert slurm_plan([], 1)["slurm"] is False


def test_execute_gpus_all_and_read_threads(monkeypatch):
    """`dynamo_execute_project NAME --gpus all` (the Slurm line when the job's GPUs differ from the project's) hands
    'all' to execute_project; the averaging reads particles with at most the CPUs this process may use."""
    from pydyn import cli, execute
    from pydyn.average import read_threads
    seen = {}
    monkeypatch.setattr(execute, "execute_project", lambda name, gpus=None, **kw: seen.update(name=name, gpus=gpus) or True)
    assert cli.execute_main(["p1", "--gpus", "all"]) == 0 and seen == {"name": "p1", "gpus": "all"}
    assert cli.execute_main(["p1", "--gpus", "2,3"]) == 0 and seen["gpus"] == [2, 3]
    assert read_threads() == max(1, min(8, len(os.sched_getaffinity(0))))
    assert read_threads(most=1) == 1


def test_vpr_save_without_change_leaves_the_file(tmp_path, monkeypatch):
    """Saving parameters that are already on disk does not rewrite virtual_project.mat (only the MAT header's date
    would differ), so its file time keeps meaning "parameters changed"; a real change is written."""
    from pydyn import vpr as V
    from pydyn.dcptool import dcp
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    f = os.path.join("p1", "settings", "virtual_project.mat")
    v = V.read_vpr_file(f)
    before = (os.stat(f).st_mtime_ns, open(f, "rb").read())
    import time
    time.sleep(0.02)
    V.write_vpr_file(f, v)
    assert (os.stat(f).st_mtime_ns, open(f, "rb").read()) == before
    V.write_vpr_file(f, V.modify(v, "cone_range_r1", 20.0))
    assert os.stat(f).st_mtime_ns != before[0] and V.read_vpr_file(f)["cone_range_r1"] == 20.0
