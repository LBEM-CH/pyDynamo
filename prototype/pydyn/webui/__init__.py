"""The dcp GUI of pyDynamo: an HTML view (Bootstrap 5) in a Qt window (Qt WebEngine), on Dynamo 1.1.514's windows.

  dcp                         the project manager in the current folder
  dcp --p NAME                open ./NAME (as Dynamo's dcp(name): a missing project is only reported)
  dcp --p NAME --d DATA --template REF.em --masks default --t TABLE.tbl [--m F] [--cm F] [--fm F] [--sm F]
                              create it first (dcp.new with Dynamo's options and checks), then open it;
                              --no-gui creates only; --force replaces an existing project (dcp --help)
  dcp --serve [--p NAME]      no window: serve the view and print its address (for a browser on this machine)

From Python (from pydyn import *): dcp(), dcp.open('p') and dcp.new(...) open the same window in a separate
process, so the console stays free, as MATLAB's does with Dynamo's GUI.

All logic is in pydyn.dcpcontrol.DcpController; pydyn.webui.api only exposes its window actions to the view.
Spec: docs/analysis/08c_dcp_gui_1.1.514.md (windows), docs/DESIGN.md (requirements).
"""
import os
import subprocess
import sys


def display_problem():
    """None if a window can be opened, else the reason. An offscreen Qt platform does not count: `dcp` would then open
    an invisible window and wait (tests and screenshots call pydyn.webui.shell.run_window directly)."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return ("no X display: the DISPLAY variable is not set. Run 'dcp' from the remote desktop (or a VNC "
                "session); everything the GUI does is also available from Python: "
                "from pydyn import dcp, dvput, dvcheck, dvunfold")
    return None


def run(project=None, serve_only=False):
    """Serve a session on `project` and show it (blocks until the window is closed)."""
    from .server import serve
    if not serve_only:
        problem = display_problem()
        if problem:
            print("[pydyn dcp] cannot open the GUI: %s" % problem)
            return 2
        from .shell import webengine_problem
        problem = webengine_problem()
        if problem:
            print("[pydyn dcp] %s" % problem)
            return 2
    srv = serve(project)
    if serve_only:
        print("[pydyn dcp] serving %s  (open this address in a browser on this machine; Ctrl+C stops)"
              % srv.start_url, flush=True)
        import time
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            srv.shutdown()
            return 0
    from .shell import run_window
    return run_window(srv)


# dcp.new's options as command-line options: Dynamo's names and short forms (pydyn.dcptool._OPTIONS, from
# S/+dpkproject/+parsers/@ParserSeeds/ParserSeeds.m and S/@dcp/new.m). One-letter forms are -x / --x; longer ones --name.
_NEW_OPTIONS = (
    ("data", ("d", "data"), "FOLDER", "data folder of the particles"),
    ("table", ("t", "table"), "TABLE", "initial table (.tbl)"),
    ("template", ("template", "tmp", "te"), "FILE", "initial template (.em)"),
    ("masks", ("masks",), "default", "'default' (Dynamo's default): make the default masks"),
    ("mask", ("m", "mask", "alignment_mask"), "FILE", "alignment mask"),
    ("cmask", ("cm", "cmask", "classification_mask"), "FILE", "classification mask"),
    ("fmask", ("fm", "fmask", "fourier_mask"), "FILE", "Fourier mask"),
    ("smask", ("sm", "smask", "smoothing_mask"), "FILE", "smoothing mask"),
)
_LONG = {w for _, ws, _, _ in _NEW_OPTIONS for w in ws if len(w) > 1} | {
    "project", "force", "fo", "forceOverwrite", "gui", "no-gui", "serve", "help"}
EXAMPLES = """examples:
  dcp                                      the project manager in this folder
  dcp --p bin8_align_1                     open an existing project
  dcp --p bin8_align_1 --d filamentsData_ctf --template raw_template_ctf.em --masks default --t particles_edit_mod.tbl
                                           create it (dcp.new, Dynamo's checks) and open it
  dcp --p bin8_align_1 --d ... --no-gui    create it only
Dynamo's own spellings work too: -template, -masks, -d, -t, ... (a single dash before a long name is read as two)."""


def _normalise(argv):
    """Dynamo's single-dash long names (-template, -masks, -forceOverwrite) as --template, ...; -d / -t stay."""
    return ["-" + a if a.startswith("-") and not a.startswith("--") and a[1:] in _LONG else a for a in argv]


def _parser():
    import argparse
    ap = argparse.ArgumentParser(prog="dcp", description="The dcp window of pyDynamo (Dynamo 1.1.514's project "
                                 "manager): open a project, or create one with dcp.new and open it.",
                                 epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help=argparse.SUPPRESS)          # dcp NAME, as Dynamo's dcp(name)
    ap.add_argument("-p", "--p", "--project", dest="project", metavar="NAME", help="the project (a folder here)")
    g = ap.add_argument_group("create the project first (dcp.new)")
    for dest, words, metavar, help_ in _NEW_OPTIONS:
        flags = [f for w in words for f in (("-" + w, "--" + w) if len(w) == 1 else ("--" + w,))]
        g.add_argument(*flags, dest=dest, metavar=metavar, default=None, help=help_)
    g.add_argument("--force", "--fo", "--forceOverwrite", dest="force", nargs="?", const=1, default=0, type=int,
                   metavar="1", help="replace an existing project of that name (Dynamo's forceOverwrite)")
    g.add_argument("--gui", dest="gui", nargs="?", const=1, default=1, type=int, choices=(0, 1), metavar="0|1",
                   help="open the window after creating (default 1)")
    g.add_argument("--no-gui", dest="gui", action="store_const", const=0, help="create only, do not open the window")
    o = ap.add_argument_group("other")
    o.add_argument("--serve", action="store_true", help="no window: serve the view and print its address")
    return ap


def main(argv=None):
    """The `dcp` command (dcp --help)."""
    argv = _normalise(sys.argv[1:] if argv is None else list(argv))
    a = _parser().parse_args(argv)
    if a.project and a.name and a.project != a.name:
        print("[pydyn dcp] two project names given: %s and %s" % (a.name, a.project))
        return 2
    name = a.project or a.name
    new_kw = {dest: getattr(a, dest) for dest, _, _, _ in _NEW_OPTIONS if getattr(a, dest) is not None}
    if a.force:
        new_kw["forceOverwrite"] = 1
    if not name:
        if new_kw:
            print("[pydyn dcp] give the project name: dcp --p NAME --d DATA ...")
            return 2
        return run(None, a.serve)
    if name == "." or "*" in name:
        print("[pydyn dcp] project search patterns (dynamo_projects 'regexp') are not ported; "
              "open 'dcp' and use the project list")
        return 2
    from .. import vpr as V
    if new_kw:
        from ..dcptool import dcp
        from ..vpr import DynamoError
        try:
            vpr = dcp.new(name, gui=0, **new_kw)
        except DynamoError as e:
            print("[pydyn dcp] %s" % e)
            return 1
        if vpr is None:                              # it exists already: Dynamo printed how to open or overwrite it
            return 1
        return run(name, a.serve) if a.gui else 0
    if not V.isproject(name):
        print('Project NOT in disk: "%s". Creating a new one.' % name)
        print("[pydyn dcp] as in Dynamo 1.1.514, dcp(name) does not create it: give the inputs, "
              "dcp --p %s --d DATA --template REF.em --masks default --t TABLE.tbl, or type the name in the window"
              % name)
        return 1
    print('Project exist in disk: "%s". Opening it.' % name)
    return run(name, a.serve)


def launch(project=None):
    """Open the GUI in a separate process (returns at once): dcp(), dcp.open and dcp.new from the Python console.
    Returns the Popen object, or None when no window can be opened (the reason is printed)."""
    problem = display_problem()
    if problem:
        print("[pydyn dcp] cannot open the GUI: %s" % problem)
        return None
    from ..execute import UNSET
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE="/dev/null", PYDYN_CLEAN_ENV="1")
    proto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = "import sys; sys.path.insert(0, %r); from pydyn.webui import main; sys.exit(main())" % proto
    args = [sys.executable, "-I", "-c", code] + ([project] if project else [])
    return subprocess.Popen(args, cwd=os.getcwd(), env=env, start_new_session=True)
