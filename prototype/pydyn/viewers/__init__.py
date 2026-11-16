"""Dynamo 1.1.514's viewers for pyDynamo, in Qt (PySide6), each in its own process.

Console (names as in Dynamo; MATLAB-style name/value pairs work):

  from pydyn.viewers import dmapview, dview, dslices
  dmapview({'results/ite_0004/averages/average_ref_001_ite_0004.em',
            'results/ite_0004/averages/average_ref_002_ite_0004.em'})   # a set is sorted; a list keeps its order
  dmapview(vol, 'bin', 1)          # a numpy [z, y, x] volume, binned once
  dview('template.em')             # one volume, all slices, x/y/z buttons
  dslices('template.em', 'z', 'overlay', 'mask.em', 'overlay_as', 'mask')

Shell:  python -I -m pydyn.viewers mapview a.em b.em | view t.em | slices t.em --kind z --overlay m.em

dcp windows (Tk, pydyn.gui) call: view_file, overlay_mask, show_files, show_last_averages, dtplot,
plot_bandpass_fsc (see each docstring).

Why a separate process: Tk and Qt each need to own the main thread's event loop, and Qt widgets must live in the
thread that created QApplication, so the two toolkits cannot share one Python process safely. Each viewer is
therefore started as `python -I -m pydyn.viewers ...` with a clean environment (no PYTHONPATH / LD_LIBRARY_PATH /
QT_PLUGIN_PATH, so the login shell's other Qt cannot be loaded into PySide6), returns at once like a MATLAB figure,
and a crash of a viewer can never take the dcp windows (or their unsaved state) down. Volumes are passed as file
names; numpy arrays go through temporary .npy files that the viewer deletes after reading.

Spec and sources: docs/analysis/11a_viewers_1.1.514.md. This module imports neither Qt nor NumPy.
"""
import os
import subprocess
import sys
import tempfile

__all__ = ["dmapview", "dynamo_mapview", "dview", "dynamo_view", "dslices", "dynamo_slices", "dtplot",
           "dynamo_tableplot", "view_file", "overlay_mask", "show_files", "show_last_averages", "plot_bandpass_fsc",
           "last_completed_iteration", "qt_problem"]

UNSET = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "LD_LIBRARY_PATH", "LD_PRELOAD",
         "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH", "QML_IMPORT_PATH")
_PROTO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def qt_problem():
    """None if a viewer window can probably open, else the reason (like pydyn.gui.display_problem)."""
    plat = os.environ.get("QT_QPA_PLATFORM", "")
    if sys.platform.startswith("linux") and not plat.startswith(("offscreen", "minimal", "vnc")) and \
            not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return ("no X display: the DISPLAY variable is not set. Run from a desktop session (or ssh -X / a VNC "
                "session).")
    try:
        import importlib.util
        if importlib.util.find_spec("PySide6") is None:
            return "PySide6 is not installed in this Python (see requirements/pydynamo-lock.txt, GUI section)"
    except Exception as e:                                       # pragma: no cover
        return "PySide6 cannot be found (%s)" % e
    return None


def _clean_env():
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE="/dev/null")
    return env


def _launch(argv, arrays=(), wait=False):
    """Start `python -I -m pydyn.viewers <argv>` (numpy arrays are appended as --npy files). Returns the Popen, or
    None when no window can be opened (the reason is printed, as Dynamo prints its graphics errors)."""
    why = qt_problem()
    if why:
        print("[pydyn viewers] cannot open the window: %s" % why)
        return None
    argv = list(argv)
    if arrays:
        import numpy as np
        d = tempfile.mkdtemp(prefix="pydyn_view_")
        for i, a in enumerate(arrays):
            f = os.path.join(d, "volume_%02d.npy" % (i + 1))
            np.save(f, np.asarray(a))
            argv += ["--npy", f]
        argv += ["--delete-inputs"]
    code = ("import sys; sys.path.insert(0, %r); from pydyn.viewers.__main__ import main; sys.exit(main())"
            % _PROTO)
    p = subprocess.Popen([sys.executable, "-I", "-c", code] + argv, env=_clean_env(), cwd=os.getcwd(),
                         start_new_session=not wait)
    if wait:
        p.wait()
    return p


def _pairs(args, kw, known):
    """MATLAB-style name/value pairs plus Python keywords -> dict; unknown names raise like MATLAB's inputParser."""
    if len(args) % 2:
        raise ValueError("parameter/value arguments must come in pairs")
    out = dict(kw)
    for k, v in zip(args[0::2], args[1::2]):
        out[str(k)] = v
    for k in out:
        if k not in known:
            raise ValueError("'%s' is not a recognized parameter. For a list of valid name-value pair arguments, "
                             "see the documentation for this function." % k)
    return out


def _split(data):
    """files and arrays of a dmapview/dview input (a single item or a list/tuple/set of items)."""
    if isinstance(data, (set, frozenset)):          # MATLAB's {a, b} typed in Python is a set: sorted file names
        items = sorted(data, key=str)
    else:
        items = list(data) if isinstance(data, (list, tuple)) else [data]
    files, arrays = [], []
    for it in items:
        if isinstance(it, (str, bytes)) or hasattr(it, "__fspath__"):
            files.append(os.fspath(it))
        else:
            arrays.append(it)
    if files and arrays:
        raise NotImplementedError("pydyn viewers: mix of file names and arrays in one call")
    return files, arrays


# =====================================================================================================================
# console functions (Dynamo names)
# =====================================================================================================================

def dmapview(data, *args, wait=False, **kw):
    """dmapview (S/shorthands/dmapview.m -> dynamo_mapview.m:79 -> dpkvol.montage): one volume or a list of volumes
    of equal size (files or numpy [z, y, x] arrays). Flags: 'bin' (k successive 2x2x2 binnings), 'mask' (external
    mask, View > External mask), 'indices' (raises, as in Dynamo 1.1.514: montage.m:102 calls the undefined
    'intesect'). Opens with volume #1 only, as Dynamo; 'shown' / "Show all volumes" show the others."""
    q = _pairs(args, kw, ("bin", "mask", "indices"))
    if q.get("indices") not in (None, []):
        raise RuntimeError("Undefined function 'intesect' for input arguments of type 'double'.")
    files, arrays = _split(data)
    argv = ["mapview"] + files
    if q.get("bin"):
        argv += ["--bin", str(int(q["bin"]))]
    extra = []
    if q.get("mask") is not None:
        m = q["mask"]
        if isinstance(m, (str, bytes)) or hasattr(m, "__fspath__"):
            argv += ["--mask", os.fspath(m)]
        else:
            extra = [m]
            argv += ["--mask-is-last-npy"]
    return _launch(argv, arrays=list(arrays) + extra, wait=wait)


dynamo_mapview = dmapview


def dview(obj, *args, wait=False, **kw):
    """dview (S/shorthands/dview.m -> dynamo_view.m -> dynamo__view.m): one volume (file or array), or a list of
    files browsed with the previous/next icons and the List menu. Flag 'title' (window title 'dview: <title>').
    dynamo_view's 'bin' never works in 1.1.514 (dynamo_view.m:241 typo): it prints the same message here."""
    q = _pairs(args, kw, ("title", "bin"))
    if q.get("bin"):
        print("No binning could be performed")
    files, arrays = _split(obj)
    if "*" in "".join(files):
        import glob
        pattern = files[0]
        files = sorted(glob.glob(pattern))
        q.setdefault("title", pattern)
    argv = ["view"] + files
    title = q.get("title", files[0] if len(files) == 1 else "")
    if title:
        argv += ["--title", str(title)]
    return _launch(argv, arrays=arrays, wait=wait)


dynamo_view = dview


def dslices(vol, *args, wait=False, **kw):
    """dslices / dynamo_slices for the calls the dcp windows make: dslices(v, 'x'|'y'|'z') (one volume or a list
    shown side by side) and dslices(template, kind, 'overlay', mask, 'overlay_as', 'mask'[, 'alpha', a,
    'overlay_color', c]). Flags 'name' (window name) and 'title' (axes title). Other dynamo_slices options raise
    NotImplementedError."""
    kind = "z"
    if len(args) % 2 == 1:
        kind, args = str(args[0]), args[1:]
    if kind not in ("x", "y", "z"):
        raise NotImplementedError("dslices orientation %r (only 'x', 'y', 'z' are ported)" % kind)
    q = _pairs(args, kw, ("overlay", "overlay_as", "alpha", "overlay_color", "name", "title"))
    files, arrays = _split(vol)
    argv = ["slices"] + files + ["--kind", kind]
    extra = []
    if q.get("overlay") is not None:
        if q.get("overlay_as", "graymap") != "mask":
            raise NotImplementedError("dslices 'overlay_as' %r (only 'mask' is ported)" % q.get("overlay_as"))
        m = q["overlay"]
        if isinstance(m, (str, bytes)) or hasattr(m, "__fspath__"):
            argv += ["--overlay", os.fspath(m)]
        else:
            extra = [m]
            argv += ["--overlay-is-last-npy"]
        if "alpha" in q:
            argv += ["--alpha", str(float(q["alpha"]))]
        if "overlay_color" in q:
            c = q["overlay_color"]
            argv += ["--color", c if isinstance(c, str) else ",".join(str(float(x)) for x in c)]
    for k in ("name", "title"):
        if q.get(k):
            argv += ["--" + k, str(q[k])]
    return _launch(argv, arrays=list(arrays) + extra, wait=wait)


dynamo_slices = dslices


def dtplot(table, *args, wait=False, **kw):
    """dtplot / dynamo_tableplot with no flags (the table window's Inspect > Plot orientations,
    dynamo__wizard_table.m:422-454): the default 'rays' modus, a point per particle on the unit sphere seen from +z.
    A file name (the .tbl extension may be skipped) or an array. Flags ('profile', 'modus', 'color', ...) are not
    ported."""
    if args or kw:
        raise NotImplementedError("dtplot flags are not ported in pydyn (only the default 'rays' plot)")
    if isinstance(table, (str, bytes)) or hasattr(table, "__fspath__"):
        return _launch(["tableplot", os.fspath(table)], wait=wait)
    return _launch(["tableplot"], arrays=[table], wait=wait)


dynamo_tableplot = dtplot


def plot_bandpass_fsc(name, vpr=None, wait=False):
    """Main window Multireference > Adaptive filtering ("golden standard") > "Plot attained resolution FSC for all
    iterations" (dynamo_project_manager_easy.m:1245-1252 -> dpkconv.plotConvergenceAll): the bandpass_fsc curve of
    every completed iteration in one figure; a left click on a curve offers dmapview of that iteration's averages."""
    last = last_completed_iteration(name, vpr)
    if last == 0:
        print("Project does not contain completed iterations!")
        return None
    return _launch(["fsc", name, "--last", str(last)], wait=wait)


# =====================================================================================================================
# hooks for the dcp windows (pydyn.gui / pydyn.dcpcontrol)
# =====================================================================================================================

def view_file(path, title=None):
    """dview of one file: template window Inspect > "dview (gray scale browser)" (dynamo__wizard_template.m:388-406),
    masks window View mask > "dview" (dynamo__wizard_masks.m:446-456, 558-571), the mask editor's "view" buttons
    (dynamo_mask_GUI.m:826-841) and `ddb ... -v` of one volume."""
    return dview(path, "title", title if title is not None else path)


def overlay_mask(template, mask, kind, mkind="mask", ref=1):
    """Masks window View mask > "overlay x|y|z" (dynamo__wizard_masks.m:460-517): figure Name
    '<mkind> overlayed on  template (reference N, view along k)' and title(mask file name)."""
    name = "%s overlayed on  template (reference %d, view along %s)" % (mkind, int(ref), kind)
    title = mask if isinstance(mask, str) else None
    return dslices(template, kind, "overlay", mask, "overlay_as", "mask", "name", name, *(("title", title)
                                                                                          if title else ()))


def show_files(files):
    """+dpkproject/quickplot.m:25-31: one file -> dview, several -> dmapview."""
    files = list(files)
    if not files:
        print("No files to show")
        return None
    if len(files) == 1:
        return view_file(files[0])
    return dmapview(files)


def project_folder_of(name, vpr=None):
    """Where dynamo__last_completed_iteration.m:42-50 looks: fullfile(vpr.path, name_project) when the vpr has a
    'path' that is an existing folder, else (path missing on disk) the current folder; without a 'path' field the
    default path '.' of dynamo_database_locate. Dynamo uses vpr.path, not the folder the project is opened from."""
    if vpr is None:
        from .. import vpr as V
        vpr = V.load(name)
    path = "."
    if "path" in vpr:
        path = str(vpr["path"]) if os.path.isdir(str(vpr["path"])) else os.getcwd()
    return os.path.join(path, str(vpr.get("name_project", name)))


def last_completed_iteration(name, vpr=None):
    """dynamo__last_completed_iteration.m: the highest iteration with an average on disk (under vpr.path, see
    project_folder_of), minus one when that iteration has fewer averages than nref of its round."""
    import glob
    import re
    if vpr is None:
        from .. import vpr as V
        vpr = V.load(name)
    files = glob.glob(os.path.join(project_folder_of(name, vpr), "results", "ite_[0-9][0-9][0-9][0-9]", "averages",
                                   "average_ref_[0-9][0-9][0-9]_ite_[0-9][0-9][0-9][0-9].em"))
    ites = []
    for f in files:
        m = re.search(r"average_ref_(\d{3})_ite_(\d{4})\.em$", f)
        if m:
            ites.append(int(m.group(2)))
    if not ites:
        return 0
    last = max(ites)
    from ..dynproject import round_array
    rounds = round_array(vpr)
    if last > len(rounds):
        return 0
    expected = int(vpr["nref_r%d" % rounds[last - 1]])
    return last if ites.count(last) == expected else last - 1


def show_last_averages(name, vpr=None):
    """Main window View > "Last computed average(s)" (dynamo_project_manager_easy.m:1099-1101 -> dcp.quickPlot
    ('lastAverage') -> +dpkproject/quickplot.m:13-31): the averages of every reference at the last completed
    iteration; one -> dview, several (the two half-maps of an even/odd project) -> dmapview."""
    from .logic import last_average_files
    ite = last_completed_iteration(name, vpr)
    return show_files(last_average_files(project_folder_of(name, vpr), ite))
