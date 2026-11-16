"""python -I -m pydyn.viewers <viewer> ...   (the viewer process; see pydyn/viewers/__init__.py)

  mapview FILE [FILE ...] [--bin N] [--mask FILE]          dmapview: one or several volumes of equal size
  view FILE [FILE ...] [--title T]                          dview: one volume (several files: browse them)
  slices FILE [FILE ...] [--kind x|y|z] [--overlay MASK] [--alpha A] [--color r|R,G,B] [--name N] [--title T]
                                                            dslices: montage, side by side, or the mask overlay
  tableplot TABLE                                           dtplot: orientations of the particles ('rays')
  fsc PROJECT [--last N]                                    Multireference > Adaptive filtering > "Plot attained
                                                            resolution FSC for all iterations"
Shell commands (once in pyproject [project.scripts]): dmapview FILE ..., dview FILE ...
Common:  --snap PNG   draw, save the window(s) as PNG and quit (tests, screenshots; works with QT_QPA_PLATFORM=offscreen)
         --do a,b,..  with --snap: apply these actions in order and save PNG_<action> after each
                      (mapview: showall optimal full auto x y z next prev thick3 norm_slice norm_volume invert;
                       view: x y z)
"""
import argparse
import os
import sys

COLORS = {"r": (1.0, 0.0, 0.0), "g": (0.0, 1.0, 0.0), "b": (0.0, 0.0, 1.0), "y": (1.0, 1.0, 0.0),
          "m": (1.0, 0.0, 1.0), "c": (0.0, 1.0, 1.0), "w": (1.0, 1.0, 1.0), "k": (0.0, 0.0, 0.0)}   # dynamo__color2rgb


def _color(s):
    if s in COLORS:
        return COLORS[s]
    return tuple(float(x) for x in s.split(","))


def parse(argv):
    ap = argparse.ArgumentParser(prog="python -m pydyn.viewers", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("viewer", choices=("mapview", "view", "slices", "tableplot", "fsc"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--npy", action="append", default=[], help="numpy volume (temporary file from the console)")
    ap.add_argument("--delete-inputs", action="store_true", help="delete the --npy files (and their folder) once read")
    ap.add_argument("--bin", type=int, default=0)
    ap.add_argument("--mask")
    ap.add_argument("--mask-is-last-npy", action="store_true")
    ap.add_argument("--title")
    ap.add_argument("--name")
    ap.add_argument("--kind", default="z", choices=("x", "y", "z"))
    ap.add_argument("--overlay")
    ap.add_argument("--overlay-is-last-npy", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--color", default="r")
    ap.add_argument("--last", type=int, default=None, help="fsc: last completed iteration (default: computed)")
    ap.add_argument("--snap")
    ap.add_argument("--do", default="")
    return ap.parse_args(argv)


def _inputs(a):
    """file names and arrays; returns (names, volumes-or-None, extra array for --*-is-last-npy)."""
    import numpy as np
    arrays = [np.load(f) for f in a.npy]
    if a.delete_inputs:
        for f in a.npy:
            try:
                os.remove(f)
            except OSError:
                pass
        if a.npy:
            try:
                os.rmdir(os.path.dirname(a.npy[0]))
            except OSError:
                pass
    extra = None
    if (a.mask_is_last_npy or a.overlay_is_last_npy) and arrays:
        extra = arrays.pop()
    if a.files:
        return list(a.files), list(a.files), extra
    return ["<numeric variable>"] * len(arrays), arrays, extra


def _figures(a):
    """tableplot and fsc: figures that do not show volumes."""
    from PySide6 import QtWidgets
    from . import logic as L
    from . import qtview as Q
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName("pydyn " + a.viewer)
    if not a.files and not a.npy:
        print("[pydyn viewers] no %s given" % ("table" if a.viewer == "tableplot" else "project"))
        return 2
    if a.viewer == "tableplot":
        _, xyz = L.dtplot_points(a.files[0] if a.files else _inputs(a)[1][0])
        w = Q.FigureWindow(Q.TablePlotCanvas(xyz), "Figure")
    else:
        name = a.files[0].rstrip("/")
        last = a.last
        if last is None:
            from . import last_completed_iteration
            last = last_completed_iteration(name)
        if last == 0:
            print("Project does not contain completed iterations!")
            return 1
        w = Q.fsc_window(name, L.bandpass_fsc_curves(name, last))
    return Q.run([w], snap=a.snap)


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    if a.viewer in ("tableplot", "fsc"):
        return _figures(a)
    names, vols, extra = _inputs(a)
    if not vols:
        print("[pydyn viewers] no volume given")
        return 2
    from PySide6 import QtWidgets
    from . import logic as L
    from . import qtview as Q
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName("pydyn " + a.viewer)
    actions = []
    if a.viewer == "mapview":
        try:
            model = L.MapviewModel(vols, bin=a.bin, names=names)
        except ValueError as e:
            print("[dmapview] %s" % e)
            print("Could not initialize the data for the mapview montage")
            return 1
        mask = None
        if a.mask or extra is not None:
            mask = L.as_volume(L.read_volume(a.mask if a.mask else extra))
            if a.bin:
                mask = L.dynamo_bin(mask, a.bin)
            if mask.shape[2] != model.volumes[0].shape[2] or mask.shape[1] != model.volumes[0].shape[1]:
                print("Volumes and mask have different sizes.")       # montage.m:118-134 (x and y only)
                return 1
        w = Q.MapviewWindow(model, mask=mask)
        m = model
        table = {"showall": m.show_all_volumes, "optimal": m.optimal_layout, "full": m.contrast_full,
                 "auto": m.autocontrast, "x": lambda: m.set_orientation("x"), "y": lambda: m.set_orientation("y"),
                 "z": lambda: m.set_orientation("z"), "next": lambda: m.page(1), "prev": lambda: m.page(-1),
                 "thick3": lambda: m.set_thick(True, 3), "norm_slice": lambda: m.set_normalization(per_slice=True),
                 "norm_volume": lambda: m.set_normalization(per_volume=True), "invert": m.op_invert}
        windows = [w]
    elif a.viewer == "view":
        items = [(n, None if isinstance(v, str) else v) for n, v in zip(names, vols)]
        w = Q.DviewWindow(items, title=a.title or (names[0] if len(names) == 1 and names[0] != "<numeric variable>"
                                                   else ""))
        table = {k: (lambda k=k: w.show_axis(k)) for k in ("x", "y", "z")}
        windows = [w]
    else:
        vv = [L.read_volume(v) for v in vols]
        if a.overlay or extra is not None:
            mask = L.read_volume(a.overlay if a.overlay else extra)
            sc = L.dslices_overlay(vv[0], mask, a.kind, alpha=a.alpha, color=_color(a.color))
            img = sc["rgb"]
        else:
            sc = L.dslices_scene(vv, a.kind)
            c = sc["cdata"]
            import numpy as np
            img = L.to_gray(c, (float(np.nanmin(c)), float(np.nanmax(c))), 64)
        w = Q.SlicesWindow(img, sc["rows"], sc["cols"], sc["tile"], name=a.name or "dynamo_slices",
                           axes_title=a.title, nvol=sc.get("nvol", 1))
        table = {}
        windows = [w]
    if refresh := getattr(windows[0], "refresh", None):
        def wrap(fn):
            def run():
                fn()
                refresh()
            return run
    else:
        def wrap(fn):
            return fn
    for name in [x for x in a.do.split(",") if x]:
        if name not in table:
            print("[pydyn viewers] unknown action %r for %s" % (name, a.viewer))
            return 2
        actions.append((name, wrap(table[name])))
    return Q.run(windows, snap=a.snap, actions=actions)


def _clean_reexec(viewer):
    """Shell commands (pyproject [project.scripts]) start in the login shell's environment, whose LD_LIBRARY_PATH may
    hold another Qt 6 (PySide6's libraries use RUNPATH, which LD_LIBRARY_PATH overrides). Restart once with the clean
    environment of pydyn.viewers._launch."""
    from . import UNSET, _PROTO, _clean_env
    if os.environ.get("PYDYN_VIEWERS_CLEAN") == "1" or not any(v in os.environ for v in UNSET):
        return
    env = _clean_env()
    env["PYDYN_VIEWERS_CLEAN"] = "1"
    code = "import sys; sys.path.insert(0, %r); from pydyn.viewers.__main__ import main; sys.exit(main())" % _PROTO
    os.execve(sys.executable, [sys.executable, "-I", "-c", code, viewer] + sys.argv[1:], env)


def dmapview_main():
    """Shell command `dmapview a.em b.em [--bin N] [--mask m.em]`."""
    _clean_reexec("mapview")
    return main(["mapview"] + sys.argv[1:])


def dview_main():
    """Shell command `dview file.em [more.em ...]`."""
    _clean_reexec("view")
    return main(["view"] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
