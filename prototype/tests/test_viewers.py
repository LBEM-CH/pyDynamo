"""pydyn.viewers: Dynamo 1.1.514's dmapview, dview and the masks-window dslices overlay.

* Logic (pure NumPy, no display): layouts, orientation of the tiles, slabs, intensity scaling, the mapview state.
* Ground truth from Dynamo 1.1.514 itself on the user's ABP data (benchmarks/08_viewers/scripts/dynamo_viewers_screens.m,
  log benchmarks/08_viewers/dynamo_screens/dynamo_run.log): CLim values, labels, layouts (skipped without test_data).
* Qt windows offscreen (QT_QPA_PLATFORM=offscreen), in this process and through the separate-process launcher
  (skipped without PySide6).
Spec: docs/analysis/11a_viewers_1.1.514.md.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from pydyn import dynio
from pydyn.viewers import logic as L

P = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TD = os.path.join(P, "test_data", "bin8_filament")
H1 = os.path.join(TD, "abp_align_eo", "results", "ite_0004", "averages", "average_ref_001_ite_0004.em")
H2 = os.path.join(TD, "abp_align_eo", "results", "ite_0004", "averages", "average_ref_002_ite_0004.em")
TPL = os.path.join(TD, "abp_align_eo", "settings", "template_initial_ref_001.em")
MSK = os.path.join(TD, "abp_align", "settings", "editorMask.em")
HAVE_TD = all(os.path.isfile(f) for f in (H1, H2, TPL, MSK))
try:
    import PySide6  # noqa: F401
    HAVE_QT = True
except Exception:
    HAVE_QT = False


def coded(nz=6, ny=5, nx=4):
    """v[z, y, x] = 10000 z + 100 y + x (0-based) so every voxel says where it is."""
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    return (10000.0 * z + 100.0 * y + x)


# ---------------------------------------------------------------------------------------------------- layouts

def test_nice_2d_distribution():
    # S/nice_2d_distribution.m: smallest square, then drop the fully empty rows
    assert L.nice_2d_distribution(64) == (8, 8)
    assert L.nice_2d_distribution(40) == (6, 7)
    assert L.nice_2d_distribution(128) == (11, 12)
    assert L.nice_2d_distribution(1) == (1, 1)
    assert L.nice_2d_distribution(3) == (2, 2)


def test_best_layout_matches_dynamo_windows():
    # screenshots 01 and 03: 950 x 420 panel; 64 frames of 64 x 64 -> 5 x 13; two volumes (128 x 64) -> 8 x 9
    assert L.best_layout((950, 420), (64, 64), 64) == (5, 13)
    assert L.best_layout((950, 420), (128, 64), 64) == (8, 9)
    r, c = L.best_layout((950, 420), (96, 96), 96)
    assert r * c >= 96 and r * c - 96 < c


# ---------------------------------------------------------------------------------------------- orientation

def test_view_tile_orientation():
    v = coded()
    nz, ny, nx = v.shape
    z = L.view_tile(v, "z", 2)                    # frame z=2: rows y from ny (top) down, columns x
    assert z.shape == (ny, nx)
    assert z[0, 0] == 10000 * 1 + 100 * (ny - 1) + 0 and z[-1, -1] == 10000 + 0 + nx - 1
    x = L.view_tile(v, "x", 3)                    # frame x=3: rows z from nz down, columns y
    assert x.shape == (nz, ny)
    assert x[0, 0] == 10000 * (nz - 1) + 0 + 2 and x[0, -1] == 10000 * (nz - 1) + 100 * (ny - 1) + 2
    y = L.view_tile(v, "y", 1)                    # frame y=1: rows z from nz down, columns x from nx down
    assert y.shape == (nz, nx)
    assert y[0, 0] == 10000 * (nz - 1) + (nx - 1) and y[-1, -1] == 0


def test_slab_mean_and_edges():
    v = coded()
    assert L.slab_indices(1, 3, 6) == [1, 2]      # getListAroundSliceBasic + intersect(1..N)
    assert L.slab_indices(3, 4, 6) == [1, 2, 3, 4]
    t = L.view_tile(v, "z", 1, thickness=3)       # mean of z=1,2 only
    np.testing.assert_allclose(t, (L.view_tile(v, "z", 1) + L.view_tile(v, "z", 2)) / 2)


def test_dview_tiles_equal_mapview_tiles():
    # dview: dynamo_rot_solid + fliplr + imagesc(fused'); mapview: readRawSlice + transpose + flipud: same picture
    v = np.random.default_rng(1).normal(size=(6, 6, 6))
    for ax in "xyz":
        sc = L.dview_scene(v, ax)
        th, tw = sc["tile"]
        raw = np.stack([L.view_tile(v, ax, k) for k in range(1, 7)])
        g = (raw - raw.min()) / (raw - raw.min()).max()
        np.testing.assert_allclose(sc["cdata"][:th, tw:2 * tw], g[1])
        assert (sc["rows"], sc["cols"]) == (2, 3)


# ---------------------------------------------------------------------------------------------- intensities

def test_maprange_and_gray():
    a = np.array([[1.0, 3.0], [np.nan, 5.0]])
    np.testing.assert_allclose(L.maprange(a), [[0, 0.5], [np.nan, 1]])
    g = L.to_gray(np.array([0.0, 0.5, 1.0, 2.0, np.nan]), (0, 1), 64)
    np.testing.assert_allclose(g, [0, 32 / 63, 1, 1, 0])


def test_dview_labels():
    v = np.zeros((4, 4, 4))
    assert L.dview_labels(v) == ("Flat value ( 0.00)", "4 x 4 x 4 voxels")
    v = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    lab, size = L.dview_labels(v)
    assert lab == "Max:    23 Min:     0 Mean:    12" and size == "4 x 3 x 2 voxels"


def test_dview_clim_interior():
    v = np.zeros((16, 16, 16))
    v[0, 0, 0] = 100.0                             # outside the interior ellipsoid: sets the global range only
    v[8, 8, 8] = 10.0                              # inside
    lo, hi = L.dview_clim(v)
    assert lo == 0 and hi == pytest.approx(0.1)
    assert L.dview_clim(np.zeros((4, 4, 4))) is None


def test_autocontrast_limits():
    x = np.concatenate([np.zeros(1000), np.linspace(0, 1, 98000), np.ones(1000)])
    lo, hi = L.autocontrast_limits(x, 255)
    assert 0 <= lo < 0.01 and 0.99 < hi <= 1


def test_dynamo_bin():
    v = np.arange(5 * 4 * 6, dtype=float).reshape(5, 4, 6)
    b = L.dynamo_bin(v, 1)
    assert b.shape == (2, 2, 3)
    assert b[0, 0, 0] == v[:2, :2, :2].mean()
    assert L.dynamo_bin(v, 2).shape == (1, 1, 1)


# -------------------------------------------------------------------------------------------- mapview model

def test_mapview_model_opening_state():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(30, 8, 8)), rng.normal(size=(30, 8, 8))
    a[25, 3, 3] = 50.0                            # outside the first 20 z-slices: not in the opening CLim
    m = L.MapviewModel([a, b])
    assert m.selection == [1]                     # montage.m:99-104
    assert m.clim == (a[:20].min(), a[:20].max())   # first imshow on the default 4 x 5 layout
    assert (m.rows, m.cols) == L.best_layout(L.MAPVIEW_PANEL, (8, 8), 30)
    m.show_all_volumes()
    sc = m.scene()
    assert sc["tile"] == (8, 16) and np.isnan(sc["image"]).any() == (m.rows * m.cols > 30)
    m.contrast_full()
    assert m.clim == (min(a.min(), b.min()), max(a.max(), b.max()))


def test_mapview_model_controls():
    v = coded(nz=10, ny=4, nx=4)
    m = L.MapviewModel([v, v + 1, v + 2])
    m.set_selection(0, 7)
    assert m.selection == [1, 2, 3]
    m.set_selection(3, 1)
    assert m.selection == [3]
    m.set_rows_cols(2, 2)
    assert m.frames_shown() == [1, 2, 3, 4]
    assert m.page(1) and m.first_frame == 5
    assert m.page(1) and m.first_frame == 9
    assert not m.page(1) and m.first_frame == 9   # past the end: refused
    assert m.page(-1, 100) and m.first_frame == 1   # clamped at the first frame
    m.set_current(6)
    m.set_interval(3)
    assert m.accessible == [3, 6, 9]
    m.set_orientation("x")
    assert m.n_frames() == 4 and m.accessible == [1, 2, 3, 4]
    m.show_only_slice(2)
    assert (m.rows, m.cols, m.frames_shown()) == (1, 1, [2])
    m.set_selection(2, 3)
    m.show_only_volume(1)                         # position in the shown list used as the index (1.1.514)
    assert m.selection == [1]


def test_mapview_mismatched_sizes():
    with pytest.raises(ValueError, match="same x length"):
        L.MapviewModel([np.zeros((4, 4, 4)), np.zeros((4, 4, 6))])


# ----------------------------------------------------------------------------------------------- dslices

def test_dslices_overlay_formula():
    t = np.random.default_rng(3).normal(size=(4, 4, 4))
    m = np.zeros((4, 4, 4))
    m[1:3, 1:3, 1:3] = 1.0
    o = L.dslices_overlay(t, m, "z")
    base = L.dslices_scene([t], "z")
    gray = L.to_gray(base["cdata"], (0.0, 1.0), 64)
    mm = L.assemble(list(L.maprange(np.stack(L.dslices_tiles(m)))), 2, 2)
    np.testing.assert_allclose(o["rgb"][..., 0], 0.8 * gray + 0.2 * mm, atol=1e-6)
    np.testing.assert_allclose(o["rgb"][..., 1], 0.8 * gray, atol=1e-6)


def test_dslices_x_uses_dynamo_rotation():
    # dynamo_rot (mex option 0): inside the sphere 'x' equals the permutation; outside it is 0
    v = np.random.default_rng(4).normal(size=(8, 8, 8))
    r = L.dslices_rotate(v, "x")
    assert r.shape == v.shape
    assert np.count_nonzero(r) < v.size


# ---------------------------------------------------------------------------------------------- API (no Qt)

def test_api_without_display(monkeypatch, capsys):
    import pydyn.viewers as V
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert V.qt_problem() is not None
    assert V.dmapview(np.zeros((4, 4, 4))) is None
    assert "cannot open the window" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="intesect"):
        V.dmapview("a.em", "indices", [1])
    with pytest.raises(ValueError, match="not a recognized parameter"):
        V.dmapview("a.em", "sym", "c2")
    with pytest.raises(NotImplementedError):
        V.dtplot("x.tbl", "profile", "shifts")          # only the default 'rays' plot is ported
    assert V.dtplot("x.tbl") is None                    # no display: message, no window


def test_last_completed_iteration(tmp_path, monkeypatch):
    import pydyn.viewers as V
    monkeypatch.chdir(tmp_path)
    vpr = {"intertwin": "1/2/3/4/5/6/7/8", "ite_r1": 5}
    for r in range(1, 9):
        vpr["nref_r%d" % r] = 2
        vpr.setdefault("ite_r%d" % r, 0)
    for ite, refs in ((1, (1, 2)), (2, (1, 2)), (3, (1,))):
        for r in refs:
            f = tmp_path / "p" / "results" / ("ite_%04d" % ite) / "averages" / ("average_ref_%03d_ite_%04d.em" % (r, ite))
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"")
    assert V.last_completed_iteration("p", vpr) == 2   # ite 3 has 1 of 2 references
    assert [os.path.basename(f) for f in L.last_average_files("p", 2)] == \
        ["average_ref_001_ite_0002.em", "average_ref_002_ite_0002.em"]


# --------------------------------------------------------------------------- Dynamo 1.1.514 ground truth

@pytest.mark.skipif(not HAVE_TD, reason="needs test_data/bin8_filament (the user's ABP project)")
def test_ground_truth_dynamo_numbers():
    # GT lines of benchmarks/08_viewers/dynamo_screens/dynamo_run.log (headless Dynamo 1.1.514, R2019a runtime)
    m = L.MapviewModel([H1, H2])
    np.testing.assert_allclose(m.clim, (-16.6978, 30.8236), atol=5e-5)
    assert (m.rows, m.cols) == (5, 13) and m.scene()["image"].shape == (320, 832)
    m.show_all_volumes()
    assert m.scene()["image"].shape == (320, 1664)
    np.testing.assert_allclose(m.clim, (-16.6978, 30.8236), atol=5e-5)   # unchanged by "Show all volumes"
    m.optimal_layout()
    assert (m.rows, m.cols) == (8, 9)
    m.contrast_full()
    np.testing.assert_allclose(m.clim, (-17.7283, 30.8236), atol=5e-5)
    for f, label, clim in ((TPL, "Max:   7.4 Min:   -11 Mean: 0.0046", (0, 1)),
                           (H1, "Max:    31 Min:   -17 Mean: -7.8e-06", (0, 0.608238)),
                           (MSK, "Max:     1 Min:     0 Mean: 0.084", (1.18255e-09, 1))):
        v = L.read_volume(f)
        assert L.dview_labels(v) == (label, "64 x 64 x 64 voxels")
        np.testing.assert_allclose(L.dview_clim(v), clim, rtol=1e-5, atol=1e-12)


# ------------------------------------------------------------------------------------------- Qt offscreen

def _qt_env():
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "QT_PLUGIN_PATH")}
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


@pytest.fixture(scope="module")
def qapp():
    if not HAVE_QT:
        pytest.skip("PySide6 is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])


def test_qt_mapview_window(qapp):
    from pydyn.viewers import qtview as Q
    rng = np.random.default_rng(5)
    m = L.MapviewModel([rng.normal(size=(12, 10, 10)), rng.normal(size=(12, 10, 10))])
    w = Q.MapviewWindow(m)
    w.show()
    qapp.processEvents()
    assert w.windowTitle() == "GUI contains 2 volume(s)"
    assert w.e_start.text() == "1" and w.e_end.text() == "1"
    w.e_end.setText("2")
    w._selection()
    assert m.selection == [1, 2] and w.canvas.qimage.width() == m.cols * 20
    w.b_axis["x"].click()
    assert m.orientation == "x" and w.b_axis["x"].font().bold()
    img = w.grab().toImage()
    assert img.width() >= 900
    w.close()


def test_qt_dview_and_slices_windows(qapp):
    from pydyn.viewers import qtview as Q
    v = np.random.default_rng(6).normal(size=(8, 8, 8))
    w = Q.DviewWindow([("<numeric variable>", v)], title="t")
    w.show()
    qapp.processEvents()
    assert w.windowTitle() == "dview: t"
    assert w.text_size.text() == "8 x 8 x 8 voxels"
    w.b["y"].click()
    assert w.axis == "y" and w.b["y"].font().bold()
    w.close()
    o = L.dslices_overlay(v, (v > 0).astype(float), "z")
    s = Q.SlicesWindow(o["rgb"], o["rows"], o["cols"], o["tile"], name="mask overlayed on  template", axes_title="m")
    s.show()
    qapp.processEvents()
    assert s.windowTitle() == "mask overlayed on  template"
    s.close()


@pytest.mark.skipif(not HAVE_QT, reason="PySide6 is not installed")
def test_viewer_process_and_launcher(tmp_path, monkeypatch):
    """The separate process: files on the command line and arrays through temporary .npy files (deleted)."""
    import tempfile
    import pydyn.viewers as V
    tmp = tmp_path / "tmpdir"
    tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp))                              # this test's own temporary folder
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    a = np.random.default_rng(7).normal(size=(8, 8, 8)).astype(np.float32)
    f1, f2 = str(tmp_path / "a.em"), str(tmp_path / "b.em")
    dynio.write_em(f1, a)
    dynio.write_em(f2, -a)
    out = str(tmp_path / "map.png")
    r = subprocess.run([sys.executable, "-I", "-m", "pydyn.viewers", "mapview", f1, f2, "--snap", out,
                        "--do", "showall,x"], cwd=os.path.join(P, "prototype"), env=_qt_env(),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(out) and os.path.isfile(str(tmp_path / "map_showall.png"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    out2 = str(tmp_path / "arr.png")
    p = V._launch(["view", "--snap", out2], arrays=[a], wait=True)
    assert p.returncode == 0 and os.path.isfile(out2)
    leftovers = [d for d in os.listdir(str(tmp)) if d.startswith("pydyn_view_")]
    assert not leftovers


@pytest.mark.skipif(not HAVE_QT, reason="PySide6 is not installed")
def test_shell_entry_restarts_clean(tmp_path):
    """dmapview_main (the future `dmapview` shell command) drops LD_LIBRARY_PATH / PYTHONPATH before Qt is loaded."""
    a = np.random.default_rng(8).normal(size=(8, 8, 8)).astype(np.float32)
    f1 = str(tmp_path / "a.em")
    dynio.write_em(f1, a)
    out = str(tmp_path / "shell.png")
    env = _qt_env()
    env["LD_LIBRARY_PATH"] = "/nonexistent/qt6/lib"
    code = ("import sys; sys.path.insert(0, %r); sys.argv = ['dmapview', %r, '--snap', %r]; "
            "from pydyn.viewers.__main__ import dmapview_main; sys.exit(dmapview_main())"
            % (os.path.join(P, "prototype"), f1, out))
    r = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(out)


# ------------------------------------------------------------------------------------------- dtplot and FSC

def test_plotangles_points_stacking():
    a = np.array([[0, 90, 0], [0, 90, 30], [0, 90, 10], [90, 90, 0], [0, 90, 5]], dtype=float)
    xyz = L.plotangles_points(a)
    np.testing.assert_allclose(np.linalg.norm(xyz[:3], axis=1), [1.0, 1.4, 1.2], atol=1e-12)   # ordered by narot
    np.testing.assert_allclose(xyz[0], [0.0, 1.0, 0.0], atol=1e-12)        # (sin td sin ti, cos td sin ti, cos ti)
    assert np.linalg.norm(xyz[4]) == pytest.approx(1.0)  # same direction as rows 1-3 but not consecutive: no stacking


@pytest.mark.skipif(not HAVE_TD, reason="needs test_data/bin8_filament (the user's ABP project)")
def test_ground_truth_dtplot_and_fsc_axes():
    gt = os.path.join(P, "benchmarks", "08_viewers", "dynamo_screens", "dtplot_xyz.txt")
    tags, xyz = L.dtplot_points(os.path.join(TD, "abp_align_eo", "results", "ite_0004", "averages",
                                             "refined_table_ref_001_ite_0004.tbl"))
    if os.path.isfile(gt):
        np.testing.assert_allclose(xyz, np.loadtxt(gt), atol=1e-8)      # Dynamo's scatter3 XData/YData/ZData
    curves = L.bandpass_fsc_curves(os.path.join(TD, "abp_align_eo"), 4)
    assert [c[0] for c in curves] == [1, 2, 3, 4] and all(c[2].size == 32 for c in curves)
    xs = np.concatenate([c[1] for c in curves])
    ys = np.concatenate([c[2] for c in curves])
    assert L.matlab_auto_limits(xs.min(), xs.max()) == (0, 35, [0, 5, 10, 15, 20, 25, 30, 35])
    lo, hi, ticks = L.matlab_auto_limits(ys.min(), ys.max())
    assert (lo, hi) == pytest.approx((0.1, 1.0)) and len(ticks) == 10


def test_qt_figures(qapp):
    from pydyn.viewers import qtview as Q
    xyz = L.plotangles_points(np.array([[0, 90, 0], [45, 45, 0]], dtype=float))
    w = Q.FigureWindow(Q.TablePlotCanvas(xyz), "Figure")
    w.show()
    qapp.processEvents()
    w.close()
    f = Q.fsc_window("p", [(1, np.arange(1, 5.0), np.array([1, 0.8, 0.5, 0.2]))])
    f.show()
    qapp.processEvents()
    assert f.windowTitle() == "Figure 1"
    f.close()


def test_zoom_keeps_the_point_under_the_cursor(qapp):
    """Wheel and click zoom (dview, dmapview, dslices canvas): the image point under the cursor stays there."""
    from PySide6 import QtCore
    from pydyn.viewers import qtview as Q
    c = Q.ImageCanvas()
    c.resize(400, 200)
    c.set_image(Q.gray_qimage(np.zeros((50, 100), np.uint8)))
    pos = QtCore.QPointF(310.0, 47.0)
    before = c.to_image(pos)
    for f in (2.0, 1.5, 1.25, 1 / 1.5):
        c.zoom_at(pos, f)
        after = c.to_image(pos)
        assert np.allclose(after, before, atol=1e-9), (f, before, after)
    assert c.zoom == pytest.approx(2.5)
    c.zoom_at(pos, 1e-3)                                       # never below the fitted view
    assert (c.zoom, c.center) == (1.0, None)
    v = Q.DviewWindow([("cube", np.random.default_rng(1).normal(size=(8, 9, 10)))])
    v.act_zoom_in.setChecked(True)
    assert v.canvas.zoom_mode == "in" and not v.canvas.pan_mode
    v.act_pan.setChecked(True)                                 # exclusive: pan switches the zoom mode off
    assert v.canvas.zoom_mode is None and v.canvas.pan_mode
    v.act_pan.setChecked(False)
    assert v.canvas.zoom_mode is None and not v.canvas.pan_mode
    v.close()


def test_zoomed_image_can_be_grabbed_and_dragged(qapp):
    """dview / dmapview / dslices canvas: once zoomed in, a left drag moves the image with the cursor (any mode); in
    Zoom In mode a click without a drag zooms at the click, a drag only moves."""
    from PySide6 import QtCore, QtGui
    from pydyn.viewers import qtview as Q
    Qt = QtCore.Qt

    def mouse(c, kind, x, y, button=Qt.LeftButton):
        t = {"press": QtCore.QEvent.MouseButtonPress, "move": QtCore.QEvent.MouseMove,
             "release": QtCore.QEvent.MouseButtonRelease}[kind]
        p = QtCore.QPointF(x, y)
        held = Qt.NoButton if kind == "release" else button
        ev = QtGui.QMouseEvent(t, p, p, button if kind != "move" else Qt.NoButton, held, Qt.NoModifier)
        {"press": c.mousePressEvent, "move": c.mouseMoveEvent, "release": c.mouseReleaseEvent}[kind](ev)

    c = Q.ImageCanvas()
    c.resize(400, 200)
    c.set_image(Q.gray_qimage(np.zeros((50, 100), np.uint8)))
    mouse(c, "press", 200, 100); mouse(c, "move", 260, 130); mouse(c, "release", 260, 130)
    assert (c.zoom, c.center) == (1.0, None)                    # the fitted image does not move
    c.zoom_at(QtCore.QPointF(200, 100), 3.0)
    grabbed = c.to_image(QtCore.QPointF(150, 80))
    mouse(c, "press", 150, 80); mouse(c, "move", 210, 115); mouse(c, "release", 210, 115)
    assert np.allclose(c.to_image(QtCore.QPointF(210, 115)), grabbed)   # the grabbed point follows the cursor
    assert c.zoom == pytest.approx(3.0)
    c.zoom_mode = "in"
    mouse(c, "press", 100, 50); mouse(c, "move", 140, 70); mouse(c, "release", 140, 70)
    assert c.zoom == pytest.approx(3.0)                         # a drag in zoom mode moves, does not zoom
    p = QtCore.QPointF(120, 60)
    before = c.to_image(p)
    mouse(c, "press", 120, 60); mouse(c, "release", 120, 60)
    assert c.zoom == pytest.approx(4.5) and np.allclose(c.to_image(p), before)   # a click zooms there
