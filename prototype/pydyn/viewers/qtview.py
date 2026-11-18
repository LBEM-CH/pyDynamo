"""Qt (PySide6) windows of the viewers. They only paint what pydyn.viewers.logic computes and forward the user's
actions to it. This module is imported only in the viewer process (python -m pydyn.viewers ...), never in the dcp window's
process. Layouts follow Dynamo 1.1.514's windows (docs/analysis/11a_viewers_1.1.514.md; screenshots of the real
windows in benchmarks/08_viewers/dynamo_screens/)."""
import math
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import logic as L

NOT_PORTED = "[not ported in pydyn]"
RED = QtGui.QColor(255, 0, 0)


def gray_qimage(gray):
    """float [0, 1] -> QImage Grayscale8 (owns a copy of the bytes)."""
    a = np.ascontiguousarray(np.round(np.clip(gray, 0, 1) * 255).astype(np.uint8))
    h, w = a.shape
    return QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()


def rgb_qimage(rgb):
    a = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    h, w, _ = a.shape
    return QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class ImageCanvas(QtWidgets.QWidget):
    """An image drawn as MATLAB's `axis image` does: scaled to fit, aspect ratio kept, pixels not smoothed. Overlays
    (grid lines, labels) are given in image pixel units.
    Zoom always keeps the point under the cursor where it is: the mouse wheel zooms at the cursor, a click in zoom mode
    (dview's Zoom In / Zoom Out, as in a MATLAB figure) zooms at the clicked point, a double click shows the whole
    image again. Once zoomed in, the image can be grabbed: dragging with the left (or middle) button moves it, in any
    mode; in a zoom mode a click without a drag zooms."""
    rightClicked = QtCore.Signal(float, float, QtCore.QPoint)    # image row, col, global position
    leftClicked = QtCore.Signal(float, float)

    def __init__(self, parent=None, background=None):
        super().__init__(parent)
        self.qimage = None
        self.vlines, self.hlines, self.line_width = [], [], 1
        self.labels = []                                           # (row, col, text) in image pixels
        self.intra = []                                            # dashed orange x positions (dslices, nvol > 1)
        self.label_font = QtGui.QFont()
        self.label_font.setPixelSize(12)
        self.zoom, self.center = 1.0, None
        self.pan_mode, self._drag = False, None
        self.zoom_mode = None                                      # None, "in" or "out" (click to zoom there)
        self.background = background
        self.setMinimumSize(100, 100)
        self.setMouseTracking(False)

    def set_image(self, qimage, vlines=(), hlines=(), line_width=1, labels=(), intra=()):
        self.qimage = qimage
        self.intra = list(intra)
        self.vlines, self.hlines, self.line_width = list(vlines), list(hlines), line_width
        self.labels = list(labels)
        self.update()

    def _geometry(self):
        """(scale, x0, y0): screen = x0 + scale * image."""
        iw, ih = self.qimage.width(), self.qimage.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        cx, cy = self.center if self.center else (iw / 2.0, ih / 2.0)
        return s, self.width() / 2.0 - s * cx, self.height() / 2.0 - s * cy

    def to_image(self, pos):
        s, x0, y0 = self._geometry()
        return (pos.y() - y0) / s, (pos.x() - x0) / s

    def zoom_at(self, pos, f):
        """Zoom by f keeping the image point under `pos` (widget coordinates) under it. Never below the fitted view."""
        if self.qimage is None:
            return
        r, c = self.to_image(pos)
        self.zoom = max(1.0, min(64.0, self.zoom * f))
        if self.zoom == 1.0:
            self.center = None
        else:
            s, _, _ = self._geometry()                             # the scale does not depend on the centre
            self.center = (c - (pos.x() - self.width() / 2.0) / s, r - (pos.y() - self.height() / 2.0) / s)
        self.update_cursor()
        self.update()

    def reset_zoom(self):
        self.zoom, self.center = 1.0, None
        self.update_cursor()
        self.update()

    def wheelEvent(self, ev):
        d = ev.angleDelta().y()
        if d and self.qimage is not None:
            self.zoom_at(ev.position(), 1.25 if d > 0 else 1 / 1.25)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.reset_zoom()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        if self.background is not None:
            p.fillRect(self.rect(), self.background)
        if self.qimage is None:
            return
        s, x0, y0 = self._geometry()
        iw, ih = self.qimage.width(), self.qimage.height()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        p.drawImage(QtCore.QRectF(x0, y0, iw * s, ih * s), self.qimage)
        pen = QtGui.QPen(RED)
        pen.setWidth(int(self.line_width))
        pen.setCosmetic(True)
        p.setPen(pen)
        right, bottom = int(round(x0 + iw * s)) - 1, int(round(y0 + ih * s)) - 1
        left, top = int(round(x0)), int(round(y0))
        if self.intra:                                            # dynamo_montage.m:168-178: '--', [0.8,0.49,0], width 1
            dpen = QtGui.QPen(QtGui.QColor(204, 125, 0))
            dpen.setStyle(QtCore.Qt.DashLine)
            dpen.setCosmetic(True)
            p.setPen(dpen)
            for x in self.intra:
                xi = int(round(x0 + x * s))
                p.drawLine(xi, top, xi, bottom)
            p.setPen(pen)
        for x in self.vlines:                                     # whole screen pixels: every line stays visible
            xi = min(max(int(round(x0 + x * s)), left), right)
            p.drawLine(xi, top, xi, bottom)
        for y in self.hlines:
            yi = min(max(int(round(y0 + y * s)), top), bottom)
            p.drawLine(left, yi, right, yi)
        if self.labels:
            p.setFont(self.label_font)
            fm = QtGui.QFontMetrics(self.label_font)
            for r, c, text in self.labels:
                x, y = x0 + c * s + 1, y0 + r * s + 1
                rect = QtCore.QRectF(x, y, fm.horizontalAdvance(text) + 2, fm.height())
                p.fillRect(rect, QtGui.QColor(255, 255, 255))
                p.setPen(QtGui.QColor(0, 0, 0))
                p.drawText(rect, QtCore.Qt.AlignCenter, text)

    def update_cursor(self):
        """Cross in a zoom mode; an open hand when the image is zoomed in (it can be grabbed); else the arrow."""
        if self._drag is not None and self._drag[2]:
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif self.zoom_mode:
            self.setCursor(QtCore.Qt.CrossCursor)
        elif self.zoom > 1.0 or self.pan_mode:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, ev):
        if self.qimage is None:
            return
        r, c = self.to_image(ev.position())
        if ev.button() == QtCore.Qt.RightButton:
            self.rightClicked.emit(r, c, ev.globalPosition().toPoint())
        elif ev.button() in (QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton):
            # grab: a drag moves the zoomed image (any mode); a left click without a drag zooms in a zoom mode
            centre = self.center or (self.qimage.width() / 2.0, self.qimage.height() / 2.0)
            self._drag = [ev.position(), centre, False, ev.button()]
            if ev.button() == QtCore.Qt.LeftButton:
                self.leftClicked.emit(r, c)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        d = ev.position() - self._drag[0]
        if not self._drag[2]:
            if abs(d.x()) + abs(d.y()) < 4:                         # a click, not yet a drag
                return
            self._drag[2] = True
            self.update_cursor()
        if self.zoom > 1.0:
            s, _, _ = self._geometry()
            self.center = (self._drag[1][0] - d.x() / s, self._drag[1][1] - d.y() / s)
            self.update()

    def mouseReleaseEvent(self, ev):
        drag, self._drag = self._drag, None
        if drag is not None and not drag[2] and drag[3] == QtCore.Qt.LeftButton and self.zoom_mode:
            self.zoom_at(ev.position(), 1.5 if self.zoom_mode == "in" else 1 / 1.5)
        self.update_cursor()


def _button(text, slot, tip=None, checkable=False):
    b = QtWidgets.QPushButton(text)
    b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    b.clicked.connect(slot)
    return b


def _not_ported(parent, what):
    QtWidgets.QMessageBox.information(parent, "pyDynamo", "%s %s" % (what, NOT_PORTED))


# =====================================================================================================================
# dmapview
# =====================================================================================================================

KEY_HELP = [("shift+Q", "previous page of frames"), ("shift+W", "next page of frames"),
            ("s / u", "select / unselect the frame under the cursor " + NOT_PORTED),
            ("1 ... 9", "anchor point at the cursor " + NOT_PORTED)]


class ContrastDialog(QtWidgets.QDialog):
    """MB/+mbimg/+contrast/@GUI ("Contrast control"): histogram of the scene in 255 bars between its min and max;
    left click sets the lower limit, right click the upper one; OK / Full / 0.01-0.99 / 0.05-0.95."""

    def __init__(self, parent, values, clim, apply, bins=255, left_sets="lower"):
        super().__init__(parent)
        self.setWindowTitle("Contrast control")
        v = np.asarray(values, dtype=np.float64).ravel()
        self.v = v[~np.isnan(v)]
        self.apply = apply
        self.left_sets = left_sets
        self.lo, self.hi = clim
        self.counts, self.edges = np.histogram(self.v, bins=bins)
        lay = QtWidgets.QVBoxLayout(self)
        self.plot = QtWidgets.QLabel()
        self.plot.setFixedSize(520, 200)
        self.plot.mousePressEvent = self._click
        lay.addWidget(self.plot)
        self.info = QtWidgets.QLabel()
        lay.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("OK", self.accept), ("Full", self._full), ("0.01 / 0.99", lambda: self._pct(0.01)),
                           ("0.05 / 0.95", lambda: self._pct(0.05))):
            row.addWidget(_button(text, slot))
        lay.addLayout(row)
        self._redraw()

    def _x2val(self, x):
        return self.edges[0] + (self.edges[-1] - self.edges[0]) * x / self.plot.width()

    def _click(self, ev):
        val = self._x2val(ev.position().x())
        lower = (ev.button() == QtCore.Qt.LeftButton) == (self.left_sets == "lower")
        if lower and val < self.hi:
            self.lo = val
        elif not lower and val > self.lo:
            self.hi = val
        self._set()

    def _full(self):
        self.lo, self.hi = float(self.v.min()), float(self.v.max())
        self._set()

    def _pct(self, f):
        """getPercentiles.m: the f and 1-f quantiles of the scene (NaN dropped)."""
        self.lo, self.hi = (float(x) for x in np.quantile(self.v, [f, 1 - f]))
        self._set()

    def _set(self):
        if self.hi > self.lo:
            self.apply(self.lo, self.hi)
        self._redraw()

    def _redraw(self):
        w, h = self.plot.width(), self.plot.height()
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtGui.QColor(255, 255, 255))
        p = QtGui.QPainter(pm)
        c = self.counts / max(1, self.counts.max())
        bw = w / len(c)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(60, 60, 60))
        for i, ci in enumerate(c):
            p.drawRect(QtCore.QRectF(i * bw, h * (1 - ci), bw, h * ci))
        span = self.edges[-1] - self.edges[0] or 1.0
        for val, col in ((self.lo, QtGui.QColor(0, 0, 255)), (self.hi, QtGui.QColor(255, 0, 0))):
            x = (val - self.edges[0]) / span * w
            p.setPen(QtGui.QPen(col, 2, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
        p.end()
        self.plot.setPixmap(pm)
        self.info.setText("lower limit %.4g   upper limit %.4g   (left click: %s, right click: %s)"
                          % (self.lo, self.hi, self.left_sets, "upper" if self.left_sets == "lower" else "lower"))


class MapviewWindow(QtWidgets.QMainWindow):
    """dpkvol.montages.aux.mapview.Montage as it opens (screenshot 01): menu bar, icon bar, the montage panel
    (950 x 420) and the 80-pixel bottom bar."""

    def __init__(self, model, mask=None):
        super().__init__()
        self.m = model
        self.mask = mask
        self.setWindowTitle("GUI contains %d volume(s)" % model.n_volumes)    # montage.m:157
        self._menus()
        self._toolbar()
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.canvas = ImageCanvas(background=QtGui.QColor(240, 240, 240))
        self.canvas.rightClicked.connect(self._context)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self._bottom_bar())
        self.setCentralWidget(central)
        self.resize(950, 500 + self.menuBar().sizeHint().height() + self.tb.sizeHint().height())
        self.refresh()

    # ---- construction ----------------------------------------------------------------------------------------
    def _menus(self):
        mb = self.menuBar()
        h = mb.addMenu("Help")
        h.addAction("Info on key actions", self._keys_help)
        t = mb.addMenu("Tools")
        c = t.addMenu("Contrast")
        c.addAction("Contrast.", self._manual_contrast, QtGui.QKeySequence("Ctrl+C"))
        c.addAction("Autocontrast.", self._autocontrast, QtGui.QKeySequence("Ctrl+A"))
        t.addAction("Annotations " + NOT_PORTED).setEnabled(False)
        s = mb.addMenu("Settings")
        s.addAction("Labels " + NOT_PORTED).setEnabled(False)
        v = mb.addMenu("View")
        self.act_norm_slice = v.addAction("normalize slice intensities separately")
        self.act_norm_slice.setCheckable(True)
        self.act_norm_slice.toggled.connect(lambda on: self._apply(self.m.set_normalization, per_slice=on))
        self.act_norm_vol = v.addAction("normalize slice intensities from each volume separately")
        self.act_norm_vol.setCheckable(True)
        self.act_norm_vol.toggled.connect(lambda on: self._apply(self.m.set_normalization, per_volume=on))
        for name in ("Selection", "Anchor points", "Chimera"):
            mb.addMenu(name).addAction(name + " " + NOT_PORTED).setEnabled(False)
        e = mb.addMenu("External mask")
        self.act_mask = e.addAction("Show external mask")
        self.act_mask.setCheckable(True)
        self.act_mask.setEnabled(self.mask is not None)
        self.act_mask.toggled.connect(lambda on: self.refresh())

    def _toolbar(self):
        tb = self.tb = self.addToolBar("tools")
        tb.setMovable(False)
        st = self.style()
        icons = QtWidgets.QStyle
        for tip, icon, slot in (
                ("refresh scene", icons.SP_BrowserReload, self.refresh),
                ("Optimal layout of rows and columns for current number of depicted of depicted frames",
                 icons.SP_FileDialogDetailedView, lambda: self._apply(self.m.optimal_layout)),
                ("Shows all available frames", icons.SP_TitleBarMaxButton, lambda: self._apply(self.m.full_video)),
                ("Launches Manual contrast tool", icons.SP_ComputerIcon, self._manual_contrast),
                ("Contrast (full)", icons.SP_DialogApplyButton, lambda: self._apply(self.m.contrast_full)),
                ("Show all volumes", icons.SP_FileDialogListView, lambda: self._apply(self.m.show_all_volumes))):
            a = tb.addAction(st.standardIcon(icon), tip, slot)
            a.setToolTip(tip)
        for tip, icon in (("click on two frames to select all the frames in the interval", icons.SP_ArrowRight),
                          ("Select an area inside the frame", icons.SP_FileDialogContentsView)):
            a = tb.addAction(st.standardIcon(icon), tip + " " + NOT_PORTED)
            a.setToolTip(tip + " " + NOT_PORTED)
            a.setEnabled(False)
        # zoom: the mouse wheel zooms at the cursor (ImageCanvas); this button shows the whole image again
        a = tb.addAction(st.standardIcon(icons.SP_FileDialogBack), "Restore zoomed area", lambda: self.canvas.reset_zoom())
        a.setToolTip("Restore zoomed area (the mouse wheel zooms at the cursor; double click also restores)")

    def _edit(self, text, slot, width=None):
        e = QtWidgets.QLineEdit(text)
        e.setAlignment(QtCore.Qt.AlignCenter)
        if width:
            e.setFixedWidth(width)
        e.returnPressed.connect(slot)
        return e

    def _bottom_bar(self):
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(80)
        g = QtWidgets.QHBoxLayout(bar)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(1)
        for text, d, chunk in (("<<", -1, None), (">>", 1, None)):
            b = _button(text, lambda _=False, d=d, c=chunk: self._apply(self.m.page, d, c))
            b.setFixedSize(46, 78)
            g.addWidget(b)
        g.addSpacing(46)
        for text, d in (("<", -1), (">", 1)):
            b = _button(text, lambda _=False, d=d: self._apply(self.m.page, d, 1))
            b.setFixedSize(46, 78)
            g.addWidget(b)
        # cols / rows
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(0)
        self.e_cols = self._edit("", lambda: self._apply(self.m.set_rows_cols, cols=self.e_cols.text()), 35)
        self.e_rows = self._edit("", lambda: self._apply(self.m.set_rows_cols, rows=self.e_rows.text()), 35)
        for r, (lab, e, attr) in enumerate((("cols", self.e_cols, "cols"), ("rows", self.e_rows, "rows"))):
            grid.addWidget(QtWidgets.QLabel(lab), r, 0)
            grid.addWidget(e, r, 1)
            pm = QtWidgets.QVBoxLayout()
            for sign in ("+", "-"):
                b = _button(sign, lambda _=False, a=attr, s=sign: self._plusminus(a, s))
                b.setFixedSize(30, 19)
                pm.addWidget(b)
            grid.addLayout(pm, r, 2)
        g.addLayout(grid)
        # current / interval (red panel in Dynamo)
        cur = QtWidgets.QGridLayout()
        self.e_current = self._edit("1", lambda: self._apply(self.m.set_current, self.e_current.text()), 90)
        self.e_interval = self._edit("1", lambda: self._apply(self.m.set_interval, self.e_interval.text()), 90)
        cur.addWidget(QtWidgets.QLabel("current"), 0, 0)
        cur.addWidget(self.e_current, 0, 1)
        cur.addWidget(QtWidgets.QLabel("interval"), 1, 0)
        cur.addWidget(self.e_interval, 1, 1)
        g.addLayout(cur)
        # graymap tools (launchGraymapToolsInBar.m)
        gt = QtWidgets.QGridLayout()
        gt.setSpacing(0)
        self.r_thick = QtWidgets.QRadioButton(" thick")
        self.r_thick.toggled.connect(lambda on: self._apply(self.m.set_thick, on, self.e_thick.text()))
        self.e_thick = self._edit("1", lambda: self._apply(self.m.set_thick, self.r_thick.isChecked(),
                                                           self.e_thick.text()), 24)
        thick = QtWidgets.QHBoxLayout()
        thick.addWidget(self.r_thick)
        thick.addWidget(self.e_thick)
        gt.addLayout(thick, 0, 0)
        for r, (text, slot) in enumerate(((" invert", self.m.op_invert), (" normalize", self.m.op_normalize),
                                          (" restore", self.m.op_restore)), start=1):
            b = _button(text, lambda _=False, s=slot: self._apply(s))
            b.setFixedHeight(19)
            gt.addWidget(b, r, 0)
        self.e_bin = self._edit("2", lambda: None, 90)
        self.e_band = self._edit("0.8", lambda: None, 90)
        self.e_sym = self._edit("c1", self._sym, 90)
        for r, (text, e, slot) in enumerate(((" bin", self.e_bin, lambda: self._apply(self.m.op_bin, self.e_bin.text())),
                                             (" band", self.e_band,
                                              lambda: self._apply(self.m.op_band, self.e_band.text())),
                                             (" sym", self.e_sym, self._sym))):
            b = _button(text, lambda _=False, s=slot: s())
            row = QtWidgets.QHBoxLayout()
            row.addWidget(b)
            row.addWidget(e)
            gt.addLayout(row, r, 1)
        g.addLayout(gt)
        # x / y / z
        xyz = QtWidgets.QVBoxLayout()
        xyz.setSpacing(0)
        self.b_axis = {}
        for a in ("x", "y", "z"):
            b = _button(a, lambda _=False, a=a: self._apply(self.m.set_orientation, a),
                        "Changes viewing direction to %s" % a, checkable=True)
            b.setFixedSize(30, 26)
            self.b_axis[a] = b
            xyz.addWidget(b)
        g.addLayout(xyz)
        # shown volumes
        sh = QtWidgets.QVBoxLayout()
        sh.setSpacing(0)
        lab = QtWidgets.QLabel("shown:")
        lab.setToolTip("Range of shown volumes (out of the full set of loaded volumes)")
        self.e_start = self._edit("1", self._selection, 79)
        self.e_start.setToolTip("First volume in the shown range")
        self.e_end = self._edit("1", self._selection, 79)
        self.e_end.setToolTip("Last volume in the shown range")
        for w in (lab, self.e_start, self.e_end):
            sh.addWidget(w)
        g.addLayout(sh)
        return bar

    # ---- actions ---------------------------------------------------------------------------------------------
    def _apply(self, fn, *a, **k):
        try:
            fn(*a, **k)
        except (ValueError, NotImplementedError, IndexError) as e:
            QtWidgets.QMessageBox.warning(self, "dmapview", str(e))
        self.refresh()

    def _plusminus(self, attr, sign):
        n = getattr(self.m, attr) + (1 if sign == "+" else -1)
        self._apply(self.m.set_rows_cols, **{attr: max(1, n)})

    def _selection(self):
        self._apply(self.m.set_selection, self.e_start.text() or 1, self.e_end.text() or 1)

    def _sym(self):
        _not_ported(self, "sym (dynamo_sym)")

    def _manual_contrast(self):
        d = ContrastDialog(self, self.m.scene()["image"], self.m.clim,
                           lambda lo, hi: (self.m.set_clim(lo, hi), self.refresh()))
        d.exec()

    def _autocontrast(self):
        self._apply(self.m.autocontrast)

    def _keys_help(self):
        QtWidgets.QMessageBox.information(self, "Info on key actions",
                                          "\n".join("%-10s %s" % kv for kv in KEY_HELP))

    def keyPressEvent(self, ev):
        """addKeysForSlidingScene.m:16-33: Shift+Q / Shift+W page backwards / forwards."""
        if ev.modifiers() == QtCore.Qt.ShiftModifier and ev.key() == QtCore.Qt.Key_Q:
            self._apply(self.m.page, -1)
        elif ev.modifiers() == QtCore.Qt.ShiftModifier and ev.key() == QtCore.Qt.Key_W:
            self._apply(self.m.page, 1)
        else:
            super().keyPressEvent(ev)

    def _context(self, r, c, gpos):
        loc = self.m.locate(r, c)
        if loc is None:
            return
        frame, pos, hor, ver = loc
        menu = QtWidgets.QMenu(self)
        menu.addAction("patch: %d [horizontal:%5.2f vertical:%5.2f]" % (frame, hor, ver)).setEnabled(False)
        menu.addAction("Show only this volume", lambda: self._apply(self.m.show_only_volume, pos))
        menu.addAction("Show only this slice", lambda: self._apply(self.m.show_only_slice, frame))
        menu.addAction("Create revolution mask " + NOT_PORTED).setEnabled(False)
        menu.exec(gpos)

    def refresh(self):
        sc = self.m.scene()
        th, tw = sc["tile"]
        gray = self.m.gray(sc)
        if self.mask is not None and self.act_mask.isChecked():
            img = self._imprint(sc, gray)
        else:
            img = gray_qimage(gray)
        rows, cols = sc["rows"], sc["cols"]
        self.canvas.set_image(img, [c * tw for c in range(cols + 1)], [r * th for r in range(rows + 1)], 1,
                              [(r * th, c * tw, t) for r, c, t in sc["labels"]])
        self.e_cols.setText(str(self.m.cols))
        self.e_rows.setText(str(self.m.rows))
        self.e_current.setText(str(self.m.first_frame))
        self.e_interval.setText(str(self.m.interval))
        self.r_thick.blockSignals(True)
        self.r_thick.setChecked(self.m.thick_on)
        self.r_thick.blockSignals(False)
        self.e_thick.setText(str(self.m.thickness))
        self.e_start.setText(str(self.m.selection[0]))
        self.e_end.setText(str(self.m.selection[-1]))
        for a, b in self.b_axis.items():
            b.setChecked(a == self.m.orientation)
            f = b.font()
            f.setBold(a == self.m.orientation)
            b.setFont(f)

    def _imprint(self, sc, gray):
        """ExternalMask/imprint.m + mbimg.overlays.imprintedMaskOnGray: the mask slices (same orientation, slab and
        tiles, repeated for each shown volume) tint the grey image red."""
        th, tw = sc["tile"]
        t = self.m.thickness if self.m.thick_on else 1
        mimg = np.zeros_like(gray, dtype=np.float64)
        for p, k in enumerate(sc["frames"]):
            r, c = divmod(p, sc["cols"])
            s = L.view_tile(self.mask, self.m.orientation, k, t)
            mimg[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = np.concatenate([s] * len(self.m.selection), axis=1)
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float64)
        rgb[..., 0] = np.clip(rgb[..., 0] + 0.1 * (mimg > 0), 0, 1)
        return rgb_qimage(rgb)


# =====================================================================================================================
# dview
# =====================================================================================================================

class DviewWindow(QtWidgets.QMainWindow):
    """dynamo__view.fig (screenshot 10): Export / List / Tools menus, zoom / pan / contrast / bandpass / previous /
    next icons, x y z buttons and the size text on top, the montage, 'temp.em' and the statistics text below."""

    def __init__(self, items, title=""):
        super().__init__()
        self.items = items                     # list of (name, array or None for a file read on demand)
        self.index = 0
        self.title = title
        self.axis = "z"
        self.cube = None
        self.setStyleSheet("QMainWindow > QWidget#central { background: rgb(179,179,179); }")
        self._menus()
        self._toolbar()
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        g = QtWidgets.QGridLayout(central)
        top = QtWidgets.QHBoxLayout()
        self.b = {}
        for a in ("x", "y", "z"):
            b = _button(a, lambda _=False, a=a: self.show_axis(a))
            b.setFixedSize(44, 27)
            self.b[a] = b
            top.addWidget(b)
        top.addStretch(1)
        self.text_size = QtWidgets.QLabel("size")
        top.addWidget(self.text_size)
        top.addStretch(1)
        g.addLayout(top, 0, 0)
        self.canvas = ImageCanvas(background=QtGui.QColor(179, 179, 179))
        g.addWidget(self.canvas, 1, 0)
        bottom = QtWidgets.QHBoxLayout()
        self.edit_save = QtWidgets.QLineEdit("temp.em")
        self.edit_save.setToolTip("save this volume with a different name")
        self.edit_save.setFixedWidth(250)
        bottom.addWidget(self.edit_save)
        bottom.addStretch(1)
        self.text_label = QtWidgets.QLabel("mean")
        bottom.addWidget(self.text_label)
        g.addLayout(bottom, 2, 0)
        g.setRowStretch(1, 1)
        self.setCentralWidget(central)
        self.resize(745, 804)
        self.load(0)

    def _menus(self):
        mb = self.menuBar()
        e = mb.addMenu("Export")
        e.addAction("Dynamo: mapview", lambda: self._to_mapview(0))
        b = e.addMenu(" bin and pass to mapview")
        for n in (1, 2, 3):
            b.addAction("bin x%d" % n, lambda n=n: self._to_mapview(n))
        e.addAction("into file", self._into_file)
        for lab in ("Dynamo: tomoshow", "Dynamo: tomoslice", "into Matlab figure", "into Chimera UCSF",
                    "Send to matlab volumeViewer", "Dynamo: isoview"):
            e.addAction(lab + " " + NOT_PORTED).setEnabled(False)
        self.menu_list = mb.addAction("List", self._list)
        t = mb.addMenu("Tools")
        t.addAction("Delete all dview figures", self.close)
        t.addAction("Set bright background", self._bright)

    def _toolbar(self):
        tb = self.addToolBar("dview")
        tb.setMovable(False)
        st = self.style()
        I = QtWidgets.QStyle
        # Zoom Out / Zoom In / Pan are modes, as in a MATLAB figure: click the image where to zoom. In every mode the
        # zoomed image can be grabbed and dragged; the wheel zooms at the cursor; a double click shows the whole image.
        self.act_zoom_out = tb.addAction(st.standardIcon(I.SP_ArrowDown), "Zoom Out (click on the image)")
        self.act_zoom_in = tb.addAction(st.standardIcon(I.SP_ArrowUp), "Zoom In (click on the image)")
        self.act_pan = tb.addAction(st.standardIcon(I.SP_DirOpenIcon), "Pan")
        group = QtGui.QActionGroup(self)
        group.setExclusionPolicy(QtGui.QActionGroup.ExclusionPolicy.ExclusiveOptional)
        for a in (self.act_zoom_out, self.act_zoom_in, self.act_pan):
            a.setCheckable(True)
            group.addAction(a)
        self.act_zoom_in.toggled.connect(lambda on: self._mode())
        self.act_zoom_out.toggled.connect(lambda on: self._mode())
        self.act_pan.toggled.connect(lambda on: self._mode())
        tb.addAction(st.standardIcon(I.SP_ComputerIcon), "Controls the contrast of image", self._contrast)
        tb.addAction(st.standardIcon(I.SP_MediaSeekForward), "bandpass filter",
                     lambda: _not_ported(self, "bandpass filter (dynamo_imfilter)"))
        self.act_prev = tb.addAction(st.standardIcon(I.SP_ArrowLeft), "previous element in the list",
                                     lambda: self.load(self.index - 1))
        self.act_next = tb.addAction(st.standardIcon(I.SP_ArrowRight), "next element in the list",
                                     lambda: self.load(self.index + 1))
        single = len(self.items) <= 1
        for a in (self.act_prev, self.act_next):
            a.setEnabled(not single)

    def load(self, i):
        """go_list (dynamo__view.m:317-365): read the item, reset to the z view and autocontrast."""
        if not 0 <= i < len(self.items):
            print("Cannot go to element")
            return
        self.index = i
        name, arr = self.items[i]
        try:
            v = L.as_volume(L.read_volume(arr if arr is not None else name))
        except Exception as e:                                      # dynamo__view.m: unreadable -> zeros(2,2,2)
            print("dview: could not read %s (%s)" % (name, e))
            v = np.zeros((2, 2, 2))
        self.cube = np.asarray(v, dtype=np.float64)
        label, size = L.dview_labels(self.cube)
        self.text_label.setText(label)
        self.text_size.setText(size)
        if len(self.items) > 1:
            self.setWindowTitle("dview: #%d %s" % (i + 1, name))
        else:
            self.setWindowTitle("dview: " + (self.title or ""))
        self.clim = None
        self.show_axis("z")

    def show_axis(self, axis):
        self.axis = axis
        sc = L.dview_gray(self.cube, axis)
        if self.clim is not None:
            sc["gray"] = L.to_gray(sc["cdata"], self.clim, 64)
        self.scene = sc
        th, tw = sc["tile"]
        xs, ys = L.montage_grid_lines(sc["rows"], sc["cols"], th, tw)
        self.canvas.zoom, self.canvas.center = 1.0, None
        self.canvas.set_image(gray_qimage(sc["gray"]), xs, ys, 2)
        for a, b in self.b.items():
            f = b.font()
            f.setBold(a == axis)
            b.setFont(f)

    def _mode(self):
        c = self.canvas
        c.zoom_mode = "in" if self.act_zoom_in.isChecked() else "out" if self.act_zoom_out.isChecked() else None
        c.pan_mode = self.act_pan.isChecked()
        c.update_cursor()

    def _contrast(self):
        """dynamo_imcontrast (dynamo__imcontrast.m): histogram of the montage CData; the limits act on CLim."""
        c = self.scene["cdata"]
        cl = self.scene["clim"]

        def apply(lo, hi):
            self.clim = (lo, hi)
            self.scene["gray"] = L.to_gray(c, self.clim, 64)
            self.canvas.qimage = gray_qimage(self.scene["gray"])
            self.canvas.update()
        # dynamo__imcontrast.m:378-397, 529-536: 100 bars; left click sets the maximum, right click the minimum
        ContrastDialog(self, c, cl, apply, bins=100, left_sets="upper").exec()

    def _bright(self):
        """dynamo__view.m:443-450: voxels outside dynamo_sphere(round(nx/2)-3) are set to max(cube), then z view."""
        nz, ny, nx = self.cube.shape
        r = round(nx / 2) - 3
        sph = L.ellipsoid(self.cube.shape, [r, r, r], [nx / 2 + 1] * 3)
        self.cube = np.where(sph, self.cube, self.cube.max())
        self.show_axis("z")

    def _to_mapview(self, nbin):
        from . import _launch
        v = L.dynamo_bin(self.cube, nbin) if nbin else self.cube
        _launch(["mapview"], arrays=[v])

    def _into_file(self):
        from ..console import dwrite
        dwrite(self.cube.astype(np.float32), self.edit_save.text())
        print("written %s" % self.edit_save.text())

    def _list(self):
        if len(self.items) <= 1:
            return
        names = ["%d  %s" % (i + 1, n) for i, (n, _) in enumerate(self.items)]
        item, ok = QtWidgets.QInputDialog.getItem(self, "List", "file", names, self.index, False)
        if ok:
            self.load(names.index(item))


# =====================================================================================================================
# dslices (masks window overlay, two maps side by side)
# =====================================================================================================================

class SlicesWindow(QtWidgets.QMainWindow):
    """figure(); dynamo_slices(...): a plain MATLAB figure (File menu, axes with the montage, red grid of width 2,
    optional axes title)."""

    def __init__(self, rgb_or_gray, rows, cols, tile, name="dynamo_slices", axes_title=None, nvol=1):
        super().__init__()
        self.setWindowTitle(name)
        f = self.menuBar().addMenu("File")
        f.addAction("Save image as ...", self._save)
        f.addAction("Close", self.close)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        if axes_title:
            t = QtWidgets.QLabel(axes_title)
            t.setAlignment(QtCore.Qt.AlignCenter)
            fnt = t.font()
            fnt.setBold(True)
            t.setFont(fnt)
            t.setWordWrap(True)
            lay.addWidget(t)
        self.canvas = ImageCanvas(background=QtGui.QColor(240, 240, 240))
        lay.addWidget(self.canvas, 1)
        lay.setContentsMargins(73, 8, 53, 46)                    # MATLAB's default axes position in a 560 x 420 figure
        self.setCentralWidget(central)
        img = rgb_qimage(rgb_or_gray) if rgb_or_gray.ndim == 3 else gray_qimage(rgb_or_gray)
        th, tw = tile
        xs, ys = L.montage_grid_lines(rows, cols, th, tw)
        intra = [c * tw + j * tw / nvol for c in range(cols) for j in range(1, nvol)]
        self.canvas.set_image(img, xs, ys, 2, intra=intra)
        self.resize(560, 420 + self.menuBar().sizeHint().height())

    def _save(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "dynamo_slices.png", "PNG (*.png)")
        if fn:
            self.grab().save(fn)


def run(windows, snap=None, actions=None, keep=False):
    """Show the windows and run the event loop. snap: save each window as a PNG after it is drawn and quit (tests and
    screenshots); actions: [(name, callable)] applied in order, with a snapshot after each."""
    app = QtWidgets.QApplication.instance()
    for w in windows:
        w.show()
    if snap:
        os.makedirs(os.path.dirname(os.path.abspath(snap)) or ".", exist_ok=True)
        for _ in range(5):
            app.processEvents()
        base, ext = os.path.splitext(snap)
        for i, w in enumerate(windows):
            w.grab().save(snap if len(windows) == 1 else "%s_%d%s" % (base, i + 1, ext or ".png"))
        for name, fn in actions or ():
            fn()
            for _ in range(5):
                app.processEvents()
            windows[0].grab().save("%s_%s%s" % (base, name, ext or ".png"))
        if not keep:
            for w in windows:
                w.close()
            return 0
    return app.exec()


# =====================================================================================================================
# MATLAB-like figures: dtplot (rays) and the adaptive-filtering FSC plot
# =====================================================================================================================

AXES_POSITION = (0.13, 0.11, 0.775, 0.815)       # MATLAB default axes Position (normalised, from the bottom left)


class FigureCanvas(QtWidgets.QWidget):
    """A 560 x 420 MATLAB figure with one axes at the default position; subclasses paint in data units."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 210)

    def axes_rect(self):
        x, y, w, h = AXES_POSITION
        W, H = self.width(), self.height()
        return QtCore.QRectF(x * W, (1 - y - h) * H, w * W, h * H)


class TablePlotCanvas(FigureCanvas):
    """dynamo_tableplot(table) default 'rays' (screenshot 30): top view (View [0 90]), axis equal and off, black axis
    lines of length 0.9 and width 2 with the labels x, y, z (font 12) at their middles (drawaxis(0.9,2,1,12)), points
    as filled circles of SizeData 40 pt^2 (8.4 px at 96 ppi) in one colour: CData 0 on parula(64) with CLim [-1 1]
    = RGB (24, 191, 181) measured on Dynamo's window [MATLAB built-in colormap]."""
    COLOR = QtGui.QColor(24, 191, 181)

    def __init__(self, xyz, parent=None):
        super().__init__(parent)
        self.xyz = np.asarray(xyz, dtype=np.float64)

    def _limits(self, r):
        xs = np.concatenate([self.xyz[:, 0], [0.0, 0.9]])
        ys = np.concatenate([self.xyz[:, 1], [0.0, 0.9]])
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        s = min(r.width() / max(x1 - x0, 1e-9), r.height() / max(y1 - y0, 1e-9))   # axis equal: one scale
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return s, r.center().x() - s * cx, r.center().y() + s * cy

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        s, ox, oy = self._limits(self.axes_rect())

        def to(x, y):
            return QtCore.QPointF(ox + s * x, oy - s * y)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setWidthF(2 * 96 / 72)
        p.setPen(pen)
        p.drawLine(to(0, 0), to(0.9, 0))
        p.drawLine(to(0, 0), to(0, 0.9))
        f = QtGui.QFont()
        f.setPointSize(12)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        for lab, (x, y) in (("x", (0.45, 0)), ("y", (0, 0.45)), ("z", (0, 0))):
            q = to(x, y)                               # text: HorizontalAlignment left, VerticalAlignment middle
            p.drawText(QtCore.QPointF(q.x(), q.y() + fm.ascent() / 2 - 1), lab)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self.COLOR)
        d = math.sqrt(40) * 96 / 72
        order = np.argsort(self.xyz[:, 2], kind="stable")   # seen from +z: higher points drawn last
        for i in order:
            q = to(self.xyz[i, 0], self.xyz[i, 1])
            p.drawEllipse(q, d / 2, d / 2)


class FscCanvas(FigureCanvas):
    """figure(1); hold on; plot(fsc) per iteration (+dpkconv/plotConvergence.m:16-21): x = 1..n, default ColorOrder,
    axes Box off, ticks out of matlab_auto_limits. Left click on a curve opens Dynamo's menu (plotConvergence.m:36-52).
    """
    curveClicked = QtCore.Signal(int, int, QtCore.QPoint)

    def __init__(self, curves, parent=None):
        super().__init__(parent)
        self.curves = curves
        xs = np.concatenate([c[1] for c in curves]) if curves else np.array([0.0, 1.0])
        ys = np.concatenate([c[2] for c in curves]) if curves else np.array([0.0, 1.0])
        self.xl = L.matlab_auto_limits(xs.min(), xs.max())
        self.yl = L.matlab_auto_limits(ys.min(), ys.max())

    def _map(self):
        r = self.axes_rect()
        (x0, x1, _), (y0, y1, _) = self.xl, self.yl

        def to(x, y):
            return QtCore.QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(),
                                  r.bottom() - (y - y0) / (y1 - y0) * r.height())
        return r, to

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        r, to = self._map()
        p.fillRect(r, QtGui.QColor(255, 255, 255))
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for i, (ite, x, y) in enumerate(self.curves):
            c = L.MATLAB_COLORORDER[i % len(L.MATLAB_COLORORDER)]
            p.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(*c), 1.0))
            p.drawPolyline(QtGui.QPolygonF([to(a, b) for a, b in zip(x, y)]))
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.setPen(QtGui.QPen(QtGui.QColor(38, 38, 38), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        p.drawLine(r.bottomLeft(), r.topLeft())
        f = QtGui.QFont()
        f.setPointSize(9)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        tl = 0.01 * max(r.width(), r.height())
        for t in self.xl[2]:
            q = to(t, self.yl[0])
            p.drawLine(q, QtCore.QPointF(q.x(), q.y() - tl))
            s = "%g" % t
            p.drawText(QtCore.QPointF(q.x() - fm.horizontalAdvance(s) / 2, q.y() + fm.ascent() + 3), s)
        for t in self.yl[2]:
            q = to(self.xl[0], t)
            p.drawLine(q, QtCore.QPointF(q.x() + tl, q.y()))
            s = "%g" % t
            p.drawText(QtCore.QPointF(q.x() - fm.horizontalAdvance(s) - 4, q.y() + fm.ascent() / 2 - 1), s)

    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self.curves:
            return
        r, to = self._map()
        pos = ev.position()
        best, bd = None, 6.0                         # a click within 6 pixels of a curve hits it
        for ite, x, y in self.curves:
            pts = [to(a, b) for a, b in zip(x, y)]
            for a, b in zip(pts[:-1], pts[1:]):
                d = _seg_dist(pos, a, b)
                if d < bd:
                    best, bd = ite, d
        if best is not None:
            (x0, x1, _) = self.xl
            xdata = x0 + (pos.x() - r.left()) / r.width() * (x1 - x0)
            self.curveClicked.emit(best, int(round(xdata)), ev.globalPosition().toPoint())


def _seg_dist(p, a, b):
    ax, ay, bx, by, px, py = a.x(), a.y(), b.x(), b.y(), p.x(), p.y()
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


class FigureWindow(QtWidgets.QMainWindow):
    """A plain MATLAB figure window (File menu) holding one canvas, 560 x 420."""

    def __init__(self, canvas, name=""):
        super().__init__()
        self.setWindowTitle(name)
        f = self.menuBar().addMenu("File")
        f.addAction("Save image as ...", self._save)
        f.addAction("Close", self.close)
        self.canvas = canvas
        self.setCentralWidget(canvas)
        self.resize(560, 420 + self.menuBar().sizeHint().height())

    def _save(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "figure.png", "PNG (*.png)")
        if fn:
            self.canvas.grab().save(fn)


def fsc_window(name, curves):
    """The adaptive-filtering FSC plot of project `name` with Dynamo's click menu."""
    canvas = FscCanvas(curves)
    w = FigureWindow(canvas, "Figure 1")

    def menu(ite, x, gpos):
        print("Plot for project %s iteration %d" % (name, ite))
        m = QtWidgets.QMenu(w)
        m.addAction("Mapview of computed averages (ite: %d)" % ite, lambda: _mapview_ite(name, ite))
        m.addAction("Mapview of computed averages (bandpassed to %d fourier pixels) %s" % (x, NOT_PORTED)
                    ).setEnabled(False)
        m.exec(gpos)
    canvas.curveClicked.connect(menu)
    return w


def _mapview_ite(name, ite):
    """plotConvergence.m:54-68 localShow: dynamo_mapview of the averages of every reference at that iteration."""
    from . import _launch
    files = L.last_average_files(name, ite)
    print("Mapview for project %s iteration %d" % (name, ite))
    if files:
        _launch(["mapview"] + files)
