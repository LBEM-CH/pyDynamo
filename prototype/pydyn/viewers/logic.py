"""Display logic of Dynamo 1.1.514's viewers, in pure NumPy (no Qt, no display): what each tile shows, how the tiles
are laid out and how intensities become grey levels. The Qt windows (pydyn.viewers.qtview) only paint these arrays.

Spec: docs/analysis/11a_viewers_1.1.514.md. S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src,
MB = /programs/x86_64-linux/dynamo/1.1.514/matlab/mbtools.

Array convention: volumes are numpy [z, y, x] (pydyn's convention, MATLAB vol(x, y, z)). A screen image is a 2-D
array [row from the top, column from the left]. Frame/slice numbers are 1-based, as Dynamo shows them.

The three viewers ported here:
* dmapview (S/dynamo_mapview.m:79 -> S/+dpkvol/montage.m -> +dpkvol/+montages/+aux/+mapview, MB/+mbvid/+guis/...):
  MapviewModel.
* dview (S/dynamo_view.m -> S/dynamo__view.m): dview_* functions.
* dslices, the masks window overlay (S/dynamo_slices.m, S/dynamo__wizard_masks.m:460-517): dslices_* functions.
"""
import math

import numpy as np

from .. import matlab_numerics as MN

AXES = ("x", "y", "z")


# =====================================================================================================================
# reading
# =====================================================================================================================

def read_volume(obj):
    """dynamo_file_checkout / dynamo_read: a file name is read with pydyn's dread (dynamo_read.m), an array is used
    as is. No normalisation on read (dynamo_read.m:286-386). Logical input becomes single (dynamo__view.m:151-153)."""
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        from ..console import dread
        v = dread(str(obj))
    else:
        v = obj
    v = np.asarray(v)
    if v.dtype == bool:
        v = v.astype(np.float32)
    if v.dtype.kind == "c":
        v = v.real
    return v


def as_volume(v):
    """2-D arrays are not shown by these windows in Dynamo (dynamo_view.m:213-228 sends them to dshow); pydyn keeps
    only volumes."""
    v = np.asarray(v)
    if v.ndim != 3:
        raise NotImplementedError("pydyn viewers show 3-D volumes only (got an array of shape %s)" % (v.shape,))
    return v


# =====================================================================================================================
# MATLAB / Dynamo helpers
# =====================================================================================================================

def maprange(v):
    """S/maprange.m:52-59 with a=0, b=1: (v - min) / max(v - min); min/max ignore NaN (MATLAB min/max). A flat input
    gives 0/0 = NaN as in MATLAB."""
    v = np.asarray(v, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = v - np.nanmin(v)
        return w / np.nanmax(w)


def mbimg_normalize(v):
    """MB/+mbimg/normalize.m:24-25: ibw - min, ./ max (same as maprange)."""
    return maprange(v)


def nice_2d_distribution(n):
    """S/nice_2d_distribution.m: (rows, columns) of dview / dslices montages. The smallest square >= n among 1..45^2
    gives the side q; then the fully empty rows are dropped."""
    ind = np.arange(1, 46) ** 2
    sq = int(ind[np.nonzero(ind - n >= 0)[0][0]])
    q = int(math.ceil(math.sqrt(sq)))
    rows = q - (q * q - n) // q
    return rows, q


def best_layout(container_wh, cell_wh, n):
    """MB/+mbgui/+layout/bestLayout.m: (rows, cols) that fill a panel of container_wh = (width, height) pixels with n
    cells of cell_wh = (width, height) data pixels."""
    wc, hc = float(cell_wh[0]), float(cell_wh[1])
    wp, hp = float(container_wh[0]), float(container_wh[1])
    final = (wp / hp) / (wc / hc)
    cols = int(math.ceil(math.sqrt(n * final)))
    rows = int(math.ceil(cols / final))
    while cols * rows - n >= cols:
        rows -= 1
    return rows, cols


def dynamo_bin(v, order):
    """S/dynamo_bin.m:86-120: `order` successive 2x2x2 mean binnings; each level floors the sizes (the last odd plane is
    dropped). Order 0 returns the input."""
    v = np.asarray(v)
    dt = v.dtype
    for _ in range(int(order)):
        if min(v.shape) < 2:
            print("Volume is too small in some dimension. Leaving unchanged.")
            return v
        nz, ny, nx = (s // 2 for s in v.shape)
        w = np.asarray(v[:2 * nz, :2 * ny, :2 * nx], dtype=np.float64)
        v = w.reshape(nz, 2, ny, 2, nx, 2).sum(axis=(1, 3, 5)) * (1.0 / 8.0)
        if dt.kind == "f":
            v = v.astype(dt)
    return v


def ellipsoid(shape_zyx, semiaxes_xyz, position_xyz):
    """dynamo_ellipsoid -> dpkgeom.sphere -> mex volume_noncubic::fillSphere (S/../mex/src/classes/
    volume_noncubic.cpp:641-653), no smoothing: voxel (x, y, z) (1-based) is 1 when sum(((r - p) / a)^2) <= 1."""
    nz, ny, nx = shape_zyx
    ax, ay, az = (float(a) for a in semiaxes_xyz)
    px, py, pz = (float(p) for p in position_xyz)
    if ax * ay * az == 0:                                     # dynamo_ellipsoid.m: prod(radii)==0 -> zeros
        return np.zeros(shape_zyx, dtype=bool)
    x = ((np.arange(1, nx + 1) - px) / ax) ** 2
    y = ((np.arange(1, ny + 1) - py) / ay) ** 2
    z = ((np.arange(1, nz + 1) - pz) / az) ** 2
    return (z[:, None, None] + y[None, :, None] + x[None, None, :]) <= 1


def to_gray(cdata, clim, levels):
    """MATLAB 'scaled' CDataMapping on a gray(levels) colormap: index = fix((C - cmin) / (cmax - cmin) * m), clamped
    to 0..m-1, grey = index / (m - 1). NaN gives the first colormap colour (black). Returns float32 in [0, 1]."""
    c = np.asarray(cdata, dtype=np.float64)
    c0, c1 = float(clim[0]), float(clim[1])
    if not c1 > c0:
        c1 = c0 + 1e-300                                      # MATLAB refuses equal limits; never reached by callers
    with np.errstate(invalid="ignore"):
        idx = np.fix((c - c0) / (c1 - c0) * levels)
    idx = np.where(np.isnan(idx), 0, np.clip(idx, 0, levels - 1))
    return (idx / (levels - 1)).astype(np.float32)


def matlab_g(fmt, *vals):
    """sprintf with %g conversions: C and Python agree for finite values; NaN/Inf print as MATLAB does."""
    out = []
    for v in vals:
        if isinstance(v, float) and math.isnan(v):
            out.append(float("nan"))
        else:
            out.append(v)
    s = fmt % tuple(out)
    return s.replace("nan", "NaN").replace("inf", "Inf")


# =====================================================================================================================
# orientation: one screen tile of a volume, seen along x, y or z
# =====================================================================================================================

def n_frames(shape_zyx, axis):
    """Frames along the viewing axis (getVideoPropertiesAtCurrentOrientation.m:17-31): x -> nx, y -> ny, z -> nz."""
    nz, ny, nx = shape_zyx
    return {"x": nx, "y": ny, "z": nz}[axis]


def slab_indices(k, thickness, n):
    """S/+dpkvol/+aux/+thickness/getListAroundSliceBasic.m, intersected with 1..n (readRawSlice.m:46-49):
    odd t -> k-(t-1)/2 .. k+(t-1)/2; even t -> k-t/2 .. k+t/2-1."""
    t = int(thickness)
    if t % 2:
        a = (t - 1) // 2
        idx = range(k - a, k + a + 1)
    else:
        idx = range(k - t // 2, k + t // 2)
    return [i for i in idx if 1 <= i <= n]


def view_tile(v, axis, k, thickness=1):
    """Screen tile of frame k (1-based) along `axis`, as mapview draws it (readRawSlice.m:50-72 then
    volumeframe2shownFragment.m:8-10: mean over the slab, flipud for y, transpose, flipud). In numpy [z, y, x]:
      z: v[k-1, ::-1, :]   rows = y from ny (top) down to 1, columns = x 1..nx
      x: v[::-1, :, k-1]   rows = z from nz down to 1,       columns = y 1..ny
      y: v[::-1, k-1, ::-1] rows = z from nz down to 1,      columns = x from nx down to 1
    dview's montage (dynamo_rot_solid + fliplr + imagesc(fused'), dynamo__view.m:245-264) gives the same tiles."""
    n = n_frames(v.shape, axis)
    idx = [i - 1 for i in slab_indices(k, thickness, n)]
    if not idx:
        raise IndexError("frame %d outside 1..%d" % (k, n))
    if axis == "z":
        s = v[idx, :, :].sum(axis=0, dtype=np.float64) / len(idx)
        return s[::-1, :]
    if axis == "x":
        s = v[:, :, idx].sum(axis=2, dtype=np.float64) / len(idx)
        return s[::-1, :]
    s = v[:, idx, :].sum(axis=1, dtype=np.float64) / len(idx)
    return s[::-1, ::-1]


def tile_shape(shape_zyx, axis):
    """(rows, cols) of one tile."""
    nz, ny, nx = shape_zyx
    return {"z": (ny, nx), "x": (nz, ny), "y": (nz, nx)}[axis]


# =====================================================================================================================
# dview (S/dynamo__view.m)
# =====================================================================================================================

def dview_labels(v):
    """dynamo__view.m:126-147: text_label and text_size."""
    v = np.asarray(v)
    with np.errstate(invalid="ignore"):
        mean_value = float(np.mean(v, dtype=np.float64))
    if math.isnan(mean_value):
        label = "Flat value (%5.2f)" % float("nan")
        label = label.replace("nan", "NaN")
    elif np.all(v == mean_value):
        label = "Flat value (%5.2f)" % mean_value
    else:
        label = matlab_g("Max: %5.2g Min: %5.2g Mean: %5.2g", float(np.nanmax(v)), float(np.nanmin(v)), mean_value)
    nz, ny, nx = v.shape
    return label, "%d x %d x %d voxels" % (nx, ny, nz)


def assemble(tiles, rows, cols, fill=0.0):
    """Tiles (all the same shape) placed left to right, then top to bottom (dynamo_montage_no_license.m:42-50,
    tile l at column i, row j with l = i + (j-1) * columns). Empty trailing tiles get `fill`."""
    th, tw = tiles[0].shape[:2]
    extra = tiles[0].shape[2:]
    out = np.full((rows * th, cols * tw) + extra, fill, dtype=np.float64)
    for p, t in enumerate(tiles):
        r, c = divmod(p, cols)
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return out


def dview_scene(v, axis="z"):
    """What dview draws for one view (dynamo__view.m:154-160, 245-264): a montage of every slice along `axis`
    (dynamo_montage(cube,[],[],0)): CData = maprange over all the slices together (dynamo_montage.m:131-146), empty
    trailing tiles 0. Returns dict(cdata, rows, cols, tile=(h, w))."""
    v = as_volume(v)
    n = n_frames(v.shape, axis)
    tiles = [view_tile(v, axis, k) for k in range(1, n + 1)]
    stack = maprange(np.stack(tiles))
    rows, cols = nice_2d_distribution(n)
    return {"cdata": assemble(list(stack), rows, cols, 0.0), "rows": rows, "cols": cols, "tile": tiles[0].shape}


def dview_clim(v):
    """dpkguis.view.setAutocontrast -> dpkvol.aux.contrast.getNormalizedCLimInInterior(cube, 'r', 2): the min and
    max of the cube inside the ellipsoid of semi-axes floor(N/2 - 2) centred at (nx/2+1)*[1,1,1]
    (dynamo_ellipsoid.m:98), mapped to the cube's [min, max] -> [0, 1] range. None when the interior is empty or the
    cube is flat ("Sorry, could not normalize the interior" / CLim left at MATLAB's automatic [0, 1])."""
    v = as_volume(v)
    nz, ny, nx = v.shape
    r = [math.floor(nx / 2 - 2), math.floor(ny / 2 - 2), math.floor(nz / 2 - 2)]
    m = ellipsoid(v.shape, r, [nx / 2 + 1] * 3)
    if not m.any():
        return None
    inner = v[m]
    cmin, cmax = float(np.nanmin(inner)), float(np.nanmax(inner))
    gmin, gmax = float(np.nanmin(v)), float(np.nanmax(v))
    g = gmax - gmin
    if g == 0:
        return None
    lo = (cmin - gmin) / g
    hi = lo + (cmax - cmin) / g
    if lo == hi:                                               # "Cannot adjust contrast: flat value detected"
        return None
    return lo, hi


def dview_gray(v, axis="z"):
    """Grey image of dview for one view: the montage CData on colormap gray (64 levels in R2019a,
    dynamo_montage_no_license.m:85) with the autocontrast CLim (or the automatic [min, max] of the CData)."""
    sc = dview_scene(v, axis)
    clim = dview_clim(v)
    if clim is None:
        c = sc["cdata"]
        clim = (float(np.nanmin(c)), float(np.nanmax(c)))
    sc["clim"] = clim
    sc["gray"] = to_gray(sc["cdata"], clim, 64)
    return sc


def montage_grid_lines(rows, cols, th, tw):
    """Red separator lines of dynamo_montage.m:183-211 (linewidth 2) in image pixel coordinates (0 = left/top edge
    of the image): verticals at every tile border, horizontals likewise; the outer frame sits on the first/last pixel
    centres (x = 1 and Lx*cols in MATLAB data units). Returns (vertical xs, horizontal ys) as pixel-edge positions."""
    xs = [c * tw for c in range(cols + 1)]
    ys = [r * th for r in range(rows + 1)]
    xs[0], xs[-1] = 0.5, cols * tw - 0.5
    ys[0], ys[-1] = 0.5, rows * th - 0.5
    return xs, ys


# =====================================================================================================================
# dslices: the masks-window overlay (S/dynamo_slices.m; dynamo__wizard_masks.m:460-517)
# =====================================================================================================================

DSLICES_EULERS = {"x": (0.0, 90.0, 90.0), "y": (0.0, 90.0, 180.0), "z": (0.0, 0.0, 0.0)}   # dynamo_slices.m:901-939


def dslices_rotate(v, kind):
    """dynamo_slices.m:1631-1636: dynamo_rot(vol, euler) unless euler == [0,0,0]. dynamo_rot of a cube is
    dynamo_shift_rot(double(v), [0,0,0], euler, all slices, 0) = mex option 0 with outscope 0 (dynamo_rot.m;
    dynamo_shift_rot.m:63), i.e. trilinear rotation with everything outside the inscribed sphere set to 0. pydyn's
    port of that mex (pydyn.dalign.mex_shift_rot) covers even cubic boxes; others raise NotImplementedError."""
    e = DSLICES_EULERS[kind]
    if e == (0.0, 0.0, 0.0):
        return np.asarray(v, dtype=np.float64)
    from ..dalign import mex_shift_rot
    return np.asarray(mex_shift_rot(np.asarray(v, dtype=np.float64), e, option=0, outscope=0.0), dtype=np.float64)


def dslices_tiles(v):
    """All slices along the 3rd (z) axis of the (rotated) volume, drawn as dynamo_montage draws them."""
    return [view_tile(v, "z", k) for k in range(1, v.shape[0] + 1)]


def dslices_scene(vols, kind="z"):
    """dynamo_slices(vols, kind) without options (dynamo_slices.m:1657-1793): every slice; for each slice the slices of
    all volumes side by side (fuse_cellarray along dim 1 = horizontally); joint maprange over everything shown
    (norm_slice false); layout nice_2d_distribution; empty trailing tiles 0. All volumes must have one size here
    (Dynamo pads them to a common cube, dynamo_cellcube; not ported)."""
    vols = [as_volume(v) for v in vols]
    if any(v.shape != vols[0].shape for v in vols):
        raise NotImplementedError("dslices of volumes of different sizes (dynamo_cellcube padding) is not ported")
    rot = [dslices_rotate(v, kind) for v in vols]
    per = [dslices_tiles(r) for r in rot]
    fused = [np.concatenate([p[k] for p in per], axis=1) for k in range(len(per[0]))]
    stack = maprange(np.stack(fused))
    rows, cols = nice_2d_distribution(len(fused))
    return {"cdata": assemble(list(stack), rows, cols, 0.0), "rows": rows, "cols": cols,
            "tile": fused[0].shape, "nvol": len(vols)}


def dslices_overlay(template, mask, kind="z", alpha=0.2, color=(1.0, 0.0, 0.0)):
    """dynamo_slices(template, kind, 'overlay', mask, 'overlay_as', 'mask') (dynamo_slices.m:1939-1981, defaults alpha
    0.2 at :695 and overlay_color 'r'): the template montage in grey (colormap gray, 64 levels, CLim automatic = [0, 1]
    of its maprange CData), then the montage of -mask (same rotation, same slices) drawn on top with AlphaData 0.2 and
    truecolor CData color * (1 - CData(-mask)) = color * maprange(mask). Empty trailing tiles have CData 0 in the
    overlay, so they are tinted fully. Returns dict(rgb [h, w, 3] float in [0,1], rows, cols, tile)."""
    t = as_volume(template)
    m = as_volume(mask)
    if t.shape != m.shape:
        raise ValueError("Mask and volume do not have the same size")
    base = dslices_scene([t], kind)
    over = dslices_scene([-np.asarray(m, dtype=np.float64)], kind)
    c = base["cdata"]
    gray = to_gray(c, (float(np.nanmin(c)), float(np.nanmax(c))), 64).astype(np.float64)
    pattern = 1.0 - np.nan_to_num(over["cdata"], nan=0.0)
    col = np.asarray(color, dtype=np.float64)
    rgb = (1.0 - alpha) * gray[..., None] + alpha * (col[None, None, :] * pattern[..., None])
    return {"rgb": np.clip(rgb, 0, 1), "rows": base["rows"], "cols": base["cols"], "tile": base["tile"]}


# =====================================================================================================================
# dmapview (S/+dpkvol/montage.m and the mapview classes)
# =====================================================================================================================

MAPVIEW_PANEL = (950, 420)     # main panel of the opened window in pixels (screenshots: figure 950 x 500, bottom 80)
MAPVIEW_DEFAULT_LAYOUT = (4, 5)  # rows, cols of the first imshow before setFullVideo (MB/+mbvid/+guis/@Montage)


class MapviewModel:
    """State of one dmapview window. Everything that changes what is drawn goes through this class; the Qt view only
    calls scene() / gray() and paints."""

    def __init__(self, volumes, bin=0, panel=MAPVIEW_PANEL, names=None):
        vols = [as_volume(read_volume(v)) for v in volumes]
        if bin:                                                # VolumeStore/initialize.m:37-39
            vols = [dynamo_bin(v, bin) for v in vols]
        if not vols:
            raise ValueError("Could not initialize the data for the mapview montage")
        for ax, name in ((2, "x"), (1, "y"), (0, "z")):        # VolumeStore/initialize.m:50-61
            if any(v.shape[ax] != vols[0].shape[ax] for v in vols):
                raise ValueError("Volumes do not have the same %s length" % name)
        self.names = list(names) if names else ["<numeric variable>"] * len(vols)
        self.original = [np.asarray(v, dtype=np.float64) for v in vols]
        self.volumes = list(self.original)                     # transformedVolumes
        self.selection = [1]                                   # montage.m:99-104 (only volume #1 on opening)
        self.orientation = "z"
        self.thick_on = False
        self.thickness = 1
        self.norm_per_slice = False                            # View menu, both unchecked
        self.norm_per_volume = False
        self.first_frame = 1
        self.rows, self.cols = MAPVIEW_DEFAULT_LAYOUT
        self.interval = 1
        self.panel = tuple(panel)
        self.accessible = list(range(1, self.n_frames() + 1))
        # first imshow(M, []) on the default layout fixes CLim (showPackageGray.m:57, Montage.m:71-72)
        self.clim = self._data_range(self.scene()["image"])
        self.full_video()                                      # setFullVideo.m:26-33

    # ---- geometry -----------------------------------------------------------------------------------------------
    @property
    def n_volumes(self):
        return len(self.volumes)

    def n_frames(self):
        return n_frames(self.volumes[0].shape, self.orientation)

    def tile_shape(self):
        th, tw = tile_shape(self.volumes[0].shape, self.orientation)
        return th, tw * len(self.selection)

    def frames_shown(self):
        """getFramesToDepict.m:4-17: rows*cols accessible frames from the current first frame."""
        if self.first_frame in self.accessible:
            i0 = self.accessible.index(self.first_frame)
        else:
            i0 = 0
        return self.accessible[i0:i0 + self.rows * self.cols]

    def fused_tile(self, k):
        """Video/read.m:38-69: the selected volumes side by side (cat(2, ...)), each optionally normalised, then the
        fused tile optionally normalised."""
        parts = []
        t = self.thickness if self.thick_on else 1
        for i in self.selection:
            s = view_tile(self.volumes[i - 1], self.orientation, k, t)
            if self.norm_per_volume:
                s = mbimg_normalize(s)
            parts.append(s)
        f = np.concatenate(parts, axis=1)
        if self.norm_per_slice:
            f = mbimg_normalize(f)
        return f

    def scene(self):
        """The fused montage: dict(image [rows*th, cols*tw] with NaN in unused tiles (assembleMatrixFromScratch.m:44),
        frames, rows, cols, tile, labels[(row, col, text)])."""
        frames = self.frames_shown()
        th, tw = self.tile_shape()
        img = np.full((self.rows * th, self.cols * tw), np.nan)
        labels = []
        for p, k in enumerate(frames):
            r, c = divmod(p, self.cols)                        # frameNumber2ColumnRow.m:23, row major
            img[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = self.fused_tile(k)
            labels.append((r, c, str(k)))                      # computeFrameLabelLocations.m:28-35
        return {"image": img, "frames": frames, "rows": self.rows, "cols": self.cols, "tile": (th, tw),
                "labels": labels, "nshown": len(self.selection)}

    def gray(self, scene=None):
        """imshow(M, []) with the stored CLim on gray(256); NaN (unused tiles) -> black."""
        sc = scene or self.scene()
        return to_gray(sc["image"], self.clim, 256)

    @staticmethod
    def _data_range(img):
        return float(np.nanmin(img)), float(np.nanmax(img))

    # ---- layout and paging -------------------------------------------------------------------------------------
    def optimal_layout(self):
        """setOptimalLayout.m:26-65 -> bestLayout(panel, [w, h] of a tile, number of accessible frames)."""
        th, tw = self.tile_shape()
        self.rows, self.cols = best_layout(self.panel, (tw, th), len(self.accessible))

    def full_video(self):
        """setFullVideo.m:26-33: all frames, first frame 1, optimal layout."""
        self.accessible = list(range(1, self.n_frames() + 1))
        self.first_frame = 1
        self.interval = 1
        self.optimal_layout()

    def page(self, direction, chunk=None):
        """getNumberNextFirstFrame.m: move by chunk accessible frames (rows*cols for << >>, 1 for < >); refused past the
        end, clamped at the first frame. Returns False when refused."""
        chunk = self.rows * self.cols if chunk is None else chunk
        if self.first_frame in self.accessible:
            i = self.accessible.index(self.first_frame) + 1 + direction * chunk
        else:
            i = 1
        if i > len(self.accessible):
            print("Cannot get next frame for current first frame %d using chunk size of %d" % (self.first_frame, chunk))
            return False
        self.first_frame = self.accessible[max(i, 1) - 1]
        return True

    def set_current(self, k):
        """The 'current' box: jump to frame k, or the nearest accessible frame."""
        k = int(round(float(k)))
        a = np.asarray(self.accessible)
        self.first_frame = int(a[np.argmin(np.abs(a - k))])

    def set_interval(self, n):
        """performIntervalChange.m:33-44: frames first_frame +- multiples of n inside the frames available at opening."""
        n = int(n)
        if n < 1:
            raise ValueError("Interval is assumed to be integer and at least than one.")
        orig = list(range(1, self.n_frames() + 1))
        f = self.first_frame
        frames = set(range(f, orig[0] - 1, -n)) | set(range(f, orig[-1] + 1, n))
        self.accessible = sorted(frames & set(orig))
        self.interval = n

    def set_rows_cols(self, rows=None, cols=None):
        if rows is not None:
            self.rows = max(1, int(rows))
        if cols is not None:
            self.cols = max(1, int(cols))

    # ---- orientation / volumes / thickness -----------------------------------------------------------------------
    def set_orientation(self, axis):
        """listenToOrienter.m:18-27: first frame, layout and contrast are kept; the frames become 1..N of the new
        axis."""
        if axis not in AXES or axis == self.orientation:
            return
        self.orientation = axis
        n = self.n_frames()
        self.accessible = list(range(1, n + 1))
        if self.first_frame > n:
            self.first_frame = 1

    def set_selection(self, start, end):
        """Selector/setControlsInGUI.m:109-238: rounded, clamped to 1..n, start <= end."""
        n = self.n_volumes
        s = min(max(int(round(float(start))), 1), n)
        e = min(max(int(round(float(end))), 1), n)
        if e < s:
            e = s
        self.selection = list(range(s, e + 1))

    def show_all_volumes(self):
        self.selection = list(range(1, self.n_volumes + 1))

    def show_only_slice(self, frame):
        """ContextManager/addSelectThisSlice.m:24-27."""
        self.first_frame = int(frame)
        self.rows, self.cols = 1, 1

    def show_only_volume(self, position):
        """ContextManager/addSelectThisVolume.m:26: the position inside the shown list is used as the volume index
        (Dynamo 1.1.514 does this even when the shown range does not start at 1)."""
        self.selection = [min(max(int(position), 1), self.n_volumes)]

    def set_thick(self, on, value=None):
        """readRawSlice.m:16-27: an empty value resets the box to 1."""
        self.thick_on = bool(on)
        if value is not None:
            try:
                self.thickness = max(1, int(float(value)))
            except ValueError:
                print("thickness set to default value of 1")
                self.thickness = 1

    # ---- volume operations (launchGraymapToolsInBar.m), cumulative on the transformed volumes ---------------------
    def op_invert(self):
        self.volumes = [-v for v in self.volumes]

    def op_normalize(self):
        """dynamo_normalize_roi without a mask: (v - mean) / std over the non-NaN voxels (std with N-1)."""
        out = []
        for v in self.volumes:
            ones = np.ones(v.shape)                             # dynamo_normalize_roi.m:106-111: no mask ->
            w = MN.normalize_roi(MN.normalize_roi(v, ones), ones)  # dynamo_normalize_roi_one twice
            out.append(w)
        self.volumes = out

    def op_restore(self):
        self.volumes = list(self.original)
        self.accessible = list(range(1, self.n_frames() + 1))
        if self.first_frame > self.n_frames():
            self.first_frame = 1

    def op_bin(self, order):
        self.volumes = [dynamo_bin(v, int(float(order))) for v in self.volumes]
        n = self.n_frames()
        self.accessible = [k for k in self.accessible if k <= n] or list(range(1, n + 1))
        if self.first_frame > n:
            self.first_frame = 1
        self.clim = self._data_range(self.scene()["image"])     # a new image size re-creates the image (CLim refit)

    def op_band(self, spec):
        """VolumeStore/bandPass.m + bandPassOne.m: V = fftshift(fftn(v)); real(ifftn(ifftshift(V .* MASK))).
        One value -> [high, low] = [0, value]; two values -> [high, low]; a third (smoothness) is ignored. If low < 1
        both are Nyquist fractions, else Fourier pixels (divided by max(size)/2). MASK = ellipsoid(size*low/2) .*
        ~ellipsoid(size*high/2) (localMaskNyquist)."""
        vals = [float(x) for x in str(spec).replace(",", " ").replace("[", " ").replace("]", " ").split()]
        if not vals:
            return
        high, low = (0.0, vals[0]) if len(vals) == 1 else (vals[0], vals[1])
        shape = self.volumes[0].shape
        if low >= 1:
            half = max(shape) / 2.0
            high, low = high / half, low / half
        m = _band_mask(shape, high, low)
        self.volumes = [np.real(np.fft.ifftn(np.fft.ifftshift(np.fft.fftshift(np.fft.fftn(v)) * m)))
                        for v in self.volumes]

    # ---- contrast -------------------------------------------------------------------------------------------------
    def contrast_full(self):
        """MB/+mbimg/+contrast/setFullContrast.m:19-26: CLim = [min, max] of the current scene (NaN ignored)."""
        lo, hi = self._data_range(self.scene()["image"])
        if lo == hi:
            print("Flat value, cannot adjust contrast")
            return False
        self.clim = (lo, hi)
        return True

    def autocontrast(self, nbars=255):
        """MB/+mbimg/autocontrast.m:83-102: histogram of the scene in 255 bars; CLim from the first bar where the
        cumulative fraction exceeds 1% to the last bar where the reverse cumulative fraction exceeds 1%, extended by
        half a bar."""
        img = self.scene()["image"]
        self.clim = autocontrast_limits(img, nbars)

    def set_clim(self, lo, hi):
        if not hi > lo:
            raise ValueError("the upper limit must be larger than the lower one")
        self.clim = (float(lo), float(hi))

    def set_normalization(self, per_slice=None, per_volume=None):
        """View menu items; each toggle then applies full contrast (FM/launchSliceNormalizationControl.m)."""
        if per_slice is not None:
            self.norm_per_slice = bool(per_slice)
        if per_volume is not None:
            self.norm_per_volume = bool(per_volume)
        self.contrast_full()

    # ---- coordinates ----------------------------------------------------------------------------------------------
    def locate(self, row, col):
        """Image pixel (row, col) of the scene -> (frame, volume position in the shown list (1-based), horizontal,
        vertical) inside the tile (FMa/@ContextHandler/prettyCoordinateLabel.m:13), or None outside the frames."""
        th, tw = self.tile_shape()
        r, c = int(row // th), int(col // tw)
        p = r * self.cols + c
        frames = self.frames_shown()
        if row < 0 or col < 0 or r >= self.rows or c >= self.cols or p >= len(frames):
            return None
        wv = tw // len(self.selection)
        h = col - c * tw
        vtx = th - (row - r * th)
        return frames[p], int(h // wv) + 1, float(h), float(vtx)


def autocontrast_limits(img, nbars=255):
    """MATLAB hist(x, nbars): nbars equal bins between min and max (centres x), NaN not counted."""
    x = np.asarray(img, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    lo, hi = float(x.min()), float(x.max())
    if hi == lo:
        return lo - 0.5, hi + 0.5
    w = (hi - lo) / nbars
    centres = lo + w * (np.arange(nbars) + 0.5)
    edges = lo + w * np.arange(1, nbars)
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=nbars)[:nbars]
    total = counts.sum()
    yplus = np.cumsum(counts) / total
    yminus = np.cumsum(counts[::-1])[::-1] / total
    first = int(np.nonzero(yplus > 0.01)[0][0])
    last = int(np.nonzero(yminus > 0.01)[0][-1])
    bottom = centres[first] - w / 2
    top = centres[last] + w / 2
    return tuple(sorted((bottom, top)))


def _band_mask(shape_zyx, high, low):
    """bandPass.m localMaskNyquist: dynamo_ellipsoid(sidelength*low/2, sidelength) .* ~dynamo_ellipsoid(sidelength*
    high/2, sidelength); centre (nx/2+1)*[1,1,1] (dynamo_ellipsoid.m:98); a zero radius gives an empty ellipsoid."""
    nz, ny, nx = shape_zyx
    c = [nx / 2 + 1] * 3
    lo = ellipsoid(shape_zyx, [nx * low / 2, ny * low / 2, nz * low / 2], c)
    hi = ellipsoid(shape_zyx, [nx * high / 2, ny * high / 2, nz * high / 2], c)
    return (lo & ~hi).astype(np.float64)


# =====================================================================================================================
# dcp hooks: the files the dcp windows hand to the viewers
# =====================================================================================================================

def last_average_files(project_folder, ite):
    """dpkproject.quickplot 'lastAverage' (+dpkproject/quickplot.m:13-31): average, ref '*', ite 'last'. The file names
    are results/ite_NNNN/averages/average_ref_RRR_ite_NNNN.em (dynamo_database_convention.m:596-601,
    dynamo_database_locate.m:559, 892-897). Returns the existing files, ref 1, 2, ..."""
    import glob
    import os
    d = os.path.join(project_folder, "results", "ite_%04d" % int(ite), "averages")
    return sorted(glob.glob(os.path.join(d, "average_ref_[0-9][0-9][0-9]_ite_%04d.em" % int(ite))))


# =====================================================================================================================
# dtplot (S/dynamo_tableplot.m, default modus 'rays') and the adaptive-filtering FSC plot (S/+dpkconv)
# =====================================================================================================================

def read_table(obj):
    """dynamo__parse_get_table: a table file (the '.tbl' extension may be skipped) or an array [rows, columns]."""
    import os
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        f = os.fspath(obj)
        if not os.path.isfile(f) and os.path.isfile(f + ".tbl"):
            f += ".tbl"
        from ..console import dread
        t = dread(f)
    else:
        t = obj
    t = np.asarray(t)
    if t.dtype.kind == "c":
        t = t.real
    return np.atleast_2d(np.asarray(t, dtype=np.float64))


def plotangles_points(angles, rmin=1.0, rmax=1.6):
    """dynamo_plotangles.m:76-201 with rays off: each row (tdrot, tilt, narot) becomes m' * [0; 0; r] (m =
    dynamo_euler2matrix, so the point is r * (sin tdrot sin tilt, cos tdrot sin tilt, cos tilt), narot does not move
    it). Points with exactly the same direction are stacked outwards by (rmax - rmin) / (largest group), ordered by
    narot. 1.1.514 compares a new point only with the most recently opened group (`for iclass=length(orientations)`),
    so equal directions stack only when they are consecutive in the table. Returns an [n, 3] array."""
    from ..dalign import euler2matrix
    a = np.asarray(angles, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    first = np.array([euler2matrix(td, ti, 0.0).T @ np.array([0.0, 0.0, rmin]) for td, ti, _ in a])
    groups = [[0]]
    for i in range(1, n):
        if np.all(first[i] == first[groups[-1][0]]):
            groups[-1].append(i)
        else:
            groups.append([i])
    step = (rmax - rmin) / max(len(g) for g in groups)
    out = np.zeros((n, 3))
    for g in groups:
        order = [g[j] for j in np.argsort(a[g, 2], kind="stable")]       # MATLAB sort is stable
        for counter, ip in enumerate(order):
            td, ti, na = a[ip]
            out[ip] = euler2matrix(td, ti, na).T @ np.array([0.0, 0.0, rmin + counter * step])
    return out


def dtplot_points(table):
    """dynamo_tableplot(table) with no flags (:959-1063): all rows, sorted by tag (dynamo_table_sort), angles from
    columns 7:9, one colour (values 0). Returns (tags, xyz)."""
    t = read_table(table)
    t = t[np.argsort(t[:, 0], kind="stable")]
    return t[:, 0].astype(int), plotangles_points(t[:, 6:9])


def bandpass_fsc_curves(name, last_ite):
    """dpkconv.plotConvergenceAll (+dpkconv/plotConvergenceAll.m, plotConvergence.m:16-29): for ite = 1..last the
    vector in results/ite_NNNN/averages/bandpass_fsc_ite_NNNN.fsc (dynamo_database_convention.m:724-729, item
    'bandpass_fsc', ID 'I'), plotted against 1..n. Missing files are skipped ("FSC file not found")."""
    import os
    out = []
    for ite in range(1, int(last_ite) + 1):
        f = os.path.join(name, "results", "ite_%04d" % ite, "averages", "bandpass_fsc_ite_%04d.fsc" % ite)
        if not os.path.isfile(f):
            print("FSC file not found")
            continue
        y = read_table(f).ravel()
        out.append((ite, np.arange(1, y.size + 1, dtype=np.float64), y))
    return out


# MATLAB R2019a default axes ColorOrder (lines cycle through it with 'hold on') [MATLAB built-in]
MATLAB_COLORORDER = [(0.0, 0.447, 0.741), (0.85, 0.325, 0.098), (0.929, 0.694, 0.125), (0.494, 0.184, 0.556),
                     (0.466, 0.674, 0.188), (0.301, 0.745, 0.933), (0.635, 0.078, 0.184)]


def matlab_auto_limits(lo, hi):
    """Automatic axis limits and ticks for a line plot: the smallest step in {1, 2, 5} x 10^k that gives at most 10
    intervals, limits rounded out to multiples of it. An approximation of MATLAB's built-in rule, checked on the
    adaptive-filtering FSC plot of the user's project (XLim [0 35] / XTick 0:5:35, YLim [0.1 1] / YTick 0.1:0.1:1)."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    k = math.floor(math.log10(span)) - 2
    while True:
        for m in (1, 2, 5):
            step = m * 10.0 ** k
            a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
            if round((b - a) / step) <= 10:
                n = int(round((b - a) / step))
                return a, b, [round(a + i * step, 12) for i in range(n + 1)]
        k += 1
