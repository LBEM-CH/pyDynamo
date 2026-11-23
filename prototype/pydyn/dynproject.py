"""Run a Dynamo dcp alignment project exactly as Dynamo 1.1.514 does (destination standalone_gpu), without MATLAB.

Specifications (every behaviour below is taken from them; section numbers are cited in the code):
  06a = docs/analysis/06a_setup_1.1.514.md          (iteration setup)
  06b = docs/analysis/06b_assemble_1.1.514.md       (iteration assembly)
  06c = docs/analysis/06c_driver_formats_1.1.514.md (driver loop and file formats)
Numerics of the GPU stages are the verified pydyn ports: engine.Engine (dynamo_compute_iteration_spp, run with
dynamo_version 1.1.514) and average.ordered_sums / average.finalize (dynamo_average, pipeline path).

What is reproduced, in order, per iteration (06c A.2 / A.4):
  setup     surviving references (06a S2), angular freezing (S4) and shift freezing (S5) for every mode,
            particle selection col 2 != 0 and task lists (S6), card_iteref (S7, S9.3), reference preprocessing
            (S8: mask_transformed, template fmask, template sym/band/norm), GPU chunks ceil(N/G) in table order
            with .tags / particle lists / .btbl written as Dynamo does (S9.2), time_start_computation (S10)
  compute   one persistent worker process per physical GPU; each runs the pydyn engine on the chunks assigned to
            it, reading the chunk's particle list and .btbl and writing refined_gpuProcRefTable_*.btbl
  assemble  gather (06b 2), columns 27-30 incl. the self-comparison and the complex column 28 (06b 3),
            thresholding col 3 (06b 6), averaging from the re-read rounded text table with the raw settings/mask.em
            and fmin 3 (06b 7), symmetrized copies (06b 8), next_iteration.txt (06b 9), propagation (06b 11),
            temp deletion, surviving references of the next iteration, card end times
  check     missing files / NaN (06c A.4 Check)

Two references without MRA (the even/odd project of dveo, 09a 6) run as two independent channels: per-reference
setup, chunks, engine runs (one resident particle set per reference and GPU worker), assembly and propagation; ite-1
comparison against settings/table_initial_ref_00R.tbl. The golden-standard adaptive bandpass (09b 4) runs at the end
of assemble (pydyn.abp, with the CPU dynamo_align port pydyn.dalign); setup reloads the vpr so the new low reaches the
next template. The even/odd adaptive bandpass, MRA and nref > 2 raise NotImplementedError.

Everything the specifications mark unsupported, or that this port does not implement, raises NotImplementedError
(see DynamoProject.validate).

Paths: Dynamo addresses the project as ./<name_project>/... relative to its working directory (06c A.1). The cards
written here keep those Dynamo strings byte for byte; on disk, './<name_project>/' is the output root given to
DynamoProject, and '<vpr.folder_data>/' is the data folder given to it.
"""
import ctypes
import glob
import json
import math
import os
import shutil
import time
import traceback

import numpy as np

from . import fdlibm as _F

from . import dynio, dynlog

DYNAMO_VERSION = "1.1.514"
DYNAMO_SRC = "/programs/x86_64-linux/dynamo/1.1.514/matlab/src"

# Parameter order of dynamo_vpr_convention.m (1.1.514): the 'nc' (global) and 'rc' (round) cells in file order.
# dynamo_pipeline_setup_card_ite.m:31-63 builds card_ite in this order. tests/test_dynproject.py checks these
# lists against the source file when it is available.
GLOBAL_PARAMS = [
    'name_project', 'path', 'file_template_initial', 'file_table_initial', 'folder_data', 'file_mask',
    'file_mask_classification', 'file_fmask_initial', 'file_mask_smoothing', 'destination', 'how_many_processors',
    'cluster_header', 'cluster_walltime', 'submit_order', 'gpu_identifier_set', 'gpu_motor',
    'matlab_workers_average', 'intertwin', 'apix', 'initial_references', 'runfrom', 'options', 'MCR_CACHE_ROOT',
    'multi_MCR_CACHE_ROOT', 'systemUsingProcessorTables', 'systemCardForParticle', 'systemNoMPIWrapUnfoldingGPU',
    'symmetrize_with_fmask', 'project_type', 'adaptive_bandpass', 'adaptive_bandpass_pushback',
    'adaptive_bandpass_threshold', 'adaptive_bandpass_initial', 'adaptive_bandpass_symmetrized',
    'abp_recompute_second_average', 'adaptive_bandpass_even_odd', 'update_fmask', 'update_with_unweighted_average',
    'fourierMinimumFractionForAverage', 'fCompensationSmoothingRadius', 'fCompensationSmoothingDecay',
    'fCompensationSmoothingMask', 'useSmoothingMaskForFourierCompensation', 'averagingImplicitRotationMask']
ROUND_PARAMS = [
    'ite', 'nref', 'cone_range', 'cone_sampling', 'cone_flip', 'cone_check_peak', 'cone_freeze_reference',
    'inplane_range', 'inplane_sampling', 'inplane_flip', 'inplane_check_peak', 'inplane_freeze_reference', 'refine',
    'refine_factor', 'high', 'low', 'sym', 'dim', 'area_search', 'area_search_modus', 'separation_in_tomogram',
    'limit_xy_check_peak', 'limit_z_check_peak', 'use_CC', 'localnc', 'mra', 'threshold', 'threshold_modus',
    'threshold2', 'threshold2_modus', 'ccmatrix', 'ccmatrix_type', 'ccmatrix_batch', 'Xmatrix', 'Xmatrix_maxMb',
    'PCA', 'PCA_neigs', 'kmeans', 'kmeans_ncluster', 'kmeans_ncoefficients', 'nclass', 'plugin_align',
    'plugin_post', 'plugin_iter', 'plugin_align_order', 'plugin_post_order', 'plugin_iter_order', 'flags',
    'convergence_type', 'convergence', 'rings', 'rings_random', 'dynamic_mask']

# area_search_modus values whose engine behaviour (06a E.5) the pydyn engine reproduces
ENGINE_SHIFT_MODES = {0, 1, 11, 2, 21, 3, 31, 4, 41, 5, 51, 6, 61}


def _datestr(t=None):
    """MATLAB datestr(now): 'dd-mmm-yyyy HH:MM:SS' with an English month abbreviation (06a S10)."""
    import calendar
    lt = time.localtime(time.time() if t is None else t)
    return "%02d-%s-%04d %02d:%02d:%02d" % (lt.tm_mday, calendar.month_abbr[lt.tm_mon], lt.tm_year,
                                            lt.tm_hour, lt.tm_min, lt.tm_sec)


def check_report_path(path, project_dir):
    """Absolute path of the opt-in diagnostics report (--report PATH / $PYDYN_REPORT), or None. A path inside the
    project folder is refused: the project folder holds exactly the files Dynamo writes (docs/DESIGN.md)."""
    if not path:
        return None
    path = os.path.abspath(path)
    proj = os.path.abspath(project_dir)
    if path == proj or path.startswith(proj.rstrip(os.sep) + os.sep):
        raise ValueError("report path %s is inside the project folder %s; a Dynamo project folder holds only "
                         "Dynamo's files, give a path outside it" % (path, proj))
    return path


# =================================================================================================
# vpr and rounds
# =================================================================================================

def load_vpr(path):
    """settings/virtual_project.mat (MAT v5, struct 'card') -> dict in MATLAB field order."""
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m["card"]
    out = {}
    for f in s._fieldnames:
        v = getattr(s, f)
        if isinstance(v, np.ndarray) and v.dtype.kind == "U":
            v = "".join(v.ravel().tolist())          # empty char arrays -> ''
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        elif isinstance(v, np.generic):
            v = v.item()
        out[f] = v
    return out


def _str2num_vector(s):
    """str2num of one intertwin part: space/comma separated integers or a:b / a:s:b ranges."""
    vals = []
    for tok in s.replace(",", " ").split():
        if ":" in tok:
            p = [float(x) for x in tok.split(":")]
            a, st, b = (p[0], 1.0, p[1]) if len(p) == 2 else (p[0], p[1], p[2])
            vals += list(np.arange(a, b + st / 2, st))
        else:
            vals.append(float(tok))
    if not vals:
        raise NotImplementedError("intertwin part %r cannot be parsed (str2num/eval not emulated)" % s)
    return vals


def round_array(vpr):
    """dynamo_zaux_intertwin_round_identifier_array.m:54-102 (06a S1.2)."""
    s = str(vpr["intertwin"])
    if s.startswith("/"):
        s = s[1:]
    parts = [_str2num_vector(p) for p in s.split("/")]
    heads = [int(p[0]) for p in parts] + [9]                       # max_round = 8
    arr = []
    for i in range(len(heads) - 1):
        n_head = int(vpr["ite_r%d" % heads[i]])
        members = list(range(heads[i], heads[i + 1]))
        for _ in range(n_head):
            arr += members
    return arr


def ite2round(vpr, ite):
    """dynamo_intertwin_ite2round.m:41-65: round_array(ite), None outside 1..len."""
    arr = round_array(vpr)
    return arr[ite - 1] if 1 <= ite <= len(arr) else None


def round_leaders(vpr):
    """dynamo_vpr_status(vpr, 'round_leaders'), dynamo_vpr_status.m:336-366 (06a S1.3). 1-based iterations."""
    arr = round_array(vpr)
    leaders = [1]
    for i in range(2, len(arr) + 1):
        if arr[i - 1] != arr[i - 2] and arr[i - 2] not in arr[i - 1:]:
            leaders.append(i)
    return leaders


def _last_leader_before(vpr, ite):
    L = sorted(round_leaders(vpr))
    before = [x for x in L if x < ite]                             # strict '<' (06a S4 quirk)
    return before[-1]


def card_ite_from_vpr(vpr, ite):
    """dynamo_pipeline_setup_card_ite.m (06a S1.1)."""
    name = vpr["name_project"]
    rnd = ite2round(vpr, ite)
    folder = "./%s/results/ite_0001/starting_values" % name     # 'ite',1 hard-coded (:16-18)
    card = {"name_project": name, "database_ite": ite,
            "database_folder_starting_reference": folder,
            "database_folder_starting_table": folder,
            "database_folder_fmask_starting_reference": folder}
    for f, v in vpr.items():                                       # :22-27
        if "feature_" in f:
            card[f] = v
    for p in ROUND_PARAMS:                                         # :33-42
        k = "%s_r%d" % (p, rnd)
        if k in vpr:
            card[p] = vpr[k]
    for p in GLOBAL_PARAMS:                                        # :46-67
        if p in card:
            continue
        if p in vpr:
            card[p] = vpr[p]
        elif p != "path":
            raise NotImplementedError("global parameter %r is missing from the vpr; the convention default is not "
                                      "ported" % p)
    card.pop("ite", None)                                          # :70-72
    return card


# =================================================================================================
# MATLAB's scalar trigonometry (fdlibm, pydyn.fdlibm) for columns 27-30 (06b 3.3, 12.1)
# =================================================================================================

class FdlibmTrig:
    """sin / cos as MATLAB R2019a computes them (fdlibm; bit-identical, pydyn.fdlibm), so that column 28 matches
    Dynamo's text exactly. NumPy's / the C library's sin and cos differ from MATLAB's in about 3 % of arguments."""
    sin = staticmethod(_F.sin_array)
    cos = staticmethod(_F.cos_array)


def _acos_deg(d):
    """MATLAB acos(d)*180/pi of a real scalar (fdlibm acos); complex for d > 1 (06b 3.3)."""
    if -1 <= d <= 1:
        return complex(_F.acos(d) * 180 / math.pi, 0)
    if d > 1:
        return complex(0, math.acosh(d) * 180 / math.pi)
    if math.isnan(d):
        return complex(float("nan"), 0)
    raise NotImplementedError("acos of a dot product below -1: MATLAB's complex branch is uncertain (06b 3.3)")


def compare_columns(cur, prev, trig, symorder=1.0, sym_kind=None):
    """dynamo_table_compare(previous, current, 'sym', op) columns 27-30 for rows already matched by tag
    (06b 3.2-3.3, 12.1). Inputs are real [rows, >=10] arrays of the values as read from text. `symorder` is
    pydyn.symmetry.symorder(op) (dynamo_rigid_compare.m:73-97: 1 for c1, N for cN/dN, 360/dphi for helical): column
    29 = min(|r|, |r - 360/symorder|) with r = mod(dnarot, 360/symorder) (:111-114; 10a 5.1); symorder = inf (helical
    dphi = 0, 360/0 in MATLAB) gives 360/symorder = 0, mod(x, 0) = x and column 29 = |dnarot|. sym_kind 'd' (the kind
    returned by dynamo__parse_sym): the axis difference is folded into [-90, 90] (dynamo_rigid_compare.m:117-127, on
    the real part for the comparisons, as MATLAB compares complex numbers)."""
    a1 = (-cur[:, 6]) * math.pi / 180
    t1 = (-cur[:, 7]) * math.pi / 180
    a2 = (-prev[:, 6]) * math.pi / 180
    t2 = (-prev[:, 7]) * math.pi / 180
    s1, c1, st1, ct1 = trig.sin(a1), trig.cos(a1), trig.sin(t1), trig.cos(t1)
    s2, c2, st2, ct2 = trig.sin(a2), trig.cos(a2), trig.sin(t2), trig.cos(t2)
    z1 = (s1 * st1, -c1 * st1, ct1)
    z2 = (s2 * st2, -c2 * st2, ct2)
    d = (z1[0] * z2[0] + z1[2] * z2[2]) + z1[1] * z2[1]            # MATLAB dot order (p1+p3)+p2
    c28 = np.array([_acos_deg(float(x)) for x in d], dtype=complex)
    if sym_kind == "d":                                             # dynamo_rigid_compare.m:117-127
        hi, lo = c28.real > 90, c28.real < -90
        c28 = np.where(hi, c28 - 180, c28)
        c28 = np.where(lo, c28 + 180, c28)
    ds = np.sqrt(((-cur[:, 3]) - (-prev[:, 3])) ** 2 + ((-cur[:, 4]) - (-prev[:, 4])) ** 2
                 + ((-cur[:, 5]) - (-prev[:, 5])) ** 2)
    x = (-cur[:, 6]) - (-prev[:, 6])
    per = 360.0 / symorder if np.isfinite(symorder) else 0.0
    r = matlab_mod(x, per)                                          # mod(x, 360/symorder)
    c29 = np.minimum(np.abs(r), np.abs(r - per))
    c30 = cur[:, 9] - prev[:, 9]
    return ds, c28, c29, c30


def matlab_mod(x, y):
    """MATLAB mod(x, y) (R2019a MCR of Dynamo 1.1.514) for an array x and a scalar y != 0.
    Integer y: x - floor(x./y).*y (06b 3.2, c1 tables). Non-integer y (helical symorder, 10a 5.1): 0 where x./y is
    within round-off of an integer (|q - round(q)| <= eps*|q|), else x - floor(q)*y rounded once (exact product).
    [RUN: headless 1.1.514 mod(x, -1.4) for 204 probes x = k*y + j*eps(k*y), k in {+-3, 4, +-6, 7, 50, +-129, 257,
    1000, 2571}, j = -8..8, and the user's three ite-1 col-29 rows: benchmarks/06_abp_align_eo/implA/modprobe.m, modfit.py.]"""
    from fractions import Fraction
    x = np.asarray(x, dtype=np.float64)
    if y == 0:                                  # mod(x, 0) = x
        return x.copy()
    if y == int(y):
        return x - np.floor(x / y) * y
    q = x / y
    out = np.empty_like(x)
    fy = Fraction(y)
    eps = np.finfo(np.float64).eps
    for i, (xi, qi) in enumerate(zip(x.ravel().tolist(), q.ravel().tolist())):
        if not (math.isfinite(xi) and math.isfinite(qi)):
            out.flat[i] = float("nan")
        elif abs(qi - round(qi)) <= eps * abs(qi):
            out.flat[i] = math.copysign(0.0, y)
        else:
            out.flat[i] = float(Fraction(xi) - math.floor(qi) * fy)
    return out


# =================================================================================================
# volume processing used by setup (06a S8)
# =================================================================================================

def _centered_coords(n):
    r = np.arange(n, dtype=np.float64) - (n // 2)
    return np.meshgrid(r, r, r, indexing="ij")                      # [z, y, x]


def isotropic_shell(n, r1, r2, smoothing):
    """dynamo__shell_mex -> volume_noncubic::fillIsotropicShell (mex/src/classes/volume_noncubic.cpp:694-790),
    double arithmetic, stored as float32 (06a S8.0 band)."""
    rz, ry, rx = _centered_coords(n)
    c1 = (rx / r1) * (rx / r1) + (ry / r1) * (ry / r1) + (rz / r1) * (rz / r1)
    c2 = (rx / r2) * (rx / r2) + (ry / r2) * (ry / r2) + (rz / r2) * (rz / r2)
    rad = np.sqrt(rx * rx + ry * ry + rz * rz)
    sm = np.zeros_like(rad)
    m2 = c2 > 1
    libm_exp = np.frompyfunc(math.exp, 1, 1)                        # the mex's C exp (NumPy's SIMD exp is host-dependent)
    sm[m2] = libm_exp(-((r2 - rad[m2]) ** 2) / (smoothing * smoothing)).astype(np.float64)
    m1 = c1 < 1
    sm[m1] = libm_exp(-((r1 - rad[m1]) ** 2) / (smoothing * smoothing)).astype(np.float64)
    v = np.where((c1 >= 1) & (c2 <= 1), 1.0, np.where(sm > 0.13, sm, 0.0))
    return v.astype(np.float32).astype(np.float64)


def ellipsoid_inside(n, radius):
    """dynamo_sphere / dynamo_ellipsoid (dpkgeom.sphere -> dynamo__sphere_mex, volume_noncubic.cpp:608-653):
    voxel inside when sum(((x - n/2)/r)^2) <= 1 (0-based), cubic even box."""
    if n % 2:
        raise NotImplementedError("odd box sizes: the sphere centre conventions differ (06a D.7-8)")
    rz, ry, rx = _centered_coords(n)
    return ((rx / radius) ** 2 + (ry / radius) ** 2 + (rz / radius) ** 2) <= 1


def normalize_roi(v, mask, threads=1):
    """dynamo_normalize_roi.m:130-184: roi = v[mask != 0] without NaN; (v - mean) / std(roi - mean), with
    MATLAB's sums for `threads` computational threads (pydyn.matlab_numerics)."""
    from .matlab_numerics import normalize_roi as _n
    return _n(v, mask, threads)


# =================================================================================================
# table helpers
# =================================================================================================

def _tags(T):
    return np.real(T[:, 0]).astype(np.int64)


def _unique_index(tags):
    """tag -> row index, only for tags present exactly once (dynamo_tag2ptable returns [] otherwise)."""
    u, cnt = np.unique(tags, return_counts=True)
    once = set(u[cnt == 1].tolist())
    return {int(t): i for i, t in enumerate(tags.tolist()) if t in once}


def table_merge(tables):
    """dynamo_table_merge.m (06b 2.3): pad columns, concatenate, unique ascending tags, first occurrence wins."""
    tables = [t for t in tables if t.size]
    if not tables:
        return np.zeros((0, 0))
    w = max(t.shape[1] for t in tables)
    cplx = any(np.iscomplexobj(t) for t in tables)
    pad = [np.concatenate([t, np.zeros((t.shape[0], w - t.shape[1]), dtype=t.dtype)], axis=1) for t in tables]
    merged = np.concatenate(pad, axis=0).astype(complex if cplx else float)
    tags = _tags(merged)
    _, first = np.unique(tags, return_index=True)                   # ascending tags, first occurrence
    return merged[first]


# =================================================================================================
# GPU worker (one process per physical GPU; particles and their spectra stay resident across iterations)
# =================================================================================================

def _resolve(path, out_root, name_project, folder_data, data_dir):
    p = path
    for pre in ("./" + name_project + "/", name_project + "/"):
        if p.startswith(pre):
            return os.path.join(out_root, p[len(pre):])
    fd = folder_data.rstrip("/") + "/"
    for pre in ("./" + fd, fd):
        if p.startswith(pre):
            return os.path.join(data_dir, p[len(pre):])
    return p


class _Progress:
    """The engine's per-particle files (dynamo_compute_iteration_spp.cu:2160-2174, 06c/10f 1.1), per GPU chunk:
    completedTasks_gpuProcRef_*.btbl = the number of particles done (EM, one float, rewritten after every particle) and,
    at every multiple of 1000 particles of the chunk, refined_gpuProcRefTable_*.btbl holding the rows done so far (the
    rest zero: the engine's table is formatted but not filled). The final refined table is written by the caller."""

    def __init__(self, chunks, sizes, ncols):
        self.chunks, self.sizes, self.ncols = chunks, sizes, ncols
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.done = 0

    def __call__(self, hi, out):
        lo, self.done = self.done, hi
        for ch, n, o in zip(self.chunks, self.sizes, self.starts):
            a, b = max(lo, o) - o, min(hi, o + n) - o
            if b <= a:
                continue
            for k in range(a // 1000 * 1000 + 1000, b + 1, 1000):     # multiples of 1000 in (a, b]
                part = np.zeros((n, self.ncols), dtype=np.float32)
                part[:k] = out[o:o + k]
                dynio.write_btbl(ch["refined_btbl"], part)
            dynio.write_btbl(ch["refined_btbl"].replace("refined_gpuProcRefTable", "completedTasks_gpuProcRef"),
                             np.array([[float(b)]], dtype=np.float32))


def _gpu_worker(device, conn, cfg):
    try:
        import cupy as cp
        from . import emio
        from .card import Card
        from .engine import Engine
        from .average import ordered_sums, pool_sums
        cp.cuda.Device(device).use()
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
        return
    res = lambda p: _resolve(p, cfg["out_root"], cfg["name_project"], cfg["folder_data"], cfg["data_dir"])
    slots = {}                                   # one resident particle set + engine per reference (msg "slot")
    prop = cp.cuda.runtime.getDeviceProperties(0)       # the engine prints device 0 of the visible set (spp.cu:4726)
    nm = prop["name"]
    conn.send({"ok": True, "device": device, "gpu_name": nm.decode() if isinstance(nm, bytes) else str(nm),
               "multiprocessors": int(prop["multiProcessorCount"])})
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            t0 = time.perf_counter()
            cpu0 = time.process_time()
            st = slots.setdefault(msg.get("slot", 1),
                                  {"paths": None, "particles": None, "tags": None, "eng": None, "eng_key": None})
            if msg["op"] == "align":
                card = Card(msg["card"])                             # parsed as the CUDA engine parses it
                card.dynamo_version = cfg["dynamo_version"]
                paths, rows, sizes = [], [], []
                for ch in msg["chunks"]:
                    with open(ch["particle_list"]) as f:
                        pl = [ln.strip() for ln in f if len(ln.strip()) > 4]
                    r = dynio.read_btbl(ch["starting_btbl"])
                    if len(r) < len(pl):
                        raise RuntimeError("starting table has fewer rows than the particle list")
                    paths += [res(p) for p in pl]
                    rows.append(r[:len(pl)])                        # engine pairs list line k with btbl row k
                    sizes.append(len(pl))
                rows = np.concatenate(rows).astype(np.float32)
                t_read = 0.0
                if st["paths"] != paths:
                    t1 = time.perf_counter()
                    st["particles"] = emio.read_em_stack(paths)
                    if st["eng"] is not None:
                        st["eng"].close()                           # cuFFT 8 plans are not in CuPy's pool
                    st["paths"], st["eng"], st["eng_key"] = paths, None, None
                    st["tags"] = rows[:, 0].astype(np.int64)
                    t_read = time.perf_counter() - t1
                template = emio.read_em(res(card.file_template)).astype(np.float32)
                mask = emio.read_em(res(card.file_mask)).astype(np.float32)
                key = (card.high, card.low, card.shrink, card.use_CC, rows[:, 12:19].tobytes())
                t1 = time.perf_counter()
                if st["eng"] is None or st["eng_key"] != key:
                    # memory only: release every slot's engine (rebuilt on demand, ~0.4 s) before building one, so the
                    # new engine's residency plan and batch calibration see the whole GPU. Keeping the engines of
                    # both references plus the new one left a few GB free in ite 2 of the e/o ABP run (the low pass
                    # changes every iteration) and the ref-2 alignment ran > 8x slower (benchmarks/05_abp_bin4_eo/REPORT.md).
                    for other in slots.values():
                        if other["eng"] is not None:
                            other["eng"].close()                    # cuFFT 8 plans are not in CuPy's pool (review H1)
                        other["eng"], other["eng_key"] = None, None
                    eng = None
                    cp.cuda.Device().synchronize()
                    cp.fft.config.get_plan_cache().clear()
                    cp.get_default_memory_pool().free_all_blocks()
                    eng = Engine(template, mask, card, batch="auto", exact=cfg.get("engine", "exact") == "exact",
                                 exact_sms=cfg.get("exact_sms"))
                    eng.prepare_particles(st["particles"], rows)
                    st["eng"], st["eng_key"] = eng, key
                else:
                    eng = st["eng"]
                    eng.card = card
                    eng.set_template(template, mask)
                t_prep = time.perf_counter() - t1
                eng.timings = {}
                t1 = time.perf_counter()
                refined = eng.refine(rows, progress=_Progress(msg["chunks"], sizes, rows.shape[1]))
                cp.cuda.Device().synchronize()
                t_ref = time.perf_counter() - t1
                o = 0
                for ch, n in zip(msg["chunks"], sizes):
                    dynio.write_btbl(ch["refined_btbl"], refined[o:o + n])
                    o += n
                conn.send({"ok": True, "device": device, "particles": len(paths), "read_s": round(t_read, 3),
                           "prepare_s": round(t_prep, 3), "refine_s": round(t_ref, 3),
                           "pairs": int(sum(eng.orientation_counts)), "residency": eng.residency,
                           "cpu_s": round(time.process_time() - cpu0, 3),
                           "wall_s": round(time.perf_counter() - t0, 3)})
            elif msg["op"] == "load":                               # particles resident without aligning (tests)
                paths, rows = [], []
                for ch in msg["chunks"]:
                    with open(ch["particle_list"]) as f:
                        pl = [ln.strip() for ln in f if len(ln.strip()) > 4]
                    paths += [res(p) for p in pl]
                    rows.append(dynio.read_btbl(ch["starting_btbl"])[:len(pl)])
                st["particles"] = emio.read_em_stack(paths)
                if st["eng"] is not None:
                    st["eng"].close()
                st["paths"], st["eng"], st["eng_key"] = paths, None, None
                st["tags"] = np.concatenate(rows)[:, 0].astype(np.int64)
                conn.send({"ok": True, "device": device, "particles": len(paths)})
            elif msg["op"] == "tags":                              # resident tags of a slot (averaging order)
                conn.send({"ok": True, "device": device,
                           "tags": [] if st["tags"] is None else st["tags"].astype(np.int64)})
            elif msg["op"] == "average":
                # add this worker's run of the ascending-tag order to the running sums (average.ordered_sums); the
                # parent chains the state through the workers, so each half is one sequential sum as in Dynamo
                rows = msg["rows"]
                pos = {int(t): i for i, t in enumerate(st["tags"].tolist())}
                pidx = np.array([pos[int(t)] for t in rows[:, 0]], dtype=np.int64)
                contiguous = len(pidx) > 0 and np.array_equal(pidx, np.arange(pidx[0], pidx[0] + len(pidx)))
                parts = st["particles"][pidx[0]:pidx[0] + len(pidx)] if contiguous else st["particles"][pidx]
                if msg.get("pieces") is None:
                    state = ordered_sums(parts, rows.astype(np.float32), msg["half"], msg["nh"],
                                         msg["mask"], msg["state"], threads=int(msg.get("threads", 1)))
                else:                                          # MATLAB pool: parfor subranges (average.pool_sums)
                    state = pool_sums(parts, rows.astype(np.float32), msg["half"], msg["nh"], msg["mask"],
                                      msg["state"], msg["pieces"], threads=int(msg.get("threads", 1)))
                cp.cuda.Device().synchronize()
                cp.fft.config.get_plan_cache().clear()
                cp.get_default_memory_pool().free_all_blocks()
                conn.send({"ok": True, "device": device, "state": state, "wall_s": round(time.perf_counter() - t0, 3)})
            else:
                raise ValueError("unknown op %r" % msg["op"])
        except Exception:
            conn.send({"ok": False, "device": device, "error": traceback.format_exc()})
    conn.close()


# =================================================================================================
# the project
# =================================================================================================

class FatalIterationError(RuntimeError):
    pass


def worker_devices(gpus, gpu_ids_card, how_many_processors=1):
    """The physical devices that get a GPU worker: `gpus` (default gpu_identifier_set) cut to the number of GPU
    chunks of a reference, how_many_processors * len(gpu_identifier_set) (setup_gpu, 06a S9.2). compute() deals the
    chunks over the workers in turn, so a worker beyond that number would get no chunk (as --gpus all, execute.py)."""
    devs = list(gpus) if gpus else list(gpu_ids_card)
    return devs[:len(gpu_ids_card) * max(1, int(float(how_many_processors)))]


class DynamoProject:
    """A Dynamo 1.1.514 dcp project, re-executed under `out` (the project root).

    source: an unfolded Dynamo project folder (settings/virtual_project.mat and the settings/ files are read);
    data:   the data folder (particle_<tag>.em); its name in the cards stays vpr.folder_data;
    gpus:   physical device ids to run on (default: the vpr's gpu_identifier_set). The GPU chunks and their file
            names always follow the vpr's gpu_identifier_set (06a S9.2); chunk i runs on gpus[i % len(gpus)].
            Devices beyond the number of chunks of a reference are not used (worker_devices).
    """

    def __init__(self, source, data, out, gpus=None, keep_temp=False, table_initial=None,
                 echo=True, engine=None, exact_sms=None, report=None, progress_log=None, matlab_threads=None):
        from .engine import preflight
        preflight(engine)                                            # before any project, setup or engine work (M1)
        self.source = os.path.abspath(source)
        self.data = os.path.abspath(data)
        self.out = os.path.abspath(out)
        self.keep_temp = keep_temp
        self.echo = echo
        self.vpr = load_vpr(os.path.join(self.source, "settings", "virtual_project.mat"))
        self.name = str(self.vpr["name_project"])
        self.rounds = round_array(self.vpr)
        self.N = len(self.rounds)
        self.gpu_ids_card = [int(g) for g in np.atleast_1d(self.vpr["gpu_identifier_set"]).tolist()]
        self.gpus = list(gpus) if gpus else list(self.gpu_ids_card)
        from .engine import engine_mode
        self.engine = engine_mode(engine)                            # GPU alignment engine: exact | fast
        self.exact_sms = exact_sms                                   # exact engine: emulate this SM count's cuBLAS 8
        # MATLAB thread count of dynamo_execute_project, for the steps that run inside it (the adaptive bandpass's
        # dynamo_align): default MATLAB's own, one per available physical core (pydyn.matlab_numerics.matlab_threads)
        from .matlab_numerics import matlab_threads as _mt
        self.matlab_threads = int(matlab_threads) if matlab_threads else _mt()
        self.trig, self.trig_name = FdlibmTrig(), "fdlibm"           # MATLAB's sin/cos, bit for bit (pydyn.fdlibm)
        # vpr.file_table_initial is relative to Dynamo's working directory, the folder that contains the project
        fti = str(self.vpr["file_table_initial"])
        self.cwd = os.path.dirname(self.source)                      # Dynamo's working directory for vpr paths
        self.table_initial = table_initial or (fti if os.path.isabs(fti)
                                               else os.path.join(os.path.dirname(self.source), fti))
        self.workers = []
        out_dir = self.out
        self.report_path = check_report_path(report or os.environ.get("PYDYN_REPORT"), out_dir)
        self.progress_log = check_report_path(progress_log, out_dir)
        used = worker_devices(self.gpus, self.gpu_ids_card, self.vpr.get("how_many_processors", 1))
        if len(used) < len(self.gpus):
            self.log("GPUs %s requested; the project splits the particles of a reference in %d part%s "
                     "(gpu_identifier_set): using %s" % (",".join(str(g) for g in self.gpus), len(used),
                                                         "" if len(used) == 1 else "s",
                                                         ",".join(str(g) for g in used)))
        self.gpus = used
        self.gpu_name, self.multiprocessors = "", 0                 # device 0 properties, sent by the workers
        from . import symmetry
        symmetry.set_device(self.gpus[0])                            # main-process symmetry GPU work (M2)
        self.report = {"dynamo_version": DYNAMO_VERSION, "source": self.source, "data": self.data, "out": self.out,
                       "name_project": self.name, "round_array": self.rounds, "gpus": self.gpus,
                       "gpu_identifier_set": self.gpu_ids_card, "trig": self.trig_name, "engine": self.engine,
                       "iterations": []}
        self.validate()
        self.facts = self._data_info()

    # ------------------------------------------------------------------ paths
    def p(self, *rel):
        return os.path.join(self.out, *rel)

    def dstr(self, rel):
        """Dynamo's database string for a project file (06c A.1)."""
        return "./%s/%s" % (self.name, rel)

    @staticmethod
    def _sv(ite):
        return "results/ite_%04d/starting_values" % ite

    @staticmethod
    def _av(ite):
        return "results/ite_%04d/averages" % ite

    def f_starting_table(self, ref, ite):
        return self.p(self._sv(ite), "starting_table_ref_%03d_ite_%04d.tbl" % (ref, ite))

    def f_starting_reference(self, ref, ite):
        return self.p(self._sv(ite), "starting_reference_ref_%03d_ite_%04d.em" % (ref, ite))

    def f_fmask_starting_reference(self, ref, ite):
        return self.p(self._sv(ite), "fmask_starting_reference_ref_%03d_ite_%04d.em" % (ref, ite))

    def r_mask_transformed(self, ite):
        return self._sv(ite) + "/mask_transformed_ite_%04d.em" % ite

    def r_fmask_transformed(self, ref, ite):
        return self._sv(ite) + "/fmask_starting_reference_transformed_ref_%03d_ite_%04d.em" % (ref, ite)

    def r_template_transformed(self, ref, ite):
        return self._sv(ite) + "/starting_reference_transformed_ref_%03d_ite_%04d.em" % (ref, ite)

    def f_av(self, kind, ref, ite, ext="em"):
        return self.p(self._av(ite), "%s_ref_%03d_ite_%04d.%s" % (kind, ref, ite, ext))

    def f_surv(self, ite):
        return self.p("results/ite_%04d" % ite, "currently_surviving_references_ite_%04d.txt" % ite)

    def f_card_ite(self, ite):
        return self.p("cards/ite_%04d" % ite, "card_ite_ite_%04d.card" % ite)

    def f_card_iteref(self, ref, ite):
        return self.p("cards/ite_%04d" % ite, "card_iteref_ref_%03d_ite_%04d.card" % (ref, ite))

    def r_task(self, ite):
        return "temp/ite_%04d/task_assignations" % ite

    def r_inter(self, ite):
        return "temp/ite_%04d/intermediate_results" % ite

    def rv(self, param, rnd):
        return self.vpr["%s_r%d" % (param, rnd)]

    # ------------------------------------------------------------------ logging
    def log(self, msg):
        """pydyn's own progress line: stdout only (and --progress-log PATH, outside the project). The project's
        log.txt holds Dynamo's lines only (dynlog; docs/analysis/10f section 1.5)."""
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        if self.echo:
            print(line, flush=True)
        if self.progress_log:
            with open(self.progress_log, "a") as f:
                f.write(line + "\n")

    def append_log(self, lines):
        """dynamo_aux_appendlog: lines to <project>/log.txt, fprintf('%s\\n') each."""
        os.makedirs(self.out, exist_ok=True)
        dynlog.append_log(self.out, lines)

    @staticmethod
    def _datestr(t=None):
        return dynlog.datestr(t)

    # ------------------------------------------------------------------ validation
    def validate(self):
        """Raise NotImplementedError for every option this port does not reproduce."""
        v = self.vpr
        ni = NotImplementedError

        def need(cond, what):
            if not cond:
                raise ni(what)
        need(self.N > 0, "project has no iterations")
        need(v.get("destination") == "standalone_gpu",
             "destination %r: only standalone_gpu is implemented" % v.get("destination"))
        need(int(v["how_many_processors"]) == 1, "how_many_processors != 1 (several proc chunks per GPU)")
        need(int(v.get("systemUsingProcessorTables", 0)) == 0, "systemUsingProcessorTables = 1 (Dynamo errors on GPU)")
        need(int(v.get("systemCardForParticle", 0)) == 0, "systemCardForParticle = 1 (per-particle cards)")
        need(str(v.get("initial_references", "*")) == "*", "initial_references other than '*'")
        need(str(v.get("project_type", "general")) == "general", "project_type other than 'general'")
        if int(v.get("adaptive_bandpass", 0)) != 0:                  # golden standard (09b 4): pydyn.abp
            from . import abp
            need(int(v.get("adaptive_bandpass", 0)) == 1, "adaptive_bandpass %s (only 0/1)" % v.get("adaptive_bandpass"))
            abp.check_supported(v, self.rounds, self)
            if self.source == self.out:
                abp.vpr_write_target(self)                           # refuses before any computation (09b 4.9)
        need(int(v.get("adaptive_bandpass_even_odd", 0)) == 0, "adaptive bandpass even/odd (09b 6: not verified)")
        need(float(v.get("fourierMinimumFractionForAverage", -1)) == -1, "fourierMinimumFractionForAverage != -1")
        need(float(v.get("fCompensationSmoothingRadius", 0)) == 0, "Fourier-compensation smoothing")
        need(int(v.get("useSmoothingMaskForFourierCompensation", 0)) == 0, "smoothing mask for Fourier compensation")
        need(int(v.get("averagingImplicitRotationMask", 0)) == 0, "averagingImplicitRotationMask")
        need(str(v.get("options", "")) == "", "vpr.options for averaging")
        for f in ("fmin", "pivoting", "force_mask"):
            need(f not in v, "vpr.%s (averaging option) is not ported; only its default is" % f)
        need(str(v.get("feature_data_type", "dynamo_folder")) == "dynamo_folder", "data type other than dynamo_folder")
        need(str(v.get("feature_extension_data", "em")) == "em", "particle extension other than em")
        L = int(v["feature_data_sidelength"])
        need(int(v.get("feature_template_sidelength", L)) == L, "template sidelength != data sidelength")
        nref1 = int(v["nref_r1"])
        need(nref1 in (1, 2), "nref_r1 = %d (only 1, or 2 independent references without MRA, 09a 6)" % nref1)
        for rnd in sorted(set(self.rounds)):
            r = lambda p: self.rv(p, rnd)
            need(int(r("nref")) == nref1, "round %d: nref %s != nref_r1 %d" % (rnd, r("nref"), nref1))
            need(int(r("mra")) == 0, "round %d: MRA" % rnd)
            need(int(r("dim")) == L, "round %d: binning (dim %s != sidelength %d)" % (rnd, r("dim"), L))
            need(int(r("localnc")) == 1, "round %d: localnc 0" % rnd)
            need(int(r("use_CC")) in (0, 1), "round %d: use_CC = %s (template fmask path)" % (rnd, r("use_CC")))
            from .abp import check_sym                              # none (c1), cN, dN, helical (10a)
            if check_sym(r("sym"), L) != "none":
                need("symmetrize_with_fmask" in v, "round %d: symmetry %s without vpr.symmetrize_with_fmask"
                     % (rnd, r("sym")))
            need(float(r("low")) >= 1, "round %d: Nyquist-fraction low-pass" % rnd)
            need(float(r("high")) > 0, "round %d: high = 0 (sphere branch of dynamo_shell)" % rnd)
            need(int(r("threshold_modus")) == 0, "round %d: threshold_modus %s (only 0)" % (rnd, r("threshold_modus")))
            need(int(r("threshold2_modus")) == 0, "round %d: second thresholding" % rnd)
            need(float(r("separation_in_tomogram")) == 0, "round %d: separation_in_tomogram" % rnd)
            for c in ("cone_check_peak", "inplane_check_peak", "limit_xy_check_peak", "limit_z_check_peak"):
                need(float(r(c)) == 0, "round %d: %s (peak check)" % (rnd, c))
            need(int(r("plugin_post")) == 0, "round %d: plugin_post" % rnd)
            need(int(r("convergence_type")) == 0, "round %d: convergence_type %s (only 0)" % (rnd, r("convergence_type")))
            m = float(r("area_search_modus"))
            need(m == int(m) and int(m) in ENGINE_SHIFT_MODES, "round %d: area_search_modus %s" % (rnd, m))
            for c in ("cone_freeze_reference", "inplane_freeze_reference"):
                need(float(r(c)) in (0, 1, 2), "round %d: %s = %s is an error in Dynamo" % (rnd, c, r(c)))

    def _data_info(self):
        """dynamo_data_info (part used here): particle file padding and box size."""
        files = sorted(glob.glob(os.path.join(self.data, "particle_*.em")))
        if not files:
            raise FileNotFoundError("no particle_*.em in %s" % self.data)
        digits = os.path.basename(files[0])[len("particle_"):-3]
        hdr = open(files[0], "rb").read(16)
        nx, ny, nz = np.frombuffer(hdr[4:16], "<u4")
        L = int(self.vpr["feature_data_sidelength"])
        if not (nx == ny == nz == L):
            raise RuntimeError("particle size %s differs from feature_data_sidelength %d" % ((nx, ny, nz), L))
        return {"padding": len(digits), "l": L, "folder_name": str(self.vpr["folder_data"]).rstrip("/"),
                "extension": "em"}

    def particle_line(self, tag):
        """@dDataOld/getParticleFile.m: <folder_name>/particle_<tag padded>.<ext> (06a S9.2)."""
        return "%s/particle_%0*d.%s" % (self.facts["folder_name"], self.facts["padding"], int(tag),
                                       self.facts["extension"])

    # ------------------------------------------------------------------ unfold-equivalent initialisation
    def initialize(self):
        """The state Dynamo's unfold leaves (06c A.5): settings/ copied from the source project; ite 1 starting
        values = byte copies of settings/{table_initial,template_initial,fmask_initial}_ref_001 (06a S5.4, S8.2
        [RUN]); card_ite for every iteration from the vpr (06a S1.1); rescaled_data/data_<dim> folders
        (dynamo_vpr_unfold.m:503-520); next_iteration.txt = 1.

        In place (source == out, Stage 3 '<name>.exe' after dvunfold): unfold already wrote this state and
        dynamo_execute_project writes nothing before the loop (08a 7), so nothing is copied or logged here. The loop
        reads next_iteration.txt at every turn; a missing file is a dynamo_read error (tolerance 0)."""
        if self.source == self.out:
            if not os.path.exists(self.p("next_iteration.txt")):
                raise FileNotFoundError('[dread: error] cannot find requested file "%s"' % self.p("next_iteration.txt"))
            return
        if os.path.exists(self.p("next_iteration.txt")):
            self.log("project root already initialised: resuming from next_iteration.txt")
            return
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.p("settings"), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(self.source, "settings"))):
            src = os.path.join(self.source, "settings", f)
            if os.path.isfile(src):
                shutil.copyfile(src, self.p("settings", f))
        nref = int(self.rv("nref", 1))
        for ref in range(1, nref + 1):
            os.makedirs(self.p(self._sv(1)), exist_ok=True)
            shutil.copyfile(self.p("settings", "table_initial_ref_%03d.tbl" % ref), self.f_starting_table(ref, 1))
            shutil.copyfile(self.p("settings", "template_initial_ref_%03d.em" % ref), self.f_starting_reference(ref, 1))
            shutil.copyfile(self.p("settings", "fmask_initial_ref_%03d.em" % ref),
                            self.f_fmask_starting_reference(ref, 1))
        self._unfold_abp_initial()
        for ite in range(1, self.N + 1):
            dynio.write_card(self.f_card_ite(ite), card_ite_from_vpr(self.vpr, ite))
        for rnd in range(1, 9):
            d = self.vpr.get("dim_r%d" % rnd, 0)
            if d:
                os.makedirs(self.p("rescaled_data", "data_%d" % int(d)), exist_ok=True)
        os.makedirs(self.p("temp"), exist_ok=True)
        dynio.write_txt(self.p("next_iteration.txt"), 1)
        self.log("initialised %s from %s (%d iterations, rounds %s)" % (self.out, self.source, self.N, self.rounds))

    def _unfold_abp_initial(self):
        """dynamo_vpr_unfold step 4, +dpkproject/unfoldCheckAdaptiveBandpassInitial.m:8-17 (ported in pydyn.vpr
        unfold): with adaptive_bandpass on and adaptive_bandpass_initial != low_r1 (e.g. a source project whose vpr
        holds the low written by a previous ABP run, 10c G4), low_r1 := adaptive_bandpass_initial before the cards are
        written; the run's vpr is saved with it (as unfold saves the vpr)."""
        v = self.vpr
        if not (float(v.get("adaptive_bandpass", 0)) and "adaptive_bandpass_initial" in v
                and float(v["adaptive_bandpass_initial"]) != float(v["low_r1"])):
            return
        from . import vpr as V
        print(" [adaptive filtering] Overruling current value for low pass in first iteration: %s"
              % dynio._matlab_d(v["low_r1"]), flush=True)
        f = self.p("settings", "virtual_project.mat")
        cur = V.add_missing_convention_parameters(V.read_vpr_file(f))
        new = V.modify(cur, "low_r1", float(v["adaptive_bandpass_initial"]))
        V.write_vpr_file(f, V.correct_defaults(new, False))
        v["low_r1"] = float(v["adaptive_bandpass_initial"])
        print(" [adaptive filtering] Using initial lowpass: %s" % dynio._matlab_d(v["low_r1"]), flush=True)
        self.log("unfold rule: low_r1 := adaptive_bandpass_initial = %s" % dynio._matlab_d(v["low_r1"]))

    # ------------------------------------------------------------------ surviving references (06a S2)
    def get_surviving_refs(self, ite):
        if ite == 1:
            return list(range(1, int(self.rv("nref", 1)) + 1))
        f = self.f_surv(ite)
        if not os.path.exists(f):
            raise FatalIterationError("missing %s" % f)
        r = dynio.read_txt(f)
        if r.size == 0:
            raise FatalIterationError("empty %s" % f)
        return [int(x) for x in np.real(r).ravel().tolist()]

    def set_surviving_refs(self, ite, refs):
        dynio.write_txt(self.f_surv(ite), np.array(refs, dtype=float).reshape(1, -1))

    # ------------------------------------------------------------------ freezing (06a S4, S5)
    def _change_starting_tables(self, ite, refs, seed, cols):
        for ref in refs:
            cur_f, seed_f = self.f_starting_table(ref, ite), self.f_starting_table(ref, seed)
            cur, sd = dynio.read_tbl(cur_f), dynio.read_tbl(seed_f)
            if np.iscomplexobj(sd) and not np.iscomplexobj(cur):
                cur = cur.astype(complex)
            idx = _unique_index(_tags(sd))
            for i, t in enumerate(_tags(cur).tolist()):
                j = idx.get(int(t))
                if j is None:
                    continue                                        # missing or repeated in the seed: unchanged
                cur[i, cols] = sd[j, cols]
            dynio.write_tbl(cur_f, cur)

    def freeze_angles(self, ite, refs):
        """dpkproject.setup_freezing_angular_references (06a S4)."""
        if ite == 1:
            return
        if "cone_freeze_reference_r1" not in self.vpr:
            self.append_log(dynlog.setup_freeze_lines(ite, False, None, None))     # "Project does not have ..."
            return
        rnd = ite2round(self.vpr, ite)
        fc, fi = float(self.rv("cone_freeze_reference", rnd)), float(self.rv("inplane_freeze_reference", rnd))
        if fc == 0 and fi == 0:
            return                                                   # setup_freezing_angular_references returns silently
        begin = _last_leader_before(self.vpr, ite)

        def seed(f):
            if f == 0:
                return ite
            if f == 1:
                return begin
            if f == 2:
                return 1
            raise RuntimeError("Invalid input check project.")
        sc, si = seed(fc), seed(fi)
        self.append_log(dynlog.setup_freeze_lines(ite, True, sc, si))      # 10f 3.1 (setup_freezing_..._references.m)
        if sc == ite and si == ite:
            return
        if sc == si != ite:
            self._change_starting_tables(ite, refs, sc, [6, 7, 8])
        else:
            if sc != ite:
                self._change_starting_tables(ite, refs, sc, [6, 7])
            if si != ite:
                self._change_starting_tables(ite, refs, si, [8])

    def freeze_shifts(self, ite, refs):
        """dpkproject.setupFreezingShifts (06a S5)."""
        if ite == 1:
            return
        if not refs:
            raise FatalIterationError("no references for shift freezing")
        rnd = ite2round(self.vpr, ite)
        mode = float(self.rv("area_search_modus", rnd))
        if 2 < mode < 5:
            if mode == 3:
                seed = _last_leader_before(self.vpr, ite)
            elif mode == 4:
                seed = 1
            else:
                raise RuntimeError("Invalid option for area_search_modus %s (Dynamo errors)" % mode)
        else:
            seed = None
        self.append_log(dynlog.setup_shifts_lines(ite, mode, seed))        # every mode logs a line (10f 3, :54-104)
        if seed is None or (mode == 3 and ite == seed):
            return
        self._change_starting_tables(ite, refs, seed, [3, 4, 5])

    # ------------------------------------------------------------------ setup
    def setup(self, ite):
        # dynamo_iteration_setup loads the vpr from disk (06a S1, 09b 5): a low pass written by the adaptive bandpass
        # of the previous iteration reaches the template preprocessing this way
        t_start = time.time()                                         # datestr(now), dynamo_iteration_setup.m:109
        self.vpr = load_vpr(self.p("settings", "virtual_project.mat"))
        rnd = ite2round(self.vpr, ite)
        refs = self.get_surviving_refs(ite)                          # S2
        self.set_surviving_refs(ite, refs)
        self.append_log(dynlog.setup_banner(                          # S3 (dynamo_iteration_setup.m:109-110, 184-189)
            ite, self.dstr("cards/ite_%04d/card_ite_ite_%04d.card" % (ite, ite)), dynlog.datestr(t_start)))
        self.freeze_angles(ite, refs)                                 # S4
        self.freeze_shifts(ite, refs)                                 # S5
        nref = int(self.rv("nref", rnd))
        # S6: task lists
        tags_in_ref, master = {}, []
        for ref in range(1, nref + 1):
            if ref not in refs:
                self.append_log([dynlog.setup_skip_reference_line(ref)])   # dynamo_iteration_setup.m:269
                continue
            T = dynio.read_tbl(self.f_starting_table(ref, ite))
            sel = np.real(T[:, 1]) != 0                              # column 2; NaN counts as nonzero
            tags_in_ref[ref] = _tags(T)[sel]
            master += [[ref, int(t)] for t in tags_in_ref[ref].tolist()]
        task = self.r_task(ite)
        dynio.write_rt(self.p(task, "assign_master_ite_%04d.rt" % ite), np.array(master, dtype=float))
        if not master:
            raise NotImplementedError("no particle selected for alignment (empty task list)")
        per_proc = math.ceil(len(master) / int(self.vpr["how_many_processors"]))
        procs = {}
        for i, (ref, tag) in enumerate(master, 1):
            procs.setdefault(math.ceil(i / per_proc) - 1, []).append([ref, tag])
        for ref, tg in tags_in_ref.items():
            dynio.write_tags(self.p(task, "assign_ref_ref_%03d_ite_%04d.tags" % (ref, ite)), tg)
        for pr, rows in procs.items():
            dynio.write_rt(self.p(task, "assign_proc_proc_%05d_ite_%04d.rt" % (pr, ite)), np.array(rows, dtype=float))
        # S7: first card_iteref
        for ref in refs:
            c = dynio.read_card(self.f_card_ite(ite))
            c["database_ref"] = ref
            dynio.write_card(self.f_card_iteref(ref, ite), c)
        # S8: reference preprocessing
        self.preprocess_references(ite, refs)
        # S9: GPU chunks and the final card_iteref
        chunks = self.setup_gpu(ite, refs)
        # S10
        c = dynio.read_card(self.f_card_ite(ite))
        c["time_start_computation"] = _datestr()
        dynio.write_card(self.f_card_ite(ite), c)
        self.append_log([dynlog.setup_done_line(c["time_start_computation"])])  # :562-566, the card's own string
        return refs, chunks

    def preprocess_references(self, ite, refs):
        """dpkproject.preprocessReferences (06a S8)."""
        rnd = ite2round(self.vpr, ite)
        L = int(self.vpr["feature_data_sidelength"])
        m = dynio.read_em(self.p("settings", "mask.em"))
        if m.shape != (L, L, L):
            raise NotImplementedError("settings/mask.em size %s != sidelength %d (resize)" % (m.shape, L))
        mask_bin = (m > 0.5).astype(np.float64)                     # DVP 'mask', ops {'dim'} (S8.1)
        dynio.write_em(self.p(self.r_mask_transformed(ite)), mask_bin)
        high, low = float(self.rv("high", rnd)), float(self.rv("low", rnd))
        from .symmetry import dynamo_sym
        from .matlab_numerics import bandpass_filter
        op = self.rv("sym", rnd)
        with_fmask = float(self.vpr.get("symmetrize_with_fmask", 0)) != 0
        for ref in refs:
            f = dynio.read_em(self.f_fmask_starting_reference(ref, ite))
            ft = (f > 0.5).astype(np.float64)                        # DVP 'fmask_template', ops {'sym','dim'} (S8.2)
            fb = ft
            # preprocessReferences.m:183-234, DVP without 'fmask' (10a 3.2): ft = dynamo_sym(F > 0.5, op); for c1 the
            # identity; helical: hsym with dz (non-binary); then the "full" test on dynamo_sphere(31)
            ft = dynamo_sym(ft, op)[0]
            if np.all(ft[ellipsoid_inside(L, L / 2 - 1)] > 0):
                ft = np.ones_like(ft)
            dynio.write_em(self.p(self.r_fmask_transformed(ref, ite)), ft)
            T = dynio.read_em(self.f_starting_reference(ref, ite)).astype(np.float64)
            # S8.3: DVP 'template', ops dim -> sym -> band -> norm, 'fmask' = the raw starting fmask.
            # sym: DVP calls dynamo_sym(map, sym_r<rnd>[, 'fmask', fmask_rescaled]) (dynamo_database_volume_process.m
            # :424-464). 'c1' parses to kind 'none' (dynamo_zaux_parse_sym.m:119-127) and dynamo_sym_one returns the
            # volume unchanged before any csym / fmask-weighted division (dynamo_sym.m:380-385), whatever the fmask
            # and symmetrize_with_fmask. So the sym step is an exact identity for c1 (the only symmetry validated).
            # [review 07: 06a S8.0 describes a csym + std test + dynamo_fweight_divide path that dynamo_sym never
            # reaches for c1; it coincided with the identity on this run only because the std was 0.]
            # Other symmetries (10a 3.1): T = dynamo_sym(T, op, 'fmask', F > 0.5) if symmetrize_with_fmask (DVP
            # :432-433), else dynamo_sym(T, op) (:435).
            T = dynamo_sym(T, op, fb if with_fmask else None)[0]
            shell = isotropic_shell(L, high, low, 2.0)               # dynamo_bandpass smp = 2
            T = bandpass_filter(T, np.fft.fftshift(shell))             # real(ifftn(fftn(T).*shell)), FFTW
            T = normalize_roi(T, mask_bin, self._mthreads())                # setup runs in dynamo_execute_project
            dynio.write_em(self.p(self.r_template_transformed(ref, ite)), T)

    def setup_gpu(self, ite, refs):
        """dynamo__iteration_setup_gpu (06a S9.2, S9.3)."""
        task, inter = self.r_task(ite), self.r_inter(ite)
        G = len(self.gpu_ids_card)
        hmp = int(self.vpr["how_many_processors"])
        pad = int(self.vpr["feature_data_padding"])
        chunks = []
        for ref in refs:
            tags = _tags(dynio.read_dlm(self.p(task, "assign_ref_ref_%03d_ite_%04d.tags" % (ref, ite))))
            per = math.ceil(len(tags) / (hmp * G))
            T = dynio.read_tbl(self.f_starting_table(ref, ite))
            if len(np.unique(_tags(T))) != len(T):
                raise NotImplementedError("repeated tags in the starting table (engine row pairing quirk, 06a S9.2)")
            c = 1
            for proc in range(hmp):
                for gpu in self.gpu_ids_card:
                    tg = tags[(c - 1) * per: c * per]
                    if len(tg) == 0:
                        raise NotImplementedError("empty GPU chunk: Dynamo would then send the whole table (06a S9.2)")
                    base = "proc_%05d_ite_%04d_ref_%03d_gpu_%02d" % (proc, ite, ref, gpu)
                    dynio.write_tags(self.p(task, "assign_gpu_%s.tags" % base), tg, padding=pad)
                    pl = self.p(task, "assign_gpu_particleList_%s.txt" % base)
                    dynio.write_lines(pl, [self.particle_line(t) for t in tg.tolist()])
                    tab = T[np.isin(_tags(T), tg)]                    # dynamo_table_grep: T order
                    sb = self.p(inter, "starting_gpuProcRefTable_ite_%04d_ref_%03d_proc_%05d_gpu_%02d.btbl"
                                % (ite, ref, proc, gpu))
                    dynio.write_btbl(sb, tab)
                    rb = self.p(inter, "refined_gpuProcRefTable_ite_%04d_ref_%03d_proc_%05d_gpu_%02d.btbl"
                                % (ite, ref, proc, gpu))
                    chunks.append({"ref": ref, "gpu": gpu, "n": int(len(tg)), "particle_list": pl,
                                   "starting_btbl": sb, "refined_btbl": rb})
                    c += 1
            sc = dynio.read_card(self.f_card_ite(ite))               # S9.3
            sc["database_ref"] = ref
            sc["database_gpu_folder_data_iteration"] = str(self.vpr["folder_data"])
            sc["database_gpu_file_template_iteration"] = self.dstr(self.r_template_transformed(ref, ite))
            sc["database_gpu_file_fmask_iteration"] = self.dstr(self.r_fmask_transformed(ref, ite))
            sc["database_gpu_file_mask"] = self.dstr(self.r_mask_transformed(ite))
            sc["database_gpu_root_file_tags"] = self.dstr(task + "/assign_gpu_proc_00000_ite_%04d_ref_%03d_gpu_"
                                                          % (ite, ref))
            sc["database_gpu_root_file_particleList"] = self.dstr(
                task + "/assign_gpu_particleList_proc_00000_ite_%04d_ref_%03d_gpu_" % (ite, ref))
            sc["database_gpu_starting_ptable_root"] = self.dstr(inter + "/starting_ptable_ite_%04d_ref_%03d_tag_"
                                                                % (ite, ref))
            sc["database_gpu_refined_ptable_root"] = self.dstr(inter + "/refined_ptable_ite_%04d_ref_%03d_tag_"
                                                               % (ite, ref))
            sc["database_gpu_shifts_shrink_factor"] = 1               # dim == sidelength (S9.1)
            sc["database_gpu_file_particle_extension"] = str(self.vpr["feature_extension_data"])
            sc["gpu_identifier_set"] = self.vpr["gpu_identifier_set"]
            dynio.write_card(self.f_card_iteref(ref, ite), sc)
        return chunks

    # ------------------------------------------------------------------ compute
    def _start_workers(self):
        if self.workers:
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        cfg = {"out_root": self.out, "name_project": self.name, "folder_data": str(self.vpr["folder_data"]),
               "data_dir": self.data, "dynamo_version": DYNAMO_VERSION, "engine": self.engine,
               "exact_sms": self.exact_sms}
        for dev in self.gpus:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_gpu_worker, args=(dev, child, cfg), daemon=True)
            proc.start()
            self.workers.append((dev, parent, proc))
        for dev, conn, _ in self.workers:
            r = conn.recv()
            if not r.get("ok"):
                raise RuntimeError("GPU worker %d failed to start:\n%s" % (dev, r.get("error")))
            self.gpu_name, self.multiprocessors = r["gpu_name"], r["multiprocessors"]   # device 0 of the node

    def close(self):
        """Stop the GPU workers: an idle worker ends on None. A worker whose reply was not read (another worker
        failed first in _ask_all) may still be aligning and would keep its GPU and write into temp/ after the run
        stopped: it is terminated, as is any worker still alive after the join timeout. Every reply that was read
        came after the worker had written its files, so no result changes."""
        busy = getattr(self, "_busy", set())
        for k, (_, conn, proc) in enumerate(self.workers):
            if k in busy:
                proc.terminate()
                continue
            try:
                conn.send(None)
            except Exception:
                pass
        for _, _, proc in self.workers:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self.workers = []
        self._busy = set()

    def _ask_all(self, msgs):
        """One message to every worker, then the replies in worker order. On a failure the workers whose reply is
        not read stay in self._busy (close() terminates them)."""
        self._busy = set()
        for k, ((dev, conn, _), m) in enumerate(zip(self.workers, msgs)):
            self._busy.add(k)
            conn.send(m)
        out = []
        for k, (dev, conn, _) in enumerate(self.workers):
            r = conn.recv()
            self._busy.discard(k)
            if not r.get("ok"):
                raise RuntimeError("GPU worker on device %d failed:\n%s" % (dev, r.get("error")))
            out.append(r)
        return out

    def compute(self, ite, refs, chunks):
        """One engine run per surviving reference (06c A.2 computeFunctionMatlabGPU); the chunks of the card's GPUs
        are distributed over the physical workers. Per run the log gets the engine's stdout (the deterministic
        equivalent of the threads' output, device blocks in gpu_identifier_set order) and the 'Returning control'
        line, and <cwd>/gpulog.txt holds the stdout as the engine run left it (10f 1.1, 8.3;
        dynamo_pipeline_system_check_output_gpu.m:95-104)."""
        self._start_workers()
        res = []
        for ref in refs:
            card = self.f_card_iteref(ref, ite)
            mine = [[] for _ in self.workers]
            ref_chunks = [c for c in chunks if c["ref"] == ref]
            for i, ch in enumerate(ref_chunks):
                mine[i % len(self.workers)].append(ch)
            t_enter = time.time()
            try:
                r = self._ask_all([{"op": "align", "card": card, "chunks": m, "slot": ref} for m in mine])
            except Exception:
                # the engine run failed (Dynamo: the engine exits != 0, e.g. on a particle file it cannot open):
                # dynamo_pipeline_system_check_output_gpu.m:40-92 appends its warning block for the order
                # sprintf('%s %s',myMotor,cardfile) (dynamo_execute_project.m:296-297; getGPUmotor.m:6), and no
                # engine stdout, gpulog.txt or 'Returning control' line, then error() stops the run
                self.append_log(dynlog.gpu_failure_lines("dynamo_compute_iteration_spp "
                                                         + dynlog.card_iteref_path(self.name, ref, ite)))
                raise
            res += r
            ecard = dynlog.EngineCard(card, self.dstr("cards/ite_%04d/card_iteref_ref_%03d_ite_%04d.card" % (ite, ref, ite)))
            dims = {k: dynlog.em_dims(self.p(f(ref, ite) if k != "mask" else f(ite)))
                    for k, f in (("template", self.r_template_transformed), ("mask", self.r_mask_transformed),
                                 ("tfmask", self.r_fmask_transformed))}
            devices = [dynlog.DeviceRun(ch["gpu"], dynio.read_lines(ch["particle_list"]), self.gpu_name,
                                        self.multiprocessors) for ch in ref_chunks]
            out = dynlog.engine_stdout(ecard, devices, dims, dynlog.asctime(t_enter), dynlog.asctime(time.time()),
                                       sum(x.get("cpu_s", 0.0) for x in r))
            self.append_log(dynlog.stdout_to_log_lines(out))
            with open(os.path.join(os.path.dirname(self.out), "gpulog.txt"), "w") as f:
                f.write(out)                                              # fprintf(fid,'%s',standard_output)
            self.append_log([dynlog.returning_control_line(dynlog.datestr())])
        return res

    # ------------------------------------------------------------------ assemble (06b)
    def gather_table(self, ite, ref):
        """06b 2: gather the per-GPU float32 tables and write refined_table as %.5g text (write 1)."""
        files = sorted(glob.glob(self.p(self.r_inter(ite), "refined_gpuProcRefTable_ite_%04d_ref_%03d_proc_*_gpu_*.btbl"
                                        % (ite, ref))))
        if not files:
            raise FatalIterationError("no refined GPU tables for ite %d ref %d" % (ite, ref))
        parts = []
        for f in files:
            t = dynio.read_btbl(f)
            parts.append(t[_tags(t) != 0])                              # gatherGPUforIteRef.m:54-55
        merged = table_merge(parts)
        f_ref = self.f_av("refined_table", ref, ite, "tbl")
        dynio.write_tbl(f_ref, merged)                                 # write 1

    def compare_table(self, ite, ref):
        """06b 3: columns 27-30 from the re-read text table and the previous table (write 2)."""
        info = {}
        f_ref = self.f_av("refined_table", ref, ite, "tbl")
        table = dynio.read_tbl(f_ref)
        prev_f = self.table_initial if ite <= 1 else f_ref              # ite >= 2: compared with itself (06b 3.1)
        if ite <= 1 and int(self.vpr["nref_r1"]) > 1:
            # linked_table_initial of this ref: dynamo_check_initial_folders(vpr.file_table_initial, ...) (09a 6.3)
            from .vpreo import check_initial_folders
            files, _ = check_initial_folders(self.table_initial, "table", int(self.vpr["nref_r1"]))
            prev_f = files[ref - 1] if len(files) >= ref and files[ref - 1] else os.path.join(
                self.table_initial, "table_initial_ref_%03d.tbl (missing)" % ref)
        try:
            if not os.path.exists(prev_f):
                raise FileNotFoundError(prev_f)
            prev = dynio.read_tbl(prev_f)
            ct, pt = _tags(table), _tags(prev)
            if len(np.unique(ct)) != len(ct) or len(np.unique(pt)) != len(pt):
                raise ValueError("repeated tags in a compared table")
            common = np.intersect1d(ct, pt)
            if not np.array_equal(common, ct):
                raise NotImplementedError("table tags not all present (sorted) in the previous table: "
                                          "dynamo_table_set mapping quirk (06b 3.4) not ported")
            pi = {int(t): i for i, t in enumerate(pt.tolist())}
            prow = np.real(prev[[pi[int(t)] for t in common.tolist()]])
            from .symmetry import symorder
            from .symmetry import parse_sym
            sym_op = self.rv("sym", ite2round(self.vpr, ite))
            c27, c28, c29, c30 = compare_columns(np.real(table), prow, self.trig, symorder(sym_op),
                                                 sym_kind=parse_sym(sym_op)[0])
            if np.any(c28.imag != 0):
                table = table.astype(complex)
                table[:, 27] = c28
            else:
                table[:, 27] = c28.real
            table[:, 26], table[:, 28], table[:, 29] = c27, c29, c30
            dynio.write_tbl(f_ref, table)                              # write 2
            info["complex_col28_rows"] = int(np.sum(c28.imag != 0))
        except NotImplementedError:
            raise
        except Exception as e:                                         # non-fatal in Dynamo (assemble :219-235)
            self.log("ite %d: table comparison failed (%s); table not rewritten" % (ite, e))
        return info

    def threshold_table(self, ite, ref):
        """06b 6: thresholding, modus 0 (dynamo_pipeline_threshold.m); peak check and overlap removal are no-ops
        (validated). Write 3."""
        info = {}
        f_ref = self.f_av("refined_table", ref, ite, "tbl")
        T = dynio.read_tbl(f_ref)
        processed = np.real(T[:, 1]) != 0
        contrib = processed & (np.real(T[:, 9]) > -np.inf)
        T[:, 2] = 0
        T[contrib, 2] = 1
        dynio.write_tbl(f_ref, T)                                      # write 3
        info["contributors"] = int(contrib.sum())
        return info

    def assemble_tables(self, ite, ref):
        """Gather, compare and threshold one reference's table (06b 2-6) without the log lines: the three phases
        of assemble() in sequence (harness / test entry point)."""
        self.gather_table(ite, ref)
        info = dict(self.compare_table(ite, ref))
        info.update(self.threshold_table(ite, ref))
        return info

    def assemble(self, ite, refs):
        """dynamo_iteration_assemble.m in its phases (10f 1.1, 3): gather -> compare -> overlap -> threshold for
        every reference, then the average of every reference, so that log.txt (appended at each phase, in
        Dynamo's order) is also right when a phase fails."""
        t_begin = _datestr()
        rnd = ite2round(self.vpr, ite)
        card_ite = dynio.read_card(self.f_card_ite(ite))
        nref = int(float(card_ite["nref"]))
        head = dynlog.assemble_head_lines()
        info = {}
        self.append_log(head[0:1])
        for ref in refs:
            try:
                self.gather_table(ite, ref)
            except FatalIterationError:
                self._fatal_assemble(ite, "[assemble] failed to create table files, even before thresholding, "
                                          "averaging and convergence checking")
                raise
        self.append_log(head[1:2])
        for ref in refs:
            info.update(self.compare_table(ite, ref))
        self.append_log(head[2:3])
        self.append_log(head[3:4])
        for ref in refs:
            info.update(self.threshold_table(ite, ref))
        avg_lines = dynlog.assemble_average_lines(ite, nref, refs)
        self.append_log(avg_lines[:1])
        t_avg0 = time.time()
        for iref in range(1, nref + 1):
            if iref not in refs:
                self.append_log([avg_lines[iref]])
                continue
            # 7. averaging
            t0 = time.perf_counter()
            try:
                self.average(ite, iref)
            except FatalIterationError:
                # pipeline.average returned ok = false: the echo comes before the ok test (assemble :343-355)
                self.append_log([avg_lines[iref]])
                self._fatal_assemble(ite, "[assemble] did create apparently valid table files, but it failed while "
                                          "using them  for averaging")
                raise
            info["average_s"] = round(time.perf_counter() - t0, 3)
            self.append_log([avg_lines[iref]])
        self.append_log([dynlog.assemble_average_done_line(time.time() - t_avg0)])
        # 8. dynamo_sym(average, op, 'fmask', fmask_average, 'foutput', 'binary') (assemble :457-479; 10a 5.2), in
        # its own loop after every reference is averaged (:331-360) and the 'done' line (:362-369).
        # c1: dynamo_sym returns its inputs (06b 8)
        op = self.rv("sym", rnd)
        from .symmetry import parse_sym, dynamo_sym
        for ref in range(1, nref + 1):
            if ref not in refs:
                continue
            if parse_sym(op)[0] == "none":
                dynio.write_em(self.f_av("average_symmetrized", ref, ite),
                               dynio.read_em(self.f_av("average", ref, ite)))
                dynio.write_em(self.f_av("fmask_average_symmetrized", ref, ite),
                               dynio.read_em(self.f_av("fmask_average", ref, ite)))
            else:
                As, Fs = dynamo_sym(dynio.read_em(self.f_av("average", ref, ite)), op,
                                    dynio.read_em(self.f_av("fmask_average", ref, ite)), "binary")
                dynio.write_em(self.f_av("average_symmetrized", ref, ite), As)
                dynio.write_em(self.f_av("fmask_average_symmetrized", ref, ite), Fs)
        # 9. convergence (type 0 never saturates; writes nothing) -> next = ite + 1
        nxt = ite + 1
        dynio.write_txt(self.p("next_iteration.txt"), nxt)
        self.append_log(dynlog.convergence_lines(nxt))                 # assemble.m:529-599, type 0: never saturated
        # 11. propagation
        src = ["average", "fmask_average", "refined_table"]
        if "update_fmask" in card_ite and not float(card_ite["update_fmask"]):
            src[1] = "fmask_starting_reference"
        if "update_with_unweighted_average" in card_ite and float(card_ite["update_with_unweighted_average"]):
            src[0] = "average_unweighted"
        for ref in refs:
            dst = [self.f_starting_reference(ref, nxt), self.f_fmask_starting_reference(ref, nxt),
                   self.f_starting_table(ref, nxt)]
            for s, d in zip(src, dst):
                sf = (self.f_fmask_starting_reference(ref, ite) if s == "fmask_starting_reference"
                      else self.f_av(s, ref, ite, "tbl" if s == "refined_table" else "em"))
                if not os.path.isfile(sf):
                    raise FatalIterationError("file not found: %s" % sf)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(sf, d)
        get_rid = float(self.rv("threshold_modus", rnd)) > 10
        surviving = list(refs)
        for ref in refs:
            f = self.f_starting_table(ref, nxt)
            T = dynio.read_tbl(f)
            if T.size == 0:
                surviving.remove(ref)
                self.set_surviving_refs(nxt, surviving)
                continue
            if get_rid:
                avg = np.real(T[:, 2]) == 1
                T[:, 1] = 0
                T[avg, 1] = 1
            dynio.write_tbl(f, T)
        if not self.keep_temp:
            shutil.rmtree(self.p("temp", "ite_%04d" % ite), ignore_errors=True)
        self.set_surviving_refs(nxt, surviving)
        # golden-standard adaptive bandpass: the last step of assemble, after the propagation above
        # (dynamo_iteration_assemble.m:812-822; 09a 6.3, 09b 3). The vpr is the one loaded by assemble.
        if float(load_vpr(self.p("settings", "virtual_project.mat")).get("adaptive_bandpass", 0)):
            from . import abp
            t0 = time.perf_counter()
            info["abp"] = abp.adaptive_filtering(self, ite)
            info["abp_s"] = round(time.perf_counter() - t0, 3)
        c = dynio.read_card(self.f_card_ite(ite))
        c["time_end_refinement"] = t_begin
        c["time_end_iteration"] = t_end = _datestr()                   # one value for the card and the log (10f 8.4h)
        dynio.write_card(self.f_card_ite(ite), c)
        self.append_log(dynlog.ending_lines(self.dstr("cards/ite_%04d/card_ite_ite_%04d.card" % (ite, ite)),
                                            c["time_start_computation"], t_begin, t_end))
        return info

    def _fatal_assemble(self, ite, reason):
        """dynamo_iteration_assemble.m:858-874 fatal_error: next_iteration := last possible + 1 and the log lines."""
        dynio.write_txt(self.p("next_iteration.txt"), self.N + 1)
        self.append_log(dynlog.assemble_fatal_lines(ite, reason))

    def _mthreads(self):
        """MATLAB computational threads of Dynamo's dynamo_execute_project to reproduce (--matlab-threads; default
        MATLAB's own, one per available physical core)."""
        t = getattr(self, "matlab_threads", None)
        if not t:
            from .matlab_numerics import matlab_threads
            t = matlab_threads()
        return int(t)

    def average_from_table(self, ite, ref, table, fsc_sym=None):
        """The dynamo_average call of pipeline/average.m and genericInput2Average.m (06b 7): rows with col 3 == 1,
        nmask = settings/mask.em (raw), fmin 3, even/odd halves of the sorted tags. `table` is used as given (the
        re-read text table for the pipeline average; the in-memory transformed table for the ABP re-average of
        reference 2, 09b 4.5). Averaged on the GPU workers that hold reference `ref`. Returns finalize()'s dict."""
        from .average import finalize, halves_for, new_state
        T = np.real(np.asarray(table))
        rows = T[T[:, 2] == 1]
        if len(rows) == 0:
            # genericInput2Average.m:298-301 returns before its output 'flags' is assigned (:316), which
            # pipeline/average.m:100 requests: a MATLAB error, not ok = false. Nothing catches it (assemble :343,
            # dynamo_execute_project.m:185): no 'Computing average for ref' line, no fatal_error, next_iteration.txt
            # unchanged. A plain RuntimeError, which assemble does not turn into _fatal_assemble.
            raise RuntimeError("no particle with col 3 == 1 for averaging: Dynamo stops with a MATLAB error here "
                               "(genericInput2Average.m:298-301, 'Sorry, not tags survive: aborting')")
        if len(rows) == 1:                    # genericInput2Average.m:292-294: one tag -> no fsc, a different path
            raise NotImplementedError("averaging a single particle (dynamo_average without the even/odd FSC) is not "
                                      "ported")
        bad = set(np.unique(rows[:, 12]).astype(int).tolist()) - {0, 1, 2}
        if bad:
            raise NotImplementedError("wedge types %s in averaging" % sorted(bad))
        rows = rows[np.argsort(rows[:, 0], kind="stable")]            # intersect(): ascending tags (05 3.2)
        half = halves_for(rows)                                        # odd = tags(1:2:end), even = tags(2:2:end)
        nh = np.bincount(half, minlength=2).astype(np.float64)
        mask = dynio.read_em(self.p("settings", "mask.em"))
        n = mask.shape[0]
        self._start_workers()
        held = self._ask_all([{"op": "tags", "slot": ref} for _ in self.workers])
        owner = {}
        for w, r in enumerate(held):
            for t in np.asarray(r["tags"], dtype=np.int64).tolist():
                owner.setdefault(t, w)
        tags = rows[:, 0].astype(np.int64)
        if any(int(t) not in owner for t in tags):
            raise RuntimeError("averaging tags are not all resident on the GPU workers")
        # MATLAB threads (benchmarks/14_msa_b4/abp_probe, 18_untested): dynamo_average runs inside
        # dynamo_execute_project, one computational thread per core. Without a pool: runs of consecutive tags held
        # by the same worker, the state going through them in order (one serial sum per half, as dynamo_average's
        # serial loop). With matlab_workers_average > 0 Dynamo opens a parpool and the particle and weight loops
        # are parfor loops on its single-threaded workers (below). The finalisation always runs in the main process.
        W = int(float((getattr(self, "vpr", None) or {}).get("matlab_workers_average", 0) or 0))
        w_of = np.array([owner[int(t)] for t in tags])
        m32 = mask.astype(np.float32)
        if W <= 0:
            cuts = np.flatnonzero(np.diff(w_of)) + 1
            state = new_state(n)
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(rows)]):
                dev, conn, _ = self.workers[int(w_of[a])]
                conn.send({"op": "average", "rows": rows[a:b], "half": half[a:b], "nh": nh,
                           "mask": m32, "state": state, "slot": ref, "threads": self._mthreads()})
                r = conn.recv()
                if not r.get("ok"):
                    raise RuntimeError("GPU worker on device %d failed:\n%s" % (dev, r.get("error")))
                state = r["state"]
        else:
            # dynamo_average with a pool: per half, MATLAB's parfor subranges of the half's ascending tags
            # (matlab_numerics.parfor_subranges), each run in descending order on a single-threaded worker, the
            # partial sums added in subrange order (MATLAB's usual order; benchmarks/19_parfor). A subrange split
            # between GPU workers continues its partial sum through them.
            from .average import new_pool_state
            from .matlab_numerics import parfor_subranges
            order, pieces = [], []                               # processing order (row index) and pieces over it
            for k in (0, 1):
                idx = np.flatnonzero(half == k)
                a = 0
                for size in parfor_subranges(len(idx), W):
                    sub = idx[a:a + size][::-1]
                    a += size
                    cut = np.flatnonzero(np.diff(w_of[sub])) + 1
                    bounds = list(zip(np.r_[0, cut], np.r_[cut, len(sub)]))
                    for j, (u, v) in enumerate(bounds):
                        pieces.append([len(order), len(order) + v - u, k, j == 0, j == len(bounds) - 1])
                        order.extend(sub[u:v].tolist())
            order = np.asarray(order, dtype=np.int64)
            state = new_pool_state(n)
            g = 0
            while g < len(pieces):                               # consecutive pieces on one worker: one message
                w = int(w_of[order[pieces[g][0]]])
                h = g
                while h < len(pieces) and int(w_of[order[pieces[h][0]]]) == w:
                    h += 1
                p0, p1 = pieces[g][0], pieces[h - 1][1]
                sel = order[p0:p1]
                rel = [(q[0] - p0, q[1] - p0, q[2], q[3], q[4]) for q in pieces[g:h]]
                dev, conn, _ = self.workers[w]
                conn.send({"op": "average", "rows": rows[sel], "half": half[sel], "nh": nh, "mask": m32,
                           "state": state, "slot": ref, "threads": 1, "pieces": rel})
                r = conn.recv()
                if not r.get("ok"):
                    raise RuntimeError("GPU worker on device %d failed:\n%s" % (dev, r.get("error")))
                state = r["state"]
                g = h
        return finalize(state, nh, mask, n, fmin=3.0, sym=fsc_sym, threads=self._mthreads())

    def average(self, ite, ref):
        """dpkproject.pipeline.average (06b 7): table = the refined_table TEXT after thresholding, tags col3 == 1,
        nmask = settings/mask.em (raw), fmin 3, fsc on even/odd halves."""
        # genericInput2Average.m:241-258: fsc_actions = 'sym <sym_r<round>>;' -> eo_fsc of the symmetrized halves
        out = self.average_from_table(ite, ref, dynio.read_tbl(self.f_av("refined_table", ref, ite, "tbl")),
                                      fsc_sym=self.rv("sym", ite2round(self.vpr, ite)))
        # pipeline/average.m:127-184 checks output.average, .average_raw, .fmask, .fweight and .fsc (fsc_option is
        # true here) for NaN; any NaN -> ok = false, nothing is written, and dynamo_iteration_assemble.m:347-355
        # calls fatal_error (:858-866): next_iteration.txt := dynamo_vpr_status(vpr,'end') + 1, then return.
        for k in ("average", "average_unweighted", "fmask_average", "fweight_average_raw", "eo_fsc"):
            if np.isnan(np.asarray(out[k])).any():
                dynio.write_txt(self.p("next_iteration.txt"), self.N + 1)
                raise FatalIterationError("NaN in %s: pipeline/average.m fails, assemble fatal_error "
                                          "(next_iteration := %d)" % (k, self.N + 1))
        fmask_average = (np.asarray(out["fmask_average"]) >= 0.5).astype(np.float64)   # dynamo_binarize(., 0.5)
        dynio.write_em(self.f_av("average", ref, ite), out["average"])
        dynio.write_em(self.f_av("average_unweighted", ref, ite), out["average_unweighted"])
        dynio.write_em(self.f_av("fweight_average_raw", ref, ite), out["fweight_average_raw"])
        dynio.write_em(self.f_av("fmask_average", ref, ite), fmask_average)
        dynio.write_fsc(self.f_av("eo_fsc", ref, ite, "fsc"), np.asarray(out["eo_fsc"]).reshape(1, -1))

    # ------------------------------------------------------------------ check (06c A.4)
    def check(self, ite):
        """dynamo_iteration_check.m:60-114: every item of every surviving reference must exist and hold no NaN
        (item-major order as in the source); the reason of a fatal error is the entity text of the last failure.
        Returns True, or False after fatal_error (:106, :118-149), which returns normally: the loop of
        dynamo_execute_project.m then finds next_iteration > N and returns (:144-147), or ends. Missing or empty
        surviving references of ite + 1 are errors in Dynamo (:62-64) and raise here."""
        refs = self.get_surviving_refs(ite + 1)
        problems, reason = [], None
        for kind, ext in (("refined_table", "tbl"), ("average", "em"), ("fmask_average", "em")):
            for ref in refs:
                f = self.f_av(kind, ref, ite, ext)
                entity = 'database item "%s", ite:%d, ref:%d  ' % (kind, ite, ref)
                if not os.path.exists(f):
                    problems.append("%s FAILED (missing)" % entity)
                    reason = entity
                    continue
                v = dynio.read_tbl(f) if ext == "tbl" else dynio.read_em(f)
                n_nan = int(np.isnan(np.real(v)).sum() + (np.isnan(v.imag).sum() if np.iscomplexobj(v) else 0))
                if n_nan:
                    problems.append("%s FAILED (%d NaN entries)" % (entity, n_nan))
                    reason = entity
        if problems:
            dynio.write_txt(self.p("next_iteration.txt"), self.N + 1)  # fatal_error: deactivate the rest
            self.append_log(dynlog.check_fatal_lines(ite, reason))
            self.log("FATAL ERROR DETECTED IN ITERATION %d: %s" % (ite, "; ".join(problems)))
            return False
        self.append_log(dynlog.check_ok_lines(ite))
        return True

    # ------------------------------------------------------------------ driver (06c A.2, A.5)
    def run(self, iterations=None):
        t_all = time.perf_counter()
        self.initialize()
        last = self.N if iterations is None else min(self.N, int(iterations))
        try:
            for ite in range(1, self.N + 1):
                if ite > last:
                    break
                nx = dynio.read_txt(self.p("next_iteration.txt"))
                nx = int(np.real(nx).ravel()[0]) if nx.size else None
                if nx is not None and nx > self.N:
                    self.log("Maximal number of iterations attained: %d" % nx)
                    break
                if nx is not None and nx > ite:
                    continue
                if not os.path.exists(self.f_card_ite(ite)):
                    raise FatalIterationError("missing card %s" % self.f_card_ite(ite))
                rnd = ite2round(self.vpr, ite)
                it = {"ite": ite, "round": rnd}
                self.log("iteration %d (round %d): setup" % (ite, rnd))
                t0 = time.perf_counter()
                refs, chunks = self.setup(ite)
                it["setup_s"] = round(time.perf_counter() - t0, 3)
                it["chunks"] = [{"gpu": c["gpu"], "n": c["n"]} for c in chunks]
                self.log("iteration %d: compute (%d particles, %d chunks on devices %s)"
                         % (ite, sum(c["n"] for c in chunks), len(chunks), self.gpus))
                t0 = time.perf_counter()
                it["workers"] = self.compute(ite, refs, chunks)
                it["compute_s"] = round(time.perf_counter() - t0, 3)
                self.log("iteration %d: assemble" % ite)
                t0 = time.perf_counter()
                it.update(self.assemble(ite, refs))
                it["assemble_s"] = round(time.perf_counter() - t0, 3)
                t0 = time.perf_counter()
                ok = self.check(ite)                  # False: next_iteration.txt = N + 1, the next pass stops
                it["check_s"] = round(time.perf_counter() - t0, 3)
                it["wall_s"] = round(it["setup_s"] + it["compute_s"] + it["assemble_s"] + it["check_s"], 3)
                if not ok:
                    it["fatal"] = True
                self.report["iterations"].append(it)
                self.log("iteration %d %s (setup %.1f s, compute %.1f s, assemble %.1f s)"
                         % (ite, "ok" if ok else "failed the check", it["setup_s"], it["compute_s"],
                            it["assemble_s"]))
                self._write_report(t_all)
        finally:
            self.close()
            self._write_report(t_all)
        return self.report

    def _write_report(self, t_all):
        """The opt-in diagnostics report (--report PATH / $PYDYN_REPORT; never inside the project folder)."""
        self.report["total_wall_s"] = round(time.perf_counter() - t_all, 3)
        if not self.report_path:
            return
        with open(self.report_path, "w") as f:
            json.dump(self.report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
