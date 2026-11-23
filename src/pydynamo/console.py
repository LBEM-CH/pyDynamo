"""Dynamo 1.1.514 console functions of the user's journey: dread, dwrite, dtrandomize_azimuth, daverage (and rng).

Specification: docs/analysis/08b_console_functions_1.1.514.md ("08b"). S = /programs/x86_64-linux/dynamo/1.1.514/
matlab/src. Arrays follow pydynamo: volumes are numpy [z, y, x] (C order = MATLAB's column-major bytes), tables are
[rows, columns]. Paths are relative to the current working directory, as in MATLAB.

Everything 08b marks uncertain, and every option not ported, raises NotImplementedError.
"""
import os
import time

import numpy as np

from pydynamo.files import formats

_TEXT_TABLE = ("txt", "tbl", "tags", "rt", "ts", "fsc", "ctf", "coords", "hdf", "xyz")
_LINES = ("doc", "m", "sh", "sel", "exe", "cmd", "cmm", "bat", "vll", "ms", "mfl")


from pydynamo.vpr import DynamoError  # noqa: E402  (one error type for the whole Stage 3 API)


# =====================================================================================================================
# dread (shorthands/dread.m -> dynamo_read.m:138-386, 08b section 1)
# =====================================================================================================================

def read_mrc(path):
    """dynamo_read_mrc_simple.m:57-81 + dynamo_read_header.m:488-640, mode 2 only (08b 1.3): float32 [z, y, x],
    data offset 1024 + nsymbt."""
    with open(path, "rb") as f:
        h = f.read(1024)
        hw = np.frombuffer(h, "<i4")
        nx, ny, nz, mode = (int(x) for x in hw[:4])
        nsymbt = int(hw[23])
        if mode != 2:
            raise NotImplementedError("MRC mode %d: the class Dynamo's MEX reader returns is not verified (08b 1.3)"
                                      % mode)
        f.seek(1024 + nsymbt)
        v = np.frombuffer(f.read(4 * nx * ny * nz), "<f4").reshape(nz, ny, nx)
    return v.copy()


def mrc_bytes(vol):
    """dynamo_write_mrc_simple.m:62-150 (08b 2.3): 256 int32 words, words 1-3 = size, 4 = 2, 8-10 = 1,
    17-19 = 1,2,3, everything else 0; then single(v) little-endian from byte 1024."""
    v = np.asarray(vol)
    if np.iscomplexobj(v):
        raise NotImplementedError("complex volumes")
    if v.ndim == 1:
        v = v.reshape(1, 1, -1)
    elif v.ndim == 2:
        v = v.reshape((1,) + v.shape)
    nz, ny, nx = v.shape
    h = np.zeros(256, "<i4")
    h[0:3] = (nx, ny, nz)
    h[3] = 2
    h[7:10] = 1
    h[16:19] = (1, 2, 3)
    return h.tobytes() + np.ascontiguousarray(v, dtype="<f4").tobytes()


def write_mrc(path, vol):
    formats._mkparent(path)
    with open(path, "wb") as f:
        f.write(mrc_bytes(vol))


def _o_error5(lines):
    """mbs.o: o.error(5) prints 5 lines of 60 '#', then each o.e(...) line [RUN e_review_checks.log:13-20]."""
    for _ in range(5):
        print("#" * 60)
    for x in lines:
        print(x)


def dread(file_name, tolerance_level=0, fmt=None):
    """dread(f): .tbl/.txt/... -> float64 (complex if a token is complex) [rows, cols]; .em/.sem -> float64
    [z, y, x] (uint8 for EM type 1); .btbl -> the table [rows, cols] (nx rows, ny columns: MATLAB's nx x ny matrix of
    dynamo_read_emfile), [z, y, x] if it has nz > 1; .mrc mode 2 -> float32; line files -> list of str; .card ->
    dict; an existing file without extension -> dynamo_vpr_load. fmt replaces the extension, without its dots
    (dynamo_read.m:270-271). A missing file raises (tolerance 0) or returns None
    (tolerance 1); a folder raises as in Dynamo (dynamo_read.m:212-266: "no policy defined for this type of folder",
    the output is never assigned), also for a project folder: use pydynamo.vpr.load for a project."""
    if isinstance(file_name, (list, tuple)):
        return [dread(f, tolerance_level, fmt) for f in file_name]
    if not isinstance(tolerance_level, (int, float)):
        raise NotImplementedError("dread with a char tolerance_level (assignment to the base workspace)")
    if "*" in file_name:
        raise NotImplementedError("dread with '*' expansion")
    if "$" in file_name:
        raise NotImplementedError("dread with a $VARIABLE in the name (mbsys.access.isfile)")
    if not os.path.isfile(file_name):                # exist(file_name,'file') ~= 2 (dynamo_read.m:212-266)
        if os.path.isdir(file_name):                 # :226-251
            if os.path.splitext(file_name)[1] in (".PCA", ".AWF") or \
                    os.path.isfile(os.path.join(file_name, "object.mat")):
                raise NotImplementedError("dread of a .PCA / .AWF / object.mat folder")
            _o_error5(["no policy defined for this type of folder"])
            # the function returns without assigning its output [RUN review/reference/logs/e_review_checks.log]
            raise DynamoError('Output argument "my_object" (and maybe others) not assigned during call to '
                              '"dynamo_read>localReadOne".')
        if tolerance_level:
            return None
        _o_error5(["Error in dynamo_read", "Missing file!!!", "  * missing file detected: %s" % file_name])
        raise DynamoError('[dread: error] cannot find requested file "%s"' % file_name)
    if fmt:
        ext = fmt.replace(".", "")                   # :270-271: strrep(my_format,'.','')
    else:
        ext = os.path.splitext(file_name)[1].lstrip(".")
        if ext == "":                                # :274-279, an existing file without extension
            from pydynamo import vpr
            return vpr.load(file_name)
    if ext in _TEXT_TABLE:
        try:
            return formats.read_dlm(file_name)
        except Exception:
            return np.zeros((0, 0))                  # dynamo_read.m:317-324: any exception -> []
    if ext == "btbl":
        v = formats.read_em(file_name)                 # :329-330, dynamo_read_emfile: reshape(data, nx, ny, nz)
        if v.shape[0] == 1:                          # nz = 1: the nx x ny matrix, i.e. the table [rows, cols]
            return np.ascontiguousarray(v[0].T)
        return v
    if ext in ("em", "sem"):
        return formats.read_em(file_name)
    if ext in ("mrc", "map", "rec", "ali", "st", "mrcs", "preali"):
        return read_mrc(file_name)
    if ext in _LINES:
        return formats.read_lines(file_name)
    if ext == "card":
        return formats.read_card(file_name)
    raise NotImplementedError("dread of .%s files (08b 1)" % ext)


# =====================================================================================================================
# dwrite (shorthands/dwrite.m -> dynamo_write.m:114-281, 08b section 2)
# =====================================================================================================================

def dwrite(obj, file_name):
    """dwrite(obj, f): dispatch on the extension; missing parent folders are created (dynamo_write.m:134-138)."""
    ext = os.path.splitext(str(file_name))[1].lstrip(".")
    if ext == "tbl":
        formats.write_tbl(file_name, np.asarray(obj))
    elif ext in ("em", "sem", "btbl"):
        a = np.asarray(obj)
        if a.dtype == bool:
            a = a.astype(np.float32)
        if ext == "btbl" and a.ndim <= 2:
            # a table [rows, cols] (a 1-D array: one row, as a MATLAB row vector): dynamo_write_em_simple.m:60,92-94
            # writes size(T) = rows, cols, 1 as nx, ny, nz and T column-major, i.e. the [z, y, x] volume T.T[None]
            a = np.ascontiguousarray(a.reshape(1, -1).T if a.ndim < 2 else a.T)[None]
        formats.write_em(file_name, a)
    elif ext in ("mrc", "rec", "ali", "mrcs"):
        write_mrc(file_name, obj)
    elif ext in ("txt", "rt", "coords"):
        formats.write_txt(file_name, np.asarray(obj))
    elif ext == "tlt":
        formats.write_txt(file_name, np.asarray(obj, dtype=float).reshape(-1, 1))
    elif ext in ("fsc", "ctf", "xyz"):
        formats.write_fsc(file_name, np.asarray(obj))
    elif ext in ("doc", "m", "sh", "exe", "cmd", "cmm", "bat", "star", "vll", "mfl", "list"):
        if isinstance(obj, str):
            obj = [obj]
        if not all(isinstance(x, str) for x in obj):
            raise NotImplementedError("dwrite of a non-cell object to .%s (evalc disp)" % ext)
        formats.write_lines(file_name, list(obj))
    elif ext == "card":
        formats.write_card(file_name, obj)
    elif ext == "tags":
        formats.write_tags(file_name, obj)
    else:
        raise NotImplementedError("dwrite of .%s files (08b 2)" % ext)
    return True


# =====================================================================================================================
# MATLAB's global random stream (08b section 3)
# =====================================================================================================================

_stream = None


def _rs():
    global _stream
    if _stream is None:
        _stream = np.random.RandomState(5489)       # fresh MATLAB session: twister, seed 0 = init_genrand(5489)
    return _stream


def rng(seed="default"):
    """rng('default') / rng(0): restart the stream as in a fresh MATLAB session. Other seeds: not verified."""
    global _stream
    if seed in ("default", 0):
        _stream = np.random.RandomState(5489)
        return
    raise NotImplementedError("rng(%r): only 'default' / 0 are verified (08b 3)" % (seed,))


def rand(*shape):
    """MATLAB rand(m, n, ...) from the shared stream (values drawn in column-major order)."""
    if len(shape) == 1:
        shape = (shape[0], shape[0])
    n = int(np.prod(shape))
    return _rs().random_sample(n).reshape(shape, order="F")


# =====================================================================================================================
# dtrandomize_azimuth (dynamo_table_randomize_azimuth.m:8-40)
# =====================================================================================================================

def _pairs(args, kw, names):
    """MATLAB-style name/value pairs (also '-name') merged with keyword arguments."""
    if len(args) % 2:
        raise ValueError("name/value arguments must come in pairs")
    out = {}
    for i in range(0, len(args), 2):
        k = args[i]
        if not isinstance(k, str):
            raise ValueError("parameter name expected, got %r" % (k,))
        out[k.lstrip("-")] = args[i + 1]
    for k, v in kw.items():
        out[k.lstrip("-")] = v
    bad = sorted(set(out) - set(names))
    if bad:
        raise NotImplementedError("options %s are not ported" % bad)
    return out


def dtrandomize_azimuth(tIn, *args, **kw):
    """tOut = tIn with column 9 = 360*rand(N,1) (dynamo_table_randomize_azimuth.m:36); 'o' writes tOut."""
    q = _pairs(args, kw, ("o",))
    if isinstance(tIn, str):
        tIn = dread(tIn)                             # dynamo_file_checkin (:22)
    t = np.array(tIn, copy=True)
    if t.ndim != 2 or t.dtype.kind not in "fiuc":
        raise DynamoError("dtrandomize_azimuth: input must be a numeric table")
    if not np.iscomplexobj(t):
        t = t.astype(np.float64)
    t[:, 8] = 360 * rand(t.shape[0], 1)[:, 0]
    if q.get("o"):
        dwrite(t, q["o"])
    return t


# =====================================================================================================================
# daverage (dynamo_average.m, default console path; 08b section 4)
# =====================================================================================================================

class AverageResult(dict):
    """The output struct of daverage (serial field set, 08b 4.3) with attribute access: oa.average, ..."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


_DAVERAGE_OPTIONS = {"table", "t", "tags", "matlab_workers", "mw", "o", "v", "device"}
_DAVERAGE_DEFAULTS = {"fmask": "table", "actions": "align;nan_zero;", "a": None, "nmask": None, "fcompensate": None,
                      "fc": None, "fsc": False, "implicitRotationMasking": 0, "irm": None, "explicitBalanceNan": 1,
                      "ebn": None, "pivoting": False}


def daverage(data, *args, **kw):
    """oa = daverage(data_folder, 't', table, 'mw', N) -> AverageResult.

    The default console path of dynamo_average (08b 4.2): all tags of the table that have a particle file
    (ascending), each particle rotated with dynamo_shift_rot option 4 (mex-exact kernel), NaN -> 0, summed in float64
    in ascending tag order, then divided per voxel by the number of non-NaN contributions. No normalisation, no
    wedge compensation. 'mw' is accepted and has no numerical effect (08b 4.5). device='auto'|'gpu'|'cpu' is a
    pydynamo addition (where to compute; the result is identical)."""
    if len(args) % 2:
        raise ValueError("name/value arguments must come in pairs")
    q = {}
    for i in range(0, len(args), 2):
        q[str(args[i]).lstrip("-")] = args[i + 1]
    q.update({k.lstrip("-"): v for k, v in kw.items()})
    for k, v in list(q.items()):
        if k in _DAVERAGE_OPTIONS:
            continue
        if k in _DAVERAGE_DEFAULTS and (v == _DAVERAGE_DEFAULTS[k] or (v is None) or
                                        (isinstance(v, (list, np.ndarray)) and np.size(v) == 0)):
            continue
        raise NotImplementedError("daverage option %r = %r is not ported (08b 4.1)" % (k, v))
    t0 = time.perf_counter()
    table = q.get("table", q.get("t"))
    if table is None:
        raise DynamoError("daverage: no table given")
    if isinstance(table, str):
        table = dread(table)
    table = np.real(np.asarray(table, dtype=complex if np.iscomplexobj(table) else float))
    if not (isinstance(data, str) and os.path.isdir(data)):
        raise NotImplementedError("daverage with a non-folder data source (accept_generic_input)")
    from pydynamo.vpr import data_info
    facts = data_info(data)
    if not facts["correct"]:
        raise DynamoError("[daverage] fatal error: data facts are not correct for %s" % data)
    tags_req = q.get("tags")
    if tags_req is None or (isinstance(tags_req, str) and tags_req in ("all", "*")) or \
            (not isinstance(tags_req, str) and np.size(tags_req) == 0):
        tags = table[:, 0]
    elif isinstance(tags_req, str):
        raise NotImplementedError("daverage tags=%r (dynamo_table_restrict)" % tags_req)
    else:
        tags = np.ravel(np.asarray(tags_req, dtype=float))
        if np.setdiff1d(tags, table[:, 0]).size:
            print("Warning: tag overflow detected. You were trying to access tags not in the table. Your input tags "
                  "have been readjusted. ")
            tags = np.intersect1d(table[:, 0], tags)
    tags = np.intersect1d(tags, np.asarray(facts["tags"], dtype=float))   # dynamo_average.m:1301, sorted unique
    if tags.size == 0:
        raise DynamoError("daverage: no particle to average")
    sel_rows = [np.nonzero(table[:, 0] == t)[0] for t in tags]
    rep = [int(t) for t, r in zip(tags, sel_rows) if len(r) > 1]
    if rep:
        # dynamo_tag2subtable.m:66-70 returns every row with the tag; dynamo_average.m:2663,2665 then index the
        # multi-row ptable linearly (-ptable(4:6), -ptable(9:-1:7)), mixing columns of different rows: not ported
        raise NotImplementedError("daverage: tags repeated in the table (%s%s): Dynamo indexes the multi-row subtable "
                                  "linearly, which is not ported" % (rep[:5], " ..." if len(rep) > 5 else ""))
    rows = np.stack([table[r[0]] for r in sel_rows])                  # dynamo_tag2subtable: the only row
    file_of = dict(zip(facts["tags"], facts["files"]))
    n = int(facts["l"])
    if list(facts["size"]) != [n, n, n]:
        raise NotImplementedError("non-cubic particles")
    files = [file_of[int(t)] for t in tags]

    def read_particle(i):
        v = formats.read_em(files[i])
        if v.shape != (n, n, n):
            raise NotImplementedError("particle %s has size %s (mixed sizes)" % (files[i], v.shape))
        if np.isnan(v).any():
            raise NotImplementedError("particle %s contains NaN: Dynamo's exclusion path is not ported" % files[i])
        return v.astype(np.float32)

    from pydynamo.averaging.average import console_average_sums
    t1 = time.perf_counter()
    S, C = console_average_sums(read_particle, rows, n, device=q.get("device", "auto"))
    time_loop = time.perf_counter() - t1
    contrib = len(tags) - C                         # average_equalize_NaN (dynamo_average.m:2778-2786)
    contrib[contrib == 0] = 1
    avg = S / contrib
    N = len(tags)
    out = AverageResult(A=np.ones(N - 1), K=np.zeros(N - 1), time_loop=time_loop, tags=tags.copy(),
                        excluded_tags=np.zeros(0), average_raw=avg, nanContributions=C, average=avg.copy(),
                        fmask=np.ones((n, n, n)), fweight=np.ones((n, n, n)), time=time.perf_counter() - t0,
                        dtype="average")
    if q.get("o"):
        o = q["o"]
        d, b = os.path.split(o)
        dwrite(out.average, o)
        dwrite(out.average_raw, os.path.join(d, "raw_" + b))
        dwrite(out.fweight, os.path.join(d, "fweight_" + b))
    if q.get("v"):
        print("[pydynamo] daverage: %d particles averaged in %.1f s (mw %s has no numerical effect); Dynamo's own "
              "verbose messages are not reproduced"
              % (N, out.time, q.get("mw", q.get("matlab_workers", 0))))
    return out
