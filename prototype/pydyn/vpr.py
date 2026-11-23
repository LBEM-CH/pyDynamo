"""Dynamo 1.1.514 virtual projects (settings/virtual_project.mat) and their command-line life cycle, without MATLAB.

Specification: docs/analysis/08a_project_creation_1.1.514.md (written "08a" below). S = the 1.1.514 MATLAB source
(/programs/x86_64-linux/dynamo/1.1.514/matlab/src); every function cites the lines it ports.

Like Dynamo, every function works relative to the current working directory: a project 'p' is the folder ./p,
its vpr is ./p/settings/virtual_project.mat, and the file parameters of the vpr are relative to the cwd.

Python representation of a vpr: a dict in MATLAB field order. Values:
  char    -> str ('' is MATLAB's 0x0 char)
  double  -> float (1x1) or 1-D numpy float64 array (1xN); NaN stays NaN
  logical -> bool
  cell    -> MatCell (a list; 1xN cell)
Files are written with scipy.io.savemat (compressed MAT 5, 08a section 3 [RUN, run C]).

Errors that Dynamo reports by printing and returning an empty object raise DynamoError with Dynamo's message.
Behaviour that is not ported raises NotImplementedError.
"""
import math
import os
import re
import shutil
import struct
import zlib

import numpy as np

from . import convention as _conv
from . import dynio

NC = _conv.NC
RC = _conv.RC
_NC = {p["name"]: p for p in NC}
_RC = {p["name"]: p for p in RC}
# hidden round parameters: dynamo_vpr_convention.m:1538-1541 (in RC order)
HIDDEN_ROUND_NAMES = [p["name"] for p in RC if p["hidden"]]
FEATURE_FIELDS = ["feature_data_sidelength", "feature_template_sidelength", "feature_extension_data",
                  "feature_extension_template", "feature_data_batch", "feature_data_padding",
                  "feature_data_fractioned", "feature_data_type"]      # dynamo_vpr_unfold.m:371-385 (08a 3)


class DynamoError(RuntimeError):
    """Dynamo printed an error and returned an empty object (or raised a MATLAB error)."""


class MatCell(list):
    """A MATLAB 1xN cell array."""


def _say(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


# =====================================================================================================================
# values
# =====================================================================================================================

def default_value(p):
    """Convention default (str / float / 1-D array / bool) of a convention entry."""
    v = p["value"]
    if isinstance(v, tuple):
        return np.array(v, dtype=np.float64)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def _is_empty(v):
    """MATLAB isempty."""
    if isinstance(v, str):
        return v == ""
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    if isinstance(v, np.ndarray):
        return v.size == 0
    return v is None


def _isnumeric(v):
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in "iuf"
    if isinstance(v, (list, tuple)) and not isinstance(v, MatCell):
        return all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) for x in v)
    return False


def _islogical(v):
    return isinstance(v, (bool, np.bool_)) or (isinstance(v, np.ndarray) and v.dtype == bool)


def as_matlab_value(v):
    """Normalise a Python value into the vpr representation (numbers as float / 1-D float64 array)."""
    if isinstance(v, MatCell):
        return MatCell(as_matlab_value(x) for x in v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        a = np.asarray(v)
        if a.dtype == bool:
            return bool(a.ravel()[0]) if a.size == 1 else a.ravel()
        a = a.astype(np.float64).ravel()
        return float(a[0]) if a.size == 1 else a
    raise NotImplementedError("value of type %s in a vpr" % type(v).__name__)


def values_equal(a, b):
    """isequaln for vpr values."""
    if isinstance(a, MatCell) or isinstance(b, MatCell):
        return isinstance(a, MatCell) and isinstance(b, MatCell) and len(a) == len(b) and \
            all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    x, y = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
    return x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))


# =====================================================================================================================
# MAT-file layout (08a section 3)
# =====================================================================================================================

_MCLS = {1: "cell", 2: "struct", 3: "object", 4: "char", 5: "sparse", 6: "double", 7: "single", 8: "int8",
         9: "uint8", 10: "int16", 11: "uint16", 12: "int32", 13: "uint32", 14: "int64", 15: "uint64"}


def _mat_tags(b, off, end):
    while off < end:
        t, n = struct.unpack_from("<II", b, off)
        if t >> 16:                                  # small data element
            n = t >> 16
            t &= 0xffff
            yield t, b[off + 4:off + 4 + n]
            off += 8
        else:
            yield t, b[off + 8:off + 8 + n]
            off += 8 + n + ((8 - n % 8) % 8)


def _mat_matrix(d):
    it = list(_mat_tags(d, 0, len(d)))
    fl = struct.unpack_from("<II", it[0][1])[0]
    dims = list(struct.unpack("<%di" % (len(it[1][1]) // 4), it[1][1]))
    cls = _MCLS[fl & 0xff]
    return {"class": cls + (" logical" if fl & 0x200 else ""), "dims": dims}, it[3:]


def _meta_of(d):
    info, items = _mat_matrix(d)
    if info["class"] == "cell":
        info["cells"] = [_meta_of(dd) for tt, dd in items]
    return info


def mat_fields_meta(path, var="card"):
    """[(field, {'class','dims'[, 'cells']})] of the struct `var` in file order, read from the MAT 5 tags (the
    MATLAB class and logical flag, which scipy does not report)."""
    b = open(path, "rb").read()
    for t, d in _mat_tags(b, 128, len(b)):
        if t == 15:
            d = zlib.decompress(d)
            t, d = next(_mat_tags(d, 0, len(d)))
        it = list(_mat_tags(d, 0, len(d)))           # flags, dims, name, field-name length, names, fields
        if it[2][1].decode() != var:
            continue
        fl = struct.unpack("<i", it[3][1][:4])[0]
        names = it[4][1]
        fns = [names[i:i + fl].split(b"\0")[0].decode() for i in range(0, len(names), fl)]
        return [(fn, _meta_of(dd)) for fn, (tt, dd) in zip(fns, it[5:])]
    raise KeyError("variable %r not in %s" % (var, path))


def _from_loaded(a, meta):
    cls = meta["class"]
    if cls == "cell":
        return MatCell(_from_loaded(x, m) for x, m in zip(np.asarray(a, dtype=object).ravel(order="F"),
                                                          meta["cells"]))
    if cls == "char":
        return "".join(np.asarray(a).ravel().tolist()) if np.asarray(a).size else ""
    if cls.endswith("logical"):
        x = np.asarray(a).astype(bool).ravel()
        return bool(x[0]) if x.size == 1 else x
    if cls in ("double", "single") or cls.startswith("int") or cls.startswith("uint"):
        x = np.asarray(a, dtype=np.float64).ravel(order="F")
        if x.size == 1:
            return float(x[0])
        if sum(d > 1 for d in meta["dims"]) > 1 or (x.size and meta["dims"][0] != 1):
            raise NotImplementedError("vpr field with dims %s (only scalars and row vectors)" % meta["dims"])
        return x
    raise NotImplementedError("MATLAB class %s in a vpr" % cls)


def read_vpr_file(path):
    """virtual_project.mat -> dict (field order and MATLAB classes kept)."""
    import scipy.io as sio
    meta = mat_fields_meta(path)
    m = sio.loadmat(path, squeeze_me=False, struct_as_record=True, mat_dtype=True)
    c = m["card"][0, 0]
    return {f: _from_loaded(c[f], mi) for f, mi in meta}


def _to_mat(v):
    if isinstance(v, MatCell):
        a = np.empty((1, len(v)), dtype=object)
        for i, x in enumerate(v):
            a[0, i] = _to_mat(x)
        return a
    if isinstance(v, str):
        return np.empty((0, 0), dtype="U1") if v == "" else np.array([v])
    if isinstance(v, (bool, np.bool_)):
        return np.array(bool(v), dtype=bool).reshape(1, 1)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.array(float(v), dtype=np.float64).reshape(1, 1)
    a = np.asarray(v)
    if a.dtype == bool:
        return a.reshape(1, -1)
    a = a.astype(np.float64)
    return np.zeros((0, 0)) if a.size == 0 else a.reshape(1, -1)


def write_vpr_file(path, vpr):
    """save(location,'card','-mat') equivalent with scipy (08a section 3, verified in run C).

    A save that would not change the file leaves it as it is: same bytes after the 128-byte MAT header, whose text only
    holds the save date. MATLAB would rewrite it with a new date; skipping that keeps the file time meaningful, since
    the window reads "parameters changed after unfold" from it (re-entering a value, opening the project)."""
    import io
    import scipy.io as sio
    card = {k: _to_mat(v) for k, v in vpr.items()}
    buf = io.BytesIO()
    sio.savemat(buf, {"card": card}, do_compression=True, oned_as="row", format="5", long_field_names=True)
    data = buf.getvalue()
    try:
        with open(path, "rb") as f:
            old = f.read()
        if len(old) == len(data) and old[128:] == data[128:]:
            return
    except OSError:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def dump_vpr(path):
    """[(name, class, dims, value)] for comparisons (class/dims from the MAT tags, value as loaded)."""
    meta = mat_fields_meta(path)
    vals = read_vpr_file(path)
    return [(f, m["class"], m["dims"], vals[f]) for f, m in meta]


# =====================================================================================================================
# locations (dynamo_database_locate, dynamo_database_convention.m:416-534)
# =====================================================================================================================

def loc(kind, name, ref=None, ite=None):
    """Dynamo database string of a project file, './<name>/...' (06c A.1)."""
    b = "./%s/" % name
    R = "_ref_%03d" % ref if ref is not None else ""
    table = {
        "virtual_project": "settings/virtual_project.mat",
        "next_iteration": "next_iteration.txt",
        "mask": "settings/mask.em",
        "mask_classification": "settings/mask_classification.em",
        "mask_smoothing": "settings/mask_smoothing.em",
        "template_initial": "settings/template_initial%s.em" % R,
        "table_initial": "settings/table_initial%s.tbl" % R,
        "fmask_initial": "settings/fmask_initial%s.em" % R,
    }
    if kind in table:
        return b + table[kind]
    if kind in ("starting_reference", "starting_table", "fmask_starting_reference"):
        ext = "tbl" if kind == "starting_table" else "em"
        return b + "results/ite_%04d/starting_values/%s%s_ite_%04d.%s" % (ite, kind, R, ite, ext)
    if kind == "card_ite":
        return b + "cards/ite_%04d/card_ite_ite_%04d.card" % (ite, ite)
    raise NotImplementedError("database location %r" % kind)


def vpr_location(name, path=None):
    """dynamo_database_locate('virtual_project', name, 'path', path): <path>/<name>/settings/virtual_project.mat."""
    base = path if path else "."
    return os.path.join(base, name, "settings", "virtual_project.mat")


def isproject(name):
    """dynamo_isproject.m: <name>/settings/virtual_project.mat exists."""
    if not isinstance(name, str) or not name:
        return False
    return os.path.isfile(os.path.join(name, "settings", "virtual_project.mat"))


# =====================================================================================================================
# creation, defaults, save, load
# =====================================================================================================================

def blank(name):
    """dynamo_vpr.m:598-729 (no options): name_project, path = pwd, the other globals, then <p>_r1.._r8 for every
    round parameter, all at the convention defaults (468 fields, 08a 2.2)."""
    vpr = {"name_project": name}
    for p in NC:                                     # the loop sets path = pwd before and after its own entry
        if p["name"] == "name_project":
            vpr["path"] = os.getcwd()
            continue
        vpr[p["name"]] = default_value(p)
        if p["name"] != "path":
            vpr["path"] = os.getcwd()
    for p in RC:
        for r in range(1, 9):
            vpr["%s_r%d" % (p["name"], r)] = default_value(p)
    return vpr


_CAN_BE_EMPTY = ("path", "options", "MCR_CACHE_ROOT", "fCompensationSmoothingMask")


def correct_defaults(vpr, verbose=True):
    """dynamo_vpr_correct_defaults.m:43-102."""
    for p in NC:
        n = p["name"]
        if n not in vpr:
            _say('Initializing field "%s" in virtual project' % n, verbose)
            vpr[n] = os.getcwd() if n == "path" else default_value(p)
        if _is_empty(vpr[n]) and n not in _CAN_BE_EMPTY:
            vpr[n] = default_value(p)
            from .dcpcontrol import format_str              # dynamo__format_str (:63-64)
            _say('Empty parameter "%s", initializing it to default value "%s" ' % (n, format_str(vpr[n])), verbose)
    for p in RC:
        for r in range(1, 9):
            f = "%s_r%d" % (p["name"], r)
            if f not in vpr:
                _say('Initializing field "%s" ' % f, verbose)
                vpr[f] = default_value(p)
            if _is_empty(vpr[f]) and "plugin" not in p["name"] and "order" not in p["name"] and p["name"] != "flags":
                _say("Giving default value to empty field %s " % f, verbose)
                vpr[f] = default_value(p)
    if vpr.get("path") == "current directory":
        vpr["path"] = os.getcwd()
    return vpr


def save(vpr, location="./"):
    """dynamo_vpr_save.m:60-90: correct_defaults, then save into <location>/<name>/settings/virtual_project.mat."""
    vpr = correct_defaults(vpr)
    f = vpr_location(vpr["name_project"], location)
    write_vpr_file(f, vpr)
    return vpr


def add_missing_convention_parameters(vpr):
    """+dpkproject/+vpr/addMissingConventionParameters.m (in memory only)."""
    for p in NC:
        if p["name"] not in vpr:
            vpr[p["name"]] = default_value(p)
    return vpr


def load(name, path=None, field=None, verbose=True):
    """dynamo_vpr_load.m:106-152 (dvload). Returns None (and prints Dynamo's warning) if the file is missing.
    field: one field; a generic round name gives the MatCell of its 8 rounds."""
    f = vpr_location(name, path)
    if not os.path.isfile(f):
        _say("WARNING: file %s could not be identified for project:  %s.\rReturning empty object." % (f, name), verbose)
        return None
    if path:
        raise NotImplementedError("dynamo_vpr_load 'path' option (dynamo_vpr_absolute_path) is not ported")
    vpr = add_missing_convention_parameters(read_vpr_file(f))
    if field:
        if field in vpr:
            return vpr[field]
        if "%s_r1" % field in vpr:
            return MatCell(vpr["%s_r%d" % (field, i)] for i in range(1, 9))
        _say("WARNING: field of type %s could not be identified in the project:  %s.\rReturning empty object."
             % (field, name))
        return None
    return vpr


dvload = load


# =====================================================================================================================
# dynamo_vpr_modify (dynamo_vpr_modify.m)
# =====================================================================================================================

def _find_parameter_type(name):
    """dynamo_vpr_modify.m find_parameter_type: 'global' | 'round_general' | 'round_single' | 'invalid'."""
    if name in _NC:
        return "global", _NC[name]["type"]
    if name in _RC:
        return "round_general", _RC[name]["type"]
    root = name[:-3]
    if root in _RC and name[-3:-1] == "_r" and name[-1].isdigit() and 0 < int(name[-1]) < 9:
        return "round_single", _RC[root]["type"]
    return "invalid", None


def _check_format(input_type, name, value):
    """dynamo_vpr_modify.m check_format (:419-446)."""
    if input_type in (0, 3) and not isinstance(value, str):
        raise DynamoError(' Error: parameter "%s" should be a string' % name)
    if input_type == 1 and not _isnumeric(value):
        raise DynamoError(' Error: parameter "%s" should be numeric' % name)
    if input_type == 2 and not (_isnumeric(value) or _islogical(value)):
        raise DynamoError(' Error: parameter "%s" should be one or zero' % name)


_SHORTHANDS = [("table", "file_table_initial"), ("mask", "file_mask"), ("data", "folder_data"),
               ("fmask", "file_fmask_initial"), ("cmask", "file_mask_classification"),
               ("template", "file_template_initial")]


def round2all(vpr):
    """dynamo__vpr_modify_round2all.m:42-105: plugin_<k>_order_all, then <hidden>_all, as 1x8 cells."""
    for kind in ("align", "post", "iter"):
        vals = MatCell(vpr["plugin_%s_order_r%d" % (kind, r)] for r in range(1, 9)
                       if "plugin_%s_order_r%d" % (kind, r) in vpr)
        vpr["plugin_%s_order_all" % kind] = vals
    for kind in HIDDEN_ROUND_NAMES:
        vals = MatCell(vpr["%s_r%d" % (kind, r)] for r in range(1, 9) if "%s_r%d" % (kind, r) in vpr)
        vpr["%s_all" % kind] = vals
    return vpr


def modify(vpr, *pairs):
    """dynamo_vpr_modify(vpr, p1, v1, ...) for plain name/value pairs (dynamo_vpr_modify.m:150-316).
    Shorthands table/mask/data/fmask/cmask/template; a generic round name takes a list/MatCell of per-round values;
    ends with round2all. Returns a new dict."""
    if vpr is None:
        raise DynamoError("warning: empty project. Aborting dynamo_vpr_modify.")
    if pairs and isinstance(pairs[0], str) and pairs[0] in ("disk", "disc", "check"):
        raise NotImplementedError("dynamo_vpr_modify 'disk'/'check' modes (use dvput)")
    if len(pairs) % 2:
        raise DynamoError("odd number of arguments")
    params = list(pairs)
    for short, legal in _SHORTHANDS:                 # replace_shorthand: only the first occurrence, odd positions
        # string comparison only: a numeric value (e.g. a numpy array for gpu_identifier_set) never equals a name
        # (a list `in` test would compare arrays elementwise and raise)
        hits = [j for j, x in enumerate(params) if isinstance(x, str) and x == short]
        if hits:
            i = hits[0]
            if i % 2 == 0:
                params[i] = legal
                val = params[i + 1]
                if short != "data" and _isnumeric(val):
                    raise NotImplementedError("numeric %s passed to dynamo_vpr_modify (written to worspace_*)" % short)
    new = dict(vpr)
    for i in range(0, len(params), 2):
        name, value = params[i], params[i + 1]
        if not isinstance(name, str):
            raise DynamoError("ERROR: position %d of the input should be a parameter name, but it is not a string."
                              % (i + 2))
        kind, itype = _find_parameter_type(name)
        if kind == "invalid":
            raise DynamoError('ERROR: position %d of the input should be a parameter name, but "%s" is not a valid '
                              'parameter ' % (i + 2, name))
        if kind in ("global", "round_single"):
            _check_format(itype, name, value)
            if "plugin" in name and "order" in name:
                value = value.replace('"', "'")
            new[name] = as_matlab_value(value)
        else:
            if not isinstance(value, (list, tuple, MatCell)):
                raise DynamoError('ERROR: position %d in the input is the general round parameter "%s". Its '
                                  'arguments should be a cellarray, each entry representing the values for a round'
                                  % (i + 2, name))
            for r, vr in enumerate(value, 1):
                fr = "%s_r%d" % (name, r)
                _check_format(itype, fr, vr)
                if "plugin" in fr and "order" in fr:
                    vr = vr.replace('"', "'")
                new[fr] = as_matlab_value(vr)
    return round2all(new)


# =====================================================================================================================
# data folder facts (dynamo_data_info.m -> dynamo_info_data.m, classic 'dynamo_folder' route)
# =====================================================================================================================

def data_info(folder, volume_type="particle"):
    r"""dynamo_info_data.m:186-420 for a Dynamo data folder: tags (ascending), N, padding, l / size / extension of the
    sample particle (the largest tag). Padding (:385-409): files are scanned in name order (dynamo_regexp2files,
    ASCII order); until a padding is found, each file is tried with 7 down to 2 digits; the first match fixes the
    padding and the pattern particle_\d{padding}, which is then used for every later file (a file of another width,
    or a stray particle_* name, is skipped and does not change the padding). [RUN] review/reference/logs/
    e_review_checks.log: {0000004, 000001, 000002, 00003, abc} -> padding 7, tags [4]; {000001, 000002, zz} ->
    padding 6, tags [1 2]; {000010, 00009} -> padding 6, tags [10]."""
    facts = {"folder_name": folder, "correct": 0, "N": 0, "tags": [], "l": float("nan"), "padding": float("nan"),
             "extension": "UNKNOWN", "source_type": "dynamo_folder", "files": [], "asproject": False}
    if not isinstance(folder, str) or folder == "":
        return facts
    if os.path.isfile(folder):
        raise NotImplementedError("data source %r is a file (.sel / particle list / dBoxes routes not ported)" % folder)
    if isproject(folder):
        raise NotImplementedError("data source %r is a project (dynamo_info_data project route not ported)" % folder)
    if "*" in folder or "?" in folder:
        raise NotImplementedError("regexp data sources are not ported")
    if not os.path.isdir(folder):
        facts["exist"] = 0
        return facts
    facts["exist"] = 1
    import glob
    all_files = sorted(glob.glob(os.path.join(folder, volume_type + "_*")))
    files, tags, padding = [], [], None
    for f in all_files:
        root, ext = os.path.splitext(os.path.basename(f))
        if padding is not None:                                       # :387-388, fixed particle_file_format
            m = re.search(r"%s_\d{%d}" % (volume_type, padding), root)
        else:
            m = None
            for pad in range(7, 1, -1):                               # :390-401
                m = re.search(r"%s_\d{%d}" % (volume_type, pad), root)
                if m:
                    padding = pad                                     # :407
                    break
        if m and m.group(0) == root:                                  # :411-418
            files.append(f)
            tags.append(int(root[len(volume_type) + 1:]))
    order = np.argsort(tags, kind="stable")
    tags = [tags[i] for i in order]
    files = [files[i] for i in order]
    facts["tags"], facts["files"], facts["N"] = tags, files, len(files)
    facts["padding"] = padding if padding is not None else []
    if not files:
        facts["empty"] = 1
        return facts
    facts["correct"] = 1
    sample = files[-1]
    ext = os.path.splitext(sample)[1].lstrip(".")
    if ext != "em":
        raise NotImplementedError("particle extension .%s (only .em is ported)" % ext)
    with open(sample, "rb") as fh:
        h = fh.read(16)
    if len(h) < 16:
        facts["correct"] = 0
        return facts
    end = {0: ">", 3: ">", 5: ">", 6: "<"}.get(h[0], "<")
    nx, ny, nz = (int(x) for x in np.frombuffer(h[4:16], end + "u4"))
    facts["l"], facts["size"], facts["extension"] = nx, [nx, ny, nz], ext
    return facts


# =====================================================================================================================
# default masks (dynamo_vpr_hint_files.m)
# =====================================================================================================================

def _ellipsoid_default(l):
    """dynamo_ellipsoid(l/2-1, l, (l/2+1)*[1 1 1]) == pydyn.masks.default_mask (byte-checked, 08a 2.3)."""
    if l % 2:
        raise NotImplementedError("odd sidelength %d: the sphere centre of dynamo_ellipsoid is not verified" % l)
    from .masks import default_mask
    return default_mask(int(l)).astype(np.float64)


_HINT_MUTATIONS = [(("file_mask", "mask"), "mask"),
                   (("mask_classification", "file_mask_classification", "cmask"), "mask_classification"),
                   (("template_initial", "template", "file_template_initial"), "template_initial"),
                   (("table", "table_initial", "file_table_initial"), "table_initial"),
                   (("fmask", "fmask_initial", "file_fmask_initial"), "fmask_initial"),
                   (("file_mask_smoothing", "smask", "mask_smoothing"), "mask_smoothing")]


def hint_files(vpr, items, verbose=True):
    """dynamo_vpr_hint_files(vpr, 'item', items) for the mask items (dynamo_vpr_hint_files.m:200-548)."""
    chosen = []
    for it in items:
        for muts, root in _HINT_MUTATIONS:
            if it in muts:
                chosen.append(root)
    if not chosen:
        _say("[vpr_hint_files] Warning: No items to generate: returning.", verbose)
        return vpr
    if "template_initial" in chosen or "table_initial" in chosen:
        raise NotImplementedError("dynamo_vpr_hint_files for template/table (dynamo_initial_template, "
                                  "dynamo_table_blank) is not ported")
    facts = data_info(vpr["folder_data"])
    if not facts["correct"]:
        _say("Data cannot be found, aborting", verbose)
        raise DynamoError("Incorrect data in project!!!")
    l = facts["l"]
    nref = int(vpr["nref_r1"])
    _say("Using %d reference(s)" % nref, verbose)
    bars = "*" * 60
    name = vpr["name_project"]
    if "mask" in chosen:
        _say(bars, verbose)
        _say("Generating alignment mask (sidelength %d)" % l, verbose)
        vpr = _archive(vpr, _ellipsoid_default(l), loc("mask", name), "file_mask", verbose)
    _say(bars, verbose)
    if "mask_classification" in chosen:
        _say("Generating classification mask (sidelength %d)" % l, verbose)
        vpr = _archive(vpr, _ellipsoid_default(l), loc("mask_classification", name), "file_mask_classification",
                       verbose)
    if "fmask_initial" in chosen:
        _say(bars, verbose)
        _say("Generating fmask_initial (sidelength %d)" % l, verbose)
        if nref != 1:
            raise NotImplementedError("default fmasks for nref > 1 (no ground truth, 08a 11.8)")
        fm = float(l) * np.ones((l, l, l))                       # dynamo_vpr_hint_files.m:474-485
        vpr = _multireference_setfiles(vpr, [fm], "fmask", verbose)
    if "mask_smoothing" in chosen:
        _say(bars, verbose)
        _say("Generating smoothing mask (sidelength %d)" % l, verbose)
        vpr = _archive(vpr, np.ones((l, l, l)), loc("mask_smoothing", name), "file_mask_smoothing", verbose)
    vpr = save(vpr)
    _say(bars, verbose)
    _say(" ", verbose)
    _say("Done. Files have been created and linked into the virtual project", verbose)
    return vpr


def _archive(vpr, obj, target, parameter, verbose):
    """dynamo_vpr_hint_files.m archive_settings: dynamo_write, then dynamo_vpr_modify(parameter, target)."""
    dynio.write_em(target, obj)
    _say("writing %s" % target, verbose)
    return modify(vpr, parameter, target)


def _multireference_setfiles(vpr, objects, item, verbose):
    """dynamo_vpr_multireference_setfiles.m + dynamo_write_multireference.m:448-467 for one reference."""
    name = vpr["name_project"]
    folder = "./%s/settings" % name
    names = []
    for i, o in enumerate(objects, 1):
        f = "%s/%s_initial_ref_%03d.em" % (folder, item, i)
        dynio.write_em(f, o)
        _say("[write_multireference] Reference #%d.  File written: %s " % (i, f), verbose)
        names.append(f)
    dynio.write_lines("%s/multisettings_%s_initial.sel" % (folder, item), ["%s 1" % n for n in names])
    return modify(vpr, {"fmask": "file_fmask_initial", "template": "file_template_initial",
                        "table": "file_table_initial"}[item], names[0])


# =====================================================================================================================
# dvput (dynamo_vpr_put.m)
# =====================================================================================================================

def _shortform2form(name):
    """dynamo__shortform2form_ifshortform.m: short names (cr, cs, ...) to long names, keeping a _r<k> suffix."""
    m = re.fullmatch(r"(\S*)(_r\d)", name)
    root, suffix = (m.group(1), m.group(2)) if m else (name, "")
    for p in NC + RC:
        if p["short"] == root:
            return p["name"] + suffix
    return name


def _parameter_kind(name):
    """dynamo_parameter.m .kind: later matches override earlier ones, as in the source order."""
    kind = None
    if name in _NC or name in [p["short"] for p in NC]:
        kind = "global"
    if name in _RC or name in [p["short"] for p in RC]:
        kind = "generic_round"
    m = re.fullmatch(r"(.+)_r([1-8])", name)
    if m and (m.group(1) in _RC or m.group(1) in [p["short"] for p in RC]):
        kind = "round"
    return kind


# names that MATLAB's str2num (eval of '[' s ']') evaluates to numbers; a value containing one of them, or ':' / ';'
# (ranges, matrices), is not ported (NotImplementedError) instead of being kept as text
_MATLAB_NUMERIC_NAMES = {"pi", "i", "j", "eps", "inf", "Inf", "nan", "NaN", "NA", "true", "false", "realmax",
                         "realmin", "intmax", "intmin", "flintmax", "rand", "randn", "randi", "now", "clock", "tic",
                         "toc", "ones", "zeros", "eye", "magic", "date", "cputime", "end"}


def _str2num_ifnum(v):
    """str2num_ifnum (S/str2num_ifnum.m, called by dynamo__system_parse_char_arguments_EVEN): object = str2num(v),
    and v itself when that is empty. pydyn evaluates with the restricted dcpcontrol.str2num (numbers, Inf/NaN,
    true/false, vectors, integer a:b:c); text that MATLAB would evaluate otherwise raises NotImplementedError, plain
    text (paths, names) stays text. A '|' (dynamo__parse_bar2cell_ifbar, dynamo_vpr_put.m:439) is not ported."""
    if not isinstance(v, str) or v == "":
        return v
    s = v.strip()
    if "{" in s:
        raise NotImplementedError("cell syntax in a dvput string argument")
    if "|" in s:
        raise NotImplementedError("'|' in a dvput string argument (dynamo__parse_bar2cell_ifbar) is not ported")
    from .dcpcontrol import str2num
    try:
        x = str2num(s)
    except NotImplementedError:
        if ":" in s or ";" in s or any(t in _MATLAB_NUMERIC_NAMES for t in re.findall(r"[A-Za-z_]\w*", s)):
            raise NotImplementedError("dvput string value %r: MATLAB's str2num would evaluate it; not ported" % v)
        return v
    if isinstance(x, bool):
        return x
    if x.size == 0:                                  # str2num('[]') = [] -> the text is kept
        return v
    return float(x[0]) if x.size == 1 else x


def put(project, option, *args, verbose=True):
    """dvput(project, option, p1, v1, ...) = dynamo_vpr_put (dynamo_vpr_put.m:165-650).

    option: 'd'/'disk'/'disc' (save), 'c'/'check', 'cd', 'u'/'unfold', 'cu'. With an odd number of trailing
    arguments the option is taken as the first parameter name and 'disk' is assumed (:214-225).
    Generic round names get '_r1'; a numeric vector or list for a generic round name expands to _r1.._rn.
    Saves into vpr['path'] (:602-605), as Dynamo does: see 08a 5.2.
    Returns the modified vpr."""
    if isinstance(option, str) and "." in option:   # parameter file (dynamo_vpr_put.m:193-211), before the swap
        if not os.path.isfile(option):
            raise DynamoError("Invalid parameter file, cannot update project")
        raise NotImplementedError("dvput with a parameter file (dpkproject.aux.input.fromCard)")
    args = list(args)
    if len(args) % 2 == 1:
        _say('Assuming intended "option" [second argument] is "disk" (changes will be save in disk).', verbose)
        args = [option] + args
        option = "disk"
    if not args:
        raise DynamoError("Syntax error! You didn't specify a new value!")
    if isinstance(args[0], str):
        if args[0] in ("seeds", "-seeds"):
            v = None
            for p in ("template", "table", "fmask"):
                v = put(project, option, p, args[1], verbose=verbose)
            return v
        if args[0] in ("cm", "-cm", "cmask", "-cmask"):
            args[0] = "file_mask_classification"
    flags = {"cu": (1, 1, 1), "cd": (1, 1, 0), "check": (1, 0, 0), "c": (1, 0, 0), "unfold": (0, 1, 1),
             "u": (0, 1, 1), "disk": (0, 1, 0), "disc": (0, 1, 0), "d": (0, 1, 0)}
    if option not in flags:
        check = disk = unfold_ = 0
    else:
        check, disk, unfold_ = flags[option]
    if not isproject(project):
        _say("INPUT ERROR", verbose)
        raise DynamoError('[standalone vpr_modify] Your input project  "%s" can not be found, aborting '
                          'dynamos_vpr_modify.' % project)
    vpr = load(project, verbose=verbose)
    # dynamo__system_parse_char_arguments_EVEN: strip a leading '-', convert numeric strings
    mods = []
    for i in range(0, len(args), 2):
        n = args[i]
        if not isinstance(n, str):
            raise DynamoError("parameter names must be strings")
        mods += [n[1:] if n.startswith("-") else n, _str2num_ifnum(args[i + 1])]
    if "ref" in mods[0::2]:
        raise NotImplementedError("dvput 'ref' option (dynamo_dbmultiput)")
    inround = 0
    if "inround" in mods[0::2]:
        j = mods[0::2].index("inround") * 2
        inround = int(mods[j + 1])
        del mods[j:j + 2]
    if inround == 0:                                 # :433-503 expansion (Python list/tuple/array = numeric vector,
        extra = []                                   # MatCell = cell)
        for i in range(0, len(mods), 2):
            n, v = mods[i], mods[i + 1]
            expandable = (_isnumeric(v) and np.size(v) > 1 and _parameter_kind(n) == "generic_round") or \
                isinstance(v, MatCell)
            if expandable:
                vals = list(v) if isinstance(v, MatCell) else list(np.ravel(np.asarray(v, dtype=float)))
                mods[i], mods[i + 1] = "%s_r1" % n, vals[0]
                for r in range(2, len(vals) + 1):
                    extra += ["%s_r%d" % (n, r), vals[r - 1]]
        mods = mods + extra
    for i in range(0, len(mods), 2):
        n = _shortform2form(mods[i])
        if inround > 0 and n in _RC:
            n = "%s_r%d" % (n, inround)
        if n in _RC:
            n = n + "_r1"
        valid = n in _NC or _find_parameter_type(n)[0] == "round_single"
        _say('[modify]  flag #%d: "%s" (%s) ' % (i // 2 + 1, n, "valid project parameter" if n in _NC else
                                                 "valid round project parameter" if valid else
                                                 "ERROR: unknown parameter"), verbose)
        if not valid:
            _say("Aborting because of syntax error. Use vpr_help for a list of legal parameters.", verbose)
            raise DynamoError("NO MODIFICATIONS HAVE BEEN OPERATED.")
        mods[i] = n
    new = modify(vpr, *mods)
    if check:
        from .vprcheck import check as _check
        rep = _check(new, verbosity=1)
        if rep.fields_error:
            _say("[standalone modify] Check not passed: aborting", verbose)
            return new
    if disk:
        _save_into_vpr_path(new)
        _say("modification saved in disk.", verbose)
    if unfold_:
        if str(new["destination"]).startswith("mpi"):
            raise NotImplementedError("unfolding MPI projects")
        unfold(new)                                  # dynamo_vpr_put.m:632 dynamo_vpr_unfold(vpr_new)
    return new


def _save_into_vpr_path(vpr):
    """dynamo_vpr_save(vpr_new, vpr_new.path) (dynamo_vpr_put.m:602-605). Dynamo would create
    <path>/<name>/settings if missing (dynamo_filename_mkdir); pydyn refuses to create a project outside the current
    tree that way and raises instead (08a 5.2: for a copied project, vpr.path is the original author's folder)."""
    target = vpr_location(vpr["name_project"], vpr["path"])
    if not os.path.isfile(target):
        raise DynamoError("dvput saves into vpr.path (dynamo_vpr_put.m:602-605): %s does not exist. Dynamo would "
                          "create it; pydyn refuses. Use dcp.put (saves into ./) or set 'path' first." % target)
    save(vpr, vpr["path"])


dvput = put


# =====================================================================================================================
# dvunfold (dynamo_vpr_unfold.m)
# =====================================================================================================================

def _copyfile(src, dst, read_and_write_if_ext_mismatch=False, log=None):
    """dynamo_copyfile.m:73-201. MATLAB copyfile keeps the mtime (copy2). Deviation (08a 6.1, 11.2): a symlinked
    source is copied by content (Dynamo copies the link itself and later writes through it)."""
    if not os.path.isfile(src):
        print("Sorry, file %s could not be located. Desisting copyfile for this one" % src)
        raise DynamoError("Aborting: copyfile error")
    if src == dst:                                   # strcmp of the strings (:117-123)
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if read_and_write_if_ext_mismatch and os.path.splitext(src)[1] != os.path.splitext(dst)[1]:
        from .console import dread, dwrite
        dwrite(dread(src), dst)
        return
    if os.path.exists(dst) and os.path.samefile(src, dst):
        # one file under two spellings (p/settings/mask.em, ./p/settings/mask.em, an absolute path): with output
        # arguments MATLAB's copyfile does not throw, it returns success 0 and its message (MCR 9.6
        # resources/MATLAB/en/FILE.xml, SourceAndDestinationSame); Dynamo prints them and goes on (:170-177)
        # [UNC: that copyfile resolves both spellings to one file is not in the source; to confirm in a run]
        print("[dcopyfile] Error copying %s into %s" % (src, dst))
        print("Cannot copy or move a file or directory onto itself.")
        print(" ")
        return
    if os.path.islink(src):
        print("[pydyn] %s is a symbolic link: copying its content (Dynamo 1.1.514 would copy the link)" % src)
    shutil.copy2(src, dst)


def _links2settings(vpr):
    """dynamo_pipeline_links2settings.m:73-210 (single and multi reference). Both copy blocks always run; each stops
    at its first missing file (enter_file_array_in_settings :129-135, ..._multireference_single :152-158). Returns
    error_fields = unique(cell_stretch(catcells(...))) (:111-112): the {link field, database name} pairs (or the
    items of the multireference branch), flattened and sorted."""
    name = vpr["name_project"]

    def copy_block(pairs, ref):
        for field, kind in pairs:
            src = vpr[field]
            dst = loc(kind, name, ref=ref) if ref else loc(kind, name)
            if not os.path.isfile(src):
                print(" Error importing file %s into database location %s. File not found" % (src, dst))
                return [field, kind]
            _copyfile(src, dst, True)
        return []

    errors = copy_block([("file_mask", "mask"), ("file_mask_classification", "mask_classification"),
                         ("file_mask_smoothing", "mask_smoothing")], None)
    if int(vpr["nref_r1"]) != 1:
        # enter_file_array_in_settings_multireference_multiple (dynamo_pipeline_links2settings.m:166-205, 09a 4.2):
        # per item, the files of dynamo_check_initial_folders go through dynamo_vpr_multireference_setfiles (a copy
        # onto itself when the folder is ./<name>/settings); the returned vpr is discarded.
        from .vpreo import check_initial_folders, multireference_setfiles
        for item in ("template", "table", "fmask"):
            field = "file_%s_initial" % item
            files, rep = check_initial_folders(vpr[field], field, int(vpr["nref_r1"]))
            if rep == "ok":
                multireference_setfiles(vpr, files, item)
            else:
                errors.append(item)
                print("Error checking initial folder")
    else:
        errors += copy_block([("file_template_initial", "template_initial"), ("file_table_initial", "table_initial"),
                              ("file_fmask_initial", "fmask_initial")], 1)
    return sorted(set(errors))


def _settings2first_references(vpr):
    """dynamo_pipeline_settings2first_references.m:45-84."""
    name, nref = vpr["name_project"], int(vpr["nref_r1"])
    for iref in range(1, nref + 1):
        f = loc("table_initial", name, ref=iref)
        T = dynio.read_tbl(f)
        if T.shape[1] < 34:
            T = np.concatenate([T, np.zeros((T.shape[0], 34 - T.shape[1]), dtype=T.dtype)], axis=1)
        T[:, 33] = iref
        dynio.write_tbl(f, T)
    for s, r in (("template_initial", "starting_reference"), ("table_initial", "starting_table"),
                 ("fmask_initial", "fmask_starting_reference")):
        for ref in range(1, nref + 1):
            src, dst = loc(s, name, ref=ref), loc(r, name, ref=ref, ite=1)
            if os.path.splitext(src)[1] == os.path.splitext(dst)[1]:
                _copyfile(src, dst)
            else:
                from .console import dread, dwrite
                dwrite(dread(src), dst)


def template_and_data_features(vpr):
    """+dpkproject/templateAndDataFeatures.m:49-105."""
    s = {"data_sidelength": float("nan"), "template_sidelength": float("nan"), "extension_data": "UNKNOWN",
         "extension_template": "UNKNOWN", "data_fractioned": float("nan"), "data_batch": float("nan"),
         "data_padding": float("nan"), "ok": True}
    facts = data_info(vpr["folder_data"])
    s["data_sidelength"], s["extension_data"], s["data_padding"] = facts["l"], facts["extension"], facts["padding"]
    s["data_type"] = "dynamo_folder"
    t = vpr["file_template_initial"]
    if int(vpr["nref_r1"]) != 1:                    # linked template of ref 1 (check_initial_folders, 09a 4.2)
        from .vpreo import linked_files
        t = linked_files(vpr, "file_template_initial", [1])[0] or ""
    if not os.path.isfile(t):
        s["ok"] = False
        return s
    from .console import dread
    v = dread(t)
    s["template_sidelength"] = v.shape[-1]          # size(template,1) = nx ([z, y, x] arrays)
    s["extension_template"] = os.path.splitext(t)[1].lstrip(".")
    return s


def execution_script_lines(vpr):
    """Lines of <name>.exe, byte for byte what Dynamo 1.1.514 writes (+dpkproject/+executionScript/createForStandalone.m:
    5 comment lines and 'dynamo_execute_project <name>'). With pyDynamo's environment active the `dynamo_execute_project`
    command (console script pydyn.cli:execute_main) runs pyDynamo; without it, the command is Dynamo's."""
    name = vpr["name_project"]
    return ["# ", "# Just source this script ", "# or write %s.sh on your shell" % name,
            "#  You may need to chmod the script to give yourself execution permission", " ",
            "dynamo_execute_project %s" % name]


def execution_script(vpr):
    """dynamo_vpr_execution_script.m + +dpkproject/executionScript.m: the file name and content by destination."""
    name, dest = vpr["name_project"], vpr["destination"]
    if dest in ("matlab", "matlab_gpu", "matlab_parfor"):
        f = "%s.m" % name                            # createForMatlab.m:6-13
        dynio.write_lines(f, ["% ", "% Just type this command in the matlab window: ", "% ", " ",
                              "dynamo_execute_project %s" % name])
        print("[pydyn] destination %r: wrote Dynamo's MATLAB script %s; pydyn executes only standalone_gpu "
              "projects" % (dest, f))
        return f
    if dest in ("standalone", "system_gpu", "standalone_gpu"):
        f = "%s.exe" % name
        dynio.write_lines(f, execution_script_lines(vpr))
        return f
    raise NotImplementedError("execution script for destination %r (dpkproject.executionScript.oldStyle)" % dest)


def unfold(name, next=1, runfrom=None, verbose=True, **project_flags):
    """dvunfold(name) = dynamo_vpr_unfold (dynamo_vpr_unfold.m:117-594; 08a section 6, steps 1-15).
    Returns (name_script, status) like Dynamo ('ok' / 'error')."""
    from .dynproject import card_ite_from_vpr, round_array
    if project_flags:
        raise NotImplementedError("dvunfold with project flags / options %s" % sorted(project_flags))
    vpr = load(name, verbose=False) if isinstance(name, str) else dict(name)
    if vpr is None:
        print('[vunfold] Invalid project name: "%s"' % name)
        print("         Aborting project unfolding.")
        return None, "error"
    # step 3 (:283-306)
    if "gpu" in str(vpr["destination"]):
        if float(vpr.get("systemUsingProcessorTables", 0)):
            _say('Warning: gpu-related projects cannot use "processor tables". Changing default setting.', verbose)
        vpr["systemUsingProcessorTables"] = 0.0
    else:
        if not float(vpr.get("systemUsingProcessorTables", 1)):
            _say('Warning: NON gpu-related projects should use "processor tables". Changing default setting.',
                 verbose)
        vpr["systemUsingProcessorTables"] = 1.0
    vpr = save(vpr)
    # step 4: unfoldCheckAdaptiveBandpassInitial (+dpkproject/unfoldCheckAdaptiveBandpassInitial.m:8-23)
    # o.echo(fmt, v) = disp(sprintf(fmt, v)) whatever the unfold verbosity (+pparse/@output/output.m:108-150, :206-213)
    if float(vpr["adaptive_bandpass"]) and float(vpr["adaptive_bandpass_initial"]) != float(vpr["low_r1"]):
        print(" [adaptive filtering] Overruling current value for low pass in first iteration: %s"
              % dynio._matlab_d(vpr["low_r1"]), flush=True)                                      # :11
        vpr["low_r1"] = float(vpr["adaptive_bandpass_initial"])
        print(" [adaptive filtering] Using initial lowpass: %s" % dynio._matlab_d(vpr["low_r1"]), flush=True)  # :13
    if float(vpr["adaptive_bandpass_even_odd"]) and float(vpr["adaptive_bandpass_initial"]) != float(vpr["low_r1"]):
        print(" [adaptive filtering eo] Overruling current value for low pass first iteration: %s"
              % dynio._matlab_d(vpr["low_r1"]), flush=True)                                      # :21
        raise DynamoError("Undefined function or variable 'adaptive_bandpass_initial' "
                          "(+dpkproject/unfoldCheckAdaptiveBandpassInitial.m:23, a Dynamo 1.1.514 bug)")
    # step 5
    dynio.write_txt(loc("next_iteration", vpr["name_project"]), next)
    # step 6
    s = template_and_data_features(vpr)
    for k, f in zip(("data_sidelength", "template_sidelength", "extension_data", "extension_template",
                     "data_batch", "data_padding", "data_fractioned", "data_type"), FEATURE_FIELDS):
        v = s.get(k)
        if isinstance(v, list) and not v:            # MATLAB [] (e.g. padding of a folder without digits)
            v = np.zeros(0)
        vpr[f] = v if isinstance(v, str) else as_matlab_value(v)
    if not s["ok"]:
        print(" Sorry, analysis of dimensions of templates and data failed!")
        raise DynamoError("[aborting]!")
    vpr = save(vpr)
    # steps 7-8
    if _links2settings(vpr):
        print("[vunfold] Error while copying linked files (in [Project files]) into the project database.")
        return None, "error"
    _settings2first_references(vpr)
    # step 9
    rounds = round_array(vpr)
    if not rounds:
        print("[vunfold] Fatal error determining iteration sequence... are all rounds empty?")
        return None, "error"
    # step 10
    for ite in range(1, len(rounds) + 1):
        c = card_ite_from_vpr(vpr, ite)
        c["destination"] = vpr["destination"]
        dynio.write_card(loc("card_ite", vpr["name_project"], ite=ite), c)
    # step 11
    if vpr.get("project_type") in ("ccmatrix", "post_ccmatrix"):
        raise NotImplementedError("ccmatrix projects")
    # step 12
    if "gpu" in str(vpr["destination"]):
        for r in range(1, 9):
            d = vpr["dim_r%d" % r]
            if float(d) == 0:
                continue
            os.makedirs(os.path.join(vpr["name_project"], "rescaled_data", "data_%d" % int(d)), exist_ok=True)
    # step 13
    if vpr.get("MCR_CACHE_ROOT"):
        os.makedirs(vpr["MCR_CACHE_ROOT"], exist_ok=True)
        if float(vpr["multi_MCR_CACHE_ROOT"]):
            for i in range(int(vpr["how_many_processors"])):
                os.makedirs(os.path.join(vpr["MCR_CACHE_ROOT"], "temp_thread%d" % i), exist_ok=True)
    # step 14
    if runfrom is not None:
        vpr["runfrom"] = float(max(int(math.floor(runfrom + 0.5)), 1))
    script = execution_script(vpr)
    # step 15: unix('chmod u=rwx <script>')
    if vpr["destination"] in ("system_omp", "system", "system_gpu", "standalone", "standalone_gpu"):
        st = os.stat(script).st_mode
        os.chmod(script, (st & ~0o700) | 0o700)
    _say('Unfolding completed: "%s"' % vpr["name_project"], verbose)
    return script, "ok"


dvunfold = unfold


def check(name, verbosity=1):
    """dvcheck = dynamo_vpr_check (see pydyn.vprcheck)."""
    from .vprcheck import check as _check
    return _check(name, verbosity=verbosity)


dvcheck = check
