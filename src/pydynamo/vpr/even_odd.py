"""dveo = dynamo_vpr_eo (Dynamo 1.1.514): derive the two-reference even/odd ("golden standard") project from a
single-reference alignment project, plus the multireference file helpers it needs.

Specification: docs/analysis/09a_even_odd_project_1.1.514.md ("09a" below), sections 1-4.
S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src; every function cites the lines it ports.

Like Dynamo, everything works relative to the current working directory (see pydynamo.vpr).

Ported: dveo(project[, 'name', n][, 'dynamic_bandpass'|'dbp'|'abf', 1][, 'save', 'd'|'u'][, 'v', 0/1]) on a
single-reference original with ite = 0. Not ported (NotImplementedError): 'ite' > 0 (09a O1, source only),
'ref' subsets and multireference originals (never run).
"""
import os
import re

import numpy as np

from pydynamo.files import formats
from pydynamo import vpr as V


# =====================================================================================================================
# file-name helpers
# =====================================================================================================================

def filename_compact(name):
    """pparse.filenameCompact (unix branch): strip leading './', collapse '/./' and '//' (repeat until stable)."""
    if not name:
        return name
    while re.search(r"/\./|^\./", name):
        name = re.sub(r"^\./", "", name)
        name = name.replace("/./././", "/")
        name = name.replace("/././", "/")
        name = name.replace("/./", "/")
        name = name.replace("//", "/")
    if name.startswith("~"):
        name = os.environ.get("HOME", "") + "/" + name[1:]
    return name


def force_abs(name):
    """dynamo_filename(name, 'force_abs', true) (S/dynamo_filename.m:252-381) for the cases reached here: an
    absolute path is kept (compacted); a relative one becomes pwd/<compacted name>. '..' paths are not ported."""
    if name == "":
        return ""
    n = filename_compact(name)
    if n.startswith("/"):
        return n
    if n.startswith(".."):
        raise NotImplementedError("dynamo_filename force_abs of a '..' path (%r)" % name)
    # existing: which() or pwd/<name> (:323-335); missing: pwd/<name> (:370-377) followed by Java's
    # getCanonicalPath (:379-381), which only matters for symlinked folders; it is applied here only to missing files
    # (they are never read through this path in dveo).
    if os.path.isfile(n) or os.path.isdir(n):
        return os.getcwd() + "/" + n
    return os.path.realpath(os.getcwd() + "/" + n)


def absolute_path_vpr(vpr):
    """dynamo_vpr_absolute_path(vpr, 'old_path', '.') (S/dynamo_vpr_absolute_path.m), as called by
    dynamo_database_locate for '<LINKED>' items with its default 'path' '.' (S/dynamo_database_locate.m:209,
    :371-377): the file fields become absolute, path := '.'."""
    new = dict(vpr)
    for f in ("file_template_initial", "file_table_initial", "folder_data", "file_mask", "file_mask_classification",
              "file_mask_smoothing", "file_fmask_initial", "cluster_header"):
        v = vpr.get(f, "")
        new[f] = "" if v == "" else force_abs(v)
    new["path"] = "."
    return new


# =====================================================================================================================
# multireference folders
# =====================================================================================================================

_PLAIN = {"file_template_initial": "template", "file_table_initial": "table", "file_fmask_initial": "fmask",
          "template_initial": "template", "table_initial": "table", "fmask_initial": "fmask",
          "template": "template", "table": "table", "fmask": "fmask"}


def check_initial_folders(folder_or_selfile, kind, minimum_files):
    """dynamo_check_initial_folders.m -> (files, 'ok'|'error'). Folder: for each ref, the single file matching
    <folder>/<kind>_initial_ref_<ref:03d>* (fullfile, so './x/settings' keeps its './'). A .sel input is not
    ported."""
    item = _PLAIN[kind]
    if os.path.isdir(folder_or_selfile) and not os.path.isfile(folder_or_selfile):
        files, ok = [], True
        for ref in range(1, int(minimum_files) + 1):
            root = "%s_initial_ref_%03d" % (item, ref)
            hits = sorted(f for f in os.listdir(folder_or_selfile) if f.startswith(root))
            if len(hits) == 1:
                files.append(_fullfile(folder_or_selfile, hits[0]))
            else:
                files.append(None)
                ok = False
        return files, ("ok" if ok else "error")
    if os.path.isfile(folder_or_selfile) and folder_or_selfile.endswith(".sel"):
        raise NotImplementedError("dynamo_check_initial_folders with a .sel file")
    return [], "error"


def _fullfile(folder, name):
    """MATLAB fullfile(folder, name) on unix: join with one '/'."""
    return folder.rstrip("/") + "/" + name if folder else name


def linked_files(vpr, field, refs):
    """dynamo_database_linked_file_read(vpr, item, refs) (S/dynamo_database_linked_file_read.m:179-200): nref 1 ->
    the vpr value; nref > 1 -> dynamo_check_initial_folders(value, item, nref)(ref). Entries whose file does not
    exist become None."""
    nref = int(vpr["nref_r1"])
    out = []
    for ref in refs:
        if ref > nref:
            out.append(None)
            continue
        if nref == 1:
            loc = vpr.get(field)
        else:
            files, rep = check_initial_folders(vpr[field], field, nref)
            loc = files[ref - 1] if rep == "ok" else None
        out.append(loc if isinstance(loc, str) and os.path.isfile(loc) else None)
    return out


def write_multireference(objects, item, folder, refs=None, verbose=True):
    """dynamo_write_multireference(objects, item, folder, 'refs', refs) (S/dynamo_write_multireference.m:329-470)
    for numeric tables and file names. Returns the written names."""
    root = {"table": "table_initial", "template": "template_initial", "fmask": "fmask_initial"}[_PLAIN[item]]
    objects = list(objects) if isinstance(objects, (list, tuple)) else [objects]
    if refs is None or len(refs) == 0:
        refs = list(range(1, len(objects) + 1))
    if len(objects) == 1 and len(refs) > 1:                  # :376-378
        objects = objects * len(refs)
    if len(objects) != len(refs):
        raise NotImplementedError("write_multireference with %d objects for %d refs" % (len(objects), len(refs)))
    if root == "fmask_initial" and any(isinstance(o, str) and o == "full" for o in objects):
        raise NotImplementedError("fmask 'full' in write_multireference")
    default_ext = ".tbl" if root == "table_initial" else ".em"
    names = []
    for iref, obj in zip(refs, objects):
        if isinstance(obj, str):
            if not os.path.isfile(obj):
                print('Warning: file "%s" does not exist. Returning empty-handed from dynamo_write_multireference'
                      % obj)
                return []
            ext = os.path.splitext(obj)[1]
        else:
            ext = default_ext
        name = "%s/%s_ref_%03d%s" % (folder, root, iref, ext)          # :448 [folder, filesep, ...]
        if isinstance(obj, str):
            if obj == name:
                V._say("source %s is the same as the destination" % obj, True)   # dynamo_copyfile verbosity 1
            V._copyfile(obj, name)
            doc_append("%s/original_locations_%s.doc" % (folder, root), [obj])
        else:
            formats.write_tbl(name, obj) if ext == ".tbl" else formats.write_em(name, obj)
        V._say("[write_multireference] Reference #%d.  File written: %s " % (iref, name), verbose)
        names.append(name)
    if refs:
        formats.write_lines("%s/multisettings_%s.sel" % (folder, root), ["%s 1" % n for n in names])
    return names


def doc_append(file, lines):
    """dynamo_doc_append.m: create the .doc with the lines or rewrite it with old lines + new lines."""
    old = formats.read_lines(file) if os.path.isfile(file) else []
    formats.write_lines(file, old + list(lines))


def multireference_setfiles(vpr, objects, item, refs=None, verbose=True):
    """dynamo_vpr_multireference_setfiles.m:105-160: write the per-reference files into ./<name>/settings and set
    file_<item>_initial to the first file (nref_r1 == 1) or to the folder (nref_r1 > 1)."""
    folder = "./%s/settings" % vpr["name_project"]                   # dynamo_database_locate('mask', 'folder')
    names = write_multireference(objects, item, folder, refs, verbose)
    if not names:
        print('Warning: Empty-handed return from dynamo_vpr_multireference_setfiles')
        return vpr
    param = "file_%s_initial" % _PLAIN[item]
    value = names[0] if int(vpr["nref_r1"]) == 1 else os.path.dirname(names[0])
    return V.modify(vpr, param, value)


# =====================================================================================================================
# dynamo_table_eo
# =====================================================================================================================

def table_eo(table):
    """dynamo_table_eo(table) with the defaults used by dveo (selection 'crop', column 2, label 1;
    S/dynamo_table_eo.m:179-246): rows with col 2 == 1 in table order; subtables[0] ('odd') = rows 2, 4, 6, ...;
    subtables[1] ('even') = rows 1, 3, 5, ..., stopping at end-1 (an odd count loses its last row)."""
    T = np.asarray(table)
    sel = T[np.real(T[:, 1]) == 1]                               # dynamo_table_restrict: ismember(col 2, 1)
    n = len(sel)
    return [sel[1::2], sel[0:max(n - 1, 0):2]]


# =====================================================================================================================
# dynamo_vpr(name, 'seedsfrom', source)
# =====================================================================================================================

def copy_of(name, source):
    """dynamo_vpr(name, 'seedsfrom', source, 'v', false) without other options (S/dynamo_vpr.m:620-728): every
    convention field from the source (default if absent), path := pwd, then <p>_r1.._r8 from the source. Fields
    outside the convention (feature_*, *_all) are not copied."""
    out = {"name_project": name}
    for p in V.NC:
        n = p["name"]
        val = source[n] if n in source else V.default_value(p)
        if n != "name_project":
            out[n] = val
        if n != "path":
            out["path"] = os.getcwd()
    for p in V.RC:
        for r in range(1, 9):
            f = "%s_r%d" % (p["name"], r)
            out[f] = source[f] if f in source else V.default_value(p)
    return out


# =====================================================================================================================
# dveo
# =====================================================================================================================

_FLAGS = {"name": "name", "v": "v", "disk": "disk", "save": "save", "dynamic_bandpass": "dynamic_bandpass",
          "abf": "dynamic_bandpass", "dbp": "dynamic_bandpass", "o": "o", "sel": "sel", "selection": "selection",
          "s": "s", "ws": "ws", "handles": "handles", "ref": "ref", "ite": "ite"}


def _parse(args, kw):
    q = {"name": None, "v": True, "save": "d", "dynamic_bandpass": False, "ref": None, "ite": 0}
    if len(args) % 2:
        raise V.DynamoError("dynamo_vpr_eo: flags must come in name/value pairs")
    pairs = list(zip(args[0::2], args[1::2])) + list(kw.items())
    for k, v in pairs:
        k = k[1:] if isinstance(k, str) and k.startswith("-") else k
        if k not in _FLAGS:
            raise V.DynamoError("dynamo_vpr_eo: unknown flag %r" % k)
        q[_FLAGS[k]] = V._str2num_ifnum(v) if isinstance(v, str) and _FLAGS[k] not in ("name", "save") else v
    return q


def dynamo_vpr_eo(project, *args, **kw):
    """dveo(project, flags...) (S/dynamo_vpr_eo.m:101-415). Returns the e/o vpr (None if Dynamo aborts)."""
    from pydynamo.vpr.checks import check as _check
    q = _parse(args, kw)
    verbose = bool(float(q["v"])) if not isinstance(q["v"], bool) else q["v"]
    src = V.load(project, verbose=False) if isinstance(project, str) else dict(project)
    if src is None:
        raise V.DynamoError("dynamo_vpr_eo: project %r cannot be loaded" % project)
    pname = src["name_project"]
    bars = "-" * 60
    if verbose:
        for ln in (bars, " ", "Creation of two-reference project to compute the FSC of an alignment",
                   "by two independents procedures, each with half of the table.", " "):
            print(ln)
    if not V.isproject(pname):
        print('[vpr_eo] Invalid project "%s"' % pname)
    rep = _check(src, verbosity=0)                                     # :215-222
    if rep.fields_error:
        print(rep.fields_error)
        print('[vpr_eo] Project "%s" contains errors, aborting' % pname)
        return None
    if int(src["nref_r1"]) != 1:
        raise NotImplementedError("dveo of a multireference original project (never run, 09a O1)")
    if q["ref"] not in (None, 1, [1]) and not (isinstance(q["ref"], np.ndarray) and q["ref"].tolist() == [1.0]):
        raise NotImplementedError("dveo 'ref' other than 1")
    if float(q["ite"]) != 0:
        raise NotImplementedError("dveo 'ite' > 0 (grep_average on a refined table; source only, 09a O1)")
    if q["save"] not in ("d", "u", "unfold"):      # 'unfold': the otherwise branch, no unfold [RUN h_dveo_save_u_unfold]
        raise NotImplementedError("dveo 'save' %r" % q["save"])
    refs = [1]
    name_eo = q["name"] if q["name"] else pname + "_eo"                    # :242-246
    eo = copy_of(name_eo, V.load(pname, verbose=False))               # :247 (seedsfrom: loaded by name)
    V.save(eo)                                                           # dynamo_vpr 'save' 'd' (:1180-1185)
    nref = 2 * len(refs)
    for r in range(1, 9):                                                # :248-252
        eo["nref_r%d" % r] = float(nref)
    print('Creating a total of %d references in project "%s"' % (nref, name_eo))
    src_abs = absolute_path_vpr(V.load(pname, verbose=False))           # linked_* of a project name (09a 1.2)
    for i, oref in enumerate(refs, 1):
        branched = [2 * i - 1, 2 * i]
        f_table = linked_files(src_abs, "file_table_initial", [oref])[0]
        if f_table is None:
            print("Cannot find table for original reference %d" % oref)
            return None
        sub = table_eo(formats.read_tbl(f_table))
        for st in (1, 2):
            n = int(np.sum(np.real(sub[st - 1][:, 1]) == 1)) if len(sub[st - 1]) else 0
            print("  * subtable %d for original reference %d aligns %d particles." % (st, oref, n))
        print("Original reference chanel %d branches into %d and %d in even/odd project." % (oref, *branched))
        eo = multireference_setfiles(eo, sub, "table", branched)
        f_tpl = linked_files(src_abs, "file_template_initial", [oref])[0]
        eo = multireference_setfiles(eo, f_tpl if f_tpl else "", "template", branched)
        f_fm = linked_files(src_abs, "file_fmask_initial", [oref])[0]
        eo = multireference_setfiles(eo, f_fm if f_fm else "", "fmask", branched)
    if q["dynamic_bandpass"] not in (False, 0, 0.0):                     # :359-381
        print("my_order =\n\n    'golden_standard_bandpass 0.5 2'\n")
        eo["adaptive_bandpass"] = 1.0
        eo["adaptive_bandpass_pushback"] = 2.0
        eo["adaptive_bandpass_initial"] = src["low_r1"]
    rep = _check(eo, verbosity=0)                                       # :385-392
    if rep.fields_error:
        print(rep.fields_error)
        print('[vpr_eo] E/O project "%s" contains errors, aborting.' % name_eo)
        return None
    eo = V.save(eo)                                                      # :396
    if q["save"] == "u":                                                 # :397-403 (case 'u' only)
        # `case 'u','unfold'`: the label is 'u'; ", 'unfold'" is a statement whose value MATLAB R2019a displays
        # [RUN: headless 1.1.514, 'save' 'u' prints this before unfolding; 'save' 'unfold' falls to otherwise and
        # does not unfold (benchmarks/05_abp_bin4_eo/dynamo/reference/logs/h_dveo_save_u_unfold.log)]
        print("\nans =\n\n    'unfold'\n", flush=True)
        V.unfold(eo)
    if verbose:
        print(" ")
        print(' [vpr_eo:ok] project "%s" created ' % name_eo)
        print(" ")
        print(bars)
    return eo


dveo = dynamo_vpr_eo
