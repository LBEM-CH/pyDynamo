"""dvcheck = dynamo_vpr_check (Dynamo 1.1.514, S/dynamo_vpr_check.m) with dynamo_vpr_statistics (S/dynamo_vpr_statistics.m).

The 33 checks run in Dynamo's order (dynamo_vpr_check.m:122-159); report lines, warning fields and error fields are
concatenated as stretch_cell does (empty strings vanish). It writes nothing (08a 5.3 [RUN]).

Deviations (stated where they occur):
* check_gpu_motor: Dynamo looks for the CUDA binary dynamo_compute_iteration_<gpu_motor>; pydyn's GPU motor is its
  own engine, so the check asks for CuPy and gpu_motor 'spp' instead.
* Branches that need MATLAB (parallel toolbox licence), plugins, multireference .sel files or non-EM volumes raise
  NotImplementedError. Multireference projects (nref_r1 > 1) with settings folders follow 09a section 5.
"""
import collections
import os
import shutil

import numpy as np

from . import dynio
from . import vpr as V

CheckResult = collections.namedtuple("CheckResult", "report fields_warning fields_error warning_state st")


def _read(f):
    from .console import dread
    return dread(f)


def _size1(f, obj):
    """size(object, 1): rows of a table, nx of a volume ([z, y, x] arrays)."""
    if os.path.splitext(f)[1] in (".tbl", ".txt", ".rt"):
        return obj.shape[0]
    return obj.shape[-1]


def whatis(s):
    """dynamo_whatis.m (file and folder kinds used by the checks)."""
    if not isinstance(s, str):
        return "not a char"
    if not os.path.isfile(s) and not os.path.isdir(s):
        return "not in file system"
    if os.path.isfile(s):
        ext = os.path.splitext(s)[1]
        kinds = {".tbl": "table", ".sel": "sel file", ".exe": "exe file", ".txt": "txt", ".tags": "tags file",
                 ".bat": "bat", ".BAT": "bat", ".ctf": "ctf file", ".ms": "multisymmetry", ".ns": "settings",
                 ".cmd": "Chimera commands", ".cmm": "Chimera markerfile", ".rbt": "transform", ".fsc": "fsc file",
                 ".doc": "doc file", ".sh": "shell script", ".m": "matlab script", ".card": "card", ".pdf": "pdf",
                 ".stl": "stl", ".sem": "stack", ".star": "star", ".vll": "volume list", ".omd": "model",
                 ".mdl": "model list", ".ctlg": "catalogue", ".reg": "region"}
        if ext in kinds:
            return kinds[ext]
        if ext == ".em":
            h = open(s, "rb").read(16)
            if len(h) < 16 or h[0] not in (0, 3, 5, 6):
                return "unreadable file"
            end = "<" if h[0] == 6 else ">"
            nx = int(np.frombuffer(h[4:8], end + "u4")[0])
            return "volume" if nx > 1 else "image"
        if ext in (".vol", ".xmp", ".mrc", ".spi", ".map", ".rec", ".st", ".ali", ".preali"):
            raise NotImplementedError("dynamo_whatis for %s files (dynamo_read_header)" % ext)
        return "file"
    if V.isproject(s):
        return "project"
    import glob
    if glob.glob(os.path.join(s, "particle_*.*")):
        return "data folder"
    if os.path.isfile(s + ".ctlg"):
        return "catalogue"
    return "folder"


def statistics(vpr):
    """dynamo_vpr_statistics.m (st, errors, facts)."""
    a = {k: float("nan") for k in ("size_particles", "size_reference", "size_mask", "size_cmask", "size_smask",
                                   "size_table", "largest_tilt_range", "smallest_tilt_range",
                                   "available_data_files", "table_overflow", "table_repetitions", "size_fmask",
                                   "NaN_reference")}
    errors = []
    facts = V.data_info(vpr["folder_data"])
    if not facts["correct"]:
        errors.append("data folder is not correct")
    if facts["l"] == 0:
        errors.append("particle files exist but are empty!")
    a["available_data_files"], a["size_particles"] = facts["N"], facts["l"]
    nref = int(vpr["nref_r1"])
    for kind, field, f, nmin in (("template", "size_reference", vpr["file_template_initial"], nref),
                                 ("fmask", "size_fmask", vpr["file_fmask_initial"], nref),
                                 ("table", "size_table", vpr["file_table_initial"], nref),
                                 ("mask", "size_mask", vpr["file_mask"], 1),
                                 ("mask_classification", "size_cmask", vpr["file_mask_classification"], 1),
                                 ("mask_smoothing", "size_smask", vpr["file_mask_smoothing"], 1)):
        if nmin > 1:                                 # control_possibly_multi, multireference branch (:148-170)
            from .vpreo import check_initial_folders
            if whatis(f) not in ("folder", "sel file"):
                errors.append("the folder or sel file suppossed to contain initial %ss does not exist" % kind)
                continue
            files, _ = check_initial_folders(f, kind, nmin)
            files = [x for x in files if x is not None]
            if len(files) < nmin:
                errors.append("not enough correct %ss passed for the required multireference setting" % kind)
                continue
            f = files[0]                             # size of the first file, read without try (:168-169)
            a[field] = _size1(f, _read(f))
            continue
        if not os.path.isfile(f):
            errors.append("the file passed as initial %s does not exist" % kind)
            continue
        try:
            a[field] = _size1(f, _read(f))
        except NotImplementedError:
            raise
        except Exception:
            errors.append("the file passed as initial %s could not be read" % kind)
    # linked_table_initial for refs 1:nref, 'check' 'disk' (:129-140); control_table per table, the tilt ranges of
    # the last table win (covered_range is local to each call)
    tables = linked_list(vpr, "file_table_initial")
    if nref == 1:
        tables = [vpr["file_table_initial"]]
    elif any(x is None for x in tables):
        raise NotImplementedError("dvcheck of a multireference project with missing initial tables "
                                  "(cell_compact of the located list)")
    for t in tables:
        a.update(_control_table(t, facts, errors))
    a["errors"] = errors
    return a, errors, facts


def linked_list(vpr, field):
    """dynamo_database_locate('linked_<item>', vpr, 'ref', '*'): one entry per reference (None if missing)."""
    from .vpreo import linked_files
    return linked_files(vpr, field, list(range(1, int(vpr["nref_r1"]) + 1)))


def _control_table(t, facts, errors):
    """dynamo_vpr_statistics.m control_table (:173-271)."""
    a = {}
    if not os.path.isfile(t):
        errors.append("table not found: it is not a file")
        a["table_overflow"] = a["table_repetitions"] = float("nan")
    else:
        T = np.real(dynio.read_tbl(t))
        if T.size == 0:
            errors.append("table exists but it is useless, as it does not match any of the data tags")
            a["table_overflow"] = a["table_repetitions"] = float("nan")
        else:
            tags = T[:, 0]
            a["table_overflow"] = len(np.setdiff1d(tags, facts["tags"]))
            if a["table_overflow"] > 0:
                errors.append("your table has %d tag(s) pointing to non existent files" % a["table_overflow"])
            a["table_repetitions"] = len(tags) - len(np.unique(tags))
            if a["table_repetitions"] > 0:
                errors.append("your tables points to the same file several times ")
            cov = []
            for row in T:
                ft = row[12] if len(row) > 12 else 0
                if ft == 1:
                    cov.append(row[14] - row[13])
                elif ft == 2:
                    cov.append(row[16] - row[15])
                elif ft == 3:
                    cov.append(min(row[16] - row[15], row[14] - row[13]))
                else:
                    cov.append(180.0)
            a["largest_tilt_range"], a["smallest_tilt_range"] = max(cov), min(cov)
    return a


def _linked(vpr, field):
    """dynamo_database_linked_file_read for reference 1: the file if it exists, else None."""
    return linked_list(vpr, field)[0]


def check(vpr, verbosity=1):
    """dynamo_vpr_check(vpr | name, verbosity) -> CheckResult(report, fields_warning, fields_error, warning_state,
    st)."""
    name = vpr if isinstance(vpr, str) else None
    if name is not None:
        vpr = V.load(name)
    if vpr is None:
        print("unreadable input, aborting dynamo_vpr_check")
        return CheckResult(["error"], "Aborted", "Aborted", "Aborted", None)
    st, _, facts = statistics(vpr)
    wisdom = {"facts": facts}
    R, W, E = [], [], []

    def add(r=(), w=(), e=()):
        R.extend(x for x in r if x != "")
        W.extend(w)
        E.extend(e)

    nan = lambda x: isinstance(x, float) and np.isnan(x)     # comparisons with NaN are false in MATLAB
    gt = lambda a, b: not (nan(a) or nan(b)) and a > b
    ne = lambda a, b: not (nan(a) or nan(b)) and a != b
    lt = lambda a, b: not (nan(a) or nan(b)) and a < b
    dest = vpr["destination"]
    ite_on = [r for r in range(1, 9) if float(vpr["ite_r%d" % r]) > 0]

    # 1 check_existence_files (:209-279)
    r, e = [], []
    dep = "file" if int(vpr["nref_r1"]) == 1 else "dir"     # depends_on_multireference (:211-215)
    for field, kind, label in (("file_template_initial", dep, "file/folder with the initial template"),
                               ("file_table_initial", dep, "file/folder that defines the initial configuration"),
                               ("file_fmask_initial", dep,
                                "file/folder that defines the initial count of fourier components"),
                               ("file_mask", "file", "file with the mask"),
                               ("file_mask_classification", "file", "file with the mask for classification")):
        ok = field in vpr and whatis(vpr[field]) in (("volume", "table", "shell script") if kind == "file" else
                                                     ("folder", "sel file", "data folder"))
        if not ok:
            r += ['ATTENTION on parameter "%s"' % field, "   %s does not seem to exist." % label,
                  '   Apparent current value in project: "%s".' % vpr.get(field)]
            e.append(field)
    if not os.path.isdir(vpr["folder_data"]) and not os.path.isfile(vpr["folder_data"]):
        e.append("folder_data")
    add(r, (), e)
    # 2 check_cluster_header (:281-308)
    if not os.path.isfile(vpr["cluster_header"]) and dest.startswith("mpi"):
        add(['[check:error] Project parameter "cluster header" (where you defined the cluster syntax) ',
             '              This parameter is pointing to an unknown file: "%s" ' % vpr["cluster_header"],
             " [ERROR] This is a fatal error for MPI computation ",
             "         Please provide a correct configuration file or use a different \"destination\" (not mpi) "],
            (), ["cluster_header"])
    # 3 check_multireference_files: nref 1 -> nothing (:356-397)
    nref = int(vpr["nref_r1"])
    if nref > 1:
        from .vpreo import check_initial_folders
        for field in ("file_template_initial", "file_table_initial", "file_fmask_initial"):
            kind_initial = field.replace("file_", "")
            files, irep = check_initial_folders(vpr[field], field, nref)
            files = [x for x in files if x is not None]
            if irep == "error":
                add(['[ERROR]  concerning "%s" detected.' % field], (), [field])
            if len(files) < nref:
                add(["      only %d correctly names files in folder %s, %d required to provide a set of '%s' for "
                     "multireference" % (len(files), vpr[field], nref, kind_initial)], (), [field])
    # 4 check_some_particle_survive (:399-435): every linked table (dynamo_database_linked_file_read 1:Nref)
    t = _linked(vpr, "file_table_initial")
    tabs = [x for x in linked_list(vpr, "file_table_initial") if x is not None]
    if not tabs:
        add(["Attention!", "Unable to read your initial table"], ["file_table_initial"])
    rep4 = []
    for i, tf_ in enumerate(tabs, 1):
        T = np.real(dynio.read_tbl(tf_))
        if T.size == 0 or not np.any(T[:, 1] != 0):
            rep4 = ["Attention!", "In the initial table # %d there are not any particles marked for processing" % i,
                    "Remember to fill column number 2 with ones"]           # report{1..3} overwritten per table
            add((), ["file_table_initial"])
    add(rep4)
    # 5 check_particle_size_ok
    sp = st["size_particles"]
    if not nan(sp) and (sp <= 10 or sp > 512):
        add(["problem with data  size (%d). Please check " % sp], ["folder_data"])
    # 6 check_wedge_reasonable
    if lt(st["largest_tilt_range"], 10):
        add([" Maximum wedge (%f) seems too small: . Please check table. " % st["largest_tilt_range"]],
            ["file_table_initial"])
    if lt(st["smallest_tilt_range"], 10):
        add([" Minimum wedge (%f) seems too small: . Please check table. " % st["smallest_tilt_range"]],
            ["file_table_initial"])
    # 7 check_some_round_has_iterations (:601-625)
    if not ite_on:
        add(['Apparently all the rounds are empty: no iterations are executed. Check the "ite" parameter'],
            ["ite_r1"])
    # 8 check_data_folder_has_data
    if lt(st["available_data_files"], 2):
        add([" only %d files with right name convention " % st["available_data_files"]], ["folder_data"])
    # 9 check_table_is_compatible_with_data: st has no table_largest_tag -> the try always fails (inert)
    # 10 check_filter_size_data (:508-554)
    if vpr["project_type"] == "general":
        for i in ite_on:
            d = float(vpr["dim_r%d" % i])
            if gt(d, sp):
                add([" too large particle dimension in round %d!!! Check \"dim_r%d\" parameter! " % (i, i)],
                    ["dim_r%d" % i])
            if d == 0:
                add([" particle dimension is zero in round %d!!! Check \"dim_r%d\" parameter!" % (i, i)],
                    ["dim_r%d" % i])
            if d % 2 != 0:
                add([" particle dimension is not evenin round %d!!! Check \"dim_r%d\" parameter!" % (i, i)],
                    ["dim_r%d" % i])
    # 11-13 sizes
    if ne(st["size_reference"], st["size_mask"]):
        add(["Mask has size %d, reference has size %d. Please check " % (st["size_mask"], st["size_reference"])],
            ["file_mask", "file_template_initial"])
    if ne(st["size_reference"], st["size_smask"]):
        add(["Smoothing FSC Mask has size %d, reference has size %d. Please check "
             % (st["size_smask"], st["size_reference"])], ["file_mask_smoothing", "file_template_initial"])
    if gt(st["size_reference"], sp):
        add(["reference has size %d, bigger than particle size %d. Please check " % (st["size_reference"], sp)],
            ["folder_data", "file_template_initial"])
    # 14 check_template_equal_data_localnc (:675-707; note errors{1}/{2} are overwritten, not appended)
    r14, e14 = [], []
    for r_ in range(1, 9):
        if float(vpr["localnc_r%d" % r_]):
            continue
        if ne(st["size_reference"], sp):
            r14.append('reference size (%d) differs from particle size (%d). This is not allowed when "localnc" is '
                       "set to zero (as in round %d). Please check. " % (st["size_reference"], sp, r_))
            e14 = ["folder_data", "file_template_initial"]
    add(r14, (), e14)
    # 15 check_template_is_not_flat (:1023-1104)  [runs before 16 in the call list]
    tf = _linked(vpr, "file_template_initial")
    tfs = linked_list(vpr, "file_template_initial")
    if nref > 1 and any(x is None for x in tfs):
        raise NotImplementedError("dvcheck of a multireference project with missing templates")
    if vpr["project_type"].find("ccmatrix") < 0 and tf is not None:
        mf = vpr["file_mask"]
        if not os.path.isfile(mf):
            add(["mask does not exist: "], (), ["file_mask"])   # sprintf('mask does not exist: %s') without
            #                                                     argument (dynamo_vpr_check.m:1044)
        else:
            mask = _read(mf)
            mx, my, mz = mask.shape[::-1]
            for i, tfi in enumerate(tfs, 1):          # the first failing reference returns (:1057-1088)
                tmpl = _read(tfi)
                tx, ty, tz = tmpl.shape[::-1]
                if (mx, my, ty) != (tx, ty, tz):          # sic: [mx,my,ty]~=[tx,ty,tz] (:1067)
                    add(["template for reference %d and mask have different dimensions." % i,
                         "template: %d x %d x %d   mask: %d x %d x %d" % (tx, ty, tz, mx, my, mz)], (),
                        ["file_template_initial", "file_mask"])
                    break
                idx = mask != 0
                if idx.sum() < 2:
                    add(["template for reference %d is zero in the mask area." % i], (),
                        ["file_template_initial", "file_mask"])
                    break
                roi = tmpl[idx]
                if np.std(roi, ddof=1) == 0:
                    add(["template for reference %d is flat in the mask area (mean value is %f)." % (i, roi.mean())],
                        (), ["file_template_initial", "file_mask"])
                    break
    # 16 check_template_not_NaNs (:991-1021)
    if vpr["project_type"].find("ccmatrix") < 0 and tf is not None:
        bad = [i for i, tfi in enumerate(tfs, 1) if np.isnan(_read(tfi)).any()]
        if bad:
            if len(bad) > 1:
                raise NotImplementedError("mbparse.prettystr of several references with NaN")
            add(["NAN found in template! Refes: %d" % bad[0]], (), ["file_template_initial"])
    # 17-18 inactive; 19 check_symmetry_syntax (:709-736)
    for r_ in range(1, 9):
        if not _sym_ok(vpr["sym_r%d" % r_]):
            add([' Wrong symmetry definition of parameter  "sym_r%d"' % r_,
                 '    Cannot understand the string  "%s" as operator or multisymmetry file.' % vpr["sym_r%d" % r_]],
                (), ["sym_r%d" % r_])
    # 20-21 plugins (:738-902)
    for kind in ("align", "post", "iter"):
        for r_ in range(1, 9):
            if float(vpr["plugin_%s_r%d" % (kind, r_)]):
                raise NotImplementedError("dvcheck of active plugins (plugin_%s_r%d)" % (kind, r_))
    # 22 check_size_fourier_count
    if ne(st["size_fmask"], st["size_reference"]):
        add(["fmask has size %d, reference has size %d. Please check " % (st["size_fmask"], st["size_reference"])],
            (), ["folder_data", "file_fmask_initial"])
    # 23 check_reasonable_destination (:337-354)
    reasonable = ["matlab", "matlab_parfor", "matlab_gpu", "matlab_multicore", "standalone", "system_omp", "mpi",
                  "system_gpu", "mpi_gpu", "standalone_gpu"]
    if dest not in reasonable:
        add(['Warning: "%s" is not an acceptable value for parameter "destination"' % dest, "   Use one of this list:",
             " ".join(reasonable) + " "], ["destination"])
    # 24 check_processors_defined (:924-949)
    hmp = vpr["how_many_processors"]
    if V._is_empty(hmp):
        add(['ERROR on parameter "how_many_processors": unknown number of processors'], (), ["how_many_processors"])
    elif float(hmp) > 1 and dest in ("system_gpu", "matlab_gpu"):
        add(['ATTENTION: inconsistency "how_many_processors" (%d), and "destination" (%s) too many  processors'
             % (hmp, dest)], (), ["how_many_processors"])
    elif float(hmp) > 1 and dest == "matlab":
        raise NotImplementedError("dvcheck for destination matlab with several cores (parallel toolbox licence check)")
    # 25 check_gpu_motor (:951-989) -- deviation, see the module docstring
    if "gpu" in dest and "gpu_motor" in vpr:
        problem = vpr["gpu_motor"] != "spp"
        try:
            import cupy  # noqa: F401
        except Exception:
            problem = True
        if problem:
            add(['WARNING on parameter "gpu_motor". The GPU motor function could not be located in the expected '
                 "location:", "pydyn engine (CuPy) for gpu_motor %r" % vpr["gpu_motor"]], ["gpu_motor"])
    # 26 check_pfmasks_are_available (:1160-1246)
    wisdom["ftype5"] = False
    for tt in tabs:
        T = np.real(dynio.read_tbl(tt))
        if T.size and T.shape[1] >= 13 and np.any((T[:, 1] == 1) & (T[:, 12] == 5)):
            raise NotImplementedError("dvcheck with ftype 5 particles (pfmask data_info)")
    # 27 check_legal_threshold (:556-599)
    if vpr["project_type"] == "general":
        legal = [0, 1, 2, 3, 4, 5, 11, 21, 31, 41, 51]
        for i in ite_on:
            for f, lab in (("threshold_modus_r%d" % i, "Threshold"), ("threshold2_modus_r%d" % i, "Threshold 2")):
                if float(vpr[f]) not in legal:
                    add(['  Invalid %s type "%d" in round %d!!! Check "%s"' % (lab, vpr[f], i, f)], (), [f])
    # 28 check_equal_size_data_template_gpu_ftype5 (:1106-1158)
    if any(float(vpr["use_CC_r%d" % r_]) == 2 for r_ in range(1, 9)) and tf is not None:
        for tfi in tfs:
            tm = _read(tfi)
            if tm.shape != (facts["l"],) * 3:
                add(["Size mismatch between particle and template.", "template: %d, particle %d" % (tm.shape[-1],
                                                                                                  facts["l"]),
                     "This is not allowed for this particular combination of wedge treatment and gpu_execution"], (),
                    ["file_template_initial"])
    # 29 check_goldenStandardReferences
    if int(vpr["nref_r1"]) != 2 and float(vpr["adaptive_bandpass"]) == 1:
        add(["Cannot run adaptive bandpass for %d references. You need two." % vpr["nref_r1"]], (),
            ["nref_r1", "adaptive_bandpass"])
    # 30 check_MPIforGPUwithMultipleCores
    if float(hmp) > 1 and dest == "mpi_gpu" and float(vpr["systemNoMPIWrapUnfoldingGPU"]) == 1:
        add(['You are using several cpu cores, but the parameter "systemNoMPIWrapUnfoldingGPU" is tuned to "1"',
             "  this can create problems during execution!"], ["how_many_processors", "systemNoMPIWrapUnfoldingGPU"])
    # 31 check_NoMRAforOneReference
    if int(vpr["nref_r1"]) == 1:
        for r_ in range(1, 9):
            if float(vpr["mra_r%d" % r_]) != 0:
                add(["  mra is tuned to run multireference, but you only have one reference!"], ["mra_r%d" % r_])
    # 32 check_multicores_being_used_under_Matlab (:310-335)
    if not (dest == "matlab" and float(hmp) == 1):
        if dest in ("matlab", "matlab_parfor", "matlab_multicore", "standalone") and \
                float(hmp) != float(vpr["matlab_workers_average"]):
            add(['Warning: Weird configuration of resources for "destination" "%s"' % dest,
                 '   cores during alignment: "%d"' % hmp, '   cores during averaging: "%d"'
                 % vpr["matlab_workers_average"]], ["destination", "how_many_processors", "matlab_workers_average"])
    # 33 check_all_required_data_is_there (:1248-1288)
    for iref, tt in enumerate(linked_list(vpr, "file_table_initial"), 1):
        if tt is None:
            if nref > 1:
                raise NotImplementedError("dynamo_read of a missing multireference table")
            continue
        T = np.real(dynio.read_tbl(tt))
        if T.size:
            asked = T[T[:, 2] == 1, 0]
            n1 = len(np.setdiff1d(asked, facts["tags"]))
            if n1 > 0:
                add(["Table #%d asks to align %d tag(s) not found in data" % (iref, n1)], (),
                    ["folder_data", "file_table_initial"])
            n2 = len(np.setdiff1d(T[:, 0], facts["tags"]))
            if n2 > 0:
                add(["Table #%d contains %d tag(s) not found in data" % (iref, n2)],
                    ["folder_data", "file_table_initial"])
    state = "warning" if (W or E) else "none"
    if verbosity:
        for ln in R:
            print(ln)
        if not R:
            print('ok, project "%s" seems safe enough.' % vpr["name_project"])
    return CheckResult(R, W, E, state, st)


def _sym_ok(chain):
    """dynamo_zaux_parse_sym(chain, 0) 'ok' output (S/dynamo_zaux_parse_sym.m; pydyn.symmetry.parse_sym, 10a 1.1-1.2).
    A helical string with fewer than 2 numbers is a MATLAB error() inside the parser (:174-178): ValueError."""
    from .symmetry import parse_sym
    if not isinstance(chain, str):
        return True                                  # numeric -> 'c<n>'
    if chain == "":
        return True
    return bool(parse_sym(chain)[2])


dvcheck = check
