"""Adaptive bandpass of Dynamo 1.1.514, golden-standard mode (vpr.adaptive_bandpass = 1, exactly two references):
+dpkproject/adaptiveFiltering.m, called at the very end of dynamo_iteration_assemble (:812-822).

Specification: docs/analysis/09b_adaptive_bandpass_1.1.514.md ("09b"), sections 4 and 9; 09a section 6.3/7.3.
S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src; line numbers below are adaptiveFiltering.m unless stated.

Not ported (NotImplementedError): the even/odd mode (adaptive_bandpass_even_odd: its per-reference template low in
the next setup is not verified), the symmetry kinds pydyn.symmetry refuses (i, cbo, multisymmetry), and every
dynamo_align branch listed in pydyn.dalign. Symmetry (adaptive_bandpass_symmetrized, 10a 6): pydyn.symmetry. abp_recompute_second_average = 0 uses the mex option-0 rotation (verified with shifts and a
non-trivial rotation on Dynamo's dynamo_rigid_apply, benchmarks/05_abp_bin4_eo/checks/impl_gt/gt_align).
"""
import math
import os

import numpy as np

from . import dynio


# =====================================================================================================================
# numerics
# =====================================================================================================================

def fsc(a, b):
    """dynamo_fsc(a, b).fsc as called here (dynamo_fsc.m:168-438, dynamo_fsc_fouriermask.m:59-135; 09b 9.1): N/2
    one-pixel shells R in [k, k+1), k = 1..N/2, fsc = real(sum A conj B) / (sqrt(sum A conj A) sqrt(sum B conj B)),
    sums over the shell voxels in column-major (C order of [z, y, x]). Returns None if a volume has a NaN."""
    if np.isnan(a).any() or np.isnan(b).any():
        return None
    n = a.shape[0]
    if a.shape != (n, n, n) or n % 2:
        raise NotImplementedError("dynamo_fsc of a non-cubic or odd box")
    A = np.fft.fftshift(np.fft.fftn(a))
    B = np.fft.fftshift(np.fft.fftn(b))
    Nm, A2, B2 = A * np.conj(B), A * np.conj(A), B * np.conj(B)
    u = np.arange(-n // 2, n // 2, dtype=np.float64)
    uz, uy, ux = np.meshgrid(u, u, u, indexing="ij")
    R = np.sqrt(ux * ux + uy * uy + uz * uz)
    out = np.empty(n // 2)
    for k in range(1, n // 2 + 1):
        s = (R >= k) & (R < k + 1)
        out[k - 1] = np.real(np.sum(Nm[s])) / (math.sqrt(np.real(np.sum(A2[s]))) * math.sqrt(np.real(np.sum(B2[s]))))
    return out


def fsc_resolution(curve, threshold):
    """dynamo_fsc_resolution(fsc, 'threshold', t) (dynamo_fsc_resolution.m:74-161; 09b 9.3): the first shell
    i >= 3 with fsc(i) <= t (1-based), 3 if none is found before a NaN-only tail, N/2 if all fsc(3:end) > t."""
    c = np.asarray(curve, dtype=np.float64).ravel()
    fp = 3
    found = False
    for i in range(3, len(c) + 1):
        if c[i - 1] <= threshold:
            fp, found = i, True
            break
    if not found and np.all(c[2:] > threshold):
        fp = len(c)
    return fp


def _E(e):
    from .dalign import euler2matrix
    return euler2matrix(*e)


def table_align(T, tp_shifts, tp_eulers):
    """dynamo_rigid_apply(table, Tp) = dynamo_table_align(table, 'T', Tp) for Tp of type 'shiftrot'
    (dynamo_table_align.m:39-48, dynamo_rigid_compose.m, dynamo_rigid_intotable.m, dynamo_rigid_switch.m,
    dynamo_rigid_inverse.m; 09b 4.4). Columns 4-9 change, every other column is kept."""
    from .dalign import matrix2euler, mm3
    out = np.array(T, copy=True)
    if np.iscomplexobj(out):
        if np.any(np.imag(out[:, 3:9]) != 0):
            raise NotImplementedError("complex shifts/angles in dynamo_table_align")
    tps = np.asarray(tp_shifts, dtype=np.float64)
    Ep = _E(tp_eulers)
    tags = np.real(out[:, 0])
    if len(np.unique(tags)) != len(tags):
        raise NotImplementedError("repeated tags in dynamo_table_align (dynamo_table_marktag sets every match)")
    for i in range(out.shape[0]):
        t = np.real(out[i])
        s0 = -t[3:6]
        e0 = (-t[8], -t[7], -t[6])                                   # dynamo_rigid('goal','align_particle')
        E0 = _E(e0)
        en = matrix2euler(mm3(E0, Ep))                               # compose: eulers
        sn = s0 + mm3(E0, tps)                                       # shiftrot: T1.shifts + R1 * T2.shifts
        sf = mm3(_E(en).T, sn)                                       # force rotshift
        ei = (-en[2], -en[1], -en[0])                                # inverse: shiftrot
        si = -sf
        se = mm3(_E(ei).T, si)                                       # force rotshift
        out[i, 3:6] = se
        out[i, 6:9] = ei
    return out


# =====================================================================================================================
# project helpers
# =====================================================================================================================

def _sym(vol, op):
    """dynamo_sym(vol, op) without fmask (adaptiveFiltering.m:212-216, :302-308): identity for c1 (dynamo_sym.m:382-383),
    cN/dN/helical through pydyn.symmetry (10a 6)."""
    from .symmetry import dynamo_sym
    return dynamo_sym(vol, op)[0]


def _read_rel(proj, path):
    """A vpr file string, relative to Dynamo's working directory (the folder that contains the project)."""
    if os.path.isabs(path):
        return path
    for pre in ("./" + proj.name + "/", proj.name + "/"):
        if path.startswith(pre):
            return proj.p(path[len(pre):])
    return os.path.join(proj.cwd, path)


def _bars(proj, s=None):
    print("-" * 60 if s is None else s, flush=True)


class _Echo:
    """mbs.o with leading 'bandpass #<ite>' and intofile <project>/log.txt (:137-144)."""

    def __init__(self, proj, ite):
        self.proj, self.lead = proj, "[bandpass #%d] " % ite

    def __call__(self, msg):
        line = self.lead + msg
        print(line, flush=True)
        with open(self.proj.p("log.txt"), "a") as f:
            f.write(line + "\n")


def vpr_write_target(proj, vpr=None):
    """dynamo_vpr_put saves into vpr.path (09b 4.9). In place (source == out) the path must be the folder that
    contains the project, otherwise Dynamo would write another project's vpr: pydyn refuses. For a re-execution
    under another root (source != out) the run folder is taken as vpr.path (documented deviation)."""
    vpr = proj.vpr if vpr is None else vpr
    if proj.source != proj.out:
        return proj.out
    target = os.path.join(str(vpr["path"]), proj.name)
    if os.path.realpath(target) != os.path.realpath(proj.out):
        raise RuntimeError("adaptive bandpass: vpr.path is %r, so Dynamo 1.1.514 would save the new low into "
                           "%s/settings/virtual_project.mat, not into this project (09b 4.9). pydyn refuses; set "
                           "'path' to the folder that contains the project first." % (vpr["path"], target))
    return proj.out


def check_sym(op, n):
    """Refuse before any computation a symmetry string pydyn.symmetry cannot apply (10a 1.1, 2.1): kinds none, c, d
    and h (with dz > 0 and an integer repetition count) are ported."""
    from . import symmetry
    kind, p, ok = symmetry.parse_sym(op)
    if not ok:
        raise NotImplementedError("symmetry %r does not parse (dynamo_vpr_check errors)" % (op,))
    if kind in ("c", "d"):
        symmetry._order(p, kind)
    elif kind == "h":
        if n <= 0:
            raise NotImplementedError("helical symmetry without feature_data_sidelength")
        symmetry.helical_reps(n, p)
    elif kind == "none":
        if list(p) != [1.0]:
            # 'n' / 'none' / '': parameters [] -> dynamo_rigid_compare's symorder is [] (not determined)
            raise NotImplementedError("symmetry string %r (only c1 for no symmetry)" % (op,))
    else:
        raise NotImplementedError("symmetry %r (kind %s) is not ported" % (op, kind))
    return kind


def check_supported(vpr, rounds, proj=None):
    """Raise NotImplementedError for golden-mode settings this port does not reproduce, before any computation.
    With `proj`, also the vpr.file_mask that dynamo_align reads (cwd-relative, dynamo_align.m:963, :974)."""
    if int(float(vpr["nref_r1"])) != 2:
        raise NotImplementedError("adaptive bandpass needs nref 2 (dynamo_vpr_check errors otherwise)")
    from .dalign import BASIC, motor_area, check_motor_area
    for rnd in sorted(set(rounds)):
        check_sym(vpr["sym_r%d" % rnd], int(float(vpr.get("feature_data_sidelength", 0) or 0)))
        if int(float(vpr["localnc_r%d" % rnd])) != 1:
            raise NotImplementedError("dynamo_align with localnc 0")
        for p in BASIC:
            if "%s_r%d" % (p, rnd) not in vpr:
                raise NotImplementedError("round parameter %s_r%d missing" % (p, rnd))
        m = int(float(vpr["area_search_modus_r%d" % rnd]))
        if m != 0:
            # dynamo_align passes the particle box as both s and feature_data_sidelength (:1329, :1815), so the
            # motor sees ceil(area_search) (dynamo__align_motor.m:233-248 for the zero-component branch)
            check_motor_area(motor_area(vpr["area_search_r%d" % rnd], 1, 1), m)
    if proj is not None:
        # a missing or non-3-D file_mask makes dynamo_align fall back to dynamo_ellipsoid(floor(N/2), N) with an
        # unterminated 'r=floor(d/2)' echo (dynamo_file_checkout.m returns the string, dynamo_align.m:1068-1082)
        f = str(vpr.get("file_mask", ""))
        for pre in ("./" + proj.name + "/", proj.name + "/"):          # checked in the source project: the run
            if f.startswith(pre):                                      # folder may not be filled yet
                f = os.path.join(proj.source, f[len(pre):])
                break
        else:
            f = f if os.path.isabs(f) else os.path.join(proj.cwd, f)
        if not str(vpr.get("file_mask", "")) or not os.path.isfile(f):
            raise NotImplementedError("adaptive bandpass with a missing file_mask %r (dynamo_align.m:1068-1082 "
                                      "default ellipsoid)" % vpr.get("file_mask"))
        if np.ndim(dynio.read_em(f)) != 3:
            raise NotImplementedError("adaptive bandpass with a file_mask that is not 3-D (dynamo_align.m:1072-1076)")


# =====================================================================================================================
# the hook
# =====================================================================================================================

def adaptive_filtering(proj, ite):
    """dpkproject.adaptiveFiltering(card_ite of ite) for a pydyn.dynproject.DynamoProject `proj`."""
    from . import dalign
    from .dynproject import load_vpr, ite2round, card_ite_from_vpr
    from . import vpr as V
    card = dynio.read_card(proj.f_card_ite(ite))                     # :107-131
    vpr = load_vpr(proj.p("settings", "virtual_project.mat"))
    this_round = ite2round(vpr, ite)
    threshold = float(vpr["adaptive_bandpass_threshold"])
    pushback = float(vpr["adaptive_bandpass_pushback"])
    o = _Echo(proj, ite)
    _bars(proj)
    print(" \n   [Dynamic bandpass] (Pipeline-embedded version)\n ", flush=True)
    nref = int(float(card["nref"]))
    if nref == 1:
        print("Only one reference available. Desisting [adaptive bandpass] module")
        return None
    if nref > 2:
        print("Two many references: this module will work for simple refinements. Desisting [adaptive bandpass] "
              "plugin")
        o(" ... but you can always write your own plugin!")
        return None
    files = [proj.f_av("average", r, ite) for r in (1, 2)]           # :173-194
    if not all(os.path.isfile(f) for f in files):
        raise NotImplementedError("adaptive bandpass with a missing average (ref '*' skips missing files)")
    A = [dynio.read_em(f) for f in files]
    op = vpr["sym_r%d" % this_round]
    template, particle = A[0], A[1]
    if float(vpr["adaptive_bandpass_symmetrized"]):
        o("Symmetrizing both references before aligning ref 2 onto ref 1")
        template, particle = _sym(template, op), _sym(particle, op)
    o("Aligning reference 2 onto reference 1 before computing resolution.")
    mask = dynio.read_em(_read_rel(proj, str(vpr["file_mask"])))     # dynamo_align: q.mask = vpr.file_mask
    sal = dalign.dynamo_align(particle, template, vpr, this_round, mask=mask)
    o("... done")
    o(" shifts on reference 2: ")                                    # '%5.2 ' prints nothing (09b 4.3)
    o(" angles on reference 2: ")
    write_sal(proj.p(proj._av(ite), "alignmentRef2ToRef1_ite_%04d.sal" % ite), sal, template, particle)
    tp_s, tp_e = sal["Tp"]["shifts"], sal["Tp"]["eulers"]
    f_t2 = proj.f_av("refined_table", 2, ite, "tbl")                 # :235-243
    if not os.path.isfile(proj.f_av("refined_table", 1, ite, "tbl")) or not os.path.isfile(f_t2):
        raise RuntimeError("Sorry! could not locate table for reference")
    T2n = table_align(dynio.read_tbl(f_t2), tp_s, tp_e)
    dynio.write_tbl(f_t2, T2n)
    if float(vpr["abp_recompute_second_average"]):                  # :250-256
        A2n = proj.average_from_table(ite, 2, T2n)["average"]       # genericInput2Average: in-memory table, double
    else:
        A2n = dalign.mex_shift_rot(A[1], tp_e, tp_s, option=0, outscope=0.0)
    dynio.write_em(proj.f_av("average_not_redefined", 1, ite), A[0])  # :263-270
    dynio.write_em(proj.f_av("average_not_redefined", 2, ite), A[1])
    T_backup = dynio.read_tbl(f_t2)                                   # read after the overwrite (Dynamo bug, 09b 4.6)
    # database extension 'em' -> dynamo_write_em_simple, the same writer as .btbl (dynamo_write.m:241-242): a complex
    # column 28 goes in as its real part [RUN: Dynamo 1.1.514's table_not_redefined_ref_002_ite_0002.em of
    # benchmarks/05_abp_bin4_eo/dynamo/run/bin4_align_1_eo, header [6 0 0 5], dims [4711 42 1], equals float32(real(T)) of
    # refined_table_ref_002_ite_0002.tbl; its col-28 entries 0+1.2074e-06i (rows 16, 40, 43) are stored as 0]
    if np.iscomplexobj(T_backup) and np.any(np.imag(T_backup) != 0):
        print("[pydyn] table_not_redefined: complex column(s) written as real parts (dynamo_write_em_simple, 06a U)")
    dynio.write_btbl(proj.f_av("table_not_redefined", 2, ite), T_backup)
    F2 = dynio.read_em(proj.f_av("fmask_average", 2, ite))           # :274-279 (rotation only, option 0)
    F2n = dalign.mex_shift_rot(F2, tp_e, (0.0, 0.0, 0.0), option=0, outscope=0.0)
    from .symmetry import dynamo_sym
    # [As2, Fs2] = dynamo_sym(A2', op, 'fmask', F2', 'foutput', 'binary') (:283-292; 10a 6): c1 returns F2' itself
    # (trivial_checkout, 09b 4.6); h: weights hsym(F2', dz 0), flat-weights test, binary at 1/(2R)
    symA2n, symF2n = dynamo_sym(A2n, op, F2n, "binary")
    dynio.write_em(proj.f_av("average", 2, ite), A2n)                # :283-292
    dynio.write_em(proj.f_av("average_symmetrized", 2, ite), symA2n)
    dynio.write_em(proj.f_av("fmask_average_symmetrized", 2, ite), symF2n)
    if float(vpr["adaptive_bandpass_symmetrized"]):                  # :302-318
        o('Computing resolution with symmetry "%s"' % op)
        V1, V2 = _sym(A[0], op), _sym(A2n, op)
    else:
        o("Computing resolution without symmetry.")
        V1, V2 = A[0], A2n
    V1, V2 = apply_smoothing_mask(proj, vpr, V1), apply_smoothing_mask(proj, vpr, V2)
    curve = fsc(V1, V2)
    if curve is None:
        raise RuntimeError("dynamo_fsc returned [] (NaN in a volume): Dynamo crashes on .fsc")
    loc = proj.dstr(proj._av(ite) + "/bandpass_fsc_ite_%04d.fsc" % ite)
    dynio.write_fsc(proj.p(proj._av(ite), "bandpass_fsc_ite_%04d.fsc" % ite), curve.reshape(1, -1))
    o("Adaptive bandpass FSC for resolution estimation saved in file:")
    o(loc)
    fp = fsc_resolution(curve, threshold)                            # :326-337
    new_low = fp - pushback
    o("Resolution estimated for iteration %d: %5.2f (threshold %5.2f)" % (ite, fp, threshold))
    o("Low pass computed for next iteration %d: %5.2f (pushback %5.2f)" % (ite + 1, new_low, pushback))
    f_res = proj.p("results", "bandpass_resolution.txt")             # :342-352
    if not os.path.isfile(f_res):
        res = np.array([float(fp)])
    else:
        res = np.real(dynio.read_txt(f_res)).ravel()
        if len(res) < ite:
            res = np.concatenate([res, np.zeros(ite - len(res))])
        res[ite - 1] = fp
    for i, r in enumerate(res, 1):
        print("ite: %03d measured resolution (fourier pixels):%d" % (i, r))
    dynio.write_txt(f_res, res.reshape(1, -1))
    info = {"fp": int(fp), "new_low": new_low, "Tp_shifts": [float(x) for x in tp_s],
            "Tp_eulers": [float(x) for x in tp_e], "ccmax": [float(x) for x in sal["ccmax"]]}
    if new_low < 2:                                                  # :356-364
        o(" ")
        o("New low pass is too low! You are killing all the frequencies!!")
        raise RuntimeError("aborting!")
    next_round = ite2round(vpr, ite + 1)                             # :368-373
    if next_round is None:
        # dynamo_intertwin_ite2round.m:54-57 disp()s this before returning [] [RUN: eo_dynamo514_run.log:529994]
        print("Warning: accessing an iteration out of bounds", flush=True)
        o("Could not identify round number for next iteration (%d)" % (ite + 1))
        return info
    # dynamo_vpr_put(name, 'disk', 'inround', next_round, 'low', new_low)  (:396): load from the cwd, save into
    # vpr.path; then dynamo_vpr_unfold(name, 'refresh_cards', ite+1): only card_ite of ite+1 (:403)
    cur = V.read_vpr_file(proj.p("settings", "virtual_project.mat"))
    cur = V.add_missing_convention_parameters(cur)
    target = vpr_write_target(proj, cur)
    new = V.modify(cur, "low_r%d" % next_round, float(new_low))
    V.write_vpr_file(os.path.join(target, "settings", "virtual_project.mat"), V.correct_defaults(new, False))
    v2 = load_vpr(proj.p("settings", "virtual_project.mat"))
    c = card_ite_from_vpr(v2, ite + 1)
    c["destination"] = v2["destination"]
    dynio.write_card(proj.f_card_ite(ite + 1), c)
    info["next_round"] = next_round
    return info


def apply_smoothing_mask(proj, vpr, v):
    """+dpkproject/+mask/applySmoothingMask.m:7-17: v .* raw file_mask_smoothing (cwd-relative); a missing field or
    file leaves the volume unmasked."""
    f = vpr.get("file_mask_smoothing")
    if not f or not os.path.isfile(_read_rel(proj, str(f))):
        print("[smoothing mask] no smoothing mask file: volume used without mask")
        return v
    m = dynio.read_em(_read_rel(proj, str(f)))
    if m.shape != v.shape:
        raise NotImplementedError("smoothing mask of another size")
    return v * m


def write_sal(path, sal, template, particle):
    """dynamo_write(sal, '.sal') = save(file, '-mat', 'contents') of the dynamo_align output (dynamo_write.m:266-269),
    with Dynamo's field names and order (dynamo_align.m:1919-1930 scard..time_alignment, :1634-1642 t/p eulers and
    shifts, Tp, Tt, :1654-1657 basic parameters, :1661-1666 timings and dtype 'single_alignment', :1682-1692 the
    full_output volumes, :817 the *_name fields). The call of adaptiveFiltering.m:221 passes particle and template
    as volumes and no mask/table/tfmask/pfmask, so record_original_name (:2064-2078) gives 'NUMERIC' for the first two
    and 'DEFAULT' for the rest. aligned_particle = dynamo_rigid_apply(particle, Tp) (shiftrot: dynamo_shift_rot, mex
    option 0, border 0) and aligned_template = dynamo_rigid_apply(template, Tt) (rotshift: dynamo_rot_shift).
    The three timings are pydyn's own wall clock. Not reproduced: MATLAB's on-disk storage classes (e.g. a double
    saved as uint8) and the file bytes. The pipeline never reads the file back."""
    import scipy.io as sio
    from . import dalign
    pt = np.asarray(sal["ptable"], dtype=np.float64)
    obj = lambda xs: np.array([np.asarray(x, dtype=np.float64) for x in xs] + [None], dtype=object)[:-1]
    num = lambda v: np.asarray(v, dtype=np.float64).reshape(1, -1) if np.size(v) > 1 else float(np.ravel(v)[0])
    sc = sal["scard"]
    tp_s, tp_e = np.asarray(sal["Tp"]["shifts"], np.float64), np.asarray(sal["Tp"]["eulers"], np.float64)
    tt_e, tt_s = pt[6:9].copy(), pt[3:6].copy()
    c = {"scard": {"use_CC": float(sc["use_CC"]), "localnc": float(sc["localnc"]),
                   "area_search": num(sc["area_search"]), "area_search_modus": float(sc["area_search_modus"]),
                   "feature_data_sidelength": float(sc["feature_data_sidelength"])},
         "ccmax": np.asarray(sal["ccmax"]).reshape(1, -1),
         "ptable_refine": obj([r.reshape(1, -1) for r in sal["ptable_refine"]]),
         "ptable": pt.reshape(1, -1), "angles": obj(sal["angles"]),
         "cc_angles": obj([c_.reshape(-1, 1) for c_ in sal["cc_angles"]]),
         "angles_all": np.concatenate([np.asarray(a, np.float64).reshape(-1, 3) for a in sal["angles"]], axis=0),
         "cc_angles_all": np.concatenate([np.ravel(c_) for c_ in sal["cc_angles"]]).reshape(-1, 1),
         "time_alignment": float(sal["time_alignment"]),
         "t_eulers": tt_e.reshape(1, -1), "t_shifts": tt_s.reshape(1, -1), "p_shifts": -pt[3:6].reshape(1, -1),
         "p_eulers": -pt[8:5:-1].reshape(1, -1),
         "Tp": {"type": "shiftrot", "shifts": tp_s.reshape(1, -1), "eulers": tp_e.reshape(1, -1)},
         "Tt": {"type": "rotshift", "eulers": tt_e.reshape(1, -1), "shifts": tt_s.reshape(1, -1)}}
    for p in dalign.BASIC:
        c[p] = num(sal["q"][p])
    ap = dalign.mex_shift_rot(particle, tp_e, tp_s, option=0, outscope=0.0)
    at = dalign.rot_shift(template, tt_e, tt_s, 0.0)
    c.update({"time_overall": float(sal["time_overall"]), "time_total": float(sal["time_total"]),
              "dtype": "single_alignment",
              "template": np.transpose(template, (2, 1, 0)), "particle": np.transpose(particle, (2, 1, 0)),
              "aligned_particle": np.transpose(ap, (2, 1, 0)), "aligned_template": np.transpose(at, (2, 1, 0)),
              "particle_name": "NUMERIC", "template_name": "NUMERIC", "mask_name": "DEFAULT", "table_name": "DEFAULT",
              "tfmask_name": "DEFAULT", "pfmask_name": "DEFAULT"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.savemat(path, {"contents": c}, format="5", do_compression=False, oned_as="row")
