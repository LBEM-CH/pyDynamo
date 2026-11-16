"""Dynamo 1.1.514 `log.txt` and GPU-engine stdout, line by line.

Spec: docs/analysis/10f_output_layout_and_log_1.1.514.md (every format below cites its 1.1.514 source line there).
  S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src
  C = /programs/x86_64-linux/dynamo/1.1.514/cuda/src  (compiled into cuda/bin/dynamo_compute_iteration_spp)

Everything here is a pure formatter (values in, lines out) except `append_log` (the only writer) and the
`*_from_project` readers, which only read a project's own files. A "line" is a str without its trailing '\n';
log.txt is "\n".join(lines) + "\n" (dynamo_aux_appendlog and mbparse.output.echo both write fprintf('%s\n')).

The engine runs one pthread per GPU (C/dynamo_compute_iteration_spp.cu:4853-4870). Their cout/printf fragments
interleave nondeterministically in Dynamo's log. `engine_stdout` writes the deterministic equivalent: the
single-threaded header, then each device's block whole, in gpu_identifier_set order, then the tail.
"""
import ctypes
import ctypes.util
import glob
import math
import os
import re
import time

import numpy as np

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
STARS = "*" * 48


# =====================================================================================================
# value formatting
# =====================================================================================================

def datestr(t=None):
    """MATLAB datestr(now), format 0 'dd-mmm-yyyy HH:MM:SS' (English month). t: epoch seconds or struct_time."""
    tm = time.localtime(time.time() if t is None else t) if not isinstance(t, time.struct_time) else t
    return "%02d-%s-%04d %02d:%02d:%02d" % (tm.tm_mday, MONTHS[tm.tm_mon - 1], tm.tm_year, tm.tm_hour, tm.tm_min,
                                            tm.tm_sec)


def asctime(t=None):
    """C asctime(localtime(&t)) without its '\\n': '%.3s %.3s%3d %.2d:%.2d:%.2d %d' (C locale)."""
    tm = time.localtime(time.time() if t is None else t) if not isinstance(t, time.struct_time) else t
    return "%s %s%3d %02d:%02d:%02d %d" % (WDAYS[tm.tm_wday], MONTHS[tm.tm_mon - 1], tm.tm_mday, tm.tm_hour,
                                           tm.tm_min, tm.tm_sec, tm.tm_year)


def datestr_to_struct(s):
    """'12-Nov-2026 01:12:28' -> struct_time (to derive placeholder values from card times); epoch 0 if the text
    is not a datestr (e.g. 'RECORDING PROBLEMS IN THE GPU')."""
    try:
        d, mo, rest = s.split("-", 2)
        y, hms = rest.split(" ")
        hh, mm, ss = (int(x) for x in hms.split(":"))
        return time.localtime(time.mktime((int(y), MONTHS.index(mo) + 1, int(d), hh, mm, ss, 0, 0, -1)))
    except (ValueError, IndexError):
        return time.localtime(0)


def cout_float(x):
    """C++ `ostream << float` with default flags: %g, precision 6 (the value is a float32)."""
    return "%g" % float(np.float32(x))


def c_f(x):
    """C printf %f of a float argument (promoted to double)."""
    return "%f" % float(np.float32(x))


def matlab_f(x):
    """MATLAB sprintf('%f') of a double."""
    return "%f" % float(x)


# =====================================================================================================
# MATLAB side: setup (S/dynamo_iteration_setup.m, S/+dpkproject/setup*.m)
# =====================================================================================================

def card_ite_path(name_project, ite):
    """dynamo_database_locate('card_ite', name_project, 'ite', ite) as passed by dynamo_execute_project.m:159."""
    return "./%s/cards/ite_%04d/card_ite_ite_%04d.card" % (name_project, ite, ite)


def card_iteref_path(name_project, ref, ite):
    """dynamo_database_locate('card_iteref', ...) (dynamo_execute_project.m:296)."""
    return "./%s/cards/ite_%04d/card_iteref_ref_%03d_ite_%04d.card" % (name_project, ite, ref, ite)


def setup_banner(ite, card_ite, t_start):
    """dynamo_iteration_setup.m:109-110 (texts) and :184-189 (appendlog). t_start: datestr string."""
    return [STARS,
            "**********    NEW ITERATION      ***************",
            "**********    iteration: %d      " % ite,
            STARS,
            "Starting iteration at %s" % t_start,
            "[setup] Parameter set is in %s" % card_ite]


def setup_freeze_lines(ite, has_freeze_fields, cone_ite_ref, inplane_ite_ref):
    """setup_freezing_angular_references.m:7-87 (echoes :17, :57, :66, :78, :86), o.leading 'setup:freeze',
    o.intofile = log.

    cone_ite_ref / inplane_ite_ref: getIteReference() results (ite, round leader or 1); None when both freeze
    flags of the round are 0 (the function returns silently). Only the lines that reach log.txt: after the
    'onlyCone' branch `o` is replaced by changeStartingTables' pparse.output() without intofile (:79), so the
    'Initial inplane' line of a cone+inplane split is screen-only."""
    if ite == 1:
        return []
    if not has_freeze_fields:
        return ["[setup:freeze] Project does not have parameters for reference freezing."]
    if cone_ite_ref is None and inplane_ite_ref is None:
        return []
    if inplane_ite_ref == ite and cone_ite_ref == ite:
        return ["[setup:freeze] No special policy for angular references in iteration %d." % ite]
    if cone_ite_ref == inplane_ite_ref and cone_ite_ref != ite:
        return ["[setup:freeze] Initial cone and azymuth in iteration %d will be read from iteration %d"
                % (ite, cone_ite_ref)]
    out = []
    if cone_ite_ref != inplane_ite_ref and cone_ite_ref != ite:
        out.append("[setup:freeze] Initial cone in iteration %d will be read from iteration %d" % (ite, cone_ite_ref))
        return out                                     # o lost its intofile (see docstring)
    if cone_ite_ref != inplane_ite_ref and inplane_ite_ref != ite:
        out.append("[setup:freeze] Initial inplane in iteration %d will be read from iteration %d"
                   % (ite, inplane_ite_ref))
    return out


def setup_shifts_lines(ite, area_search_modus, seed_ite=None):
    """setupFreezingShifts.m:9-105 (echoes :54, :58, :71, :83, :86, :90, :94, :104), o.leading 'setup:shifts',
    o.intofile = log. seed_ite: for modus 3 the last round leader < ite (strict), for modus 4 always 1."""
    if ite == 1:
        return []
    m = area_search_modus
    lead = "[setup:shifts] "
    if 2 < m < 5:
        if m == 3:
            if ite == seed_ite:
                return [lead + "Iteration %d starts a round Shift limits taken from previous iteration." % seed_ite]
            out = [lead + "Area search modus chosen to be of type 3 [read shifts from first iteration on the round]"]
        elif m == 4:
            seed_ite = 1
            out = [lead + "Area search modus chosen to be of type 4 [read shift origins from first iteration on project]"]
        else:                                          # non-integer in (2,5): o.error + error -> not reproduced
            raise ValueError("area_search_modus %r: Dynamo errors" % m)
        return out + [lead + "Shifts origin in iteration %d will be read from iteration %d" % (ite, seed_ite)]
    if m == 0:
        return [lead + "No shift limitations"]
    if m == 1:
        return [lead + "Shift limits are defined from subtomogram center"]
    if m == 2:
        return [lead + "Shift limits are defined from positions computed on the previous iteration"]
    return [lead + "Warning: speciallized policy for origin of shifts [%s]" % _matlab_d(m)]


def setup_skip_reference_line(ref):
    """dynamo_iteration_setup.m:269 (o.leading 'setup')."""
    return "[setup] skipping reference %d (possibly extinct in previous iteration)" % ref


def setup_empty_table_line(ref):
    """dynamo_iteration_setup.m:274: the text already starts with '[setup]', o adds a second one."""
    return '[setup] [setup] Warning: empty "starting table" for  ref %d' % ref


def setup_no_tags_line(ref, column=2):
    """dynamo_iteration_setup.m:281."""
    return "[setup] [setup] Warning: no  tags to align in ref %d (according to column %d)" % (ref, column)


def setup_done_line(time_start_computation):
    """dynamo_iteration_setup.m:562-566 (o.leading = '' before the echo)."""
    return "[setup] done with general set up of card files, starting computations: %s" % time_start_computation


# =====================================================================================================
# MATLAB side: compute and assemble
# =====================================================================================================

def returning_control_line(t_datestr):
    """dynamo_pipeline_system_check_output_gpu.m:102-104."""
    return " Returning control to Matlab  at %s. GPU round was successful" % t_datestr


def assemble_head_lines():
    """dynamo_iteration_assemble.m:137, 164, 267, 277 (o.leading 'assemble'). 'Applying  thresholding:' has 2
    spaces in the source."""
    return ["[assemble] Gathering refined tables.",
            "[assemble] Computing table comparisons against previous iteration.",
            "[assemble] getting rid of overlapping particles:",
            "[assemble] Applying  thresholding:"]


def assemble_average_lines(ite, nref, current_refs):
    """dynamo_iteration_assemble.m:332-345. The per-ref line passes {ite,iref} to 'ref %d ite %d' (swapped)."""
    out = ["[assemble] Computing average for %d multireference channels." % nref]
    for iref in range(1, nref + 1):
        if iref not in current_refs:
            out.append("[assemble] Skipping extinct reference %d" % iref)
            continue
        out.append("[assemble] Computing average for ref %d ite %d" % (ite, iref))
    return out


def assemble_average_done_line(seconds, average_needed=True):
    """dynamo_iteration_assemble.m:362-369: sprintf('... %f seconds. %\\n') -- MATLAB stops at the invalid '%\\n'
    conversion, so the line ends with 'seconds. ' and no empty line follows."""
    if not average_needed:
        return "average has been skipped"
    return "done assembling intermediate average in %s seconds. " % matlab_f(seconds)


def convergence_lines(next_iteration, kind="not_converged", next_round=None, error_text=None):
    """dynamo_iteration_assemble.m:529-599, one appendlog of the cell array.

    kind: 'not_converged' (some reference did not saturate; next = ite+1), 'converged_end' (all saturated, no later
    round leader), 'next_round' (all saturated; next = first leader after ite, next_round its round), 'error'
    (the try failed: error_text = lasterr, may hold '\\n'), 'no_saturation_info' (saturations empty: assumed not
    saturated, so the not_converged lines follow)."""
    m = []
    if kind == "no_saturation_info":
        m += ["  Saturation could not be extracted.", "  Assuming saturation was not achieved."]
        kind = "not_converged"
    if kind == "not_converged":
        m += ["   ", "At least one reference did not achieve convergence: keeping normal execution path"]
    elif kind == "converged_end":
        m += ["   ", "Project has attained convergence. Next iterations will be skipped "]
    elif kind == "next_round":
        m += ["   ", "All references have saturated, moving to next round %d" % next_round,
              "Next iteration will be %d   (the first on the new round)." % next_iteration]
    elif kind == "error":
        bar = " ----------------------------------------------------------------------  "
        m += [bar, "   ", "[DYNAMO ERROR]: non fatal: recovering", "Convergence computation could not be completed."]
        m += str(error_text).split("\n")
        m += ["Keeping usual execution path for next iteration (%d)." % next_iteration, "   ", bar]
    else:
        raise ValueError(kind)
    m.append(" [Assemble]: preparing next iteration: (%d)" % next_iteration)
    return m


def bandpass_lines(ite, name_project, symmetrized, sym_operator, fourier_pixels, threshold, pushback,
                   next_round_exists):
    """dpkproject.adaptiveFiltering (S/+dpkproject/adaptiveFiltering.m:138-144 o, echoes :213-337 and :371; o.leading
    'bandpass #<ite>', o.intofile = log) for nref == 2 and a successful resolution estimate. The two
    ' shifts/angles on reference 2: %5.2 ...' formats stop at the invalid '%5.2 ' conversion."""
    lead = "[bandpass #%d] " % ite
    out = []
    if symmetrized:
        out.append(lead + "Symmetrizing both references before aligning ref 2 onto ref 1")
    out += [lead + "Aligning reference 2 onto reference 1 before computing resolution.",
            lead + "... done",
            lead + " shifts on reference 2: ",
            lead + " angles on reference 2: "]
    out.append(lead + ('Computing resolution with symmetry "%s"' % sym_operator if symmetrized
                       else "Computing resolution without symmetry."))
    out += [lead + "Adaptive bandpass FSC for resolution estimation saved in file:",
            lead + "./%s/results/ite_%04d/averages/bandpass_fsc_ite_%04d.fsc" % (name_project, ite, ite),
            lead + "Resolution estimated for iteration %d: %5.2f (threshold %5.2f)" % (ite, fourier_pixels, threshold),
            lead + "Low pass computed for next iteration %d: %5.2f (pushback %5.2f)"
            % (ite + 1, fourier_pixels - pushback, pushback)]
    if not next_round_exists:
        out.append(lead + "Could not identify round number for next iteration (%d)" % (ite + 1))
    return out


def bandpass_too_many_references_line(ite):
    """adaptiveFiltering.m:165-168: nref > 2 echoes one line into the log (nref == 1 only disp()s) and returns."""
    return "[bandpass #%d]  ... but you can always write your own plugin!" % ite


def ending_lines(card_ite, time_start_computation, time_begin_str, time_end_iteration):
    """dynamo_iteration_assemble.m:838-849: 'The card %s has been completed \\n' gives the line and an empty line."""
    return [" ", " ", "The card %s has been completed " % card_ite, "",
            "Computations for iterative refinement started at %s" % time_start_computation,
            "Assembling results and averaging  started at %s" % time_begin_str,
            "Iteration finished at %s" % time_end_iteration, " "]


def check_ok_lines(ite):
    """dynamo_iteration_check.m:109-114."""
    return ["iteration %d ok" % ite, " ", " ", " "]


def check_fatal_lines(ite, reason):
    """dynamo_iteration_check.m:118-149 (fatal_error)."""
    bars = ["%" * 59] * 6
    return [" "] + bars + [" ", " FATAL ERROR DETECTED IN ITERATION %d" % ite, "   reason: %s" % reason,
                           "   Deactivating all subsequent iterations.", " "] + bars


def assemble_fatal_lines(ite, reason):
    """dynamo_iteration_assemble.m:858-874 (fatal_error)."""
    return [" FATAL ERROR DETECTED IN ITERATION %d" % ite, "   reason: %s" % reason,
            "   Deactivating subsequent rounds."]


def gpu_failure_lines(order):
    """dynamo_pipeline_system_check_output_gpu.m:42-90: appended when the engine exits != 0 (then error())."""
    a = ["&" * 63] * 8
    b = ["&" * 62] * 8
    return a + ["GPU: Warning!!!",
                "something went wrong inside the GPU application while the system tried to execute the order:",
                "%s " % order, "", "Dynamo will abort project execution at this point."] + b


# =====================================================================================================
# engine: card parsing and the angular quantities it prints
# =====================================================================================================

def extract_field(card_lines, field):
    """settings_card::extract_field (C/dynamo_compute_iteration_spp.cu:1876-1919): the LAST line that contains
    `field` with '=' at index len(field); value = text after '=' up to ';' (or end)."""
    val = ""
    for line in card_lines:
        if line.find(field) < 0:
            continue
        eq = line.find("=")
        if eq != len(field):
            continue
        col = line.find(";")
        val = line[eq + 1:] if col < 0 else line[eq + 1:col]
    return val


def _atoi(s):
    m = re.match(r"\s*([+-]?\d+)", s)
    return int(m.group(1)) if m else 0


def _atof(s):
    m = re.match(r"\s*([+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)", s)
    return float(m.group(1)) if m else 0.0


def _matlab_d(x):
    x = float(x)
    return "%d" % int(x) if x == int(x) else ("%e" % x)


class EngineCard:
    """The card fields the engine prints, parsed as the engine parses them (atoi/atof into int/float)."""

    def __init__(self, path, path_as_given=None):
        self.path = path
        # the card path as Dynamo passes it on the command line (printed by the engine), e.g.
        # './<name>/cards/ite_0001/card_iteref_ref_001_ite_0001.card'
        self.path_as_given = path if path_as_given is None else path_as_given
        with open(path) as f:
            lines = f.read().split("\n")
        g = lambda k: extract_field(lines, k)
        self.name_project = g("name_project")
        self.ite, self.ref = _atoi(g("database_ite")), _atoi(g("database_ref"))
        self.file_tags = g("database_gpu_root_file_tags")
        self.file_particleList = g("database_gpu_root_file_particleList")
        self.folder_data = g("database_gpu_folder_data_iteration")
        self.file_template = g("database_gpu_file_template_iteration")
        self.file_tfmask = g("database_gpu_file_fmask_iteration")
        self.file_mask = g("database_gpu_file_mask")
        self.gpu_identifier_set = g("gpu_identifier_set")
        self.cone_range, self.cone_sampling = np.float32(_atof(g("cone_range"))), np.float32(_atof(g("cone_sampling")))
        self.inplane_range = np.float32(_atof(g("inplane_range")))
        self.inplane_sampling = np.float32(_atof(g("inplane_sampling")))
        self.cone_flip, self.inplane_flip = _atoi(g("cone_flip")), _atoi(g("inplane_flip"))
        self.rings = _atoi(g("rings"))
        self.refine = _atoi(g("refine"))
        self.refine_factor = np.float32(_atof(g("refine_factor")))
        self.padding = _atoi(g("feature_data_padding"))

    def devices(self):
        """main(): `while (os >> temp)` over the gpu_identifier_set string, atoi of each token."""
        return [(tok, _atoi(tok)) for tok in self.gpu_identifier_set.split()]


_libm = ctypes.CDLL(ctypes.util.find_library("m"))
for _n in ("cosf", "sinf", "acosf", "ceilf"):
    getattr(_libm, _n).restype = ctypes.c_float
    getattr(_libm, _n).argtypes = [ctypes.c_float]
F = np.float32
DEG2RAD_F = F(0.0174532925199432954743716805978692718781530857086181640625)
RAD2DEG_F = F(57.29577951308232286464772187173366546630859375)


def _fill_interval(first, interval, last):
    """fill_interval (C/classes/gpuAngles.cpp:285-311) in float."""
    first, interval, last = F(first), F(interval), F(last)
    if interval == 0:
        return [F(0)]
    n = int(F(F(last - first) / interval) + F(1))
    return [F(first + F(F(i) * interval)) for i in range(max(n, 0))]


def tdrot_prints(cone_range, cone_sampling):
    """The (cone_sampling_tdrot, coneSamplingEquispaced) pairs printed by ONE orientation_set::fill_orientations
    (C/classes/gpuAngles.cpp:343-435): one pair per tilt_north of fill_interval(0, cone_sampling, aperture_half)
    with tilt_north not 0/180 and arg^2 < 1. Depends only on the level's cone_range/cone_sampling (not on the
    seed orientation, the in-plane range or the flips). All float math, glibc cosf/sinf/acosf/ceilf."""
    cone_range, cs = F(cone_range), F(cone_sampling)
    aperture_half = F(cone_range / F(2.0))
    if cone_range == F(180.0):
        aperture_half = F(179.999)
    out = []
    cos_cs = F(_libm.cosf(F(cs * DEG2RAD_F)))
    for tn in _fill_interval(0, cs, aperture_half):
        c = F(_libm.cosf(F(tn * DEG2RAD_F)))
        s = F(_libm.sinf(F(tn * DEG2RAD_F)))
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = F(F(cos_cs - F(c * c)) / F(s * s))
        if tn == 0 or tn == 180:
            continue
        if F(arg * arg) >= 1:                          # as in C: a NaN arg is NOT skipped (it would print 'nan')
            continue
        cst = F(F(_libm.acosf(arg)) * RAD2DEG_F)
        eq = F(F(360) / F(_libm.ceilf(F(F(360) / cst))))
        out.append((cst, eq))
    return out


def level_ranges(card):
    """scan_range of each refinement level: level 0 from the card, then scan_range::shrink(refine_factor)
    (gpuAngles.cpp:51-63) `refine` times. Returns [(cone_range, cone_sampling, inplane_range, inplane_sampling)]."""
    cr, cs, ir, isa = F(card.cone_range), F(card.cone_sampling), F(card.inplane_range), F(card.inplane_sampling)
    f = F(card.refine_factor)
    out = [(cr, cs, ir, isa)]
    for _ in range(card.refine):
        if cr != 0:
            cr = F(cs * f)
        cs = F(cs / F(2.0))
        if ir != 0:
            ir = F(isa * f)
        isa = F(isa / F(2.0))
        out.append((cr, cs, ir, isa))
    return out


def fill_print_lines(cone_range, cone_sampling):
    """The stdout lines of one fill_orientations call: 'cone sampling tdrot<g>' then 'Equispaced<g>' per tilt."""
    out = []
    for cst, eq in tdrot_prints(cone_range, cone_sampling):
        out += ["cone sampling tdrot" + cout_float(cst), "Equispaced" + cout_float(eq)]
    return out


def per_particle_lines(card):
    """One particle's refinement_process (dynamo_compute_iteration_spp.cu:2530-2602): `refine`+1 align() calls,
    each building one orientation_set (:3628)."""
    out = []
    for cr, cs, _, _ in level_ranges(card):
        out += fill_print_lines(cr, cs)
    return out


def multigrid_counts(card):
    """main()'s ' multigrid step: %d  totaling %d triplets' values (dynamo_compute_iteration_spp.cu:4824-4841):
    orientation_set((0,0,0), range) with the card's flips kept at EVERY level (main never resets them), so for
    cone_flip > 0 the counts are twice the cone part. Uses pydyn.angles.orientation_set (the verified port)."""
    from . import angles as A
    rng = A.ScanRange(card.cone_range, card.cone_sampling, card.inplane_range, card.inplane_sampling,
                      card.cone_flip, card.inplane_flip)
    counts = [len(A.orientation_set((0.0, 0.0, 0.0), rng))]
    for _ in range(card.refine):
        rng.shrink(card.refine_factor)
        counts.append(len(A.orientation_set((0.0, 0.0, 0.0), rng)))
    return counts


# =====================================================================================================
# engine stdout
# =====================================================================================================

class DeviceRun:
    """One device's share of an engine run.
    device: physical id from gpu_identifier_set; particle_lines: the lines of its assign_gpu_particleList file
    (only lines longer than 4 characters are processed, :2110); gpu_name / multiprocessors: properties of device 0
    of the visible set (cudaGetDeviceProperties(&p, 0), :4726) -- run dependent."""

    def __init__(self, device, particle_lines, gpu_name="", multiprocessors=0):
        self.device = int(device)
        self.particle_lines = [p for p in particle_lines if len(p) > 4]
        self.gpu_name, self.multiprocessors = gpu_name, int(multiprocessors)


def _load_block(card, device, dims):
    """iteration::load (dynamo_compute_iteration_spp.cu:1988-2038). dims: {'template','mask','tfmask'}:(nx,ny,nz)."""
    t, m, f = dims["template"], dims["mask"], dims["tfmask"]
    gpustr = "gpu_%02d" % device
    tag = "ite_%04d" % card.ite
    start_tbl = "%s/temp/%s/intermediate_results/starting_gpuProcRefTable_%s_ref_%03d_proc_%05d_%s.btbl" % (
        card.name_project, tag, tag, card.ref, 0, gpustr)
    return ["Ready to load auxiliary files: template, mask, fmask",
            "Reference file read with %d x %d x %d = %d elements " % (t[0], t[1], t[2], t[0] * t[1] * t[2]),
            "loading reference from file " + card.file_template,
            "loading mask from file " + card.file_mask,
            "Mask file read with %d x %d x %d = %d elements " % (m[0], m[1], m[2], m[0] * m[1] * m[2]),
            "tfmask file read with %d x %d x %d = %d elements " % (f[0], f[1], f[2], f[0] * f[1] * f[2]),
            "loading tfmask from file " + card.file_tfmask,
            "Ready to load Starting table",
            "Loading: starting table for this gpu device and processor from file: " + start_tbl]


def _loop_block(card, dev, particle_lines_each):
    """iteration::loop (dynamo_compute_iteration_spp.cu:2046-2189)."""
    d2 = "%02d" % dev.device
    out = [" Original file_tags " + card.file_tags,
           "   Current particleListFile for the selected gpu device " + card.file_particleList + d2 + ".txt"]
    for k, line in enumerate(dev.particle_lines):
        if k < 3:
            out.append("  [showing only first 3 assignations]... particle file %s to be aligned by device %s"
                       % (line, d2))
        out += particle_lines_each
        if (k + 1) % 1000 == 0:
            out.append("saving intermediate results for %d particles" % (k + 1))
    return out


def device_block(card, dev, dims, particle_lines_each=None):
    """All the lines one device thread prints, in its own order (branched_routine, :4717-4734)."""
    if particle_lines_each is None:
        particle_lines_each = per_particle_lines(card)
    return ([" Your card %s -identified as %d in your system- has %d multiprocessors "
             % (dev.gpu_name, dev.device, dev.multiprocessors),
             "I am device %d on CPU node %d with orders to open %s" % (dev.device, 0, card.path_as_given)]
            + _load_block(card, dev.device, dims) + _loop_block(card, dev, particle_lines_each))


def engine_header(card, counts=None, t_enter=None):
    """main() up to the thread start (dynamo_compute_iteration_spp.cu:4736-4842), as stdout TEXT (str).
    t_enter: asctime string. The trailing printf(" \\n ") leaves ' ' pending: the next fragment (of whichever
    thread prints first) starts after it."""
    counts = multigrid_counts(card) if counts is None else counts
    s = "Date/time when entering the GPU iteration was: %s\n   \n" % t_enter
    s += " Text file that controls current iteration: %s\n" % card.path_as_given
    s += " \n GPU system settings \n ------------------- \n \n"
    s += "   Project ((parameter 'gpu_identifier_set') is invoking devices: %s\n" % card.gpu_identifier_set
    s += "   parsing the requested GPU identifiers\n"
    devs = card.devices()
    for c, (tok, d) in enumerate(devs):
        s += "%s\n   %d: device_id:%d selected \n" % (tok, c, d)
    s += "   You asked for a total of %d GPUS \n" % len(devs)
    s += " \n  Angular scan parameters: \n ------------------------ \n \n "
    s += "scanned directions will be defined inside this cone: \n"
    s += " cone_range       : %s \n" % c_f(card.cone_range)
    s += " cone_sampling    : %s \n" % c_f(card.cone_sampling)
    s += " cone_flip        : %d \n" % card.cone_flip
    s += " inplane_range    : %s \n" % c_f(card.inplane_range)
    s += " inplane_sampling : %s \n" % c_f(card.inplane_sampling)
    s += " inplane_flip     : %d \n" % card.inplane_flip
    s += " rings            : %d \n" % card.rings
    for lev, (cr, cs, _, _) in enumerate(level_ranges(card)):
        s += "".join(x + "\n" for x in fill_print_lines(cr, cs))
        s += " multigrid step: %d  totaling %d triplets \n" % (lev, counts[lev])
    s += " \n "
    return s


def engine_stdout(card, devices, dims, t_enter, t_leave, cpu_seconds, counts=None):
    """The complete stdout of one `dynamo_compute_iteration_spp <card_iteref>` call, deterministic equivalent:
    device blocks whole, in gpu_identifier_set order. devices: [DeviceRun] in that order (len == number of ids).
    t_enter / t_leave: asctime strings; cpu_seconds: clock() difference (run dependent)."""
    each = per_particle_lines(card)
    s = engine_header(card, counts, t_enter)
    if len(devices) > 1:
        for dev in devices:
            s += "".join(x + "\n" for x in device_block(card, dev, dims, each))
        s += " cpu time invested in  multiGPU modus: %s seconds \n" % c_f(cpu_seconds)
    else:                                                   # single GPU (:4895-4904) [no ground-truth log]
        dev = devices[0]
        s += "".join(x + "\n" for x in _load_block(card, dev.device, dims))
        s += "name_project %s\n" % card.name_project
        s += "".join(x + "\n" for x in _loop_block(card, dev, each))
        s += " cpu time invested analyzing all particles in one single GPU (%d): %s seconds \n" % (
            dev.device, c_f(cpu_seconds))
    s += "Date/time when leaving  the GPU iteration was: %s\n " % t_leave
    return s


def stdout_to_log_lines(stdout):
    """dynamo_aux_appendlog(standard_output) writes fprintf('%s\\n', standard_output)."""
    return (stdout + "\n").split("\n")[:-1]


# =====================================================================================================
# writer
# =====================================================================================================

def append_log(project_dir, lines):
    """The only side effect of this module: append lines to <project>/log.txt ('a', '%s\\n' each)."""
    with open(os.path.join(project_dir, "log.txt"), "a") as f:
        f.write("".join(x + "\n" for x in lines))


# =====================================================================================================
# reconstruction of the expected log.txt of a finished project from its own files
# =====================================================================================================

def em_dims(path):
    """nx, ny, nz from an EM header (bytes 4..16, little endian int32)."""
    with open(path, "rb") as f:
        h = f.read(16)
    return tuple(int(x) for x in np.frombuffer(h[4:16], "<i4"))


def _num(tok):
    """Real part of a dlmwrite token (1.1.514 may leave complex entries such as '1.5+0.2i')."""
    m = re.match(r"[+-]?(nan|inf|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)", tok, re.I)
    return float(m.group(0)) if m else float("nan")


def read_table_cols12(path):
    """Columns 1 (tag) and 2 (aligned flag) of a .tbl text table."""
    tags, flags = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2:
                tags.append(int(_num(t[0])))
                flags.append(_num(t[1]))
    return tags, flags


def read_card_dict(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v[:-1] if v.endswith(";") else v
    return out


def load_vpr(path):
    import scipy.io as sio
    s = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["card"]
    out = {}
    for f in s._fieldnames:
        v = getattr(s, f)
        if isinstance(v, np.ndarray) and v.dtype.kind == "U":
            v = "".join(v.ravel().tolist())
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        elif isinstance(v, np.generic):
            v = v.item()
        out[f] = v
    return out


def _str2num_vector(s):
    vals = []
    for tok in s.replace(",", " ").split():
        if ":" in tok:
            p = [float(x) for x in tok.split(":")]
            a, st, b = (p[0], 1.0, p[1]) if len(p) == 2 else (p[0], p[1], p[2])
            vals += list(np.arange(a, b + st / 2, st))
        else:
            vals.append(float(tok))
    return vals


def round_array(vpr):
    """dynamo_zaux_intertwin_round_identifier_array.m:54-102 (same port as dynproject.round_array)."""
    s = str(vpr["intertwin"])
    if s.startswith("/"):
        s = s[1:]
    parts = [_str2num_vector(p) for p in s.split("/")]
    heads = [int(p[0]) for p in parts] + [9]
    arr = []
    for i in range(len(heads) - 1):
        for _ in range(int(vpr["ite_r%d" % heads[i]])):
            arr += list(range(heads[i], heads[i + 1]))
    return arr


def round_leaders(arr):
    """dynamo_vpr_status(vpr,'round_leaders') (dynamo_vpr_status.m:336-366, as dynproject.round_leaders)."""
    lead = [1]
    for i in range(2, len(arr) + 1):
        if arr[i - 1] != arr[i - 2] and arr[i - 2] not in arr[i - 1:]:
            lead.append(i)
    return lead


def gpu_chunks(tags, devices, how_many_processors=1):
    """dynamo__iteration_setup_gpu.m:222-273: ceil(n/(hmp*G)) tags per chunk, chunk c on gpu_identifier_set[c]
    (only proc 0 is read by the engine)."""
    per = math.ceil(len(tags) / (how_many_processors * len(devices)))
    return [tags[i * per:(i + 1) * per] for i in range(len(devices))]


def _project_file(project_dir, cwd, name_project, path):
    """A card path ('./<name>/results/...', relative to Dynamo's working directory) on disk. A project folder
    that was renamed after the run (the user's bin8_align_1 still says name_project=bin4_align_1) is found
    through project_dir."""
    for pre in ("./%s/" % name_project, "%s/" % name_project):
        if path.startswith(pre):
            return os.path.join(project_dir, path[len(pre):])
    return path if os.path.isabs(path) else os.path.join(cwd, path)


def _data_padding(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "particle_*.em")))
    if not files:
        raise FileNotFoundError("no particle_*.em in %s: cannot determine the particle name padding" % data_dir)
    return len(os.path.basename(files[0])[len("particle_"):-3])


class RunValues:
    """Run-dependent values (dates, times, durations, GPU model, multiprocessor count). Missing values fall back to
    placeholders derived from the card times, which the checker normalises away. A live pyDynamo run does not need
    this class: it passes its own clock and device query straight to the formatters.
    keys of `values`: ('t_start', ite) datestr of setup entry; ('t_enter', ite, ref) / ('t_leave', ite, ref)
    asctime strings of the engine run; ('cpu', ite, ref) seconds; ('t_return', ite, ref) datestr after the engine;
    ('average_s', ite) seconds of the averaging."""

    def __init__(self, gpu_name="<GPU>", multiprocessors=0, values=None):
        self.gpu_name, self.multiprocessors = gpu_name, multiprocessors
        self.v = values or {}

    def get(self, key, default):
        return self.v.get(key, default)


def expected_log_lines(project_dir, cwd=None, data_dir=None, iterations=None, run=None, convergence=None,
                       name_project=None):
    """The expected log.txt lines of ONE complete execution (iterations 1..N, or the given list) of a finished
    1.1.514 standalone_gpu project, rebuilt from the project's own files:
      cards/ite_N/card_ite_ite_N.card (time_start_computation, time_end_refinement, time_end_iteration, nref,
      area_search_modus, freeze flags) and card_iteref_ref_R_ite_N.card (every engine field);
      results/ite_N/currently_surviving_references_ite_N.txt; results/ite_N/starting_values/starting_table_*
      (the GPU chunks: temp/ is deleted by assemble) and the *_transformed .em headers (dimensions);
      settings/virtual_project.mat (rounds, adaptive bandpass parameters); results/bandpass_resolution.txt.
    cwd: Dynamo's working directory (default: the project's parent); data_dir: the particle folder (default
    cwd/<folder_data>), used only for the file-name padding. convergence: {ite: dict(kind=..., error_text=...)}
    (default 'not_converged', the branch of a convergence_type-0 project). run: RunValues."""
    pd = os.path.abspath(project_dir)
    cwd = cwd or os.path.dirname(pd)
    run = run or RunValues()
    convergence = convergence or {}
    vpr = load_vpr(os.path.join(pd, "settings", "virtual_project.mat"))
    arr = round_array(vpr)
    N = len(arr)
    iterations = list(range(1, N + 1)) if iterations is None else list(iterations)
    res_file = os.path.join(pd, "results", "bandpass_resolution.txt")
    resolutions = None
    if os.path.exists(res_file):
        with open(res_file) as f:
            resolutions = [_num(t) for t in f.read().split()]
    lines = []
    for ite in iterations:
        rnd = arr[ite - 1]
        ci = read_card_dict(os.path.join(pd, "cards", "ite_%04d" % ite, "card_ite_ite_%04d.card" % ite))
        name = name_project or ci["name_project"]
        cip = card_ite_path(name, ite)
        tsc = ci.get("time_start_computation", "RECORDING PROBLEMS IN THE GPU")
        tbeg = ci.get("time_end_refinement", tsc)
        tend = ci.get("time_end_iteration", tsc)
        lines += setup_banner(ite, cip, run.get(("t_start", ite), tsc))
        # freeze (angles), then shifts
        has_ff = "cone_freeze_reference_r1" in vpr
        leaders = sorted(round_leaders(arr))
        begin = ([x for x in leaders if x < ite] or [ite])[-1]
        if has_ff and ite > 1:
            fc, fi = float(vpr["cone_freeze_reference_r%d" % rnd]), float(vpr["inplane_freeze_reference_r%d" % rnd])
            if fc == 0 and fi == 0:
                lines += setup_freeze_lines(ite, True, None, None)
            else:
                sel = lambda f: {0: ite, 1: begin, 2: 1}[int(f)]
                lines += setup_freeze_lines(ite, True, sel(fc), sel(fi))
        else:
            lines += setup_freeze_lines(ite, has_ff, None, None)
        lines += setup_shifts_lines(ite, _num(ci["area_search_modus"]), begin)
        refs_txt = os.path.join(pd, "results", "ite_%04d" % ite, "currently_surviving_references_ite_%04d.txt" % ite)
        with open(refs_txt) as f:
            refs = [int(_num(t)) for t in f.read().split()]
        nref = int(_num(ci["nref"]))
        for r in range(1, nref + 1):
            if r not in refs:
                lines.append(setup_skip_reference_line(r))
        lines.append(setup_done_line(tsc))
        # compute: one engine run per surviving reference
        for ref in refs:
            cpath = os.path.join(pd, "cards", "ite_%04d" % ite, "card_iteref_ref_%03d_ite_%04d.card" % (ref, ite))
            card = EngineCard(cpath, card_iteref_path(name, ref, ite))
            dims = {"template": em_dims(_project_file(pd, cwd, card.name_project, card.file_template)),
                    "mask": em_dims(_project_file(pd, cwd, card.name_project, card.file_mask)),
                    "tfmask": em_dims(_project_file(pd, cwd, card.name_project, card.file_tfmask))}
            tags, flags = read_table_cols12(os.path.join(
                pd, "results", "ite_%04d" % ite, "starting_values", "starting_table_ref_%03d_ite_%04d.tbl" % (ref, ite)))
            sel_tags = [t for t, fl in zip(tags, flags) if fl != 0]
            devs = card.devices()
            folder = card.folder_data.rstrip("/")
            pad = _data_padding(data_dir or os.path.join(cwd, folder))
            chunks = gpu_chunks(sel_tags, devs, int(_num(ci.get("how_many_processors", "1"))))
            druns = [DeviceRun(d, ["%s/particle_%0*d.em" % (folder, pad, t) for t in ch], run.gpu_name,
                               run.multiprocessors) for (_, d), ch in zip(devs, chunks)]
            t_enter = run.get(("t_enter", ite, ref), asctime(datestr_to_struct(tsc)))
            t_leave = run.get(("t_leave", ite, ref), asctime(datestr_to_struct(tbeg)))
            out = engine_stdout(card, druns, dims, t_enter, t_leave, run.get(("cpu", ite, ref), 0.0))
            lines += stdout_to_log_lines(out)
            lines.append(returning_control_line(run.get(("t_return", ite, ref), tbeg)))
        # assemble
        lines += assemble_head_lines()
        lines += assemble_average_lines(ite, nref, refs)
        lines.append(assemble_average_done_line(run.get(("average_s", ite), 0.0)))
        conv = dict(convergence.get(ite, {}))
        kind = conv.pop("kind", "not_converged")
        nxt = conv.pop("next_iteration", ite + 1)
        lines += convergence_lines(nxt, kind, **conv)
        if int(_num(str(vpr.get("adaptive_bandpass", 0)))):            # dynamo_iteration_assemble.m:813-822
            if nref > 2:
                lines.append(bandpass_too_many_references_line(ite))
            elif nref == 2:
                if resolutions is None or len(resolutions) < ite:
                    raise FileNotFoundError("results/bandpass_resolution.txt lacks iteration %d" % ite)
                lines += bandpass_lines(ite, name, bool(vpr.get("adaptive_bandpass_symmetrized", 0)),
                                        str(vpr["sym_r%d" % rnd]), resolutions[ite - 1],
                                        float(vpr["adaptive_bandpass_threshold"]),
                                        float(vpr["adaptive_bandpass_pushback"]), ite + 1 <= N)
        lines += ending_lines(cip, tsc, tbeg, tend)
        lines += check_ok_lines(ite)
    return lines


# =====================================================================================================
# project folder layout (for checks; spec section 1)
# =====================================================================================================

def project_tree(project_dir):
    """Sorted relative paths (dirs end with '/') under a project folder."""
    out = []
    for root, dirs, files in os.walk(project_dir):
        rel = os.path.relpath(root, project_dir)
        for d in dirs:
            out.append(os.path.normpath(os.path.join(rel, d)) + "/")
        for f in files:
            out.append(os.path.normpath(os.path.join(rel, f)))
    return sorted(out)


PYDYN_ONLY_FILES = ("pydyn_log.txt", "pydyn_report.json")
