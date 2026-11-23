"""CPU tests of pydynamo.project.logfile: the Dynamo 1.1.514 log.txt / engine stdout formats (spec docs/analysis/10f).

Literal expectations are copied from Dynamo's own logs:
  [L1] benchmarks/05_abp_bin4_eo/dynamo/run/bin4_align_1_eo/log.txt (Dynamo 1.1.514, 2 refs, 4 iterations)
  [L2] test_data/bin8_filament/bin8_align_1/log.txt (the user's run of 26-Jun, from line 2565356)
  [L3] test_data/bin8_filament/abp_align_eo/log.txt (the user's run of 26-Jun, from line 962541)
  [G]  benchmarks/05_abp_bin4_eo/dynamo/run/gpulog.txt (engine stdout of ite 4 ref 2 of [L1])
"""
import os
import re
import time
import collections

import pytest

from pydynamo.project import logfile as D

from expected_log import expected_log_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tm(y, mo, d, h, mi, s):
    return time.localtime(time.mktime((y, mo, d, h, mi, s, 0, 0, -1)))


def _card(tmp_path, **kw):
    f = dict(name_project="bin4_align_1_eo", database_ite=1, database_ref=1,
             database_gpu_folder_data_iteration="filamentsData_ctf",
             database_gpu_file_template_iteration="./bin4_align_1_eo/results/ite_0001/starting_values/"
                                                  "starting_reference_transformed_ref_001_ite_0001.em",
             database_gpu_file_fmask_iteration="./bin4_align_1_eo/results/ite_0001/starting_values/"
                                               "fmask_starting_reference_transformed_ref_001_ite_0001.em",
             database_gpu_file_mask="./bin4_align_1_eo/results/ite_0001/starting_values/mask_transformed_ite_0001.em",
             database_gpu_root_file_tags="./bin4_align_1_eo/temp/ite_0001/task_assignations/"
                                         "assign_gpu_proc_00000_ite_0001_ref_001_gpu_",
             database_gpu_root_file_particleList="./bin4_align_1_eo/temp/ite_0001/task_assignations/"
                                                 "assign_gpu_particleList_proc_00000_ite_0001_ref_001_gpu_",
             cone_range=20, cone_sampling=3, cone_flip=1, inplane_range=360, inplane_sampling=15, inplane_flip=0,
             refine=2, refine_factor=2, rings=0, gpu_identifier_set="0  1  2  3", feature_data_padding=6)
    f.update(kw)
    p = tmp_path / "card_iteref_ref_001_ite_0001.card"
    p.write_text("".join("%s=%s;\n" % (k, v) for k, v in f.items()))
    return D.EngineCard(str(p), "./%s/cards/ite_%04d/card_iteref_ref_%03d_ite_%04d.card"
                        % (f["name_project"], f["database_ite"], f["database_ref"], f["database_ite"]))


# ------------------------------------------------------------------------------------------------ values
def test_datestr_and_asctime():
    t = _tm(2026, 11, 12, 1, 12, 28)
    assert D.datestr(t) == "12-Nov-2026 01:12:28"                              # [L1:5]
    assert D.asctime(t) == "Thu Nov 12 01:12:28 2026"                          # [L1:8]
    assert D.asctime(_tm(2026, 6, 5, 9, 3, 7)) == "Fri Jun  5 09:03:07 2026"   # asctime pads the day to width 3
    assert D.datestr(D.datestr_to_struct("26-Jun-2026 16:51:17")) == "26-Jun-2026 16:51:17"


def test_cout_float_is_g6():
    assert D.cout_float(60.0) == "60"
    assert D.cout_float(27.692308) == "27.6923"
    assert D.c_f(20) == "20.000000" and D.matlab_f(59.947974) == "59.947974"


def test_tdrot_prints_per_level():
    """[G] lines 31-48 and [L3]: the pairs depend only on the level's cone range / sampling."""
    s = lambda cr, cs: [(D.cout_float(a), D.cout_float(b)) for a, b in D.tdrot_prints(cr, cs)]
    assert s(20, 3) == [("60.0228", "60"), ("29.0062", "27.6923"), ("19.2663", "18.9474")]
    assert s(6, 1.5) == [("60.0066", "60"), ("28.9665", "27.6923")]
    assert s(3, 0.75) == [("60.0065", "60"), ("28.9577", "27.6923")]
    assert s(1.5, 0.375) == [("60.0309", "60"), ("28.9651", "27.6923")]           # [L3] multigrid step 3
    assert s(0, 3) == [] and s(20, 0) == []                                        # tilt 0 only: no print


def test_level_ranges_and_per_particle_lines(tmp_path):
    c = _card(tmp_path)
    assert [tuple(float(x) for x in r[:2]) for r in D.level_ranges(c)] == [(20, 3), (6, 1.5), (3, 0.75)]
    pp = D.per_particle_lines(c)
    assert len(pp) == 14 and pp[0] == "cone sampling tdrot60.0228" and pp[-1] == "Equispaced27.6923"


def test_multigrid_counts(tmp_path):
    assert D.multigrid_counts(_card(tmp_path)) == [1888, 188, 188]                                    # [L1:44-54]
    assert D.multigrid_counts(_card(tmp_path, inplane_range=45, inplane_sampling=5)) == [684, 190, 190]  # [G]
    assert D.multigrid_counts(_card(tmp_path, cone_flip=2, inplane_flip=2, inplane_range=90, inplane_sampling=5,
                                               refine=3)) == \
        [2880, 380, 380, 380]                                                                         # [L3]


def test_extract_field_engine_semantics():
    lines = ["refine=2;", "refine_factor=3;", "xrefine=9;", "gpu_identifier_set=0  1  2  3;", "refine=4"]
    assert D.extract_field(lines, "refine") == "4"                 # the LAST matching line, ';' optional
    assert D.extract_field(lines, "refine_factor") == "3"          # '=' must sit at len(field)
    assert D.extract_field(lines, "gpu_identifier_set") == "0  1  2  3"
    assert D.extract_field(lines, "missing") == ""


# ------------------------------------------------------------------------------------------------ engine
L1_HEADER = "\n".join([                                                    # [L1:8-55] + the pending ' ' of :56
    "Date/time when entering the GPU iteration was: Thu Nov 12 01:12:28 2026",
    "   ",
    " Text file that controls current iteration: ./bin4_align_1_eo/cards/ite_0001/card_iteref_ref_001_ite_0001.card",
    " ",
    " GPU system settings ",
    " ------------------- ",
    " ",
    "   Project ((parameter 'gpu_identifier_set') is invoking devices: 0  1  2  3",
    "   parsing the requested GPU identifiers",
    "0", "   0: device_id:0 selected ",
    "1", "   1: device_id:1 selected ",
    "2", "   2: device_id:2 selected ",
    "3", "   3: device_id:3 selected ",
    "   You asked for a total of 4 GPUS ",
    " ",
    "  Angular scan parameters: ",
    " ------------------------ ",
    " ",
    " scanned directions will be defined inside this cone: ",
    " cone_range       : 20.000000 ",
    " cone_sampling    : 3.000000 ",
    " cone_flip        : 1 ",
    " inplane_range    : 360.000000 ",
    " inplane_sampling : 15.000000 ",
    " inplane_flip     : 0 ",
    " rings            : 0 ",
    "cone sampling tdrot60.0228", "Equispaced60",
    "cone sampling tdrot29.0062", "Equispaced27.6923",
    "cone sampling tdrot19.2663", "Equispaced18.9474",
    " multigrid step: 0  totaling 1888 triplets ",
    "cone sampling tdrot60.0066", "Equispaced60",
    "cone sampling tdrot28.9665", "Equispaced27.6923",
    " multigrid step: 1  totaling 188 triplets ",
    "cone sampling tdrot60.0065", "Equispaced60",
    "cone sampling tdrot28.9577", "Equispaced27.6923",
    " multigrid step: 2  totaling 188 triplets ",
    " ",
    " "])


def test_engine_header_is_log1(tmp_path):
    c = _card(tmp_path)
    assert D.engine_header(c, None, "Thu Nov 12 01:12:28 2026") == L1_HEADER


def test_device_block_lines(tmp_path):
    """[L1:56-87] device 0 lines (order within one thread)."""
    c = _card(tmp_path)
    dims = {"template": (64, 64, 64), "mask": (64, 64, 64), "tfmask": (64, 64, 64)}
    dev = D.DeviceRun(0, ["filamentsData_ctf/particle_%06d.em" % t for t in (2, 4, 6, 8)], "NVIDIA A40", 84)
    b = D.device_block(c, dev, dims)
    sv = "./bin4_align_1_eo/results/ite_0001/starting_values/"
    assert b[:14] == [
        " Your card NVIDIA A40 -identified as 0 in your system- has 84 multiprocessors ",
        "I am device 0 on CPU node 0 with orders to open ./bin4_align_1_eo/cards/ite_0001/card_iteref_ref_001_ite_0001.card",
        "Ready to load auxiliary files: template, mask, fmask",
        "Reference file read with 64 x 64 x 64 = 262144 elements ",
        "loading reference from file " + sv + "starting_reference_transformed_ref_001_ite_0001.em",
        "loading mask from file " + sv + "mask_transformed_ite_0001.em",
        "Mask file read with 64 x 64 x 64 = 262144 elements ",
        "tfmask file read with 64 x 64 x 64 = 262144 elements ",
        "loading tfmask from file " + sv + "fmask_starting_reference_transformed_ref_001_ite_0001.em",
        "Ready to load Starting table",
        "Loading: starting table for this gpu device and processor from file: bin4_align_1_eo/temp/ite_0001/"
        "intermediate_results/starting_gpuProcRefTable_ite_0001_ref_001_proc_00000_gpu_00.btbl",
        " Original file_tags ./bin4_align_1_eo/temp/ite_0001/task_assignations/assign_gpu_proc_00000_ite_0001_ref_001_gpu_",
        "   Current particleListFile for the selected gpu device ./bin4_align_1_eo/temp/ite_0001/task_assignations/"
        "assign_gpu_particleList_proc_00000_ite_0001_ref_001_gpu_00.txt",
        "  [showing only first 3 assignations]... particle file filamentsData_ctf/particle_000002.em to be aligned by "
        "device 00"]
    showing = [x for x in b if "[showing only first 3" in x]
    assert len(showing) == 3                                     # only the first 3 particles (:2122-2125)
    assert sum(x.startswith("cone sampling tdrot") for x in b) == 4 * 7


def test_saving_every_1000_and_tail(tmp_path):
    c = _card(tmp_path, gpu_identifier_set="0  1")
    dims = {"template": (8, 8, 8), "mask": (8, 8, 8), "tfmask": (8, 8, 8)}
    devs = [D.DeviceRun(0, ["d/particle_%06d.em" % t for t in range(1, 2002)], "G", 1),
            D.DeviceRun(1, ["d/particle_%06d.em" % t for t in range(3000, 3002)] + ["", "abc"], "G", 1)]
    out = D.engine_stdout(c, devs, dims, "Thu Nov 12 01:12:28 2026", "Thu Nov 12 01:44:11 2026", 5718.306641)
    lines = D.stdout_to_log_lines(out)
    assert [x for x in lines if x.startswith("saving")] == ["saving intermediate results for 1000 particles",
                                                            "saving intermediate results for 2000 particles"]
    assert lines[-3:] == [" cpu time invested in  multiGPU modus: 5718.306641 seconds ",
                          "Date/time when leaving  the GPU iteration was: Thu Nov 12 01:44:11 2026", " "]
    first = [x for x in lines if "Your card" in x]
    assert first[0].startswith("  Your card") and first[1].startswith(" Your card")   # pending ' ' of printf(" \n ")
    # (2001 + 2) particles + main(): 7 pairs each; lines of <= 4 characters are skipped by the engine (:2104)
    assert sum(x.startswith("cone sampling tdrot") for x in lines) == (2001 + 2 + 1) * 7


def test_single_gpu_path(tmp_path):
    c = _card(tmp_path, gpu_identifier_set="2")
    dims = {"template": (8, 8, 8), "mask": (8, 8, 8), "tfmask": (8, 8, 8)}
    out = D.engine_stdout(c, [D.DeviceRun(2, ["d/particle_000001.em"])], dims, "A", "B", 1.5)
    lines = D.stdout_to_log_lines(out)
    assert " Ready to load auxiliary files: template, mask, fmask" in lines      # after the pending ' '
    assert "name_project bin4_align_1_eo" in lines
    assert not any("Your card" in x for x in lines)
    assert " cpu time invested analyzing all particles in one single GPU (2): 1.500000 seconds " in lines


def test_gpu_chunks():
    tags = list(range(1, 4712))
    ch = D.gpu_chunks(tags, [0, 1, 2, 3])
    assert [len(x) for x in ch] == [1178, 1178, 1178, 1177]                 # [eo_dynamo514_run.log:53]


# ------------------------------------------------------------------------------------------------ MATLAB side
def test_setup_lines():
    assert D.setup_banner(1, D.card_ite_path("bin4_align_1_eo", 1), "12-Nov-2026 01:12:10") == [   # [L1:1-6]
        "************************************************",
        "**********    NEW ITERATION      ***************",
        "**********    iteration: 1      ",
        "************************************************",
        "Starting iteration at 12-Nov-2026 01:12:10",
        "[setup] Parameter set is in ./bin4_align_1_eo/cards/ite_0001/card_ite_ite_0001.card"]
    assert D.setup_shifts_lines(1, 4) == []
    assert D.setup_shifts_lines(2, 4) == [                                                           # [L1 ite 2]
        "[setup:shifts] Area search modus chosen to be of type 4 [read shift origins from first iteration on project]",
        "[setup:shifts] Shifts origin in iteration 2 will be read from iteration 1"]
    assert D.setup_shifts_lines(3, 3, seed_ite=1)[-1] == "[setup:shifts] Shifts origin in iteration 3 will be read from iteration 1"
    assert D.setup_shifts_lines(2, 1) == ["[setup:shifts] Shift limits are defined from subtomogram center"]
    assert D.setup_shifts_lines(2, 11) == ["[setup:shifts] Warning: speciallized policy for origin of shifts [11]"]
    assert D.setup_done_line("12-Nov-2026 01:12:28") == \
        "[setup] done with general set up of card files, starting computations: 12-Nov-2026 01:12:28"


def test_setup_freeze_lines():
    assert D.setup_freeze_lines(1, True, 1, 1) == []
    assert D.setup_freeze_lines(2, False, None, None) == ["[setup:freeze] Project does not have parameters for "
                                                          "reference freezing."]
    assert D.setup_freeze_lines(2, True, None, None) == []
    assert D.setup_freeze_lines(3, True, 1, 1) == ["[setup:freeze] Initial cone and azymuth in iteration 3 will be "
                                                   "read from iteration 1"]
    # cone and inplane from different iterations: the 'Initial inplane' echo is screen-only (o lost intofile)
    assert D.setup_freeze_lines(3, True, 1, 2) == ["[setup:freeze] Initial cone in iteration 3 will be read from "
                                                   "iteration 1"]
    assert D.setup_freeze_lines(3, True, 3, 1) == ["[setup:freeze] Initial inplane in iteration 3 will be read from "
                                                   "iteration 1"]


def test_assemble_and_ending_lines():
    """[L1] end of iteration 1."""
    got = (D.assemble_head_lines() + D.assemble_average_lines(1, 2, [1, 2])
           + [D.assemble_average_done_line(59.947974)] + D.convergence_lines(2)
           + D.bandpass_lines(1, "bin4_align_1_eo", True, "c1", 28, 0.5, 2, True)
           + D.ending_lines(D.card_ite_path("bin4_align_1_eo", 1), "12-Nov-2026 01:12:28", "12-Nov-2026 02:15:55",
                            "12-Nov-2026 02:19:05") + D.check_ok_lines(1))
    assert got == [
        "[assemble] Gathering refined tables.",
        "[assemble] Computing table comparisons against previous iteration.",
        "[assemble] getting rid of overlapping particles:",
        "[assemble] Applying  thresholding:",
        "[assemble] Computing average for 2 multireference channels.",
        "[assemble] Computing average for ref 1 ite 1",
        "[assemble] Computing average for ref 1 ite 2",
        "done assembling intermediate average in 59.947974 seconds. ",
        "   ",
        "At least one reference did not achieve convergence: keeping normal execution path",
        " [Assemble]: preparing next iteration: (2)",
        "[bandpass #1] Symmetrizing both references before aligning ref 2 onto ref 1",
        "[bandpass #1] Aligning reference 2 onto reference 1 before computing resolution.",
        "[bandpass #1] ... done",
        "[bandpass #1]  shifts on reference 2: ",
        "[bandpass #1]  angles on reference 2: ",
        '[bandpass #1] Computing resolution with symmetry "c1"',
        "[bandpass #1] Adaptive bandpass FSC for resolution estimation saved in file:",
        "[bandpass #1] ./bin4_align_1_eo/results/ite_0001/averages/bandpass_fsc_ite_0001.fsc",
        "[bandpass #1] Resolution estimated for iteration 1: 28.00 (threshold  0.50)",
        "[bandpass #1] Low pass computed for next iteration 2: 26.00 (pushback  2.00)",
        " ", " ",
        "The card ./bin4_align_1_eo/cards/ite_0001/card_ite_ite_0001.card has been completed ",
        "",
        "Computations for iterative refinement started at 12-Nov-2026 01:12:28",
        "Assembling results and averaging  started at 12-Nov-2026 02:15:55",
        "Iteration finished at 12-Nov-2026 02:19:05",
        " ",
        "iteration 1 ok", " ", " ", " "]
    assert D.bandpass_lines(4, "p", True, "c1", 26, 0.5, 2, False)[-1] == \
        "[bandpass #4] Could not identify round number for next iteration (5)"
    assert D.bandpass_lines(1, "p", True, "h[-1.4,0.3]", 32, 0.143, 0, True)[-2:] == [                # [L3]
        "[bandpass #1] Resolution estimated for iteration 1: 32.00 (threshold  0.14)",
        "[bandpass #1] Low pass computed for next iteration 2: 32.00 (pushback  0.00)"]
    assert D.assemble_average_lines(2, 3, [1, 3])[2] == "[assemble] Skipping extinct reference 2"
    assert D.bandpass_too_many_references_line(3) == "[bandpass #3]  ... but you can always write your own plugin!"
    assert D.assemble_average_done_line(1.0, average_needed=False) == "average has been skipped"


def test_convergence_error_branch():
    """[L2]: the catch branch with a two-line lasterr."""
    bar = " ----------------------------------------------------------------------  "
    assert D.convergence_lines(2, "error", error_text="Error using containers.Map/subsref\n"
                                                      "The specified key is not present in this container.") == [
        bar, "   ", "[DYNAMO ERROR]: non fatal: recovering", "Convergence computation could not be completed.",
        "Error using containers.Map/subsref", "The specified key is not present in this container.",
        "Keeping usual execution path for next iteration (2).", "   ", bar,
        " [Assemble]: preparing next iteration: (2)"]


def test_returning_control_and_stdout_split():
    assert D.returning_control_line("12-Nov-2026 01:44:12") == \
        " Returning control to Matlab  at 12-Nov-2026 01:44:12. GPU round was successful"        # [L1]
    assert D.stdout_to_log_lines("a\nb\n ") == ["a", "b", " "]


def test_gpu_failure_lines():
    """dynamo_pipeline_system_check_output_gpu.m:44-57, appended at :90 for the order of dynamo_execute_project.m:297
    (sprintf('%s %s',myMotor,cardfile)); sprintf('%s \\n', order) gives the order line and an empty line, as
    dynamo_aux_appendlog writes fprintf('%s\\n') per element."""
    order = "dynamo_compute_iteration_spp " + D.card_iteref_path("p", 1, 1)
    assert order == "dynamo_compute_iteration_spp ./p/cards/ite_0001/card_iteref_ref_001_ite_0001.card"
    assert D.gpu_failure_lines(order) == ["&" * 63] * 8 + [
        "GPU: Warning!!!",
        "something went wrong inside the GPU application while the system tried to execute the order:",
        order + " ", "", "Dynamo will abort project execution at this point."] + ["&" * 62] * 8


def test_append_log(tmp_path):
    D.append_log(str(tmp_path), ["x", " "])
    D.append_log(str(tmp_path), ["y"])
    assert (tmp_path / "log.txt").read_text() == "x\n \ny\n"


# ------------------------------------------------------------------------------------------------ ground truth
GT = os.environ.get("DYNLOG_GT_PROJECT", os.path.join(ROOT, "benchmarks", "05_abp_bin4_eo", "dynamo", "run", "bin4_align_1_eo"))


def _gt_present():
    need = ["log.txt", "settings/virtual_project.mat", "results/ite_0001/starting_values/mask_transformed_ite_0001.em",
            "cards/ite_0001/card_iteref_ref_001_ite_0001.card"]
    import glob
    return all(os.path.exists(os.path.join(GT, n)) for n in need) and \
        bool(glob.glob(os.path.join(os.path.dirname(GT), "filamentsData_ctf", "particle_*.em")))


@pytest.mark.skipif(not _gt_present(), reason="Dynamo 1.1.514 ground-truth run (with its data) absent")
def test_iteration1_matches_dynamo_log():
    """Iteration 1 of the Dynamo run rebuilt from the project's files: after N1-N4 normalisation the text has the
    same characters (fragment interleaving permutes them only) and the same ordered MATLAB/header lines."""
    exp = expected_log_lines(GT, iterations=[1])
    with open(os.path.join(GT, "log.txt"), encoding="latin-1") as f:
        whole = f.read().split("\n")
    act = whole[:len(exp)]
    assert act[-1] == " " and whole[len(exp)].startswith("****")    # iteration 1 ends where iteration 2 begins
    rules = [(r"\d{2}-[A-Z][a-z]{2}-\d{4} \d{2}:\d{2}:\d{2}", "<D>"),
             (r"[A-Z][a-z]{2} [A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2} \d{4}", "<A>"),
             (r"(in  multiGPU modus: |intermediate average in )[\d.]+", r"\1<S>"),
             (r"(Your card ).*?( -identified as \d+ in your system- has )\d+", r"\1<G>\2<M>")]

    def norm(x):
        for rx, rep in rules:
            x = re.sub(rx, rep, x)
        return x
    E, A = [norm(x) for x in exp], [norm(x) for x in act]
    assert collections.Counter("\n".join(E)) == collections.Counter("\n".join(A))

    def ordered(L):
        """Drop the thread regions: after the ' ' that follows the last ' multigrid step' line, up to ' cpu time'."""
        out, inreg = [], False
        for i, x in enumerate(L):
            if inreg:
                if x.startswith(" cpu time invested"):
                    inreg = False
                    out.append(x)
                continue
            out.append(x)
            inreg = x == " " and i > 0 and L[i - 1].startswith(" multigrid step:")
        return out
    assert ordered(E) == ordered(A)
    assert len(ordered(E)) < 200
