"""DynamoProject (pydyn.dynproject): Dynamo 1.1.514 project logic.

CPU-only. The first group uses synthetic inputs. The second group replays setup and the table part of assembly on
the files of the user's run (test_data/bin8_filament, read-only) and requires byte-identical outputs; it is skipped
when the data is absent.
"""
import os
import re
import shutil

import numpy as np
import pytest

from pydyn import dynio
from pydyn import dynproject as dp

RUNROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "test_data", "bin8_filament")
RUN = os.path.join(RUNROOT, "bin8_align_1")
DATA = os.path.join(RUNROOT, "filamentsData_ctf")
have_run = pytest.mark.skipif(not os.path.isdir(RUN), reason="test_data/bin8_filament not available")


def base_vpr(rounds=2):
    v = {"name_project": "p", "intertwin": "1/2/3/4/5/6/7/8", "destination": "standalone_gpu",
         "how_many_processors": 1, "gpu_identifier_set": np.array([0, 1, 2, 3], dtype=np.uint8),
         "systemUsingProcessorTables": 0, "systemCardForParticle": 0, "initial_references": "*",
         "project_type": "general", "adaptive_bandpass": 0, "adaptive_bandpass_even_odd": 0,
         "fourierMinimumFractionForAverage": -1, "fCompensationSmoothingRadius": 0,
         "useSmoothingMaskForFourierCompensation": 0, "averagingImplicitRotationMask": 0, "options": "",
         "feature_data_type": "dynamo_folder", "feature_extension_data": "em", "feature_data_sidelength": 64,
         "feature_template_sidelength": 64}
    vals = {"nref": 1, "mra": 0, "dim": 64, "localnc": 1, "use_CC": 1, "sym": "c1", "low": 25, "high": 2,
            "threshold_modus": 0, "threshold2_modus": 0, "separation_in_tomogram": 0, "cone_check_peak": 0,
            "inplane_check_peak": 0, "limit_xy_check_peak": 0, "limit_z_check_peak": 0, "plugin_post": 0,
            "convergence_type": 0, "area_search_modus": 4, "cone_freeze_reference": 0, "inplane_freeze_reference": 0}
    for r in range(1, 9):
        v["ite_r%d" % r] = 2 if r <= rounds else 0
        for k, x in vals.items():
            v["%s_r%d" % (k, r)] = x
    return v


def bare_project(tmp_path, vpr):
    pr = dp.DynamoProject.__new__(dp.DynamoProject)
    pr.vpr, pr.out, pr.name, pr.echo = vpr, str(tmp_path), vpr["name_project"], False
    pr.rounds = dp.round_array(vpr)
    pr.N = len(pr.rounds)
    return pr


# ------------------------------------------------------------------ synthetic

def test_round_array_leaders_and_ite2round():
    v = base_vpr(2)
    assert dp.round_array(v) == [1, 1, 2, 2]
    assert dp.round_leaders(v) == [1, 3]
    assert [dp.ite2round(v, i) for i in (0, 1, 2, 3, 4, 5)] == [None, 1, 1, 2, 2, None]
    v["intertwin"] = "1 2/3/4/5/6/7/8"                # rounds 1 and 2 alternate ite_r1 times; ite_r2 is ignored
    v["ite_r1"], v["ite_r2"], v["ite_r3"] = 2, 7, 1
    assert dp.round_array(v) == [1, 2, 1, 2, 3]
    v["intertwin"] = "1/3"                            # a series runs over head .. next head - 1
    assert dp.round_array(v) == [1, 2, 1, 2] + [3, 4, 5, 6, 7, 8]
    assert dp._last_leader_before(base_vpr(2), 3) == 1      # strict '<': a round leader gets the previous leader


def test_convention_lists_match_dynamo_source():
    src = os.path.join(dp.DYNAMO_SRC, "dynamo_vpr_convention.m")
    if not os.path.exists(src):
        pytest.skip("Dynamo 1.1.514 source not available")
    import re
    nc, rc = [], []
    for ln in open(src).read().splitlines():
        m = re.match(r"^\s*(nc|rc)\{c\}\.name\s*=\s*'([^']*)'\s*;", ln.split("%")[0])
        if m:
            (nc if m.group(1) == "nc" else rc).append(m.group(2))
    assert nc == dp.GLOBAL_PARAMS and rc == dp.ROUND_PARAMS


def test_validate_rejects_unsupported(tmp_path):
    bare_project(tmp_path, base_vpr()).validate()
    for key, val in [("nref_r2", 2), ("mra_r1", 1), ("dim_r1", 32), ("localnc_r1", 0), ("sym_r2", "c6"),
                     ("adaptive_bandpass", 1), ("threshold_modus_r1", 3), ("use_CC_r1", 2), ("low_r1", 0.4),
                     ("area_search_modus_r1", 7), ("how_many_processors", 2), ("convergence_type_r1", 1),
                     ("destination", "matlab_parfor")]:
        v = base_vpr()
        v[key] = val
        with pytest.raises(NotImplementedError):
            bare_project(tmp_path, v).validate()
    v = base_vpr()
    v["fmin"] = 2
    with pytest.raises(NotImplementedError):
        bare_project(tmp_path, v).validate()


def test_acos_branches_and_self_comparison():
    assert dp._acos_deg(1.0) == 0
    z = dp._acos_deg(1 + 2.0 ** -52)
    assert z.real == 0 and dynio.dlm_field(z) == "0+1.2074e-06i"
    assert dynio.dlm_field(dp._acos_deg(1 + 2 * 2.0 ** -52)) == "0+1.7075e-06i"
    with pytest.raises(NotImplementedError):
        dp._acos_deg(-1 - 2.0 ** -52)
    rng = np.random.default_rng(1)
    t = np.zeros((50, 42))
    t[:, 3:6] = rng.normal(size=(50, 3))
    t[:, 6:9] = rng.uniform(-180, 180, size=(50, 3))
    t[:, 9] = rng.uniform(size=50)
    c27, c28, c29, c30 = dp.compare_columns(t, t, dp.NumpyTrig)
    assert not c27.any() and not c29.any() and not c30.any()
    assert np.all(np.abs(c28) < 1e-5)                 # rounding artefact only


def test_compare_columns_values():
    cur = np.zeros((1, 10))
    prev = np.zeros((1, 10))
    cur[0, 3:6] = [3, 4, 0]
    cur[0, 6:10] = [102.46, 170.14, 80.936, 0.09848]
    prev[0, 6:9] = [90, 0, 0]
    c27, c28, c29, c30 = dp.compare_columns(cur, prev, dp.NumpyTrig)
    assert c27[0] == 5 and dynio.dlm_field(c29[0]) == "12.46" and c30[0] == 0.09848
    assert dynio.dlm_field(c28[0]) == "170.14"        # angle between the z axes = tilt difference


def test_table_merge_sorted_first_occurrence():
    a = np.array([[3, 1, 7.0], [1, 1, 1.0]])
    b = np.array([[2, 1, 2.0, 9.0], [3, 1, 8.0, 9.0], [0, 0, 0, 0]])
    m = dp.table_merge([a, b[b[:, 0] != 0]])
    assert m[:, 0].tolist() == [1, 2, 3] and m[2, 2] == 7.0 and m.shape[1] == 4 and m[0, 3] == 0


def test_shift_and_angle_freezing_modes(tmp_path):
    v = base_vpr(2)
    v["area_search_modus_r1"] = v["area_search_modus_r2"] = 3
    pr = bare_project(tmp_path, v)
    pr.log = lambda m: None

    def tab(ite, val, tags=(1, 2, 3)):
        t = np.zeros((len(tags), 42))
        t[:, 0] = tags
        t[:, 3:9] = val
        dynio.write_tbl(pr.f_starting_table(1, ite), t)
    tab(1, 1.0)
    tab(2, 2.0)
    tab(3, 3.0, tags=(1, 2, 3, 4))
    tab(4, 4.0)
    for ite in (2, 3, 4):
        pr.freeze_shifts(ite, [1])
    s = {k: dynio.read_tbl(pr.f_starting_table(1, k)) for k in (2, 3, 4)}
    assert np.all(s[2][:, 3:6] == 1) and np.all(s[2][:, 6:9] == 2)             # seed: leader 1
    assert np.all(s[3][:3, 3:6] == 1) and np.all(s[3][3, 3:6] == 3)           # seed 1 (strict '<'); tag 4 missing
    assert np.all(s[4][:, 3:6] == 1)                                         # seed 3, already rewritten from 1
    v["cone_freeze_reference_r2"], v["inplane_freeze_reference_r2"] = 2, 0
    tab(1, 1.0)
    tab(4, 4.0)
    pr.freeze_angles(4, [1])
    t4 = dynio.read_tbl(pr.f_starting_table(1, 4))
    assert np.all(t4[:, 6:8] == 1) and np.all(t4[:, 8] == 4) and np.all(t4[:, 3:6] == 4)
    v["area_search_modus_r2"] = 2
    tab(4, 4.0)
    pr.freeze_shifts(4, [1])
    assert np.all(dynio.read_tbl(pr.f_starting_table(1, 4))[:, 3:6] == 4)    # mode 2: nothing


def test_template_sym_c1_is_identity_even_with_partial_fmask(tmp_path):
    """dynamo_sym(map,'c1','fmask',f) returns map unchanged ('c1' -> kind 'none', dynamo_sym.m:380-385), so a
    template fmask with holes inside the r = L/2-2 sphere must not trigger a Fourier-weight division."""
    L = 16
    v = base_vpr(1)
    v["feature_data_sidelength"] = v["feature_template_sidelength"] = L
    v["dim_r1"], v["high_r1"], v["low_r1"] = L, 2, 6
    v["symmetrize_with_fmask"] = 1
    pr = bare_project(tmp_path, v)
    rng = np.random.default_rng(3)
    mask = np.zeros((L, L, L))
    mask[4:12, 4:12, 4:12] = 0.8
    fmask = np.ones((L, L, L))
    fmask[:, :, 8:11] = 0                                   # wedge-like hole inside the inner sphere
    T = rng.normal(size=(L, L, L)).astype(np.float32)
    dynio.write_em(pr.p("settings", "mask.em"), mask)
    dynio.write_em(pr.f_fmask_starting_reference(1, 1), fmask)
    dynio.write_em(pr.f_starting_reference(1, 1), T)
    pr.preprocess_references(1, [1])
    got = dynio.read_em(pr.p(pr.r_template_transformed(1, 1)))
    shell = dp.isotropic_shell(L, 2, 6, 2.0)
    ref = np.real(np.fft.ifftn(np.fft.fftn(T.astype(np.float64)) * np.fft.fftshift(shell)))
    ref = dp.normalize_roi(ref, mask > 0.5)
    assert np.array_equal(got, ref.astype(np.float32).astype(np.float64))
    assert np.array_equal(dynio.read_em(pr.p(pr.r_fmask_transformed(1, 1))), (fmask > 0.5).astype(np.float64))


def test_isotropic_shell_is_float32_valued():
    s = dp.isotropic_shell(64, 2, 25, 2.0)
    assert s[32, 32, 40] == 1 and s[32, 32, 32] == np.float32(np.exp(-1.0))
    assert np.array_equal(s, s.astype(np.float32).astype(np.float64)) and s.max() == 1 and (s[s > 0] > 0.13).all()


# ------------------------------------------------------------------ replays on the user's run

def _run_file(ite, name):
    sub = "averages" if name.startswith(("refined_table", "average", "fmask_average", "eo_fsc", "fweight")) \
        else "starting_values"
    return os.path.join(RUN, "results", "ite_%04d" % ite, sub, name)


def _strip_times(b):
    return b"".join(l + b"\n" for l in b.split(b"\n")[:-1] if not l.startswith(b"time_"))


@have_run
def test_card_ite_from_vpr_matches_run():
    vpr = dp.load_vpr(os.path.join(RUN, "settings", "virtual_project.mat"))
    assert dp.round_array(vpr) == [1, 1, 2, 2] and dp.round_leaders(vpr) == [1, 3]
    for k in (1, 2, 3, 4):
        mine = dynio.card_bytes(dp.card_ite_from_vpr(vpr, k))
        run = open(os.path.join(RUN, "cards", "ite_%04d" % k, "card_ite_ite_%04d.card" % k), "rb").read()
        assert mine == _strip_times(run), k


def _project(tmp_path, matlab_trig=False):
    pr = dp.DynamoProject(RUN, DATA, str(tmp_path / "proj"), matlab_trig=matlab_trig, echo=False)
    os.makedirs(pr.p("settings"), exist_ok=True)
    for f in os.listdir(os.path.join(RUN, "settings")):
        shutil.copyfile(os.path.join(RUN, "settings", f), pr.p("settings", f))
    return pr


@have_run
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_setup_reproduces_run(tmp_path, k):
    """Setup of iteration k from the run's inputs: the rewritten starting table, mask_transformed, template fmask,
    transformed template and card_iteref must be byte-identical to the run; GPU chunks as in the engine log."""
    pr = _project(tmp_path)
    sv = lambda i: pr.p(pr._sv(i))
    os.makedirs(sv(1), exist_ok=True)
    os.makedirs(sv(k), exist_ok=True)
    shutil.copyfile(os.path.join(RUN, "settings", "table_initial_ref_001.tbl"), pr.f_starting_table(1, 1))
    if k > 1:
        shutil.copyfile(_run_file(k - 1, "refined_table_ref_001_ite_%04d.tbl" % (k - 1)), pr.f_starting_table(1, k))
        dynio.write_txt(pr.f_surv(k), 1)
    for name in ("starting_reference", "fmask_starting_reference"):
        shutil.copyfile(_run_file(k, "%s_ref_001_ite_%04d.em" % (name, k)), pr.p(pr._sv(k), "%s_ref_001_ite_%04d.em"
                                                                                % (name, k)))
    dynio.write_card(pr.f_card_ite(k), dp.card_ite_from_vpr(pr.vpr, k))
    refs, chunks = pr.setup(k)
    assert refs == [1]
    same = lambda a, b: open(a, "rb").read() == open(b, "rb").read()
    assert same(pr.f_starting_table(1, k), _run_file(k, "starting_table_ref_001_ite_%04d.tbl" % k))
    for rel in (pr.r_mask_transformed(k), pr.r_fmask_transformed(1, k), pr.r_template_transformed(1, k)):
        assert same(pr.p(rel), os.path.join(RUN, rel)), rel
    assert same(pr.f_card_iteref(1, k), os.path.join(RUN, "cards", "ite_%04d" % k,
                                                     "card_iteref_ref_001_ite_%04d.card" % k))
    card = dynio.read_card(pr.f_card_ite(k))
    assert "time_start_computation" in card
    assert same(pr.f_surv(k), os.path.join(RUN, "results", "ite_%04d" % k,
                                           "currently_surviving_references_ite_%04d.txt" % k))
    assert [c["gpu"] for c in chunks] == [0, 1, 2, 3] and [c["n"] for c in chunks] == [2356, 2356, 2356, 2354]
    # log.txt: Dynamo's setup banner (dynamo_iteration_setup.m:184-189), as in the user's log.txt apart from the date
    log = open(pr.p("log.txt")).read().splitlines()
    with open(os.path.join(RUN, "log.txt"), errors="replace") as f:
        user = f.read().splitlines()
    at = max(i for i, x in enumerate(user) if x == "**********    iteration: %d      " % k) - 2
    done = next(i for i in range(at + 6, len(user)) if user[i].startswith("[setup] done with general set up"))
    # banner, then the freezing / shift / skipped-reference lines (dynlog) and the 'done' line, all as in Dynamo's log
    assert log[:4] == user[at:at + 4] and log[5] == user[at + 5] and log[6:-1] == user[at + 6:done]
    assert log[-1] == "[setup] done with general set up of card files, starting computations: %s" % \
        card["time_start_computation"]
    assert re.fullmatch(r"Starting iteration at \d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d:\d\d", log[4])
    assert re.fullmatch(r"Starting iteration at \d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d:\d\d", user[at + 4])
    assert not os.path.exists(pr.p("pydyn_report.json"))
    first = [open(c["particle_list"]).read().splitlines()[:3] for c in chunks]
    assert first[0] == ["filamentsData_ctf/particle_%06d.em" % t for t in (1, 2, 3)]
    assert first[1][0].endswith("002357.em") and first[2][0].endswith("004713.em") and first[3][0].endswith("007069.em")
    S = np.real(dynio.read_tbl(pr.f_starting_table(1, k)))
    b3 = dynio.read_btbl(chunks[3]["starting_btbl"])
    assert np.array_equal(b3, S[7068:].astype(np.float32).astype(np.float64))


def _matlab_trig_available():
    try:
        dp.MatlabTrig()
        return True
    except OSError:
        return False


@have_run
@pytest.mark.parametrize("trig", ["matlab", "numpy"])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_assemble_tables_reproduce_run_refined_table(tmp_path, k, trig):
    """Gather + columns 27-30 + thresholding on per-GPU float32 tables equal to what the engine produced
    (starting table of the run with its columns 4-10 from the run's refined table): the refined_table must be
    byte-identical with MATLAB's sin/cos; with numpy's, every column except 28 must be."""
    mt = trig == "matlab"
    if mt and not _matlab_trig_available():
        pytest.skip("MCR libmwmathutil not available")
    pr = _project(tmp_path, matlab_trig=mt)
    S = dynio.read_tbl(_run_file(k, "starting_table_ref_001_ite_%04d.tbl" % k))
    R = dynio.read_tbl(_run_file(k, "refined_table_ref_001_ite_%04d.tbl" % k))
    eng = np.real(S).copy()
    eng[:, 3:10] = np.real(R[:, 3:10])
    per = -(-len(eng) // 4)
    for g in range(4):
        dynio.write_btbl(pr.p(pr.r_inter(k), "refined_gpuProcRefTable_ite_%04d_ref_001_proc_00000_gpu_%02d.btbl"
                              % (k, g)), eng[g * per:(g + 1) * per])
    if k == 1:
        pr.table_initial = os.path.join(RUNROOT, "particles_edit_mod.tbl")
    info = pr.assemble_tables(k, 1)
    mine = open(pr.f_av("refined_table", 1, k, "tbl"), "rb").read().split(b"\n")
    run = open(_run_file(k, "refined_table_ref_001_ite_%04d.tbl" % k), "rb").read().split(b"\n")
    assert len(mine) == len(run)
    bad = {}
    for a, b in zip(mine, run):
        for j, (x, y) in enumerate(zip(a.split(b" "), b.split(b" "))):
            if x != y:
                bad[j + 1] = bad.get(j + 1, 0) + 1
    assert info["contributors"] == 9422
    if mt:
        assert not bad, bad
    else:
        assert set(bad) <= {28}, bad
