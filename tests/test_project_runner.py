"""DynamoProject (pydynamo.project.runner): Dynamo 1.1.514 project logic.

CPU-only. The first group uses synthetic inputs. The second group replays setup and the table part of assembly on
the files of the user's run (test_data/bin8_filament, read-only) and requires byte-identical outputs; it is skipped
when the data is absent.
"""
import os
import re
import shutil

import numpy as np
import pytest

from pydynamo.files import formats
from pydynamo.project import runner as dp

RUNROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


def test_convention_lists_match_the_generated_convention_table():
    """dynproject's parameter lists equal pydynamo.vpr.convention (generated from dynamo_vpr_convention.m; the re-parse of
    the source itself is tests/test_console_and_lifecycle.py::test_convention_table_matches_source)."""
    from pydynamo.vpr import convention
    assert dp.GLOBAL_PARAMS == [p["name"] for p in convention.NC]
    assert dp.ROUND_PARAMS == [p["name"] for p in convention.RC]


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
    assert z.real == 0 and formats.dlm_field(z) == "0+1.2074e-06i"
    assert formats.dlm_field(dp._acos_deg(1 + 2 * 2.0 ** -52)) == "0+1.7075e-06i"
    with pytest.raises(NotImplementedError):
        dp._acos_deg(-1 - 2.0 ** -52)
    rng = np.random.default_rng(1)
    t = np.zeros((50, 42))
    t[:, 3:6] = rng.normal(size=(50, 3))
    t[:, 6:9] = rng.uniform(-180, 180, size=(50, 3))
    t[:, 9] = rng.uniform(size=50)
    c27, c28, c29, c30 = dp.compare_columns(t, t, dp.FdlibmTrig)
    assert not c27.any() and not c29.any() and not c30.any()
    assert np.all(np.abs(c28) < 1e-5)                 # rounding artefact only


def test_compare_columns_values():
    cur = np.zeros((1, 10))
    prev = np.zeros((1, 10))
    cur[0, 3:6] = [3, 4, 0]
    cur[0, 6:10] = [102.46, 170.14, 80.936, 0.09848]
    prev[0, 6:9] = [90, 0, 0]
    c27, c28, c29, c30 = dp.compare_columns(cur, prev, dp.FdlibmTrig)
    assert c27[0] == 5 and formats.dlm_field(c29[0]) == "12.46" and c30[0] == 0.09848
    assert formats.dlm_field(c28[0]) == "170.14"        # angle between the z axes = tilt difference


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
        formats.write_tbl(pr.f_starting_table(1, ite), t)
    tab(1, 1.0)
    tab(2, 2.0)
    tab(3, 3.0, tags=(1, 2, 3, 4))
    tab(4, 4.0)
    for ite in (2, 3, 4):
        pr.freeze_shifts(ite, [1])
    s = {k: formats.read_tbl(pr.f_starting_table(1, k)) for k in (2, 3, 4)}
    assert np.all(s[2][:, 3:6] == 1) and np.all(s[2][:, 6:9] == 2)             # seed: leader 1
    assert np.all(s[3][:3, 3:6] == 1) and np.all(s[3][3, 3:6] == 3)           # seed 1 (strict '<'); tag 4 missing
    assert np.all(s[4][:, 3:6] == 1)                                         # seed 3, already rewritten from 1
    v["cone_freeze_reference_r2"], v["inplane_freeze_reference_r2"] = 2, 0
    tab(1, 1.0)
    tab(4, 4.0)
    pr.freeze_angles(4, [1])
    t4 = formats.read_tbl(pr.f_starting_table(1, 4))
    assert np.all(t4[:, 6:8] == 1) and np.all(t4[:, 8] == 4) and np.all(t4[:, 3:6] == 4)
    v["area_search_modus_r2"] = 2
    tab(4, 4.0)
    pr.freeze_shifts(4, [1])
    assert np.all(formats.read_tbl(pr.f_starting_table(1, 4))[:, 3:6] == 4)    # mode 2: nothing


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
    formats.write_em(pr.p("settings", "mask.em"), mask)
    formats.write_em(pr.f_fmask_starting_reference(1, 1), fmask)
    formats.write_em(pr.f_starting_reference(1, 1), T)
    pr.preprocess_references(1, [1])
    got = formats.read_em(pr.p(pr.r_template_transformed(1, 1)))
    shell = dp.isotropic_shell(L, 2, 6, 2.0)
    ref = np.real(np.fft.ifftn(np.fft.fftn(T.astype(np.float64)) * np.fft.fftshift(shell)))
    ref = dp.normalize_roi(ref, mask > 0.5)
    assert np.array_equal(got, ref.astype(np.float32).astype(np.float64))
    assert np.array_equal(formats.read_em(pr.p(pr.r_fmask_transformed(1, 1))), (fmask > 0.5).astype(np.float64))


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
        mine = formats.card_bytes(dp.card_ite_from_vpr(vpr, k))
        run = open(os.path.join(RUN, "cards", "ite_%04d" % k, "card_ite_ite_%04d.card" % k), "rb").read()
        assert mine == _strip_times(run), k


def _project(tmp_path):
    pr = dp.DynamoProject(RUN, DATA, str(tmp_path / "proj"), echo=False)
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
        formats.write_txt(pr.f_surv(k), 1)
    for name in ("starting_reference", "fmask_starting_reference"):
        shutil.copyfile(_run_file(k, "%s_ref_001_ite_%04d.em" % (name, k)), pr.p(pr._sv(k), "%s_ref_001_ite_%04d.em"
                                                                                % (name, k)))
    formats.write_card(pr.f_card_ite(k), dp.card_ite_from_vpr(pr.vpr, k))
    refs, chunks = pr.setup(k)
    assert refs == [1]
    same = lambda a, b: open(a, "rb").read() == open(b, "rb").read()
    assert same(pr.f_starting_table(1, k), _run_file(k, "starting_table_ref_001_ite_%04d.tbl" % k))
    for rel in (pr.r_mask_transformed(k), pr.r_fmask_transformed(1, k), pr.r_template_transformed(1, k)):
        assert same(pr.p(rel), os.path.join(RUN, rel)), rel
    assert same(pr.f_card_iteref(1, k), os.path.join(RUN, "cards", "ite_%04d" % k,
                                                     "card_iteref_ref_001_ite_%04d.card" % k))
    card = formats.read_card(pr.f_card_ite(k))
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
    # banner, then the freezing / shift / skipped-reference lines (logfile) and the 'done' line, all as in Dynamo's log
    assert log[:4] == user[at:at + 4] and log[5] == user[at + 5] and log[6:-1] == user[at + 6:done]
    assert log[-1] == "[setup] done with general set up of card files, starting computations: %s" % \
        card["time_start_computation"]
    assert re.fullmatch(r"Starting iteration at \d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d:\d\d", log[4])
    assert re.fullmatch(r"Starting iteration at \d\d-[A-Z][a-z]{2}-\d{4} \d\d:\d\d:\d\d", user[at + 4])
    assert not os.path.exists(pr.p("pydynamo_report.json"))
    first = [open(c["particle_list"]).read().splitlines()[:3] for c in chunks]
    assert first[0] == ["filamentsData_ctf/particle_%06d.em" % t for t in (1, 2, 3)]
    assert first[1][0].endswith("002357.em") and first[2][0].endswith("004713.em") and first[3][0].endswith("007069.em")
    S = np.real(formats.read_tbl(pr.f_starting_table(1, k)))
    b3 = formats.read_btbl(chunks[3]["starting_btbl"])
    assert np.array_equal(b3, S[7068:].astype(np.float32).astype(np.float64))


@have_run
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_assemble_tables_reproduce_run_refined_table(tmp_path, k):
    """Gather + columns 27-30 + thresholding on per-GPU float32 tables equal to what the engine produced
    (starting table of the run with its columns 4-10 from the run's refined table): the refined_table is
    byte-identical (MATLAB's sin/cos/acos through pydynamo.matlab.fdlibm)."""
    pr = _project(tmp_path)
    S = formats.read_tbl(_run_file(k, "starting_table_ref_001_ite_%04d.tbl" % k))
    R = formats.read_tbl(_run_file(k, "refined_table_ref_001_ite_%04d.tbl" % k))
    eng = np.real(S).copy()
    eng[:, 3:10] = np.real(R[:, 3:10])
    per = -(-len(eng) // 4)
    for g in range(4):
        formats.write_btbl(pr.p(pr.r_inter(k), "refined_gpuProcRefTable_ite_%04d_ref_001_proc_00000_gpu_%02d.btbl"
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
    assert not bad, bad


def test_average_chains_one_state_through_the_workers_in_ascending_tag_order(monkeypatch):
    """average_from_table: rows sorted by tag, odd/even by position, and one running state sent to the workers' runs
    of consecutive tags in that order (dynamo_average's single-thread loop), then finalize on the last state."""
    from pydynamo.averaging import average as AV
    n = 4
    held = {0: [5, 7, 9, 40], 1: [11, 12, 2], 2: [30, 31]}             # tags resident on three workers

    class Conn:
        def __init__(self, w):
            self.w, self.out = w, None
        def send(self, m):
            if m["op"] == "tags":
                self.out = {"ok": True, "tags": np.array(held[self.w])}
            else:
                calls.append((self.w, m["rows"][:, 0].astype(int).tolist(), m["half"].tolist(), m["state"]["k"]))
                st = dict(m["state"]); st["k"] = m["state"]["k"] + 1
                self.out = {"ok": True, "state": st}
        def recv(self):
            return self.out

    calls = []
    pr = object.__new__(dp.DynamoProject)
    pr.workers = [(d, Conn(d), None) for d in range(3)]
    pr._start_workers = lambda: None
    pr.p = lambda *a: "mask.em"
    monkeypatch.setattr(dp.formats, "read_em", lambda f: np.ones((n, n, n), np.float32))
    monkeypatch.setattr(AV, "new_state", lambda n: {"k": 0})
    monkeypatch.setattr(AV, "finalize", lambda state, nh, mask, n, fmin, sym, threads=1: (state, nh.tolist()))
    tags = [40, 2, 31, 5, 12, 9, 30, 7, 11]
    T = np.zeros((len(tags), 35))
    T[:, 0], T[:, 2] = tags, 1
    state, nh = pr.average_from_table(1, 1, T)
    # ascending: 2 5 7 9 11 12 30 31 40 -> runs by owner: [2](1) [5 7 9](0) [11 12](1) [30 31](2) [40](0)
    assert [c[0] for c in calls] == [1, 0, 1, 2, 0]
    assert [c[1] for c in calls] == [[2], [5, 7, 9], [11, 12], [30, 31], [40]]
    assert sum((c[2] for c in calls), []) == [0, 1, 0, 1, 0, 1, 0, 1, 0]      # odd/even by position
    assert [c[3] for c in calls] == [0, 1, 2, 3, 4] and state["k"] == 5      # each run gets the previous state
    assert nh == [5.0, 4.0]


# ------------------------------------------------------------------ GPU workers, failure paths and the driver loop

class _Proc:
    """Stand-in for a worker process: it ends on None (unless stuck) or when terminated."""

    def __init__(self, stuck=False):
        self.alive, self.stuck, self.got_none, self.terminated = True, stuck, False, False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated, self.alive = True, False

    def join(self, timeout=None):
        if self.got_none and not self.stuck:
            self.alive = False


class _Conn:
    """Parent end of a worker pipe; `reply` is the worker's answer, None for a worker still busy (never answers)."""

    def __init__(self, proc, reply):
        self.proc, self.reply, self.sent = proc, reply, []

    def send(self, m):
        self.sent.append(m)
        if m is None:
            self.proc.got_none = True

    def recv(self):
        assert self.reply is not None, "the reply of a busy worker must not be awaited"
        return self.reply


def _workers(*replies, stuck=()):
    out = []
    for k, rep in enumerate(replies):
        p = _Proc(stuck=k in stuck)
        out.append((k, _Conn(p, rep), p))
    return out


def test_worker_devices_are_cut_to_the_gpu_chunks():
    """setup_gpu makes how_many_processors * len(gpu_identifier_set) chunks per reference and compute() deals them over
    the workers: a device beyond that number would get an empty chunk list (np.concatenate([]) in the worker)."""
    assert dp.worker_devices([0, 1], [0]) == [0]
    assert dp.worker_devices([0, 0], [0], 1) == [0]
    assert dp.worker_devices([2, 3, 4, 5, 6], [0, 1], 1.0) == [2, 3]
    assert dp.worker_devices([5], [0, 1, 2, 3]) == [5]
    assert dp.worker_devices(None, [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert dp.worker_devices([0, 1, 2, 3], [0, 1], 2) == [0, 1, 2, 3]


def test_dynamoproject_starts_no_worker_without_a_gpu_chunk(tmp_path, monkeypatch):
    """DynamoProject(gpus=[0, 1]) on a project with gpu_identifier_set 0 (one chunk per reference) uses device 0 only;
    a list not longer than the chunk count is kept as given."""
    from pydynamo.averaging import symmetry as Y
    v = base_vpr(1)
    v["gpu_identifier_set"], v["file_table_initial"] = 0, "t.tbl"
    monkeypatch.setattr(dp, "load_vpr", lambda path: dict(v))
    monkeypatch.setattr(dp.DynamoProject, "validate", lambda self: None)
    monkeypatch.setattr(dp.DynamoProject, "_data_info", lambda self: {})
    devs = []
    monkeypatch.setattr(Y, "set_device", lambda d: devs.append(d))
    monkeypatch.delenv("PYDYNAMO_REPORT", raising=False)
    monkeypatch.setattr("pydynamo.engine.preflight", lambda: None)           # no Dynamo installation needed here
    mk = lambda gpus: dp.DynamoProject(str(tmp_path / "src"), str(tmp_path / "data"), str(tmp_path / "out"),
                                       gpus=gpus, echo=False, matlab_threads=1)
    pr = mk([0, 1])
    assert pr.gpus == [0] and pr.report["gpus"] == [0] and devs == [0]
    v["gpu_identifier_set"] = np.array([0, 1, 2, 3], dtype=np.uint8)
    assert mk([4, 5]).gpus == [4, 5] and mk(None).gpus == [0, 1, 2, 3]


def test_close_terminates_workers_still_busy_after_a_failure(tmp_path):
    """_ask_all raises on the first failed reply; the workers whose reply was not read may still be aligning and are
    terminated by close(). An idle worker ends on None; one still alive after the join timeout is terminated."""
    pr = bare_project(tmp_path, base_vpr())
    pr.workers = _workers({"ok": False, "error": "out of memory"}, None, {"ok": True})
    w = list(pr.workers)
    with pytest.raises(RuntimeError, match="device 0 failed"):
        pr._ask_all([{"op": "align"}] * 3)
    pr.close()
    assert pr.workers == []
    assert not w[0][2].terminated and not w[0][2].alive and w[0][1].sent[-1] is None
    assert w[1][2].terminated and None not in w[1][1].sent
    assert w[2][2].terminated                                       # its reply was not read either
    pr.workers = _workers({"ok": True}, {"ok": True}, stuck=(1,))
    w = list(pr.workers)
    assert len(pr._ask_all([{"op": "tags"}] * 2)) == 2
    pr.close()
    assert not w[0][2].terminated and not w[0][2].alive
    assert w[1][2].terminated and w[1][1].sent[-1] is None


def test_failed_engine_run_appends_dynamo_gpu_warning_block(tmp_path):
    """dynamo_pipeline_system_check_output_gpu.m:40-92: a failed engine run appends the warning block with the order
    (dynamo_execute_project.m:296-297), then the run stops; no engine stdout, gpulog.txt or 'Returning control'."""
    pr = bare_project(tmp_path / "p", base_vpr())
    pr._start_workers = lambda: None
    pr.workers = _workers({"ok": False, "error": "Error: could not open input file"})
    chunks = [{"ref": 1, "gpu": 0, "n": 1, "particle_list": "pl", "starting_btbl": "s", "refined_btbl": "r"}]
    with pytest.raises(RuntimeError):
        pr.compute(1, [1], chunks)
    order = "dynamo_compute_iteration_spp ./p/cards/ite_0001/card_iteref_ref_001_ite_0001.card"
    assert (tmp_path / "p" / "log.txt").read_text().split("\n")[:-1] == ["&" * 63] * 8 + [
        "GPU: Warning!!!",
        "something went wrong inside the GPU application while the system tried to execute the order:",
        order + " ", "", "Dynamo will abort project execution at this point."] + ["&" * 62] * 8
    assert not (tmp_path / "gpulog.txt").exists()


def _assemble_project(tmp_path, nref):
    """A bare project whose assemble() runs from the averaging on (gather, compare and threshold stubbed out)."""
    v = base_vpr(1)
    for r in range(1, 9):
        v["nref_r%d" % r] = nref
    pr = bare_project(tmp_path, v)
    pr.log, pr.keep_temp = (lambda m: None), False
    formats.write_card(pr.f_card_ite(1), {"nref": nref, "time_start_computation": "01-Jan-2026 00:00:00"})
    formats.write_txt(pr.p("next_iteration.txt"), 1)
    pr.gather_table = lambda ite, ref: None
    pr.compare_table = lambda ite, ref: {}
    pr.threshold_table = lambda ite, ref: {}
    for ref in range(1, nref + 1):
        formats.write_tbl(pr.f_av("refined_table", ref, 1, "tbl"), np.ones((2, 35)))
    return pr


def _average_out(n, value):
    return {"average": np.full((n, n, n), value), "average_unweighted": np.full((n, n, n), value),
            "fmask_average": np.ones((n, n, n)), "fweight_average_raw": np.ones((n, n, n)), "eo_fsc": np.ones(n)}


HEAD = ["[assemble] Gathering refined tables.", "[assemble] Computing table comparisons against previous iteration.",
        "[assemble] getting rid of overlapping particles:", "[assemble] Applying  thresholding:"]


def test_failed_average_logs_its_line_before_the_fatal_block_and_symmetrizes_nothing(tmp_path):
    """dynamo_iteration_assemble.m:343-355: the 'Computing average for ref' echo comes before the ok test, so a NaN
    average (pipeline/average.m ok = false) has its line before fatal_error's block. The symmetrized averages come from
    a later loop (:457-479), so a failure in reference 2 leaves none of reference 1."""
    pr = _assemble_project(tmp_path, 2)
    outs = {1: _average_out(4, 0.5), 2: _average_out(4, np.nan)}
    pr.average_from_table = lambda ite, ref, table, fsc_sym=None: outs[ref]
    with pytest.raises(dp.FatalIterationError):
        pr.assemble(1, [1, 2])
    assert (tmp_path / "log.txt").read_text().split("\n")[:-1] == HEAD + [
        "[assemble] Computing average for 2 multireference channels.",
        "[assemble] Computing average for ref 1 ite 1",
        "[assemble] Computing average for ref 1 ite 2",
        " FATAL ERROR DETECTED IN ITERATION 1",
        "   reason: [assemble] did create apparently valid table files, but it failed while using them  for averaging",
        "   Deactivating subsequent rounds."]
    assert formats.read_txt(pr.p("next_iteration.txt")).ravel().tolist() == [pr.N + 1]
    assert os.path.exists(pr.f_av("average", 1, 1)) and not os.path.exists(pr.f_av("average", 2, 1))
    assert not os.path.exists(pr.f_av("average_symmetrized", 1, 1))
    assert not os.path.exists(pr.f_av("fmask_average_symmetrized", 1, 1))


def test_assemble_symmetrizes_after_every_average_and_the_done_line(tmp_path, monkeypatch):
    """Success path of the split loops: the per-reference lines, then the 'done' line; the c1 symmetrized files are
    byte copies of the averages (06b 8)."""
    pr = _assemble_project(tmp_path, 2)
    monkeypatch.setattr(dp, "load_vpr", lambda path: pr.vpr)          # the adaptive bandpass test reads settings/
    outs = {1: _average_out(4, 0.5), 2: _average_out(4, 0.25)}
    pr.average_from_table = lambda ite, ref, table, fsc_sym=None: outs[ref]
    pr.assemble(1, [1, 2])
    log = (tmp_path / "log.txt").read_text().split("\n")[:-1]
    assert log[:7] == HEAD + ["[assemble] Computing average for 2 multireference channels.",
                              "[assemble] Computing average for ref 1 ite 1", "[assemble] Computing average for ref 1 ite 2"]
    assert re.fullmatch(r"done assembling intermediate average in \d+\.\d{6} seconds\. ", log[7])
    for ref in (1, 2):
        for k in ("average", "fmask_average"):
            assert open(pr.f_av(k + "_symmetrized", ref, 1), "rb").read() == open(pr.f_av(k, ref, 1), "rb").read()
    assert formats.read_txt(pr.p("next_iteration.txt")).ravel().tolist() == [2]


def test_empty_average_selection_is_dynamos_matlab_error_not_a_fatal_block(tmp_path):
    """genericInput2Average.m:298-301 returns before its output 'flags' is assigned (:316), which pipeline/average.m:100
    requests: MATLAB stops with an error. No 'Computing average for ref' line, no fatal_error block, next_iteration.txt
    unchanged (the iteration can be run again)."""
    pr = _assemble_project(tmp_path, 1)
    T = np.zeros((3, 35))
    T[:, 0], T[:, 1] = [1, 2, 3], 1                                   # aligned, none marked in column 3
    formats.write_tbl(pr.f_av("refined_table", 1, 1, "tbl"), T)
    with pytest.raises(RuntimeError) as e:
        pr.assemble(1, [1])
    assert not isinstance(e.value, dp.FatalIterationError)
    log = (tmp_path / "log.txt").read_text().split("\n")[:-1]
    assert log == HEAD + ["[assemble] Computing average for 1 multireference channels."]
    assert formats.read_txt(pr.p("next_iteration.txt")).ravel().tolist() == [1]


def test_check_fatal_error_returns_and_the_run_ends_normally(tmp_path):
    """dynamo_iteration_check.m:106, :118-149: fatal_error writes next_iteration = N + 1 and its block and returns;
    the next pass of dynamo_execute_project.m finds next_iteration > N and returns (:144-147), without an error."""
    pr = bare_project(tmp_path, base_vpr(1))                          # 2 iterations
    pr.log = lambda m: None
    pr.workers, pr.report, pr.report_path, pr.gpus = [], {"iterations": []}, None, [0]
    pr.initialize = lambda: None
    formats.write_txt(pr.p("next_iteration.txt"), 1)
    for ite in (1, 2):
        formats.write_card(pr.f_card_ite(ite), {"nref": 1})
    calls = []
    pr.setup = lambda ite: calls.append(ite) or ([1], [])
    pr.compute = lambda ite, refs, chunks: []

    def assemble(ite, refs):                                          # no averages: the check finds them missing
        formats.write_txt(pr.p("next_iteration.txt"), ite + 1)
        pr.set_surviving_refs(ite + 1, refs)
        return {}
    pr.assemble = assemble
    rep = pr.run()
    assert calls == [1]
    assert [it["ite"] for it in rep["iterations"]] == [1] and rep["iterations"][0]["fatal"] is True
    assert formats.read_txt(pr.p("next_iteration.txt")).ravel().tolist() == [pr.N + 1]
    bars = ["%" * 59] * 6
    assert (tmp_path / "log.txt").read_text().split("\n")[:-1] == [" "] + bars + [
        " ", " FATAL ERROR DETECTED IN ITERATION 1", '   reason: database item "fmask_average", ite:1, ref:1  ',
        "   Deactivating all subsequent iterations.", " "] + bars
    formats.write_tbl(pr.f_av("refined_table", 1, 1, "tbl"), np.ones((2, 35)))
    for k in ("average", "fmask_average"):
        formats.write_em(pr.f_av(k, 1, 1), np.ones((2, 2, 2)))
    assert pr.check(1) is True
    assert (tmp_path / "log.txt").read_text().endswith("iteration 1 ok\n \n \n \n")
