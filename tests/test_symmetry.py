"""pydynamo.averaging.symmetry (dynamo_sym / dynamo_hsym / csym / dsym of Dynamo 1.1.514; docs/analysis/10a_symmetry_1.1.514.md)
and its use in setup, assemble, the averaging FSC, the adaptive bandpass and dvcheck.

Ground truth:
  benchmarks/06_abp_align_eo/sym_reader/gt   headless Dynamo 1.1.514 (gt1.m): small 32^3 tests, the user's ite-1 pieces
  benchmarks/06_abp_align_eo/implA           headless 1.1.514 mod() probe (modprobe.m -> modprobe.txt)
  test_data/bin8_filament/abp_align_eo  the user's helical e/o ABP project (read only; copied into tmp_path)
"""
import hashlib
import os
import shutil

import numpy as np
import pytest

from pydynamo.bandpass import adaptive
from pydynamo.files import formats
from pydynamo.project import runner as DP
from pydynamo.averaging import symmetry as Y
from pydynamo import vpr as V
from pydynamo.vpr import checks as vpr_check

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(HERE)
U = os.path.join(P, "test_data", "bin8_filament")
E = os.path.join(U, "abp_align_eo")
GT = os.path.join(P, "benchmarks", "06_abp_align_eo", "sym_reader", "gt")
IA = os.path.join(P, "benchmarks", "06_abp_align_eo", "implA")
HAVE_GT = os.path.isfile(os.path.join(GT, "small_out.mat"))
HAVE_E = os.path.isfile(os.path.join(E, "results", "ite_0004", "averages", "alignmentRef2ToRef1_ite_0004.sal"))
OP = "h[-1.4,0.3]"
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()


def _mat(f):
    import scipy.io as sio
    return {k: (v.transpose(2, 1, 0) if getattr(v, "ndim", 0) == 3 else v)
            for k, v in sio.loadmat(os.path.join(GT, f)).items() if not k.startswith("__")}


# ------------------------------------------------------------------ parsing, dvcheck, refusals

def test_parse_sym_like_dynamo():
    # [RUN, 10a 1.1] dynamo_zaux_parse_sym of these strings in headless 1.1.514
    for s in ("h[-1.4,0.3]", "H[-1.4, 0.3]", "h-1.4,0.3"):
        assert Y.parse_sym(s) == ("h", [-1.4, 0.3], True)
    assert Y.parse_sym("c1") == ("none", [1.0], True)
    assert Y.parse_sym("c3")[:2] == ("c", [3.0]) and Y.parse_sym("3")[:2] == ("c", [3.0])
    assert Y.parse_sym("d2")[:2] == ("d", [2.0])
    assert Y.parse_sym("h[-1.4,0.3,,1]")[1] == [-1.4, 0.3, 0.0, 1.0]           # ',,' -> ',0,'
    assert not Y.parse_sym("x2")[2] and not Y.parse_sym("c")[2]
    with pytest.raises(ValueError):                                         # MATLAB error() (:174-178)
        Y.parse_sym("h[1.4]")
    with pytest.raises(NotImplementedError):
        Y.parse_sym("h[-1.4,0.1+0.2]")                                      # needs MATLAB's evaluator


def test_helical_repetitions():
    # dynamo_sym.m:405-440: 2*floor((64/0.3-2)/2)+1 = 211
    assert Y.helical_reps(64, [-1.4, 0.3]) == (-1.4, 0.3, 211)
    assert Y.helical_reps(64, [-1.4, 0.3, 0]) == (-1.4, 0.3, 211)
    assert Y.helical_reps(64, [-1.4, 0.3, 21, 3]) == (-1.4, 0.3, 21)      # 4th value ignored (cSymmetry = 1)
    assert len(Y._replist(64, -1.4, 0.0, 211)) == 211                       # fmask path, dz forced to 0
    with pytest.raises(NotImplementedError):
        Y.helical_reps(64, [-1.4, -0.3])
    assert Y.symorder(OP) == 360 / -1.4 and Y.symorder("c1") == 1.0 and Y.symorder("d3") == 3.0


def test_dvcheck_and_abp_accept_helical():
    assert vpr_check._sym_ok(OP) and vpr_check._sym_ok("c1") and not vpr_check._sym_ok("x2")
    assert adaptive.check_sym(OP, 64) == "h" and adaptive.check_sym("c1", 64) == "none" and adaptive.check_sym("d2", 64) == "d"
    for bad in ("ico", "cbo", "none"):
        with pytest.raises(NotImplementedError):
            adaptive.check_sym(bad, 64)


@pytest.mark.skipif(not os.path.isfile(os.path.join(IA, "modprobe.txt")), reason="mod probe output absent")
def test_matlab_mod_non_integer_divisor_vs_mcr():
    """MATLAB mod(x, -1.4) of headless 1.1.514 (204 probes around multiples of -1.4 and the user's 3 rows)."""
    n = 0
    for line in open(os.path.join(IA, "modprobe.txt")):
        t = line.split()
        if t[0] == "I":
            x, m = float(t[1]), float(t[2])
            assert DP.matlab_mod(np.array([x]), 360.0)[0] == m
            continue
        x, m = (float(t[1]), float(t[3])) if t[0] == "U" else (float(t[2]), float(t[4]))
        assert DP.matlab_mod(np.array([x]), -1.4)[0] == m
        n += 1
    assert n == 207


@pytest.mark.skipif(not HAVE_E, reason="user abp_align_eo project absent")
def test_col29_helical_fold_matches_user_table():
    """Column 29 of refined_table_ref_001_ite_0001 (dynamo_rigid_compare, symorder 360/-1.4): 4711/4711 rows."""
    cur = formats.read_tbl(os.path.join(E, "results/ite_0001/averages/refined_table_ref_001_ite_0001.tbl"))
    prev = formats.read_tbl(os.path.join(E, "settings/table_initial_ref_001.tbl"))
    pi = {int(t): i for i, t in enumerate(np.real(prev[:, 0]).tolist())}
    pr = np.real(prev[[pi[int(t)] for t in np.real(cur[:, 0]).tolist()]])
    _, _, c29, _ = DP.compare_columns(np.real(cur), pr, DP.FdlibmTrig(), Y.symorder(OP))
    assert np.array_equal(np.array([float("%.5g" % v) for v in c29]), np.real(cur[:, 28]))


def _project_copy(tmp_path, low=None):
    root = tmp_path / "root"
    Pp = root / "abp_align_eo"
    shutil.copytree(os.path.join(E, "settings"), str(Pp / "settings"))
    os.chmod(str(Pp / "settings"), 0o755)
    for f in os.listdir(str(Pp / "settings")):
        os.chmod(str(Pp / "settings" / f), 0o644)
    os.symlink(os.path.join(U, "abp_align"), str(root / "abp_align"))
    os.symlink(os.path.join(U, "filamentsData_ctf"), str(root / "filamentsData_ctf"))
    fv = str(Pp / "settings" / "virtual_project.mat")
    v = V.modify(V.read_vpr_file(fv), "path", str(root))
    if low is not None:
        v = V.modify(v, "low_r1", float(low))
    V.write_vpr_file(fv, v)
    return root, Pp


@pytest.mark.skipif(not HAVE_E, reason="user abp_align_eo project absent")
def test_user_project_validates_and_dvchecks(tmp_path):
    """The user's helical e/o ABP project passes DynamoProject.validate (incl. adaptive.check_supported) and dvcheck."""
    root, Pp = _project_copy(tmp_path)
    pr = DP.DynamoProject(str(Pp), str(root / "filamentsData_ctf"), str(Pp), echo=False)
    assert str(pr.vpr["sym_r1"]) == OP
    v = DP.load_vpr(str(Pp / "settings" / "virtual_project.mat"))
    assert all(vpr_check._sym_ok(v["sym_r%d" % r]) for r in range(1, 9))


@pytest.mark.skipif(not HAVE_E, reason="user abp_align_eo project absent")
def test_rerun_initialize_restores_initial_low(tmp_path):
    """10c G4: re-running the user's project (vpr low_r1 = 32, written by ABP) under another root applies unfold's
    unfoldCheckAdaptiveBandpassInitial rule: card ite 1 gets low = adaptive_bandpass_initial = 25 as Dynamo's did."""
    root, Pp = _project_copy(tmp_path)
    out = tmp_path / "run" / "abp_align_eo"
    pr = DP.DynamoProject(str(Pp), str(root / "filamentsData_ctf"), str(out), echo=False)
    pr.initialize()
    c = formats.read_card(pr.f_card_ite(1))
    assert float(c["low"]) == 25.0
    assert float(V.read_vpr_file(str(out / "settings" / "virtual_project.mat"))["low_r1"]) == 25.0
    lines = open(os.path.join(E, "cards/ite_0001/card_ite_ite_0001.card")).read().splitlines()
    assert "low=25;" in lines and "low=25;" in open(pr.f_card_ite(1)).read().splitlines()


# ------------------------------------------------------------------ the operator vs headless Dynamo (small)

def _check_small(dev):
    I, O = _mat("small_in.mat"), _mat("small_out.mat")
    v, f = I["v"], I["f"]
    assert np.array_equal(Y.cylinder(32), O["cyl"])
    hs, R = Y.hsym(v, -1.4, 0.3, 105, device=dev)
    assert R == 105 and np.array_equal(hs, O["hs"])                           # float64-exact
    s, _ = Y.dynamo_sym(v, OP, device=dev)
    assert np.array_equal(s, O["s_h"])
    s, F = Y.dynamo_sym(v, OP, f, "binary", device=dev)                       # random fmask: Fourier division
    assert np.array_equal(s.astype(np.float32), O["s_hf"].astype(np.float32)) and np.array_equal(F, O["f_hf"])
    _, W = Y.dynamo_sym(v, OP, f, "weights", device=dev)
    assert np.array_equal(W, O["w_hf"])
    for op, k in (("c3", "c3"), ("d2", "d2")):
        s, _ = Y.dynamo_sym(v, op, device=dev)
        assert np.array_equal(s, O["s_" + k])
        s, F = Y.dynamo_sym(v, op, f, "binary", device=dev)
        assert np.array_equal(s.astype(np.float32), O["s_%sf" % k].astype(np.float32))
        assert np.array_equal(F, O["f_%sf" % k])
    ps = os.path.join(IA, "particle_sym.mat")
    if os.path.isfile(ps):                    # dynamo_particle 'sym' action with a random fmask (the eo_fsc path)
        import scipy.io as sio
        m = {k: v_.transpose(2, 1, 0) for k, v_ in sio.loadmat(ps).items() if not k.startswith("__")}
        assert np.array_equal(Y.dynamo_sym(v, OP, f, device=dev)[0].astype(np.float32), m["vp"].astype(np.float32))
        assert np.array_equal(Y.dynamo_sym(v, "c3", f, device=dev)[0].astype(np.float32), m["vc"].astype(np.float32))
    for op, fn in ((OP, "small_sym_h.em"), ("c3", "small_sym_c3.em"), ("d2", "small_sym_d2.em")):
        assert np.array_equal(Y.dynamo_sym(v, op, device=dev)[0].astype(np.float32),
                              formats.read_em(os.path.join(GT, fn)))


@pytest.mark.skipif(not HAVE_GT, reason="symmetry ground truth absent")
def test_small_symmetry_cpu_vs_dynamo():
    _check_small("cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_GT, reason="symmetry ground truth absent")
def test_small_symmetry_gpu_vs_dynamo():
    _check_small("gpu")


# ------------------------------------------------------------------ the user's files (GPU: 422 resamplings / 64^3)

@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_E, reason="user abp_align_eo project absent")
@pytest.mark.parametrize("ite", [1, 4])
def test_setup_helical_matches_user_files(tmp_path, ite):
    """preprocessReferences with sym h[-1.4,0.3] and symmetrize_with_fmask 1 (10a 3.1-3.2): the transformed
    templates, template fmasks and mask of the user's iterations 1 (low 25) and 4 (low 32) are byte-identical.
    (Iterations 2 and 3: one voxel of ref 1's transformed template differs by one float32 ulp, a band-pass FFT
    round-off at a float32 rounding midpoint, benchmarks/06_abp_align_eo/implA/v_setup_ulp.log; all else identical.)"""
    root, Pp = _project_copy(tmp_path, low=25 if ite == 1 else 32)
    sv = "results/ite_%04d/starting_values/" % ite
    os.makedirs(str(Pp / sv))
    for r in (1, 2):
        for k in ("starting_reference", "fmask_starting_reference"):
            nm = "%s_ref_%03d_ite_%04d.em" % (k, r, ite)
            shutil.copyfile(os.path.join(E, sv, nm), str(Pp / sv / nm))
    pr = DP.DynamoProject(str(Pp), str(root / "filamentsData_ctf"), str(Pp), echo=False)
    pr.preprocess_references(ite, [1, 2])
    names = ["mask_transformed_ite_%04d.em" % ite] + ["%s_ref_%03d_ite_%04d.em" % (k, r, ite) for r in (1, 2) for k in
                                                     ("starting_reference_transformed",
                                                      "fmask_starting_reference_transformed")]
    for nm in names:
        assert md5(str(Pp / sv / nm)) == md5(os.path.join(E, sv, nm)), nm


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_E, reason="user abp_align_eo project absent")
@pytest.mark.parametrize("ite", [1, 2, 3, 4])
def test_assemble_symmetrized_from_user_average(tmp_path, ite):
    """dynamo_sym(average, op, 'fmask', fmask_average, 'foutput', 'binary') on the user's ref-1 average: the user's
    average_symmetrized and (all-ones) fmask_average_symmetrized, byte-identical."""
    av = os.path.join(E, "results/ite_%04d/averages/" % ite)
    f = lambda k: av + "%s_ref_001_ite_%04d.em" % (k, ite)
    As, Fs = Y.dynamo_sym(formats.read_em(f("average")), OP, formats.read_em(f("fmask_average")), "binary")
    formats.write_em(str(tmp_path / "a.em"), As)
    formats.write_em(str(tmp_path / "f.em"), Fs)
    assert md5(str(tmp_path / "a.em")) == md5(f("average_symmetrized"))
    assert md5(str(tmp_path / "f.em")) == md5(f("fmask_average_symmetrized")) and np.all(Fs == 1)


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_GT, reason="symmetry ground truth absent")
def test_eo_fsc_helical_from_dynamo_halves():
    """fsc_local with fsc_actions 'sym h[-1.4,0.3];' (10a 5.4) on Dynamo's own odd/even averages and fmasks of the
    user's ite-1 ref-1 average: the symmetrized halves equal dynamo_particle's, and the interpolated FSC equals the
    user's eo_fsc_ref_001_ite_0001.fsc bytes."""
    from pydynamo.averaging.average import fsc_fouriermask
    H, S = _mat("eo_halves_ref001_ite0001.mat"), _mat("eo_sym_halves_ref001_ite0001.mat")
    vs = [Y.dynamo_sym(H[a], OP, (H[m] >= 0.5).astype(np.float64))[0]
          for a, m in (("oa", "ofm"), ("ea", "efm"))]
    assert np.array_equal(vs[0], S["vp"]) and np.array_equal(vs[1], S["vpe"])
    fs = fsc_fouriermask(vs[0], vs[1], 16)
    eo = np.interp(np.linspace(1, 32, 32), np.linspace(1, 32, 16), fs)
    with open(os.path.join(GT, "eo_fsc_h_ref001_ite0001.fsc"), "rb") as fh:
        ref = fh.read()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        formats.write_fsc(os.path.join(d, "e.fsc"), eo.reshape(1, -1))
        assert open(os.path.join(d, "e.fsc"), "rb").read() == ref
