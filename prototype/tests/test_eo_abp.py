"""Even/odd project (dveo) and golden-standard adaptive bandpass (09a, 09b).

CPU tests on synthetic data, and replays against Dynamo 1.1.514's own files:
  benchmarks/05_abp_bin4_eo/dynamo/reference          the reader's dveo/check/unfold runs (eo_project_pristine.tar, snaps)
  benchmarks/05_abp_bin4_eo/checks/impl_gt/gt_align    headless dynamo_align / angleset / kernels / table transform (gt_align.m)
  benchmarks/05_abp_bin4_eo/checks/impl_gt/harness     adaptiveFiltering on the user's data (snap3 = input, snap4 = Dynamo output)
  test_data/bin8_filament            the user's project (read only; copied into tmp_path)
"""
import hashlib
import math
import os
import shutil
import subprocess
import tarfile

import numpy as np
import pytest

from pydyn import abp, dalign as D, dynio, dynproject as DP, vpr as V, vpreo, vprcheck

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(os.path.dirname(HERE))
TD = os.path.join(P, "test_data", "bin8_filament")
REF = os.path.join(P, "benchmarks", "05_abp_bin4_eo", "dynamo", "reference")
GT = os.path.join(P, "benchmarks", "05_abp_bin4_eo", "checks", "impl_gt")
# Dynamo's snapshot files record the absolute folder they were made in. The folder had its pre-2026-11-13 name then
# (benchmarks/README.md, "Old names"), so that recorded form is normalised too.
REF_RECORDED = os.path.join(P, "benchmarks", "abp", "reference")
recorded = lambda path: path.replace(REF, REF_RECORDED)
md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
f2c = lambda a: np.transpose(a, (2, 1, 0))


def _gt():
    import scipy.io as sio
    return sio.loadmat(os.path.join(GT, "gt_align", "gt_align.mat"), squeeze_me=True)


def _user_copy(w):
    """The layout of the reader's work folders: a real, writable copy of the user's project as bin4_align_1, real
    copies of the linked table/template, the data folder as a symlink (nothing writes particles)."""
    os.makedirs(w, exist_ok=True)
    subprocess.check_call(["cp", "-r", os.path.join(TD, "bin8_align_1"), os.path.join(w, "bin4_align_1")])
    subprocess.check_call(["chmod", "-R", "u+w", os.path.join(w, "bin4_align_1")])
    for f in ("particles_edit_mod.tbl", "raw_template_ctf.em"):
        shutil.copyfile(os.path.join(TD, f), os.path.join(w, f))
    os.symlink(os.path.join(TD, "filamentsData_ctf"), os.path.join(w, "filamentsData_ctf"))


HAVE_USER = os.path.isdir(os.path.join(TD, "bin8_align_1"))
HAVE_GT = os.path.isfile(os.path.join(GT, "gt_align", "gt_align.mat"))
HAVE_HARNESS = os.path.isdir(os.path.join(GT, "harness", "snap4_after_abp"))


# =====================================================================================================================
# synthetic
# =====================================================================================================================

def test_table_eo_split_rule():
    T = np.zeros((9, 35))
    T[:, 0] = np.arange(101, 110)
    T[:, 1] = [1, 1, 0, 1, 1, 1, 1, 0, 1]            # 7 selected rows: an odd count drops the last one
    odd, even = vpreo.table_eo(T)
    sel = T[T[:, 1] == 1, 0]
    assert odd[:, 0].tolist() == sel[1::2].tolist()   # rows 2, 4, 6 -> ref 1
    assert even[:, 0].tolist() == sel[0:6:2].tolist()  # rows 1, 3, 5 (end-1) -> ref 2
    assert 109 not in odd[:, 0] and 109 not in even[:, 0]


def test_matlab_colon():
    assert D.matlab_colon(0.0, 3.0, 10.0).tolist() == [0, 3, 6, 9]
    assert D.matlab_colon(0.0, 1.5, 3.0).tolist() == [0, 1.5, 3]
    assert D.matlab_colon(5.0, -2.5, 0.0).tolist() == [5, 2.5, 0]
    assert D.matlab_colon(1.0, 1.0, 0.0).size == 0
    a, d = 0.1, 0.1                                   # the right end snaps to b within 2*eps*max(|a|,|b|)
    assert D.matlab_colon(a, d, 0.3)[-1] == 0.3


def test_fsc_resolution_rules():
    # 09b 9.3 [RUN]
    assert abp.fsc_resolution([0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3], 0.5) == 6
    assert abp.fsc_resolution([0.3, 0.2, 0.9, 0.8, 0.6, 0.4], 0.5) == 6
    assert abp.fsc_resolution([0.9] * 4, 0.5) == 4
    assert abp.fsc_resolution([0.9, 0.9, np.nan, np.nan], 0.5) == 3


def test_check_golden_needs_two_references():
    # dynamo_vpr_check.m:1290-1301 (check_goldenStandardReferences) and the run-time validation
    vpr = {"nref_r1": 1.0}
    with pytest.raises(NotImplementedError):
        abp.check_supported(vpr, [1])


def test_check_supported_refuses_bandpass_only_failures(tmp_path):
    """Settings that would only fail inside the bandpass step, after a full iteration, are refused up front: a zero
    search-area component with modus > 0 (dynamo__align_motor.m:233-248) and a missing / non-3-D file_mask
    (dynamo_align.m:1068-1082 default ellipsoid)."""
    import types
    vpr = {"nref_r1": 2.0, "sym_r1": "c1", "localnc_r1": 1.0, "area_search_modus_r1": 4.0,
           "area_search_r1": np.array([4.0, 4.0, 2.0]), "file_mask": "p/settings/mask.em"}
    for p in D.BASIC:
        vpr.setdefault("%s_r1" % p, 1.0)
    src = tmp_path / "p"
    (src / "settings").mkdir(parents=True)
    proj = types.SimpleNamespace(name="p", source=str(src), cwd=str(tmp_path))
    abp.check_supported(vpr, [1])                                   # no proj: no file check
    with pytest.raises(NotImplementedError, match="missing file_mask"):
        abp.check_supported(vpr, [1], proj)
    dynio.write_em(str(src / "settings" / "mask.em"), np.ones((8, 8, 8), dtype=np.float32))
    abp.check_supported(vpr, [1], proj)
    for bad in (np.array([4.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0])):
        with pytest.raises(NotImplementedError, match="zero component"):
            abp.check_supported(dict(vpr, area_search_r1=bad), [1], proj)
    abp.check_supported(dict(vpr, area_search_r1=np.array([4.0, 0.0, 2.0]), area_search_modus_r1=0.0), [1], proj)
    abp.check_supported(dict(vpr, area_search_r1=np.array([0.5, 4.0, 2.0])), [1], proj)   # ceil(0.5) = 1


def test_fsc_shells_synthetic():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16, 16))
    c = abp.fsc(a, a)
    assert c.shape == (8,) and np.allclose(c, 1, atol=1e-12)
    assert np.allclose(abp.fsc(a, -a), -1, atol=1e-12)
    b = a.copy()
    b[0, 0, 0] = np.nan
    assert abp.fsc(a, b) is None


def test_table_align_zero_rotation_keeps_the_transform():
    rng = np.random.default_rng(2)
    T = np.zeros((20, 35))
    T[:, 0] = np.arange(1, 21)
    T[:, 3:6] = rng.uniform(-3, 3, (20, 3))
    T[:, 6:9] = rng.uniform(-170, 170, (20, 3))
    T[:, 7] = np.abs(T[:, 7])
    out = abp.table_align(T, [0.0, 0.0, 0.0], [-0.0, -0.0, -0.0])
    for r0, r1 in zip(T, out):                        # same rotation matrix, same shifts, other columns kept
        assert np.allclose(D.euler2matrix(*r0[6:9]), D.euler2matrix(*r1[6:9]), atol=1e-12)
        assert np.allclose(r0[3:6], r1[3:6], atol=1e-12)
    assert np.array_equal(np.delete(T, range(3, 9), 1), np.delete(out, range(3, 9), 1))


def test_mex_kernels_synthetic():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((16, 16, 16))
    r4 = D.mex_shift_rot(v, (10, 20, 30), (0.5, -0.25, 0), option=4, outscope=np.nan)
    r1, m1 = D.mex_shift_rot(v, (10, 20, 30), (0.5, -0.25, 0), option=1, outscope=0.0, mask=np.ones_like(v))
    inside = ~np.isnan(r4)
    assert np.array_equal(m1.astype(bool), inside)
    assert np.array_equal(r1[inside], r4[inside]) and np.all(r1[~inside] == 0)
    r0 = D.mex_shift_rot(v, (0, 0, 0), (0, 0, 0), option=0, outscope=-1.0)
    g = np.arange(16) - 8
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    rad = np.sqrt((x * x + y * y + z * z).astype(np.float32))
    keep = (rad < 8) & (x < 7) & (y < 7) & (z < 7)    # the +1 pivot must stay in the box
    assert np.array_equal(r0[keep], v[keep].astype(np.float32).astype(np.float64)) and np.all(r0[rad >= 8] == -1)


def test_peak_subpixel_symmetric_peak():
    v = np.zeros((8, 8, 8))
    v[4, 3, 2] = 1.0                                  # [z, y, x]
    v[4, 3, 1] = v[4, 3, 3] = 0.5
    pos, val = D.peak_subpixel(v)
    assert pos.tolist() == [3, 4, 5] and val == 1.0   # 1-based [x, y, z]


# =====================================================================================================================
# replays against Dynamo 1.1.514
# =====================================================================================================================

@pytest.mark.skipif(not (HAVE_USER and os.path.isfile(os.path.join(REF, "eo_project_pristine.tar"))),
                    reason="user project or Dynamo dveo reference absent")
def test_dveo_dbp_matches_dynamo(tmp_path, monkeypatch):
    w = str(tmp_path / "w")
    _user_copy(w)
    monkeypatch.chdir(w)
    with pytest.raises(NotImplementedError):          # 'ite' > 0: source only (09a O1)
        vpreo.dveo("bin4_align_1", "ite", 4)
    assert not os.path.exists("bin4_align_1_eo")
    vpreo.dveo("bin4_align_1", "dbp", 1)
    rep = vprcheck.check("bin4_align_1_eo", 0)
    assert rep.fields_error == [] and rep.fields_warning == []
    assert V.unfold("bin4_align_1_eo", verbose=False)[1] == "ok"
    ref = str(tmp_path / "ref")
    with tarfile.open(os.path.join(REF, "eo_project_pristine.tar")) as t:
        t.extractall(ref)
    ref_cwd = os.path.join(REF, "work_dbp")
    n = 0
    for d, _, fs in os.walk(os.path.join(ref, "bin4_align_1_eo")):
        for f in fs:
            b = os.path.join(d, f)
            rel = os.path.relpath(b, ref)
            a = os.path.join(w, rel)
            assert os.path.isfile(a), rel
            if f == "virtual_project.mat":
                da, db = V.dump_vpr(a), V.dump_vpr(b)
                key = lambda d: [x[:3] if x[0] != "path" else x[:2] for x in d]
                assert key(da) == key(db)                                    # names, order, classes, dims
                bad = [x[0] for x, y in zip(da, db) if x[0] != "path" and not V.values_equal(x[3], y[3])]
                assert bad == [] and dict((x[0], x[3]) for x in da)["path"] == w
                continue
            xa, xb = open(a, "rb").read(), open(b, "rb").read()
            if f.endswith((".doc", ".card")):
                xa = xa.replace(w.encode(), b"<CWD>")
                xb = xb.replace(ref_cwd.encode(), b"<CWD>").replace(recorded(ref_cwd).encode(), b"<CWD>")
            assert xa == xb, rel
            n += 1
    assert n == 26


@pytest.mark.skipif(not (HAVE_USER and os.path.isdir(os.path.join(REF, "snaps", "dvput3_after_unfold"))),
                    reason="user project or Dynamo dveo snapshots absent")
@pytest.mark.parametrize("variant", ["default", "dvput"])
def test_dveo_variants_match_dynamo(tmp_path, monkeypatch, capsys, variant):
    """dveo without -dbp, and dveo + 'dvput adaptive_bandpass 1' (unfold then forces low_r1 := initial 16)."""
    w = str(tmp_path / "w")
    _user_copy(w)
    monkeypatch.chdir(w)
    vpreo.dveo("bin4_align_1", "v", 0)
    if variant == "dvput":
        V.dvput("bin4_align_1_eo", "d", "adaptive_bandpass", 1, verbose=False)
    capsys.readouterr()
    assert V.unfold("bin4_align_1_eo", verbose=False)[1] == "ok"
    echo = [x for x in capsys.readouterr().out.splitlines() if "[adaptive filtering" in x]
    # Dynamo's own stdout of the same unfold: benchmarks/05_abp_bin4_eo/dynamo/reference/logs/a_dveo_variants.log:128-129
    assert echo == ([" [adaptive filtering] Overruling current value for low pass in first iteration: 25",
                     " [adaptive filtering] Using initial lowpass: 16"] if variant == "dvput" else [])
    ref = os.path.join(REF, "snaps", "%s3_after_unfold" % variant)
    ref_cwd = os.path.join(REF, "work_%s" % variant)
    for d, _, fs in os.walk(ref):
        for f in fs:
            b = os.path.join(d, f)
            a = os.path.join(w, "bin4_align_1_eo", os.path.relpath(b, ref))
            if f == "virtual_project.mat":
                da, db = V.dump_vpr(a), V.dump_vpr(b)
                assert [x[0] for x in da] == [x[0] for x in db]
                assert [x[0] for x, y in zip(da, db) if x[0] != "path" and not V.values_equal(x[3], y[3])] == []
                continue
            xa, xb = open(a, "rb").read(), open(b, "rb").read()
            if f.endswith((".doc", ".card")):
                xa = xa.replace(w.encode(), b"<CWD>")
                xb = xb.replace(ref_cwd.encode(), b"<CWD>").replace(recorded(ref_cwd).encode(), b"<CWD>")
            assert xa == xb, os.path.relpath(b, ref)
    low = V.load("bin4_align_1_eo", verbose=False)["low_r1"]
    assert low == (16 if variant == "dvput" else 25)


@pytest.mark.skipif(not (HAVE_USER and os.path.isfile(os.path.join(REF, "logs", "h_dveo_save_u_unfold.log"))),
                    reason="user project or Dynamo dveo 'save' log absent")
@pytest.mark.parametrize("save", ["u", "unfold"])
def test_dveo_save_u_unfold_matches_dynamo(tmp_path, monkeypatch, capsys, save):
    """dynamo_vpr_eo.m:397-403 `case 'u','unfold'`: 'u' displays ans = 'unfold' and unfolds; 'unfold' does not
    unfold. Stdout equals Dynamo's line for line (benchmarks/05_abp_bin4_eo/dynamo/reference/logs/h_dveo_save_u_unfold.log)."""
    w = str(tmp_path / "w")
    _user_copy(w)
    monkeypatch.chdir(w)
    capsys.readouterr()
    vpreo.dveo("bin4_align_1", "save", save)
    out = capsys.readouterr().out.splitlines()
    log = open(os.path.join(REF, "logs", "h_dveo_save_u_unfold.log")).read().splitlines()
    assert out == log[log.index("GT ===== save %s" % save) + 1:log.index("GT ===== after save %s" % save)]
    assert os.path.isfile("bin4_align_1_eo.exe") == (save == "u")
    assert os.path.isdir("bin4_align_1_eo/cards") == (save == "u")


@pytest.mark.skipif(not (HAVE_USER and os.path.isfile(os.path.join(REF, "eo_project_pristine.tar"))),
                    reason="user project or Dynamo dveo reference absent")
def test_setup_ite1_eo_matches_user_dynamo(tmp_path):
    """Both references' transformed template/fmask and the mask equal the user's own 1.1.514 ite-1 files (same
    template, low and mask); 1178/1178/1178/1177 particles per GPU and reference; surviving refs '1 2'."""
    w = str(tmp_path)
    with tarfile.open(os.path.join(REF, "eo_project_pristine.tar")) as t:
        t.extractall(w)
    subprocess.check_call(["cp", "-r", os.path.join(TD, "bin8_align_1"), os.path.join(w, "bin4_align_1")])
    os.symlink(os.path.join(TD, "filamentsData_ctf"), os.path.join(w, "filamentsData_ctf"))
    proj = DP.DynamoProject(os.path.join(w, "bin4_align_1_eo"), os.path.join(w, "filamentsData_ctf"),
                            os.path.join(w, "out", "bin4_align_1_eo"), echo=False)
    proj.initialize()
    refs, chunks = proj.setup(1)
    U = os.path.join(TD, "bin8_align_1", "results", "ite_0001", "starting_values")
    for ref in (1, 2):
        assert md5(proj.p(proj.r_template_transformed(ref, 1))) == \
            md5(os.path.join(U, "starting_reference_transformed_ref_001_ite_0001.em"))
        assert md5(proj.p(proj.r_fmask_transformed(ref, 1))) == \
            md5(os.path.join(U, "fmask_starting_reference_transformed_ref_001_ite_0001.em"))
        c = dynio.read_card(proj.f_card_iteref(ref, 1))
        assert int(c["database_ref"]) == ref and int(c["nref"]) == 2
    assert md5(proj.p(proj.r_mask_transformed(1))) == md5(os.path.join(U, "mask_transformed_ite_0001.em"))
    assert open(proj.f_surv(1)).read() == "1 2\n" and refs == [1, 2]
    assert [(c["ref"], c["n"]) for c in chunks] == [(1, 1178), (1, 1178), (1, 1178), (1, 1177),
                                                    (2, 1178), (2, 1178), (2, 1178), (2, 1177)]


@pytest.mark.skipif(not HAVE_GT, reason="dynamo_align ground truth absent")
def test_dalign_kernels_and_angles_vs_dynamo():
    m = _gt()
    A1 = dynio.read_em(os.path.join(GT, "gt_align", "in", "average_ref_001_ite_0001.em"))
    M = dynio.read_em(os.path.join(TD, "bin8_align_1", "settings", "editorMask.em"))
    rt, rm = D.mex_shift_rot(A1, (12, 5, -33), (0, 0, 0), option=1, outscope=0.0, mask=M)
    assert np.array_equal(rt, f2c(m["rt"])) and np.array_equal(rm, f2c(m["rm"]))
    E = D.ellipsoid([4, 4, 2], 64)
    assert np.array_equal(E, f2c(m["E"]))
    assert np.array_equal(D.rot_shift(E, (12, 5, -33), (1, -1, 0), -1.0), f2c(m["Er"]))
    P_ = dynio.read_em(os.path.join(GT, "gt_align", "in", "particle_rot.em"))
    assert np.max(np.abs(D.normalize_whole(P_) - f2c(m["pn"]))) < 1e-12
    for k, args in (("a1", (20, 3, 360, 15, (0, 0, 0))), ("a2", (6, 1.5, 30, 7.5, (30, 12, -50))),
                    ("a3", (3, 0.75, 15, 3.75, (-170, 175, 33))), ("a4", (20, 3, 45, 5, (0, 0, 0))),
                    ("a5", (6, 1.5, 10, 2.5, (0, -7, 12)))):
        a, g = D.angleset(*args), np.atleast_2d(m[k])
        assert a.shape == g.shape and np.max(np.abs(a - g)) < 1e-12, k     # same set and order; trig roundoff


@pytest.mark.skipif(not HAVE_GT, reason="dynamo_align ground truth absent")
def test_table_align_fmask_rotation_fsc_vs_dynamo():
    m = _gt()
    T = dynio.read_tbl(os.path.join(GT, "gt_align", "in", "refined_table_ref_002_ite_0001.tbl"))
    out = np.real(abp.table_align(T, m["r1_Tp_shifts"], m["r1_Tp_eulers"]))
    assert np.array_equal(out, m["Ttab"])              # MATLAB's trig (pydyn.fdlibm): bit-identical doubles
    F2 = dynio.read_em(os.path.join(GT, "gt_align", "in", "fmask_average_ref_002_ite_0001.em"))
    assert np.array_equal(D.mex_shift_rot(F2, m["r1_Tp_eulers"], (0, 0, 0), option=0), f2c(m["F2r"]))
    A2 = dynio.read_em(os.path.join(GT, "gt_align", "in", "average_ref_002_ite_0001.em"))
    assert np.array_equal(D.mex_shift_rot(A2, m["r1_Tp_eulers"], m["r1_Tp_shifts"], option=0), f2c(m["A2r"]))
    A1 = dynio.read_em(os.path.join(GT, "gt_align", "in", "average_ref_001_ite_0001.em"))
    M = dynio.read_em(os.path.join(TD, "bin8_align_1", "settings", "editorMask.em"))
    c = abp.fsc(A1 * M, A2 * M)
    assert np.max(np.abs(c - m["fscA"])) < 1e-12 and abp.fsc_resolution(c, 0.5) == int(m["fpA"]) == 28


@pytest.mark.skipif(not (HAVE_GT and HAVE_USER), reason="dynamo_align ground truth absent")
def test_dynamo_align_rotated_round2_vs_dynamo():
    m = _gt()
    vpr = DP.load_vpr(os.path.join(GT, "harness", "snap3_before_abp", "settings", "virtual_project.mat"))
    A1 = dynio.read_em(os.path.join(GT, "gt_align", "in", "average_ref_001_ite_0001.em"))
    P_ = dynio.read_em(os.path.join(GT, "gt_align", "in", "particle_rot.em"))
    M = dynio.read_em(os.path.join(TD, "bin8_align_1", "settings", "editorMask.em"))
    s = D.dynamo_align(P_, A1, vpr, 2, mask=M, verbose=False)
    assert s["ptable"].shape == m["r2_ptable"].shape
    assert np.max(np.abs(s["ptable"] - m["r2_ptable"])) < 1e-11
    assert np.max(np.abs(s["ccmax"] - m["r2_ccmax"])) < 1e-12
    for lev in range(3):
        ga, gc = np.atleast_2d(m["r2_angles"][lev]), np.ravel(m["r2_cc"][lev])
        assert s["angles"][lev].shape == ga.shape and np.max(np.abs(s["cc_angles"][lev] - gc)) < 1e-12
        assert int(np.flatnonzero(s["cc_angles"][lev] == s["cc_angles"][lev].max())[-1]) == \
            int(np.flatnonzero(gc == gc.max())[-1])


def _harness(w):
    S3 = os.path.join(GT, "harness", "snap3_before_abp")
    subprocess.check_call(["cp", "-r", S3, os.path.join(w, "bin4_align_1_eo")])
    subprocess.check_call(["cp", "-r", os.path.join(TD, "bin8_align_1"), os.path.join(w, "bin4_align_1")])
    subprocess.check_call(["chmod", "-R", "u+w", w])
    os.symlink(os.path.join(TD, "filamentsData_ctf"), os.path.join(w, "filamentsData_ctf"))
    root = os.path.join(w, "bin4_align_1_eo")
    with pytest.raises(RuntimeError, match="vpr.path"):          # in place, vpr.path is another folder: refused
        DP.DynamoProject(root, os.path.join(w, "filamentsData_ctf"), root, echo=False)
    f = os.path.join(root, "settings", "virtual_project.mat")
    v = V.read_vpr_file(f)
    v["path"] = w                                                 # as if dveo had been run here
    V.write_vpr_file(f, v)
    return root


def _compare_harness(root, files):
    S4 = os.path.join(GT, "harness", "snap4_after_abp")
    av = "results/ite_0001/averages/"
    for name in files:
        assert md5(os.path.join(root, av + name)) == md5(os.path.join(S4, av + name)), name
    assert md5(os.path.join(root, "results/bandpass_resolution.txt")) == \
        md5(os.path.join(S4, "results/bandpass_resolution.txt"))
    a = dynio.read_lines(os.path.join(root, "cards/ite_0002/card_ite_ite_0002.card"))
    b = dynio.read_lines(os.path.join(S4, "cards/ite_0002/card_ite_ite_0002.card"))
    assert [x for x in a if not x.startswith("path=")] == [x for x in b if not x.startswith("path=")]
    va = V.read_vpr_file(os.path.join(root, "settings/virtual_project.mat"))
    vb = V.read_vpr_file(os.path.join(S4, "settings/virtual_project.mat"))
    assert va["low_r1"] == vb["low_r1"] == 26
    assert [k for k in vb if k != "path" and not V.values_equal(va.get(k), vb[k])] == []
    lines = lambda f: [x for x in open(f).read().splitlines() if x.startswith("[bandpass")]
    assert lines(os.path.join(root, "log.txt")) == lines(os.path.join(S4, "log.txt"))


@pytest.mark.skipif(not (HAVE_HARNESS and HAVE_USER), reason="ABP harness absent")
def test_abp_harness_replay_cpu(tmp_path):
    """adaptiveFiltering on Dynamo's own ite-1 averages; the GPU re-average of reference 2 is replaced by Dynamo's
    float32 result. Every file Dynamo writes is byte-identical (card: except the test folder's path line)."""
    root = _harness(str(tmp_path))
    proj = DP.DynamoProject(root, os.path.join(str(tmp_path), "filamentsData_ctf"), root, echo=False)
    a2 = dynio.read_em(os.path.join(GT, "harness/snap4_after_abp/results/ite_0001/averages/"
                                        "average_ref_002_ite_0001.em"))
    proj.average_from_table = lambda ite, ref, table: {"average": a2}
    info = abp.adaptive_filtering(proj, 1)
    assert info["fp"] == 28 and info["new_low"] == 26 and info["next_round"] == 1
    _compare_harness(root, ("refined_table_ref_002_ite_0001.tbl", "average_not_redefined_ref_001_ite_0001.em",
                            "average_not_redefined_ref_002_ite_0001.em", "table_not_redefined_ref_002_ite_0001.em",
                            "fmask_average_symmetrized_ref_002_ite_0001.em", "bandpass_fsc_ite_0001.fsc"))


@pytest.mark.gpu
@pytest.mark.skipif(not (HAVE_HARNESS and HAVE_USER), reason="ABP harness absent")
def test_abp_harness_replay_gpu(tmp_path):
    """Same replay with pydyn's own GPU re-average of reference 2 (genericInput2Average): the resolution, the new low
    and the vpr/card must match; the FSC curve within the averaging kernel's tolerance (08b 4.6)."""
    root = _harness(str(tmp_path))
    proj = DP.DynamoProject(root, os.path.join(str(tmp_path), "filamentsData_ctf"), root, echo=False)
    try:
        # the re-average needs the particles of reference 2 resident on the workers (left there by 'align' in a run)
        proj._start_workers()
        T2 = dynio.read_tbl(proj.f_av("refined_table", 2, 1, "tbl"))
        _load_slot(proj, 2, np.real(T2))
        info = abp.adaptive_filtering(proj, 1)
    finally:
        proj.close()
    assert info["fp"] == 28 and info["new_low"] == 26
    x = dynio.read_fsc(os.path.join(root, "results/ite_0001/averages/bandpass_fsc_ite_0001.fsc")).ravel()
    y = dynio.read_fsc(os.path.join(GT, "harness/snap4_after_abp/results/ite_0001/averages/"
                                        "bandpass_fsc_ite_0001.fsc")).ravel()
    # measured 2026-11-12: all 32 shells equal at the file's %.5g precision (max|d| 0)
    assert md5(os.path.join(root, "results/ite_0001/averages/bandpass_fsc_ite_0001.fsc")) == \
        md5(os.path.join(GT, "harness/snap4_after_abp/results/ite_0001/averages/bandpass_fsc_ite_0001.fsc"))
    assert np.array_equal(x, y)


def _load_slot(proj, slot, T):
    """Put the particles of table T into worker slot `slot` (what the 'align' step leaves resident)."""
    import tempfile
    d = tempfile.mkdtemp(dir=proj.out)
    per = math.ceil(len(T) / len(proj.workers))
    msgs = []
    for i, _ in enumerate(proj.workers):
        rows = T[i * per:(i + 1) * per]
        pl, sb = os.path.join(d, "pl_%d.txt" % i), os.path.join(d, "s_%d.btbl" % i)
        dynio.write_lines(pl, [proj.particle_line(t) for t in rows[:, 0]])
        dynio.write_btbl(sb, rows)
        msgs.append({"op": "load", "slot": slot, "chunks": [{"particle_list": pl, "starting_btbl": sb}]})
    proj._ask_all(msgs)
    shutil.rmtree(d, ignore_errors=True)


# =====================================================================================================================
# Dynamo 1.1.514's own 4-iteration e/o + ABP run (benchmarks/05_abp_bin4_eo/dynamo/run): ite 4 has a non-trivial rotation
# (Tp eulers [9.2357, -1.0637, -9.2373]); every input below is Dynamo's file of that run
# =====================================================================================================================

RUN = os.path.join(P, "benchmarks", "05_abp_bin4_eo", "dynamo", "run")
RUNP = os.path.join(RUN, "bin4_align_1_eo")
HAVE_RUN = os.path.isfile(os.path.join(RUNP, "results", "ite_0004", "averages", "alignmentRef2ToRef1_ite_0004.sal"))


def _run_av(ite, name):
    return os.path.join(RUNP, "results", "ite_%04d" % ite, "averages", name)


def _run_sal(ite):
    import scipy.io as sio
    return sio.loadmat(_run_av(ite, "alignmentRef2ToRef1_ite_%04d.sal" % ite), squeeze_me=False,
                       struct_as_record=False)["contents"][0, 0]


@pytest.mark.skipif(not HAVE_RUN, reason="Dynamo 1.1.514 e/o ABP run absent")
def test_abp_ite4_real_run_cpu(tmp_path):
    """dynamo_align on Dynamo's two ite-4 averages vs its .sal; the .sal pydyn writes (fields, order, volumes);
    dynamo_table_align of the pre-ABP table with Dynamo's Tp vs Dynamo's rewritten table (text bytes); the fmask
    rotation; and table_not_redefined (ite 2) = float32 real part of the table, complex col 28 included."""
    import scipy.io as sio
    ite = 4
    vpr = DP.load_vpr(os.path.join(RUNP, "settings", "virtual_project.mat"))
    G = _run_sal(ite)
    A1 = dynio.read_em(_run_av(ite, "average_not_redefined_ref_001_ite_0004.em")).astype(np.float64)
    A2 = dynio.read_em(_run_av(ite, "average_not_redefined_ref_002_ite_0004.em")).astype(np.float64)
    assert np.array_equal(f2c(G.template), A1) and np.array_equal(f2c(G.particle), A2)
    s = D.dynamo_align(A2, A1, vpr, DP.ite2round(vpr, ite), mask=dynio.read_em(os.path.join(RUN, str(vpr["file_mask"]))),
                       verbose=False)
    assert np.max(np.abs(s["ptable"] - np.ravel(G.ptable))) < 1e-12           # measured 7.1e-15
    assert np.max(np.abs(s["ccmax"] - np.ravel(G.ccmax))) < 1e-12
    for lev in range(3):
        ga, gc = np.atleast_2d(G.angles[0, lev]), np.ravel(G.cc_angles[0, lev])
        assert s["angles"][lev].shape == ga.shape and np.max(np.abs(s["angles"][lev] - ga)) < 1e-12
        assert np.max(np.abs(s["cc_angles"][lev] - gc)) < 1e-12
    f = str(tmp_path / "x.sal")
    abp.write_sal(f, s, A1, A2)
    Pm = sio.loadmat(f, squeeze_me=False, struct_as_record=False)["contents"][0, 0]
    assert list(Pm._fieldnames) == list(G._fieldnames)                          # 38 fields, Dynamo's order
    assert str(Pm.dtype[0]) == "single_alignment"
    for k in ("aligned_particle", "aligned_template", "template", "particle"):
        assert np.array_equal(getattr(Pm, k), getattr(G, k)), k
    for k in D.BASIC + ("angles_all", "cc_angles_all", "t_eulers", "p_eulers"):
        a, b = np.asarray(getattr(Pm, k), np.float64), np.asarray(getattr(G, k), np.float64)
        assert a.shape == b.shape and np.max(np.abs(a - b)) < 1e-12, k
    for k in ("particle_name", "template_name", "mask_name", "table_name", "tfmask_name", "pfmask_name"):
        assert str(getattr(Pm, k)[0]) == str(getattr(G, k)[0]), k
    assert list(Pm.scard[0, 0]._fieldnames) == list(G.scard[0, 0]._fieldnames)
    tps, tpe = np.ravel(G.Tp[0, 0].shifts), np.ravel(G.Tp[0, 0].eulers)
    T_pre = dynio.read_tbl(os.path.join(RUNP, "results/ite_0005/starting_values/starting_table_ref_002_ite_0005.tbl"))
    T_dyn = dynio.read_tbl(_run_av(ite, "refined_table_ref_002_ite_0004.tbl"))
    assert dynio.tbl_bytes(abp.table_align(T_pre, tps, tpe)) == dynio.tbl_bytes(T_dyn)
    F2 = dynio.read_em(_run_av(ite, "fmask_average_ref_002_ite_0004.em"))
    Fd = dynio.read_em(_run_av(ite, "fmask_average_symmetrized_ref_002_ite_0004.em"))
    assert np.array_equal(D.mex_shift_rot(F2, tpe, (0, 0, 0), option=0).astype(np.float32), Fd.astype(np.float32))
    E = np.squeeze(dynio.read_em(_run_av(2, "table_not_redefined_ref_002_ite_0002.em")))
    T2 = dynio.read_tbl(_run_av(2, "refined_table_ref_002_ite_0002.tbl"))
    assert np.iscomplexobj(T2) and np.any(np.imag(T2[:, 27]) != 0)
    E = E if E.shape == T2.shape else E.T
    assert np.array_equal(E, np.real(T2).astype(np.float32))


@pytest.mark.gpu
@pytest.mark.skipif(not (HAVE_RUN and HAVE_USER), reason="Dynamo 1.1.514 e/o ABP run absent")
def test_abp_ite4_reaverage_real_run_gpu(tmp_path):
    """genericInput2Average of reference 2 on the GPU from Dynamo's ite-4 transformed table: A2' within 1e-5 of
    Dynamo's file (measured 8.6e-7, 9 voxels differ in float32) and bandpass_fsc_ite_0004.fsc byte-identical."""
    ite = 4
    G = _run_sal(ite)
    tps, tpe = np.ravel(G.Tp[0, 0].shifts), np.ravel(G.Tp[0, 0].eulers)
    T_pre = dynio.read_tbl(os.path.join(RUNP, "results/ite_0005/starting_values/starting_table_ref_002_ite_0005.tbl"))
    T2 = abp.table_align(T_pre, tps, tpe)
    shutil.copytree(os.path.join(RUNP, "settings"), str(tmp_path / "out" / "settings"))       # mask.em
    proj = DP.DynamoProject(RUNP, os.path.join(RUN, "filamentsData_ctf"), str(tmp_path / "out"), echo=False)
    try:
        proj._start_workers()
        _load_slot(proj, 2, np.real(T2))
        A2n = proj.average_from_table(ite, 2, T2)["average"]
    finally:
        proj.close()
    Ad = dynio.read_em(_run_av(ite, "average_ref_002_ite_0004.em")).astype(np.float64)
    assert np.max(np.abs(A2n - Ad)) < 1e-5
    A1 = dynio.read_em(_run_av(ite, "average_not_redefined_ref_001_ite_0004.em")).astype(np.float64)
    M = dynio.read_em(os.path.join(RUN, "bin4_align_1", "settings", "editorMask.em"))
    c = abp.fsc(A1 * M, A2n * M)
    assert dynio.dlmwrite_bytes(c.reshape(1, -1), 5) == open(_run_av(ite, "bandpass_fsc_ite_0004.fsc"), "rb").read()
    assert abp.fsc_resolution(c, 0.5) == 26


# ------------------------------------------------------------------ helical golden ABP on the user's abp_align_eo (10a 6)

EUSER = os.path.join(TD, "abp_align_eo")
HAVE_EUSER = os.path.isfile(os.path.join(EUSER, "results", "ite_0005", "starting_values", "starting_table_ref_002_ite_0005.tbl"))


@pytest.mark.gpu
@pytest.mark.skipif(not HAVE_EUSER, reason="user abp_align_eo project absent")
def test_abp_helical_ite4_user_files(tmp_path):
    """adaptiveFiltering of the user's iteration 4 (sym h[-1.4,0.3], adaptive_bandpass_symmetrized 1, threshold 0.143,
    pushback 0): input = the user's averages, with ref 2's pre-ABP average (average_not_redefined) and pre-ABP table
    (starting_table_ref_002_ite_0005, propagated before ABP and untouched by a setup 5); the ref-2 re-average is the
    user's float32 average_ref_002 (the symmetrized outputs depend on float32(A2') only). Outputs vs the user's
    files: average_symmetrized / fmask_average_symmetrized of ref 2, bandpass_fsc, refined_table_ref_002,
    table_not_redefined, bandpass_resolution.txt byte-identical; .sal Tp/ccmax to 1e-12; log lines identical."""
    import scipy.io as sio
    root = tmp_path / "root"
    Pp = root / "abp_align_eo"
    shutil.copytree(os.path.join(EUSER, "settings"), str(Pp / "settings"))
    os.chmod(str(Pp / "settings"), 0o755)
    for f in os.listdir(str(Pp / "settings")):
        os.chmod(str(Pp / "settings" / f), 0o644)
    os.makedirs(str(Pp / "cards" / "ite_0004"))
    shutil.copyfile(os.path.join(EUSER, "cards/ite_0004/card_ite_ite_0004.card"), str(Pp / "cards/ite_0004/card_ite_ite_0004.card"))
    os.symlink(os.path.join(TD, "abp_align"), str(root / "abp_align"))
    os.symlink(os.path.join(TD, "filamentsData_ctf"), str(root / "filamentsData_ctf"))
    av = "results/ite_0004/averages/"
    os.makedirs(str(Pp / av))
    f = lambda k, r, ext="em": av + "%s_ref_%03d_ite_0004.%s" % (k, r, ext)
    for src, dst in ((f("average", 1), f("average", 1)), (f("fmask_average", 1), f("fmask_average", 1)),
                     (f("fmask_average", 2), f("fmask_average", 2)), (f("average_not_redefined", 2), f("average", 2)),
                     (f("refined_table", 1, "tbl"), f("refined_table", 1, "tbl")),
                     ("results/ite_0005/starting_values/starting_table_ref_002_ite_0005.tbl", f("refined_table", 2, "tbl"))):
        shutil.copyfile(os.path.join(EUSER, src), str(Pp / dst))
    res = np.ravel(np.real(dynio.read_txt(os.path.join(EUSER, "results/bandpass_resolution.txt"))))
    dynio.write_txt(str(Pp / "results" / "bandpass_resolution.txt"), res[:3].reshape(1, -1))
    fv = str(Pp / "settings" / "virtual_project.mat")
    V.write_vpr_file(fv, V.modify(V.read_vpr_file(fv), "path", str(root)))
    proj = DP.DynamoProject(str(Pp), str(root / "filamentsData_ctf"), str(Pp), echo=False)
    a2n = dynio.read_em(os.path.join(EUSER, f("average", 2)))
    proj.average_from_table = lambda ite, ref, table, fsc_sym=None: {"average": a2n}
    info = abp.adaptive_filtering(proj, 4)
    assert info["fp"] == 32 and info["new_low"] == 32 and "next_round" not in info
    for nm in (f("average_symmetrized", 2), f("fmask_average_symmetrized", 2), av + "bandpass_fsc_ite_0004.fsc",
               f("refined_table", 2, "tbl"), f("table_not_redefined", 2), f("average_not_redefined", 1),
               "results/bandpass_resolution.txt"):
        assert md5(str(Pp / nm)) == md5(os.path.join(EUSER, nm)), nm
    load = lambda p: sio.loadmat(p, squeeze_me=True, struct_as_record=False)["contents"]
    m, u = load(str(Pp / av / "alignmentRef2ToRef1_ite_0004.sal")), load(os.path.join(EUSER, av, "alignmentRef2ToRef1_ite_0004.sal"))
    for a, b in ((m.Tp.shifts, u.Tp.shifts), (m.Tp.eulers, u.Tp.eulers), (m.ccmax, u.ccmax)):
        assert np.max(np.abs(np.ravel(a).astype(float) - np.ravel(b).astype(float))) < 1e-12
    mine = [l for l in open(str(Pp / "log.txt")).read().splitlines() if l.startswith("[bandpass")]
    with open(os.path.join(EUSER, "log.txt"), errors="replace") as fh:
        user = [l for l in fh.read().splitlines()[962542:] if l.startswith("[bandpass #4]")]
    assert mine == user
