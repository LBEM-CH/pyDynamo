"""The user's abp_align -> abp_align_eo journey (docs/analysis/10b_abp_align_journey_1.1.514.md) in pyDynamo:
mask editor (pydynamo.masks, dynamo_mask_GUI -> dynamo_mask cylinder + dynamo_mollify), the multireference side of the
dcp controller (open of a two-reference project, Multireference > Adaptive filtering > Derive a project, the
adaptive-bandpass parameter editor), and vpr.modify with numpy values.

Reference values are Dynamo 1.1.514's own (headless MCR runs in benchmarks/06_abp_align_eo/implB: k1_mask_dump.m,
k4_blob_dump.m; md5s of the MATLAB double dumps are inlined below). The replay on the user's data is
benchmarks/06_abp_align_eo/implB/scripts/j_pydyn_journey.py (test_user_journey_* runs it when the data is present).
"""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from pydynamo.files import formats
from pydynamo import masks
from pydynamo import vpr as V
from pydynamo.manager.controller import ABP_PARAMETERS, DcpController

from test_console_and_lifecycle import make_data

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, ".."))
T = os.path.join(P, "test_data", "bin8_filament")
JOURNEY = os.path.join(P, "benchmarks", "06_abp_align_eo", "implB", "scripts", "j_pydyn_journey.py")

# md5 of the MATLAB double arrays written with fwrite (x fastest = the bytes of a C-order [z, y, x] float64 array)
MD5_BLOB_64_6 = "6aaff71e5d1b2193f00a08387073c68f"      # dynamo_blob(64, 6)
MD5_CYL_15_15_15 = "bfc39723bd975bcf3c5deddf76d04243"   # dynamo_cylinder([15 15 15], [64 64 64], [], 0)
MD5_MASK_DOUBLE = "26f1c2c7708be1928e3be3eb58a99da0"    # dynamo_mask(... 'r',15,'height',30,'gaussian',3 ...)
# EM files written by dynamo_write (float32)
MD5_EDITOR_MASK = "90f4bb01a9f2fc670bd758ea3aeb58dd"    # = the user's abp_align/settings/editorMask.em
MD5_TUTORIAL_R12 = "5fa60032fde0e2288b51823298625da8"   # r 12 / h 30 / G 3 (journey_dynamo/m1)


def md5b(b):
    return hashlib.md5(b).hexdigest()


def md5(f):
    return md5b(open(f, "rb").read())


def needs_fftw():
    if masks.fft_backend() == "scipy":
        pytest.skip("no FFTW 3 library (set PYDYNAMO_FFTW_LIB): the mask is not byte-identical with scipy.fft")


# ---------------------------------------------------------------------------------------------------- masks
def test_matlab_exp_and_geometry_are_bit_identical():
    """MATLAB's exp (fdlibm) and the cylinder grid, bit for bit [RUN k4_blob_dump.m, k1_mask_dump.m]."""
    assert md5b(masks.dynamo_blob(64, 6).astype("<f8").tobytes()) == MD5_BLOB_64_6
    assert md5b(masks.dynamo_cylinder([15, 15, 15], [64, 64, 64], None, 0).astype("<f8").tobytes()) == \
        MD5_CYL_15_15_15
    # numpy's exp is not MATLAB's: +-1 ulp on 24,518 of the 262,144 blob values
    i = np.arange(-32, 32, dtype=float) / np.sqrt(2 * (6 / 2.35482) ** 2)
    z, y, x = np.meshgrid(i, i, i, indexing="ij")
    arg = -x ** 2 - y ** 2 - z ** 2
    assert np.count_nonzero(np.exp(arg) != masks.matlab_exp(arg)) > 0
    assert masks.matlab_exp(np.array([0.0, -1e-30]))[0] == 1.0


def test_dynamo_mask_cylinder_bytes_match_dynamo(tmp_path):
    needs_fftw()
    m = masks.dynamo_mask("size", 64, "type", "cylinder", "r", 15, "height", 30, "gaussian", 3, "invert", False)
    assert md5b(m.astype("<f8").tobytes()) == MD5_MASK_DOUBLE
    f = str(tmp_path / "m.em")
    formats.write_em(f, m)
    assert md5(f) == MD5_EDITOR_MASK
    # the same bytes for any height in [30, 32) and radius in [15, sqrt(226)) (10b 2.5)
    for kw in (dict(r=15, height=31), dict(r=15.03, height=30), dict(radius=15, h=30)):
        formats.write_em(f, masks.dynamo_mask(size=64, type="cylinder", gaussian=3, invert=False, **kw))
        assert md5(f) == MD5_EDITOR_MASK, kw
    formats.write_em(f, masks.dynamo_mask(size=64, type="cylinder", r=12, height=30, gaussian=3, invert=False))
    assert md5(f) == MD5_TUTORIAL_R12


def test_scipy_fallback_matches_above_the_noise_floor(monkeypatch):
    needs_fftw()
    ref = masks.dynamo_mask(size=64, type="cylinder", r=15, height=30, gaussian=3).astype(np.float32)
    monkeypatch.setitem(masks._BACKEND, "b", None)
    with pytest.warns(UserWarning, match="scipy.fft"):
        alt = masks.dynamo_mask(size=64, type="cylinder", r=15, height=30, gaussian=3).astype(np.float32)
    big = np.abs(ref) > 1e-6
    assert np.array_equal(ref[big], alt[big])
    assert np.abs(ref.astype(float) - alt).max() < 1e-12


def test_mask_editor_arguments_and_unported_types(tmp_path):
    ed = masks.MaskEditor(size=[64, 64, 64], mask=str(tmp_path / "e.em"))
    assert ed.size == "64  64  64" and ed.type == "sphere" and not any(ed.radio.values())
    a0 = ed.read_mask_settings()
    assert a0[0] == "size" and np.ravel(a0[1]).tolist() == [64, 64, 64] and a0[2:] == ["type", "sphere", "invert",
                                                                                         False]
    ed.set_type("cylinder")
    ed.select("gaussian", "3")
    ed.select("r", "15")
    ed.select("height", "30")
    a = ed.read_mask_settings()
    # radio order of dynamo_mask_GUI.m:188-190, then 'invert' (always passed)
    assert [x for x in a if isinstance(x, str)] == ["size", "type", "cylinder", "r", "height", "gaussian", "invert"]
    assert a[-1] is False
    ed.set_type("sphere")
    with pytest.raises(NotImplementedError):
        ed.save()
    with pytest.raises(ValueError):
        ed.set_type("cube")
    with pytest.raises(NotImplementedError):
        masks.dynamo_mask(size=64, type="cylinder", r=10, eulers=[0, 30, 0])


# ---------------------------------------------------------------------------------------------------- controller
def _single_project(root, monkeypatch, name="p1"):
    monkeypatch.chdir(root)
    make_data(str(root), N=4)
    c = DcpController()
    c.new(name, "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
          "particles_edit_mod.tbl")
    return c


def test_mask_editor_from_the_masks_window(tmp_path, monkeypatch):
    c = _single_project(tmp_path, monkeypatch)
    for kind, fname, box in (("align", "editorMask.em", "mask"), ("cmask", "editorCMask.em", "cmask"),
                             ("smask", "editorSmoothingMask.em", "cmask")):          # sic, wizard_masks.m:373
        ed, b, f = c.mask_editor(kind)
        assert (b, f) == (box, "p1/settings/" + fname) and ed.save_file == f
        assert ed.size == "64  64  64" and ed.volume == "raw_template_ctf.em"
    ed, _, f = c.mask_editor("align")
    ed.set_type("cylinder")
    ed.select("r", "15")
    ed.select("height", "30")
    ed.select("gaussian", "3")
    ed.save()
    assert not os.path.exists("p1/settings/editorMask.em") or V.load("p1")["file_mask"] != f   # not yet entered
    assert c.mask_enter(f).ok and V.load("p1")["file_mask"] == f
    assert formats.read_em(f).shape == (64, 64, 64)
    assert c.messages[-1][0].startswith('Parameter "') or "accepted" in c.messages[-1][0]


def test_vpr_modify_accepts_numpy_values(tmp_path, monkeypatch):
    _single_project(tmp_path, monkeypatch)
    V.put("p1", "d", "gpu_identifier_set", np.array([0, 1, 2, 3]), "destination", "standalone_gpu")
    assert np.ravel(V.load("p1")["gpu_identifier_set"]).tolist() == [0, 1, 2, 3]


def test_golden_standard_derive_open_and_abp_editor(tmp_path, monkeypatch):
    c = _single_project(tmp_path, monkeypatch)
    assert c.abp_parameters() is None
    assert c.messages[-1] == ("This option is valid only for projects with two references.", "warning")
    assert c.golden_standard_derive() == "p1_eo"
    assert c.messages[-3][0] == "Created even/odd project for golden standard/dynamic bandpass: p1_eo"
    assert c.name == "p1"                                   # the derived project is not opened
    assert c.enter_project_name("p1_eo") == "opened"
    assert float(c.runtime["nref_r1"]) == 2 and all(c.correct[k] for k in ("data", "table", "template", "mask"))
    assert c.template_size_cache == [64, 64, 64]
    rows = c.abp_parameters()
    assert [r[0] for r in rows] == list(ABP_PARAMETERS)
    assert [r[1] for r in rows] == ["adaptive bandpass", "adaptive bandpass threshold", "recompute rotated reference",
                                    "adaptive initial lowpass", "adaptive bandpass pushback",
                                    "symmetrized resolution", "adaptive bandpass even/odd"]
    v0 = V.load("p1_eo")
    assert (v0["adaptive_bandpass"], v0["adaptive_bandpass_threshold"], v0["adaptive_bandpass_pushback"]) == \
        (1.0, 0.5, 2.0)
    for n, s in (("adaptive_bandpass_threshold", "0.143"), ("adaptive_bandpass_initial", "25"),
                 ("adaptive_bandpass_pushback", "0")):
        assert c.abp_set(n, s).ok
    assert not c.abp_set("adaptive_bandpass_pushback", "1 2").ok
    v = V.load("p1_eo")
    assert (v["adaptive_bandpass_threshold"], v["adaptive_bandpass_initial"], v["adaptive_bandpass_pushback"]) == \
        (0.143, 25.0, 0.0)
    assert len(v) == 468 and list(v) == list(V.blank("p1_eo"))       # GUI-style save (getVpr)
    assert not any(k.endswith("_all") for k in v)
    # a missing reference in a seed folder is reported (setMultireferenceProject.m)
    os.remove("p1_eo/settings/template_initial_ref_002.em")
    c.enter_project_name("p1_eo")
    assert any(m.startswith("Current template cannot be used as seed for multireference project")
               for m, _ in c.messages)
    assert c.correct["template"] is False


# ---------------------------------------------------------------------------------------------------- user data
@pytest.mark.skipif(not os.path.isfile(os.path.join(T, "abp_align", "settings", "editorMask.em")),
                    reason="the user's bin8_filament data is not present")
def test_user_journey_matches_dynamo_and_user_files(tmp_path):
    """The full replay (dcp.new, mask editor, parameters, derive, ABP editor, edits, check, unfold, history):
    every step against the Dynamo 1.1.514 snapshots of journey_dynamo, the end against the user's files."""
    if masks.fft_backend() == "scipy":
        pytest.skip("no FFTW library")
    if not os.path.isdir(os.path.join(P, "benchmarks", "06_abp_align_eo", "journey_dynamo", "snaps")):
        pytest.skip("Dynamo journey snapshots not present")
    r = subprocess.run([sys.executable, "-I", JOURNEY, str(tmp_path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    summ = r.stdout.split("==== SUMMARY")[1].strip().splitlines()
    cmp_lines = [s for s in summ if "files compared" in s]
    assert len(cmp_lines) == 12 and all(", 0 differ [];" in s and s.endswith("only in pydynamo: []")
                                        for s in cmp_lines), "\n".join(summ)
    assert any("editorMask.em md5 %s" % MD5_EDITOR_MASK in s and s.endswith("equal True)") for s in summ)
    assert any(s.startswith("5b put vs s7: True vpr n=496/496 order_same=True") for s in summ)
    assert "6c card ite 1 vs USER: diffs []" in "\n".join(summ)
    assert "6c starting_values (6 files written by unfold) vs USER: differ []" in summ
