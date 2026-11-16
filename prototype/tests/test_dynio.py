"""Dynamo 1.1.514 file formats (pydyn.dynio).

CPU-only. The first group needs no data. The second group round-trips every Dynamo-written .tbl/.fsc/.txt/.em/.card
of the user's run (test_data/bin8_filament, read-only) through read -> write and requires identical bytes; it is
skipped when the data is absent.
"""
import glob
import os

import numpy as np
import pytest

from pydyn import dynio

RUNROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "test_data", "bin8_filament")
RUN = os.path.join(RUNROOT, "bin8_align_1")
have_run = pytest.mark.skipif(not os.path.isdir(RUN), reason="test_data/bin8_filament not available")


# ------------------------------------------------------------------ no data needed

def test_dlm_fields_as_matlab():
    f = dynio.dlm_field
    assert f(9422.0) == "9422" and f(-63.99) == "-63.99" and f(50.0) == "50"
    assert f(0.00057602) == "0.00057602" and f(3.4332e-05) == "3.4332e-05" and f(7.0167e-15) == "7.0167e-15"
    assert f(102.4567891) == "102.46" and f(100000.0) == "1e+05"
    assert f(complex(0, 1.2074e-06)) == "0+1.2074e-06i" and f(complex(0, 1.7075e-06)) == "0+1.7075e-06i"
    assert f(complex(3.5, 0)) == "3.5"                     # real element of a complex matrix: plain
    assert f(float("nan")) == "NaN" and f(float("inf")) == "Inf" and f(-float("inf")) == "-Inf"


def test_tbl_text_and_reader_keep_complex(tmp_path):
    t = np.zeros((2, 42), dtype=complex)
    t[:, 0] = [1, 2]
    t[0, 27] = complex(0, 1.2074e-06)
    t[1, 9] = 0.052038
    p = str(tmp_path / "a.tbl")
    dynio.write_tbl(p, t)
    b = open(p, "rb").read()
    assert b.split(b"\n")[0].split(b" ")[27] == b"0+1.2074e-06i"
    assert b.endswith(b"\n") and not b.endswith(b" \n") and b"\r" not in b
    r = dynio.read_tbl(p)
    assert np.iscomplexobj(r) and r[0, 27] == complex(0, 1.2074e-06) and r[1, 9] == 0.052038
    dynio.write_tbl(p, r)
    assert open(p, "rb").read() == b


def test_txt_precision9_fsc_row_tags(tmp_path):
    dynio.write_txt(str(tmp_path / "n.txt"), 5)
    assert open(tmp_path / "n.txt", "rb").read() == b"5\n"
    dynio.write_txt(str(tmp_path / "r.txt"), np.array([[1.0, 2.0, 3.0]]))
    assert open(tmp_path / "r.txt", "rb").read() == b"1 2 3\n"
    dynio.write_fsc(str(tmp_path / "e.fsc"), np.array([0.99166, 0.025445]))
    assert open(tmp_path / "e.fsc", "rb").read() == b"0.99166 0.025445\n"
    dynio.write_tags(str(tmp_path / "t.tags"), [1, 2357], padding=6)
    assert open(tmp_path / "t.tags", "rb").read() == b"000001\n002357\n"
    dynio.write_tags(str(tmp_path / "t5.tags"), [7], padding=3)    # padding = max(padding, 5)
    assert open(tmp_path / "t5.tags", "rb").read() == b"00007\n"


def test_num2str_card_values():
    assert dynio.num2str(np.array([4, 4, 2], dtype=np.uint8)) == "4  4  2"
    assert dynio.num2str(np.array([0, 1, 2, 3])) == "0  1  2  3"
    assert dynio.num2str(np.array([0.5, 0.01])) == "0.5        0.01"
    assert dynio.num2str(0.2) == "0.2" and dynio.num2str(float("nan")) == "NaN" and dynio.num2str(64) == "64"
    assert dynio.num2str(np.array([], dtype="<U1")) == "" and dynio.num2str("c1") == "c1"


def test_card_order_and_roundtrip(tmp_path):
    card = {"database_ite": 2, "name_project": "p", "nref": 1, "database_folder_x": "./p/x", "database_ref": 1,
            "area_search": [4.0, 4.0, 2.0], "intertwin": "1/2/3", "time_start_computation": "26-Jun-2026 12:08:30"}
    p = str(tmp_path / "c.card")
    dynio.write_card(p, card)
    lines = open(p).read().splitlines()
    assert lines[:4] == ["name_project=p;", "database_ite=2;", "database_ref=1;", "database_folder_x=./p/x;"]
    assert "area_search=4  4  2;" in lines and "intertwin=1/2/3;" in lines
    c = dynio.read_card(p)
    assert c["intertwin"] == "1/2/3" and c["area_search"] == [4.0, 4.0, 2.0]
    assert c["time_start_computation"] == "26-Jun-2026 12:08:30"
    dynio.write_card(str(tmp_path / "d.card"), c)
    assert open(tmp_path / "d.card", "rb").read() == open(p, "rb").read()


def test_em_and_btbl_layout(tmp_path):
    v = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)       # [z, y, x]
    p = str(tmp_path / "v.em")
    dynio.write_em(p, v)
    b = open(p, "rb").read()
    assert b[:4] == bytes([6, 0, 0, 5]) and b[16:512] == bytes(496) and len(b) == 512 + 4 * 24
    assert tuple(np.frombuffer(b[4:16], "<u4")) == (4, 3, 2)          # nx, ny, nz
    assert np.frombuffer(b[512:516], "<f4")[0] == 0 and np.frombuffer(b[516:520], "<f4")[0] == 1   # x fastest
    r = dynio.read_em(p)
    assert r.dtype == np.float64 and np.array_equal(r, v)
    t = np.array([[1, 1, 0, 0.5], [2, 1, 0, -1.25]])
    q = str(tmp_path / "t.btbl")
    dynio.write_btbl(q, t)
    b = open(q, "rb").read()
    assert tuple(np.frombuffer(b[4:16], "<u4")) == (2, 4, 1)          # n_rows x n_cols x 1
    assert list(np.frombuffer(b[512:528], "<f4")) == [1, 2, 1, 1]      # column-major
    assert np.array_equal(dynio.read_btbl(q), t)
    x = np.array([[1.0, 0.1]])
    dynio.write_btbl(q, x)
    assert dynio.read_btbl(q)[0, 1] == float(np.float32(0.1))        # values pass through float32


def test_dpktbl_branch_above_99999():
    b = dynio.tbl_bytes(np.array([[100000.0, 1.0, 0.0, 0.5]]))
    assert b == b"100000 1 0 0.50000000 \n"


# ------------------------------------------------------------------ every Dynamo-written file of the user's run

def _run_files(pattern):
    return sorted(glob.glob(os.path.join(RUN, pattern), recursive=True))


@have_run
def test_roundtrip_every_tbl():
    files = _run_files("**/*.tbl") + [os.path.join(RUNROOT, "particles_edit_mod.tbl")]
    assert len(files) >= 14
    bad = [f for f in files if dynio.tbl_bytes(dynio.read_tbl(f)) != open(f, "rb").read()]
    assert not bad, bad
    ncplx = sum(np.iscomplexobj(dynio.read_tbl(f)) for f in files)
    assert ncplx >= 5                                                # refined ite 1-4 and starting ite 2-5


@have_run
def test_roundtrip_every_fsc_and_txt():
    fsc = _run_files("**/*.fsc")
    txt = _run_files("results/**/currently_surviving_references_*.txt") + [os.path.join(RUN, "next_iteration.txt")]
    assert len(fsc) >= 4 and len(txt) >= 5
    bad = [f for f in fsc if dynio.dlmwrite_bytes(dynio.read_fsc(f), 5) != open(f, "rb").read()]
    bad += [f for f in txt if dynio.dlmwrite_bytes(dynio.read_txt(f), 9) != open(f, "rb").read()]
    assert not bad, bad


@have_run
def test_roundtrip_every_card():
    files = _run_files("cards/*/*.card")
    assert len(files) >= 8
    bad = [f for f in files if dynio.card_bytes(dynio.read_card(f)) != open(f, "rb").read()]
    assert not bad, bad


@have_run
def test_roundtrip_every_em_and_sample_particles():
    files = _run_files("**/*.em") + [os.path.join(RUNROOT, "raw_template_ctf.em")]
    files += sorted(glob.glob(os.path.join(RUNROOT, "filamentsData_ctf", "particle_*.em")))[:50]
    assert len(files) >= 60
    bad = [f for f in files if dynio.em_bytes(dynio.read_em(f)) != open(f, "rb").read()]
    assert not bad, bad
