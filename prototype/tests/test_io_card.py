import numpy as np
import pytest

from pydyn import dynio, emio
from pydyn.card import Card, atof, parse_area_search, read_card


def test_em_roundtrip_keeps_xyz_order(tmp_path):
    v = np.random.RandomState(0).rand(4, 5, 6).astype(np.float32)        # [z, y, x]
    p = tmp_path / "v.em"
    emio.write_em(p, v)
    raw = open(p, "rb").read()
    assert raw[0] == 6 and raw[3] == 5                                     # PC, float
    assert list(np.frombuffer(raw[4:16], "<i4")) == [6, 5, 4]              # nx, ny, nz
    assert np.array_equal(emio.read_em(p), v)


def test_em_stack_threaded_equals_serial_read(tmp_path):
    rs = np.random.RandomState(3)
    paths = []
    for k in range(37):
        p = tmp_path / f"particle_{k}.em"
        emio.write_em(p, rs.standard_normal((6, 5, 4)).astype(np.float32))
        paths.append(p)
    serial = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
    for threads in (1, 8):
        got = emio.read_em_stack(paths, threads=threads)
        assert got.dtype == np.float32 and np.array_equal(got.view(np.int32), serial.view(np.int32))


def test_btbl_and_tbl_roundtrip(tmp_path):
    t = np.arange(3 * 35, dtype=np.float32).reshape(3, 35)
    emio.write_btbl(tmp_path / "t.btbl", t)
    assert np.array_equal(emio.read_btbl(tmp_path / "t.btbl"), t)
    emio.write_tbl(tmp_path / "t.tbl", t)
    assert np.allclose(emio.read_tbl(tmp_path / "t.tbl"), t)


def test_card_parsing_like_dynamo_engine(tmp_path):
    card = tmp_path / "c.card"
    card.write_text("name_project=p1;\n% comment=x;\ncone_range=30;\ncone_range=45; trailing\n"
                    "area_search=4  4  4;\ngpu_identifier_set=0 1 2 3;\n dim=99;\n"
                    "database_ite=2;\ndatabase_ref=1;\ndatabase_gpu_file_template_iteration=t.em;\n"
                    "database_gpu_file_mask=m.em;\ndatabase_gpu_root_file_particleList=list_gpu_;\n")
    f = read_card(card)
    assert f["cone_range"] == "45"                                         # last match wins
    assert "dim" not in f                                                  # key must start at column 0
    c = Card(card)
    assert c.gpu_ids == [0, 1, 2, 3] and c.area_search == [4, 4, 4]
    assert c.particle_list(3) == "list_gpu_03.txt"
    assert c.refined_table(1).endswith("refined_gpuProcRefTable_ite_0002_ref_001_proc_00000_gpu_01.btbl")


def _card_with_area_search(tmp_path, text):
    card = tmp_path / "a.card"
    card.write_text("name_project=p1;\ndatabase_gpu_file_template_iteration=t.em;\ndatabase_gpu_file_mask=m.em;\n"
                    "database_gpu_root_file_particleList=list_gpu_;\narea_search_modus=1;\n"
                    + ("" if text is None else "area_search=%s;\n" % text))
    return card


@pytest.mark.parametrize("text, want", [
    ("4  4  4", [4, 4, 4]),
    ("4  4  2", [4, 4, 2]),
    ("10  10   4", [10, 10, 4]),
    ("4", [4, 4, 4]),                          # one value: no space, so each field is the whole text
    ("4 2", [4, 2, 2]),
    ("4           4         2.5", [4, 0, 4]),  # num2str([4 4 2.5]): the first gap is wider than 3 spaces
    ("5    5  100", [5, 0, 5]),                # num2str([5 5 100])
    (" 4 4 4", [0, 4, 4]),                     # the engine does not trim the value
    ("4 4 ", [4, 4, 0]),
    ("1e1 2.5x 3", [10, 2.5, 3]),              # atof: the longest numeric prefix
    (None, [0, 0, 0]),                         # no area_search line: extract_field returns ""
])
def test_area_search_parsed_as_the_gpu_engine_parses_it(tmp_path, text, want):
    """settings_card::settings_card (dynamo_compute_iteration_spp.cu:1821-1852), not a whitespace split."""
    assert Card(_card_with_area_search(tmp_path, text)).area_search == want
    assert parse_area_search("" if text is None else text) == want


def test_area_search_of_cards_written_with_num2str():
    """The card text Dynamo writes (dynamo_card_write.m: num2str, dynio.num2str) and what its engine then reads."""
    assert dynio.num2str([4, 4, 2.5]) == "4           4         2.5"
    assert parse_area_search(dynio.num2str([4, 4, 2.5])) == [4, 0, 4]
    assert parse_area_search(dynio.num2str(4)) == [4, 4, 4]
    assert parse_area_search(dynio.num2str([4, 4, 2])) == [4, 4, 2]


def test_read_card_unstripped_and_c_atof(tmp_path):
    card = tmp_path / "c.card"
    card.write_text("area_search= 4 4 4 ;\nname_project=p1\n")
    assert read_card(card, strip=False) == {"area_search": " 4 4 4 ", "name_project": "p1"}
    assert read_card(card) == {"area_search": "4 4 4", "name_project": "p1"}
    assert atof("  -2.5e1abc") == -25.0 and atof("7.") == 7.0 and atof(".5") == 0.5 and atof("1e") == 1.0
    assert atof("") == 0.0 and atof("abc") == 0.0 and atof(".") == 0.0 and atof("0x") == 0.0
    assert atof("0x10") == 16.0 and atof("inf") == float("inf") and np.isnan(atof("NaN"))
