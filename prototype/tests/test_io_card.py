import numpy as np

from pydyn import emio
from pydyn.card import Card, read_card


def test_em_roundtrip_keeps_xyz_order(tmp_path):
    v = np.random.RandomState(0).rand(4, 5, 6).astype(np.float32)        # [z, y, x]
    p = tmp_path / "v.em"
    emio.write_em(p, v)
    raw = open(p, "rb").read()
    assert raw[0] == 6 and raw[3] == 5                                     # PC, float
    assert list(np.frombuffer(raw[4:16], "<i4")) == [6, 5, 4]              # nx, ny, nz
    assert np.array_equal(emio.read_em(p), v)


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
