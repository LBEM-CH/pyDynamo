import os

import numpy as np
import pytest

from pydyn import emio
from pydyn.project import Project, default_mask, shape_mask


def test_default_mask_is_dynamos():
    m = default_mask(64)
    assert int(m.sum()) == 124471                       # voxel count of dcp 'masks default' at l = 64
    # Dynamo's double-precision test loses 16 of the 198 lattice points lying exactly on the sphere |r|^2 = 31^2
    # (its rounding depends on the x+y+z summation order, so the mask is NOT exactly axis-symmetric).
    g = np.arange(64) - 32
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    r2 = x ** 2 + y ** 2 + z ** 2
    ideal = (r2 <= 961).astype(np.float32)
    diff = np.argwhere(ideal != m)
    assert len(diff) == 16 and all(r2[tuple(d)] == 961 for d in diff)


def test_shape_masks():
    l = 32
    assert shape_mask(l, "ones").sum() == l ** 3
    s = shape_mask(l, "sphere", radius=10)
    assert s[l // 2, l // 2, l // 2] == 1 and s[0, 0, 0] == 0
    e = shape_mask(l, "ellipsoid", semiaxes=[12, 8, 4])
    assert e[l // 2, l // 2, l // 2 + 11] == 1 and e[l // 2 + 6, l // 2, l // 2] == 0      # x long, z short
    c = shape_mask(l, "cylinder", radius=6, height=10)
    assert c[l // 2 + 4, l // 2, l // 2] == 1 and c[l // 2 + 7, l // 2, l // 2] == 0
    soft = shape_mask(l, "sphere", radius=8, edge=2)
    assert 0 < soft[l // 2, l // 2, l // 2 + 9] < 1


def make_inputs(tmp, n=16, particles=4, mask_size=None):
    data = tmp / "data"
    data.mkdir()
    rng = np.random.RandomState(1)
    for t in range(1, particles + 1):
        emio.write_em(data / f"particle_{t:05d}.em", rng.rand(n, n, n).astype(np.float32))
    table = np.zeros((particles, 35), dtype=np.float32)
    table[:, 0] = np.arange(1, particles + 1)
    table[:, 1] = table[:, 2] = 1
    table[:, 12], table[:, 13], table[:, 14] = 1, -60, 60
    emio.write_tbl(tmp / "start.tbl", table)
    emio.write_em(tmp / "ref.em", rng.rand(n, n, n).astype(np.float32))
    ms = mask_size or n
    emio.write_em(tmp / "mask.em", default_mask(ms))
    return data


def test_project_check_unfold_and_job_exe(tmp_path):
    data = make_inputs(tmp_path)
    pr = Project(tmp_path / "proj", data=str(data), table=str(tmp_path / "start.tbl"),
                 template=str(tmp_path / "ref.em"), mask=str(tmp_path / "mask.em"))
    errors, facts = pr.check()
    assert errors == [] and facts == {"particles": 4, "box": 16, "wedge_types": [1]}
    exe = pr.unfold()
    assert os.access(exe, os.X_OK)
    text = open(exe).read()
    assert "--project" in text and "#SBATCH --gres=gpu:4" in text and "unset PYTHONPATH" in text
    again = Project.load(tmp_path / "proj")
    assert again.p["table"] == str(tmp_path / "start.tbl")
    assert "--cone-range" in again.run_args()


def test_project_check_reports_problems(tmp_path):
    data = make_inputs(tmp_path, mask_size=20)
    os.remove(data / "particle_00003.em")
    pr = Project(tmp_path / "proj", data=str(data), table=str(tmp_path / "start.tbl"),
                 template=str(tmp_path / "ref.em"), mask=str(tmp_path / "mask.em"), sym="x7", low=0.5)
    errors, _ = pr.check()
    text = " | ".join(errors)
    assert "have no file" in text and "low" in text and "sym" in text
    with pytest.raises(ValueError):
        pr.unfold()
