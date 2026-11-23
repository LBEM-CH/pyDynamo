"""Compare two Dynamo tables (.btbl or .tbl) particle by particle (rows matched by tag, column 1).

python scripts/compare_tables.py A B [--truth T]   ->  JSON summary
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from pydynamo.files import em_stack  # noqa: E402
from pydynamo.engine.angles import rotation_matrices  # noqa: E402


def load(path):
    return em_stack.read_btbl(path) if ".btbl" in os.path.basename(path) else em_stack.read_tbl(path).astype(np.float32)


def angular_distance(a, b):
    """Rotation angle (deg) between orientations given as Dynamo Euler triplets [N, 3]."""
    Ma = rotation_matrices(a).reshape(-1, 3, 3).astype(np.float64)
    Mb = rotation_matrices(b).reshape(-1, 3, 3).astype(np.float64)
    tr = np.einsum("nij,nij->n", Ma, Mb)          # trace(Ma^T Mb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def compare(A, B):
    tags = np.intersect1d(A[:, 0][A[:, 0] > 0], B[:, 0][B[:, 0] > 0])
    ia = {t: i for i, t in enumerate(A[:, 0])}
    ib = {t: i for i, t in enumerate(B[:, 0])}
    a = A[[ia[t] for t in tags]]
    b = B[[ib[t] for t in tags]]
    dang = angular_distance(a[:, 6:9], b[:, 6:9])
    dsh = np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=1)
    dcc = np.abs(a[:, 9] - b[:, 9])
    return {
        "particles": int(len(tags)),
        "same_orientation_lt_0.01deg": int((dang < 0.01).sum()),
        "orientation_within_1deg": int((dang < 1).sum()),
        "angle_diff_deg_median": float(np.median(dang)), "angle_diff_deg_max": float(dang.max()),
        "shift_diff_px_median": float(np.median(dsh)), "shift_diff_px_max": float(dsh.max()),
        "shift_within_0.5px": int((dsh < 0.5).sum()),
        "cc_absdiff_median": float(np.median(dcc)), "cc_absdiff_max": float(dcc.max()),
        "cc_mean": [float(a[:, 9].mean()), float(b[:, 9].mean())],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--truth", default=None)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    out = {"a_vs_b": compare(A, B)}
    if x.truth:
        T = load(x.truth)
        out["a_vs_truth"] = compare(A, T)
        out["b_vs_truth"] = compare(B, T)
    print(json.dumps(out, indent=1))
