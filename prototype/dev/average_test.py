"""Average a Dynamo iteration on the GPUs and compare with Dynamo's own averaging outputs, bit for bit.
python dev/average_test.py <dynamo_work_dir> <project> <ite> <data_folder> <ngpus>
Uses the project's refined table of iteration <ite> and settings/mask.em, the same inputs as Dynamo's assemble. The
particles (ascending tags) are split into <ngpus> consecutive runs and one running state goes through the GPUs in
that order, as the project runner does (pydyn.average.ordered_sums).
"""
import json
import os
import sys
import time

T0 = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # prototype/


def main():
    import numpy as np
    import cupy as cp
    from pydyn import emio
    from pydyn.average import finalize, halves_for, new_state, ordered_sums
    work, proj, ite, data, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    avg = os.path.join(work, proj, "results", f"ite_{ite:04d}", "averages")
    rows = np.real(emio.read_tbl(os.path.join(avg, f"refined_table_ref_001_ite_{ite:04d}.tbl")))
    rows = rows[rows[:, 2] == 1]
    rows = rows[np.argsort(rows[:, 0], kind="stable")].astype(np.float32)
    half = halves_for(rows)
    nh = np.bincount(half, minlength=2).astype(np.float64)
    mask = emio.read_em(os.path.join(work, proj, "settings", "mask.em")).astype(np.float32)
    n = mask.shape[0]
    paths = [emio.particle_path(os.path.join(work, data), t) for t in rows[:, 0]]
    state = new_state(n)
    per_gpu = []
    for g, c in enumerate(np.array_split(np.arange(len(rows)), ngpu)):
        t = time.perf_counter()
        parts = emio.read_em_stack([paths[i] for i in c])
        t_read = time.perf_counter() - t
        with cp.cuda.Device(g):
            state = ordered_sums(parts, rows[c], half[c], nh, mask, state)
        per_gpu.append({"gpu": g, "particles": len(c), "t_read_s": round(t_read, 2), "sums_s": round(state["seconds"], 2)})
    t2 = time.perf_counter()
    out = finalize(state, nh, mask, n)
    report = {"particles": int(len(rows)), "halves": out["n"], "gpus": ngpu, "finalize_s": round(time.perf_counter() - t2, 2),
              "wall_s": round(time.perf_counter() - T0, 2), "per_gpu": per_gpu}
    for name in ("average", "average_unweighted", "fweight_average_raw", "fmask_average"):
        ref = emio.read_em(os.path.join(avg, f"{name}_ref_001_ite_{ite:04d}.em")).astype(np.float32)
        mine = out[name].astype(np.float32)
        report[name] = {"voxels_differ": int((mine.view(np.int32) != ref.view(np.int32)).sum()),
                        "max_abs_diff": float(np.abs(mine.astype(np.float64) - ref).max())}
    tok = open(os.path.join(avg, f"eo_fsc_ref_001_ite_{ite:04d}.fsc")).read().split()
    mine = ["%.5g" % x for x in out["eo_fsc"]]
    report["eo_fsc"] = {"tokens": len(tok), "tokens_differ": [i + 1 for i, (a, b) in enumerate(zip(tok, mine))
                                                              if float(a) != float(b)],
                        "max_abs_diff": float(np.abs(np.array(tok, float) - out["eo_fsc"]).max())}
    if os.environ.get("AVG_DUMP"):                                   # full-precision outputs for comparisons
        np.savez(os.environ["AVG_DUMP"], **{k: out[k] for k in ("average", "average_unweighted", "eo_fsc")})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
