"""Average a Dynamo iteration on the GPUs and compare with Dynamo's own averaging outputs.

python average_test.py <dynamo_work_dir> <project> <ite> <data_folder> <ngpus>
Uses the project's refined table of iteration <ite> and settings/mask.em, the same inputs as Dynamo's assemble.
"""
import json
import multiprocessing as mp
import os
import sys
import time

T0 = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker(args):
    rows, half, paths, mask, gpu = args
    import numpy as np
    import cupy as cp
    from pydyn import emio
    from pydyn.average import partial_sums
    cp.cuda.Device(gpu).use()
    t = time.perf_counter()
    parts = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
    t_read = time.perf_counter() - t
    log = []
    r = partial_sums(parts, rows, half, mask, batch="auto", balance_log=log)
    r.update({"gpu": gpu, "t_read_s": t_read, "balancer": log})
    return r


def main():
    import numpy as np
    from pydyn import emio
    from pydyn.average import finalize, halves_for
    work, proj, ite, data, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    avg = os.path.join(work, proj, "results", f"ite_{ite:04d}", "averages")
    rows = emio.read_tbl(os.path.join(avg, f"refined_table_ref_001_ite_{ite:04d}.tbl")).astype(np.float32)
    half = halves_for(rows)
    keep = half >= 0
    rows, half = rows[keep], half[keep]
    mask = emio.read_em(os.path.join(work, proj, "settings", "mask.em")).astype(np.float32)
    paths = [emio.particle_path(os.path.join(work, data), t) for t in rows[:, 0]]
    chunks = np.array_split(np.arange(len(rows)), ngpu)
    jobs = [(rows[c], half[c], [paths[i] for i in c], mask, g) for g, c in enumerate(chunks)]
    t1 = time.perf_counter()
    if ngpu == 1:
        parts = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(ngpu) as pool:
            parts = pool.map(worker, jobs)
    t_gpu = time.perf_counter() - t1
    t2 = time.perf_counter()
    out = finalize(parts, mask, mask.shape[0])
    t_fin = time.perf_counter() - t2
    wall = time.perf_counter() - T0

    report = {"particles": int(len(rows)), "halves": out["n"], "gpus": ngpu, "wall_s": round(wall, 2),
              "gpu_phase_s": round(t_gpu, 2), "finalize_s": round(t_fin, 2),
              "per_gpu": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items() if k in ("gpu", "seconds", "t_read_s", "balancer")} for p in parts]}
    for name in ("average", "average_unweighted", "fweight_average_raw", "fmask_average"):
        ref = emio.read_em(os.path.join(avg, f"{name}_ref_001_ite_{ite:04d}.em")).astype(np.float64)
        mine = out[name]
        d = np.abs(mine - ref)
        report[name] = {"max_abs_diff": float(d.max()), "ref_absmax": float(np.abs(ref).max()),
                        "corr": float(np.corrcoef(mine.ravel(), ref.ravel())[0, 1]),
                        "mismatch_voxels": int((d > 1e-4 * max(1.0, float(np.abs(ref).max()))).sum())}
    fsc_ref = np.loadtxt(os.path.join(avg, f"eo_fsc_ref_001_ite_{ite:04d}.fsc")).ravel()
    np.save(os.environ.get("AVG_DUMP", "/tmp/avg_dump.npy"), {k: out[k] for k in ("average", "eo_fsc")}, allow_pickle=True)
    report["eo_fsc_full"] = {"mine": [round(x, 4) for x in out["eo_fsc"]], "dynamo": [round(x, 4) for x in fsc_ref]}
    ref = emio.read_em(os.path.join(avg, f"average_ref_001_ite_{ite:04d}.em")).astype(np.float64)
    d = np.abs(out["average"] - ref)
    idx = np.argwhere(d > 1e-4 * np.abs(ref).max())
    report["average_mismatch_voxels_zyx"] = idx.tolist()[:6]
    report["eo_fsc"] = {"max_abs_diff": float(np.abs(out["eo_fsc"] - fsc_ref).max()), "len": [len(out["eo_fsc"]), len(fsc_ref)],
                        "mine_first5": [round(x, 4) for x in out["eo_fsc"][:5]], "dynamo_first5": [round(x, 4) for x in fsc_ref[:5]]}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
