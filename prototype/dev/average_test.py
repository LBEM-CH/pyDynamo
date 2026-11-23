"""Average a Dynamo iteration on the GPUs and compare with Dynamo's own averaging outputs, bit for bit.
python dev/average_test.py <dynamo_work_dir> <project> <ite> <data_folder> <ngpus>
Uses the project's refined table of iteration <ite> and settings/mask.em, the same inputs as Dynamo's assemble, and
follows the project runner (DynamoProject.average_from_table):
- MATLAB's computational thread count of dynamo_execute_project (matlab_numerics.matlab_threads(); set
  $PYDYN_MATLAB_THREADS to the core count of the node Dynamo ran on, e.g. 64);
- the even/odd FSC of the halves symmetrized with the round's sym_r<round> (genericInput2Average.m:241-258);
- matlab_workers_average > 0: MATLAB's parfor subranges on single-threaded workers (average.pool_sums), else one
  running state through the GPUs (average.ordered_sums).
The particles (ascending tags) are split into <ngpus> consecutive runs, one per GPU.
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
    from pydyn.average import finalize, halves_for, new_pool_state, new_state, ordered_sums, pool_sums
    from pydyn.dynproject import ite2round, load_vpr
    from pydyn.matlab_numerics import matlab_threads, parfor_subranges
    work, proj, ite, data, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    avg = os.path.join(work, proj, "results", f"ite_{ite:04d}", "averages")
    vpr = load_vpr(os.path.join(work, proj, "settings", "virtual_project.mat"))
    W = int(float(vpr.get("matlab_workers_average", 0) or 0))
    sym = vpr["sym_r%d" % ite2round(vpr, ite)]
    T = matlab_threads()
    rows = np.real(emio.read_tbl(os.path.join(avg, f"refined_table_ref_001_ite_{ite:04d}.tbl")))
    rows = rows[rows[:, 2] == 1]
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    half = halves_for(rows)
    nh = np.bincount(half, minlength=2).astype(np.float64)
    mask = emio.read_em(os.path.join(work, proj, "settings", "mask.em"))
    m32 = mask.astype(np.float32)
    n = mask.shape[0]
    paths = [emio.particle_path(os.path.join(work, data), t) for t in rows[:, 0]]
    r32 = rows.astype(np.float32)
    g_of = np.zeros(len(rows), dtype=np.int64)                       # GPU of each row: consecutive runs
    for g, c in enumerate(np.array_split(np.arange(len(rows)), ngpu)):
        g_of[c] = g
    per_gpu = []

    def run(g, sel, fn):
        t = time.perf_counter()
        parts = emio.read_em_stack([paths[i] for i in sel])
        t_read = time.perf_counter() - t
        with cp.cuda.Device(g):
            out = fn(parts)
        per_gpu.append({"gpu": g, "particles": len(sel), "t_read_s": round(t_read, 2),
                        "sums_s": round(out["seconds"], 2)})
        return out

    if W <= 0:
        state = new_state(n)
        for g in range(ngpu):
            sel = np.flatnonzero(g_of == g)
            state = run(g, sel, lambda p, s=state, c=sel: ordered_sums(p, r32[c], half[c], nh, m32, s, threads=T))
    else:                                                            # as dynproject.average_from_table, pool branch
        order, pieces = [], []
        for k in (0, 1):
            idx = np.flatnonzero(half == k)
            a = 0
            for size in parfor_subranges(len(idx), W):
                sub = idx[a:a + size][::-1]
                a += size
                cut = np.flatnonzero(np.diff(g_of[sub])) + 1
                bounds = list(zip(np.r_[0, cut], np.r_[cut, len(sub)]))
                for j, (u, v) in enumerate(bounds):
                    pieces.append([len(order), len(order) + v - u, k, j == 0, j == len(bounds) - 1])
                    order.extend(sub[u:v].tolist())
        order = np.asarray(order, dtype=np.int64)
        state = new_pool_state(n)
        i = 0
        while i < len(pieces):                                       # consecutive pieces on one GPU: one call
            g = int(g_of[order[pieces[i][0]]])
            j = i
            while j < len(pieces) and int(g_of[order[pieces[j][0]]]) == g:
                j += 1
            p0, p1 = pieces[i][0], pieces[j - 1][1]
            sel = order[p0:p1]
            rel = [(q[0] - p0, q[1] - p0, q[2], q[3], q[4]) for q in pieces[i:j]]
            state = run(g, sel, lambda p, s=state, c=sel, r=rel: pool_sums(p, r32[c], half[c], nh, m32, s, r,
                                                                            threads=1))
            i = j
    t2 = time.perf_counter()
    out = finalize(state, nh, mask, n, fmin=3.0, sym=sym, threads=T)
    report = {"particles": int(len(rows)), "halves": out["n"], "gpus": ngpu, "matlab_threads": T,
              "matlab_workers_average": W, "sym": sym, "finalize_s": round(time.perf_counter() - t2, 2),
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
    if W > 0:
        report["note"] = ("with a MATLAB pool Dynamo's own reruns differ in the last eo_fsc points (a shell that "
                          "holds only rounding noise), so those points are not expected to match")
    if os.environ.get("AVG_DUMP"):                                   # full-precision outputs for comparisons
        np.savez(os.environ["AVG_DUMP"], **{k: out[k] for k in ("average", "average_unweighted", "eo_fsc")})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
