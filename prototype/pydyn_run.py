"""pydyn_run: run a full subtomogram-averaging job (align + average + next reference) on all GPUs, no MATLAB.

python pydyn_run.py --data <folder> --table <initial.tbl> --template <ref.em> --mask <mask.em> --out <project_dir>
                    [--ite 1] [--gpus 0,1,2,3] [--cone-range 30 --cone-sampling 10 --inplane-range 30
                    --inplane-sampling 10 --refine 1 --refine-factor 2 --high 2 --low 20 --area-search 4 4 4
                    --area-search-modus 0 --sym c1] [--fmask <fmask.em>]

The math follows Dynamo's GPU engine and Dynamo's pipeline averaging, verified against Dynamo outputs.
Results are written in Dynamo's layout:
  <out>/results/ite_XXXX/averages/{refined_table,average,average_unweighted,fweight_average_raw,fmask_average,eo_fsc}
  <out>/results/ite_XXXX/starting_values/{starting_reference,starting_reference_transformed,...}
One persistent worker process per GPU loads its particles once and keeps them resident across iterations. Each
iteration it aligns, then computes its partial averaging sums on the same GPU. The master merges them.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

T_START = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Params:
    """The subset of Dynamo card fields the engine uses."""
    def __init__(self, a, n):
        self.dim = self.sidelength = n
        self.shrink = 1.0
        self.high, self.low = a.high, a.low
        self.use_CC, self.localnc = 1, 1
        self.area_search, self.area_search_modus = list(a.area_search), a.area_search_modus
        self.cone_range, self.cone_sampling, self.cone_flip = a.cone_range, a.cone_sampling, 0
        self.inplane_range, self.inplane_sampling, self.inplane_flip = a.inplane_range, a.inplane_sampling, 0
        self.refine, self.refine_factor = a.refine, a.refine_factor
        self.dynamo_version = a.dynamo_version
        self.engine, self.exact_sms = a.engine, a.exact_sms


def worker_main(gpu, conn):
    import numpy as np
    import cupy as cp
    from pydyn import emio
    from pydyn.engine import Engine
    from pydyn.average import partial_sums
    cp.cuda.Device(gpu).use()
    msg = conn.recv()
    paths, rows, half, mask, params = msg["paths"], msg["rows"], msg["half"], msg["mask"], msg["params"]
    mask_raw = msg["mask_raw"]                              # averaging normalises with the raw project mask
    t = time.perf_counter()
    particles = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
    t_read = time.perf_counter() - t
    eng = None
    conn.send({"ready": True, "t_read_s": t_read})
    while True:
        msg = conn.recv()
        if msg is None:
            break
        timing = {}
        t = time.perf_counter()
        if eng is None:
            eng = Engine(msg["template"], mask, params, batch="auto", exact=params.engine == "exact",
                         exact_sms=params.exact_sms)
            eng.prepare_particles(particles, rows)
        else:
            eng.set_template(msg["template"])
        timing["prepare_s"] = time.perf_counter() - t
        t = time.perf_counter()
        eng.timings = {}
        rows = eng.refine(rows)
        cp.cuda.Device().synchronize()
        timing["align_s"] = time.perf_counter() - t
        t = time.perf_counter()
        sel = half >= 0
        ps = partial_sums(particles[sel], rows[sel], half[sel], mask_raw, batch="auto")
        timing["average_sums_s"] = time.perf_counter() - t
        conn.send({"rows": rows, "partial": ps, "timing": timing, "pairs": sum(eng.orientation_counts),
                   "balancer": eng.balance_log[:1]})
    conn.close()


def write_dynamo_layout(out, ite, refined, avg):
    import numpy as np
    from pydyn import emio
    d = os.path.join(out, "results", f"ite_{ite:04d}", "averages")
    os.makedirs(d, exist_ok=True)
    emio.write_tbl(os.path.join(d, f"refined_table_ref_001_ite_{ite:04d}.tbl"), refined)
    for name in ("average", "average_unweighted", "fweight_average_raw", "fmask_average"):
        emio.write_em(os.path.join(d, f"{name}_ref_001_ite_{ite:04d}.em"), avg[name])
    np.savetxt(os.path.join(d, f"eo_fsc_ref_001_ite_{ite:04d}.fsc"), avg["eo_fsc"], fmt="%.6f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--fmask", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ite", type=int, default=1)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--cone-range", type=float, default=30)
    ap.add_argument("--cone-sampling", type=float, default=10)
    ap.add_argument("--inplane-range", type=float, default=30)
    ap.add_argument("--inplane-sampling", type=float, default=10)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--refine-factor", type=float, default=2)
    ap.add_argument("--high", type=float, default=2)
    ap.add_argument("--low", type=float, default=20)
    ap.add_argument("--area-search", type=float, nargs=3, default=[4, 4, 4])
    ap.add_argument("--area-search-modus", type=int, default=0)
    ap.add_argument("--sym", default="c1")
    ap.add_argument("--dynamo-version", default="1.1.514", help="Dynamo build whose exact arithmetic to reproduce")
    ap.add_argument("--engine", choices=("exact", "fast"), default=None,
                    help="exact: Dynamo 1.1.514's own kernels, cuFFT 8 and cuBLAS 8; fast: pyDynamo's batched engine. "
                         "Default: $PYDYN_ENGINE, else pydyn.engine.DEFAULT_ENGINE")
    ap.add_argument("--exact-sms", type=int, default=None,
                    help="exact engine: reproduce the cuBLAS 8 reduction order of a GPU with this many SMs")
    ap.add_argument("--symmetrize-with-fmask", type=int, default=1, choices=(0, 1),
                    help="vpr.symmetrize_with_fmask for the template symmetrization (Dynamo's default 1)")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="write the timing report (JSON) here ($PYDYN_REPORT); never inside --out")
    argv = sys.argv[1:]
    if "--project" in argv:                       # job.exe: settings come from <folder>/project.json
        i = argv.index("--project")
        from pydyn.project import Project
        pr = Project.load(argv[i + 1])
        argv = pr.run_args() + argv[:i] + argv[i + 2:]   # later flags (e.g. --gpus) override the project's
    a = ap.parse_args(argv)
    from pydyn.engine import engine_mode, preflight
    a.engine = engine_mode(a.engine)
    preflight(a.engine)                                   # before any data or worker work (review M1)
    from pydyn.dynproject import check_report_path
    report_path = check_report_path(a.report or os.environ.get("PYDYN_REPORT"), a.out)

    import numpy as np
    from pydyn import emio
    from pydyn.average import finalize, halves_for
    from pydyn.reference import prepare_template

    gpus = [int(g) for g in a.gpus.split(",")]
    rows = emio.read_tbl(a.table).astype(np.float32)
    from pydyn.reference import binary_mask
    mask_raw = emio.read_em(a.mask).astype(np.float32)      # Dynamo's file_mask, as given
    mask = binary_mask(mask_raw)                            # alignment + template normalisation (mask_transformed)
    n = mask.shape[0]
    fmask = emio.read_em(a.fmask).astype(np.float64) if a.fmask else np.ones((n, n, n))
    reference = emio.read_em(a.template).astype(np.float64)
    params = Params(a, n)
    half = halves_for(rows)
    paths = [emio.particle_path(a.data, t) for t in rows[:, 0]]
    chunks = np.array_split(np.arange(len(rows)), len(gpus))

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for g, c in zip(gpus, chunks):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker_main, args=(g, child), daemon=True)
        p.start()
        parent.send({"paths": [paths[i] for i in c], "rows": rows[c], "half": half[c], "mask": mask,
                     "mask_raw": mask_raw, "params": params})
        conns.append(parent)
        procs.append(p)
    ready = [c.recv() for c in conns]
    t_ready = time.perf_counter() - T_START
    report = {"particles": int(len(rows)), "gpus": gpus, "startup_and_read_s": round(t_ready, 2),
              "read_s_per_gpu": [round(r["t_read_s"], 2) for r in ready], "iterations": []}
    for ite in range(1, a.ite + 1):
        t0 = time.perf_counter()
        template = prepare_template(reference, mask, a.sym, a.high, a.low, fmask,
                                    symmetrize_with_fmask=bool(a.symmetrize_with_fmask))
        sv = os.path.join(a.out, "results", f"ite_{ite:04d}", "starting_values")
        os.makedirs(sv, exist_ok=True)
        emio.write_em(os.path.join(sv, f"starting_reference_ref_001_ite_{ite:04d}.em"), reference)
        emio.write_em(os.path.join(sv, f"starting_reference_transformed_ref_001_ite_{ite:04d}.em"), template)
        emio.write_tbl(os.path.join(sv, f"starting_table_ref_001_ite_{ite:04d}.tbl"), rows)
        t_prep = time.perf_counter() - t0
        for c in conns:
            c.send({"template": template})
        res = [c.recv() for c in conns]
        t_gpu = time.perf_counter() - t0 - t_prep
        t1 = time.perf_counter()
        rows = np.concatenate([r["rows"] for r in res])
        avg = finalize([r["partial"] for r in res], mask_raw, n, sym=a.sym)
        write_dynamo_layout(a.out, ite, rows, avg)
        t_fin = time.perf_counter() - t1
        reference, fmask = avg["average"], avg["fmask_average"]
        it = {"ite": ite, "wall_s": round(time.perf_counter() - t0, 2), "reference_prep_s": round(t_prep, 2),
              "gpu_phase_s": round(t_gpu, 2), "finalize_write_s": round(t_fin, 2),
              "per_gpu": [{k: round(v, 2) for k, v in r["timing"].items()} for r in res],
              "pairs": int(sum(r["pairs"] for r in res)), "balancer": res[0]["balancer"]}
        report["iterations"].append(it)
        print(json.dumps(it), flush=True)
    for c in conns:
        c.send(None)
    for p in procs:
        p.join()
    report["total_wall_s"] = round(time.perf_counter() - T_START, 2)
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps({"total_wall_s": report["total_wall_s"], "startup_and_read_s": report["startup_and_read_s"]}))


if __name__ == "__main__":
    main()
