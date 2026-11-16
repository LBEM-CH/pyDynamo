"""Drop-in replacement for Dynamo's GPU engine: python run_engine.py <card> [--engine exact|fast] [--batch B] [--out-suffix S].

Run from the same working directory Dynamo's engine would run in (paths in the card are relative to it).
One worker process per id in the card's gpu_identifier_set (ids index the visible CUDA devices), each reading
its own particle list and starting table and writing refined_gpuProcRefTable_..._gpu_DD.btbl (plus --out-suffix).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

T_START = time.perf_counter()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker(args):
    card_path, gpu, batch, suffix, shared, max_res, dynv, mode, sms = args
    batch = batch if batch == "auto" else int(batch)
    t0 = time.perf_counter()
    import numpy as np
    import cupy as cp
    from pydyn.card import Card
    from pydyn import emio
    from pydyn.engine import Engine
    cp.cuda.Device(gpu).use()
    card = Card(card_path)
    card.dynamo_version = dynv
    t_import = time.perf_counter() - t0

    t1 = time.perf_counter()
    with open(card.particle_list(gpu)) as f:
        paths = [ln.strip() for ln in f if len(ln.strip()) > 4]
    rows = emio.read_btbl(card.starting_table(gpu))[: len(paths)]
    particles = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
    template = emio.read_em(card.file_template).astype(np.float32)
    mask = emio.read_em(card.file_mask).astype(np.float32)
    t_io = time.perf_counter() - t1

    eng = Engine(template, mask, card, batch=batch, exact=(mode == "exact"), exact_sms=sms)
    eng.shared_grid = shared
    eng.prepare_particles(particles, rows, max_resident=max_res)
    t2 = time.perf_counter()
    refined = eng.refine(rows)
    cp.cuda.Device().synchronize()
    t_refine = time.perf_counter() - t2

    out = card.refined_table(gpu) + suffix
    emio.write_btbl(out, refined)
    free, total = cp.cuda.runtime.memGetInfo()
    peak = cp.get_default_memory_pool().total_bytes()
    return {"gpu": gpu, "engine": eng.mode, "particles": len(paths), "pairs_per_level": eng.orientation_counts,
            "t_import_s": round(t_import, 3), "t_read_s": round(t_io, 3),
            "t_prepare_s": round(eng.timings.get("prepare_particles", 0), 3),
            "t_refine_s": round(t_refine, 3), "t_angles_s": round(eng.timings.get("angles", 0), 3),
            "t_align_s": round(eng.timings.get("align", 0), 3),
            "worker_wall_s": round(time.perf_counter() - t0, 3), "pool_bytes": int(peak),
            "balancer": eng.balance_log, "residency": eng.residency, "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("--engine", choices=("exact", "fast"), default=None,
                    help="exact: Dynamo 1.1.514's own kernels, cuFFT 8 and cuBLAS 8 (bit-identical to the official engine "
                         "on the same GPU model); fast: pyDynamo's batched engine. Default: $PYDYN_ENGINE, else "
                         "pydyn.engine.DEFAULT_ENGINE")
    ap.add_argument("--exact-sms", type=int, default=None,
                    help="exact engine: reproduce the cuBLAS 8 reduction order of a GPU with this many SMs "
                         "(e.g. 64 = RTX A5000, 84 = A40); default: the GPU in use ($PYDYN_EXACT_SMS)")
    ap.add_argument("--batch", default="auto", help="pairs per batch, or auto (memory balancer)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--max-resident", type=int, default=None, help="force particle chunks of this size (testing)")
    ap.add_argument("--dynamo-version", default="1.1.514", help="Dynamo build whose exact arithmetic to reproduce")
    ap.add_argument("--shared-grid", action="store_true",
                    help="one orientation set for all particles (seed 0,0,0) at level 0: reference-major batching; NOT Dynamo-identical")
    a = ap.parse_args()
    from pydyn.engine import engine_mode, preflight
    mode = engine_mode(a.engine)
    preflight(mode)                       # before the card is read or any worker starts (review M1)
    from pydyn.card import Card
    gpus = Card(a.card).gpu_ids
    if mode == "exact" and a.shared_grid:
        ap.error("--shared-grid is a fast-engine option (not Dynamo-identical)")
    jobs = [(a.card, g, a.batch, a.out_suffix, a.shared_grid, a.max_resident, a.dynamo_version, mode, a.exact_sms)
            for g in gpus]
    if len(jobs) == 1:
        results = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(worker, jobs)
    wall = time.perf_counter() - T_START
    pairs = sum(sum(r["pairs_per_level"]) for r in results)
    print(json.dumps({"engine": "pydyn-" + mode, "shared_grid": a.shared_grid, "gpus": gpus, "batch": a.batch, "wall_s": round(wall, 3),
                      "particle_orientations": pairs,
                      "particle_orientations_per_s": round(pairs / wall, 1), "workers": results}, indent=1))


if __name__ == "__main__":
    main()
