"""Drop-in replacement for Dynamo's GPU engine on one card: python scripts/align_card.py <card> [--sms N] [--batch B] [--out-suffix S].

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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def worker(args):
    card_path, gpu, batch, suffix, sms = args
    t0 = time.perf_counter()
    import numpy as np
    import cupy as cp
    from pydynamo.engine.card import Card
    from pydynamo.files import em_stack
    from pydynamo.engine import Engine
    cp.cuda.Device(gpu).use()
    card = Card(card_path)
    t_import = time.perf_counter() - t0

    t1 = time.perf_counter()
    with open(card.particle_list(gpu)) as f:
        paths = [ln.strip() for ln in f if len(ln.strip()) > 4]
    rows = em_stack.read_btbl(card.starting_table(gpu))[: len(paths)]
    particles = em_stack.read_em_stack(paths)
    template = em_stack.read_em(card.file_template).astype(np.float32)
    mask = em_stack.read_em(card.file_mask).astype(np.float32)
    t_io = time.perf_counter() - t1

    eng = Engine(template, mask, card, batch=batch, sms=sms)
    eng.prepare_particles(particles, rows)
    t2 = time.perf_counter()
    refined = eng.refine(rows)
    cp.cuda.Device().synchronize()
    t_refine = time.perf_counter() - t2

    out = card.refined_table(gpu) + suffix
    em_stack.write_btbl(out, refined)
    peak = cp.get_default_memory_pool().total_bytes()
    return {"gpu": gpu, "particles": len(paths), "pairs_per_level": eng.orientation_counts,
            "t_import_s": round(t_import, 3), "t_read_s": round(t_io, 3), "t_refine_s": round(t_refine, 3),
            "t_align_s": round(eng.timings.get("align", 0), 3), "host": eng.timings.get("host"),
            "worker_wall_s": round(time.perf_counter() - t0, 3), "pool_bytes": int(peak), "out": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("--sms", type=int, default=None,
                    help="reproduce the cuBLAS 8 reduction order of a GPU with this many streaming multiprocessors "
                         "(e.g. 64 = RTX A5000, 84 = A40); default: $PYDYNAMO_SMS, else the GPU in use")
    ap.add_argument("--batch", default="auto",
                    help="orientations per step, or auto (128, fewer when the free GPU memory holds fewer); "
                         "1 = Dynamo's own loop; the bits do not depend on it")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    from pydynamo.engine import preflight
    preflight()                           # before the card is read or any worker starts (review M1)
    from pydynamo.engine.card import Card
    gpus = Card(a.card).gpu_ids
    batch = a.batch if a.batch == "auto" else int(a.batch)
    jobs = [(a.card, g, batch, a.out_suffix, a.sms) for g in gpus]
    if len(jobs) == 1:
        results = [worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(worker, jobs)
    wall = time.perf_counter() - T_START
    pairs = sum(sum(r["pairs_per_level"]) for r in results)
    print(json.dumps({"gpus": gpus, "batch": a.batch, "wall_s": round(wall, 3), "particle_orientations": pairs,
                      "particle_orientations_per_s": round(pairs / wall, 1), "workers": results}, indent=1))


if __name__ == "__main__":
    main()
