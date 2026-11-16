"""Stage-by-stage GPU time of the pydyn engine on one GPU's share of a case: python profile_engine.py <card> [gpu] [batch] [max_particles]."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cupy as cp
from pydyn.card import Card
from pydyn import emio
from pydyn.engine import Engine
card = Card(sys.argv[1]); gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 512; maxp = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
paths = [l.strip() for l in open(card.particle_list(gpu)) if len(l.strip()) > 4][:maxp]
rows = emio.read_btbl(card.starting_table(gpu))[:len(paths)]
parts = np.stack([emio.read_em(p).astype(np.float32) for p in paths])
eng = Engine(emio.read_em(card.file_template).astype(np.float32), emio.read_em(card.file_mask).astype(np.float32), card, batch=batch)
eng.prepare_particles(parts, rows)
eng.refine(rows[:2])  # warm-up (plans, kernels)
eng.profile, eng._events, eng.timings = True, [], {}
t0 = time.perf_counter(); eng.refine(rows); cp.cuda.Device().synchronize(); wall = time.perf_counter() - t0
prof = eng.profile_report(); tot = sum(prof.values()); pairs = sum(eng.orientation_counts)
print(json.dumps({"particles": len(paths), "pairs": pairs, "batch": batch, "refine_wall_s": round(wall, 3),
                  "gpu_stage_sum_s": round(tot, 3), "pairs_per_s": round(pairs / wall, 1),
                  "stages_pct": {k: round(100 * v / tot, 1) for k, v in prof.items()},
                  "host_angles_s": round(eng.timings.get("angles", 0), 3)}, indent=1))
