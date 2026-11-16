# pyDynamo plan (updated 2026-11-11)

## Where we are

| Done | Evidence |
|---|---|
| Dynamo analysed (`dcp`, execution, GPU engine, alignment math, averaging) | `docs/analysis/01–05` |
| Python GPU alignment engine. It reads the same card and files as `dynamo_compute_iteration_spp` and is about 3.5× faster on 4 × A40, with per-particle parity (≤ 0.022°) | `benchmarks/02_engine_tutorial_ribosome/REPORT.md` |
| GPU averaging, symmetry (c/d), next-reference preparation; parity with Dynamo's files | same |
| Full iteration in Python, about 20× faster than Dynamo (local, 1,024 particles) and 4.5× (global 360°/15°, 256 particles) | same |
| Memory balancer, `job.exe`, CLI, a first Tk GUI, CI (GitHub Actions, CPU) | `prototype/`, `.github/` |

Shared-grid mode was rejected because it changes the result. Binning is out of scope because it is not used.

---

## Step 1: scale test with 7,000 tutorial particles (4 × A40)

**Data:** Dynamo's own tutorial generator (`dynamo_tutorial`), with its default thermosome template, symmetry c8 as the tutorial sets it, a ±60° missing wedge and noise 0.1.

| Set | Particles | Box | How generated |
|---|---|---|---|
| `tut7000_64` | 7,000 | 64 | `dynamo_tutorial(..., 'M', 7000)` |
| `tut7000_128` | 7,000 | 128 | Thermosome template upsampled to 80³, so the generator's box (1.6 × template) is 128. If generating 7,000 is too slow (estimated 1–2 h), generate 1,000 unique particles and replicate them to 7,000 for timing only, and say so |

**Runs**, each a full iteration of the project as `dcp` would create it, all sandboxed in job 11313, one at a time:

| Run | Search | Dynamo (4 GPUs) | pydyn (4 GPUs) |
|---|---|---|---|
| A | local (cone 30/10, in-plane 30/10, refine 1×2), box 64 | full project, 1 iteration (about 10 min estimated) | 3 iterations |
| B | global 360°/15°, box 64 | full project, 1 iteration (about 80 min estimated) | 1 iteration |
| C | local, box 128 | full project, 1 iteration | 3 iterations, which needs Step 2 |
| D | global 360°/15°, box 128 | a 500-particle subset, extrapolated (the full run would take about 10 h) | full 7,000 |

**Checks for every run:**
- Parity with Dynamo per particle: orientation, shift and CC; the averages and FSC.
- Accuracy against the tutorial's true table, **modulo c8 symmetry**: the thermosome is symmetric, so an orientation error is the smallest over the 8 symmetry-equivalent orientations.
- Timing per stage (alignment, averaging, I/O), GPU memory and the balancer's decisions.

**Deliverable: a human-readable report.** `benchmarks/02_engine_tutorial_ribosome/REPORT.md` will hold:
- the purpose, the data, the exact commands and the sandbox used;
- one table per run: Dynamo against pydyn, per stage and per iteration, with the speed-up;
- parity and accuracy tables;
- memory use (per GPU, the balancer's chosen batch, any out-of-memory back-offs);
- plain-language conclusions and open issues.

A top-level `benchmarks/README.md` will index every report (phase 1, ribosome, scale7000), so results can be found later.

## Step 2: box 128 with the memory balancer (how Dynamo copes, and what pydyn will do)

**How Dynamo copes:** it never holds the data set in memory. The GPU engine reads, aligns and discards **one particle at a time** (324 MB of GPU memory per run). The MATLAB averaging also streams particles from disk one by one. That is memory-safe but slow: 1 particle per GPU at a time, with disk reads inside the loop.

**pydyn today** keeps every particle's two spectra resident on the GPU:
- box 64: about 2.2 MB per particle, so 7,000 particles need about 15 GB, which fits;
- box 128: about 17 MB per particle, so 7,000 particles need about 120 GB, which does not fit in 4 × 46 GB.

**Change:** keep the particles resident where they fit, and stream the rest.
- Raw particles stay in host RAM (7,000 × 8 MB = 56 GB at box 128; the node has 251 GB).
- The balancer splits each GPU's memory between (a) as many resident particle spectra as fit and (b) the working batches of orientations.
- Particles beyond that are processed in chunks: upload the chunk, compute its spectra on the GPU, run all its orientations, drop it. The extra cost is a few FFTs per particle per iteration, small next to thousands of orientations.
- Averaging streams the same way, as Dynamo does, but in batches.
- **Test:** box 128 with 7,000 particles completes with no out-of-memory error, and the report shows the split the balancer chose.

## Step 3: the Python replacement for your Dynamo workflow (only if Steps 1–2 are good)

Your current journey, and its pyDynamo equivalent:

| Today (Dynamo, MATLAB Runtime) | pyDynamo |
|---|---|
| type `dynamo` in a Linux terminal → console | **no special console**: activate the pyDynamo environment and `from pydyn import dread, dwrite, dtrandomize_azimuth, daverage, dcp` in any Python script, IPython or Jupyter (decided 2026-11-11) |
| `T = dread('particles_b8_edit.tbl');` | `T = dread('particles_b8_edit.tbl')` (`.tbl`/`.em`/`.mrc`) |
| `T2 = dtrandomize_azimuth(T);` | `T2 = dtrandomize_azimuth(T)`: column 9 (narot) = 360·rand, as in `dynamo_table_randomize_azimuth.m` |
| `dwrite(T2,'particles_b8_edit_mod.tbl');` | `dwrite(T2, 'particles_b8_edit_mod.tbl')` |
| `oa = daverage('filamentsData_b8_ctf','t','particles_b8_edit_mod.tbl','mw',50);` | `oa = daverage('filamentsData_b8_ctf', t='particles_b8_edit_mod.tbl')`: GPU averaging. `mw` is accepted but GPUs replace MATLAB workers. Ported from `daverage`'s default path and checked against Dynamo's output on the same table |
| `dwrite(oa.average,'raw_template_ctf.em');` | `dwrite(oa.average, 'raw_template_ctf.em')` |
| `dcp.new('bin8_align_1','d','filamentsData_b8_ctf','template','raw_template_ctf.em','masks','default','t','particles_b8_edit_mod.tbl');` | the same call in the console, **or** `dcp` in a terminal to open the GUI. Same default parameters as Dynamo's `dcp.new`, same default masks (already identical) |
| set the parameters, then fold/unfold → `bin8_align_1.exe` | the GUI's parameter and computing windows, then **Unfold** → `bin8_align_1.exe` in the current folder, as Dynamo names it |
| `./bin8_align_1.exe` outside the MCR | `./bin8_align_1.exe`: pure Python and GPU, no MATLAB |

**GUI work** (Tk, launched by typing `dcp`):
- the wizard steps as in Dynamo: project, data, table, template, masks, numerical parameters **with rounds** (several rounds, as in `dcp`), computing environment, check, unfold;
- each window checks its input immediately, as `dcp` does;
- the masks window previews the mask on the template (already built);
- packaging: `pip install -e .` puts `dcp`, `pydynamo` and `pydyn` on your PATH without long paths.

**Tests:**
- CPU tests in CI for every console function and the project/rounds logic.
- A parity test: your `dcp.new` call in Dynamo and in pyDynamo gives the same parameters and masks.
- An end-to-end run on your real filament data (bin 8, CTF-corrected) compared with Dynamo. **Needs from you:** the path to that data folder and table, and you trying the GUI on your desktop session.

## Later (not planned in detail yet)

- Multireference alignment and classification.
- Adaptive filtering (`adaptive_bandpass`, even/odd).
- "Derive a project", e.g. even/odd (gold-standard) half sets.
- Other wedge types; shift-limit modes 1 and 2 compared with Dynamo; symmetry used to shrink the search.

## Rules kept throughout

- All tests run in the sandbox: node block job 11313, `sbx_run.sh`, one run at a time.
- No watchdog polling: report when a job finishes.
- Nothing is committed or pushed by the assistant; you push.
- Every claim of parity is measured against Dynamo's own output.
