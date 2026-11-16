# pydyn prototype: GPU subtomogram averaging with Dynamo's math, no MATLAB

**Status:** a feasibility prototype. It matches Dynamo 1.1.557 on the tested paths (see `../benchmarks/02_engine_tutorial_ribosome/REPORT.md`).

## Stage 3 usage

This is the Dynamo 1.1.514 workflow, done in Python alone, with no MATLAB or MCR at run time. The names, files and
on-disk project layout are Dynamo's. The specifications are `docs/analysis/08a_project_creation_1.1.514.md` (dcp
projects) and `docs/analysis/08b_console_functions_1.1.514.md` (console functions).

**Install once** into the project environment. This puts `dcp` and `pydynamo` into `.envs/pydynamo/bin`:

```bash
PIP_CONFIG_FILE=/dev/null .envs/pydynamo/bin/python -I -m pip install -e . --no-deps --no-build-isolation
export PATH=$PWD/.envs/pydynamo/bin:$PATH        # optional
```

**Console steps.** Run these in Python, from the folder that holds the data, just as you would in the Dynamo console:

```python
from pydyn import dread, dwrite, dtrandomize_azimuth, daverage, dcp, dvput, dvcheck, dvunfold, dvload

T  = dread('particles_b8_edit.tbl')
T2 = dtrandomize_azimuth(T)                   # MATLAB's start-up random stream: same numbers as a fresh Dynamo session
dwrite(T2, 'particles_b8_edit_mod.tbl')       # %.5g text, byte-identical to Dynamo
oa = daverage('filamentsData_b8_ctf', 't', 'particles_b8_edit_mod.tbl', 'mw', 50)   # GPU (CPU fallback), mex-exact
dwrite(oa.average, 'raw_template_ctf.em')     # byte-identical to Dynamo's raw_template_ctf.em on the test data

dcp.new('bin8_align_1', d='filamentsData_b8_ctf', template='raw_template_ctf.em', masks='default',
        t='particles_b8_edit_mod.tbl', gui=0)  # MATLAB style also works: dcp.new('bin8_align_1', 'd', '...', ...)
dvput('bin8_align_1', 'd', 'destination', 'standalone_gpu', 'gpu_identifier_set', [0, 1, 2, 3],
      'ite', [2, 2], 'cone_range_r1', 20, 'cone_sampling_r1', 3)   # 'ite' [2 2] -> ite_r1 = ite_r2 = 2, as in Dynamo
r = dvcheck('bin8_align_1')                   # r.report, r.fields_warning, r.fields_error, r.warning_state
dvunfold('bin8_align_1')                      # writes bin8_align_1.exe next to the project
```

* Python lists or arrays are numeric vectors (MATLAB `[...]`). `pydyn.vpr.MatCell([...])` is a MATLAB cell.
* `dvget` does not exist in Dynamo. `dvload(name, field='ite')` is the Dynamo way to read a field.
* `dcp.put(p, v[, round])` edits the current project (the one opened with `dcp.open`) and saves into `./`.
  `dvput` saves into the vpr's `path`, as Dynamo does. pydyn refuses to create that folder if it is missing.

**The dcp GUI.** `dcp` (or `dcp <project>`) in a terminal, from the folder that holds the data, opens the
project manager of Dynamo 1.1.514: an HTML view in a Qt WebEngine window (`pydyn/webui`), with Dynamo's steps
in a sidebar, each step's window as a page and the message list at the bottom. `dcp.open('p')`, `dcp()` and
`dcp.new(...)` open the same window from Python, in its own process. `dcp --tk` opens the older Tk view
(`pydyn/gui.py`). It needs an X display (the remote desktop or VNC); without one it prints why and exits. The windows follow Dynamo's
order and enable each other as Dynamo's do: project (type a name: open it, or create a blank one) -> data ->
table -> template (or "average all") -> masks (ellipsoid, copy mask, fmask/smask "use ones", "use defaults",
enter a mask file, overlay preview on the template) -> numerical parameters (the 8 round columns of Dynamo, area
filters, advanced, general view; double-click a cell, Return saves) -> computing environment (destination,
`gpu_identifier_set`, ...) -> check -> unfold. Every value is written into the project as soon as it is entered,
as in Dynamo. All logic is in `pydyn/dcpcontrol.py`, which was replayed against Dynamo's own dcp windows run headless
(`benchmarks/04_journey_bin8_align_1/checks/gui/parity_gui.md`). Not ported (shown as "[not ported in pydyn]"): the mask editor,
blank/random tables, random options and Fourier compensation of "average all", multireference.

**Running.** From a terminal in the folder that contains the project:

```bash
./bin8_align_1.exe                    # pure Python, in a clean environment (no SBGrid PYTHONPATH/LD_LIBRARY_PATH)
./bin8_align_1.exe --gpus 0,1         # other GPUs than gpu_identifier_set
pydynamo info bin8_align_1            # rounds, iterations, next_iteration
pydynamo check bin8_align_1           # = dvcheck
pydynamo unfold bin8_align_1          # = dvunfold
prototype/bin/pydynamo-shell          # a shell with pyDynamo's environment, no SBGrid, with Slurm's sbatch
dcp [project]                         # the project window (pydyn/webui; needs an X display; dcp --help)
dcp --p NAME --d DATA --template REF.em --masks default --t TABLE.tbl   # dcp.new + the window, in one line
pydynamo                              # Python console with Dynamo's commands: dcp.new(...), daverage(...)
dtrandomize_azimuth --i particles_edit.tbl --o particles_edit_mod.tbl          # Dynamo's console steps, from the shell
daverage --d filamentsData_ctf --t particles_edit_mod.tbl --o raw_template_ctf.em
```

The `.exe` keeps Dynamo's five comment lines. Its last line runs
`python -I -m pydyn execute_project <name>`, which calls `DynamoProject` in place on `./<name>`, instead of
`dynamo_execute_project <name>`. On the cluster, start it inside an allocation (`benchmarks/slurm/sbx_run.sh`).

**Differences from Dynamo 1.1.514, by design or for safety:**

| What | Dynamo | pydyn |
|---|---|---|
| last line of `<name>.exe` | `dynamo_execute_project <name>` | the Python launcher above |
| unfold copies a symlinked seed | copies the link and then writes column 34 through it (this damaged `test_data`, 08a §6.1) | copies the content (and says so) |
| `dvput` when `vpr.path/<name>` does not exist | creates it there | raises |
| `dvcheck` GPU motor check | `which dynamo_compute_iteration_spp` | CuPy available and `gpu_motor` `'spp'` |
| `matlab_workers_average` | opens a MATLAB pool | ignored (averaging runs on the GPU) |
| column 28 of refined tables | MATLAB `sin`/`cos` | numpy by default; `--matlab-trig` uses the MCR library for bit-exact text |

Everything else that is not ported raises `NotImplementedError`. This includes multireference, `.mrc` modes other
than 2, `rng(s)` with s ≠ 0, non-default `daverage` options and plugins.

**Checks.** `benchmarks/04_journey_bin8_align_1/checks/dcpnew/run_check.py` repeats the reference runs A and B of 08a in Python and
compares every file and every vpr field with headless Dynamo (`parity.md`). The CPU tests
`prototype/tests/test_stage3.py` compare with Dynamo's own output, kept as small text fixtures in
`tests/fixtures/stage3`.

## Older prototype tools

`pydyn_cli.py`, `pydyn_gui.py` and their `project.json` format were removed in Stage 3; `prototype/bin/dcp` now opens
the Stage 3 GUI above. `pydyn/project.py` (its `default_mask` is Dynamo's, byte-checked) and `pydyn_run.py` (the
multi-GPU driver used by the benchmark scripts, `--project` reads a `project.json`) are kept for the benchmarks only.

## Tests

```bash
python -m pytest -m "not gpu"            # CPU tests; this is what GitHub Actions runs (.github/workflows/ci.yml)
# GPU tests, on the cluster inside the sandbox:
benchmarks/slurm/sbx_run.sh --cpus 4 --gpus 0 --mem 16G --timeout 600 --clean -- python -I -m pytest
```

The CI also fails if a data file (`.em`, `.mrc`, `.btbl`, ...) or any file over 1 MB is tracked.

## Inputs and outputs

**Inputs:** the same as a `dcp` project:
- a data folder of `particle_NNNNN.em`;
- a Dynamo `.tbl`;
- a template `.em`, an alignment mask `.em`, and optionally a template fmask.

**Outputs** use Dynamo's layout in the project folder:
- `results/ite_XXXX/averages/`: `refined_table`, `average`, `average_unweighted`, `fweight_average_raw`, `fmask_average`, `eo_fsc`;
- `results/ite_XXXX/starting_values/`: the reference and the prepared template for each iteration;
- nothing else: the project folder holds exactly the files Dynamo writes (`log.txt` line for line, `gpulog.txt` in the working folder). Timings and diagnostics go to stdout; `--report PATH` (or `$PYDYN_REPORT`) writes a JSON report to a path outside the project.

## Parts

| File | What it does |
|---|---|
| `pydyn/engine.py` | Alignment, the same math as Dynamo's CUDA engine `dynamo_compute_iteration_spp`, batched on the GPU (`fast` mode) |
| `pydyn/dynamo_exact.py` | `exact` mode of the engine: Dynamo 1.1.514's own kernels (PTX from the installed binary), cuFFT 8 and cuBLAS 8 in Dynamo's order; refined tables bit-identical to the official engine on the same GPU model |
| `pydyn/angles.py` | Dynamo's orientation generator (float32 semantics through glibc) |
| `pydyn/average.py` | Dynamo's pipeline average: odd/even halves, normalisation, wedge compensation, FSC |
| `pydyn/reference.py` | Next-iteration reference: symmetry c1/cN/dN, band-pass, normalisation |
| `pydyn/balancer.py` | Chooses the largest batch that fits in free GPU memory and halves it on out-of-memory |
| `pydyn/project.py` | Dynamo's default mask (identical); the older `project.json` format used by `pydyn_run.py --project` |
| `pydyn_run.py` | Multi-GPU driver: one persistent worker per GPU, particles resident across iterations |
| `run_engine.py` | Drop-in replacement for Dynamo's GPU engine on a Dynamo card file (used for the head-to-head tests) |
| `pydyn/dcpcontrol.py`, `pydyn/webui/`, `pydyn/gui.py`, `bin/dcp` | The dcp GUI: controller (no display), the HTML view in a Qt WebEngine window (`dcp`), and the older Tk view (`dcp --tk`) |

## Not supported yet

The checks reject a project that uses any of these:
- wedge types other than 0 and 1;
- binning (`dim` smaller than the box);
- several references or MRA, and Dynamo's thresholding options;
- `area_search_modus` other than 0, 1 and 2;
- Nyquist-fraction low-pass values;
- several rounds.

The shared-grid mode (`run_engine.py --shared-grid`) is an experiment that changes results. It is off by default and not used.

## Engine modes

`--engine exact|fast` (`pydynamo execute_project`, `NAME.exe`, `run_engine.py`), or `PYDYN_ENGINE=exact|fast`;
the default is `exact`:
- `exact` executes Dynamo 1.1.514's own arithmetic: the engine PTX extracted from
  `/programs/x86_64-linux/dynamo/1.1.514/cuda/bin/dynamo_compute_iteration_spp` (md5-checked, cached in
  `$PYDYN_CACHE_DIR`, default `~/.cache/pydyn`), cuFFT 8.0.44 and cuBLAS 8.0.45 from
  `/programs/x86_64-linux/cuda/8.0/lib64` (`$PYDYN_CUDA8_LIB`), float32 host arithmetic as in `spp.cu`. Orientations
  are processed in batches (`--batch`, default 128; 1 = Dynamo's own one-at-a-time loop) with bit-identical results.
  Verified against the official binary orientation by orientation (`benchmarks/09_engine_parity/exact`).
- `--exact-sms N` (or `PYDYN_EXACT_SMS`): Dynamo's results depend on the GPU model through cuBLAS 8's `Sdot` grid
  (12 blocks per SM). With `--exact-sms 64` the exact engine on an A40 reproduces runs made on an RTX A5000 (the
  user's abp_align_eo iteration 1, ref 1: all 4,711 refined rows equal to the text table's precision).
- `fast` is pyDynamo's batched R2C engine; it differs from Dynamo at the 7th digit, which can flip near-ties.
