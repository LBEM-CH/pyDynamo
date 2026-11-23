# pyDynamo: Dynamo 1.1.514's dcp workflow in Python, on GPUs

**Status:** pyDynamo re-implements the subtomogram-averaging workflow of Dynamo 1.1.514 (console steps, `dcp`
projects, the GPU alignment and the averaging) in Python, without MATLAB. The tables, averages and cards it writes are
bit-identical to Dynamo's in these conditions: the exact engine runs on the same GPU model (or with `--exact-sms`),
and an FFTW 3 library is available. The `.exe` is byte-identical, and `log.txt` follows Dynamo's line by line.
With adaptive_bandpass = 1, Dynamo realigns reference 2 inside `dynamo_execute_project`, which MATLAB runs with one
thread per core of the job, and MATLAB's sums change with the thread count. pyDynamo reproduces it with
`--matlab-threads N` (default: one per physical core available, as MATLAB; e.g. 64 for a Dynamo run on a whole
64-core node), measured bit-identical for 16 and 64 threads. MATLAB's own `fftn` / `ifftn` calls are reproduced bit
for bit (measured for boxes 16 to 320, odd, non-power-of-two and non-cubic sizes).

Names: **pyDynamo** is the project (MIT licence, see `LICENSE` and `NOTICE`). **pydyn** is the Python package
(`prototype/pydyn`). **pydynamo** is the environment (`.envs/pydynamo`) and also the console command. Paths below are
relative to the repository root.

Some docstrings cite detailed specifications and benchmark records; these are kept outside the public repository.

## Install

The environment is a uv virtual environment with CPython 3.11.18. It holds the exact, hash-checked wheels of
`requirements/pydynamo-lock.txt` (NumPy, SciPy, CuPy for CUDA 12, PySide6 for the windows) and pydyn installed in
editable mode. `requirements/README-env.txt` covers how to create, check and move it. In short:

```bash
bash requirements/recreate.sh "$PWD/.envs/pydynamo"   # from the repository root; needs .envs/tools/bin/uv
source .envs/pydynamo/bin/activate                     # the prompt starts with (pydynamo); `deactivate` leaves
```

- Dynamo parity was verified with the locked versions only. Do not add packages to this environment or upgrade any.
- `pyproject.toml` also declares the extras `gpu` (`cupy-cuda12x`), `gui` (PySide6) and `test` (pytest), for use in
  other environments.
- The exact GPU engine loads Dynamo 1.1.514's engine binary from a local Dynamo installation, and loads the CUDA 8
  cuFFT and cuBLAS libraries. `external/README.md` describes these files and how to check an installation.
- Bit-identical averages need an FFTW 3 library: the system's `libfftw3.so.3`, or the one `PYDYN_FFTW_LIB` names.

## The user workflow

The workflow is Dynamo's, with the same names, files and project layout. Run everything from the folder that
holds the data, as you would in Dynamo.

**1. Activate the environment:** `source .envs/pydynamo/bin/activate`. Or start `prototype/bin/pydynamo-shell`,
an interactive shell with the environment active and nothing from the login profile, a module system (such as an
SBGrid-style software tree) or conda. That shell keeps HOME, the X display, the terminal and Slurm's commands.

**2. Console steps.** Type them in `pydynamo` (a Python console with Dynamo's commands loaded) or after
`from pydyn import *`:

```python
dtrandomize_azimuth('particles.tbl', 'o', 'particles_rand.tbl')   # MATLAB's start-up random stream
oa = daverage('particles_data', 't', 'particles_rand.tbl', 'mw', 50)  # GPU if visible, else CPU (same result)
dwrite(oa.average, 'template.em')
dcp.new('myproject', d='particles_data', template='template.em', masks='default',
        t='particles_rand.tbl')              # opens the dcp window, as in Dynamo; gui=0 creates only
dvput('myproject', 'd', 'destination', 'standalone_gpu', 'gpu_identifier_set', [0, 1])
r = dvcheck('myproject')                      # r.report, r.fields_warning, r.fields_error, r.warning_state
dvunfold('myproject')                         # writes myproject.exe next to the project
```

- MATLAB-style name/value pairs also work: `dcp.new('p', 'd', 'data', 'template', 'x.em', ...)`.
- Python lists and arrays are numeric vectors (MATLAB `[...]`). `pydyn.vpr.MatCell([...])` is a MATLAB cell.
- `dvload(name, field='ite')` reads a field. Dynamo has no `dvget`.
- `dcp.put(p, v[, round])` edits the current project (the one opened with `dcp.open`) and saves into `./`. `dvput`
  saves into the vpr's `path`, as Dynamo does.
- Also exported: `dread`, `dwrite`, `rng`, `dveo` (the even/odd project), `dmask` / `dmollify` / `dcylinder`,
  and the viewers `dmapview`, `dview`, `dslices` and `dtplot`.
- From the shell: `dtrandomize_azimuth --i IN.tbl --o OUT.tbl` and
  `daverage --d DATA --t TABLE.tbl --o AVERAGE.em [--mw N] [--device auto|gpu|cpu] [--all-outputs]`.

**3. The dcp window.** `dcp`, or `dcp NAME`, opens Dynamo 1.1.514's project manager (`pydyn.webui`). It is an HTML
view in a Qt WebEngine window. `dcp()`, `dcp.open('p')` and `dcp.new(...)` open the same window from Python, in its
own process.
- `dcp --p NAME --d DATA --template REF.em --masks default --t TABLE.tbl` creates the project (`dcp.new`, with
  Dynamo's checks) and opens it. Add `--no-gui` to create it only. See `dcp --help`. Dynamo's single-dash spellings
  (`-template`, `-masks`) also work.
- The window needs an X display. Without one, it prints the reason and exits. `dcp --serve` opens no window: it
  serves the view on 127.0.0.1 and prints the address, for a browser on the same machine.
- The sidebar holds Dynamo's steps in Dynamo's order: Data, Table, Template (or "average all"), Masks, Numerical
  parameters, Computing environment, Check, Unfold and Run. Each step enables the next as Dynamo's windows do.
  Every value is checked and written into the project as soon as it is entered.
- Also ported: the Multireference menu (even/odd project, golden-standard derive, adaptive-bandpass editor), the mask
  editor (cylinder masks) and Dynamo's viewers. Each viewer runs as its own Qt process.
- Actions that are not ported show `[not ported in pydyn]` in the message list (see "Not supported").
- All logic is in `pydyn/dcpcontrol.py`, which needs no display. The view only draws.

**4. Check and unfold** from the window, or with `dvcheck` / `dvunfold` (or `pydynamo check NAME` /
`pydynamo unfold NAME`).

**5. Run** the project from a terminal in the same folder:

```bash
./myproject.exe                                  # Dynamo's own script; takes no arguments
dynamo_execute_project myproject --gpus 0,1      # other devices than gpu_identifier_set
dynamo_execute_project myproject --gpus all      # every GPU this process sees (e.g. a Slurm job's)
```

`NAME.exe` is byte-identical to the one Dynamo writes: five header lines (four comments and a blank line), then
`dynamo_execute_project NAME`.
While the environment is active, that command is pyDynamo's console script. Without the environment, it is Dynamo's
own command. The command takes these options:
- `--engine exact|fast` and `--exact-sms N` (see "GPU engine");
- `--iterations N` (stop after that iteration);
- `--keep-temp` (keep `temp/ite_XXXX`, which Dynamo deletes);
- `--report PATH` (opt-in JSON diagnostics, refused inside the project folder).

`pydynamo execute_project NAME [options]` does the same. The command restarts itself once in a clean environment,
without `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `PYTHONUSERBASE`, `LD_LIBRARY_PATH`, `LD_PRELOAD` or the user
site. A login environment that adds its own Python, Qt 6 or CUDA libraries therefore cannot leak in. With the exact
engine, a start-up check confirms that the Dynamo files are present before any project work.

**From the dcp window: this machine or Slurm.** The Computing environment page has a **Submit to Slurm** box, saved
into the project. Unticked, the Run step runs the project on this machine's GPUs (`gpu_identifier_set`), in the
background. Its output goes to `pydyn_NAME.out` next to the project. Ticked, the Run step submits it with `sbatch` as
a job asking for as many GPUs as the project has, on one machine:
- the job's CPU cores, RAM (empty: estimated from the data) and partitions (empty: the shared partitions that have
  machines with that many GPUs, read with `sinfo`) are pyDynamo's `pydyn_slurm_cpus`, `pydyn_slurm_mem` and
  `pydyn_slurm_partition`;
- its time limit and the submit command are Dynamo's own `cluster_walltime` (`--time`; Dynamo's default is
  00:10:00) and `submit_order`. Dynamo writes both into the cards, so changing them needs a new unfold;
- inside the job the run uses the job's GPUs (`--gpus all`), and its output goes to `slurm-<job id>.out`.

The Run page shows the state of the last run (`squeue`, then `sacct`) and stops it (`scancel` for a job). The last run
of a project is recorded in `pydyn_NAME.job.json` next to it, so it is shown again when the window is reopened.
The run is pyDynamo's `dynamo_execute_project NAME`, started with the environment's Python in a clean environment.
The particle files are read with up to 8 of the job's CPU cores.

pyDynamo's `pydyn_slurm*` fields are stored in `settings/virtual_project.mat` only while they differ from their
defaults, so a project that never ticks the box keeps Dynamo's file as it is. Dynamo reads only the fields it knows,
and its own dcp window does not keep these when it saves a project.

`pydynamo info NAME` prints a project's rounds, iterations, next iteration, data and seeds.

## Inputs and outputs

**Inputs** are those of a `dcp` project:
- a data folder of `particle_<tag>.em`;
- a Dynamo `.tbl`;
- a template `.em`;
- masks (or `masks='default'`).

**Outputs** follow Dynamo's layout in the project folder: `settings/`, `cards/`, `results/ite_XXXX/starting_values/`,
`results/ite_XXXX/averages/`, `temp/`, `log.txt` and `next_iteration.txt`. As in Dynamo, `temp/ite_XXXX` is deleted
at the end of each iteration's assembly.
- `averages/` holds `refined_table`, `average`, `average_unweighted`, `fweight_average_raw`, `fmask_average`,
  `eo_fsc`, `average_symmetrized` and `fmask_average_symmetrized`. As in Dynamo, the last two are written for every
  symmetry; for c1 they are copies of `average` and `fmask_average`.
- The project folder contains no file that Dynamo does not write. `gpulog.txt` goes to the working folder, as Dynamo
  writes it, and so do the Run step's `pydyn_NAME.out` and `pydyn_NAME.job.json`. Only the Submit to Slurm settings
  add fields to `settings/virtual_project.mat`, and only while they are set (see "From the dcp window").
- pyDynamo's progress lines and timings go to stdout.

## Exactness

### GPU engine

`--engine exact|fast` (or `PYDYN_ENGINE`) selects the engine. The default is `exact`.

- **exact** runs Dynamo 1.1.514's own CUDA kernels. Their PTX is extracted at run time from the local Dynamo engine
  binary (`dynamo_compute_iteration_spp`), md5-checked and cached. The FFTs use cuFFT 8.0 and the reductions use
  legacy cuBLAS 8.0's order (by default pyDynamo kernels checked bit for bit against cuBLAS 8 at start; cuBLAS 8 itself
  is the fallback), both from the CUDA 8 libraries. The host arithmetic is float32, as in Dynamo's
  source. The refined tables are bit-identical to the official engine's on the same GPU model.
  - Dynamo's results depend on the GPU model through cuBLAS 8's reduction order. `--exact-sms N` (or
    `PYDYN_EXACT_SMS`) reproduces the order of a GPU with N streaming multiprocessors, e.g. `--exact-sms 64` for an
    RTX A5000.
  - Each particle is aligned on its own, and its orientations are processed 128 at a time (the batched path) with the
    same bits. Fewer are taken when the free GPU memory holds fewer, and an out-of-memory error halves the number;
    the bits do not depend on it. This path reads results back once per refinement level and runs the reductions in cuBLAS 8's order. It
    uploads the rotation matrices once per level, skips the pad copy, and merges steps that are a single IEEE
    operation. The batched reductions and the merged steps are checked at start against legacy cuBLAS 8 and
    Dynamo's kernels on the GPU in use. A failed check turns that option off (the cuBLAS / per-kernel path gives the
    same bits) and prints a note on stderr.
  - `PYDYN_EXACT_OPTS` turns the options off one by one, e.g. `sync=batch` (the earlier per-batch path),
    `red=cublas`, `mats=0`, `nopad=0`, `fuse=0`. An option that is set explicitly and then fails its check raises
    an error.
- **fast** is pyDynamo's own batched CuPy engine (real-to-complex FFTs, particle spectra kept on the GPU). It needs
  nothing from the Dynamo installation. It is close to Dynamo but not identical: float32 differences at the 7th digit
  can flip near-ties between orientations. Its batch size comes from the GPU memory balancer (`pydyn/balancer.py`),
  which only this engine uses. It refuses odd box sizes, `localnc` 0 and `use_CC` other than 0 and 1
  (NotImplementedError).

### Averaging

The averages are bit-identical to Dynamo's when `matlab_workers_average` is 0. For `standalone_gpu`, Dynamo runs
set-up and assembly (and so the averaging and the adaptive bandpass) inside `dynamo_execute_project`, where MATLAB
uses one computational thread per core; its sums over 89,000 or more elements depend on that count. pyDynamo
reproduces this in four ways:
- it uses MATLAB's sums for the same thread count, `--matlab-threads` above (`pydyn/matlab_numerics.py`);
- it adds the particles in ascending tag order and passes one running state through the GPU workers;
- it normalises each particle with MATLAB's sums;
- it computes `fftn` / `ifftn` with MATLAB's own sequence of FFTW calls (1-D transforms one dimension at a time,
  4 lines per plan, read from MATLAB with an FFTW call logger; `pydyn/masks.py`).

With `matlab_workers_average` > 0, Dynamo averages each half on a pool of single-threaded MATLAB workers (a
`parfor` loop). MATLAB splits the particles into subranges, each worker adds its subranges, and the partial sums are
added in the order the workers finish. That order changes from run to run, so Dynamo does not repeat its own
averages in double precision. In measured cases, three or four Dynamo re-averages of the same particles (9,422 and
9,965 particles, 50 workers) differed from each other in about 90 % of the voxels, by up to 1.1e-14, and all rounded
to the same float32 file. pyDynamo splits the particles into MATLAB's subranges (`matlab_numerics.parfor_subranges`,
measured in MATLAB for 8 to 64 workers and 1 to 30,000 particles). It adds each subrange in MATLAB's descending
order with one thread, then adds the subrange sums in order (`average.pool_sums`). Its averages fall within Dynamo's
own run-to-run spread, and its float32 files matched Dynamo's in every measured case.

The even/odd FSC can still differ in its last points. Its last shell (radius n/2 to n/2 + 2, the corners beyond
Nyquist) has Fourier weight 0 in the measured projects, so Dynamo sets those components to 0, and the shell keeps
only rounding noise. Its FSC value, which the last two or three points of `eo_fsc` interpolate (points 30 to 32 of
32 for box 64), changes with any change in the last bit of the half maps. It changes between Dynamo's own pooled
runs, and between pyDynamo and Dynamo. Without a pool, these points are bit-identical too.

FFTW is loaded through ctypes: `$PYDYN_FFTW_LIB` if it is set (then only that library is tried), else the system's
`libfftw3.so.3`. No MATLAB library is used. In measurements, the system FFTW 3.3.8 gave the same bits as MATLAB's own
FFTW 3.3.3. Without FFTW, NumPy FFTs are used. The averages are then not bit-identical, and a one-time note on
stderr says so. The mask editor's `dmollify` falls back the same way, to SciPy FFTs with a warning.

### Table columns 27-30

MATLAB R2019a, the runtime of Dynamo 1.1.514, computes its scalar trigonometry with fdlibm. `pydyn/fdlibm.py` ports
fdlibm's sin, cos, atan, atan2 and acos operation for operation, so it returns MATLAB's values bit for bit. pyDynamo
always uses it for table columns 27-30 (column 28 is complex), so the text matches Dynamo's. It also uses it for the
angle math of `dynamo_align` in the adaptive bandpass. No MATLAB library is needed. NumPy's and the C library's sin
and cos differ from MATLAB's in about 3 % of the arguments. sin and cos of arguments with |x| > 2^19·π/2 raise
`NotImplementedError`.

## Runtime paths and environment variables

| Variable | Effect |
|---|---|
| `PYDYN_ENGINE` | `exact` or `fast`, when `--engine` is not given (default `exact`) |
| `PYDYN_EXACT_SMS` | default of `--exact-sms` (unset: the GPU in use) |
| `PYDYN_EXACT_OPTS` | comma-separated `key=value` list that turns options of the exact engine's batched path off (see above) |
| `PYDYN_EXACT_TIMING` | `1`: host-side time per phase of the exact engine, reported by `run_engine.py` |
| `PYDYN_DYNAMO_ROOT` | the Dynamo 1.1.514 installation whose `cuda/bin/dynamo_compute_iteration_spp` the exact engine reads (default `/programs/x86_64-linux/dynamo/1.1.514`) |
| `PYDYN_CUDA8_LIB` | the folder with `libcufft.so.8.0` and `libcublas.so.8.0` (default `/programs/x86_64-linux/cuda/8.0/lib64`) |
| `PYDYN_DYNAMO_PTX` | a PTX file to use instead of extracting one (it must have the 1.1.514 engine's md5) |
| `PYDYN_CACHE_DIR` | the PTX cache folder (default `$XDG_CACHE_HOME/pydyn` or `~/.cache/pydyn`, then a per-user folder in the system temp folder) |
| `PYDYN_FFTW_LIB` | the FFTW 3 library to use (only that one is tried) |
| `PYDYN_REPORT` | default of `--report PATH` |
| `PYDYN_WEBUI_GPU` | set: keep Chromium's GPU paths in the dcp window (default `--disable-gpu`, for remote desktops) |
| `PYDYN_WEBUI_DEBUG` | set: log the dcp window's local server requests and all page console messages |
| `PYDYN_GPU_TESTS` | `1`: run the GPU tests |

## Differences from Dynamo 1.1.514

| What | Dynamo | pyDynamo |
|---|---|---|
| unfold, seed file that is a symbolic link | copies the link, then writes through it | copies the content (and says so) |
| `dvput` when the project at `vpr.path` does not exist | creates it | raises; use `dcp.put` or set `path` |
| `dvcheck` GPU-motor check | looks for `dynamo_compute_iteration_<gpu_motor>` | CuPy importable and `gpu_motor` `'spp'` |
| `matlab_workers_average` > 0 | opens a MATLAB pool; the averages' last double bits and the last 2–3 `eo_fsc` points change from run to run | no pool; the GPU workers sum in MATLAB's `parfor` subranges: same float32 averages, last `eo_fsc` points differ as between two Dynamo runs (see Averaging) |
| dcp's rounds-table edits (`getVpr`) | write `./decoy/settings/virtual_project.mat` | no `./decoy` folder |
| GUI strings (`str2num`) | any MATLAB expression | numbers, Inf/NaN, true/false, vectors and integer `a:b` / `a:s:b` ranges; anything else raises |
| GPUs used | `gpu_identifier_set` | the same by default; `--gpus LIST` or `--gpus all` overrides it (the particle split still follows `gpu_identifier_set`) |

## Not supported

These cases raise `NotImplementedError`, or appear as `[not ported in pydyn]` in the dcp window.

**Project runs** are checked before any computation (`DynamoProject.validate` in `pydyn/dynproject.py`):
- Execution settings:
  - any destination other than `standalone_gpu`;
  - `how_many_processors` other than 1;
  - `systemUsingProcessorTables` or `systemCardForParticle` set to 1.
- Project setup:
  - `initial_references` other than `'*'`, or `project_type` other than `'general'`;
  - data other than a Dynamo data folder of `.em` particles;
  - a template box that differs from the data box;
  - binning (`dim` smaller than the box).
- References and adaptive bandpass:
  - more than two references, or MRA. With `nref` 2, the two references run independently; this is the even/odd
    project of `dveo`;
  - a reference count that changes between rounds;
  - `adaptive_bandpass` other than 0 or 1 (1 is the golden-standard mode, with two references);
  - the even/odd adaptive bandpass (`adaptive_bandpass_even_odd`).
- Averaging options:
  - `fourierMinimumFractionForAverage` other than -1;
  - Fourier-compensation smoothing (a radius or a mask);
  - `averagingImplicitRotationMask`;
  - `vpr.options`, `fmin`, `pivoting` and `force_mask`.
- Round parameters:
  - `localnc` 0;
  - `use_CC` other than 0 and 1;
  - symmetries other than c1, cN, dN and helical (and a symmetry without `vpr.symmetrize_with_fmask`);
  - a Nyquist-fraction low-pass (`low` < 1), or `high` 0;
  - `threshold_modus` other than 0, or a second thresholding;
  - `separation_in_tomogram`;
  - peak checks (`cone_check_peak`, `inplane_check_peak`, `limit_xy_check_peak`, `limit_z_check_peak`);
  - `plugin_post`;
  - `convergence_type` other than 0;
  - `area_search_modus` outside 0, 1, 11, 2, 21, 3, 31, 4, 41, 5, 51, 6 and 61.

The averaging step also refuses Fourier-sampling (wedge) types other than 0, 1 and 2 (table column 13). Other cases
that are not ported raise the same error at the point where they occur.

**The dcp window** (`pydyn/dcpcontrol.py`) does not port:
- mask types other than the cylinder in the mask editor;
- blank and random tables;
- the random options and the Fourier compensation of "average all", and "average all" for multireference projects;
- database queries (`name:...`) and `*` selections in file boxes;
- the plugin manager;
- the show, progress, rewind and reset windows;
- project search patterns in `dcp`;
- the "Last computed table" and "All seed files" views.

**Console functions** do not support:
- `.mrc` modes other than 2;
- non-default `daverage` options;
- `rng` seeds other than `'default'` / 0;
- `dveo` with `'ite'` > 0;
- MATLAB expressions or `|` in `dvput` strings.

## Developer harnesses

These scripts run from a checkout. They are not installed by `pip install`. Use the environment's Python (`python -I`).

| Script | What it does |
|---|---|
| `prototype/run_engine.py CARD [--engine] [--exact-sms N] [--batch B\|auto]` | Drop-in for Dynamo's GPU engine on one card. Run it from the folder Dynamo's engine would run in. It starts one worker per id in the card's `gpu_identifier_set`, writes `refined_gpuProcRefTable_*.btbl` and prints a JSON summary. `--shared-grid` is a fast-engine experiment that is not Dynamo-identical. |
| `prototype/run_dynamo_project.py --source P --data D --out NEW` | Re-executes an unfolded Dynamo project into a new folder. The source is only read. It takes the engine options of `dynamo_execute_project`, plus `--table-initial` and `--progress-log`. |
| `prototype/dev/average_test.py WORK PROJECT ITE DATA NGPUS` | Averages one iteration on the GPUs as the project runner does (MATLAB thread count, the round's FSC symmetry, the pool's subranges when `matlab_workers_average` > 0) and compares the result with Dynamo's averaging outputs, bit for bit. Set `PYDYN_MATLAB_THREADS` to the core count of the node Dynamo ran on. `AVG_DUMP=file.npz` also saves the full-precision outputs. |
| `prototype/dev/compare_tables.py A B [--truth T]` | Compares two tables (`.tbl` or `.btbl`) particle by particle and prints a JSON summary of the orientation, shift and cc differences. |
| `prototype/bin/dcp` | The `dcp` command, run with `.envs/pydynamo`'s Python without the editable install |
| `prototype/bin/pydynamo-shell` | A clean interactive shell with the environment active (see "The user workflow") |
| `prototype/tools/gen_convention.py`, `gen_dcp_help.py`, `gen_dcp_parameters.py` | Regenerate `pydyn/convention.py`, `pydyn/dcp_help.py` and `pydyn/dcp_parameters.py` from Dynamo 1.1.514's MATLAB sources. The output is checked in, and the tests compare it with the sources when those are present. |

## Tests

```bash
python -I -m pytest -m "not gpu"                                # CPU tests, from the repository root
PYDYN_GPU_TESTS=1 python -I -m pytest -m gpu prototype/tests    # GPU tests: opt-in, inside a GPU job
```

- The CPU tests run anywhere. GitHub Actions runs them without a GPU, a Dynamo installation, data or FFTW.
- Run them in `prototype/bin/pydynamo-shell` (or with `env -u LD_LIBRARY_PATH`): pytest imports PySide6 in its own
  process, and a login `LD_LIBRARY_PATH` that holds another Qt 6 build breaks that import.
- Tests that need a Dynamo installation, its sources or local reference data are skipped when those are absent.
  `pytest -rs`, which is on by default, lists the reasons.
- The GPU tests never run without `PYDYN_GPU_TESTS=1`, even on a machine that sees a GPU.
- CI also fails if a data file (`.em`, `.mrc`, `.btbl`, ...) or any file over 1 MB is tracked.

## Parts

| Module | What it does |
|---|---|
| `pydyn/__init__.py` | The console API under Dynamo's names (`from pydyn import *`), loaded lazily |
| `pydyn/__main__.py` | `python -I -m pydyn ...`, the same as `pydynamo` |
| `pydyn/cli.py` | Console scripts: `pydynamo`, `dcp`, `dynamo_execute_project`, `dtrandomize_azimuth`, `daverage`; restart in a clean environment |
| `pydyn/console.py` | `dread`, `dwrite`, `dtrandomize_azimuth`, `daverage`, `rng` (Dynamo's console functions) |
| `pydyn/dcptool.py` | `dcp`: `dcp.new`, `dcp.open`, `dcp.put`, `dcp()` |
| `pydyn/vpr.py` | Virtual projects (`settings/virtual_project.mat`): load and save, `dvput`, `dvload`, `dvunfold`, the `.exe` script |
| `pydyn/vprcheck.py` | `dvcheck` (`dynamo_vpr_check`, its checks in Dynamo's order) |
| `pydyn/vpreo.py` | `dveo`: the even/odd (golden-standard) two-reference project |
| `pydyn/convention.py` | Dynamo 1.1.514's parameter convention table (generated) |
| `pydyn/convention_parse.py` | Parser of `dynamo_vpr_convention.m`, used to generate and to check `convention.py` |
| `pydyn/dcp_parameters.py` | The dcp windows' parameter classes: labels, areas, types (generated) |
| `pydyn/dcp_help.py` | Dynamo's detailed help text for project parameters (generated) |
| `pydyn/dcpcontrol.py` | Controller of the dcp window (no display): Dynamo's dcp callbacks |
| `pydyn/webui/` | The dcp window: local server (`server.py`), session API (`api.py`), Qt WebEngine window (`shell.py`), HTML view (`static/`, vendored Bootstrap 5.3.3 and Bootstrap Icons 1.11.3) |
| `pydyn/viewers/` | `dmapview`, `dview`, `dslices`, `dtplot` and the adaptive-filtering FSC plot: display logic (`logic.py`) and Qt windows (`qtview.py`), each viewer in its own process |
| `pydyn/execute.py` | `dynamo_execute_project`: destination and GPU checks, then runs the project in place |
| `pydyn/launch.py` | The Run step of the dcp window: this machine's GPUs or a Slurm job (`pydyn_slurm*` fields, the `sbatch` line, `sinfo` partitions, run state, stop) |
| `pydyn/dynproject.py` | `DynamoProject`: the iteration loop (setup, GPU workers, assembly, averaging, propagation), `validate`, columns 27-30 |
| `pydyn/engine.py` | The alignment engine: mode selection and start-up check, and the `fast` batched CuPy engine |
| `pydyn/dynamo_exact.py` | The `exact` engine: Dynamo 1.1.514's PTX, cuFFT 8, cuBLAS 8, the batched path and its self-checks |
| `pydyn/balancer.py` | GPU memory balancer: the largest batch that fits, halved on out-of-memory (`fast` engine) |
| `pydyn/angles.py` | Dynamo's GPU orientation generator, with C float semantics through glibc |
| `pydyn/average.py` | Dynamo's pipeline average on the GPU: ordered sums, wedges, finalize (average, weights, eo FSC); the console `daverage` sums |
| `pydyn/matlab_numerics.py` | MATLAB numerics: sums for a given thread count, mean and std, FFTW `fftn` / `ifftn`, FSC, `linspace`, `interp1`, `parfor` subranges |
| `pydyn/fdlibm.py` | fdlibm's sin, cos, atan, atan2 and acos (MATLAB's scalar trigonometry), bit for bit |
| `pydyn/masks.py` | `default_mask`, `dmask` (cylinder), `dmollify`, `dcylinder`, the mask editor's state, MATLAB's `exp`, the FFTW loader |
| `pydyn/symmetry.py` | `dynamo_sym`: c, d and helical symmetry operators, with symmetrized Fourier weights |
| `pydyn/abp.py` | Golden-standard adaptive bandpass (`adaptive_bandpass` = 1) |
| `pydyn/dalign.py` | CPU `dynamo_align`, as the adaptive bandpass uses it |
| `pydyn/dynio.py` | Byte-exact Dynamo file I/O: `.tbl`, `.em`, `.btbl`, `.card`, `.fsc`, `.txt`, scripts |
| `pydyn/emio.py` | EM volumes and binary tables; reads particle stacks in parallel threads |
| `pydyn/card.py` | Dynamo task cards, parsed as the GPU engine reads them |
| `pydyn/dynlog.py` | `log.txt` and the GPU engine's stdout, line by line as Dynamo writes them |
