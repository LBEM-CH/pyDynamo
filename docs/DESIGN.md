# pyDynamo design: a drop-in for Dynamo users

pyDynamo re-implements Dynamo 1.1.514 by reverse engineering it. Every behaviour comes from Dynamo's source (cite file:line) and is confirmed against Dynamo's own output. Nothing is invented.

## The user and their workflow (requirements set by the user, 2026-11-12)

1. **Who the user is:** a Dynamo user, who knows Dynamo's commands, GUI and project folders.
2. **Why they switch:** they use pyDynamo to make the same runs faster, on GPUs, without MATLAB.
3. **How they activate it:** they turn pyDynamo on and off like any environment (conda, or something better). When it is active, the Dynamo commands they use run pyDynamo.
4. **Making the inputs:** they run the same console steps as in Dynamo, for example `dtrandomize_azimuth` and `daverage`, to make the reference `.em` and the particle `.tbl`.
5. **Making the project:** they create it with `dcp.new`, edit it in the dcp GUI, then check and unfold it.
6. **Running it:** they run `./<project>.exe`.
7. **The unfolded project:** it is a folder exactly like Dynamo's, with `settings/`, `cards/`, `results/`, `temp/`, `log.txt` and `next_iteration.txt`.
8. **During the run:** each iteration's starting values and results are written into the project folder, at the same moments and under the same names as Dynamo writes them.

**Rule for the output:** the output must be exactly Dynamo's. That means:
- no extra pyDynamo files in the project folder;
- `log.txt` line for line, except values that also differ between two Dynamo runs (dates, durations, GPU model);
- the `.exe` byte-identical to the one Dynamo writes.

## How each requirement is met, and what is missing

| # | Requirement | Status | Gap or work item |
|---|---|---|---|
| 1, 4 | Dynamo console functions under Dynamo's names and argument style | `from pydyn import *` gives `dread`, `dwrite`, `dtrandomize_azimuth`, `daverage`, `dcp`, `dvput`, `dvcheck`, `dvunfold`, `dvload`, `dveo`. MATLAB-style name/value pairs work, e.g. `dcp.new('abp_align','d','filamentsData_b8_ctf','template','x.em','masks','default','t','y.tbl');`. The trailing `;` is valid Python. | Acceptance test with the user's exact commands for both journeys (`bin8_align_1` and `abp_align` → `abp_align_eo`). |
| 3 | Activation with a free environment named `pydynamo`; the user does not want conda | Now: a conda prefix env, `.envs/pydynamo`. Its Python comes from Anaconda's defaults channel; every package in it is a pip wheel from PyPI. The `dcp` and `pydynamo` commands restart themselves in a clean environment, so SBGrid's `PYTHONPATH`/`LD_LIBRARY_PATH` cannot leak in. | Replace it with a `uv` venv (free; python-build-standalone CPython 3.11.18; exact pinned wheels in `requirements/pydynamo-lock.txt`), activated with `source …/pydynamo/bin/activate`. The switch is allowed only after the new env is shown to give bit-identical results to the verified one (in progress). |
| 5 | `dcp.new` opens the GUI; `dcp` in a terminal opens the GUI; check and unfold from the GUI | Done (Stage 3); `dcp.new` opens the GUI by default (`gui` defaults to 1), as in Dynamo. | Desktop test by the user. Mask editor (cylinder) and multireference > adaptive filtering menus: in progress (workflow `abp-align-helical-parity-gpu`). |
| 6 | `./<project>.exe` | **Done (2026-11-13):** pyDynamo writes Dynamo's `.exe` byte for byte (five comment lines, then `dynamo_execute_project <name>`; checked against `test_data/bin8_filament/abp_align_eo.exe`). The console script `dynamo_execute_project = pydyn.cli:execute_main` (pyproject) runs pyDynamo when the environment is active (clean-environment restart kept); without the environment it is Dynamo's command. | The entry point is written but the environment is not reinstalled yet: after the proof run, `pip install --no-deps -e .` writes `bin/dynamo_execute_project`, `bin/dmapview`, `bin/dview`. |
| 7, 8 | Project folder and files exactly Dynamo's, during and after the run | **Done (2026-11-13):** `pydyn_log.txt` and `pydyn_report.json` are no longer written (diagnostics go to stdout; opt-in `--report PATH` / `$PYDYN_REPORT`, refused inside the project). `log.txt` is written line for line through `pydyn.dynlog` (setup banner, freeze/shift lines, engine blocks in device order, `Returning control`, assemble phases, convergence, ending, check; the `[bandpass]` lines as before). `gpulog.txt` is written in the working folder. During the run `temp/ite_X/intermediate_results/` gets `completedTasks_gpuProcRef_*.btbl` after every particle (exact engine) and the refined `.btbl` every 1000 particles. Acceptance: `benchmarks/07_output_layout/check_log.py --project` on a pyDynamo run (see the Step-1 report). | Fast engine: the progress files are written once per spectra chunk, not per particle (its batches run level by level). |
| 5, 8 | Dynamo's viewers: the map viewer (`dmapview`) for templates, masks and the two half-maps after an ABP run, the view buttons in the dcp windows, and table viewing | **Wired (2026-11-13):** `pydyn` exports `dmapview`, `dview`, `dslices`, `dtplot` (lazy); the dcp windows have Dynamo's menus (main: View > Last computed average(s), Multireference > Adaptive filtering > Plot attained resolution FSC; template: Inspect > dview; table: Inspect > Plot orientations; masks: View mask / View Classification mask > dview, overlay x/y/z; mask editor: Show panel and view buttons). The viewers run as a separate Qt (PySide6) process. pyproject: `packages = ["pydyn", "pydyn.viewers"]`, the `gui` extra, `dmapview` and `dview` scripts. Spec: `docs/analysis/11a_viewers_1.1.514.md`. | Desktop test by the user; PySide6 is installed into the new env only (switch after the proof run). Not ported: table "General info", "Last computed table", "All seed files", mask editor "filtered". |
| 5 | GUI on a published UI framework, fast to start, launched on the remote desktop (user, 2026-11-13: the Tk version is "super slow and very ugly"; 2026-11-16: "Qt + HTML", no ssh tunnel) | `pydyn/webui`: an HTML view (Bootstrap 5.3.3 + Bootstrap Icons, vendored, sha256 in `static/vendor/MANIFEST.txt`) in a Qt WebEngine window (PySide6-Addons 6.11.3), over a local token-protected server; all logic stays in `dcpcontrol.py`. One window: Dynamo's steps in a sidebar with their enabled/correct state, each step's window as a page with Dynamo's title, the message list as a dock, dialogs for the mask editor, the ABP parameters and a file chooser of the node's files; Dynamo's View and Multireference menus. `dcp --p NAME --d DATA --template REF --masks default --t TABLE` creates and opens a project in one line (dcp.new with Dynamo's option names; Dynamo's single-dash spellings also work); `dtrandomize_azimuth --i IN --o OUT` and `daverage --d DATA --t TABLE --o AVG` run those console steps from the shell; `pydynamo` alone opens a Python console with Dynamo's commands (type `dcp.new(...)` there); `dcp.new`/`dcp.open`/`dcp()` open the window in its own process. `dcp --tk` keeps the old view. Start-up about 1.4 s from the NFS env. | **Done (2026-11-16):** `tests/test_webui.py` replays Dynamo's own GUI journey (run G) through the view's API against the same fixtures, checks the server's refusals, the one-line start and the console, and runs the HTML view in Qt WebEngine (offscreen) through every step without script errors. Desktop test by the user. |
| — | Anything not ported | pyDynamo raises `NotImplementedError` before writing anything. | Keep it that way: the user can then deactivate pyDynamo and run Dynamo on the same project. |

## Acceptance test (to build)

A scripted "Dynamo user" session, run for each of the user's journeys:
1. a fresh login shell, then `conda activate` pyDynamo;
2. Python with the user's exact console commands;
3. `dcp.new` and the GUI, driven headless, up to check and unfold;
4. `./<name>.exe`.

It then compares the whole project folder (tree, every file, `log.txt` after normalising run-dependent values) and the `.exe` bytes with Dynamo 1.1.514's own folder for the same journey.
