# External: Dynamo

## What is here

| Path | Content |
|---|---|
| `download/dynamoForMatlab.tar` | Dynamo **1.1.558**, "MATLAB version" from https://www.dynamo-em.org/w/index.php?title=Downloads (Google Drive id `1x5he7ctkC4BUCNFhfXLk6heggGohELoM`, 307 MB, sha256 `b46c9d5ffcf68a375e51ac58db66bc822a03b56a6afbae1aed22fa1911870c69`) |
| `dynamo-1.1.558/` | The extracted tarball, without the macOS `._*` / `.DS_Store` metadata files. It has about 31k `.m` files (only 3 `.p`), CUDA sources in `cuda/src`, and C++/MEX sources in `C/src` and `mex/src` |

## How to use it on this cluster

* **In MATLAB** (needs a MATLAB license, and none was found on the cluster):
  `run <repository>/external/dynamo-1.1.558/dynamo_activate.m`
* **Without MATLAB (what works today):** use the SBGrid standalone build 1.1.557.
  It ships the MATLAB Runtime R2024a. Compared with 1.1.558:
  * The CUDA sources and the CUDA engine binary are byte-identical. SBGrid only wraps the binary to set `LD_LIBRARY_PATH` to CUDA 12.1.
  * The classic `dynamo_execute_project` / `dynamo_iteration_*` / `dynamo__align_motor` / `@dcp` files are unchanged.
  * 173 other files differ: mostly GUI code, the newer `+dpksta` project framework, rebuilt MEX binaries, and `dynamo_bandpass_nyquist.m`.
    `dynamo_bandpass_nyquist.m` now builds its band ellipsoids with `dpkgeom.sphere` and `dpkgeom.conventions.center.dynamo`, which can matter for bit-level parity.

  Usage:

  ```bash
  # runs a Dynamo/MATLAB script headless (about 50 s runtime start-up)
  export DYNAMO_X=1.1.557              # override the 1.1.514 pin in ~/.sbgrid.conf
  export MCR_CACHE_ROOT=$HOME/.dynamo   # or a local scratch dir
  /programs/x86_64-linux/dynamo/1.1.557/bin.capsules/dynamo myscript.m
  ```

  SBGrid traps on this cluster:
  * `~/.sbgrid.conf` pins `DYNAMO_X=1.1.514`.
    Dynamo projects call `dynamo_execute_project`, `dynamo_compute_iteration_spp`, etc. by bare name through SBGrid shims.
    So even a project started from the 1.1.557 path silently runs 1.1.514 (MATLAB Runtime 9.6 instead of 24.1).
    Always `export DYNAMO_X=1.1.557`, and check that the logs say `MATLAB Runtime version 24.1`.
  * The login shell's `PYTHONPATH`, `LD_LIBRARY_PATH` and `PATH` carry 4d / SBGrid / conda entries.
    Run Python work through `../benchmarks/slurm/clean_env.sh`, a wrapper that uses `env -i` and keeps only the Slurm/CUDA device variables.
  * The user pip config adds an NVIDIA index host (`pypi.ngc.nvidia.com`) that does not resolve here. `clean_env.sh` sets `PIP_CONFIG_FILE=/dev/null`.
  * Run tests inside a Slurm allocation (`srun --jobid=<id> --overlap ...`) so that the scheduler knows which GPUs are in use.

  Gotchas:
  * The launcher joins all script lines with `;` and evaluates them as one line, so:
    * do not use `%` comments;
    * a single error aborts the whole script, so wrap risky steps in `try ... end`.
  * MATLAB `gpuArray` does not work inside this runtime (a CUDA library symbol mismatch).
    Dynamo's GPU alignment runs through the separate executable `cuda/bin/dynamo_compute_iteration_spp`.

Analysis of `dcp` and the alignment pipeline: see `../docs/DCP_ANALYSIS.md` and `../docs/analysis/`.
