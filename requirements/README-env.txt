pyDynamo Python environment (uv, no conda)
==========================================

What it is
----------
A plain Python virtual environment named "pydynamo", made with uv (MIT/Apache-2.0, free).
  * Python: CPython 3.11.18 from python-build-standalone (release 20261026, build "cpython-3.11.18-linux-x86_64-gnu",
    i.e. the baseline x86-64 build, not the x86_64_v3 one), downloaded by uv. No Anaconda channel is used.
  * Packages: wheels from PyPI only, exact versions and sha256 hashes in requirements/pydynamo-lock.txt.
    They are the versions of the conda env that pyDynamo's Dynamo parity was verified with (2026-11-12).
  * pydyn itself is installed editable (pip install -e), so edits under prototype/pydyn take effect at once.

Everything lives under .envs/ of the repository (nothing in $HOME):
  .envs/tools         small venv of /usr/bin/python3 that holds the uv binary (.envs/tools/bin/uv)
  .envs/uv-python     the uv-managed CPython 3.11.18 (UV_PYTHON_INSTALL_DIR)
  .envs/uv-cache      uv's download/wheel cache (UV_CACHE_DIR); can be deleted, it only speeds up re-creation
  .envs/pydynamo      the environment itself

Use it
------
  source <repository>/.envs/pydynamo/bin/activate
      the prompt now starts with (pydynamo); python, pip, pytest, pydynamo and dcp come from the env
  deactivate
      leaves it (restores PATH and the prompt)

Without activating: <repository>/.envs/pydynamo/bin/python -I ...
The console scripts pydynamo and dcp restart themselves in a clean environment (pydyn.cli._clean_reexec), so
SBGrid/conda variables of the login shell do not leak in.

Re-create it from scratch (exact same files)
--------------------------------------------
Run inside the Slurm sandbox (benchmarks/slurm/sbx_run.sh ... --clean), from the repository root P:

  P=<repository>
  export UV_CACHE_DIR=$P/.envs/uv-cache UV_PYTHON_INSTALL_DIR=$P/.envs/uv-python UV_PYTHON_BIN_DIR=$P/.envs/uv-python/bin
  export UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy PIP_CONFIG_FILE=/dev/null
  UV=$P/.envs/tools/bin/uv

  # 0. only if .envs/tools is missing (the user's pip.conf adds an unreachable NVIDIA index: PIP_CONFIG_FILE=/dev/null)
  /usr/bin/python3 -m venv --without-pip $P/.envs/tools
  $P/.envs/tools/bin/python -I /usr/lib/python3/dist-packages/pip install --isolated --no-cache-dir uv
  # 1. the interpreter (no-op when already installed)
  $UV python install --no-bin cpython-3.11.18-linux-x86_64-gnu
  # 2. the venv; ask for the exact patch version, so that pyvenv.cfg points at .../cpython-3.11.18-linux-x86_64-gnu
  #    and NOT at the minor-version link cpython-3.11-linux-x86_64-gnu (which a later 'uv python install/upgrade
  #    3.11' would silently move to another patch release)
  $UV venv --relocatable --prompt pydynamo --python cpython-3.11.18-linux-x86_64-gnu $P/.envs/pydynamo
  # 3. exactly the locked packages, hash-checked (sync also removes anything not in the lock)
  $UV pip sync --python $P/.envs/pydynamo/bin/python --require-hashes $P/requirements/pydynamo-lock.txt
  # 4. pydyn, editable, without touching the pinned packages
  $P/.envs/pydynamo/bin/python -I -m pip install --no-deps --no-cache-dir -e $P
  #    (setuptools writes prototype/pydyn.egg-info while doing this; it is in .gitignore and can be deleted)

Check: grep home $P/.envs/pydynamo/pyvenv.cfg  ->  .../uv-python/cpython-3.11.18-linux-x86_64-gnu/bin

Rules
-----
  * Do not 'pip install' or 'uv pip install' other packages into this env, and never upgrade one: the outputs must
    stay bit-identical to Dynamo, and that was verified with exactly these versions. Use a separate venv for tools.
  * Do not run 'uv python upgrade' or 'uv python install --reinstall' with this UV_PYTHON_INSTALL_DIR.
  * CUDA: CuPy uses the node's NVIDIA driver (libcuda) plus the CUDA 12.4 wheels of the lock: cuFFT 11.2.1.3,
    cuBLAS/cuBLASLt 12.4.5.8, NVRTC 12.4.127, and the CUDA 12.4.127 headers of nvidia-cuda-runtime-cu12 (NVRTC
    compiles). CuPy 14.2.0 links the CUDA 12.9 runtime statically, so cupy.cuda.runtime.runtimeGetVersion() reports
    12090; the pydyn workloads load no libcudart.so. curand and nvJitLink are not loaded by pydyn and are not installed; if pydyn ever
    needs them, add nvidia-curand-cu12==10.3.5.147 / nvidia-nvjitlink-cu12==12.4.127 (the verified env's versions)
    to the lock, otherwise CuPy would fall back to the system CUDA under /usr/local/cuda.
  * The dcp window (pydyn.webui) is Qt WebEngine (PySide6-Addons, GUI section of the lock). The older Tk view
    (dcp --tk) uses Tcl/Tk 9.0.4, which is built into python-build-standalone (the conda env had Tk 8.6.15).

Moving the env
--------------
The env is made with --relocatable: python, pip, pytest, the activate scripts and every script installed by uv find
the env from their own location. Only the two scripts that pip writes for 'pip install -e' (bin/pydynamo and
bin/dcp) contain the absolute path of the env. After 'mv', run step 4 again in the new place (it rewrites them).

GUI section (Qt viewers, added 2026-11-12)
------------------------------------------
The lock ends with a separate GUI section: pyside6-essentials==6.11.3, pyside6-addons==6.11.3 (added 2026-11-16,
for Qt WebEngine: the dcp window) and shiboken6==6.11.3 (Qt for Python, LGPLv3). They are used only by pydyn.viewers (dmapview, dview, the masks-window
overlay, dtplot, the adaptive-filtering FSC plot), which always runs in its own process; no numerical code imports them.
  * Recreate: recreate.sh installs them together with the rest (uv pip sync of the whole lock).
  * Add them to an existing env without touching anything else (what was done for .envs/pydynamo-uv):
      sed -n '/^# GUI section/,$p' requirements/pydynamo-lock.txt | grep -v '^#' > gui.txt
      (echo "--index-url https://pypi.org/simple"; cat gui.txt) > gui_req.txt
      $UV pip install --python <env>/bin/python --require-hashes --no-deps -r gui_req.txt
  * Verify: <env>/bin/python -I -B requirements/verify_env.py requirements/pydynamo-manifest.json \
        requirements/pydynamo-gui-manifest.json <env>/lib/python3.11/site-packages
    (with the numerical manifest alone, the only complaint must be "EXTRA distributions: ['pyside6-essentials',
    'shiboken6']").
  * The X11 window needs the system libraries of Qt's xcb plugin (libxcb-cursor0, libxkbcommon-x11-0, libxcb-icccm4,
    libxcb-image0, libxcb-keysyms1, libxcb-render-util0, libxcb-xinerama0, ...); all present on the cluster's GPU nodes (Ubuntu
    22.04). Headless tests use QT_QPA_PLATFORM=offscreen.
  * Never start a viewer with the login shell's LD_LIBRARY_PATH: it holds ~/4d/qt6/lib (Qt 6.2.4), which overrides
    PySide6's RUNPATH and breaks the import (undefined symbol in libQt6DBus). pydyn.viewers restarts itself without it.
