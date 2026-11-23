pyDynamo Python environment (uv venv)
=====================================

What it is
----------
"pydynamo" is a plain Python virtual environment made with uv (MIT/Apache-2.0). No conda or Anaconda channel is used.
  * Python: CPython 3.11.18 from python-build-standalone (release 20261026, build "cpython-3.11.18-linux-x86_64-gnu",
    i.e. the baseline x86-64 build, not the x86_64_v3 one), downloaded by uv.
  * Packages: wheels from PyPI only, with exact versions and sha256 hashes in requirements/pydynamo-lock.txt. pyDynamo's
    bit-identical Dynamo parity was verified with exactly these versions.
  * pydynamo itself is installed editable (pip install -e), so edits under src/pydynamo take effect at once.
  * requirements/recreate.sh builds it; requirements/verify_env.py checks it (see below).

Everything lives under .envs/ of the repository (nothing in $HOME; .envs/ is in .gitignore):
  .envs/tools         small venv of /usr/bin/python3 that holds the uv binary (.envs/tools/bin/uv)
  .envs/uv-python     the uv-managed CPython 3.11.18 (UV_PYTHON_INSTALL_DIR)
  .envs/uv-cache      uv's download/wheel cache (UV_CACHE_DIR); can be deleted, it only speeds up re-creation
  .envs/pydynamo      the environment itself

Use it
------
  source <repository>/.envs/pydynamo/bin/activate
      the prompt now starts with (pydynamo); python, pip, pytest and the console scripts pydynamo, dcp,
      dynamo_execute_project, dtrandomize_azimuth, daverage, dmapview and dview come from the env
  deactivate
      leaves it (restores PATH and the prompt)
  <repository>/bin/pydynamo-shell
      instead of activating in the login shell: a new interactive shell with this env active and a minimal
      environment (HOME, USER, the terminal, the X display, PATH /usr/bin:/bin plus the directory of sbatch);
      exit leaves it

Without activating: <repository>/.envs/pydynamo/bin/python -I ...
The console scripts restart themselves once in a clean environment (pydynamo.cli._clean_reexec; the viewers through
pydynamo.viewers.__main__._clean_reexec): without PYTHONPATH, PYTHONHOME, PYTHONSTARTUP, PYTHONUSERBASE, LD_LIBRARY_PATH
and LD_PRELOAD (dmapview and dview also without QT_PLUGIN_PATH, QT_QPA_PLATFORM_PLUGIN_PATH, QML2_IMPORT_PATH and
QML_IMPORT_PATH), so settings of the login shell (a module system such as an SBGrid-style software tree, conda, another
Python) do not leak in.

Re-create it from scratch (exact same files)
--------------------------------------------
  bash requirements/recreate.sh <repository>/.envs/pydynamo    (an absolute path: the script works in a temporary
                                                              folder; e.g. "$PWD/.envs/pydynamo" from the repository root)

recreate.sh refuses to overwrite an existing env (move the old one away first), sets the uv variables below itself
and works in a temporary folder. uv downloads the interpreter and the wheels; no GPU is needed. It expects the uv
binary at .envs/tools/bin/uv (step 0). Steps 1 to 4 are what it runs; P is the repository root:

  P=<repository>
  export UV_CACHE_DIR=$P/.envs/uv-cache UV_PYTHON_INSTALL_DIR=$P/.envs/uv-python UV_PYTHON_BIN_DIR=$P/.envs/uv-python/bin
  export UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy PIP_CONFIG_FILE=/dev/null
  UV=$P/.envs/tools/bin/uv

  # 0. only if .envs/tools is missing (not part of recreate.sh). --isolated and PIP_CONFIG_FILE=/dev/null: no pip
  #    configuration of the user or the site is read (for example one that adds other package indexes). The second
  #    line runs the system's pip module (Debian/Ubuntu path); any pip that can install uv into .envs/tools will do.
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
  # 4. pydynamo, editable, without touching the pinned packages
  $P/.envs/pydynamo/bin/python -I -m pip install --no-deps --no-cache-dir -e $P
  #    (setuptools writes src/pydynamo.egg-info while doing this; recreate.sh deletes it, it is in .gitignore)

The current env was made with uv 0.13.7 (pyvenv.cfg records the version).

Check it
--------
  grep home $P/.envs/pydynamo/pyvenv.cfg  ->  .../uv-python/cpython-3.11.18-linux-x86_64-gnu/bin
  $P/.envs/pydynamo/bin/python -I -B requirements/verify_env.py requirements/pydynamo-manifest.json \
      requirements/pydynamo-gui-manifest.json $P/.envs/pydynamo/lib/python3.11/site-packages
      ->  verify_env: OK (18 distributions, 9878 files)
verify_env.py compares the sha256 of every installed file with the manifests and reports any other distribution
(pydynamo's own editable metadata may differ).

Rules
-----
  * Do not 'pip install' or 'uv pip install' other packages into this env, and never upgrade one: the outputs must
    stay bit-identical to Dynamo, and that was verified with exactly these versions. Use a separate venv for tools.
  * Do not run 'uv python upgrade' or 'uv python install --reinstall' with this UV_PYTHON_INSTALL_DIR.
  * CUDA: CuPy uses the node's NVIDIA driver (libcuda) plus the CUDA 12.4 wheels of the lock: cuFFT 11.2.1.3,
    cuBLAS/cuBLASLt 12.4.5.8, NVRTC 12.4.127, and the CUDA 12.4.127 headers of nvidia-cuda-runtime-cu12 (NVRTC
    compiles). CuPy 14.2.0 links the CUDA 12.9 runtime statically, so cupy.cuda.runtime.runtimeGetVersion() reports
    12090; the pydynamo workloads load no libcudart.so. curand and nvJitLink are not loaded by pydynamo and are not
    installed; if pydynamo ever needs them, add nvidia-curand-cu12==10.3.5.147 / nvidia-nvjitlink-cu12==12.4.127 to the
    lock, otherwise CuPy would fall back to a system CUDA installation (e.g. /usr/local/cuda).
  * Not in the lock: the default exact engine also loads Dynamo 1.1.514's own kernels from the local Dynamo
    installation ($PYDYNAMO_DYNAMO_ROOT) and the CUDA 8 libraries libcufft.so.8.0 and libcublas.so.8.0
    ($PYDYNAMO_CUDA8_LIB); bit-identical averages need FFTW 3 ($PYDYNAMO_FFTW_LIB if set, else the system's
    libfftw3.so.3). See the Requirements section of README.md.

Moving the env
--------------
The env is made with --relocatable: python, pip, pytest, the activate scripts and every script installed by uv find
the env from their own location. Only the files that 'pip install -e' writes hold absolute paths: the console
scripts in bin/ (pydynamo, dcp, dynamo_execute_project, dtrandomize_azimuth, daverage, dmapview, dview; their first
line names the env's python) and pydynamo's editable-install .pth file and direct_url.json in site-packages (they name
the repository). After moving the env, run step 4 again in the new place (it rewrites them). bin/python is an
absolute symlink into .envs/uv-python, and pyvenv.cfg names the same folder, so after moving the whole repository,
re-create the env with recreate.sh.

GUI section (Qt)
----------------
The lock ends with a separate GUI section: pyside6-essentials==6.11.3, pyside6-addons==6.11.3 and shiboken6==6.11.3
(Qt for Python, LGPLv3). PySide6-Essentials holds QtCore, QtGui, QtWidgets and the xcb/offscreen platform plugins;
PySide6-Addons holds Qt WebEngine, which shows the dcp window; shiboken6 is their binding runtime. Only pydynamo.viewers
(dmapview, dview, dslices, dtplot, the masks-window overlay, the adaptive-filtering FSC plot; each viewer runs in its
own process) and the dcp window (pydynamo.webui.shell) import them; no numerical code does.
  * Recreate: recreate.sh installs them together with the rest (uv pip sync of the whole lock).
  * Add them to an existing env without touching anything else:
      sed -n '/^# GUI section/,$p' requirements/pydynamo-lock.txt | grep -v '^#' > gui.txt
      (echo "--index-url https://pypi.org/simple"; cat gui.txt) > gui_req.txt
      $UV pip install --python <env>/bin/python --require-hashes --no-deps -r gui_req.txt
  * Verify: as in "Check it". With the numerical manifest alone, the only complaint must be
    "EXTRA distributions: ['pyside6-addons', 'pyside6-essentials', 'shiboken6']".
  * The X11 window needs the system libraries of Qt's xcb plugin (libxcb-cursor0, libxkbcommon-x11-0, libxcb-icccm4,
    libxcb-image0, libxcb-keysyms1, libxcb-render-util0, libxcb-xinerama0, ...); install them with the system's
    package manager if they are missing. Headless tests use QT_QPA_PLATFORM=offscreen.
  * A login environment that adds its own Qt 6 libraries to LD_LIBRARY_PATH overrides PySide6's RUNPATH and can break
    the import (for example an undefined symbol in libQt6DBus). This is why the console scripts drop LD_LIBRARY_PATH
    and pydynamo.viewers starts each viewer without it and without the Qt plugin path variables.
