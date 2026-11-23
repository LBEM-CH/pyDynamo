#!/bin/bash
# Create the pydynamo env from scratch exactly as requirements/README-env.txt says.
# Usage: bash requirements/recreate.sh <repository>/.envs/pydynamo
# Then check: <env>/bin/python -I -B requirements/verify_env.py requirements/pydynamo-manifest.json requirements/pydynamo-gui-manifest.json <env>/lib/python3.11/site-packages
set -euxo pipefail
P="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"   # the repository
T=$1
export UV_CACHE_DIR=$P/.envs/uv-cache UV_PYTHON_INSTALL_DIR=$P/.envs/uv-python UV_PYTHON_BIN_DIR=$P/.envs/uv-python/bin
export UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy PIP_CONFIG_FILE=/dev/null
UV=$P/.envs/tools/bin/uv
cd "$(mktemp -d)"     # neutral working folder, so nothing is picked up from the current directory
date -Iseconds
$UV python install --no-bin cpython-3.11.18-linux-x86_64-gnu
test ! -e $T     # refuses to overwrite: move the old env away first
$UV venv --relocatable --prompt pydynamo --python cpython-3.11.18-linux-x86_64-gnu $T
$UV pip sync --python $T/bin/python --require-hashes $P/requirements/pydynamo-lock.txt
test ! -e $P/src/pydynamo.egg-info
$T/bin/python -I -m pip install --no-deps --no-cache-dir -e $P
rm -rf $P/src/pydynamo.egg-info
grep home $T/pyvenv.cfg
ls -la $T/lib/python3.11/site-packages/nvidia/
date -Iseconds
