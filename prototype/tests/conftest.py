import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _gpu_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_available()


def _cupy_importable():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


# dvcheck of a GPU project: no remark when pyDynamo's GPU engine can run (CuPy installed); without CuPy (e.g. CI) the
# one warning Dynamo gives for a GPU motor it cannot locate (vprcheck, check_gpu_motor)
CHECK_CLEAN = ([], [], [], "none") if _cupy_importable() else (
    ['WARNING on parameter "gpu_motor". The GPU motor function could not be located in the expected location:',
     "pydyn engine (CuPy) for gpu_motor 'spp'"], ["gpu_motor"], [], "warning")


def pytest_collection_modifyitems(config, items):
    if GPU:
        return
    skip = pytest.mark.skip(reason="needs CuPy and a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
