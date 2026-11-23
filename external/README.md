# external/

`external/` is an optional local download area. Git ignores everything in it except this README
(`.gitignore`: `external/*`, `!external/README.md`), so downloads and unpacked archives stay local.

pyDynamo reads nothing from `external/` at run time. The docstring of `pydynamo/bandpass/realign.py` cites MEX sources of a
Dynamo download unpacked here (`external/dynamo-1.1.558/mex/src`). That citation is for reading only.

## What pyDynamo needs from a Dynamo 1.1.514 installation

pyDynamo does not run MATLAB or Dynamo, and loads no MATLAB library. For the exact engine it reads Dynamo 1.1.514's
engine binary from a local Dynamo installation and loads the CUDA 8.0 cuFFT / cuBLAS libraries. By default it looks
in an SBGrid-style software tree under `/programs/x86_64-linux/`.

### Exact GPU engine (`--engine exact`, the default)

| File | Default location | Environment variable |
|---|---|---|
| Dynamo's engine binary `dynamo_compute_iteration_spp` | `$PYDYNAMO_DYNAMO_ROOT/cuda/bin/` | `PYDYNAMO_DYNAMO_ROOT` (default `/programs/x86_64-linux/dynamo/1.1.514`) |
| cuFFT 8 `libcufft.so.8.0` and cuBLAS 8 `libcublas.so.8.0` | `$PYDYNAMO_CUDA8_LIB/` | `PYDYNAMO_CUDA8_LIB` (default `/programs/x86_64-linux/cuda/8.0/lib64`) |

- At run time pyDynamo extracts Dynamo's CUDA kernels (PTX) from the engine binary and checks the md5 of the PTX
  (`da35c4c571db739ad40cd0edd9703242`). The md5 of the 1.1.514 engine binary itself is
  `8d14a9a39d622ca71a697b832ee0ede9`.
- The extracted PTX is cached as `dynamo_spp_1.1.514_<md5>.ptx`. The cache folder is `$PYDYNAMO_CACHE_DIR` if that
  variable is set (only that folder is tried). Otherwise pyDynamo uses `$XDG_CACHE_HOME/pydynamo` (default
  `~/.cache/pydynamo`), or `pydynamo-cache-<uid>` in the system temporary folder when the first one cannot be written.
  A cached file with the wrong md5 is extracted again.
- `PYDYNAMO_DYNAMO_PTX` names a PTX file to use instead of extracting one. It must have the md5 above. The engine
  binary must still be present, because the start-up check looks for it.
- Before any project work starts, a start-up check confirms that the engine binary, both CUDA 8 libraries and the
  PTX are available. If any of them is missing, the run stops with a message that lists what is missing.
  `--engine fast` (or `PYDYNAMO_ENGINE=fast`) needs none of these files, but its results are not bit-identical to
  Dynamo's.

### FFTW 3 (averaging and masks, with either engine)

MATLAB's `fftn` / `ifftn` are FFTW. pyDynamo calls FFTW 3 itself (`PYDYNAMO_FFTW_LIB` if it is set, then only that
library is tried; else the system's `libfftw3.so.3`, e.g. the `libfftw3-3` package). The system FFTW 3.3.8 gives
the same bits as MATLAB's own FFTW 3.3.3. Without any FFTW library numpy/scipy FFTs are used: the averages are then
not bit-identical, and a one-time note is printed to stderr.

MATLAB's scalar trigonometry (table columns 27-30, the ABP angle math) is fdlibm, which pyDynamo ports
(`pydynamo/matlab/fdlibm.py`); no MATLAB library is loaded.

## Checking an installation

With the environment active (`source .envs/pydynamo/bin/activate`), run:

```bash
python -I -c "from pydynamo.engine import preflight; print(preflight('exact'))"   # prints: exact
python -I -c "from pydynamo import masks; print(masks.fft_backend())"             # fftw:<library> or scipy
```

## Not needed at run time

- The CPU test suite runs without any of these files (CI has no Dynamo installation).
- Dynamo 1.1.514's MATLAB sources (`matlab/src` in the same installation) are read only by the generators in
  `tools/` and by some tests, which are skipped when the sources are absent.
