# pyDynamo

pyDynamo re-implements the dcp subtomogram-averaging workflow of Dynamo 1.1.514 in Python, on NVIDIA GPUs. It is
meant for Dynamo users. It has the same console steps (`dread`, `dtrandomize_azimuth`, `daverage`, `dcp.new`,
`dvcheck`, `dvunfold`), the dcp project window (a few parts are not ported) and the same project folders. The unfolded `NAME.exe` is
byte-identical to Dynamo's. The default `exact` engine runs Dynamo's own CUDA kernels with cuFFT 8; its reductions
follow legacy cuBLAS 8's order (pyDynamo kernels checked bit for bit against cuBLAS 8 at start, cuBLAS 8 itself as
the fallback).
With it, the tables, averages and cards are bit-identical to Dynamo's on the same GPU model (the averages when an
FFTW 3 library is present), and `log.txt` follows Dynamo's line by line. With adaptive_bandpass = 1, Dynamo realigns reference 2 inside `dynamo_execute_project`, which MATLAB runs with one thread per core of the job, and MATLAB's sums change with the thread count; pyDynamo reproduces it with `--matlab-threads N` (default: one per physical core available, as MATLAB; e.g. 64 for a Dynamo run on a whole 64-core node), measured bit-identical for 16 and 64 threads. MATLAB's own `fftn`/`ifftn` calls are reproduced bit for bit (measured for boxes 16 to 320, odd, non-power-of-two and non-cubic sizes). pyDynamo does not run MATLAB and loads no
MATLAB library: it reproduces MATLAB's numbers with FFTW 3 (measured: FFTW 3.3.8, sse2-avx build, gives the same bits as MATLAB's
own FFTW 3.3.3; other builds are not verified) and a port of fdlibm
(MATLAB's sin/cos/atan2/acos, `pydyn/fdlibm.py`).

## Names

| Name | What it is |
|---|---|
| pyDynamo | the project and this repository |
| `pydyn` | the Python package (`prototype/pydyn`) |
| `pydynamo` | the Python environment (`.envs/pydynamo`, a uv venv that `requirements/recreate.sh` builds from `requirements/pydynamo-lock.txt`), and the console command `pydynamo` |

## Requirements

- Linux x86-64. uv downloads the environment's Python (CPython 3.11.18) and the PyPI wheels of the lock file.
- An NVIDIA GPU and its driver. CuPy and its CUDA 12 libraries are wheels in the lock file.
- For the default `exact` engine, two more things:
  - a local Dynamo 1.1.514 installation (Dynamo's GPU kernels are extracted from its engine binary at run time);
  - the CUDA 8 libraries `libcufft.so.8.0` and `libcublas.so.8.0`.

  `PYDYN_DYNAMO_ROOT` and `PYDYN_CUDA8_LIB` tell the engine where they are. Without them, use `--engine fast`. It is
  pyDynamo's own CuPy engine: close to Dynamo, but not bit-identical.
- FFTW 3 for bit-identical averages: `PYDYN_FFTW_LIB` if it is set, else the system's `libfftw3.so.3`. Without FFTW,
  numpy FFTs are used, the averages are not bit-identical, and a note on stderr says so.
- For the dcp window and the viewers, an X display. Qt for Python (PySide6, in the lock file) also needs the
  system's xcb libraries.

## Where to start

- `requirements/README-env.txt` says how to build the `pydynamo` environment (`requirements/recreate.sh`) and how to check it.
- `prototype/README.md` describes usage, in short:

```bash
source .envs/pydynamo/bin/activate      # or prototype/bin/pydynamo-shell for a clean shell
dcp                                     # the dcp project window
./NAME.exe                              # after check and unfold; runs pyDynamo while the environment is active
```

- `docs/DESIGN.md` holds the design notes.

To run the tests from the repository root:

```bash
python -m pytest -m "not gpu"                                   # CPU tests; CI runs them without GPU, Dynamo, data or FFTW
PYDYN_GPU_TESTS=1 python -I -m pytest -m gpu prototype/tests    # GPU tests, opt-in, inside a GPU job
```

## License

pyDynamo is under the MIT license (see `LICENSE`). `NOTICE` gives Dynamo's license notice. pyDynamo ports parts of
Dynamo's implementation, but Dynamo itself is not included in this repository.
