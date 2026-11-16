"""Exact engine mode: Dynamo 1.1.514's own GPU arithmetic, executed step by step from Python.

The fast engine (engine.py) reproduces Dynamo's *mathematics*; its float32 results differ from Dynamo's at the 7th
digit, which flips near-ties between orientations (docs/analysis/12a, 12d section 7). This module instead executes
the *same operations* as the official engine `dynamo_compute_iteration_spp` 1.1.514, in the same order, with the same
code (sources: /programs/x86_64-linux/dynamo/1.1.514/cuda/src/dynamo_compute_iteration_spp.cu, cited `spp.cu:LINE`):

* device kernels: Dynamo's own PTX, embedded (LZ4-compressed) in the fatbin of bin/dynamo_compute_iteration_spp,
  extracted at run time (md5-checked) and loaded with cupy.RawModule. On sm_86 the official binary also runs this
  PTX through the driver JIT, so the machine code is the same (docs/analysis/12a section 0, 12d section 1.1);
* FFTs: cuFFT 8.0.44 (libcufft.so.8.0, the library Dynamo links), C2C `cufftPlan3d`, in place, through ctypes on
  CuPy memory (spp.cu:3821-3824);
* reductions: legacy cuBLAS 8.0.45 `cublasSasum` / `cublasSdot` / `cublasIsamax` (spp.cu:4182, 4192, 4207, 4423);
* host side: rotation and limit matrices in float from float-rounded cos/sin (spp.cu:3799-3800, 4011-4033,
  4060-4083), the scalars mean = sum/nnz, sqrt, cc - 1, x + subpixel, x - n/2 in float (spp.cu:4193, 4208, 4469,
  4574, 4597), the first-strict-maximum rule and Dynamo's defaults when no orientation beats -1 (spp.cu:3961-3983,
  4562).

One particle at a time, as particle::align (spp.cu:3607-4650); its orientations one at a time (batch=1, Dynamo's
loop) or B at a time with the same operations per orientation (DynamoExact, batch>1). The particle spectra
(spp.cu:3835-3905) are computed once per particle and reused for its refinement levels: Dynamo recomputes them in
every align() call from the same host volume with the same deterministic calls, so the bits are the same.
Optionally (sdot_sms) cublasSdot's summation order of a GPU model with another SM count is reproduced (SdotOrder):
Dynamo's own results depend on the GPU model through it.

Supported (raises otherwise): localnc=1, use_CC 0/1, every fsampling type but 5 (custom pfmask files; types without a
case in spp.cu:4135-4161 apply no mask, as in Dynamo), every area_search_modus, data box == template box (no padding
or binning, as the fast engine).
"""
import ctypes
import hashlib
import math
import os
import re
import struct

import numpy as np

try:
    import cupy as cp
except ImportError:                     # CPU-only environments can still import the host helpers
    cp = None

from . import angles as A

F = np.float32
DYNAMO_ROOT = os.environ.get("PYDYN_DYNAMO_ROOT", "/programs/x86_64-linux/dynamo/1.1.514")
CUDA8_LIB = os.environ.get("PYDYN_CUDA8_LIB", "/programs/x86_64-linux/cuda/8.0/lib64")
ENGINE_BINARY = os.path.join(DYNAMO_ROOT, "cuda", "bin", "dynamo_compute_iteration_spp")
# md5 of the PTX entry of the official 1.1.514 binary (md5 8d14a9a39d622ca71a697b832ee0ede9), 80,777 bytes;
# identical to benchmarks/09_engine_parity/harness/ptx/dynamo_spp_1.1.514_0.ptx
PTX_MD5 = "da35c4c571db739ad40cd0edd9703242"
D2R = 0.0174532925199432954743716805978692718781530857086181640625      # spp.cu:3800 `const double degrees2rad`
CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE = 0x29, -1, 1

# mangled names of the kernels in Dynamo's PTX (signatures as in spp.cu)
KERNELS = {
    "putblanks": "_Z9putblanksP6float2i",                                     # spp.cu:4090
    "rotate": "_Z6rotateP6float2S0_S0_S0_iPf",                                 # spp.cu:4112 (localnc=1)
    "bandpass": "_Z21aplyfourierfilter_fftP6float2ifff",                       # spp.cu:3875, 4168
    "wedge1": "_Z14applywedge_fftP6float2iff",                                 # ftype 1
    "wedge2": "_Z15applywedgex_fftP6float2iff",                                # ftype 2
    "wedge4": "_Z17apply_pyramid_fftP6float2iffff",                            # ftype 4
    "wedge6": "_Z28applywedge_tilt_z_beam_y_fftP6float2iff",                   # ftype 6
    "wedge7": "_Z28applywedge_tilt_x_beam_y_fftP6float2iff",                   # ftype 7
    "normalize_fft": "_Z13normalize_fftP6float2l",                             # spp.cu:3877, 4170
    "squared_copy": "_Z12squared_copyP6float2S0_i",                            # spp.cu:3879
    "c2f": "_Z16copy_cufft2floatP6float2Pfi",                                  # spp.cu:4179, 4188
    "nf2c": "_Z21normalize_float2cufftP6float2PfS1_fi",                        # spp.cu:4198
    "pad": "_Z3padP6float2S0_ii",                                              # spp.cu:4220-4224
    "mbc": "_Z21multiply_by_conjugateP6float2S0_S0_i",                         # spp.cu:4297-4301
    "roseman": "_Z16assemble_rosemanP6float2S0_S0_iff",                        # spp.cu:4328
    "limits": "_Z22limit_shifts_ellipsoidP6float2fffiiiiiPf",                  # spp.cu:4355-4414
    "plus_one": "_Z25copy_cufft2float_plus_oneP6float2Pfi",                    # spp.cu:4420
}


# ----------------------------------------------------------------------------------------------- Dynamo's PTX
def _lz4_block(src, start, end):
    """LZ4 block decoder (the fatbin PTX entry is LZ4-compressed, entry flag 0x2000)."""
    out = bytearray()
    i = start
    while i < end:
        tok = src[i]
        i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                x = src[i]
                i += 1
                lit += x
                if x != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= end:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                x = src[i]
                i += 1
                ml += x
                if x != 255:
                    break
        ml += 4
        for _ in range(ml):
            out.append(out[-off])
    return bytes(out)


def extract_ptx(binary=ENGINE_BINARY):
    """The PTX entries (kind 1) of the binary's fatbins, decoded (benchmarks/.../harness/tools/extract_ptx.py)."""
    b = open(binary, "rb").read()
    magic = struct.pack("<I", 0xBA55ED50)
    found = []
    i = b.find(magic)
    while i >= 0:
        _, _, hsz, fsz = struct.unpack_from("<IHHQ", b, i)
        p, end = i + hsz, i + hsz + fsz
        while p < end:
            kind, _, esz, psz = struct.unpack_from("<HHIQ", b, p)
            if kind == 1:
                hdr = b[p:p + esz]
                usize = struct.unpack_from("<Q", hdr, 56)[0]
                found.append(_lz4_block(b, p + esz, p + esz + psz)[:usize])
            p += esz + psz
        i = b.find(magic, end)
    return found


def _cache_dirs():
    """Cache folders in order of preference: $PYDYN_CACHE_DIR (only that one), else ~/.cache/pydyn, then a per-user
    folder under the system temp dir (used when ~/.cache is not writable, e.g. a read-only home)."""
    if os.environ.get("PYDYN_CACHE_DIR"):
        return [os.environ["PYDYN_CACHE_DIR"]]
    import tempfile
    home = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "pydyn")
    return [home, os.path.join(tempfile.gettempdir(), "pydyn-cache-%s" % os.getuid())]


def _cache_dir():
    return _cache_dirs()[0]


def _md5_of(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _cache_store(name, data, md5):
    """Path of cache file `name` holding `data` (bytes) with the given md5: an existing file with the right md5 is
    reused; a missing or corrupt one (md5 differs) is rewritten; a folder that cannot be written is skipped."""
    last = None
    for d in _cache_dirs():
        p = os.path.join(d, name)
        if _md5_of(p) == md5:
            return p
        try:
            os.makedirs(d, exist_ok=True)
            tmp = f"{p}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
            return p
        except OSError as e:
            last = e
    raise RuntimeError(f"cannot write the PTX cache in any of {_cache_dirs()} ({last}); set $PYDYN_CACHE_DIR to a "
                       "writable folder")


def ptx_file():
    """Path of Dynamo 1.1.514's engine PTX: $PYDYN_DYNAMO_PTX, or extracted from the installed binary into the cache
    (see _cache_dirs). A cached file whose md5 is not that of the 1.1.514 engine is extracted again."""
    p = os.environ.get("PYDYN_DYNAMO_PTX")
    if p:
        if _md5_of(p) != PTX_MD5:
            raise RuntimeError(f"{p}: md5 differs from Dynamo 1.1.514's engine PTX {PTX_MD5}")
        return p
    name = f"dynamo_spp_1.1.514_{PTX_MD5}.ptx"
    for d in _cache_dirs():
        if _md5_of(os.path.join(d, name)) == PTX_MD5:
            return os.path.join(d, name)
    txt = [t for t in extract_ptx() if hashlib.md5(t).hexdigest() == PTX_MD5]
    if not txt:
        raise RuntimeError(f"{ENGINE_BINARY}: no PTX entry with md5 {PTX_MD5} (not Dynamo 1.1.514?)")
    return _cache_store(name, txt[0], PTX_MD5)


def preflight():
    """Everything the exact engine needs from the Dynamo installation, checked before any project work: the engine
    binary, the CUDA 8 cuFFT and cuBLAS libraries, and the PTX extraction. Raises RuntimeError with the list of
    problems and the way out (the batched `fast` engine needs none of them)."""
    problems = []
    if not os.path.isfile(ENGINE_BINARY):
        problems.append(f"Dynamo engine binary not found: {ENGINE_BINARY}")
    for so in ("libcufft.so.8.0", "libcublas.so.8.0"):
        if not os.path.isfile(os.path.join(CUDA8_LIB, so)):
            problems.append(f"CUDA 8 library not found: {os.path.join(CUDA8_LIB, so)}")
    if not problems:
        try:
            ptx_file()
        except Exception as e:
            problems.append(f"PTX extraction from {ENGINE_BINARY} failed: {e}")
    if problems:
        raise RuntimeError("the exact engine (Dynamo 1.1.514's own kernels) is not available:\n  "
                           + "\n  ".join(problems)
                           + "\nInstall/point to Dynamo 1.1.514 ($PYDYN_DYNAMO_ROOT, $PYDYN_CUDA8_LIB), or run with "
                             "--engine fast (pyDynamo's batched engine; float32 differences at the 7th digit can flip "
                             "near-ties against Dynamo).")


# pointer parameters of the kernels (from the signatures in spp.cu; normalize_fft's `long n` is .u64 but no pointer)
POINTER_PARAMS = {"putblanks": (0,), "rotate": (0, 1, 2, 3, 5), "bandpass": (0,), "wedge1": (0,), "wedge2": (0,),
                  "wedge4": (0,), "wedge6": (0,), "wedge7": (0,), "normalize_fft": (0,), "c2f": (0, 1),
                  "nf2c": (0, 1, 2), "pad": (0, 1), "mbc": (0, 1, 2), "roseman": (0, 1, 2), "limits": (0, 9),
                  "plus_one": (0, 1)}
BLAS_STREAMS = 4                         # concurrent cuBLAS 8 handles in the batched path
# per-volume float parameters of the batched variants (kernel key -> parameter indices); see batched_ptx()
PER_VOLUME_FLOATS = {"nf2c": (3,), "roseman": (4, 5)}
BATCHED = ("putblanks", "rotate", "bandpass", "wedge1", "wedge2", "wedge4", "wedge6", "wedge7", "normalize_fft",
           "c2f", "nf2c", "pad", "mbc", "roseman", "limits", "plus_one")


def _entry_text(ptx, mangled):
    """The text of kernel `mangled` in the PTX module, from `.visible .entry` to its closing brace."""
    i = ptx.index(".visible .entry " + mangled + "(")
    ends = [ptx.find(t, i + 1) for t in ("\n.visible .entry", "\n.entry", "\n.visible .func", "\n.func", "\n.global",
                                         "\n.const")]
    ends = [e for e in ends if e > 0]
    ent = ptx[i:min(ends) if ends else len(ptx)].rstrip()
    assert ent.endswith("\n}"), mangled
    return ent


_SWAP_XZ = re.compile(r"%(tid|ctaid)\.x\b")


def _swap_xz(line):
    """%tid.x <-> %ctaid.x: in Dynamo's kernels voxel (x, y, z) is (blockIdx.x, blockIdx.y, threadIdx.x), so the 64
    threads of a block touch addresses n*n*8 bytes apart. Swapped, thread x of block (z, y) computes voxel (x, y, z):
    the same voxels, each with the same instructions, but consecutive threads access consecutive addresses. Every
    kernel of BATCHED computes each voxel independently (no shared memory, no reduction, no other special
    register: docs/analysis/12a section 2), so which thread computes a voxel cannot change its value."""
    return _SWAP_XZ.sub(lambda m: "%ctaid.x" if m.group(1) == "tid" else "%tid.x", line)


def _batched_entry(ptx, mangled, ptrs, per_volume=(), coalesce=True):
    """A copy of Dynamo's kernel `mangled` that processes volume blockIdx.z of a batch.

    Dynamo's kernels address one volume with (blockIdx.x, blockIdx.y, threadIdx.x) (spp.cu:94-97). The copy keeps
    every instruction of the original and only adds integer address arithmetic at the parameter loads:
    * every pointer parameter k gets a byte stride parameter `<name>_bz_stride_k` (appended in order); its loaded
      address becomes  address + blockIdx.z * stride  (stride 0 = shared by the batch);
    * a float parameter listed in `per_volume` becomes a pointer to an array; the value is loaded from
      element blockIdx.z instead of from the parameter space (same register, same use).
    The floating-point instruction sequence is untouched (docs/analysis/12a section 6, "Keeping the speed").
    coalesce: also swap threadIdx.x and blockIdx.x (_swap_xz); launch grid (n, n, B), block n, as before."""
    lines = _entry_text(ptx, mangled).split("\n")
    new = mangled + "_bz"
    close = lines.index(")")
    params = []
    for ln in lines[1:close]:
        typ, name = ln.strip().rstrip(",").split()[1:3]
        params.append((typ, name))
    k_of = {name: k for k, (_, name) in enumerate(params)}
    assert all(params[k][0] == ".u64" for k in ptrs), mangled
    hdr = [f".visible .entry {new}("]
    plist = []
    for k, (typ, name) in enumerate(params):
        plist.append(f".param {'.u64' if k in per_volume else typ} {new}_param_{k}")
    plist += [f".param .u64 {new}_stride_{k}" for k in ptrs]
    hdr += [p + ("," if q < len(plist) - 1 else "") for q, p in enumerate(plist)]
    body = lines[close:]
    last_reg = max(q for q, ln in enumerate(body) if ln.startswith(".reg"))
    out = body[:last_reg + 1] + [".reg .b32 %bzr<1>;", ".reg .b64 %bzd<40>;",
                                 "mov.u32 %bzr0, %ctaid.z;", "cvt.u64.u32 %bzd0, %bzr0;"]
    nload = 0
    for ln in body[last_reg + 1:]:
        st = ln.strip()
        if st.startswith("ld.param"):
            op, rest = st.split(None, 1)
            reg, ref = [x.strip() for x in rest.rstrip(";").split(",")]
            k = k_of[ref.strip("[]")]
            nload += 1
            if k in per_volume:
                assert op == "ld.param.f32", (mangled, st)
                a, b = f"%bzd{1 + 2 * k}", f"%bzd{2 + 2 * k}"
                out += [f"ld.param.u64 {a}, [{new}_param_{k}];", f"cvta.to.global.u64 {a}, {a};",
                        f"shl.b64 {b}, %bzd0, 2;", f"add.s64 {a}, {a}, {b};", f"ld.global.f32 {reg}, [{a}];"]
                continue
            out.append(f"{op} {reg}, [{new}_param_{k}];")
            if k in ptrs:
                assert op == "ld.param.u64", (mangled, st)
                a = f"%bzd{1 + 2 * k}"
                out += [f"ld.param.u64 {a}, [{new}_stride_{k}];", f"mad.lo.s64 {reg}, %bzd0, {a}, {reg};"]
            continue
        assert "_param_" not in st, (mangled, st)
        assert not re.search(r"%(n?tid|n?ctaid)\.[yz]|%n(tid|ctaid)\.x", st.replace("%ctaid.y", "")), (mangled, st)
        out.append(_swap_xz(ln) if coalesce else ln)
    assert nload > 0 and len(params) <= 19, mangled
    return "\n".join(hdr + out), ptrs


def batched_ptx(ptx_text):
    """Dynamo's PTX plus a batched copy (suffix _bz) of every kernel in BATCHED; returns (text, pointer params)."""
    extra, ptrs = [], {}
    for key in BATCHED:
        e, ptrs[key] = _batched_entry(ptx_text, KERNELS[key], POINTER_PARAMS[key], PER_VOLUME_FLOATS.get(key, ()))
        extra.append(e)
    return ptx_text + "\n\n" + "\n\n".join(extra) + "\n", ptrs


def batched_ptx_file():
    src = ptx_file()
    txt, ptrs = batched_ptx(open(src).read())
    data = txt.encode()
    h = hashlib.md5(data).hexdigest()
    return _cache_store(f"dynamo_spp_1.1.514_{PTX_MD5}_bz_{h}.ptx", data, h), ptrs


# ----------------------------------------------------------------------------------------------- libraries
class _Libs:
    """Per-process (per-device) handles: Dynamo's PTX module, cuFFT 8, legacy cuBLAS 8."""
    _by_device = {}

    def __init__(self):
        cp.zeros(1)                                          # make the device's primary context current first
        self.module = cp.RawModule(path=ptx_file())
        self.k = {key: self.module.get_function(name) for key, name in KERNELS.items()}
        self.cufft = ctypes.CDLL(os.path.join(CUDA8_LIB, "libcufft.so.8.0"))
        self.cufft.cufftPlan3d.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int]
        self.cufft.cufftExecC2C.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.cufft.cufftDestroy.argtypes = [ctypes.c_int]
        b = ctypes.CDLL(os.path.join(CUDA8_LIB, "libcublas.so.8.0"))
        b.cublasSasum.restype = ctypes.c_float
        b.cublasSasum.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        b.cublasSdot.restype = ctypes.c_float
        b.cublasSdot.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        b.cublasIsamax.restype = ctypes.c_int
        b.cublasIsamax.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        b.cublasGetError.restype = ctypes.c_int
        r = b.cublasInit()                                   # spp.cu:3858
        if r != 0:
            raise RuntimeError(f"cublasInit (cuBLAS 8) failed: {r}")
        self.cublas = b
        self.kb = None                                       # batched kernels, loaded on first use
        self.h2 = None                                       # cuBLAS 8 v2 handle (device pointer mode)

    def batched(self):
        """Batched copies of Dynamo's kernels (batched_ptx) and a cuBLAS 8 v2 handle whose results stay on the GPU."""
        if self.kb is None:
            path, self.kb_ptrs = batched_ptx_file()
            self.module_b = cp.RawModule(path=path)
            self.kb = {key: self.module_b.get_function(KERNELS[key] + "_bz") for key in BATCHED}
            b = self.cublas
            h = ctypes.c_void_p()
            r = b.cublasCreate_v2(ctypes.byref(h))
            if r != 0:
                raise RuntimeError(f"cublasCreate_v2 (cuBLAS 8) failed: {r}")
            b.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
            r = b.cublasSetPointerMode_v2(h, 1)              # CUBLAS_POINTER_MODE_DEVICE
            if r != 0:
                raise RuntimeError(f"cublasSetPointerMode_v2 failed: {r}")
            for nm in ("cublasSasum_v2", "cublasIsamax_v2"):
                getattr(b, nm).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            b.cublasSdot_v2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_void_p]
            self.h2 = h
            # BLAS_STREAMS more handles on their own streams: the per-volume reductions of a batch are independent
            # calls on independent data, so they may run concurrently (fork/join with events on the legacy stream)
            b.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.streams, self.h2s = [], []
            for _ in range(BLAS_STREAMS):
                st = cp.cuda.Stream(non_blocking=True)
                hs = ctypes.c_void_p()
                if b.cublasCreate_v2(ctypes.byref(hs)) or b.cublasSetPointerMode_v2(hs, 1) or \
                        b.cublasSetStream_v2(hs, ctypes.c_void_p(st.ptr)):
                    raise RuntimeError("cuBLAS 8 v2 stream handle setup failed")
                self.streams.append(st)
                self.h2s.append(hs)
            self.cufft.cufftPlanMany.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int]
        return self.kb

    def plan_many(self, n, B):
        """cufftPlanMany: B contiguous n^3 C2C volumes (same transform per volume as cufftPlan3d, see tests)."""
        h = ctypes.c_int()
        dims = (ctypes.c_int * 3)(n, n, n)
        N = n ** 3
        r = self.cufft.cufftPlanMany(ctypes.byref(h), 3, dims, None, 1, N, None, 1, N, CUFFT_C2C, B)
        if r != 0:
            raise RuntimeError(f"cufftPlanMany (cuFFT 8) failed: {r}")
        return h.value

    def fft_ptr(self, plan, ptr, direction):
        r = self.cufft.cufftExecC2C(plan, ptr, ptr, direction)
        if r != 0:
            raise RuntimeError(f"cufftExecC2C (cuFFT 8) failed: {r}")

    def asum_dev(self, n, x_ptr, out_ptr, h=None):
        r = self.cublas.cublasSasum_v2(h or self.h2, n, x_ptr, 1, out_ptr)
        if r != 0:
            raise RuntimeError(f"cublasSasum_v2 failed: {r}")

    def dot_dev(self, n, x_ptr, y_ptr, out_ptr, h=None):
        r = self.cublas.cublasSdot_v2(h or self.h2, n, x_ptr, 1, y_ptr, 1, out_ptr)
        if r != 0:
            raise RuntimeError(f"cublasSdot_v2 failed: {r}")

    def iamax_dev(self, n, x_ptr, out_ptr, h=None):
        r = self.cublas.cublasIsamax_v2(h or self.h2, n, x_ptr, 1, out_ptr)
        if r != 0:
            raise RuntimeError(f"cublasIsamax_v2 failed: {r}")

    def fork(self):
        """The BLAS streams wait for everything queued so far on the legacy default stream."""
        ev = cp.cuda.Event(disable_timing=True)
        ev.record(cp.cuda.Stream.null)
        for st in self.streams:
            st.wait_event(ev)

    def join(self):
        """The legacy default stream waits for everything queued on the BLAS streams."""
        for st in self.streams:
            ev = cp.cuda.Event(disable_timing=True)
            ev.record(st)
            cp.cuda.Stream.null.wait_event(ev)

    @classmethod
    def get(cls):
        dev = cp.cuda.runtime.getDevice()
        if dev not in cls._by_device:
            cls._by_device[dev] = cls()
        return cls._by_device[dev]

    def plan3d(self, n):
        h = ctypes.c_int()
        r = self.cufft.cufftPlan3d(ctypes.byref(h), n, n, n, CUFFT_C2C)          # spp.cu:3821, 3824
        if r != 0:
            raise RuntimeError(f"cufftPlan3d (cuFFT 8) failed: {r}")
        return h.value

    def fft(self, plan, a, direction):                                          # cufftExecC2C(plan, a, a, dir)
        r = self.cufft.cufftExecC2C(plan, a.data.ptr, a.data.ptr, direction)
        if r != 0:
            raise RuntimeError(f"cufftExecC2C (cuFFT 8) failed: {r}")

    def asum(self, x):
        return F(self.cublas.cublasSasum(x.size, x.data.ptr, 1))

    def dot(self, x, y):
        return F(self.cublas.cublasSdot(x.size, x.data.ptr, 1, y.data.ptr, 1))

    def iamax(self, x):                                                         # 1-based (FORTRAN convention)
        return int(self.cublas.cublasIsamax(x.size, x.data.ptr, 1))


# ----------------------------------------------------------------------------------------------- other GPU models
# cuBLAS 8 chooses the grid of cublasSdot's dot_kernel in host code from the device: G = 12 blocks of 128 threads per
# SM for n = 64^3 on sm_86 (A40: 84 SMs, G = 1008, docs/analysis/12a section 5.1). The summation order, hence sigma and
# the mean of every rotated template, therefore depends on the GPU model, and so do Dynamo's tables (the user's RTX
# A5000 runs differ from the A40 runs in near-ties). SdotOrder runs cuBLAS 8's two dot kernels, as read from cuBLAS
# 8.0.45's PTX (12a section 2; CPU model benchmarks/09_engine_parity/fftcc/sdot_model.py), with a chosen grid, so that
# the exact engine can reproduce a GPU model with another SM count. Intrinsics only (__fmaf_rn, __fadd_rn: no
# contraction); NVRTC's -ftz (CuPy) only matters for denormal operands or partial sums. At construction the
# emulation is checked bit for bit against cublasSdot on this GPU with this GPU's own grid.
BLOCKS_PER_SM = 12
_SDOT_SRC = r"""
extern "C" __global__ void sdot_stage1(const float* x, const float* y, float* part, const long long n, const int G)
{   // dot_kernel<float,float,float,128,...>: grid-stride fma chain per thread, then the shared-memory tree
    __shared__ float s[128];
    int t = threadIdx.x, g = blockIdx.x;
    float acc = 0.f;
    for (long long i = (long long)g * 128 + t; i < n; i += (long long)G * 128) acc = __fmaf_rn(x[i], y[i], acc);
    s[t] = acc;
    __syncthreads();
    for (int k = 64; k > 0; k >>= 1) {
        float v = (t < k) ? __fadd_rn(s[t], s[t + k]) : 0.f;
        __syncthreads();
        if (t < k) s[t] = v;
        __syncthreads();
    }
    if (t == 0) part[g] = s[0];
}
extern "C" __global__ void sdot_stage2(const float* part, float* out, const int G)
{   // reduce_1Block_kernel<float,float,float,128,7>, evaluated by one thread in the kernel's order
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s[192], tmp[128];
    for (int t = 0; t < 192; t++) s[t] = 0.f;
    for (int t = 0; t < 128; t++) {
        float acc = t < G ? part[t] : 0.f;
        for (int i = t + 128; i < G; i += 128) acc = __fadd_rn(acc, part[i]);
        s[t] = acc;
    }
    if (G >= 33) {
        if (G > 64) { for (int t = 0; t < 128; t++) tmp[t] = __fadd_rn(s[t], s[t + 64]); for (int t = 0; t < 128; t++) s[t] = tmp[t]; }
        if (G > 32) { for (int t = 0; t < 128; t++) tmp[t] = __fadd_rn(s[t], s[t + 32]); for (int t = 0; t < 128; t++) s[t] = tmp[t]; }
    }
    float u[4];
    for (int t = 0; t < 4; t++) { float acc = 0.f; for (int j = t; j < 32; j += 4) acc = __fadd_rn(acc, s[j]); u[t] = acc; }
    float o = __fadd_rn(0.f, u[0]);
    o = __fadd_rn(o, u[1]); o = __fadd_rn(o, u[2]); o = __fadd_rn(o, u[3]);
    *out = o;
}
"""


class SdotOrder:
    """cublasSdot (unit strides, n >= 33*128) in the order cuBLAS 8 uses on a GPU with `sms` multiprocessors."""

    def __init__(self, sms, libs, n_check=262144):
        self.G = int(BLOCKS_PER_SM * sms)
        if self.G < 33:
            raise ValueError("SdotOrder: grids below 33 blocks are not modelled")
        mod = cp.RawModule(code=_SDOT_SRC)
        self.k1, self.k2 = mod.get_function("sdot_stage1"), mod.get_function("sdot_stage2")
        self.part = cp.zeros(max(self.G, 2048), cp.float32)
        self.one = cp.zeros(1, cp.float32)
        own = BLOCKS_PER_SM * cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["multiProcessorCount"]
        rs = np.random.RandomState(7)
        for k in range(4):                         # this GPU's own grid must give cublasSdot's bits
            x = cp.asarray(rs.standard_normal(n_check).astype(np.float32))
            y = x if k % 2 else cp.asarray((rs.random_sample(n_check) < 0.3).astype(np.float32))
            self._run(n_check, x.data.ptr, y.data.ptr, self.one.data.ptr, own)
            if self.one.get()[0].view(np.int32) != libs.dot(x, y).view(np.int32):
                raise RuntimeError(f"SdotOrder does not reproduce cublasSdot on this GPU (grid {own}): "
                                   "cannot emulate another GPU model")

    def _run(self, n, x_ptr, y_ptr, out_ptr, G):
        mp = lambda p, b: cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(p, b, None), 0)
        x = cp.ndarray((n,), cp.float32, mp(x_ptr, 4 * n))
        y = cp.ndarray((n,), cp.float32, mp(y_ptr, 4 * n))
        o = cp.ndarray((1,), cp.float32, mp(out_ptr, 4))
        self.k1((G,), (128,), (x, y, self.part, np.int64(n), np.int32(G)))
        self.k2((1,), (1,), (self.part, o, np.int32(G)))

    def dot_dev(self, n, x_ptr, y_ptr, out_ptr):
        self._run(n, x_ptr, y_ptr, out_ptr, self.G)

    def dot(self, x, y):
        self._run(x.size, x.data.ptr, y.data.ptr, self.one.data.ptr, self.G)
        return F(self.one.get()[0])


# ----------------------------------------------------------------------------------------------- host arithmetic
_trig_cache = {}


def _cos_sin(a):
    """float(cos(a*degrees2rad)), float(sin(...)) for a float angle a: double product and libm double cos/sin, as
    the C++ code `cotd=cos(tdrot*degrees2rad)` with `float cotd` and `float tdrot` (spp.cu:3797-3800, 4063-4068)."""
    key = F(a).view(np.int32).item()
    v = _trig_cache.get(key)
    if v is None:
        x = float(F(a)) * D2R
        v = (F(math.cos(x)), F(math.sin(x)))
        if len(_trig_cache) < 1_000_000:
            _trig_cache[key] = v
    return v


def matrices_float(angles):
    """h_rotmatrix (spp.cu:4075-4083) / h_rotmatrix_limits (spp.cu:4025-4033) for [K, 3] angles: float cos/sin and
    nine float expressions evaluated left to right with one rounding per operation (x86-64 SSE, no FMA)."""
    ang = np.asarray(angles, dtype=F).reshape(-1, 3)
    cs = np.array([[_cos_sin(a) for a in row] for row in ang], dtype=F).reshape(-1, 3, 2)
    cotd, sitd = cs[:, 0, 0], cs[:, 0, 1]
    coti, siti = cs[:, 1, 0], cs[:, 1, 1]
    cona, sina = cs[:, 2, 0], cs[:, 2, 1]
    m = np.empty((len(ang), 9), dtype=F)
    m[:, 0] = cotd * cona - (sitd * coti) * sina
    m[:, 1] = (-cona) * sitd - (cotd * coti) * sina
    m[:, 2] = sina * siti
    m[:, 3] = cotd * sina + (cona * sitd) * coti
    m[:, 4] = (cotd * cona) * coti - sitd * sina
    m[:, 5] = (-cona) * siti
    m[:, 6] = sitd * siti
    m[:, 7] = cotd * siti
    m[:, 8] = coti
    return m


def rawpeak2space(rawpeak, nsmall, nlarge):                                   # spp.cu:3433-3441
    nlf = F(nlarge)
    xs = rawpeak + nlarge // 2 - nlarge * int(math.floor(F(rawpeak + nlarge // 2) / nlf))
    xflip = nlarge - xs - 1
    xm = xflip + (nlarge + nsmall) // 2 + 1
    return xm - nlarge * int(math.floor(F(xm) / nlf))


def spacepeak2raw(space, nsmall, nlarge):                                     # spp.cu:3442-3448
    s = nlarge - space + nsmall // 2
    return s - nlarge * int(math.floor(F(s) / F(nlarge)))


def space2memory(x, y, z, nr, nd):                                            # spp.cu:3449-3458 (never -1)
    return nd * nd * spacepeak2raw(z, nr, nd) + nd * spacepeak2raw(y, nr, nd) + spacepeak2raw(x, nr, nd)


def _cdiv(a, b):
    """C integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def rawpeak2space_v(r, n):
    """rawpeak2space for arrays, nsmall == nlarge == n: the floor() arguments are non-negative small integers whose
    float quotient never rounds across an integer, so floor(float(a)/float(n)) == a // n (checked in the tests)."""
    xs = r + n // 2 - n * ((r + n // 2) // n)
    xm = (n - xs - 1) + (n + n) // 2 + 1
    return xm - n * (xm // n)


def spacepeak2raw_v(s, n):
    t = n - s + n // 2
    return t - n * (t // n)


# ----------------------------------------------------------------------------------------------- the engine
class DynamoExact:
    """particle::refinement_process + particle::align of dynamo_compute_iteration_spp 1.1.514 (one GPU).

    batch = 1: one orientation at a time, every call exactly as in spp.cu (the reference implementation).
    batch = B > 1: B orientations of the same particle and level per step. Each orientation still gets exactly
    Dynamo's operations on its own buffers, only interleaved with the other orientations of the batch:
    * kernels: batched copies of Dynamo's PTX kernels (batched_ptx: same instructions, volume index blockIdx.z);
    * FFTs: cuFFT 8 cufftPlanMany over B contiguous volumes (fft_many=True) or B cufftPlan3d calls (False);
    * reductions: cuBLAS 8 Sasum/Sdot/Isamax through the v2 API in device pointer mode, one call per volume (spread
      over BLAS_STREAMS handles and streams, joined before the results are used), so the results stay on the GPU
      until the batch needs them on the host;
    * host scalars (mean = sum/nnz, sqrt, cc - 1, ...) as float32 NumPy array operations (IEEE, one rounding each).
    That each of these gives the bits of the per-orientation calls is checked by tests/test_exact.py and by the
    harness (benchmarks/09_engine_parity/exact).
    """

    def __init__(self, template, mask, card, powf=True, batch=1, fft_many=True, sdot_sms=None):
        from .engine import subpixel_spline
        self._spline = subpixel_spline
        self.powf = powf
        self.n = n = int(template.shape[0])
        self.N = n ** 3
        self.L = _Libs.get()
        self.k = self.L.k
        # cublasSdot of this GPU (default), or in the order of a GPU with `sdot_sms` multiprocessors (SdotOrder)
        self.sdot = self.L if not sdot_sms else SdotOrder(sdot_sms, self.L, n_check=self.N)
        self.sdot_sms = sdot_sms
        self.plan = self.L.plan3d(n)                                            # spp.cu:3821 (data box)
        self.plan_small = self.L.plan3d(n)                                      # spp.cu:3824 (template box)
        self.grid, self.block = (n, n), (n,)                                    # spp.cu:3771-3786
        z = lambda: cp.zeros(self.N, dtype=cp.complex64)
        self.d_rr, self.d_rm, self.d_prr, self.d_prm = z(), z(), z(), z()
        self.d_RD, self.d_DMDM, self.d_D2M = z(), z(), z()
        self.d_data, self.d_data_square = z(), z()
        self.f_rr = cp.zeros(self.N, dtype=cp.float32)                          # cublasAlloc, spp.cu:3859-3861
        self.f_rm = cp.zeros(self.N, dtype=cp.float32)
        self.f_ccf = cp.zeros(self.N, dtype=cp.float32)
        self.d_rot = cp.zeros(9, dtype=cp.float32)
        self.d_lim = cp.zeros(9, dtype=cp.float32)
        self.set_template(template, mask)
        self.card = card
        self.batch = max(1, int(batch))
        self.fft_many = bool(fft_many)
        self._bB = 0                                                            # allocated batch capacity
        self._plans_many = {}
        self.record = None                     # list: per-orientation trace (harness comparisons)
        self.counts = []

    def close(self):
        for p in [self.plan, self.plan_small] + list(self._plans_many.values()):
            if p is not None:
                self.L.cufft.cufftDestroy(p)
        self.plan = self.plan_small = None
        self._plans_many = {}

    # -- helpers
    def _kern(self, name, *args):
        self.k[name](self.grid, self.block, args)

    @staticmethod
    def _cplx(vol):
        """cufftComplexVolume::import (spp.cu:2716-2720): .x = value, .y = 0."""
        h = np.zeros(int(np.prod(vol.shape)), dtype=np.complex64)
        h.real = np.ascontiguousarray(vol, dtype=np.float32).ravel()
        return cp.asarray(h)

    def set_template(self, template, mask=None):
        self.d_reference = self._cplx(template)
        if mask is not None:
            self.d_mask = self._cplx(mask)

    def _constants(self):
        """Card-derived floats of align() (spp.cu:3839-3851); card fields are float in Dynamo (gpuAngles.h)."""
        c = self.card
        if int(getattr(c, "localnc", 1)) != 1:
            raise NotImplementedError("exact engine: only localnc=1 (spp.cu:4100-4112)")
        if int(c.use_CC) not in (0, 1):
            raise NotImplementedError("exact engine: use_CC=2 (rotated tfmask, spp.cu:4250-4292) is not implemented")
        s = F(c.shrink)
        low_d, high_d, win_d = s * F(c.low), s * F(c.high), F(3) * s            # spp.cu:3839-3841
        fr = F(self.n) / F(self.n)                                              # factor_reference_data, 3842
        self.low_d, self.high_d, self.win_d = low_d, high_d, win_d
        self.low_r, self.high_r, self.win_r = low_d * fr, high_d * fr, win_d * fr
        a = [F(x) for x in c.area_search]
        self.sax, self.say, self.saz = a[0] * s, a[1] * s, a[2] * s             # spp.cu:3847-3849
        self.shrink = s

    # -- spp.cu:3739-3905: particle transfer, band-pass, D^ and (D^2)^
    def load_particle(self, vol):
        n = self.n
        self._constants()
        self.d_data = self._cplx(vol)                                           # 3739
        self.L.fft(self.plan, self.d_data, CUFFT_FORWARD)                       # 3835
        self._kern("bandpass", self.d_data, np.int32(n), self.high_d, self.low_d, self.win_d)   # 3875
        self.L.fft(self.plan, self.d_data, CUFFT_INVERSE)                       # 3876
        self._kern("normalize_fft", self.d_data, np.int64(n))                   # 3877
        self._kern("squared_copy", self.d_data, self.d_data_square, np.int32(n))  # 3879
        self.L.fft(self.plan, self.d_data, CUFFT_FORWARD)                       # 3904
        self.L.fft(self.plan, self.d_data_square, CUFFT_FORWARD)                # 3905

    def _wedge_args(self, ftype, wedge):
        """applywedge*_fft of fsampling type `ftype` (spp.cu:4133-4161); None = no kernel (types 0, 3)."""
        if self.card.use_CC <= 0:
            return None
        return {1: ("wedge1", (wedge[13], wedge[14])), 2: ("wedge2", (wedge[15], wedge[16])),
                4: ("wedge4", (wedge[13], wedge[14], wedge[15], wedge[16])),
                6: ("wedge6", (wedge[17], wedge[18])), 7: ("wedge7", (wedge[17], wedge[18]))}.get(ftype)

    # -- spp.cu:4058-4560: one orientation (batch = 1)
    def _orientation(self, mat, ftype, wedge, origin):
        n, N, L, kern = self.n, self.N, self.L, self._kern
        self.d_rot.set(mat)                                                     # 4086
        kern("putblanks", self.d_rr, np.int32(n))                               # 4090
        kern("putblanks", self.d_rm, np.int32(n))                               # 4091
        kern("rotate", self.d_reference, self.d_rr, self.d_mask, self.d_rm, np.int32(n), self.d_rot)   # 4112
        L.fft(self.plan_small, self.d_rr, CUFFT_FORWARD)                        # 4119
        w = self._wedge_args(ftype, wedge)                                      # 4133-4161
        if w is not None:
            kern(w[0], self.d_rr, np.int32(n), *w[1])
        kern("bandpass", self.d_rr, np.int32(n), self.high_r, self.low_r, self.win_r)   # 4168
        L.fft(self.plan_small, self.d_rr, CUFFT_INVERSE)                        # 4169
        kern("normalize_fft", self.d_rr, np.int64(n))                           # 4170
        kern("c2f", self.d_rm, self.f_rm, np.int32(n))                          # 4179
        nnz = L.asum(self.f_rm)                                                 # 4182
        kern("c2f", self.d_rr, self.f_rr, np.int32(n))                          # 4188
        ssum = self.sdot.dot(self.f_rr, self.f_rm)                              # 4192
        mean = F(ssum / nnz)                                                    # 4193 (float / float)
        kern("nf2c", self.d_rr, self.f_rr, self.f_rm, mean, np.int32(n))        # 4198
        norm = F(math.sqrt(self.sdot.dot(self.f_rr, self.f_rr)))                # 4207-4208 (sqrt correctly rounded)
        kern("putblanks", self.d_prr, np.int32(n))                              # 4219
        kern("pad", self.d_rr, self.d_prr, np.int32(n), np.int32(n))            # 4220
        kern("putblanks", self.d_prm, np.int32(n))                              # 4223
        kern("pad", self.d_rm, self.d_prm, np.int32(n), np.int32(n))            # 4224
        L.fft(self.plan, self.d_prm, CUFFT_FORWARD)                             # 4238
        L.fft(self.plan, self.d_prr, CUFFT_FORWARD)                             # 4240
        kern("mbc", self.d_prr, self.d_data, self.d_RD, np.int32(n))            # 4297
        kern("mbc", self.d_prm, self.d_data, self.d_DMDM, np.int32(n))          # 4300
        kern("mbc", self.d_prm, self.d_data_square, self.d_D2M, np.int32(n))    # 4301
        L.fft(self.plan, self.d_RD, CUFFT_INVERSE)                              # 4309
        L.fft(self.plan, self.d_D2M, CUFFT_INVERSE)                             # 4315
        L.fft(self.plan, self.d_DMDM, CUFFT_INVERSE)                            # 4316
        kern("roseman", self.d_RD, self.d_DMDM, self.d_D2M, np.int32(n), nnz, norm)   # 4328
        if origin is not None:                                                  # 4355-4414
            kern("limits", self.d_RD, self.sax, self.say, self.saz, np.int32(origin[0]), np.int32(origin[1]),
                 np.int32(origin[2]), np.int32(n), np.int32(n), self.d_lim)
        kern("plus_one", self.d_RD, self.f_ccf, np.int32(n))                    # 4420
        pos = L.iamax(self.f_ccf) - 1                                           # 4423, 4441
        # copy_float2cufft_minus_one (4443) only rewrites d_RD, which the next orientation overwrites: no output
        rz = pos // (n * n)                                                     # 4446-4448
        ry = (pos - n * n * rz) // n
        rx = pos - n * n * rz - n * ry
        bx, by, bz = rawpeak2space(rx, n, n), rawpeak2space(ry, n, n), rawpeak2space(rz, n, n)   # 4450-4456
        idx = [pos] + [space2memory(x, y, z, n, n) for (x, y, z) in               # 4493-4498
                       ((bx - 1, by, bz), (bx + 1, by, bz), (bx, by - 1, bz), (bx, by + 1, bz),
                        (bx, by, bz - 1), (bx, by, bz + 1))]
        v = self.f_ccf[cp.asarray(np.array(idx, dtype=np.int64))].get()        # 4466, 4507-4512 (reads only)
        cc = F(v[0]) - F(1)                                                     # 4469
        nb = np.array([F(x) - F(1) for x in v[1:]], dtype=F)                    # 4520-4525 (x-, x+, y-, y+, z-, z+)
        return cc, (bx, by, bz), nb, (nnz, ssum, mean, norm)

    # -- the same for B orientations at once (batch > 1)
    def _buffers(self, B):
        if self._bB < B:
            self._bufs = None
            cp.get_default_memory_pool().free_all_blocks()
            c = lambda: cp.zeros((B, self.N), dtype=cp.complex64)
            f = lambda: cp.zeros((B, self.N), dtype=cp.float32)
            self._bufs = dict(rr=c(), rm=c(), prr=c(), prm=c(), RD=c(), DMDM=c(), D2M=c(), f_rr=f(), f_rm=f(),
                              f_ccf=f(), nnz=cp.zeros(B, cp.float32), ssum=cp.zeros(B, cp.float32),
                              nrm=cp.zeros(B, cp.float32), pos=cp.zeros(B, cp.int32))
            self._bB = B
        return self._bufs

    def _fftb(self, a, B, direction):
        """In-place C2C of volumes a[0:B] (contiguous [B, N] complex64)."""
        if not self.fft_many:
            for b in range(B):
                self.L.fft_ptr(self.plan, a[b].data.ptr, direction)
            return
        o = 0
        while o < B:                                       # plans for batch sizes = powers of two (and batch)
            m = self.batch if B - o >= self.batch else 1 << ((B - o).bit_length() - 1)
            if m not in self._plans_many:
                self._plans_many[m] = self.L.plan_many(self.n, m)
            self.L.fft_ptr(self._plans_many[m], a[o].data.ptr, direction)
            o += m

    def _orientations(self, mats, ftype, wedge, origin):
        n, N, L = self.n, self.N, self.L
        B = len(mats)
        kb = L.batched()
        bf = self._buffers(B)
        rr, rm, prr, prm = bf["rr"], bf["rm"], bf["prr"], bf["prm"]
        RD, DMDM, D2M, f_rr, f_rm, f_ccf = bf["RD"], bf["DMDM"], bf["D2M"], bf["f_rr"], bf["f_rm"], bf["f_ccf"]
        g, blk = (n, n, B), (n,)
        C, R, Z, n32 = np.uint64(8 * N), np.uint64(4 * N), np.uint64(0), np.int32(n)
        d_rot = cp.asarray(np.ascontiguousarray(mats, dtype=F).ravel())          # 4086 (one 9-float matrix each)
        kb["putblanks"](g, blk, (rr, n32, C))                                    # 4090
        kb["putblanks"](g, blk, (rm, n32, C))                                    # 4091
        kb["rotate"](g, blk, (self.d_reference, rr, self.d_mask, rm, n32, d_rot, Z, C, Z, C, np.uint64(36)))  # 4112
        self._fftb(rr, B, CUFFT_FORWARD)                                         # 4119
        w = self._wedge_args(ftype, wedge)                                       # 4133-4161
        if w is not None:
            kb[w[0]](g, blk, (rr, n32) + tuple(w[1]) + (C,))
        kb["bandpass"](g, blk, (rr, n32, self.high_r, self.low_r, self.win_r, C))  # 4168
        self._fftb(rr, B, CUFFT_INVERSE)                                         # 4169
        kb["normalize_fft"](g, blk, (rr, np.int64(n), C))                        # 4170
        kb["c2f"](g, blk, (rm, f_rm, n32, C, R))                                 # 4179
        kb["c2f"](g, blk, (rr, f_rr, n32, C, R))                                 # 4188
        p_nnz, p_sum, p_nrm, p_pos = bf["nnz"].data.ptr, bf["ssum"].data.ptr, bf["nrm"].data.ptr, bf["pos"].data.ptr
        hs, lib_dot = L.h2s, self.sdot is L
        L.fork()
        for b in range(B):
            h = hs[b % len(hs)]
            L.asum_dev(N, f_rm[b].data.ptr, p_nnz + 4 * b, h)                    # 4182
            if lib_dot:
                L.dot_dev(N, f_rr[b].data.ptr, f_rm[b].data.ptr, p_sum + 4 * b, h)   # 4192
        L.join()
        if not lib_dot:
            for b in range(B):
                self.sdot.dot_dev(N, f_rr[b].data.ptr, f_rm[b].data.ptr, p_sum + 4 * b)
        nnz = bf["nnz"][:B].get()
        ssum = bf["ssum"][:B].get()
        mean = (ssum / nnz).astype(F)                                            # 4193 (float / float, IEEE)
        d_mean = cp.asarray(mean)
        kb["nf2c"](g, blk, (rr, f_rr, f_rm, d_mean, n32, C, R, R))               # 4198
        if lib_dot:
            L.fork()
            for b in range(B):
                L.dot_dev(N, f_rr[b].data.ptr, f_rr[b].data.ptr, p_nrm + 4 * b, hs[b % len(hs)])   # 4207
            L.join()
        else:
            for b in range(B):
                self.sdot.dot_dev(N, f_rr[b].data.ptr, f_rr[b].data.ptr, p_nrm + 4 * b)
        norm = np.sqrt(bf["nrm"][:B].get()).astype(F)                            # 4208 (IEEE sqrt, float)
        d_nnz, d_norm = cp.asarray(nnz), cp.asarray(norm)
        kb["putblanks"](g, blk, (prr, n32, C))                                   # 4219
        kb["pad"](g, blk, (rr, prr, n32, n32, C, C))                             # 4220
        kb["putblanks"](g, blk, (prm, n32, C))                                   # 4223
        kb["pad"](g, blk, (rm, prm, n32, n32, C, C))                             # 4224
        self._fftb(prm, B, CUFFT_FORWARD)                                        # 4238
        self._fftb(prr, B, CUFFT_FORWARD)                                        # 4240
        kb["mbc"](g, blk, (prr, self.d_data, RD, n32, C, Z, C))                  # 4297
        kb["mbc"](g, blk, (prm, self.d_data, DMDM, n32, C, Z, C))                # 4300
        kb["mbc"](g, blk, (prm, self.d_data_square, D2M, n32, C, Z, C))          # 4301
        self._fftb(RD, B, CUFFT_INVERSE)                                         # 4309
        self._fftb(D2M, B, CUFFT_INVERSE)                                        # 4315
        self._fftb(DMDM, B, CUFFT_INVERSE)                                       # 4316
        kb["roseman"](g, blk, (RD, DMDM, D2M, n32, d_nnz, d_norm, C, C, C))      # 4328
        if origin is not None:                                                   # 4355-4414
            kb["limits"](g, blk, (RD, self.sax, self.say, self.saz, np.int32(origin[0]), np.int32(origin[1]),
                                  np.int32(origin[2]), n32, n32, self.d_lim, C, Z))
        kb["plus_one"](g, blk, (RD, f_ccf, n32, C, R))                           # 4420
        L.fork()
        for b in range(B):
            L.iamax_dev(N, f_ccf[b].data.ptr, p_pos + 4 * b, hs[b % len(hs)])    # 4423
        L.join()
        pos = bf["pos"][:B].get().astype(np.int64) - 1                           # 4441
        rz = pos // (n * n)                                                      # 4446-4456
        ry = (pos - n * n * rz) // n
        rx = pos - n * n * rz - n * ry
        bx, by, bz = rawpeak2space_v(rx, n), rawpeak2space_v(ry, n), rawpeak2space_v(rz, n)
        mem = lambda x, y, z: n * n * spacepeak2raw_v(z, n) + n * spacepeak2raw_v(y, n) + spacepeak2raw_v(x, n)
        idx = np.stack([pos, mem(bx - 1, by, bz), mem(bx + 1, by, bz), mem(bx, by - 1, bz), mem(bx, by + 1, bz),
                        mem(bx, by, bz - 1), mem(bx, by, bz + 1)], axis=1)       # 4493-4498
        v = f_ccf[cp.arange(B)[:, None], cp.asarray(idx)].get()                  # 4466, 4507-4512 (reads only)
        cc = v[:, 0] - F(1)                                                      # 4469
        nb = v[:, 1:] - F(1)                                                     # 4520-4525
        return cc, np.stack([bx, by, bz], axis=1), nb, (nnz, ssum, mean, norm)

    # -- spp.cu:2530-2610 + 3950-4615: all refinement levels of the loaded particle
    def refine_particle(self, row):
        card, n = self.card, self.n
        self._constants()
        s = self.shrink
        row = np.asarray(row, dtype=F)
        ftype = int(row[12])                                                    # 3618 (float -> int)
        if card.use_CC > 0 and ftype == 5:                                       # 4150: apply_custom_filter_fft
            raise NotImplementedError("exact engine: fsampling type 5 (custom pfmask per particle) is not implemented")
        wedge = [F(x) for x in row]                                             # minwedge..maxwedge_z, 3988-3993
        ang = (F(row[6]), F(row[7]), F(row[8]))                                 # transfer2variables, 2219-2226
        sh = [F(row[3]) * s, F(row[4]) * s, F(row[5]) * s]
        cc_out = F(row[9])
        self.d_lim.set(matrices_float([row[6:9]])[0])                           # 4011-4035 (table_row angles)
        rng = A.ScanRange(card.cone_range, card.cone_sampling, card.inplane_range, card.inplane_sampling,
                          card.cone_flip, card.inplane_flip)
        modus = int(card.area_search_modus)
        counts = []
        for level in range(int(card.refine) + 1):                               # 2571
            oset = A.orientation_set(tuple(float(x) for x in ang), rng)
            mats = matrices_float(oset)
            O = len(oset)
            counts.append(O)
            if modus == 0:
                origin = None
            elif modus in (1, 11, 5, 51):                                       # 4363-4367, 4390-4394
                origin = (_cdiv(n, 2), _cdiv(n, 2), _cdiv(n, 2))
            elif modus in (2, 21, 3, 31, 4, 41, 6, 61):                         # 4379-4388, 4396-4401: int(float)
                origin = tuple(int(F(_cdiv(n, 2)) + sh[a]) for a in range(3))
            else:
                origin = None                                                   # switch without a case: no limit
            cc = np.empty(O, dtype=F)
            sp = np.empty((O, 3), dtype=np.int64)
            nb = np.empty((O, 6), dtype=F)
            info = []
            if self.batch == 1:
                for o in range(O):
                    cc[o], sp[o], nb[o], inf = self._orientation(mats[o], ftype, wedge, origin)
                    info.append(inf)
            else:
                for o0 in range(0, O, self.batch):
                    o1 = min(O, o0 + self.batch)
                    cc[o0:o1], sp[o0:o1], nb[o0:o1], inf = self._orientations(mats[o0:o1], ftype, wedge, origin)
                    info += list(zip(*inf))
            if self.record is not None:
                for o in range(O):
                    self.record.append((level, oset[o].copy(), mats[o].copy(), cc[o], tuple(int(x) for x in sp[o]),
                                        info[o] + (nb[o].copy(),)))
            # 3961-3983 defaults; 4562 `if (correlation_best_angles<max_correlation_in_this_orientation)`: the first
            # strict maximum among the values > -1 (a NaN never compares true); np.argmax returns the first maximum
            valid = cc > F(-1)
            if valid.any():
                k = int(np.argmax(np.where(valid, cc, F(-np.inf))))
                best = cc[k]
                bang = (F(oset[k, 0]), F(oset[k, 1]), F(oset[k, 2]))
                # 4527-4529: subpixel_spline is a pure function, evaluated by Dynamo for every orientation; only the
                # value of the best orientation reaches the output (4568-4576), so only that one is evaluated here
                sub = [F(self._spline(nb[k, 2 * a], cc[k], nb[k, 2 * a + 1], self.powf)) for a in range(3)]
                bpos = [F(F(sp[k, a]) + sub[a]) for a in range(3)]
            else:
                best, bang = F(-1), (F(-1001), F(-1002), F(-1003))
                bpos = [F(-4001), F(-4002), F(-4003)]
            sh = [F(bpos[a] - F(n) / F(2)) for a in range(3)]                   # 4597-4599
            ang, cc_out = bang, best                                            # 4601-4614 old_results.copy
            rng.shrink(card.refine_factor)                                      # 2580-2605
            rng.cone_flip = int(card.cone_flip > 1)
            rng.inplane_flip = int(card.inplane_flip > 1)
        out = row.copy()                                                        # markPtableVector, 1419-1425
        out[3:6] = [sh[a] / s for a in range(3)]
        out[6:9] = ang
        out[9] = cc_out
        self.counts = counts
        return out
