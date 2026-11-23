"""Dynamo 1.1.514 mask generation as the dcp mask editor does it (dynamo_mask_GUI -> dynamo_mask).

S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src. Spec: docs/analysis/10b_abp_align_journey_1.1.514.md 2.

Ported (bit for bit, checked against headless Dynamo 1.1.514 and the user's abp_align/settings/editorMask.em):

* dynamo_blob(l, fwhm)                      S/dynamo_blob.m:59-85 (with MATLAB's exp: matlab_exp)
* dynamo_cylinder(radii, size, [], 0)        S/dynamo_cylinder.m:61-122 (centred, no smoothing)
* dynamo_mollify(vol, hwhm)                  S/dynamo_mollify.m:107-113 + maprange.m (hwhm 1 or 3 values)
* dynamo_mask('size',..,'type','cylinder', 'r'|'radius'|'sx'|'sy'|'r1', 'height'|'h', 'gaussian', 'invert')
                                             S/dynamo_mask.m:262-575
* MaskEditor: the state of the dynamo_mask_GUI window and its "save" button (read_mask_settings,
  createAndSaveMask, S/dynamo_mask_GUI.m:104-210,291-390).

Arrays are [z, y, x] (C order), the memory layout of a MATLAB (x, y, z) array, as everywhere in pydyn.

Floating point. dynamo_mollify convolves with fftn / ifftn, and dynamo_blob uses exp. MATLAB R2019a (MCR 9.6, the
runtime of Dynamo 1.1.514) computes them with FFTW 3.3.3 ('estimate' planner; same result with 1 or 8 threads) and
with fdlibm's exp. The low-order bits of the convolution (the round-off
"noise floor" outside the mask, |v| < 1e-13) depend on both, and they survive the [0, 1] rescaling and the float32
write. Measured against headless Dynamo dumps (benchmarks/06_abp_align_eo/implB, scripts k1-k4):

* exp: matlab_exp below (fdlibm e_exp.c, vectorised) is bit-identical to MATLAB on all 262,144 blob arguments;
  numpy / glibc exp differ there by +-1 ulp in 24,518 values;
* fftn / ifftn: MATLAB R2019a's own sequence of FFTW calls (_FFTW.fftn_real / ifftn / ifftn_real, read from
  MATLAB with an FFTW call logger, benchmarks/15_fft_sizes): 1-D plans along one dimension at a time, 4 lines per
  execution; a symmetric spectrum (as here) is inverted with a complex-to-real last step. Bit-identical to MATLAB
  (measured for boxes 16 to 320 incl. odd and non-cubic sizes; plan pattern logged for 8 to 512);
* with both, a real dcp mask-editor mask is reproduced byte for byte with the system FFTW 3.3.8 (libfftw3.so.3),
  exactly as with MATLAB's own FFTW 3.3.3; the averaging (pydyn.matlab_numerics) is bit-identical with it too. With numpy / scipy.fft only the voxels with
  |v| > 1e-6 are bit-identical (max |diff| below 1e-15 before the float32 write).

So the FFT backend is FFTW 3 through ctypes (environment variable PYDYN_FFTW_LIB, else the system libfftw3.so.3; no
MATLAB library is used), otherwise scipy.fft with a printed warning. `fft_backend()` tells which one is used. Console shorthands as in Dynamo
(S/shorthands): dmask = dynamo_mask, dmollify = dynamo_mollify, dcylinder = dynamo_cylinder.

Not ported (NotImplementedError): mask types other than 'cylinder' (sphere, ellipsoid, tube, shell, blob,
missing_wedge, missing_pyramid), and a non-empty 'band', 'eulers', 'shifts', 'position', 'for', merges or 'ws'.
A cylinder does not read 'smooth' (S/dynamo_mask.m:484 sets it to 0), 'fwhm', 'tilt', 'sz', 'r2' or 'thick', and
'r1' only stands in for 'r' (:358-360), so these are accepted; 'nfig' is never read. An empty value ([], as the
editor passes for a ticked radio button with an empty box) is the parameter not given, as Dynamo's isempty tests.
"""
import ctypes
import ctypes.util
import os
import warnings

import numpy as np

from . import dynio

# S/dynamo_mask_GUI.m:180-193
GUI_TYPES = ("sphere", "ellipsoid", "cylinder", "tube", "shell", "blob", "missing_wedge", "missing_pyramid")
GUI_RADIO_SETTINGS = ("r", "r1", "r2", "sx", "sy", "sz", "height", "tilt", "fwhm", "band", "gaussian", "eulers",
                      "shifts", "position")
GUI_JUST_RADIO = ("invert",)

# S/dynamo__wizard_masks.m:362-374: file and edit box per editor button
EDITOR_FILES = {"align": ("editorMask.em", "mask"), "cmask": ("editorCMask.em", "cmask"),
                "smask": ("editorSmoothingMask.em", "cmask")}      # sic: smask writes into edit_cmask (:373)


# ===================================================================================================== default mask
def default_mask(l):
    """Dynamo's default alignment/classification mask: dynamo_ellipsoid(l/2-1, l, (l/2+1)*[1 1 1]) (fillSphere)."""
    r, c = l / 2 - 1, l / 2 + 1
    x = np.arange(l, dtype=np.float64)
    t = ((x + 1 - c) / r) ** 2
    # C++ order is (x-term + y-term) + z-term, with x the fastest index (last axis of our [z, y, x] arrays)
    s = t[None, None, :] + t[None, :, None]
    s = s + t[:, None, None]
    return (s <= 1).astype(np.float32)


# ===================================================================================================== FFT backend
class _FFTW:
    """ctypes binding of FFTW 3 (double, FFTW_ESTIMATE, one thread) and MATLAB R2019a's fftn / ifftn on it
    (fftn_real, ifftn, ifftn_real; see the section below). Those three hold a lock; the whole-volume helpers
    (rfftn_full, cfftn, cfft_axis, c2r_axis0) are only called through them."""
    ESTIMATE = 1 << 6
    FORWARD, BACKWARD = -1, 1

    def __init__(self, path):
        lib = ctypes.CDLL(path)
        self.path = path
        try:                                                # e.g. 'fftw-3.3.8-sse2-avx': which build gives the bits
            # const char fftw_version[] (an array, not a pointer): read the string at the symbol itself
            self.version = ctypes.string_at(ctypes.addressof(ctypes.c_char.in_dll(lib, "fftw_version"))).decode()
        except (ValueError, AttributeError):
            self.version = "unknown version"
        self._keep = []
        import threading
        self._lock = threading.RLock()                     # FFTW planning is not thread-safe; plan buffers are shared
        if hasattr(lib, "fftw_set_malloc_fcn"):
            # the MathWorks build routes fftw_malloc through hooks that MATLAB installs; without MATLAB they are
            # unset (fftw_malloc segfaults), so install aligned libc allocators
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.aligned_alloc.restype = ctypes.c_void_p
            libc.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            libc.free.argtypes = [ctypes.c_void_p]
            MALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
            FREE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
            m = MALLOC(lambda n: libc.aligned_alloc(64, ((max(int(n), 1) + 63) // 64) * 64))
            f = FREE(lambda p: libc.free(p))
            lib.fftw_set_malloc_fcn.argtypes = [MALLOC]
            lib.fftw_set_free_fcn.argtypes = [FREE]
            lib.fftw_set_malloc_fcn(m)
            lib.fftw_set_free_fcn(f)
            self._keep += [m, f, libc]
        vp, ci, cu = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        lib.fftw_plan_dft_r2c_3d.restype = vp
        lib.fftw_plan_dft_r2c_3d.argtypes = [ci, ci, ci, vp, vp, cu]
        lib.fftw_plan_dft_3d.restype = vp
        lib.fftw_plan_dft_3d.argtypes = [ci, ci, ci, vp, vp, ci, cu]
        lib.fftw_execute.argtypes = [vp]
        lib.fftw_destroy_plan.argtypes = [vp]
        self.lib = lib

    @staticmethod
    def _aligned(shape, dtype):
        dtype = np.dtype(dtype)
        n = int(np.prod(shape)) * dtype.itemsize
        raw = np.empty(n + 64, dtype=np.uint8)
        off = (-raw.ctypes.data) % 64
        return raw[off:off + n].view(dtype).reshape(shape)

    def rfftn_full(self, x):
        """Whole-volume r2c 3-D plus conjugate completion: the fast candidate of fftn_real, MATLAB's bits only where
        _fast_ok('fftn', shape) proved it (e.g. 32^3, 64^3, 128^3; not 48^3, 96^3, 256^3). Use fftn_real."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        n0, n1, n2 = x.shape
        a = self._aligned(x.shape, np.float64)
        o = self._aligned((n0, n1, n2 // 2 + 1), np.complex128)
        p = self.lib.fftw_plan_dft_r2c_3d(n0, n1, n2, a.ctypes.data, o.ctypes.data, self.ESTIMATE)
        if not p:
            raise RuntimeError("FFTW could not plan r2c %s" % (x.shape,))
        a[...] = x
        self.lib.fftw_execute(p)
        self.lib.fftw_destroy_plan(p)
        h = o.copy()
        full = np.empty(x.shape, dtype=np.complex128)
        m = n2 // 2 + 1
        full[:, :, :m] = h
        if n2 - m > 0:
            i0 = (-np.arange(n0)) % n0
            i1 = (-np.arange(n1)) % n1
            i2 = n2 - np.arange(m, n2)
            full[:, :, m:] = np.conj(h[i0[:, None, None], i1[None, :, None], i2[None, None, :]])
        return full

    def cfftn(self, x, sign):
        x = np.ascontiguousarray(x, dtype=np.complex128)
        a = self._aligned(x.shape, np.complex128)
        o = self._aligned(x.shape, np.complex128)
        p = self.lib.fftw_plan_dft_3d(*x.shape, a.ctypes.data, o.ctypes.data, sign, self.ESTIMATE)
        if not p:
            raise RuntimeError("FFTW could not plan c2c %s" % (x.shape,))
        a[...] = x
        self.lib.fftw_execute(p)
        self.lib.fftw_destroy_plan(p)
        return o.copy()

    def _guru(self):
        if not hasattr(self, "_iodim"):
            class IODim(ctypes.Structure):
                _fields_ = [("n", ctypes.c_int), ("is_", ctypes.c_int), ("os_", ctypes.c_int)]
            vp, ci, cu, pd = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(IODim)
            self.lib.fftw_plan_guru_dft.restype = vp
            self.lib.fftw_plan_guru_dft.argtypes = [ci, pd, ci, pd, vp, vp, ci, cu]
            self.lib.fftw_plan_guru_dft_c2r.restype = vp
            self.lib.fftw_plan_guru_dft_c2r.argtypes = [ci, pd, ci, pd, vp, vp, cu]
            self._iodim = IODim
        return self._iodim

    def cfft_axis(self, x, axis, sign):
        """1-D complex transforms along one axis of a 3-D array (unnormalised), all other lines at once."""
        IODim = self._guru()
        x = np.ascontiguousarray(x, dtype=np.complex128)
        st = [x.strides[k] // 16 for k in range(3)]
        a = self._aligned(x.shape, np.complex128)
        o = self._aligned(x.shape, np.complex128)
        d = (IODim * 1)(IODim(x.shape[axis], st[axis], st[axis]))
        h = (IODim * 2)(*[IODim(x.shape[k], st[k], st[k]) for k in range(3) if k != axis])
        p = self.lib.fftw_plan_guru_dft(1, d, 2, h, a.ctypes.data, o.ctypes.data, sign, self.ESTIMATE)
        if not p:
            raise RuntimeError("FFTW could not plan c2c along axis %d of %s" % (axis, x.shape))
        a[...] = x
        self.lib.fftw_execute(p)
        self.lib.fftw_destroy_plan(p)
        return o.copy()

    def c2r_axis0(self, x, n0):
        """1-D complex-to-real backward transforms along axis 0 (unnormalised): x holds planes 0..n0//2 of a
        spectrum that is conjugate-symmetric along axis 0; returns the real [n0, n1, n2] result."""
        IODim = self._guru()
        x = np.ascontiguousarray(x, dtype=np.complex128)
        h0, n1, n2 = x.shape
        if h0 != n0 // 2 + 1:
            raise ValueError("c2r_axis0: %d planes given for length %d" % (h0, n0))
        a = self._aligned(x.shape, np.complex128)
        o = self._aligned((n0, n1, n2), np.float64)
        d = (IODim * 1)(IODim(n0, n1 * n2, n1 * n2))
        h = (IODim * 2)(IODim(n1, n2, n2), IODim(n2, 1, 1))
        p = self.lib.fftw_plan_guru_dft_c2r(1, d, 2, h, a.ctypes.data, o.ctypes.data, self.ESTIMATE)
        if not p:
            raise RuntimeError("FFTW could not plan c2r along axis 0 of %s" % ((n0, n1, n2),))
        a[...] = x
        self.lib.fftw_execute(p)
        self.lib.fftw_destroy_plan(p)
        return o.copy()

    # ---------------------------------------------------------------- MATLAB R2019a's fftn / ifftn (libmwmfl_fft)
    # Read with an LD_PRELOAD logger of its FFTW calls (benchmarks/15_fft_sizes/fftw_shim.c): MATLAB transforms one
    # dimension at a time with 1-D plans (guru64, FFTW_ESTIMATE, one thread) made on buffers aligned 16 bytes past
    # a 64-byte boundary, for 4 lines at a time (howmany 4) and for single left-over lines (howmany 1). The lines
    # are grouped 4 by 4 along dim 1 over all columns, along dim 2 over x within each z slice, and along dim 3 over
    # the flattened (x, y) index. ifft divides each part by n after each dimension. Which 1-D plan transforms a
    # line decides its bits for sizes that are not powers of two.
    GROUP = 4

    def _plan1(self, kind, n, hm, sign):
        """Cached 1-D plan ('c2c', 'r2c', 'c2r') of length n for hm contiguous lines, planned like MATLAB on
        buffers at 16 mod 64; returns (plan, in buffer, out buffer)."""
        if not hasattr(self, "_p1"):
            self._p1 = {}
            class IO64(ctypes.Structure):
                _fields_ = [("n", ctypes.c_ssize_t), ("is_", ctypes.c_ssize_t), ("os_", ctypes.c_ssize_t)]
            vp, ci, cu, pio = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(IO64)
            for nm, extra in (("fftw_plan_guru64_dft", [ci, cu]), ("fftw_plan_guru64_dft_r2c", [cu]),
                              ("fftw_plan_guru64_dft_c2r", [cu])):
                fn = getattr(self.lib, nm)
                fn.restype = vp
                fn.argtypes = [ci, pio, ci, pio, vp, vp] + extra
            for nm in ("fftw_execute_dft", "fftw_execute_dft_r2c", "fftw_execute_dft_c2r"):
                getattr(self.lib, nm).argtypes = [vp, vp, vp]
                getattr(self.lib, nm).restype = None
            self._io64 = IO64
        key = (kind, n, hm, sign)
        if key not in self._p1:
            IO64, h = self._io64, n // 2 + 1
            nin, nout = {"c2c": (n, n), "r2c": (n, h), "c2r": (h, n)}[kind]
            tin = np.float64 if kind == "r2c" else np.complex128
            tout = np.float64 if kind == "c2r" else np.complex128
            a, o = self._aligned16((hm, nin), tin), self._aligned16((hm, nout), tout)
            d = (IO64 * 1)(IO64(n, 1, 1))
            m = (IO64 * 1)(IO64(hm, nin, nout))
            if kind == "c2c":
                p = self.lib.fftw_plan_guru64_dft(1, d, 1, m, a.ctypes.data, o.ctypes.data, sign, self.ESTIMATE)
            elif kind == "r2c":
                p = self.lib.fftw_plan_guru64_dft_r2c(1, d, 1, m, a.ctypes.data, o.ctypes.data, self.ESTIMATE)
            else:
                p = self.lib.fftw_plan_guru64_dft_c2r(1, d, 1, m, a.ctypes.data, o.ctypes.data, self.ESTIMATE)
            if not p:
                raise RuntimeError("FFTW could not plan %s n=%d howmany=%d" % (kind, n, hm))
            self._p1[key] = (p, a, o)
        return self._p1[key]

    @staticmethod
    def _aligned16(shape, dtype, zero=True):
        """Array whose data start 16 bytes past a 64-byte boundary (MATLAB's FFT buffers); zeroed unless zero=False."""
        dtype = np.dtype(dtype)
        n = int(np.prod(shape)) * dtype.itemsize
        raw = (np.zeros if zero else np.empty)(n + 128, dtype=np.uint8)
        off = ((-raw.ctypes.data) % 64) + 16
        return raw[off:off + n].view(dtype).reshape(shape)

    def _lines(self, X, kind, sign, n, block):
        """Transform the rows of X ([lines, len], C order, data at 16 mod 64) as MATLAB does: 4 consecutive rows
        per execution of the howmany-4 plan, restarting at every multiple of `block` rows; left-over rows one by
        one. Returns a new array (at 16 mod 64)."""
        nl = X.shape[0]
        h = n // 2 + 1
        nout = {"c2c": n, "r2c": h, "c2r": n}[kind]
        tout = np.float64 if kind == "c2r" else np.complex128
        Y = self._aligned16((nl, nout), tout, zero=False)
        ex = {"c2c": self.lib.fftw_execute_dft, "r2c": self.lib.fftw_execute_dft_r2c,
              "c2r": self.lib.fftw_execute_dft_c2r}[kind]
        xb, yb = X.ctypes.data, Y.ctypes.data
        xs, ys = X.strides[0], Y.strides[0]
        g = self.GROUP
        p4 = self._plan1(kind, n, g, sign)[0] if block >= g else None
        p1, a1, o1 = self._plan1(kind, n, 1, sign)
        for b0 in range(0, nl, block):
            s, end = b0, b0 + block
            while s + g <= end:
                ip, op = xb + s * xs, yb + s * ys
                if ip % 32 == 16 and op % 32 == 16:                 # the planning alignment: run in place
                    ex(p4, ip, op)
                else:
                    _, a4, o4 = self._plan1(kind, n, g, sign)
                    a4[...] = X[s:s + g]
                    ex(p4, a4.ctypes.data, o4.ctypes.data)
                    Y[s:s + g] = o4
                s += g
            while s < end:
                ip, op = xb + s * xs, yb + s * ys
                if ip % 32 == 16 and op % 32 == 16:
                    ex(p1, ip, op)
                else:
                    a1[0] = X[s]
                    ex(p1, a1.ctypes.data, o1.ctypes.data)
                    Y[s] = o1[0]
                s += 1
        return Y

    def _stage(self, V, axis, dtype):
        """V [z, y, x] -> rows of lines along `axis` at 16 mod 64, in MATLAB's line order: axis 2 rows (z, y);
        axis 1 rows (z, x); axis 0 rows (y, x). Copied slice by slice (2-D transposes stay in cache)."""
        nz, ny, nx = V.shape
        if axis == 2:
            X = self._aligned16((nz, ny, nx), dtype, zero=False)
            X[...] = V
        elif axis == 1:
            X = self._aligned16((nz, nx, ny), dtype, zero=False)
            for z in range(nz):
                X[z] = V[z].T
        else:
            X = self._aligned16((ny, nx, nz), dtype, zero=False)
            for y in range(ny):
                X[y] = np.ascontiguousarray(V[:, y, :]).T     # contiguous slab first: no 1-MB-stride gathers
        return X

    @staticmethod
    def _unstage(Y, axis):
        """Inverse of _stage for the transformed rows Y (3-D, last axis = the transformed line)."""
        if axis == 2:
            return Y
        if axis == 1:
            nz, nx, m = Y.shape
            out = np.empty((nz, m, nx), Y.dtype)
            for z in range(nz):
                out[z] = Y[z].T
            return out
        ny, nx, m = Y.shape
        out = np.empty((m, ny, nx), Y.dtype)
        for y in range(ny):
            out[:, y, :] = Y[y].T
        return out

    def _along(self, V, axis, kind, sign):
        """MATLAB's 1-D transforms of the [z, y, x] array V along axis (2 = MATLAB dim 1, 1 = dim 2, 0 = dim 3),
        unnormalised; lines grouped 4 by 4 over all columns (axis 2), over x within each z slice (axis 1), over the
        flattened (x, y) index (axis 0)."""
        nz, ny, nx = V.shape
        n = V.shape[axis]
        X = self._stage(V, axis, np.float64 if kind == "r2c" else np.complex128)
        block = {2: nz * ny, 1: nx, 0: ny * nx}[axis]
        Y = self._lines(X.reshape(-1, n), kind, sign, n, block)
        return self._unstage(Y.reshape(X.shape[0], X.shape[1], -1), axis)

    @staticmethod
    def _complete_x(H, nx):
        """Full x lines from r2c halves: F[k] = conj(H[nx - k]) per line."""
        nz, ny, h = H.shape
        F = np.empty((nz, ny, nx), np.complex128)
        F[:, :, :h] = H
        if nx - h > 0:
            F[:, :, h:] = np.conj(H[:, :, 1:nx - h + 1][:, :, ::-1])
        return F

    @staticmethod
    def _hermitize(G):
        """MATLAB's real fftn output: the redundant half replaced by conjugate mirrors of the computed one (z > n/2;
        then y > n/2 in the planes z = 0, n/2; then x > n/2 on the lines with z, y in {0, n/2}), and an imaginary
        part of exactly 0 at the self-conjugate points."""
        S = G.copy()
        nz, ny, nx = S.shape
        mz, my, mx = ((-np.arange(k)) % k for k in (nz, ny, nx))
        hz, hy, hx = nz // 2 + 1, ny // 2 + 1, nx // 2 + 1
        selfs = lambda k: [0] + ([k // 2] if k % 2 == 0 else [])
        if hz < nz:
            S[hz:] = np.conj(G[mz[hz:]][:, my][:, :, mx])
        for z0 in selfs(nz):
            if hy < ny:
                S[z0, hy:] = np.conj(G[z0, my[hy:]][:, mx])
            for y0 in selfs(ny):
                if hx < nx:
                    S[z0, y0, hx:] = np.conj(G[z0, y0, mx[hx:]])
                for x0 in selfs(nx):
                    S[z0, y0, x0] = complex(S[z0, y0, x0].real, 0.0)
        return S

    @staticmethod
    def _div(V, n):
        """MATLAB complex ./ real scalar: each part divided."""
        out = np.empty(V.shape, np.complex128)
        out.real = V.real / n
        out.imag = V.imag / n
        return out

    def _fftn_real_exact(self, x):
        """MATLAB fftn of a real 3-D array: r2c along x and per-line completion, c2c along y and z, then the
        conjugate mirroring of _hermitize."""
        F = self._complete_x(self._along(x, 2, "r2c", 0), x.shape[2])
        F = self._along(F, 1, "c2c", self.FORWARD)
        F = self._along(F, 0, "c2c", self.FORWARD)
        return self._hermitize(F)

    def _ifftn_exact(self, V):
        """MATLAB ifftn of a 3-D array that is not conjugate-symmetric: ifft along x, y, z, each / n."""
        for ax in (2, 1, 0):
            V = self._div(self._along(V, ax, "c2c", self.BACKWARD), V.shape[ax])
        return V

    def _ifftn_sym_exact(self, V):
        """MATLAB ifftn of a conjugate-symmetric 3-D array (real result): ifft along x and y (complex), then a
        complex-to-real ifft along z, each / n."""
        nz = V.shape[0]
        A = self._div(self._along(V, 2, "c2c", self.BACKWARD), V.shape[2])
        A = self._div(self._along(A[: nz // 2 + 1], 1, "c2c", self.BACKWARD), V.shape[1])
        X = self._stage(A, 0, np.complex128)               # lines along z (planes 0..nz/2), rows ordered (y, x)
        Y = self._lines(X.reshape(-1, X.shape[2]), "c2r", 0, nz, V.shape[1] * V.shape[2])
        return self._unstage(Y.reshape(X.shape[0], X.shape[1], nz), 0) / nz

    # the same results through whole-volume plans (fast); used for a shape only after _fast_ok proved them
    # bit-identical to the exact path on that shape (they are for 64^3, not for 45^3, 48^3, 50^3)
    def _fftn_real_fast(self, x):
        return self.rfftn_full(x)

    def _ifftn_fast(self, V):
        return self._div(self.cfftn(V, self.BACKWARD), V.size)

    def _ifftn_sym_fast(self, V):
        n0 = V.shape[0]
        A = self.cfft_axis(V[: n0 // 2 + 1], 2, self.BACKWARD)
        A = self.cfft_axis(A, 1, self.BACKWARD)
        return self.c2r_axis0(A, n0) / V.size

    FAST = True                                             # False: always the exact (grouped 1-D) path

    def _fast_ok(self, op, shape):
        """Once per (operation, shape): do the whole-volume plans give exactly the bits of MATLAB's grouped 1-D
        transforms? Checked on a random input (the FFT arithmetic does not depend on the data, so one generic
        input decides); the answer is cached."""
        if not self.FAST:
            return False
        if not hasattr(self, "_fast"):
            self._fast = {}
        key = (op, tuple(shape))
        if key not in self._fast:
            rng = np.random.default_rng(20261119)
            x = rng.standard_normal(shape)
            if op == "fftn":
                a, b = self._fftn_real_exact(x), self._fftn_real_fast(x)
            elif op == "ifftn":
                q = x + 1j * rng.standard_normal(shape)
                a, b = self._ifftn_exact(q), self._ifftn_fast(q)
            else:
                F = self._fftn_real_exact(x)
                a, b = self._ifftn_sym_exact(F), self._ifftn_sym_fast(F)
            same = lambda u, v: np.array_equal(np.real(u).view(np.int64), np.real(v).view(np.int64))
            self._fast[key] = bool(same(a, b) and same(np.imag(a), np.imag(b)))
        return self._fast[key]

    def fftn_real(self, x):
        """MATLAB fftn of a real 3-D array (pydyn [z, y, x] = MATLAB (x, y, z))."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        with self._lock:
            return self._fftn_real_fast(x) if self._fast_ok("fftn", x.shape) else self._fftn_real_exact(x)

    def ifftn(self, V):
        """MATLAB ifftn of a 3-D array that is not conjugate-symmetric (complex result)."""
        V = np.ascontiguousarray(V, dtype=np.complex128)
        with self._lock:
            return self._ifftn_fast(V) if self._fast_ok("ifftn", V.shape) else self._ifftn_exact(V)

    def ifftn_real(self, V):
        """real(ifftn(V)) as MATLAB R2019a computes it. MATLAB first tests V for exact conjugate symmetry
        (V(k) == conj(V(-k)) for every k): a symmetric V takes the real path (_ifftn_sym_exact), any other V the
        complex ifftn, real part."""
        V = np.ascontiguousarray(V, dtype=np.complex128)
        if not conj_symmetric(V):
            return np.real(self.ifftn(V))
        with self._lock:
            return self._ifftn_sym_fast(V) if self._fast_ok("ifftn_sym", V.shape) else self._ifftn_sym_exact(V)


def conj_symmetric(V):
    """True when V(k) == conj(V(-k mod n)) holds exactly for every element (MATLAB's ifftn symmetry test;
    NaN never compares equal)."""
    if not np.iscomplexobj(V):
        V = V.astype(np.complex128)
    idx = [(-np.arange(s)) % s for s in V.shape]
    M = V[np.ix_(*idx)]
    return bool(np.array_equal(V.real, M.real) and np.array_equal(V.imag, -M.imag))


_BACKEND = {}


def _fftw_candidates():
    env = os.environ.get("PYDYN_FFTW_LIB")
    if env:
        return [env]
    out = []
    sysl = ctypes.util.find_library("fftw3")
    out += [sysl] if sysl else []
    out += ["libfftw3.so.3"]
    return out


def fft_backend():
    """'fftw:<library> (<fftw_version>)' or 'scipy' (see the module doc). Bit-identity with MATLAB was measured with
    fftw-3.3.8-sse2-avx; other builds are not verified."""
    if "b" not in _BACKEND:
        _BACKEND["b"] = None
        for path in _fftw_candidates():
            if path and (os.path.isfile(path) or not os.path.isabs(path)):
                try:
                    _BACKEND["b"] = _FFTW(path)
                    break
                except OSError:
                    continue
    b = _BACKEND["b"]
    return "scipy" if b is None else "fftw:%s (%s)" % (b.path, b.version)


def _filter_local(a, filt):
    """filterLocal of dynamo_mollify.m:111-113: real(ifftn(fftn(a).*fftn(ifftshift(filt))))."""
    fr = np.fft.ifftshift(filt)
    fft_backend()
    b = _BACKEND["b"]
    if b is not None and a.ndim == 3:
        from .matlab_numerics import cmul
        return b.ifftn_real(cmul(b.fftn_real(a), b.fftn_real(fr)))
    warnings.warn("pydyn.masks: no FFTW library found; dynamo_mollify uses scipy.fft. Voxels above 1e-6 are "
                  "bit-identical to Dynamo, the round-off floor below 1e-13 is not (set PYDYN_FFTW_LIB).")
    import scipy.fft as sf
    return np.real(sf.ifftn(sf.fftn(a) * sf.fftn(fr)))


# ===================================================================================================== exp
# fdlibm e_exp.c constants
_LN2HI = np.array([6.93147180369123816490e-01, -6.93147180369123816490e-01])
_LN2LO = np.array([1.90821492927058770002e-10, -1.90821492927058770002e-10])
_INVLN2 = 1.44269504088896338700e+00
_HALF = np.array([0.5, -0.5])
_P1, _P2, _P3, _P4, _P5 = (1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05,
                           -1.65339022054652515390e-06, 4.13813679705723846039e-08)
_O_THRESHOLD, _U_THRESHOLD = 7.09782712893383973096e+02, -7.45133219101941108420e+02

# matlab_exp is ported from fdlibm's e_exp.c:
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is preserved.
def matlab_exp(x):
    """exp as MATLAB R2019a computes it (fdlibm e_exp.c, no FMA), vectorised.
    Bit-identical to MATLAB on all 262,144 arguments of dynamo_blob(64, 6) (k4_blob_dump.m); numpy / glibc exp
    differ there by +-1 ulp in 24,518 values. |x| >= 709.78 is e_exp.c's filter: NaN -> x+x, exp(+-inf) = {inf, 0},
    x > o_threshold -> huge*huge (inf), x < u_threshold -> twom1000*twom1000 (0); the rest of that range takes the
    reduction below, where e_exp.c's k < -1021 scaling (y*2^(k+1000))*2^-1000 is the one rounding of ldexp(y, k)
    into the subnormal range (e.g. the corners of dynamo_blob(128, 6), arguments down to -946)."""
    x = np.asarray(x, dtype=np.float64)
    hxs = (x.view(np.uint64) >> np.uint64(32)).astype(np.int64)
    xsb = (hxs >> 31) & 1
    hx = hxs & 0x7fffffff
    filt = hx >= 0x40862E42                             # if |x| >= 709.78...
    nan = filt & np.isnan(x)
    over = filt & (x > _O_THRESHOLD)                    # overflow, and exp(+inf) = +inf
    under = filt & (x < _U_THRESHOLD)                   # underflow, and exp(-inf) = 0
    x_in = x
    x = np.where(nan | over | under, 0.0, x)            # those return above; keep the reduction finite there
    k = np.zeros(x.shape, dtype=np.int64)
    hi = x.copy()
    lo = np.zeros_like(x)
    red = hx > 0x3fd62e42
    small = red & (hx < 0x3FF0A2B2)
    big = red & ~small
    hi = np.where(small, x - _LN2HI[xsb], hi)
    lo = np.where(small, _LN2LO[xsb], lo)
    k = np.where(small, 1 - xsb - xsb, k)
    kb = np.trunc(_INVLN2 * x + _HALF[xsb]).astype(np.int64)
    t = kb.astype(np.float64)
    hi = np.where(big, x - t * _LN2HI[0], hi)
    lo = np.where(big, t * _LN2LO[0], lo)
    k = np.where(big, kb, k)
    xr = np.where(red, hi - lo, x)
    tiny = (~red) & (hx < 0x3e300000)
    t = xr * xr
    c = xr - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    y0 = 1.0 - ((xr * c) / (c - 2.0) - xr)
    y1 = 1.0 - ((lo - (xr * c) / (2.0 - c)) - hi)
    y1 = np.ldexp(y1, k)
    out = np.where(k == 0, y0, y1)
    out = np.where(tiny, 1.0 + x, out)
    xn = np.where(nan, x_in, 0.0)
    out = np.where(nan, xn + xn, out)                   # NaN: x+x
    out = np.where(over, np.inf, out)
    out = np.where(under, 0.0, out)
    return out

# ===================================================================================================== geometry
def _size3(my_size):
    s = np.ravel(np.asarray(my_size, dtype=float))
    if s.size == 1:
        s = s[0] * np.ones(3)
    return s


def dynamo_blob(l, fwhm):
    """S/dynamo_blob.m: exp(-x1^2-x2^2-x3^2) on the grid -l/2:l/2-1 (even l), x_i = i/sqrt(2 c^2),
    c = fwhm/2.35482. Returned as [z, y, x] (x1 is the first MATLAB index = last axis here)."""
    if fwhm is None or np.size(fwhm) == 0:              # set_default_or_empty('fwhm',l/3) :59
        fwhm = int(l) / 3
    fw = np.ravel(np.asarray(fwhm, dtype=float))
    if fw.size == 2:
        raise ValueError("Please give just one value or three [x,y,z] for the fwmh parameter")
    if fw.size == 1:
        fw = fw[0] * np.ones(3)
    c = fw / 2.35482
    l = int(l)
    i = np.arange(-l // 2, l // 2, dtype=float) if l % 2 == 0 else np.arange(-(l // 2), l // 2 + 1, dtype=float)
    x3, x2, x1 = np.meshgrid(i, i, i, indexing="ij")
    # MATLAB's scalar c^2 is c*c (MCR muDoubleScalarPower(x, 2) == x*x); NumPy's scalar ** 2 is C pow(), which is not
    # x*x for every x (e.g. hwhm 2.85, 5.7, 11.4, 22.8, 37.81)
    x1 = x1 / np.sqrt(2 * (c[0] * c[0]))
    x2 = x2 / np.sqrt(2 * (c[1] * c[1]))
    x3 = x3 / np.sqrt(2 * (c[2] * c[2]))
    return matlab_exp(-x1 ** 2 - x2 ** 2 - x3 ** 2)


def dynamo_cylinder(radii, my_size, position=None, smoothing_pixels=0):
    """S/dynamo_cylinder.m:61-122: axis z, sqrt((x-px)^2/rx^2 + (y-py)^2/ry^2) <= 1 and |z-pz| <= h. The default
    position is (size(1)/2+1)*[1 1 1] (1-based). Returned as [z, y, x]."""
    s = _size3(my_size)
    radii = list(np.ravel(np.asarray(radii, dtype=float)))
    if len(radii) == 1:
        rr, h = [radii[0], radii[0]], s[2] / 2
    elif len(radii) == 2:
        rr, h = [radii[0], radii[0]], radii[1]
    else:
        rr, h = [radii[0], radii[1]], radii[2]
    if position is None or np.size(position) == 0:
        position = (s[0] / 2 + 1) * np.ones(3)
    position = np.ravel(np.asarray(position, dtype=float))
    if position.size == 1:
        position = position[0] * np.ones(3)
    s1, s2, s3 = (int(v) for v in s)
    if rr[0] * rr[1] == 0:
        return np.zeros((s3, s2, s1))
    if smoothing_pixels and smoothing_pixels > 0:
        raise NotImplementedError("dynamo_cylinder with smoothing_pixels > 0")
    z, y, x = np.meshgrid(np.arange(1, s3 + 1, dtype=float), np.arange(1, s2 + 1, dtype=float),
                          np.arange(1, s1 + 1, dtype=float), indexing="ij")
    c1 = (x - position[0]) ** 2 / (rr[0] * rr[0])                  # radii(i)^2 = r*r (scalar ** 2 is pow())
    c2 = (y - position[1]) ** 2 / (rr[1] * rr[1])
    vol = (np.sqrt(c1 + c2) <= 1).astype(np.float64)
    vol[np.abs(z - position[2]) > h] = 0
    return vol


def dynamo_mollify(volume, hwhm=None):
    """S/dynamo_mollify.m:107-113 (3-D, useMatlab false): blob of FWHM 2*hwhm, circular FFT convolution, then
    maprange to [min(volume), max(volume)]. hwhm*2 is elementwise: 3 values give dynamo_blob a width per axis
    [x, y, z], 2 values are its error (dynamo_blob.m:60-62). There is no default for hwhm: dynamo_mollify.m:67 is
    set_default('fwhm',3), a different variable, so Dynamo fails at hwhm*2 (:107) with "Undefined function or
    variable 'hwhm'" when it is not given; so does this function."""
    if hwhm is None:
        raise TypeError("dynamo_mollify: Undefined function or variable 'hwhm' (the source's set_default('fwhm',3) "
                        "sets another variable; pass hwhm)")
    v = np.asarray(volume, dtype=np.float64)
    if v.ndim != 3:
        raise NotImplementedError("dynamo_mollify of a 2-D image (imgaussfilt)")
    blob = dynamo_blob(v.shape[0], 2 * np.ravel(np.asarray(hwhm, dtype=float)))     # hwhm*2 :107
    f = _filter_local(v, blob)
    return maprange(f, v.min(), v.max())


def maprange(x, a=0.0, b=1.0):
    """maprange.m: (x - min) / max(x - min) * (b - a) + a."""
    x = x - x.min()
    out = x / x.max()
    out = out * (b - a)
    return out + a


# ===================================================================================================== dynamo_mask
_UNPORTED = ("smooth", "band", "fwhm", "eulers", "shifts", "position", "tilt", "thick", "r1", "r2", "sz", "for",
             "merge_with", "mw", "or", "subtract_from", "sf", "ws")       # 'nfig' is declared (:191), never read
# not read by the cylinder case: q.smooth=0 (:484); tilt only in :516-524, fwhm :508-512, sz :480, r2 and thick
# :489-502; r1 only stands in for r (:358-360) (their errors of :395 and :405 are kept in dynamo_mask)
_NOT_READ_BY_CYLINDER = ("smooth", "fwhm", "tilt", "sz", "r1", "r2", "thick")


def _empty(v):
    """MATLAB isempty of a parameter as passed: None (not given), '' or an array of no elements. dynamo__str2num
    (:272) keeps a char that is not a number, so a non-empty char stays non-empty."""
    if v is None:
        return True
    if isinstance(v, str):
        return v == ""
    return np.size(v) == 0


def _num(v):
    if v is None:
        return None
    if isinstance(v, str):
        from .dcpcontrol import str2num                 # dynamo__str2num of a char parameter
        v = str2num(v)
    a = np.ravel(np.asarray(v, dtype=float))
    return None if a.size == 0 else a


def _size_argument(raw):
    """The 'size' of dynamo_mask.m:274-292. dynamo__str2num keeps a char that is not a number (a file name), then
    ischar(q.size) -> q.size = dynamo_read(file) and, for any array with ndims > 2, q.size = squeeze(size(array))
    = [nx ny nz] (this port stores volumes as [z, y, x], hence the reversed shape)."""
    if isinstance(raw, str):
        try:
            return _num(raw)
        except NotImplementedError:                       # not a number: a file name, dynamo_read(q.size)
            if not os.path.isfile(raw):
                raise FileNotFoundError('[dread: error] cannot find requested file "%s"' % raw)
            if not raw.lower().endswith(".em"):
                raise NotImplementedError("dynamo_mask size from %r: only EM files are read" % raw)
            return np.array(dynio.read_em(raw).shape[::-1], dtype=float)
    if isinstance(raw, np.ndarray) and raw.ndim > 2:
        return np.array(raw.shape[::-1], dtype=float)
    return _num(raw)


def dynamo_mask(*args, **kw):
    """dynamo_mask(...) (S/dynamo_mask.m) for 'type','cylinder'. Arguments as Dynamo's name/value pairs, as
    positional pairs or keywords: size, type, r (or radius), sx, sy, r1, height (or h), gaussian, invert, o
    (output file); an empty value ([]) is the parameter not given, as Dynamo's isempty tests. Returns the mask as a
    float64 [z, y, x] array (None where Dynamo prints and returns empty)."""
    if len(args) % 2:
        raise TypeError("dynamo_mask takes name/value pairs")
    q = {}
    for k, v in list(zip(args[0::2], args[1::2])) + list(kw.items()):
        q[str(k).lstrip("-")] = v                        # dynamo__parse_remove_minus
    if not _empty(q.get("radius")):                      # dynamo__parse_short2real(q,'radius','r') :262
        q["r"] = q["radius"]
    if not _empty(q.get("h")):                           # :263
        q["height"] = q["h"]
    mtype = q.get("type")
    for k in _UNPORTED:
        if _empty(q.get(k)) or (mtype == "cylinder" and k in _NOT_READ_BY_CYLINDER):
            continue
        if not (k == "smooth" and np.all(_num(q[k]) == 0)):
            raise NotImplementedError("dynamo_mask parameter %r is not ported" % k)
    size = _size_argument(q.get("size"))
    invert = q.get("invert")
    if _empty(invert):                                   # if q.invert (:572) of [] is false
        invert = False
    invert = bool(invert) if not isinstance(invert, str) else invert.strip().lower() in ("1", "true", "yes", "on")
    if size is None:
        print("Mask needs a size, aborting.")
        return None
    if size.size == 1:
        size = size[0] * np.ones(3)
    if size.size != 3:
        print("Please provide a valid size: aborting dynamo_mask.")
        return None
    if mtype != "cylinder":
        raise NotImplementedError("dynamo_mask type %r (only 'cylinder' is ported)" % (mtype,))
    nx = size[0]
    r, sx, sy, r1 = _num(q.get("r")), _num(q.get("sx")), _num(q.get("sy")), _num(q.get("r1"))
    height, gaussian = _num(q.get("height")), _num(q.get("gaussian"))
    if r is None and sx is None and r1 is None:          # :350-352
        r = np.array([np.floor(nx / 2)])
    if r is None and sx is not None:                     # :354-356
        r = sx
    if r is None and r1 is not None:                     # :358-360
        r = r1
    if sx is None and sy is None:                        # :363-366
        sx, sy = r, r
    if sx is not None and sy is None:
        sy = sx
    if height is None:                                   # :384-386 sets q.heigh (sic), then :416-418
        height = np.array([np.floor(nx / 2)])
    # :388-406 do not change a cylinder; their one effect is MATLAB's error for vectors of unequal lengths in
    # q.r1+q.thick and q.r2-q.r1, which NumPy's broadcasting raises alike (ValueError)
    r1 = r if r1 is None else r1                         # :388-390
    r2, thick = _num(q.get("r2")), _num(q.get("thick"))
    if r2 is None:                                       # :392-400 (q.thickess=2, sic, when thick is empty)
        r2 = r1 + (2 if thick is None else thick)
    np.subtract(r2, r1)                                  # :403-406 q.thick=q.r2-q.r1
    mask = dynamo_cylinder([sx[0], sy[0], height[0] / 2], size, None, 0)     # :483-485 (q.smooth = 0)
    mask = np.nan_to_num(mask, nan=0.0)                  # dynamo_nan(mask,0) :541
    if gaussian is not None:                             # :564-566, the whole vector (dynamo_mollify)
        mask = dynamo_mollify(mask, gaussian)
    if invert:                                           # :572-574
        mask = -mask + np.ones(mask.shape)
    if q.get("o"):
        dynio.write_em(q["o"], mask)
    return mask


# ===================================================================================================== mask editor
class MaskEditor:
    """The dynamo_mask_GUI window, as opened by the dcp masks wizard (S/dynamo__wizard_masks.m:406):
    edit_size = num2str(size), edit_save = the target file; type popup 'sphere' and every radio button off
    (dynamo_mask_GUI.fig defaults, 10b 2.1). `select(name, value)` ticks a radio button and fills its edit box,
    `save()` is the save button (createAndSaveMask)."""

    def __init__(self, size=None, mask=None, volume=None):
        self.volume = volume
        self.size = "" if size is None else "  ".join("%g" % float(s) for s in np.ravel(size))   # num2str
        self.save_file = mask or ""
        self.type = "sphere"
        self.radio = {p: False for p in GUI_RADIO_SETTINGS + GUI_JUST_RADIO}
        self.edit = {p: "" for p in GUI_RADIO_SETTINGS}
        self.messages = []

    def set_type(self, t):
        if t not in GUI_TYPES:
            raise ValueError("mask type %r is not in the popup menu %s" % (t, GUI_TYPES))
        self.type = t

    def select(self, name, value=None, on=True):
        if name not in self.radio:
            raise ValueError("no radio button %r in dynamo_mask_GUI" % name)
        self.radio[name] = bool(on)
        if value is not None and name in self.edit:
            self.edit[name] = value if isinstance(value, str) else " ".join("%g" % v for v in np.ravel(value))

    def read_mask_settings(self):
        """read_mask_settings (S/dynamo_mask_GUI.m:291-362): the argument list of dynamo_mask."""
        from .dcpcontrol import str2num
        my_size = str2num(self.size) if self.size.strip() else np.zeros(0)
        if np.size(my_size) == 0:
            if self.volume and os.path.isfile(self.volume):
                my_size = np.array(dynio.read_em(self.volume).shape[::-1], dtype=float)
            else:
                self.messages.append("Cannot determine intended size, type it or provide a reference file")
                return []
        args = ["size", my_size, "type", self.type]
        for p in GUI_RADIO_SETTINGS:
            if self.radio[p]:
                args += [p, str2num(self.edit[p]) if self.edit[p].strip() else np.zeros(0)]
        for p in GUI_JUST_RADIO:
            args += [p, bool(self.radio[p])]
        return args

    def show(self, mode="mask", view="z", alpha=0.5, color="r"):
        """The "Show" panel (dynamo_mask_GUI.m:944-1100, pushbutton_show): the mask built from the current settings
        shown with dslices along x|y|z: mode 'mask' (title "Mask: <view> view", update_show_mask) or 'overlay' on the
        reference volume (edit_example; update_show_overlay: alpha, overlay colour). 'filtered' is not ported. Opens the
        viewer process and returns at once; returns the Popen or None."""
        from . import viewers
        if mode not in ("mask", "overlay"):
            raise NotImplementedError("mask editor show mode %r (only 'mask' and 'overlay' are ported)" % (mode,))
        if view not in ("x", "y", "z"):
            self.messages.append("Choose a view x|y|z. Aborting depiction.")
            return None
        args = self.read_mask_settings()
        if not args:
            self.messages.append("Desisting depiction: input error")
            return None
        mask = dynamo_mask(*args)
        if mask is None:
            self.messages.append("Desisting depiction: mask cannot be formed")
            return None
        title = "Mask: %s view" % view
        if mode == "mask":
            p = viewers.dslices(mask, view, "title", title)
        else:
            if not (self.volume and os.path.isfile(self.volume)):
                self.messages.append("volume not found")
                self.messages.append("Invalid reference")
                return None
            vol = dynio.read_em(self.volume)
            if vol.shape != mask.shape:
                self.messages += ["size mismatch of volume and mask.",
                                  "volume : %d x %d x %d" % vol.shape[::-1], "mask   : %d x %d x %d" % mask.shape[::-1]]
                return None
            p = viewers.dslices(vol, view, "title", title, "overlay", mask, "overlay_as", "mask", "alpha", float(alpha),
                                "overlay_color", color)
        self.messages += ["done: mask depicted" if mode == "mask" else "done: depicting overlay of mask on reference",
                          "File NOT saved yet."]
        return p

    def view_file(self, path):
        """The "view" buttons beside the file fields (dynamo_mask_GUI.m:826-841 generic_view -> dynamo_view)."""
        from . import viewers
        if not (path and os.path.isfile(path)):
            self.messages.append("file not found")
            return None
        return viewers.view_file(path)

    def save(self):
        """createAndSaveMask (S/dynamo_mask_GUI.m:368-390): dynamo_mask(args{:}), dynamo_write(mask, edit_save)."""
        args = self.read_mask_settings()
        if not args:
            return None
        mask = dynamo_mask(*args)
        if mask is None:
            self.messages.append("Mask was not produced, aborting")
            return None
        dynio.write_em(self.save_file, mask)
        self.messages.append("Mask created and saved")
        return mask


# Dynamo console shorthands (S/shorthands/dmask.m, dmollify.m, dcylinder.m)
dmask = dynamo_mask
dmollify = dynamo_mollify
dcylinder = dynamo_cylinder
