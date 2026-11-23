"""dynamo_sym of Dynamo 1.1.514: the symmetry operator used by setup, assemble, the averaging FSC and the adaptive
bandpass (docs/analysis/10a_symmetry_1.1.514.md, "10a"; S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src).

* parse_sym           dynamo_zaux_parse_sym(chain, 0) (S/dynamo_zaux_parse_sym.m:54-180; 10a 1.1)
* hsym                dynamo_hsym 'addition' method (S/dynamo_hsym.m:276-424; 10a 2.2): 2*floor(reps/2)+1 copies
                      i = -h..h, copy i = dynamo_shift_rot(V, [0 0 i*dz], [0 0 i*dphi], [max(1,floor(i*dz)) n], NaN)
                      times the equally moved cylinder [floor(n/2)-1, floor(n/2)-2] (outscope 0); voxels where the
                      moved cylinder is < 0.99 are dropped; the float64 sum (i ascending) is divided by the per-voxel
                      number of copies kept (0 where none).
* csym / dsym         dynamo_csym / dynamo_dsym (S/dynamo_csym.m:72-78, S/dynamo_dsym.m:53-78; 10a 2.3)
* dynamo_sym          dynamo_sym_one (S/dynamo_sym.m:273-613; 10a 2.1): the operator, then with an fmask the
                      symmetrized weights (local_sym_fmask :651-716, helical: dz forced to 0), the flat-weights test
                      std(Fs inside radius round(n/2-2)) < 1e-5 (MATLAB std, N-1) -> ones, else dynamo_fweight_divide
                      with threshold 1/(2R); foutput 'weights' or 'binary' (Fs >= 1/(2R)).

Resampling is dynamo_shift_rot_mex masking option 0 (implicit sphere |r| >= n/2 -> outscope, pivots outside the box
-> outscope), bit-exact: on the GPU a CUDA port of the mex arithmetic (the rot_mex0 kernel of pydynamo.averaging.average plus the
z shift and the slice range), on the CPU pydynamo.bandpass.realign.mex_shift_rot. Both give the same float64 result.

Not ported (NotImplementedError): icosahedral / cuboctahedral / multisymmetry operators, MATLAB str2num expressions
in symmetry strings, helical strings with dz <= 0 or a non-integer repetition count, non-cubic boxes.
"""
import math
import re

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

_NUMRE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_MATLAB_CONST = ("pi", "inf", "nan", "eps", "i", "j", "true", "false", "realmax", "realmin", "intmax", "intmin")


# =====================================================================================================================
# parsing
# =====================================================================================================================

def _str2num(s):
    """MATLAB str2num (eval) for a plain numeric row: optional [ ], numbers separated by commas or blanks.
    Returns a list of floats; [] for a plain unknown identifier (str2num catches the eval error); NotImplementedError
    for anything that would need MATLAB's evaluator (expressions, ranges, constants such as pi)."""
    t = s.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    if t.strip() == "":
        return []
    if re.fullmatch(r"[a-z_][a-z0-9_]*", t.strip()) and t.strip() not in _MATLAB_CONST:
        return []
    toks = t.replace(",", " ").split()
    if (t.strip().startswith(",") or t.strip().endswith(",") or ",," in t.replace(" ", "")
            or not all(_NUMRE.fullmatch(x) for x in toks)):
        raise NotImplementedError("str2num of %r (MATLAB eval) in a symmetry string" % s)
    return [float(x) for x in toks]


def parse_sym(chain):
    """dynamo_zaux_parse_sym(chain, 0) -> (kind, parameters, ok). kind in 'none', 'c', 'd', 'h', 'i', 'cbo',
    'multisymmetry'. A helical string with fewer than 2 numbers raises ValueError (MATLAB error(), :174-178)."""
    if chain is None:
        chain = ""
    if not isinstance(chain, str):
        v = float(np.ravel(chain)[0])
        chain = "c" + (str(int(v)) if v == int(v) else repr(v))            # :54-56 num2str
    if chain.isdigit():                                                      # :63-65
        chain = "c" + chain
    chain = chain.lower().strip()                                            # :66-67
    if chain == "":
        return "none", [], True                                              # empty chain: no operator
    if chain in ("i", "ico"):
        return "i", [], True
    if chain == "cbo":
        return "cbo", [], True
    if chain in ("n", "none"):
        return "none", [], True
    if "." in chain:                                                         # :97-104
        import os
        if os.path.splitext(chain)[1] == ".ms" and os.path.isfile(chain):
            return "multisymmetry", chain, True
    if len(chain) == 1:                                                      # :112-117
        return None, [], False
    k, rest = chain[0], chain[1:]
    if k in ("c", "d"):                                                      # :121-151
        p = _str2num(rest)
        if not p:
            return None, [], False
        if k == "c" and len(p) == 1 and p[0] == 1:
            return "none", p, True
        return k, p, True
    if k == "h":                                                             # :154-180
        rest = rest.replace(",,", ",0,").replace(",[],", ",0,")
        p = _str2num(rest)
        if len(p) < 2:
            raise ValueError("not enough arguments for helical symmetrization")
        return "h", p, True
    return None, [], False


def _order(p, what):
    if len(p) != 1 or p[0] != int(p[0]) or p[0] < 1:
        raise NotImplementedError("%s symmetry with parameters %s" % (what, p))
    return int(p[0])


def helical_reps(n, p):
    """dynamo_sym.m:405-440: (dphi, dz, reps) for a box with size(vol, 3) = n."""
    if len(p) > 4:
        raise NotImplementedError("helical symmetry with more than 4 parameters (o.error in dynamo_sym.m:430-432)")
    dphi, dz = float(p[0]), float(p[1])
    if not dz > 0:
        raise NotImplementedError("helical symmetry with dz = %r <= 0 (not verified)" % dz)
    reps_default = 2 * math.floor((n / dz - 2) / 2) + 1
    reps = p[2] if len(p) >= 3 else 0
    if reps == 0:
        reps = reps_default
    if reps != int(reps) or reps < 1:
        raise NotImplementedError("helical symmetry with %r repetitions (not verified)" % reps)
    return dphi, dz, int(reps)


# =====================================================================================================================
# helical operator (dynamo_hsym)
# =====================================================================================================================

def cylinder(n):
    """dynamo_cylinder([floor(n/2)-1, floor(n/2)-2], [n n n]) (S/dynamo_cylinder.m:61-122): 1-based grid, centre
    n/2+1, sqrt((x-c)^2/R^2 + (y-c)^2/R^2) <= 1 and |z-c| <= H, binary float64 ([z, y, x])."""
    rad, h = n // 2 - 1, n // 2 - 2
    pos = n / 2 + 1
    g = np.arange(1, n + 1, dtype=np.float64)
    Z, Y, X = np.meshgrid(g, g, g, indexing="ij")
    v = (np.sqrt((X - pos) ** 2 / rad ** 2 + (Y - pos) ** 2 / rad ** 2) <= 1).astype(np.float64)
    v[np.abs(Z - pos) > h] = 0
    return v


def _replist(n, dphi, dz, reps):
    max_reps = 2 * math.floor(n / dz) if dz != 0 else math.inf                  # dynamo_hsym.m:282-289
    reps = min(max_reps, reps)
    half = math.floor(reps / 2)                                                 # :299-306
    return list(range(-half, half + 1))


_HSYM_SRC = r"""
extern "C" __global__
void hsym_accum(const float* V, const int B, const float* K, const double* M, const float* SZ, const int* Z1,
                const int L, double* S, double* C, const int n)
{   // dynamo_hsym.m:359-403 per voxel, copies in ascending order. Copy i: dynamo_shift_rot_mex option 0 (implicit
    // sphere and box test -> outscope), shift (0, 0, SZ[i]) (float32), matrix M[i] (double of float32 angles), output
    // slices z >= Z1[i] computed, the others keep 0 (mex constructor). vm = moved cylinder K (outscope 0),
    // vr = moved volume (outscope NaN); vm < 0.99 -> dropped (vr := 0, C += 1); else S += double(vr) * double(vm).
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long N = (long)n * n * n;
    if (v >= N) return;
    int z = (int)(v / ((long)n * n)), y = (int)((v / n) % n), x = (int)(v % n);
    int c = n / 2;
    float rx = (float)(x - c), ry = (float)(y - c), rz = (float)(z - c);
    float dist = sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(rx, rx), __fmul_rn(ry, ry)), __fmul_rn(rz, rz)));
    bool sph = dist >= (float)(n / 2);
    double acc[4] = {0.0, 0.0, 0.0, 0.0};
    double cnt = 0.0;
    long nn = (long)n * n;
    for (int i = 0; i < L; i++) {
        bool bad = true;
        float vm = 0.f;
        long px = 0, PX = 0, py = 0, PY = 0, pz = 0, PZ = 0;
        float xf = 0.f, yf = 0.f, zf = 0.f;
        if (z >= Z1[i] && !sph) {
            const double* m = M + 9 * i;
            float orx = (float)(m[0] * (double)rx + m[1] * (double)ry + m[2] * (double)rz - 0.0);
            float ory = (float)(m[3] * (double)rx + m[4] * (double)ry + m[5] * (double)rz - 0.0);
            float orz = (float)(m[6] * (double)rx + m[7] * (double)ry + m[8] * (double)rz - (double)SZ[i]);
            int fx = (int)floor((double)orx), fy = (int)floor((double)ory), fz = (int)floor((double)orz);
            int onx = fx + c, ony = fy + c, onz = fz + c, onX = onx + 1, onY = ony + 1, onZ = onz + 1;
            if (!(onx < 0 || onx >= n || ony < 0 || ony >= n || onz < 0 || onz >= n ||
                  onX < 0 || onX >= n || onY < 0 || onY >= n || onZ < 0 || onZ >= n)) {
                xf = orx - (float)fx; yf = ory - (float)fy; zf = orz - (float)fz;
                px = onx; PX = onX; py = (long)ony * n; PY = (long)onY * n; pz = (long)onz * nn; PZ = (long)onZ * nn;
                const float* t = K;
                float vxyz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+pz], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
                float vxyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+PZ], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), zf);
                float vxYz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+pz], __fsub_rn(1.f, xf)), yf), __fsub_rn(1.f, zf));
                float vxYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+PZ], __fsub_rn(1.f, xf)), yf), zf);
                float vXyz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+pz], xf), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
                float vXyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+PZ], xf), __fsub_rn(1.f, yf)), zf);
                float vXYz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+pz], xf), yf), __fsub_rn(1.f, zf));
                float vXYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+PZ], xf), yf), zf);
                float a = __fadd_rn(vxyz, vXyz); a = __fadd_rn(a, vxYz); a = __fadd_rn(a, vxyZ);
                a = __fadd_rn(a, vXyZ); a = __fadd_rn(a, vxYZ); a = __fadd_rn(a, vXYz); a = __fadd_rn(a, vXYZ);
                vm = a;
                bad = ((double)vm < 0.99);
            }
        }
        if (bad) {
            cnt = __dadd_rn(cnt, 1.0);
            for (int b = 0; b < B; b++) acc[b] = __dadd_rn(acc[b], 0.0);
            continue;
        }
        for (int b = 0; b < B; b++) {
            const float* t = V + (long)b * N;
            float vxyz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+pz], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
            float vxyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+PZ], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), zf);
            float vxYz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+pz], __fsub_rn(1.f, xf)), yf), __fsub_rn(1.f, zf));
            float vxYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+PZ], __fsub_rn(1.f, xf)), yf), zf);
            float vXyz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+pz], xf), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
            float vXyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+PZ], xf), __fsub_rn(1.f, yf)), zf);
            float vXYz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+pz], xf), yf), __fsub_rn(1.f, zf));
            float vXYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+PZ], xf), yf), zf);
            float a = __fadd_rn(vxyz, vXyz); a = __fadd_rn(a, vxYz); a = __fadd_rn(a, vxyZ);
            a = __fadd_rn(a, vXyZ); a = __fadd_rn(a, vxYZ); a = __fadd_rn(a, vXYz); a = __fadd_rn(a, vXYZ);
            acc[b] = __dadd_rn(acc[b], __dmul_rn((double)a, (double)vm));
        }
    }
    for (int b = 0; b < B; b++) S[(long)b * N + v] = acc[b];
    C[v] = cnt;
}
"""
_hk = {}


_DEVICE = [None]          # physical GPU of the main process's symmetry work (review M2); None = CuPy's current device


def set_device(device):
    """Run the main-process GPU symmetry work (hsym, cN/dN rotations) on this physical GPU, e.g. the first GPU of the
    project, instead of device 0 (which another job may hold). None restores CuPy's current device."""
    _DEVICE[0] = None if device is None else int(device)


def _device_ctx():
    import contextlib
    if _DEVICE[0] is None or cp is None:
        return contextlib.nullcontext()
    return cp.cuda.Device(_DEVICE[0])


def _gpu_ok():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > (_DEVICE[0] or 0)
    except Exception:
        return False


def _hsym_gpu(vols, L, mats, sz, z1):
    with _device_ctx():
        return _hsym_gpu_on_current(vols, L, mats, sz, z1)


def _hsym_gpu_on_current(vols, L, mats, sz, z1):
    dev = cp.cuda.Device().id
    if (dev, "k") not in _hk:
        _hk[(dev, "k")] = cp.RawKernel(_HSYM_SRC, "hsym_accum", options=("--fmad=false",))
    n = vols[0].shape[0]
    N = n ** 3
    out_S, out_C = [], None
    for b0 in range(0, len(vols), 4):
        grp = vols[b0:b0 + 4]
        V = cp.asarray(np.stack([np.asarray(v, dtype=np.float64).astype(np.float32) for v in grp]))
        K = cp.asarray(cylinder(n).astype(np.float32))
        S = cp.empty((len(grp), N), dtype=cp.float64)
        C = cp.empty(N, dtype=cp.float64)
        _hk[(dev, "k")](((N + 127) // 128,), (128,),
                 (V, np.int32(len(grp)), K, cp.asarray(mats.ravel()), cp.asarray(sz), cp.asarray(z1),
                  np.int32(len(L)), S, C, np.int32(n)))
        out_S += [S[k].get().reshape(n, n, n) for k in range(len(grp))]
        out_C = C.get().reshape(n, n, n)
        del V, K, S, C
    cp.get_default_memory_pool().free_all_blocks()      # the project's main process shares GPU 0 with a worker
    return out_S, out_C


def _hsym_cpu(vols, L, dphi, dz):
    from pydynamo.bandpass.realign import mex_shift_rot
    n = vols[0].shape[0]
    mask = cylinder(n)
    S = [np.zeros((n, n, n)) for _ in vols]
    C = np.zeros((n, n, n))
    for i in L:
        phi, z = i * dphi, i * dz
        z1 = max(1, math.floor(z)) - 1
        vm = mex_shift_rot(mask, (0.0, 0.0, phi), (0.0, 0.0, z), option=0, outscope=0.0)
        vm[:z1] = 0
        bad = vm < 0.99
        C = C + bad
        for k, v in enumerate(vols):
            vr = mex_shift_rot(v, (0.0, 0.0, phi), (0.0, 0.0, z), option=0, outscope=np.nan)
            vr[:z1] = 0
            vr = vr * vm
            vr[bad] = 0
            S[k] = S[k] + vr
    return S, C


def hsym(vols, dphi, dz, reps, device="auto"):
    """dynamo_hsym(vol, 'phi', dphi, 'dz', dz, 'reps', reps) for each volume of `vols` (they share the mask copies).
    Returns (list of float64 volumes, number of copies)."""
    single = isinstance(vols, np.ndarray)
    vols = [vols] if single else list(vols)
    n = vols[0].shape[0]
    for v in vols:
        if v.shape != (n, n, n):
            raise NotImplementedError("helical symmetrization of a non-cubic box (dynamo_hsym.m:276 returns [])")
        if n % 2:
            raise NotImplementedError("helical symmetrization of an odd box")
    L = _replist(n, dphi, dz, reps)
    use_gpu = device == "gpu" or (device == "auto" and _gpu_ok())
    if use_gpu:
        from pydynamo.averaging.average import mex_rotation_matrix
        mats = np.array([mex_rotation_matrix(0.0, 0.0, i * dphi) for i in L], dtype=np.float64)
        sz = np.array([i * dz for i in L], dtype=np.float32)
        z1 = np.array([max(1, math.floor(i * dz)) - 1 for i in L], dtype=np.int32)
        S, C = _hsym_gpu(vols, L, mats, sz, z1)
    else:
        S, C = _hsym_cpu(vols, L, dphi, dz)
    contrib = len(L) * np.ones((n, n, n)) - C                                   # :415-424
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for s in S:
            s = s / contrib
            s[contrib == 0] = 0
            out.append(s)
    return (out[0] if single else out), len(L)


# =====================================================================================================================
# cN / dN (dynamo_csym, dynamo_dsym) and the operator dispatch
# =====================================================================================================================

def _rot0(vol, eulers, device="auto"):
    """dynamo_rot(vol, eulers): mex option 0, outscope 0 (float64 of the float32 result)."""
    if device != "cpu" and _gpu_ok():
        from pydynamo.averaging.average import rot_sphere0
        with _device_ctx():
            return rot_sphere0(vol, [eulers])[0]
    from pydynamo.bandpass.realign import mex_shift_rot
    return mex_shift_rot(vol, eulers, (0.0, 0.0, 0.0), option=0, outscope=0.0)


def csym(vol, N, device="auto"):
    """dynamo_csym.m:72-78: out = vol; out = out + dynamo_rot(vol, [(360/N)*(i-1), 0, 0]) for i = 2..N; out / N."""
    out = np.asarray(vol, dtype=np.float64)
    for i in range(2, N + 1):
        out = out + _rot0(vol, ((360.0 / N) * (i - 1), 0.0, 0.0), device)
    return out / N


def dsym(vol, N, device="auto"):
    """dynamo_dsym.m:53-78: vol' = 0.5*(vol + dynamo_rot(vol, [0 90 90])), then the cN loop on vol'."""
    v = np.asarray(vol, dtype=np.float64)
    v = 0.5 * (v + _rot0(v, (0.0, 90.0, 90.0), device))
    return csym(v, N, device)


def _apply(vol, kind, p, device, fmask_path=False):
    """The operator of dynamo_sym_one (:382-449) or of local_sym_fmask (:651-716, fmask_path) -> (vol, repetitions)."""
    if kind == "c":
        N = _order(p, "c")
        return csym(vol, N, device), N
    if kind == "d":
        N = _order(p, "d")
        return dsym(vol, N, device), 2 * N
    if kind == "h":
        n = vol.shape[2]
        dphi, dz, reps = helical_reps(n, p)
        return hsym(vol, dphi, 0.0 if fmask_path else dz, reps, device)
    raise NotImplementedError("symmetry operator of kind %r (not verified)" % kind)


def flat_weights(F):
    """dynamo_sym.m:554-561: std(F inside dynamo_ellipsoid(round(n/2-2))) < 1e-5 (MATLAB std, N-1)."""
    from pydynamo.project.runner import ellipsoid_inside
    n = F.shape[0]
    inner = ellipsoid_inside(n, math.floor(n / 2 - 2 + 0.5))
    from pydynamo.matlab.numerics import mstd
    return mstd(F[inner]) < 1e-5                                   # single-thread MATLAB std (sequential sums)


def dynamo_sym(vol, op, fmask=None, foutput="weights", device="auto", divide="allow"):
    """dynamo_sym(vol, op [, 'fmask', fmask, 'foutput', foutput]) -> (symvol float64, fout).
    kind 'none' (c1, n, none): (vol, fmask) unchanged (:382-385, trivial_checkout returns the raw fmask).
    divide='refuse': raise NotImplementedError instead of taking the dynamo_fweight_divide branch (callers whose
    weights path differs from the plain 'fmask' one and is not verified)."""
    kind, p, ok = parse_sym(op)
    if not ok:
        raise ValueError("cannot parse symmetry %r" % (op,))
    if kind == "none":
        return vol, fmask
    v = np.asarray(vol, dtype=np.float64)
    symvol, _ = _apply(v, kind, p, device)
    if fmask is None:
        return symvol, None
    F, R = _apply(np.asarray(fmask, dtype=np.float64), kind, p, device, fmask_path=True)
    thr = (1.0 / R) / 2
    if flat_weights(F):
        F = np.ones_like(F)
    else:
        if divide == "refuse":
            raise NotImplementedError("dynamo_sym %s with non-flat symmetrized weights in this context "
                                      "(Fourier division branch not verified here)" % op)
        from pydynamo.averaging.average import fweight_divide
        symvol = fweight_divide(symvol, F, thr)
    if foutput in ("weights", "fweight"):
        return symvol, F
    if foutput in ("binary", "fmask"):
        return symvol, (F >= thr).astype(np.float64)
    raise NotImplementedError("dynamo_sym foutput %r" % foutput)


def symorder(op):
    """dynamo_rigid_compare.m:73-97: the order used to fold the narot difference (column 29)."""
    kind, p, ok = parse_sym(op)
    if not ok:
        raise ValueError("cannot parse symmetry %r" % (op,))
    if kind in ("c", "d"):
        return float(_order(p, kind))
    if kind == "h":
        # MATLAB 360/0 = Inf (dphi = 0): 360/symorder = 0, mod(x, 0) = x, so column 29 = |dnarot| (rigid_compare.m:87, :113)
        return 360.0 / float(p[0]) if float(p[0]) != 0 else float("inf")
    if kind == "none" and len(p) == 1:
        return float(p[0])
    raise NotImplementedError("table comparison with symmetry %r (symorder not determined)" % (op,))
