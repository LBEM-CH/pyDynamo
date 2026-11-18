"""GPU re-implementation of Dynamo's pipeline average (docs/analysis/05_averaging_assembly.md sections 3-4).

`ordered_sums` brings particles into the reference frame and adds them, in ascending tag order, to the running sums
of their half (odd/even by position in the ascending tag list):
* S = sum of mask-normalized particles;
* C = number of NaN contributions per voxel;
* F = sum of rotated wedges / Ntags.
The project runner chains one state through its GPU workers in tag order, so each half is one sequential double
sum as in dynamo_average's single-thread MATLAB loop (Dynamo's executables run -singleCompThread); the per-particle
normalisation uses MATLAB's sequential sums (pydyn.matlab_numerics), and the FFTs are MATLAB's FFTW. Checked byte
for byte against the user's Dynamo averages (benchmarks/12_avg_order). `finalize` computes the outputs Dynamo
writes: average, average_unweighted, fweight_average_raw, fmask_average and eo_fsc.

Rotations follow dynamo_shift_rot_mex exactly (08b 4.4; mex/src/classes/volume_float.cpp:416-620): the particle with
masking option 4 (sharpShiftRot, NaN outside the box, dynamo_average.m:2663-2665), the wedge with option 0
(dynamo_rot: implicit sphere |r| >= n/2 and out-of-box -> 0, dynamo_rot.m / dynamo_shift_rot.m, mex :218-223).
Normalisation (dynamo_normalize_roi) and accumulation are in double, as in MATLAB.

Arrays are [z, y, x]; Dynamo's centre is floor(N/2) (0-based) for rotations, wedges and fftshift.
"""
import os
import time

import numpy as np
try:
    import cupy as cp
except ImportError:                     # CPU-only environments: finalize() and helpers still work
    cp = None

from . import angles as A
from .emio import read_threads  # noqa: F401  (the console daverage path and tests use it from here)
from . import matlab_numerics as MN


def _fill_wedge(n, theta_min, theta_max, eulers):
    """volume_noncubic::fillWedge with crop (mex/src/classes/volume_noncubic.cpp:477-603), float32 arithmetic."""
    F = np.float32
    pi = 3.1415926535897
    tanmax, tanmin = F(np.tan(float(F(theta_max)) * pi / 180.0)), F(np.tan(float(F(theta_min)) * pi / 180.0))
    cx = F(np.floor(F(n) / 2) + 0.5)
    r = (np.arange(n, dtype=np.float32) - cx + F(0.5)).astype(np.float32)
    rz, ry, rx = np.meshgrid(r, r, r, indexing="ij")
    m = A.euler_matrix_double(*eulers)                      # orientation(tdrot, tilt, narot).m, double
    urx = (F(m[0]) * rx + F(m[1]) * ry + F(m[2]) * rz).astype(np.float32)
    urz = (F(m[6]) * rx + F(m[7]) * ry + F(m[8]) * rz).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        uzx = urz / urx
        W = ((uzx < tanmax) & (uzx > tanmin)).astype(np.float32)
    W[(urx == 0) & (urz == 0)] = 1
    W[(urx == 0) & (urz != 0)] = 0
    rn = rx / cx, ry / cx, rz / cx
    W[(rn[0] * rn[0] + rn[1] * rn[1] + rn[2] * rn[2]) >= 1] = 0
    return W


def wedge_mask(n, row):
    """Unrotated, centred Fourier sampling of one particle, as dynamo_wedge_mask builds it from table columns 13-19:
    type 1 (tilt about y): range cols 14-15, eulers [0, 0, 0];
    type 2 (tilt about x): range cols 16-17, eulers [90, 0, 0] (dynamo_wedge_mask.m:198-199)."""
    ftype = int(row[12])
    if ftype == 0:
        W = np.ones((n, n, n), dtype=np.float32)
        c = n // 2
        r = np.arange(n, dtype=np.float32) - np.float32(c)
        rz, ry, rx = np.meshgrid(r, r, r, indexing="ij")
        W[(rx * rx + ry * ry + rz * rz) / np.float32((c + 0.5) ** 2) >= 1] = 0
        return W
    if ftype == 1:
        return _fill_wedge(n, row[13], row[14], (0.0, 0.0, 0.0))
    if ftype == 2:
        return _fill_wedge(n, row[15], row[16], (90.0, 0.0, 0.0))
    raise NotImplementedError(f"ftype {ftype} not supported by the prototype averaging")


def mex_matrices(rows):
    """Double matrices and float32 shifts of dynamo_shift_rot(p, -ptable(4:6), -ptable(9:-1:7)) for every row, as
    dynamo_shift_rot_mex builds them (sharp_shift_rot_params, 08b 4.4)."""
    params = [sharp_shift_rot_params(r) for r in np.asarray(rows, dtype=np.float64)]
    M = np.stack([p[0] for p in params]).astype(np.float64) if params else np.zeros((0, 9))
    S = np.stack([p[1] for p in params]).astype(np.float32) if params else np.zeros((0, 3), np.float32)
    return M, S


def new_state(n):
    """Running sums of both halves (odd = 0, even = 1): S = sum of normalised particles, C = NaN counts,
    F = sum of rotated wedges / Ntags; float64 [2, n^3] each, all zero (totalFmask = zeros, dynamo_average.m:2160;
    average_raw starts from the first particle, and 0 + r = r)."""
    N = n ** 3
    return {"S": np.zeros((2, N)), "C": np.zeros((2, N)), "F": np.zeros((2, N))}


def ordered_sums(particles, rows, half, nh, mask, state, batch=None):
    """Add particles to the running sums of their half, in the order given, as dynamo_average does in single-thread
    MATLAB (dynamo_average.m alignParticle :2647-2681, main loop :1974-1997, weight loop :2167-2187):
      r = sharpShiftRot(p, -t(4:6), -t(9:-1:7))     mex option 4 (NaN outside the box)
      nanmap = isnan(r); r(nanmap) = 0; r = dynamo_normalize_roi(r, mask); r(nanmap) = 0
      S += r; C += nanmap; F += dynamo_rot(wedge, -t(9:-1:7)) / Ntags
    The caller passes the particles in ascending tag order (intersect() in dynamo_average.m:1301) and chains the
    state through every GPU worker in that order, so each half is one sequential sum as in Dynamo.
    particles [P, n, n, n] float32 (host or device); rows [P, cols]; half[P] in {0, 1}; nh = (Ntags_odd,
    Ntags_even) of the whole average; mask: the project mask (find(mask) = voxels != 0). Returns the new state."""
    t0 = time.perf_counter()
    P, n = len(rows), particles.shape[1]
    N = n ** 3
    idx = cp.asarray(np.flatnonzero(np.asarray(mask).ravel() != 0))
    nvox = int(idx.size)
    keys, widx = {}, np.zeros(P, dtype=np.int64)
    for p in range(P):
        key = (int(rows[p, 12]),) + tuple(float(x) for x in rows[p, 13:19])
        widx[p] = keys.setdefault(key, len(keys))
    Wstack = cp.asarray(np.stack([wedge_mask(n, np.array([0] * 12 + list(k) + [0] * 16, dtype=np.float32)) for k in keys])) \
        if P else None
    mats, shifts = mex_matrices(rows)
    S, C, F = (cp.asarray(state[k]) for k in ("S", "C", "F"))
    nh_d = cp.asarray(np.asarray(nh, dtype=np.float64))
    half = np.asarray(half, dtype=np.int32)
    kr, kw, km, ka = (_mex_kernel(k) for k in ("shiftrot_mex4", "rot_mex0", "roi_moments", "accum_ordered"))
    if batch is None:                                   # from free GPU memory; the sums' order does not depend on it
        per = 4 * N * 4 + nvox * (4 + 8 + 8)            # in/out/wedge float32 [B, N], gathered ROI float32/64 + transpose
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        batch = max(1, min(256, int(0.5 * free) // per))
    for b0 in range(0, P, int(batch)):
        b1 = min(P, b0 + int(batch))
        B = b1 - b0
        M = cp.asarray(mats[b0:b1].ravel())
        out = cp.empty((B, n, n, n), dtype=cp.float32)
        kr(((B * N + 255) // 256,), (256,), (cp.asarray(particles[b0:b1], dtype=cp.float32), M,
                                             cp.asarray(shifts[b0:b1].ravel()), out, np.int32(n), np.int64(B * N)))
        V = out.reshape(B, N)
        X = cp.nan_to_num(V[:, idx].astype(cp.float64), copy=False, nan=0.0).T.copy()   # [nvox, B], r(nanmap) = 0
        mu, sd = cp.empty(B, cp.float64), cp.empty(B, cp.float64)
        km(((B + 127) // 128,), (128,), (X, np.int32(B), np.int64(nvox), mu, sd))
        del X
        R = cp.empty((B, n, n, n), dtype=cp.float32)
        kw(((B * N + 255) // 256,), (256,), (Wstack, cp.asarray(widx[b0:b1]), M, R, np.float32(0.0), np.int32(n),
                                             np.int64(B * N)))
        ka(((N + 255) // 256,), (256,), (V, R.reshape(B, N), mu, sd, cp.asarray(half[b0:b1]), nh_d, np.int32(B),
                                         np.int64(N), S, C, F))
        del out, R
    cp.cuda.Device().synchronize()
    return {"S": S.get(), "C": C.get(), "F": F.get(), "seconds": time.perf_counter() - t0}


def normalize_roi(v, mask):
    """dynamo_normalize_roi(v, mask) in single-thread MATLAB (pydyn.matlab_numerics)."""
    return MN.normalize_roi(v, mask)


def fweight_divide(v, F, tau):
    """dynamo_fweight_divide(v, F, tau) with MATLAB's FFTW (pydyn.matlab_numerics)."""
    return MN.fweight_divide(v, F, tau)


def fsc_fouriermask(a, b, ns):
    """dynamo_fsc_fouriermask(a, b, ns) (dynamo_fsc_fouriermask.m): shells R in [k*dr, (k+1)*dr), k = 1..ns,
    dr = (n/2)/ns, R from ndgrid(-n/2:n/2-1); sequential sums in index order; FFTW."""
    n = a.shape[0]
    u = np.arange(-n // 2, n // 2, dtype=np.float64)
    uz, uy, ux = np.meshgrid(u, u, u, indexing="ij")
    R = np.sqrt(ux * ux + uy * uy + uz * uz)
    dr = (n / 2) / ns
    shells = [(R >= k * dr) & (R < k * dr + dr) for k in range(1, ns + 1)]
    return MN.shell_fsc(a, b, shells)


def finalize(state, nh, mask, n, fmin=3.0, sym=None):
    """Dynamo's averaging outputs from the chained sums (dynamo_average.m FSC path, 05 3.5-3.8), float64, host:
    per half average_raw = normalize_roi(S ./ max(Ntags - C, 1)); fweight = F / max(F);
    average = normalize_roi(fweight_divide(average_raw, fweight, fthreshold)), fthreshold = fmin/Ntags (or 1/Ntags);
    then average_raw = 0.5 (odd + even), fweight = 0.5 (odd + even), average = fweight_divide(average_raw, fweight,
    0.1), fmask = fweight >= 1/(Ntags + 1), eo_fsc from the two half averages.
    sym: the fsc_actions 'sym <op>;' of the pipeline (genericInput2Average.m:241-258): before the FSC each half
    goes through dynamo_particle(half, 'actions', 'sym <op>;', 'fmask', binarize(half fmask, 0.5)) (fsc_local,
    dynamo_average.m:2808-2823), which is dynamo_sym(half, op, 'fmask', that fmask) [RUN: headless 1.1.514, max abs 0
    for h[-1.4,0.3] and c3 with a random binary fmask, i.e. with the Fourier division: benchmarks/06_abp_align_eo/implA/
    particle_sym.m/.log]; c1 parses as 'none' (dynamo_zaux_parse_sym.m): no action."""
    S, C, F = state["S"], state["C"], state["F"]
    if min(int(nh[0]), int(nh[1])) < 1:
        raise NotImplementedError("finalize needs particles in both halves (the even/odd FSC path of dynamo_average)")
    halves = []
    for k in (0, 1):
        nk = int(nh[k])
        D = nk * np.ones(S.shape[1]) - C[k]
        D[D == 0] = 1
        A_raw = normalize_roi((S[k] / D).reshape(n, n, n), mask)
        Fk = F[k].reshape(n, n, n)
        Fk = Fk / Fk.max()
        tau = (1.0 / nk) * fmin if fmin < nk else 1.0 / nk                  # dynamo_average.m:2215-2219
        Ak = normalize_roi(fweight_divide(A_raw, Fk, tau), mask)
        halves.append((A_raw, Fk, Ak, Fk >= tau))
    ns = n // 4
    vh = [halves[0][2], halves[1][2]]
    if sym is not None:
        from .symmetry import parse_sym, dynamo_sym
        if parse_sym(sym)[0] != "none":
            vh = [dynamo_sym(h[2], sym, (np.asarray(h[3], dtype=np.float64) >= 0.5).astype(np.float64))[0]
                  for h in halves]
    fs = fsc_fouriermask(vh[0], vh[1], ns)
    # fsc_local (dynamo_average.m:2826-2840): interp1(linspace(1, s1/2, numshells), fsc, linspace(1, s1/2, s1/2)),
    # all NaN if a shell value is NaN
    eo_fsc = (MN.interp1(MN.linspace(1, n / 2, ns), fs, MN.linspace(1, n / 2, n // 2)) if not np.isnan(fs).any()
              else np.full(n // 2, np.nan))
    A_raw = 0.5 * (halves[0][0] + halves[1][0])
    Fm = 0.5 * (halves[0][1] + halves[1][1])
    average = fweight_divide(A_raw, Fm, 0.1)
    ntags = int(nh[0]) + int(nh[1])
    fmask = (Fm >= 1.0 / (ntags + 1)).astype(np.float32)
    return {"average": average, "average_unweighted": A_raw, "fweight_average_raw": Fm,
            "fmask_average": fmask, "eo_fsc": eo_fsc, "n": [int(nh[0]), int(nh[1])]}


def halves_for(rows):
    """Odd/even half per row: position in the ascending sorted list of selected tags (col 3 == 1)."""
    sel = rows[:, 2] == 1
    tags = np.sort(rows[sel, 0])
    pos = {t: i for i, t in enumerate(tags)}
    half = np.array([pos[t] % 2 if s else -1 for t, s in zip(rows[:, 0], sel)], dtype=np.int64)
    return half


# =====================================================================================================================
# mex-exact rotation (dynamo_shift_rot_mex): option 4 (sharpShiftRot, particles: console daverage and the project
# average) and option 0 (dynamo_rot: wedges, symmetrisation).
# Spec: docs/analysis/08b_console_functions_1.1.514.md section 4.4 (mex/src/dynamo_shift_rot_mex.cpp:84-112,
# 218-223, 256-259, 297-304; mex/src/classes/volume_float.cpp:416-620; orientation.cpp:49-102;
# shift_parameters.cpp:38-43). Option 4 is checked bit-exact against Dynamo's oa.average
# (benchmarks/04_journey_bin8_align_1/dynamo/console). The older float-matrix kernels shift_rot_nan / rot_sphere0 were removed
# in the Stage 3 review (08b 4.6: 83 voxels with a different NaN count on the console average).
# =====================================================================================================================
_D2R = 0.0174532925199432954743716805978692718781530857086181640625

_MEX_SRC = r"""
extern "C" __global__
void shiftrot_mex4(const float* in, const double* M, const float* S, float* out, const int n, const long total)
{
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    long N = (long)n * n * n, b = gid / N, v = gid - b * N;
    int z = (int)(v / ((long)n * n)), y = (int)((v / n) % n), x = (int)(v % n);
    int c = n / 2;
    float rx = (float)(x - c), ry = (float)(y - c), rz = (float)(z - c);
    const double* m = M + 9 * b; const float* s = S + 3 * b;
    float orx = (float)(m[0] * (double)rx + m[1] * (double)ry + m[2] * (double)rz - (double)s[0]);
    float ory = (float)(m[3] * (double)rx + m[4] * (double)ry + m[5] * (double)rz - (double)s[1]);
    float orz = (float)(m[6] * (double)rx + m[7] * (double)ry + m[8] * (double)rz - (double)s[2]);
    int fx = (int)floor((double)orx), fy = (int)floor((double)ory), fz = (int)floor((double)orz);
    int onx = fx + c, ony = fy + c, onz = fz + c, onX = onx + 1, onY = ony + 1, onZ = onz + 1;
    if (onx < 0 || onx >= n || ony < 0 || ony >= n || onz < 0 || onz >= n ||
        onX < 0 || onX >= n || onY < 0 || onY >= n || onZ < 0 || onZ >= n) { out[gid] = __int_as_float(0x7fc00000); return; }
    float xf = orx - (float)fx, yf = ory - (float)fy, zf = orz - (float)fz;
    const float* t = in + b * N;
    long nn = (long)n * n;
    long px = onx, PX = onX, py = (long)ony * n, PY = (long)onY * n, pz = (long)onz * nn, PZ = (long)onZ * nn;
    float vxyz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+pz], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
    float vxyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+PZ], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), zf);
    float vxYz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+pz], __fsub_rn(1.f, xf)), yf), __fsub_rn(1.f, zf));
    float vxYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+PZ], __fsub_rn(1.f, xf)), yf), zf);
    float vXyz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+pz], xf), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
    float vXyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+PZ], xf), __fsub_rn(1.f, yf)), zf);
    float vXYz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+pz], xf), yf), __fsub_rn(1.f, zf));
    float vXYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+PZ], xf), yf), zf);
    float acc = __fadd_rn(vxyz, vXyz); acc = __fadd_rn(acc, vxYz); acc = __fadd_rn(acc, vxyZ);
    acc = __fadd_rn(acc, vXyZ); acc = __fadd_rn(acc, vxYZ); acc = __fadd_rn(acc, vXYz); acc = __fadd_rn(acc, vXYZ);
    out[gid] = acc;
}
extern "C" __global__
void rot_mex0(const float* W, const long long* widx, const double* M, float* out, const float outscope, const int n,
              const long total)
{   // dynamo_shift_rot_mex masking option 0 (implicitMask = 1), zero shift: volume_float.cpp:490-494 skips voxels with
    // sqrt(rx*rx+ry*ry+rz*rz) >= n/2 (value outscope), then the same arithmetic as option 4 (:519-616), outscope
    // value for pivots outside the box. dynamo_rot passes outscope 0 (dynamo_rot.m: dynamo_shift_rot(..,[z1,z2],0)).
    long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    long N = (long)n * n * n, b = gid / N, v = gid - b * N;
    int z = (int)(v / ((long)n * n)), y = (int)((v / n) % n), x = (int)(v % n);
    int c = n / 2;
    float rx = (float)(x - c), ry = (float)(y - c), rz = (float)(z - c);
    float dist = sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(rx, rx), __fmul_rn(ry, ry)), __fmul_rn(rz, rz)));
    if (dist >= (float)(n / 2)) { out[gid] = outscope; return; }
    const double* m = M + 9 * b;
    float orx = (float)(m[0] * (double)rx + m[1] * (double)ry + m[2] * (double)rz - 0.0);
    float ory = (float)(m[3] * (double)rx + m[4] * (double)ry + m[5] * (double)rz - 0.0);
    float orz = (float)(m[6] * (double)rx + m[7] * (double)ry + m[8] * (double)rz - 0.0);
    int fx = (int)floor((double)orx), fy = (int)floor((double)ory), fz = (int)floor((double)orz);
    int onx = fx + c, ony = fy + c, onz = fz + c, onX = onx + 1, onY = ony + 1, onZ = onz + 1;
    if (onx < 0 || onx >= n || ony < 0 || ony >= n || onz < 0 || onz >= n ||
        onX < 0 || onX >= n || onY < 0 || onY >= n || onZ < 0 || onZ >= n) { out[gid] = outscope; return; }
    float xf = orx - (float)fx, yf = ory - (float)fy, zf = orz - (float)fz;
    const float* t = W + widx[b] * N;
    long nn = (long)n * n;
    long px = onx, PX = onX, py = (long)ony * n, PY = (long)onY * n, pz = (long)onz * nn, PZ = (long)onZ * nn;
    float vxyz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+pz], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
    float vxyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+py+PZ], __fsub_rn(1.f, xf)), __fsub_rn(1.f, yf)), zf);
    float vxYz = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+pz], __fsub_rn(1.f, xf)), yf), __fsub_rn(1.f, zf));
    float vxYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[px+PY+PZ], __fsub_rn(1.f, xf)), yf), zf);
    float vXyz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+pz], xf), __fsub_rn(1.f, yf)), __fsub_rn(1.f, zf));
    float vXyZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+py+PZ], xf), __fsub_rn(1.f, yf)), zf);
    float vXYz = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+pz], xf), yf), __fsub_rn(1.f, zf));
    float vXYZ = __fmul_rn(__fmul_rn(__fmul_rn(t[PX+PY+PZ], xf), yf), zf);
    float acc = __fadd_rn(vxyz, vXyz); acc = __fadd_rn(acc, vxYz); acc = __fadd_rn(acc, vxyZ);
    acc = __fadd_rn(acc, vXyZ); acc = __fadd_rn(acc, vxYZ); acc = __fadd_rn(acc, vXYz); acc = __fadd_rn(acc, vXYZ);
    out[gid] = acc;
}
extern "C" __global__
void accum_serial(const float* V, double* S, double* C, const int B, const long N, const int init)
{   // per voxel, in particle order: NaN -> 0 (r(nanmap) = 0), S += double(r), C += nanmap. With init the first
    // particle initialises S and C (dynamo_average.m:1974-1997), so a -0 of the first particle is kept as in MATLAB.
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    double s = 0.0, c = 0.0;
    int b0 = 0;
    if (init) { float x = V[v]; int nn = isnan(x); s = nn ? 0.0 : (double)x; c = nn ? 1.0 : 0.0; b0 = 1; }
    else { s = S[v]; c = C[v]; }
    for (int b = b0; b < B; b++) {
        float x = V[(long)b * N + v];
        int nn = isnan(x);
        s = __dadd_rn(s, nn ? 0.0 : (double)x);
        c = __dadd_rn(c, nn ? 1.0 : 0.0);
    }
    S[v] = s; C[v] = c;
}
extern "C" __global__
void roi_moments(const double* X, const int B, const long n, double* mu, double* sd)
{   // dynamo_normalize_roi_one in single-thread MATLAB: X is [n, B] (mask voxels in index order, one column per
    // particle); miroi = sum(iroi)/n, y = iroi - miroi, stdiroi = sqrt(sum(abs(y - sum(y)/n).^2)/(n-1)); every sum
    // sequential from 0, one rounding per operation.
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    double s = 0.0;
    for (long k = 0; k < n; k++) s = __dadd_rn(s, X[k * B + b]);
    double m = __ddiv_rn(s, (double)n);
    double s2 = 0.0;
    for (long k = 0; k < n; k++) s2 = __dadd_rn(s2, __dsub_rn(X[k * B + b], m));
    double yb = __ddiv_rn(s2, (double)n);
    double s3 = 0.0;
    for (long k = 0; k < n; k++) { double d = __dsub_rn(__dsub_rn(X[k * B + b], m), yb); s3 = __dadd_rn(s3, __dmul_rn(d, d)); }
    mu[b] = m;
    sd[b] = n > 1 ? __dsqrt_rn(__ddiv_rn(s3, (double)(n - 1))) : 0.0;
}
extern "C" __global__
void accum_ordered(const float* V, const float* R, const double* mu, const double* sd, const int* half,
                   const double* nh, const int B, const long N, double* S, double* C, double* F)
{   // dynamo_average.m main loop (:1974-1997) and weight loop (:2167-2187) per half, particles in the given
    // (ascending tag) order: r = sharpShiftRot(p) (V, NaN outside); r(nan) = 0; r = (r - miroi) ./ stdiroi;
    // r(nan) = 0; average_raw += r; average_nan += nan; totalFmask += fmask / Ntags (R = rotated wedge).
    long v = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    double s0 = S[v], s1 = S[N + v], c0 = C[v], c1 = C[N + v], f0 = F[v], f1 = F[N + v];
    for (int b = 0; b < B; b++) {
        float x = V[(long)b * N + v];
        int nn = isnan(x);
        double r = nn ? 0.0 : __ddiv_rn(__dsub_rn((double)x, mu[b]), sd[b]);
        double w = __ddiv_rn((double)R[(long)b * N + v], nh[half[b]]);
        if (half[b] == 0) { s0 = __dadd_rn(s0, r); c0 = __dadd_rn(c0, nn ? 1.0 : 0.0); f0 = __dadd_rn(f0, w); }
        else              { s1 = __dadd_rn(s1, r); c1 = __dadd_rn(c1, nn ? 1.0 : 0.0); f1 = __dadd_rn(f1, w); }
    }
    S[v] = s0; S[N + v] = s1; C[v] = c0; C[N + v] = c1; F[v] = f0; F[N + v] = f1;
}
"""
_mex_k = {}


def _mex_kernel(name):
    if name not in _mex_k:
        _mex_k[name] = cp.RawKernel(_MEX_SRC, name, options=("--fmad=false",))
    return _mex_k[name]


def mex_rotation_matrix(tdrot, tilt, narot):
    """orientation(double, double, double): angles cast to float, matrix filled in double with glibc cos/sin
    (Python's math module), products left to right (08b 4.4)."""
    import math
    a, b, c = (float(np.float32(v)) for v in (tdrot, tilt, narot))
    cotd, sitd = math.cos(a * _D2R), math.sin(a * _D2R)
    coti, siti = math.cos(b * _D2R), math.sin(b * _D2R)
    cona, sina = math.cos(c * _D2R), math.sin(c * _D2R)
    return [cotd * cona - sitd * coti * sina, - cona * sitd - cotd * coti * sina, sina * siti,
            cotd * sina + cona * sitd * coti, cotd * cona * coti - sitd * sina, -cona * siti,
            sitd * siti, cotd * siti, coti]


def rot_sphere0(vol, eulers):
    """dynamo_rot(vol, eulers) on the GPU: dynamo_shift_rot_mex masking option 0 with outscope 0 (dynamo_rot.m,
    dynamo_shift_rot.m; mex-exact kernel average.rot_mex0: double matrix of float32(eulers), implicit sphere
    |r| >= n/2 -> 0, pivots outside the box -> 0). The volume enters as float32, as the mex converts it."""
    n = vol.shape[0]
    N = n ** 3
    E = np.atleast_2d(np.asarray(eulers, dtype=np.float64))
    mats = np.array([mex_rotation_matrix(*e) for e in E], dtype=np.float64)
    out = cp.empty((len(E), n, n, n), dtype=cp.float32)
    _mex_kernel("rot_mex0")(((len(E) * N + 255) // 256,), (256,),
                            (cp.asarray(vol, dtype=cp.float32), cp.zeros(len(E), dtype=cp.int64), cp.asarray(mats.ravel()),
                             out, np.float32(0.0), np.int32(n), np.int64(len(E) * N)))
    return out.get().astype(np.float64)


def sharp_shift_rot_params(row):
    """dynamo_shift_rot(p, -ptable(4:6), -ptable(9:-1:7), [], [], 0) (sharpShiftRot.m:13): the double matrix of
    e = (-t9, -t8, -t7) and the float32 shift s = (-t4, -t5, -t6)."""
    m = mex_rotation_matrix(-row[8], -row[7], -row[6])
    s = np.array([-row[3], -row[4], -row[5]], dtype=np.float32)
    return np.array(m, dtype=np.float64), s


def shift_rot_mex_cpu(vol, m, s, implicit_sphere=False, outscope=np.nan):
    """CPU (numpy) version of shiftrot_mex4 (option 4, outscope NaN) with the same operation order: [z, y, x] float32
    in and out. implicit_sphere=True, outscope=0 and s=0 give rot_mex0 (option 0, dynamo_rot)."""
    F = np.float32
    n = vol.shape[0]
    c = n // 2
    g = (np.arange(n) - c).astype(F)
    rz, ry, rx = np.meshgrid(g, g, g, indexing="ij")
    rx, ry, rz = rx.astype(np.float64), ry.astype(np.float64), rz.astype(np.float64)
    orx = (m[0] * rx + m[1] * ry + m[2] * rz - float(s[0])).astype(F)
    ory = (m[3] * rx + m[4] * ry + m[5] * rz - float(s[1])).astype(F)
    orz = (m[6] * rx + m[7] * ry + m[8] * rz - float(s[2])).astype(F)
    fx = np.floor(orx.astype(np.float64)).astype(np.int64)
    fy = np.floor(ory.astype(np.float64)).astype(np.int64)
    fz = np.floor(orz.astype(np.float64)).astype(np.int64)
    onx, ony, onz = fx + c, fy + c, fz + c
    out_ = (onx < 0) | (onx + 1 >= n) | (ony < 0) | (ony + 1 >= n) | (onz < 0) | (onz + 1 >= n)
    onx, ony, onz = np.clip(onx, 0, n - 2), np.clip(ony, 0, n - 2), np.clip(onz, 0, n - 2)
    xf, yf, zf = orx - fx.astype(F), ory - fy.astype(F), orz - fz.astype(F)
    one = F(1)
    ux, uy, uz = one - xf, one - yf, one - zf
    t = vol.astype(F, copy=False)
    T = lambda dz, dy, dx: t[onz + dz, ony + dy, onx + dx]
    vxyz = T(0, 0, 0) * ux * uy * uz
    vxyZ = T(1, 0, 0) * ux * uy * zf
    vxYz = T(0, 1, 0) * ux * yf * uz
    vxYZ = T(1, 1, 0) * ux * yf * zf
    vXyz = T(0, 0, 1) * xf * uy * uz
    vXyZ = T(1, 0, 1) * xf * uy * zf
    vXYz = T(0, 1, 1) * xf * yf * uz
    vXYZ = T(1, 1, 1) * xf * yf * zf
    acc = vxyz + vXyz
    acc = acc + vxYz
    acc = acc + vxyZ
    acc = acc + vXyZ
    acc = acc + vxYZ
    acc = acc + vXYz
    acc = acc + vXYZ
    acc[out_] = outscope
    if implicit_sphere:                                   # volume_float.cpp:490-494
        acc[np.sqrt(rx * rx + ry * ry + rz * rz).astype(F) >= F(n // 2)] = outscope
    return acc.astype(F)


def console_average_sums(read_particle, rows, n, device="auto", batch=512):
    """The console daverage backend (pydyn.console.daverage): serial-order sums of dynamo_average's default console
    path (08b 4.2): for each row (ascending tags), rotate
    the particle with the mex-exact kernel, NaN -> 0, S += r, C += isnan(r). Returns float64 [z, y, x] S and C.
    read_particle(i) returns the float32 [z, y, x] particle of rows[i]."""
    P = len(rows)
    params = [sharp_shift_rot_params(r) for r in rows]
    use_gpu = cp is not None and device != "cpu"
    if use_gpu:
        try:
            use_gpu = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            use_gpu = False
    if device == "gpu" and not use_gpu:
        raise RuntimeError("daverage: no GPU available")
    N = n ** 3
    if not use_gpu:
        S = np.zeros((n, n, n))
        C = np.zeros((n, n, n))
        for i in range(P):
            r = shift_rot_mex_cpu(read_particle(i), *params[i])
            nan = np.isnan(r)
            x = np.where(nan, 0.0, r.astype(np.float64))
            if i == 0:
                S, C = x, nan.astype(np.float64)
            else:
                S = S + x
                C = C + nan
        return S, C
    from concurrent.futures import ThreadPoolExecutor
    S = cp.zeros(N, dtype=cp.float64)
    C = cp.zeros(N, dtype=cp.float64)
    kr, ka = _mex_kernel("shiftrot_mex4"), _mex_kernel("accum_serial")
    with ThreadPoolExecutor(read_threads()) as ex:
        for b0 in range(0, P, batch):
            b1 = min(P, b0 + batch)
            B = b1 - b0
            vin = cp.asarray(np.stack(list(ex.map(read_particle, range(b0, b1)))).astype(np.float32))
            M = cp.asarray(np.stack([params[i][0] for i in range(b0, b1)]).ravel())
            Sh = cp.asarray(np.stack([params[i][1] for i in range(b0, b1)]).ravel())
            out = cp.empty((B, n, n, n), dtype=cp.float32)
            kr(((B * N + 255) // 256,), (256,), (vin, M, Sh, out, np.int32(n), np.int64(B * N)))
            ka(((N + 255) // 256,), (256,), (out, S, C, np.int32(B), np.int64(N), np.int32(b0 == 0)))
            del vin, out
    cp.cuda.Device().synchronize()
    return S.get().reshape(n, n, n), C.get().reshape(n, n, n)
