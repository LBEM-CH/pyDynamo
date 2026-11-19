"""dynamo_align (Dynamo 1.1.514, CPU 'pipeline' destination) as used by the golden-standard adaptive bandpass:
dynamo_align(particle, template, 'settings_from', project, 'settings_round', round), plus the mex rotation
kernels and geometry helpers it needs.

Specification: docs/analysis/09b_adaptive_bandpass_1.1.514.md 4.3 and docs/analysis/03_alignment_math.md (CPU
motor, written against 1.1.558; the files used here are byte-identical in 1.1.514: dynamo_align.m,
dynamo__align_motor.m, dynamo_peak_subpixel.m, dynamo_angleset.m, dynamo__multigrid_convention.m,
dynamo_normalize_roi.m, dynamo_shift_rot*.m, dynamo_rigid*.m, dynamo_table_align.m, dynamo_euler2matrix.m,
dynamo_matrix2euler.m). S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src, MEX = external/dynamo-1.1.558/mex/src.

Arrays are numpy [z, y, x] (Dynamo x = MATLAB dim 1 = numpy last axis). Column-major MATLAB order (find, max(:))
is C order of a [z, y, x] array.

Precision: the rotation kernels follow the mex operation by operation (float32, double matrix from glibc cos/sin of
the float32 angles, no FMA). Everything else is double with MATLAB R2019a's own numerics: sequential sums
(pydyn.matlab_numerics), FFTW fftn / ifftn as MATLAB calls it (a conjugate-symmetric spectrum takes MATLAB's real
inverse), MATLAB's complex multiply and conj, fdlibm sin/cos/atan2/acos/pow and MATLAB's atan2d builtin
(pydyn.fdlibm), MATLAB's colon operator, and the sub-pixel fit in real arithmetic. MATLAB's sums depend on its
thread count (pydyn.matlab_numerics.msum): Dynamo runs this alignment (adaptive bandpass) inside dynamo_execute_project,
with one MATLAB thread per core of the job, so `threads` must be that count (the project runner's matlab_threads).
Measured bit-identical to Dynamo 1.1.514 (ptable, ccmax, every per-angle score, every angle list): the stored .sal of
the user's bin8 run and of the multi-tomogram run (64 threads, iterations 1-4), a box-48 project run here (16
threads), headless dynamo_align with maxNumCompThreads(1) on both box-64 projects, and a synthetic pair
(tests/test_matlab_exact.py). A negative radicand in the sub-pixel fit is evaluated with Python complex powers (not
verified).

Ported branches: area_search_modus 0 (localnc 1) and 2/3/4 (symmetric: dynamo_move; asymmetric: dynamo_rot_shift),
use_CC 0/1, no 'band', no 'dim', no 'actions', no table (dynamo_table_blank), full Fourier sampling. Everything else
raises NotImplementedError.
"""
import math

import numpy as np

from . import fdlibm as _F
from . import matlab_numerics as MN

F32 = np.float32
_D2R = 0.017453292519943295


# =====================================================================================================================
# MATLAB helpers
# =====================================================================================================================

def matlab_colon(a, d, b):
    """a:d:b of MATLAB (double), the algorithm MathWorks documents for the colon operator (colonop, C. Moler,
    "The colon operator", 2019): n = round((b-a)/d), reduced by one if a+n*d overshoots b by more than
    tol = 2*eps*max(|a|,|b|); the right end c = a+n*d is snapped to b within tol; elements a+k*d from the left and
    c-k*d from the right. For even n the middle element of MATLAB R2019a's builtin is a + (c-a)/2 when a and c have
    the same sign and (a+c)/2 otherwise (measured: 0 of 20,000 random ranges differ; Moler's (a+c)/2 alone differs
    in 72)."""
    eps = 2.220446049250313e-16
    if d == 0 or (d > 0 and a > b) or (d < 0 and a < b) or any(map(math.isnan, (a, d, b))):
        return np.zeros(0)
    tol = 2.0 * eps * max(abs(a), abs(b))
    sig = 1.0 if d > 0 else -1.0
    n = int(math.floor((b - a) / d + 0.5)) if (b - a) / d >= 0 else int(round((b - a) / d))
    if sig * (a + n * d - b) > tol:
        n -= 1
    c = a + n * d
    if sig * (c - b) > -tol:
        c = b
    v = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    v[k] = a + k * d
    v[n - k] = c - k * d
    if n % 2 == 0:
        v[n // 2] = a + (c - a) / 2 if a * c > 0 else (a + c) / 2
    return v


class _MatlabTrig:
    """MATLAB-side scalar trigonometry as MATLAB R2019a computes it (fdlibm, pydyn.fdlibm: bit-identical), for every
    MATLAB-side angle computation (euler2matrix, matrix2euler, angleset, table transforms; tests/test_eo_abp.py).
    The C library's sin/cos/atan2/acos differ from MATLAB's in the last bit for some arguments."""
    sin, cos, atan2, acos = staticmethod(_F.sin), staticmethod(_F.cos), staticmethod(_F.atan2), staticmethod(_F.acos)
    name = "fdlibm"


_T = _MatlabTrig()


def mm3(a, b):
    """MATLAB a*b for 3x3 (or 3x3 times 3-vector) doubles, summed left to right without FMA ([RUN]: with the MATLAB
    trig this reproduces Dynamo's table transforms bit for bit)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        return np.array([(a[i, 0] * b[0] + a[i, 1] * b[1]) + a[i, 2] * b[2] for i in range(3)])
    return np.array([[(a[i, 0] * b[0, j] + a[i, 1] * b[1, j]) + a[i, 2] * b[2, j] for j in range(3)]
                     for i in range(3)])


def euler2matrix(tdrot, tilt, narot):
    """dynamo_euler2matrix.m:119-127 (double)."""
    td, ti, na = tdrot * math.pi / 180, tilt * math.pi / 180, narot * math.pi / 180
    ctd, cna, cti = _T.cos(td), _T.cos(na), _T.cos(ti)
    std, sna, sti = _T.sin(td), _T.sin(na), _T.sin(ti)
    return np.array([[ctd * cna - std * cti * sna, - cna * std - ctd * cti * sna, sna * sti],
                     [ctd * sna + cna * std * cti, ctd * cna * cti - std * sna, -cna * sti],
                     [std * sti, ctd * sti, cti]])


def matrix2euler(rm):
    """dynamo_matrix2euler.m:67-100."""
    tol = 1e-4
    if abs(rm[2, 2] - 1) < tol:
        return 0.0, 0.0, _T.atan2(rm[1, 0], rm[0, 0]) * 180 / math.pi
    if abs(rm[2, 2] + 1) < tol:
        return 0.0, 180.0, _T.atan2(rm[1, 0], rm[0, 0]) * 180 / math.pi
    tdrot = _T.atan2(rm[2, 0], rm[2, 1])
    tilt = _T.acos(rm[2, 2])
    narot = _T.atan2(rm[0, 2], -rm[1, 2])
    return tdrot * 180 / math.pi, tilt * 180 / math.pi, narot * 180 / math.pi


def compose_rotations(e1, e2):
    """dynamo_compose_rotations.m: euler of M(e1) * M(e2)."""
    return matrix2euler(mm3(euler2matrix(*e1), euler2matrix(*e2)))


# =====================================================================================================================
# angles (dynamo_angleset -> dynamo_angleincrement2list)
# =====================================================================================================================

def _best_narot(eu1, eu2):
    """+dpkgeom/+angles/closeFormulaBestTdrotForIncrementList.m."""
    td1, ti1, na1 = (x * math.pi / 180 for x in eu1)
    td2, ti2 = eu2[0] * math.pi / 180, eu2[1] * math.pi / 180
    alpha = _T.cos(td1) * _T.cos(na1) - _T.sin(td1) * _T.cos(ti1) * _T.sin(na1)
    beta = _T.cos(td1) * _T.sin(na1) + _T.sin(td1) * _T.cos(ti1) * _T.cos(na1)
    A = alpha * _T.cos(td2) + beta * _T.sin(td2) * _T.cos(ti2)
    B = beta * _T.cos(td2) - alpha * _T.sin(td2) * _T.cos(ti2)
    return _F.atan2d(B, A)                                       # MATLAB's atan2d builtin (pydyn.fdlibm)


def angleincrement2list(aperture, interval, inplane_range=0, inplane_sample=1, old_angles=(0, 0, 0)):
    """dynamo_angleincrement2list.m (rings = 0)."""
    old = [float(x) for x in old_angles]
    if old[1] < 0 or old[1] > 180:
        old = list(matrix2euler(euler2matrix(*old)))
    td_old, ti_old, na_old = old
    tilts = matlab_colon(0.0, float(interval), aperture / 2)
    if aperture == 0:
        tilts = np.zeros(1)
    angles = []
    for tilt_north in tilts:
        if tilt_north == 0 or tilt_north == 180:
            tdrots = [0.0]
        else:
            c, s = _T.cos(tilt_north * math.pi / 180), _T.sin(tilt_north * math.pi / 180)
            arg = (_T.cos(interval * math.pi / 180) - c * c) / (s * s)       # MATLAB c^2 = c*c
            if arg > 1 or arg < -1:                                   # imag(acos) ~= 0
                print("correcting numerical instability generating angles, probably the south  pole")
                tdrots = [0.0]
            else:
                max_interval = _T.acos(arg) * 180 / math.pi
                nsec = math.floor(360 / max_interval) + 1
                tdrots = list(MN.linspace(0, 360, nsec + 1)[:-1])           # MATLAB linspace
        for td_n in tdrots:
            td, ti, na_t = compose_rotations((td_n, tilt_north, 0.0), (td_old, ti_old, 0.0))
            seed = _best_narot((td_old, ti_old, na_old), (td, ti, na_t))
            pos = matlab_colon(seed, float(inplane_sample), seed + inplane_range / 2)
            neg = matlab_colon(seed - inplane_sample, -float(inplane_sample), seed - inplane_range / 2)
            nar = list(pos) + list(neg)
            if not nar:
                nar = [seed]
            for na in nar:
                if angles:
                    m = euler2matrix(td, ti, na)
                    mo = euler2matrix(*angles[-1])
                    if np.max(np.abs(mo - m)) > 1e-8:
                        angles.append((td, ti, na))
                else:
                    angles.append((td, ti, na))
    return np.array(angles, dtype=np.float64).reshape(-1, 3)


def angleset(cone_range, cone_sampling, inplane_range, inplane_sampling, old_angles=(0, 0, 0), cone_flip=False,
             inplane_flip=False):
    """dynamo_angleset.m: increment list, then the in-plane flip copies, then the cone flip copies."""
    a = angleincrement2list(cone_range, cone_sampling, inplane_range, inplane_sampling, old_angles)
    if inplane_flip:
        f = a.copy()
        f[:, 2] += 180
        a = np.concatenate([a, f])
    if cone_flip:
        f = a.copy()
        f[:, 1] += 180
        a = np.concatenate([a, f])
    return a


def multigrid(refine_factor, cone_range, cone_sampling, inplane_range, inplane_sampling):
    """dynamo__multigrid_convention.m."""
    return (refine_factor * cone_sampling if cone_range != 0 else 0, cone_sampling / 2,
            refine_factor * inplane_sampling if inplane_range != 0 else 0, inplane_sampling / 2)


# =====================================================================================================================
# mex rotation kernels (MEX/classes/volume_float.cpp)
# =====================================================================================================================

def mex_matrix(eulers):
    """orientation(double tdrot, tilt, narot) (MEX/classes/orientation.cpp:49-58, 94-102): angles cast to float,
    matrix in double from glibc cos/sin (08b 4.4)."""
    a, b, c = (float(F32(v)) for v in eulers)
    cotd, sitd = math.cos(a * _D2R), math.sin(a * _D2R)
    coti, siti = math.cos(b * _D2R), math.sin(b * _D2R)
    cona, sina = math.cos(c * _D2R), math.sin(c * _D2R)
    return [cotd * cona - sitd * coti * sina, - cona * sitd - cotd * coti * sina, sina * siti,
            cotd * sina + cona * sitd * coti, cotd * cona * coti - sitd * sina, -cona * siti,
            sitd * siti, cotd * siti, coti]


def mex_shift_rot(vol, eulers, shifts=(0.0, 0.0, 0.0), option=0, outscope=0.0, mask=None):
    """dynamo_shift_rot_mex(vol, shifts, eulers, slices, outscope, border 0, option, mask) for a cubic even box.
    option 0: implicit sphere (output r >= n/2 -> outscope) + box test (shift_and_rotation_of, implicitMask 1);
    option 4: box test only (implicitMask 0); option 1: 'from mask' (from_mask_shift_and_rotation_of): a voxel is
    interpolated only if its 8 pivots are in the box and in (bool)mask, else outscope; returns (vol, rotated mask).
    Output double (from float)."""
    n = vol.shape[0]
    if vol.shape != (n, n, n) or n % 2:
        raise NotImplementedError("mex rotation of a non-cubic or odd box")
    if option not in (0, 1, 4):
        raise NotImplementedError("dynamo_shift_rot_mex option %r" % option)
    m = mex_matrix(eulers)
    s = [float(F32(x)) for x in shifts]
    c = n // 2
    g = np.arange(n, dtype=np.float64) - c
    rz, ry, rx = np.meshgrid(g, g, g, indexing="ij")
    orx = (m[0] * rx + m[1] * ry + m[2] * rz - s[0]).astype(F32)
    ory = (m[3] * rx + m[4] * ry + m[5] * rz - s[1]).astype(F32)
    orz = (m[6] * rx + m[7] * ry + m[8] * rz - s[2]).astype(F32)
    fx = np.floor(orx).astype(np.int64)
    fy = np.floor(ory).astype(np.int64)
    fz = np.floor(orz).astype(np.int64)
    onx, ony, onz = fx + c, fy + c, fz + c
    out_ = (onx < 0) | (onx + 1 >= n) | (ony < 0) | (ony + 1 >= n) | (onz < 0) | (onz + 1 >= n)
    onx, ony, onz = np.clip(onx, 0, n - 2), np.clip(ony, 0, n - 2), np.clip(onz, 0, n - 2)
    xf, yf, zf = orx - fx.astype(F32), ory - fy.astype(F32), orz - fz.astype(F32)
    one = F32(1)
    ux, uy, uz = one - xf, one - yf, one - zf
    t = np.asarray(vol, dtype=np.float64).astype(F32)
    T = lambda a, dz, dy, dx: a[onz + dz, ony + dy, onx + dx]
    acc = T(t, 0, 0, 0) * ux * uy * uz                                        # vxyz
    acc = acc + T(t, 0, 0, 1) * xf * uy * uz                                  # + vXyz
    acc = acc + T(t, 0, 1, 0) * ux * yf * uz                                  # + vxYz
    acc = acc + T(t, 1, 0, 0) * ux * uy * zf                                  # + vxyZ
    acc = acc + T(t, 1, 0, 1) * xf * uy * zf                                  # + vXyZ
    acc = acc + T(t, 1, 1, 0) * ux * yf * zf                                  # + vxYZ
    acc = acc + T(t, 0, 1, 1) * xf * yf * uz                                  # + vXYz
    acc = acc + T(t, 1, 1, 1) * xf * yf * zf                                  # + vXYZ
    o = F32(outscope)
    if option == 1:
        mb = np.asarray(mask) != 0                                            # copy_mask_from_matlab: (bool)
        inside = ~out_
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    inside &= T(mb, dz, dy, dx)
        acc = np.where(inside, acc, o)
        return acc.astype(np.float64), inside.astype(np.float64)
    acc = np.where(out_, o, acc)
    if option == 0:
        r = np.sqrt((rx * rx + ry * ry + rz * rz).astype(F32))               # float sqrt of exact integers
        acc = np.where(r >= F32(n / 2), o, acc)
    return acc.astype(np.float64)


def rot_shift(vol, eulers, shifts, outscope):
    """dynamo_rot_shift.m (5 args): shifts_transformed = (M(eulers) * shifts')', then the mex with option 0."""
    st = mm3(euler2matrix(*eulers), np.asarray(shifts, dtype=np.float64))
    return mex_shift_rot(vol, eulers, st, option=0, outscope=outscope)


# =====================================================================================================================
# geometry
# =====================================================================================================================

def ellipsoid(radii, n):
    """dynamo_ellipsoid(radii, n) -> dpkgeom.sphere (mex) with centre n/2+1 (1-based) and no smoothing: voxel inside
    when sum(((x - n/2)/r_x)^2 + ...) <= 1 (0-based; x = MATLAB dim 1)."""
    r = np.atleast_1d(np.asarray(radii, dtype=np.float64))
    if r.size == 1:
        r = np.repeat(r, 3)
    if n % 2:
        raise NotImplementedError("dynamo_ellipsoid in an odd box")
    if np.prod(r) == 0:
        return np.zeros((n, n, n))
    g = np.arange(n, dtype=np.float64) - n // 2
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    return (((x / r[0]) ** 2 + (y / r[1]) ** 2 + (z / r[2]) ** 2) <= 1).astype(np.float64)


def normalize_whole(v, threads=1):
    """dynamo_normalize_roi(v) without mask: dynamo_normalize_roi_one is called twice (dynamo_normalize_roi.m,
    the no-mask branch), each time (v - mean) / std(v - mean) over all non-NaN voxels, with MATLAB's sums for
    `threads` computational threads."""
    ones = np.ones(np.shape(v))
    return MN.normalize_roi(MN.normalize_roi(v, ones, threads), ones, threads)


# =====================================================================================================================
# peak (dynamo_peak_subpixel.m)
# =====================================================================================================================

def _rdiv(a, b):
    """MATLAB real scalar a / b (IEEE: x/0 = +-Inf, 0/0 = NaN)."""
    if b != 0:
        return a / b
    if a == 0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _cdiv(a, b):
    """MATLAB complex / real scalar: each part divided (b real)."""
    return complex(_rdiv(a.real, b), _rdiv(a.imag, b))


def _subpixel_direction(y1, y2, y3):
    """dynamo_peak_subpixel_direction (S/dynamo_peak_subpixel.m) with MATLAB's scalar arithmetic: x^2 = x*x, x^(1/2)
    and x^(3/2) = fdlibm pow (pow(x, 0.5) is sqrt(x)), left to right as written. Candidates 1-2 use only
    z1^(1/2), candidates 3-4 only z2^(1/2) and z2^(3/2); a pair is complex in MATLAB only when its own radicand is
    negative, and its values then have a nonzero imaginary part and are set to -Inf (ind_imag). That complex pair is
    evaluated here with Python complex powers, only to test the imaginary part (not verified bit for bit)."""
    if y1 == y3:
        return y2, 0.0
    if abs(y2 - y3) >= 2 or abs(y2 - y1) >= 2:
        return y2, 0.0
    z1 = (- (y1 * y1) / 4 + (y1 * y3) / 2 + y2 * y2 - 2 * y2 * y3 + (3 * (y3 * y3)) / 4)
    z2 = ((3 * (y1 * y1)) / 4 - 2 * y1 * y2 + (y1 * y3) / 2 + y2 * y2 - (y3 * y3) / 4)
    d1, d2 = (y1 + 2 * y2 - 3 * y3), (2 * y2 - 3 * y1 + y3)
    if z1 >= 0:
        s1 = _F.pow(z1, 0.5)
        pv = [_rdiv(-(2 * s1), d1), _rdiv(2 * s1, d1)]
        v = [_rdiv(2 * y1 * y2 - 3 * y1 * y3 + y1 * s1 - 2 * y2 * s1 + y3 * s1 + y1 * y1, d1),
             _rdiv(2 * y1 * y2 - 3 * y1 * y3 - y1 * s1 + 2 * y2 * s1 - y3 * s1 + y1 * y1, d1)]
    else:
        s1 = complex(z1) ** 0.5
        pv = [_cdiv(-(2 * s1), d1), _cdiv(2 * s1, d1)]
        v = [_cdiv(2 * y1 * y2 - 3 * y1 * y3 + y1 * s1 - 2 * y2 * s1 + y3 * s1 + y1 * y1, d1),
             _cdiv(2 * y1 * y2 - 3 * y1 * y3 - y1 * s1 + 2 * y2 * s1 - y3 * s1 + y1 * y1, d1)]
    if z2 >= 0:
        s2, t2 = _F.pow(z2, 0.5), _F.pow(z2, 1.5)
        pv += [_rdiv(2 * y2 - 3 * y1 + y3 + 2 * s2, d2), _rdiv(-(3 * y1 - 2 * y2 - y3 + 2 * s2), d2)]
        v += [y3 - s2 / 2 + _rdiv(2 * t2, d2 * d2) - _rdiv(2 * (y2 - y3) * s2, d2),
              y3 + s2 / 2 - _rdiv(2 * t2, d2 * d2) + _rdiv(2 * (y2 - y3) * s2, d2)]
    else:
        s2, t2 = complex(z2) ** 0.5, complex(z2) ** 1.5
        pv += [_cdiv(2 * y2 - 3 * y1 + y3 + 2 * s2, d2), _cdiv(-(3 * y1 - 2 * y2 - y3 + 2 * s2), d2)]
        v += [y3 - s2 / 2 + _cdiv(2 * t2, d2 * d2) - _cdiv(2 * (y2 - y3) * s2, d2),
              y3 + s2 / 2 - _cdiv(2 * t2, d2 * d2) + _cdiv(2 * (y2 - y3) * s2, d2)]
    vals = []
    for vi, pi in zip(v, pv):
        vi, pi = complex(vi), complex(pi)
        x = vi.real if (vi.imag == 0 and math.isfinite(vi.real)) else -math.inf
        if pi.real < 0 or pi.real > 1:                                 # MATLAB compares real parts
            x = -math.inf
        vals.append(x)
    ip = int(np.argmax(vals))                                          # max: first of equal values
    pos = complex(pv[ip])
    if ip < 2:
        pos = pos - 1
    if pos.imag != 0:
        raise NotImplementedError("complex sub-pixel position (all candidates invalid)")
    return vals[ip], pos.real


def peak_subpixel(vol, vol0=None):
    """[position (1-based [x,y,z]), value] = dynamo_peak_subpixel(vol, vol0) on a [z, y, x] array."""
    if vol0 is None:
        vol0 = vol
    i = int(np.argmax(vol))                                            # max(volume(:)): first maximum
    if np.isnan(vol.flat[i]):
        raise NotImplementedError("NaN maximum in dynamo_peak_subpixel")
    z, y, x = np.unravel_index(i, vol.shape)
    pint = [x + 1, y + 1, z + 1]
    pval = float(vol.flat[i])
    sizes = [vol.shape[2], vol.shape[1], vol.shape[0]]
    pos, vals = [], []
    for d in range(3):
        if 1 < pint[d] < sizes[d]:
            plus, minus = list(pint), list(pint)
            plus[d] += 1
            minus[d] -= 1
            yp = float(vol0[plus[2] - 1, plus[1] - 1, plus[0] - 1])
            ym = float(vol0[minus[2] - 1, minus[1] - 1, minus[0] - 1])
        else:
            yp = ym = pval
        v, s = _subpixel_direction(ym, pval, yp)
        vals.append(v)
        pos.append(pint[d] + s)
    return np.array(pos), max(vals)


# =====================================================================================================================
# correlation (dynamo__align_motor.m local_correlation_general, :537-637)
# =====================================================================================================================

def local_correlation(small_template, small_mask, conjP, conjP2, want_matrix, threads=1):
    """local_correlation_general with MATLAB's numerics (pydyn.matlab_numerics: sequential mean and sum, FFTW
    fftn / ifftn as MATLAB calls it, MATLAB's complex multiply); conjP / conjP2 = MN.conj of the spectra."""
    m = (small_mask == 1).astype(np.float64)
    t = small_template * m
    idx = m != 0
    t[idx] = t[idx] - MN.mmean(t[idx], threads)
    nmask = int(idx.sum())
    x = t[idx]
    norm_t = math.sqrt(MN.msum(x * x, threads))
    tp = t * m
    P1 = MN.fftn(tp)
    PM = MN.fftn(m)
    mc1 = MN.ifftn_real(MN.cmul(P1, conjP))
    mcn = MN.ifftn_real(MN.cmul(PM, conjP2))
    mca = MN.ifftn_real(MN.cmul(PM, conjP))
    joint = mcn - 1 * (mca * mca) / nmask
    # sqrt of a negative value is imaginary in MATLAB; real(i*s*norm) = 0, which is then a 'problem' entry
    with np.errstate(invalid="ignore"):
        final = np.where(joint >= 0, np.sqrt(np.where(joint >= 0, joint, 0)) * norm_t, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = mc1 / final
    mc[final == 0] = -1
    if not want_matrix:
        pd, pv = peak_subpixel(mc)
        nl = ns = mc.shape[0]
        f = pd - 1 + nl / 2 - np.floor((pd - 1 + nl / 2) / nl) * nl + 1
        fl = nl - f + 1
        rs = fl - 1 + ((nl + ns) / 2 + 1) - nl * np.floor((fl - 1 + ((nl + ns) / 2 + 1)) / nl) + 1
        return min(pv, 1.0), rs, None
    # nargout == 3: fftshift, flip the three dims, circshift by (m1+m2)/2 + 1 (the unshifted peak only feeds the
    # unused restriction_drift and is not computed here)
    mcs = np.fft.fftshift(mc)[::-1, ::-1, ::-1]
    k = (mc.shape[0] + conjP.shape[0]) // 2 + 1
    mcs = np.roll(mcs, (k, k, k), axis=(0, 1, 2))
    return None, None, mcs


# =====================================================================================================================
# the motor (dynamo__align_motor.m)
# =====================================================================================================================

def align_motor(scard, template, mask, conjP, conjP2, angles, ptable_input, threads=1):
    use_cc = int(scard["use_CC"])
    if use_cc not in (0, 1):
        raise NotImplementedError("use_CC %s in dynamo_align" % use_cc)
    if not scard["localnc"]:
        raise NotImplementedError("localnc 0 (correlation_general) in dynamo_align")
    if ptable_input[12] != 0:
        raise NotImplementedError("dynamo_align with ftype %s" % ptable_input[12])
    # ftype 0: W = full sampling (ones), filtermask_template = ones .* ones -> rotated template FFT round trip
    s = conjP.shape[0]
    area = motor_area(scard["area_search"], s, scard["feature_data_sidelength"])
    modus = int(scard["area_search_modus"])
    search_area = None
    if modus > 0:
        check_motor_area(area, modus)
        asym = not (area[0] == area[1] == area[2])
        north = ellipsoid(area, s)
        shifts = [_mround(x) for x in ptable_input[3:6]]
        if asym:
            search_area = rot_shift(north, ptable_input[6:9], shifts, -1.0)
        else:
            search_area = np.roll(north, (int(shifts[2]), int(shifts[1]), int(shifts[0])), axis=(0, 1, 2))
        search_area = np.where(search_area < 0.1, 0.0, search_area)
        search_area = (search_area != 0).astype(np.float64)
    cc_list = np.zeros(len(angles))
    best = -1.0
    best_ang, best_pos = None, None
    for i, (td, ti, na) in enumerate(angles):
        rt_raw, rmask = mex_shift_rot(template, (td, ti, na), (0, 0, 0), option=1, outscope=0.0, mask=mask)
        rt = MN.ifftn_real(MN.fftn(rt_raw))                        # dynamo_applyfiltermask(., ones)
        if modus == 0:
            cc, pos, _ = local_correlation(rt, rmask, conjP, conjP2, False, threads)
        else:
            _, _, ccm = local_correlation(rt, rmask, conjP, conjP2, True, threads)
            masked = ccm * search_area + (-1.0 * (search_area == 0))
            pos, cc = peak_subpixel(masked, ccm)
        if cc > 1:
            if cc < 1.05:
                cc = 1.0
            else:
                print("Attention: large interpolation artifact for interpolated cc: %f" % cc)
        cc_list[i] = cc
        if cc >= best:
            best, best_ang, best_pos = cc, (td, ti, na), pos
    out = np.array(ptable_input, dtype=np.float64).copy()
    sh = np.asarray(best_pos, dtype=np.float64) - (s / 2 + 1)
    out[3:6] = sh
    out[6:9] = best_ang
    out[9] = best
    return out, best, cc_list


def motor_area(area_search, s, feature_data_sidelength):
    """The search area as dynamo__align_motor sees it: a scalar is expanded to 3 values, then
    ceil(area * s / feature_data_sidelength)."""
    area = np.atleast_1d(np.asarray(area_search, dtype=np.float64))
    if area.size < 3:
        area = area * np.ones(3)
    return np.ceil(area * (s / feature_data_sidelength))


def check_motor_area(area, modus):
    """The search-area branches of dynamo__align_motor.m that are ported: modus 2/3/4 with no zero component."""
    if modus not in (2, 3, 4):
        raise NotImplementedError("area_search_modus %d in dynamo_align" % modus)
    if np.all(area == 0) or not np.all(area):
        raise NotImplementedError("area_search with a zero component (all(area_search)==0 branch)")


def _mround(x):
    return math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1)


# =====================================================================================================================
# dynamo_align
# =====================================================================================================================

BASIC = ("cone_range", "cone_sampling", "inplane_range", "inplane_sampling", "refine", "refine_factor", "use_CC",
         "area_search", "area_search_modus", "localnc")


def blank_ptable():
    """dynamo_table_blank([], 'tags', 1) (S/dynamo_table_blank.m; 'fsampling' 1, 'ytilt'/'xtilt' [-60, 60]): one
    35-column row with tag 1, cols 2-3 = 1, col 13 = 1, cols 14-17 = -60 60 -60 60, col 32 = 1 [RUN: the .sal
    ptable of the harness, with col 13 reset to 0 by the 'full' pfmask]."""
    t = np.zeros(35)
    t[0] = t[1] = t[2] = 1
    t[12] = 1
    t[13:17] = (-60, 60, -60, 60)
    t[31] = 1
    return t


def dynamo_align(particle, template, vpr, settings_round, mask=None, verbose=True, threads=1):
    """sal = dynamo_align(particle, template, 'settings_from', vpr, 'settings_round', r) (S/dynamo_align.m), the
    'pipeline' destination. particle/template: [z, y, x] double arrays. mask: the volume read from vpr.file_mask
    (the caller reads it, cwd-relative). Returns a dict with ptable (1x n columns), ccmax, ptable_refine, angles,
    cc_angles, Tp (shifts, eulers, type 'shiftrot'), scard, the basic parameters q (:744-745, :1654-1657) and
    pydyn's own wall-clock timings (time_alignment: the refinement loop, :1820/:1930; time_overall = time_total: the
    whole call, :1661-1662; pydyn has no separate input-parsing step)."""
    import time
    t_all = time.time()
    q = {p: vpr["%s_r%d" % (p, settings_round)] for p in BASIC}           # dynamo_align_one :934-966
    for p in ("cone_range", "cone_sampling", "inplane_range", "inplane_sampling", "refine", "refine_factor",
              "use_CC", "area_search_modus", "localnc"):
        q[p] = float(q[p])
    if np.isnan(particle).any() or np.isnan(template).any():
        raise NotImplementedError("NaN in dynamo_align inputs (Dynamo returns an empty result)")
    n = particle.shape[0]
    if mask is None or np.ndim(mask) != 3:
        raise NotImplementedError("dynamo_align without a 3-D mask (dynamo_ellipsoid(floor(N/2), N) default)")
    if template.shape != particle.shape or mask.shape != particle.shape:
        raise NotImplementedError("dynamo_align with different particle/template/mask sizes")
    ptable = blank_ptable()
    angles = angleset(q["cone_range"], q["cone_sampling"], q["inplane_range"], q["inplane_sampling"],
                      ptable[6:9], False, False)
    if verbose:
        print(" Assuming full Fourier sampling by default (no pfmask passed)")
    ptable[12] = 0                                                       # pfmask struct 'full' (:1258-1262)
    # filtermask_data = ones: particle_transformed = real(ifftn(fftn(p) .* fftshift(ones)))  (:1338-1340)
    pt = MN.ifftn_real(MN.fftn(particle))
    pt = normalize_whole(pt, threads)                                    # reproduce_pipeline :1796
    P = MN.fftn(pt)
    P2 = MN.fftn(pt * pt)
    conjP, conjP2 = MN.conj(P), MN.conj(P2)
    scard = {"use_CC": q["use_CC"], "localnc": q["localnc"], "area_search": np.asarray(q["area_search"], float),
             "area_search_modus": q["area_search_modus"], "feature_data_sidelength": n}
    levels, ccmax, cclist, alist = [], [], [], [angles]
    t_ops = time.time()
    row, cc, ccl = align_motor(scard, template, mask, conjP, conjP2, angles, ptable, threads)
    levels.append(row)
    ccmax.append(cc)
    cclist.append(ccl)
    cr, cs, ir, is_ = q["cone_range"], q["cone_sampling"], q["inplane_range"], q["inplane_sampling"]
    for lev in range(1, int(q["refine"]) + 1):
        cr, cs, ir, is_ = multigrid(q["refine_factor"], cr, cs, ir, is_)
        ang = angleset(cr, cs, ir, is_, levels[-1][6:9], False, False)    # flips forced off (q.*_flip > 1 false)
        alist.append(ang)
        row, cc, ccl = align_motor(scard, template, mask, conjP, conjP2, ang, levels[-1], threads)
        levels.append(row)
        ccmax.append(cc)
        cclist.append(ccl)
        if verbose:
            print("   Refining level: %d (%d triplets). cone_range: %5.2f. inplane_range: %5.2f. "
                  % (lev, len(ang), cr, ir))
    pt_final = levels[-1]                                                 # expansion_factor 1
    time_alignment = time.time() - t_ops
    time_total = time.time() - t_all
    return {"ptable": pt_final, "ccmax": np.array(ccmax), "ptable_refine": levels, "angles": alist,
            "cc_angles": cclist, "Tp": {"type": "shiftrot", "shifts": -pt_final[3:6].copy(),
                                        "eulers": -pt_final[8:5:-1].copy()},
            "scard": scard, "q": q, "time_alignment": time_alignment, "time_overall": time_total,
            "time_total": time_total}
