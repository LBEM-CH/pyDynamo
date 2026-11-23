"""Port of Dynamo's GPU-engine angle generator (cuda/src/classes/gpuAngles.cpp), with C float semantics.

The C++ code mixes float and double math. With <cmath> and `using namespace std`, cos/sin/acos/ceil of a float
argument resolve to the float overloads (cosf, ...). The orientation counts depend on float32 truncation in
fill_interval (e.g. 95 instead of 100 orientations at refinement level 1 for cone 30/10, inplane 30/10), so this
module calls glibc's libm float and double functions through ctypes and does float arithmetic with np.float32.
"""
import ctypes
import ctypes.util

import numpy as np

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
for _name in ("cosf", "sinf", "acosf", "ceilf"):
    getattr(_libm, _name).restype = ctypes.c_float
    getattr(_libm, _name).argtypes = [ctypes.c_float]
for _name in ("cos", "sin", "acos", "atan2"):
    getattr(_libm, _name).restype = ctypes.c_double
_libm.cos.argtypes = _libm.sin.argtypes = _libm.acos.argtypes = [ctypes.c_double]
_libm.atan2.argtypes = [ctypes.c_double, ctypes.c_double]

F = np.float32
cosf, sinf, acosf, ceilf = _libm.cosf, _libm.sinf, _libm.acosf, _libm.ceilf
cosd, sind, acosd, atan2d = _libm.cos, _libm.sin, _libm.acos, _libm.atan2

DEG2RAD_F = F(0.0174532925199432954743716805978692718781530857086181640625)
RAD2DEG_F = F(57.29577951308232286464772187173366546630859375)
DEG2RAD_D = 0.0174532925199432954743716805978692718781530857086181640625
PI_D = 3.141592653589793115997963468544185161590576171875
TOL_F = F(0.00001)


def euler_matrix_double(tdrot, tilt, narot):
    """orientation::fill_matrix (double); row-major m[3*i+j] = Rz(narot) Rx(tilt) Rz(tdrot)."""
    ctd, std = cosd(float(F(tdrot)) * DEG2RAD_D), sind(float(F(tdrot)) * DEG2RAD_D)
    cti, sti = cosd(float(F(tilt)) * DEG2RAD_D), sind(float(F(tilt)) * DEG2RAD_D)
    cna, sna = cosd(float(F(narot)) * DEG2RAD_D), sind(float(F(narot)) * DEG2RAD_D)
    return [ctd * cna - std * cti * sna, -cna * std - ctd * cti * sna, sna * sti,
            ctd * sna + cna * std * cti, ctd * cna * cti - std * sna, -cna * sti,
            std * sti, ctd * sti, cti]


def fill_interval(first, interval, last):
    first, interval, last = F(first), F(interval), F(last)
    if interval == 0:
        return [F(0)]
    n = int(F(F(last - first) / interval) + F(1))  # C: (int) of a float expression, truncation toward 0
    if n <= 0:
        return []
    return [F(first + F(F(i) * interval)) for i in range(n)]


def _matrix_to_angles(m):
    """update_as_composition tail: angles from a double matrix, stored as float."""
    one_p, one_m = float(F(1) + TOL_F), float(F(1) - TOL_F)
    if one_m < m[8] < one_p:
        return F(0), F(0), F(atan2d(m[3], m[0]) * 180.0 / PI_D)
    if float(F(-1) - TOL_F) < m[8] < float(F(-1) + TOL_F):
        return F(atan2d(m[6], m[7]) * 180.0 / PI_D), F(180.0), F(atan2d(m[2], -m[5]) * 180.0 / PI_D)
    return (F(atan2d(m[6], m[7]) * 180.0 / PI_D), F(acosd(m[8]) * 180.0 / PI_D),
            F(atan2d(m[2], -m[5]) * 180.0 / PI_D))


def _compose(m1, m2):
    return [sum(m1[3 * i + k] * m2[3 * k + j] for k in range(3)) for i in range(3) for j in range(3)]


def _force_tilt_in_range(td, ti, na):
    if ti < 0 or ti > 360:
        m = euler_matrix_double(td, ti, na)
        if float(F(1) - TOL_F) < m[8] < float(F(1) + TOL_F):
            return F(0), F(0), F(atan2d(m[3], m[0]) * 180.0 / PI_D)
        return (F(atan2d(m[6], m[7]) * 180.0 / PI_D), F(acosd(m[8]) * 180.0 / PI_D),
                F(atan2d(m[2], -m[5]) * 180.0 / PI_D))
    return td, ti, na


def _best_narot(seed, td2_deg, ti2_deg):
    """orientation::bestNarotForPerturbation (float trig of float args, combined in double)."""
    td1, ti1, na1 = (F(F(a) * DEG2RAD_F) for a in seed)
    td2, ti2 = F(F(td2_deg) * DEG2RAD_F), F(F(ti2_deg) * DEG2RAD_F)
    alpha = F(F(cosf(td1)) * F(cosf(na1)) - F(F(F(sinf(td1)) * F(cosf(ti1))) * F(sinf(na1))))
    beta = F(F(cosf(td1)) * F(sinf(na1)) + F(F(F(sinf(td1)) * F(cosf(ti1))) * F(cosf(na1))))
    a = float(alpha) * cosf(td2) + float(beta) * sinf(td2) * cosf(ti2)
    b = float(beta) * cosf(td2) - float(alpha) * sinf(td2) * cosf(ti2)
    return F(atan2d(b, a) * float(RAD2DEG_F))


class ScanRange:
    def __init__(self, cone_range, cone_sampling, inplane_range, inplane_sampling, cone_flip=0, inplane_flip=0):
        self.cone_range, self.cone_sampling = F(cone_range), F(cone_sampling)
        self.inplane_range, self.inplane_sampling = F(inplane_range), F(inplane_sampling)
        self.cone_flip, self.inplane_flip = int(cone_flip), int(inplane_flip)

    def shrink(self, factor):
        """scan_range::shrink."""
        factor = F(factor)
        if self.cone_range != 0:
            self.cone_range = F(self.cone_sampling * factor)
        self.cone_sampling = F(self.cone_sampling / 2.0)
        if self.inplane_range != 0:
            self.inplane_range = F(self.inplane_sampling * factor)
        self.inplane_sampling = F(self.inplane_sampling / 2.0)

    def copy(self):
        return ScanRange(self.cone_range, self.cone_sampling, self.inplane_range, self.inplane_sampling,
                         self.cone_flip, self.inplane_flip)


def orientation_set(seed, rng):
    """orientation_set::fill_orientations. seed = (tdrot, tilt, narot) floats. Returns float32 array [N, 3]."""
    td0, ti0, na0 = _force_tilt_in_range(F(seed[0]), F(seed[1]), F(seed[2]))
    seed_f = (td0, ti0, na0)
    aperture_half = F(179.999) if rng.cone_range == F(180.0) else F(rng.cone_range / 2.0)
    inplane_half = F(rng.inplane_range / 2.0)
    m_seed = euler_matrix_double(td0, ti0, 0.0)
    cos_cs = cosf(F(rng.cone_sampling * DEG2RAD_F))
    out = []
    for tn in fill_interval(0, rng.cone_sampling, aperture_half):
        c = F(cosf(F(tn * DEG2RAD_F)))
        s = F(sinf(F(tn * DEG2RAD_F)))
        arg = F((cos_cs - float(F(c * c))) / float(F(s * s))) if s != 0 else F(np.inf)
        if tn == 0 or tn == 180:
            tdrots = fill_interval(0, 1, 0)
        elif F(arg * arg) >= 1:
            tdrots = fill_interval(0, 1, 0)
        else:
            cst = F(acosf(arg) * RAD2DEG_F)
            eq = F(F(360) / F(ceilf(F(F(360) / cst))))
            tdrots = fill_interval(0, eq, F(F(360) - eq))
        for td in tdrots:
            m = _compose(euler_matrix_double(td, tn, 0.0), m_seed)
            rtd, rti, _ = _matrix_to_angles(m)
            nu = _best_narot(seed_f, rtd, rti)
            narots = fill_interval(nu, rng.inplane_sampling, F(nu + inplane_half))
            if rng.inplane_sampling != 0:
                narots += fill_interval(F(nu - rng.inplane_sampling), F(-rng.inplane_sampling), F(nu - inplane_half))
            out.extend((rtd, rti, n) for n in narots)
    if rng.cone_flip > 0:
        out += [(a, F(b + 180), c) for a, b, c in out]
    if rng.inplane_flip > 0:
        out += [(a, b, F(c + 180)) for a, b, c in out]
    return np.array(out, dtype=np.float32).reshape(-1, 3)


def rotation_matrices(angles):
    """Per-orientation float32 matrices as used by the rotate kernel (computed in double, stored float)."""
    a = np.asarray(angles, dtype=np.float64) * DEG2RAD_D
    td, ti, na = a[:, 0], a[:, 1], a[:, 2]
    ctd, std, cti, sti, cna, sna = np.cos(td), np.sin(td), np.cos(ti), np.sin(ti), np.cos(na), np.sin(na)
    m = np.stack([ctd * cna - std * cti * sna, -cna * std - ctd * cti * sna, sna * sti,
                  ctd * sna + cna * std * cti, ctd * cna * cti - std * sna, -cna * sti,
                  std * sti, ctd * sti, cti], axis=1)
    return m.astype(np.float32)
