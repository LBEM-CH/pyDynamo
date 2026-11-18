"""fdlibm's sin, cos, atan, atan2 and acos (Sun's freely distributable libm 5.3: k_sin.c, k_cos.c, e_rem_pio2.c,
s_sin.c, s_cos.c, s_atan.c, e_atan2.c, e_acos.c), ported operation for operation in double precision (no FMA).

Dynamo 1.1.514 runs on MATLAB R2019a, whose scalar trigonometry is fdlibm: these functions return MATLAB's values
bit for bit (2,000,000 angles for sin/cos against the MATLAB runtime's own routines). pyDynamo uses them wherever
Dynamo's MATLAB code computes trigonometry that reaches its output files (table columns 27-30, the rigid-body
and angle-set computations of dynamo_align), so no MATLAB library is needed. The C library's and NumPy's sin/cos
differ from MATLAB's in about 3 % of the arguments.

Only arguments with |x| <= 2^20 * pi/2 (about 1,647,099 rad) are supported for sin/cos (fdlibm's medium-size argument
reduction); larger ones raise NotImplementedError instead of guessing.
"""
# fdlibm (from which this module is ported):
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is preserved.

import math
import struct

import numpy as np


def _hi(x):
    return struct.unpack("<q", struct.pack("<d", x))[0] >> 32


def _lo(x):
    return struct.unpack("<q", struct.pack("<d", x))[0] & 0xffffffff


def _with_lo_zero(x):
    b = struct.unpack("<q", struct.pack("<d", x))[0] & ~0xffffffff
    return struct.unpack("<d", struct.pack("<q", b))[0]


def _from_hi(hi):
    return struct.unpack("<d", struct.pack("<q", hi << 32))[0]


# ----------------------------------------------------------------------------------------------- k_sin.c, k_cos.c
_S1, _S2, _S3 = -1.66666666666666324348e-01, 8.33333333332248946124e-03, -1.98412698298579493134e-04
_S4, _S5, _S6 = 2.75573137070700676789e-06, -2.50507602534068634195e-08, 1.58969099521155010221e-10
_C1, _C2, _C3 = 4.16666666666666019037e-02, -1.38888888888741095749e-03, 2.48015872894767294178e-05
_C4, _C5, _C6 = -2.75573143513906633035e-07, 2.08757232129817482790e-09, -1.13596475577881948265e-11


def _k_sin(x, y, iy):
    if (_hi(x) & 0x7fffffff) < 0x3e400000 and int(x) == 0:     # |x| < 2^-27: x itself (keeps -0.0)
        return x
    z = x * x
    v = z * x
    r = _S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))
    if iy == 0:
        return x + v * (_S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


def _k_cos(x, y):
    ix = _hi(x) & 0x7fffffff
    if ix < 0x3e400000 and int(x) == 0:                        # |x| < 2^-27
        return 1.0
    z = x * x
    r = z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    if ix < 0x3FD33333:                                        # |x| < 0.3
        return 1.0 - (0.5 * z - (z * r - x * y))
    qx = 0.28125 if ix > 0x3fe90000 else _from_hi(ix - 0x00200000)
    hz = 0.5 * z - qx
    a = 1.0 - qx
    return a - (hz - (z * r - x * y))


# ----------------------------------------------------------------------------------------------- e_rem_pio2.c
_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1, _PIO2_1T = 1.57079632673412561417e+00, 6.07710050650619224932e-11
_PIO2_2, _PIO2_2T = 6.07710050630396597660e-11, 2.02226624879595063154e-21
_PIO2_3, _PIO2_3T = 2.02226624871116645580e-21, 8.47842766036889956997e-32


def _rem_pio2(x):
    """n and the reduced argument y0 + y1 = x - n*pi/2, for |x| <= 2^20 * pi/2."""
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix <= 0x3fe921fb:
        return 0, x, 0.0
    if ix < 0x4002d97c:                                        # |x| < 3pi/4: n = +-1
        if hx > 0:
            z = x - _PIO2_1
            if ix != 0x3ff921fb:
                y0 = z - _PIO2_1T
                return 1, y0, (z - y0) - _PIO2_1T
            z -= _PIO2_2
            y0 = z - _PIO2_2T
            return 1, y0, (z - y0) - _PIO2_2T
        z = x + _PIO2_1
        if ix != 0x3ff921fb:
            y0 = z + _PIO2_1T
            return -1, y0, (z - y0) + _PIO2_1T
        z += _PIO2_2
        y0 = z + _PIO2_2T
        return -1, y0, (z - y0) + _PIO2_2T
    if ix <= 0x413921fb:                                       # |x| <= 2^20 (pi/2): medium size
        t = abs(x)
        n = int(t * _INVPIO2 + 0.5)
        fn = float(n)
        r = t - fn * _PIO2_1
        w = fn * _PIO2_1T
        j = ix >> 20
        y0 = r - w
        if j - ((_hi(y0) >> 20) & 0x7ff) > 16:                 # second iteration, good to 118 bits
            t = r
            w = fn * _PIO2_2
            r = t - w
            w = fn * _PIO2_2T - ((t - r) - w)
            y0 = r - w
            if j - ((_hi(y0) >> 20) & 0x7ff) > 49:             # third iteration, 151 bits
                t = r
                w = fn * _PIO2_3
                r = t - w
                w = fn * _PIO2_3T - ((t - r) - w)
                y0 = r - w
        y1 = (r - y0) - w
        return (-n, -y0, -y1) if hx < 0 else (n, y0, y1)
    raise NotImplementedError("fdlibm sin/cos: |x| > 2^20*pi/2 (Payne-Hanek reduction) is not ported")


# ----------------------------------------------------------------------------------------------- s_sin.c, s_cos.c
def sin(x):
    x = float(x)
    ix = _hi(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        return _k_sin(x, 0.0, 0)
    if ix >= 0x7ff00000:                                       # inf or NaN
        return x - x
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _k_sin(y0, y1, 1)
    if n == 1:
        return _k_cos(y0, y1)
    if n == 2:
        return -_k_sin(y0, y1, 1)
    return -_k_cos(y0, y1)


def cos(x):
    x = float(x)
    ix = _hi(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        return _k_cos(x, 0.0)
    if ix >= 0x7ff00000:
        return x - x
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _k_cos(y0, y1)
    if n == 1:
        return -_k_sin(y0, y1, 1)
    if n == 2:
        return -_k_cos(y0, y1)
    return _k_sin(y0, y1, 1)


# ----------------------------------------------------------------------------------------------- s_atan.c
_ATANHI = (4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01,
           1.57079632679489655800e+00)
_ATANLO = (2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17,
           6.12323399573676603587e-17)
_AT = (3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01,
       -1.11111104054623557880e-01, 9.09088713343650656196e-02, -7.69187620504482999495e-02,
       6.66107313738753120669e-02, -5.83357013379057348645e-02, 4.97687799461593236017e-02,
       -3.65315727442169155270e-02, 1.62858201153657823623e-02)


def atan(x):
    x = float(x)
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix >= 0x44100000:                                       # |x| >= 2^66
        if ix > 0x7ff00000 or (ix == 0x7ff00000 and _lo(x) != 0):
            return x + x                                       # NaN
        return _ATANHI[3] + _ATANLO[3] if hx > 0 else -_ATANHI[3] - _ATANLO[3]
    if ix < 0x3fdc0000:                                        # |x| < 0.4375
        if ix < 0x3e200000:                                    # |x| < 2^-29
            return x
        idx = -1
    else:
        x = abs(x)
        if ix < 0x3ff30000:                                    # |x| < 1.1875
            if ix < 0x3fe60000:                                # 7/16 <= |x| < 11/16
                idx, x = 0, (2.0 * x - 1.0) / (2.0 + x)
            else:                                              # 11/16 <= |x| < 19/16
                idx, x = 1, (x - 1.0) / (x + 1.0)
        elif ix < 0x40038000:                                  # |x| < 2.4375
            idx, x = 2, (x - 1.5) / (1.0 + 1.5 * x)
        else:                                                  # 2.4375 <= |x| < 2^66
            idx, x = 3, -1.0 / x
    z = x * x
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    if idx < 0:
        return x - x * (s1 + s2)
    z = _ATANHI[idx] - ((x * (s1 + s2) - _ATANLO[idx]) - x)
    return -z if hx < 0 else z


# ----------------------------------------------------------------------------------------------- e_atan2.c
_PI_O_4, _PI_O_2 = 7.8539816339744827900E-01, 1.5707963267948965580E+00
_PI, _PI_LO = 3.1415926535897931160E+00, 1.2246467991473531772E-16
_TINY = 1.0e-300


def atan2(y, x):
    y, x = float(y), float(x)
    if math.isnan(x) or math.isnan(y):
        return x + y
    hx, lx, hy, ly = _hi(x), _lo(x), _hi(y), _lo(y)
    ix, iy = hx & 0x7fffffff, hy & 0x7fffffff
    if ((hx - 0x3ff00000) | lx) == 0:                          # x = 1.0
        return atan(y)
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2)                    # 2*sign(x) + sign(y)
    if (iy | ly) == 0:                                         # y = 0
        if m in (0, 1):
            return y
        return _PI + _TINY if m == 2 else -_PI - _TINY
    if (ix | lx) == 0:                                         # x = 0
        return -_PI_O_2 - _TINY if hy < 0 else _PI_O_2 + _TINY
    if ix == 0x7ff00000:                                       # x = +-inf
        if iy == 0x7ff00000:
            return (_PI_O_4 + _TINY, -_PI_O_4 - _TINY, 3.0 * _PI_O_4 + _TINY, -3.0 * _PI_O_4 - _TINY)[m]
        return (0.0, -0.0, _PI + _TINY, -_PI - _TINY)[m]
    if iy == 0x7ff00000:                                       # y = +-inf
        return -_PI_O_2 - _TINY if hy < 0 else _PI_O_2 + _TINY
    k = (iy - ix) >> 20
    if k > 60:                                                 # |y/x| > 2^60
        z = _PI_O_2 + 0.5 * _PI_LO
    elif hx < 0 and k < -60:                                   # |y|/x < -2^60
        z = 0.0
    else:
        z = atan(abs(y / x))
    if m == 0:
        return z
    if m == 1:
        return -z
    if m == 2:
        return _PI - (z - _PI_LO)
    return (z - _PI_LO) - _PI


# ----------------------------------------------------------------------------------------------- e_acos.c
_PIO2_HI, _PIO2_LO = 1.57079632679489655800e+00, 6.12323399573676603587e-17
_PS = (1.66666666666666657415e-01, -3.25565818622400915405e-01, 2.01212532134862925881e-01,
       -4.00555345006794114027e-02, 7.91534994289814532176e-04, 3.47933107596021167570e-05)
_QS = (-2.40339491173441421878e+00, 2.02094576023350569471e+00, -6.88283971605453293030e-01,
       7.70381505559019352791e-02)


def _pq(z):
    p = z * (_PS[0] + z * (_PS[1] + z * (_PS[2] + z * (_PS[3] + z * (_PS[4] + z * _PS[5])))))
    q = 1.0 + z * (_QS[0] + z * (_QS[1] + z * (_QS[2] + z * _QS[3])))
    return p, q


def acos(x):
    """acos of a real in [-1, 1]; NaN outside (MATLAB's complex branch for |x| > 1 is the caller's)."""
    x = float(x)
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix >= 0x3ff00000:                                       # |x| >= 1
        if ((ix - 0x3ff00000) | _lo(x)) == 0:
            return 0.0 if hx > 0 else _PI + 2.0 * _PIO2_LO
        return float("nan")
    if ix < 0x3fe00000:                                        # |x| < 0.5
        if ix <= 0x3c600000:                                   # |x| < 2^-57
            return _PIO2_HI + _PIO2_LO
        z = x * x
        p, q = _pq(z)
        r = p / q
        return _PIO2_HI - (x - (_PIO2_LO - x * r))
    if hx < 0:                                                 # x < -0.5
        z = (1.0 + x) * 0.5
        p, q = _pq(z)
        s = math.sqrt(z)
        r = p / q
        w = r * s - _PIO2_LO
        return _PI - 2.0 * (s + w)
    z = (1.0 - x) * 0.5                                        # x > 0.5
    s = math.sqrt(z)
    df = _with_lo_zero(s)
    c = (z - df * df) / (s + df)
    p, q = _pq(z)
    r = p / q
    w = r * s + c
    return 2.0 * (df + w)


# ----------------------------------------------------------------------------------------------- arrays
def _vectorise(f):
    uf = np.frompyfunc(f, 1, 1)

    def g(x):
        a = np.asarray(x, dtype=np.float64)
        return uf(a).astype(np.float64) if a.ndim else float(f(float(a)))
    g.__name__, g.__doc__ = f.__name__ + "_array", "%s of every element (float64 array)" % f.__name__
    return g


sin_array = _vectorise(sin)
cos_array = _vectorise(cos)
