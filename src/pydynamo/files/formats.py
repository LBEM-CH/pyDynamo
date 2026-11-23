"""Byte-exact Dynamo 1.1.514 file I/O (MATLAB R2019a runtime), as used by a dcp project run.

Specification: docs/analysis/06c_driver_formats_1.1.514.md section B (verified Python formatter of B.10) and
docs/analysis/06b_assemble_1.1.514.md section 10. Sources (1.1.514, matlab/src):

  dynamo_write.m:225-226   .txt .rt .coords -> dlmwrite(f, m, 'delimiter', ' ', 'precision', 9)
  dynamo_write.m:228-229   .fsc .ctf .xyz   -> dlmwrite(f, m, ' ')             (default precision: %.5g)
  dynamo_write.m:232-239   .tbl -> dlmwrite(f, m, ' ') if max(m(:,1)) <= 99999, else dpktbl.write
  dynamo_write.m:241-242   .em .sem .btbl   -> dynamo_write_em_simple
  dynamo_write.m:145-146   .card            -> dynamo_card_write
  dynamo_write.m:163-180   .exe .sh .m ...  -> fprintf('%s\\n') per line
  dynamo_tags_write.m      .tags            -> '%0<max(padding,5)>d\\n'
  dynamo_read.m:317-330    .tbl .txt .fsc   -> dlmread(f, ' ') (complex tokens parsed); .em/.btbl -> dynamo_read_emfile
  dynamo_card_read.m, dynamo_card_write.m, dynamo_zaux_card_order.m

Array conventions
* Text tables are 2-D numpy arrays [rows, columns]. A table with at least one complex token (column 28 of
  1.1.514 tables, e.g. '0+1.2074e-06i') is returned as complex128; others as float64. MATLAB keeps such a
  matrix complex; dlmwrite prints only the elements with a non-zero imaginary part in complex form.
* EM volumes follow the rest of pydynamo: numpy arrays indexed [z, y, x] (C order), so the C-order bytes are
  MATLAB's column-major (x fastest) order. read_em returns float64 (MATLAB fread(...,'float32') -> double),
  except uint8 data. write_em writes single(v) little-endian with the 512-byte header 6,0,0,5 + uint32 nx,ny,nz
  + 496 zero bytes (dynamo_write_em_simple.m:53-123).
* .btbl tables are EM files of size (n_rows, n_cols, 1): read_btbl / write_btbl convert from/to [rows, columns].

Every writer takes (path, data). The *_bytes functions return the exact bytes for testing.
Uncertain corner cases (06c B.3 items 5-7: NaN/Inf, -0, complex with non-zero real part or negative imaginary
part) follow MATLAB's documented sprintf behaviour by analogy; none occurs in the reference run.
"""
import math
import os
import re

import numpy as np

# ------------------------------------------------------------------------------------------------
# MATLAB sprintf numeric conversions (C printf semantics, MATLAB spellings of NaN / Inf)
# ------------------------------------------------------------------------------------------------


def _g(x, prec):
    x = float(x)
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    return '%.*g' % (prec, x)


def _gplus(x, prec):
    x = float(x)
    if math.isnan(x):
        return '+NaN'
    if math.isinf(x):
        return '+Inf' if x > 0 else '-Inf'
    return '%+.*g' % (prec, x)


def dlm_field(v, prec=5):
    """One element as MATLAB dlmwrite prints it: '%.<prec>g'; complex with imag != 0 -> 'a+bi'."""
    if isinstance(v, (complex, np.complexfloating)):
        if v.imag == 0:
            return _g(v.real, prec)
        return _g(v.real, prec) + _gplus(v.imag, prec) + 'i'
    return _g(v, prec)


def dlmwrite_bytes(m, prec=5, dlm=' '):
    """dlmwrite(file, m, dlm[, 'precision', prec]) on Linux: one line per row, '\\n' after every row."""
    a = np.asarray(m)
    if a.size == 0:
        return b''
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:                       # vectors written by Dynamo are row vectors (one line)
        a = a.reshape(1, -1)
    return ''.join(dlm.join(dlm_field(v, prec) for v in row.tolist()) + '\n' for row in a).encode('ascii')


# ------------------------------------------------------------------------------------------------
# dpktbl.write (+dpktbl/write.m:4-31): only when the largest tag is above 99999 (never in the reference run)
# ------------------------------------------------------------------------------------------------
_DPKTBL_INT_COLS = {1, 2, 3, 12, 13, 20, 21, 22, 31, 32, 34, 35}


def _matlab_d(x):
    x = float(x)
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    if x == math.floor(x):
        return '%d' % int(x)
    return '%e' % x                          # MATLAB '%d' with a non-integer falls back to %e [UNC, 06c B.4]


def _matlab_f(x):
    x = float(x)
    if math.isnan(x):
        return '  NaN'
    if math.isinf(x):
        return '  Inf' if x > 0 else ' -Inf'
    return '%5.8f' % x


def dpktbl_write_bytes(t):
    t = np.real(np.asarray(t))
    return ''.join(''.join((_matlab_d(v) if j + 1 in _DPKTBL_INT_COLS else _matlab_f(v)) + ' '
                           for j, v in enumerate(row)) + '\n' for row in t).encode('ascii')


# ------------------------------------------------------------------------------------------------
# text writers (dynamo_write)
# ------------------------------------------------------------------------------------------------


def _mkparent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)       # dynamo_write.m:133-138 creates a missing parent folder


def tbl_bytes(table):
    t = np.asarray(table)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    if t.size == 0:
        return b''
    return dlmwrite_bytes(t, 5) if np.max(np.real(t[:, 0])) <= 99999 else dpktbl_write_bytes(t)


def write_tbl(path, table):
    """.tbl: dynamo_write.m:232-239."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(tbl_bytes(table))


def write_fsc(path, vec):
    """.fsc/.ctf/.xyz: dlmwrite(f, m, ' ') (dynamo_write.m:228-229)."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(dlmwrite_bytes(vec, 5))


def write_txt(path, x):
    """.txt/.rt/.coords: dlmwrite(f, m, 'delimiter', ' ', 'precision', 9) (dynamo_write.m:225-226)."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(dlmwrite_bytes(x, 9))


write_rt = write_txt


def write_lines(path, lines):
    """.exe/.sh/.m/.doc/...: fprintf(fid, '%s\\n', line) per line (dynamo_write.m:163-180); also the GPU particle
    lists of createParticleFileListAssignations.m (fopen 'wt', fprintf('%s\\n'))."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(''.join(ln + '\n' for ln in lines).encode())


def write_tags(path, tags, padding=6):
    """dynamo_tags_write.m:1-12: padding = max(padding, 5); '%0<padding>d\\n' per tag."""
    _mkparent(path)
    p = max(int(padding), 5)
    with open(path, 'wb') as f:
        f.write(''.join('%0*d\n' % (p, int(t)) for t in np.ravel(np.real(np.asarray(tags)))).encode())


# ------------------------------------------------------------------------------------------------
# dlmread(file, ' ') as used by dynamo_read for .tbl/.txt/.fsc/.rt/.tags
# ------------------------------------------------------------------------------------------------
_NUM = r'(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
_CPLX = re.compile(r'^([-+]?' + _NUM + r'|[-+]?(?:NaN|Inf))([-+]' + _NUM + r'|[-+](?:NaN|Inf))i$')


def _tok(t):
    if t == '':
        return 0.0                          # empty field between two delimiters -> 0
    if t.endswith('i'):
        m = _CPLX.match(t)
        if m:
            return complex(float(m.group(1)), float(m.group(2)))
    return float(t)


def read_dlm(path):
    """dlmread(path, ' '): numeric matrix; complex128 when any token is complex; short rows padded with 0."""
    rows = []
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            if line == '':
                continue
            rows.append([_tok(t) for t in line.split(' ')])
    if not rows:
        return np.zeros((0, 0))
    w = max(len(r) for r in rows)
    cplx = any(isinstance(v, complex) for r in rows for v in r)
    a = np.zeros((len(rows), w), dtype=complex if cplx else float)
    for i, r in enumerate(rows):
        a[i, :len(r)] = r
    return a


def read_tbl(path):
    """dynamo_read of a .tbl (dlmread). Keeps complex entries (1.1.514 column 28)."""
    return read_dlm(path)


read_txt = read_dlm
read_fsc = read_dlm


def read_lines(path):
    """Cell of lines (dynamo_read.m:291-311, fgetl)."""
    with open(path, 'r') as f:
        return f.read().splitlines()


# ------------------------------------------------------------------------------------------------
# MATLAB num2str (one argument) for the values that occur in cards (06c B.6)
# ------------------------------------------------------------------------------------------------


def num2str(x):
    if isinstance(x, str):
        return x
    xs = np.atleast_1d(np.asarray(x, dtype=float)).ravel().tolist()
    if not xs:
        return ''
    neg = any(v < 0 for v in xs)
    if all(math.isfinite(v) and v == math.floor(v) for v in xs):
        if len(xs) == 1:
            return '%d' % int(xs[0])
        xmax = max(abs(v) for v in xs)
        d = 1 if xmax == 0 else int(math.floor(math.log10(xmax))) + 1
        return ''.join('%*d' % (d + 2 + neg, int(v)) for v in xs).strip()
    fin = [abs(v) for v in xs if math.isfinite(v)]
    xmax = max(fin) if fin else 0.0
    ndgt = min(max((0 if xmax == 0 else int(math.floor(math.log10(xmax)))) + 5, 5), 16)

    def g(v):
        if math.isnan(v):
            return 'NaN'
        if math.isinf(v):
            return 'Inf' if v > 0 else '-Inf'
        return '%.*g' % (ndgt, v)
    if len(xs) == 1:
        return g(xs[0])
    return ''.join(g(v).rjust(ndgt + 7 + neg) for v in xs).strip()


# ------------------------------------------------------------------------------------------------
# cards
# ------------------------------------------------------------------------------------------------
# official string-typed (type 0/3) and "other" (type 6) parameters of dynamo_vpr_convention.m
CARD_STRING_FIELDS = {
    'name_project', 'path', 'file_template_initial', 'file_table_initial', 'folder_data',
    'file_mask', 'file_mask_classification', 'file_fmask_initial', 'file_mask_smoothing',
    'destination', 'cluster_header', 'cluster_walltime', 'submit_order', 'gpu_motor',
    'intertwin', 'options', 'MCR_CACHE_ROOT', 'project_type', 'fCompensationSmoothingMask',
    'sym', 'ccmatrix_type', 'plugin_align_order', 'plugin_post_order', 'plugin_iter_order',
    'flags', 'initial_references'}
_NUMVEC = re.compile(r'^\s*[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|NaN|Inf)'
                     r'(?:\s+[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|NaN|Inf))*\s*$')


def read_card(path):
    """dynamo_card_read: ordered dict name -> str, float or list of float."""
    card = {}
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            if not line or line[0] == '%':
                continue
            parts = line.split('=')
            if len(parts) != 2:
                continue                    # dynamo_card_read warns and skips (:164-168)
            par, val = parts[0].strip(), parts[1].strip()
            if not par or par.startswith('#'):
                continue
            if val.endswith(';'):
                val = val[:-1]
            val = val.strip()
            if par not in CARD_STRING_FIELDS and _NUMVEC.match(val):
                nums = [float(t) for t in val.split()]
                card[par] = nums[0] if len(nums) == 1 else nums
            else:
                card[par] = val
    return card


def card_order(card):
    """dynamo_zaux_card_order.m:37-60: database_* fields first (struct order), then the rest; then database_class,
    database_tag, database_ref, database_ite and name_project are each moved to the top, in that sequence."""
    keys = [k for k in card if k.startswith('database_')] + [k for k in card if not k.startswith('database_')]
    for pref in ('database_class', 'database_tag', 'database_ref', 'database_ite', 'name_project'):
        if pref in keys:
            keys.remove(pref)
            keys.insert(0, pref)
    return {k: card[k] for k in keys}


def card_bytes(card):
    return ''.join('%s=%s;\n' % (k, num2str(v)) for k, v in card_order(card).items()).encode()


def write_card(path, card):
    """dynamo_card_write.m:58-91: 'name=num2str(value);\\n' per field, in card_order."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(card_bytes(card))


# ------------------------------------------------------------------------------------------------
# EM (.em / .btbl)
# ------------------------------------------------------------------------------------------------


def read_em(path):
    """dynamo_read_emfile: returns [z, y, x]; float64 unless the file holds uint8."""
    with open(path, 'rb') as f:
        b = f.read()
    end = {0: '>', 3: '>', 5: '>', 6: '<'}[b[0]]
    nx, ny, nz = (int(v) for v in np.frombuffer(b[4:16], end + 'u4'))
    code = b[3]
    dt = {1: 'u1', 2: end + 'u2', 5: end + 'f4'}[code]
    v = np.frombuffer(b[512:], dt, count=nx * ny * nz).reshape((nz, ny, nx))
    return v.copy() if code == 1 else v.astype(np.float64)


def em_bytes(vol_zyx):
    v = np.asarray(vol_zyx)
    if np.iscomplexobj(v):
        raise NotImplementedError('complex volumes are never written by this pipeline')
    if v.ndim == 1:
        v = v.reshape(1, 1, -1)
    elif v.ndim == 2:
        v = v.reshape((1,) + v.shape)
    nz, ny, nx = v.shape
    code = 1 if v.dtype == np.uint8 else 5
    hdr = bytes([6, 0, 0, code]) + np.array([nx, ny, nz], '<u4').tobytes() + bytes(496)
    return hdr + np.ascontiguousarray(v, dtype='u1' if code == 1 else '<f4').tobytes()


def write_em(path, vol_zyx):
    """dynamo_write_em_simple: single(v) (round to nearest), little-endian, zero-filled 512-byte header."""
    _mkparent(path)
    with open(path, 'wb') as f:
        f.write(em_bytes(vol_zyx))


def read_btbl(path):
    """Binary table (EM, size n_rows x n_cols x 1, float32) -> float64 [rows, cols] (dynamo_read_emfile)."""
    v = read_em(path)                       # [1, n_cols, n_rows]
    return np.array(v[0].T, dtype=np.float64)


def write_btbl(path, table):
    """dynamo_write(T, '.btbl') -> dynamo_write_em_simple(single(T)). For a complex T the mex receives the real
    part (mxGetData under the separate-complex API; 06a section U [UNC]; the imaginary parts are 0 here)."""
    t = np.real(np.asarray(table))
    write_em(path, np.ascontiguousarray(t.T)[None])
