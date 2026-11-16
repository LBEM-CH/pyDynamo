"""Dynamo EM-format I/O (volumes and binary tables).

EM layout: 512-byte header; byte 0 machine, byte 3 data type; int32 nx, ny, nz at bytes 4, 8, 12.
Data follow with x fastest (MATLAB first index).

Arrays returned here are numpy arrays indexed [z, y, x] (C order), so arr.ravel() matches
Dynamo's linear index x + nx*(y + ny*z).
"""
import numpy as np

_EM_TYPES = {1: np.int8, 2: np.int16, 4: np.int32, 5: np.float32, 8: np.complex64, 9: np.float64}


def read_em(path):
    """Return a [z, y, x] array in the file's data type."""
    with open(path, "rb") as f:
        buf = f.read()
    dtype_code = buf[3]
    nx, ny, nz = np.frombuffer(buf[4:16], dtype="<i4")
    dt = _EM_TYPES[dtype_code]
    count = int(nx) * int(ny) * int(nz)
    data = np.frombuffer(buf, dtype=dt, count=count, offset=512)
    return data.reshape(int(nz), int(ny), int(nx))


def write_em(path, arr_zyx):
    """Write a float32 [z, y, x] array (or a 2-D/1-D array) as EM float."""
    a = np.ascontiguousarray(arr_zyx, dtype=np.float32)
    if a.ndim == 1:
        nz, ny, nx = 1, 1, a.shape[0]
    elif a.ndim == 2:
        nz, (ny, nx) = 1, a.shape
    else:
        nz, ny, nx = a.shape
    hdr = np.zeros(512, dtype=np.uint8)
    hdr[0], hdr[3] = 6, 5  # PC, float -- same as Dynamo's GPU engine writer
    hdr[4:16] = np.frombuffer(np.array([nx, ny, nz], dtype="<i4").tobytes(), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(a.tobytes())


def read_btbl(path):
    """Dynamo binary table (.btbl): EM with nx = particles, ny = columns. Returns [particles, cols] float32."""
    v = read_em(path)  # shape (1, ncols, nparticles)
    return np.array(v[0].T, dtype=np.float32)


def write_btbl(path, table):
    t = np.asarray(table, dtype=np.float32)
    write_em(path, t.T)  # [cols, particles] -> nx = particles


def read_tbl(path):
    """Dynamo text table (.tbl). MATLAB sometimes writes complex entries ('0+1.2e-06i'); the real part is kept."""
    with open(path) as f:
        text = f.read()
    if "i" not in text:
        return np.loadtxt(path, dtype=np.float64, ndmin=2)
    rows = []
    for line in text.splitlines():
        toks = line.split()
        if toks:
            rows.append([complex(t.replace("i", "j")).real if t.endswith("i") else float(t) for t in toks])
    return np.array(rows, dtype=np.float64, ndmin=2)


def particle_path(folder, tag, _cache={}):
    """Path of particle `tag` in a Dynamo data folder; the number width (5 or 6 digits, ...) is detected once."""
    import glob
    import os
    if folder not in _cache:
        width = 5
        for p in glob.iglob(os.path.join(folder, "particle_*.em")):
            digits = os.path.basename(p)[len("particle_"):-3]
            if digits.isdigit():
                width = len(digits)
                break
        _cache[folder] = width
    return os.path.join(folder, f"particle_{int(tag):0{_cache[folder]}d}.em")


def write_tbl(path, table):
    np.savetxt(path, np.asarray(table), fmt="%.6g")
