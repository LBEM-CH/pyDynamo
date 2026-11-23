"""Parse a Dynamo task card exactly as the GPU engine does.

A line matches when it starts with the key followed directly by '='. The value runs up to the
first ';'. The last matching line wins.
"""
import re


def read_card(path, strip=True):
    """key -> value text. strip=False: the text exactly as settings_card::extract_field returns it
    (dynamo_compute_iteration_spp.cu:1871-1922: from after the '=' up to the first ';', not trimmed)."""
    fields = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, rest = line.partition("=")
            if not key or key != key.strip() or key.startswith("%"):
                continue
            v = rest.rstrip("\n").split(";")[0]
            fields[key] = v.strip() if strip else v
    return fields


def _num(fields, key, default=0.0):
    v = fields.get(key, "")
    try:
        return float(v)
    except ValueError:
        return default


# C atof (strtod, "C" locale): leading white space skipped, the longest numeric prefix converted (decimal, hex,
# inf/infinity, nan), 0.0 when there is none
_ATOF = re.compile(r"[ \t\n\v\f\r]*([+-]?(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?"
                   r"|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[iI][nN][fF](?:[iI][nN][iI][tT][yY])?"
                   r"|[nN][aA][nN]))")


def atof(s):
    m = _ATOF.match(s)
    if not m:
        return 0.0
    t = m.group(1)
    return float.fromhex(t) if "x" in t or "X" in t else float(t)


def parse_area_search(v):
    """area_search[3] as settings_card::settings_card reads it (dynamo_compute_iteration_spp.cu:1821-1852): the text
    up to the first ' ' (the whole text without one); then, after that space and at most two more, the text up to
    the next ' ' (an empty text gives 0); then the rest. So '4' gives [4, 4, 4], '4 2' gives [4, 2, 2], and a first
    gap of 4 or more spaces (num2str of [4 4 2.5]: '4           4         2.5') gives [4, 0, 4]."""
    i = v.find(" ")                                       # 1823 (npos as int: -1)
    a0 = atof(v if i < 0 else v[:i])                      # 1827-1828 (substr(0, npos): the whole text)
    rest = v[i + 1:]                                      # 1831 (npos + 1 = 0: the whole text)
    for _ in range(2):                                    # 1834-1839
        if rest[:1] == " ":
            rest = rest[1:]
    j = rest.find(" ")                                    # 1843
    a1 = atof(rest if j < 0 else rest[:j])                # 1845-1847
    a2 = atof(rest[j + 1:])                               # 1850-1852
    return [a0, a1, a2]


class Card:
    def __init__(self, path):
        f = read_card(path)
        self.raw = f
        self.name_project = f["name_project"]
        self.ite = int(_num(f, "database_ite"))
        self.ref = int(_num(f, "database_ref"))
        self.file_template = f["database_gpu_file_template_iteration"]
        self.file_mask = f["database_gpu_file_mask"]
        self.root_particle_list = f["database_gpu_root_file_particleList"]
        self.shrink = _num(f, "database_gpu_shifts_shrink_factor", 1.0)
        self.sidelength = int(_num(f, "feature_data_sidelength"))
        self.dim = int(_num(f, "dim"))
        self.cone_range = _num(f, "cone_range")
        self.cone_sampling = _num(f, "cone_sampling")
        self.cone_flip = int(_num(f, "cone_flip"))
        self.inplane_range = _num(f, "inplane_range")
        self.inplane_sampling = _num(f, "inplane_sampling")
        self.inplane_flip = int(_num(f, "inplane_flip"))
        self.refine = int(_num(f, "refine"))
        self.refine_factor = _num(f, "refine_factor")
        self.high = _num(f, "high")
        self.low = _num(f, "low")
        self.use_CC = int(_num(f, "use_CC"))
        self.localnc = int(_num(f, "localnc", 1.0))
        self.area_search = parse_area_search(read_card(path, strip=False).get("area_search", ""))
        self.area_search_modus = int(_num(f, "area_search_modus"))
        self.gpu_ids = [int(x) for x in f.get("gpu_identifier_set", "0").split()]

    def starting_table(self, gpu):
        return (f"{self.name_project}/temp/ite_{self.ite:04d}/intermediate_results/"
                f"starting_gpuProcRefTable_ite_{self.ite:04d}_ref_{self.ref:03d}_proc_00000_gpu_{gpu:02d}.btbl")

    def refined_table(self, gpu):
        return (f"{self.name_project}/temp/ite_{self.ite:04d}/intermediate_results/"
                f"refined_gpuProcRefTable_ite_{self.ite:04d}_ref_{self.ref:03d}_proc_00000_gpu_{gpu:02d}.btbl")

    def particle_list(self, gpu):
        return f"{self.root_particle_list}{gpu:02d}.txt"
