"""Parse a Dynamo task card exactly as the GPU engine does.

A line matches when it starts with the key followed directly by '='. The value runs up to the
first ';'. The last matching line wins.
"""


def read_card(path):
    fields = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, rest = line.partition("=")
            if not key or key != key.strip() or key.startswith("%"):
                continue
            fields[key] = rest.split(";")[0].strip()
    return fields


def _num(fields, key, default=0.0):
    v = fields.get(key, "")
    try:
        return float(v)
    except ValueError:
        return default


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
        self.area_search = [float(x) for x in f.get("area_search", "0 0 0").split()][:3]
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
