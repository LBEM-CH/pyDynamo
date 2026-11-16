"""pydyn project: the settings of one alignment job, saved as <folder>/project.json, and the job.exe that runs it.

Parameter names follow Dynamo's (cone_range, inplane_sampling, area_search_modus, ...), without the _r1 suffix: a
project has one round. Masks follow Dynamo's conventions:
* default mask = dynamo_ellipsoid(l/2-1, l, (l/2+1)*[1 1 1]), reproduced exactly (124,471 voxels at l = 64);
* sphere/ellipsoid/cylinder masks with an optional Gaussian edge (Dynamo-style; not bit-identical to Dynamo's
  smoothing).
"""
import json
import os
import shlex
import stat

import numpy as np

from . import emio

PYDYN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # .../prototype
ENV_PYTHON = os.path.join(os.path.dirname(PYDYN), ".envs", "pydynamo", "bin", "python")

DEFAULTS = {
    "name": "",
    "data": "", "table": "", "template": "", "mask": "", "fmask": "",
    "ite": 3,
    "cone_range": 30.0, "cone_sampling": 10.0, "inplane_range": 30.0, "inplane_sampling": 10.0,
    "refine": 1, "refine_factor": 2.0,
    "high": 2.0, "low": 20.0,
    "area_search": [4.0, 4.0, 4.0], "area_search_modus": 0,
    "sym": "c1",
    "gpus": [0, 1, 2, 3],
    "slurm": {"partition": "", "nodelist": "", "time": "24:00:00", "cpus_per_task": 16, "mem": "128G"},
}


class Project:
    def __init__(self, folder, **kw):
        self.folder = os.path.abspath(folder)
        self.p = json.loads(json.dumps(DEFAULTS))
        self.p["name"] = os.path.basename(self.folder)
        self.update(**kw)

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, "project.json")) as f:
            data = json.load(f)
        pr = cls(folder)
        pr.update(**data)
        return pr

    def update(self, **kw):
        for k, v in kw.items():
            if k == "slurm" and isinstance(v, dict):
                self.p["slurm"].update(v)
            elif k in self.p:
                self.p[k] = v
            else:
                raise KeyError(f"unknown project parameter: {k}")
        return self

    def path(self, key):
        """Absolute path of a file parameter (relative paths are relative to the project folder)."""
        v = self.p[key]
        return v if not v or os.path.isabs(v) else os.path.normpath(os.path.join(self.folder, v))

    def save(self):
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, "project.json"), "w") as f:
            json.dump(self.p, f, indent=1)
        return os.path.join(self.folder, "project.json")

    # ---------------------------------------------------------------- checks
    def check(self):
        """Return (errors, facts). Errors are human-readable strings; empty list = ready to unfold."""
        errors, facts = [], {}
        for key in ("data", "table", "template", "mask"):
            if not self.p[key]:
                errors.append(f"{key}: not set")
            elif not os.path.exists(self.path(key)):
                errors.append(f"{key}: not found: {self.path(key)}")
        if errors:
            return errors, facts
        rows = emio.read_tbl(self.path("table"))
        facts["particles"] = int(len(rows))
        if rows.shape[1] < 20:
            errors.append(f"table: {rows.shape[1]} columns, Dynamo tables have at least 20")
        missing = [int(t) for t in rows[:, 0] if not os.path.exists(self.particle_path(t))]
        if missing:
            errors.append(f"data: {len(missing)} particles of the table have no file, e.g. "
                          f"{os.path.basename(self.particle_path(missing[0]))}")
        else:
            box = emio.read_em(self.particle_path(rows[0, 0])).shape
            facts["box"] = box[0]
            for key in ("template", "mask") + (("fmask",) if self.p["fmask"] else ()):
                shp = emio.read_em(self.path(key)).shape
                if shp != box:
                    errors.append(f"{key}: size {shp} differs from the particle size {box}")
            if len(set(box)) != 1 or box[0] % 2:
                errors.append(f"particles must be cubic with an even side; got {box}")
        ftypes = sorted({int(x) for x in rows[:, 12]})
        facts["wedge_types"] = ftypes
        if any(f not in (0, 1, 2) for f in ftypes):
            errors.append(f"wedge types {ftypes}: the prototype averaging supports ftype 0, 1 and 2 only")
        if self.p["area_search_modus"] not in (0, 1, 2):
            errors.append("area_search_modus: supported values are 0, 1, 2")
        if not (self.p["low"] >= 1):
            errors.append("low: Nyquist-fraction low-pass (< 1) is not implemented")
        s = str(self.p["sym"]).lower()
        if not (s in ("c1", "none") or (s[0] in "cd" and s[1:].isdigit())):
            errors.append(f"sym {self.p['sym']}: supported are c1, cN, dN")
        if not self.p["gpus"]:
            errors.append("gpus: none selected")
        return errors, facts

    def particle_path(self, tag):
        return emio.particle_path(self.path("data"), tag)

    # ---------------------------------------------------------------- job.exe
    def unfold(self):
        """Write project.json and an executable job.exe that runs the job from a terminal or through sbatch."""
        errors, facts = self.check()
        if errors:
            raise ValueError("project not ready:\n  " + "\n  ".join(errors))
        self.save()
        s = self.p["slurm"]
        sb = [f"#SBATCH --job-name=pydyn_{self.p['name']}", f"#SBATCH --gres=gpu:{len(self.p['gpus'])}",
              f"#SBATCH --cpus-per-task={s['cpus_per_task']}", f"#SBATCH --mem={s['mem']}", f"#SBATCH --time={s['time']}",
              f"#SBATCH --output={self.folder}/job_%j.log"]
        if s.get("partition"):
            sb.append(f"#SBATCH --partition={s['partition']}")
        if s.get("nodelist"):
            sb.append(f"#SBATCH --nodelist={s['nodelist']}")
        gpus = ",".join(str(g) for g in self.p["gpus"])
        script = f"""#!/bin/bash
# pydyn job for project '{self.p['name']}' ({facts['particles']} particles, box {facts.get('box')})
# Run directly:        ./job.exe
# Or through Slurm:    sbatch job.exe
{chr(10).join(sb)}
# Clean environment: no SBGrid / 4d / conda / user-site Python packages leak in.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
# Inside a Slurm allocation the scheduler sets CUDA_VISIBLE_DEVICES; outside, use the project's GPUs.
if [ -z "${{SLURM_JOB_ID:-}}" ]; then export CUDA_VISIBLE_DEVICES={gpus}; fi
NGPU=$(echo "${{CUDA_VISIBLE_DEVICES}}" | awk -F, '{{print NF}}')
GPU_LIST=$(seq -s, 0 $((NGPU-1)))
cd {shlex.quote(self.folder)}
exec {shlex.quote(ENV_PYTHON)} -I {shlex.quote(os.path.join(PYDYN, 'pydyn_run.py'))} --project {shlex.quote(self.folder)} --gpus "$GPU_LIST" "$@"
"""
        exe = os.path.join(self.folder, "job.exe")
        with open(exe, "w") as f:
            f.write(script)
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        return exe

    def run_args(self):
        """Command-line arguments for pydyn_run equivalent to this project."""
        p = self.p
        args = ["--data", self.path("data"), "--table", self.path("table"), "--template", self.path("template"),
                "--mask", self.path("mask"), "--out", self.folder, "--ite", str(p["ite"]),
                "--cone-range", str(p["cone_range"]), "--cone-sampling", str(p["cone_sampling"]),
                "--inplane-range", str(p["inplane_range"]), "--inplane-sampling", str(p["inplane_sampling"]),
                "--refine", str(p["refine"]), "--refine-factor", str(p["refine_factor"]),
                "--high", str(p["high"]), "--low", str(p["low"]),
                "--area-search", *[str(x) for x in p["area_search"]], "--area-search-modus", str(p["area_search_modus"]),
                "--sym", str(p["sym"])]
        if p["fmask"]:
            args += ["--fmask", self.path("fmask")]
        return args


# -------------------------------------------------------------------- masks
def default_mask(l):
    """Dynamo's default alignment/classification mask: dynamo_ellipsoid(l/2-1, l, (l/2+1)*[1 1 1]) (fillSphere)."""
    r, c = l / 2 - 1, l / 2 + 1
    x = np.arange(l, dtype=np.float64)
    t = ((x + 1 - c) / r) ** 2
    # C++ order is (x-term + y-term) + z-term, with x the fastest index (last axis of our [z, y, x] arrays)
    s = t[None, None, :] + t[None, :, None]
    s = s + t[:, None, None]
    return (s <= 1).astype(np.float32)


def shape_mask(l, kind="sphere", radius=None, semiaxes=None, height=None, edge=0.0, center=None):
    """Dynamo-style masks: 'sphere' (radius), 'ellipsoid' (semiaxes x, y, z), 'cylinder' (radius, height along z),
    'ones'. edge > 0 adds a Gaussian fall-off of that width (voxels) outside the hard boundary."""
    if kind == "ones":
        return np.ones((l, l, l), dtype=np.float32)
    c = np.asarray(center if center is not None else [l // 2] * 3, dtype=np.float64)   # (x, y, z), 0-based
    g = np.arange(l, dtype=np.float64)
    z, y, x = np.meshgrid(g - c[2], g - c[1], g - c[0], indexing="ij")
    if kind == "sphere":
        radius = radius if radius is not None else l / 2 - 1
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) - radius
    elif kind == "ellipsoid":
        ax = np.asarray(semiaxes, dtype=np.float64)
        rho = np.sqrt((x / ax[0]) ** 2 + (y / ax[1]) ** 2 + (z / ax[2]) ** 2)
        d = (rho - 1) * ax.min()
    elif kind == "cylinder":
        radius = radius if radius is not None else l / 2 - 1
        height = height if height is not None else l - 2
        d = np.maximum(np.sqrt(x ** 2 + y ** 2) - radius, np.abs(z) - height / 2)
    else:
        raise ValueError(f"unknown mask kind {kind}")
    m = (d <= 0).astype(np.float64)
    if edge > 0:
        soft = np.exp(-(np.clip(d, 0, None) / edge) ** 2)
        m = np.where(d <= 0, 1.0, np.where(soft > 0.13, soft, 0.0))
    return m.astype(np.float32)


def write_default_masks(folder, l):
    """settings/mask.em (Dynamo default ellipsoid) and settings/fmask_initial.em (l * ones), as dcp 'masks default'."""
    d = os.path.join(folder, "settings")
    os.makedirs(d, exist_ok=True)
    emio.write_em(os.path.join(d, "mask.em"), default_mask(l))
    emio.write_em(os.path.join(d, "fmask_initial.em"), np.full((l, l, l), float(l), dtype=np.float32))
    return os.path.join(d, "mask.em"), os.path.join(d, "fmask_initial.em")
