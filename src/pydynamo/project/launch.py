"""Where the Run step of the dcp window runs a project: this machine's GPUs, or a Slurm job.

pyDynamo's own computing parameters, kept in settings/virtual_project.mat next to Dynamo's. Dynamo 1.1.514 reads only
the fields it knows (dynamo_vpr_load / dynamo_vpr_check test them with isfield) and its dcp window does not keep other
fields when it saves. A field is stored only while it differs from its default, so a project that never sets them
keeps Dynamo's file as it is.
  pydynamo_slurm            0: the Run step runs the project on this machine's GPUs (gpu_identifier_set);
                         1: it submits the project as a Slurm job
  pydynamo_slurm_cpus       the job's CPU cores (--cpus-per-task)
  pydynamo_slurm_mem        the job's main memory (--mem, e.g. 64G); '' = estimated from the data (memory_estimate)
  pydynamo_slurm_partition  the job's partitions (--partition, comma-separated); '' = the shared partitions whose
                         machines have the project's GPU count (slurm_plan)
The job also takes three of Dynamo's own computing parameters: the number of devices in gpu_identifier_set
(--gres=gpu:N; inside the job the run uses the job's GPUs, --gpus all), cluster_walltime (--time) and submit_order (the
submit command, sbatch).

The run itself is pyDynamo's dynamo_execute_project (pydynamo.cli.execute_main) started with this environment's Python in
a clean environment (as the console scripts restart themselves), from the folder that holds the project. A local run
writes its output to pydynamo_<name>.out and a Slurm job to Slurm's slurm-<job id>.out, both next to the project folder,
never inside it. The last run of a project is recorded in pydynamo_<name>.job.json next to it, so that its state can be
shown (and the run stopped) after the window is opened again.
"""
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from pydynamo.project.execute import UNSET

PARAMS = {                      # name: (default, kind)
    "pydynamo_slurm": (0.0, "flag"),
    "pydynamo_slurm_cpus": (32.0, "count"),
    "pydynamo_slurm_mem": ("", "mem"),
    "pydynamo_slurm_partition": ("", "partitions"),
}
PRETTY = {"pydynamo_slurm": "submit to Slurm", "pydynamo_slurm_cpus": "Slurm CPU cores",
          "pydynamo_slurm_mem": "Slurm memory (RAM)", "pydynamo_slurm_partition": "Slurm partitions"}
SLURM_BIN = "/usr/local/slurm/bin"            # Slurm's commands may sit outside PATH (as on this cluster)
SLURM_CACHE_S = 30
_SINFO = {"t": 0.0, "parts": []}
_MEM = re.compile(r"^[1-9][0-9]*[KMGT]?$")
_PARTS = re.compile(r"^[A-Za-z0-9_.\-]+(,[A-Za-z0-9_.\-]+)*$")
_TIME = re.compile(r"^(?:[0-9]+-)?[0-9]+(?::[0-9]{1,2}){0,2}$|^(?:UNLIMITED|INFINITE)$")
_CHILDREN = {}                                 # pid -> Popen of local runs started by this process


class LaunchError(RuntimeError):
    """A run cannot be started (or a setting is invalid); the message says why."""


# ---------------------------------------------------------------------------------------------- settings in the vpr
def get(vpr, name):
    """The value of a pyDynamo computing parameter in a vpr (its default when the field is absent)."""
    default, kind = PARAMS[name]
    v = (vpr or {}).get(name, default)
    if kind in ("flag", "count"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return v if isinstance(v, str) else default


def parse(name, string):
    """The value of `string` for parameter `name`, or LaunchError."""
    if name not in PARAMS:
        raise LaunchError('"%s" is not a pyDynamo computing parameter' % name)
    kind = PARAMS[name][1]
    s = ("1" if string else "0") if isinstance(string, bool) else str(string).strip()
    if kind == "flag":
        if s not in ("0", "1"):
            raise LaunchError("%s is 0 or 1 (passed value: %r)" % (PRETTY[name], s))
        return float(s)
    if kind == "count":
        if not s.isdigit() or int(s) < 1:
            raise LaunchError("%s is a whole number >= 1 (passed value: %r)" % (PRETTY[name], s))
        return float(int(s))
    if kind == "mem":
        s = s.upper()
        if s and not _MEM.match(s):
            raise LaunchError("%s is a number of megabytes or a number with K, M, G or T, e.g. 64G (passed value: %r)"
                              % (PRETTY[name], s))
        return s
    if s and not _PARTS.match(s):
        raise LaunchError("%s is a comma-separated list of partition names (passed value: %r)" % (PRETTY[name], s))
    return s


def put(vpr, name, string):
    """Set a pyDynamo computing parameter in the vpr dict: stored while it differs from its default, removed when it
    equals it. Returns the value."""
    v = parse(name, string)
    if v == PARAMS[name][0]:
        vpr.pop(name, None)
    else:
        vpr[name] = v
    return v


def pydynamo_fields(vpr):
    """The pyDynamo computing parameters a vpr holds ({} for a project that never set one)."""
    return {k: vpr[k] for k in PARAMS if k in (vpr or {})}


# ---------------------------------------------------------------------------------------------- Slurm
def slurm_bin(command):
    """Path of a Slurm command (PATH, else SLURM_BIN), or None without Slurm."""
    p = shutil.which(command)
    if p:
        return p
    q = os.path.join(SLURM_BIN, command)
    return q if os.access(q, os.X_OK) else None


def parse_sinfo(text):
    """`sinfo -h -N -o "%N|%P|%a|%G|%f"` (one line per machine and partition) -> partitions: name, default, up,
    nodes, node_gpus {machine: GPUs}, node_features {machine: [...]}, gpus (the most one machine has), features
    (Slurm's node features; on this cluster the GPU generation, e.g. Ampere)."""
    parts = {}
    for line in text.splitlines():
        f = line.strip().split("|")
        if len(f) != 5:
            continue
        node, name, avail, gres, feats = f
        default = name.endswith("*")
        name = name.rstrip("*")
        gpus = 0
        for g in gres.split(","):                     # gpu:4, gpu:a40:4, gpu:4(S:0-1)
            g = g.split("(")[0]
            if g.startswith("gpu"):
                try:
                    gpus = max(gpus, int(g.split(":")[-1]))
                except ValueError:
                    pass
        p = parts.setdefault(name, {"name": name, "default": False, "up": avail == "up", "nodes": [],
                                    "node_gpus": {}, "node_features": {}, "gpus": 0, "features": []})
        p["default"] = p["default"] or default
        if node not in p["node_gpus"]:
            p["nodes"].append(node)
        p["node_gpus"][node] = gpus
        p["node_features"][node] = [x for x in feats.split(",") if x and x != "(null)"]
        p["gpus"] = max(p["gpus"], gpus)
        p["features"] += [x for x in feats.split(",") if x and x != "(null)" and x not in p["features"]]
    return list(parts.values())


def slurm_partitions():
    """The cluster's partitions (parse_sinfo), read with sinfo at most every SLURM_CACHE_S seconds; [] without Slurm."""
    if time.time() - _SINFO["t"] < SLURM_CACHE_S:
        return _SINFO["parts"]
    sinfo = slurm_bin("sinfo")
    parts = []
    if sinfo:
        try:
            r = subprocess.run([sinfo, "-h", "-N", "-o", "%N|%P|%a|%G|%f"], capture_output=True, text=True, timeout=8)
            parts = parse_sinfo(r.stdout)
        except (OSError, subprocess.SubprocessError):
            parts = []
    _SINFO.update(t=time.time(), parts=parts)
    return parts


def slurm_plan(parts, n):
    """Where a job asking for n GPUs on one machine can run: the shared partitions (those spanning several machines)
    that have machines with n GPUs, one partition per set of machines (e.g. a general partition and program-specific
    ones over the same machines: the general one is kept, its name has "gpu"). sbatch takes them as a list and starts
    the job in whichever offers the earliest start (sbatch --partition). Partitions of a single machine are left out:
    the job goes to the shared pool, not to one chosen machine."""
    shared = [p for p in parts if p["up"] and len(p["nodes"]) > 1 and p["gpus"] >= n]
    shared.sort(key=lambda p: (-len(p["nodes"]), "gpu" not in p["name"], p["name"]))
    chosen = []
    for p in shared:
        if not any(set(p["nodes"]) <= set(q["nodes"]) for q in chosen):
            chosen.append(p)
    machines = []
    for p in chosen:
        machines += [m for m in p["nodes"] if p["node_gpus"].get(m, 0) >= n and m not in machines]
    feats = []
    for p in chosen:
        for m in p["nodes"]:
            if p["node_gpus"].get(m, 0) >= n:
                feats += [x for x in p["node_features"].get(m, []) if x not in feats]
    most = max((p["gpus"] for p in parts if p["up"]), default=0)
    return {"slurm": bool(parts), "gpus": n, "partitions": [p["name"] for p in chosen],
            "machines": [m.split(".")[0] for m in machines], "features": feats, "most_gpus": most,
            "default": next((p["name"] for p in parts if p["default"]), None)}


def memory_estimate(particles, sidelength, gpus):
    """The job's RAM (--mem), not GPU memory: pyDynamo keeps the particles in RAM during the alignment (float32,
    shared among the GPU processes) and holds twice that while reading them; a margin of 4 GB per GPU process and
    4 GB for the main one; rounded up to 8 GB, at least 16 GB. An estimate."""
    l = sidelength
    vox = (l[0] * l[1] * l[2]) if isinstance(l, (list, tuple)) and len(l) == 3 else (float(l or 0) ** 3)
    gb = float(particles or 0) * vox * 4 / 2 ** 30
    return "%dG" % max(16, int(math.ceil((2 * gb + 4 * int(gpus) + 4) / 8)) * 8)


def project_gpus(vpr):
    """The devices of gpu_identifier_set, as ints."""
    g = vpr.get("gpu_identifier_set", 0)
    try:
        return [int(float(x)) for x in (g.split() if isinstance(g, str) else list(_flat(g)))]
    except (TypeError, ValueError):
        return []


def _flat(v):
    try:
        import numpy as np
        return np.atleast_1d(np.asarray(v, dtype=float)).ravel().tolist()
    except Exception:
        return [v]


# ---------------------------------------------------------------------------------------------- commands
def _clean_env():
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE="/dev/null", PYDYNAMO_CLEAN_ENV="1")
    return env


def execute_argv(name, gpus_all=False):
    """pyDynamo's `dynamo_execute_project NAME` with this environment's Python, isolated, the package folder on
    sys.path (as pydynamo.cli._clean_reexec, so it works without 'pip install -e'); `--gpus all` inside a Slurm job."""
    proto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # src/
    code = ("import sys; sys.path.insert(0, %r); from pydynamo.cli import execute_main as f; sys.exit(f())" % proto)
    return [sys.executable, "-I", "-c", code, name] + (["--gpus", "all"] if gpus_all else [])


def describe(vpr, name, data_facts=None, parts=None):
    """What the Run step will do for this project: mode, the command (argv and as a shell line), and for Slurm the
    job's resources and where it can start. parts: slurm_partitions() (read when None and needed)."""
    slurm = get(vpr, "pydynamo_slurm") == 1
    ids = project_gpus(vpr)
    n = max(1, len(ids))
    out = {"name": name, "mode": "slurm" if slurm else "local", "gpus": n, "devices": ids}
    df = data_facts or {}
    est = memory_estimate(df.get("N"), df.get("l"), n)
    out.update(cpus=int(get(vpr, "pydynamo_slurm_cpus")), mem=get(vpr, "pydynamo_slurm_mem"), mem_estimate=est,
               partition=get(vpr, "pydynamo_slurm_partition"), walltime=str(vpr.get("cluster_walltime", "") or ""),
               submit_order=str(vpr.get("submit_order", "") or ""))
    if not slurm:
        argv = execute_argv(name)
        out.update(argv=argv, line=shlex.join(argv), out=out_file(name), short="dynamo_execute_project %s" % name)
        return out
    if parts is None:
        parts = slurm_partitions()
    plan = slurm_plan(parts, n)
    partitions = out["partition"] or ",".join(plan["partitions"])
    wrap = shlex.join(execute_argv(name, gpus_all=True))
    argv = [out["submit_order"] or "sbatch", "--job-name=%s" % name, "--gres=gpu:%d" % n,
            "--cpus-per-task=%d" % out["cpus"], "--mem=%s" % (out["mem"] or est)]
    if out["walltime"]:
        argv.append("--time=%s" % out["walltime"])
    if partitions:
        argv.append("--partition=%s" % partitions)
    argv.append("--wrap=%s" % wrap)
    short = argv[:-1] + ["--wrap=dynamo_execute_project %s --gpus all" % name]
    out.update(argv=argv, line=shlex.join(argv), plan=plan, partitions=partitions, wrap=wrap, short=shlex.join(short))
    return out


# ---------------------------------------------------------------------------------------------- runs
def out_file(name):
    return "pydynamo_%s.out" % name


def job_file(name):
    return "pydynamo_%s.job.json" % name


def _read_job(name):
    try:
        with open(job_file(name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_job(name, job):
    with open(job_file(name), "w") as f:
        json.dump(job, f, indent=1)


def _check_names(d):
    """The values that go on the sbatch line, checked (a value from an edited project could hold anything)."""
    if d["mode"] != "slurm":
        return
    if d["walltime"] and not _TIME.match(d["walltime"]):
        raise LaunchError("cluster_walltime %r is not a Slurm time limit (minutes, MM:SS, HH:MM:SS, D-HH, "
                          "D-HH:MM or D-HH:MM:SS)" % d["walltime"])
    so = d["submit_order"] or "sbatch"
    if not re.match(r"^[A-Za-z0-9_./\-]+$", so):
        raise LaunchError("submit_order %r is not a command name" % so)
    if d["mem"] and not _MEM.match(d["mem"]):
        raise LaunchError("pydynamo_slurm_mem %r is not a Slurm memory size" % d["mem"])
    if d["partition"] and not _PARTS.match(d["partition"]):
        raise LaunchError("pydynamo_slurm_partition %r is not a list of partition names" % d["partition"])


def start(vpr, name, data_facts=None, parts=None):
    """Run the project as its parameters say (describe): start it on this machine's GPUs in the background, or submit
    it to Slurm. Refused while the last recorded run of the project is still running or queued. Returns the job."""
    st = status(name)
    if st and st.get("active"):
        raise LaunchError("the last run of %s is still %s (%s): stop it first" % (name, st["state"], st["id"]))
    d = describe(vpr, name, data_facts, parts)
    _check_names(d)
    job = {"mode": d["mode"], "line": d["line"], "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": os.getcwd()}
    if d["mode"] == "local":
        log = open(d["out"], "ab")
        try:
            p = subprocess.Popen(d["argv"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=_clean_env(), start_new_session=True)
        finally:
            log.close()
        _CHILDREN[p.pid] = p
        job.update(id=str(p.pid), pid=p.pid, out=d["out"])
    else:
        cmd = d["argv"]
        exe = slurm_bin(cmd[0]) if os.sep not in cmd[0] else cmd[0]
        if not exe:
            raise LaunchError("%s was not found (PATH, %s): is Slurm installed here?" % (cmd[0], SLURM_BIN))
        try:
            r = subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            raise LaunchError("%s failed: %s" % (cmd[0], e))
        m = re.search(r"Submitted batch job ([0-9]+)", r.stdout)
        if r.returncode != 0 or not m:
            raise LaunchError("%s refused the job: %s" % (cmd[0], (r.stderr or r.stdout).strip() or
                                                          "exit status %d" % r.returncode))
        job.update(id=m.group(1), job_id=m.group(1), out="slurm-%s.out" % m.group(1))
    _write_job(name, job)
    return status(name)


def _pid_runs(pid, name):
    """The recorded local run is alive (and is still that run: its command line names the project)."""
    p = _CHILDREN.get(pid)
    if p is not None:
        if p.poll() is not None:
            return False, p.returncode
        return True, None
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            args = f.read().split(b"\0")
    except OSError:
        return False, None
    return name.encode() in args, None


def _slurm_state(job_id):
    """The job's state: squeue while it is queued or running, else sacct (its final state), else ''."""
    sq = slurm_bin("squeue")
    if sq:
        try:
            r = subprocess.run([sq, "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True, timeout=10)
            s = r.stdout.strip().splitlines()
            if r.returncode == 0 and s:
                return s[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    sa = slurm_bin("sacct")
    if sa:
        try:
            r = subprocess.run([sa, "-n", "-X", "-P", "-j", job_id, "-o", "State"], capture_output=True, text=True,
                               timeout=10)
            s = r.stdout.strip().splitlines()
            if r.returncode == 0 and s:
                return s[0].split()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def status(name):
    """The last recorded run of the project: mode, id, state, active (running or queued), its output file and the
    line it was started with; None when there is none."""
    job = _read_job(name)
    if not job:
        return None
    out = dict(job)
    if job.get("mode") == "local":
        alive, rc = _pid_runs(int(job["pid"]), name)
        out["active"] = alive
        out["state"] = "RUNNING" if alive else ("ENDED" if rc is None else "ENDED (exit status %d)" % rc)
    else:
        s = _slurm_state(str(job["job_id"]))
        out["state"] = s or "ENDED"
        out["active"] = s in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
                              "RESIZING")
    return out


def stop(name):
    """Stop the last recorded run: the local run's process group gets SIGTERM, a Slurm job is cancelled (scancel)."""
    st = status(name)
    if not st or not st.get("active"):
        raise LaunchError("%s has no run in progress" % name)
    if st["mode"] == "local":
        try:
            os.killpg(os.getpgid(int(st["pid"])), signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        sc = slurm_bin("scancel")
        if not sc:
            raise LaunchError("scancel was not found (PATH, %s)" % SLURM_BIN)
        r = subprocess.run([sc, str(st["job_id"])], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise LaunchError("scancel %s failed: %s" % (st["job_id"], (r.stderr or r.stdout).strip()))
    return status(name)
