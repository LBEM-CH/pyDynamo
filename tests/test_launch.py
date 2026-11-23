"""The Run step's computing parameters (pydynamo.project.launch): this machine's GPUs or a Slurm job, saved in the project only
while they differ from their defaults, the sbatch line, and starting, following and stopping a run (with stand-in
Slurm commands and a stand-in run, no GPU)."""
import os
import shlex
import stat
import sys
import time

import numpy as np
import pytest

from pydynamo.project import launch
from pydynamo import vpr as V

from test_console_and_lifecycle import make_data

SINFO = """\
gpu-a.cluster.example|gpus|up|gpu:4|Ampere
gpu-b.cluster.example|gpus|up|gpu:4|Ampere
pc-1.cluster.example|pc*|up|gpu:2|(null)
pc-2.cluster.example|pc*|up|gpu:2|(null)
"""


def test_parameters_are_stored_only_while_they_differ_from_the_default():
    vpr = {}
    assert launch.put(vpr, "pydynamo_slurm", "1") == 1.0 and vpr == {"pydynamo_slurm": 1.0}
    assert launch.put(vpr, "pydynamo_slurm", True) == 1.0
    launch.put(vpr, "pydynamo_slurm", "0")
    assert vpr == {}                                                  # back to the default: the field is gone
    launch.put(vpr, "pydynamo_slurm_cpus", "32")
    assert vpr == {}
    launch.put(vpr, "pydynamo_slurm_cpus", "16")
    launch.put(vpr, "pydynamo_slurm_mem", "64g")
    launch.put(vpr, "pydynamo_slurm_partition", "gpus,pc")
    assert vpr == {"pydynamo_slurm_cpus": 16.0, "pydynamo_slurm_mem": "64G", "pydynamo_slurm_partition": "gpus,pc"}
    assert launch.get({}, "pydynamo_slurm_cpus") == 32.0 and launch.get(vpr, "pydynamo_slurm_mem") == "64G"
    for name, bad in (("pydynamo_slurm", "2"), ("pydynamo_slurm_cpus", "0"), ("pydynamo_slurm_cpus", "1.5"),
                      ("pydynamo_slurm_mem", "64GB"), ("pydynamo_slurm_partition", "a;b"), ("cluster_walltime", "1")):
        with pytest.raises(launch.LaunchError):
            launch.put(vpr, name, bad)


def test_memory_estimate():
    assert launch.memory_estimate(1000, 64, 2) == "16G"              # 0.98 GB of particles: the 16 GB floor
    assert launch.memory_estimate(9422, 64, 4) == "40G"              # 2 x 9.2 GB + 4 x 4 GB + 4 GB = 38.4 GB
    assert launch.memory_estimate(9422, [64, 64, 64], 4) == "40G"
    assert launch.memory_estimate(None, None, 1) == "16G"


def test_describe_local_and_slurm():
    vpr = {"gpu_identifier_set": np.array([0.0, 1.0]), "cluster_walltime": "02:00:00", "submit_order": "sbatch"}
    d = launch.describe(vpr, "p1", {"N": 1000, "l": 64}, parts=[])
    assert d["mode"] == "local" and d["devices"] == [0, 1] and d["short"] == "dynamo_execute_project p1"
    assert d["argv"][0] == sys.executable and d["argv"][-1] == "p1" and "--gpus" not in d["argv"]
    assert "execute_main" in d["argv"][3] and d["out"] == "pydynamo_p1.out"
    launch.put(vpr, "pydynamo_slurm", "1")
    launch.put(vpr, "pydynamo_slurm_cpus", "16")
    d = launch.describe(vpr, "p1", {"N": 1000, "l": 64}, parts=launch.parse_sinfo(SINFO))
    assert d["mode"] == "slurm" and d["partitions"] == "gpus,pc"
    assert d["argv"][:-1] == ["sbatch", "--job-name=p1", "--gres=gpu:2", "--cpus-per-task=16", "--mem=16G",
                              "--time=02:00:00", "--partition=gpus,pc"]
    wrap = shlex.split(d["argv"][-1].split("=", 1)[1])
    assert wrap[0] == sys.executable and wrap[-3:] == ["p1", "--gpus", "all"]     # the job's own GPUs
    assert shlex.split(d["short"])[-1] == "--wrap=dynamo_execute_project p1 --gpus all"
    launch.put(vpr, "pydynamo_slurm_partition", "mine")
    launch.put(vpr, "pydynamo_slurm_mem", "100G")
    vpr["cluster_walltime"] = ""
    d = launch.describe(vpr, "p1", {"N": 1000, "l": 64}, parts=launch.parse_sinfo(SINFO))
    assert "--partition=mine" in d["argv"] and "--mem=100G" in d["argv"]
    assert not any(a.startswith("--time") for a in d["argv"])                      # no limit: the partition's


def _fake_slurm(tmp_path, monkeypatch, squeue_state="PENDING"):
    """Stand-ins for sbatch / squeue / scancel: they record their arguments next to them."""
    b = tmp_path / "slurm_bin"
    b.mkdir()
    scripts = {
        "sbatch": '#!/bin/sh\nprintf "%s\\n" "$@" > "$(dirname "$0")/sbatch.args"\necho "Submitted batch job 4242"\n',
        "squeue": '#!/bin/sh\ncat "$(dirname "$0")/state" 2>/dev/null\n',
        "scancel": '#!/bin/sh\necho "$@" > "$(dirname "$0")/scancel.args"\nrm -f "$(dirname "$0")/state"\n',
    }
    for name, text in scripts.items():
        p = b / name
        p.write_text(text)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (b / "state").write_text(squeue_state + "\n")
    monkeypatch.setattr(launch, "slurm_bin", lambda c: str(b / c) if (b / c).exists() else None)
    return b


def test_submit_follow_and_cancel_a_slurm_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = _fake_slurm(tmp_path, monkeypatch)
    vpr = {"gpu_identifier_set": 0.0, "cluster_walltime": "01:00:00", "submit_order": "sbatch", "pydynamo_slurm": 1.0}
    job = launch.start(vpr, "p1", {"N": 3, "l": 64}, parts=[])
    assert (job["mode"], job["id"], job["state"], job["active"]) == ("slurm", "4242", "PENDING", True)
    assert job["out"] == "slurm-4242.out" and os.path.isfile("pydynamo_p1.job.json")
    args = (b / "sbatch.args").read_text().splitlines()
    assert args[:3] == ["--job-name=p1", "--gres=gpu:1", "--cpus-per-task=32"] and "--time=01:00:00" in args
    with pytest.raises(launch.LaunchError, match="still PENDING"):
        launch.start(vpr, "p1", {"N": 3, "l": 64}, parts=[])          # one run of a project at a time
    st = launch.stop("p1")
    assert (b / "scancel.args").read_text().split() == ["4242"]
    assert st["state"] == "ENDED" and st["active"] is False              # no longer in the queue, no sacct here
    with pytest.raises(launch.LaunchError):
        launch.stop("p1")


def test_a_refused_submission_is_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = _fake_slurm(tmp_path, monkeypatch)
    (b / "sbatch").write_text('#!/bin/sh\necho "sbatch: error: Invalid partition name specified" >&2\nexit 1\n')
    vpr = {"gpu_identifier_set": 0.0, "cluster_walltime": "01:00:00", "submit_order": "sbatch", "pydynamo_slurm": 1.0}
    with pytest.raises(launch.LaunchError, match="Invalid partition"):
        launch.start(vpr, "p1", {}, parts=[])
    assert not os.path.exists("pydynamo_p1.job.json")
    vpr["cluster_walltime"] = "ten minutes"
    with pytest.raises(launch.LaunchError, match="not a Slurm time limit"):
        launch.start(vpr, "p1", {}, parts=[])


def test_local_run_in_the_background_and_stop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = "import sys, time; print('running', sys.argv[1], flush=True); time.sleep(60)"
    monkeypatch.setattr(launch, "execute_argv", lambda name, gpus_all=False: [sys.executable, "-c", code, name])
    job = launch.start({"gpu_identifier_set": 0.0}, "p1", {})
    assert (job["mode"], job["state"], job["active"]) == ("local", "RUNNING", True) and job["out"] == "pydynamo_p1.out"
    t = time.time()
    while "running p1" not in open("pydynamo_p1.out").read() and time.time() - t < 20:
        time.sleep(0.05)
    assert "running p1" in open("pydynamo_p1.out").read()
    pid = int(job["pid"])
    launch.stop("p1")
    launch._CHILDREN[pid].wait(20)
    st = launch.status("p1")
    assert st["active"] is False and st["state"].startswith("ENDED")
    launch._CHILDREN.pop(pid)                                            # as in a new window: read from /proc
    assert launch.status("p1")["active"] is False


def test_the_window_keeps_the_run_settings_and_a_dynamo_project_file(tmp_path, monkeypatch):
    """Ticking 'submit to Slurm' adds the field to virtual_project.mat; Dynamo-style edits of the window keep it;
    unticking removes it, and the file holds Dynamo's fields again, unchanged."""
    from pydynamo.manager.controller import DcpController
    from pydynamo.manager.tool import dcp
    monkeypatch.chdir(tmp_path)
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    f = os.path.join("p1", "settings", "virtual_project.mat")
    c = DcpController(echo=lambda *a, **k: None)
    assert c.enter_project_name("p1") == "opened"
    # a round edit saves the whole project as Dynamo's window does (getVpr: the convention's fields; the *_all fields
    # dcp.new leaves are not among them): that file, without the box ticked, is the reference
    assert c.set_value("cone_range", "20", 1).ok
    before = V.read_vpr_file(f)
    assert c.pydynamo_edit("pydynamo_slurm", "1").ok and V.load("p1")["pydynamo_slurm"] == 1.0
    assert c.set_value("cone_range", "20", 1).ok                         # the whole save keeps the box
    assert V.load("p1")["pydynamo_slurm"] == 1.0 and c.get_vpr()["pydynamo_slurm"] == 1.0
    assert not c.pydynamo_edit("pydynamo_slurm_mem", "lots").ok                # refused, nothing saved
    assert "pydynamo_slurm_mem" not in V.load("p1")
    assert c.pydynamo_edit("pydynamo_slurm", "0").ok and "pydynamo_slurm" not in c.runtime
    after = V.read_vpr_file(f)
    assert list(after) == list(before)                                   # Dynamo's fields only, in Dynamo's order
    assert all(V.values_equal(after[k], before[k]) for k in before)
    c2 = DcpController(echo=lambda *a, **k: None)
    assert c.pydynamo_edit("pydynamo_slurm", "1").ok
    assert c2.enter_project_name("p1") == "opened" and c2.runtime["pydynamo_slurm"] == 1.0   # opened again


def test_run_step_of_the_window(tmp_path, monkeypatch):
    """The Run step's actions: its settings, a pyDynamo setting keeps the unfold current while Dynamo's
    cluster_walltime (in the cards) needs a new unfold, and a run is refused before the unfold."""
    from pydynamo.manager.tool import dcp
    from pydynamo.webui.api import Session
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYDYNAMO_CACHE_DIR", str(tmp_path / "cache"))
    make_data(str(tmp_path), N=4)
    dcp.new("p1", "d", "filamentsData_ctf", "template", "raw_template_ctf.em", "masks", "default", "t",
            "particles_edit_mod.tbl", "gui", 0)
    s = Session()
    assert s.call("enter_project_name", ["p1"])["ok"]
    r = s.call("run_start")
    assert r["ok"] and r["result"]["started"] is False and "Unfold the project first" in r["result"]["error"]
    assert s.call("set_value", ["ite", "1", 1])["ok"] and s.call("unfold_and_hint")["result"][1] == "ok"
    assert s._unfold_info()["current"] is True
    rs = s.call("run_settings")["result"]
    assert rs["mode"] == "local" and rs["status"] is None and rs["unfold"]["current"] is True
    time.sleep(0.05)
    assert s.call("run_set", ["pydynamo_slurm", "1"])["result"]["ok"]
    assert s._unfold_info()["current"] is True                           # not in the cards
    rs = s.call("run_settings")["result"]
    assert rs["mode"] == "slurm" and rs["argv"][0] == "sbatch"
    time.sleep(0.05)
    assert s.call("run_set", ["cluster_walltime", "04:00:00"])["result"]["ok"]
    assert V.load("p1")["cluster_walltime"] == "04:00:00"
    assert s._unfold_info()["current"] is False                          # Dynamo writes it into the cards
    assert not s.call("run_set", ["name_project", "x"])["ok"]
