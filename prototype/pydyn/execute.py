"""dynamo_execute_project in pure Python: run an unfolded project in place (what <name>.exe does).

Spec: 06c A.2 (S/dynamo_execute_project.m:1-264) and 08a section 7. The iterations themselves are
pydyn.dynproject.DynamoProject (06a/06b/06c), run with source == out == ./<name_project>.

Before the loop Dynamo writes nothing; this function also writes nothing before DynamoProject's first iteration.
Deviations, printed when they apply:
* matlab_workers_average > 0: Dynamo opens a MATLAB pool (mbparse.multicore.checkPool) for the averaging; pydyn
  averages on the GPU and opens no pool.
* the GPU check (dpkproject.runtime.checks.gpu) is done with CuPy instead of mbgpu / `which` of the CUDA binary.
"""
import os

# variables that must not leak into the pydyn run (SBGrid / 4d / conda / user site, see 06c and the cluster notes)
UNSET = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "LD_LIBRARY_PATH", "LD_PRELOAD")


def _bars(n=80):
    print("-" * n)


def execute_project(name, gpus=None, destination=None, matlab_trig=False, keep_temp=False, iterations=None,
                    engine=None, exact_sms=None, report=None):
    """dynamo_execute_project(name): run the unfolded project ./<name> from the current folder.
    gpus: physical device ids to use (default: the vpr's gpu_identifier_set), or "all": every visible GPU, up to
    the number of particle parts (pyDynamo option for Slurm jobs). engine: 'exact' or 'fast' GPU
    alignment engine (pydyn.engine.engine_mode; default $PYDYN_ENGINE, else pydyn.engine.DEFAULT_ENGINE).
    exact_sms: exact engine only, reproduce the cuBLAS 8 reduction order of a GPU with this many SMs.
    report: opt-in JSON diagnostics path ($PYDYN_REPORT), refused inside the project folder.
    Returns True when the loop ends normally (o.ok), False when Dynamo would print a message and return."""
    from .engine import preflight
    preflight(engine)                                # before any project, setup or engine work (review M1)
    from . import vpr as V
    from .dynproject import DynamoProject, round_array, check_report_path
    check_report_path(report or os.environ.get("PYDYN_REPORT"), os.path.abspath(name))   # refuse early
    vpr = V.load(name)
    if vpr is None:
        print("Sorry, the project has not beeinf")
        return False
    final = destination or vpr["destination"]
    if final in ("matlab", "standalone", "matlab_parfor"):
        raise NotImplementedError("destination %r (CPU alignment, dynamo_iteration_compute) is not ported" % final)
    if final in ("system_omp", "mpi", "system_gpu", "mpi_gpu"):
        print("This destination cannot be executed through this program")
        print("   execution motor: dynamo_execute_project")
        print('   execution destination: "%s"' % final)
        return False
    if final not in ("matlab_gpu", "standalone_gpu"):
        print('Unknown destination: "%s"' % final)
        return False
    facts = V.data_info(vpr["folder_data"])
    rounds = round_array(vpr)
    _bars()
    print(" ")
    print("Project: %s" % vpr["name_project"])
    print("  contains %d rounds, totalling %d iterations " % (len(set(rounds)), len(rounds)))
    print('  will run on destination "%s" on %d cores' % (final, int(vpr["how_many_processors"])))
    print(" ")
    _bars()
    hmp, mwa = int(vpr["how_many_processors"]), int(vpr["matlab_workers_average"])
    if hmp > 1 or mwa > 0:
        print(" ")
        print("Your project invokes multicore operations (%d cores required)." % max(hmp, mwa))
        if hmp != mwa:
            print("Alignment step: %d cores,  Averaging step: %d cores" % (hmp, mwa))
        print("[pydyn] no MATLAB pool is opened: the averaging runs on the GPU")
        print(" ")
    card_ids = _as_list(vpr["gpu_identifier_set"])
    ids = None if gpus == "all" else [int(g) for g in (gpus if gpus else card_ids)]
    try:
        import cupy as cp
        ndev = cp.cuda.runtime.getDeviceCount()
    except Exception as e:                           # dpkproject.runtime.checks.gpu fails
        print("Problems with GPU (%s)" % e)
        return False
    if ids is None:
        # pyDynamo option (--gpus all): every GPU this process sees; in a Slurm job those are the job's own, numbered
        # 0..n-1 (CUDA_VISIBLE_DEVICES). The particles are still split as gpu_identifier_set says (the files of the
        # project follow it); the parts go to the GPUs in turn, so GPUs beyond the number of parts would stay idle.
        parts = len(card_ids) * max(1, hmp)
        ids = list(range(min(ndev, parts)))
        print("[pydyn] GPUs: %d visible%s; the project splits its particles in %d part%s (gpu_identifier_set): "
              "using %s" % (ndev, " (CUDA_VISIBLE_DEVICES=%s)" % os.environ["CUDA_VISIBLE_DEVICES"]
                            if os.environ.get("CUDA_VISIBLE_DEVICES") else "", parts, "" if parts == 1 else "s",
                            ",".join(str(i) for i in ids) or "none"))
    if not ids or max(ids) >= ndev or min(ids) < 0:
        print("Problems with GPU: devices %s requested, %d visible" % (ids, ndev))
        return False
    if not facts["correct"]:
        print("[pydyn] warning: data folder %r is not correct (dynamo_data_info)" % vpr["folder_data"])
    root = os.path.abspath(vpr["name_project"])
    nxt = os.path.join(root, "next_iteration.txt")
    if not os.path.isfile(nxt):                      # dynamo_read([name,'/next_iteration.txt']) errors (tolerance 0)
        raise FileNotFoundError('[dread: error] cannot find requested file "%s/next_iteration.txt"'
                                % vpr["name_project"])
    if "feature_data_sidelength" not in V.load(vpr["name_project"], verbose=False):
        # dynamo_iteration_setup.m:141-146 stops when the project is not unfolded
        raise RuntimeError("project %s is not unfolded (no feature_data_sidelength): run dvunfold first"
                           % vpr["name_project"])
    data = vpr["folder_data"]
    data = data if os.path.isabs(data) else os.path.abspath(data)
    pr = DynamoProject(root, data, root, gpus=ids, matlab_trig=matlab_trig, keep_temp=keep_temp, engine=engine,
                       exact_sms=exact_sms, report=report)
    pr.run(iterations=iterations)
    return True


dynamo_execute_project = execute_project


def _as_list(v):
    import numpy as np
    return [int(x) for x in np.atleast_1d(np.asarray(v, dtype=float)).tolist()]
