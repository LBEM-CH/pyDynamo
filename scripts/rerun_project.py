"""Re-execute a Dynamo 1.1.514 dcp alignment project (standalone_gpu) with pydynamo, without MATLAB.

python scripts/rerun_project.py --source <dynamo project folder> --data <data folder> --out <new project root>
                             [--gpus 0,1,2,3] [--iterations N] [--keep-temp] [--table-initial F]
                             [--sms N] [--report PATH] [--progress-log PATH]

--source      an unfolded Dynamo project (its settings/ folder, incl. settings/virtual_project.mat, is read;
              nothing is written there)
--data        the data folder (particle_<tag>.em)
--out         the new project root: results/, cards/, settings/, next_iteration.txt, ... with Dynamo's names,
              exactly Dynamo's file set; log.txt is written line for line as Dynamo does (pydynamo.project.logfile); the
              engine's stdout also goes to gpulog.txt in the folder that contains --out, as Dynamo writes it
--gpus        physical devices to run on (default: the project's gpu_identifier_set). The GPU chunks and their
              file names always follow the project's gpu_identifier_set, as in Dynamo.
--iterations  stop after this global iteration (default: all iterations of the project)
--sms         reproduce the cuBLAS 8 reduction order of a GPU with this many streaming multiprocessors
--matlab-threads  MATLAB thread count of the Dynamo run to reproduce (adaptive bandpass realignment; Dynamo uses
              one per core of its job); default $PYDYNAMO_MATLAB_THREADS, else one per physical core available here
--report      opt-in JSON diagnostics ($PYDYNAMO_REPORT); refused inside --out
--progress-log  opt-in copy of the progress lines on stdout; refused inside --out
--keep-temp   keep temp/ite_XXXX (Dynamo deletes it at the end of every assemble)
--table-initial  the table used by the iteration-1 comparison (default: vpr.file_table_initial relative to the
              folder that contains --source, i.e. Dynamo's working directory)
See src/pydynamo/project/runner.py for the specification and the list of unsupported options.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--table-initial", default=None)
    ap.add_argument("--sms", type=int, default=None)
    ap.add_argument("--matlab-threads", type=int, default=None)
    ap.add_argument("--report", default=None, metavar="PATH")
    ap.add_argument("--progress-log", default=None, metavar="PATH")
    a = ap.parse_args(argv)
    if os.path.abspath(a.out) == os.path.abspath(a.source):
        ap.error("--out must differ from --source (the source project is read-only)")
    from pydynamo.engine import preflight
    preflight()                                           # before any project work (review M1)
    from pydynamo.project.runner import DynamoProject
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else None
    pr = DynamoProject(a.source, a.data, a.out, gpus=gpus, keep_temp=a.keep_temp,
                       table_initial=a.table_initial, sms=a.sms, report=a.report,
                       progress_log=a.progress_log, matlab_threads=a.matlab_threads)
    rep = pr.run(iterations=a.iterations)
    print(json.dumps({"total_wall_s": rep.get("total_wall_s"),
                      "iterations": [{k: it[k] for k in ("ite", "wall_s", "setup_s", "compute_s", "assemble_s")}
                                     for it in rep["iterations"]]}))


if __name__ == "__main__":
    main()
