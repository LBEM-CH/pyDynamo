"""Re-execute a Dynamo 1.1.514 dcp alignment project (standalone_gpu) with pydyn, without MATLAB.

python run_dynamo_project.py --source <dynamo project folder> --data <data folder> --out <new project root>
                             [--gpus 0,1,2,3] [--iterations N] [--keep-temp] [--table-initial F]
                             [--engine exact|fast] [--exact-sms N] [--report PATH] [--progress-log PATH]

--source      an unfolded Dynamo project (its settings/ folder, incl. settings/virtual_project.mat, is read;
              nothing is written there)
--data        the data folder (particle_<tag>.em)
--out         the new project root: results/, cards/, settings/, next_iteration.txt, ... with Dynamo's names,
              exactly Dynamo's file set; log.txt is written line for line as Dynamo does (pydyn.dynlog); the
              engine's stdout also goes to gpulog.txt in the folder that contains --out, as Dynamo writes it
--gpus        physical devices to run on (default: the project's gpu_identifier_set). The GPU chunks and their
              file names always follow the project's gpu_identifier_set, as in Dynamo.
--iterations  stop after this global iteration (default: all iterations of the project)
--engine      exact (default; Dynamo 1.1.514's own kernels, needs its engine binary and the CUDA 8 libraries) or fast
--exact-sms   exact engine: reproduce the cuBLAS 8 reduction order of a GPU with this many SMs
--report      opt-in JSON diagnostics ($PYDYN_REPORT); refused inside --out
--progress-log  opt-in copy of the progress lines on stdout; refused inside --out
--keep-temp   keep temp/ite_XXXX (Dynamo deletes it at the end of every assemble)
--table-initial  the table used by the iteration-1 comparison (default: vpr.file_table_initial relative to the
              folder that contains --source, i.e. Dynamo's working directory)
See pydyn/dynproject.py for the specification and the list of unsupported options.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--table-initial", default=None)
    ap.add_argument("--engine", choices=("exact", "fast"), default=None,
                    help="GPU alignment engine (default: $PYDYN_ENGINE, else exact)")
    ap.add_argument("--exact-sms", type=int, default=None)
    ap.add_argument("--report", default=None, metavar="PATH")
    ap.add_argument("--progress-log", default=None, metavar="PATH")
    a = ap.parse_args(argv)
    if os.path.abspath(a.out) == os.path.abspath(a.source):
        ap.error("--out must differ from --source (the source project is read-only)")
    from pydyn.engine import preflight
    preflight(a.engine)                                   # before any project work (review M1)
    from pydyn.dynproject import DynamoProject
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else None
    pr = DynamoProject(a.source, a.data, a.out, gpus=gpus, keep_temp=a.keep_temp,
                       table_initial=a.table_initial, engine=a.engine, exact_sms=a.exact_sms, report=a.report,
                       progress_log=a.progress_log)
    rep = pr.run(iterations=a.iterations)
    print(json.dumps({"total_wall_s": rep.get("total_wall_s"),
                      "iterations": [{k: it[k] for k in ("ite", "wall_s", "setup_s", "compute_s", "assemble_s")}
                                     for it in rep["iterations"]]}))


if __name__ == "__main__":
    main()
