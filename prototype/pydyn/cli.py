"""Command-line entry points.

  pydynamo execute_project NAME [--gpus 0,1,2,3|all] [--engine exact|fast] [--exact-sms N] [--keep-temp]
                         [--iterations N] [--report PATH]
  dynamo_execute_project NAME [same options]   Dynamo's command name: what <name>.exe (byte-identical to Dynamo's, last
                         line `dynamo_execute_project <name>`) runs when pyDynamo's environment is active
  pydynamo check NAME          dvcheck (exit status 1 if Dynamo reports errors)
  pydynamo unfold NAME         dvunfold (writes NAME.exe)
  pydynamo info NAME           rounds, iterations, next iteration, data and seeds of a project
  pydynamo                     an interactive Python console with Dynamo's commands loaded (type dcp.new(...) there)
  dtrandomize_azimuth --i TABLE --o TABLE           Dynamo's dtrandomize_azimuth on files
  daverage --d DATA --t TABLE --o AVERAGE.em [--mw N] [--device auto|gpu|cpu] [--all-outputs]   Dynamo's daverage, the average written
  dcp [args]                   opens the dcp window (pydyn.webui.main; dcp --help)
  dmapview FILE / dview FILE   Dynamo's dmapview / dview viewers (pydyn.viewers)

Run from the folder that contains the project (as Dynamo). 'python -I -m pydyn ...' is the same as 'pydynamo'.
The console scripts restart themselves once in a clean environment (no PYTHONPATH / LD_LIBRARY_PATH / user site),
so SBGrid or other Python settings of the login shell cannot leak in.
"""
import argparse
import os
import sys

from .execute import UNSET


def _clean_reexec(entry):
    """Restart this entry point once as `python -I -c ...` without the leaking variables. The package folder is put
    on sys.path explicitly, so this also works when pydyn is not pip-installed."""
    if os.environ.get("PYDYN_CLEAN_ENV") == "1":
        return
    if sys.flags.isolated and not any(v in os.environ for v in UNSET):
        return
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE="/dev/null", PYDYN_CLEAN_ENV="1")
    proto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "import sys; sys.path.insert(0, %r); from pydyn.cli import %s as f; sys.exit(f())" % (proto, entry)
    os.execve(sys.executable, [sys.executable, "-I", "-c", code] + sys.argv[1:], env)


def _info(name):
    from . import vpr as V
    from .dynproject import round_array
    v = V.load(name)
    if v is None:
        return 1
    rounds = round_array(v)
    nxt = os.path.join(v["name_project"], "next_iteration.txt")
    print("project            %s (name_project %s)" % (name, v["name_project"]))
    print("path (creation)    %s" % v["path"])
    print("data               %s" % v["folder_data"])
    print("table / template   %s / %s" % (v["file_table_initial"], v["file_template_initial"]))
    print("mask / fmask       %s / %s" % (v["file_mask"], v["file_fmask_initial"]))
    print("destination        %s, gpus %s" % (v["destination"], v["gpu_identifier_set"]))
    print("rounds per ite     %s (%d iterations)" % (rounds, len(rounds)))
    print("unfolded           %s" % ("yes" if "feature_data_sidelength" in v else "no"))
    if os.path.isfile(nxt):
        print("next_iteration     %s" % open(nxt).read().strip())
    return 0


def _execute_args(e):
    """Arguments of `pydynamo execute_project` and of the `dynamo_execute_project` command."""
    e.add_argument("name")
    e.add_argument("--gpus", default=None,
                   help="comma-separated device ids, or all: every GPU this process sees (in a Slurm job, the job's "
                        "own), up to the number of particle parts. Default: gpu_identifier_set, as Dynamo")
    e.add_argument("--engine", choices=("exact", "fast"), default=None,
                   help="GPU alignment engine. exact: Dynamo 1.1.514's own kernels, cuFFT 8 and cuBLAS 8, bit-identical "
                        "to the official engine on the same GPU model; fast: pyDynamo's batched engine (float32 "
                        "differences at the 7th digit can flip near-ties). Default: $PYDYN_ENGINE, else exact")
    e.add_argument("--exact-sms", type=int, default=None,
                   help="exact engine: reproduce the cuBLAS 8 reduction order of a GPU with this many SMs, e.g. 64 for "
                        "an RTX A5000 (Dynamo's tables depend on the GPU model through it); default: the GPU in use")
    e.add_argument("--keep-temp", action="store_true")
    e.add_argument("--iterations", type=int, default=None, help="stop after this iteration")
    e.add_argument("--report", default=None, metavar="PATH",
                   help="opt-in JSON diagnostics ($PYDYN_REPORT); refused inside the project folder")


def _run_execute(a):
    from .execute import execute_project
    gpus = "all" if a.gpus == "all" else ([int(g) for g in a.gpus.split(",")] if a.gpus else None)
    ok = execute_project(a.name, gpus=gpus, keep_temp=a.keep_temp,
                         iterations=a.iterations, engine=a.engine, exact_sms=a.exact_sms, report=a.report)
    return 0 if ok else 1


def execute_main(argv=None):
    """The `dynamo_execute_project NAME` command (console script). With pyDynamo's environment active, the
    byte-identical Dynamo `<name>.exe` runs pyDynamo; it restarts itself once in a clean environment like `pydynamo`."""
    if argv is None:
        _clean_reexec("execute_main")
    ap = argparse.ArgumentParser(prog="dynamo_execute_project", description="Run an unfolded dcp project (pyDynamo).")
    _execute_args(ap)
    return _run_execute(ap.parse_args(argv))


def main(argv=None):
    if argv is None:
        _clean_reexec("main")
    if not (sys.argv[1:] if argv is None else argv):
        return console_main()
    ap = argparse.ArgumentParser(prog="pydynamo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("execute_project", help="run an unfolded project (what NAME.exe does)")
    _execute_args(e)
    for c in ("check", "unfold", "info"):
        sub.add_parser(c).add_argument("name")
    a = ap.parse_args(argv)
    if a.cmd == "execute_project":
        return _run_execute(a)
    if a.cmd == "check":
        from .vprcheck import check
        r = check(a.name)
        return 1 if (r.fields_error or r.warning_state == "Aborted") else 0
    if a.cmd == "unfold":
        from .vpr import unfold
        script, status = unfold(a.name)
        return 0 if status == "ok" else 1
    if a.cmd == "info":
        return _info(a.name)
    return 2


def dcp_main():
    """The `dcp` command: the dcp window (pydyn.webui)."""
    _clean_reexec("dcp_main")
    from . import webui
    return webui.main(sys.argv[1:])


def _io_option(ap, short, words, **kw):
    """An option given as -x, --x or --word (e.g. -i / --i / --input)."""
    ap.add_argument("-" + short, "--" + short, *["--" + w for w in words], **kw)


def dtrandomize_azimuth_main(argv=None):
    """Shell command: dtrandomize_azimuth --i TABLE --o TABLE (Dynamo's dtrandomize_azimuth(tIn, 'o', tOut))."""
    if argv is None:
        _clean_reexec("dtrandomize_azimuth_main")
    ap = argparse.ArgumentParser(
        prog="dtrandomize_azimuth",
        description="Dynamo's dtrandomize_azimuth on table files: column 9 (narot) := 360*rand. Each command starts the "
                    "random stream as a fresh MATLAB session does, so it gives the same angles as Dynamo's first "
                    "dtrandomize_azimuth after starting MATLAB.")
    _io_option(ap, "i", ["input"], required=True, dest="input", metavar="TABLE", help="input table (.tbl)")
    _io_option(ap, "o", ["output"], required=True, dest="output", metavar="TABLE", help="output table (.tbl)")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    from .console import dtrandomize_azimuth
    from .vpr import DynamoError
    try:
        t = dtrandomize_azimuth(a.input, "o", a.output)
    except (DynamoError, OSError, NotImplementedError, ValueError) as e:
        print("dtrandomize_azimuth: %s" % e)
        return 1
    print("dtrandomize_azimuth: %d particles, azimuths randomized -> %s" % (len(t), a.output))
    return 0


def daverage_main(argv=None):
    """Shell command: daverage --d DATA --t TABLE --o FILE (Dynamo's oa = daverage(DATA, 't', TABLE);
    dwrite(oa.average, FILE))."""
    if argv is None:
        _clean_reexec("daverage_main")
    ap = argparse.ArgumentParser(
        prog="daverage",
        description="Dynamo's daverage of a data folder with a table (rotate, shift and sum every particle, as "
                    "dynamo_average does from the console). --o writes the average, as dwrite(oa.average, FILE) does; "
                    "Dynamo's own 'o' option, which also writes raw_ and fweight_ files, is --all-outputs.")
    _io_option(ap, "d", ["data"], required=True, dest="data", metavar="FOLDER", help="data folder of the particles")
    _io_option(ap, "t", ["table"], required=True, dest="table", metavar="TABLE", help="table (.tbl)")
    _io_option(ap, "o", ["output"], required=True, dest="output", metavar="FILE", help="the average (.em)")
    ap.add_argument("--mw", type=int, default=None, metavar="N",
                    help="accepted as in Dynamo; no effect on the result")
    ap.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                    help="where to compute (the result is the same); default: a GPU if one is visible")
    ap.add_argument("--all-outputs", action="store_true",
                    help="Dynamo's 'o' option: also write raw_<FILE> and fweight_<FILE> next to it")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    from .console import daverage, dwrite
    from .vpr import DynamoError
    kw = {"device": a.device}
    if a.mw is not None:
        kw["mw"] = a.mw
    if a.all_outputs:
        kw["o"] = a.output
    try:
        oa = daverage(a.data, "t", a.table, **kw)
    except (DynamoError, OSError, NotImplementedError, ValueError) as e:
        print("daverage: %s" % e)
        return 1
    if not a.all_outputs:
        dwrite(oa.average, a.output)
    print("daverage: average of %s with %s -> %s" % (a.data, a.table, a.output))
    return 0


CONSOLE_BANNER = """pyDynamo console (Python %s). Dynamo's commands are loaded (from pydyn import *), e.g.
  dtrandomize_azimuth('particles_edit.tbl', o='particles_edit_mod.tbl')
  oa = daverage('filamentsData_ctf', t='particles_edit_mod.tbl')
  dwrite(oa.average, 'raw_template_ctf.em')
  dcp.new('bin8_align_1', d='filamentsData_ctf', template='raw_template_ctf.em', masks='default', t='particles_edit_mod.tbl')
Also: dcp.open('p'), dcp(), dread, dvput, dvcheck, dvunfold, dveo. From the shell, without this console:
  dtrandomize_azimuth --i IN.tbl --o OUT.tbl     daverage --d DATA --t TABLE.tbl --o AVERAGE.em     dcp --p NAME --d DATA ...
Folder: %s      exit() or Ctrl-D leaves."""


def console_main():
    """`pydynamo` without arguments: an interactive Python console with Dynamo's commands loaded, the place to
    type dcp.new(...) as at the MATLAB prompt (the shell itself cannot read that syntax)."""
    import code
    import platform
    try:
        import readline  # noqa: F401  (line editing and history)
    except ImportError:                              # pragma: no cover
        pass
    import pydyn
    ns = {"__name__": "__console__"}
    exec("from pydyn import *", ns)
    ns["pydyn"] = pydyn
    code.interact(banner=CONSOLE_BANNER % (platform.python_version(), os.getcwd()), local=ns, exitmsg="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
