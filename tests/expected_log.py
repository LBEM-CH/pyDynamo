"""The expected log.txt of a finished Dynamo 1.1.514 standalone_gpu project, rebuilt from its own files with
pydynamo.project.logfile's line builders: the reference that test_logfile compares Dynamo's log.txt with."""
import glob
import os

from pydynamo.project.logfile import DeviceRun, EngineCard, _num, asctime, assemble_average_done_line, assemble_average_lines, assemble_head_lines, bandpass_lines, bandpass_too_many_references_line, card_ite_path, card_iteref_path, check_ok_lines, convergence_lines, datestr_to_struct, em_dims, ending_lines, engine_stdout, gpu_chunks, load_vpr, returning_control_line, round_array, round_leaders, setup_banner, setup_done_line, setup_freeze_lines, setup_shifts_lines, setup_skip_reference_line, stdout_to_log_lines


def read_table_cols12(path):
    """Columns 1 (tag) and 2 (aligned flag) of a .tbl text table."""
    tags, flags = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 2:
                tags.append(int(_num(t[0])))
                flags.append(_num(t[1]))
    return tags, flags


def read_card_dict(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v[:-1] if v.endswith(";") else v
    return out


def _project_file(project_dir, cwd, name_project, path):
    """A card path ('./<name>/results/...', relative to Dynamo's working directory) on disk. A project folder
    that was renamed after the run (the user's bin8_align_1 still says name_project=bin4_align_1) is found
    through project_dir."""
    for pre in ("./%s/" % name_project, "%s/" % name_project):
        if path.startswith(pre):
            return os.path.join(project_dir, path[len(pre):])
    return path if os.path.isabs(path) else os.path.join(cwd, path)


def _data_padding(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "particle_*.em")))
    if not files:
        raise FileNotFoundError("no particle_*.em in %s: cannot determine the particle name padding" % data_dir)
    return len(os.path.basename(files[0])[len("particle_"):-3])


class RunValues:
    """Run-dependent values (dates, times, durations, GPU model, multiprocessor count). Missing values fall back to
    placeholders derived from the card times, which the checker normalises away. A live pyDynamo run does not need
    this class: it passes its own clock and device query straight to the formatters.
    keys of `values`: ('t_start', ite) datestr of setup entry; ('t_enter', ite, ref) / ('t_leave', ite, ref)
    asctime strings of the engine run; ('cpu', ite, ref) seconds; ('t_return', ite, ref) datestr after the engine;
    ('average_s', ite) seconds of the averaging."""

    def __init__(self, gpu_name="<GPU>", multiprocessors=0, values=None):
        self.gpu_name, self.multiprocessors = gpu_name, multiprocessors
        self.v = values or {}

    def get(self, key, default):
        return self.v.get(key, default)


def expected_log_lines(project_dir, cwd=None, data_dir=None, iterations=None, run=None, convergence=None,
                       name_project=None):
    """The expected log.txt lines of ONE complete execution (iterations 1..N, or the given list) of a finished
    1.1.514 standalone_gpu project, rebuilt from the project's own files:
      cards/ite_N/card_ite_ite_N.card (time_start_computation, time_end_refinement, time_end_iteration, nref,
      area_search_modus, freeze flags) and card_iteref_ref_R_ite_N.card (every engine field);
      results/ite_N/currently_surviving_references_ite_N.txt; results/ite_N/starting_values/starting_table_*
      (the GPU chunks: temp/ is deleted by assemble) and the *_transformed .em headers (dimensions);
      settings/virtual_project.mat (rounds, adaptive bandpass parameters); results/bandpass_resolution.txt.
    cwd: Dynamo's working directory (default: the project's parent); data_dir: the particle folder (default
    cwd/<folder_data>), used only for the file-name padding. convergence: {ite: dict(kind=..., error_text=...)}
    (default 'not_converged', the branch of a convergence_type-0 project). run: RunValues."""
    pd = os.path.abspath(project_dir)
    cwd = cwd or os.path.dirname(pd)
    run = run or RunValues()
    convergence = convergence or {}
    vpr = load_vpr(os.path.join(pd, "settings", "virtual_project.mat"))
    arr = round_array(vpr)
    N = len(arr)
    iterations = list(range(1, N + 1)) if iterations is None else list(iterations)
    res_file = os.path.join(pd, "results", "bandpass_resolution.txt")
    resolutions = None
    if os.path.exists(res_file):
        with open(res_file) as f:
            resolutions = [_num(t) for t in f.read().split()]
    lines = []
    for ite in iterations:
        rnd = arr[ite - 1]
        ci = read_card_dict(os.path.join(pd, "cards", "ite_%04d" % ite, "card_ite_ite_%04d.card" % ite))
        name = name_project or ci["name_project"]
        cip = card_ite_path(name, ite)
        tsc = ci.get("time_start_computation", "RECORDING PROBLEMS IN THE GPU")
        tbeg = ci.get("time_end_refinement", tsc)
        tend = ci.get("time_end_iteration", tsc)
        lines += setup_banner(ite, cip, run.get(("t_start", ite), tsc))
        # freeze (angles), then shifts
        has_ff = "cone_freeze_reference_r1" in vpr
        leaders = sorted(round_leaders(arr))
        begin = ([x for x in leaders if x < ite] or [ite])[-1]
        if has_ff and ite > 1:
            fc, fi = float(vpr["cone_freeze_reference_r%d" % rnd]), float(vpr["inplane_freeze_reference_r%d" % rnd])
            if fc == 0 and fi == 0:
                lines += setup_freeze_lines(ite, True, None, None)
            else:
                sel = lambda f: {0: ite, 1: begin, 2: 1}[int(f)]
                lines += setup_freeze_lines(ite, True, sel(fc), sel(fi))
        else:
            lines += setup_freeze_lines(ite, has_ff, None, None)
        lines += setup_shifts_lines(ite, _num(ci["area_search_modus"]), begin)
        refs_txt = os.path.join(pd, "results", "ite_%04d" % ite, "currently_surviving_references_ite_%04d.txt" % ite)
        with open(refs_txt) as f:
            refs = [int(_num(t)) for t in f.read().split()]
        nref = int(_num(ci["nref"]))
        for r in range(1, nref + 1):
            if r not in refs:
                lines.append(setup_skip_reference_line(r))
        lines.append(setup_done_line(tsc))
        # compute: one engine run per surviving reference
        for ref in refs:
            cpath = os.path.join(pd, "cards", "ite_%04d" % ite, "card_iteref_ref_%03d_ite_%04d.card" % (ref, ite))
            card = EngineCard(cpath, card_iteref_path(name, ref, ite))
            dims = {"template": em_dims(_project_file(pd, cwd, card.name_project, card.file_template)),
                    "mask": em_dims(_project_file(pd, cwd, card.name_project, card.file_mask)),
                    "tfmask": em_dims(_project_file(pd, cwd, card.name_project, card.file_tfmask))}
            tags, flags = read_table_cols12(os.path.join(
                pd, "results", "ite_%04d" % ite, "starting_values", "starting_table_ref_%03d_ite_%04d.tbl" % (ref, ite)))
            sel_tags = [t for t, fl in zip(tags, flags) if fl != 0]
            devs = card.devices()
            folder = card.folder_data.rstrip("/")
            pad = _data_padding(data_dir or os.path.join(cwd, folder))
            chunks = gpu_chunks(sel_tags, devs, int(_num(ci.get("how_many_processors", "1"))))
            druns = [DeviceRun(d, ["%s/particle_%0*d.em" % (folder, pad, t) for t in ch], run.gpu_name,
                               run.multiprocessors) for (_, d), ch in zip(devs, chunks)]
            t_enter = run.get(("t_enter", ite, ref), asctime(datestr_to_struct(tsc)))
            t_leave = run.get(("t_leave", ite, ref), asctime(datestr_to_struct(tbeg)))
            out = engine_stdout(card, druns, dims, t_enter, t_leave, run.get(("cpu", ite, ref), 0.0))
            lines += stdout_to_log_lines(out)
            lines.append(returning_control_line(run.get(("t_return", ite, ref), tbeg)))
        # assemble
        lines += assemble_head_lines()
        lines += assemble_average_lines(ite, nref, refs)
        lines.append(assemble_average_done_line(run.get(("average_s", ite), 0.0)))
        conv = dict(convergence.get(ite, {}))
        kind = conv.pop("kind", "not_converged")
        nxt = conv.pop("next_iteration", ite + 1)
        lines += convergence_lines(nxt, kind, **conv)
        if int(_num(str(vpr.get("adaptive_bandpass", 0)))):            # dynamo_iteration_assemble.m:813-822
            if nref > 2:
                lines.append(bandpass_too_many_references_line(ite))
            elif nref == 2:
                if resolutions is None or len(resolutions) < ite:
                    raise FileNotFoundError("results/bandpass_resolution.txt lacks iteration %d" % ite)
                lines += bandpass_lines(ite, name, bool(vpr.get("adaptive_bandpass_symmetrized", 0)),
                                        str(vpr["sym_r%d" % rnd]), resolutions[ite - 1],
                                        float(vpr["adaptive_bandpass_threshold"]),
                                        float(vpr["adaptive_bandpass_pushback"]), ite + 1 <= N)
        lines += ending_lines(cip, tsc, tbeg, tend)
        lines += check_ok_lines(ite)
    return lines
