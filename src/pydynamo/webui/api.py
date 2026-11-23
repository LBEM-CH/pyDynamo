"""One dcp session behind the HTML view: the DcpController, its messages, and JSON-ready results.

The HTML view (pydynamo.webui.static) has no logic of its own. Every action is a call of a whitelisted DcpController
method (ACTIONS) or one of the few view-level actions (VIEW_ACTIONS: closing the masks
window, the mask editor, the run hint after unfold). Errors are handled as Dynamo's dcp shows them:
Dynamo errors and unported paths become red messages, never a crash.

Calls are serialised with one lock (the controller is the global DYNCP of a dcp session, not thread-safe). Reading
messages does not take the lock, so the view keeps showing progress during a long call ("average all").
"""
import os
import threading
import time
import traceback

# DcpController methods the view may call (window actions of S/@dcp and dynamo_project_manager_easy)
ACTIONS = frozenset((
    "enter_project_name", "create", "open", "list_projects",
    "data_enter", "table_enter", "table_blank", "table_random", "template_enter", "template_average",
    "mask_enter", "cmask_enter", "fmask_enter", "smask_enter", "masks_window_opened", "masks_ellipsoid",
    "masks_copy_mask", "masks_use_ones", "smask_use_ones", "masks_use_defaults", "template_size",
    "create_even_odd_project", "golden_standard_derive", "abp_parameters", "abp_set",
    "view_template", "plot_orientations", "view_mask", "overlay_mask", "view_last_averages",
    "plot_adaptive_filtering_fsc",
    "rounds_table", "comment", "parameter_help", "set_value", "active_rounds", "iteration_rounds", "add_round", "remove_round",
    "environment_values", "set_destination", "environment_edit",
    "check", "unfold", "execution_order",
))
VIEW_ACTIONS = frozenset(("masks_closed", "editor_open", "editor_save", "editor_show", "editor_view", "editor_slices",
                          "editor_load", "use_gpus",
                          "unfold_and_hint", "run_hint",
                          "run_settings", "run_set", "run_start", "run_status", "run_stop"))
# runtime fields shown by the view (main window and the step windows)
FIELDS = ("name_project", "folder_data", "file_table_initial", "file_template_initial", "file_mask",
          "file_mask_classification", "file_fmask_initial", "file_mask_smoothing", "destination", "gpu_identifier_set",
          "nref_r1")
MASK_FIELDS = {"mask": "file_mask", "cmask": "file_mask_classification", "fmask": "file_fmask_initial",
               "smask": "file_mask_smoothing"}


class ApiError(Exception):
    pass


def jsonable(x):
    """Controller results as JSON values: Check and result objects become dicts with their attributes."""
    import numpy as np
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x) if isinstance(x, (float, np.floating)) else int(x)
        return v if v == v and v not in (float("inf"), float("-inf")) else str(v)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):            # namedtuple, e.g. vpr_check.CheckResult
        d = {k: jsonable(v) for k, v in zip(x._fields, x)}
        d["__type__"] = type(x).__name__
        return d
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if hasattr(x, "__dict__"):
        d = {k: jsonable(v) for k, v in vars(x).items() if not k.startswith("_")}
        d["__type__"] = type(x).__name__
        return d
    return str(x)


def data_summary(df):
    """The data window's facts without the per-particle lists (a data folder can hold 100,000 particles: the tags
    and file paths stay on the server): the number, the box size, and the file names Dynamo found, as a pattern
    (particle_ + the tag written with `padding` digits + the extension) with the first and last name."""
    if not df:
        return None
    out = {"N": int(df.get("N") or 0), "l": jsonable(df.get("l")), "size": jsonable(df.get("size"))}
    tags, pad, ext = df.get("tags") or [], df.get("padding"), df.get("extension") or "em"
    if isinstance(pad, (int, float)) and pad == pad and len(tags):
        p = int(pad)
        out["pattern"] = "particle_%s.%s" % ("#" * p, ext)
        out["first"] = "particle_%0*d.%s" % (p, int(tags[0]), ext)
        out["last"] = "particle_%0*d.%s" % (p, int(tags[-1]), ext)
    return out


def _tiles(out, vols, axis, k):
    """Screen tiles (pydynamo.viewers.logic.view_tile, Dynamo's mapview orientation) of the volumes in `vols`
    ({"t": template, "m": mask}, each {"v": [z, y, x] array, "stats": ...}) at frame k (1-based, default central),
    base64 float32, into `out`. A mask whose size differs from the template is dropped with an error."""
    import base64
    import numpy as np
    from pydynamo.viewers.logic import n_frames, view_tile
    if "t" in vols and "m" in vols and vols["t"]["v"].shape != vols["m"]["v"].shape:
        st, sm = vols["t"]["v"].shape, vols["m"]["v"].shape
        out["error"] = "size mismatch: template %s, mask %s (x y z)" % (" ".join(map(str, st[::-1])),
                                                                         " ".join(map(str, sm[::-1])))
        vols = {"t": vols["t"]}
    shape = next(iter(vols.values()))["v"].shape
    n = n_frames(shape, axis)
    k = n // 2 + 1 if k in (None, "") else max(1, min(n, int(k)))
    out.update(k=k, n=n, shape_xyz=list(shape[::-1]))
    for key, h in vols.items():
        tile = np.ascontiguousarray(view_tile(h["v"], axis, k), dtype=np.float32)
        out["h"], out["w"] = tile.shape
        out[key] = base64.b64encode(tile.tobytes()).decode("ascii")
        out[key + "stats"] = h["stats"]
    return out


def _mask_log_dir():
    """The folder of the mask editor's logs: pydynamo's per-user cache ($PYDYNAMO_CACHE_DIR, else $XDG_CACHE_HOME/pydynamo or
    ~/.cache/pydynamo, as pydynamo.engine.kernels), subfolder mask_logs. Not the project: Dynamo's mask editor writes the mask
    only (dynamo_mask_GUI.m:385 dynamo_write(mask,file_output)), so the project folder holds Dynamo's files only."""
    base = os.environ.get("PYDYNAMO_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "pydynamo")
    return os.path.join(base, "mask_logs")


def mask_log_path(mask_file):
    """The log pydynamo writes for a mask made in the mask editor: in _mask_log_dir, named after the mask's real path
    (its file name + a hash of the path)."""
    import hashlib
    real = os.path.realpath(str(mask_file))
    return os.path.join(_mask_log_dir(), "%s.%s.log" % (os.path.basename(real),
                                                         hashlib.sha1(os.fsencode(real)).hexdigest()[:16]))


def find_mask_log(mask_file):
    """The log to read for `mask_file`: the one in the cache (mask_log_path), else '<mask>.log' next to the mask (a
    log carried with a copied mask, or written there by earlier pydynamo versions; only read, never written). None
    without one."""
    if not mask_file:
        return None
    for p in (mask_log_path(mask_file), str(mask_file) + ".log"):
        if os.path.isfile(p):
            return p
    return None


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_mask_log(mask_file, settings):
    """Write (overwrite) the log of the settings that created `mask_file` (at mask_log_path, outside the project):
    plain 'name = value' lines, only the ticked settings, and the mask's md5 so a later change of the mask can be
    told."""
    from pydynamo.masks import GUI_RADIO_SETTINGS
    lines = ["# pyDynamo mask editor: the settings that created %s (Create mask, %s)"
             % (os.path.basename(mask_file), time.strftime("%d-%b-%Y %H:%M:%S")),
             "# Loading this mask in the mask editor fills in these settings; the next Create mask overwrites this file.",
             "file = %s" % os.path.realpath(str(mask_file)), "md5 = %s" % _md5(mask_file),
             "type = %s" % settings.get("type", ""), "size = %s" % str(settings.get("size") or "").strip()]
    for p in GUI_RADIO_SETTINGS:
        if (settings.get("radio") or {}).get(p):
            lines.append("%s = %s" % (p, str((settings.get("edit") or {}).get(p, "")).strip()))
    lines.append("invert = %d" % int(bool(settings.get("invert"))))
    p = mask_log_path(mask_file)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".%d.tmp" % os.getpid()
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, p)


def read_mask_log(mask_file):
    """The settings in the log of `mask_file` (find_mask_log), or None without one. 'matches': the mask is still the
    one the log was written for (same md5); 'created': the time in its header; 'log': its path."""
    from pydynamo.masks import GUI_RADIO_SETTINGS
    p = find_mask_log(mask_file)
    if p is None:
        return None
    kv, created = {}, None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# pyDynamo mask editor") and "(Create mask, " in line:
                created = line.rsplit("(Create mask, ", 1)[1].rstrip(")")
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = (x.strip() for x in line.split("=", 1))
            kv[k] = v
    return {"type": kv.get("type") or "sphere", "size": kv.get("size"),
            "radio": {q: q in kv for q in GUI_RADIO_SETTINGS}, "edit": {q: kv.get(q, "") for q in GUI_RADIO_SETTINGS},
            "invert": kv.get("invert", "0") not in ("0", "", "false"), "created": created, "log": p,
            "matches": bool(os.path.isfile(mask_file) and kv.get("md5") == _md5(mask_file))}


def _settings_text(rec):
    from pydynamo.masks import GUI_RADIO_SETTINGS
    on = [q for q in GUI_RADIO_SETTINGS if (rec.get("radio") or {}).get(q)]
    return " \u00b7 ".join([rec.get("type", "")] + ["%s %s" % (q, str((rec.get("edit") or {}).get(q, "")).strip())
                                                     for q in on] + (["inverted"] if rec.get("invert") else []))


# Slurm's partitions and where a job can run: pydynamo.project.launch (also the Run step's local / Slurm runs)
from pydynamo.project.launch import SLURM_CACHE_S, parse_sinfo, slurm_partitions, slurm_plan   # noqa: E402,F401


import socket as _socket
_HOST = _socket.gethostname().split(".")[0]


def _int(s):
    try:
        return int(float(s))
    except ValueError:
        return None


def _vol_stats(v):
    """Whole-volume numbers for the inspector: contrast limits of the template, the levels of a (soft) mask."""
    import numpy as np
    f = v.ravel()
    sample = f[:: max(1, f.size // 2000000)]
    p1, p99 = np.percentile(sample, [1, 99])
    mn, mx = float(f.min()), float(f.max())
    soft = int(((f > 0) & (f < 1)).sum())
    return {"min": mn, "max": mx, "mean": float(f.mean(dtype=np.float64)), "std": float(f.std(dtype=np.float64)),
            "p1": float(p1), "p99": float(p99), "n_gt0": int((f > 0).sum()), "n_gt05": int((f > 0.5).sum()),
            "n_ge1": int((f >= 1).sum()), "n_soft": soft, "n": int(f.size)}


class Session:
    """The state of one dcp window (one DcpController) for the HTML view."""

    def __init__(self, project=None):
        from pydynamo.manager.controller import DcpController
        self.lock = threading.RLock()
        self._mlock = threading.Lock()
        self.log = []                                # [(seq, text, color)]
        self.busy = None                             # name of the call in progress
        self.c = DcpController(echo=self._echo)
        self.editor = None                           # (MaskEditor, box, file) of the open mask editor
        self._vols = {}                              # volumes read for the mask inspector
        self._ed_cache = {}                          # the mask editor's last live mask
        self._ed_mem = {}                            # (project, kind) -> the editor's settings in this session
        self._ed_last = {}                           # project -> the editor's last created or loaded mask file
        self._ed_kind = "align"
        self._unfold_ok_until = {}                   # project -> parameters' mtime still covered by its unfold
        self._ed_source = "default"
        self._ed_log_matches = None
        self.start_project = project
        self._echo("Folder: %s" % os.getcwd())

    # ------------------------------------------------------------------------------------------------ messages
    def _echo(self, text, color="k"):
        with self._mlock:
            self.log.append((len(self.log) + 1, text, color))

    def messages(self, after=0):
        with self._mlock:
            return [{"seq": s, "text": t, "color": c} for s, t, c in self.log[int(after):]]

    # ------------------------------------------------------------------------------------------------ calls
    def call(self, method, args=(), kwargs=None):
        if method not in ACTIONS and method not in VIEW_ACTIONS:
            raise ApiError("unknown action %r" % method)
        if not self.lock.acquire(timeout=0.2):
            return {"ok": False, "busy": self.busy, "error": "busy: %s is still running" % self.busy}
        try:
            self.busy = method
            fn = getattr(self, "_v_" + method) if method in VIEW_ACTIONS else getattr(self.c, method)
            opening = method in ("enter_project_name", "open") and args
            was_current = self._unfold_current_on_disk(str(args[0])) if opening else False
            vpr_before = self._vpr_mtime(self.c.name) if method == "check" else None
            out = self._guard(fn, *list(args), **dict(kwargs or {}))
            if opening and out.get("ok") and was_current and self.c.name:
                # opening re-saves the parameters (Dynamo's setSingleReferenceProject / setMultireferenceProject):
                # that save does not change them, so the unfold stays current
                self._unfold_ok_until[self.c.name] = self._vpr_mtime(self.c.name)
            if method == "check" and vpr_before is not None and self.c.name in self._unfold_ok_until \
                    and self._unfold_ok_until[self.c.name] >= vpr_before:
                self._unfold_ok_until[self.c.name] = self._vpr_mtime(self.c.name)
            if method in ("enter_project_name", "open", "create", "check") and out.get("ok"):
                self._restore_steps()
            return out
        finally:
            self.busy = None
            self.lock.release()

    def _guard(self, fn, *a, **kw):
        """Dynamo errors and unported paths become red messages."""
        from pydynamo.vpr import DynamoError
        try:
            r = fn(*a, **kw)
        except NotImplementedError as e:
            self._echo("[not ported in pydynamo] %s" % e, "r")
            return {"ok": False, "error": "not ported: %s" % e}
        except DynamoError as e:
            self._echo(str(e), "r")
            return {"ok": False, "error": str(e)}
        except Exception as e:                       # show, do not crash the GUI
            self._echo("ERROR: %s" % e, "r")
            self._echo(traceback.format_exc(), "r")
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        return {"ok": True, "result": jsonable(r)}

    # ------------------------------------------------------------------------------------------------ state
    def state(self):
        """What the view draws: project, step buttons (enabled / correct), the window fields, rounds summary."""
        if not self.lock.acquire(timeout=0.2):
            return {"busy": self.busy, "seq": len(self.log)}
        try:
            c = self.c
            rt = c.runtime or {}
            st = {"cwd": os.getcwd(), "project": c.name, "busy": None, "seq": len(self.log), "host": _HOST,
                  "start_project": self.start_project,
                  "enabled": dict(c.enabled), "correct": dict(c.correct), "data_facts": data_summary(c.data_facts),
                  "template_size": jsonable(c.template_size_cache), "semiaxis": c.semiaxis,
                  "mask_box": jsonable(c.mask_box), "fields": {f: jsonable(rt.get(f)) for f in FIELDS},
                  "rounds": None, "exe": None, "order": None, "editor_file": self._editor_file(),
                  "editor_info": self._editor_info(), "unfold_info": self._unfold_info()}
            if c.name is not None:
                try:
                    st["nref"] = int(float(rt["nref_r1"]))
                except (KeyError, TypeError, ValueError):
                    st["nref"] = None
                try:
                    st["rounds"] = {"active": c.active_rounds(), "iterations": jsonable(c.iteration_rounds())}
                except Exception:
                    pass
                try:
                    script = c.execution_script_name()
                    st["exe"] = script if script and os.path.isfile(script) else None
                    st["order"] = c.execution_order()
                except Exception:
                    pass
            return st
        finally:
            self.lock.release()

    def meta(self):
        """Constants of the dcp windows (labels as in Dynamo's .fig files and parameter classes)."""
        from pydynamo.manager.controller import DESTINATIONS, ENVIRONMENT_PARAMETERS, PYDYNAMO_EXECUTES, ROUND_AREAS, ROUND_AREAS_ON, STEP_ORDER, parameter
        from pydynamo.masks import GUI_RADIO_SETTINGS, GUI_TYPES
        return {"steps": list(STEP_ORDER), "round_areas": list(ROUND_AREAS), "round_areas_on": list(ROUND_AREAS_ON),
                "destinations": list(DESTINATIONS), "pydynamo_executes": list(PYDYNAMO_EXECUTES),
                "environment": [{"name": p, "pretty": parameter(p)["pretty"] or p,
                                 "comment": parameter(p)["comment"] or ""} for p in ENVIRONMENT_PARAMETERS],
                "editor_types": list(GUI_TYPES), "editor_settings": list(GUI_RADIO_SETTINGS)}

    # ------------------------------------------------------------------------------------------------ mask inspector
    def _volume(self, path):
        """A volume as [z, y, x] float32, cached by (path, mtime, size) so slicing through it stays fast."""
        import numpy as np
        from pydynamo.console import dread
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
        hit = self._vols.get(key)
        if hit is None:
            v = np.ascontiguousarray(np.asarray(dread(path), dtype=np.float32))
            hit = {"v": v, "stats": _vol_stats(v)}
            if len(self._vols) > 6:
                self._vols.clear()
            self._vols[key] = hit
        return hit

    def slices(self, which="mask", axis="z", k=None, mfile=None):
        """One screen tile of the template and of a mask, as Dynamo's mapview draws it (pydynamo.viewers.logic.view_tile:
        z: rows y from ny down to 1, columns x; x: rows z from nz down, columns y; y: rows z from nz down, columns x
        from nx down). k is 1-based (default: the central frame). The mask is the project's (`which`) or, when `mfile`
        is given, that file (e.g. the mask editor's output before it is entered)."""
        rt = self.c.runtime or {}
        tfile = rt.get("file_template_initial") or ""
        mfile = mfile or rt.get(MASK_FIELDS.get(which, "file_mask")) or ""
        out = {"which": which, "axis": axis, "template_file": tfile, "mask_file": mfile}
        if axis not in ("x", "y", "z"):
            raise ApiError("axis must be x, y or z")
        vols = {}
        for key, f in (("t", tfile), ("m", mfile)):
            if f and os.path.isfile(f):
                vols[key] = self._volume(f)
        if not vols:
            out["error"] = "neither the template nor the mask is a readable file"
            return out
        return _tiles(out, vols, axis, k)

    def _v_editor_slices(self, form, axis="z", k=None):
        """The mask editor's live view: the mask the current settings make (dynamo_mask of read_mask_settings, what
        Save would write, built in memory and cached per setting) and the editor's reference volume (the project
        template), as screen tiles like slices(). Nothing is written and no message goes to the main window."""
        import numpy as np
        from pydynamo.masks import dynamo_mask
        self._need_editor()
        if axis not in ("x", "y", "z"):
            raise ApiError("axis must be x, y or z")
        self._editor_apply(form)
        ed = self.editor[0]
        out = {"which": self.editor[1], "axis": axis, "template_file": ed.volume or "", "mask_file": ed.save_file,
               "live": True}
        vols = {}
        if ed.volume and os.path.isfile(ed.volume):
            vols["t"] = self._volume(ed.volume)
        try:
            args = ed.read_mask_settings()
            key = repr(args)
            if not args:
                out["error"] = "; ".join(ed.messages) or "the settings cannot make a mask yet"
            elif self._ed_cache.get("key") == key:
                vols["m"] = self._ed_cache["vol"]
            else:
                m = dynamo_mask(*args)
                if m is None:
                    out["error"] = "; ".join(ed.messages) or "the mask cannot be formed with these settings"
                else:
                    m = np.ascontiguousarray(np.asarray(m, dtype=np.float32))
                    self._ed_cache = {"key": key, "vol": {"v": m, "stats": _vol_stats(m)}}
                    vols["m"] = self._ed_cache["vol"]
        except NotImplementedError as e:
            out["error"] = "not ported: %s" % e
        except Exception as e:                       # a half-typed value: report it in the view, do not echo
            out["error"] = "%s: %s" % (type(e).__name__, e)
        finally:
            ed.messages.clear()
        if "m" not in vols and ed.save_file and os.path.isfile(ed.save_file):
            vols["m"] = self._volume(ed.save_file)
            out["note"] = "showing the file %s (the settings do not make a mask%s)" % (
                ed.save_file, ": " + out.pop("error") if out.get("error") else "")
        if not vols:
            out.setdefault("error", "no reference volume")
            return out
        return _tiles(out, vols, axis, k)

    @staticmethod
    def slurm_partitions():
        """The Slurm partitions (sinfo, read only, cached SLURM_CACHE_S): see parse_sinfo. [] without Slurm."""
        return slurm_partitions()

    def slurm_plan(self, gpus):
        """Where a Slurm job with `gpus` GPUs on one machine can run (see slurm_plan)."""
        try:
            n = max(1, int(gpus))
        except (TypeError, ValueError):
            n = 1
        return slurm_plan(slurm_partitions(), n)

    @staticmethod
    def gpus():
        """The GPUs of this machine (nvidia-smi): index, name, memory and current load, for the environment window."""
        import subprocess
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.SubprocessError):
            return []
        out = []
        for line in r.stdout.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 5 and p[0].isdigit():
                out.append({"index": int(p[0]), "name": p[1], "mem_total": _int(p[2]), "mem_used": _int(p[3]),
                             "util": _int(p[4])})
        return out

    # ------------------------------------------------------------------------------------------------ file browser
    @staticmethod
    def listdir(path=".", size_limit=2000):
        """Folder listing for the file chooser (the files of this machine, relative to the project folder). Every
        entry is returned: a data folder can hold 100,000 particles. Folders and files are told apart from the
        directory listing itself (d_type, no per-file call on NFS); sizes are read only for folders of at most
        `size_limit` files (one stat each)."""
        base = os.path.abspath(path or ".")
        if not os.path.isdir(base):
            raise ApiError("not a folder: %s" % path)
        dirs, files = [], []
        with os.scandir(base) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    (dirs if e.is_dir() else files).append(e.name)
                except OSError:
                    continue
        key = str.lower
        dirs.sort(key=key)
        files.sort(key=key)
        sizes = None
        if len(files) <= size_limit:
            sizes = []
            for n in files:
                try:
                    sizes.append(os.stat(os.path.join(base, n)).st_size)
                except OSError:
                    sizes.append(None)
        return {"abs": base, "rel": os.path.relpath(base), "parent": os.path.dirname(base),
                "dirs": dirs, "files": files, "sizes": sizes}

    # ------------------------------------------------------------------------------------------------ view actions
    def _v_masks_closed(self):
        """dcp.closeRequestMasks masks step right/wrong from the isCorrect flags."""
        c = self.c
        ok = all(c.correct.get(k) for k in ("mask", "cmask", "fmask"))
        if not ok:
            self._echo("Masks are not yet correct!", "r")
        c._step_status("masks", "right" if ok else "wrong")
        return ok

    def _unfold_info(self):
        """Is the open project unfolded, read from its files: Dynamo's unfold saves the parameters first and writes the
        execution script last, so the script is current while it is not older than the project's parameters
        (settings/virtual_project.mat). Nothing of this is stored by pyDynamo."""
        c = self.c
        if c.name is None:
            return None
        try:
            script = c.execution_script_name()
        except Exception:
            return None
        vpr_file = os.path.join(c.name, "settings", "virtual_project.mat")
        if not (script and os.path.isfile(script) and os.path.isfile(vpr_file)):
            return None
        t_exe, t_vpr = os.stat(script).st_mtime, os.stat(vpr_file).st_mtime
        ok_until = self._unfold_ok_until.get(c.name)
        return {"exe": script, "at": time.strftime("%d-%b-%Y %H:%M", time.localtime(t_exe)),
                "current": t_exe >= t_vpr or (ok_until is not None and t_vpr <= ok_until)}

    @staticmethod
    def _vpr_mtime(name):
        f = os.path.join(str(name), "settings", "virtual_project.mat") if name else ""
        return os.stat(f).st_mtime if f and os.path.isfile(f) else None

    @staticmethod
    def _unfold_current_on_disk(name):
        """Before opening: the project's .exe is not older than its parameters."""
        exe, t_vpr = "%s.exe" % name, Session._vpr_mtime(name)
        return t_vpr is not None and os.path.isfile(exe) and os.stat(exe).st_mtime >= t_vpr

    def _restore_steps(self):
        """A project that is already unfolded (and not changed since) keeps its unfold and run steps open when it is
        opened again, instead of asking for check and unfold anew."""
        info = self._unfold_info()
        if info and info["current"]:
            self.c._enable("unfold", True)
            self.c._enable("run", True)

    def _editor_info(self):
        """The mask editor's mask, for the masks page without opening the editor: file, creation time, the settings
        that made it (from its log), whether the mask changed since, and the boxes that use it now."""
        f = self._editor_file()
        if not f:
            return None
        rec = read_mask_log(f)
        settings = _settings_text(rec) if rec else None
        rt = self.c.runtime or {}
        used = [box for box, field in (("alignment", "file_mask"), ("classification", "file_mask_classification"),
                                       ("smoothing", "file_mask_smoothing"))
                if rt.get(field) and os.path.normpath(str(rt[field])) == os.path.normpath(f)]
        return {"file": f, "saved_at": time.strftime("%d-%b-%Y %H:%M", time.localtime(os.stat(f).st_mtime)),
                "settings": settings, "used_as": used, "log": rec["log"] if rec else None,
                "log_matches": rec["matches"] if rec else None}

    def _editor_file(self):
        """The mask the editor last created or loaded in this project, else the editor's default file if it exists."""
        name = self.c.name
        if name is None:
            return None
        for f in (self._ed_last.get(name), "%s/settings/editorMask.em" % name):
            if f and os.path.isfile(f):
                return f
        return None

    def _editor_state(self):
        ed, box, file = self.editor
        f = ed.save_file
        exists = bool(f) and os.path.isfile(f)
        return {"size": ed.size, "type": ed.type, "radio": dict(ed.radio), "edit": dict(ed.edit),
                "invert": bool(ed.radio.get("invert")), "save_file": f, "box": box, "file": file,
                "exists": exists, "saved_at": time.strftime("%d-%b-%Y %H:%M", time.localtime(os.stat(f).st_mtime))
                if exists else None, "source": self._ed_source,
                "log": find_mask_log(f),
                "log_matches": self._ed_log_matches}

    def _editor_settings(self, form):
        """Put settings (a form or a remembered record) into the editor, without touching anything else."""
        from pydynamo.masks import GUI_RADIO_SETTINGS
        ed = self.editor[0]
        if form.get("size") is not None:
            ed.size = form["size"]
        ed.set_type(form.get("type", ed.type))
        for p in GUI_RADIO_SETTINGS:
            ed.select(p, (form.get("edit") or {}).get(p, ""), on=bool((form.get("radio") or {}).get(p, False)))
        ed.radio["invert"] = bool(form.get("invert", False))

    def _editor_apply(self, form):
        from pydynamo.masks import GUI_RADIO_SETTINGS
        ed = self.editor[0]
        ed.size = form.get("size", ed.size)
        ed.set_type(form.get("type", ed.type))
        for p in GUI_RADIO_SETTINGS:
            ed.select(p, form.get("edit", {}).get(p, ed.edit[p]), on=bool(form.get("radio", {}).get(p, False)))
        ed.radio["invert"] = bool(form.get("invert", False))
        if "save_file" in form:
            ed.save_file = str(form["save_file"]).strip()
        # the settings of this session: reopening the editor brings them back
        self._ed_mem[(self.c.name, self._ed_kind)] = {k: form.get(k) for k in ("size", "type", "radio", "edit",
                                                                                 "invert", "save_file")}

    def _editor_flush(self, color="k"):
        ed = self.editor[0]
        for msg in ed.messages:
            self._echo("[mask editor] %s" % msg, color)
        ed.messages.clear()

    def _v_editor_open(self, kind="align"):
        """dynamo_mask_GUI from the masks window. Reopening it brings back this session's settings; otherwise the
        settings in the log of its mask file (written by Create mask); otherwise Dynamo's defaults."""
        r = self.c.mask_editor(kind)
        if not r:
            self.editor = None
            return None
        self.editor, self._ed_kind = r, kind
        mem = self._ed_mem.get((self.c.name, kind))
        if mem:
            self._editor_settings(mem)
            if mem.get("save_file"):
                self.editor[0].save_file = mem["save_file"]
            self._ed_source = "session"
            rec = read_mask_log(self.editor[0].save_file)
            self._ed_log_matches = rec["matches"] if rec else None
        else:
            rec = read_mask_log(self.editor[0].save_file)
            if rec:
                self._editor_settings(rec)
            self._ed_source = "log" if rec else "default"
            self._ed_log_matches = rec["matches"] if rec else None
        return self._editor_state()

    def _v_editor_load(self, path):
        """Load an existing mask file into the editor: it becomes the editor's file (shown, transferable). With a
        log (find_mask_log: written by Create mask, or carried next to the mask), its settings fill the editor;
        without one they are cleared and the view shows the file itself."""
        self._need_editor()
        path = str(path).strip()
        if not os.path.isfile(path):
            raise ApiError("not a file: %s" % path)
        ed = self.editor[0]
        ed.save_file = path
        rec = read_mask_log(path)
        self._editor_settings(rec or {"size": ed.size, "type": "sphere", "radio": {}, "edit": {}, "invert": False})
        self._ed_source = "log" if rec else "file"
        self._ed_log_matches = rec["matches"] if rec else None
        self._ed_last[self.c.name] = path
        self._ed_mem[(self.c.name, self._ed_kind)] = dict(save_file=path, size=ed.size, type=ed.type,
                                                         radio=dict(ed.radio), edit=dict(ed.edit),
                                                         invert=bool(ed.radio.get("invert")))
        return self._editor_state()

    def _need_editor(self):
        if self.editor is None:
            raise ApiError("the mask editor is not open")

    def _v_editor_save(self, form):
        self._need_editor()
        self._editor_apply(form)
        m = None
        try:
            m = self.editor[0].save()
        finally:
            self._editor_flush("k" if m is not None else "r")
        if m is not None:                            # the log of what made this file (overwritten each time)
            ed = self.editor[0]
            self._ed_last[self.c.name] = ed.save_file
            try:
                write_mask_log(ed.save_file, {"size": ed.size, "type": ed.type, "radio": dict(ed.radio),
                                              "edit": dict(ed.edit), "invert": bool(ed.radio.get("invert"))})
                self._echo("[mask editor] settings logged in %s" % mask_log_path(ed.save_file))
                self._ed_log_matches = True
            except OSError as e:
                self._echo("[mask editor] could not write the log %s: %s" % (mask_log_path(ed.save_file), e), "r")
            self._ed_source = "log"
        return self._editor_state()

    def _v_editor_show(self, form, mode="mask", view="z", alpha=0.5, color="r"):
        self._need_editor()
        self._editor_apply(form)
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            alpha = 0.5
        try:
            return self.editor[0].show(mode, view, alpha, (color or "r").strip() or "r")
        finally:
            self._editor_flush()

    def _v_editor_view(self, path):
        self._need_editor()
        try:
            return self.editor[0].view_file(str(path).strip())
        finally:
            self._editor_flush()

    def _ensure_gpu_destination(self):
        """pyDynamo runs projects on the GPUs: Dynamo's destination standalone_gpu (the one DynamoProject runs).
        Set it, as choosing that radio button in Dynamo's environment window would, when the project has another."""
        if self.c.name is not None and (self.c.runtime or {}).get("destination") != "standalone_gpu":
            self.c.set_destination("standalone_gpu")
            self._echo("[pyDynamo] destination set to standalone_gpu: the project runs on GPUs", "b")

    def _v_use_gpus(self):
        self._ensure_gpu_destination()
        return self.c.runtime.get("destination")

    def _v_unfold_and_hint(self):
        """Main window "unfold": unfold, then the command to run. The .exe is written
        for the GPUs (destination standalone_gpu)."""
        self._ensure_gpu_destination()
        r = self.c.unfold()
        if r and r[1] == "ok":
            self._echo("Run it from a terminal in %s:   %s" % (os.getcwd(), self.c.execution_order()), "b")
        return r

    def _v_run_hint(self):
        """Main window "run": the execution order of the project."""
        order = self.c.execution_order()
        if order:
            self._echo("Execution order: %s  (run it from a terminal in %s)" % (order, os.getcwd()), "b")
        return {"order": order, "cwd": os.getcwd()}

    # ------------------------------------------------------------------------------------------------ run (pydynamo)
    # The Run step runs the project where its computing parameters say (pydynamo.project.launch): pydynamo_slurm 0 on this
    # machine's GPUs, 1 as a Slurm job.
    def _need_project(self):
        from pydynamo.vpr import DynamoError
        if self.c.name is None:
            raise DynamoError("No project is open")

    def _v_run_settings(self):
        """Where and how the Run step runs the open project (launch.describe), the state of its last run, and
        whether the project is unfolded with its current parameters."""
        from pydynamo.project import launch
        self._need_project()
        d = launch.describe(self.c.get_vpr(), self.c.name, self.c.data_facts)
        d["status"] = launch.status(self.c.name)
        d["unfold"] = self._unfold_info()
        return d

    def _v_run_set(self, name, value):
        """A computing parameter of the Run step: pyDynamo's (launch.PARAMS) or Dynamo's cluster_walltime /
        submit_order. pyDynamo's are not in the cards, so a project unfolded before stays unfolded; Dynamo's two are
        (card_ite), so changing them needs a new unfold, as in Dynamo."""
        from pydynamo.project import launch
        self._need_project()
        if name in launch.PARAMS:
            info = self._unfold_info()
            c = self.c.pydynamo_edit(name, value)
            if c.ok and info and info["current"]:
                self._unfold_ok_until[self.c.name] = self._vpr_mtime(self.c.name)
            return c
        if name in ("cluster_walltime", "submit_order"):
            return self.c.environment_edit(name, value)
        raise ApiError("%s is not a parameter of the Run step" % name)

    def _v_run_start(self):
        """Run the project (launch.start): on this machine's GPUs or as a Slurm job. Only an unfolded project whose
        parameters did not change since its unfold."""
        from pydynamo.project import launch
        self._need_project()
        info = self._unfold_info()
        try:
            if not info or not info["current"]:
                raise launch.LaunchError("Unfold the project first: the Run step runs the unfolded project, and its "
                                         "parameters changed after the last unfold" if info else
                                         "Unfold the project first: the Run step runs the unfolded project")
            job = launch.start(self.c.get_vpr(), self.c.name, self.c.data_facts)
        except launch.LaunchError as e:
            self._echo(str(e), "r")
            return {"started": False, "error": str(e)}
        if job["mode"] == "local":
            self._echo("[pyDynamo] %s runs on this machine's GPUs (process %s); its output goes to %s"
                       % (self.c.name, job["id"], job["out"]), "b")
        else:
            self._echo("[pyDynamo] %s submitted to Slurm: job %s (%s); its output goes to %s"
                       % (self.c.name, job["id"], job["state"], job["out"]), "b")
        return {"started": True, "job": job}

    def _v_run_status(self):
        from pydynamo.project import launch
        self._need_project()
        return launch.status(self.c.name)

    def _v_run_stop(self):
        from pydynamo.project import launch
        self._need_project()
        try:
            st = launch.stop(self.c.name)
        except launch.LaunchError as e:
            self._echo(str(e), "r")
            return None
        self._echo("[pyDynamo] %s: %s %s stopped" % (self.c.name, "process" if st["mode"] == "local" else "Slurm job",
                                                     st["id"]), "b")
        return st
