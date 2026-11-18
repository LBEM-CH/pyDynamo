"""Controller of the dcp GUI (Dynamo 1.1.514 `dcp` / dynamo_project_manager_easy), without a display.

The view (pydyn.webui: HTML in a Qt WebEngine window) only draws and forwards the user's input here. Everything that reads or writes a
project goes through the project API: pydyn.vpr (load/save/unfold/data_info/...), pydyn.dcptool.dcp
(new/put/set_project), pydyn.vprcheck.check (dvcheck), pydyn.console (dread/dwrite/daverage). There is no second
project format: the project is ./<name>/settings/virtual_project.mat, as in Dynamo.

S = /programs/x86_64-linux/dynamo/1.1.514/matlab/src. Spec: docs/analysis/08a_project_creation_1.1.514.md
sections 2.4, 5.1, 10. Each method cites the Dynamo callback it ports.

The in-memory "runtime" mirrors Dynamo's dProject.runtime (S/@dProject/setFromVpr.m): it is read from the vpr
when a project is opened, every dcp.put updates it (S/@dcp/put.m:51-64), and a rounds-table edit saves the
vpr rebuilt from it (S/@dProject/getVpr.m, setParameter.m:57-64): the 468 convention fields only, so *_all and
feature_* are dropped, exactly like a GUI edit in Dynamo (08a 5.1). Like Dynamo, the runtime is not reloaded
after an unfold.

Deviations from Dynamo (each also printed where it applies):
* getVpr starts from dynamo_vpr('decoy'), which writes ./decoy/settings/virtual_project.mat; pydyn does not
  create ./decoy (no project file changes, 08a 11.3).
* GUI strings are converted with a restricted str2num: numbers, Inf/NaN, true/false, vectors ("4 4 2",
  "[4,4,2]") and integer a:b / a:s:b ranges. Any other MATLAB expression raises NotImplementedError (08a 11.7).
* Multireference projects (nref > 1) are opened (setMultireferenceProject) and checked (seed folders); the
  Multireference menu (even/odd project, golden-standard derive, adaptive-bandpass editor) and the mask editor
  (cylinder masks, pydyn.masks) are ported. Not ported (NotImplementedError): the other mask types, blank and
  random tables, random options of "average all", Fourier compensation of "average all", database queries
  ('name:...') and '*' selections in file boxes, the plugin manager, the show/progress/rewind/reset windows.
"""
import math
import os
import re

import numpy as np

from . import convention as _conv
from . import dcp_parameters as _cls
from . import dynio
from . import vpr as V
from .vpr import DynamoError

NC = {p["name"]: p for p in _conv.NC}
RC = {p["name"]: p for p in _conv.RC}

# S/@dcp/stepOrdering.m
STEP_ORDER = ("data", "table", "template", "masks", "settings", "destination", "unfold", "run")
# S/dynamo__wizard_rounds.m updateTable: presentAreas (the area check boxes that filter the table)
ROUND_AREAS = ("general", "angles", "plugin", "convergence", "threshold", "classification", "limits", "correlation",
               "system")
# initial check-box values of dynamo__wizard_rounds.fig (read with headless Dynamo 1.1.514, run G:
# benchmarks/04_journey_bin8_align_1/dynamo/gui/rounds_window_state.txt); 'advanced' starts unticked, the table shows round
# parameters ("show general parameters" toggle), columns "round 1".."round 8"
ROUND_AREAS_ON = ("general", "angles", "threshold", "limits", "correlation")
# S/dynamo__wizard_environment.m OpeningFcn: radio buttons, and the parameters with an edit box
DESTINATIONS = ("matlab", "matlab_parfor", "standalone", "matlab_gpu", "standalone_gpu", "mpi", "mpi_gpu")
ENVIRONMENT_PARAMETERS = ("how_many_processors", "gpu_identifier_set", "gpu_motor", "cluster_header",
                          "cluster_walltime", "submit_order", "matlab_workers_average")
PYDYN_EXECUTES = ("standalone_gpu", "matlab_gpu")      # pydyn.execute.execute_project
# S/+dpkproject/+aux/+gui/@ABPManager/launchGUI.m: the parameters of the "Adaptive bandpass parameters" editor
ABP_PARAMETERS = ("adaptive_bandpass", "adaptive_bandpass_threshold", "abp_recompute_second_average",
                  "adaptive_bandpass_initial", "adaptive_bandpass_pushback", "adaptive_bandpass_symmetrized",
                  "adaptive_bandpass_even_odd")


class Check:
    """pparse.output of a parameter check: ok, report lines, and the converted value (valueRightClass)."""

    def __init__(self, ok=False, report=None, value=None):
        self.ok = ok
        self.report = list(report or [])
        self.value = value

    def __bool__(self):
        return bool(self.ok)

    def __repr__(self):
        return "Check(ok=%r, report=%r, value=%r)" % (self.ok, self.report, self.value)


# =====================================================================================================================
# MATLAB helpers
# =====================================================================================================================

_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_SPECIAL = {"inf": math.inf, "+inf": math.inf, "-inf": -math.inf, "nan": math.nan, "+nan": math.nan,
            "-nan": math.nan}


def _num_token(t):
    if re.fullmatch(_NUM, t):
        return float(t)
    if t.lower() in _SPECIAL and t.lstrip("+-") in ("Inf", "inf", "NaN", "nan"):
        return _SPECIAL[t.lower()]
    raise NotImplementedError("str2num(%r): MATLAB would evaluate this expression; pydyn evaluates only numbers, "
                              "Inf/NaN, true/false, vectors and integer a:b:c ranges" % t)


def str2num(s):
    """Restricted MATLAB str2num (eval of '[' s ']'). Returns a float64 1-D array (row vector), or a bool for a
    lone true/false. '' and '[]' give an empty array."""
    if not isinstance(s, str):
        raise TypeError("str2num needs a string")
    body = s.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    if ";" in body or "'" in body or '"' in body:
        raise NotImplementedError("str2num(%r): matrices / strings are not ported" % s)
    toks = body.replace(",", " ").split()
    if len(toks) == 1 and toks[0] in ("true", "false"):
        return toks[0] == "true"
    vals = []
    for t in toks:
        if ":" in t:
            parts = t.split(":")
            if len(parts) not in (2, 3) or any(p == "" for p in parts):
                raise NotImplementedError("str2num(%r): range syntax" % s)
            p = [_num_token(x) for x in parts]
            a, st, b = (p[0], 1.0, p[1]) if len(p) == 2 else p
            if not all(float(x).is_integer() for x in (a, st, b)) or st == 0:
                raise NotImplementedError("str2num(%r): only integer a:b / a:s:b ranges are ported" % s)
            n = int(math.floor((b - a) / st)) + 1
            vals += [a + i * st for i in range(max(n, 0))]
        else:
            vals.append(_num_token(t))
    return np.array(vals, dtype=np.float64)


def format_str(value, n_printable=30):
    """dynamo__format_str.m: how dcp shows a value in its edit boxes and tables."""
    if isinstance(value, V.MatCell):
        return "".join(" " + format_str(x) for x in value)
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, str):
        return "[]" if value == "" else value
    a = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    if a.size == 0:
        return "[]"

    def one(x):
        if math.isnan(x):
            return "  NaN"                            # sprintf('%5.2f', NaN)
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"          # x == round(x) is true for Inf: sprintf('%d', Inf)
        if x == round(x):
            return "%d" % int(x)
        return "%5.2f" % x
    out = "".join(" " + one(float(x)) for x in a[:n_printable])
    if a.size >= n_printable:
        out += " .... %d in total" % a.size
    return out


def read_size(f):
    """dynamo_read_size.m for volumes: [lx, ly, lz] from the header (em), or from the volume (other formats)."""
    if os.path.splitext(f)[1] == ".em":
        with open(f, "rb") as fh:
            h = fh.read(16)
        if len(h) < 16 or h[0] not in (0, 3, 5, 6):
            raise DynamoError("[read_size] invalid header: %s" % f)
        end = "<" if h[0] == 6 else ">"
        return [int(x) for x in np.frombuffer(h[4:16], end + "u4")]
    from .console import dread
    v = np.asarray(dread(f))
    if v.ndim != 3:
        raise NotImplementedError("dynamo_read_size of %s (not a volume)" % f)
    return [int(x) for x in v.shape[::-1]]


def ellipsoid(radii, my_size, position=None):
    """dynamo_ellipsoid(radii, my_size[, position]) (S/dynamo_ellipsoid.m) -> float64 array [z, y, x].

    Scalar radii/size are repeated three times; the default position is (my_size(1)/2+1)*[1 1 1] (1-based, uses
    the x size only: dynamo_ellipsoid.m, 1.1.514). The volume is dpkgeom.sphere -> dynamo__sphere_mex ->
    volume_noncubic::fillSphere (mex/src/classes/volume_noncubic.cpp): voxel (x,y,z) (0-based) is 1 iff
    rnx*rnx + rny*rny + rnz*rnz <= 1 with rnx = (x - position[0] + 1) / semiaxes[0], in double precision and in
    this order. smoothing is 0 (the dcp buttons never pass one)."""
    r = np.atleast_1d(np.asarray(radii, dtype=np.float64)).ravel()
    s = np.atleast_1d(np.asarray(my_size, dtype=np.float64)).ravel()
    if r.size == 1:
        r = np.repeat(r, 3)
    if s.size == 1:
        s = np.repeat(s, 3)
    if r.size != 3 or s.size != 3:
        raise NotImplementedError("dynamo_ellipsoid with %d semiaxes / %d sizes (dpkgeom.sphere needs three)"
                                  % (r.size, s.size))
    if not all(float(x).is_integer() for x in s):
        raise DynamoError("Size needs to be passed as integer.")           # dpkgeom/sphere.m
    nx, ny, nz = (int(x) for x in s)
    pos = np.repeat((s[0] / 2 + 1), 3) if position is None else \
        np.resize(np.asarray(position, dtype=np.float64).ravel(), 3)
    if np.prod(r) == 0:
        return np.zeros((nz, ny, nx))
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = ((np.arange(nx, dtype=np.float64) - pos[0]) + 1) / r[0]
        ty = ((np.arange(ny, dtype=np.float64) - pos[1]) + 1) / r[1]
        tz = ((np.arange(nz, dtype=np.float64) - pos[2]) + 1) / r[2]
        c = (tx * tx)[None, None, :] + (ty * ty)[None, :, None]
        c = c + (tz * tz)[:, None, None]
    return (c <= 1).astype(np.float64)


# =====================================================================================================================
# parameter checks (S/+dpkproject/+parameters)
# =====================================================================================================================

def parameter(name):
    """The dcp GUI object of a global or round parameter (generic round name, e.g. 'cone_range'):
    S/+dpkproject/+parameters/+list/@<name>/<name>.m (pydyn.dcp_parameters). Its type equals the convention's;
    pretty, comment, functionality and advanced differ for some parameters, and the GUI uses the class values."""
    if name not in NC and name not in RC:
        raise DynamoError('"%s" is not a parameter of the convention' % name)
    return _cls.PARAMETERS[name]


def class_value(p):
    """obj.value of a parameter class (+dpkproject/+parameters/+list/@<name>/<name>.m), from its MATLAB literal
    (pydyn.dcp_parameters keeps obj.default, which has the same text as obj.value in all 1.1.514 classes): a quoted
    literal gives the char value, a numeric literal a double (scalar float, or a 1-D array otherwise)."""
    text = p["default"].strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    v = str2num(text)
    if isinstance(v, bool):
        return v
    return float(v[0]) if v.size == 1 else v


def _type_ok(ptype, value):
    """@Parameter/checkValue.m: the switch on obj.type (types 0, 1, 2, 6; type 3 is not checked)."""
    if ptype == 0:
        return isinstance(value, str)
    if ptype == 1:
        return V._isnumeric(value) or (isinstance(value, np.ndarray) and value.size == 0)
    if ptype == 2:
        return V._isnumeric(value) or V._islogical(value) or (isinstance(value, np.ndarray) and value.size == 0)
    if ptype == 6:
        return V._isnumeric(value) or isinstance(value, str)
    return True


class DcpController:
    """State of one dcp session (the global DYNCP of Dynamo) and the actions of its windows.

    Works relative to the current working directory, as Dynamo does. `echo(text, color)` receives every message
    (Dynamo writes them into the listbox of the main window); messages are also kept in self.messages."""

    def __init__(self, echo=None):
        self._echo_sink = echo
        self.messages = []
        self._reset()

    def _reset(self):
        self.name = None
        self.runtime = None
        self.correct = {k: None for k in ("data", "table", "template", "mask", "cmask", "fmask", "smask")}
        self.data_facts = None
        self.template_size_cache = None
        self.enabled = {s: False for s in STEP_ORDER}
        self.semiaxis = ""                           # edit_semiaxis of the masks window
        self.mask_box = None                         # (box, file) set by the mask editor (blue text)

    # ------------------------------------------------------------------------------------------------ messages
    def echo(self, text, color="k"):
        for line in ([text] if isinstance(text, str) else list(text)):
            self.messages.append((line, color))
            if self._echo_sink:
                self._echo_sink(line, color)

    # ------------------------------------------------------------------------------------------------ steps
    def _enable(self, step, on):
        self.enabled[step] = bool(on)

    def _cascade(self, step, on):
        """S/@dcp/enableCascade.m."""
        for s in STEP_ORDER[STEP_ORDER.index(step):]:
            self._enable(s, on)

    def _step_status(self, step, status):
        """S/@dcp/stepStatus2GUI.m."""
        i = STEP_ORDER.index(step)
        nxt = STEP_ORDER[i + 1] if i + 1 < len(STEP_ORDER) else None
        if status == "wrong":
            if nxt:
                self._cascade(nxt, False)
            return
        if step == "data":
            self._enable("data", True)
        if nxt:
            self._enable(nxt, True)

    def _steps_status_as_read(self):
        """S/@dcp/stepsStatusAsRead2GUI.m."""
        for s in ("data", "table", "template"):
            self._step_status(s, "right" if self.correct[s] else "wrong")
        all_ok = True
        if not self.correct["mask"]:
            self.echo("The mask is not correct.", "r")
            all_ok = False
        if not self.correct["cmask"]:
            self.echo("The classification mask is not correct.", "r")
            all_ok = False
        if not self.correct["fmask"]:
            self.echo("The fourier mask is not correct.", "r")
            all_ok = False
        self._step_status("masks", "right" if all_ok else "wrong")
        self._enable("destination", self.enabled["settings"])

    # ------------------------------------------------------------------------------------------------ project
    def require_project(self):
        if self.name is None:
            raise DynamoError("No current project.")
        return self.name

    @staticmethod
    def list_projects():
        """Projects in the current folder (folders with settings/virtual_project.mat, dynamo_isproject.m)."""
        return sorted(d for d in os.listdir(".") if os.path.isdir(d) and V.isproject(d))

    def enter_project_name(self, name):
        """Main window edit_project_Callback / getProject (S/dynamo_project_manager_easy.m:209-243,486-560).
        Returns 'opened', 'not_a_project' (the GUI then offers dcp.create) or 'invalid'."""
        name = (name or "").strip()
        if "*" in name:
            raise NotImplementedError("project search with '*' (dynamo_projects 'regexp')")
        if " " in name or name == "":
            self.echo("Blanks not allowed inside the name of a project.", "warning")
            return "invalid"
        if not V.isproject(name):
            self.echo('The string "%s" is not a project currently in file system' % name, "warning")
            return "not_a_project"
        self.open(name)
        self.echo('Current project: "%s" ' % name)
        self._enable("data", True)
        return "opened"

    def create(self, name):
        """dcp.create(name) (S/@dcp/create.m): dynamo_vpr(name) (which saves), dynamo_vpr_save, dcp.open."""
        if " " in name or not name:
            raise DynamoError("Blanks not allowed inside the name of a project.")
        vpr = V.blank(name)
        vpr = V.save(vpr)                            # dynamo_vpr(name): 'save','d' default (dynamo_vpr.m:1179-1188)
        V.save(vpr)                                  # create.m: dynamo_vpr_save(vpr)
        self.echo('A blank project with name "%s" has been saved' % name)
        self.open(name)
        self.echo('A new project has been created: "%s"' % name)
        self._enable("data", True)
        self.echo("Click on [particles] to start populating it by providing a data folder.")
        return name

    def new(self, name, *args, **kw):
        """dcp.new(name, ...) with gui=0, then open it here (the console call of the user's journey)."""
        from .dcptool import dcp
        kw = dict(kw)
        kw["gui"] = 0
        v = dcp.new(name, *args, **kw)
        if v is None:
            return None
        self.open(name)
        return name

    def open(self, name):
        """dcp.open(name) (S/@dcp/open.m) and the opening function of the main window
        (S/dynamo_project_manager_easy.m:60-112): runtime from the vpr, checkSteps, stepsStatusAsRead2GUI,
        setSingleReferenceProject."""
        from .dcptool import dcp
        vpr = V.load(name)
        if vpr is None:
            raise DynamoError('Cannot create a "dProject" object out of string "%s" ' % name)
        if not vpr.get("name_project") or vpr["name_project"] == "[]":
            raise DynamoError('Cannot create a "dProject" object out of string "%s". Empty project name. ' % name)
        if vpr["name_project"] != name:
            # Dynamo would go on with the internal name (d.runtime.name_project), i.e. another folder
            raise NotImplementedError('project folder "%s" holds name_project "%s": Dynamo then works on '
                                      './%s, which is not handled here' % (name, vpr["name_project"],
                                                                         vpr["name_project"]))
        self._reset()
        self.runtime = self._runtime_from_vpr(vpr)
        self.name = name
        dcp.set_project(name)
        self.echo('Project currently in scope: "%s"' % name)
        self.check_steps()
        self._steps_status_as_read()
        nref = int(float(self.runtime["nref_r1"]))
        if nref == 1:                                # dynamo_project_manager_easy.m:108-112
            self._set_single_reference_project()
        else:
            self._set_multireference_project(nref)
        return name

    @staticmethod
    def _runtime_from_vpr(vpr):
        """S/@dProject/setFromVpr.m: every convention parameter (globals, then <p>_r1.._r8) from the vpr; a missing
        field keeps the value of the parameter class (obj.value, equal to obj.default in every 1.1.514 class), with
        Dynamo's warning text (setFromVpr.m: global and round messages differ)."""
        rt = {}
        for k in V.blank(vpr["name_project"]):
            if k in vpr:
                rt[k] = vpr[k]
            else:
                if k in NC:
                    base = k
                    print('[warning] : the vpr being read misses parameter of type "%s"' % base)
                else:
                    base, r = k.rsplit("_r", 1)
                    print('[warning] : vpr misses a parameter of type"%s" in round %d' % (base, int(r)))
                rt[k] = class_value(parameter(base))
        return rt

    def _set_single_reference_project(self):
        """S/@dcp/setSingleReferenceProject.m:16-28: nref = 1 in the runtime and in the vpr (load, set, save)."""
        for r in range(1, 9):
            self.runtime["nref_r%d" % r] = 1.0
        vpr = V.load(self.name, verbose=False)
        for r in range(1, 9):
            vpr["nref_r%d" % r] = 1.0
        V.save(vpr)

    def _set_multireference_project(self, nref):
        """S/@dcp/setMultireferenceProject.m: nref in the runtime and in the vpr (load, set nref_r1..8, save; the
        other fields are kept), then the seed checks of template, fmask and table (messages only). The table
        message prints the fmask value, as in Dynamo (setMultireferenceProject.m, checkTable branch)."""
        for r in range(1, 9):
            self.runtime["nref_r%d" % r] = float(nref)
        vpr = V.load(self.name, verbose=False)
        for r in range(1, 9):
            vpr["nref_r%d" % r] = float(nref)
        V.save(vpr)
        for item, field, shown, label in (("template", "file_template_initial", "file_template_initial", "template"),
                                          ("fmask", "file_fmask_initial", "file_fmask_initial", "masks"),
                                          ("table", "file_table_initial", "file_fmask_initial", "table")):
            seeds = self.available_seeds(item)
            if len([r for r in range(1, nref + 1) if r in seeds["availableRefs"]]) < nref:
                what = {"template": "Current template", "fmask": "Current Fourier mask",
                        "table": "Current table"}[item]
                sep = " " if item == "template" else "  "
                self.echo("%s cannot be used as seed for multireference%sproject: %s"
                          % (what, sep, self.runtime[shown]), "warning")
                self.echo("You need to provide a folder that contains all references from 1 to %d" % nref)

    def scan_seeds_in_folder(self, value, item):
        """S/+dpkproject/+parameters/scanSeedsInFolder.m: a file is a single seed (the '.sel' test compares the
        extension with 'sel' without the dot and never fires); a folder is scanned for
        <item>_initial_ref_<ddd>*.* (MATLAB dir order: sorted names)."""
        short = {"template": "template_initial", "table": "table_initial", "fmask": "fmask_initial",
                 "file_template_initial": "template_initial", "file_table_initial": "table_initial",
                 "file_fmask_initial": "fmask_initial"}[item]
        o = {"nrefs": 0, "formattedAsSingle": False, "files": [], "availableRefs": [], "ok": True}
        if value is None or (isinstance(value, str) and value == ""):
            return o
        if not (os.path.isfile(value) or os.path.isdir(value)):
            self.echo('The string "%s" is neither a file nor a folder.' % value, "r")
            o["ok"] = False
            return o
        if os.path.isfile(value):
            o.update(nrefs=1, formattedAsSingle=True, files=[value])
            return o
        pat = re.compile(r"%s_ref_(?P<ref>\d\d\d).*" % short)
        for f in sorted(os.listdir(value)):
            if not (f.startswith(short + "_ref_") and "." in f[len(short) + 5:]):
                continue
            m = pat.match(f)
            if m is None:
                print("Incorrect file name: %s" % f)
            else:
                o["files"].append("%s/%s" % (value, f))
                o["availableRefs"].append(int(m.group("ref")))
        o["nrefs"] = len(o["files"])
        return o

    def available_seeds(self, item, get_sizes=False):
        """S/@dcp/availableSeeds.m: the seed files of template / table / fmask (single: the value itself)."""
        field = {"template": "file_template_initial", "table": "file_table_initial",
                 "fmask": "file_fmask_initial"}[item]
        if int(float(self.runtime["nref_r1"])) == 1:
            o = {"files": [self.runtime[field]], "availableRefs": [], "ok": True}
        else:
            o = self.scan_seeds_in_folder(self.runtime[field], item)
            o["ok"] = bool(o["files"])
        if get_sizes:
            sizes = [read_size(f) for f in o["files"]]
            if any(sizes[i] != sizes[i - 1] for i in range(1, len(sizes))):
                self.echo("Not all templates have the same dimensionality.")
                o.update(ok=False, size=None)
            else:
                o.update(ok=True, size=sizes[0] if sizes else None)
        return o

    # ------------------------------------------------------------------------------------------------ dcp.put
    def put(self, parameter_name, value, round=None):
        """dcp.put (S/@dcp/put.m): load the vpr, set the field, save into ./; then the runtime value."""
        from .dcptool import dcp
        self.require_project()
        dcp.set_project(self.name)
        dcp.put(parameter_name, value, round)
        key = parameter_name if round is None else "%s_r%d" % (parameter_name, int(round))
        self.runtime[key] = V.as_matlab_value(value)
        return Check(True)

    def _gui_save(self):
        """S/@dProject/setParameter.m:57-64 with 'inDisk': vpr = getVpr() (getVpr.m: the convention fields in
        dynamo_vpr order, values of the runtime), dynamo_vpr_save(vpr). ./decoy is not created (module doc)."""
        vpr = {k: self.runtime[k] for k in V.blank(self.name)}
        V.save(vpr)

    def get_vpr(self):
        """@dProject/getVpr (in memory)."""
        return {k: self.runtime[k] for k in V.blank(self.name)}

    # ------------------------------------------------------------------------------------------------ checks
    def template_size(self, force=True):
        """dcp.getTemplateSize (S/@dcp/getTemplateSize.m): the stored size, else checkSteps('template')."""
        if self.name is None:
            return None
        if self.template_size_cache is None and force:
            self.check_steps(["template"])
        ts = self.template_size_cache
        if ts is None or not np.any(ts):
            return None
        return list(ts)

    def check_value(self, name, value):
        """Parameter.checkValue with the parameter-specific overrides (S/+dpkproject/+parameters/+list/@*/
        checkValue.m) for a parameter bound to the current project (dProject set, as in every GUI path).

        The 'options' test of @Parameter/checkValue.m sets o.ok = false and is then overwritten by o.ok = true:
        it never rejects a value (Dynamo 1.1.514 behaviour, kept)."""
        p = parameter(name)
        if not _type_ok(p["type"], value):
            kind = {0: "a char value", 1: "numeric", 2: "numeric/logical", 6: "numeric or char"}[p["type"]]
            return Check(False, ['Parameter "%s" should be %s (passed value: "%s")' % (name, kind,
                                                                                    format_str(value))])
        if name == "folder_data":
            facts = V.data_info(value)
            c = Check(bool(facts["correct"]))
            c.facts = facts
            return c
        if name in ("file_table_initial", "file_template_initial", "file_fmask_initial"):
            return self._multireference_entry(p, value)
        if name == "file_mask":
            if not os.path.isfile(value):            # isValueAFile.m
                return Check(False, ['check[%s] File does not exist: "%s"' % (p["pretty"], value)])
            return self._mask_size(value, "mask", require_file=True)
        if name in ("file_mask_classification", "file_mask_smoothing"):
            return self._mask_size(value, "classification mask" if name == "file_mask_classification"
                                   else "smoothing mask", require_file=False)
        if name == "gpu_motor":
            if value not in ("spp", "cuda"):
                return Check(False, ['check[gpu_motor] The input value "%s" is incorrect. Use only "spp" or "cuda" '
                                     "(advanced)" % format_str(value)])
            return Check(True)
        if name == "initial_references":
            if isinstance(value, str) and value != "*":
                return Check(False, ['The string code "%s" is not valid for parameter "%s"' % (value, name)])
            return Check(True)
        if name == "cluster_header":
            if not os.path.isfile(value):
                return Check(False, ['check[%s] File does not exist: "%s"' % (p["pretty"], value)])
            return Check(True)
        return Check(True)

    def _multireference_entry(self, p, value):
        """@Parameter/isValueMultireferenceEntry.m."""
        nref = int(float(self.runtime["nref_r1"])) if self.runtime else 1
        if nref != 1:
            if not os.path.isdir(value):
                return Check(False, ['Checking a value for "%s" (multireference): "%s" not a folder'
                                     % (p["pretty"], value)])
            seeds = self.scan_seeds_in_folder(value, p["name"])
            short = {"file_template_initial": "template_initial", "file_table_initial": "table_initial",
                     "file_fmask_initial": "fmask_initial"}[p["name"]]
            if not seeds["nrefs"] > 0:
                return Check(False, ['No files inside folder "%s" with the structure %s_ref_xxx.ext' % (value, short)])
            if seeds["nrefs"] >= nref:
                return Check(True)
            return Check(False, ['Not enough files inside folder "%s" with the structure file_%s_initial_ref_xxx.ext. '
                                 'You have %d but need %d.' % (value, short, seeds["nrefs"], nref)])
        if not os.path.isfile(value):
            return Check(False, ['Checking a value for "%s" (single reference): File does not exist: "%s"'
                                 % (p["pretty"], value)])
        return Check(True)

    def _mask_size(self, value, label, require_file):
        """file_mask / file_mask_classification / file_mask_smoothing checkValue.m: size against the template. The
        texts are Dynamo's, including its typos in @file_mask/checkValue.m:20 ('as  mask') and :39 ('correspondo')."""
        ts = self.template_size()
        if ts is None:
            return Check(True)
        is_mask = label == "mask"
        if not os.path.isfile(value):
            return Check(False, ["File provided as %s does not exist: %s" % (" mask" if is_mask else label, value)])
        sm = read_size(value)
        if len(sm) != len(ts):
            return Check(False, ["Number of elements in %s (%d) does not correspond with number of elements in "
                                 "template (%d)" % (label, len(sm), len(ts))])
        if sm != ts:
            return Check(False, ["Size of %s %s does not %s to size of template %s"
                                 % (label, format_str(sm), "correspondo" if is_mask else "correspond",
                                    format_str(ts))])
        return Check(True)

    def value_from_string(self, name, s):
        """@Parameter/checkValuePassedAsString.m: type 0 keeps the string, type 1 str2num, type 2
        double(str2num); then checkValue. Types 3 and 6 have no conversion in 1.1.514 (MATLAB error)."""
        p = parameter(name)
        t = p["type"]
        if t == 0:
            v = s
        elif t in (1, 2):
            v = str2num(s)
            if t == 2 and isinstance(v, bool):
                v = np.array([1.0 if v else 0.0])
            if isinstance(v, np.ndarray) and v.size == 0 and s != "":
                return Check(False, ['Parameter "%s": input error. String "%s" cannot be converted into  type %d'
                                     % (name, s, t)])
            if isinstance(v, np.ndarray) and v.size:
                v = V.as_matlab_value(v)
        else:
            return Check(False, ["Undefined function or variable 'valueRightClass'. (checkValuePassedAsString.m "
                                 "has no case for parameter type %d: Dynamo 1.1.514 cannot set \"%s\" from the GUI)"
                                 % (t, name)])
        c = self.check_value(name, v)
        c.value = v
        return c

    def check_steps(self, items=None):
        """dcp.checkSteps (S/@dcp/checkSteps.m): the isCorrect* flags of data, table, template, masks."""
        items = items or ["data", "table", "template", "mask", "fmask", "cmask"]
        rt = self.runtime
        ok_all = True
        wrong = []
        if "data" in items:
            c = self.check_value("folder_data", rt["folder_data"])
            if not c.ok:
                self.echo("Data not found.", "r")
                self.echo(c.report, "r")
                self.correct["data"] = False
                ok_all = False
                wrong.append("data")
            else:
                self.correct["data"] = True
                self.data_facts = c.facts
        if "table" in items:
            c = self.check_value("file_table_initial", rt["file_table_initial"])
            if not c.ok:
                self.echo("table not found.", "r")
                self.echo(c.report, "r")
                self.correct["table"] = False
                ok_all = False
                wrong.append("table")
            else:
                self.correct["table"] = True
        if "template" in items:
            c = self.check_value("file_template_initial", rt["file_template_initial"])
            if not c.ok:
                self.echo("Problem with template found.", "r")
                self.echo(c.report, "r")
                self.correct["template"] = False
                ok_all = False
                wrong.append("template")
            else:
                seeds = self.available_seeds("template", get_sizes=True)          # checkSteps.m:56-66
                if not seeds.get("size"):
                    self.echo("Problem with template found: cannot determine size!", "r")
                    self.correct["template"] = False
                    ok_all = False
                    wrong.append("template")
                else:
                    self.correct["template"] = True
                    self.template_size_cache = seeds["size"]
        for item, field, label in (("mask", "file_mask", "Problem with mask  found."),
                                   ("cmask", "file_mask_classification", "Problem with classification mask found."),
                                   ("fmask", "file_fmask_initial", "Problem with fourier maks.")):
            if item not in items:
                continue
            c = self.check_value(field, rt[field])
            if not c.ok:
                self.echo(label, "r")
                self.echo(c.report, "r")
                self.correct[item] = False
                ok_all = False
                wrong.append("masks")
            else:
                self.correct[item] = True
        return ok_all, wrong

    # ------------------------------------------------------------------------------------------------ data window
    @staticmethod
    def _plain_input(s):
        """dynamo__query2folder_iffolder / pgui.editFieldFile: a string with ':' is a database query, '*' a
        selection list; both are not ported."""
        if ":" in s:
            raise NotImplementedError("database queries in file boxes (dynamo_db): %r" % s)
        if "*" in s:
            raise NotImplementedError("'*' file selection (pgui.editFieldFile): %r" % s)
        return s

    def data_enter(self, folder):
        """dcp.dataEnter (S/@dcp/dataEnter.m)."""
        self.require_project()
        if not folder:
            self.echo("Please enter a valid data.")
            return Check(False)
        folder = self._plain_input(folder)
        facts = V.data_info(folder)
        if not facts["correct"]:
            if not os.path.isdir(folder):
                self.echo('Your input data "%s" is not a folder.' % folder, "r")
            self.echo('Cannot use your input data "%s"' % folder, "r")
            self.correct["data"] = False
            self._step_status("data", "wrong")
            return Check(False, ['Cannot use your input data "%s"' % folder])
        self.echo("Data container %s with %d particles of size %d" % (folder, facts["N"], facts["l"]), "waiting")
        self.put("folder_data", folder)
        self.echo("Accepted.")
        self._step_status("data", "right")            # dcp.dataIsRight (dcp.m:396-402)
        self.correct["data"] = True
        self.data_facts = facts
        if float(self.runtime["dim_r1"]) == 0:
            self.put("dim", float(facts["l"]), 1)
            self.put("low", float(math.floor(facts["l"] / 3 + 0.5)), 1)     # MATLAB round (l/3 is never x.5)
        if float(self.runtime["ite_r1"]) == 0:
            self.put("ite", 1.0, 1)
        return Check(True)

    # ------------------------------------------------------------------------------------------------ table / template
    def table_enter(self, file):
        """dcp.tableEnter (S/@dcp/tableEnter.m)."""
        self.require_project()
        file = self._plain_input(file)
        c = self.check_value("file_table_initial", file)
        if not c.ok:
            self.echo('Cannot use your input table "%s"' % file, "r")
            self.echo(c.report, "r")
            self.correct["table"] = False
            self._step_status("table", "wrong")
            return c
        self.echo("Table entered", "waiting")
        self.put("file_table_initial", file)
        self.echo('Table accepted: "%s"' % file)
        self.correct["table"] = True
        self._step_status("table", "right")
        return c

    def _enter_file_parameter(self, value, name):
        """dcp.enterFileParameter (S/@dcp/enterFileParameter.m)."""
        self.require_project()
        value = self._plain_input(value)
        c = self.check_value(name, value)
        pretty = parameter(name)["pretty"]
        if not c.ok:
            self.echo('check [%s] Cannot use your input "%s"' % (pretty, value), "r")
            self.echo(c.report, "r")
            return c
        self.echo('.... testing input for parameter "%s" ' % pretty, "waiting")
        self.put(name, value)
        self.echo('Parameter "%s" accepted into project "%s" with value: "%s"' % (pretty, self.name,
                                                                                format_str(value)))
        return c

    def template_enter(self, file):
        """dcp.templateEnter (S/@dcp/templateEnter.m)."""
        c = self._enter_file_parameter(file, "file_template_initial")
        if c.ok:
            self.correct["template"] = True
            self._enable("masks", True)
            self.template_size_cache = read_size(file)
        else:
            self.correct["template"] = False
            self._cascade("masks", False)
        return c

    def template_average(self, random_sample=False, random_table=False, fcompensate=False, device="auto"):
        """"average all" of the template window (S/@dcp/callbackTemplateAverage.m, single reference): dynamo_average
        of the project data with the project table and all its tags, written to <name>/settings/computedAverage.em,
        then templateEnter. fcompensate 0 and [] take the same path in dynamo_average.m (:732, :2150), so this is
        pydyn.daverage's default path."""
        self.require_project()
        if int(float(self.runtime["nref_r1"])) > 1:
            raise NotImplementedError("average all for multireference projects")
        if random_sample or random_table:
            raise NotImplementedError("average all with random samples / random table (randperm, "
                                      "dynamo_table_random)")
        if fcompensate:
            raise NotImplementedError("average all with Fourier compensation (dynamo_average 'fcompensate' 1)")
        facts = self.data_facts
        if not facts or not facts["correct"]:
            self.echo("Data is not valid.", "warning")
            return Check(False)
        from .console import daverage, dread, dwrite
        self.echo("Creating initial template through averaging of particles.", "waiting")
        T = np.asarray(dread(self.runtime["file_table_initial"]))
        tags = np.unique(np.real(T[:, 0]))
        oa = daverage(self.runtime["folder_data"], "table", T, "tags", tags, device=device)
        file = "%s/settings/computedAverage.em" % self.name
        dwrite(oa.average, file)
        self.echo("Averaging completed. If in blue, click on the name of the produced average to accept it.")
        return self.template_enter(file)

    def table_blank(self):
        raise NotImplementedError("blank table (dcp.callbackTableBlank -> dynamo_table_blank) is not ported")

    def table_random(self):
        raise NotImplementedError("random table (dcp.callbackTableRandom -> dynamo_table_random) is not ported")

    # ------------------------------------------------------------------------------------------------ masks window
    def _mask_enter(self, value, field, flag):
        c = self._enter_file_parameter(value, field)
        self.correct[flag] = bool(c.ok)
        return c

    def mask_enter(self, value):
        """dcp.maskEnter (S/@dcp/maskEnter.m)."""
        return self._mask_enter(value, "file_mask", "mask")

    def cmask_enter(self, value):
        """dcp.cmaskEnter."""
        return self._mask_enter(value, "file_mask_classification", "cmask")

    def fmask_enter(self, value):
        """dcp.fmaskEnter."""
        return self._mask_enter(value, "file_fmask_initial", "fmask")

    def smask_enter(self, value):
        """dcp.smaskEnter."""
        return self._mask_enter(value, "file_mask_smoothing", "smask")

    def masks_window_opened(self):
        """dynamo__wizard_masks_OpeningFcn: the semiaxis box shows dynamo__format_str(sizeTemplate/2-[1,1,1])
        (sizeTemplate = dcp.fetch('templateSize'), the stored size)."""
        ts = self.template_size_cache
        if ts is None:
            self.echo("Template is still invalid!", "r")
            return None
        self.semiaxis = format_str(np.asarray(ts, dtype=float) / 2 - 1)
        return self.semiaxis

    def masks_ellipsoid(self, semiaxis=None):
        """"ellipsoid" button (S/@dcp/callbackMasksEllipsoid.m) -> <name>/settings/automatic_mask.em, maskEnter."""
        self.require_project()
        if semiaxis is not None:
            self.semiaxis = semiaxis
        length_mask = self.template_size()
        if length_mask is None:
            return Check(False, ["unknown template size"])
        my_axis = str2num(self.semiaxis) if self.semiaxis.strip() else np.zeros(0)
        if isinstance(my_axis, bool) or np.size(my_axis) == 0:
            return Check(False, ["empty semiaxis"])
        vol = ellipsoid(my_axis, length_mask)
        folder = "%s/settings" % self.name
        os.makedirs(folder, exist_ok=True)
        f = "%s/automatic_mask.em" % folder
        if not os.path.isfile(f):                    # sic: the test is inverted in callbackMasksEllipsoid.m
            self.echo("Overriding previously generated automatic mask for alignment.")
        dynio.write_em(f, vol)
        return self.mask_enter(f)

    def masks_copy_mask(self):
        """"copy mask" button (S/@dcp/callbackMasksCopyMask.m) -> <name>/settings/automatic_cmask.em, cmaskEnter."""
        self.require_project()
        fm = self.runtime["file_mask"]
        if not os.path.isfile(fm):
            self.echo("Current alignment mask is not a file.", "r")
            return Check(False, ["Current alignment mask is not a file."])
        if not self.check_value("file_mask", fm).ok:
            self.echo("Current alignment mask is not valid.", "r")
            return Check(False, ["Current alignment mask is not valid."])
        folder = "%s/settings" % self.name
        os.makedirs(folder, exist_ok=True)
        f = "%s/automatic_cmask.em" % folder
        V._copyfile(fm, f)                           # dynamo_copyfile (default options)
        return self.cmask_enter(f)

    def masks_use_ones(self):
        """fmask "use ones" (S/@dcp/callbackMasksUseOnes.m -> S/dynamo_fourier_count_full.m), every reference:
        for ref = 1:nref, the linked initial template (dynamo_database_linked_file_read: nref 1 the vpr value, nref > 1
        dynamo_check_initial_folders of it) gives the size, and ones(size(template)) is written to the database
        location of fmask_initial ref. An empty link stops the loop ("You probably need to save the project
        first!"); a missing template is an error. Then the fmask box gets the file (nref 1) or its folder
        (fileparts, nref > 1) and fmaskEnter runs."""
        from .vpreo import check_initial_folders
        self.require_project()
        vpr = self.get_vpr()
        nref = int(float(vpr["nref_r1"]))
        files, errors = [], []
        for ref in range(1, nref + 1):
            if nref == 1:
                loc = vpr["file_template_initial"]
            else:
                linked, rep = check_initial_folders(vpr["file_template_initial"], "file_template_initial", nref)
                loc = linked[ref - 1] if rep == "ok" else ""
            if not loc:
                self.echo("[fourier count full] You probably need to save the project first! ", "k")
                break
            if not os.path.isfile(loc):
                errors.append("initial template not found")
                continue
            size_xyz = read_size(loc)
            f = V.loc("fmask_initial", self.name, ref=ref)
            dynio.write_em(f, np.ones(size_xyz[::-1]))
            files.append(f)
        if not files:
            self.echo('Cannot create full fmask on project "%s"' % self.name, "r")
            return Check(False, errors or ["no initial template"])
        if errors:
            self.echo(errors, "r")
            return Check(False, errors)
        return self.fmask_enter(files[0] if nref == 1 else os.path.dirname(files[0]))

    def smask_use_ones(self):
        """smask "use ones" (S/@dcp/callbackSmoothingMasksUseOnes.m) -> <name>/settings/smoothingMaskOnes.em."""
        self.require_project()
        f = "%s/settings/smoothingMaskOnes.em" % self.name
        size = self.template_size()
        if size is None:
            raise NotImplementedError("smask use ones without a template size (ones([]))")
        dynio.write_em(f, np.ones(size[::-1]))
        return self.smask_enter(f)

    def masks_use_defaults(self):
        """"use defaults" (S/@dcp/callbackMasksDefaults.m): semiaxes floor(l/2-1), then ellipsoid, copy mask,
        fmask use ones, smask use ones."""
        self.require_project()
        lm = self.template_size()
        if lm is None:
            return Check(False, ["unknown template size"])
        r = [int(math.floor(x / 2 - 1)) for x in lm]
        self.semiaxis = "%d %d %d" % tuple(r)
        out = [self.masks_ellipsoid(), self.masks_copy_mask(), self.masks_use_ones(), self.smask_use_ones()]
        return Check(all(c.ok for c in out), sum((c.report for c in out), []))

    def mask_editor(self, kind="align"):
        """"mask editor" buttons of the masks window (S/dynamo__wizard_masks.m:340-350,679 -> editorMask :352-414):
        file <name>/settings/editorMask.em (align), editorCMask.em (cmask) or editorSmoothingMask.em (smask), the
        template size (dcp.getTemplateSize) and the first template seed, then dynamo_mask_GUI('volume', template,
        'size', size, 'mask', file) = a pydyn.masks.MaskEditor. The masks-window box is set to the file (in blue; for
        'smask' Dynamo sets the classification box, :373); the mask enters the project only when Enter is pressed
        in that box (mask_enter / cmask_enter / smask_enter). Returns (editor, box, file) or None."""
        from .masks import EDITOR_FILES, MaskEditor
        if self.name is None:
            self.echo("No current project available, cannot open mask editor.", "warning")
            return None
        fname, box = EDITOR_FILES[kind]
        file_mask = "%s/settings/%s" % (self.name, fname)
        my_size = self.template_size()
        if my_size is None:
            self.echo("Unknown size of current template.", "warning")
            return None
        seeds = self.available_seeds("template")
        if not seeds["ok"] or not seeds["files"] or not os.path.isfile(seeds["files"][0]):
            self.echo("Cannot identify a valid template in the project.", "warning")
            return None
        editor = MaskEditor(size=my_size, mask=file_mask, volume=seeds["files"][0])
        self.mask_box = (box, file_mask)
        self.echo("Press on enter onto the mask file %s" % file_mask)
        self.echo("To confirm that you want to use it in the project.")
        return editor, box, file_mask

    # ------------------------------------------------------------------------------------------------ multireference
    def _open_tool(self, tool):
        """openTool (S/dynamo_project_manager_easy.m:842-929) for 'deriveEO' and 'goldenStandard'."""
        from .vpreo import dynamo_vpr_eo
        if self.name is None:
            self.echo("No current project", "warning")
            return None
        name = self.runtime["name_project"]
        if not V.isproject(name):
            self.echo("Invalid project %s" % name, "warning")
            return None
        self.echo('Opening tool "%s".' % tool, "waiting")
        self.echo('launchig commmand: "dynamo_vpr_eo" on project "%s"' % name, "waiting")
        eo = dynamo_vpr_eo(name) if tool == "deriveEO" else dynamo_vpr_eo(name, "dynamic_bandpass", 1)
        if eo is None:
            self.echo('Sorry. Failed to create e/o project for "%s" ' % name, "warning")
            return None
        name_eo = eo["name_project"]
        self.echo(("Created even/odd project: %s" if tool == "deriveEO" else
                   "Created even/odd project for golden standard/dynamic bandpass: %s") % name_eo)
        self.echo("You need to run it and possibly realign afterwards the two obtained averages.")
        self.echo("%s" % name_eo)
        return name_eo

    def create_even_odd_project(self):
        """Multireference > Create an even/odd project (uimenu_multireferece_even_odd -> openTool('deriveEO')):
        dynamo_vpr_eo(name)."""
        return self._open_tool("deriveEO")

    def golden_standard_derive(self):
        """Multireference > Adaptive filtering ("golden standard") > Derive a project.
        (uimenu_golden_standard_create_Callback, :1171-1172 -> openTool('goldenStandard')):
        dynamo_vpr_eo(name, 'dynamic_bandpass', 1). The new project <name>_eo is not opened: type its name into
        the project box (enter_project_name) to work on it."""
        return self._open_tool("goldenStandard")

    def abp_parameters(self):
        """Multireference > Adaptive filtering > Edit parameters for Adaptive Filtering run
        (uimenu_golden_standard_parameters_Callback :1355-1366 -> ABPManager/launchGUI.m): the 7 parameters of the
        editor "Adaptive bandpass parameters" as (name, label, string of the current value). None (with Dynamo's
        warning) unless the project has two references."""
        if self.name is None:
            self.echo("No current project", "warning")
            return None
        if float(self.runtime["nref_r1"]) != 2:
            self.echo("This option is valid only for projects with two references.", "warning")
            return None
        return [(n, parameter(n)["pretty"] or n, format_str(self.runtime[n])) for n in ABP_PARAMETERS]

    def abp_set(self, name, string):
        """An edit in the adaptive-bandpass editor: the value goes into the project parameter and
        dcp.setParameter -> dProject.setParameter(...,'inDisk',true) -> getVpr + dynamo_vpr_save
        (projectParameter2mbParameter.m:35-48, @dcp/setParameter.m, @dProject/setParameter.m:57-64): a GUI-style
        save (the 468 convention fields). Type 1 parameters are mbs.p.Scalar (singleScalar/check.m: numeric, one
        element), type 2 mbs.p.Switcher. The text is converted with pydyn's str2num (the parameterEditor's own
        text conversion is not ported; the stored values of the user's project are reproduced)."""
        if name not in ABP_PARAMETERS:
            raise DynamoError("%s is not a parameter of the adaptive bandpass editor" % name)
        if self.abp_parameters() is None:
            return Check(False, ["This option is valid only for projects with two references."])
        v = str2num(string) if isinstance(string, str) else string
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        a = np.ravel(np.asarray(v, dtype=float))
        if a.size != 1:
            msg = 'Input for parameter "%s" has length %d (expected 1)' % (name, a.size)
            self.echo(msg, "r")
            return Check(False, [msg])
        self.runtime[name] = float(a[0])
        self._gui_save()
        return Check(True, value=float(a[0]))

    def preview(self, which="mask"):
        """Volumes for the overlay view of the masks window (dynamo_slices 'overlay' of the mask on the template):
        (template, mask) as [z, y, x] arrays, or None when one is missing or their sizes differ."""
        from .console import dread
        t = self.runtime["file_template_initial"]
        m = self.runtime[{"mask": "file_mask", "cmask": "file_mask_classification", "smask": "file_mask_smoothing",
                          "fmask": "file_fmask_initial"}[which]]
        if not (os.path.isfile(t) and os.path.isfile(m)):
            return None
        tv, mv = np.asarray(dread(t)), np.asarray(dread(m))
        if tv.shape != mv.shape:
            self.echo("Dimension mistmatch: template %s, mask %s" % (format_str(list(tv.shape[::-1])),
                                                                     format_str(list(mv.shape[::-1]))), "r")
            return None
        return tv, mv

    # ------------------------------------------------------------------------------------------------ viewers
    # The menu entries that open Dynamo's viewers (pydyn.viewers, a separate Qt process: each call returns at once).
    # Messages are Dynamo's (dynamo__wizard_masks.m, dynamo__wizard_template.m, dynamo__wizard_table.m,
    # dynamo_project_manager_easy.m); docs/analysis/11a_viewers_1.1.514.md section 8. Tests monkeypatch
    # pydyn.viewers._launch.
    def _seed_file(self, item, ref):
        seeds = self.available_seeds(item)
        files = seeds.get("files") or []
        if int(float(self.runtime["nref_r1"])) == 1:
            return files[0] if files else ""
        return files[ref - 1] if 0 < ref <= len(files) else ""

    def view_template(self, ref=1):
        """Template window, Inspect > "dview (gray scale browser)" (dynamo__wizard_template.m:388-406)."""
        from . import viewers
        self.require_project()
        file = self._seed_file("template", ref)
        if not file:
            self.echo("Could not recover template for reference %d" % ref, "r")
            return None
        self.echo("Showing file in dview: %s" % file)
        p = viewers.view_file(file)
        self.echo("Done", "k")
        return p

    def plot_orientations(self, ref=1):
        """Table window, Inspect > "Plot orientations" (dynamo__wizard_table.m:422-454 showTable('dtplot'))."""
        from . import viewers
        self.echo("Looking up table.", "waiting")
        if self.name is None:
            self.echo("No current project.")
            return None
        file = self._seed_file("table", ref)
        if not file:
            self.echo("Could not retrieve initial table file for reference %d" % ref, "warning")
            return None
        self.echo("Found table for reference %d" % ref, "waiting")
        self.echo(file, "waiting")
        return viewers.dtplot(file)

    def _mask_file(self, mkind):
        field = {"mask": "file_mask", "cmask": "file_mask_classification"}[mkind]
        value = self.runtime[field]
        return value if isinstance(value, str) else ""

    def view_mask(self, mkind="mask"):
        """Masks window, View mask > "dview" (dynamo__wizard_masks.m:446-456; cmask: the same for the classification
        mask)."""
        from . import viewers
        self.echo("Retrieving current %s" % ("mask" if mkind == "mask" else "classification mask"), "waiting")
        mask = self._mask_file(mkind)
        if not mask:
            self.echo("Cannot retrieve current mask")
            return None
        self.echo("Mask in project:", "waiting")
        self.echo(mask, "waiting")
        p = viewers.view_file(mask)
        self.echo("Done", "k")
        return p

    def overlay_mask(self, kind, mkind="mask", ref=1):
        """Masks window, View mask > "overlay x|y|z" (dynamo__wizard_masks.m:458-517), with the size check."""
        from . import viewers
        self.echo("Retrieving current %s" % ("mask" if mkind == "mask" else "classification mask"), "waiting")
        mask = self._mask_file(mkind)
        if not mask:
            self.echo("Cannot retrieve current mask", "warning")
            return None
        self.echo("Mask in project:", "waiting")
        self.echo(mask, "waiting")
        template = self._seed_file("template", ref)
        if not template:
            self.echo("Cannot retrieve current template (for ref %d)" % ref, "warning")
            return None
        st, sm = read_size(template), read_size(mask)
        if st != sm:
            self.echo("Dimension mistmatch: ", "r")
            self.echo("Mask      File      : %s " % mask, "r")
            self.echo("          Dimension : %s " % format_str(list(sm)), "r")
            self.echo("Template  File      : %s " % template, "r")
            self.echo("          Dimension : %s " % format_str(list(st)), "r")
            return None
        p = viewers.overlay_mask(template, mask, kind, mkind, ref)
        self.echo("Done overlay of %s" % mkind, "k")
        return p

    def view_last_averages(self):
        """Main window, View > "Last computed average(s)" (dynamo_project_manager_easy.m:1099-1101)."""
        from . import viewers
        self.require_project()
        return viewers.show_last_averages(self.name)

    def plot_adaptive_filtering_fsc(self):
        """Main window, Multireference > Adaptive filtering > "Plot attained resolution FSC for all iterations"
        (dynamo_project_manager_easy.m:1245-1252)."""
        from . import viewers
        if self.name is None:
            self.echo("No current project", "warning")
            return None
        p = viewers.plot_bandpass_fsc(self.name)
        self.echo("done plotting convergence story.", "k")
        return p

    # ------------------------------------------------------------------------------------------------ rounds window
    def rounds_table(self, areas=ROUND_AREAS_ON, include_advanced=False, round_parameters=True):
        """dProject.runtimeRoundPublish (S/@dProject/runtimeRoundPublish.m) as filtered by the rounds window
        (dynamo__wizard_rounds.m updateTable). Returns rows (name, label, [cell strings]) and the column names."""
        self.require_project()
        areas = list(areas)
        if not areas:
            self.echo("Man, there are no functionalities left... please edit some parameter...")
            return [], []
        plist = _conv.RC if round_parameters else _conv.NC           # dProject.roundParameterList / general...
        rows = []
        for pc in plist:
            p = parameter(pc["name"])
            if p["functionality"] not in areas:
                continue
            if not include_advanced and p["advanced"]:
                continue
            label = p["pretty"] or p["name"]
            if round_parameters:
                cells = [format_str(self.runtime["%s_r%d" % (p["name"], r)]) for r in range(1, 9)]
            else:
                cells = [format_str(self.runtime[p["name"]])]
            rows.append((p["name"], label, cells))
        cols = ["round %d" % r for r in range(1, 9)] if round_parameters else ["value"]
        return rows, cols

    def comment(self, name):
        """htable_CellSelectionCallback: the comment of the selected parameter."""
        return parameter(name)["comment"] or "no comment available on this parameter, sorry."

    @staticmethod
    def parameter_help(name):
        """dynamo_vpr_help(name) (S/dynamo_vpr_help.m:68 -> S/dynamo__help_parameters.m): Dynamo's detailed help
        lines of a parameter, verbatim (pydyn.dcp_help), or [] when Dynamo has none for it."""
        from .dcp_help import HELP
        return list(HELP.get(name, ()))

    def set_value(self, name, string, round=None):
        """htable_CellEditCallback (dynamo__wizard_rounds.m): checkValuePassedAsString, setValue (no check),
        dcp.setParameter -> dProject.setParameter(...,'inDisk',true) -> getVpr + dynamo_vpr_save."""
        self.require_project()
        is_round = name in RC
        if is_round and round is None:
            raise DynamoError("round parameter %s needs a round" % name)
        if not is_round and round is not None:
            raise DynamoError("%s is not a round parameter" % name)
        c = self.value_from_string(name, string)
        if not c.ok:
            self.echo(c.report, "r")
            self.echo('Parameter "%s" change rejected.' % name, "r")
            return c
        key = "%s_r%d" % (name, int(round)) if is_round else name
        self.runtime[key] = V.as_matlab_value(c.value) if not (isinstance(c.value, np.ndarray) and
                                                               c.value.size == 0) else c.value
        self._gui_save()
        return c

    def active_rounds(self):
        """Rounds with ite_r > 0 (the only meaning of 'a round' in a Dynamo project: there are always 8)."""
        def scalar(v):
            a = np.ravel(np.asarray(v, dtype=float))
            return float(a[0]) if a.size else 0.0
        return [r for r in range(1, 9) if scalar(self.runtime["ite_r%d" % r]) > 0]

    def iteration_rounds(self):
        """The round of every iteration (dynamo_zaux_intertwin_round_identifier_array), from the runtime."""
        from .dynproject import round_array
        return round_array(self.runtime)

    def add_round(self, n_iterations="1"):
        """Shortcut: type n into the 'iterations' cell of the first round with ite 0 after the last used one."""
        act = self.active_rounds()
        r = (max(act) + 1) if act else 1
        if r > 8:
            raise DynamoError("a Dynamo project has 8 rounds")
        return r, self.set_value("ite", n_iterations, r)

    def remove_round(self):
        """Shortcut: type 0 into the 'iterations' cell of the last used round (its other values are kept, as in
        the user's project where rounds 3-4 hold values but ite 0)."""
        act = self.active_rounds()
        if not act:
            raise DynamoError("no round has iterations")
        r = max(act)
        return r, self.set_value("ite", "0", r)

    # ------------------------------------------------------------------------------------------------ environment
    def environment_values(self):
        """Edit boxes of the environment window: (string, valid) per parameter (OpeningFcn)."""
        out = {}
        for p in ENVIRONMENT_PARAMETERS:
            v = self.runtime[p]
            out[p] = (format_str(v), self.check_value(p, v).ok)
        return out

    def set_destination(self, destination):
        """Destination radio buttons (dynamo__wizard_environment.m setDestination): dcp.put('destination', tag)."""
        if destination not in DESTINATIONS:
            raise DynamoError("This GUI cannot handle destination \"%s\"" % destination)
        self.put("destination", destination)
        if destination not in PYDYN_EXECUTES:
            self.echo('[pydyn] destination "%s" is stored as in Dynamo, but pydyn executes only %s projects'
                      % (destination, " / ".join(PYDYN_EXECUTES)), "warning")
        return Check(True)

    def environment_edit(self, name, string):
        """dcp.parameterGUIEditCallback (S/@dcp/parameterGUIEditCallback.m) for an environment edit box; for
        how_many_processors under matlab_parfor the value is copied to matlab_workers_average
        (edit_how_many_processors_Callback)."""
        if name not in ENVIRONMENT_PARAMETERS:
            raise DynamoError("%s has no edit box in the environment window" % name)
        c = self._gui_edit(name, string)
        if c.ok and name == "how_many_processors" and self.runtime["destination"] == "matlab_parfor":
            self._gui_edit("matlab_workers_average", string)
        return c

    def _gui_edit(self, name, string):
        c = self.value_from_string(name, string)
        if not c.ok:
            self.echo('Invalid value for parameter "%s"' % parameter(name)["pretty"], "r")
            self.echo(c.report, "r")
            return c
        self.put(name, c.value)
        self.echo('"%s"  in disk  set to "%s"' % (parameter(name)["pretty"], string))
        return c

    # ------------------------------------------------------------------------------------------------ check / unfold
    def execution_script_name(self):
        """dpkproject.executionScript(name): <name>.exe for standalone / system destinations, <name>.m for matlab."""
        dest = self.runtime["destination"]
        if dest in ("matlab", "matlab_gpu", "matlab_parfor"):
            return "%s.m" % self.name
        if dest in ("standalone", "system_gpu", "standalone_gpu"):
            return "%s.exe" % self.name
        return None

    def check(self):
        """Main window "check" (dynamo_project_manager_easy.m pushbutton_check_Callback): checkSteps,
        stepsStatusAsRead2GUI, dynamo_vpr_check; unfold is enabled when there is no error or the only error is
        how_many_processors."""
        from .vprcheck import check as dvcheck
        self.require_project()
        self.echo("Checking project consistency", "waiting")
        self.check_steps()
        self._steps_status_as_read()
        r = dvcheck(self.name, verbosity=0)
        if not r.report:
            self.echo('No errors found in project : "%s"' % self.name)
        else:
            self.echo(r.report, r.warning_state)
        errors = list(r.fields_error) if r.fields_error != "Aborted" else ["Aborted"]
        if errors == ["how_many_processors"]:
            self.echo("Error is related to number of cores")
            self.echo("You can unfold the project, but be carefull!")
            self.echo("You probably need to use the order <dpkmulticore.run('%s');>" % self.name)   # :407
            errors = []
        if not errors:
            script = self.execution_script_name()
            if not (script and os.path.isfile(script)):
                self.echo('Cannot find file "%s"' % script, r.warning_state)
                self.echo("No execution script yet ready, please unfold the project.", r.warning_state)
            self._enable("unfold", True)
        else:
            self._enable("run", False)
            self._enable("unfold", False)
        return r

    def unfold(self):
        """Main window "unfold" (pushbutton_unfold_Callback): dynamo_vpr_unfold(name) without a check."""
        self.require_project()
        self.echo("Unfolding the project. Please wait!", "waiting")
        if not V.isproject(self.name):
            self.echo("Invalid project.", "warning")
            return None, "error"
        script, status = V.unfold(self.name)
        self.echo("Done unfolding. Execution script ready:")
        self.echo("%s" % script)
        if status == "ok":
            self._enable("run", True)
            if self.runtime["destination"] not in PYDYN_EXECUTES:
                self.echo("[pydyn] destination %s: pydyn executes only %s projects"
                          % (self.runtime["destination"], " / ".join(PYDYN_EXECUTES)), "warning")
        return script, status

    def execution_order(self):
        """pushbutton_run_Callback: the order Dynamo would pass to the system (pydyn shows it; run it in a
        terminal)."""
        dest = self.runtime["destination"]
        if dest in ("standalone", "system_gpu", "standalone_gpu", "system", "omp", "system_omp"):
            return "./%s.exe" % self.name
        if dest in ("matlab", "matlab_gpu", "matlab_parfor"):
            return "run %s" % self.name
        if dest in ("mpi", "mpi_gpu"):
            return "%s %s.sh" % (self.runtime["submit_order"], self.name)
        raise DynamoError('[wizard] unknown destination "%s"' % dest)
