"""dcp: the Dynamo 1.1.514 project tool, without MATLAB (module pydynamo.manager.tool; exported as pydynamo.dcp).

  from pydynamo import dcp
  dcp.new('bin8_align_1', d='filamentsData_b8_ctf', template='raw_template_ctf.em', masks='default',
          t='particles_b8_edit_mod.tbl')                        # or MATLAB style: dcp.new('p', 'd', 'data', ...)
  dcp.open('bin8_align_1')                                       # opens the GUI on the project
  dcp.put('destination', 'standalone_gpu')                       # current (opened) project, saved into ./
  dcp()                                                          # opens the GUI

Specification: docs/analysis/08a_project_creation_1.1.514.md sections 2 and 5.1 (S/@dcp/new.m:30-121,
S/+dpkproject/+parsers/@ParserSeeds/ParserSeeds.m:13-95, S/@dcp/put.m).
"""
import os
import shutil

from pydynamo import vpr as V
from pydynamo.vpr import DynamoError

# ParserSeeds.m:16-24 and new.m:40-41: option -> short forms
_OPTIONS = {"data": ("d",), "table": ("t",), "template": ("tmp", "te"), "mask": ("m", "alignment_mask"),
            "fmask": ("fm", "fourier_mask"), "smask": ("sm", "smoothing_mask"),
            "cmask": ("cm", "classification_mask"), "masks": (), "gui": ("g", "show"),
            "forceOverwrite": ("force", "fo")}
_DEFAULTS = {"data": "", "table": "", "template": "", "mask": "", "fmask": "", "smask": "", "cmask": "",
             "masks": "default", "gui": True, "forceOverwrite": False}


def _parse(args, kw):
    alias = {}
    for k, shorts in _OPTIONS.items():
        alias[k] = k
        for s in shorts:
            alias[s] = k
    if len(args) % 2:
        raise DynamoError("dcp.new: options must come as name/value pairs")
    q = dict(_DEFAULTS)
    items = [(args[i], args[i + 1]) for i in range(0, len(args), 2)] + list(kw.items())
    for k, v in items:
        if not isinstance(k, str) or k.lstrip("-") not in alias:
            raise DynamoError("dcp.new: unknown option %r" % (k,))
        q[alias[k.lstrip("-")]] = v
    for k in ("gui", "forceOverwrite"):
        v = q[k]
        q[k] = bool(int(v)) if isinstance(v, str) and v.strip().lstrip("-").isdigit() else bool(v)
    return q


def _delete_project(name):
    """dynamo_delete(name) (S/dynamo_delete.m): the folder and <name>{_comment.doc,.m,.exe,.sh,.bat,.cmp,.bdl,.seq,
    .ctlg}."""
    print("Trying to delete 1 primary objects")
    if os.path.isdir(name):
        shutil.rmtree(name)
    for suf in ("_comment.doc", ".m", ".exe", ".sh", ".bat", ".cmp", ".bdl", ".seq", ".ctlg"):
        if os.path.isfile(name + suf):
            os.remove(name + suf)


class _Dcp:
    """The `dcp` command: dcp() opens the GUI; dcp.new / dcp.open / dcp.put as in Dynamo."""

    def __init__(self):
        self.current = None                          # name of the project opened in the GUI (dcp.getProject)

    def __call__(self, *args):
        """dcp() / dcp('name'): the dcp window in its own process (pydynamo.webui)."""
        from pydynamo import webui
        return webui.launch(args[0] if args else None)

    def __repr__(self):
        return "<pydynamo dcp (Dynamo 1.1.514 project tool); current project: %r>" % self.current

    # ----------------------------------------------------------------------------------------------------- new
    def new(self, name=None, *args, **kw):
        """dcp.new(name, 'd', data, 'template', file, 'masks', 'default', 't', table, ...) (S/@dcp/new.m:30-121).
        Returns the vpr dict, or None when the project exists already (Dynamo prints and returns)."""
        if name is None:
            print("You need to provide a name for the project.")
            return None
        q = _parse(args, kw)
        if q["data"]:
            facts = V.data_info(q["data"])
            if not facts["correct"]:
                print("Passed data: %s" % q["data"])
                raise DynamoError("Cannot create project, invalid data!")
        for k, msg in (("table", "table"), ("template", "template"), ("mask", "mask")):
            if q[k] and isinstance(q[k], str) and not os.path.exists(q[k]):
                raise DynamoError("Cannot create project, invalid %s" % msg)
            if q[k] and not isinstance(q[k], str):
                raise NotImplementedError("dcp.new with a non-file %s" % k)
        if V.isproject(name):
            if not q["forceOverwrite"]:
                print(' cannot use "dcp.new".  The project "%s" exist already ' % name)
                print(" If you want to open it use:")
                print(" dcp.open('%s');" % name)
                print('If you want to overwrite it, use the "forceOverwrite" flag (switching it to 1)')
                return None
            print("Deleting previous project with the same name.")
            _delete_project(name)
        vpr = V.blank(name)                          # new.m:102: dynamo_vpr(name) saves the blank project
        vpr = V.save(vpr)
        print(" ")
        print("[vpr:ok] dynamo_vpr finished successfully.")
        print(" ")
        vpr = self._set_project_seeds(vpr, q)        # new.m:107
        vpr = V.save(vpr)                            # new.m:108
        if q["gui"]:
            self.open(name)
            print("The Wizard should have been opened")
        return vpr

    @staticmethod
    def _set_project_seeds(vpr, q):
        """ParserSeeds.setProjectSeeds (ParserSeeds.m:32-95)."""
        for s in ("data", "table", "template"):
            if q[s]:
                vpr = V.modify(vpr, s, q[s])
        if q["template"]:
            if not isinstance(q["template"], str):
                raise DynamoError("Sorry, the template must be passed as a string")
            if q["template"] == "average":
                raise DynamoError("on the fly average not ready")
            if q["template"] == "randomAverage":
                raise DynamoError("on the fly random average not ready")
            vpr = V.modify(vpr, "template", q["template"])
        all_masks = ["mask", "cmask", "smask", "fmask"]
        explicit = [m for m in all_masks if q[m]]
        final_default = sorted(set(all_masks) - set(explicit))      # MATLAB setdiff returns sorted values
        if q["masks"] in ("d", "def", "default", "hint"):
            if q["data"]:
                print("Creating %d default masks" % len(final_default))
                vpr = V.hint_files(vpr, final_default)
            else:
                print("warning: No data has been passed, cannot create hints.")
        for m in explicit:
            print("Passing %d mask that were explicitely passed." % len(explicit))
            if m == "smask":
                # dynamo_vpr_modify has no 'smask' shorthand: Dynamo 1.1.514 gets an empty vpr here (a bug)
                raise DynamoError('ERROR: "smask" is not a valid parameter (dynamo_vpr_modify, called from '
                                  "ParserSeeds.m:88): Dynamo 1.1.514 cannot pass an explicit smoothing mask in dcp.new")
            vpr = V.modify(vpr, m, q[m])
        return vpr

    # ----------------------------------------------------------------------------------------------------- open
    def open(self, name):
        """dcp.open(name) (S/@dcp/open.m): makes `name` the current project and opens the GUI on it."""
        if not V.isproject(name):
            print('Project "%s" not found in %s' % (name, os.getcwd()))
            return None
        self.current = name
        from pydynamo import webui                           # its own process: the console stays free, as MATLAB's
        return name if webui.launch(name) is not None else None

    def set_project(self, name):
        """dcp.setProject: make `name` the current project without opening the GUI (used by the GUI)."""
        if not V.isproject(name):
            raise DynamoError('Project "%s" not found' % name)
        self.current = name

    # ----------------------------------------------------------------------------------------------------- put
    def put(self, parameter, value, round=None):
        """dcp.put(p, v[, round]) (S/@dcp/put.m): load the current project, set the field (no type check, no
        round2all, so existing *_all fields stay as they are), save into ./ (08a 5.1)."""
        if self.current is None:
            print("Aborting [dcp.put] for modification of parameter in disk for current project.")
            return None
        vpr = V.load(self.current)
        key = parameter if round is None else "%s_r%d" % (parameter, int(round))
        vpr[key] = V.as_matlab_value(value)
        return V.save(vpr)


dcp = _Dcp()
