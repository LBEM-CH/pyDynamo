"""pyDynamo: Dynamo 1.1.514 subtomogram averaging in Python (GPU), without MATLAB.

Dynamo's console commands, with Dynamo's names:

  from pydynamo import dread, dwrite, dtrandomize_azimuth, daverage, dcp, dvput, dvcheck, dvunfold, dvload

  T  = dread('particles_b8_edit.tbl')
  T2 = dtrandomize_azimuth(T)
  dwrite(T2, 'particles_b8_edit_mod.tbl')
  oa = daverage('filamentsData_b8_ctf', 't', 'particles_b8_edit_mod.tbl', 'mw', 50)
  dwrite(oa.average, 'raw_template_ctf.em')
  dcp.new('bin8_align_1', d='filamentsData_b8_ctf', template='raw_template_ctf.em', masks='default',
          t='particles_b8_edit_mod.tbl', gui=0)
  m = dmask('size', 64, 'type', 'cylinder', 'r', 15, 'height', 30, 'gaussian', 3, 'invert', False)
  dwrite(m, 'abp_align/settings/editorMask.em')      # = the dcp mask editor's save (pydynamo.masks)

Exports are loaded lazily, so `import pydynamo` does not import NumPy, SciPy or CuPy.
"""
_EXPORTS = {
    "dread": "console", "dwrite": "console", "dtrandomize_azimuth": "console", "daverage": "console",
    "rng": "console", "dcp": "manager.tool", "dvput": "vpr", "dvcheck": "vpr.checks", "dvunfold": "vpr",
    "dvload": "vpr", "dynamo_execute_project": "project.execute", "execute_project": "project.execute",
    "dveo": "vpr.even_odd", "dynamo_vpr_eo": "vpr.even_odd",
    "dmask": "masks", "dynamo_mask": "masks", "dmollify": "masks", "dynamo_mollify": "masks",
    "dcylinder": "masks", "dynamo_cylinder": "masks",
    "dmapview": "viewers", "dynamo_mapview": "viewers", "dview": "viewers", "dynamo_view": "viewers",
    "dslices": "viewers", "dynamo_slices": "viewers", "dtplot": "viewers", "dynamo_tableplot": "viewers",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        mod = importlib.import_module("." + _EXPORTS[name], __name__)
        return getattr(mod, name)
    raise AttributeError("module 'pydynamo' has no attribute %r" % name)
