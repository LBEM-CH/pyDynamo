"""pydyn: Dynamo 1.1.514 subtomogram averaging in Python (GPU), without MATLAB.

Stage 3 console API (names as in Dynamo):

  from pydyn import dread, dwrite, dtrandomize_azimuth, daverage, dcp, dvput, dvcheck, dvunfold, dvload

  T  = dread('particles_b8_edit.tbl')
  T2 = dtrandomize_azimuth(T)
  dwrite(T2, 'particles_b8_edit_mod.tbl')
  oa = daverage('filamentsData_b8_ctf', 't', 'particles_b8_edit_mod.tbl', 'mw', 50)
  dwrite(oa.average, 'raw_template_ctf.em')
  dcp.new('bin8_align_1', d='filamentsData_b8_ctf', template='raw_template_ctf.em', masks='default',
          t='particles_b8_edit_mod.tbl', gui=0)
  m = dmask('size', 64, 'type', 'cylinder', 'r', 15, 'height', 30, 'gaussian', 3, 'invert', False)
  dwrite(m, 'abp_align/settings/editorMask.em')      # = the dcp mask editor's save (pydyn.masks)

Exports are loaded lazily, so `import pydyn` does not import NumPy, SciPy or CuPy.
"""
_EXPORTS = {
    "dread": "console", "dwrite": "console", "dtrandomize_azimuth": "console", "daverage": "console",
    "rng": "console", "dcp": "dcptool", "dvput": "vpr", "dvcheck": "vprcheck", "dvunfold": "vpr",
    "dvload": "vpr", "dynamo_execute_project": "execute", "execute_project": "execute",
    "dveo": "vpreo", "dynamo_vpr_eo": "vpreo",
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
    raise AttributeError("module 'pydyn' has no attribute %r" % name)
