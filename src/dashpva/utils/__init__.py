# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
_LAZY_IMPORTS = {
    "HDF5Writer": "dashpva.utils.hdf5_writer",
    "HDF5Loader": "dashpva.utils.hdf5_loader",
    "HDF5Handler": "dashpva.utils.hdf5_handler",
    "PVAReader": "dashpva.utils.pva_reader",
    "SizeManager": "dashpva.utils.size_manager",
    "rotation_cycle": "dashpva.utils.generators",
    "DashAnalysis": "dashpva.utils.dash_analysis",
    "RSMConverter": "dashpva.utils.rsm_converter",
    "RotationAxis": "dashpva.utils.rsm_geometry",
    "DetectorModel": "dashpva.utils.rsm_geometry",
    "RSMGeometry": "dashpva.utils.rsm_geometry",
    "BuiltRSMGeometry": "dashpva.utils.rsm_geometry",
    "build_hxrd": "dashpva.utils.rsm_geometry",
    "calculate_q": "dashpva.utils.rsm_geometry",
    "MaskManager": "dashpva.utils.mask_manager",
}


def __getattr__(name):  # noqa: F811
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
