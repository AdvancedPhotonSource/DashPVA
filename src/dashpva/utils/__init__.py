_LAZY_IMPORTS = {
    "HDF5Writer": "dashpva.utils.hdf5_writer",
    "HDF5Loader": "dashpva.utils.hdf5_loader",
    "HDF5Handler": "dashpva.utils.hdf5_handler",
    "PVAReader": "dashpva.utils.pva_reader",
    "SizeManager": "dashpva.utils.size_manager",
    "rotation_cycle": "dashpva.utils.generators",
    "DashAnalysis": "dashpva.utils.dash_analysis",
    "RSMConverter": "dashpva.utils.rsm_converter",
    "MaskManager": "dashpva.utils.mask_manager",
    "load_last":"utils.user_config",
    "save_last":"utils.user_config"
}


def __getattr__(name):  # noqa: F811
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
