# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

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
