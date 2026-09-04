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

"""Smoke tests verifying all packages import cleanly after the src/ restructure."""

import importlib

import pytest


CORE_MODULES = [
    "dashpva",
    "dashpva.settings",
    "dashpva.cli",
    "dashpva.gui",
    "dashpva.database",
    "dashpva.database.db",
    "dashpva.database.interface",
    "dashpva.database.models.profile",
    "dashpva.database.models.setting_value",
    "dashpva.database.models.settings",
    "dashpva.utils",
    "dashpva.utils.stats_analysis",
    "dashpva.utils.roi_ops",
    "dashpva.utils.generators",
    "dashpva.utils.mask_manager",
    "dashpva.utils.log_manager",
    "dashpva.utils.config",
    "dashpva.utils.config.source",
    "dashpva.viewer.launcher.registry",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_import_module(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_version_is_string():
    import dashpva

    assert isinstance(dashpva.__version__, str)
    assert len(dashpva.__version__) > 0


def test_version_consistency():
    import dashpva
    import dashpva.settings as settings

    assert dashpva.__version__ == settings.__VERSION__


def test_settings_project_root_exists():
    import dashpva.settings as settings

    assert settings.PROJECT_ROOT.exists()
    assert (settings.PROJECT_ROOT / "pyproject.toml").exists()


def test_hdf5_structure_is_dict():
    import dashpva.settings as settings

    assert isinstance(settings.HDF5_STRUCTURE, dict)
    assert "nexus" in settings.HDF5_STRUCTURE


def test_lazy_import_dash_analysis():
    from dashpva.utils import DashAnalysis

    assert DashAnalysis is not None


def test_lazy_import_mask_manager():
    from dashpva.utils import MaskManager

    assert MaskManager is not None


def test_lazy_import_size_manager():
    from dashpva.utils import SizeManager

    assert SizeManager is not None


def test_database_interface_importable():
    from dashpva.database import DatabaseInterface

    assert DatabaseInterface is not None


def test_cli_group_importable():
    from dashpva.cli import cli

    assert cli is not None
