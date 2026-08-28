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

"""A corrupted profile must degrade CONFIG, not make dashpva.settings unimportable.

Regression: resolve_profile_config() was called unguarded in both the
module-level reload() (run at import time) and Settings.reload() (run from
__init__). A profile with a non-list IOC_RSM_PARAMETER.SAMPLE_AXES -- e.g.
one corrupted by the separate workflow.py config-tree bug -- raised straight
through both, which would have bricked every DashPVA entry point including
the HKL Setup tool meant to repair the profile.
"""

import toml


def _corrupted_profile_toml(tmp_path):
    path = tmp_path / "corrupted.toml"
    path.write_text(
        toml.dumps(
            {
                "IOC_PREFIX": "test:",
                "IOC_RSM_PARAMETER": {
                    "SCHEMA_VERSION": 1,
                    # Should be a list of axis tables; simulates the
                    # workflow.py tree bug stringifying it.
                    "SAMPLE_AXES": "[{'LABEL': 'Mu'}]",
                    "DETECTOR_AXES": [],
                },
            }
        )
    )
    return str(path)


def test_module_reload_degrades_on_corrupted_profile(monkeypatch, isolated_settings, tmp_path):
    s = isolated_settings
    path = _corrupted_profile_toml(tmp_path)

    s.set_locator(path)
    s.reload()  # must not raise

    assert s.CONFIG_ERROR is not None
    assert "SAMPLE_AXES" in s.CONFIG_ERROR
    # Falls back to the raw, unresolved config rather than losing everything.
    assert s.CONFIG.get("IOC_PREFIX") == "test:"


def test_module_reload_clears_config_error_on_clean_profile(
    monkeypatch, isolated_settings, tmp_path, tmp_toml
):
    s = isolated_settings
    s.set_locator(_corrupted_profile_toml(tmp_path))
    s.reload()
    assert s.CONFIG_ERROR is not None

    s.set_locator(tmp_toml)
    s.reload()
    assert s.CONFIG_ERROR is None


def test_settings_class_degrades_on_corrupted_profile(tmp_path):
    import dashpva.settings as settings

    path = _corrupted_profile_toml(tmp_path)
    s = settings.Settings.from_toml(path)  # must not raise

    assert s.CONFIG_ERROR is not None
    assert "SAMPLE_AXES" in s.CONFIG_ERROR
    assert s.CONFIG.get("IOC_PREFIX") == "test:"
