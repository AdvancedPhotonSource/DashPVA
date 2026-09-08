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

"""Analysis control/status channels resolve from the active Workflow setup."""

from types import SimpleNamespace

import pytest

import dashpva.settings as app_settings


class _Database:
    def __init__(self, saved):
        self.saved = saved

    def get_setting_by_path(self, path):
        assert path == ["APP_DATA", "workflow", "analysis"]
        return SimpleNamespace(id=17)

    def get_setting_value(self, setting_id, name):
        assert (setting_id, name) == (17, "last")
        return self.saved


def test_workflow_channels_override_profile_fallback(monkeypatch):
    monkeypatch.setattr(
        app_settings,
        "ANALYSIS",
        {"CONTROL_CHANNEL": "profile:control", "STATUS_CHANNEL": "profile:status"},
    )
    database = _Database(
        {
            "control_channel": "workflow:*:control",
            "status_channel": "workflow:*:status",
            "n_consumers": 1,
        }
    )

    assert app_settings.get_analysis_transport_channels(database) == (
        "workflow:1:control",
        "workflow:1:status",
    )


def test_profile_channels_are_a_toml_only_fallback(monkeypatch):
    monkeypatch.setattr(
        app_settings,
        "ANALYSIS",
        {"CONTROL_CHANNEL": "profile:control", "STATUS_CHANNEL": "profile:status"},
    )
    assert app_settings.get_analysis_transport_channels(_Database(None)) == (
        "profile:control",
        "profile:status",
    )


def test_stateful_grid_refuses_multiple_consumers(monkeypatch):
    monkeypatch.setattr(app_settings, "ANALYSIS", {})
    database = _Database(
        {
            "control_channel": "workflow:*:control",
            "status_channel": "workflow:*:status",
            "n_consumers": 2,
        }
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        app_settings.get_analysis_transport_channels(database)
