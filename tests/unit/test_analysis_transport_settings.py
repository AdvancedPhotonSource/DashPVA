# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
