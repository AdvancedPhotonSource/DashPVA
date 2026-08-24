# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
