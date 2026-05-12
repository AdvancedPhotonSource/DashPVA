"""Tests for dashpva.settings — configuration loading, defaults, and priority."""

import os

import pytest


class TestSettingsDefaults:

    def test_reload_with_no_config(self, isolated_settings):
        s = isolated_settings
        assert s.DETECTOR_PREFIX is None
        assert s.OUTPUT_FILE_LOCATION is None
        assert isinstance(s.METADATA_CA, dict)
        assert isinstance(s.METADATA_PVA, dict)

    def test_project_root_is_path(self, isolated_settings):
        from pathlib import Path

        assert isinstance(isolated_settings.PROJECT_ROOT, Path)

    def test_version_matches_package(self, isolated_settings):
        import dashpva

        assert isolated_settings.__VERSION__ == dashpva.__version__

    def test_hdf5_structure_has_nexus(self, isolated_settings):
        assert "nexus" in isolated_settings.HDF5_STRUCTURE
        assert "default" in isolated_settings.HDF5_STRUCTURE["nexus"]


class TestSettingsReload:

    def test_reload_from_toml(self, monkeypatch, tmp_toml):
        import dashpva.settings as settings

        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        settings.set_locator(tmp_toml)
        settings.reload()

        assert settings.DETECTOR_PREFIX == "13SIM1:"
        assert settings.CACHING_MODE == "disk"
        assert settings.ALIGNMENT_MAX_CACHE_SIZE == 100
        assert settings.SCAN_THRESHOLD == 0.5
        assert settings.BIN_COUNT == 10

        settings.set_locator(None)
        monkeypatch.setattr(settings, "_locator_internal", None)
        settings.reload()

    def test_reload_parses_scan_options(self, monkeypatch, tmp_toml):
        import dashpva.settings as settings

        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        settings.set_locator(tmp_toml)
        settings.reload()

        assert settings.SCAN_START_SCAN is True
        assert settings.SCAN_STOP_SCAN is False
        assert settings.SCAN_MAX_CACHE_SIZE == 50

        settings.set_locator(None)
        monkeypatch.setattr(settings, "_locator_internal", None)
        settings.reload()

    def test_reload_parses_metadata_sections(self, monkeypatch, tmp_toml):
        import dashpva.settings as settings

        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        settings.set_locator(tmp_toml)
        settings.reload()

        assert settings.METADATA_CA.get("FLAG_PV") == "test:flag"
        assert settings.METADATA_PVA.get("CHANNEL") == "test:pva"

        settings.set_locator(None)
        monkeypatch.setattr(settings, "_locator_internal", None)
        settings.reload()


class TestGetEffectiveLocator:

    def test_programmatic_takes_priority(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setenv("DASPVA_CONFIG_LOCATOR", "/env/path.toml")
        settings.set_locator("/programmatic/path.toml")
        eff = settings._get_effective_locator()
        assert eff == "/programmatic/path.toml"

        settings.set_locator(None)
        monkeypatch.setattr(settings, "_locator_internal", None)

    def test_env_var_fallback(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.setenv("DASPVA_CONFIG_LOCATOR", "/env/config.toml")
        eff = settings._get_effective_locator()
        assert eff == "/env/config.toml"

        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)

    def test_env_var_int_conversion(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.setenv("DASPVA_CONFIG_LOCATOR", "42")
        eff = settings._get_effective_locator()
        assert eff == 42

        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)

    def test_none_when_nothing_set(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        eff = settings._get_effective_locator()
        assert eff is None
