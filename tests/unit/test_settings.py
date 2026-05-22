"""Tests for dashpva.settings — configuration loading, defaults, and priority."""


class TestSettingsDefaults:

    def test_reload_with_no_config(self, monkeypatch, isolated_settings):
        s = isolated_settings
        monkeypatch.setattr(s, "ConfigSource", None)
        s.reload()
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

        assert settings.METADATA_CA.get("ENERGY") == "test:energy"
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
        monkeypatch.setattr(settings, "_STATE_FILE", settings.PROJECT_ROOT / ".dashpva_locator_test_nonexistent")
        eff = settings._get_effective_locator()
        assert eff is None

    def test_state_file_fallback(self, monkeypatch, tmp_path):
        import dashpva.settings as settings

        state_file = tmp_path / ".dashpva_locator"
        state_file.write_text("/some/config.toml")
        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        monkeypatch.setattr(settings, "_STATE_FILE", state_file)
        eff = settings._get_effective_locator()
        assert eff == "/some/config.toml"

    def test_state_file_int_locator(self, monkeypatch, tmp_path):
        import dashpva.settings as settings

        state_file = tmp_path / ".dashpva_locator"
        state_file.write_text("42")
        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        monkeypatch.setattr(settings, "_STATE_FILE", state_file)
        eff = settings._get_effective_locator()
        assert eff == 42

    def test_set_locator_writes_state_file(self, monkeypatch, tmp_path):
        import dashpva.settings as settings

        state_file = tmp_path / ".dashpva_locator"
        monkeypatch.setattr(settings, "_STATE_FILE", state_file)
        settings.set_locator("/test/path.toml")
        assert state_file.read_text() == "/test/path.toml"
        monkeypatch.setattr(settings, "_locator_internal", None)


class TestInputChannel:

    def test_get_input_channel_from_explicit(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "INPUT_CHANNEL", "custom:Channel")
        monkeypatch.setattr(settings, "DETECTOR_PREFIX", "somePrefix")
        assert settings.get_input_channel() == "custom:Channel"

    def test_get_input_channel_from_prefix(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "INPUT_CHANNEL", None)
        monkeypatch.setattr(settings, "DETECTOR_PREFIX", "4idEiger")
        assert settings.get_input_channel() == "4idEiger:Pva1:Image"

    def test_get_input_channel_fallback(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "INPUT_CHANNEL", None)
        monkeypatch.setattr(settings, "DETECTOR_PREFIX", None)
        assert settings.get_input_channel() == "pvapy:image"

    def test_get_input_channel_custom_fallback(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "INPUT_CHANNEL", None)
        monkeypatch.setattr(settings, "DETECTOR_PREFIX", None)
        assert settings.get_input_channel("vit:1:input_phase") == "vit:1:input_phase"

    def test_save_input_channel_updates_global(self, monkeypatch):
        import dashpva.settings as settings

        monkeypatch.setattr(settings, "INPUT_CHANNEL", None)
        monkeypatch.setattr(settings, "_locator_internal", None)
        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        monkeypatch.setattr(settings, "_STATE_FILE", settings.PROJECT_ROOT / ".dashpva_locator_test_nonexistent")
        settings.save_input_channel("test:Channel")
        assert settings.INPUT_CHANNEL == "test:Channel"
        monkeypatch.setattr(settings, "INPUT_CHANNEL", None)

    def test_save_and_reload_input_channel(self, monkeypatch, tmp_toml, tmp_path):
        import dashpva.settings as settings

        state_file = tmp_path / ".dashpva_locator"
        monkeypatch.setattr(settings, "_STATE_FILE", state_file)
        monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
        settings.set_locator(tmp_toml)
        settings.reload()
        assert settings.INPUT_CHANNEL is None

        settings.save_input_channel("saved:Pva1:Image")
        assert settings.INPUT_CHANNEL == "saved:Pva1:Image"

        settings.reload()
        assert settings.INPUT_CHANNEL == "saved:Pva1:Image"

        monkeypatch.setattr(settings, "_locator_internal", None)
        settings.reload()
