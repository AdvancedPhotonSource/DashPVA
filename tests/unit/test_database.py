"""Tests for dashpva.database — profile CRUD and setting values."""

import pytest


class TestProfileCRUD:

    def test_create_profile(self, tmp_db):
        prof = tmp_db.create_profile("test_profile", "A test profile")
        assert prof is not None
        assert prof.name == "test_profile"
        assert prof.id is not None

    def test_get_profile_by_id(self, tmp_db):
        prof = tmp_db.create_profile("by_id_test")
        retrieved = tmp_db.get_profile_by_id(prof.id)
        assert retrieved is not None
        assert retrieved.name == "by_id_test"

    def test_get_profile_by_name(self, tmp_db):
        tmp_db.create_profile("by_name_test")
        retrieved = tmp_db.get_profile_by_name("by_name_test")
        assert retrieved is not None
        assert retrieved.name == "by_name_test"

    def test_get_all_profiles(self, tmp_db):
        tmp_db.create_profile("profile_a")
        tmp_db.create_profile("profile_b")
        profiles = tmp_db.get_all_profiles()
        names = [p.name for p in profiles]
        assert "profile_a" in names
        assert "profile_b" in names

    def test_update_profile_name(self, tmp_db):
        prof = tmp_db.create_profile("old_name")
        result = tmp_db.update_profile_name(prof.id, "new_name")
        assert result is True
        updated = tmp_db.get_profile_by_id(prof.id)
        assert updated.name == "new_name"

    def test_delete_profile(self, tmp_db):
        prof = tmp_db.create_profile("to_delete")
        result = tmp_db.delete_profile(prof.id)
        assert result is True
        assert tmp_db.get_profile_by_id(prof.id) is None

    def test_nonexistent_profile_returns_none(self, tmp_db):
        assert tmp_db.get_profile_by_id(99999) is None


class TestProfileConfigImportExport:

    def test_import_and_export_toml_dict(self, tmp_db, sample_config_dict):
        prof = tmp_db.create_profile("import_test")
        tmp_db.import_toml_to_profile(prof.id, sample_config_dict)
        exported = tmp_db.export_profile_to_toml(prof.id)
        assert isinstance(exported, dict)
        assert exported.get("DETECTOR_PREFIX") == "13SIM1:"

    def test_import_toml_file(self, tmp_db, tmp_toml):
        prof = tmp_db.create_profile("file_import_test")
        tmp_db.import_toml_file(prof.id, tmp_toml)
        exported = tmp_db.export_profile_to_toml(prof.id)
        assert exported.get("DETECTOR_PREFIX") == "13SIM1:"


class TestSettingValue:

    def test_auto_detect_int(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value(42)
        assert sv.value_type == "int"
        assert sv.get_value() == 42

    def test_auto_detect_float(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value(3.14)
        assert sv.value_type == "float"
        assert abs(sv.get_value() - 3.14) < 1e-10

    def test_auto_detect_string(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value("hello")
        assert sv.value_type == "string"
        assert sv.get_value() == "hello"

    def test_auto_detect_dict_json(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value({"a": 1, "b": [2, 3]})
        assert sv.value_type == "json"
        result = sv.get_value()
        assert result == {"a": 1, "b": [2, 3]}

    def test_auto_detect_list_json(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value([1, 2, 3])
        assert sv.value_type == "json"
        assert sv.get_value() == [1, 2, 3]

    def test_bool_stored_as_string(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value(True)
        assert sv.value_type == "string"

    def test_explicit_value_type(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="", value_type="string")
        sv.set_value("42", value_type="int")
        assert sv.value_type == "int"

    def test_get_value_invalid_int_returns_raw(self):
        from dashpva.database.models.setting_value import SettingValue

        sv = SettingValue(key="test", value="abc", value_type="int")
        assert sv.get_value() == "abc"
