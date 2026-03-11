"""
DatabaseInterface: Public facade for all database operations in DashPVA.

This interface centralizes profile and configuration management behind a single
import surface. External modules (GUIs, config repository, tests) should depend
on this interface rather than internal SQLAlchemy manager/models.

Usage:
    from database import DatabaseInterface

    db = DatabaseInterface()
    profiles = db.get_all_profiles()
    prof = db.create_profile("my_profile", "Example")
    db.import_toml_file(prof.id, "pv_configs/metadata_pvs.toml")
    cfg_dict = db.export_profile_to_toml(prof.id)

Notes:
- Initializes the SQLite database on construction.
- Wraps internal ProfileManager methods with a stable, GUI/service-friendly API.
"""

from typing import List, Optional, Dict, Any, Union
from database.db import init_database, create_tables
from database.managers.settings import SettingsManager

# NOTE: profile model/manager not yet implemented — ProfileManager and Profile
# are unavailable until database/models/profile.py and database/managers/profile.py
# are added.
Profile = None
ProfileConfig = None
ProfileManager = None


class DatabaseInterface:
    """Facade over ProfileManager providing a stable public API."""

    def __init__(self) -> None:
        # Ensure DB file/tables exist
        init_database()
        # Ensure any new tables are created (e.g., 'settings')
        try:
            create_tables()
        except Exception:
            pass
        # Internal manager implementation
        self._mgr = ProfileManager()
        # Settings manager (simple name/type/desc)
        self._settings_mgr = SettingsManager()

    # Profiles CRUD

    def create_profile(self, name: str, description: Optional[str] = None) -> Optional[Profile]:
        return self._mgr.create_profile(name, description)

    def get_all_profiles(self) -> List[Profile]:
        return self._mgr.get_all_profiles()

    def get_profile_by_id(self, profile_id: int) -> Optional[Profile]:
        return self._mgr.get_profile_by_id(profile_id)

    def get_profile_by_name(self, name: str) -> Optional[Profile]:
        return self._mgr.get_profile_by_name(name)

    def get_selected_profile(self) -> Optional[Profile]:
        return self._mgr.get_selected_profile()

    def update_profile_name(self, profile_id: int, new_name: str) -> bool:
        return self._mgr.update_profile_name(profile_id, new_name)

    def update_profile_description(self, profile_id: int, description: str) -> bool:
        return self._mgr.update_profile_description(profile_id, description)

    def delete_profile(self, profile_id: int) -> bool:
        return self._mgr.delete_profile(profile_id)
    
    # Detector CRUD
    #

    # System CRUD

    # Selected / Default flags

    def set_selected_profile(self, profile_id: int) -> bool:
        return self._mgr.set_selected_profile(profile_id)

    def clear_selected_profiles(self) -> bool:
        return self._mgr.clear_selected_profiles()

    def get_selected_profile(self) -> Optional[Profile]:
        return self._mgr.get_selected_profile()

    def set_default_profile(self, profile_id: int) -> bool:
        return self._mgr.set_default_profile(profile_id)

    def unset_default_profile(self, profile_id: int) -> bool:
        return self._mgr.unset_default_profile(profile_id)

    def get_default_profile(self) -> Optional[Profile]:
        return self._mgr.get_default_profile()

    def any_default_exists(self) -> bool:
        return self._mgr.any_default_exists()

    def profile_exists(self, name: str) -> bool:
        return self._mgr.profile_exists(name)

    # Configuration entries

    def add_profile_config(
        self,
        profile_id: int,
        config_type: str,
        config_key: str,
        config_value: str,
        config_section: Optional[str] = None,
    ) -> bool:
        return self._mgr.add_profile_config(profile_id, config_type, config_key, config_value, config_section)

    def get_profile_configs(self, profile_id: int, config_type: Optional[str] = None) -> List[ProfileConfig]:
        return self._mgr.get_profile_configs(profile_id, config_type)

    def clear_profile_configs(self, profile_id: int) -> bool:
        return self._mgr.clear_profile_configs(profile_id)

    def update_config_value(self, config_id: int, new_value: str) -> bool:
        return self._mgr.update_config_value(config_id, new_value)

    def delete_config_entry(self, config_id: int) -> bool:
        return self._mgr.delete_config_entry(config_id)

    def rename_config_type(self, profile_id: int, old_type: str, new_type: str) -> bool:
        return self._mgr.rename_config_type(profile_id, old_type, new_type)

    # Import / Export TOML

    def import_toml_to_profile(self, profile_id: int, toml_data: Dict[str, Any]) -> bool:
        return self._mgr.import_toml_to_profile(profile_id, toml_data)

    def import_toml_file(self, profile_id: int, toml_file_path: str) -> bool:
        return self._mgr.import_toml_file(profile_id, toml_file_path)

    def export_profile_to_toml(self, profile_id: int) -> Dict[str, Any]:
        return self._mgr.export_profile_to_toml(profile_id)

    def export_profile_to_toml_file(self, profile_id: int, output_path: str) -> bool:
        return self._mgr.export_profile_to_toml_file(profile_id, output_path)

    # Defaults / Seeding

    def ensure_shipped_default_profile(self, toml_file_path: str, name: str = "device:metadata:default") -> Optional[Profile]:
        return self._mgr.ensure_shipped_default_profile(toml_file_path, name)

    def seed_system_defaults_from_toml(self, toml_file_path: str, name: str = "device:metadata:default") -> bool:
        """
        Convenience wrapper used by UI setup to ensure a default profile seeded from TOML.

        Returns True if the default profile exists or was created successfully.
        """
        prof = self.ensure_shipped_default_profile(toml_file_path, name=name)
        return prof is not None

    # Utilities

    def clone_profile_configs(self, source_profile_id: int, dest_profile_id: int) -> bool:
        return self._mgr.clone_profile_configs(source_profile_id, dest_profile_id)

    # Settings CRUD wrappers with individual setting values
    def create_setting(self, name: str, type_: str, desc: Optional[str] = None, parent_id: Optional[int] = None):
        return self._settings_mgr.create_setting(name, type_, desc, parent_id)

    def create_child_setting(self, parent_id: int, name: str, type_: str, desc: Optional[str] = None):
        return self._settings_mgr.create_child_setting(parent_id, name, type_, desc)

    def get_all_settings(self):
        return self._settings_mgr.get_all_settings()

    def get_settings_by_type(self, type_: str):
        return self._settings_mgr.get_settings_by_type(type_)

    def get_setting_by_name(self, name: str):
        return self._settings_mgr.get_setting_by_name(name)

    def get_setting_by_id(self, id_: int):
        return self._settings_mgr.get_setting_by_id(id_)

    def get_distinct_setting_types(self):
        return self._settings_mgr.get_distinct_types()

    def update_setting_desc(self, id_: int, desc: str) -> bool:
        return self._settings_mgr.update_setting_desc(id_, desc)

    def delete_setting(self, id_: int) -> bool:
        return self._settings_mgr.delete_setting(id_)

    # Setting Value operations
    def add_setting_value(self, setting_id: int, key: str, value: Union[str, int]) -> bool:
        return self._settings_mgr.add_setting_value(setting_id, key, value)

    def add_setting_value_by_name(self, setting_name: str, key: str, value: Union[str, int]) -> bool:
        return self._settings_mgr.add_setting_value_by_name(setting_name, key, value)

    def update_setting_value(self, setting_id: int, key: str, value: Union[str, int]) -> bool:
        return self._settings_mgr.update_setting_value(setting_id, key, value)

    def update_setting_value_by_name(self, setting_name: str, key: str, value: Union[str, int]) -> bool:
        return self._settings_mgr.update_setting_value_by_name(setting_name, key, value)

    def get_setting_value(self, setting_id: int, key: str) -> Optional[Union[str, int]]:
        return self._settings_mgr.get_setting_value(setting_id, key)

    def get_setting_value_by_name(self, setting_name: str, key: str) -> Optional[Union[str, int]]:
        return self._settings_mgr.get_setting_value_by_name(setting_name, key)

    def remove_setting_value(self, setting_id: int, key: str) -> bool:
        return self._settings_mgr.remove_setting_value(setting_id, key)

    def remove_setting_value_by_name(self, setting_name: str, key: str) -> bool:
        return self._settings_mgr.remove_setting_value_by_name(setting_name, key)

    def get_all_setting_values(self, setting_id: int) -> Dict[str, Union[str, int]]:
        return self._settings_mgr.get_all_setting_values(setting_id)

    def get_all_setting_values_by_name(self, setting_name: str) -> Dict[str, Union[str, int]]:
        return self._settings_mgr.get_all_setting_values_by_name(setting_name)

    # Hierarchical settings operations
    def get_root_settings(self):
        return self._settings_mgr.get_root_settings()

    def get_setting_children(self, parent_id: int):
        return self._settings_mgr.get_children(parent_id)

    def get_setting_tree(self):
        return self._settings_mgr.get_setting_tree()

    def get_setting_by_path(self, path: List[str]):
        return self._settings_mgr.get_setting_by_path(path)

    def move_setting(self, setting_id: int, new_parent_id: Optional[int]) -> bool:
        return self._settings_mgr.move_setting(setting_id, new_parent_id)
