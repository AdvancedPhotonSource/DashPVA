import json
import toml
from typing import List, Optional, Dict, Any
from database.db import get_session
from database.models.profile import Profile, ProfileConfig


class ProfileManager:

    # ------------------------------------------------------------------ #
    # Profiles CRUD
    # ------------------------------------------------------------------ #

    def create_profile(self, name: str, description: Optional[str] = None) -> Optional[Profile]:
        session = get_session()
        try:
            profile = Profile(name=name, description=description)
            session.add(profile)
            session.commit()
            session.refresh(profile)
            return profile
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_all_profiles(self) -> List[Profile]:
        session = get_session()
        try:
            return session.query(Profile).order_by(Profile.id).all()
        except Exception:
            return []
        finally:
            session.close()

    def get_profile_by_id(self, profile_id: int) -> Optional[Profile]:
        session = get_session()
        try:
            return session.query(Profile).filter_by(id=profile_id).first()
        except Exception:
            return None
        finally:
            session.close()

    def get_profile_by_name(self, name: str) -> Optional[Profile]:
        session = get_session()
        try:
            return session.query(Profile).filter_by(name=name).first()
        except Exception:
            return None
        finally:
            session.close()

    def update_profile_name(self, profile_id: int, new_name: str) -> bool:
        session = get_session()
        try:
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                return False
            profile.name = new_name
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def update_profile_description(self, profile_id: int, description: str) -> bool:
        session = get_session()
        try:
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                return False
            profile.description = description
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def delete_profile(self, profile_id: int) -> bool:
        session = get_session()
        try:
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                return False
            session.delete(profile)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def profile_exists(self, name: str) -> bool:
        session = get_session()
        try:
            return session.query(Profile).filter_by(name=name).count() > 0
        except Exception:
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Selected / Default flags
    # ------------------------------------------------------------------ #

    def set_selected_profile(self, profile_id: int) -> bool:
        session = get_session()
        try:
            session.query(Profile).update({Profile.is_selected: False})
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                session.rollback()
                return False
            profile.is_selected = True
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def clear_selected_profiles(self) -> bool:
        session = get_session()
        try:
            session.query(Profile).update({Profile.is_selected: False})
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_selected_profile(self) -> Optional[Profile]:
        session = get_session()
        try:
            return session.query(Profile).filter_by(is_selected=True).first()
        except Exception:
            return None
        finally:
            session.close()

    def set_default_profile(self, profile_id: int) -> bool:
        session = get_session()
        try:
            session.query(Profile).update({Profile.is_default: False})
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                session.rollback()
                return False
            profile.is_default = True
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def unset_default_profile(self, profile_id: int) -> bool:
        session = get_session()
        try:
            profile = session.query(Profile).filter_by(id=profile_id).first()
            if not profile:
                return False
            profile.is_default = False
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_default_profile(self) -> Optional[Profile]:
        session = get_session()
        try:
            return session.query(Profile).filter_by(is_default=True).first()
        except Exception:
            return None
        finally:
            session.close()

    def any_default_exists(self) -> bool:
        session = get_session()
        try:
            return session.query(Profile).filter_by(is_default=True).count() > 0
        except Exception:
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Configuration entries
    # ------------------------------------------------------------------ #

    def add_profile_config(
        self,
        profile_id: int,
        config_type: str,
        config_key: str,
        config_value: str,
        config_section: Optional[str] = None,
    ) -> bool:
        session = get_session()
        try:
            config = ProfileConfig(
                profile_id=profile_id,
                config_type=config_type,
                config_section=config_section,
                config_key=config_key,
                config_value=str(config_value),
            )
            session.add(config)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_profile_configs(self, profile_id: int, config_type: Optional[str] = None) -> List[ProfileConfig]:
        session = get_session()
        try:
            q = session.query(ProfileConfig).filter_by(profile_id=profile_id)
            if config_type:
                q = q.filter_by(config_type=config_type)
            # Exclude internal JSON blob records
            q = q.filter(ProfileConfig.config_type != '__toml__')
            return q.all()
        except Exception:
            return []
        finally:
            session.close()

    def clear_profile_configs(self, profile_id: int) -> bool:
        session = get_session()
        try:
            session.query(ProfileConfig).filter_by(profile_id=profile_id).delete()
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def update_config_value(self, config_id: int, new_value: str) -> bool:
        session = get_session()
        try:
            config = session.query(ProfileConfig).filter_by(id=config_id).first()
            if not config:
                return False
            config.config_value = str(new_value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def delete_config_entry(self, config_id: int) -> bool:
        session = get_session()
        try:
            config = session.query(ProfileConfig).filter_by(id=config_id).first()
            if not config:
                return False
            session.delete(config)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def rename_config_type(self, profile_id: int, old_type: str, new_type: str) -> bool:
        session = get_session()
        try:
            session.query(ProfileConfig).filter_by(
                profile_id=profile_id, config_type=old_type
            ).update({ProfileConfig.config_type: new_type})
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    # ------------------------------------------------------------------ #
    # Import / Export TOML
    # ------------------------------------------------------------------ #

    def import_toml_to_profile(self, profile_id: int, toml_data: Dict[str, Any]) -> bool:
        """Store the full TOML dict as a JSON blob for reliable round-trip export."""
        session = get_session()
        try:
            # Remove any existing JSON blob for this profile
            session.query(ProfileConfig).filter_by(
                profile_id=profile_id, config_type='__toml__', config_key='__data__'
            ).delete()
            blob = ProfileConfig(
                profile_id=profile_id,
                config_type='__toml__',
                config_section=None,
                config_key='__data__',
                config_value=json.dumps(toml_data),
            )
            session.add(blob)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def import_toml_file(self, profile_id: int, toml_file_path: str) -> bool:
        try:
            with open(toml_file_path, 'r') as f:
                data = toml.load(f)
            return self.import_toml_to_profile(profile_id, data)
        except Exception:
            return False

    def export_profile_to_toml(self, profile_id: int) -> Dict[str, Any]:
        """Return the stored TOML dict for this profile (empty dict if nothing stored)."""
        session = get_session()
        try:
            blob = session.query(ProfileConfig).filter_by(
                profile_id=profile_id, config_type='__toml__', config_key='__data__'
            ).first()
            if blob:
                return json.loads(blob.config_value)
            return {}
        except Exception:
            return {}
        finally:
            session.close()

    def export_profile_to_toml_file(self, profile_id: int, output_path: str) -> bool:
        try:
            data = self.export_profile_to_toml(profile_id)
            with open(output_path, 'w') as f:
                toml.dump(data, f)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Defaults / Seeding
    # ------------------------------------------------------------------ #

    def ensure_shipped_default_profile(
        self, toml_file_path: str, name: str = 'device:metadata:default'
    ) -> Optional[Profile]:
        existing = self.get_profile_by_name(name)
        if existing:
            return existing
        profile = self.create_profile(name, description='Shipped default profile')
        if profile and toml_file_path:
            self.import_toml_file(profile.id, toml_file_path)
        return profile

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def clone_profile_configs(self, source_profile_id: int, dest_profile_id: int) -> bool:
        session = get_session()
        try:
            source_configs = session.query(ProfileConfig).filter_by(
                profile_id=source_profile_id
            ).all()
            for cfg in source_configs:
                clone = ProfileConfig(
                    profile_id=dest_profile_id,
                    config_type=cfg.config_type,
                    config_section=cfg.config_section,
                    config_key=cfg.config_key,
                    config_value=cfg.config_value,
                )
                session.add(clone)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()
