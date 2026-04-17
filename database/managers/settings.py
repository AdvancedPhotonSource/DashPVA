from typing import List, Optional, Union, Dict, Any
from sqlalchemy.orm import Session, selectinload
from database.db import get_session
from database.models.settings import Settings
from database.models.setting_value import SettingValue

# Eager-load options reused across read methods so callers never hit
# DetachedInstanceError when accessing .values or .children after session close.
_SETTING_OPTS = [selectinload(Settings.values), selectinload(Settings.children)]

class SettingsManager:
    """
    CRUD operations for the Settings table with individual setting values.
    """
    def __init__(self) -> None:
        pass

    def _session(self) -> Session:
        return get_session()

    # Create
    def create_setting(self, name: str, type_: str, desc: Optional[str] = None, parent_id: Optional[int] = None) -> Optional[Settings]:
        session = self._session()
        try:
            # Check for existing setting with same name and parent
            existing = (
                session.query(Settings)
                .filter(Settings.name == name, Settings.parent_id == parent_id)
                .options(*_SETTING_OPTS)
                .first()
            )
            if existing:
                session.expunge_all()
                return existing

            obj = Settings(name=name, type=type_, desc=desc or "", parent_id=parent_id)
            session.add(obj)
            session.commit()
            session.refresh(obj)
            session.expunge_all()
            return obj
        except Exception:
            session.rollback()
            return None
        finally:
            session.close()

    def create_child_setting(self, parent_id: int, name: str, type_: str, desc: Optional[str] = None) -> Optional[Settings]:
        """Create a child setting under a parent setting."""
        return self.create_setting(name, type_, desc, parent_id)

    # Read
    def get_all_settings(self) -> List[Settings]:
        session = self._session()
        try:
            objs = (
                session.query(Settings)
                .options(*_SETTING_OPTS)
                .order_by(Settings.type, Settings.name)
                .all()
            )
            session.expunge_all()
            return objs
        finally:
            session.close()

    def get_settings_by_type(self, type_: str) -> List[Settings]:
        session = self._session()
        try:
            objs = (
                session.query(Settings)
                .filter(Settings.type == type_)
                .options(*_SETTING_OPTS)
                .order_by(Settings.name)
                .all()
            )
            session.expunge_all()
            return objs
        finally:
            session.close()

    def get_distinct_types(self) -> List[str]:
        session = self._session()
        try:
            rows = session.query(Settings.type).distinct().all()
            return sorted([t[0] for t in rows if t and t[0]])
        finally:
            session.close()

    def get_setting_by_name(self, name: str) -> Optional[Settings]:
        session = self._session()
        try:
            obj = (
                session.query(Settings)
                .filter(Settings.name == name)
                .options(*_SETTING_OPTS)
                .first()
            )
            if obj:
                session.expunge_all()
            return obj
        finally:
            session.close()

    def get_setting_by_id(self, id_: int) -> Optional[Settings]:
        session = self._session()
        try:
            obj = (
                session.query(Settings)
                .filter(Settings.id == id_)
                .options(*_SETTING_OPTS)
                .first()
            )
            if obj:
                session.expunge_all()
            return obj
        finally:
            session.close()

    # Update
    def update_setting_desc(self, id_: int, desc: str) -> bool:
        session = self._session()
        try:
            obj = session.query(Settings).filter(Settings.id == id_).first()
            if not obj:
                return False
            obj.desc = desc
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def update_setting(self, id_: int, name: str, type_: str) -> bool:
        session = self._session()
        try:
            obj = session.query(Settings).filter(Settings.id == id_).first()
            if not obj:
                return False
            obj.name = name
            obj.type = type_
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    # Setting Value operations
    def add_setting_value(self, setting_id: int, key: str, value, value_type: Optional[str] = None) -> bool:
        """Add a new key-value pair to a setting (no-op if key already exists)."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.id == setting_id).first()
            if not setting:
                return False

            # Issue 6: dedup on (setting_id, key) — consistent with seed script
            existing = session.query(SettingValue).filter(
                SettingValue.setting_id == setting_id,
                SettingValue.key == key,
            ).first()
            if existing:
                return True

            setting_value = SettingValue(setting_id=setting_id, key=key)
            setting_value.set_value(value, value_type)
            session.add(setting_value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def add_setting_value_by_name(self, setting_name: str, key: str, value: Union[str, int]) -> bool:
        """Add a new key-value pair to a setting by setting name."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.name == setting_name).first()
            if not setting:
                return False
            
            setting_value = SettingValue(setting_id=setting.id, key=key)
            setting_value.set_value(value)
            session.add(setting_value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def update_setting_value(self, setting_id: int, key: str, value, value_type: Optional[str] = None) -> bool:
        """Update an existing setting value."""
        session = self._session()
        try:
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting_id,
                SettingValue.key == key
            ).first()

            if not setting_value:
                return False

            setting_value.set_value(value, value_type)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def update_setting_value_by_name(self, setting_name: str, key: str, value: Union[str, int]) -> bool:
        """Update an existing setting value by setting name."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.name == setting_name).first()
            if not setting:
                return False
            
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting.id,
                SettingValue.key == key
            ).first()
            
            if not setting_value:
                return False
            
            setting_value.set_value(value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_setting_value(self, setting_id: int, key: str) -> Optional[Union[str, int]]:
        """Get a specific setting value."""
        session = self._session()
        try:
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting_id,
                SettingValue.key == key
            ).first()
            
            if not setting_value:
                return None
            
            return setting_value.get_value()
        finally:
            session.close()

    def get_setting_value_by_name(self, setting_name: str, key: str) -> Optional[Union[str, int]]:
        """Get a specific setting value by setting name."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.name == setting_name).first()
            if not setting:
                return None
            
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting.id,
                SettingValue.key == key
            ).first()
            
            if not setting_value:
                return None
            
            return setting_value.get_value()
        finally:
            session.close()

    def remove_setting_value(self, setting_id: int, key: str) -> bool:
        """Remove a setting value."""
        session = self._session()
        try:
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting_id,
                SettingValue.key == key
            ).first()
            
            if not setting_value:
                return False
            
            session.delete(setting_value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def remove_setting_value_by_name(self, setting_name: str, key: str) -> bool:
        """Remove a setting value by setting name."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.name == setting_name).first()
            if not setting:
                return False
            
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting.id,
                SettingValue.key == key
            ).first()
            
            if not setting_value:
                return False
            
            session.delete(setting_value)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    def get_all_setting_values(self, setting_id: int) -> Dict[str, Union[str, int]]:
        """Get all values for a setting as a dictionary."""
        session = self._session()
        try:
            setting = (
                session.query(Settings)
                .filter(Settings.id == setting_id)
                .options(selectinload(Settings.values))
                .first()
            )
            if not setting:
                return {}
            return setting.get_all_values()
        finally:
            session.close()

    def get_all_setting_values_with_type(self, setting_id: int) -> List[tuple]:
        """Return (key, value, value_type) tuples for a setting."""
        session = self._session()
        try:
            setting = (
                session.query(Settings)
                .filter(Settings.id == setting_id)
                .options(selectinload(Settings.values))
                .first()
            )
            if not setting:
                return []
            return [(sv.key, sv.get_value(), sv.value_type) for sv in setting.values]
        finally:
            session.close()

    def get_all_setting_values_by_name(self, setting_name: str) -> Dict[str, Union[str, int]]:
        """Get all values for a setting by name as a dictionary."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.name == setting_name).first()
            if not setting:
                return {}
            return setting.get_all_values()
        finally:
            session.close()

    # Hierarchical operations
    def get_root_settings(self) -> List[Settings]:
        """Get all root settings (settings with no parent)."""
        session = self._session()
        try:
            # Load all so that children references across objects are populated
            all_objs = session.query(Settings).options(*_SETTING_OPTS).all()
            roots = [o for o in all_objs if o.parent_id is None]
            session.expunge_all()
            return roots
        finally:
            session.close()

    def get_children(self, parent_id: int) -> List[Settings]:
        """Get all direct children of a setting."""
        session = self._session()
        try:
            objs = (
                session.query(Settings)
                .filter(Settings.parent_id == parent_id)
                .options(*_SETTING_OPTS)
                .order_by(Settings.name)
                .all()
            )
            session.expunge_all()
            return objs
        finally:
            session.close()

    def get_setting_tree(self) -> List[Settings]:
        """Get all settings organised as a tree structure."""
        session = self._session()
        try:
            # Load everything with values eager-loaded; then touch .children for
            # every node so SQLAlchemy populates the InstrumentedLists from the
            # identity map before we expunge — no extra round-trips needed.
            all_settings = (
                session.query(Settings)
                .options(selectinload(Settings.values))
                .all()
            )
            for s in all_settings:
                _ = s.children  # populate from identity map while session is live
            roots = [s for s in all_settings if s.parent_id is None]
            session.expunge_all()
            return roots
        finally:
            session.close()

    def get_setting_by_path(self, path: List[str]) -> Optional[Settings]:
        """Get a setting by its hierarchical path."""
        if not path:
            return None

        session = self._session()
        try:
            current = None
            for name in path:
                if current is None:
                    current = (
                        session.query(Settings)
                        .filter(Settings.name == name, Settings.parent_id.is_(None))
                        .options(*_SETTING_OPTS)
                        .first()
                    )
                else:
                    current = (
                        session.query(Settings)
                        .filter(Settings.name == name, Settings.parent_id == current.id)
                        .options(*_SETTING_OPTS)
                        .first()
                    )
                if current is None:
                    return None
            if current:
                session.expunge_all()
            return current
        finally:
            session.close()

    def move_setting(self, setting_id: int, new_parent_id: Optional[int]) -> bool:
        """Move a setting to a new parent (or make it root if new_parent_id is None)."""
        session = self._session()
        try:
            setting = session.query(Settings).filter(Settings.id == setting_id).first()
            if not setting:
                return False

            if new_parent_id is not None:
                parent = session.query(Settings).filter(Settings.id == new_parent_id).first()
                if not parent:
                    return False

                # Check for circular reference
                current = parent
                while current:
                    if current.id == setting_id:
                        return False
                    current = current.parent

            setting.parent_id = new_parent_id
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()

    # Delete
    def delete_setting(self, id_: int) -> bool:
        session = self._session()
        try:
            obj = session.query(Settings).filter(Settings.id == id_).first()
            if not obj:
                return False
            session.delete(obj)
            session.commit()
            return True
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()
