from typing import List, Optional, Union, Dict, Any
from sqlalchemy.orm import Session
from database.db import get_session
from database.models.settings import Settings
from database.models.setting_value import SettingValue

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
            existing = session.query(Settings).filter(
                Settings.name == name,
                Settings.parent_id == parent_id
            ).first()
            if existing:
                return existing
            
            obj = Settings(name=name, type=type_, desc=desc or "", parent_id=parent_id)
            session.add(obj)
            session.commit()
            session.refresh(obj)
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
            return session.query(Settings).order_by(Settings.type, Settings.name).all()
        finally:
            session.close()

    def get_settings_by_type(self, type_: str) -> List[Settings]:
        session = self._session()
        try:
            return (
                session.query(Settings)
                .filter(Settings.type == type_)
                .order_by(Settings.name)
                .all()
            )
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
            return session.query(Settings).filter(Settings.name == name).first()
        finally:
            session.close()

    def get_setting_by_id(self, id_: int) -> Optional[Settings]:
        session = self._session()
        try:
            return session.query(Settings).get(id_)
        finally:
            session.close()

    # Update
    def update_setting_desc(self, id_: int, desc: str) -> bool:
        session = self._session()
        try:
            obj = session.query(Settings).get(id_)
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

    # Setting Value operations
    def add_setting_value(self, setting_id: int, key: str, value: Union[str, int]) -> bool:
        """Add a new key-value pair to a setting."""
        session = self._session()
        try:
            setting = session.query(Settings).get(setting_id)
            if not setting:
                return False
            
            setting_value = SettingValue(setting_id=setting_id, key=key)
            setting_value.set_value(value)
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

    def update_setting_value(self, setting_id: int, key: str, value: Union[str, int]) -> bool:
        """Update an existing setting value."""
        session = self._session()
        try:
            setting_value = session.query(SettingValue).filter(
                SettingValue.setting_id == setting_id,
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
            setting = session.query(Settings).get(setting_id)
            if not setting:
                return {}
            return setting.get_all_values()
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
            return session.query(Settings).filter(Settings.parent_id.is_(None)).order_by(Settings.type, Settings.name).all()
        finally:
            session.close()

    def get_children(self, parent_id: int) -> List[Settings]:
        """Get all direct children of a setting."""
        session = self._session()
        try:
            return session.query(Settings).filter(Settings.parent_id == parent_id).order_by(Settings.name).all()
        finally:
            session.close()

    def get_setting_tree(self) -> List[Settings]:
        """Get all settings organized as a tree structure."""
        session = self._session()
        try:
            # Get all settings with their relationships loaded
            settings = session.query(Settings).all()
            # Return only root settings - children will be accessible via relationships
            return [s for s in settings if s.parent_id is None]
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
                    # Looking for root setting
                    current = session.query(Settings).filter(
                        Settings.name == name,
                        Settings.parent_id.is_(None)
                    ).first()
                else:
                    # Looking for child setting
                    current = session.query(Settings).filter(
                        Settings.name == name,
                        Settings.parent_id == current.id
                    ).first()
                
                if current is None:
                    return None
            
            return current
        finally:
            session.close()

    def move_setting(self, setting_id: int, new_parent_id: Optional[int]) -> bool:
        """Move a setting to a new parent (or make it root if new_parent_id is None)."""
        session = self._session()
        try:
            setting = session.query(Settings).get(setting_id)
            if not setting:
                return False
            
            # Check for circular reference
            if new_parent_id is not None:
                parent = session.query(Settings).get(new_parent_id)
                if not parent:
                    return False
                
                # Check if new parent is a descendant of this setting
                current = parent
                while current:
                    if current.id == setting_id:
                        return False  # Would create circular reference
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
            obj = session.query(Settings).get(id_)
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
