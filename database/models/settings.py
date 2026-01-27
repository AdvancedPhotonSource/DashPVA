from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base
from typing import List, Dict, Any, Union, Optional

class Settings(Base):
    __tablename__ = 'settings'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    type = Column(String(100), nullable=False)
    desc = Column(Text, nullable=True)
    parent_id = Column(Integer, ForeignKey('settings.id'), nullable=True)
    
    # Self-referential relationship for hierarchy
    parent = relationship("Settings", remote_side=[id], back_populates="children")
    children = relationship("Settings", back_populates="parent", cascade="all, delete-orphan")
    
    # Relationship to setting values
    values = relationship("SettingValue", back_populates="setting", cascade="all, delete-orphan")
    
    def add_value(self, key: str, value: Union[str, int]) -> None:
        """Add a new key-value pair to this setting."""
        from database.models.setting_value import SettingValue
        setting_value = SettingValue(key=key, setting_id=self.id)
        setting_value.set_value(value)
        self.values.append(setting_value)
    
    def get_value(self, key: str) -> Optional[Union[str, int]]:
        """Get a specific value by key."""
        for setting_value in self.values:
            if setting_value.key == key:
                return setting_value.get_value()
        return None
    
    def update_value(self, key: str, value: Union[str, int]) -> bool:
        """Update an existing value by key."""
        for setting_value in self.values:
            if setting_value.key == key:
                setting_value.set_value(value)
                return True
        return False
    
    def remove_value(self, key: str) -> bool:
        """Remove a value by key."""
        for setting_value in self.values:
            if setting_value.key == key:
                self.values.remove(setting_value)
                return True
        return False
    
    def get_all_values(self) -> Dict[str, Union[str, int]]:
        """Get all key-value pairs as a dictionary."""
        return {sv.key: sv.get_value() for sv in self.values}
    
    def get_values_list(self) -> List[Dict[str, Any]]:
        """Get all values as a list of dictionaries."""
        return [sv.to_dict() for sv in self.values]
    
    def add_child(self, name: str, type_: str, desc: Optional[str] = None) -> 'Settings':
        """Add a child setting to this setting."""
        child = Settings(name=name, type=type_, desc=desc, parent_id=self.id)
        self.children.append(child)
        return child
    
    def get_path(self) -> List[str]:
        """Get the full path from root to this setting."""
        path = []
        current = self
        while current:
            path.insert(0, current.name)
            current = current.parent
        return path
    
    def get_full_path(self) -> str:
        """Get the full path as a string separated by '/'."""
        return '/'.join(self.get_path())
    
    def is_root(self) -> bool:
        """Check if this is a root setting (no parent)."""
        return self.parent_id is None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf setting (has values but no children)."""
        return len(self.children) == 0 and len(self.values) > 0
    
    def is_container(self) -> bool:
        """Check if this is a container setting (has children)."""
        return len(self.children) > 0
    
    def get_descendants(self) -> List['Settings']:
        """Get all descendant settings recursively."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'desc': self.desc,
            'parent_id': self.parent_id,
            'values': self.get_all_values(),
            'children': [child.to_dict() for child in self.children]
        }
    
    def to_tree_dict(self) -> Dict[str, Any]:
        """Convert to a nested dictionary structure for tree display."""
        result = {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'desc': self.desc,
            'values': self.get_all_values()
        }
        
        if self.children:
            result['children'] = {child.name: child.to_tree_dict() for child in self.children}
        
        return result
