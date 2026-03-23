import json as _json
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base
from typing import Union, Optional, Any

VALUE_TYPES = ('string', 'int', 'float', 'json')

class SettingValue(Base):
    __tablename__ = 'setting_values'

    id = Column(Integer, primary_key=True, autoincrement=True)
    setting_id = Column(Integer, ForeignKey('settings.id'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)  # Store as string, convert as needed
    value_type = Column(String(20), nullable=False)  # 'string', 'int', 'float', or 'json'

    # Relationship back to the parent setting
    setting = relationship("Settings", back_populates="values")

    def set_value(self, value: Any, value_type: Optional[str] = None) -> None:
        """Set the value. If value_type is given it is used directly; otherwise auto-detected."""
        if value_type and value_type in VALUE_TYPES:
            self.value_type = value_type
            self.value = _json.dumps(value) if value_type == 'json' else str(value)
        elif isinstance(value, bool):
            self.value = str(value)
            self.value_type = 'string'
        elif isinstance(value, int):
            self.value = str(value)
            self.value_type = 'int'
        elif isinstance(value, float):
            self.value = str(value)
            self.value_type = 'float'
        elif isinstance(value, (dict, list)):
            self.value = _json.dumps(value)
            self.value_type = 'json'
        else:
            self.value = str(value)
            self.value_type = 'string'

    def get_value(self) -> Any:
        """Get the value with proper type conversion."""
        if self.value_type == 'int':
            try:
                return int(self.value)
            except (ValueError, TypeError):
                return self.value
        if self.value_type == 'float':
            try:
                return float(self.value)
            except (ValueError, TypeError):
                return self.value
        if self.value_type == 'json':
            try:
                return _json.loads(self.value)
            except (ValueError, TypeError):
                return self.value
        return self.value
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'setting_id': self.setting_id,
            'key': self.key,
            'value': self.get_value(),
            'value_type': self.value_type
        }
