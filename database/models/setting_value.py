from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base
from typing import Union, Optional

class SettingValue(Base):
    __tablename__ = 'setting_values'

    id = Column(Integer, primary_key=True, autoincrement=True)
    setting_id = Column(Integer, ForeignKey('settings.id'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)  # Store as string, convert as needed
    value_type = Column(String(20), nullable=False)  # 'string' or 'int'
    
    # Relationship back to the parent setting
    setting = relationship("Settings", back_populates="values")
    
    def set_value(self, value: Union[str, int]) -> None:
        """Set the value and automatically determine the type."""
        if isinstance(value, int):
            self.value = str(value)
            self.value_type = 'int'
        else:
            self.value = str(value)
            self.value_type = 'string'
    
    def get_value(self) -> Union[str, int]:
        """Get the value with proper type conversion."""
        if self.value_type == 'int':
            try:
                return int(self.value)
            except ValueError:
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
