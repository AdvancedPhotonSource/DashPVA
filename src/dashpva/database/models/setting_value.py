# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import json as _json
from typing import Any, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from dashpva.database.db import Base

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
