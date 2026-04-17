from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.db import Base


class Profile(Base):
    __tablename__ = 'profiles'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    is_default = Column(Boolean, nullable=False, default=False)
    is_selected = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    configs = relationship('ProfileConfig', back_populates='profile', cascade='all, delete-orphan')


class ProfileConfig(Base):
    __tablename__ = 'profile_configs'

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey('profiles.id'), nullable=False)
    config_type = Column(String(100), nullable=False)
    config_section = Column(String(255), nullable=True)
    config_key = Column(String(255), nullable=False)
    config_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship('Profile', back_populates='configs')
