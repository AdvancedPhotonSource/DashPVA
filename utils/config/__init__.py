"""
Configuration repository and sources for DashPVA settings.

This package provides a consistent abstraction for loading/saving configuration
data from different backends (TOML files and database profiles).
"""

from .interfaces import ConfigSource
from .repository import ConfigRepository

__all__ = [
    "ConfigSource",
    "ConfigRepository",
]
