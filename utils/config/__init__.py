"""
Configuration repository and sources for DashPVA settings.

This package provides a consistent abstraction for loading/saving configuration
data from different backends (TOML files and database profiles).
"""

from .source import ConfigSource

__all__ = ["ConfigSource"]
