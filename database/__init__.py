"""
Database package public API.

Importing `database` exposes the DatabaseInterface facade, which is the single
public entry point for all database operations (profiles, configs, import/export).
Internal modules (manager, models, migrations) are implementation details.

Usage:
    from database import DatabaseInterface
    db = DatabaseInterface()
"""

from .interface import DatabaseInterface

__all__ = ["DatabaseInterface"]
