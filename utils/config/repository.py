from __future__ import annotations
from typing import Optional, Union, Dict, Any
import os
import tempfile
import toml

from .interfaces import ConfigSource
from .sources import TomlConfigSource, DbProfileConfigSource
from typing import Any


class ConfigRepository:
    """
    Resolve and load configuration from either a TOML path or a DB profile.

    Locator can be:
      - str path to a TOML file
      - int profile id
      - "profile:<name>" string
    """

    def __init__(self, db: Any = None) -> None:
        self.db = db

    def resolve(self, locator: Optional[Union[int, str]]) -> Optional[ConfigSource]:
        if locator is None:
            # When no explicit locator is provided, prefer the currently selected DB profile
            if self.db:
                try:
                    sel = self.db.get_selected_profile()
                except Exception:
                    sel = None
                if sel is not None:
                    return DbProfileConfigSource(self.db, sel.id)
            return None

        # TOML path
        if isinstance(locator, str) and (locator.endswith('.toml') or os.path.exists(locator)):
            return TomlConfigSource(locator)

        # DB profile by id or name
        if isinstance(locator, int) or (isinstance(locator, str) and locator.startswith('profile:')):
            return DbProfileConfigSource(self.db, locator)

        # If a plain string name is given, treat as profile name
        if isinstance(locator, str):
            return DbProfileConfigSource(self.db, f"profile:{locator}")

        return None

    def load(self, locator: Optional[Union[int, str]]) -> Dict[str, Any]:
        src = self.resolve(locator)
        return src.load() if src else {}

    def ensure_path(self, locator: Optional[Union[int, str]]) -> Optional[str]:
        """
        Ensure a TOML file path is available for the current source.
          - TOML: return original path
          - DB: export to a temporary TOML and return its path
        """
        src = self.resolve(locator)
        if src is None:
            return None
        # If it's a TOML source with a 'path', return it
        if getattr(src, 'source_type', None) == 'toml' and hasattr(src, 'path'):
            return getattr(src, 'path')

        # For DB-backed source, export to a temp TOML
        try:
            data = src.load()
            fd, tmp = tempfile.mkstemp(suffix='.toml', prefix='dashpva_')
            os.close(fd)
            with open(tmp, 'w') as f:
                toml.dump(data, f)
            return tmp
        except Exception:
            return None
