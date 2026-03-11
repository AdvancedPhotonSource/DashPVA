"""
Unified configuration source for DashPVA.

Three source classes, one entry point:

  TomlConfigSource      – reads/writes a TOML file on disk
  DbProfileConfigSource – reads/writes a database profile
  ConfigSource          – single class to import; auto-selects the right backend

Usage (settings.py only ever imports ConfigSource):

    src = ConfigSource('/path/to/config.toml')   # → TOML backend
    src = ConfigSource(42)                        # → DB backend (profile id)
    src = ConfigSource('profile:my_profile')      # → DB backend (by name)
    src = ConfigSource()                          # → auto-detects selected DB profile;
                                                  #   falls back to empty dict so
                                                  #   settings.py uses minimal defaults

    cfg = src.load()   # always returns a plain dict
"""

from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, Optional, Union
import toml


# ---------------------------------------------------------------------------
# Concrete source backends
# ---------------------------------------------------------------------------

class TomlConfigSource:
    """Load/save configuration from a TOML file on disk."""

    source_type: str = "toml"

    def __init__(self, path: str) -> None:
        self.path = str(path)

    def load(self) -> Dict[str, Any]:
        try:
            return toml.load(self.path)
        except Exception:
            return {}

    def save(self, update: Dict[str, Any]) -> bool:
        try:
            existing = self.load()
            existing.update(update or {})
            with open(self.path, "w") as f:
                toml.dump(existing, f)
            return True
        except Exception:
            return False


class DbProfileConfigSource:
    """Load/save configuration from a database profile (by id or 'profile:<name>')."""

    source_type: str = "db"

    def __init__(self, db: Any, locator: Union[int, str]) -> None:
        self.db = db
        self.locator = locator

    def _resolve_profile_id(self) -> Optional[int]:
        if self.db is None:
            return None
        loc = self.locator
        if isinstance(loc, int):
            return loc
        if isinstance(loc, str) and loc.startswith("profile:"):
            name = loc[len("profile:"):]
            try:
                prof = self.db.get_profile_by_name(name)
                return prof.id if prof else None
            except Exception:
                return None
        if isinstance(loc, str):
            try:
                prof = self.db.get_profile_by_name(loc)
                return prof.id if prof else None
            except Exception:
                return None
        return None

    def load(self) -> Dict[str, Any]:
        profile_id = self._resolve_profile_id()
        if profile_id is None or self.db is None:
            return {}
        try:
            return self.db.export_profile_to_toml(profile_id) or {}
        except Exception:
            return {}

    def save(self, update: Dict[str, Any]) -> bool:
        profile_id = self._resolve_profile_id()
        if profile_id is None or self.db is None:
            return False
        try:
            return self.db.import_toml_to_profile(profile_id, update or {})
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

class ConfigSource:
    """
    Single configuration source — the only class settings.py needs to import.

    Detects the backend automatically from the locator:
      - str path ending in '.toml' or an existing file → TomlConfigSource
      - int or 'profile:<name>'                        → DbProfileConfigSource
      - None                                           → no source (empty dict;
                                                         settings.py uses its
                                                         hard-coded defaults)

    Args:
        locator: TOML path, DB profile id, 'profile:<name>', or None.
    """

    def __init__(
        self,
        locator: Optional[Union[int, str]] = None,
    ) -> None:
        self.locator = locator
        self._db: Any = None
        self.source_type: str = self._detect()
        self._backend: Optional[Union[TomlConfigSource, DbProfileConfigSource]] = None
        self._build_backend()

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _detect(self) -> str:
        loc = self.locator
        if loc is None:
            return "none"
        if isinstance(loc, int):
            return "db"
        if isinstance(loc, str):
            if loc.startswith("profile:"):
                return "db"
            if loc.endswith(".toml") or os.path.exists(loc):
                return "toml"
        return "none"

    def _get_db(self) -> Any:
        if self._db is None:
            try:
                from database import DatabaseInterface
                self._db = DatabaseInterface()
            except Exception:
                pass
        return self._db

    def _build_backend(self) -> None:
        if self.source_type == "toml":
            self._backend = TomlConfigSource(self.locator)
        elif self.source_type == "db":
            self._backend = DbProfileConfigSource(self._get_db(), self.locator)
        else:
            # locator is None — try the currently selected DB profile before giving up
            db = self._get_db()
            if db is not None:
                try:
                    sel = db.get_selected_profile()
                    if sel is not None:
                        self._backend = DbProfileConfigSource(db, sel.id)
                        self.source_type = "db"
                        return
                except Exception:
                    pass
            self._backend = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        """Return the full configuration dict for the current source."""
        if self._backend is not None:
            return self._backend.load()
        return {}

    def save(self, update: Dict[str, Any]) -> bool:
        """Persist an updated configuration dict back to the current source."""
        if self._backend is not None:
            return self._backend.save(update)
        return False

    def ensure_path(self) -> Optional[str]:
        """Return a file path to a TOML representation of the config.

        - TOML source: returns the original file path (already on disk).
        - DB source: exports the config to a temporary TOML file and returns that path.
        - None source: returns None.
        """
        if self.source_type == "toml":
            return self._backend.path
        if self.source_type == "db":
            try:
                data = self.load()
                fd, tmp = tempfile.mkstemp(suffix=".toml", prefix="dashpva_")
                os.close(fd)
                with open(tmp, "w") as f:
                    toml.dump(data, f)
                return tmp
            except Exception:
                return None
        return None
