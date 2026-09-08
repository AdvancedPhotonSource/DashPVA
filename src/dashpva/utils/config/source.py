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

import contextlib
import hashlib
import os
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import toml

from dashpva.utils.config.revision import mapping_revision

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

_LOCAL_LOCK = threading.RLock()


class ConfigSourceError(RuntimeError):
    """A strict configuration read or write could not be completed."""


class ConfigSaveStatus(str, Enum):
    """Outcome of a strict compare-and-swap save."""

    SAVED = "saved"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass(frozen=True)
class ConfigSaveResult:
    """Result returned by :meth:`ConfigSource.replace_if_revision`."""

    status: ConfigSaveStatus
    revision: Optional[str] = None
    error: Optional[str] = None

    @property
    def saved(self) -> bool:
        return self.status is ConfigSaveStatus.SAVED


def _bytes_revision(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@contextlib.contextmanager
def _config_lock(path: Path, *, exclusive: bool):
    lock_path = path.with_name(f".{path.name}.lock")
    with _LOCAL_LOCK:
        with lock_path.open("a+b") as lock_file:
            if fcntl is not None:
                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_file.fileno(), mode)
            elif msvcrt is not None:  # pragma: no cover - Windows
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                mode = msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK
                while True:
                    try:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), mode, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _read_toml_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _parse_toml(payload: bytes, path: Path) -> Dict[str, Any]:
    try:
        return toml.loads(payload.decode("utf-8")) if payload else {}
    except Exception as exc:
        raise ConfigSourceError(f"could not parse configuration {path}: {exc}") from exc


def _write_temp_config(config: Dict[str, Any]) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".toml", prefix="dashpva_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            toml.dump(config, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return tmp

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
            config, _ = self.load_snapshot()
            return config
        except Exception:
            return {}

    def load_snapshot(self) -> tuple[Dict[str, Any], str]:
        path = Path(self.path)
        try:
            payload = _read_toml_bytes(path)
            return _parse_toml(payload, path), _bytes_revision(payload)
        except ConfigSourceError:
            raise
        except Exception as exc:
            raise ConfigSourceError(f"could not read configuration {path}: {exc}") from exc

    def replace_if_revision(
        self,
        full_config: Dict[str, Any],
        revision: str,
    ) -> ConfigSaveResult:
        path = Path(self.path)
        if not isinstance(full_config, dict):
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error="configuration must be a dict")
        try:
            replacement = toml.dumps(full_config).encode("utf-8")
        except Exception as exc:
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error=f"could not encode TOML: {exc}")

        tmp_path: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _config_lock(path, exclusive=True):
                current = _read_toml_bytes(path)
                if _bytes_revision(current) != revision:
                    return ConfigSaveResult(ConfigSaveStatus.CONFLICT)

                fd, tmp_name = tempfile.mkstemp(
                    dir=str(path.parent),
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                )
                tmp_path = Path(tmp_name)
                if path.exists():
                    try:
                        os.fchmod(fd, stat.S_IMODE(path.stat().st_mode))
                    except (AttributeError, OSError):
                        pass
                with os.fdopen(fd, "wb") as stream:
                    stream.write(replacement)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp_path, path)
                tmp_path = None
                try:
                    directory_fd = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
            return ConfigSaveResult(
                ConfigSaveStatus.SAVED,
                revision=_bytes_revision(replacement),
            )
        except Exception as exc:
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error=str(exc))
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def save(self, update: Dict[str, Any]) -> bool:
        for _ in range(3):
            try:
                existing, revision = self.load_snapshot()
            except ConfigSourceError:
                return False
            existing.update(update or {})
            result = self.replace_if_revision(existing, revision)
            if result.status is ConfigSaveStatus.SAVED:
                return True
            if result.status is ConfigSaveStatus.ERROR:
                return False
        return False

    def resolved_identity(self) -> Optional[str]:
        """Which configuration this source actually reads: its file path."""
        return self.path


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

    def load_snapshot(self) -> tuple[Dict[str, Any], str]:
        profile_id = self._resolve_profile_id()
        if profile_id is None or self.db is None:
            raise ConfigSourceError("database profile could not be resolved")
        try:
            if hasattr(self.db, "load_profile_toml_strict"):
                config = self.db.load_profile_toml_strict(profile_id)
            else:
                profile = self.db.get_profile_by_id(profile_id)
                if profile is None:
                    raise ConfigSourceError(f"database profile {profile_id} does not exist")
                config = self.db.export_profile_to_toml(profile_id) or {}
            return config, mapping_revision(config)
        except ConfigSourceError:
            raise
        except Exception as exc:
            raise ConfigSourceError(f"could not read database profile {profile_id}: {exc}") from exc

    def replace_if_revision(
        self,
        full_config: Dict[str, Any],
        revision: str,
    ) -> ConfigSaveResult:
        profile_id = self._resolve_profile_id()
        if profile_id is None or self.db is None:
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error="database profile could not be resolved")
        if not isinstance(full_config, dict):
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error="configuration must be a dict")
        try:
            status = self.db.replace_profile_toml_if_revision(
                profile_id,
                full_config,
                revision,
            )
        except Exception as exc:
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error=str(exc))
        if status == ConfigSaveStatus.SAVED.value:
            return ConfigSaveResult(
                ConfigSaveStatus.SAVED,
                revision=mapping_revision(full_config),
            )
        if status == ConfigSaveStatus.CONFLICT.value:
            return ConfigSaveResult(ConfigSaveStatus.CONFLICT)
        return ConfigSaveResult(ConfigSaveStatus.ERROR, error="database replacement failed")

    def save(self, update: Dict[str, Any]) -> bool:
        for _ in range(3):
            try:
                existing, revision = self.load_snapshot()
            except ConfigSourceError:
                return False
            existing.update(update or {})
            result = self.replace_if_revision(existing, revision)
            if result.status is ConfigSaveStatus.SAVED:
                return True
            if result.status is ConfigSaveStatus.ERROR:
                return False
        return False

    def resolved_identity(self) -> Optional[int]:
        """Which profile this source actually reads: its resolved database id.

        Not the same as ``self.locator`` -- a locator of None (or a
        'profile:<name>' locator) still resolves to one concrete profile id,
        and it's that resolved id callers need to detect "the active profile
        changed" even when the locator itself never changes.
        """
        return self._resolve_profile_id()


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
                from dashpva.database import DatabaseInterface
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

    def load_snapshot(self) -> tuple[Dict[str, Any], str]:
        """Return raw configuration and an opaque compare-and-swap revision."""
        if self._backend is None:
            raise ConfigSourceError("no configuration source is available")
        return self._backend.load_snapshot()

    def replace_if_revision(
        self,
        full_config: Dict[str, Any],
        revision: str,
    ) -> ConfigSaveResult:
        """Replace the raw profile only when *revision* is still current."""
        if self._backend is None:
            return ConfigSaveResult(ConfigSaveStatus.ERROR, error="no configuration source is available")
        return self._backend.replace_if_revision(full_config, revision)

    def save(self, update: Dict[str, Any]) -> bool:
        """Persist an updated configuration dict back to the current source."""
        if self._backend is not None:
            return self._backend.save(update)
        return False

    def resolved_identity(self) -> Any:
        """Which configuration source is actually active right now.

        Unlike ``self.locator`` (which is often None under DB auto-detect),
        this is resolved fresh from the backend built at construction time --
        a TOML path, a DB profile id, or None if no source is available.
        Constructing a new ConfigSource on every check (rather than caching
        one) is what makes this reflect a DB selection change made elsewhere,
        even when the locator passed in stays the same (e.g. always None).
        """
        if self._backend is None:
            return None
        return self._backend.resolved_identity()

    def ensure_path(self, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Return a file path to a TOML representation of the config.

        - TOML source: returns the original file path (already on disk).
        - DB source: exports the config to a temporary TOML file and returns that path.
        - None source: returns None.
        """
        if config is not None:
            try:
                return _write_temp_config(config)
            except Exception:
                return None
        if self.source_type == "toml":
            return self._backend.path
        if self.source_type == "db":
            try:
                return _write_temp_config(self.load())
            except Exception:
                return None
        return None
