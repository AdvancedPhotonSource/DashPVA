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

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, List, Pattern

import dashpva.settings as settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PyQt5.QtWidgets import QApplication

# PyQt5 is deliberately NOT imported at module scope. LogMixin is used by
# HDF5Writer and other pure-write code that must be importable inside a
# pvaccess consumer process, where mixing in PyQt5 core-dumps (see the
# two-process note in consumers/ioc_rsm_parameter). Qt is only touched when a
# GUI process actually asks for the running QApplication.


def _running_qapplication():
    """The live QApplication, or None outside a GUI process."""
    try:
        from PyQt5.QtWidgets import QApplication as _QApplication
    except Exception:
        return None
    try:
        return _QApplication.instance()
    except Exception:
        return None


class UINoiseFilter(logging.Filter):
    """
    Filter that drops low-value INFO messages (typically chatty UI status updates)
    before they reach the rotating file handler.

    Patterns are regexes matched against the final log message text. Only applies
    to INFO level; WARNING/ERROR/DEBUG are always allowed through.

    Configure patterns via env var DASHPVA_LOG_INFO_DROP as a comma-separated list
    of regexes. If not provided, sensible defaults are used.
    """

    def __init__(self, patterns: List[Pattern[str]]):
        super().__init__()
        self._patterns = patterns or []

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        # Only filter chatty INFO messages; allow everything else
        if record.levelno != logging.INFO:
            return True
        try:
            msg = record.getMessage()
        except Exception:
            # Fallback to raw message
            msg = str(record.msg)
        for pat in self._patterns:
            try:
                if pat.search(msg):
                    return False
            except Exception:
                # If a pattern misbehaves, fail open
                continue
        return True


class LogManager:
    """
    Central logging manager with a single RotatingFileHandler.
    - Log file: <settings.LOG_PATH>/general.log when available, otherwise logs/general.log
      (ensures directory exists)
    - Format: "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    - Rotation defaults: 10 MB per file, 10 backups
      Override via env: DASHPVA_LOG_MAX_BYTES, DASHPVA_LOG_BACKUPS
      Optional level override via env: DASHPVA_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)

    Noise filtering:
      A UINoiseFilter is attached to the file handler to drop chatty UI INFO
      updates (e.g., Vmin/Vmax set, Selected:, Collapsed/Expanded, Playback toggles,
      Log scale toggles). Configure patterns with DASHPVA_LOG_INFO_DROP (comma-
      separated regex). If not set, defaults are applied.
    """

    def __init__(self, app: QApplication = None, app_name: str = "DashPVA", log_file: str = "logs/general.log"):
        self.app = app
        self.app_name = app_name
        # Resolve log path from settings module; fallback to provided default
        if settings.LOG_PATH:
            log_path = Path(settings.LOG_PATH).expanduser()
            self.log_file = str(log_path / "general.log")
        else:
            self.log_file = log_file
        

        # Rotation configuration with env overrides
        default_max_bytes = 10 * 1024 * 1024
        default_backup_count = 10
        try:
            # Prefer new env names; fall back to legacy if present
            max_bytes = int(os.environ.get("DASHPVA_LOG_MAX_BYTES", os.environ.get("DASHPVA_LOG_MAX_BYTES", default_max_bytes)))
        except Exception:
            max_bytes = default_max_bytes
        try:
            backup_count = int(os.environ.get("DASHPVA_LOG_BACKUPS", os.environ.get("DASHPVA_LOG_BACKUPS", default_backup_count)))
        except Exception:
            backup_count = default_backup_count

        level_name = os.environ.get("DASHPVA_LOG_LEVEL", os.environ.get("DASHPVA_LOG_LEVEL", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        self.level = level

        # Ensure logs directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        self._handler = RotatingFileHandler(
            self.log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        self._handler.setLevel(level)
        # Formatter without milliseconds in the timestamp
        self._handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # Attach a UI noise filter to drop low-value INFO messages from persisting
        drop_specs = os.environ.get("DASHPVA_LOG_INFO_DROP", "").strip()
        if drop_specs:
            raw_patterns = [p.strip() for p in drop_specs.split(",") if p.strip()]
        else:
            # Sensible defaults
            raw_patterns = [
                r"^(Vmin|Vmax) set to:",
                r"^Selected:",
                r"^Collapsed",
                r"^Expanded",
                r"^Playback (started|paused)",
                r"^Log scale (enabled|disabled)",
            ]

        compiled: List[Pattern[str]] = []
        for rp in raw_patterns:
            try:
                compiled.append(re.compile(rp))
            except Exception:
                # Ignore invalid patterns
                continue
        self._handler.addFilter(UINoiseFilter(compiled))

        # Track which loggers we've already attached the handler to
        self._attached = set()

        # Install global excepthook
        self.enable_global_excepthook()

    def get_logger(self, name: str) -> logging.Logger:
        """Return a logger named as provided, ensuring our rotating file handler is attached once."""
        logger = logging.getLogger(str(name) if name else self.app_name)
        logger.setLevel(self.level)
        # Avoid duplicate propagation to root to keep single sink
        logger.propagate = False
        if logger.name not in self._attached:
            logger.addHandler(self._handler)
            self._attached.add(logger.name)
        return logger

    def enable_global_excepthook(self) -> None:
        """Route uncaught exceptions to the logging system."""
        # Only install once
        if getattr(self, "_excepthook_installed", False):
            return

        def _hook(exc_type, exc_value, exc_tb):
            try:
                # Ensure our rotating handler is attached and use it for uncaught exceptions
                logger = self.get_logger(self.app_name)
                logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
            except Exception:
                # Fallback to stderr if logging fails for any reason
                import traceback
                traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

        sys.excepthook = _hook
        self._excepthook_installed = True


# Singleton access
_default_manager = None

def get_default_manager(app: QApplication = None) -> LogManager:
    """Return the singleton LogManager, creating it if necessary."""
    global _default_manager
    if _default_manager is None:
        if app is None:
            app = _running_qapplication()
        _default_manager = LogManager(app=app)
    return _default_manager


class LogMixin:
    """Mixin providing a drop-in self.logger backed by LogManager."""

    def set_log_manager(self, manager: LogManager = None, viewer_name: str = None) -> None:
        mgr = manager or get_default_manager()
        # Allow caller to set/override viewer_name
        if viewer_name is not None:
            try:
                self.viewer_name = str(viewer_name)
            except Exception:
                self.viewer_name = None
        name = getattr(self, "viewer_name", None) or f"{self.__module__}.{self.__class__.__name__}"
        self.logger = mgr.get_logger(name)

        