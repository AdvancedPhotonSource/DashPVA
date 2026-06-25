"""Named Bayesian optimizer setups stored in the app's central profile.

The optimizer's DOF/objective PVs are exportable configuration, so they live in the
same central profile the consumers and the area-detector reader read from (Workflow
editor -> ``dashpva.db`` / TOML, surfaced via ``settings`` / ``ConfigSource``), under
a top-level ``[BAYESIAN]`` table keyed by setup name.  Each named setup is an
``OptimizerConfig``::

    [BAYESIAN.beam_collimation]   # == OptimizerConfig.to_dict()
    [BAYESIAN.crl_focusing]

When no profile / database is available (offline sim, fresh machine) there is no
active source; callers fall back to local QSettings storage.

This module is pure (no Qt) so it can be unit-tested against a temporary TOML file.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

from dashpva.viewer.bayesian.blop_adapter import OptimizerConfig

logger = logging.getLogger(__name__)

# Top-level table in the profile config that holds the named Bayesian setups.
SECTION = "BAYESIAN"


def active_source() -> Tuple[Optional[Any], str]:
    """Resolve the app's active config source for reading/writing setups.

    Returns ``(source, label)`` where ``source`` is a ``ConfigSource`` bound to the
    currently-selected profile (or active TOML file), and ``label`` is a short human
    name used for display and the local last-used key.  Returns ``(None, "")`` when no
    profile/DB is resolvable, so the caller falls back to local storage.
    """
    try:
        from dashpva import settings
        from dashpva.utils.config.source import ConfigSource
    except Exception:  # noqa: BLE001 - config stack unavailable -> local fallback
        return None, ""

    # Re-resolve the locator so we pick up a profile selected in another window or
    # process (the Workflow editor writes the choice to the locator state file); the
    # optimizer runs as its own process with a cached settings.LOCATOR otherwise.
    try:
        settings.reload()
    except Exception:  # noqa: BLE001
        pass

    locator = getattr(settings, "LOCATOR", None)
    try:
        src = ConfigSource(locator)
    except Exception:  # noqa: BLE001
        return None, ""

    if getattr(src, "source_type", "none") == "none":
        return None, ""
    return src, _label(src, locator)


def _label(src: Any, locator: Any) -> str:
    """Best-effort human name for the active source (profile name or TOML filename)."""
    if getattr(src, "source_type", "") == "toml":
        return os.path.basename(getattr(src, "locator", "") or "config.toml")
    # DB profile: resolve the profile's display name.
    try:
        from dashpva.database import DatabaseInterface

        db = DatabaseInterface()
        if isinstance(locator, int):
            prof = db.get_profile_by_id(locator)
        elif isinstance(locator, str) and locator.startswith("profile:"):
            prof = db.get_profile_by_name(locator[len("profile:"):])
        elif isinstance(locator, str) and locator:
            prof = db.get_profile_by_name(locator)
        else:
            prof = db.get_selected_profile()
        if prof is not None:
            return prof.name
    except Exception:  # noqa: BLE001
        pass
    return "profile"


def _read_table(src: Any) -> dict:
    try:
        cfg = src.load() or {}
    except Exception:  # noqa: BLE001
        return {}
    table = cfg.get(SECTION, {})
    return dict(table) if isinstance(table, dict) else {}


def list_setups(src: Any) -> List[str]:
    """Sorted names of the Bayesian setups stored in the source's profile."""
    if src is None:
        return []
    return sorted(_read_table(src).keys())


def load_setup(src: Any, name: str) -> Optional[OptimizerConfig]:
    """Return the named setup as an ``OptimizerConfig``, or ``None`` if absent."""
    if src is None or not name:
        return None
    data = _read_table(src).get(name)
    if not isinstance(data, dict):
        return None
    try:
        return OptimizerConfig.from_dict(data)
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse Bayesian setup %r: %s", name, exc)
        return None


def save_setup(src: Any, name: str, cfg: OptimizerConfig) -> bool:
    """Write ``cfg`` under ``name``, preserving sibling setups (load-modify-save)."""
    if src is None or not name:
        return False
    table = _read_table(src)
    table[name] = cfg.to_dict()
    try:
        return bool(src.save({SECTION: table}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save Bayesian setup %r: %s", name, exc)
        return False


def delete_setup(src: Any, name: str) -> bool:
    """Remove the named setup, preserving siblings.  False if it did not exist."""
    if src is None or not name:
        return False
    table = _read_table(src)
    if name not in table:
        return False
    table.pop(name)
    try:
        return bool(src.save({SECTION: table}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete Bayesian setup %r: %s", name, exc)
        return False
