"""Persist small key/value bits of UI state (e.g. last-typed prefix) across launches.

Backed by ``<repo-root>/.dashpva_data.json`` — the same file ``settings.py``
falls back to when the DB isn't available. Read-modify-write per call so
unrelated keys are preserved.
"""

import json
import logging
from typing import Any

from dashpva import settings as app_settings

_CONFIG_FILE = app_settings.PROJECT_ROOT / ".dashpva_data.json"

logger = logging.getLogger(__name__)


def load_last(key: str, default: Any = "") -> Any:
    try:
        with open(_CONFIG_FILE, "r") as f:
            return json.load(f).get(key, default)
    except FileNotFoundError:
        return default
    except Exception as e:
        logger.debug("user_config.load_last(%r) failed: %s", key, e)
        return default


def save_last(key: str, value: Any) -> None:
    try:
        try:
            with open(_CONFIG_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, ValueError):
            data = {}
        data[key] = value
        with open(_CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.debug("user_config.save_last(%r) failed: %s", key, e)
