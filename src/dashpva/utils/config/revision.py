# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Stable revisions for configuration compare-and-swap operations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def mapping_revision(config: Mapping[str, Any]) -> str:
    """Return a deterministic revision for a TOML-shaped mapping."""
    payload = json.dumps(
        config,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
