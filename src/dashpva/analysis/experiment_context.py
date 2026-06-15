"""
ExperimentContext — loads optional prior knowledge about the current
experiment (material, energy, expected phases, free-form notes) so the
SessionAnalyzer can inject it into the LLM prompt.

Supported file formats: .toml, .json, .txt / anything else (treated as plain
text). Missing file or empty path → silently returns ''.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class ExperimentContext:
    """
    Load and format optional experiment context from a TOML / JSON / text file.

    Use ``to_text()`` to get a string ready to drop into an LLM prompt.
    """

    def __init__(self, filepath: str | None):
        self.filepath = (filepath or '').strip()
        self.data: dict | str = {}
        self._load()

    def _load(self) -> None:
        if not self.filepath:
            return
        p = Path(self.filepath).expanduser()
        if not p.is_file():
            return
        suffix = p.suffix.lower()
        try:
            if suffix == '.toml':
                with p.open('rb') as f:
                    self.data = tomllib.load(f)
            elif suffix == '.json':
                with p.open('r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = p.read_text(encoding='utf-8').strip()
        except Exception:
            # Silently treat unreadable files as no-context — the user can
            # see the dock-side status and pick a different file.
            self.data = {}

    def to_text(self) -> str:
        """Return a human-readable string suitable for prompt injection."""
        if not self.data:
            return ''
        if isinstance(self.data, str):
            return self.data
        lines = []
        for key, value in self.data.items():
            if isinstance(value, (list, tuple)):
                value_str = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                # One-level nested dicts: flatten as "key.subkey: value"
                nested = ', '.join(f"{k}={v}" for k, v in value.items())
                value_str = f"{{{nested}}}"
            else:
                value_str = str(value)
            lines.append(f"  {key}: {value_str}")
        return "Experiment context:\n" + "\n".join(lines)