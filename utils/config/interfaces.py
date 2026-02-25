from typing import Dict, Any, Protocol


class ConfigSource(Protocol):
    """
    Interface for a configuration source.

    Implementations should provide .load() to return a dict of settings,
    and .save(update) to persist a dict of settings.
    May optionally expose attributes like 'source_type' and for TOML
    sources, a 'path'.
    """

    source_type: str

    def load(self) -> Dict[str, Any]:
        ...

    def save(self, update: Dict[str, Any]) -> bool:
        ...
