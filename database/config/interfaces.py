"""
Interfaces for configuration sources (DB and TOML) used by DashPVA.
"""

from typing import Dict, Any


class ConfigSource:
    """
    Abstract interface for a configuration source.

    Implementations:
      - DbProfileConfigSource: loads/saves from database profiles
      - TomlConfigSource: loads/saves from TOML files

    Attributes:
      source_type: Literal "db" or "toml"
    """

    source_type: str = ""

    def load(self) -> Dict[str, Any]:
        """
        Load configuration as a TOML-shaped dictionary.
        """
        raise NotImplementedError("load() must be implemented by subclasses")

    def save(self, update: Dict[str, Any]) -> bool:
        """
        Save the provided configuration dictionary back to the source.

        Returns:
          True if successful, False otherwise.
        """
        raise NotImplementedError("save() must be implemented by subclasses")
