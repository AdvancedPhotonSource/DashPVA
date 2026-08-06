# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from typing import Any


class BaseManager:
    """Shared helpers for database managers."""

    def clean(self, value):
        """Strip stray '*' UI edit-tracking markers before persisting."""
        return self._strip_trailing_asterisks(value)

    def _strip_trailing_asterisks(self, obj: Any) -> Any:
        """Recursively strip trailing '*' UI edit-tracking markers from strings, dict keys, and list items.

        If both a clean key and its '*'-suffixed duplicate exist in the same dict,
        the clean key's value always wins, regardless of dict iteration order.
        """
        if isinstance(obj, dict):
            cleaned = [
                (key, key.rstrip('*') if isinstance(key, str) else key, self._strip_trailing_asterisks(value))
                for key, value in obj.items()
            ]
            result = {clean_key: value for key, clean_key, value in cleaned if key == clean_key}
            for key, clean_key, value in cleaned:
                if key != clean_key:
                    result.setdefault(clean_key, value)
            return result
        if isinstance(obj, list):
            return [self._strip_trailing_asterisks(item) for item in obj]
        if isinstance(obj, str):
            return obj.rstrip('*')
        return obj
