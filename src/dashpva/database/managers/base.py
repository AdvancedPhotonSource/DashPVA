# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from typing import Any


class BaseManager:
    """Shared helpers for database managers."""

    def clean(self, value):
        """Strip stray '*' UI edit-tracking markers before persisting."""
        return self._strip_trailing_asterisks(value)

    def _strip_trailing_asterisks(self, obj: Any) -> Any:
        """Recursively strip trailing '*' UI edit-tracking markers from strings, dict keys, and list items."""
        if isinstance(obj, dict):
            return {
                (key.rstrip('*') if isinstance(key, str) else key): self._strip_trailing_asterisks(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [self._strip_trailing_asterisks(item) for item in obj]
        if isinstance(obj, str):
            return obj.rstrip('*')
        return obj
