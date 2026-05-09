"""DashPVA viewer.launcher package.

Ensures relative imports in submodules (e.g., launcher.py) work reliably
when invoked via "python -m viewer.launcher.launcher".
"""

# Re-export views registry for convenience
from .registry import VIEWS  # noqa: F401
