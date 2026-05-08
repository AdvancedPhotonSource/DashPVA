"""
bluesky_compat.py
=================
Compatibility layer for importing Bluesky / Ophyd packages from a
beamline conda environment into the uv-managed DashPVA venv.

DashPVA uses ``uv`` for dependency management, but Bluesky and Ophyd are
installed in the beamline's conda environment.  This module injects the
conda env's ``site-packages`` into ``sys.path`` so that
``import bluesky`` and ``import ophyd`` resolve correctly.

Configuration (pick one, in priority order):
    1. Pass ``root`` directly to ``ensure_bluesky(root="/path/to/conda/env")``
    2. Set env var ``DASHPVA_BLUESKY_ROOT=/path/to/conda/env``
    3. Falls back to the APS 6-IDB default

Usage
-----
    from viewer.bayesian.bluesky_compat import ensure_bluesky
    ensure_bluesky()          # call once, idempotent
    import bluesky            # now works
"""

from __future__ import annotations

import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

_DEFAULT_CONDA_ENV = "/home/beams/USER6IDB/.conda/envs/6idb-bits"

_already_injected = False
_configured_root: str | None = None


def _find_site_packages(env_root: str) -> str | None:
    """Auto-detect the ``site-packages`` directory inside a conda env."""
    pattern = os.path.join(env_root, "lib", "python*", "site-packages")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return None


def get_bluesky_root() -> str:
    """Return the currently configured conda env root path."""
    if _configured_root:
        return _configured_root
    return os.getenv("DASHPVA_BLUESKY_ROOT", _DEFAULT_CONDA_ENV)


def ensure_bluesky(root: str | None = None) -> None:
    """Inject beamline conda site-packages into ``sys.path``.

    Parameters
    ----------
    root : str, optional
        Path to the conda environment containing bluesky/ophyd.
        If provided on the first call, overrides the env var and default.
        Subsequent calls are no-ops regardless of the ``root`` argument.

    Raises
    ------
    RuntimeError
        If the conda environment or bluesky cannot be found.
    """
    global _already_injected, _configured_root
    if _already_injected:
        return

    env_root = root or os.getenv("DASHPVA_BLUESKY_ROOT") or _DEFAULT_CONDA_ENV
    _configured_root = env_root

    sp = _find_site_packages(env_root)
    if sp is None or not os.path.isdir(sp):
        raise RuntimeError(
            f"Cannot find site-packages in conda env: {env_root}\n"
            "Expected a directory matching lib/python*/site-packages.\n"
            "Set DASHPVA_BLUESKY_ROOT or pass the path in the viewer GUI."
        )

    if sp not in sys.path:
        sys.path.insert(0, sp)
        logger.info("Injected conda site-packages: %s", sp)

    # Auto-discover instrument source (e.g. 6idb-bits/src) next to conda env
    bits_src = os.getenv("DASHPVA_BLUESKY_BITS_SRC", "")
    if bits_src and os.path.isdir(bits_src) and bits_src not in sys.path:
        sys.path.insert(0, bits_src)
        logger.info("Injected instrument src: %s", bits_src)

    _already_injected = True

    try:
        import bluesky  # noqa: F401
        logger.info("bluesky %s imported successfully", bluesky.__version__)
    except ImportError as exc:
        _already_injected = False
        raise RuntimeError(
            f"bluesky import failed after path injection: {exc}\n"
            f"site-packages used: {sp}"
        ) from exc
