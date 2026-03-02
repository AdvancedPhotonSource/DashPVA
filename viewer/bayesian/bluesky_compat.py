"""
bluesky_compat.py
=================
Compatibility layer for importing Bluesky / Ophyd packages from the
``6idb-bits`` conda environment into the uv-managed DashPVA venv.

DashPVA uses ``uv`` for dependency management, but Bluesky and Ophyd are
installed in a separate conda environment (``6idb-bits``).  This module
injects the conda env's ``site-packages`` into ``sys.path`` so that
``import bluesky`` and ``import ophyd`` resolve correctly.

Usage
-----
    from viewer.bayesian.bluesky_compat import ensure_bluesky
    ensure_bluesky()          # call once, idempotent
    import bluesky            # now works
    from bluesky import RunEngine
"""

from __future__ import annotations

import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded paths — edit these if the environment moves
# ---------------------------------------------------------------------------

#: Root of the conda environment that contains bluesky / ophyd
CONDA_ENV_ROOT = "/home/beams/USER6IDB/.conda/envs/6idb-bits"

#: Source tree of the 6idb-bits instrument package (for ``id6_b`` devices)
BITS_SRC_DIR = "/home/beams/USER6IDB/6idb-bits/src"

_already_injected = False


def _find_site_packages(env_root: str) -> str | None:
    """Auto-detect the ``site-packages`` directory inside a conda env.

    Globs ``lib/python*/site-packages`` so the exact Python version
    (3.11, 3.12, …) does not need to be hardcoded.
    """
    pattern = os.path.join(env_root, "lib", "python*", "site-packages")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]  # pick the highest Python version if multiple
    return None


def ensure_bluesky() -> None:
    """Inject ``6idb-bits`` conda site-packages into ``sys.path``.

    Safe to call multiple times — subsequent calls are no-ops.

    Raises
    ------
    RuntimeError
        If the conda environment or its ``site-packages`` directory cannot
        be found on disk.
    """
    global _already_injected
    if _already_injected:
        return

    # --- conda env site-packages ---
    sp = _find_site_packages(CONDA_ENV_ROOT)
    if sp is None or not os.path.isdir(sp):
        raise RuntimeError(
            f"Cannot find site-packages in conda env: {CONDA_ENV_ROOT}\n"
            "Expected a directory matching lib/python*/site-packages.\n"
            "Please verify that the 6idb-bits conda environment is installed."
        )

    if sp not in sys.path:
        sys.path.insert(0, sp)
        logger.info("Injected conda site-packages: %s", sp)

    # --- 6idb-bits instrument source (for id6_b devices) ---
    if os.path.isdir(BITS_SRC_DIR) and BITS_SRC_DIR not in sys.path:
        sys.path.insert(0, BITS_SRC_DIR)
        logger.info("Injected 6idb-bits src: %s", BITS_SRC_DIR)

    _already_injected = True

    # Quick smoke test
    try:
        import bluesky  # noqa: F401
        logger.info("bluesky %s imported successfully", bluesky.__version__)
    except ImportError as exc:
        _already_injected = False
        raise RuntimeError(
            f"bluesky import failed after path injection: {exc}\n"
            f"site-packages used: {sp}"
        ) from exc
