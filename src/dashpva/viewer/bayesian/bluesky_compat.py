"""
bluesky_compat.py
=================
Compatibility layer for importing Bluesky / Ophyd / blop.

DashPVA's ``bayesian`` install (``bash install.sh --bayesian``) pulls
bluesky, ophyd and blop straight into the uv-managed DashPVA venv, so in the
normal case **nothing needs to be injected** — ``import bluesky`` already works.

This module only does something when bluesky is *not* importable from the
current interpreter (e.g. DashPVA was installed lean and you want to borrow a
beamline conda env's site-packages).  In that case it injects the env's
``site-packages`` onto ``sys.path``.

Configuration (only consulted when bluesky is missing), in priority order:
    1. Pass ``root`` directly: ``ensure_bluesky(root="/path/to/conda/env")``
    2. Set env var ``DASHPVA_BLUESKY_ROOT=/path/to/conda/env``
    3. (no built-in default) -> raise a clear error telling the user what to set

There is deliberately **no** hard-coded facility default: the previous
``/home/beams/USER6IDB/.conda/envs/6idb-bits`` (an APS 6-IDB path) was wrong for
every other site.  If you need to borrow an env, name it explicitly.

Usage
-----
    from dashpva.viewer.bayesian.bluesky_compat import ensure_bluesky
    ensure_bluesky()          # call once, idempotent; no-op if bluesky present
    import bluesky            # now works
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

# No built-in facility default.  ``None`` means "only use bluesky if it is
# already importable, or if a root is supplied explicitly / via env var".
_DEFAULT_CONDA_ENV: str | None = None

_already_injected = False
_configured_root: str | None = None


def _bluesky_importable() -> bool:
    """True if ``import bluesky`` would succeed without any path surgery."""
    try:
        return importlib.util.find_spec("bluesky") is not None
    except (ImportError, ValueError):
        return False


def _find_site_packages(env_root: str) -> str | None:
    """Auto-detect the ``site-packages`` directory inside a conda env."""
    pattern = os.path.join(env_root, "lib", "python*", "site-packages")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    return None


def get_bluesky_root() -> str | None:
    """Return the currently configured conda env root path, or ``None``.

    ``None`` means "no external env configured; rely on the DashPVA venv".
    """
    if _configured_root:
        return _configured_root
    return os.getenv("DASHPVA_BLUESKY_ROOT") or _DEFAULT_CONDA_ENV


def ensure_bluesky(root: str | None = None) -> None:
    """Make ``import bluesky`` work; a no-op when it already does.

    Parameters
    ----------
    root : str, optional
        Path to a conda environment containing bluesky/ophyd to borrow from,
        used **only** if bluesky is not already importable.  If provided on the
        first call it overrides the env var.  Subsequent calls are no-ops.

    Raises
    ------
    RuntimeError
        If bluesky is not importable and no valid env root can be found.
    """
    global _already_injected, _configured_root
    if _already_injected:
        return

    # Fast path: the DashPVA venv already has bluesky (the --bayesian install).
    if root is None and _bluesky_importable():
        _already_injected = True
        logger.debug("bluesky already importable; no path injection needed.")
        return

    env_root = root or os.getenv("DASHPVA_BLUESKY_ROOT") or _DEFAULT_CONDA_ENV
    if not env_root:
        # Nothing to inject and bluesky isn't here -> actionable error.
        raise RuntimeError(
            "bluesky is not importable in this interpreter and no external "
            "conda env was configured.\n"
            "Either install the bayesian extra (`bash install.sh --bayesian`) "
            "so bluesky/ophyd/blop live in the DashPVA venv, or point DashPVA "
            "at a conda env that has them:\n"
            "    export DASHPVA_BLUESKY_ROOT=/path/to/conda/env\n"
            "or call ensure_bluesky(root='/path/to/conda/env')."
        )

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

    # Optionally inject an instrument source tree (e.g. a *-bits/src package).
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
