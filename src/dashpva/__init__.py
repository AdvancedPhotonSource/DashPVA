# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""DashPVA: Distributed Analysis and Streaming Hub with Process Variable Access."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version():
    # Prefer the version in the checked-out pyproject.toml so an in-place update
    # (git checkout of a release tag) is reflected without reinstalling; fall back
    # to installed metadata for wheel installs that don't ship pyproject.toml.
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        with open(pyproject, "rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        try:
            return version("DashPVA")
        except PackageNotFoundError:
            return "0.0.0"


__version__ = _version()

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass
