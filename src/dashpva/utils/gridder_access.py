# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Guarded access to xrayutilities' Gridder3D accumulators.

``Gridder3D`` exposes only ``.data``, which copies the whole numerator array
and divides on every access -- unaffordable for a live grid refreshing once a
second, where a 256^3 volume means a fresh 134 MB allocation per snapshot. It
also leaves un-hit voxels at zero, indistinguishable from a measured zero.

So we read ``_gdata`` (the running sum of gridded values) and ``_gnorm`` (the
per-voxel contribution count) directly. Those are private, so this module is
the single place that touches them, and :func:`verify_gridder_contract`
pins the *semantics* rather than merely checking the attributes exist -- a
future xrayutilities could keep the names and change the meaning, and a
``hasattr`` check would sail straight past that.
"""

from __future__ import annotations

import numpy as np
import xrayutilities as xu

__all__ = [
    "GridderContractError",
    "gridder_coverage",
    "gridder_numerator",
    "verify_gridder_contract",
]


class GridderContractError(RuntimeError):
    """Raised when Gridder3D no longer matches the accumulator contract."""


def gridder_numerator(gridder: "xu.Gridder3D") -> np.ndarray:
    """The running sum of gridded values (``_gdata``), not a copy."""
    try:
        return gridder._gdata
    except AttributeError as exc:  # pragma: no cover - guarded at startup
        raise GridderContractError(
            "xrayutilities Gridder3D no longer exposes _gdata; the live grid "
            "and the offline coverage volume depend on it."
        ) from exc


def gridder_coverage(gridder: "xu.Gridder3D") -> np.ndarray:
    """Per-voxel contribution count (``_gnorm``), not a copy."""
    try:
        return gridder._gnorm
    except AttributeError as exc:  # pragma: no cover - guarded at startup
        raise GridderContractError(
            "xrayutilities Gridder3D no longer exposes _gnorm; the live grid "
            "and the offline coverage volume depend on it."
        ) from exc


def verify_gridder_contract() -> None:
    """Assert _gdata sums values and _gnorm counts contributions.

    Grids two known points into one voxel and a third into another, then
    checks the accumulators hold the sum and the count. This is what makes the
    guard meaningful: it fails if the fields are renamed *or* if they start
    meaning something else.
    """
    gridder = xu.Gridder3D(2, 2, 2)
    gridder.KeepData(True)
    gridder.dataRange(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, fixed=True)

    # Two samples into the (0,0,0) corner, one into (1,1,1).
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.array([0.0, 0.0, 1.0])
    values = np.array([2.0, 3.0, 7.0])
    gridder(x, y, z, values)

    numerator = np.asarray(gridder_numerator(gridder))
    coverage = np.asarray(gridder_coverage(gridder))

    if numerator.shape != (2, 2, 2) or coverage.shape != (2, 2, 2):
        raise GridderContractError(
            f"Gridder3D accumulators have unexpected shapes "
            f"{numerator.shape} / {coverage.shape}; expected (2, 2, 2)."
        )
    if not np.isclose(numerator[0, 0, 0], 5.0):
        raise GridderContractError(
            f"Gridder3D._gdata is no longer a running sum: expected 2 + 3 = 5 in "
            f"the first voxel, got {numerator[0, 0, 0]!r}."
        )
    if not np.isclose(coverage[0, 0, 0], 2.0):
        raise GridderContractError(
            f"Gridder3D._gnorm is no longer a contribution count: expected 2 in "
            f"the first voxel, got {coverage[0, 0, 0]!r}."
        )
    if not np.isclose(numerator[1, 1, 1], 7.0) or not np.isclose(coverage[1, 1, 1], 1.0):
        raise GridderContractError(
            "Gridder3D accumulators did not record the single-sample voxel as "
            f"sum 7 / count 1, got {numerator[1, 1, 1]!r} / {coverage[1, 1, 1]!r}."
        )
    if not np.isclose(coverage[0, 0, 1], 0.0):
        raise GridderContractError(
            "Gridder3D._gnorm is non-zero in a voxel nothing was gridded into."
        )
