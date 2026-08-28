# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""Incremental reciprocal-space accumulation for the live gridded preview.

Qt-free by construction: this runs inside a pvaccess consumer process, where
importing PyQt5 core-dumps (see the two-process note in
``consumers/ioc_rsm_parameter``).

Bounds are fixed before the first frame and never change. ``Gridder3D`` latches
``fixed_range`` on its first call when ``KeepData`` is set, after which points
outside the first batch's range are silently dropped -- and rebinning mid-scan
would change what a voxel *means*, so an accumulated volume could not be
interpreted. Changing bounds therefore requires a new accumulator, not a
``Clear()``: ``Gridder.Clear()`` zeroes the accumulators but keeps the latched
range.

Two grids are maintained over the *same accepted samples*: the full-resolution
one that gets saved, and an independently gridded coarse one for the preview.
Gridding twice costs one extra C call per frame on already-computed arrays;
block-reducing the fine grid instead would mean sweeping hundreds of megabytes
of accumulator on every snapshot, in the same process running the hot loop.

Aggregation matches the offline builder exactly -- an unweighted mean of the
gridded values, which are per-frame monitor-normalized when a monitor is
supplied. Summing instead would make a voxel bright wherever the scan path
happened to sample it more often, which is an artifact of the trajectory
rather than of the scattering.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import xrayutilities as xu

import dashpva.settings as app_settings
from dashpva.utils.gridder_access import gridder_coverage, gridder_numerator
from dashpva.utils.volume_io import finite_intensity_range, mean_with_nan_empty

__all__ = [
    "GridBoundsSpec",
    "LiveVolumeAccumulator",
    "PreviewPayload",
    "coarse_shape_for_budget",
]

#: Preview payload ceiling. The full-resolution volume never crosses the wire:
#: a 256^3 float32 volume is 64 MiB, and its float64 accumulators are 128 MiB
#: each, so publishing them at 1 Hz would saturate the link.
DEFAULT_PREVIEW_BUDGET_BYTES = app_settings.RSM_GRID_PREVIEW_BUDGET_BYTES

#: Gridder3D.axis() returns a bare float rather than an array for n == 1, and a
#: single-bin axis carries no spatial information anyway.
MIN_GRID_DIMENSION = 2


@dataclass(frozen=True)
class GridBoundsSpec:
    """Fixed HKL bounds and resolution, locked for an accumulation run."""

    hmin: float
    hmax: float
    kmin: float
    kmax: float
    lmin: float
    lmax: float
    nx: int
    ny: int
    nz: int

    def __post_init__(self) -> None:
        for low, high, axis in (
            (self.hmin, self.hmax, "H"),
            (self.kmin, self.kmax, "K"),
            (self.lmin, self.lmax, "L"),
        ):
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(f"{axis} bounds must be finite, got ({low}, {high}).")
            if not high > low:
                raise ValueError(
                    f"{axis} bounds must satisfy max > min, got ({low}, {high})."
                )
        for value, axis in ((self.nx, "nx"), (self.ny, "ny"), (self.nz, "nz")):
            if int(value) < MIN_GRID_DIMENSION:
                raise ValueError(
                    f"{axis} must be at least {MIN_GRID_DIMENSION}, got {value}."
                )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.nx), int(self.ny), int(self.nz))

    @property
    def voxel_count(self) -> int:
        return int(self.nx) * int(self.ny) * int(self.nz)

    def grid_bytes(self) -> int:
        """Resident bytes for the fine accumulators.

        Gridder3D allocates ``_gdata`` and ``_gnorm`` as float64 and offers no
        dtype knob, so this is a floor, not an estimate.
        """
        return self.voxel_count * 8 * 2

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (
            (self.hmax - self.hmin) / (self.nx - 1),
            (self.kmax - self.kmin) / (self.ny - 1),
            (self.lmax - self.lmin) / (self.nz - 1),
        )

    def contains_mask(self, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
        """Boolean mask of samples inside the grid.

        xrayutilities drops out-of-range points silently; counting them here is
        what turns "nothing appeared" into a number the user can act on.
        """
        return (
            (qx >= self.hmin) & (qx <= self.hmax)
            & (qy >= self.kmin) & (qy <= self.kmax)
            & (qz >= self.lmin) & (qz <= self.lmax)
        )


def coarse_shape_for_budget(
    shape: Sequence[int],
    budget_bytes: int = DEFAULT_PREVIEW_BUDGET_BYTES,
    itemsize: int = 4,
) -> tuple[int, int, int]:
    """Largest shape within ``budget_bytes`` keeping the fine aspect ratio.

    A single uniform divisor is wrong for an anisotropic grid -- it either
    overshoots the budget or wastes most of it -- so each axis is scaled by a
    common factor and then floored independently.
    """
    dims = tuple(max(MIN_GRID_DIMENSION, int(value)) for value in shape)
    max_voxels = max(MIN_GRID_DIMENSION ** 3, budget_bytes // itemsize)
    total = dims[0] * dims[1] * dims[2]
    if total <= max_voxels:
        return dims
    scale = (max_voxels / total) ** (1.0 / 3.0)
    coarse = [max(MIN_GRID_DIMENSION, min(dim, int(dim * scale))) for dim in dims]
    # Flooring three axes can still land just over budget; shrink the largest
    # axis until it fits rather than rescaling everything again.
    while coarse[0] * coarse[1] * coarse[2] > max_voxels:
        axis = int(np.argmax(coarse))
        if coarse[axis] <= MIN_GRID_DIMENSION:
            break
        coarse[axis] -= 1
    return (coarse[0], coarse[1], coarse[2])


@dataclass
class PreviewPayload:
    """Everything the GUI needs to render, and nothing full-resolution."""

    mean: np.ndarray                      # float32, NaN where uncovered
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]    # cell corner, not bin centre
    spacing: tuple[float, float, float]
    intensity_range: list
    frames_accepted: int
    frames_rejected: int
    points_binned: int
    points_out_of_range: int
    points_nonfinite: int
    points_masked: int
    voxels_filled: int
    aggregation: str
    geometry_fingerprint: str

    def nbytes(self) -> int:
        return int(self.mean.nbytes)


@dataclass
class _Counters:
    frames_accepted: int = 0
    frames_rejected: int = 0
    points_binned: int = 0
    points_out_of_range: int = 0
    points_nonfinite: int = 0
    points_masked: int = 0


@dataclass
class LiveVolumeAccumulator:
    """Fixed-bounds incremental gridder with a coarse preview alongside.

    Repeated passes over the same region need no special handling: KeepData
    keeps accumulating, so the mean is over every contributing pixel across all
    passes and coverage records how many contributed.
    """

    bounds: GridBoundsSpec
    monitor_name: Optional[str] = None
    mask: Optional[np.ndarray] = None
    geometry_fingerprint: str = ""
    preview_budget_bytes: int = DEFAULT_PREVIEW_BUDGET_BYTES
    counters: _Counters = field(default_factory=_Counters)

    def __post_init__(self) -> None:
        self._fine = self._new_gridder(self.bounds.shape)
        self._coarse_shape = coarse_shape_for_budget(
            self.bounds.shape, self.preview_budget_bytes
        )
        self._coarse = self._new_gridder(self._coarse_shape)
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)

    def _new_gridder(self, shape: Sequence[int]) -> "xu.Gridder3D":
        gridder = xu.Gridder3D(*[int(value) for value in shape])
        gridder.KeepData(True)
        # Must precede the first call: the range latches on first use.
        gridder.dataRange(
            self.bounds.hmin, self.bounds.hmax,
            self.bounds.kmin, self.bounds.kmax,
            self.bounds.lmin, self.bounds.lmax,
            fixed=True,
        )
        return gridder

    # -- accumulation ------------------------------------------------------

    def add_frame(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        qz: np.ndarray,
        intensity: np.ndarray,
        *,
        monitor: Optional[float] = None,
    ) -> int:
        """Grid one frame. Returns the number of samples actually binned.

        ``monitor`` divides the frame before gridding, matching the offline
        builder's per-frame I0 normalization; without it, frames taken at
        different exposure or attenuation disagree where they overlap.
        """
        qx = np.asarray(qx, dtype=float).ravel()
        qy = np.asarray(qy, dtype=float).ravel()
        qz = np.asarray(qz, dtype=float).ravel()
        values = np.asarray(intensity, dtype=float).ravel()
        if not (qx.size == qy.size == qz.size == values.size):
            self.counters.frames_rejected += 1
            raise ValueError(
                f"Frame arrays disagree in length: qx={qx.size}, qy={qy.size}, "
                f"qz={qz.size}, intensity={values.size}."
            )

        if monitor is not None:
            monitor = float(monitor)
            if not math.isfinite(monitor) or monitor <= 0:
                self.counters.frames_rejected += 1
                raise ValueError(
                    f"Monitor value must be finite and positive, got {monitor!r}. "
                    "Normalizing by it would corrupt the whole accumulation."
                )
            values = values / monitor

        keep = np.ones(values.shape, dtype=bool)

        if self.mask is not None:
            flat_mask = self.mask.ravel()
            if flat_mask.size != values.size:
                self.counters.frames_rejected += 1
                raise ValueError(
                    f"Mask has {flat_mask.size} pixels but the frame has "
                    f"{values.size}; a mask captured at a different detector "
                    f"ROI or binning cannot be applied to this scan."
                )
            # Excluded, never zeroed: a zeroed-but-present pixel would count as
            # a real intensity-zero measurement and bias its voxel low.
            keep &= ~flat_mask
            self.counters.points_masked += int(np.count_nonzero(flat_mask))

        finite = np.isfinite(qx) & np.isfinite(qy) & np.isfinite(qz) & np.isfinite(values)
        self.counters.points_nonfinite += int(np.count_nonzero(keep & ~finite))
        keep &= finite

        inside = self.bounds.contains_mask(qx, qy, qz)
        self.counters.points_out_of_range += int(np.count_nonzero(keep & ~inside))
        keep &= inside

        binned = int(np.count_nonzero(keep))
        if binned:
            kx, ky, kz, kv = qx[keep], qy[keep], qz[keep], values[keep]
            self._fine(kx, ky, kz, kv)
            self._coarse(kx, ky, kz, kv)
        self.counters.points_binned += binned
        self.counters.frames_accepted += 1
        return binned

    # -- results -----------------------------------------------------------

    @property
    def coverage(self) -> np.ndarray:
        """Per-voxel contribution count of the full-resolution grid."""
        return np.asarray(gridder_coverage(self._fine))

    @property
    def mean(self) -> np.ndarray:
        """Full-resolution per-voxel mean, NaN where nothing contributed."""
        return mean_with_nan_empty(gridder_numerator(self._fine), self.coverage)

    @property
    def aggregation(self) -> str:
        return (
            "mean_of_counts_over_monitor" if self.monitor_name else "unweighted_mean"
        )

    @property
    def origin(self) -> tuple[float, float, float]:
        """Cell-corner origin: gridder axes are bin centres, PyVista wants corners."""
        dh, dk, dl = self.bounds.spacing
        return (
            self.bounds.hmin - dh / 2.0,
            self.bounds.kmin - dk / 2.0,
            self.bounds.lmin - dl / 2.0,
        )

    def preview(self) -> PreviewPayload:
        """A coarse, budget-capped snapshot. Never carries the fine volume."""
        coarse_coverage = np.asarray(gridder_coverage(self._coarse))
        coarse_mean = mean_with_nan_empty(
            gridder_numerator(self._coarse), coarse_coverage
        ).astype(np.float32)
        nx, ny, nz = self._coarse_shape
        spacing = (
            (self.bounds.hmax - self.bounds.hmin) / (nx - 1),
            (self.bounds.kmax - self.bounds.kmin) / (ny - 1),
            (self.bounds.lmax - self.bounds.lmin) / (nz - 1),
        )
        origin = (
            self.bounds.hmin - spacing[0] / 2.0,
            self.bounds.kmin - spacing[1] / 2.0,
            self.bounds.lmin - spacing[2] / 2.0,
        )
        return PreviewPayload(
            mean=coarse_mean,
            shape=self._coarse_shape,
            origin=origin,
            spacing=spacing,
            intensity_range=finite_intensity_range(coarse_mean),
            voxels_filled=int(np.count_nonzero(coarse_coverage)),
            aggregation=self.aggregation,
            geometry_fingerprint=self.geometry_fingerprint,
            **asdict(self.counters),
        )

    def to_metadata(self, extra: Optional[dict] = None) -> dict:
        """Metadata matching the offline builder's volume conventions."""
        volume = self.mean
        metadata = {
            "voxel_spacing": list(self.bounds.spacing),
            "grid_origin": list(self.origin),
            "volume_shape": list(self.bounds.shape),
            "grid_dimensions_cells": list(self.bounds.shape),
            "array_order": "F",
            "axes_labels": ["H", "K", "L"],
            "coordinate_system": "HKL",
            "intensity_range": finite_intensity_range(volume),
            "aggregation": self.aggregation,
            "monitor_dataset": self.monitor_name or "",
            "geometry_fingerprint": self.geometry_fingerprint,
            "source": "live_grid",
            **{f"live_{key}": value for key, value in asdict(self.counters).items()},
        }
        if extra:
            metadata.update(extra)
        return metadata

    def clear(self) -> None:
        """Discard accumulated data, keeping the same locked bounds."""
        self._fine.Clear()
        self._coarse.Clear()
        self.counters = _Counters()
