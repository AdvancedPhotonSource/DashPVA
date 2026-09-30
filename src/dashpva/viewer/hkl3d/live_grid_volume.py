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

"""Flicker-free volume rendering of the live grid preview."""

from typing import Optional

import numpy as np
import pyvista as pyv
from matplotlib import colormaps
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

import dashpva.settings as app_settings

_OPACITY_SAMPLES = 256
_COLOR_SAMPLES = 64
_COLORMAP = "viridis"
# Slice of the transfer range below the colour floor reserved for empty voxels.
_EMPTY_BAND = 1.0 / 64.0


def display_values(mean: np.ndarray, log: bool) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (values, empty_mask, auto_low, auto_high) for a preview mean.

    Empty (NaN) and, in log mode, non-positive voxels are flagged in the mask;
    the renderer parks them below the colour floor where opacity is zero.
    """
    data = np.asarray(mean, dtype=np.float32)
    if log:
        values = np.full(data.shape, np.nan, dtype=np.float32)
        positive = data > 0
        values[positive] = np.log10(data[positive])
    else:
        values = data.copy()
    empty = ~np.isfinite(values)
    filled = values[~empty]
    if filled.size == 0:
        values[empty] = 0.0
        return values, empty, 0.0, 1.0
    low = float(np.percentile(filled, app_settings.RSM_GRID_AUTOSCALE_LOW_PERCENTILE))
    high = float(np.percentile(filled, app_settings.RSM_GRID_AUTOSCALE_HIGH_PERCENTILE))
    if not high > low:
        low, high = float(filled.min()), float(filled.max())
    if not high > low:
        high = low + 1.0
    values[empty] = low
    return values, empty, low, high


def transfer_range(low: float, high: float) -> tuple[float, float]:
    """Colour/opacity range: the empty band sits just under ``low``."""
    if not high > low:
        high = low + 1e-6
    return low - (high - low) * _EMPTY_BAND / (1.0 - _EMPTY_BAND), high


def opacity_ramp(min_opacity: float, max_opacity: float) -> np.ndarray:
    """Zero over the empty band; every filled voxel gets at least the floor opacity."""
    floor = max(min_opacity, app_settings.RSM_GRID_FILLED_MIN_OPACITY)
    band = max(1, int(round(_OPACITY_SAMPLES * _EMPTY_BAND)))
    ramp = np.empty(_OPACITY_SAMPLES)
    ramp[:band] = 0.0
    # Quadratic so background stays a faint haze and bright voxels stand out.
    t = np.linspace(0.0, 1.0, _OPACITY_SAMPLES - band) ** 2
    ramp[band:] = floor + (max(floor, max_opacity) - floor) * t
    return ramp


def park_values(values: np.ndarray, empty: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clamp filled voxels up to ``low`` (always visible) and empties into the band."""
    out = np.maximum(values, low)
    out[empty] = transfer_range(low, high)[0]
    return out


class LiveGridVolume:
    """Owns one volume actor; new previews update its scalars in place."""

    def __init__(self, plotter):
        self.plotter = plotter
        self.actor = None
        self._key = None
        self._raw: Optional[np.ndarray] = None
        self._empty: Optional[np.ndarray] = None
        self._clim = (0.0, 1.0)
        self._opacity = (0.0, 1.0)
        self.auto_range = (0.0, 1.0)

    def clear(self) -> None:
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        self.actor = None
        self._key = None
        self._raw = None
        self._empty = None

    def update(self, payload, *, log: bool, clim=None, opacity=(0.0, 1.0)) -> tuple[float, float]:
        """Show a preview. ``clim=None`` autoscales; returns the range in use."""
        values, empty, low, high = display_values(payload.mean, log)
        shape = tuple(int(v) for v in payload.shape)
        spacing = np.asarray(payload.spacing, dtype=float)
        origin = np.asarray(payload.origin, dtype=float)
        key = (shape, tuple(origin), tuple(spacing))
        rebuild = self.actor is None or key != self._key
        if rebuild:
            self.clear()
            self._key = key
        self.auto_range = (low, high)
        self._raw = values.flatten(order="F")
        self._empty = empty.flatten(order="F")
        self._clim = self.auto_range if clim is None else (float(clim[0]), float(clim[1]))
        self._opacity = (float(opacity[0]), float(opacity[1]))
        parked = park_values(self._raw, self._empty, *self._clim)

        if rebuild:
            # Point data at voxel centres, so in-place scalar updates stay valid.
            mesh = pyv.ImageData(
                dimensions=shape,
                spacing=tuple(spacing),
                origin=tuple(origin + spacing / 2.0),
            )
            mesh.point_data["intensity"] = parked
            self.actor = self.plotter.add_volume(
                mesh,
                scalars="intensity",
                cmap=_COLORMAP,
                clim=transfer_range(*self._clim),
                show_scalar_bar=True,
                reset_camera=True,
            )
            # Voxels are bins: no blending of a filled bin into its empty neighbours.
            self.actor.prop.interpolation_type = "nearest"
            # Opacity is per voxel. VTK's default unit distance is far larger
            # than an HKL voxel (~1e-5 r.l.u.), which renders it invisible.
            self.actor.prop.SetScalarOpacityUnitDistance(float(spacing.min()))
            self.plotter.add_axes(xlabel="H", ylabel="K", zlabel="L")
            self.plotter.show_bounds(
                xtitle="H Axis", ytitle="K Axis", ztitle="L Axis", fmt="%.3f"
            )
        else:
            self._push(parked)
        self._apply_transfer()
        return self._clim

    def set_display(self, clim, opacity) -> None:
        """Re-apply colour range/opacity to the current volume without new data."""
        if self.actor is None:
            return
        self._clim = (float(clim[0]), float(clim[1]))
        self._opacity = (float(opacity[0]), float(opacity[1]))
        self._push(park_values(self._raw, self._empty, *self._clim))
        self._apply_transfer()

    def _push(self, parked: np.ndarray) -> None:
        dataset = self.actor.mapper.dataset
        dataset.point_data["intensity"][:] = parked
        dataset.GetPointData().GetArray("intensity").Modified()

    def _apply_transfer(self) -> None:
        # Built directly in VTK: pyvista's LookupTable.apply_opacity +
        # VolumeProperty.apply_lookup_table leaves the volume fully transparent.
        low, high = self._clim
        floor, top = transfer_range(low, high)
        ramp = opacity_ramp(*self._opacity)
        opacity = vtkPiecewiseFunction()
        for value, alpha in zip(np.linspace(floor, top, ramp.size), ramp):
            opacity.AddPoint(float(value), float(alpha))
        color = vtkColorTransferFunction()
        cmap = colormaps[_COLORMAP]
        for t in np.linspace(0.0, 1.0, _COLOR_SAMPLES):
            r, g, b, _alpha = cmap(t)
            color.AddRGBPoint(float(low + t * (top - low)), r, g, b)
        color.AddRGBPoint(float(floor), *cmap(0.0)[:3])
        self.actor.prop.SetScalarOpacity(opacity)
        self.actor.prop.SetColor(color)
        self.actor.mapper.scalar_range = (floor, top)
        self.plotter.render()
