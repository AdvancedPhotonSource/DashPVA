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

"""Two-pass reciprocal-space volume gridding for DashPVA (RSM Volume Builder).

Merges one or more HDF5 scan files into a single interpolated 3D volume via
xrayutilities' Gridder3D, mirroring rsMap3D's QGridMapper.processMap on top of
DashPVA's per-frame Q-conversion in rsm_converter.py.

Output is in HKL (reciprocal lattice units), not Q: rsm_converter always passes
each file's own UB to Ang2Q.area, which returns hkl rather than Q.

Two passes are required, not merely an optimization. Gridder3D latches
``fixed_range = True`` on its first call when KeepData is set (gridder3d.py
``_checktransinput``), after which every point outside the first batch's range
is silently dropped. So the global bounds must be known and set via
``dataRange(..., fixed=True)`` before any gridding call.

Masked pixels are excluded from the arrays entirely, never zeroed:
``Gridder3D.data`` is a per-bin mean, so a zeroed-but-present pixel would count
as a real "intensity = 0" measurement and bias its voxel low.

Known limitations vs rsMap3D: no flat-field correction, no user-specified grid
crop (we always auto-range), and no detector tilt (rot1/rot2/rot3 from PONI
files are not applied -- a pre-existing rsm_converter limitation).
"""
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import psutil
import xrayutilities as xu

import dashpva.settings as app_settings
from dashpva.utils.gridder_access import gridder_coverage, gridder_numerator
from dashpva.utils.rsm_converter import RSMConverter
from dashpva.utils.volume_io import finite_intensity_range, mean_with_nan_empty

logger = logging.getLogger(__name__)

DEFAULT_ENERGY_RTOL = app_settings.RSM_GRID_ENERGY_RELATIVE_TOLERANCE
DEFAULT_UB_ATOL = app_settings.RSM_GRID_UB_ABSOLUTE_TOLERANCE
# Per-batch budget for the qx/qy/qz float64 arrays plus their mask-filtered
# copies. rsMap3D budgets bytes rather than frames for the same reason: a fixed
# frame count that is fine on a 512^2 detector needs many GB on a 2048^2 one.
DEFAULT_BATCH_BYTES = app_settings.RSM_GRID_BATCH_MEMORY_BYTES
CA_GROUP = "entry/data/metadata/ca"


class RSMMergeError(Exception):
    """Raised when gridding cannot proceed safely (bad grid size, no files,
    mask/detector shape mismatch, unusable monitor data, nothing left after
    masking). Always names the offending file or value."""


@dataclass
class GridBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def union(self, other: "GridBounds") -> "GridBounds":
        return GridBounds(
            xmin=min(self.xmin, other.xmin), xmax=max(self.xmax, other.xmax),
            ymin=min(self.ymin, other.ymin), ymax=max(self.ymax, other.ymax),
            zmin=min(self.zmin, other.zmin), zmax=max(self.zmax, other.zmax),
        )


@dataclass
class FileValidationInfo:
    filename: str
    energy_eV: float
    ub: np.ndarray
    num_frames: int
    detector_shape: Tuple[int, int]


@dataclass
class VolumeResult:
    volume: np.ndarray            # per-voxel mean, shape (nx, ny, nz)
    xaxis: np.ndarray             # bin centers
    yaxis: np.ndarray
    zaxis: np.ndarray
    per_file_info: List[FileValidationInfo]
    num_points_binned: int
    num_points_excluded_by_mask: int
    num_points_excluded_nonfinite: int
    monitor_dataset: Optional[str]
    mask_applied: bool
    mask_transposed: bool
    batch_bytes: int
    memory_estimate: "GridMemoryEstimate"
    # Per-voxel contribution count (Gridder3D's _gnorm). Lets a reader tell an
    # unmeasured voxel from a measured zero, and see where repeated passes
    # overlapped. Not sufficient to reconstruct a flux-weighted mean: from
    # sum(I/m) and N you cannot recover sum(I) and sum(m) separately unless the
    # monitor was constant across that voxel's contributions.
    coverage: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GridMemoryEstimate:
    peak_bytes: int
    grid_bytes: int
    batch_bytes: int
    output_bytes: int


def estimate_grid_memory(
    nx: int,
    ny: int,
    nz: int,
    detector_shapes: Sequence[Tuple[int, int]] = (),
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> GridMemoryEstimate:
    """Conservative peak for Gridder3D plus the largest processing batch.

    Gridder3D holds two float64 voxel arrays and ``.data`` materializes a
    third. The batch allowance covers coordinate/intensity originals, masked
    copies, and worst-case finite-filter copies.
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise RSMMergeError(f"Grid dimensions must be positive, got {(nx, ny, nz)}.")
    if batch_bytes < 1:
        raise RSMMergeError(f"batch_bytes must be positive, got {batch_bytes}.")

    voxels = int(nx) * int(ny) * int(nz)
    grid_bytes = voxels * 24
    output_bytes = voxels * 4
    actual_batch_bytes = batch_bytes
    if detector_shapes:
        actual_batch_bytes = 0
        for shape in detector_shapes:
            per_frame = (
                int(np.prod(shape)) * app_settings.RSM_GRID_WORKING_BYTES_PER_PIXEL
            )
            frames = max(1, batch_bytes // max(per_frame, 1))
            actual_batch_bytes = max(actual_batch_bytes, frames * per_frame)
    return GridMemoryEstimate(
        peak_bytes=grid_bytes + actual_batch_bytes,
        grid_bytes=grid_bytes,
        batch_bytes=actual_batch_bytes,
        output_bytes=output_bytes,
    )


def ensure_memory_available(
    estimate: GridMemoryEstimate,
    available_bytes: Optional[int] = None,
    max_fraction: float = app_settings.RSM_GRID_MAX_MEMORY_FRACTION,
) -> None:
    """Reject a build whose conservative peak exceeds the safe RAM budget."""
    if not np.isfinite(max_fraction) or not 0 < max_fraction <= 1:
        raise RSMMergeError(
            f"max_fraction must be finite and in (0, 1], got {max_fraction!r}."
        )
    available = int(
        psutil.virtual_memory().available if available_bytes is None else available_bytes
    )
    budget = int(available * max_fraction)
    if estimate.peak_bytes > budget:
        required_gib = estimate.peak_bytes / 1024**3
        budget_gib = budget / 1024**3
        raise RSMMergeError(
            f"Estimated peak memory is {required_gib:.2f} GiB, exceeding the "
            f"safe budget of {budget_gib:.2f} GiB ({max_fraction:.0%} of "
            "currently available RAM). Reduce nx, ny, or nz."
        )


def detector_shapes_for_files(
    filenames: Sequence[str],
) -> List[Tuple[int, int]]:
    """Read detector frame shapes once for memory estimation."""
    detector_shapes = []
    for filename in filenames:
        try:
            with h5py.File(filename, "r") as h5_file:
                data_shape = h5_file["entry/data/data"].shape
                if len(data_shape) != 3 or data_shape[0] < 1:
                    raise ValueError(
                        "entry/data/data must have shape "
                        f"(frames, direction1, direction2), got {data_shape}."
                    )
                detector_shapes.append(tuple(data_shape[1:]))
        except Exception as exc:
            raise RSMMergeError(f"Invalid scan file '{filename}': {exc}") from exc
    return detector_shapes


def estimate_files_memory(
    filenames: Sequence[str],
    nx: int,
    ny: int,
    nz: int,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
) -> GridMemoryEstimate:
    """Read only detector shapes and estimate the requested build's peak."""
    return estimate_grid_memory(
        nx,
        ny,
        nz,
        detector_shapes=detector_shapes_for_files(filenames),
        batch_bytes=batch_bytes,
    )


def _resolve_mask(mask_manager, use_mask: bool, mask_transposed: bool) -> Optional[np.ndarray]:
    """Return the boolean mask to apply (True = masked), or None.

    MaskManager holds the mask in the *viewer's display* orientation
    (see mask_manager.export_json_mask), and the viewer's transpose state is
    runtime-only -- it is never persisted, so an offline tool cannot detect it.
    The caller therefore states it explicitly via mask_transposed.
    """
    if not use_mask or mask_manager is None:
        return None
    mask = getattr(mask_manager, "mask", None)
    if mask is None:
        return None
    return mask.T if mask_transposed else mask


def _mask_and_ravel(qx, qy, qz, mask, intensity=None):
    """Flatten a (B, H, W) batch, dropping masked pixels from every array.

    Returns (qx, qy, qz, intensity_or_None, num_excluded). Pass intensity=None
    during bounds discovery, where only the coordinates are needed.
    """
    if mask is None:
        flat_i = None if intensity is None else intensity.ravel()
        qx_f, qy_f, qz_f = qx.ravel(), qy.ravel(), qz.ravel()
        excluded = 0
    else:
        if mask.shape != qx.shape[1:]:
            raise RSMMergeError(
                f"Mask shape {mask.shape} does not match detector frame shape "
                f"{qx.shape[1:]}. A mask captured at one detector binning/ROI "
                f"cannot be applied to a scan taken at a different one. If the "
                f"mask was made with the viewer transposed, set mask_transposed."
            )
        keep = ~mask
        qx_f, qy_f, qz_f = qx[:, keep].ravel(), qy[:, keep].ravel(), qz[:, keep].ravel()
        flat_i = None if intensity is None else intensity[:, keep].ravel()
        excluded = int(np.count_nonzero(mask)) * qx.shape[0]

    finite = np.isfinite(qx_f) & np.isfinite(qy_f) & np.isfinite(qz_f)
    if flat_i is not None:
        finite &= np.isfinite(flat_i)
    num_nonfinite = int(finite.size - np.count_nonzero(finite))
    if num_nonfinite:
        qx_f, qy_f, qz_f = qx_f[finite], qy_f[finite], qz_f[finite]
        if flat_i is not None:
            flat_i = flat_i[finite]
    return qx_f, qy_f, qz_f, flat_i, excluded, num_nonfinite


def _batch_size_for(detector_shape: Tuple[int, int], batch_bytes: int) -> int:
    """Frames per batch that keep the qx/qy/qz arrays and their masked copies
    within batch_bytes. Always at least 1."""
    per_frame = (
        int(np.prod(detector_shape)) * app_settings.RSM_GRID_WORKING_BYTES_PER_PIXEL
    )
    return max(1, batch_bytes // max(per_frame, 1))


def _iter_batches(n_frames: int, batch_size: int):
    for start in range(0, n_frames, batch_size):
        yield start, min(start + batch_size, n_frames)


def _read_monitor(h5_file, monitor_dataset: Optional[str], n_frames: int,
                   filename: str) -> Optional[np.ndarray]:
    """Per-frame monitor (I0) values from entry/data/metadata/ca/<name>.

    rsMap3D divides each image by its monitor counts before gridding; without
    it, merging scans taken at different exposure or attenuation produces an
    intensity step at the seam.
    """
    if not monitor_dataset:
        return None
    path = f"{CA_GROUP}/{monitor_dataset}"
    if path not in h5_file:
        raise RSMMergeError(
            f"Monitor dataset '{monitor_dataset}' not found at {path} in "
            f"'{filename}'. Available: {sorted(h5_file[CA_GROUP].keys()) if CA_GROUP in h5_file else 'none'}."
        )
    values = np.ravel(h5_file[path][...]).astype(float)
    if values.size < n_frames:
        raise RSMMergeError(
            f"Monitor '{monitor_dataset}' in '{filename}' has {values.size} "
            f"values but the scan has {n_frames} frames."
        )
    values = values[:n_frames]
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise RSMMergeError(
            f"Monitor '{monitor_dataset}' in '{filename}' must contain only "
            f"finite, strictly positive values."
        )
    return values


def compute_file_bounds(
    filename: str,
    converter: RSMConverter,
    mask: Optional[np.ndarray] = None,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[GridBounds, FileValidationInfo]:
    """Pass 1 for one file: sweep every frame computing Q only, to find this
    file's exact HKL bounds (mask-aware, so excluded pixels don't widen the
    range) and its energy/UB for the cross-file consistency check."""
    with h5py.File(filename, "r") as f:
        n_frames = f["entry/data/data"].shape[0]
        geom = converter.build_file_geometry(f)
        batch_size = _batch_size_for(tuple(geom.shape[1:]), batch_bytes)

        xmin = ymin = zmin = np.inf
        xmax = ymax = zmax = -np.inf
        for start, stop in _iter_batches(n_frames, batch_size):
            qx, qy, qz = converter.q_for_frames(geom, f, start, stop)
            qx_f, qy_f, qz_f, _, _, _ = _mask_and_ravel(qx, qy, qz, mask)
            if qx_f.size:
                xmin, xmax = min(xmin, float(qx_f.min())), max(xmax, float(qx_f.max()))
                ymin, ymax = min(ymin, float(qy_f.min())), max(ymax, float(qy_f.max()))
                zmin, zmax = min(zmin, float(qz_f.min())), max(zmax, float(qz_f.max()))
            if progress_cb is not None:
                progress_cb(stop, n_frames)

    if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
        raise RSMMergeError(
            f"No finite, unmasked pixels remain in '{filename}' -- cannot determine a "
            f"grid range. Check that the mask is not excluding the whole detector."
        )

    bounds = GridBounds(xmin, xmax, ymin, ymax, zmin, zmax)
    info = FileValidationInfo(filename=filename, energy_eV=geom.energy_eV, ub=geom.ub,
                               num_frames=n_frames, detector_shape=tuple(geom.shape[1:]))
    return bounds, info


def validate_consistency(
    infos: Sequence[FileValidationInfo],
    energy_rtol: float = DEFAULT_ENERGY_RTOL,
    ub_atol: float = DEFAULT_UB_ATOL,
    warn: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Warn (do not block) when merged files differ in energy or UB.

    These are warnings rather than errors because each file's own UB is applied
    per-file, so the output already lands in a common crystal-fixed HKL frame --
    that is precisely why rsMap3D applies UB per scan, and it does not block
    either. A re-refined UB between scans is routine, and HKL is geometrically
    energy-independent. What differing energy does affect is intensity
    comparability (cross-section, absorption), which is a interpretation caveat
    for the user, not a geometric error.

    Returns the list of warning messages (also logged, and passed to `warn`).
    """
    messages: List[str] = []
    if len(infos) <= 1:
        return messages
    ref = infos[0]
    for info in infos[1:]:
        rel_e = abs(info.energy_eV - ref.energy_eV) / abs(ref.energy_eV) if ref.energy_eV else 0.0
        if rel_e > energy_rtol:
            messages.append(
                f"Photon energy in '{info.filename}' is {info.energy_eV:.3f} eV vs "
                f"{ref.energy_eV:.3f} eV in '{ref.filename}' (relative difference "
                f"{rel_e:.2e}). Reflections still land at the same HKL, but "
                f"intensities are not directly comparable between these scans."
            )
        ub_diff = float(np.max(np.abs(info.ub - ref.ub)))
        if ub_diff > ub_atol:
            messages.append(
                f"UB matrix in '{info.filename}' differs from '{ref.filename}' by "
                f"up to {ub_diff:.2e} per element. Each file's own UB is applied, "
                f"so the merge stays in a common HKL frame -- but confirm this is a "
                f"re-refinement of the same crystal and not a different sample."
            )
    for msg in messages:
        logger.warning(msg)
        if warn is not None:
            warn(msg)
    return messages


def build_volume(
    filenames: Sequence[str],
    nx: int,
    ny: int,
    nz: int,
    use_mask: bool = True,
    mask_manager=None,
    mask_transposed: bool = False,
    monitor_dataset: Optional[str] = None,
    batch_bytes: int = DEFAULT_BATCH_BYTES,
    energy_rtol: float = DEFAULT_ENERGY_RTOL,
    ub_atol: float = DEFAULT_UB_ATOL,
    memory_limit_fraction: Optional[float] = app_settings.RSM_GRID_MAX_MEMORY_FRACTION,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    warn: Optional[Callable[[str], None]] = None,
    fixed_bounds: Optional[GridBounds] = None,
) -> VolumeResult:
    """Merge scan file(s) into one gridded HKL volume.

    Args:
        filenames: DashPVA HDF5 scan files (entry/data/data + metadata/HKL/...).
        nx, ny, nz: grid resolution, each >= 2 (xrayutilities' axis() returns a
            bare float rather than an array for n == 1).
        use_mask, mask_manager, mask_transposed: see _resolve_mask.
        monitor_dataset: name under entry/data/metadata/ca/ to divide intensity
            by (I0 normalization); None to skip.
        batch_bytes: per-batch memory budget for the coordinate arrays.
        energy_rtol, ub_atol: consistency-warning thresholds (see
            validate_consistency) -- these warn, they do not block.
        memory_limit_fraction: safe fraction of currently available RAM;
            ``None`` disables the guard for an explicitly managed script/HPC job.
        progress_cb: called with (frames_done, frames_total) across both passes.
        warn: called with each consistency-warning message.
    """
    if nx < 2 or ny < 2 or nz < 2:
        raise RSMMergeError(f"nx, ny, nz must each be >= 2 (got {nx}, {ny}, {nz}).")
    if not filenames:
        raise RSMMergeError("No input files given.")
    if batch_bytes < 1:
        raise RSMMergeError(f"batch_bytes must be positive, got {batch_bytes}.")

    memory_estimate = estimate_files_memory(
        filenames, nx, ny, nz, batch_bytes=batch_bytes
    )
    if memory_limit_fraction is not None:
        ensure_memory_available(memory_estimate, max_fraction=memory_limit_fraction)

    converter = RSMConverter()
    mask = _resolve_mask(mask_manager, use_mask, mask_transposed)

    # Pass 1: bounds + consistency. Frame counts are discovered here, so the
    # progress denominator only becomes exact once this pass completes.
    per_file_bounds: List[GridBounds] = []
    per_file_info: List[FileValidationInfo] = []
    for index, filename in enumerate(filenames):
        if progress_cb is not None:
            progress_cb(index, len(filenames))
        try:
            bounds, info = compute_file_bounds(
                filename, converter, mask=mask, batch_bytes=batch_bytes
            )
        except RSMMergeError:
            raise
        except Exception as exc:
            raise RSMMergeError(f"Invalid RSM metadata in '{filename}': {exc}") from exc
        per_file_bounds.append(bounds)
        per_file_info.append(info)

    validate_consistency(per_file_info, energy_rtol=energy_rtol, ub_atol=ub_atol, warn=warn)

    if fixed_bounds is not None:
        # Caller-supplied bounds: used to reproduce a live accumulation offline
        # at exactly the same grid. Points outside are dropped by the gridder,
        # the same way the live path drops them.
        global_bounds = fixed_bounds
    else:
        global_bounds = per_file_bounds[0]
        for b in per_file_bounds[1:]:
            global_bounds = global_bounds.union(b)

    # Pass 2: fixed-range accumulation.
    gridder = xu.Gridder3D(nx, ny, nz)
    gridder.KeepData(True)
    # Must precede the first gridder(...) call -- see module docstring.
    gridder.dataRange(global_bounds.xmin, global_bounds.xmax,
                       global_bounds.ymin, global_bounds.ymax,
                       global_bounds.zmin, global_bounds.zmax, fixed=True)

    total_frames = sum(info.num_frames for info in per_file_info)
    frames_done = 0
    num_points_binned = 0
    num_points_excluded_by_mask = 0
    num_points_excluded_nonfinite = 0

    for filename, info in zip(filenames, per_file_info):
        with h5py.File(filename, "r") as f:
            geom = converter.build_file_geometry(f)
            data_ds = f["entry/data/data"]
            monitor = _read_monitor(f, monitor_dataset, info.num_frames, filename)
            batch_size = _batch_size_for(info.detector_shape, batch_bytes)
            for start, stop in _iter_batches(info.num_frames, batch_size):
                qx, qy, qz = converter.q_for_frames(geom, f, start, stop)
                intensity = np.asarray(data_ds[start:stop], dtype=float)
                if monitor is not None:
                    intensity = intensity / monitor[start:stop, np.newaxis, np.newaxis]
                qx_f, qy_f, qz_f, int_f, n_excl, n_nonfinite = _mask_and_ravel(
                    qx, qy, qz, mask, intensity
                )
                if qx_f.size:
                    gridder(qx_f, qy_f, qz_f, int_f)
                num_points_binned += int(qx_f.size)
                num_points_excluded_by_mask += n_excl
                num_points_excluded_nonfinite += n_nonfinite
                frames_done += (stop - start)
                if progress_cb is not None:
                    progress_cb(frames_done, total_frames)

    if num_points_binned == 0:
        raise RSMMergeError("No finite, unmasked detector points remain to grid.")
    if num_points_excluded_nonfinite:
        message = (
            f"Excluded {num_points_excluded_nonfinite} point(s) with non-finite "
            "coordinates or intensity."
        )
        logger.warning(message)
        if warn is not None:
            warn(message)

    # Read the accumulators directly rather than via Gridder3D.data: .data
    # copies the whole numerator array and divides on every access, and it
    # leaves un-hit voxels at zero. mean_with_nan_empty applies the same
    # division but marks empty voxels NaN, and coverage is kept alongside.
    coverage = np.array(gridder_coverage(gridder), copy=True)
    return VolumeResult(
        volume=mean_with_nan_empty(gridder_numerator(gridder), coverage),
        coverage=coverage,
        xaxis=gridder.xaxis,
        yaxis=gridder.yaxis,
        zaxis=gridder.zaxis,
        per_file_info=per_file_info,
        num_points_binned=num_points_binned,
        num_points_excluded_by_mask=num_points_excluded_by_mask,
        num_points_excluded_nonfinite=num_points_excluded_nonfinite,
        monitor_dataset=monitor_dataset,
        mask_applied=mask is not None,
        mask_transposed=bool(mask is not None and mask_transposed),
        batch_bytes=batch_bytes,
        memory_estimate=memory_estimate,
    )


def list_monitor_candidates(filename: str) -> List[str]:
    """Names under entry/data/metadata/ca/ usable as a monitor, for the GUI."""
    try:
        with h5py.File(filename, "r") as f:
            if CA_GROUP not in f:
                return []
            return sorted(k for k, v in f[CA_GROUP].items() if isinstance(v, h5py.Dataset))
    except Exception:
        return []


def volume_result_to_metadata(result: VolumeResult, extra: Optional[dict] = None) -> dict:
    """Metadata dict for HDF5Loader.save_vol_to_h5, matching the convention in
    dash_analysis.py/hkl_3d_plot_dock.py (array_order 'F', grid_dimensions_cells
    = volume.shape, no transpose).

    grid_origin is shifted half a voxel from the axis minima: Gridder3D's axes
    are bin centers, but the PyVista viewer treats grid_origin as a cell corner.
    """
    xaxis, yaxis, zaxis = result.xaxis, result.yaxis, result.zaxis
    dx = float(xaxis[1] - xaxis[0]) if len(xaxis) > 1 else 0.0
    dy = float(yaxis[1] - yaxis[0]) if len(yaxis) > 1 else 0.0
    dz = float(zaxis[1] - zaxis[0]) if len(zaxis) > 1 else 0.0
    volume = result.volume
    ref = result.per_file_info[0]
    ub_matrices = np.stack([info.ub for info in result.per_file_info])

    metadata = {
        "voxel_spacing": [dx, dy, dz],
        "grid_origin": [float(xaxis[0]) - dx / 2.0,
                         float(yaxis[0]) - dy / 2.0,
                         float(zaxis[0]) - dz / 2.0],
        "volume_shape": list(volume.shape),
        "original_shape": list(ref.detector_shape),
        "array_order": "F",
        "grid_dimensions_cells": list(volume.shape),
        "axes_labels": ["H", "K", "L"],
        # Finite-only: empty voxels are NaN, and volume.min() would propagate
        # that, leaving intensity_range as [nan, nan] and breaking every
        # downstream colour scale that trusts it.
        "intensity_range": finite_intensity_range(volume),
        "source_files": [info.filename for info in result.per_file_info],
        "source_energies_eV": [info.energy_eV for info in result.per_file_info],
        "source_ub_matrices": ub_matrices.ravel().tolist(),
        "source_ub_matrices_shape": list(ub_matrices.shape),
        "coordinate_system": "HKL",
        "gridder": "xrayutilities.Gridder3D",
        "monitor_dataset": result.monitor_dataset or "",
        "mask_applied": result.mask_applied,
        "mask_transposed": result.mask_transposed,
        "batch_memory_bytes": result.batch_bytes,
        "num_points_binned": result.num_points_binned,
        "num_points_excluded_by_mask": result.num_points_excluded_by_mask,
        "num_points_excluded_nonfinite": result.num_points_excluded_nonfinite,
    }
    if extra:
        metadata.update(extra)
    return metadata
