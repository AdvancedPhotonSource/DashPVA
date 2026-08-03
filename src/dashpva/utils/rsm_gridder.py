# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
import xrayutilities as xu

from dashpva.utils.rsm_converter import RSMConverter

logger = logging.getLogger(__name__)

DEFAULT_ENERGY_RTOL = 1e-4
DEFAULT_UB_ATOL = 1e-4
# Per-batch budget for the qx/qy/qz float64 arrays plus their mask-filtered
# copies. rsMap3D budgets bytes rather than frames for the same reason: a fixed
# frame count that is fine on a 512^2 detector needs many GB on a 2048^2 one.
DEFAULT_BATCH_BYTES = 256 * 1024 * 1024
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
    volume: np.ndarray            # Gridder3D.data, shape (nx, ny, nz)
    xaxis: np.ndarray             # bin centers
    yaxis: np.ndarray
    zaxis: np.ndarray
    per_file_info: List[FileValidationInfo]
    num_points_binned: int
    num_points_excluded_by_mask: int


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
        return qx.ravel(), qy.ravel(), qz.ravel(), flat_i, 0
    if mask.shape != qx.shape[1:]:
        raise RSMMergeError(
            f"Mask shape {mask.shape} does not match detector frame shape "
            f"{qx.shape[1:]}. A mask captured at one detector binning/ROI "
            f"cannot be applied to a scan taken at a different one. If the "
            f"mask was made with the viewer transposed, set mask_transposed."
        )
    keep = ~mask
    flat_i = None if intensity is None else intensity[:, keep].ravel()
    num_excluded = int(np.count_nonzero(mask)) * qx.shape[0]
    return qx[:, keep].ravel(), qy[:, keep].ravel(), qz[:, keep].ravel(), flat_i, num_excluded


def _batch_size_for(detector_shape: Tuple[int, int], batch_bytes: int) -> int:
    """Frames per batch that keep the qx/qy/qz arrays and their masked copies
    within batch_bytes. Always at least 1."""
    per_frame = int(np.prod(detector_shape)) * 8 * 6  # 3 coord arrays, doubled by masked copies
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
    if np.any(values == 0):
        raise RSMMergeError(
            f"Monitor '{monitor_dataset}' in '{filename}' contains zero values; "
            f"cannot normalize (would divide by zero)."
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
            qx_f, qy_f, qz_f, _, _ = _mask_and_ravel(qx, qy, qz, mask)
            if qx_f.size:
                xmin, xmax = min(xmin, float(qx_f.min())), max(xmax, float(qx_f.max()))
                ymin, ymax = min(ymin, float(qy_f.min())), max(ymax, float(qy_f.max()))
                zmin, zmax = min(zmin, float(qz_f.min())), max(zmax, float(qz_f.max()))
            if progress_cb is not None:
                progress_cb(stop, n_frames)

    if not np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax]).all():
        raise RSMMergeError(
            f"No unmasked pixels remain in '{filename}' -- cannot determine a "
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
    progress_cb: Optional[Callable[[int, int], None]] = None,
    warn: Optional[Callable[[str], None]] = None,
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
        progress_cb: called with (frames_done, frames_total) across both passes.
        warn: called with each consistency-warning message.
    """
    if nx < 2 or ny < 2 or nz < 2:
        raise RSMMergeError(f"nx, ny, nz must each be >= 2 (got {nx}, {ny}, {nz}).")
    if not filenames:
        raise RSMMergeError("No input files given.")

    converter = RSMConverter()
    mask = _resolve_mask(mask_manager, use_mask, mask_transposed)

    # Pass 1: bounds + consistency. Frame counts are discovered here, so the
    # progress denominator only becomes exact once this pass completes.
    per_file_bounds: List[GridBounds] = []
    per_file_info: List[FileValidationInfo] = []
    for index, filename in enumerate(filenames):
        if progress_cb is not None:
            progress_cb(index, len(filenames))
        bounds, info = compute_file_bounds(filename, converter, mask=mask,
                                            batch_bytes=batch_bytes)
        per_file_bounds.append(bounds)
        per_file_info.append(info)

    validate_consistency(per_file_info, energy_rtol=energy_rtol, ub_atol=ub_atol, warn=warn)

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
                qx_f, qy_f, qz_f, int_f, n_excl = _mask_and_ravel(qx, qy, qz, mask, intensity)
                if qx_f.size:
                    gridder(qx_f, qy_f, qz_f, int_f)
                num_points_binned += int(qx_f.size)
                num_points_excluded_by_mask += n_excl
                frames_done += (stop - start)
                if progress_cb is not None:
                    progress_cb(frames_done, total_frames)

    return VolumeResult(
        volume=gridder.data,
        xaxis=gridder.xaxis,
        yaxis=gridder.yaxis,
        zaxis=gridder.zaxis,
        per_file_info=per_file_info,
        num_points_binned=num_points_binned,
        num_points_excluded_by_mask=num_points_excluded_by_mask,
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
        "intensity_range": [float(volume.min()), float(volume.max())] if volume.size else [0.0, 0.0],
        "source_files": [info.filename for info in result.per_file_info],
        "energy_eV": ref.energy_eV,
        # Flat 9 elements so save_vol_to_h5 stores it as a numeric dataset
        # rather than a stringified repr.
        "ub_matrix": ref.ub.ravel().tolist(),
        "num_points_binned": result.num_points_binned,
        "num_points_excluded_by_mask": result.num_points_excluded_by_mask,
    }
    if extra:
        metadata.update(extra)
    return metadata
