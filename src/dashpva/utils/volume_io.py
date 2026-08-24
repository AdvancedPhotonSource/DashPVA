# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Qt-free reciprocal-space volume persistence.

Deliberately imports nothing from PyQt5. The live grid accumulator runs inside
a pvaccess consumer process, and this repository documents that mixing
pvaccess with PyQt5 in one process core-dumps (see the two-process note in
``consumers/ioc_rsm_parameter``). ``HDF5Writer`` and ``HDF5Loader`` delegate
here so the consumer can write a volume without importing Qt at all.

The layout matches what ``HDF5Loader.save_vol_to_h5`` has always produced --
``/entry/data/data`` plus ``/entry/data/metadata`` -- so Workbench opens these
files with no viewer change. Coverage is an optional sibling dataset at
``/entry/data/coverage``; ``load_h5_volume_3d`` addresses ``data`` by name and
iterates only ``metadata``, so the extra dataset is inert to existing readers.

Empty voxels are NaN, not zero. ``Gridder3D`` leaves un-hit bins at zero, which
is indistinguishable from a measured zero: a voxel nothing was ever scattered
into would otherwise read as a real, confident measurement of no intensity.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import h5py
import numpy as np

__all__ = [
    "DATA_PATH",
    "COVERAGE_PATH",
    "finite_intensity_range",
    "mean_with_nan_empty",
    "save_volume",
]

DATA_PATH = "entry/data/data"
COVERAGE_PATH = "entry/data/coverage"

#: Chunk targets ~1 MiB of float32 so a large volume streams to disk instead of
#: being handed to h5py as one enormous contiguous write.
_TARGET_CHUNK_ELEMENTS = 262_144


def mean_with_nan_empty(
    numerator: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """Per-voxel mean, NaN where nothing contributed.

    Mirrors ``Gridder3D.data`` (``_gdata / _gnorm`` where ``_gnorm != 0``) but
    marks empty voxels NaN rather than leaving them at zero.
    """
    numerator = np.asarray(numerator, dtype=float)
    coverage = np.asarray(coverage)
    if numerator.shape != coverage.shape:
        raise ValueError(
            f"Numerator shape {numerator.shape} and coverage shape "
            f"{coverage.shape} must match."
        )
    result = np.full(numerator.shape, np.nan, dtype=float)
    filled = coverage != 0
    np.divide(numerator, coverage, out=result, where=filled)
    result[~filled] = np.nan
    return result


def finite_intensity_range(volume: np.ndarray) -> list[float]:
    """``[min, max]`` over finite voxels only.

    ``volume.min()`` propagates NaN, so once empty voxels are NaN the stored
    ``intensity_range`` becomes ``[nan, nan]`` and every downstream colour
    scale that trusts it breaks. An all-empty volume reports ``[0, 0]``.
    """
    data = np.asarray(volume, dtype=float)
    if data.size == 0:
        return [0.0, 0.0]
    finite = np.isfinite(data)
    if not finite.any():
        return [0.0, 0.0]
    return [float(np.min(data[finite])), float(np.max(data[finite]))]


def _chunks_for(shape: Sequence[int]) -> Optional[tuple[int, ...]]:
    """A chunk shape near the target size, or None for arrays already small."""
    dims = tuple(int(value) for value in shape)
    if not dims or any(value <= 0 for value in dims):
        return None
    if int(np.prod(dims)) <= _TARGET_CHUNK_ELEMENTS:
        return None
    chunk = list(dims)
    # Shrink the slowest-varying axis first so each chunk stays contiguous.
    for axis in range(len(chunk)):
        if int(np.prod(chunk)) <= _TARGET_CHUNK_ELEMENTS:
            break
        remaining = int(np.prod(chunk[axis + 1:])) or 1
        chunk[axis] = max(1, min(chunk[axis], _TARGET_CHUNK_ELEMENTS // remaining))
    return tuple(chunk)


def _write_metadata(group: h5py.Group, metadata: Mapping[str, Any]) -> None:
    """Write metadata preserving numeric type.

    Numbers must land as numbers; stringifying them reintroduces the repr bug
    fixed in 687e943, where readers got back values like '[6 6 6]'.
    """
    text_dtype = h5py.string_dtype(encoding="utf-8")
    for key, value in metadata.items():
        if value is None:
            continue
        try:
            if isinstance(value, str):
                group.create_dataset(key, data=value, dtype=text_dtype)
            elif isinstance(value, (bool, int, float, np.number)):
                group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)
                if array.size and array.dtype.kind in ("i", "u", "f", "b"):
                    group.create_dataset(key, data=array)
                else:
                    # h5py has no conversion path from a numpy <U dtype; hand it
                    # a list of Python str and let the vlen dtype do the work.
                    group.create_dataset(
                        key,
                        data=[str(item) for item in np.asarray(value).ravel()],
                        dtype=text_dtype,
                    )
            else:
                group.create_dataset(key, data=str(value), dtype=text_dtype)
        except (TypeError, ValueError):
            # A failed create can still have claimed the name, so clear it
            # before the stringified retry rather than colliding with itself.
            if key in group:
                del group[key]
            group.create_dataset(key, data=str(value), dtype=text_dtype)


def save_volume(
    file_path: str,
    volume: np.ndarray,
    *,
    coverage: Optional[np.ndarray] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    compress: bool = True,
) -> bool:
    """Write a gridded volume, and optionally its coverage, to HDF5.

    Args:
        file_path: destination; overwritten.
        volume: 2-D slice or 3-D volume. Stored float32, NaN preserved.
        coverage: per-voxel contribution count, same shape as ``volume``.
        metadata: written under ``/entry/data/metadata``. ``intensity_range``
            is filled from finite voxels only when absent.
        compress: gzip the datasets (chunked writes only).

    Returns True on success.
    """
    array = np.asarray(volume)
    if array.size == 0:
        raise ValueError("Volume array cannot be empty.")
    if array.ndim not in (2, 3):
        raise ValueError(f"Volume must be 2-D or 3-D, got ndim={array.ndim}.")

    meta: dict[str, Any] = dict(metadata or {})
    meta.setdefault("data_type", "volume" if array.ndim == 3 else "slice")
    meta.setdefault("creation_timestamp", str(np.datetime64("now")))
    if array.ndim == 3:
        meta.setdefault("volume_shape", [int(value) for value in array.shape])
    else:
        meta.setdefault("slice_shape", [int(value) for value in array.shape])
    meta.setdefault("intensity_range", finite_intensity_range(array))

    stored = array.astype(np.float32, copy=False)
    chunks = _chunks_for(stored.shape)
    options: dict[str, Any] = {}
    if chunks is not None:
        options["chunks"] = chunks
        if compress:
            options["compression"] = "gzip"
            options["compression_opts"] = 4

    if coverage is not None:
        coverage_array = np.asarray(coverage)
        if coverage_array.shape != stored.shape:
            raise ValueError(
                f"Coverage shape {coverage_array.shape} must match volume shape "
                f"{stored.shape}."
            )

    with h5py.File(file_path, "w") as handle:
        entry_group = handle.require_group("entry")
        data_group = entry_group.require_group("data")
        data_group.create_dataset("data", data=stored, **options)
        # Discovery attributes HDF5Loader.load_h5_volume_3d reads.
        entry_group.attrs["data_type"] = meta["data_type"]
        data_group.attrs["array_rank"] = stored.ndim
        data_group.attrs["array_shape"] = np.array(stored.shape, dtype=np.int64)
        if coverage is not None:
            coverage_array = np.asarray(coverage)
            # Counts are integral; keep them integral so "how many frames hit
            # this voxel" stays exactly answerable.
            if coverage_array.dtype.kind == "f" and np.all(
                coverage_array == np.floor(coverage_array)
            ):
                coverage_array = coverage_array.astype(np.uint32)
            data_group.create_dataset("coverage", data=coverage_array, **options)
        _write_metadata(data_group.require_group("metadata"), meta)
    return True
