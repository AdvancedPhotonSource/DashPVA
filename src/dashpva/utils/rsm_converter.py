# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import xrayutilities as xu

import dashpva.settings as app_settings

"""Utilities for converting detector frames into reciprocal space (RSM).
This module provides a concise RSMConverter focused on the essential
pipeline: reading metadata, building geometry, and computing Q-space.
"""


@dataclass
class FileGeometry:
    """Frame-invariant xrayutilities geometry for one HDF5 scan file.

    Energy, UB, beam directions, and detector setup are constant across every
    frame of a scan; only sample/detector circle *positions* vary per frame.
    Built once via RSMConverter.build_file_geometry() so callers processing
    many frames (e.g. a gridding pass) don't rebuild QConversion/HXRD on every
    frame the way create_rsm() does (kept as-is there for backward
    compatibility with existing single-frame callers).
    """
    hxrd: "xu.HXRD"
    ub: np.ndarray
    energy_eV: float
    shape: Tuple[int, ...]


class Data:
    """Simple container for 3D points and intensities."""
    def __init__(self, points: np.ndarray, intensities: np.ndarray, metadata: dict = None, num_images: int = 0, shape: tuple = None):
        self.points = points
        self.intensities = intensities
        self.metadata = metadata
        self.num_images = num_images
        self.shape = shape

class RSMConverter:
    """Compute reciprocal space mapping (RSM) from HDF5 detector data.

    Responsibilities:
    - Read HKL metadata and detector setup from file
    - Build xrayutilities geometry and convert Angles → Q-space
    - Provide a compact public API for loading and computing
    """

    # Public API
    def load_data(self, file_path: Optional[str] = None) -> Data:
        """Load points and intensities from an HDF5 file and return Data.
        If precomputed qx, qy, qz are absent, compute Q-space via RSM.
        """
        path = file_path or getattr(self, "file_path", None)
        if not path:
            raise FileNotFoundError("No file path provided to load_data and none set in DashAnalysis.")
        points_3d, intensities, num_images, shape = self.load_h5_to_3d(path)
        return Data(points=points_3d, intensities=intensities, metadata=None, num_images=num_images, shape=shape)

    def load_h5_to_3d(self, filename: str):
        """Load q-points and intensities; compute Q if not present in file."""
        with h5py.File(filename, "r") as f:
            shape = f["entry/data/data"].shape
        qx = self.take_data_by_key(filename, "qx")
        if qx is not None:
            qy = self.take_data_by_key(filename, "qy")
            qz = self.take_data_by_key(filename, "qz")
            q = np.column_stack((np.ravel(qx), np.ravel(qy), np.ravel(qz)))
        else:
            q = self.get_q_points(filename)
        intensity = self.get_intensity(filename)
        return q, intensity, shape[0], shape[1:]

    def create_rsm(self, filename: str, frame: int):
        """Create reciprocal space mapping for a single frame using xrayutilities."""
        try:
            with h5py.File(filename, "r") as f:
                shape = f["entry/data/data"].shape
                sc_dir, sc_pos, dc_dir, dc_pos = self.get_sample_and_detector_circles(f, frame)
                primary, inplane, surface, ub, energy = self.get_physics_params(f)
                qconv = xu.experiment.QConversion(sc_dir, dc_dir, primary)
                hxrd = xu.HXRD(inplane, surface, en=energy, qconv=qconv)
                p_dir1, p_dir2, cch1, cch2, nch1, nch2, pw1, pw2, dist, roi = self.get_detector_setup(f, shape)
                hxrd.Ang2Q.init_area(
                    p_dir1, p_dir2,
                    cch1=cch1, cch2=cch2,
                    Nch1=nch1, Nch2=nch2,
                    pwidth1=pw1, pwidth2=pw2,
                    distance=dist,
                    roi=roi,
                )
                angles = [*sc_pos, *dc_pos]
                return hxrd.Ang2Q.area(*angles, UB=ub)
        except Exception:
            raise

    def get_q_points(self, filename: str) -> np.ndarray:
        """Compute Q points for all frames and return flattened (N, 3) array."""
        with h5py.File(filename, "r") as f:
            n_frames = f["entry/data/data"].shape[0]
        qxyz_stack = np.stack([self.create_rsm(filename, i) for i in range(n_frames)], axis=0)
        return np.column_stack((
            qxyz_stack[:, 0, ...].ravel(),
            qxyz_stack[:, 1, ...].ravel(),
            qxyz_stack[:, 2, ...].ravel(),
        ))

    # Physics & Metadata
    def get_physics_params(self, h5_file: h5py.File):
        """Extract beam directions, UB matrix, and energy from HKL metadata."""
        meta = h5_file["entry/data/metadata/HKL"]
        primary = [float(self._static_numeric(
            meta[f"PRIMARY_BEAM_DIRECTION/AXIS_NUMBER_{i}"], 1,
            f"primary beam direction axis {i}",
        )[0]) for i in range(1, 4)]
        inplane = [float(self._static_numeric(
            meta[f"INPLANE_REFERENCE_DIRECITON/AXIS_NUMBER_{i}"], 1,
            f"in-plane reference direction axis {i}",
        )[0]) for i in range(1, 4)]
        surface = [float(self._static_numeric(
            meta[f"SAMPLE_SURFACE_NORMAL_DIRECITON/AXIS_NUMBER_{i}"], 1,
            f"sample surface normal direction axis {i}",
        )[0]) for i in range(1, 4)]
        ub = self.get_ub_matrix_from_file(h5_file)
        energy = float(self._static_numeric(
            meta["SPEC/ENERGY_VALUE"], 1, "photon energy",
            rtol=app_settings.RSM_GRID_ENERGY_RELATIVE_TOLERANCE,
        )[0]) * 1000.0
        vectors = np.asarray([primary, inplane, surface], dtype=float)
        if not np.isfinite(vectors).all():
            raise ValueError("Beam and sample reference directions must be finite.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Beam and sample reference directions must be non-zero.")
        if np.linalg.norm(np.cross(inplane, surface)) == 0:
            raise ValueError("In-plane and sample-surface directions must not be parallel.")
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError(f"Photon energy must be finite and positive, got {energy!r} eV.")
        if not np.isfinite(ub).all() or np.linalg.matrix_rank(ub) < 3:
            raise ValueError("UB matrix must be finite and full rank.")
        return primary, inplane, surface, ub, energy

    def get_intensity(self, filename: str) -> np.ndarray:
        """Return detector intensities as a flattened array."""
        with h5py.File(filename, "r") as f:
            return f["entry/data/data"][:].ravel()

    # Detector parameters
    def get_detector_setup(self, h5_file: h5py.File, shape: tuple):
        """Return detector setup: directions, center pixels, size, pixel widths, distance, roi."""
        det = h5_file["entry/data/metadata/HKL/DETECTOR_SETUP"]
        roi = [0, shape[1], 0, shape[2]]
        p_dir1 = self._static_str(det["PIXEL_DIRECTION_1"], "pixel direction 1")
        p_dir2 = self._static_str(det["PIXEL_DIRECTION_2"], "pixel direction 2")
        # Flatten before indexing — PVs stored per-frame may have shape (n_frames, N)
        cch = self._static_numeric(det["CENTER_CHANNEL_PIXEL"], 2, "detector center")
        cch1 = float(cch[0])
        cch2 = float(cch[1])
        size = self._static_numeric(det["SIZE"], 2, "detector size")
        pw1 = float(size[0]) / float(shape[1])
        pw2 = float(size[1]) / float(shape[2])
        dist = float(self._static_numeric(det["DISTANCE"], 1, "detector distance")[0])
        numeric = np.asarray([cch1, cch2, pw1, pw2, dist], dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Detector center, pixel sizes, and distance must be finite.")
        if pw1 <= 0 or pw2 <= 0 or dist <= 0:
            raise ValueError(
                f"Detector pixel sizes and distance must be positive, got "
                f"({pw1}, {pw2}, {dist})."
            )
        return p_dir1, p_dir2, cch1, cch2, shape[1], shape[2], pw1, pw2, dist, roi

    # Geometry extraction
    def _read_position(self, h5_file: h5py.File, axis_path: str, frame: int) -> float:
        """Read the per-frame position for a circle axis.

        Requires the dedicated POSITION dataset under the axis group. Using a
        different circle's position as fallback would silently corrupt HKL.
        """
        pos_path = f"{axis_path}/POSITION"
        if pos_path in h5_file:
            arr = np.ravel(h5_file[pos_path][...])
            if arr.size == 1:
                value = float(arr[0])
            elif frame >= arr.size:
                raise ValueError(
                    f"Circle position dataset '{pos_path}' has {arr.size} values "
                    f"but frame {frame} was requested."
                )
            else:
                value = float(arr[frame])
            if not np.isfinite(value):
                raise ValueError(f"Circle position dataset '{pos_path}' contains non-finite values.")
            return value
        raise KeyError(f"No POSITION dataset found at {pos_path}")

    def get_sample_and_detector_circles(self, h5_file: h5py.File, frame: int):
        """Return lists of direction strings and positions for sample and detector circles."""
        sample_paths, detector_paths = self._resolve_circle_paths(h5_file)
        sc_dir = [self._static_str(
            h5_file[f"{p}/DIRECTION_AXIS"], f"sample circle direction at {p}"
        ) for p in sample_paths]
        dc_dir = [self._static_str(
            h5_file[f"{p}/DIRECTION_AXIS"], f"detector circle direction at {p}"
        ) for p in detector_paths]
        sc_pos = [self._read_position(h5_file, p, frame) for p in sample_paths]
        dc_pos = [self._read_position(h5_file, p, frame) for p in detector_paths]
        return sc_dir, sc_pos, dc_dir, dc_pos

    def _resolve_circle_paths(self, h5_file: h5py.File) -> Tuple[List[str], List[str]]:
        """Return (sample_paths, detector_paths): the resolved HKL group paths
        for sample/detector circles, in circle order. Prefers the numbered
        SAMPLE_CIRCLE_AXIS_1..4 / DETECTOR_CIRCLE_AXIS_1..2 groups, falling
        back to the legacy MU/ETA/CHI/PHI / NU/DELTA named groups — same
        precedence get_sample_and_detector_circles has always used.
        """
        hkl_base = "entry/data/metadata/HKL"
        sample_priority = ["MU", "ETA", "CHI", "PHI"]
        detector_priority = ["NU", "DELTA"]

        hkl = h5_file[hkl_base]
        unsupported = [
            name for name in hkl
            if (
                name.startswith("SAMPLE_CIRCLE_AXIS_")
                and self._axis_number(name) > 4
            ) or (
                name.startswith("DETECTOR_CIRCLE_AXIS_")
                and self._axis_number(name) > 2
            )
        ]
        if unsupported:
            raise ValueError(
                "Unsupported circle metadata: " + ", ".join(sorted(unsupported))
                + ". This converter currently supports up to four sample and "
                "two detector circles; geometry-agnostic support is tracked in #132."
            )

        sample_paths = [f"{hkl_base}/SAMPLE_CIRCLE_AXIS_{i}" for i in range(1, 5)
                         if f"{hkl_base}/SAMPLE_CIRCLE_AXIS_{i}" in h5_file]
        if not sample_paths:
            sample_paths = [f"{hkl_base}/{axis}" for axis in sample_priority
                             if f"{hkl_base}/{axis}" in h5_file]

        detector_paths = [f"{hkl_base}/DETECTOR_CIRCLE_AXIS_{i}" for i in range(1, 3)
                           if f"{hkl_base}/DETECTOR_CIRCLE_AXIS_{i}" in h5_file]
        if not detector_paths:
            detector_paths = [f"{hkl_base}/{axis}" for axis in detector_priority
                               if f"{hkl_base}/{axis}" in h5_file]

        return sample_paths, detector_paths

    @staticmethod
    def _axis_number(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[-1])
        except ValueError:
            return 10**9

    def get_circle_directions(self, h5_file: h5py.File) -> Tuple[List[str], List[str]]:
        """Return (sc_dir, dc_dir): the frame-invariant circle axis-direction
        strings. Split out of get_sample_and_detector_circles so a caller
        processing many frames of the same file can resolve directions once
        instead of once per frame (see build_file_geometry)."""
        sample_paths, detector_paths = self._resolve_circle_paths(h5_file)
        sc_dir = [self._static_str(
            h5_file[f"{p}/DIRECTION_AXIS"], f"sample circle direction at {p}"
        ) for p in sample_paths]
        dc_dir = [self._static_str(
            h5_file[f"{p}/DIRECTION_AXIS"], f"detector circle direction at {p}"
        ) for p in detector_paths]
        return sc_dir, dc_dir

    def get_circle_positions_batch(self, h5_file: h5py.File, start: int, stop: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (sc_pos_arrays, dc_pos_arrays): one 1-D array of length
        (stop - start) per circle axis, covering frames [start, stop). Used by
        q_for_frames to make a single batched Ang2Q.area call instead of one
        call per frame."""
        sample_paths, detector_paths = self._resolve_circle_paths(h5_file)
        sc_pos = [self._read_position_batch(h5_file, p, start, stop) for p in sample_paths]
        dc_pos = [self._read_position_batch(h5_file, p, start, stop) for p in detector_paths]
        return sc_pos, dc_pos

    def _read_position_batch(self, h5_file: h5py.File, axis_path: str, start: int, stop: int) -> np.ndarray:
        """Batch version of _read_position: positions for frames [start, stop)."""
        pos_path = f"{axis_path}/POSITION"
        if pos_path in h5_file:
            arr = np.ravel(h5_file[pos_path][...])
            if arr.size == 1:
                values = np.full(stop - start, float(arr[0]), dtype=float)
            elif arr.size < stop:
                raise ValueError(
                    f"Circle position dataset '{pos_path}' has {arr.size} values "
                    f"but frame {stop - 1} was requested."
                )
            else:
                values = arr[start:stop].astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"Circle position dataset '{pos_path}' contains non-finite values.")
            return values
        raise KeyError(f"No POSITION dataset found at {pos_path}")

    def build_file_geometry(self, h5_file: h5py.File) -> FileGeometry:
        """Build QConversion/HXRD/Ang2Q.init_area ONCE for this file, instead
        of once per frame the way create_rsm() does. Reuse the returned
        FileGeometry across every frame/batch of this file via q_for_frames."""
        shape = h5_file["entry/data/data"].shape
        sc_dir, dc_dir = self.get_circle_directions(h5_file)
        primary, inplane, surface, ub, energy = self.get_physics_params(h5_file)
        qconv = xu.experiment.QConversion(sc_dir, dc_dir, primary)
        hxrd = xu.HXRD(inplane, surface, en=energy, qconv=qconv)
        p_dir1, p_dir2, cch1, cch2, nch1, nch2, pw1, pw2, dist, roi = self.get_detector_setup(h5_file, shape)
        hxrd.Ang2Q.init_area(
            p_dir1, p_dir2,
            cch1=cch1, cch2=cch2,
            Nch1=nch1, Nch2=nch2,
            pwidth1=pw1, pwidth2=pw2,
            distance=dist,
            roi=roi,
        )
        return FileGeometry(hxrd=hxrd, ub=ub, energy_eV=energy, shape=shape)

    def q_for_frames(self, geom: FileGeometry, h5_file: h5py.File, start: int, stop: int):
        """Angles -> Q for frames [start, stop) via a single batched
        Ang2Q.area call. Returns (qx, qy, qz), each shaped (stop - start, H, W)
        -- not raveled and not mask-filtered, so callers can boolean-index out
        masked pixels before raveling.
        """
        sc_pos, dc_pos = self.get_circle_positions_batch(h5_file, start, stop)
        angles = [*sc_pos, *dc_pos]
        qx, qy, qz = geom.hxrd.Ang2Q.area(*angles, UB=geom.ub)
        n = stop - start
        if n == 1 and qx.ndim == 2:
            # xrayutilities' Ang2Q.area drops the leading batch axis when
            # Npoints == 1 (experiment.py: `if Npoints == 1: ... return
            # qpos[:, :, 0], ...`) -- restore it so callers can always rely on
            # a (n_frames, H, W) shape, even for a batch of exactly one frame
            # (e.g. the last partial batch of a scan).
            qx, qy, qz = qx[np.newaxis, ...], qy[np.newaxis, ...], qz[np.newaxis, ...]
        return qx, qy, qz

    # UB helpers
    def get_ub_matrix_from_file(self, h5_file: h5py.File) -> np.ndarray:
        """Return UB 3x3 by slicing first 9 values from file-based path."""
        path = "entry/data/metadata/HKL/SPEC/UB_MATRIX_VALUE"
        if path in h5_file:
            return self._static_numeric(h5_file[path], 9, "UB matrix").reshape(3, 3)
        raise KeyError(f"UB Matrix link missing at {path}")

    # HDF5 utilities
    def take_data_by_key(self, file_path, target_key):
        """Return dataset whose path ends with target_key, or None if not found."""
        with h5py.File(file_path, "r") as f:
            found_path = None
            def find_key(name, obj):
                nonlocal found_path
                if name.endswith(target_key):
                    found_path = name
                    return True
            f.visititems(find_key)
            if found_path:
                ds = f[found_path]
                return ds.asstr()[:] if ds.dtype == "O" else ds[:]
            else:
                print(f"Key '{target_key}' not found in file.")
                return None

    # Internal helpers
    def _static_str(self, dataset, label: str) -> str:
        """Return one static string metadata value, rejecting changes."""
        try:
            raw = dataset.asstr()[...]
        except Exception:
            raw = dataset[...]
        values = []
        for value in np.asarray(raw).ravel():
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            values.append(str(value))
        if not values:
            raise ValueError(f"{label} is empty.")
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Per-frame varying {label} is not supported.")
        return values[0]

    def _static_numeric(
        self,
        dataset,
        width: int,
        label: str,
        rtol: float = app_settings.RSM_STATIC_METADATA_RELATIVE_TOLERANCE,
        atol: float = app_settings.RSM_STATIC_METADATA_ABSOLUTE_TOLERANCE,
    ) -> np.ndarray:
        """Return one static metadata record, rejecting per-frame changes."""
        values = np.asarray(dataset[...], dtype=float).ravel()
        if values.size < width or values.size % width:
            raise ValueError(
                f"{label} requires records of {width} value(s), got {values.size}."
            )
        records = values.reshape(-1, width)
        if not np.isfinite(records).all():
            raise ValueError(f"{label} must contain only finite values.")
        if records.shape[0] > 1 and not np.allclose(
            records, records[0], rtol=rtol, atol=atol
        ):
            raise ValueError(
                f"Per-frame varying {label} is not supported by the current RSM conversion."
            )
        return records[0]
