import toml
import numpy as np
import h5py
import xrayutilities as xu
from typing import Dict, Optional

"""Utilities for converting detector frames into reciprocal space (RSM).
This module provides a concise RSMConverter focused on the essential
pipeline: reading metadata, building geometry, and computing Q-space.
"""

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
    def __init__(self, config_path: str = "pv_configs/s6lambda.toml"):
        """Initialize with a TOML HKL config path (used by other parts of the app)."""
        self.hkl_config: Dict[str, Dict[str, str]] = {}
        config = toml.load(config_path)
        if "HKL" in config:
            self.hkl_config = config["HKL"]

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
        primary = [meta[f"PRIMARY_BEAM_DIRECTION/AXIS_NUMBER_{i}"][0] for i in range(1, 4)]
        inplane = [meta[f"INPLANE_REFERENCE_DIRECITON/AXIS_NUMBER_{i}"][0] for i in range(1, 4)]
        surface = [meta[f"SAMPLE_SURFACE_NORMAL_DIRECITON/AXIS_NUMBER_{i}"][0] for i in range(1, 4)]
        ub = self.get_ub_matrix_from_file(h5_file)
        energy = float(meta["SPEC/ENERGY_VALUE"][0]) * 1000.0
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
        p_dir1 = self._first_str(det["PIXEL_DIRECTION_1"])
        p_dir2 = self._first_str(det["PIXEL_DIRECTION_2"])
        cch1 = int(det["CENTER_CHANNEL_PIXEL"][0])
        cch2 = int(det["CENTER_CHANNEL_PIXEL"][1])
        size = det["SIZE"][...]
        pw1 = float(size[0]) / float(shape[1])
        pw2 = float(size[1]) / float(shape[2])
        dist = float(det["DISTANCE"][0])
        return p_dir1, p_dir2, cch1, cch2, shape[1], shape[2], pw1, pw2, dist, roi

    # Geometry extraction
    def get_sample_and_detector_circles(self, h5_file: h5py.File, frame: int):
        """Return lists of direction strings and positions for sample and detector circles."""
        sc_dir, sc_pos, dc_dir, dc_pos = [], [], [], []
        hkl_base = "entry/data/metadata/HKL"
        sample_priority = ["MU", "ETA", "CHI", "PHI"]
        detector_priority = ["NU", "DELTA"]

        # Prefer fallback SAMPLE_CIRCLE_AXIS_1..4 if available, else canonical
        fallback_found = False
        for i in range(1, 5):
            path = f"{hkl_base}/SAMPLE_CIRCLE_AXIS_{i}"
            if path in h5_file:
                fallback_found = True
                dir_val = self._first_str(h5_file[f"{path}/DIRECTION_AXIS"])
                sc_dir.append(dir_val)
                sc_pos.append(float(h5_file[f"{path}/POSITION"][frame]))
        if not fallback_found:
            for axis in sample_priority:
                path = f"{hkl_base}/{axis}"
                if path in h5_file:
                    dir_val = self._first_str(h5_file[f"{path}/DIRECTION_AXIS"])
                    sc_dir.append(dir_val)
                    sc_pos.append(float(h5_file[f"{path}/POSITION"][frame]))

        # Prefer fallback DETECTOR_CIRCLE_AXIS_1..2 if available, else canonical
        fallback_d_found = False
        for i in range(1, 3):
            path = f"{hkl_base}/DETECTOR_CIRCLE_AXIS_{i}"
            if path in h5_file:
                fallback_d_found = True
                dir_val = self._first_str(h5_file[f"{path}/DIRECTION_AXIS"])
                dc_dir.append(dir_val)
                dc_pos.append(float(h5_file[f"{path}/POSITION"][frame]))
        if not fallback_d_found:
            for axis in detector_priority:
                path = f"{hkl_base}/{axis}"
                if path in h5_file:
                    dir_val = self._first_str(h5_file[f"{path}/DIRECTION_AXIS"])
                    dc_dir.append(dir_val)
                    dc_pos.append(float(h5_file[f"{path}/POSITION"][frame]))

        return list(sc_dir), list(sc_pos), list(dc_dir), list(dc_pos)

    # UB helpers
    def get_ub_matrix_from_file(self, h5_file: h5py.File) -> np.ndarray:
        """Return UB 3x3 by slicing first 9 values from file-based path."""
        path = "entry/data/metadata/HKL/SPEC/UB_MATRIX_VALUE"
        if path in h5_file:
            return self._ub_from_values(h5_file[path][...])
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
    def _first_str(self, ds) -> str:
        """Return first element of a dataset as string, decoding bytes when needed."""
        try:
            return ds.asstr()[0]
        except Exception:
            val = ds[0]
            if isinstance(val, (bytes, np.bytes_)):
                try:
                    return val.decode("utf-8")
                except Exception:
                    return str(val)
            return str(val)

    def _ub_from_values(self, ub_values) -> np.ndarray:
        """Take first 9 elements and reshape to 3×3 UB matrix."""
        arr = np.asarray(ub_values).ravel()
        if arr.size < 9:
            raise ValueError(f"UB matrix requires at least 9 elements, got {arr.size}")
        return arr[:9].reshape(3, 3)
