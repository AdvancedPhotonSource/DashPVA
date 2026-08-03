# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Shared synthetic HDF5 scan-file builder for the RSM Volume Builder tests.

Writes a minimal but structurally faithful DashPVA scan file: the exact
HDF5 paths dashpva.utils.rsm_converter.RSMConverter reads (including the
real 'INPLANE_REFERENCE_DIRECITON'/'SAMPLE_SURFACE_NORMAL_DIRECITON' typos
present in that module -- must match, not "fix"), with a deliberately
simple single-sample-circle / single-detector-circle geometry (a rocking
curve: MU sweeps across frames, DELTA fixed) so it's cheap to reason about
without needing a full multi-circle diffractometer.
"""
from typing import Optional, Tuple

import h5py
import numpy as np


def make_synthetic_scan_h5(
    path: str,
    n_frames: int = 3,
    shape: Tuple[int, int] = (4, 4),
    energy_eV: float = 10000.0,
    ub: Optional[np.ndarray] = None,
    hot_pixel: Optional[Tuple[int, int, int]] = None,
    hot_value: float = 1e6,
    background: float = 1.0,
    mu_start_deg: float = 0.0,
    mu_stop_deg: float = 10.0,
    delta_deg: float = 20.0,
    ca_monitor: Optional[Tuple[str, "np.ndarray"]] = None,
) -> None:
    """Write a synthetic scan HDF5 file at `path`.

    Geometry: one sample circle ("MU", direction "z-") sweeping linearly from
    mu_start_deg to mu_stop_deg across n_frames, one fixed detector circle
    ("DELTA", direction "z-") at delta_deg -- a simple rocking-curve scan.
    Primary beam along +y, inplane reference +x, sample surface normal +z.
    Detector: shape (H, W), pixel directions "x+"/"z+" pixel size 1mm
    (SIZE = shape in mm), center channel at the array center, 500mm distance.

    Args:
        path: output HDF5 file path.
        n_frames: number of frames (scan points).
        shape: (H, W) detector frame shape.
        energy_eV: photon energy in eV (written to file as keV, since
            RSMConverter.get_physics_params multiplies the stored value by
            1000 to get eV).
        ub: 3x3 UB matrix; defaults to the identity.
        hot_pixel: optional (frame, row, col) to set to `hot_value`; every
            other pixel is `background`.
        hot_value, background: intensity values.
        mu_start_deg, mu_stop_deg: MU sweep range (degrees) across frames.
        delta_deg: fixed DELTA position (degrees).
        ca_monitor: optional (name, values) written to
            entry/data/metadata/ca/<name>, mirroring how HDF5Writer stores
            per-frame CA PVs -- used to exercise I0 normalization.
    """
    h, w = shape
    if ub is None:
        ub = np.eye(3)
    ub = np.asarray(ub, dtype=float)

    images = np.full((n_frames, h, w), float(background), dtype=np.float32)
    if hot_pixel is not None:
        f, r, c = hot_pixel
        images[f, r, c] = float(hot_value)

    mu_positions = np.linspace(mu_start_deg, mu_stop_deg, n_frames)
    delta_positions = np.full(n_frames, float(delta_deg))

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("entry/data")
        data_grp.create_dataset("data", data=images)

        hkl = f.create_group("entry/data/metadata/HKL")

        def _axis_group(name, values):
            g = hkl.create_group(name)
            for i, v in enumerate(values, start=1):
                g.create_dataset(f"AXIS_NUMBER_{i}", data=float(v))

        _axis_group("PRIMARY_BEAM_DIRECTION", [0.0, 1.0, 0.0])
        _axis_group("INPLANE_REFERENCE_DIRECITON", [1.0, 0.0, 0.0])  # sic, matches rsm_converter.py
        _axis_group("SAMPLE_SURFACE_NORMAL_DIRECITON", [0.0, 0.0, 1.0])  # sic

        spec = hkl.create_group("SPEC")
        spec.create_dataset("ENERGY_VALUE", data=energy_eV / 1000.0)  # file stores keV
        spec.create_dataset("UB_MATRIX_VALUE", data=ub.ravel())

        det = hkl.create_group("DETECTOR_SETUP")
        det.create_dataset("PIXEL_DIRECTION_1", data=np.array(["x+"], dtype=object), dtype=str_dt)
        det.create_dataset("PIXEL_DIRECTION_2", data=np.array(["z+"], dtype=object), dtype=str_dt)
        det.create_dataset("CENTER_CHANNEL_PIXEL", data=np.array([w // 2, h // 2], dtype=float))
        det.create_dataset("SIZE", data=np.array([float(w), float(h)], dtype=float))  # mm, 1mm/pixel
        det.create_dataset("DISTANCE", data=500.0)  # mm

        mu = hkl.create_group("MU")
        mu.create_dataset("DIRECTION_AXIS", data=np.array(["z-"], dtype=object), dtype=str_dt)
        mu.create_dataset("POSITION", data=mu_positions)

        delta = hkl.create_group("DELTA")
        delta.create_dataset("DIRECTION_AXIS", data=np.array(["z-"], dtype=object), dtype=str_dt)
        delta.create_dataset("POSITION", data=delta_positions)

        if ca_monitor is not None:
            name, values = ca_monitor
            f.create_group("entry/data/metadata/ca").create_dataset(
                name, data=np.asarray(values, dtype=float))
