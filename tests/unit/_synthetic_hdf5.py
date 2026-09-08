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

    Geometry: one numbered sample circle (direction "z-") sweeping linearly from
    mu_start_deg to mu_stop_deg across n_frames, one fixed detector circle
    ("DELTA", direction "z-") at delta_deg -- a simple rocking-curve scan.
    Primary beam along +y, inplane reference +x, sample surface normal +z.
    Detector: shape (direction1, direction2), pixel directions "x+"/"z+" pixel size 1mm
    (SIZE = shape in mm), center channel at the array center, 500mm distance.

    Args:
        path: output HDF5 file path.
        n_frames: number of frames (scan points).
        shape: (direction1, direction2) detector frame shape, matching
            NTNDArray/DashPVA rather than conventional image row/column names.
        energy_eV: photon energy in eV (written to file as keV, since
            RSMConverter.get_physics_params multiplies the stored value by
            1000 to get eV).
        ub: 3x3 UB matrix; defaults to the identity.
        hot_pixel: optional (frame, direction1, direction2) to set to `hot_value`; every
            other pixel is `background`.
        hot_value, background: intensity values.
        mu_start_deg, mu_stop_deg: MU sweep range (degrees) across frames.
        delta_deg: fixed DELTA position (degrees).
        ca_monitor: optional (name, values) written to
            entry/data/metadata/ca/<name>, mirroring how HDF5Writer stores
            per-frame CA PVs -- used to exercise I0 normalization.
    """
    nch1, nch2 = shape
    if ub is None:
        ub = np.eye(3)
    ub = np.asarray(ub, dtype=float)

    images = np.full((n_frames, nch1, nch2), float(background), dtype=np.float32)
    if hot_pixel is not None:
        frame, ch1, ch2 = hot_pixel
        images[frame, ch1, ch2] = float(hot_value)

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
        det.create_dataset(
            "CENTER_CHANNEL_PIXEL", data=np.array([nch1 // 2, nch2 // 2], dtype=float)
        )
        det.create_dataset(
            "SIZE", data=np.array([float(nch1), float(nch2)], dtype=float)
        )  # mm, 1mm/pixel
        det.create_dataset("DISTANCE", data=500.0)  # mm

        mu = hkl.create_group("SAMPLE_CIRCLE_AXIS_1")
        mu.create_dataset("DIRECTION_AXIS", data=np.array(["z-"], dtype=object), dtype=str_dt)
        mu.create_dataset("POSITION", data=mu_positions)

        delta = hkl.create_group("DETECTOR_CIRCLE_AXIS_1")
        delta.create_dataset("DIRECTION_AXIS", data=np.array(["z-"], dtype=object), dtype=str_dt)
        delta.create_dataset("POSITION", data=delta_positions)

        if ca_monitor is not None:
            name, values = ca_monitor
            f.create_group("entry/data/metadata/ca").create_dataset(
                name, data=np.asarray(values, dtype=float))
