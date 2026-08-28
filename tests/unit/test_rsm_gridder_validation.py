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

"""Cross-file merge behavior and input guards for the RSM Volume Builder.

Energy/UB differences WARN rather than block: each file's own UB is applied
per-file, so the merge already lands in a common crystal-fixed HKL frame (this
is why rsMap3D applies UB per scan and does not block either), and HKL is
geometrically energy-independent. Genuinely unusable input still raises.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dashpva.utils.rsm_gridder import (
    RSMMergeError,
    build_volume,
    ensure_memory_available,
    estimate_grid_memory,
)

from ._synthetic_hdf5 import make_synthetic_scan_h5


class _FakeMaskManager:
    def __init__(self, mask):
        self.mask = mask


class TestConsistencyWarnings:

    def test_mismatched_ub_warns_but_still_merges(self, tmp_path):
        path_a = str(tmp_path / "scan_a.h5")
        path_b = str(tmp_path / "scan_b.h5")
        make_synthetic_scan_h5(path_a, n_frames=2, shape=(4, 4))
        bad_ub = np.eye(3)
        bad_ub[0, 0] = 5.0
        make_synthetic_scan_h5(path_b, n_frames=2, shape=(4, 4), ub=bad_ub)

        warnings = []
        result = build_volume([path_a, path_b], nx=4, ny=5, nz=6, use_mask=False,
                              warn=warnings.append)

        assert result.num_points_binned == (2 + 2) * 4 * 4
        assert any("scan_b.h5" in w and "UB matrix" in w for w in warnings)

    def test_mismatched_energy_warns_but_still_merges(self, tmp_path):
        path_a = str(tmp_path / "scan_a.h5")
        path_b = str(tmp_path / "scan_b.h5")
        make_synthetic_scan_h5(path_a, n_frames=2, shape=(4, 4), energy_eV=10000.0)
        make_synthetic_scan_h5(path_b, n_frames=2, shape=(4, 4), energy_eV=10050.0)

        warnings = []
        result = build_volume([path_a, path_b], nx=4, ny=5, nz=6, use_mask=False,
                              warn=warnings.append)

        assert result.num_points_binned == (2 + 2) * 4 * 4
        assert any("scan_b.h5" in w and "energy" in w.lower() for w in warnings)

    def test_consistent_files_merge_without_warnings(self, tmp_path):
        path_a = str(tmp_path / "scan_a.h5")
        path_b = str(tmp_path / "scan_b.h5")
        make_synthetic_scan_h5(path_a, n_frames=2, shape=(4, 4), mu_start_deg=0.0, mu_stop_deg=5.0)
        make_synthetic_scan_h5(path_b, n_frames=3, shape=(4, 4), mu_start_deg=5.5, mu_stop_deg=10.0)

        warnings = []
        result = build_volume([path_a, path_b], nx=4, ny=5, nz=6, use_mask=False,
                              warn=warnings.append)

        assert warnings == []
        assert result.num_points_binned == (2 + 3) * 4 * 4
        assert len(result.per_file_info) == 2


class TestInputGuards:

    def test_no_files_raises(self):
        with pytest.raises(RSMMergeError):
            build_volume([], nx=4, ny=4, nz=4)

    def test_grid_resolution_below_two_raises(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(4, 4))
        with pytest.raises(RSMMergeError):
            build_volume([path], nx=1, ny=4, nz=4)

    def test_fully_masked_file_raises_instead_of_nan_volume(self, tmp_path):
        # With every pixel masked the bounds stay at +/-inf, which would reach
        # dataRange() and yield NaN axes with no error at all.
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(4, 4))
        mask_manager = _FakeMaskManager(np.ones((4, 4), dtype=bool))

        with pytest.raises(RSMMergeError, match="No finite, unmasked pixels"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=True, mask_manager=mask_manager)

    def test_nonfinite_intensity_is_excluded_and_reported(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(3, 5))
        with h5py.File(path, "r+") as h5_file:
            h5_file["entry/data/data"][0, 1, 2] = np.nan
        warnings = []
        result = build_volume(
            [path], nx=4, ny=5, nz=6, use_mask=False, warn=warnings.append
        )
        assert result.num_points_binned == 2 * 3 * 5 - 1
        assert result.num_points_excluded_nonfinite == 1
        assert any("non-finite" in warning for warning in warnings)

    def test_scalar_circle_position_is_broadcast(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=3, shape=(3, 5))
        circle_path = (
            "entry/data/metadata/HKL/DETECTOR_CIRCLE_AXIS_1/POSITION"
        )
        with h5py.File(path, "r+") as h5_file:
            del h5_file[circle_path]
            h5_file["entry/data/metadata/HKL/DETECTOR_CIRCLE_AXIS_1"].create_dataset(
                "POSITION", data=20.0
            )
        result = build_volume([path], nx=4, ny=5, nz=6, use_mask=False)
        assert result.num_points_binned == 3 * 3 * 5

    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_monitor_must_be_finite_and_positive(self, tmp_path, value):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(
            path, n_frames=2, shape=(3, 5),
            ca_monitor=("I0", np.array([1.0, value])),
        )
        with pytest.raises(RSMMergeError, match="strictly positive"):
            build_volume(
                [path], nx=4, ny=5, nz=6, use_mask=False, monitor_dataset="I0"
            )

    def test_singular_ub_is_rejected(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, shape=(3, 5), ub=np.zeros((3, 3)))
        with pytest.raises(RSMMergeError, match="full rank"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=False)

    def test_varying_energy_within_file_is_rejected(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(3, 5))
        with h5py.File(path, "r+") as h5_file:
            dataset = h5_file["entry/data/metadata/HKL/SPEC/ENERGY_VALUE"]
            del h5_file[dataset.name]
            h5_file["entry/data/metadata/HKL/SPEC"].create_dataset(
                "ENERGY_VALUE", data=np.array([10.0, 10.1])
            )
        with pytest.raises(RSMMergeError, match="varying photon energy"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=False)

    def test_epics_readback_jitter_is_accepted(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(3, 5))
        with h5py.File(path, "r+") as h5_file:
            spec = h5_file["entry/data/metadata/HKL/SPEC"]
            del spec["ENERGY_VALUE"]
            spec.create_dataset("ENERGY_VALUE", data=np.array([10.0, 10.0000001]))
            detector = h5_file["entry/data/metadata/HKL/DETECTOR_SETUP"]
            del detector["DISTANCE"]
            detector.create_dataset("DISTANCE", data=np.array([500.0, 500.0001]))

        result = build_volume([path], nx=4, ny=5, nz=6, use_mask=False)

        assert result.per_file_info[0].energy_eV == pytest.approx(10000.0)

    def test_extra_numbered_circle_is_supported(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(3, 5))
        with h5py.File(path, "r+") as h5_file:
            group = h5_file["entry/data/metadata/HKL"].create_group(
                "SAMPLE_CIRCLE_AXIS_5"
            )
            group.create_dataset("DIRECTION_AXIS", data=np.bytes_("x+"))
            group.create_dataset("POSITION", data=np.zeros(2))
        result = build_volume([path], nx=4, ny=5, nz=6, use_mask=False)

        assert result.volume.shape == (4, 5, 6)


class TestMemoryGuard:

    def test_estimate_includes_dense_grid_batch_and_float32_output(self):
        estimate = estimate_grid_memory(
            10, 20, 30, detector_shapes=[(100, 200)], batch_bytes=1_000_000
        )
        assert estimate.grid_bytes == 10 * 20 * 30 * 24
        assert estimate.batch_bytes == 100 * 200 * 96
        assert estimate.output_bytes == 10 * 20 * 30 * 4
        assert estimate.peak_bytes == estimate.grid_bytes + estimate.batch_bytes

    def test_one_detector_frame_can_exceed_batch_target(self):
        estimate = estimate_grid_memory(
            2, 2, 2, detector_shapes=[(4096, 4096)], batch_bytes=1024
        )
        assert estimate.batch_bytes == 4096 * 4096 * 96

    def test_unsafe_estimate_is_rejected(self):
        estimate = estimate_grid_memory(500, 500, 500, batch_bytes=1)
        with pytest.raises(RSMMergeError, match="Estimated peak memory"):
            ensure_memory_available(estimate, available_bytes=1024**3, max_fraction=0.70)

    @pytest.mark.parametrize("fraction", [0.0, 1.1, np.nan])
    def test_invalid_memory_fraction_is_rejected(self, fraction):
        estimate = estimate_grid_memory(2, 2, 2, batch_bytes=1)
        with pytest.raises(RSMMergeError, match="max_fraction"):
            ensure_memory_available(estimate, available_bytes=1024**3, max_fraction=fraction)

    def test_programmatic_caller_can_disable_memory_guard(self, tmp_path, monkeypatch):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=1, shape=(2, 3))
        monkeypatch.setattr(
            "dashpva.utils.rsm_gridder.psutil.virtual_memory",
            lambda: SimpleNamespace(available=1),
        )

        with pytest.raises(RSMMergeError, match="Estimated peak memory"):
            build_volume([path], nx=2, ny=2, nz=2, use_mask=False)
        result = build_volume(
            [path], nx=2, ny=2, nz=2, use_mask=False, memory_limit_fraction=None
        )
        assert result.volume.shape == (2, 2, 2)
