# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""End-to-end gridding behavior for the RSM Volume Builder.

Grid dimensions are deliberately ASYMMETRIC everywhere in this file: with
nx == ny == nz an axis permutation anywhere in the pipeline still produces a
correctly-shaped array, so a transpose bug would go undetected.
"""
import h5py
import numpy as np
import pytest
import xrayutilities as xu

from dashpva.utils.hdf5_loader import HDF5Loader
from dashpva.utils.rsm_converter import RSMConverter
from dashpva.utils.rsm_gridder import (
    RSMMergeError,
    build_volume,
    volume_result_to_metadata,
)

from ._synthetic_hdf5 import make_synthetic_scan_h5


class _FakeMaskManager:
    def __init__(self, mask):
        self.mask = mask


def _expected_q(scan_path, frame, row, col):
    """Ground truth from the existing, unmodified single-frame code path, so
    this is not the new gridding code checked against itself."""
    qx, qy, qz = RSMConverter().create_rsm(scan_path, frame)
    return float(qx[row, col]), float(qy[row, col]), float(qz[row, col])


class TestHotPixelRoundtrip:

    def test_hot_pixel_maps_to_expected_voxel_after_roundtrip(self, tmp_path):
        scan_path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(scan_path, n_frames=3, shape=(3, 5),
                                hot_pixel=(1, 1, 3), hot_value=1e6)
        expected_q = _expected_q(scan_path, 1, 1, 3)

        result = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=False)
        assert result.volume.shape == (5, 6, 7)

        peak_idx = np.unravel_index(np.nanargmax(result.volume), result.volume.shape)
        nearest_idx = tuple(
            int(np.argmin(np.abs(axis - q)))
            for axis, q in zip((result.xaxis, result.yaxis, result.zaxis), expected_q)
        )
        assert peak_idx == nearest_idx

        out_path = str(tmp_path / "volume.h5")
        metadata = volume_result_to_metadata(result)
        assert HDF5Loader().save_vol_to_h5(out_path, result.volume, metadata=metadata)

        loader = HDF5Loader()
        loaded_volume, _ = loader.load_h5_volume_3d(out_path)

        assert loaded_volume.shape == (5, 6, 7)
        np.testing.assert_allclose(loaded_volume, result.volume.astype(np.float32))
        assert np.unravel_index(np.nanargmax(loaded_volume), loaded_volume.shape) == peak_idx

        assert loader.file_metadata["array_order"] == "F"
        # Output is HKL because rsm_converter always applies the UB matrix.
        assert list(loader.file_metadata["axes_labels"]) == ["H", "K", "L"]
        assert list(loader.file_metadata["grid_dimensions_cells"]) == [5, 6, 7]
        np.testing.assert_allclose(loader.file_metadata["voxel_spacing"], metadata["voxel_spacing"])
        np.testing.assert_allclose(loader.file_metadata["grid_origin"], metadata["grid_origin"])

        with h5py.File(out_path, "r") as h5_file:
            saved = h5_file["entry/data/metadata"]
            assert saved["coordinate_system"].asstr()[()] == "HKL"
            assert saved["gridder"].asstr()[()] == "xrayutilities.Gridder3D"
            assert saved["source_energies_eV"][...].tolist() == [10000.0]
            assert saved["source_ub_matrices_shape"][...].tolist() == [1, 3, 3]

    def test_grid_origin_is_cell_corner_not_bin_center(self, tmp_path):
        # Gridder3D's axes are bin CENTERS; the PyVista viewer treats
        # grid_origin as a cell CORNER, so it must sit half a voxel below.
        scan_path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(scan_path, n_frames=3, shape=(4, 4))
        result = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=False)
        meta = volume_result_to_metadata(result)

        for origin, spacing, axis in zip(meta["grid_origin"], meta["voxel_spacing"],
                                          (result.xaxis, result.yaxis, result.zaxis)):
            assert origin == pytest.approx(float(axis[0]) - spacing / 2.0)


class TestMaskHandling:

    def test_masked_pixels_are_excluded_not_zeroed(self, tmp_path):
        scan_path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(scan_path, n_frames=3, shape=(4, 4))
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        mask_manager = _FakeMaskManager(mask)

        unmasked = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=False,
                                 mask_manager=mask_manager)
        masked = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=True,
                               mask_manager=mask_manager)

        # Exclusion removes points from the gridder; zero-and-include (the bug
        # this guards against) would leave num_points_binned unchanged.
        assert unmasked.num_points_binned == 3 * 4 * 4
        assert masked.num_points_binned == 3 * (4 * 4 - 1)
        assert unmasked.num_points_excluded_by_mask == 0
        assert masked.num_points_excluded_by_mask == 3

    def test_mask_transposed_selects_different_pixels(self, tmp_path):
        # The in-app mask is stored in the viewer's display orientation, which
        # the file does not record, so the caller states it explicitly.
        scan_path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(scan_path, n_frames=2, shape=(4, 4), hot_pixel=(0, 0, 3),
                                hot_value=1e6)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 3] = True  # masks the hot pixel only when applied untransposed
        mask_manager = _FakeMaskManager(mask)

        raw = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=True,
                            mask_manager=mask_manager, mask_transposed=False)
        transposed = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=True,
                                   mask_manager=mask_manager, mask_transposed=True)

        assert raw.num_points_binned == transposed.num_points_binned
        # Untransposed drops the hot pixel; transposed masks (3,0) and keeps it.
        assert np.nanmax(raw.volume) < np.nanmax(transposed.volume)

    def test_mask_shape_mismatch_raises_loudly(self, tmp_path):
        scan_path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(scan_path, n_frames=2, shape=(4, 4))
        mask_manager = _FakeMaskManager(np.zeros((8, 8), dtype=bool))

        with pytest.raises(RSMMergeError, match="Mask shape"):
            build_volume([scan_path], nx=4, ny=5, nz=6, use_mask=True,
                          mask_manager=mask_manager)


class TestMonitorNormalization:

    def test_intensity_divided_by_per_frame_monitor(self, tmp_path):
        # Two frames with identical counts but monitor 1.0 vs 2.0: after
        # normalization the second frame's contribution must be halved.
        plain = str(tmp_path / "plain.h5")
        norm = str(tmp_path / "norm.h5")
        for path in (plain, norm):
            make_synthetic_scan_h5(path, n_frames=2, shape=(4, 4), background=100.0,
                                    ca_monitor=("I0", np.array([1.0, 2.0])))

        without = build_volume([plain], nx=5, ny=6, nz=7, use_mask=False)
        with_norm = build_volume([norm], nx=5, ny=6, nz=7, use_mask=False,
                                  monitor_dataset="I0")

        assert np.nanmax(without.volume) == pytest.approx(100.0)
        # Frame 0 -> 100/1, frame 1 -> 100/2; bins hold one or the other.
        assert np.nanmax(with_norm.volume) == pytest.approx(100.0)
        assert with_norm.volume[with_norm.volume > 0].min() == pytest.approx(50.0)

    def test_missing_monitor_dataset_raises(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(4, 4))
        with pytest.raises(RSMMergeError, match="not found"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=False, monitor_dataset="I0")

    def test_zero_monitor_value_raises_instead_of_dividing_by_zero(self, tmp_path):
        path = str(tmp_path / "scan.h5")
        make_synthetic_scan_h5(path, n_frames=2, shape=(4, 4),
                                ca_monitor=("I0", np.array([1.0, 0.0])))
        with pytest.raises(RSMMergeError, match="strictly positive"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=False, monitor_dataset="I0")


class TestGridderAssumptions:

    def test_fixed_range_latches_and_clips_out_of_range_points(self):
        # Guards the third-party behavior the whole two-pass design rests on:
        # with KeepData set, the first call latches fixed_range, after which
        # out-of-range points are silently dropped rather than rescaling.
        gridder = xu.Gridder3D(4, 5, 6)
        gridder.KeepData(True)
        gridder.dataRange(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, fixed=True)

        gridder(np.array([0.5]), np.array([0.5]), np.array([0.5]), np.array([10.0]))
        gridder(np.array([5.0]), np.array([5.0]), np.array([5.0]), np.array([999.0]))

        assert gridder.xaxis.min() >= 0.0 and gridder.xaxis.max() <= 1.0
        assert gridder.data.max() == 10.0


class TestCoverageAndEmptyVoxels:
    """Coverage volume and NaN-vs-zero for voxels nothing scattered into."""

    def test_empty_voxels_are_nan_and_coverage_counts_contributions(self, tmp_path):
        scan_path = str(tmp_path / "coverage.h5")
        make_synthetic_scan_h5(scan_path, n_frames=3, shape=(3, 5))

        # A grid far finer than the sampling guarantees unfilled voxels.
        result = build_volume([scan_path], nx=12, ny=13, nz=14, use_mask=False)

        assert result.coverage is not None
        assert result.coverage.shape == result.volume.shape

        empty = result.coverage == 0
        filled = ~empty
        assert empty.any(), "expected some voxels to receive no contribution"
        assert filled.any(), "expected some voxels to be filled"

        # The distinction this exists for: unmeasured is NaN, not a confident 0.
        assert np.all(np.isnan(result.volume[empty]))
        assert not np.any(np.isnan(result.volume[filled]))

        # Coverage counts pixels, so it must total the number actually binned.
        assert int(result.coverage.sum()) == result.num_points_binned

    def test_intensity_range_ignores_nan_empty_voxels(self, tmp_path):
        scan_path = str(tmp_path / "range.h5")
        make_synthetic_scan_h5(scan_path, n_frames=2, shape=(3, 5))

        result = build_volume([scan_path], nx=11, ny=12, nz=13, use_mask=False)
        metadata = volume_result_to_metadata(result)

        low, high = metadata["intensity_range"]
        assert np.isfinite(low) and np.isfinite(high)
        assert low == pytest.approx(float(np.nanmin(result.volume)))
        assert high == pytest.approx(float(np.nanmax(result.volume)))

    def test_coverage_survives_save_and_reopens_as_a_plain_volume(self, tmp_path):
        scan_path = str(tmp_path / "scan.h5")
        out_path = str(tmp_path / "volume.h5")
        make_synthetic_scan_h5(scan_path, n_frames=2, shape=(3, 5))

        result = build_volume([scan_path], nx=5, ny=6, nz=7, use_mask=False)
        assert HDF5Loader().save_vol_to_h5(
            out_path,
            result.volume,
            metadata=volume_result_to_metadata(result),
            coverage=result.coverage,
        )

        with h5py.File(out_path, "r") as handle:
            assert "entry/data/coverage" in handle
            stored_coverage = handle["entry/data/coverage"][()]
        np.testing.assert_array_equal(stored_coverage, result.coverage.astype(np.uint32))

        # Workbench addresses /entry/data/data by name; the sibling must be inert.
        loaded, shape = HDF5Loader().load_h5_volume_3d(out_path)
        assert shape == result.volume.shape
        np.testing.assert_allclose(
            loaded, result.volume.astype(np.float32), rtol=0, atol=0, equal_nan=True
        )
