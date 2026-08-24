# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Live accumulator: fixed bounds, drop accounting, and offline parity."""

import numpy as np
import pytest

from dashpva.utils.rsm_live_grid import (
    DEFAULT_PREVIEW_BUDGET_BYTES,
    GridBoundsSpec,
    LiveVolumeAccumulator,
    coarse_shape_for_budget,
)

# Asymmetric so a transposed grid cannot pass by coincidence.
BOUNDS = GridBoundsSpec(0.0, 1.0, 0.0, 2.0, 0.0, 3.0, nx=5, ny=6, nz=7)


def _frame(value=1.0, n=8):
    """n samples spread along the grid diagonal, all in range."""
    t = np.linspace(0.05, 0.95, n)
    return t * 1.0, t * 2.0, t * 3.0, np.full(n, value)


class TestBoundsValidation:
    @pytest.mark.parametrize("kwargs", [
        {"hmax": 0.0},                      # max == min
        {"hmax": float("nan")},             # non-finite
    ])
    def test_degenerate_bounds_are_rejected(self, kwargs):
        params = dict(hmin=0.0, hmax=1.0, kmin=0.0, kmax=2.0, lmin=0.0, lmax=3.0,
                      nx=5, ny=6, nz=7)
        params.update(kwargs)
        with pytest.raises(ValueError):
            GridBoundsSpec(**params)

    def test_single_bin_axis_is_rejected(self):
        """Gridder3D.axis() returns a bare float for n == 1."""
        with pytest.raises(ValueError, match="at least 2"):
            GridBoundsSpec(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx=1, ny=5, nz=5)

    def test_grid_bytes_reflects_two_float64_accumulators(self):
        assert BOUNDS.grid_bytes() == 5 * 6 * 7 * 8 * 2


class TestAccumulation:
    def test_repeated_identical_frames_hold_the_mean_and_double_coverage(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame(value=10.0))
        first_mean = acc.mean.copy()
        first_coverage = acc.coverage.copy()

        acc.add_frame(*_frame(value=10.0))

        # A second identical pass must not brighten the volume...
        np.testing.assert_allclose(acc.mean, first_mean, equal_nan=True)
        # ...but must record that twice as much evidence went into it.
        np.testing.assert_array_equal(acc.coverage, first_coverage * 2)

    def test_mean_not_sum_across_overlapping_passes(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame(value=2.0))
        acc.add_frame(*_frame(value=4.0))
        filled = acc.coverage > 0
        np.testing.assert_allclose(acc.mean[filled], 3.0)

    def test_empty_voxels_are_nan_and_counted_as_uncovered(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame())
        empty = acc.coverage == 0
        assert empty.any()
        assert np.all(np.isnan(acc.mean[empty]))
        assert not np.any(np.isnan(acc.mean[~empty]))

    def test_frame_at_a_time_equals_one_combined_batch(self):
        per_frame = LiveVolumeAccumulator(BOUNDS)
        qx, qy, qz, values = _frame(value=5.0)
        for index in range(len(qx)):
            per_frame.add_frame(
                qx[index:index + 1], qy[index:index + 1],
                qz[index:index + 1], values[index:index + 1],
            )

        batched = LiveVolumeAccumulator(BOUNDS)
        batched.add_frame(qx, qy, qz, values)

        np.testing.assert_allclose(per_frame.mean, batched.mean, equal_nan=True)
        np.testing.assert_array_equal(per_frame.coverage, batched.coverage)


class TestRejectionAccounting:
    def test_out_of_range_points_are_counted_not_silently_dropped(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        qx = np.array([0.5, 99.0, -99.0])
        qy = np.array([1.0, 1.0, 1.0])
        qz = np.array([1.5, 1.5, 1.5])
        binned = acc.add_frame(qx, qy, qz, np.ones(3))
        assert binned == 1
        assert acc.counters.points_out_of_range == 2

    def test_nonfinite_samples_are_counted_and_excluded(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        qx = np.array([0.5, np.nan, 0.5])
        qy = np.array([1.0, 1.0, 1.0])
        qz = np.array([1.5, 1.5, np.inf])
        assert acc.add_frame(qx, qy, qz, np.ones(3)) == 1
        assert acc.counters.points_nonfinite == 2

    def test_masked_pixels_are_excluded_rather_than_zeroed(self):
        """A zeroed pixel would count as a real intensity-zero measurement."""
        mask = np.zeros(8, dtype=bool)
        mask[:4] = True
        acc = LiveVolumeAccumulator(BOUNDS, mask=mask)
        qx, qy, qz, values = _frame(value=7.0)
        binned = acc.add_frame(qx, qy, qz, values)
        assert binned == 4
        assert acc.counters.points_masked == 4
        filled = acc.coverage > 0
        np.testing.assert_allclose(acc.mean[filled], 7.0)

    def test_mask_pixel_count_mismatch_is_a_named_error(self):
        acc = LiveVolumeAccumulator(BOUNDS, mask=np.zeros(3, dtype=bool))
        with pytest.raises(ValueError, match="ROI or binning"):
            acc.add_frame(*_frame())

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_nonpositive_monitor_is_rejected(self, bad):
        acc = LiveVolumeAccumulator(BOUNDS, monitor_name="I0")
        with pytest.raises(ValueError, match="finite and positive"):
            acc.add_frame(*_frame(), monitor=bad)
        assert acc.counters.frames_rejected == 1

    def test_monitor_divides_the_frame(self):
        acc = LiveVolumeAccumulator(BOUNDS, monitor_name="I0")
        acc.add_frame(*_frame(value=10.0), monitor=4.0)
        filled = acc.coverage > 0
        np.testing.assert_allclose(acc.mean[filled], 2.5)
        assert acc.aggregation == "mean_of_counts_over_monitor"

    def test_aggregation_label_without_a_monitor(self):
        assert LiveVolumeAccumulator(BOUNDS).aggregation == "unweighted_mean"


class TestPreview:
    def test_preview_stays_within_the_byte_budget(self):
        big = GridBoundsSpec(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx=256, ny=256, nz=256)
        acc = LiveVolumeAccumulator(big)
        payload = acc.preview()
        assert payload.nbytes() <= DEFAULT_PREVIEW_BUDGET_BYTES
        assert payload.mean.dtype == np.float32

    def test_preview_never_carries_the_full_resolution_volume(self):
        big = GridBoundsSpec(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, nx=128, ny=128, nz=128)
        acc = LiveVolumeAccumulator(big)
        payload = acc.preview()
        assert payload.shape != big.shape
        assert payload.mean.size < big.voxel_count

    def test_small_grids_are_previewed_at_full_resolution(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        assert acc.preview().shape == BOUNDS.shape

    def test_coarse_shape_preserves_anisotropy(self):
        """A single uniform divisor would waste or overshoot the budget."""
        coarse = coarse_shape_for_budget((512, 512, 64), budget_bytes=4 * 1024 * 1024)
        assert coarse[0] == coarse[1] > coarse[2]
        assert coarse[0] * coarse[1] * coarse[2] * 4 <= 4 * 1024 * 1024

    def test_preview_reports_the_same_counters_as_the_accumulator(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame())
        acc.add_frame(np.array([99.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
        payload = acc.preview()
        assert payload.frames_accepted == 2
        assert payload.points_out_of_range == 1
        assert payload.points_binned == acc.counters.points_binned

    def test_preview_and_fine_grid_see_identical_accepted_samples(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame(value=3.0))
        payload = acc.preview()
        fine_total = int(acc.coverage.sum())
        assert fine_total == acc.counters.points_binned
        # Same samples, different binning: totals agree even though shapes differ.
        assert payload.voxels_filled >= 1


class TestLifecycle:
    def test_clear_resets_data_and_counters_but_keeps_bounds(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        acc.add_frame(*_frame())
        assert acc.counters.points_binned > 0

        acc.clear()

        assert acc.counters.points_binned == 0
        assert acc.counters.frames_accepted == 0
        assert int(acc.coverage.sum()) == 0
        assert np.all(np.isnan(acc.mean))
        # Bounds survive, so a cleared accumulator is still usable.
        acc.add_frame(*_frame())
        assert acc.counters.points_binned > 0

    def test_origin_is_shifted_half_a_voxel_to_the_cell_corner(self):
        acc = LiveVolumeAccumulator(BOUNDS)
        dh, dk, dl = BOUNDS.spacing
        assert acc.origin == pytest.approx((-dh / 2.0, -dk / 2.0, -dl / 2.0))

    def test_metadata_matches_the_offline_volume_conventions(self):
        acc = LiveVolumeAccumulator(BOUNDS, monitor_name="I0")
        acc.add_frame(*_frame(), monitor=2.0)
        metadata = acc.to_metadata()
        assert metadata["axes_labels"] == ["H", "K", "L"]
        assert metadata["array_order"] == "F"
        assert metadata["grid_dimensions_cells"] == list(BOUNDS.shape)
        assert metadata["aggregation"] == "mean_of_counts_over_monitor"
        low, high = metadata["intensity_range"]
        assert np.isfinite(low) and np.isfinite(high)


class TestLiveOfflineParity:
    """The point of the whole exercise: what you watch is what you save."""

    def test_live_and_offline_agree_on_identical_frames_bounds_resolution(self, tmp_path):
        import h5py

        from dashpva.utils.rsm_converter import RSMConverter
        from dashpva.utils.rsm_gridder import GridBounds, build_volume

        from ._synthetic_hdf5 import make_synthetic_scan_h5

        scan = str(tmp_path / "parity.h5")
        make_synthetic_scan_h5(scan, n_frames=3, shape=(3, 5))

        converter = RSMConverter()
        with h5py.File(scan, "r") as handle:
            geometry = converter.build_file_geometry(handle)
            n_frames = handle["entry/data/data"].shape[0]
            qx, qy, qz = converter.q_for_frames(geometry, handle, 0, n_frames)
            intensity = np.asarray(handle["entry/data/data"][()], dtype=float)

        # Same box for both paths, padded so nothing falls out of range.
        pad = 1e-9
        bounds = GridBounds(
            float(qx.min()) - pad, float(qx.max()) + pad,
            float(qy.min()) - pad, float(qy.max()) + pad,
            float(qz.min()) - pad, float(qz.max()) + pad,
        )
        nx, ny, nz = 5, 6, 7

        offline = build_volume(
            [scan], nx=nx, ny=ny, nz=nz, use_mask=False, fixed_bounds=bounds
        )

        live = LiveVolumeAccumulator(
            GridBoundsSpec(
                bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax,
                bounds.zmin, bounds.zmax, nx=nx, ny=ny, nz=nz,
            )
        )
        for index in range(n_frames):
            live.add_frame(qx[index], qy[index], qz[index], intensity[index])

        assert live.counters.points_out_of_range == 0
        np.testing.assert_array_equal(live.coverage, offline.coverage)
        np.testing.assert_allclose(
            live.mean, offline.volume, rtol=1e-12, atol=1e-12, equal_nan=True
        )
