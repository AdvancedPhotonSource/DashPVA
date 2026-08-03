# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Cross-file merge behavior and input guards for the RSM Volume Builder.

Energy/UB differences WARN rather than block: each file's own UB is applied
per-file, so the merge already lands in a common crystal-fixed HKL frame (this
is why rsMap3D applies UB per scan and does not block either), and HKL is
geometrically energy-independent. Genuinely unusable input still raises.
"""
import numpy as np
import pytest

from dashpva.utils.rsm_gridder import RSMMergeError, build_volume

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

        with pytest.raises(RSMMergeError, match="No unmasked pixels"):
            build_volume([path], nx=4, ny=5, nz=6, use_mask=True, mask_manager=mask_manager)
