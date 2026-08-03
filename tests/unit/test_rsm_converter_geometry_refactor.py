# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Test A -- geometry-refactor equivalence.

RSMConverter.create_rsm() rebuilds QConversion/HXRD/Ang2Q.init_area from
scratch on every frame. The new build_file_geometry()/q_for_frames() split
builds that geometry once per file and reuses it across a batch of frames
via xrayutilities' vectorized Ang2Q.area() call. This test proves the new
batched path is numerically identical to the existing, unmodified per-frame
path -- both alone (batch size 1) and as a genuine multi-frame batch.
"""
import h5py
import numpy as np
import pytest

from dashpva.utils.rsm_converter import RSMConverter

from ._synthetic_hdf5 import make_synthetic_scan_h5


@pytest.fixture()
def scan_path(tmp_path):
    path = str(tmp_path / "scan.h5")
    make_synthetic_scan_h5(path, n_frames=4, shape=(4, 4))
    return path


class TestGeometryRefactorEquivalence:

    def test_single_frame_batches_match_create_rsm(self, scan_path):
        converter = RSMConverter()
        with h5py.File(scan_path, "r") as f:
            geom = converter.build_file_geometry(f)
            for frame in range(4):
                qx_old, qy_old, qz_old = converter.create_rsm(scan_path, frame)
                qx_new, qy_new, qz_new = converter.q_for_frames(geom, f, frame, frame + 1)
                assert qx_new.shape == (1,) + qx_old.shape
                np.testing.assert_allclose(qx_new[0], qx_old, atol=1e-9)
                np.testing.assert_allclose(qy_new[0], qy_old, atol=1e-9)
                np.testing.assert_allclose(qz_new[0], qz_old, atol=1e-9)

    def test_full_scan_batch_matches_per_frame_create_rsm(self, scan_path):
        converter = RSMConverter()
        with h5py.File(scan_path, "r") as f:
            n_frames = f["entry/data/data"].shape[0]
            geom = converter.build_file_geometry(f)
            qx_batch, qy_batch, qz_batch = converter.q_for_frames(geom, f, 0, n_frames)

        assert qx_batch.shape[0] == n_frames
        for frame in range(n_frames):
            qx_old, qy_old, qz_old = converter.create_rsm(scan_path, frame)
            np.testing.assert_allclose(qx_batch[frame], qx_old, atol=1e-9)
            np.testing.assert_allclose(qy_batch[frame], qy_old, atol=1e-9)
            np.testing.assert_allclose(qz_batch[frame], qz_old, atol=1e-9)

    def test_get_sample_and_detector_circles_unchanged_after_refactor(self, scan_path):
        # get_sample_and_detector_circles was rewritten to resolve its circle
        # paths through _resolve_circle_paths -- same signature, same return
        # value, zero behavior change.
        converter = RSMConverter()
        with h5py.File(scan_path, "r") as f:
            for frame in range(4):
                sc_dir, sc_pos, dc_dir, dc_pos = converter.get_sample_and_detector_circles(f, frame)
                assert sc_dir == ["z-"]
                assert dc_dir == ["z-"]
                assert len(sc_pos) == 1 and len(dc_pos) == 1
