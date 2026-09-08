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
import xrayutilities as xu

from dashpva.utils.rsm_converter import RSMConverter

from ._synthetic_hdf5 import make_synthetic_scan_h5


@pytest.fixture()
def scan_path(tmp_path):
    path = str(tmp_path / "scan.h5")
    make_synthetic_scan_h5(path, n_frames=4, shape=(3, 5))
    return path


class TestGeometryRefactorEquivalence:

    def test_detector_center_retains_subpixel_precision(self, scan_path):
        with h5py.File(scan_path, "r+") as h5_file:
            center = h5_file[
                "entry/data/metadata/HKL/DETECTOR_SETUP/CENTER_CHANNEL_PIXEL"
            ]
            center[...] = np.array([1.25, 2.75])
            setup = RSMConverter().get_detector_setup(
                h5_file, h5_file["entry/data/data"].shape
            )
        assert setup[2:4] == pytest.approx((1.25, 2.75))

    def test_asymmetric_detector_matches_independent_reference(self, scan_path):
        converter = RSMConverter()
        with h5py.File(scan_path, "r") as h5_file:
            geom = converter.build_file_geometry(h5_file)
            actual = converter.q_for_frames(geom, h5_file, 0, 4)

        qconv = xu.experiment.QConversion(["z-"], ["z-"], [0.0, 1.0, 0.0])
        reference = xu.HXRD(
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], en=10000.0, qconv=qconv
        )
        reference.Ang2Q.init_area(
            "x+", "z+", cch1=1, cch2=2, Nch1=3, Nch2=5,
            pwidth1=1.0, pwidth2=1.0, distance=500.0, roi=[0, 3, 0, 5]
        )
        expected = reference.Ang2Q.area(
            np.linspace(0.0, 10.0, 4), np.full(4, 20.0), UB=np.eye(3)
        )

        for actual_axis, expected_axis in zip(actual, expected):
            assert actual_axis.shape == (4, 3, 5)
            np.testing.assert_allclose(actual_axis, expected_axis, atol=1e-9)

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

    def test_legacy_circle_names_remain_supported(self, scan_path):
        with h5py.File(scan_path, "r+") as h5_file:
            hkl = h5_file["entry/data/metadata/HKL"]
            hkl.move("SAMPLE_CIRCLE_AXIS_1", "MU")
            hkl.move("DETECTOR_CIRCLE_AXIS_1", "DELTA")
            sc_dir, sc_pos, dc_dir, dc_pos = RSMConverter().get_sample_and_detector_circles(
                h5_file, frame=2
            )

        assert sc_dir == ["z-"]
        assert dc_dir == ["z-"]
        assert sc_pos == pytest.approx([np.linspace(0.0, 10.0, 4)[2]])
        assert dc_pos == pytest.approx([20.0])
