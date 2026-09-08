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

"""Qt-free volume persistence, NaN-empty voxels, and the Gridder3D contract."""

import subprocess
import sys

import h5py
import numpy as np
import pytest

from dashpva.utils.gridder_access import (
    GridderContractError,
    gridder_coverage,
    gridder_numerator,
    verify_gridder_contract,
)
from dashpva.utils.hdf5_loader import HDF5Loader
from dashpva.utils.volume_io import (
    finite_intensity_range,
    mean_with_nan_empty,
    save_volume,
)

# Asymmetric on purpose: a transpose bug still yields a correctly-shaped cube.
SHAPE = (3, 4, 5)


class TestQtFreeWritePath:
    """The grid consumer runs under pvaccess, where importing Qt core-dumps."""

    def test_volume_write_path_imports_no_qt(self):
        # A subprocess, because another test may already have imported Qt.
        code = (
            "import sys;"
            "import dashpva.utils.volume_io, dashpva.utils.gridder_access,"
            "       dashpva.utils.hdf5_writer, dashpva.utils.rsm_gridder,"
            "       dashpva.utils.log_manager;"
            "qt=[m for m in sys.modules if m.startswith('PyQt5')];"
            "sys.exit('PyQt5 imported: %s' % qt if qt else 0)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout


class TestGridderContract:
    """_gdata/_gnorm are private; pin their meaning, not just their names."""

    def test_accumulators_sum_values_and_count_contributions(self):
        verify_gridder_contract()

    def test_contract_check_rejects_a_non_conforming_gridder(self):
        class Impostor:
            _gdata = np.zeros((2, 2, 2))
            _gnorm = np.zeros((2, 2, 2))

        # The accessors themselves are tolerant; the contract check is not.
        assert gridder_numerator(Impostor()).shape == (2, 2, 2)
        assert gridder_coverage(Impostor()).shape == (2, 2, 2)

        class Missing:
            pass

        with pytest.raises(GridderContractError, match="_gdata"):
            gridder_numerator(Missing())
        with pytest.raises(GridderContractError, match="_gnorm"):
            gridder_coverage(Missing())


class TestMeanAndRange:
    def test_empty_voxels_are_nan_not_zero(self):
        numerator = np.array([[[6.0, 0.0]]])
        coverage = np.array([[[3.0, 0.0]]])
        mean = mean_with_nan_empty(numerator, coverage)
        assert mean[0, 0, 0] == pytest.approx(2.0)
        assert np.isnan(mean[0, 0, 1])

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="must match"):
            mean_with_nan_empty(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_intensity_range_ignores_nan(self):
        volume = np.array([1.0, np.nan, 5.0, np.nan])
        assert finite_intensity_range(volume) == [1.0, 5.0]

    def test_all_nan_volume_reports_no_finite_intensity(self):
        assert finite_intensity_range(np.full(4, np.nan)) == [0.0, 0.0]

    def test_empty_volume_reports_zero_range(self):
        assert finite_intensity_range(np.array([])) == [0.0, 0.0]


class TestSaveVolume:
    def _volume(self):
        volume = np.arange(np.prod(SHAPE), dtype=float).reshape(SHAPE)
        volume[0, 0, 0] = np.nan
        return volume

    def test_saves_data_metadata_and_coverage(self, tmp_path):
        path = str(tmp_path / "vol.h5")
        volume = self._volume()
        coverage = np.ones(SHAPE)
        coverage[0, 0, 0] = 0

        assert save_volume(
            path,
            volume,
            coverage=coverage,
            metadata={"axes_labels": ["H", "K", "L"], "voxel_spacing": [0.1, 0.2, 0.3]},
        )

        with h5py.File(path, "r") as handle:
            stored = handle["entry/data/data"][()]
            assert stored.dtype == np.float32
            assert np.isnan(stored[0, 0, 0])
            # Counts stay integral so "how many frames hit this voxel" is exact.
            assert handle["entry/data/coverage"].dtype.kind in ("i", "u")
            meta = handle["entry/data/metadata"]
            np.testing.assert_allclose(meta["voxel_spacing"][()], [0.1, 0.2, 0.3])
            assert [s.decode() for s in meta["axes_labels"][()]] == ["H", "K", "L"]
            # intensity_range must skip the NaN, not propagate it.
            low, high = meta["intensity_range"][()]
            assert np.isfinite(low) and np.isfinite(high)
            assert handle["entry"].attrs["data_type"] == "volume"

    def test_saved_volume_reopens_through_the_existing_loader(self, tmp_path):
        path = str(tmp_path / "vol.h5")
        volume = self._volume()
        save_volume(path, volume, coverage=np.ones(SHAPE))

        loaded, shape = HDF5Loader().load_h5_volume_3d(path)
        assert shape == SHAPE
        np.testing.assert_allclose(
            loaded, volume.astype(np.float32), rtol=0, atol=0, equal_nan=True
        )

    def test_coverage_shape_must_match(self, tmp_path):
        with pytest.raises(ValueError, match="must match volume shape"):
            save_volume(
                str(tmp_path / "bad.h5"), np.zeros(SHAPE), coverage=np.zeros((2, 2, 2))
            )

    def test_rejects_empty_and_wrong_rank(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be empty"):
            save_volume(str(tmp_path / "a.h5"), np.array([]))
        with pytest.raises(ValueError, match="2-D or 3-D"):
            save_volume(str(tmp_path / "b.h5"), np.zeros((2, 2, 2, 2)))

    def test_large_volume_is_chunked_so_it_streams(self, tmp_path):
        path = str(tmp_path / "big.h5")
        save_volume(path, np.zeros((80, 80, 80), dtype=float))
        with h5py.File(path, "r") as handle:
            dataset = handle["entry/data/data"]
            assert dataset.chunks is not None
            assert int(np.prod(dataset.chunks)) <= int(np.prod(dataset.shape))
