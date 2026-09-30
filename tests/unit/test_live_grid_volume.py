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

"""Live grid volume: visible autoscale and transparent empty voxels."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from dashpva.viewer.hkl3d.live_grid_volume import (
    display_values,
    opacity_ramp,
    park_values,
    transfer_range,
)


def test_log_autoscale_ignores_empty_voxels_and_hot_pixels():
    mean = np.full(20000, np.nan, dtype=np.float32)
    mean[:6000] = 10.0          # background
    mean[6000:9999] = 1000.0    # signal
    mean[9999] = 1e9            # one hot voxel (0.01%)
    values, empty, low, high = display_values(mean, log=True)

    assert empty.sum() == 10000
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(3.0)
    assert np.all(np.isfinite(values))


def test_empty_and_nonpositive_voxels_sit_at_the_transparent_floor():
    mean = np.array([np.nan, 0.0, -5.0, 1.0, 10.0, 100.0], dtype=np.float32)
    values, empty, low, _high = display_values(mean, log=True)

    assert empty.tolist() == [True, True, True, False, False, False]
    assert np.all(values[empty] == low)
    assert opacity_ramp(0.2, 0.9)[0] == 0.0


def test_all_empty_volume_is_safe():
    values, empty, low, high = display_values(np.full(8, np.nan), log=True)
    assert empty.all()
    assert (low, high) == (0.0, 1.0)
    assert np.all(values == 0.0)


def test_linear_mode_uses_raw_values():
    mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values, empty, low, high = display_values(mean, log=False)
    assert not empty.any()
    np.testing.assert_array_equal(values, mean)
    assert low < high


def test_every_filled_voxel_is_visible_and_empties_are_transparent():
    values = np.array([0.5, 1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    empty = np.array([False, False, False, False, True])
    parked = park_values(values, empty, 1.0, 3.0)
    floor, top = transfer_range(1.0, 3.0)
    ramp = opacity_ramp(0.0, 1.0)

    assert parked[0] == 1.0            # below the colour floor, still shown
    assert floor < 1.0 and top == 3.0
    assert parked[4] == floor          # empty voxel sits in the transparent band

    def opacity(value):
        return np.interp(value, np.linspace(floor, top, ramp.size), ramp)

    assert opacity(parked[4]) == 0.0
    assert opacity(parked[0]) > 0.0
