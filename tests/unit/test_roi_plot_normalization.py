#!/usr/bin/env python3
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


"""Divisor rules for offline ROI plot normalization.

Workbench divides a plotted ROI curve by a CA channel already recorded in the
scan HDF5. It is not live Area Detector normalization -- both series come off
disk, one reading per frame.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from dashpva.viewer.workbench.rois.roi_plot_dock import (
    CA_METADATA_PATH,
    load_ca_channels,
    normalize_series,
)


def test_a_matching_divisor_divides_point_for_point():
    assert list(normalize_series([10.0, 20.0], [2.0, 4.0])) == [5.0, 5.0]


def test_no_divisor_passes_the_series_through():
    assert list(normalize_series([10.0, 20.0], None)) == [10.0, 20.0]


def test_a_dropped_ca_sample_is_rejected_rather_than_truncated():
    """A short divisor shifts later readings onto the wrong frame; trimming hides it."""
    out = normalize_series([10.0, 20.0, 30.0], [2.0, 4.0])
    assert len(out) == 3
    assert np.isnan(out).all()


def test_a_longer_divisor_is_rejected_too():
    assert np.isnan(normalize_series([10.0, 20.0], [2.0, 4.0, 5.0])).all()


@pytest.mark.parametrize('divisor', [0.0, -4.0, np.nan, np.inf])
def test_a_non_positive_or_non_finite_divisor_gaps_that_frame(divisor):
    out = normalize_series([10.0, 20.0], [2.0, divisor])
    assert out[0] == 5.0
    assert np.isnan(out[1])


def test_an_empty_series_stays_empty():
    assert len(normalize_series([], [1.0, 2.0])) == 0


def _write(path, groups):
    with h5py.File(path, 'w') as h5f:
        for group, datasets in groups.items():
            grp = h5f.require_group(group)
            for name, values in datasets.items():
                grp[name] = np.asarray(values, dtype=float)


def test_only_ca_channels_are_offered_as_divisors(tmp_path):
    """Motor positions are plottable but are not per-frame flux, so they must not divide."""
    scan = tmp_path / 'scan.h5'
    _write(scan, {
        CA_METADATA_PATH: {'I0': [1.0, 2.0, 3.0], 'I1': [4.0, 5.0, 6.0]},
        'entry/data/metadata/motor_positions': {'theta': [0.1, 0.2, 0.3]},
        'entry/data/metadata': {'loose': [7.0, 8.0, 9.0]},
    })
    assert sorted(load_ca_channels(str(scan))) == ['I0', 'I1']


def test_a_single_point_channel_is_not_a_divisor(tmp_path):
    scan = tmp_path / 'scan.h5'
    _write(scan, {CA_METADATA_PATH: {'I0': [1.0], 'I1': [1.0, 2.0]}})
    assert sorted(load_ca_channels(str(scan))) == ['I1']


def test_a_missing_file_or_group_yields_no_divisors(tmp_path):
    assert load_ca_channels(None) == {}
    assert load_ca_channels(str(tmp_path / 'nope.h5')) == {}
    scan = tmp_path / 'bare.h5'
    _write(scan, {'entry/data': {}})
    assert load_ca_channels(str(scan)) == {}
