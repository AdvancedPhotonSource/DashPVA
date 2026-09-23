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
    ROIPlotDock,
    load_ca_channels,
    norm_array,
    norm_key,
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


def test_a_combo_with_no_selection_means_no_normalization():
    class _Combo:
        def currentData(self):
            return ''
    assert norm_key(_Combo()) == ''
    assert norm_array(_Combo(), {'I0': np.array([1.0, 2.0])}) is None
    assert norm_key(None) == ''


def test_a_selected_channel_resolves_to_its_readings():
    class _Combo:
        def currentData(self):
            return 'I0'
    out = norm_array(_Combo(), {'I0': [1.0, 2.0, 3.0]})
    assert list(out) == [1.0, 2.0, 3.0]
    assert norm_array(_Combo(), {}) is None


class _Spinbox:
    def __init__(self, frame):
        self._frame = frame

    def value(self):
        return self._frame


class _Dock:
    """Stand-in carrying only what ``_normalize_frame`` reads."""

    _normalize_frame = ROIPlotDock._normalize_frame

    def __init__(self, readings, frame):
        self._readings = np.asarray(readings, dtype=float)
        self.main = type('M', (), {'frame_spinbox': _Spinbox(frame)})()

    def _norm_array(self):
        return self._readings


@pytest.mark.parametrize('divisor', [0.0, -4.0, np.nan, np.inf])
def test_single_frame_leaves_an_invalid_divisor_undivided(divisor):
    """Same rule as the time-series path: a frame that gaps there cannot divide here.

    A negative reading would otherwise invert the whole projection and plot it
    under a "/ channel" label as though it were real.
    """
    values, note = _Dock([2.0, divisor], frame=1)._normalize_frame([10.0, 20.0])
    assert list(values) == [10.0, 20.0]
    assert note == ' (unavailable)'


def test_single_frame_divides_by_a_positive_reading():
    values, note = _Dock([2.0, 4.0], frame=1)._normalize_frame([10.0, 20.0])
    assert list(values) == [2.5, 5.0]
    assert note == ''


def test_single_frame_without_a_reading_for_that_frame():
    values, note = _Dock([2.0], frame=7)._normalize_frame([10.0])
    assert list(values) == [10.0]
    assert note == ' (unavailable)'
