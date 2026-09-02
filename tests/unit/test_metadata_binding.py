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

"""Fail-closed metadata binding: reject rather than mis-place a frame."""

import pytest

from dashpva.utils.metadata_binding import (
    BindingRejection,
    ChannelClass,
    ChannelSpec,
    MetadataBinder,
    classify_hkl_channels,
)

CHANNELS = [
    ChannelSpec("ioc:Mu:Position", ChannelClass.REQUIRED_DYNAMIC),
    ChannelSpec("ioc:Mu:DirectionAxis", ChannelClass.STATIC),
    ChannelSpec("ioc:Temperature", ChannelClass.OPTIONAL),
]

GOOD = {
    "ioc:Mu:Position": 12.5,
    "ioc:Mu:DirectionAxis": "z-",
    "ioc:Temperature": 300.0,
}


def _binder(**kwargs):
    return MetadataBinder(CHANNELS, **kwargs)


class TestHappyPath:
    def test_binds_a_complete_frame(self):
        binder = _binder()
        bound = binder.bind(GOOD, frame_id=1, timestamp=100.0)
        assert bound is not None
        assert bound.values["ioc:Mu:Position"] == 12.5
        assert bound.frame_id == 1
        assert binder.counters.frames_bound == 1
        assert binder.counters.frames_rejected == 0

    def test_optional_channels_never_block_a_frame(self):
        binder = _binder()
        attributes = {k: v for k, v in GOOD.items() if k != "ioc:Temperature"}
        assert binder.bind(attributes, frame_id=1, timestamp=100.0) is not None
        assert binder.counters.frames_rejected == 0

    def test_static_geometry_may_arrive_once_and_stop(self):
        """Directions and UB are latched, not republished every frame."""
        binder = _binder()
        assert binder.bind(GOOD, frame_id=1, timestamp=100.0) is not None
        later = binder.bind({"ioc:Mu:Position": 13.0}, frame_id=2, timestamp=101.0)
        assert later is not None
        assert later.values["ioc:Mu:DirectionAxis"] == "z-"


class TestFailClosed:
    def test_missing_required_dynamic_is_rejected_not_guessed(self):
        """A stale angle mis-places every pixel with no visual tell."""
        binder = _binder()
        assert binder.bind({"ioc:Mu:DirectionAxis": "z-"}, frame_id=1, timestamp=1.0) is None
        assert binder.counters.rejections[BindingRejection.MISSING_REQUIRED.value] == 1

    def test_static_missing_before_it_was_ever_seen_is_rejected(self):
        binder = _binder()
        assert binder.bind({"ioc:Mu:Position": 1.0}, frame_id=1, timestamp=1.0) is None
        assert binder.counters.frames_rejected == 1

    def test_frame_without_a_timestamp_is_rejected(self):
        binder = _binder()
        assert binder.bind(GOOD, frame_id=1, timestamp=None) is None
        assert binder.counters.rejections[BindingRejection.NO_TIMESTAMP.value] == 1

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_required_value_is_rejected(self, bad):
        binder = _binder()
        attributes = dict(GOOD, **{"ioc:Mu:Position": bad})
        assert binder.bind(attributes, frame_id=1, timestamp=1.0) is None
        assert binder.counters.rejections[BindingRejection.NONFINITE_REQUIRED.value] == 1

    def test_metadata_older_than_the_tolerance_is_rejected(self):
        binder = _binder(max_age_seconds=0.5)
        timestamps = {"ioc:Mu:Position": 99.0}
        assert binder.bind(
            GOOD, frame_id=1, timestamp=100.0, metadata_timestamps=timestamps
        ) is None
        assert binder.counters.rejections[BindingRejection.STALE_TIMESTAMP.value] == 1

    def test_metadata_within_the_tolerance_is_accepted(self):
        binder = _binder(max_age_seconds=0.5)
        timestamps = {"ioc:Mu:Position": 99.8}
        assert binder.bind(
            GOOD, frame_id=1, timestamp=100.0, metadata_timestamps=timestamps
        ) is not None

    def test_required_dynamic_timestamp_is_required_when_age_check_enabled(self):
        binder = _binder(max_age_seconds=0.5)
        assert binder.bind(GOOD, frame_id=1, timestamp=100.0) is None
        assert binder.counters.rejections[BindingRejection.NO_TIMESTAMP.value] == 1

    def test_static_channels_do_not_require_fresh_timestamps(self):
        binder = _binder(max_age_seconds=0.5)
        timestamps = {"ioc:Mu:Position": 100.0}
        assert binder.bind(
            GOOD, frame_id=1, timestamp=100.0, metadata_timestamps=timestamps
        ) is not None


class TestMonitor:
    def _monitor_binder(self):
        channels = CHANNELS + [
            ChannelSpec("ioc:I0", ChannelClass.REQUIRED_DYNAMIC, require_positive=True)
        ]
        return MetadataBinder(channels, monitor_channel="ioc:I0")

    def test_positive_monitor_is_bound(self):
        bound = self._monitor_binder().bind(
            dict(GOOD, **{"ioc:I0": 4.0}), frame_id=1, timestamp=1.0
        )
        assert bound is not None and bound.monitor == 4.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_monitor_is_rejected(self, bad):
        """Dividing by it would corrupt the accumulation, not just this frame."""
        binder = self._monitor_binder()
        assert binder.bind(dict(GOOD, **{"ioc:I0": bad}), frame_id=1, timestamp=1.0) is None
        assert binder.counters.rejections[BindingRejection.NONPOSITIVE_MONITOR.value] == 1

    def test_missing_monitor_is_rejected_when_normalizing(self):
        binder = self._monitor_binder()
        assert binder.bind(GOOD, frame_id=1, timestamp=1.0) is None
        assert binder.counters.frames_rejected == 1


class TestFrameIdentity:
    def test_duplicate_frame_id_is_rejected(self):
        binder = _binder()
        binder.bind(GOOD, frame_id=7, timestamp=1.0)
        assert binder.bind(GOOD, frame_id=7, timestamp=2.0) is None
        assert binder.counters.rejections[BindingRejection.DUPLICATE_FRAME_ID.value] == 1

    def test_gap_is_counted_and_flagged_but_the_frame_still_binds(self):
        """A gap means upstream dropped something, not that this frame is wrong."""
        binder = _binder()
        binder.bind(GOOD, frame_id=1, timestamp=1.0)
        bound = binder.bind(GOOD, frame_id=5, timestamp=2.0)
        assert bound is not None and bound.followed_gap
        assert binder.counters.id_gaps == 1
        assert binder.counters.frames_missing_upstream == 3
        assert bound.missing_before == 3

    def test_consecutive_ids_report_no_gap(self):
        binder = _binder()
        binder.bind(GOOD, frame_id=1, timestamp=1.0)
        bound = binder.bind(GOOD, frame_id=2, timestamp=2.0)
        assert bound is not None and not bound.followed_gap
        assert binder.counters.id_gaps == 0

    def test_out_of_order_ids_are_rejected_without_moving_the_high_water_mark(self):
        binder = _binder()
        binder.bind(GOOD, frame_id=5, timestamp=1.0)
        assert binder.bind(GOOD, frame_id=3, timestamp=2.0) is None
        assert binder.counters.ids_out_of_order == 1
        assert binder.bind(GOOD, frame_id=6, timestamp=3.0) is not None
        assert binder.counters.id_gaps == 0

    def test_missing_frame_id_is_rejected(self):
        binder = _binder()
        assert binder.bind(GOOD, frame_id=None, timestamp=1.0) is None
        reason = BindingRejection.MISSING_FRAME_ID.value
        assert binder.counters.rejections[reason] == 1

    def test_metadata_rejection_does_not_create_a_false_upstream_gap(self):
        binder = _binder()
        assert binder.bind(GOOD, frame_id=1, timestamp=1.0) is not None
        assert binder.bind({}, frame_id=2, timestamp=2.0) is None
        assert binder.bind(GOOD, frame_id=3, timestamp=3.0) is not None
        assert binder.counters.frames_missing_upstream == 0
        assert binder.counters.id_gap_events == 0

    def test_reset_clears_history_and_counters(self):
        binder = _binder()
        binder.bind(GOOD, frame_id=1, timestamp=1.0)
        binder.reset()
        assert binder.counters.frames_bound == 0
        # The same id is reusable after a reset, and static cache is cleared.
        assert binder.bind(GOOD, frame_id=1, timestamp=1.0) is not None


class TestClassification:
    def test_circle_positions_are_required_and_the_rest_static(self):
        specs = classify_hkl_channels({
            "SAMPLE_CIRCLE_AXIS_1": {
                "POSITION": "ioc:Mu:Position",
                "DIRECTION_AXIS": "ioc:Mu:DirectionAxis",
            },
            "DETECTOR_CIRCLE_AXIS_1": {"POSITION": "ioc:Nu:Position"},
            "SPEC": {"ENERGY_VALUE": "ioc:Energy"},
        })
        by_name = {spec.name: spec.channel_class for spec in specs}
        assert by_name["ioc:Mu:Position"] is ChannelClass.REQUIRED_DYNAMIC
        assert by_name["ioc:Nu:Position"] is ChannelClass.REQUIRED_DYNAMIC
        assert by_name["ioc:Mu:DirectionAxis"] is ChannelClass.STATIC
        assert by_name["ioc:Energy"] is ChannelClass.STATIC

    def test_monitor_is_promoted_to_required_and_positive(self):
        specs = classify_hkl_channels(
            {"SPEC": {"ENERGY_VALUE": "ioc:Energy"}}, monitor_channel="ioc:I0"
        )
        monitor = next(spec for spec in specs if spec.name == "ioc:I0")
        assert monitor.channel_class is ChannelClass.REQUIRED_DYNAMIC
        assert monitor.require_positive

    def test_optional_channels_are_marked_optional(self):
        specs = classify_hkl_channels({}, optional_channels=("ioc:Temperature",))
        assert specs[0].channel_class is ChannelClass.OPTIONAL
