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

"""Fail-closed binding of scalar metadata to detector frames.

Qt-free: this runs inside the pvaccess grid consumer.

The live grid multiplies one frame's intensities by one set of circle angles.
If those angles belong to a different frame, every pixel lands at the wrong
place in reciprocal space and the result still looks like a diffraction volume
-- there is no visual tell. So binding is fail-closed: a frame whose required
metadata is missing, stale, or unusable is **rejected and counted**, never
gridded with a guessed value.

Channels are classified because not all metadata carries equal weight:

``STATIC``
    Geometry that does not change during a scan (directions, UB, detector
    calibration). A stale value is fine -- it is the same value.
``REQUIRED_DYNAMIC``
    Circle positions and, when normalizing, the monitor. These must be fresh
    for *this* frame; a stale one silently mis-places the frame.
``OPTIONAL``
    Environment (temperature, and so on). Recorded when present, never a
    reason to reject.

Scope: frame-bound and trigger-latched metadata only, i.e. values delivered in
the frame's own attribute list. Asynchronous fly-scan interpolation and
pulse-ID joins are a separate problem and deliberately not attempted here --
guessing at them is exactly the silent mis-binding this module exists to stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "BindingCounters",
    "BindingRejection",
    "BoundFrame",
    "ChannelClass",
    "ChannelSpec",
    "MetadataBinder",
    "METADATA_TIMESTAMP_ATTRIBUTE_PREFIX",
]

METADATA_TIMESTAMP_ATTRIBUTE_PREFIX = "dashpva:metadata_timestamp:"


class ChannelClass(Enum):
    STATIC = "static"
    REQUIRED_DYNAMIC = "required_dynamic"
    OPTIONAL = "optional"


class BindingRejection(Enum):
    """Why a frame was refused. Each is counted separately so the cause is
    visible rather than showing up only as a low frame count."""

    NO_TIMESTAMP = "no_timestamp"
    STALE_TIMESTAMP = "stale_timestamp"
    MISSING_REQUIRED = "missing_required"
    NONFINITE_REQUIRED = "nonfinite_required"
    NONPOSITIVE_MONITOR = "nonpositive_monitor"
    MISSING_FRAME_ID = "missing_frame_id"
    DUPLICATE_FRAME_ID = "duplicate_frame_id"
    OUT_OF_ORDER_FRAME_ID = "out_of_order_frame_id"


@dataclass(frozen=True)
class ChannelSpec:
    """One metadata channel and how strictly it is treated."""

    name: str
    channel_class: ChannelClass
    #: Reject the frame when the value is not strictly positive. Set for the
    #: monitor: dividing by zero or a negative corrupts the accumulation.
    require_positive: bool = False


@dataclass
class BindingCounters:
    frames_bound: int = 0
    frames_rejected: int = 0
    id_gap_events: int = 0
    frames_missing_upstream: int = 0
    ids_out_of_order: int = 0
    rejections: dict = field(default_factory=dict)

    def reject(self, reason: BindingRejection) -> None:
        self.frames_rejected += 1
        self.rejections[reason.value] = self.rejections.get(reason.value, 0) + 1

    def as_dict(self) -> dict:
        return {
            "frames_bound": self.frames_bound,
            "frames_rejected": self.frames_rejected,
            "id_gap_events": self.id_gap_events,
            "frames_missing_upstream": self.frames_missing_upstream,
            "ids_out_of_order": self.ids_out_of_order,
            **{f"rejected_{key}": value for key, value in self.rejections.items()},
        }

    @property
    def id_gaps(self) -> int:
        """Compatibility alias for the old, ambiguous counter name."""
        return self.id_gap_events


@dataclass(frozen=True)
class BoundFrame:
    """A frame whose required metadata was present, fresh and usable."""

    values: Mapping[str, Any]
    frame_id: Optional[int]
    timestamp: Optional[float]
    monitor: Optional[float]
    #: True when this frame's id was not the previous id plus one. The frame is
    #: still bound -- a gap means upstream dropped something, not that this
    #: frame is wrong -- but the preview is flagged incomplete from here on.
    followed_gap: bool = False
    missing_before: int = 0


class MetadataBinder:
    """Validates and binds per-frame scalar metadata, fail-closed.

    ``max_age_seconds`` bounds how old a REQUIRED_DYNAMIC value may be relative
    to the frame timestamp. ``None`` disables the age check for setups where
    metadata carries no timestamp of its own.
    """

    def __init__(
        self,
        channels: Sequence[ChannelSpec],
        *,
        monitor_channel: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        self.channels = tuple(channels)
        self.monitor_channel = monitor_channel
        self.max_age_seconds = max_age_seconds
        self.counters = BindingCounters()
        self._last_frame_id: Optional[int] = None
        self._static_cache: dict[str, Any] = {}

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self.channels
            if spec.channel_class is ChannelClass.REQUIRED_DYNAMIC
        )

    @property
    def static_names(self) -> tuple[str, ...]:
        return tuple(
            spec.name
            for spec in self.channels
            if spec.channel_class is ChannelClass.STATIC
        )

    def _finite(self, value: Any) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            # Non-numeric channels (direction strings, units) are carried
            # through as-is; only numbers get a finiteness check.
            return value is not None
        return math.isfinite(numeric)

    def bind(
        self,
        attributes: Mapping[str, Any],
        *,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        metadata_timestamps: Optional[Mapping[str, float]] = None,
    ) -> Optional[BoundFrame]:
        """Return a BoundFrame, or None (counted) if the frame must be refused."""
        if frame_id is None:
            self.counters.reject(BindingRejection.MISSING_FRAME_ID)
            return None

        followed_gap = False
        missing_before = 0
        frame_id = int(frame_id)
        if self._last_frame_id is not None and frame_id == self._last_frame_id:
            self.counters.reject(BindingRejection.DUPLICATE_FRAME_ID)
            return None
        if self._last_frame_id is not None:
            if frame_id < self._last_frame_id:
                self.counters.ids_out_of_order += 1
                self.counters.reject(BindingRejection.OUT_OF_ORDER_FRAME_ID)
                return None
            if frame_id > self._last_frame_id + 1:
                missing_before = frame_id - self._last_frame_id - 1
                self.counters.id_gap_events += 1
                self.counters.frames_missing_upstream += missing_before
                followed_gap = True

        # Consume valid ids before metadata checks so rejected frames are not
        # later misreported as missing upstream.
        self._last_frame_id = frame_id

        if timestamp is None:
            self.counters.reject(BindingRejection.NO_TIMESTAMP)
            return None

        values: dict[str, Any] = {}
        for spec in self.channels:
            present = spec.name in attributes and attributes[spec.name] is not None
            if present:
                value = attributes[spec.name]
                if spec.channel_class is ChannelClass.STATIC:
                    self._static_cache[spec.name] = value
            elif spec.channel_class is ChannelClass.STATIC:
                # Static geometry legitimately arrives once and stops; reuse it.
                if spec.name not in self._static_cache:
                    self.counters.reject(BindingRejection.MISSING_REQUIRED)
                    return None
                value = self._static_cache[spec.name]
            elif spec.channel_class is ChannelClass.OPTIONAL:
                continue
            else:
                self.counters.reject(BindingRejection.MISSING_REQUIRED)
                return None

            if (
                spec.channel_class is ChannelClass.REQUIRED_DYNAMIC
                and self.max_age_seconds is not None
            ):
                metadata_timestamp = (metadata_timestamps or {}).get(spec.name)
                if metadata_timestamp is None:
                    self.counters.reject(BindingRejection.NO_TIMESTAMP)
                    return None
                if abs(timestamp - metadata_timestamp) > self.max_age_seconds:
                    self.counters.reject(BindingRejection.STALE_TIMESTAMP)
                    return None

            if spec.channel_class is not ChannelClass.OPTIONAL and not self._finite(value):
                self.counters.reject(BindingRejection.NONFINITE_REQUIRED)
                return None
            if spec.require_positive:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    self.counters.reject(BindingRejection.NONFINITE_REQUIRED)
                    return None
                if not (math.isfinite(numeric) and numeric > 0):
                    self.counters.reject(BindingRejection.NONPOSITIVE_MONITOR)
                    return None
            values[spec.name] = value

        monitor = None
        if self.monitor_channel:
            if self.monitor_channel not in values:
                self.counters.reject(BindingRejection.MISSING_REQUIRED)
                return None
            monitor = float(values[self.monitor_channel])

        self.counters.frames_bound += 1
        return BoundFrame(
            values=values,
            frame_id=frame_id,
            timestamp=timestamp,
            monitor=monitor,
            followed_gap=followed_gap,
            missing_before=missing_before,
        )

    def reset(self, *, keep_static: bool = False) -> None:
        """Forget per-run identity/counters, optionally retaining static values."""
        self.counters = BindingCounters()
        self._last_frame_id = None
        if not keep_static:
            self._static_cache.clear()

    def observe_inactive_frame_id(self, frame_id: Optional[int]) -> None:
        """Advance identity across an intentional stopped interval."""
        if frame_id is None:
            return
        try:
            value = int(frame_id)
        except (TypeError, ValueError):
            return
        if self._last_frame_id is None or value > self._last_frame_id:
            self._last_frame_id = value


def classify_hkl_channels(
    hkl_config: Mapping[str, Mapping[str, str]],
    *,
    monitor_channel: Optional[str] = None,
    optional_channels: Sequence[str] = (),
) -> list[ChannelSpec]:
    """Derive channel specs from a profile's effective ``[HKL]`` section.

    Circle POSITION channels are the ones that must be fresh per frame; the
    directions, UB, energy and detector calibration are static geometry.
    """
    specs: list[ChannelSpec] = []
    seen: set[str] = set()

    def _add(name: str, channel_class: ChannelClass, positive: bool = False) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        specs.append(ChannelSpec(name, channel_class, positive))

    for section, fields in (hkl_config or {}).items():
        if not isinstance(fields, Mapping):
            continue
        is_circle = section.startswith(("SAMPLE_CIRCLE", "DETECTOR_CIRCLE"))
        for key, channel in fields.items():
            if not channel:
                continue
            if is_circle and key == "POSITION":
                _add(channel, ChannelClass.REQUIRED_DYNAMIC)
            else:
                _add(channel, ChannelClass.STATIC)

    if monitor_channel:
        seen.discard(monitor_channel)
        specs = [spec for spec in specs if spec.name != monitor_channel]
        _add(monitor_channel, ChannelClass.REQUIRED_DYNAMIC, positive=True)
    for name in optional_channels:
        _add(name, ChannelClass.OPTIONAL)
    return specs
