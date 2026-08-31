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

"""Profile-driven RSM-parameter IOC and staged geometry editor.

The GUI and pvAccess IOC run in separate processes. The child reads the active
profile from the database and is restarted only after a successful
compare-and-swap save -- so by the time it starts, the database already holds
exactly the snapshot that was approved. Configuration is never handed over as
a side-channel file: per AGENTS.md the database is the config source, with
TOML reserved for import/export.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

import numpy as np

import dashpva.settings as app_settings
from dashpva.utils.rsm_geometry import DETECTOR_SETUP_FIELDS, FRAME_AXIS_ORDERS
from dashpva.utils.rsm_parameter_config import (
    RSMParameterEditSession,
    RSMParameterProfile,
    SnapshotActivationError,
    adoption_diff,
    apply_and_activate,
    merge_live_records,
    profile_from_raw,
    requires_adoption_confirmation,
    update_raw_profile,
    validate_distance,
    validate_parameter_profile,
    validate_source_pv,
    validate_ub_matrix,
)


def _flatten_config(
    value: object, prefix: str = "", *, index_lists: bool = False
) -> dict[str, Any]:
    """Flatten configuration fields for review without losing axis rows."""
    if index_lists and isinstance(value, (list, tuple)) and any(
        isinstance(item, Mapping) for item in value
    ):
        flattened: dict[str, Any] = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(
                _flatten_config(item, path, index_lists=index_lists)
            )
        return flattened
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(item, path, index_lists=index_lists))
    return flattened


def _restore_axis_row(
    rows: list[tuple[int | None, Mapping[str, Any]]],
    origin: int,
    values: Mapping[str, Any],
) -> list[tuple[int | None, dict[str, Any]]]:
    """Restore a removed loaded axis without changing any surviving identity."""
    restored = [(item_origin, dict(item)) for item_origin, item in rows]
    if any(item_origin == origin for item_origin, _item in restored):
        return restored
    insertion = len(restored)
    for row, (candidate, _item) in enumerate(restored):
        if candidate is not None and candidate >= 0 and candidate > origin:
            insertion = row
            break
    restored.insert(insertion, (origin, dict(values)))
    return restored


def _reorder_loaded_axis_rows(
    rows: list[tuple[int | None, Mapping[str, Any]]],
    order: list[int],
) -> list[tuple[int | None, dict[str, Any]]]:
    """Reorder loaded axes in place while keeping new-axis slots unchanged."""
    copied = [(origin, dict(values)) for origin, values in rows]
    loaded = [
        origin for origin, _values in copied if origin is not None and origin >= 0
    ]
    if len(order) != len(loaded) or set(order) != set(loaded):
        raise ValueError("axis order must contain every loaded identity exactly once")
    by_origin = {origin: values for origin, values in copied if origin is not None}
    ordered = iter(
        (origin, by_origin[origin]) for origin in order if origin in by_origin
    )
    return [
        next(ordered) if origin is not None and origin >= 0 else (origin, values)
        for origin, values in copied
    ]


def _review_pending_change(build_pending):
    """Keep an invalid staged RSM form visible to the close gate."""
    try:
        return build_pending(), []
    except ValueError as exc:
        return None, [
            (
                "change",
                "RSM_CONFIGURATION_INVALID",
                "valid loaded configuration",
                str(exc),
            )
        ]


def _has_pending_rsm_change(build_pending, raw: Mapping[str, Any]) -> bool:
    """Treat invalid staged values as dirty until saved or discarded."""
    try:
        return build_pending()["replacement"] != raw
    except ValueError:
        return True


class ActiveProfileChanged(RuntimeError):
    """The active profile was switched out from under this editor.

    The edit session and the running IOC subprocess are both bound to
    whichever profile was active when this window opened -- the IOC's record
    database is built from that profile at launch. If the active profile now
    resolves to something else (selected in another DashPVA window), that is
    a different situation from an activation failure: nothing went wrong, the
    target simply moved, so it gets its own type, message, and no retry.
    """

    def __init__(self, expected: object, current: object):
        super().__init__(
            f"active profile changed from {expected!r} to {current!r} in another process"
        )
        self.expected = expected
        self.current = current


class ProfileContentMismatch(RuntimeError):
    """The right profile came back, but not with the values that were saved.

    Usually a concurrent writer: something else changed the same profile
    between the save and the read-back. Carries a readable list of the fields
    that disagree, since "which settings differ" is the part a beamline user
    can actually act on.
    """

    def __init__(self, differences: list[str], count: int):
        super().__init__(f"{count} setting(s) differ from what was saved")
        self.differences = differences
        self.count = count


def _active_profile_identity() -> object:
    """Which profile the central settings currently resolve to.

    Must be the *resolved* identity, not app_settings.LOCATOR -- the locator
    is None whenever the profile is auto-detected from the database
    selection, so comparing locators would report "unchanged" across exactly
    the switch this is meant to catch.

    KNOWN LIMITATION: a transient failure here (e.g. brief DB contention) is
    indistinguishable from "genuinely no active profile" -- both return None.
    If that transient None differs from the identity resolved at a different
    call site (window-open vs. post-save recheck), _classify_activation_mismatch
    will report ActiveProfileChanged for a profile that never actually moved,
    masking what may really be a retriable content mismatch. Narrow and rare
    (requires a DB hiccup at exactly the comparison instant); not fixed here,
    since a proper fix needs a way to distinguish "couldn't resolve" from "no
    profile" that ConfigSource.resolved_identity() doesn't currently provide.
    """
    from dashpva.utils.config.source import ConfigSource

    try:
        return ConfigSource(app_settings.LOCATOR).resolved_identity()
    except Exception as exc:
        print(f"[HKL Setup] could not resolve active profile identity: {exc}", flush=True)
        return None


_AXIS_LIST_KEYS = ("SAMPLE_AXES", "DETECTOR_AXES")


def _pretty_config_path(path: str) -> str:
    """Turn a dotted raw-profile path into something readable at a beamline.

    'IOC_RSM_PARAMETER.SAMPLE_AXES.0.DIRECTION' -> 'Sample axis 1 - DIRECTION'
    """
    parts = [part for part in path.split(".") if part and part != "IOC_RSM_PARAMETER"]
    pretty: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if (
            part in _AXIS_LIST_KEYS
            and index + 1 < len(parts)
            and parts[index + 1].isdigit()
        ):
            role = "Sample" if part == "SAMPLE_AXES" else "Detector"
            pretty.append(f"{role} axis {int(parts[index + 1]) + 1}")
            index += 2
            continue
        pretty.append(part)
        index += 1
    return " - ".join(pretty) if pretty else path


def _flatten_for_diff(value: object, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings/axis-lists for the activation-mismatch diff.

    Deliberately separate from _flatten_config (used by the change-review
    diff above): this only needs to say which raw-profile fields differ
    between what was saved and what got read back, over the *whole* raw
    profile document, not to drive the change-review keep/drop UI.
    """
    if not isinstance(value, Mapping):
        return {prefix: value}
    if not value:
        # An explicitly empty section ({}) must stay distinguishable from the
        # key being entirely absent -- both used to flatten to "no entries at
        # this path", which made a real difference (one side {}, the other
        # side missing) silently vanish from the diff.
        return {prefix: {}} if prefix else {}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_for_diff(item, path))
        elif isinstance(item, list) and any(isinstance(sub, Mapping) for sub in item):
            for index, sub in enumerate(item):
                flat.update(_flatten_for_diff(sub, f"{path}.{index}"))
        else:
            flat[path] = item
    return flat


def _describe_config_differences(
    saved: Mapping[str, Any],
    reloaded: Mapping[str, Any],
    limit: int = 15,
) -> tuple[list[str], int]:
    """Field-by-field description of how a re-read profile differs from the save.

    Returns (lines, total); lines is capped at *limit* because a whole-profile
    difference runs to dozens of entries and a wall of them helps nobody.
    """
    saved_flat = _flatten_for_diff(saved)
    reloaded_flat = _flatten_for_diff(reloaded)
    missing = object()
    lines: list[str] = []
    total = 0
    for path in sorted(set(saved_flat) | set(reloaded_flat)):
        was = saved_flat.get(path, missing)
        now = reloaded_flat.get(path, missing)
        if was == now:
            continue
        total += 1
        if len(lines) < limit:
            lines.append(
                f"{_pretty_config_path(path)}: "
                f"saved {'(not set)' if was is missing else was!r}, "
                f"read back {'(not set)' if now is missing else now!r}"
            )
    if total > len(lines):
        lines.append(f"...and {total - len(lines)} more")
    return lines, total


def _profile_display_name(identity: object) -> str:
    """Name a profile the way the user chose it, never as a bare database id."""
    if identity is None:
        return "the auto-selected profile"
    if isinstance(identity, str):
        return os.path.basename(identity) or identity
    try:
        from dashpva.database.interface import DatabaseInterface

        profile = DatabaseInterface().get_profile_by_id(int(identity))
        if profile is not None and getattr(profile, "name", None):
            return str(profile.name)
    except Exception:
        pass
    return f"profile {identity}"


def _classify_activation_mismatch(
    expected: object,
    current: object,
    saved: Mapping[str, Any] | None = None,
    reloaded: Mapping[str, Any] | None = None,
) -> Exception:
    """Explain why reloaded settings don't match the snapshot being activated.

    A moved profile means the target was switched elsewhere; the same profile
    coming back with different values means something wrote to it
    concurrently. The two need different advice, so they get different types.
    """
    if current != expected:
        return ActiveProfileChanged(expected, current)
    differences, total = _describe_config_differences(saved or {}, reloaded or {})
    return ProfileContentMismatch(differences, total)


def _parse_ub_or_pv(
    text: str, current_fallback: "tuple[float, ...]"
) -> "tuple[tuple[float, ...], str]":
    """Split one 'UB Matrix (PV or value)' field into (literal, source_pv).

    Text that parses as JSON must be a valid, full-rank 9-number UB matrix or
    the whole entry is rejected -- a JSON-looking typo must never be silently
    reinterpreted as a PV name. Text that isn't JSON is a PV name (a bare CA
    PV name is never valid JSON), and the previously validated literal is
    kept as the fallback the IOC publishes if that PV is ever unavailable.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError(
            "UB Matrix must not be blank: enter a 9-number JSON array or a PV name"
        )
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        if stripped.startswith(("[", "{")):
            raise ValueError(f"UB Matrix contains malformed JSON: {exc.msg}") from exc
        return tuple(current_fallback), validate_source_pv(stripped, "UB Matrix source")
    return validate_ub_matrix(parsed, "UB Matrix"), ""


def _format_ub_or_pv(ub_matrix: "Any", source_pv: str) -> str:
    if source_pv:
        return source_pv
    return json.dumps(list(ub_matrix), sort_keys=True)


def _parse_distance_or_pv(text: str, current_fallback: float) -> "tuple[float, str]":
    """Split one 'Distance (PV or value)' field into (literal, source_pv).

    Same rule as _parse_ub_or_pv: a number that fails validation (non-finite,
    non-positive) is an error, never silently reinterpreted as a PV name.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Distance must not be blank: enter a number or a PV name")
    try:
        literal = float(stripped)
    except ValueError:
        return float(current_fallback), validate_source_pv(
            stripped, "Distance source"
        )
    return validate_distance(literal, "Distance"), ""


def _format_distance_or_pv(distance: float, source_pv: str) -> str:
    if source_pv:
        return source_pv
    return json.dumps(distance)


#: Placeholder text per detector field, so the expected shape is visible
#: without consulting the docs. Keys not listed simply get no hint.
_DETECTOR_FIELD_HINTS: dict[str, str] = {
    "PIXEL_DIRECTION_1": "z-",
    "PIXEL_DIRECTION_2": "x-",
    "CENTER_CHANNEL_PIXEL": "direction-1 pixel, direction-2 pixel",
    "SIZE": "direction-1 size, direction-2 size",
    "PIXEL_SIZE": "direction-1 size, direction-2 size",
    "DETECTOR_SHAPE": "direction-1 pixels, direction-2 pixels (full unbinned detector)",
    "BINNING": "direction-1 binning, direction-2 binning",
    "ROI": "start1, stop1, start2, stop2 (half-open, unbinned)",
    "DETROT": "degrees",
    "TILT": "degrees",
    "TILTAZIMUTH": "degrees",
    "UNITS": "mm",
    "DISTANCE_UNITS": "mm",
    "SIZE_UNITS": "mm",
    "PIXEL_SIZE_UNITS": "mm",
    "ANGLE_UNITS": "deg",
    "FRAME_AXIS_ORDER": f"one of {', '.join(FRAME_AXIS_ORDERS)}",
}

_DETECTOR_FIELD_LABELS: dict[str, str] = {
    "PIXEL_DIRECTION_1": "Pixel direction 1",
    "PIXEL_DIRECTION_2": "Pixel direction 2",
    "CENTER_CHANNEL_PIXEL": "Center channel pixel",
    "DISTANCE": "Distance (PV or value)",
    "SIZE": "Size",
    "PIXEL_SIZE": "Pixel size",
    "DETECTOR_SHAPE": "Detector shape",
    "BINNING": "Binning",
    "ROI": "ROI",
    "DETROT": "Detector rotation",
    "TILT": "Tilt",
    "TILTAZIMUTH": "Tilt azimuth",
    "UNITS": "Units",
    "DISTANCE_UNITS": "Distance units",
    "SIZE_UNITS": "Size units",
    "PIXEL_SIZE_UNITS": "Pixel size units",
    "ANGLE_UNITS": "Angle units",
    "FRAME_AXIS_ORDER": "Frame axis order",
}

_DETECTOR_FIELD_DESCRIPTIONS: dict[str, str] = {
    "PIXEL_DIRECTION_1": "Beamline [xyz][+-] direction of detector direction 1.",
    "PIXEL_DIRECTION_2": "Beamline [xyz][+-] direction of detector direction 2.",
    "CENTER_CHANNEL_PIXEL": (
        "Direct-beam center ordered as detector direction 1, direction 2 "
        "in full-frame, unbinned pixels."
    ),
    "SIZE": "Full detector-face size along detector directions 1 and 2.",
    "PIXEL_SIZE": "One unbinned pixel size along detector directions 1 and 2.",
    "DETECTOR_SHAPE": (
        "Full unbinned pixel count along detector directions 1 and 2, not array rows/columns."
    ),
    "BINNING": "Acquisition binning along detector directions 1 and 2.",
    "ROI": (
        "Half-open, unbinned bounds [start1, stop1, start2, stop2) along "
        "detector directions 1 and 2."
    ),
    "DETROT": "Detector rotation about the primary-beam direction.",
    "TILT": "Detector-plane tilt away from perpendicular to the beam.",
    "TILTAZIMUTH": "Azimuth of the detector tilt.",
    "UNITS": "Default length unit for distance and detector size.",
    "DISTANCE_UNITS": "Length unit for distance; overrides UNITS.",
    "SIZE_UNITS": "Length unit for detector size; overrides UNITS.",
    "PIXEL_SIZE_UNITS": "Length unit for pixel size; overrides UNITS.",
    "ANGLE_UNITS": "Angle unit for detector rotation, tilt, and tilt azimuth.",
    "FRAME_AXIS_ORDER": (
        "The only field that maps detector directions 1/2 onto acquired-array rows/columns."
    ),
}

_UB_INPUT_TOOLTIP = (
    "Enter a flat row-major JSON array of exactly 9 finite, full-rank numbers, "
    "or one CA PV whose waveform publishes that format."
)


def _format_detector_value(value: object) -> str:
    """Render one detector value for a plain text field (lists as 'a, b').

    Floats keep their trailing '.0' so the round-trip preserves the stored
    type -- rendering 300.0 as '300' would quietly rewrite it to an int on
    save.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_detector_value(item) for item in value)
    return str(value)


def _parse_detector_value(text: str) -> Any:
    """Inverse of _format_detector_value.

    Numbers come back as numbers and everything else stays a string, which is
    what the DETECTOR_SETUP validators expect -- directions and units are
    strings, geometry is numeric.
    """

    def one(token: str) -> Any:
        token = token.strip()
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    stripped = text.strip()
    if stripped.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid detector list: {text}") from exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"invalid detector list: {text}")
        return [one(str(part)) for part in parsed]
    if "," in stripped:
        return [one(part) for part in stripped.split(",")]
    return one(stripped)


def _ai(name: str) -> str:
    return (
        f'record(ai, "{name}") {{\n'
        '  field(DTYP, "Soft Channel")\n'
        '  field(PREC, "6")\n'
        '}\n'
    )


def _longout(name: str) -> str:
    return f'record(longout, "{name}") {{\n  field(DTYP, "Soft Channel")\n}}\n'


def _stringout(name: str) -> str:
    return f'record(stringout, "{name}") {{\n  field(DTYP, "Soft Channel")\n}}\n'


def _waveform(name: str, count: int) -> str:
    return (
        f'record(waveform, "{name}") {{\n'
        '  field(DTYP, "Soft Channel")\n'
        '  field(FTVL, "DOUBLE")\n'
        f'  field(NELM, "{count}")\n'
        '}\n'
    )


def build_ioc_database(profile: RSMParameterProfile) -> str:
    """Build an EPICS database for every configured circle without a max."""
    lines: list[str] = []
    prefix = profile.prefix
    for axis in profile.axes:
        base = f"{prefix}{axis.record_name}"
        lines.extend(
            (
                _ai(f"{base}:Position"),
                _ai(f"{base}:AxisNumber"),
                _stringout(f"{base}:DirectionAxis"),
                _stringout(f"{base}:SpecMotorName"),
            )
        )

    lines.extend(
        (
            _ai(f"{prefix}spec:Energy:Value"),
            _stringout(f"{prefix}spec:Energy:Units"),
            _waveform(f"{prefix}spec:UB_matrix:Value", 9),
        )
    )
    for group in (
        "PrimaryBeamDirection",
        "InplaneReferenceDirection",
        "SampleSurfaceNormalDirection",
    ):
        for index in range(1, 4):
            lines.append(_ai(f"{prefix}{group}:AxisNumber{index}"))

    lines.extend(
        (
            _stringout(f"{prefix}DetectorSetup:PixelDirection1"),
            _stringout(f"{prefix}DetectorSetup:PixelDirection2"),
            _waveform(f"{prefix}DetectorSetup:CenterChannelPixel", 2),
            _waveform(f"{prefix}DetectorSetup:Size", 2),
            _ai(f"{prefix}DetectorSetup:Distance"),
            _stringout(f"{prefix}DetectorSetup:Units"),
            _longout(f"{prefix}ScanOn:Value"),
            _stringout(f"{prefix}FilePath:Value"),
            _stringout(f"{prefix}FileName:Value"),
        )
    )
    return "".join(lines)


def static_ioc_values(profile: RSMParameterProfile) -> dict[str, Any]:
    """Return all frame-invariant IOC values for a validated profile."""
    prefix = profile.prefix
    values: dict[str, Any] = {}
    for axes in (profile.sample_axes, profile.detector_axes):
        for axis_number, axis in enumerate(axes, start=1):
            base = f"{prefix}{axis.record_name}"
            values[f"{base}:AxisNumber"] = float(axis_number)
            values[f"{base}:DirectionAxis"] = axis.direction
            values[f"{base}:SpecMotorName"] = axis.published_spec_motor_name

    values[f"{prefix}spec:Energy:Units"] = profile.energy_units
    values[f"{prefix}spec:UB_matrix:Value"] = list(profile.ub_matrix)
    for group, vector in (
        ("PrimaryBeamDirection", profile.primary_beam_direction),
        ("InplaneReferenceDirection", profile.inplane_reference_direction),
        ("SampleSurfaceNormalDirection", profile.sample_surface_normal_direction),
    ):
        for index, value in enumerate(vector, start=1):
            values[f"{prefix}{group}:AxisNumber{index}"] = float(value)

    detector = profile.detector_setup
    values[f"{prefix}DetectorSetup:PixelDirection1"] = detector["PIXEL_DIRECTION_1"]
    values[f"{prefix}DetectorSetup:PixelDirection2"] = detector["PIXEL_DIRECTION_2"]
    values[f"{prefix}DetectorSetup:CenterChannelPixel"] = list(
        detector["CENTER_CHANNEL_PIXEL"]
    )
    values[f"{prefix}DetectorSetup:Size"] = list(detector["SIZE"])
    values[f"{prefix}DetectorSetup:Distance"] = float(detector["DISTANCE"])
    values[f"{prefix}DetectorSetup:Units"] = detector["UNITS"]
    values[f"{prefix}ScanOn:Value"] = 0
    values[f"{prefix}FilePath:Value"] = ""
    values[f"{prefix}FileName:Value"] = ""
    return values


def all_pv_names(profile: RSMParameterProfile) -> list[tuple[str, str]]:
    """Return the dynamic display list for the IOC's records."""
    prefix = profile.prefix
    records: list[tuple[str, str]] = []
    for axis in profile.axes:
        base = f"{prefix}{axis.record_name}"
        records.extend(
            (
                (f"{base}:Position", f"{axis.label} position"),
                (f"{base}:AxisNumber", f"{axis.label} axis number"),
                (f"{base}:DirectionAxis", f"{axis.label} direction"),
                (f"{base}:SpecMotorName", f"{axis.label} label"),
            )
        )
    records.extend(
        (
            (f"{prefix}spec:Energy:Value", "Energy value"),
            (f"{prefix}spec:Energy:Units", "Energy units"),
            (f"{prefix}spec:UB_matrix:Value", "UB matrix"),
        )
    )
    for group, description in (
        ("PrimaryBeamDirection", "Primary beam"),
        ("InplaneReferenceDirection", "In-plane reference"),
        ("SampleSurfaceNormalDirection", "Sample surface normal"),
    ):
        for index in range(1, 4):
            records.append(
                (f"{prefix}{group}:AxisNumber{index}", f"{description} {index}")
            )
    records.extend(
        (
            (f"{prefix}DetectorSetup:PixelDirection1", "Detector pixel direction 1"),
            (f"{prefix}DetectorSetup:PixelDirection2", "Detector pixel direction 2"),
            (f"{prefix}DetectorSetup:CenterChannelPixel", "Detector center"),
            (f"{prefix}DetectorSetup:Size", "Detector size"),
            (f"{prefix}DetectorSetup:Distance", "Detector distance"),
            (f"{prefix}DetectorSetup:Units", "Detector units"),
            (f"{prefix}ScanOn:Value", "Scan on flag"),
            (f"{prefix}FilePath:Value", "File path"),
            (f"{prefix}FileName:Value", "File name"),
        )
    )
    return records


class _SourceMonitorCache:
    """Cached CA monitors with expectation-specific transition logging."""

    def __init__(self, pv_factory, emit) -> None:
        self._pv_factory = pv_factory
        self._emit = emit
        self._monitors: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._unavailable: set[tuple[str, str]] = set()

    def _read(self, source: str) -> Any:
        with self._lock:
            monitor = self._monitors.get(source)
            if monitor is None:
                monitor = self._pv_factory(source, auto_monitor=True)
                self._monitors[source] = monitor
            if not monitor.connected:
                raise ValueError("no connection or value")
            value = monitor.value
        if value is None:
            raise ValueError("no connection or value")
        return value

    def _failed(
        self,
        source: str,
        expectation: str,
        error: Exception,
        *,
        fallback: bool,
    ) -> None:
        key = (source, expectation)
        if key in self._unavailable:
            return
        self._unavailable.add(key)
        suffix = "; using static fallback" if fallback else ""
        self._emit(f"[IOC] source unavailable: {source!r} ({error}){suffix}")

    def _recovered(self, source: str, expectation: str) -> None:
        key = (source, expectation)
        if key not in self._unavailable:
            return
        self._unavailable.remove(key)
        self._emit(f"[IOC] source recovered: {source!r}")

    def scalar(self, source: str) -> float:
        """Resolve an axis/energy scalar, returning NaN while unavailable."""
        source = source.strip()
        try:
            result = float(source)
        except ValueError:
            try:
                raw = self._read(source)
                array = np.asarray(raw, dtype=float)
                if array.size != 1:
                    raise ValueError(f"expected one scalar, got {raw!r}")
                result = float(array.reshape(-1)[0])
                if not np.isfinite(result):
                    raise ValueError("non-finite value")
            except Exception as exc:
                self._failed(source, "scalar", exc, fallback=False)
                return float("nan")
            self._recovered(source, "scalar")
        if not np.isfinite(result):
            return float("nan")
        return result

    def vector_or_fallback(
        self,
        source: str,
        length: int,
        fallback: tuple[float, ...],
        *,
        full_rank_3x3: bool = False,
    ) -> tuple[float, ...]:
        """Resolve one flat finite waveform, retaining a validated fallback."""
        source = source.strip()
        if not source:
            return fallback
        expectation = f"vector-{length}{'-full-rank' if full_rank_3x3 else ''}"
        try:
            raw = self._read(source)
            array = np.asarray(raw, dtype=float)
            if array.ndim != 1 or array.size != length or not np.all(np.isfinite(array)):
                raise ValueError(f"expected {length} finite row-major values, got {raw!r}")
            if full_rank_3x3 and np.linalg.matrix_rank(array.reshape(3, 3)) < 3:
                raise ValueError("matrix is not full rank")
            result = tuple(float(item) for item in array)
        except Exception as exc:
            self._failed(source, expectation, exc, fallback=True)
            return fallback
        self._recovered(source, expectation)
        return result

    def positive_scalar_or_fallback(self, source: str, fallback: float) -> float:
        """Resolve one finite positive scalar, retaining a validated fallback."""
        source = source.strip()
        if not source:
            return fallback
        expectation = "positive-scalar"
        try:
            raw = self._read(source)
            array = np.asarray(raw, dtype=float)
            if array.size != 1:
                raise ValueError(f"expected one positive scalar, got {raw!r}")
            result = float(array.reshape(-1)[0])
            if not np.isfinite(result) or result <= 0:
                raise ValueError("non-finite or non-positive value")
        except Exception as exc:
            self._failed(source, expectation, exc, fallback=True)
            return fallback
        self._recovered(source, expectation)
        return result


def _source_owned_ioc_values(
    profile: RSMParameterProfile,
    sources: _SourceMonitorCache,
) -> dict[str, Any]:
    """Values that must be refreshed because a configured source owns them.

    Source-less UB/distance records are deliberately absent: their static
    fallback is written once at startup, after which a terminal caput must stay
    in place long enough for the editor's live-adoption path to observe it.
    """
    values: dict[str, Any] = {}
    if profile.ub_matrix_source_pv:
        values[f"{profile.prefix}spec:UB_matrix:Value"] = list(
            sources.vector_or_fallback(
                profile.ub_matrix_source_pv,
                9,
                profile.ub_matrix,
                full_rank_3x3=True,
            )
        )
    if profile.detector_distance_source_pv:
        values[f"{profile.prefix}DetectorSetup:Distance"] = (
            sources.positive_scalar_or_fallback(
                profile.detector_distance_source_pv,
                profile.detector_setup["DISTANCE"],
            )
        )
    return values


def _run_ioc(raw_config: Mapping[str, Any]) -> None:
    """Run the GUI-free pvAccess IOC in its dedicated process."""
    import ctypes.util

    import pvaccess as pva
    from epics import PV as EpicsPV

    profile = profile_from_raw(raw_config)
    current_values: dict[str, Any] = {}

    def ioc_put(ca_ioc, record: str, value: Any) -> None:
        if isinstance(value, (list, tuple, np.ndarray)):
            converted = [
                float(item) if isinstance(item, (int, float, np.floating)) else str(item)
                for item in value
            ]
        elif isinstance(value, bool):
            converted = int(value)
        else:
            converted = value
        current_values[record] = converted
        try:
            ca_ioc.putField(record, converted)
        except Exception as exc:
            print(f"IOC put [{record}]: {exc}", flush=True)

    if not os.environ.get("EPICS_DB_INCLUDE_PATH"):
        library = ctypes.util.find_library("pvData")
        if library:
            library = os.path.realpath(library)
            dbd = os.path.realpath(os.path.join(os.path.dirname(library), "../../dbd"))
        elif os.environ.get("EPICS_BASE"):
            dbd = os.path.join(os.environ["EPICS_BASE"], "dbd")
        else:
            dbd = os.path.join(os.path.dirname(pva.__file__), "dbd")
            if not os.path.isdir(dbd):
                raise RuntimeError(
                    "Cannot find dbd directory. Set EPICS_DB_INCLUDE_PATH."
                )
        os.environ["EPICS_DB_INCLUDE_PATH"] = dbd

    base_dbd = os.path.join(os.environ["EPICS_DB_INCLUDE_PATH"], "base.dbd")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".db") as stream:
        stream.write(build_ioc_database(profile))
        database_path = stream.name
    try:
        ca_ioc = pva.CaIoc()
        ca_ioc.loadDatabase(base_dbd, "", "")
        ca_ioc.registerRecordDeviceDriver()
        ca_ioc.loadRecords(database_path, "")
        ca_ioc.start()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(database_path)

    static_values = static_ioc_values(profile)
    for record, value in static_values.items():
        ioc_put(ca_ioc, record, value)

    stop_event = threading.Event()
    sources = _SourceMonitorCache(
        EpicsPV,
        lambda message: print(message, flush=True),
    )
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    print(
        f"IOC ready (prefix={profile.prefix!r}, axes={len(profile.axes)})",
        flush=True,
    )
    print(json.dumps({"type": "values", "data": dict(current_values)}), flush=True)
    loop_count = 0
    while not stop_event.is_set():
        started = time.monotonic()
        try:
            for axis in profile.axes:
                ioc_put(
                    ca_ioc,
                    f"{profile.prefix}{axis.record_name}:Position",
                    sources.scalar(axis.source_pv),
                )
            ioc_put(
                ca_ioc,
                f"{profile.prefix}spec:Energy:Value",
                sources.scalar(profile.energy_source_pv),
            )
            for record, value in _source_owned_ioc_values(profile, sources).items():
                ioc_put(ca_ioc, record, value)
            loop_count += 1
            if loop_count % app_settings.RSM_IOC_SNAPSHOT_EVERY == 0:
                for record in static_values:
                    try:
                        current_values[record] = ca_ioc.getField(record)
                    except Exception as exc:
                        print(f"IOC read [{record}]: {exc}", flush=True)
                print(
                    json.dumps({"type": "values", "data": dict(current_values)}),
                    flush=True,
                )
        except Exception as exc:
            print(f"IOC update error: {exc}", flush=True)
        stop_event.wait(
            max(
                0.0,
                app_settings.RSM_IOC_POLL_INTERVAL_SECONDS
                - (time.monotonic() - started),
            )
        )
    print("IOC subprocess exiting.", flush=True)


_gui_classes_cache: tuple[type, type, type] | None = None


def _build_gui_classes() -> tuple[type, type, type]:
    """Define and return (PollWorker, AxisTable, SimulatorWindow).

    PyQt5 is deliberately NOT imported at module scope: this file is also
    executed as the pvaccess IOC subprocess (`--ioc-mode`), and mixing PyQt5
    into that process core-dumps (see the two-process note in this module's
    docstring, also referenced from log_manager.py/hdf5_writer.py/etc.). This
    factory is the one place the GUI classes get defined, called by `_run_gui`
    for the real GUI process and by tests that need to instantiate them
    directly -- `--ioc-mode` never calls it, so PyQt5 is never imported there.
    Classes are defined once and cached; repeated calls return the same
    objects.
    """
    global _gui_classes_cache
    if _gui_classes_cache is not None:
        return _gui_classes_cache

    from PyQt5.QtCore import QSettings, Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QComboBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    from dashpva.viewer.core.base_window import BaseWindow

    def format_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, list):
            return "[" + ", ".join(
                f"{item:.6g}" if isinstance(item, float) else str(item)
                for item in value
            ) + "]"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    class PollWorker(QThread):
        results_ready = pyqtSignal(list)

        def __init__(
            self,
            records: list[tuple[str, str]],
            pv_values: dict[str, Any],
            pv_lock: threading.Lock,
        ):
            super().__init__()
            self.records = records
            self.pv_values = pv_values
            self.pv_lock = pv_lock
            self.running = True

        def run(self):
            previous: list[str] | None = None
            while self.running:
                with self.pv_lock:
                    snapshot = dict(self.pv_values)
                values = [
                    format_value(snapshot.get(name)) for name, _ in self.records
                ]
                if values != previous:
                    previous = values
                    self.results_ready.emit(values)
                self.msleep(50)

        def stop(self):
            self.running = False

    class AxisTable(QWidget):
        headers = (
            "Label",
            "SPEC motor name",
            "Record name",
            "Source PV / static",
            "Direction",
            "Units",
        )
        axis_keys = (
            "LABEL",
            "SPEC_MOTOR_NAME",
            "RECORD_NAME",
            "SOURCE_PV",
            "DIRECTION",
            "ANGLE_UNITS",
        )

        def __init__(self, role: str):
            super().__init__()
            self.role = role
            self._next_new_origin = -1
            layout = QVBoxLayout(self)
            self.table = QTableWidget(0, len(self.headers))
            self.table.setHorizontalHeaderLabels(self.headers)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
            self.table.setSelectionBehavior(QTableWidget.SelectRows)
            layout.addWidget(self.table)
            controls = QHBoxLayout()
            for label, callback in (
                ("Add", self.add_axis),
                ("Remove", self.remove_axis),
                ("Move up", lambda: self.move_axis(-1)),
                ("Move down", lambda: self.move_axis(1)),
            ):
                button = QPushButton(label)
                button.clicked.connect(callback)
                controls.addWidget(button)
            controls.addStretch()
            layout.addLayout(controls)

        def load_axes(self, axes) -> None:
            self.table.setRowCount(0)
            self._next_new_origin = -1
            for origin, axis in enumerate(axes):
                self._append(axis.as_mapping(), origin)

        def _append(
            self,
            values: Mapping[str, Any],
            origin: int | None = None,
            row: int | None = None,
        ) -> None:
            if origin is None:
                origin = self._next_new_origin
                self._next_new_origin -= 1
            row = self.table.rowCount() if row is None else row
            self.table.insertRow(row)
            for column, key in enumerate(self.axis_keys):
                item = QTableWidgetItem(str(values.get(key, "")))
                item.setData(Qt.UserRole, origin)
                self.table.setItem(row, column, item)
            self.table.setCurrentCell(row, 0)

        def add_axis(self) -> None:
            ordinal = self.table.rowCount() + 1
            stem = f"{self.role.title()}Axis{ordinal}"
            self._append(
                {
                    "LABEL": stem,
                    "SPEC_MOTOR_NAME": "",
                    "RECORD_NAME": stem,
                    "SOURCE_PV": "0",
                    "DIRECTION": "z-",
                    "ANGLE_UNITS": "deg",
                }
            )

        def remove_axis(self) -> None:
            row = self.table.currentRow()
            if row >= 0:
                self.table.removeRow(row)

        def move_axis(self, offset: int) -> None:
            source_row = self.table.currentRow()
            target_row = source_row + offset
            if source_row < 0 or not 0 <= target_row < self.table.rowCount():
                return
            row_values = [
                self.table.takeItem(source_row, column)
                for column in range(self.table.columnCount())
            ]
            self.table.removeRow(source_row)
            self.table.insertRow(target_row)
            for column, item in enumerate(row_values):
                self.table.setItem(target_row, column, item)
            self.table.setCurrentCell(target_row, 0)

        def values(self) -> list[dict[str, str]]:
            axes = []
            for row in range(self.table.rowCount()):
                axes.append(
                    {
                        key: (self.table.item(row, column).text().strip()
                              if self.table.item(row, column) is not None else "")
                        for column, key in enumerate(self.axis_keys)
                    }
                )
            return axes

        def origins(self) -> tuple[int | None, ...]:
            return tuple(
                self.table.item(row, 0).data(Qt.UserRole)
                if self.table.item(row, 0) is not None
                else None
                for row in range(self.table.rowCount())
            )

        def row_for_origin(self, origin: int) -> int | None:
            for row, candidate in enumerate(self.origins()):
                if candidate == origin:
                    return row
            return None

        def restore_axis(self, origin: int, values: Mapping[str, Any]) -> None:
            rows = list(zip(self.origins(), self.values()))
            restored = _restore_axis_row(rows, origin, values)
            if len(restored) == len(rows):
                return
            self.table.setRowCount(0)
            for item_origin, item_values in restored:
                self._append(item_values, item_origin)

        def remove_origin(self, origin: int) -> None:
            row = self.row_for_origin(origin)
            if row is not None:
                self.table.removeRow(row)

        def replace_axis(self, origin: int, values: Mapping[str, Any]) -> None:
            row = self.row_for_origin(origin)
            if row is None:
                self.restore_axis(origin, values)
                return
            for column, key in enumerate(self.axis_keys):
                item = QTableWidgetItem(str(values.get(key, "")))
                item.setData(Qt.UserRole, origin)
                self.table.setItem(row, column, item)

        def reorder_loaded(self, order: list[int]) -> None:
            rows = list(zip(self.origins(), self.values()))
            new_rows = _reorder_loaded_axis_rows(rows, order)
            self.table.setRowCount(0)
            for origin, values in new_rows:
                self._append(values, origin)

    class CollapsibleSection(QWidget):
        """A titled row that expands into a bordered content area.

        The whole header is the click target and the arrow is drawn by the
        style, so it reads as a disclosure control rather than a checkbox, and
        a collapsed section leaves a single line instead of an empty frame.
        ``isChecked``/``setChecked`` are kept so
        :meth:`BaseWindow.save_checkable_state` still persists it.

        Example:
            section = CollapsibleSection("Detector setup", checked=True)
            section.form.addRow("Distance", QLineEdit())
            layout.addWidget(section)
        """

        def __init__(self, title: str, checked: bool = False, parent=None,
                     layout_cls=None, area_name: str = ""):
            super().__init__(parent)
            self.header = QToolButton()
            self.header.setObjectName("collapsibleHeader")
            self.header.setText(title)
            self.header.setCheckable(True)
            self.header.setChecked(checked)
            self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            self.header.setAutoRaise(True)
            self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            self.area = QFrame()
            self.area.setObjectName(area_name or "collapsibleArea")
            self.area.setVisible(checked)
            # `form` is the content layout whatever its type -- detector setup
            # supplies a QGridLayout, the rest take the default QFormLayout.
            self.form = (layout_cls or QFormLayout)(self.area)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.header)
            layout.addWidget(self.area)

            self.header.toggled.connect(self._on_toggled)

        def _on_toggled(self, checked: bool) -> None:
            self.header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            self.area.setVisible(checked)

        def isChecked(self) -> bool:
            return self.header.isChecked()

        def setChecked(self, checked: bool) -> None:
            self.header.setChecked(checked)


    class SimulatorWindow(BaseWindow):
        def __init__(
            self,
            session: RSMParameterEditSession,
            profile: RSMParameterProfile,
            restart_ioc,
            pv_values: dict[str, Any],
            pv_lock: threading.Lock,
        ):
            super().__init__(
                viewer_name="RSM Parameter IOC", visible_actions=["Documentation"]
            )
            self.session = session
            self.restart_ioc = restart_ioc
            self.pv_values = pv_values
            self.pv_lock = pv_lock
            # Remember which profile was active when this editor opened, so a
            # switch made elsewhere can be reported as such instead of as an
            # activation failure -- see _activate_snapshot/_mark_out_of_sync.
            self._startup_locator = _active_profile_identity()
            self.profile = profile
            self.worker: PollWorker | None = None
            self.setWindowTitle("RSM Parameter IOC")
            self._build_ui()
            self._load_profile(profile)
            self._reset_record_monitor(profile)
            settings = QSettings("DashPVA", "RSMParameterIOC")
            geometry = settings.value("window_geom")
            if geometry:
                self.restoreGeometry(geometry)
            for section, key, default in (
                (self.calibration_group, "static_geometry_expanded", True),
                (self.detector_setup_group, "detector_setup_expanded", True),
                (self.advanced_group, "advanced_expanded", False),
            ):
                self.restore_checkable_state(settings, section, key, default)

        def _build_ui(self) -> None:
            central = QWidget()
            central_layout = QVBoxLayout(central)
            central_layout.setContentsMargins(0, 0, 0, 0)
            self.setCentralWidget(central)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            central_layout.addWidget(scroll)
            content = QWidget()
            root = QVBoxLayout(content)
            scroll.setWidget(content)

            self.profile_notice = QLabel()
            self.profile_notice.setWordWrap(True)
            self.profile_notice.setProperty("messageLevel", "warning")
            root.addWidget(self.profile_notice)

            general = QGroupBox("Profile-backed IOC settings")
            form = QFormLayout(general)
            self.prefix_edit = QLineEdit()
            self.prefix_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow("IOC prefix", self.prefix_edit)
            self.energy_edit = QLineEdit()
            self.energy_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow("Energy source PV / static", self.energy_edit)
            self.energy_units = QComboBox()
            self.energy_units.setEditable(True)
            self.energy_units.addItem("keV")
            self.energy_units.lineEdit().setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.energy_units.setToolTip("RSM energy is configured in keV.")
            form.addRow("Energy units", self.energy_units)
            root.addWidget(general)

            sample_group = QGroupBox("Ordered sample circles")
            sample_layout = QVBoxLayout(sample_group)
            self.sample_table = AxisTable("sample")
            sample_layout.addWidget(self.sample_table)
            root.addWidget(sample_group)

            detector_group = QGroupBox("Ordered detector circles")
            detector_layout = QVBoxLayout(detector_group)
            self.detector_table = AxisTable("detector")
            detector_layout.addWidget(self.detector_table)
            root.addWidget(detector_group)

            self.calibration_group = self._build_calibration_group()
            root.addWidget(self.calibration_group)
            self.detector_setup_group = self._build_detector_setup_group()
            root.addWidget(self.detector_setup_group)
            self.advanced_group = self._build_advanced_group()
            root.addWidget(self.advanced_group)
            root.addWidget(self._build_records_group())

            controls = QHBoxLayout()
            reload_button = QPushButton("Reload profile")
            reload_button.clicked.connect(self._reload)
            apply_button = QPushButton("Apply && Save")
            apply_button.clicked.connect(self._apply)
            self.retry_button = QPushButton("Retry IOC sync")
            self.retry_button.clicked.connect(self._retry_sync)
            self.retry_button.setVisible(False)
            controls.addWidget(reload_button)
            controls.addStretch()
            controls.addWidget(self.retry_button)
            controls.addWidget(apply_button)
            # Outside the scroll area so it stays visible without scrolling.
            central_layout.addLayout(controls)

            self.resize(1000, 1000)

        def _build_calibration_group(self) -> CollapsibleSection:
            section = CollapsibleSection("Static geometry — JSON", checked=True)
            form = section.form
            self.ub_matrix_edit = QLineEdit()
            self.ub_matrix_edit.setObjectName("lineEditUbMatrix")
            self.ub_matrix_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.ub_matrix_edit.setToolTip(_UB_INPUT_TOOLTIP)
            self.ub_matrix_edit.textChanged.connect(self._on_ub_matrix_text_changed)
            form.addRow("UB Matrix (PV or value)", self.ub_matrix_edit)
            self.calibration_values: dict[str, QLineEdit] = {}
            for key in (
                "PRIMARY_BEAM_DIRECTION",
                "INPLANE_REFERENCE_DIRECTION",
                "SAMPLE_SURFACE_NORMAL_DIRECTION",
            ):
                edit = QLineEdit()
                edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.calibration_values[key] = edit
                form.addRow(key.replace("_", " ").title(), edit)
            return section

        def _build_detector_setup_group(self) -> CollapsibleSection:
            """One field per detector value instead of a single JSON blob.

            Generated from DETECTOR_SETUP_FIELDS in a compact two-column grid
            so an unrecognized/new field never silently disappears from the
            form; blank means absent, which is how the optional fields (ROI,
            BINNING, TILT, ...) are meant to be expressed. DISTANCE gets a
            dedicated PV-or-value row instead of a plain field, in the
            position it would otherwise occupy. Center/size/pixel size/shape/
            binning are all in detector directions 1/2, not x/y -- only
            FRAME_AXIS_ORDER maps them onto array rows/columns.
            """
            section = CollapsibleSection(
                "Detector setup", checked=True,
                layout_cls=QGridLayout, area_name="groupBoxDetectorSetup",
            )
            grid = section.form
            hint = QLabel("Leave optional fields blank to omit them from the profile.")
            hint.setProperty("messageLevel", "info")
            grid.addWidget(hint, 0, 0, 1, 4)
            self.detector_setup_values: dict[str, QLineEdit] = {}
            self._detector_setup_extras: dict[str, Any] = {}
            for index, key in enumerate(DETECTOR_SETUP_FIELDS):
                field_row, column_pair = divmod(index, 2)
                row = field_row + 1
                label_col, edit_col = column_pair * 2, column_pair * 2 + 1
                if key == "DISTANCE":
                    label = QLabel(f"{_DETECTOR_FIELD_LABELS[key]}:")
                    self.detector_distance_edit = QLineEdit()
                    self.detector_distance_edit.setObjectName("lineEditDetectorDistance")
                    self.detector_distance_edit.textChanged.connect(
                        self._on_distance_text_changed
                    )
                    edit = self.detector_distance_edit
                else:
                    label = QLabel(f"{_DETECTOR_FIELD_LABELS[key]}:")
                    edit = QLineEdit()
                    edit.setPlaceholderText(_DETECTOR_FIELD_HINTS.get(key, ""))
                    self.detector_setup_values[key] = edit
                edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tooltip = _DETECTOR_FIELD_DESCRIPTIONS.get(key, "")
                label.setToolTip(tooltip)
                edit.setToolTip(tooltip)
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(label, row, label_col)
                grid.addWidget(edit, row, edit_col)
            for key in ("DISTANCE_UNITS", "UNITS"):
                self.detector_setup_values[key].textChanged.connect(
                    lambda _text: self._refresh_distance_tooltip()
                )
            return section

        def _build_advanced_group(self) -> CollapsibleSection:
            section = CollapsibleSection("Advanced")
            form = section.form
            self.sample_orientation = QComboBox()
            self.sample_orientation.setEditable(True)
            self.sample_orientation.addItems(
                ("det", "sam", "x+", "x-", "y+", "y-", "z+", "z-")
            )
            self.sample_orientation.lineEdit().setAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )
            form.addRow("Sample orientation", self.sample_orientation)
            return section

        def _calibration_edits(self) -> dict[str, Any]:
            """Parse the calibration fields, raising on the first bad entry.

            Validation proper happens in normalize_parameters; this only turns
            the text back into JSON so a typo is reported as a typo instead of
            surfacing later as an unrelated geometry error.
            """
            parsed: dict[str, Any] = {}
            for key, edit in self.calibration_values.items():
                text = edit.text().strip()
                if not text:
                    continue
                try:
                    parsed[key] = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{key} is not valid JSON: {exc}") from exc
            return parsed

        def _load_detector_setup(self, setup: Mapping[str, Any]) -> None:
            # DISTANCE_SOURCE_PV is a known, GUI-managed key (handled by the
            # dedicated distance widget below), not an arbitrary extension --
            # it must NOT be captured here, or switching the field back to a
            # literal would leak the stale PV name back in on save via
            # _detector_setup_edits's extras merge (see _parse_distance_or_pv
            # call below, which is the only place this key gets rewritten).
            shown = set(DETECTOR_SETUP_FIELDS) | {"DISTANCE_SOURCE_PV"}
            # Anything else this build doesn't know about is carried through
            # untouched rather than silently dropped on the next save.
            self._detector_setup_extras = {
                key: value for key, value in setup.items() if key not in shown
            }
            for key, edit in self.detector_setup_values.items():
                edit.setText(_format_detector_value(setup.get(key)))
            distance = setup.get("DISTANCE", 0.0)
            distance_source = setup.get("DISTANCE_SOURCE_PV", "")
            # Tracks the last known-good literal typed into this field, live,
            # independent of self.profile (which only updates on a successful
            # Apply/Reload) -- see _on_distance_text_changed.
            self._distance_literal_fallback = distance
            self.detector_distance_edit.setText(
                _format_distance_or_pv(distance, distance_source)
            )
            self._refresh_distance_tooltip()
            self._set_source_mode(self.detector_distance_edit, bool(distance_source))

        def _refresh_distance_tooltip(self) -> None:
            fallback = getattr(
                self,
                "_distance_literal_fallback",
                self.profile.detector_setup.get("DISTANCE", 0.0),
            )
            units = (
                self.detector_setup_values["DISTANCE_UNITS"].text().strip()
                or self.detector_setup_values["UNITS"].text().strip()
                or "the configured distance units"
            )
            tooltip = (
                "Enter one finite positive value, or one CA PV that publishes a "
                f"single finite positive scalar in {units}."
            )
            try:
                _distance, source = _parse_distance_or_pv(
                    self.detector_distance_edit.text(),
                    fallback,
                )
            except ValueError:
                source = ""
            if source:
                tooltip += (
                    "\nFallback if the source PV is unavailable: "
                    f"{fallback} {units}."
                )
            self.detector_distance_edit.setToolTip(tooltip)

        def _on_distance_text_changed(self, text: str) -> None:
            """Keep _distance_literal_fallback current as the user types.

            Without this, switching Distance from a freshly-typed literal
            straight to a PV name (no Apply in between) would silently revert
            the fallback to whatever was last saved, discarding what the user
            just typed -- self.profile.detector_setup only updates on a
            successful Apply/Reload/out-of-sync recovery, never on a keystroke.
            """
            try:
                self._distance_literal_fallback = validate_distance(text.strip())
            except (TypeError, ValueError):
                pass  # blank, a PV name, or not yet a valid literal -- keep the last known-good one
            try:
                _distance, source = _parse_distance_or_pv(
                    text, self._distance_literal_fallback
                )
            except ValueError:
                source = ""
            self._set_source_mode(self.detector_distance_edit, bool(source))
            self._refresh_distance_tooltip()

        def _detector_setup_edits(self) -> dict[str, Any]:
            setup: dict[str, Any] = copy.deepcopy(self._detector_setup_extras)
            for key, edit in self.detector_setup_values.items():
                text = edit.text().strip()
                if not text:
                    continue
                setup[key] = _parse_detector_value(text)
            distance, distance_source = _parse_distance_or_pv(
                self.detector_distance_edit.text(),
                self._distance_literal_fallback,
            )
            setup["DISTANCE"] = distance
            # Unconditional pop-then-set: entering a literal must clear any
            # previously configured source, not just leave a stale one in
            # place (defense in depth alongside excluding this key from
            # _detector_setup_extras in _load_detector_setup).
            setup.pop("DISTANCE_SOURCE_PV", None)
            if distance_source:
                setup["DISTANCE_SOURCE_PV"] = distance_source
            return setup

        def _set_source_mode(self, widget, active: bool) -> None:
            """Toggle the 'sourceMode' QSS property so a PV-backed field looks
            visually distinct from a literal value."""
            widget.setProperty("sourceMode", "true" if active else "false")
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        def _build_records_group(self) -> QGroupBox:
            group = QGroupBox("Live IOC records")
            layout = QVBoxLayout(group)
            self.records_table = QTableWidget(0, 2)
            self.records_table.setHorizontalHeaderLabels(("PV name", "Value"))
            self.records_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.records_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.records_table.setMinimumHeight(320)
            layout.addWidget(self.records_table)
            return group

        def _load_profile(self, profile: RSMParameterProfile) -> None:
            self.profile = profile
            raw_parameters = self.session.raw.get("IOC_RSM_PARAMETER")
            self._raw_baseline = copy.deepcopy(
                raw_parameters if isinstance(raw_parameters, Mapping) else {}
            )
            self._normalized_baseline = profile.parameter_mapping()
            self.prefix_edit.setText(profile.prefix)
            self.energy_edit.setText(profile.energy_source_pv)
            self.energy_units.setCurrentText(profile.energy_units)
            self.sample_orientation.setCurrentText(profile.sample_orientation)
            self.sample_table.load_axes(profile.sample_axes)
            self.detector_table.load_axes(profile.detector_axes)
            mapping = profile.parameter_mapping()
            for key, edit in self.calibration_values.items():
                edit.setText(json.dumps(mapping[key], sort_keys=True))
            # Tracks the last known-good literal typed into this field, live,
            # independent of self.profile (which only updates on a successful
            # Apply/Reload) -- see _on_ub_matrix_text_changed. Set before
            # setText below so the attribute always exists even if that
            # triggers the signal against a PV-name value.
            self._ub_matrix_literal_fallback = profile.ub_matrix
            self.ub_matrix_edit.setText(
                _format_ub_or_pv(profile.ub_matrix, profile.ub_matrix_source_pv)
            )
            self._refresh_ub_tooltip()
            self._set_source_mode(self.ub_matrix_edit, bool(profile.ub_matrix_source_pv))
            self._load_detector_setup(mapping.get("DETECTOR_SETUP") or {})
            if app_settings.CONFIG_ERROR:
                self.profile_notice.setText(
                    "The active profile failed to resolve and is being shown/edited as a "
                    f"fallback so it can be repaired here: {app_settings.CONFIG_ERROR}"
                )
                self.profile_notice.setProperty("messageLevel", "error")
            elif self.session.has_canonical_parameters and not requires_adoption_confirmation(self.session.raw):
                self.profile_notice.setText("Edits are staged until Apply & Save succeeds.")
                self.profile_notice.setProperty("messageLevel", "info")
            else:
                self.profile_notice.setText(
                    "This profile is not fully canonical. First Apply & Save will show "
                    "the effective HKL and static-geometry changes for confirmation."
                )
                self.profile_notice.setProperty("messageLevel", "warning")
            self.profile_notice.style().unpolish(self.profile_notice)
            self.profile_notice.style().polish(self.profile_notice)

        def _on_ub_matrix_text_changed(self, text: str) -> None:
            """Keep _ub_matrix_literal_fallback current as the user types.

            Without this, switching UB Matrix from a freshly-typed literal
            straight to a PV name (no Apply in between) would silently revert
            the fallback to whatever was last saved, discarding what the user
            just typed -- self.profile.ub_matrix only updates on a successful
            Apply/Reload/out-of-sync recovery, never on a keystroke.
            """
            try:
                self._ub_matrix_literal_fallback = validate_ub_matrix(
                    json.loads(text.strip()), "UB Matrix"
                )
            except (json.JSONDecodeError, ValueError):
                pass  # blank, a PV name, or not yet a valid literal -- keep the last known-good one
            try:
                _matrix, source = _parse_ub_or_pv(
                    text, self._ub_matrix_literal_fallback
                )
            except ValueError:
                source = ""
            self._set_source_mode(self.ub_matrix_edit, bool(source))
            self._refresh_ub_tooltip()

        def _refresh_ub_tooltip(self) -> None:
            tooltip = _UB_INPUT_TOOLTIP
            try:
                _matrix, source = _parse_ub_or_pv(
                    self.ub_matrix_edit.text(), self._ub_matrix_literal_fallback
                )
            except ValueError:
                source = ""
            if source:
                tooltip += (
                    "\nFallback if the source PV is unavailable: "
                    f"{json.dumps(list(self._ub_matrix_literal_fallback))}."
                )
            self.ub_matrix_edit.setToolTip(tooltip)

        def _parameters(self) -> dict[str, Any]:
            current = self.profile.parameter_mapping()
            current.update(
                {
                    "SAMPLE_AXES": self.sample_table.values(),
                    "DETECTOR_AXES": self.detector_table.values(),
                    "ENERGY_SOURCE_PV": self.energy_edit.text().strip(),
                    "ENERGY_UNITS": self.energy_units.currentText().strip(),
                    "SAMPLE_ORIENTATION": self.sample_orientation.currentText().strip(),
                }
            )
            current.update(self._calibration_edits())
            ub_matrix, ub_matrix_source_pv = _parse_ub_or_pv(
                self.ub_matrix_edit.text(), self._ub_matrix_literal_fallback
            )
            current["UB_MATRIX"] = list(ub_matrix)
            current["UB_MATRIX_SOURCE_PV"] = ub_matrix_source_pv
            current["DETECTOR_SETUP"] = self._detector_setup_edits()
            return current

        def _axis_origins(self) -> dict[str, tuple[int | None, ...]]:
            return {
                "SAMPLE_AXES": self.sample_table.origins(),
                "DETECTOR_AXES": self.detector_table.origins(),
            }

        def _pending_change(self) -> dict[str, Any]:
            """Build the exact candidate used by Apply, close, and review."""
            form_parameters = self._parameters()
            parameters = copy.deepcopy(form_parameters)
            validate_parameter_profile(self.prefix_edit.text(), parameters)
            with self.pv_lock:
                live = dict(self.pv_values)
            origins = self._axis_origins()
            adopted, conflicts = merge_live_records(
                parameters,
                self.profile,
                live,
                self._raw_baseline,
                self._normalized_baseline,
                axis_origins=origins,
            )
            candidate = validate_parameter_profile(
                self.prefix_edit.text(), parameters
            )
            replacement = update_raw_profile(
                self.session.raw,
                self.prefix_edit.text(),
                parameters,
                axis_origins=origins,
            )
            return {
                "form_parameters": form_parameters,
                "parameters": parameters,
                "candidate": candidate,
                "origins": origins,
                "adopted": adopted,
                "conflicts": conflicts,
                "replacement": replacement,
            }

        def _confirm_adoption(self, parameters: Mapping[str, Any]) -> bool:
            if not requires_adoption_confirmation(self.session.raw):
                return True
            try:
                details = adoption_diff(self.session.raw, self.prefix_edit.text(), parameters)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Confirm canonical RSM adoption")
            box.setText(
                "Applying will make IOC_RSM_PARAMETER authoritative for HKL channels "
                "and static geometry. Verify the diff before continuing."
            )
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            return box.exec_() == QMessageBox.Save

        def _confirm_live_adoption(self, adopted: list[str]) -> bool:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Records changed on the IOC")
            box.setText(
                "These IOC records changed since this profile was loaded. Save "
                "to adopt the live values, or Cancel to leave the profile untouched."
            )
            box.setDetailedText("\n".join(adopted))
            box.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            return box.exec_() == QMessageBox.Save

        def _apply(self) -> bool:
            try:
                pending = self._pending_change()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            if pending["conflicts"]:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("Conflicting IOC changes")
                box.setText(
                    "The same settings changed both in this editor and on the IOC. "
                    "Nothing was saved. Reconcile the values and try again."
                )
                box.setDetailedText("\n".join(pending["conflicts"]))
                box.exec_()
                return False
            if pending["adopted"] and not self._confirm_live_adoption(
                pending["adopted"]
            ):
                return False
            parameters = pending["parameters"]
            if not self._confirm_adoption(parameters):
                return False
            if self.sample_orientation.currentText().strip().lower() == "sam":
                answer = QMessageBox.warning(
                    self,
                    "Check sample orientation",
                    "SAMPLE_ORIENTATION='sam' is physically correct only when the "
                    "innermost sample circle is the azimuth motor. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return False
            try:
                result, saved_snapshot = apply_and_activate(
                    self.session,
                    self.prefix_edit.text(),
                    parameters,
                    self._activate_snapshot,
                    axis_origins=pending["origins"],
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid RSM configuration", str(exc))
                return False
            except SnapshotActivationError as exc:
                saved_profile = profile_from_raw(exc.snapshot)
                self._load_profile(saved_profile)
                self._reset_record_monitor(saved_profile)
                self._mark_out_of_sync(exc.snapshot, exc.error)
                return False
            if result.status.value == "conflict":
                QMessageBox.warning(
                    self,
                    "Profile changed",
                    "Another editor changed this profile. Nothing was saved or applied. "
                    "Reload the profile and reapply your edits.",
                )
                return False
            if not result.saved or saved_snapshot is None:
                QMessageBox.critical(
                    self,
                    "Profile save failed",
                    result.error or "The profile could not be saved. The IOC was not restarted.",
                )
                return False
            saved_profile = profile_from_raw(saved_snapshot)
            self._load_profile(saved_profile)
            self._reset_record_monitor(saved_profile)
            self.statusBar().showMessage("Profile saved atomically; IOC restarted from that snapshot")
            return True

        def _reload(self) -> None:
            answer = QMessageBox.question(
                self,
                "Discard staged edits?",
                "Reloading discards edits that have not been applied.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            try:
                profile = self.session.load()
            except Exception as exc:
                QMessageBox.critical(self, "Profile reload failed", str(exc))
                return
            self._load_profile(profile)
            self._reset_record_monitor(profile)
            try:
                self._activate_snapshot(self.session.raw)
            except Exception as exc:
                self._mark_out_of_sync(self.session.raw, exc)
                return
            self.statusBar().showMessage("Profile and IOC reloaded from the same snapshot")

        def _activate_snapshot(self, snapshot: Mapping[str, Any]) -> None:
            app_settings.reload()
            if app_settings.RAW_CONFIG != snapshot:
                # Only resolve the current profile identity on the failure path
                # -- it can hit the database, and the common case never gets here.
                raise _classify_activation_mismatch(
                    self._startup_locator,
                    _active_profile_identity(),
                    saved=snapshot,
                    reloaded=app_settings.RAW_CONFIG,
                )
            self.restart_ioc(snapshot)
            self.pending_snapshot = None
            self.retry_button.setVisible(False)

        def _show_notice(self, text: str, level: str) -> None:
            self.profile_notice.setText(text)
            self.profile_notice.setProperty("messageLevel", level)
            self.profile_notice.style().unpolish(self.profile_notice)
            self.profile_notice.style().polish(self.profile_notice)

        def _show_problem(self, icon, title: str, text: str, details: str = "") -> None:
            """Plain-language dialog up front, technical detail behind Show Details."""
            box = QMessageBox(self)
            box.setIcon(icon)
            box.setWindowTitle(title)
            box.setText(text)
            if details:
                box.setDetailedText(details)
            box.exec_()

        def _mark_out_of_sync(self, snapshot: Mapping[str, Any], error: Exception) -> None:
            """Route to whichever of the three failures actually happened."""
            if isinstance(error, ActiveProfileChanged):
                self._mark_profile_changed(error)
                return
            profile = _profile_display_name(self._startup_locator)
            self.pending_snapshot = dict(snapshot)
            self.retry_button.setVisible(True)

            if isinstance(error, ProfileContentMismatch):
                self._show_notice(
                    f"Saved, but {error.count} setting(s) read back differently from "
                    f"{profile}. The angle IOC was not restarted and is still serving "
                    "the previous values. Use Retry IOC sync, or Reload profile to take "
                    "whatever is now stored.",
                    "error",
                )
                self._show_problem(
                    QMessageBox.Warning,
                    "Saved settings don't match what was read back",
                    f"Profile '{profile}' was saved, but reading it back gave different "
                    f"values for {error.count} setting(s). The angle IOC was not "
                    "restarted, so it is still serving the previous values.\n\n"
                    "This usually means another window wrote to the same profile at the "
                    "same time. Check the values under Show Details, then use "
                    "'Retry IOC sync' -- or 'Reload profile' to take whatever is now "
                    "stored.",
                    "\n".join(error.differences),
                )
                return

            self._show_notice(
                "The angle IOC did not restart, so its PVs are still serving the "
                f"previous values. '{profile}' is saved and was not rolled back. "
                "Use Retry IOC sync.",
                "error",
            )
            self._show_problem(
                QMessageBox.Critical,
                "The angle IOC did not restart",
                f"Profile '{profile}' is saved, but the RSM parameter IOC could not be "
                "restarted with it, so its PVs are still serving the previous values. "
                "Nothing was rolled back.\n\n"
                "Use 'Retry IOC sync'. If it keeps failing, the IOC may need to be "
                "started from a terminal.",
                str(error),
            )

        def _mark_profile_changed(self, error: ActiveProfileChanged) -> None:
            """Not a failure -- the active profile moved, so retrying this snapshot
            can never succeed. Say so plainly and hide the retry button."""
            editing = _profile_display_name(error.expected)
            now_active = _profile_display_name(error.current)
            self.pending_snapshot = None
            self.retry_button.setVisible(False)
            self._show_notice(
                f"'{now_active}' is now the active profile, but HKL Setup is editing "
                f"'{editing}'. Close and reopen HKL Setup to work on '{now_active}'.",
                "error",
            )
            self._show_problem(
                QMessageBox.Warning,
                "A different profile is now active",
                f"HKL Setup is editing '{editing}', but '{now_active}' is now the "
                "active profile -- most likely selected in another DashPVA window.\n\n"
                f"Nothing was lost: your settings for '{editing}' are saved, and the "
                f"angle IOC is still serving '{editing}' PVs, so anything reading them "
                "right now is unaffected.\n\n"
                f"To work on '{now_active}', close HKL Setup and open it again. To stay "
                f"on '{editing}', make it the active profile again in the Workflow "
                "window.",
            )

        def _retry_sync(self) -> None:
            snapshot = getattr(self, "pending_snapshot", None)
            if snapshot is None:
                return
            try:
                self._activate_snapshot(snapshot)
            except Exception as exc:
                self._mark_out_of_sync(snapshot, exc)
                return
            self._load_profile(profile_from_raw(snapshot))
            self.statusBar().showMessage("IOC synchronized with the saved profile snapshot")

        def _stop_worker(self) -> None:
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait(2000)
                self.worker = None

        def _reset_record_monitor(self, profile: RSMParameterProfile) -> None:
            self._stop_worker()
            records = all_pv_names(profile)
            self.records_table.setRowCount(len(records))
            self.value_items: list[QTableWidgetItem] = []
            for row, (name, _) in enumerate(records):
                self.records_table.setItem(row, 0, QTableWidgetItem(name))
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.records_table.setItem(row, 1, item)
                self.value_items.append(item)
            self.worker = PollWorker(records, self.pv_values, self.pv_lock)
            self.worker.results_ready.connect(self._apply_results)
            self.worker.start()

        def _apply_results(self, values: list[str]) -> None:
            for item, value in zip(self.value_items, values):
                item.setText(value)
            self.statusBar().showMessage(f"Last IOC update: {time.strftime('%H:%M:%S')}")

        def has_unsaved_changes(self) -> bool:
            return _has_pending_rsm_change(self._pending_change, self.session.raw)

        def save_changes(self) -> bool:
            return self._apply()

        def _change_target(self, key: str):
            simple = {
                "PREFIX": self.prefix_edit,
                "ENERGY_SOURCE_PV": self.energy_edit,
                "ENERGY_UNITS": self.energy_units,
                "SAMPLE_ORIENTATION": self.sample_orientation,
                "UB Matrix (PV or value)": self.ub_matrix_edit,
                "Distance (PV or value)": self.detector_distance_edit,
            }
            if key in simple:
                return simple[key]
            if key in self.calibration_values:
                return self.calibration_values[key]
            if key.startswith("DETECTOR_SETUP."):
                return self.detector_setup_values.get(key.split(".", 1)[1])
            return None

        def _axis_target(self, key: str):
            parts = key.split(".")
            if len(parts) != 3 or parts[0] not in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                return None
            table = (
                self.sample_table
                if parts[0] == "SAMPLE_AXES"
                else self.detector_table
            )
            if parts[2] not in AxisTable.axis_keys:
                return None
            if parts[1].startswith("@"):
                try:
                    row = table.row_for_origin(int(parts[1][1:]))
                except ValueError:
                    return None
                if row is None:
                    return None
            elif parts[1].isdigit():
                row = int(parts[1])
            else:
                return None
            if row >= table.table.rowCount():
                return None
            return table, row, AxisTable.axis_keys.index(parts[2])

        def is_change_editable(self, key: str) -> bool:
            if key in getattr(self, "_live_only_review_keys", set()):
                return False
            parts = key.split(".")
            if len(parts) == 2 and parts[0] in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                if parts[1] == "ORDER":
                    return True
                if parts[1].startswith("@"):
                    try:
                        int(parts[1][1:])
                    except ValueError:
                        return False
                    return True
            return (
                self._change_target(key) is not None
                or self._axis_target(key) is not None
            )

        def _paired_review_row(
            self,
            rows: list[tuple[str, str, str, str]],
            key: str,
            baseline_pair: tuple[Any, str],
            current_pair: tuple[Any, str],
            form_pair: tuple[Any, str],
            formatter,
        ) -> None:
            """One review row for a (literal, source_pv) pair, never two.

            UB_MATRIX/UB_MATRIX_SOURCE_PV (and DISTANCE/DISTANCE_SOURCE_PV) are
            two canonical keys but one logical field in the GUI -- flattening
            them independently would let a keep/drop decision update only one
            half of the pair, silently orphaning the other. Both halves are
            popped out of the flattened diff by the caller and folded into
            this single synthetic row instead.
            """
            old = formatter(*baseline_pair)
            new = formatter(*current_pair)
            if old == new:
                return
            rows.append(("change", key, old, new))
            if formatter(*form_pair) == old:
                self._live_only_review_keys.add(key)

        def unsaved_changes_rows(self) -> list[tuple[str, str, str, str]]:
            pending, invalid_rows = _review_pending_change(self._pending_change)
            if pending is None:
                return invalid_rows
            baseline_parameters = copy.deepcopy(self._normalized_baseline)
            current_parameters = copy.deepcopy(pending["parameters"])
            form_parameters = copy.deepcopy(pending["form_parameters"])
            baseline_parameters.setdefault("DETECTOR_SETUP", {}).update(
                copy.deepcopy(self._detector_setup_extras)
            )
            for axis_key in ("SAMPLE_AXES", "DETECTOR_AXES"):
                baseline_parameters.pop(axis_key, None)
                current_parameters.pop(axis_key, None)
                form_parameters.pop(axis_key, None)

            rows: list[tuple[str, str, str, str]] = []
            self._live_only_review_keys: set[str] = set()

            def _pop_pair(
                mapping: dict[str, Any],
                literal_key: str,
                source_key: str,
                container: str | None = None,
            ) -> tuple[Any, str]:
                target = mapping.get(container, {}) if container else mapping
                return target.pop(literal_key, None), target.pop(source_key, "")

            self._paired_review_row(
                rows,
                "UB Matrix (PV or value)",
                _pop_pair(baseline_parameters, "UB_MATRIX", "UB_MATRIX_SOURCE_PV"),
                _pop_pair(current_parameters, "UB_MATRIX", "UB_MATRIX_SOURCE_PV"),
                _pop_pair(form_parameters, "UB_MATRIX", "UB_MATRIX_SOURCE_PV"),
                _format_ub_or_pv,
            )
            self._paired_review_row(
                rows,
                "Distance (PV or value)",
                _pop_pair(
                    baseline_parameters, "DISTANCE", "DISTANCE_SOURCE_PV", "DETECTOR_SETUP"
                ),
                _pop_pair(
                    current_parameters, "DISTANCE", "DISTANCE_SOURCE_PV", "DETECTOR_SETUP"
                ),
                _pop_pair(
                    form_parameters, "DISTANCE", "DISTANCE_SOURCE_PV", "DETECTOR_SETUP"
                ),
                _format_distance_or_pv,
            )

            baseline = _flatten_config(baseline_parameters, index_lists=True)
            current = _flatten_config(current_parameters, index_lists=True)
            form = _flatten_config(form_parameters, index_lists=True)
            if self.prefix_edit.text() != self.profile.prefix:
                rows.append(
                    (
                        "change",
                        "PREFIX",
                        self.profile.prefix,
                        self.prefix_edit.text(),
                    )
                )
            for key in sorted(set(baseline) & set(current)):
                if baseline[key] != current[key]:
                    rows.append(
                        ("change", key, str(baseline[key]), str(current[key]))
                    )
                    if form.get(key) == baseline[key]:
                        self._live_only_review_keys.add(key)
            for key in sorted(set(current) - set(baseline)):
                rows.append(("add", key, "", str(current[key])))
            for key in sorted(set(baseline) - set(current)):
                rows.append(("remove", key, str(baseline[key]), ""))
            for axis_key, table in (
                ("SAMPLE_AXES", self.sample_table),
                ("DETECTOR_AXES", self.detector_table),
            ):
                baseline_axes = self._normalized_baseline[axis_key]
                current_axes = pending["parameters"][axis_key]
                form_axes = pending["form_parameters"][axis_key]
                origins = table.origins()
                loaded_origins = [
                    origin
                    for origin in origins
                    if origin is not None and origin >= 0
                ]
                if (
                    set(loaded_origins) == set(range(len(baseline_axes)))
                    and loaded_origins != list(range(len(baseline_axes)))
                ):
                    rows.append(
                        (
                            "change",
                            f"{axis_key}.ORDER",
                            json.dumps(list(range(len(baseline_axes)))),
                            json.dumps(loaded_origins),
                        )
                    )
                for row, origin in enumerate(origins):
                    if origin is None:
                        continue
                    if origin < 0:
                        rows.append(
                            (
                                "add",
                                f"{axis_key}.@{origin}",
                                "",
                                json.dumps(current_axes[row], sort_keys=True),
                            )
                        )
                        continue
                    baseline_axis = baseline_axes[origin]
                    for field in AxisTable.axis_keys:
                        old = baseline_axis[field]
                        new = current_axes[row][field]
                        if old == new:
                            continue
                        key = f"{axis_key}.@{origin}.{field}"
                        rows.append(("change", key, str(old), str(new)))
                        if form_axes[row][field] == old:
                            self._live_only_review_keys.add(key)
                present = {origin for origin in origins if origin is not None}
                for origin, axis in enumerate(baseline_axes):
                    if origin not in present:
                        rows.append(
                            (
                                "remove",
                                f"{axis_key}.@{origin}",
                                json.dumps(axis, sort_keys=True),
                                "",
                            )
                        )
            return rows

        def apply_change_decisions(self, kept: list, dropped: list) -> None:
            for _kind, key, old, _new in dropped:
                self._write_change(key, old)
            for kind, key, _old, new in kept:
                if kind != "remove":
                    self._write_change(key, new)

        def _write_change(self, key: str, value: str) -> None:
            parts = key.split(".")
            if len(parts) == 2 and parts[0] in (
                "SAMPLE_AXES",
                "DETECTOR_AXES",
            ):
                table = (
                    self.sample_table
                    if parts[0] == "SAMPLE_AXES"
                    else self.detector_table
                )
                if parts[1] == "ORDER":
                    try:
                        table.reorder_loaded([int(item) for item in json.loads(value)])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        return
                    return
                if parts[1].startswith("@"):
                    try:
                        origin = int(parts[1][1:])
                    except ValueError:
                        return
                    if not value:
                        table.remove_origin(origin)
                        return
                    try:
                        values = json.loads(value)
                    except json.JSONDecodeError:
                        return
                    if isinstance(values, Mapping):
                        table.replace_axis(origin, values)
                    return
            axis = self._axis_target(key)
            if axis is not None:
                table, row, column = axis
                item = table.table.item(row, column)
                origin = item.data(Qt.UserRole) if item is not None else None
                replacement = QTableWidgetItem(value)
                replacement.setData(Qt.UserRole, origin)
                table.table.setItem(row, column, replacement)
                return
            target = self._change_target(key)
            if target is None:
                return
            if isinstance(target, QComboBox):
                target.setCurrentText(value)
            elif key.startswith("DETECTOR_SETUP."):
                target.setText(_format_detector_value(_parse_detector_value(value)))
            else:
                target.setText(value)

        def unsaved_changes_text(self) -> str:
            return (
                "This editor has staged form or live IOC changes that have not "
                "been saved to the profile."
            )

        def closeEvent(self, event) -> None:
            if not self.confirm_close(event):
                return
            self._stop_worker()
            settings = QSettings("DashPVA", "RSMParameterIOC")
            settings.setValue("window_geom", self.saveGeometry())
            for section, key in (
                (self.calibration_group, "static_geometry_expanded"),
                (self.detector_setup_group, "detector_setup_expanded"),
                (self.advanced_group, "advanced_expanded"),
            ):
                self.save_checkable_state(settings, section, key)
            settings.sync()
            super().closeEvent(event)

    _gui_classes_cache = (PollWorker, AxisTable, SimulatorWindow)
    return _gui_classes_cache


def _run_gui(
    session: RSMParameterEditSession,
    initial_profile: RSMParameterProfile,
    restart_ioc,
    pv_values: dict[str, Any],
    pv_lock: threading.Lock,
) -> None:
    from PyQt5.QtWidgets import QApplication

    from dashpva.gui import configure_app

    _, _, SimulatorWindow = _build_gui_classes()
    app = QApplication(sys.argv)
    configure_app(app)
    window = SimulatorWindow(session, initial_profile, restart_ioc, pv_values, pv_lock)
    window.show()
    sys.exit(app.exec_())


def _active_session():
    from dashpva.utils.config.source import ConfigSource

    return RSMParameterEditSession(ConfigSource(app_settings.LOCATOR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-driven RSM parameter IOC")
    parser.add_argument("--ioc-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prefix", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prefix is not None:
        parser.error(
            "--prefix is no longer an override; set root IOC_PREFIX in the active "
            "Workflow profile"
        )

    if args.ioc_mode:
        _run_ioc(_active_session().raw)
        return

    try:
        session = _active_session()
        profile = profile_from_raw(session.raw)
    except Exception as exc:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.warning(
            None,
            "RSM profile unavailable",
            f"{exc}\n\nSelect a valid profile in the Workflow dialog and try again.",
        )
        raise SystemExit(2) from exc

    pv_values: dict[str, Any] = {}
    pv_lock = threading.Lock()
    process_holder: list[subprocess.Popen | None] = [None]
    ready = threading.Event()

    def launch_ioc(snapshot: Mapping[str, Any]) -> subprocess.Popen:
        # The child re-reads the profile from the database rather than being
        # handed a serialized copy. Activation only happens after a successful
        # compare-and-swap save, so the database already *is* the snapshot --
        # and per AGENTS.md the DB is the config source, with TOML reserved for
        # import/export. A side-channel JSON file would be a third format.
        del snapshot
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--ioc-mode",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        process_holder[0] = process

        def forward_output() -> None:
            if process.stdout is None:
                return
            for raw_line in process.stdout:
                text = raw_line.decode(errors="replace").strip()
                try:
                    message = json.loads(text)
                    if message.get("type") == "values":
                        with pv_lock:
                            pv_values.clear()
                            pv_values.update(message["data"])
                        ready.set()
                        continue
                except (json.JSONDecodeError, AttributeError):
                    pass
                print(text, flush=True)

        threading.Thread(target=forward_output, daemon=True).start()
        return process

    def stop_ioc() -> None:
        process = process_holder[0]
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        process_holder[0] = None

    def restart_ioc(snapshot: Mapping[str, Any]) -> None:
        stop_ioc()
        with pv_lock:
            pv_values.clear()
        ready.clear()
        launch_ioc(snapshot)
        if not ready.wait(timeout=15):
            raise RuntimeError("IOC did not publish a snapshot within 15 seconds")

    launch_ioc(session.raw)
    if not ready.wait(timeout=15):
        print("Warning: IOC did not respond within 15 seconds.", flush=True)

    try:
        _run_gui(session, profile, restart_ioc, pv_values, pv_lock)
    finally:
        stop_ioc()


if __name__ == "__main__":
    main()
