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

"""Semantic access to HKL channel mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

AxisRole = Literal["sample", "detector"]

AXIS_CHANNEL_FIELDS = (
    "AXIS_NUMBER",
    "DIRECTION_AXIS",
    "POSITION",
    "SPEC_MOTOR_NAME",
)
RSM_AXIS_FIELDS = ("DIRECTION_AXIS", "POSITION")
VECTOR_FIELDS = ("AXIS_NUMBER_1", "AXIS_NUMBER_2", "AXIS_NUMBER_3")

SECTION_ALIASES = {
    "PRIMARY_BEAM_DIRECTION": ("PRIMARY_BEAM_DIRECTION",),
    "INPLANE_REFERENCE_DIRECTION": (
        "INPLANE_REFERENCE_DIRECTION",
        "INPLANE_REFERENCE_DIRECITON",
    ),
    "SAMPLE_SURFACE_NORMAL_DIRECTION": (
        "SAMPLE_SURFACE_NORMAL_DIRECTION",
        "SAMPLE_SURFACE_NORMAL_DIRECITON",
    ),
    "SPEC": ("SPEC",),
    "DETECTOR_SETUP": ("DETECTOR_SETUP",),
}

SECTION_CHANNEL_FIELDS = {
    "PRIMARY_BEAM_DIRECTION": VECTOR_FIELDS,
    "INPLANE_REFERENCE_DIRECTION": VECTOR_FIELDS,
    "SAMPLE_SURFACE_NORMAL_DIRECTION": VECTOR_FIELDS,
    "SPEC": ("ENERGY_VALUE", "UB_MATRIX_VALUE", "ENERGY_UNITS"),
    "DETECTOR_SETUP": (
        "CENTER_CHANNEL_PIXEL",
        "DISTANCE",
        "PIXEL_DIRECTION_1",
        "PIXEL_DIRECTION_2",
        "SIZE",
        "UNITS",
    ),
}

RSM_REQUIRED_SECTION_FIELDS = {
    "PRIMARY_BEAM_DIRECTION": VECTOR_FIELDS,
    "INPLANE_REFERENCE_DIRECTION": VECTOR_FIELDS,
    "SAMPLE_SURFACE_NORMAL_DIRECTION": VECTOR_FIELDS,
    "SPEC": ("ENERGY_VALUE", "UB_MATRIX_VALUE"),
    "DETECTOR_SETUP": (
        "CENTER_CHANNEL_PIXEL",
        "DISTANCE",
        "PIXEL_DIRECTION_1",
        "PIXEL_DIRECTION_2",
        "SIZE",
    ),
}

_AXIS_SECTION = re.compile(r"^(SAMPLE|DETECTOR)_CIRCLE_AXIS_([1-9][0-9]*)$")
_ROLE_PREFIX = {"sample": "SAMPLE", "detector": "DETECTOR"}
_ROLE_ORDER = {"sample": 0, "detector": 1}


@dataclass(frozen=True)
class HklAxisSection:
    """One numbered sample or detector circle mapping."""

    role: str
    index: int
    name: str
    channels: Mapping[str, Any]


@dataclass(frozen=True)
class HklChannel:
    """A semantic config field bound to one EPICS channel."""

    section: str
    field: str
    channel: str


def _channel(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def axis_group_parts(name: str) -> tuple[AxisRole, int] | None:
    """Return semantic role and one-based number for an exact axis group."""
    match = _AXIS_SECTION.fullmatch(str(name))
    if match is None:
        return None
    return ("sample" if match.group(1) == "SAMPLE" else "detector", int(match.group(2)))


def numbered_axis_group_names(names: Iterable[str], role: AxisRole) -> tuple[str, ...]:
    """Return exact numbered group names for *role* in numeric order."""
    if role not in _ROLE_PREFIX:
        raise ValueError(f"unknown HKL axis role: {role!r}")
    matches = []
    for name in names:
        parts = axis_group_parts(name)
        if parts is not None and parts[0] == role:
            matches.append((parts[1], str(name)))
    return tuple(name for _, name in sorted(matches))


def iter_hkl_axes(
    hkl_config: Mapping[str, Any],
    role: str | None = None,
) -> tuple[HklAxisSection, ...]:
    """Return numbered circle sections in role/numeric order."""
    if role is not None and role not in _ROLE_PREFIX:
        raise ValueError(f"unknown HKL axis role: {role!r}")

    axes: list[HklAxisSection] = []
    seen: set[tuple[str, int]] = set()
    for name, section in hkl_config.items():
        parts = axis_group_parts(name)
        if parts is None:
            continue
        axis_role, index = parts
        if role is not None and axis_role != role:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"HKL.{name} must be a table")
        identity = (axis_role, index)
        if identity in seen:
            raise ValueError(f"duplicate HKL {axis_role} axis number {index}")
        seen.add(identity)
        axes.append(HklAxisSection(axis_role, index, name, section))
    return tuple(sorted(axes, key=lambda axis: (_ROLE_ORDER[axis.role], axis.index)))


def get_hkl_section(
    hkl_config: Mapping[str, Any],
    semantic_name: str,
    *,
    required: bool = False,
) -> Mapping[str, Any]:
    """Return a named section, accepting canonical and legacy typo aliases."""
    aliases = SECTION_ALIASES.get(semantic_name, (semantic_name,))
    for name in aliases:
        section = hkl_config.get(name)
        if isinstance(section, Mapping):
            return section
        if section is not None:
            raise ValueError(f"HKL.{name} must be a table")
    if required:
        raise ValueError(f"missing HKL section {semantic_name}")
    return {}


def axis_field_channels(
    hkl_config: Mapping[str, Any],
    role: str,
    field: str,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Return one explicitly named field for each axis of *role*."""
    result = []
    for axis in iter_hkl_axes(hkl_config, role):
        channel = _channel(axis.channels.get(field))
        if channel is None:
            if required:
                raise ValueError(f"missing HKL.{axis.name}.{field} channel")
            continue
        result.append(channel)
    return tuple(result)


def section_field_channels(
    hkl_config: Mapping[str, Any],
    semantic_name: str,
    fields: tuple[str, ...],
    *,
    required: bool = True,
) -> tuple[str, ...]:
    """Return explicitly named fields from a canonical/legacy section."""
    section = get_hkl_section(hkl_config, semantic_name, required=required)
    result = []
    for field in fields:
        channel = _channel(section.get(field))
        if channel is None:
            if required:
                raise ValueError(f"missing HKL.{semantic_name}.{field} channel")
            continue
        result.append(channel)
    return tuple(result)


def iter_semantic_hkl_channels(
    hkl_config: Mapping[str, Any],
) -> tuple[HklChannel, ...]:
    """Return only recognized PV-valued HKL fields."""
    bindings = []
    for axis in iter_hkl_axes(hkl_config):
        for field in AXIS_CHANNEL_FIELDS:
            channel = _channel(axis.channels.get(field))
            if channel is not None:
                bindings.append(HklChannel(axis.name, field, channel))
    for section_name, fields in SECTION_CHANNEL_FIELDS.items():
        section = get_hkl_section(hkl_config, section_name)
        for field in fields:
            channel = _channel(section.get(field))
            if channel is not None:
                bindings.append(HklChannel(section_name, field, channel))
    return tuple(bindings)


def semantic_hkl_channels(hkl_config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return recognized channels once each, preserving semantic order."""
    channels = []
    seen = set()
    for binding in iter_semantic_hkl_channels(hkl_config):
        if binding.channel not in seen:
            seen.add(binding.channel)
            channels.append(binding.channel)
    return tuple(channels)


def required_rsm_channels(hkl_config: Mapping[str, Any]) -> frozenset[str]:
    """Return the channels required to calculate Q for a detector frame."""
    channels = set()
    for role in _ROLE_PREFIX:
        for field in RSM_AXIS_FIELDS:
            channels.update(axis_field_channels(hkl_config, role, field))
    for section_name, fields in RSM_REQUIRED_SECTION_FIELDS.items():
        channels.update(section_field_channels(hkl_config, section_name, fields))
    return frozenset(channels)
