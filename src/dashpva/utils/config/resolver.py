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

"""Resolve persisted beamline profiles into runtime configuration."""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

_AXIS_SECTION = re.compile(
    r"^(?:SAMPLE|DETECTOR)_CIRCLE_AXIS_[1-9][0-9]*$"
)
_LEGACY_AXIS_SECTIONS = {"MU", "ETA", "CHI", "PHI", "NU", "DELTA"}
_RECORD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_AXIS_MANAGED_FIELDS = {
    "ANGLE_UNITS",
    "AXIS_NUMBER",
    "DIRECTION_AXIS",
    "POSITION",
    "SPEC_MOTOR_NAME",
}


def _normalized_prefix(config: Mapping[str, Any]) -> str:
    prefix = config.get("IOC_PREFIX", "")
    if not isinstance(prefix, str):
        raise ValueError("IOC_PREFIX must be a string")
    prefix = prefix.strip()
    if prefix and not prefix.endswith(":"):
        prefix += ":"
    return prefix


def _pv(prefix: str, suffix: str) -> str:
    return f"{prefix}{suffix}"


def _managed_section(
    hkl: dict[str, Any],
    name: str,
    values: Mapping[str, str],
    aliases: tuple[str, ...] = (),
) -> None:
    section: dict[str, Any] = {}
    for candidate in (*aliases, name):
        existing = hkl.pop(candidate, None)
        if isinstance(existing, dict):
            section.update(existing)
    section.update(values)
    hkl[name] = section


def _axis_mapping(prefix: str, record_name: str) -> dict[str, str]:
    base = _pv(prefix, record_name)
    return {
        "AXIS_NUMBER": f"{base}:AxisNumber",
        "DIRECTION_AXIS": f"{base}:DirectionAxis",
        "POSITION": f"{base}:Position",
        "SPEC_MOTOR_NAME": f"{base}:SpecMotorName",
    }


def _canonical_axes(parameters: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    axes = parameters.get(key, [])
    if axes is None:
        return []
    if not isinstance(axes, list) or not all(isinstance(axis, Mapping) for axis in axes):
        raise ValueError(f"IOC_RSM_PARAMETER.{key} must be a list of axis tables")
    return axes


def resolve_profile_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a detached runtime view of *raw*.

    Legacy profiles are copied unchanged. A profile containing
    ``IOC_RSM_PARAMETER`` gets an effective ``HKL`` section generated from its
    canonical ordered axes and IOC prefix. The persisted mapping is never
    modified.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("profile configuration must be a mapping")

    effective = copy.deepcopy(dict(raw))
    parameters = raw.get("IOC_RSM_PARAMETER")
    if parameters is None:
        return effective
    if not isinstance(parameters, Mapping):
        raise ValueError("IOC_RSM_PARAMETER must be a table")
    schema_version = parameters.get("SCHEMA_VERSION", 1)
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(
            f"unsupported IOC_RSM_PARAMETER SCHEMA_VERSION {schema_version!r}; expected 1"
        )

    prefix = _normalized_prefix(raw)
    effective["IOC_PREFIX"] = prefix
    hkl = copy.deepcopy(raw.get("HKL", {}))
    if not isinstance(hkl, dict):
        raise ValueError("HKL must be a table")

    old_axis_sections = {
        name: value
        for name, value in hkl.items()
        if _AXIS_SECTION.fullmatch(name) and isinstance(value, dict)
    }
    for name in tuple(hkl):
        if _AXIS_SECTION.fullmatch(name) or name in _LEGACY_AXIS_SECTIONS:
            hkl.pop(name)

    seen_records: set[str] = set()
    for role, key in (("SAMPLE", "SAMPLE_AXES"), ("DETECTOR", "DETECTOR_AXES")):
        for index, axis in enumerate(_canonical_axes(parameters, key), start=1):
            record_name = axis.get("RECORD_NAME")
            if not isinstance(record_name, str) or not record_name.strip():
                raise ValueError(f"IOC_RSM_PARAMETER.{key}[{index - 1}] needs RECORD_NAME")
            record_name = record_name.strip()
            if _RECORD_NAME.fullmatch(record_name) is None:
                raise ValueError(
                    "RECORD_NAME must be an unprefixed record stem containing "
                    f"letters, digits, '_', '.', or '-': {record_name!r}"
                )
            if record_name in seen_records:
                raise ValueError(f"duplicate RECORD_NAME across RSM axes: {record_name!r}")
            seen_records.add(record_name)

            section_name = f"{role}_CIRCLE_AXIS_{index}"
            extension = {
                name: value
                for name, value in old_axis_sections.get(section_name, {}).items()
                if name not in _AXIS_MANAGED_FIELDS
            }
            extension.update(_axis_mapping(prefix, record_name))
            hkl[section_name] = extension

    spec = {
        "ENERGY_VALUE": _pv(prefix, "spec:Energy:Value"),
        "UB_MATRIX_VALUE": _pv(prefix, "spec:UB_matrix:Value"),
    }
    if parameters.get("ENERGY_UNITS") is not None:
        spec["ENERGY_UNITS"] = _pv(prefix, "spec:Energy:Units")
    _managed_section(hkl, "SPEC", spec)

    for name, record, aliases in (
        ("PRIMARY_BEAM_DIRECTION", "PrimaryBeamDirection", ()),
        (
            "INPLANE_REFERENCE_DIRECTION",
            "InplaneReferenceDirection",
            ("INPLANE_REFERENCE_DIRECITON",),
        ),
        (
            "SAMPLE_SURFACE_NORMAL_DIRECTION",
            "SampleSurfaceNormalDirection",
            ("SAMPLE_SURFACE_NORMAL_DIRECITON",),
        ),
    ):
        _managed_section(
            hkl,
            name,
            {
                f"AXIS_NUMBER_{index}": _pv(prefix, f"{record}:AxisNumber{index}")
                for index in range(1, 4)
            },
            aliases,
        )

    _managed_section(
        hkl,
        "DETECTOR_SETUP",
        {
            "CENTER_CHANNEL_PIXEL": _pv(prefix, "DetectorSetup:CenterChannelPixel"),
            "DISTANCE": _pv(prefix, "DetectorSetup:Distance"),
            "PIXEL_DIRECTION_1": _pv(prefix, "DetectorSetup:PixelDirection1"),
            "PIXEL_DIRECTION_2": _pv(prefix, "DetectorSetup:PixelDirection2"),
            "SIZE": _pv(prefix, "DetectorSetup:Size"),
            "UNITS": _pv(prefix, "DetectorSetup:Units"),
        },
    )

    effective["HKL"] = hkl
    return effective
