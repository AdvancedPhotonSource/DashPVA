# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Ordering and identity helpers for configured HKL rotation axes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from dashpva.utils.config.hkl import (
    AxisRole,
    axis_group_parts,
    numbered_axis_group_names,
)

_LEGACY_GROUPS = {
    "sample": ("MU", "ETA", "CHI", "PHI"),
    "detector": ("NU", "DELTA"),
}


def has_canonical_axis_parameters(config: Mapping[str, Any]) -> bool:
    """Whether numbered axes are governed by IOC_RSM_PARAMETER ordering."""
    return isinstance(config.get("IOC_RSM_PARAMETER"), Mapping)

def resolved_axis_groups(names: Iterable[str], role: AxisRole) -> list[str]:
    """Prefer numbered groups, falling back to the established named layout."""
    available = {str(name) for name in names}
    numbered = numbered_axis_group_names(available, role)
    if numbered:
        return numbered
    return [name for name in _LEGACY_GROUPS[role] if name in available]


def hkl_group_sort_key(name: str) -> tuple[int, int, str]:
    """Sort numbered sample/detector groups naturally and other keys stably."""
    parts = axis_group_parts(name)
    if parts is not None:
        role, number = parts
        return (0 if role == "sample" else 1, number, str(name))
    for role_index, role in enumerate(("sample", "detector"), start=2):
        try:
            return (role_index, _LEGACY_GROUPS[role].index(str(name)), str(name))
        except ValueError:
            pass
    return (4, 0, str(name))


def canonical_axis_metadata(
    config: Mapping[str, Any], group_name: str
) -> dict[str, str]:
    """Return persisted human/machine identity fields for a numbered axis."""
    parts = axis_group_parts(group_name)
    parameters = config.get("IOC_RSM_PARAMETER")
    if parts is None or not isinstance(parameters, Mapping):
        return {}
    role, number = parts
    key = "SAMPLE_AXES" if role == "sample" else "DETECTOR_AXES"
    axes = parameters.get(key)
    if not isinstance(axes, list) or number > len(axes):
        return {}
    axis = axes[number - 1]
    if not isinstance(axis, Mapping):
        return {}
    return {
        field: str(axis[field]).strip()
        for field in (
            "LABEL",
            "SPEC_MOTOR_NAME",
            "RECORD_NAME",
            "SOURCE_PV",
            "DIRECTION",
            "ANGLE_UNITS",
        )
        if axis.get(field) is not None and str(axis[field]).strip()
    }
