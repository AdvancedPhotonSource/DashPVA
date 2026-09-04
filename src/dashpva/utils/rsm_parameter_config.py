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

"""Pure configuration model for the RSM-parameter IOC and editor."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

import dashpva.settings as app_settings
from dashpva.utils.config.resolver import resolve_profile_config
from dashpva.utils.config.source import ConfigSaveResult, ConfigSaveStatus
from dashpva.utils.rsm_geometry import (
    FRAME_AXIS_ORDERS,
    RotationAxis,
    validate_sample_orientation,
)
from dashpva.utils.units import normalize_angle_units, normalize_length_units

_RECORD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_PREFIX = re.compile(r"[A-Za-z0-9_.:-]*")
_SAMPLE_DIRECTION = re.compile(r"[xyzk][+-]")
_DETECTOR_DIRECTION = re.compile(r"[xyz][+-]")
_EXPLICIT_ORIENTATION = re.compile(r"[xyz][+-]")


DEFAULT_SAMPLE_AXES = (
    {
        "LABEL": "Mu",
        "RECORD_NAME": "Mu",
        "SOURCE_PV": "6idb1:m28.RBV",
        "DIRECTION": "x+",
        "ANGLE_UNITS": "deg",
    },
    {
        "LABEL": "Eta",
        "RECORD_NAME": "Eta",
        "SOURCE_PV": "6idb1:m17.RBV",
        "DIRECTION": "z-",
        "ANGLE_UNITS": "deg",
    },
    {
        "LABEL": "Chi",
        "RECORD_NAME": "Chi",
        "SOURCE_PV": "6idb1:m19.RBV",
        "DIRECTION": "y+",
        "ANGLE_UNITS": "deg",
    },
    {
        "LABEL": "Phi",
        "RECORD_NAME": "Phi",
        "SOURCE_PV": "6idb1:m20.RBV",
        "DIRECTION": "z-",
        "ANGLE_UNITS": "deg",
    },
)

DEFAULT_DETECTOR_AXES = (
    {
        "LABEL": "Nu",
        "RECORD_NAME": "Nu",
        "SOURCE_PV": "6idb1:m29.RBV",
        "DIRECTION": "x+",
        "ANGLE_UNITS": "deg",
    },
    {
        "LABEL": "Delta",
        "RECORD_NAME": "Delta",
        "SOURCE_PV": "6idb1:m18.RBV",
        "DIRECTION": "z-",
        "ANGLE_UNITS": "deg",
    },
)

DEFAULT_ENERGY_SOURCE_PV = "6idb:spec:Energy"
DEFAULT_ENERGY_UNITS = "keV"
DEFAULT_SAMPLE_ORIENTATION = "det"
SCHEMA_VERSION = 1

DEFAULT_UB = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
DEFAULT_PRIMARY_BEAM = (0.0, 1.0, 0.0)
DEFAULT_INPLANE_REFERENCE = (0.0, 1.0, 0.0)
DEFAULT_SAMPLE_NORMAL = (0.0, 0.0, 1.0)
DEFAULT_DETECTOR = {
    "pixel_dir1": "z-",
    "pixel_dir2": "x-",
    "center": (300.0, 300.0),
    "size": (28.38, 28.38),
    "distance": 400.644,
    "units": "mm",
}

_STATIC_PARAMETER_KEYS = (
    "UB_MATRIX",
    "PRIMARY_BEAM_DIRECTION",
    "INPLANE_REFERENCE_DIRECTION",
    "SAMPLE_SURFACE_NORMAL_DIRECTION",
    "DETECTOR_SETUP",
)
_AXIS_KEYS = ("SAMPLE_AXES", "DETECTOR_AXES")
_AXIS_FIELDS = (
    "LABEL",
    "RECORD_NAME",
    "SOURCE_PV",
    "DIRECTION",
    "ANGLE_UNITS",
)
_MISSING = object()

AxisOrigins = Mapping[str, Sequence[int | None]]


class SnapshotConfigSource(Protocol):
    """Minimal strict persistence API used by the staged editor."""

    def load_snapshot(self) -> tuple[dict[str, Any], str]: ...

    def replace_if_revision(
        self,
        full_config: dict[str, Any],
        revision: str,
    ) -> ConfigSaveResult: ...


@dataclass(frozen=True, slots=True)
class AxisParameter:
    """One ordered, validated rotation circle."""

    label: str
    record_name: str
    source_pv: str
    direction: str
    angle_units: str = "deg"


    def as_mapping(self) -> dict[str, str]:
        return {
            "LABEL": self.label,
            "RECORD_NAME": self.record_name,
            "SOURCE_PV": self.source_pv,
            "DIRECTION": self.direction,
            "ANGLE_UNITS": self.angle_units,
        }


@dataclass(frozen=True, slots=True)
class RSMParameterProfile:
    """Runtime IOC configuration derived from one raw profile snapshot."""

    prefix: str
    schema_version: int
    sample_axes: tuple[AxisParameter, ...]
    detector_axes: tuple[AxisParameter, ...]
    energy_source_pv: str
    energy_units: str
    sample_orientation: str
    ub_matrix: tuple[float, ...]
    ub_matrix_source_pv: str
    primary_beam_direction: tuple[float, float, float]
    inplane_reference_direction: tuple[float, float, float]
    sample_surface_normal_direction: tuple[float, float, float]
    detector_setup: Mapping[str, Any]

    @property
    def axes(self) -> tuple[AxisParameter, ...]:
        return self.sample_axes + self.detector_axes

    @property
    def detector_distance_source_pv(self) -> str:
        """Optional live distance source, separate from the numeric fallback."""
        return str(self.detector_setup.get("DISTANCE_SOURCE_PV", ""))

    def parameter_mapping(self) -> dict[str, Any]:
        return {
            "SCHEMA_VERSION": self.schema_version,
            "SAMPLE_AXES": [axis.as_mapping() for axis in self.sample_axes],
            "DETECTOR_AXES": [axis.as_mapping() for axis in self.detector_axes],
            "ENERGY_SOURCE_PV": self.energy_source_pv,
            "ENERGY_UNITS": self.energy_units,
            "SAMPLE_ORIENTATION": self.sample_orientation,
            "UB_MATRIX": list(self.ub_matrix),
            "UB_MATRIX_SOURCE_PV": self.ub_matrix_source_pv,
            "PRIMARY_BEAM_DIRECTION": list(self.primary_beam_direction),
            "INPLANE_REFERENCE_DIRECTION": list(self.inplane_reference_direction),
            "SAMPLE_SURFACE_NORMAL_DIRECTION": list(
                self.sample_surface_normal_direction
            ),
            "DETECTOR_SETUP": copy.deepcopy(dict(self.detector_setup)),
        }


def default_parameter_mapping() -> dict[str, Any]:
    """Return a detached canonical six-circle starter configuration."""
    return {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "SAMPLE_AXES": copy.deepcopy(list(DEFAULT_SAMPLE_AXES)),
        "DETECTOR_AXES": copy.deepcopy(list(DEFAULT_DETECTOR_AXES)),
        "ENERGY_SOURCE_PV": DEFAULT_ENERGY_SOURCE_PV,
        "ENERGY_UNITS": DEFAULT_ENERGY_UNITS,
        "SAMPLE_ORIENTATION": DEFAULT_SAMPLE_ORIENTATION,
        "UB_MATRIX": list(DEFAULT_UB),
        "UB_MATRIX_SOURCE_PV": "",
        "PRIMARY_BEAM_DIRECTION": list(DEFAULT_PRIMARY_BEAM),
        "INPLANE_REFERENCE_DIRECTION": list(DEFAULT_INPLANE_REFERENCE),
        "SAMPLE_SURFACE_NORMAL_DIRECTION": list(DEFAULT_SAMPLE_NORMAL),
        "DETECTOR_SETUP": {
            "PIXEL_DIRECTION_1": DEFAULT_DETECTOR["pixel_dir1"],
            "PIXEL_DIRECTION_2": DEFAULT_DETECTOR["pixel_dir2"],
            "CENTER_CHANNEL_PIXEL": list(DEFAULT_DETECTOR["center"]),
            "SIZE": list(DEFAULT_DETECTOR["size"]),
            "DISTANCE": DEFAULT_DETECTOR["distance"],
            "UNITS": DEFAULT_DETECTOR["units"],
        },
    }


def normalize_prefix(value: object) -> str:
    """Validate and normalize the sole IOC record prefix."""
    if not isinstance(value, str):
        raise ValueError("IOC_PREFIX must be a string")
    prefix = value.strip()
    if prefix and _PREFIX.fullmatch(prefix) is None:
        raise ValueError("IOC_PREFIX may contain only letters, digits, '_', '.', '-', and ':'")
    if prefix and not prefix.endswith(":"):
        prefix += ":"
    return prefix


def _normalize_axis(role: str, index: int, values: object) -> AxisParameter:
    if not isinstance(values, Mapping):
        raise ValueError(f"{role} axis {index} must be a table")

    label = str(values.get("LABEL", "")).strip()
    record_name = str(values.get("RECORD_NAME", "")).strip()
    source_pv = str(values.get("SOURCE_PV", "")).strip()
    direction = str(values.get("DIRECTION", "")).strip().lower()
    angle_units = str(values.get("ANGLE_UNITS", "deg")).strip().lower()

    if not label:
        raise ValueError(f"{role} axis {index} needs a LABEL")
    if _RECORD_NAME.fullmatch(record_name) is None:
        raise ValueError(
            f"{role} axis {index} RECORD_NAME must be an unprefixed EPICS record stem"
        )
    if not source_pv:
        raise ValueError(f"{role} axis {index} needs a SOURCE_PV or static number")
    _validate_source(source_pv, f"{role} axis {index} SOURCE_PV")
    direction_pattern = _SAMPLE_DIRECTION if role == "sample" else _DETECTOR_DIRECTION
    if direction_pattern.fullmatch(direction) is None:
        allowed = "[xyzk][+-]" if role == "sample" else "[xyz][+-]"
        raise ValueError(f"{role} axis {index} DIRECTION must match {allowed}")
    # ANGLE_UNITS describes the *source PV*, not the published record. The IOC
    # converts to degrees on publish (see _to_degrees in ioc_rsm_parameter), so
    # every downstream consumer and the geometry model itself stay degrees-only.
    angle_units = normalize_angle_units(angle_units, f"{role} axis {index} ANGLE_UNITS")

    return AxisParameter(
        label, record_name, source_pv, direction, angle_units
    )


def _validate_source(value: str, label: str) -> None:
    try:
        static_value = float(value)
    except ValueError:
        return
    if not math.isfinite(static_value):
        raise ValueError(f"{label} static value must be finite")


def _reject_numeric_source_pv(value: str, label: str) -> None:
    """Reject text that cannot unambiguously name one CA source PV."""
    try:
        float(value)
    except ValueError:
        if any(character.isspace() for character in value):
            raise ValueError(f"{label} must be one CA PV name without whitespace")
        if value.startswith(("[", "{")):
            raise ValueError(f"{label} must be a CA PV name, not malformed JSON")
        if not any(character.isalpha() or character in "_:" for character in value):
            raise ValueError(f"{label} must be a valid CA PV name")
        return
    raise ValueError(f"{label} must be a PV name, not a static number")


def validate_source_pv(value: object, label: str) -> str:
    """Return one nonempty, nonnumeric CA source name."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a CA PV name")
    source = value.strip()
    if not source:
        raise ValueError(f"{label} must not be blank")
    _reject_numeric_source_pv(source, label)
    return source


def _finite_vector(values: object, label: str, length: int) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain {length} finite numbers") from exc
    if len(vector) != length or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return vector


def validate_ub_matrix(value: object, label: str = "UB_MATRIX") -> tuple[float, ...]:
    """Validate a flat, row-major 9-number UB matrix: finite and full rank.

    Shared by the runtime validator and the GUI's combined PV-or-literal field,
    so a JSON-shaped-but-invalid entry is rejected the same way in both places.
    """
    ub = _finite_vector(value, label, 9)
    if np.linalg.matrix_rank(np.asarray(ub).reshape(3, 3)) < 3:
        raise ValueError(f"{label} must be full rank")
    return ub


def validate_distance(value: object, label: str = "DETECTOR_SETUP.DISTANCE") -> float:
    """Validate a sample-to-detector distance: finite and strictly positive."""
    distance = float(value)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return distance


def _static_geometry(parameters: Mapping[str, Any]) -> tuple[
    tuple[float, ...],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    dict[str, Any],
]:
    ub = validate_ub_matrix(parameters.get("UB_MATRIX", DEFAULT_UB), "UB_MATRIX")

    primary = _finite_vector(
        parameters.get("PRIMARY_BEAM_DIRECTION", DEFAULT_PRIMARY_BEAM),
        "PRIMARY_BEAM_DIRECTION",
        3,
    )
    inplane = _finite_vector(
        parameters.get("INPLANE_REFERENCE_DIRECTION", DEFAULT_INPLANE_REFERENCE),
        "INPLANE_REFERENCE_DIRECTION",
        3,
    )
    surface = _finite_vector(
        parameters.get("SAMPLE_SURFACE_NORMAL_DIRECTION", DEFAULT_SAMPLE_NORMAL),
        "SAMPLE_SURFACE_NORMAL_DIRECTION",
        3,
    )
    if any(np.isclose(np.linalg.norm(vector), 0.0) for vector in (primary, inplane, surface)):
        raise ValueError("beam and sample reference directions must be non-zero")
    cosine = np.dot(inplane, surface) / (
        np.linalg.norm(inplane) * np.linalg.norm(surface)
    )
    if not np.isclose(cosine, 0.0):
        raise ValueError(
            "INPLANE_REFERENCE_DIRECTION and SAMPLE_SURFACE_NORMAL_DIRECTION "
            "must be perpendicular"
        )

    detector_value = parameters.get("DETECTOR_SETUP", {})
    if detector_value is None:
        detector_value = {}
    if not isinstance(detector_value, Mapping):
        raise ValueError("DETECTOR_SETUP must be a table")
    detector = {
        "PIXEL_DIRECTION_1": str(
            detector_value.get("PIXEL_DIRECTION_1", DEFAULT_DETECTOR["pixel_dir1"])
        ).strip().lower(),
        "PIXEL_DIRECTION_2": str(
            detector_value.get("PIXEL_DIRECTION_2", DEFAULT_DETECTOR["pixel_dir2"])
        ).strip().lower(),
        "CENTER_CHANNEL_PIXEL": list(
            _finite_vector(
                detector_value.get("CENTER_CHANNEL_PIXEL", DEFAULT_DETECTOR["center"]),
                "DETECTOR_SETUP.CENTER_CHANNEL_PIXEL",
                2,
            )
        ),
        "SIZE": list(
            _finite_vector(
                detector_value.get("SIZE", DEFAULT_DETECTOR["size"]),
                "DETECTOR_SETUP.SIZE",
                2,
            )
        ),
        "DISTANCE": validate_distance(
            detector_value.get("DISTANCE", DEFAULT_DETECTOR["distance"]),
            "DETECTOR_SETUP.DISTANCE",
        ),
        "UNITS": str(detector_value.get("UNITS", DEFAULT_DETECTOR["units"])).strip(),
    }
    for key in ("PIXEL_DIRECTION_1", "PIXEL_DIRECTION_2"):
        if _DETECTOR_DIRECTION.fullmatch(detector[key]) is None:
            raise ValueError(f"DETECTOR_SETUP.{key} must match [xyz][+-]")
    if any(value <= 0 for value in detector["SIZE"]):
        raise ValueError("DETECTOR_SETUP.SIZE values must be positive")
    if not detector["UNITS"]:
        raise ValueError("DETECTOR_SETUP.UNITS is required")
    if "DISTANCE_SOURCE_PV" in detector_value:
        distance_source_pv = validate_source_pv(
            detector_value["DISTANCE_SOURCE_PV"],
            "DETECTOR_SETUP.DISTANCE_SOURCE_PV",
        )
        detector["DISTANCE_SOURCE_PV"] = distance_source_pv

    # --- PR 3 calibration ------------------------------------------------
    # All optional: a profile that omits every one of these behaves exactly as
    # it did before, so existing beamline profiles keep working untouched.
    normalize_length_units(detector["UNITS"], "DETECTOR_SETUP.UNITS")
    for key, source_key in (
        ("DISTANCE_UNITS", "DISTANCE"),
        ("SIZE_UNITS", "SIZE"),
        ("PIXEL_SIZE_UNITS", "PIXEL_SIZE"),
    ):
        if key in detector_value:
            detector[key] = normalize_length_units(
                detector_value[key], f"DETECTOR_SETUP.{key}"
            )
        del source_key

    if "PIXEL_SIZE" in detector_value:
        pixel_size = _finite_vector(
            detector_value["PIXEL_SIZE"], "DETECTOR_SETUP.PIXEL_SIZE", 2
        )
        if any(value <= 0 for value in pixel_size):
            raise ValueError("DETECTOR_SETUP.PIXEL_SIZE values must be positive")
        detector["PIXEL_SIZE"] = list(pixel_size)

    if "DETECTOR_SHAPE" in detector_value:
        shape = _finite_vector(
            detector_value["DETECTOR_SHAPE"], "DETECTOR_SETUP.DETECTOR_SHAPE", 2
        )
        if any(value <= 0 or value != int(value) for value in shape):
            raise ValueError(
                "DETECTOR_SETUP.DETECTOR_SHAPE must be two positive integers "
                "(the full unbinned detector, not the acquired frame)"
            )
        detector["DETECTOR_SHAPE"] = [int(value) for value in shape]

    if "BINNING" in detector_value:
        binning = _finite_vector(detector_value["BINNING"], "DETECTOR_SETUP.BINNING", 2)
        if any(value < 1 or value != int(value) for value in binning):
            raise ValueError("DETECTOR_SETUP.BINNING must be two positive integers")
        detector["BINNING"] = [int(value) for value in binning]

    if "ROI" in detector_value:
        roi = _finite_vector(detector_value["ROI"], "DETECTOR_SETUP.ROI", 4)
        if any(value != int(value) for value in roi):
            raise ValueError("DETECTOR_SETUP.ROI must contain four integers")
        roi = [int(value) for value in roi]
        if not (0 <= roi[0] < roi[1] and 0 <= roi[2] < roi[3]):
            raise ValueError(
                "DETECTOR_SETUP.ROI must be half-open unbinned bounds "
                "[start1, stop1, start2, stop2) with start < stop"
            )
        detector["ROI"] = roi

    # Units are validated here but values are NOT converted: the profile keeps
    # whatever the beamline declared, and conversion to the canonical eV/mm/deg
    # happens once, when the DetectorModel is built. Converting here instead
    # would make the stored value and the stored unit disagree, so a second
    # normalization pass would convert again -- normalization must be idempotent.
    tilt_units = normalize_angle_units(
        detector_value.get("ANGLE_UNITS", "deg"), "DETECTOR_SETUP.ANGLE_UNITS"
    )
    if "ANGLE_UNITS" in detector_value:
        detector["ANGLE_UNITS"] = tilt_units
    for key in ("DETROT", "TILT", "TILTAZIMUTH"):
        if key in detector_value:
            value = float(detector_value[key])
            if not math.isfinite(value):
                raise ValueError(f"DETECTOR_SETUP.{key} must be a finite angle")
            detector[key] = value

    if "FRAME_AXIS_ORDER" in detector_value:
        order = str(detector_value["FRAME_AXIS_ORDER"]).strip().lower()
        if order not in FRAME_AXIS_ORDERS:
            raise ValueError(
                f"DETECTOR_SETUP.FRAME_AXIS_ORDER must be one of {FRAME_AXIS_ORDERS}"
            )
        detector["FRAME_AXIS_ORDER"] = order

    return ub, primary, inplane, surface, detector


def validate_parameter_profile(
    prefix: object,
    parameters: object,
) -> RSMParameterProfile:
    """Return an immutable normalized profile or raise a user-facing error."""
    normalized_prefix = normalize_prefix(prefix)
    if not isinstance(parameters, Mapping):
        raise ValueError("IOC_RSM_PARAMETER must be a table")
    schema_version = parameters.get("SCHEMA_VERSION", SCHEMA_VERSION)
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported IOC_RSM_PARAMETER SCHEMA_VERSION {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    sample_values = parameters.get("SAMPLE_AXES", [])
    detector_values = parameters.get("DETECTOR_AXES", [])
    if not isinstance(sample_values, list):
        raise ValueError("SAMPLE_AXES must be an ordered list")
    if not isinstance(detector_values, list):
        raise ValueError("DETECTOR_AXES must be an ordered list")

    sample_axes = tuple(
        _normalize_axis("sample", index, values)
        for index, values in enumerate(sample_values, start=1)
    )
    detector_axes = tuple(
        _normalize_axis("detector", index, values)
        for index, values in enumerate(detector_values, start=1)
    )

    record_names = [axis.record_name for axis in sample_axes + detector_axes]
    if len(record_names) != len(set(record_names)):
        raise ValueError("RECORD_NAME values must be unique across sample and detector axes")

    energy_source = str(parameters.get("ENERGY_SOURCE_PV", "")).strip()
    if not energy_source:
        raise ValueError("ENERGY_SOURCE_PV must be a PV name or static number")
    _validate_source(energy_source, "ENERGY_SOURCE_PV")
    energy_units = str(parameters.get("ENERGY_UNITS", DEFAULT_ENERGY_UNITS)).strip()
    if energy_units.lower() != "kev":
        raise ValueError("ENERGY_UNITS must be keV until energy conversion lands")

    sample_orientation = str(
        parameters.get("SAMPLE_ORIENTATION", DEFAULT_SAMPLE_ORIENTATION)
    ).strip().lower()
    if sample_orientation == "det":
        if not detector_axes:
            raise ValueError("SAMPLE_ORIENTATION='det' requires a detector rotation axis")
    elif sample_orientation == "sam":
        if not sample_axes:
            raise ValueError("SAMPLE_ORIENTATION='sam' requires a sample rotation axis")
    elif _EXPLICIT_ORIENTATION.fullmatch(sample_orientation) is None:
        raise ValueError("SAMPLE_ORIENTATION must be 'det', 'sam', or [xyz][+-]")

    ub, primary, inplane, surface, detector = _static_geometry(parameters)

    raw_ub_matrix_source_pv = parameters.get("UB_MATRIX_SOURCE_PV", "")
    ub_matrix_source_pv = str(raw_ub_matrix_source_pv).strip()
    if raw_ub_matrix_source_pv and not ub_matrix_source_pv:
        raise ValueError("UB_MATRIX_SOURCE_PV must not be blank if present")
    if ub_matrix_source_pv:
        ub_matrix_source_pv = validate_source_pv(
            raw_ub_matrix_source_pv, "UB_MATRIX_SOURCE_PV"
        )

    rotation_sample = tuple(
        RotationAxis("sample", axis.direction) for axis in sample_axes
    )
    rotation_detector = tuple(
        RotationAxis("detector", axis.direction) for axis in detector_axes
    )
    validate_sample_orientation(
        rotation_sample,
        rotation_detector,
        primary,
        sample_orientation,
        warn_for_sample_axis=False,
    )

    return RSMParameterProfile(
        prefix=normalized_prefix,
        schema_version=SCHEMA_VERSION,
        sample_axes=sample_axes,
        detector_axes=detector_axes,
        energy_source_pv=energy_source,
        energy_units="keV",
        sample_orientation=sample_orientation,
        ub_matrix=ub,
        ub_matrix_source_pv=ub_matrix_source_pv,
        primary_beam_direction=primary,
        inplane_reference_direction=inplane,
        sample_surface_normal_direction=surface,
        detector_setup=detector,
    )


def profile_from_raw(raw: Mapping[str, Any]) -> RSMParameterProfile:
    """Resolve canonical parameters from a raw profile without using DETECTOR_PREFIX."""
    parameters = raw.get("IOC_RSM_PARAMETER")
    if parameters is None:
        parameters = default_parameter_mapping()
    return validate_parameter_profile(raw.get("IOC_PREFIX", ""), parameters)


def _read_path(container: Any, path: tuple[Any, ...], default: Any = None) -> Any:
    for step in path:
        try:
            container = container[step]
        except (KeyError, IndexError, TypeError):
            return default
    return container


def _write_path(container: Any, path: tuple[Any, ...], value: Any) -> None:
    for step in path[:-1]:
        container = container[step]
    container[path[-1]] = value


def _stored_change(
    submitted: Any,
    normalized: Any,
    template: Any = _MISSING,
) -> Any:
    """Keep the profile's numeric container types while storing validated edits."""
    if template is not _MISSING:
        if isinstance(template, bool) and isinstance(normalized, bool):
            return normalized
        if isinstance(template, int) and not isinstance(template, bool):
            try:
                numeric = float(normalized)
                if math.isfinite(numeric) and numeric.is_integer():
                    return int(numeric)
            except (TypeError, ValueError):
                pass
        if isinstance(template, float):
            try:
                return float(normalized)
            except (TypeError, ValueError):
                pass
        if isinstance(template, (list, tuple)) and isinstance(normalized, (list, tuple)):
            values = [
                _stored_change(
                    submitted[index] if isinstance(submitted, (list, tuple)) and index < len(submitted) else item,
                    item,
                    template[index] if index < len(template) else _MISSING,
                )
                for index, item in enumerate(normalized)
            ]
            return tuple(values) if isinstance(template, tuple) else values
    if isinstance(normalized, str):
        return normalized
    if isinstance(submitted, (int, float, bool, list, tuple)):
        return copy.deepcopy(submitted)
    return copy.deepcopy(normalized)


def _infer_axis_origins(
    baseline_axes: Sequence[Mapping[str, Any]],
    submitted_axes: Sequence[Mapping[str, Any]],
) -> list[int | None]:
    """Best-effort identity for non-GUI callers; the editor supplies exact origins."""
    available = list(range(len(baseline_axes)))
    origins: list[int | None] = []
    for row, axis in enumerate(submitted_axes):
        record = str(axis.get("RECORD_NAME", "")).strip()
        matches = [
            index
            for index in available
            if str(baseline_axes[index].get("RECORD_NAME", "")).strip() == record
        ]
        if len(matches) == 1:
            origin = matches[0]
        elif row in available:
            origin = row
        else:
            origin = None
        origins.append(origin)
        if origin in available:
            available.remove(origin)
    return origins


def _patched_axes(
    raw_axes: object,
    baseline_axes: Sequence[Mapping[str, Any]],
    submitted_axes: Sequence[Mapping[str, Any]],
    normalized_axes: Sequence[Mapping[str, Any]],
    origins: Sequence[int | None] | None,
) -> list[dict[str, Any]]:
    raw_list = raw_axes if isinstance(raw_axes, list) else []
    resolved_origins = list(origins) if origins is not None else _infer_axis_origins(
        baseline_axes, submitted_axes
    )
    if len(resolved_origins) != len(normalized_axes):
        raise ValueError("axis identity count does not match the edited axis count")

    stored: list[dict[str, Any]] = []
    used: set[int] = set()
    for submitted, normalized, origin in zip(
        submitted_axes, normalized_axes, resolved_origins
    ):
        if origin is None or origin < 0:
            new_axis = {
                key: _stored_change(submitted.get(key), normalized[key])
                for key in _AXIS_FIELDS
            }
            stored.append(new_axis)
            continue
        if origin in used or not 0 <= origin < len(baseline_axes):
            raise ValueError("axis identities must reference each loaded axis at most once")
        used.add(origin)
        raw_axis = raw_list[origin] if origin < len(raw_list) else {}
        raw_axis = raw_axis if isinstance(raw_axis, Mapping) else {}
        patched = copy.deepcopy(dict(raw_axis))
        baseline = baseline_axes[origin]
        for key in _AXIS_FIELDS:
            if normalized[key] == baseline[key]:
                continue
            patched[key] = _stored_change(
                submitted.get(key), normalized[key], raw_axis.get(key, _MISSING)
            )
        stored.append(patched)
    return stored


def _patched_detector_setup(
    raw_setup: object,
    baseline: Mapping[str, Any],
    submitted: object,
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    raw_mapping = raw_setup if isinstance(raw_setup, Mapping) else {}
    submitted_mapping = submitted if isinstance(submitted, Mapping) else {}
    stored = copy.deepcopy(dict(raw_mapping))
    managed = set(baseline) | set(normalized)
    for key in managed:
        if key not in submitted_mapping:
            stored.pop(key, None)
            continue
        if key in raw_mapping and normalized.get(key) == baseline.get(key):
            continue
        if key not in raw_mapping and normalized.get(key) == baseline.get(key):
            continue
        stored[key] = _stored_change(
            submitted_mapping[key],
            normalized[key],
            raw_mapping.get(key, _MISSING),
        )
    for key, value in submitted_mapping.items():
        if key not in managed and key not in stored:
            stored[key] = copy.deepcopy(value)
    return stored


def update_raw_profile(
    raw: Mapping[str, Any],
    prefix: object,
    parameters: object,
    *,
    axis_origins: AxisOrigins | None = None,
) -> dict[str, Any]:
    """Patch a raw profile without rewriting values the form did not change.

    Runtime normalization supplies defaults and canonical Python types. Persistence
    instead starts from the exact loaded document, retaining absent keys, unknown
    extensions, and original numeric container types until the corresponding
    semantic value is edited.
    """
    normalized = validate_parameter_profile(prefix, parameters)
    submitted = parameters if isinstance(parameters, Mapping) else {}
    baseline_profile = profile_from_raw(raw)
    baseline = baseline_profile.parameter_mapping()
    candidate = normalized.parameter_mapping()
    replacement = copy.deepcopy(dict(raw))
    if normalized.prefix != baseline_profile.prefix:
        replacement["IOC_PREFIX"] = normalized.prefix

    raw_parameters = raw.get("IOC_RSM_PARAMETER")
    if not isinstance(raw_parameters, Mapping):
        replacement["IOC_PREFIX"] = normalized.prefix
        stored_candidate = copy.deepcopy(candidate)
        replacement["IOC_RSM_PARAMETER"] = stored_candidate
        return replacement

    stored = copy.deepcopy(dict(raw_parameters))
    for key in _AXIS_KEYS:
        submitted_axes = submitted.get(key, [])
        if not isinstance(submitted_axes, list) or not all(
            isinstance(axis, Mapping) for axis in submitted_axes
        ):
            raise ValueError(f"{key} must be an ordered list")
        origins = axis_origins.get(key) if axis_origins is not None else None
        patched_axes = _patched_axes(
            raw_parameters.get(key),
            baseline[key],
            submitted_axes,
            candidate[key],
            origins,
        )
        if key in raw_parameters or patched_axes:
            stored[key] = patched_axes
        else:
            stored.pop(key, None)

    patched_setup = _patched_detector_setup(
        raw_parameters.get("DETECTOR_SETUP"),
        baseline["DETECTOR_SETUP"],
        submitted.get("DETECTOR_SETUP"),
        candidate["DETECTOR_SETUP"],
    )
    if "DETECTOR_SETUP" in raw_parameters or patched_setup:
        stored["DETECTOR_SETUP"] = patched_setup
    else:
        stored.pop("DETECTOR_SETUP", None)
    for key, value in candidate.items():
        if key in (*_AXIS_KEYS, "DETECTOR_SETUP"):
            continue
        if value == baseline.get(key, _MISSING):
            continue
        stored[key] = _stored_change(
            submitted.get(key), value, raw_parameters.get(key, _MISSING)
        )
    replacement["IOC_RSM_PARAMETER"] = stored
    return replacement


def _values_match(left: object, right: object) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    if isinstance(right, (list, tuple)):
        try:
            return len(left) == len(right) and all(  # type: ignore[arg-type]
                _values_match(a, b) for a, b in zip(left, right)  # type: ignore[arg-type]
            )
        except TypeError:
            return False
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return math.isclose(
                float(left),
                float(right),
                rel_tol=app_settings.RSM_STATIC_METADATA_RELATIVE_TOLERANCE,
                abs_tol=app_settings.RSM_STATIC_METADATA_ABSOLUTE_TOLERANCE,
            )
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def _adoptable_records(
    profile: RSMParameterProfile,
) -> dict[str, tuple[Any, ...]]:
    prefix = profile.prefix
    records: dict[str, tuple[Any, ...]] = {}
    for key, axes in (
        ("SAMPLE_AXES", profile.sample_axes),
        ("DETECTOR_AXES", profile.detector_axes),
    ):
        for origin, axis in enumerate(axes):
            base = f"{prefix}{axis.record_name}"
            records[f"{base}:DirectionAxis"] = (key, origin, "DIRECTION")
    records[f"{prefix}spec:Energy:Units"] = ("ENERGY_UNITS",)
    # A source-owned record is driven by its PV every poll -- a live caput to it
    # would just be overwritten on the next cycle, and adopting it into the form
    # would silently discard the configured source. Only participate in adoption
    # when no source PV is configured, matching the per-axis SOURCE_PV records
    # above (which were never adoptable in the first place).
    if not profile.ub_matrix_source_pv:
        records[f"{prefix}spec:UB_matrix:Value"] = ("UB_MATRIX",)
    for group, key in (
        ("PrimaryBeamDirection", "PRIMARY_BEAM_DIRECTION"),
        ("InplaneReferenceDirection", "INPLANE_REFERENCE_DIRECTION"),
        ("SampleSurfaceNormalDirection", "SAMPLE_SURFACE_NORMAL_DIRECTION"),
    ):
        for index in range(3):
            records[f"{prefix}{group}:AxisNumber{index + 1}"] = (key, index)
    for suffix, key in (
        ("PixelDirection1", "PIXEL_DIRECTION_1"),
        ("PixelDirection2", "PIXEL_DIRECTION_2"),
        ("CenterChannelPixel", "CENTER_CHANNEL_PIXEL"),
        ("Size", "SIZE"),
        ("Units", "UNITS"),
    ):
        records[f"{prefix}DetectorSetup:{suffix}"] = ("DETECTOR_SETUP", key)
    if not profile.detector_distance_source_pv:
        records[f"{prefix}DetectorSetup:Distance"] = ("DETECTOR_SETUP", "DISTANCE")
    return records


def _current_form_path(
    baseline_path: tuple[Any, ...],
    axis_origins: AxisOrigins | None,
) -> tuple[Any, ...] | None:
    if baseline_path[0] not in _AXIS_KEYS or axis_origins is None:
        return baseline_path
    key, origin, *tail = baseline_path
    origins = axis_origins.get(key, ())
    try:
        row = list(origins).index(origin)
    except ValueError:
        return None
    return (key, row, *tail)


def merge_live_records(
    parameters: dict[str, Any],
    baseline_profile: RSMParameterProfile,
    live: Mapping[str, Any],
    raw_baseline: Mapping[str, Any],
    normalized_baseline: Mapping[str, Any],
    *,
    axis_origins: AxisOrigins | None = None,
) -> tuple[list[str], list[str]]:
    """Three-way merge exact loaded values, form edits, and live IOC records.

    Axis records are bound to their original row identity. Reordering or renaming
    an axis therefore cannot make a caput apply to whichever row happens to occupy
    the old index. A removed axis with a concurrent live change is a conflict.
    """
    adopted: list[str] = []
    conflicts: list[str] = []
    for record, baseline_path in _adoptable_records(baseline_profile).items():
        if record not in live:
            continue
        base_value = _read_path(normalized_baseline, baseline_path, _MISSING)
        if base_value is _MISSING:
            base_value = _read_path(raw_baseline, baseline_path, _MISSING)
        if base_value is _MISSING:
            continue
        live_value = live[record]
        seeded_value = base_value
        if _values_match(live_value, seeded_value):
            continue

        form_path = _current_form_path(baseline_path, axis_origins)
        label = ".".join(str(step) for step in baseline_path)
        if form_path is None:
            conflicts.append(
                f"{label}: loaded {base_value!r} | edited (axis removed) "
                f"| IOC {live_value!r}"
            )
            continue
        form_value = _read_path(parameters, form_path, _MISSING)
        if _values_match(live_value, form_value):
            continue
        if _values_match(form_value, base_value):
            _write_path(
                parameters,
                form_path,
                _stored_change(live_value, live_value, base_value),
            )
            adopted.append(f"{label}: {base_value!r} -> {live_value!r}")
        else:
            conflicts.append(
                f"{label}: loaded {base_value!r} | edited {form_value!r} "
                f"| IOC {live_value!r}"
            )
    return adopted, conflicts


def requires_adoption_confirmation(raw: Mapping[str, Any]) -> bool:
    """Whether an unchanged Apply would introduce canonical profile content."""
    parameters = raw.get("IOC_RSM_PARAMETER")
    if not isinstance(parameters, Mapping):
        return True
    profile = profile_from_raw(raw)
    replacement = update_raw_profile(
        raw,
        profile.prefix,
        profile.parameter_mapping(),
    )
    return replacement != raw


def adoption_diff(
    raw: Mapping[str, Any],
    prefix: object,
    parameters: object,
) -> str:
    """Describe the channel/static changes requiring first-adoption consent."""
    replacement = update_raw_profile(raw, prefix, parameters)
    before_hkl = resolve_profile_config(raw).get("HKL", {})
    after_hkl = resolve_profile_config(replacement).get("HKL", {})

    lines = ["Effective HKL channel mapping:"]
    before_flat = _flatten_mapping(before_hkl)
    after_flat = _flatten_mapping(after_hkl)
    changed_paths = sorted(set(before_flat) | set(after_flat))
    for path in changed_paths:
        before = before_flat.get(path, "<missing>")
        after = after_flat.get(path, "<missing>")
        if before != after:
            lines.append(f"  {path}: {before!r} -> {after!r}")
    if len(lines) == 1:
        lines.append("  no effective HKL channel changes")

    lines.append("")
    lines.append("Static geometry stored in IOC_RSM_PARAMETER:")
    existing = raw.get("IOC_RSM_PARAMETER", {})
    existing = existing if isinstance(existing, Mapping) else {}
    candidate = replacement["IOC_RSM_PARAMETER"]
    for key in _STATIC_PARAMETER_KEYS:
        before = existing.get(key, "<not profile-backed>")
        after = candidate.get(key, "<not profile-backed>")
        if before != after:
            lines.append(
                f"  {key}: {json.dumps(before, sort_keys=True)} -> "
                f"{json.dumps(after, sort_keys=True)}"
            )
    # UB_MATRIX_SOURCE_PV is optional (default ""), so it's intentionally not in
    # _STATIC_PARAMETER_KEYS -- adding it there would make every pre-existing
    # profile look "not fully canonical" the first time this field shipped.
    # Still worth surfacing in the diff on its own.
    before_ub_source = existing.get("UB_MATRIX_SOURCE_PV", "")
    after_ub_source = candidate.get("UB_MATRIX_SOURCE_PV", "")
    if before_ub_source != after_ub_source:
        lines.append(
            f"  UB_MATRIX_SOURCE_PV: {before_ub_source!r} -> {after_ub_source!r}"
        )
    return "\n".join(lines)


def _flatten_mapping(values: object, prefix: str = "HKL") -> dict[str, Any]:
    if not isinstance(values, Mapping):
        return {prefix: values}
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, path))
        else:
            flattened[path] = value
    return flattened


class RSMParameterEditSession:
    """Staged raw-profile editor with compare-and-swap persistence."""

    def __init__(self, source: SnapshotConfigSource):
        self.source = source
        self.raw: dict[str, Any] = {}
        self.revision = ""
        self.profile: RSMParameterProfile
        self.normalized: dict[str, Any] = {}
        self.load()

    def load(self) -> RSMParameterProfile:
        raw, revision = self.source.load_snapshot()
        self.raw = copy.deepcopy(raw)
        self.revision = revision
        self.profile = profile_from_raw(self.raw)
        self.normalized = self.profile.parameter_mapping()
        return self.profile

    @property
    def has_canonical_parameters(self) -> bool:
        return isinstance(self.raw.get("IOC_RSM_PARAMETER"), Mapping)

    def apply(
        self,
        prefix: object,
        parameters: object,
        *,
        axis_origins: AxisOrigins | None = None,
    ) -> tuple[ConfigSaveResult, dict[str, Any] | None]:
        replacement = update_raw_profile(
            self.raw, prefix, parameters, axis_origins=axis_origins
        )
        result = self.source.replace_if_revision(replacement, self.revision)
        if result.status is not ConfigSaveStatus.SAVED:
            return result, None
        self.raw = replacement
        self.profile = profile_from_raw(replacement)
        self.normalized = self.profile.parameter_mapping()
        if result.revision is not None:
            self.revision = result.revision
        return result, copy.deepcopy(replacement)


class SnapshotActivationError(RuntimeError):
    """The profile saved, but its exact snapshot could not be activated."""

    def __init__(self, snapshot: dict[str, Any], error: Exception):
        super().__init__(str(error))
        self.snapshot = copy.deepcopy(snapshot)
        self.error = error


def apply_and_activate(
    session: RSMParameterEditSession,
    prefix: object,
    parameters: object,
    activate: Callable[[Mapping[str, Any]], None],
    *,
    axis_origins: AxisOrigins | None = None,
) -> tuple[ConfigSaveResult, dict[str, Any] | None]:
    """CAS-save staged values, then activate only that exact saved snapshot."""
    result, snapshot = session.apply(
        prefix, parameters, axis_origins=axis_origins
    )
    if snapshot is None:
        return result, None
    activation_snapshot = copy.deepcopy(snapshot)
    try:
        activate(activation_snapshot)
    except Exception as exc:
        raise SnapshotActivationError(snapshot, exc) from exc
    return result, snapshot
