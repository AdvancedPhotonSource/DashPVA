# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Shared xrayutilities geometry construction for offline and live RSM."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import xrayutilities as xu

AxisRole = Literal["sample", "detector"]

_SAMPLE_DIRECTION = re.compile(r"[xyzk][+-]")
_DETECTOR_DIRECTION = re.compile(r"[xyz][+-]")
_EXPLICIT_SAMPLE_ORIENTATION = re.compile(r"[xyz][+-]")


def _vector3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{label} must contain exactly three values.")
    if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError(f"{label} must be finite and non-zero.")
    return tuple(float(component) for component in vector)


def _matrix3(value: Sequence[Sequence[float]], label: str) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} must have shape (3, 3).")
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"{label} must be finite and full rank.")
    return tuple(tuple(float(component) for component in row) for row in matrix)


def direction_vector(direction: str) -> np.ndarray:
    """Return xrayutilities' physical vector for a rotation direction."""
    try:
        vector = np.asarray(xu.math.getVector(str(direction).strip().lower()), dtype=float)
    except Exception as exc:
        raise ValueError(f"Invalid xrayutilities rotation direction {direction!r}.") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError(f"Rotation direction {direction!r} did not produce a valid vector.")
    return vector


def _parallel(left: Sequence[float], right: Sequence[float]) -> bool:
    return bool(np.isclose(np.linalg.norm(np.cross(left, right)), 0.0))


@dataclass(frozen=True, slots=True)
class RotationAxis:
    """One ordered sample or detector rotation axis.

    ``role`` comes from the profile list containing the axis. ``record_name``
    is its stable machine identity; ``label`` remains human-editable.
    """

    role: AxisRole
    direction: str
    label: str = ""
    record_name: str = ""
    source_pv: str = ""
    angle_units: str = "deg"

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        if role not in ("sample", "detector"):
            raise ValueError(f"Axis role must be 'sample' or 'detector', got {self.role!r}.")
        object.__setattr__(self, "role", role)

        direction = str(self.direction).strip().lower()
        pattern = _SAMPLE_DIRECTION if role == "sample" else _DETECTOR_DIRECTION
        if pattern.fullmatch(direction) is None:
            allowed = "[xyzk][+-]" if role == "sample" else "[xyz][+-]"
            raise ValueError(
                f"Invalid {role} rotation direction {self.direction!r}; expected {allowed}."
            )
        object.__setattr__(self, "direction", direction)

        units = str(self.angle_units).strip().lower()
        if units not in ("deg", "degree", "degrees"):
            raise ValueError(
                f"Axis {self.record_name or self.label or direction!r} uses unsupported "
                f"angle units {self.angle_units!r}; PR 1 geometry requires degrees."
            )
        object.__setattr__(self, "angle_units", "deg")

    @classmethod
    def from_mapping(cls, role: AxisRole, values: Mapping[str, object]) -> RotationAxis:
        """Build an axis from one canonical ``IOC_RSM_PARAMETER`` list item."""
        try:
            direction = str(values["DIRECTION"])
        except KeyError as exc:
            raise ValueError("Canonical rotation axis is missing DIRECTION.") from exc
        return cls(
            role=role,
            direction=direction,
            label=str(values.get("LABEL", "")),
            record_name=str(values.get("RECORD_NAME", "")),
            source_pv=str(values.get("SOURCE_PV", "")),
            angle_units=str(values.get("ANGLE_UNITS", "deg")),
        )


@dataclass(frozen=True, slots=True)
class DetectorModel:
    """Legacy area-detector calibration used by the shared PR 1 builder."""

    pixel_direction_1: str
    pixel_direction_2: str
    center_channel: tuple[float, float]
    shape: tuple[int, int]
    pixel_width: tuple[float, float]
    distance: float
    roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        for field_name in ("pixel_direction_1", "pixel_direction_2"):
            value = str(getattr(self, field_name)).strip().lower()
            if _DETECTOR_DIRECTION.fullmatch(value) is None:
                raise ValueError(f"{field_name} must use [xyz][+-] syntax, got {value!r}.")
            object.__setattr__(self, field_name, value)

        center = tuple(float(value) for value in self.center_channel)
        if len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("Detector center_channel must contain two finite values.")
        object.__setattr__(self, "center_channel", center)

        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("Detector shape must contain two positive integers.")
        object.__setattr__(self, "shape", shape)

        pixel_width = tuple(float(value) for value in self.pixel_width)
        if (
            len(pixel_width) != 2
            or not np.isfinite(pixel_width).all()
            or any(value <= 0 for value in pixel_width)
        ):
            raise ValueError("Detector pixel_width must contain two finite positive values.")
        object.__setattr__(self, "pixel_width", pixel_width)

        distance = float(self.distance)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Detector distance must be finite and positive.")
        object.__setattr__(self, "distance", distance)

        roi = self.roi if self.roi is not None else (0, shape[0], 0, shape[1])
        roi = tuple(int(value) for value in roi)
        if len(roi) != 4:
            raise ValueError("Detector roi must contain four integer bounds.")
        if not (0 <= roi[0] < roi[1] <= shape[0] and 0 <= roi[2] < roi[3] <= shape[1]):
            raise ValueError(f"Detector roi {roi!r} falls outside detector shape {shape!r}.")
        object.__setattr__(self, "roi", roi)


@dataclass(frozen=True, slots=True)
class RSMGeometry:
    """Frame-invariant diffraction geometry with ordered rotation axes."""

    sample_axes: tuple[RotationAxis, ...]
    detector_axes: tuple[RotationAxis, ...]
    primary_beam_direction: tuple[float, float, float]
    inplane_reference_direction: tuple[float, float, float]
    sample_surface_normal_direction: tuple[float, float, float]
    energy_eV: float
    ub_matrix: tuple[tuple[float, ...], ...]
    detector: DetectorModel
    sample_orientation: str = "det"

    def __post_init__(self) -> None:
        sample_axes = tuple(self.sample_axes)
        detector_axes = tuple(self.detector_axes)
        if any(not isinstance(axis, RotationAxis) or axis.role != "sample" for axis in sample_axes):
            raise ValueError("sample_axes may contain only sample RotationAxis values.")
        if any(
            not isinstance(axis, RotationAxis) or axis.role != "detector"
            for axis in detector_axes
        ):
            raise ValueError("detector_axes may contain only detector RotationAxis values.")
        object.__setattr__(self, "sample_axes", sample_axes)
        object.__setattr__(self, "detector_axes", detector_axes)

        primary = _vector3(self.primary_beam_direction, "Primary beam direction")
        inplane = _vector3(self.inplane_reference_direction, "In-plane reference direction")
        surface = _vector3(
            self.sample_surface_normal_direction, "Sample surface-normal direction"
        )
        if _parallel(inplane, surface):
            raise ValueError(
                "In-plane reference and sample surface-normal directions must be independent."
            )
        cosine = np.dot(inplane, surface) / (
            np.linalg.norm(inplane) * np.linalg.norm(surface)
        )
        if not np.isclose(cosine, 0.0):
            raise ValueError(
                "In-plane reference and sample surface-normal directions must be "
                "perpendicular; xrayutilities would otherwise adjust the in-plane vector."
            )
        object.__setattr__(self, "primary_beam_direction", primary)
        object.__setattr__(self, "inplane_reference_direction", inplane)
        object.__setattr__(self, "sample_surface_normal_direction", surface)

        energy = float(self.energy_eV)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Photon energy must be finite and positive in eV.")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "ub_matrix", _matrix3(self.ub_matrix, "UB matrix"))

        orientation = str(self.sample_orientation).strip().lower()
        validate_sample_orientation(sample_axes, detector_axes, primary, orientation)
        object.__setattr__(self, "sample_orientation", orientation)

        if not isinstance(self.detector, DetectorModel):
            raise TypeError("detector must be a DetectorModel.")


@dataclass(frozen=True, slots=True)
class BuiltRSMGeometry:
    """A validated geometry and its initialized xrayutilities object."""

    model: RSMGeometry
    hxrd: xu.HXRD

    @property
    def ub(self) -> np.ndarray:
        return np.asarray(self.model.ub_matrix, dtype=float)

    @property
    def energy_eV(self) -> float:
        return self.model.energy_eV

    @property
    def shape(self) -> tuple[int, int]:
        return self.model.detector.shape


def validate_sample_orientation(
    sample_axes: Sequence[RotationAxis],
    detector_axes: Sequence[RotationAxis],
    primary_beam_direction: Sequence[float],
    sample_orientation: str,
    *,
    warn_for_sample_axis: bool = True,
) -> None:
    """Reject cases xrayutilities would fail or silently reinterpret."""
    primary = np.asarray(primary_beam_direction, dtype=float)
    orientation = str(sample_orientation).strip().lower()

    if orientation == "det":
        if not detector_axes:
            raise ValueError("SAMPLE_ORIENTATION='det' requires a detector rotation axis.")
        innermost = direction_vector(detector_axes[-1].direction)
        if _parallel(innermost, primary):
            if len(detector_axes) < 2 or _parallel(
                direction_vector(detector_axes[-2].direction), primary
            ):
                raise ValueError(
                    "SAMPLE_ORIENTATION='det' requires an innermost detector rotation "
                    "not parallel to the primary beam (a final beam-axis rotation is ignored)."
                )
        return

    if orientation == "sam":
        if not sample_axes:
            raise ValueError("SAMPLE_ORIENTATION='sam' requires a sample rotation axis.")
        if _parallel(direction_vector(sample_axes[-1].direction), primary):
            raise ValueError(
                "SAMPLE_ORIENTATION='sam' requires the innermost sample axis not to be "
                "parallel to the primary beam."
            )
        if warn_for_sample_axis:
            warnings.warn(
                "SAMPLE_ORIENTATION='sam' is physically correct only when the innermost "
                "sample circle is the azimuth motor.",
                UserWarning,
                stacklevel=2,
            )
        return

    if _EXPLICIT_SAMPLE_ORIENTATION.fullmatch(orientation) is None:
        raise ValueError(
            "SAMPLE_ORIENTATION must be 'det', 'sam', or explicit [xyz][+-] syntax."
        )
    if _parallel(direction_vector(orientation), primary):
        raise ValueError(
            f"Explicit SAMPLE_ORIENTATION={orientation!r} is parallel to the primary "
            "beam; xrayutilities would silently substitute a different axis."
        )


def build_hxrd(model: RSMGeometry) -> BuiltRSMGeometry:
    """Validate and initialize one xrayutilities area-detector geometry."""
    q_conversion = xu.experiment.QConversion(
        [axis.direction for axis in model.sample_axes],
        [axis.direction for axis in model.detector_axes],
        model.primary_beam_direction,
    )
    hxrd = xu.HXRD(
        model.inplane_reference_direction,
        model.sample_surface_normal_direction,
        en=model.energy_eV,
        qconv=q_conversion,
        sampleor=model.sample_orientation,
    )
    detector = model.detector
    hxrd.Ang2Q.init_area(
        detector.pixel_direction_1,
        detector.pixel_direction_2,
        cch1=detector.center_channel[0],
        cch2=detector.center_channel[1],
        Nch1=detector.shape[0],
        Nch2=detector.shape[1],
        pwidth1=detector.pixel_width[0],
        pwidth2=detector.pixel_width[1],
        distance=detector.distance,
        roi=list(detector.roi),
    )
    return BuiltRSMGeometry(model=model, hxrd=hxrd)


def calculate_q(
    geometry: BuiltRSMGeometry,
    sample_angles: Sequence[object],
    detector_angles: Sequence[object],
    *,
    ub_matrix: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ordered sample/detector angles with an initialized geometry."""
    if len(sample_angles) != len(geometry.model.sample_axes):
        raise ValueError(
            f"Expected {len(geometry.model.sample_axes)} sample angles, "
            f"received {len(sample_angles)}."
        )
    if len(detector_angles) != len(geometry.model.detector_axes):
        raise ValueError(
            f"Expected {len(geometry.model.detector_axes)} detector angles, "
            f"received {len(detector_angles)}."
        )
    ub = geometry.ub if ub_matrix is None else np.asarray(
        _matrix3(ub_matrix, "UB matrix"), dtype=float
    )
    return geometry.hxrd.Ang2Q.area(
        *sample_angles,
        *detector_angles,
        UB=ub,
        deg=True,
    )
