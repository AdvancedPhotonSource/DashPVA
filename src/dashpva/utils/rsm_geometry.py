# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Shared xrayutilities geometry construction for offline and live RSM."""

from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import xrayutilities as xu

from dashpva.utils.config.hkl import SECTION_CHANNEL_FIELDS
from dashpva.utils.units import normalize_length_units, to_deg, to_mm

AxisRole = Literal["sample", "detector"]

_SAMPLE_DIRECTION = re.compile(r"[xyzk][+-]")
_DETECTOR_DIRECTION = re.compile(r"[xyz][+-]")
_EXPLICIT_SAMPLE_ORIENTATION = re.compile(r"[xyz][+-]")

#: Array axis 0 of an acquired frame runs along ``pixel_direction_1``. This is
#: the layout ``Ang2Q.area`` returns, so no transform is needed.
FRAME_AXIS_DIRECT = "direction1_direction2"
#: Array axis 0 runs along ``pixel_direction_2`` instead -- the detector is
#: read out transposed relative to the calibration. Q is transposed to match
#: the frame so intensity, mask and Q always share one indexing convention.
FRAME_AXIS_SWAPPED = "direction2_direction1"
FRAME_AXIS_ORDERS = (FRAME_AXIS_DIRECT, FRAME_AXIS_SWAPPED)


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
    """Area-detector calibration in unbinned, full-frame coordinates.

    The central invariant: ``center_channel``, ``shape`` and ``pixel_width``
    always describe the **whole physical detector at binning 1**, never the
    frame that was actually acquired. Cropping and binning are handed to
    xrayutilities as ``roi``/``Nav``, which applies them itself
    (``QConversion._get_detparam_area`` divides the centre by Nav, multiplies
    the pixel width by Nav, and maps an unbinned ROI into binned channels).

    Pre-shifting the centre or pre-scaling the pixel width to match a cropped
    frame therefore double-applies the correction. The legacy
    ``SIZE / frame_shape`` pixel width did exactly that -- it silently absorbed
    binning into the pixel size, so a 2x-binned scan produced a Q scale wrong
    by a factor of two with no error raised. That derivation is now permitted
    only for full-frame, unbinned data (see ``pixel_width_from_size``).

    ``roi`` is half-open in unbinned channels, ``[start1, stop1, start2,
    stop2)``, and its span must be exactly divisible by ``binning`` --
    xrayutilities would otherwise ``ceil`` the span and hand back a frame one
    channel larger than the detector actually produced.

    Lengths are millimetres, angles degrees; convert at the boundary with
    :mod:`dashpva.utils.units`.
    """

    pixel_direction_1: str
    pixel_direction_2: str
    center_channel: tuple[float, float]
    shape: tuple[int, int]
    pixel_width: tuple[float, float]
    distance: float
    roi: tuple[int, int, int, int] | None = None
    binning: tuple[int, int] = (1, 1)
    detrot: float = 0.0
    tilt: float = 0.0
    tiltazimuth: float = 0.0
    frame_axis_order: str = FRAME_AXIS_DIRECT

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

        binning = tuple(int(value) for value in self.binning)
        if len(binning) != 2 or any(value < 1 for value in binning):
            raise ValueError(
                f"Detector binning must be two positive integers, got {self.binning!r}."
            )
        object.__setattr__(self, "binning", binning)

        # xrayutilities ceils a non-divisible span, which would silently return
        # one channel more than the detector produced. Reject it instead.
        for index, (span, factor, axis) in enumerate(
            ((roi[1] - roi[0], binning[0], 1), (roi[3] - roi[2], binning[1], 2))
        ):
            if span % factor:
                raise ValueError(
                    f"Detector ROI span {span} along direction {axis} is not divisible by "
                    f"binning {factor}. xrayutilities would round the span up and report a "
                    f"frame larger than the detector produced; adjust the ROI or binning."
                )

        for field_name in ("detrot", "tilt", "tiltazimuth"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value):
                raise ValueError(f"Detector {field_name} must be a finite angle in degrees.")
            object.__setattr__(self, field_name, value)

        order = str(self.frame_axis_order).strip().lower()
        if order not in FRAME_AXIS_ORDERS:
            raise ValueError(
                f"Detector frame_axis_order must be one of {FRAME_AXIS_ORDERS}, got "
                f"{self.frame_axis_order!r}."
            )
        object.__setattr__(self, "frame_axis_order", order)

    @property
    def acquired_shape(self) -> tuple[int, int]:
        """Shape of one acquired frame after this ROI and binning.

        Compare against the real frame shape before converting: a mismatch
        means the calibration and the data disagree about what was read out,
        which otherwise surfaces as a plausible but wrongly-scaled volume.
        """
        roi = self.roi
        rows = (roi[1] - roi[0]) // self.binning[0]
        cols = (roi[3] - roi[2]) // self.binning[1]
        if self.frame_axis_order == FRAME_AXIS_SWAPPED:
            return (cols, rows)
        return (rows, cols)

    def require_frame_shape(self, frame_shape: Sequence[int]) -> None:
        """Raise unless ``frame_shape`` matches :attr:`acquired_shape`."""
        actual = tuple(int(value) for value in tuple(frame_shape)[:2])
        expected = self.acquired_shape
        if actual != expected:
            raise ValueError(
                f"Acquired frame shape {actual} does not match the calibration: ROI "
                f"{self.roi} with binning {self.binning} and frame axis order "
                f"'{self.frame_axis_order}' implies {expected}. Check the detector ROI, "
                f"binning, or frame axis order in the profile."
            )

    def orient_frame_array(self, array: np.ndarray) -> np.ndarray:
        """Map a detector-ordered array onto the acquired frame layout.

        ``Ang2Q.area`` always returns ``(direction1, direction2)``. When the
        detector reads out transposed, the trailing two axes are swapped so Q,
        intensity and mask share one indexing convention. Leading axes (a frame
        batch) are preserved.
        """
        if self.frame_axis_order == FRAME_AXIS_DIRECT:
            return array
        data = np.asarray(array)
        if data.ndim < 2:
            raise ValueError("Frame-axis reordering needs at least two dimensions.")
        return np.swapaxes(data, -1, -2)


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
    # Nch/cch/pwidth are the unbinned full-frame calibration; roi and Nav are
    # passed through so xrayutilities applies cropping and binning itself.
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
        detrot=detector.detrot,
        tilt=detector.tilt,
        tiltazimuth=detector.tiltazimuth,
        roi=list(detector.roi),
        Nav=list(detector.binning),
    )
    return BuiltRSMGeometry(model=model, hxrd=hxrd)


def pixel_width_from_size(
    size: Sequence[float],
    frame_shape: Sequence[int],
    detector: Mapping[str, object] | None = None,
    *,
    binning: Sequence[int] = (1, 1),
    roi: Sequence[int] | None = None,
) -> tuple[float, float]:
    """Derive pixel width from a total detector size, for legacy data only.

    ``SIZE / frame_shape`` is only the physical pixel width when the frame is
    the whole unbinned detector. Under an ROI it divides the full width by a
    cropped channel count; under binning it divides by too few channels. Both
    yield a wrong Q scale that still looks reasonable, so this refuses rather
    than guessing whenever the frame is not full-frame and unbinned.
    """
    del detector  # accepted for call-site symmetry; nothing here needs it
    binning = tuple(int(value) for value in binning)
    if any(value != 1 for value in binning):
        raise ValueError(
            f"Cannot derive pixel width from total detector SIZE when binning is "
            f"{binning}: the division would absorb the binning factor. Provide an "
            f"explicit unbinned PIXEL_SIZE in the profile."
        )
    shape = tuple(int(value) for value in tuple(frame_shape)[:2])
    if roi is not None:
        roi = tuple(int(value) for value in roi)
        if (roi[1] - roi[0], roi[3] - roi[2]) != shape:
            raise ValueError(
                f"Cannot derive pixel width from total detector SIZE under ROI {roi}: "
                f"the division would use a cropped channel count. Provide an explicit "
                f"unbinned PIXEL_SIZE in the profile."
            )
    values = tuple(float(value) for value in tuple(size)[:2])
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"Detector SIZE must be two positive lengths, got {size!r}.")
    if any(value <= 0 for value in shape):
        raise ValueError(f"Detector frame shape must be positive, got {frame_shape!r}.")
    return (values[0] / shape[0], values[1] / shape[1])


#: DETECTOR_SETUP fields that detector_model_from_setup understands. Live paths
#: resolve each through the profile's channel map, so a beamline exposes only
#: the ones it actually publishes.
DETECTOR_SETUP_FIELDS = (
    "PIXEL_DIRECTION_1",
    "PIXEL_DIRECTION_2",
    "CENTER_CHANNEL_PIXEL",
    "DISTANCE",
    "SIZE",
    "PIXEL_SIZE",
    "DETECTOR_SHAPE",
    "BINNING",
    "ROI",
    "DETROT",
    "TILT",
    "TILTAZIMUTH",
    "UNITS",
    "DISTANCE_UNITS",
    "SIZE_UNITS",
    "PIXEL_SIZE_UNITS",
    "ANGLE_UNITS",
    "FRAME_AXIS_ORDER",
)

_DETECTOR_REQUIRED_FIELDS = (
    "PIXEL_DIRECTION_1",
    "PIXEL_DIRECTION_2",
    "CENTER_CHANNEL_PIXEL",
    "DISTANCE",
)


def detector_setup_from_channels(
    section: Mapping[str, str],
    values: Mapping[str, object],
    canonical: Mapping[str, object] | None = None,
) -> dict:
    """Resolve a DETECTOR_SETUP channel map into a literal setup mapping.

    ``section`` maps a field name to the PV/attribute name that carries it;
    ``values`` maps that name to the value most recently seen. ``canonical`` is
    the profile's own ``IOC_RSM_PARAMETER.DETECTOR_SETUP`` table, used as the
    base so calibration the IOC does not publish as a record (pixel size, ROI,
    binning, tilt, frame axis order) still reaches live geometry. Only the six
    fields in the public IOC ``DETECTOR_SETUP`` channel contract may overlay
    that base. Both inputs are copied so constructing live geometry cannot
    mutate the raw profile or a frame's metadata. Fields neither side supplies
    are omitted so
    :func:`detector_model_from_setup` falls back to its legacy defaults rather
    than reading a ``None`` as a number.
    """
    if canonical is not None and not isinstance(canonical, Mapping):
        raise TypeError("Canonical DETECTOR_SETUP must be a mapping.")
    setup: dict = copy.deepcopy(dict(canonical or {}))
    for field in DETECTOR_SETUP_FIELDS:
        if setup.get(field) is None:
            setup.pop(field, None)
    for field in SECTION_CHANNEL_FIELDS["DETECTOR_SETUP"]:
        channel = section.get(field)
        if not channel or channel not in values:
            continue
        value = values[channel]
        if value is None:
            continue
        setup[field] = copy.deepcopy(value)
    missing = [field for field in _DETECTOR_REQUIRED_FIELDS if field not in setup]
    if missing:
        raise ValueError(
            f"DETECTOR_SETUP is missing required value(s): {', '.join(missing)}."
        )
    if "SIZE" not in setup and "PIXEL_SIZE" not in setup:
        raise ValueError(
            "DETECTOR_SETUP needs either PIXEL_SIZE (preferred) or SIZE to determine "
            "the physical pixel width."
        )
    return setup


def detector_model_from_setup(
    setup: Mapping[str, object],
    frame_shape: Sequence[int],
    *,
    verify_frame_shape: bool = True,
) -> DetectorModel:
    """Build a DetectorModel from a DETECTOR_SETUP mapping, converting units.

    This is the single boundary where declared units become canonical mm and
    degrees. Profiles keep whatever the beamline declared; nothing downstream
    of here sees a unit string.

    A setup with none of the PR 3 keys reproduces the previous behavior
    exactly -- full-frame, unbinned, no tilt, pixel width derived from SIZE --
    so existing beamline profiles convert identically.
    """
    def _length(key: str, unit_key: str, default_unit: object) -> object:
        return setup.get(unit_key, default_unit) if key in setup else default_unit

    base_units = setup.get("UNITS", "mm")
    normalize_length_units(base_units, "DETECTOR_SETUP.UNITS")

    distance = to_mm(
        setup["DISTANCE"],
        _length("DISTANCE", "DISTANCE_UNITS", base_units),
        "DETECTOR_SETUP.DISTANCE",
    )

    frame = tuple(int(value) for value in tuple(frame_shape)[:2])
    binning = tuple(int(value) for value in setup.get("BINNING", (1, 1)))
    order = str(setup.get("FRAME_AXIS_ORDER", FRAME_AXIS_DIRECT)).strip().lower()
    if order not in FRAME_AXIS_ORDERS:
        raise ValueError(
            f"DETECTOR_SETUP.FRAME_AXIS_ORDER must be one of {FRAME_AXIS_ORDERS}."
        )

    # The acquired frame is in read-out order; the calibration is in detector
    # (direction1, direction2) order. Undo the swap before reasoning about ROI.
    detector_frame = (frame[1], frame[0]) if order == FRAME_AXIS_SWAPPED else frame

    if "DETECTOR_SHAPE" in setup:
        shape = tuple(int(value) for value in setup["DETECTOR_SHAPE"])
    else:
        # Legacy: the frame is assumed to be the whole unbinned detector.
        shape = (detector_frame[0] * binning[0], detector_frame[1] * binning[1])

    roi = setup.get("ROI")
    roi = tuple(int(value) for value in roi) if roi is not None else (0, shape[0], 0, shape[1])

    if "PIXEL_SIZE" in setup:
        raw = tuple(float(value) for value in setup["PIXEL_SIZE"])
        units = _length("PIXEL_SIZE", "PIXEL_SIZE_UNITS", base_units)
        pixel_width = (
            to_mm(raw[0], units, "DETECTOR_SETUP.PIXEL_SIZE"),
            to_mm(raw[1], units, "DETECTOR_SETUP.PIXEL_SIZE"),
        )
    else:
        units = _length("SIZE", "SIZE_UNITS", base_units)
        size_mm = tuple(
            to_mm(value, units, "DETECTOR_SETUP.SIZE") for value in tuple(setup["SIZE"])[:2]
        )
        pixel_width = pixel_width_from_size(
            size_mm, shape, binning=binning, roi=roi if "ROI" in setup else None
        )

    angle_units = setup.get("ANGLE_UNITS", "deg")
    tilts = {
        name.lower(): to_deg(setup[name], angle_units, f"DETECTOR_SETUP.{name}")
        for name in ("DETROT", "TILT", "TILTAZIMUTH")
        if name in setup
    }

    model = DetectorModel(
        pixel_direction_1=str(setup["PIXEL_DIRECTION_1"]),
        pixel_direction_2=str(setup["PIXEL_DIRECTION_2"]),
        center_channel=tuple(float(value) for value in tuple(setup["CENTER_CHANNEL_PIXEL"])[:2]),
        shape=shape,
        pixel_width=pixel_width,
        distance=distance,
        roi=roi,
        binning=binning,
        frame_axis_order=order,
        **tilts,
    )
    if verify_frame_shape:
        model.require_frame_shape(frame)
    return model


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
    qx, qy, qz = geometry.hxrd.Ang2Q.area(
        *sample_angles,
        *detector_angles,
        UB=ub,
        deg=True,
    )
    # Ang2Q.area returns (direction1, direction2); hand back the acquired frame
    # layout so callers can index Q, intensity and mask identically.
    orient = geometry.model.detector.orient_frame_array
    return orient(qx), orient(qy), orient(qz)
