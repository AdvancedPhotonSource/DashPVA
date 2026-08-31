# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Canonical unit normalization for RSM geometry.

DashPVA's internal geometry is expressed in exactly three units: **eV** for
photon energy, **mm** for every length, and **degrees** for every angle. Values
arriving from a profile, an IOC record, or an HDF5 file are converted here, at
the boundary, so nothing downstream has to guess.

This replaces two silent assumptions that predate the canonical geometry model:
a blind ``energy * 1000`` (correct only when the source publishes keV) and an
unread ``DETECTOR_SETUP.UNITS`` key (so millimetres were assumed regardless of
what the profile declared). Both produced a plausible-looking but wrong Q with
no error, which is the failure mode the canonical model exists to remove.

Unknown units raise rather than falling back to a default -- a mis-declared
unit is a calibration error, and silently guessing would reintroduce exactly
the class of bug this module removes.
"""

from __future__ import annotations

import math
from typing import Final, Mapping

__all__ = [
    "ANGLE_TO_DEG",
    "ENERGY_TO_EV",
    "LENGTH_TO_MM",
    "normalize_angle_units",
    "normalize_energy_units",
    "normalize_length_units",
    "to_deg",
    "to_eV",
    "to_mm",
]

LENGTH_TO_MM: Final[Mapping[str, float]] = {
    "m": 1.0e3,
    "meter": 1.0e3,
    "meters": 1.0e3,
    "cm": 1.0e1,
    "centimeter": 1.0e1,
    "centimeters": 1.0e1,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "um": 1.0e-3,
    "µm": 1.0e-3,
    "micron": 1.0e-3,
    "microns": 1.0e-3,
    "micrometer": 1.0e-3,
    "micrometers": 1.0e-3,
    "nm": 1.0e-6,
    "nanometer": 1.0e-6,
    "nanometers": 1.0e-6,
}

ENERGY_TO_EV: Final[Mapping[str, float]] = {
    "ev": 1.0,
    "electronvolt": 1.0,
    "electronvolts": 1.0,
    "kev": 1.0e3,
    "kiloelectronvolt": 1.0e3,
    "kiloelectronvolts": 1.0e3,
    "mev": 1.0e6,
    "megaelectronvolt": 1.0e6,
    "megaelectronvolts": 1.0e6,
}

ANGLE_TO_DEG: Final[Mapping[str, float]] = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "rad": 180.0 / math.pi,
    "radian": 180.0 / math.pi,
    "radians": 180.0 / math.pi,
}


def _normalize(units: object, table: Mapping[str, float], kind: str, label: str) -> str:
    """Return the lowercase key for ``units`` or raise naming what was allowed."""
    key = str(units).strip().lower()
    if key not in table:
        allowed = ", ".join(sorted({name for name in table}))
        raise ValueError(
            f"{label} declares unsupported {kind} units {units!r}. "
            f"Supported {kind} units: {allowed}."
        )
    return key


def normalize_length_units(units: object, label: str = "Value") -> str:
    """Validate a length unit without converting a value."""
    return _normalize(units, LENGTH_TO_MM, "length", label)


def normalize_energy_units(units: object, label: str = "Value") -> str:
    """Validate an energy unit without converting a value."""
    return _normalize(units, ENERGY_TO_EV, "energy", label)


def normalize_angle_units(units: object, label: str = "Value") -> str:
    """Validate an angle unit without converting a value."""
    return _normalize(units, ANGLE_TO_DEG, "angle", label)


def _convert(value: object, factor: float, label: str) -> float:
    try:
        magnitude = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a real number, got {value!r}.") from exc
    if not math.isfinite(magnitude):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return magnitude * factor


def to_mm(value: object, units: object, label: str = "Length") -> float:
    """Convert a length to millimetres, the canonical internal length unit."""
    return _convert(value, LENGTH_TO_MM[normalize_length_units(units, label)], label)


def to_eV(value: object, units: object, label: str = "Photon energy") -> float:
    """Convert a photon energy to eV, the unit xrayutilities' ``en=`` expects."""
    return _convert(value, ENERGY_TO_EV[normalize_energy_units(units, label)], label)


def to_deg(value: object, units: object, label: str = "Angle") -> float:
    """Convert an angle to degrees, matching the ``deg=True`` conversion path."""
    return _convert(value, ANGLE_TO_DEG[normalize_angle_units(units, label)], label)
