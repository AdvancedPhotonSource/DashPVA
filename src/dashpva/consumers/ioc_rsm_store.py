# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""IOC RSM Parameter axis/geometry config stored in the app's central profile.

The IOC simulator's UB/beam/detector defaults live under a top-level
``[IOC_RSM_PARAMETER]`` table, same as the consumers and area detector reader
(Workflow editor -> ``dashpva.db`` / TOML, surfaced via ``settings`` /
``ConfigSource``)::

    [IOC_RSM_PARAMETER]   # == IOCRSMConfig.to_dict()

The axis list itself is geometry that other HKL/RSM consumers already read, so
it is stored alongside them under the profile's ``[HKL]`` table instead::

    [HKL]
    AXES = [ {name, source_pv, axis_number, direction}, ... ]

``load_config`` reads the resolved axis list from ``settings.HKL_AXES`` and the
active profile's display name from ``settings.PROFILE_NAME`` (both populated by
``settings.reload()``), rather than this module doing its own separate
``ConfigSource``/``DatabaseInterface`` lookups — so every HKL/RSM consumer and
this tool agree on the same axis list and profile name. ``save_config`` still
writes directly via ``ConfigSource`` (settings.py has no generic nested-array
save helper), then calls ``settings.reload()`` so ``settings.HKL_AXES``
reflects the change immediately.

When a profile has no ``HKL.AXES`` entry yet (or no profile/database is
available), callers fall back to :data:`DEFAULT_AXES` — 4 generic, unconfigured
axes with no source PV, never a specific beamline's geometry.

This module is pure (no Qt) so it can be unit-tested against a temporary TOML file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Top-level table in the profile config that holds UB/beam/detector geometry.
SECTION = "IOC_RSM_PARAMETER"

# Top-level table (shared with other HKL/RSM consumers) that holds the axis list.
AXES_SECTION = "HKL"
AXES_KEY = "AXES"

# Generic, unconfigured fallback template — used only when a profile has no
# HKL.AXES entry yet. Deliberately not a specific beamline's axis layout.
DEFAULT_AXES = [
    {'name': f'Axis{i}', 'source_pv': '', 'axis_number': i, 'direction': 'x+', 'role': 'sample'}
    for i in range(1, 5)
]

DEFAULT_ENERGY_SOURCE_PV = ''
DEFAULT_UB            = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_PRIMARY_BEAM  = [0, 1, 0]
DEFAULT_INPLANE_REF   = [0, 1, 0]
DEFAULT_SAMPLE_NORMAL = [0, 0, 1]
DEFAULT_DETECTOR = {
    'pixel_dir1': 'z-',
    'pixel_dir2': 'x-',
    'center':     [300, 300],
    'size':       [28.38, 28.38],
    'distance':   400.644,
    'units':      'mm',
}


@dataclass
class AxisSpec:
    """One motor axis: name, its live position PV, static geometry (axis
    number/direction), and which HKL circle group (sample/detector) it
    belongs to. This is the single representation shared by the IOC
    simulator and every real HKL/RSM consumer (see settings.HKL_SAMPLE_CIRCLES
    / HKL_DETECTOR_CIRCLES)."""

    name: str
    source_pv: str = ""
    axis_number: int = 1
    direction: str = "x+"
    role: str = "sample"


def _axes_from_raw(raw: Any) -> Optional[List[AxisSpec]]:
    """Parse a list of axis dicts (as read from TOML/DB) into AxisSpecs, or
    None if raw is not a non-empty list."""
    if not isinstance(raw, list) or not raw:
        return None
    return [
        AxisSpec(
            name=a.get("name", ""),
            source_pv=a.get("source_pv", ""),
            axis_number=int(a.get("axis_number", 1)),
            direction=a.get("direction", "x+"),
            role=a.get("role", "sample"),
        )
        for a in raw
    ]


@dataclass
class IOCRSMConfig:
    """Full axis list + UB/beam/detector geometry, as edited in the GUI."""

    axes: List[AxisSpec] = field(default_factory=list)
    energy_source_pv: str = DEFAULT_ENERGY_SOURCE_PV
    ub_matrix: List[float] = field(default_factory=lambda: list(DEFAULT_UB))
    primary_beam: List[float] = field(default_factory=lambda: list(DEFAULT_PRIMARY_BEAM))
    inplane_ref: List[float] = field(default_factory=lambda: list(DEFAULT_INPLANE_REF))
    sample_normal: List[float] = field(default_factory=lambda: list(DEFAULT_SAMPLE_NORMAL))
    detector: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_DETECTOR))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the UB/beam/detector geometry — everything except the axis
        list, which is stored separately under HKL.AXES (see save_config)."""
        return {
            "ENERGY_SOURCE_PV": self.energy_source_pv,
            "UB_MATRIX": list(self.ub_matrix),
            "PRIMARY_BEAM": list(self.primary_beam),
            "INPLANE_REF": list(self.inplane_ref),
            "SAMPLE_NORMAL": list(self.sample_normal),
            "DETECTOR": dict(self.detector),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], axes: Optional[List[AxisSpec]] = None) -> "IOCRSMConfig":
        data = data or {}
        axes = axes if axes else [AxisSpec(**a) for a in DEFAULT_AXES]
        return cls(
            axes=axes,
            energy_source_pv=data.get("ENERGY_SOURCE_PV", DEFAULT_ENERGY_SOURCE_PV),
            ub_matrix=list(data.get("UB_MATRIX", DEFAULT_UB)),
            primary_beam=list(data.get("PRIMARY_BEAM", DEFAULT_PRIMARY_BEAM)),
            inplane_ref=list(data.get("INPLANE_REF", DEFAULT_INPLANE_REF)),
            sample_normal=list(data.get("SAMPLE_NORMAL", DEFAULT_SAMPLE_NORMAL)),
            detector=dict(data.get("DETECTOR", DEFAULT_DETECTOR)),
        )

    def to_full_dict(self) -> Dict[str, Any]:
        """Serialize everything, including axes — used only to hand this config
        off to the headless IOC subprocess via a one-shot temp file (see
        ioc_rsm_parameter._launch_ioc_proc), not for profile persistence."""
        d = self.to_dict()
        d["AXES"] = [vars(a).copy() for a in self.axes]
        return d

    @classmethod
    def from_full_dict(cls, data: Dict[str, Any]) -> "IOCRSMConfig":
        data = data or {}
        return cls.from_dict(data, axes=_axes_from_raw(data.get("AXES")))


def default_config() -> IOCRSMConfig:
    """4 generic, unconfigured axes — used whenever no profile config is available."""
    return IOCRSMConfig(axes=[AxisSpec(**a) for a in DEFAULT_AXES])


def active_source() -> Tuple[Optional[Any], str]:
    """Resolve the app's active config source for reading/writing this config.

    Returns ``(source, label)`` where ``source`` is a ``ConfigSource`` bound to the
    currently-selected profile (or active TOML file), and ``label`` is
    ``settings.PROFILE_NAME`` — a short human name for display. Returns
    ``(None, "")`` when no profile/DB is resolvable, so the caller falls back to
    the generic default template.
    """
    try:
        from dashpva import settings
        from dashpva.utils.config.source import ConfigSource
    except Exception:  # noqa: BLE001 - config stack unavailable -> fallback
        return None, ""

    # Re-resolve the locator so we pick up a profile selected in another window or
    # process (this tool runs as its own process with a cached settings.LOCATOR
    # otherwise).
    try:
        settings.reload()
    except Exception:  # noqa: BLE001
        pass

    locator = getattr(settings, "LOCATOR", None)
    try:
        src = ConfigSource(locator)
    except Exception:  # noqa: BLE001
        return None, ""

    if getattr(src, "source_type", "none") == "none":
        return None, ""
    return src, getattr(settings, "PROFILE_NAME", None) or "profile"


def load_config(src: Any) -> IOCRSMConfig:
    """Read ``IOC_RSM_PARAMETER`` from ``src``; axes come from
    ``settings.HKL_AXES`` (already re-resolved by ``active_source()``'s
    ``settings.reload()`` call), so this module and every other HKL/RSM
    consumer agree on the same axis list. Falls back to the generic 4-axis
    template if absent or if ``src`` is ``None`` (offline / no active
    profile). Never attempts a PV connection — this is pure config I/O."""
    data: Dict[str, Any] = {}
    if src is not None:
        try:
            data = (src.load() or {}).get(SECTION) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load IOC RSM parameter config: %s", exc)
    try:
        from dashpva import settings
        axes = _axes_from_raw(settings.HKL_AXES)
    except Exception:  # noqa: BLE001
        axes = None
    return IOCRSMConfig.from_dict(data, axes=axes)


def save_config(src: Any, cfg: IOCRSMConfig) -> bool:
    """Write ``cfg``: UB/beam/detector to ``IOC_RSM_PARAMETER``, axes to
    ``HKL.AXES``. Load-modify-save on the HKL table so other HKL keys already
    in the profile (SPEC, DETECTOR_SETUP, ...) are preserved. Reloads
    ``settings`` afterward so ``settings.HKL_AXES`` reflects the change
    immediately."""
    if src is None:
        return False
    try:
        full = src.load() or {}
        hkl = dict(full.get(AXES_SECTION) or {})
        hkl[AXES_KEY] = [vars(a).copy() for a in cfg.axes]
        ok = bool(src.save({SECTION: cfg.to_dict(), AXES_SECTION: hkl}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save IOC RSM parameter config: %s", exc)
        return False
    if ok:
        try:
            from dashpva import settings
            settings.reload()
        except Exception:  # noqa: BLE001
            pass
    return ok
