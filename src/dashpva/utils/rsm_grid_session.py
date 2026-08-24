# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""State machine and control surface for a live gridded accumulation.

Qt-free and pvaccess-free on purpose: the pvapy processor is a thin adapter
over this, so the control logic -- which is where the mistakes hide -- can be
tested without a live stream or an EPICS network.

States::

    idle --start--> running --stop--> stopped --save--> stopped
                       ^                  |
                       +------ start -----+
    (clear returns running/stopped to idle)

Configuration is *locked* at start: geometry fingerprint, mask, monitor,
bounds and resolution. Changing any of them mid-accumulation would mean the
voxels already accumulated and the ones still arriving describe different
things, and the volume could no longer be interpreted. A change therefore
requires an explicit clear/restart rather than silently taking effect.

Saving is only allowed from ``stopped``. Writing while frames are still
arriving would capture a torn accumulator, and the resulting file would be
neither the volume you watched nor a complete one.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

import numpy as np

from dashpva.utils.rsm_gridder import (
    GridMemoryEstimate,
    RSMMergeError,
    ensure_memory_available,
    estimate_grid_memory,
)
from dashpva.utils.rsm_live_grid import (
    DEFAULT_PREVIEW_BUDGET_BYTES,
    GridBoundsSpec,
    LiveVolumeAccumulator,
    PreviewPayload,
)
from dashpva.utils.volume_io import save_volume

__all__ = [
    "GridSessionState",
    "LiveGridSession",
    "SessionError",
    "geometry_fingerprint",
]


class GridSessionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    SAVING = "saving"


class SessionError(RuntimeError):
    """An operation was requested that the current state does not permit."""


def geometry_fingerprint(*parts: Any) -> str:
    """Stable short digest of everything that must not change mid-run.

    Compared rather than trusted: if the profile is edited while a run is in
    progress, the mismatch is what stops the accumulation instead of the two
    halves quietly disagreeing.
    """
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class _LockedConfig:
    bounds: GridBoundsSpec
    fingerprint: str
    monitor_name: Optional[str]
    mask_signature: Optional[str]


@dataclass
class _SessionCounters:
    frames_seen_running: int = 0
    frames_rejected_binding: int = 0
    frames_rejected_processing: int = 0


class LiveGridSession:
    """Owns one accumulator and the rules for driving it."""

    def __init__(
        self,
        *,
        output_dir: str,
        preview_budget_bytes: int = DEFAULT_PREVIEW_BUDGET_BYTES,
        save_fn: Callable[..., bool] = save_volume,
        memory_guard: Callable[[GridMemoryEstimate], None] = ensure_memory_available,
    ) -> None:
        self.output_dir = output_dir
        self.preview_budget_bytes = preview_budget_bytes
        self._save_fn = save_fn
        self._memory_guard = memory_guard
        self._lock = threading.RLock()
        self._state = GridSessionState.IDLE
        self._accumulator: Optional[LiveVolumeAccumulator] = None
        self._locked: Optional[_LockedConfig] = None
        self._incomplete = False
        self._last_error: str = ""
        self._save_counter = 0
        self._counters = _SessionCounters()
        self._memory_estimate: Optional[GridMemoryEstimate] = None

    # -- introspection -----------------------------------------------------

    @property
    def state(self) -> GridSessionState:
        return self._state

    @property
    def accumulator(self) -> Optional[LiveVolumeAccumulator]:
        return self._accumulator

    @property
    def incomplete(self) -> bool:
        """True once any frame was dropped or a frame-id gap was seen."""
        return self._incomplete

    def get_state(self) -> dict:
        with self._lock:
            accepted = 0
            payload: dict[str, Any] = {
                "state": self._state.value,
                "incomplete": self._incomplete,
                "last_error": self._last_error,
                "frames_seen_running": self._counters.frames_seen_running,
                "frames_rejected_binding": self._counters.frames_rejected_binding,
                "frames_rejected_processing": (
                    self._counters.frames_rejected_processing
                ),
            }
            if self._locked is not None:
                payload.update(
                    grid_shape=list(self._locked.bounds.shape),
                    grid_bounds=[
                        self._locked.bounds.hmin,
                        self._locked.bounds.hmax,
                        self._locked.bounds.kmin,
                        self._locked.bounds.kmax,
                        self._locked.bounds.lmin,
                        self._locked.bounds.lmax,
                    ],
                    geometry_fingerprint=self._locked.fingerprint,
                    monitor=self._locked.monitor_name or "",
                )
            if self._memory_estimate is not None:
                payload["memory_peak_bytes"] = self._memory_estimate.peak_bytes
            if self._accumulator is not None:
                counters = self._accumulator.counters
                accepted = counters.frames_accepted
                payload.update(
                    frames_accepted=accepted,
                    points_binned=counters.points_binned,
                    points_out_of_range=counters.points_out_of_range,
                    points_nonfinite=counters.points_nonfinite,
                    points_masked=counters.points_masked,
                )
            rejected = (
                self._counters.frames_rejected_binding
                + self._counters.frames_rejected_processing
            )
            payload["frames_rejected"] = rejected
            payload["accounting_consistent"] = (
                self._counters.frames_seen_running == accepted + rejected
            )
            return payload

    # -- transitions -------------------------------------------------------

    def start(
        self,
        bounds: GridBoundsSpec,
        *,
        fingerprint: str,
        monitor_name: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        mask_signature: Optional[str] = None,
    ) -> dict:
        """Lock configuration and begin accepting frames."""
        with self._lock:
            if self._state is GridSessionState.SAVING:
                raise SessionError("Cannot start while a save is in progress.")
            if self._state is GridSessionState.RUNNING:
                raise SessionError("The live grid is already running.")

            locked = _LockedConfig(
                bounds=bounds,
                fingerprint=fingerprint,
                monitor_name=monitor_name,
                mask_signature=mask_signature,
            )
            if self._state is GridSessionState.STOPPED and self._locked is not None:
                if self._config_changed(locked):
                    raise SessionError(
                        "Configuration changed since this run started (geometry, "
                        "bounds, mask, or monitor). Clear the grid before starting "
                        "again -- resuming would mix incompatible data."
                    )
                self._state = GridSessionState.RUNNING
                return self.get_state()

            estimate = estimate_grid_memory(*bounds.shape)
            try:
                self._memory_guard(estimate)
            except RSMMergeError as exc:
                raise SessionError(str(exc)) from exc
            self._memory_estimate = estimate
            self._accumulator = LiveVolumeAccumulator(
                bounds,
                monitor_name=monitor_name,
                mask=mask,
                geometry_fingerprint=fingerprint,
                preview_budget_bytes=self.preview_budget_bytes,
            )
            self._locked = locked
            self._counters = _SessionCounters()
            self._incomplete = False
            self._last_error = ""
            self._state = GridSessionState.RUNNING
            return self.get_state()

    def validate_geometry(self, fingerprint: str) -> None:
        """Latch the first accepted frame geometry and fail closed on change."""
        with self._lock:
            if self._state is not GridSessionState.RUNNING or self._locked is None:
                return
            if not self._locked.fingerprint:
                self._locked.fingerprint = fingerprint
                if self._accumulator is not None:
                    self._accumulator.geometry_fingerprint = fingerprint
                return
            if self._locked.fingerprint != fingerprint:
                self._incomplete = True
                self._last_error = (
                    "Static RSM geometry changed during accumulation. The grid "
                    "was stopped before incompatible frames could be mixed. Clear "
                    "and restart after the geometry is stable."
                )
                self._state = GridSessionState.STOPPED
                raise SessionError(self._last_error)

    def _config_changed(self, candidate: _LockedConfig) -> bool:
        current = self._locked
        assert current is not None
        return (
            current.bounds != candidate.bounds
            or current.fingerprint != candidate.fingerprint
            or current.monitor_name != candidate.monitor_name
            or current.mask_signature != candidate.mask_signature
        )

    def stop(self) -> dict:
        """Stop accepting frames. The accumulated volume is retained."""
        with self._lock:
            if self._state is GridSessionState.SAVING:
                raise SessionError("Cannot stop while a save is in progress.")
            if self._state is GridSessionState.IDLE:
                raise SessionError("Nothing is running.")
            self._state = GridSessionState.STOPPED
            return self.get_state()

    def clear(self) -> dict:
        """Discard the accumulation and return to idle."""
        with self._lock:
            if self._state is GridSessionState.SAVING:
                raise SessionError("Cannot clear while a save is in progress.")
            self._accumulator = None
            self._locked = None
            self._counters = _SessionCounters()
            self._memory_estimate = None
            self._incomplete = False
            self._last_error = ""
            self._state = GridSessionState.IDLE
            return self.get_state()

    # -- frames ------------------------------------------------------------

    def add_frame(
        self,
        qx: np.ndarray,
        qy: np.ndarray,
        qz: np.ndarray,
        intensity: np.ndarray,
        *,
        monitor: Optional[float] = None,
        followed_gap: bool = False,
    ) -> int:
        """Grid one frame. Frames arriving outside RUNNING are ignored.

        Ignoring rather than raising: frames keep arriving from the detector
        after the user presses Stop, and that is normal, not an error.
        """
        with self._lock:
            if self._state is not GridSessionState.RUNNING or self._accumulator is None:
                return 0
            if followed_gap:
                self._incomplete = True
            try:
                binned = self._accumulator.add_frame(
                    qx, qy, qz, intensity, monitor=monitor
                )
            except ValueError as exc:
                self.note_processing_rejection(str(exc))
                return 0
            if self._accumulator.counters.points_out_of_range:
                self._incomplete = True
            return binned

    def note_frame_seen(self) -> None:
        with self._lock:
            self._counters.frames_seen_running += 1

    def note_binding_rejection(self, message: str = "") -> None:
        with self._lock:
            self._counters.frames_rejected_binding += 1
            self._incomplete = True
            if message:
                self._last_error = message

    def note_processing_rejection(self, message: str = "") -> None:
        with self._lock:
            self._counters.frames_rejected_processing += 1
            self._incomplete = True
            if message:
                self._last_error = message

    def note_rejected_frame(self) -> None:
        """Record that upstream refused a frame, marking the preview partial."""
        self.note_processing_rejection()

    # -- output ------------------------------------------------------------

    def preview(self) -> Optional[PreviewPayload]:
        with self._lock:
            if self._accumulator is None:
                return None
            return self._accumulator.preview()

    def save(self, filename: str, *, extra_metadata: Optional[Mapping] = None) -> dict:
        """Write the full-resolution volume and coverage under ``output_dir``.

        Only from STOPPED: saving mid-flight would capture a torn accumulator.
        The path is confined to ``output_dir`` -- this runs in a consumer
        process reachable over the network, so an arbitrary caller-supplied
        path would be a write-anywhere primitive.
        """
        import os

        with self._lock:
            if self._state is not GridSessionState.STOPPED:
                raise SessionError(
                    f"Save is only allowed from '{GridSessionState.STOPPED.value}'; "
                    f"currently '{self._state.value}'. Stop the accumulation first."
                )
            if self._accumulator is None:
                raise SessionError("Nothing has been accumulated.")

            base = os.path.basename(str(filename)).strip()
            if not base or base in (".", ".."):
                raise SessionError(f"Invalid output filename {filename!r}.")
            if not base.endswith((".h5", ".hdf5")):
                base += ".h5"
            target = os.path.join(self.output_dir, base)
            if os.path.exists(target):
                raise SessionError(
                    f"Refusing to overwrite existing file '{target}'. "
                    "Choose another name."
                )

            self._state = GridSessionState.SAVING
            metadata = self._accumulator.to_metadata(
                {
                    "live_incomplete": bool(self._incomplete),
                    **(dict(extra_metadata) if extra_metadata else {}),
                }
            )
            volume = self._accumulator.mean
            coverage = self._accumulator.coverage

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self._save_fn(
                target,
                volume,
                coverage=coverage,
                metadata=metadata,
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._state = GridSessionState.STOPPED
            raise
        with self._lock:
            self._save_counter += 1
            self._state = GridSessionState.STOPPED
            result = self.get_state()
            result["saved_path"] = target
            result["job_id"] = f"save-{self._save_counter}"
            return result
