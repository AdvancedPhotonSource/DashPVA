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

"""Stateful live RSM grid consumer using pvapy control/status records.

The full grid stays in this process. The standard pvapy analysis control
channel carries namespaced ``rsm_grid`` commands and ``userStats.rsm_grid`` on
the standard status channel carries a budget-capped float32 preview.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pvaccess as pva

import dashpva.settings as app_settings
from dashpva.consumers.hpc.analysis.hpc_rsm_consumer import HpcRsmProcessor
from dashpva.utils.metadata_binding import (
    METADATA_TIMESTAMP_ATTRIBUTE_PREFIX,
    MetadataBinder,
    classify_hkl_channels,
)
from dashpva.utils.rsm_grid_session import (
    GridSessionState,
    LiveGridSession,
    SessionError,
    geometry_fingerprint,
)
from dashpva.utils.rsm_grid_transport import (
    RSM_GRID_NAMESPACE,
    RSM_GRID_PROTOCOL_VERSION,
    GridTransportError,
    parse_command_envelope,
)
from dashpva.utils.rsm_live_grid import GridBoundsSpec, PreviewPayload


class HpcRsmGridProcessor(HpcRsmProcessor):
    """Compute Q and accumulate one fixed grid without publishing Q arrays."""

    def __init__(self, configDict={}):
        self._grid_lock = threading.RLock()
        self.session: Optional[LiveGridSession] = None
        self.binder: Optional[MetadataBinder] = None
        self._preview_interval = app_settings.RSM_GRID_PREVIEW_INTERVAL_SECONDS
        self._last_preview = 0.0
        self._preview_publishes = 0
        self._latest_preview: Optional[PreviewPayload] = None
        self._grid_defaults: dict[str, Any] = {}
        self._ack_request_id = ""
        self._ack_command = ""
        self._command_error = ""
        self._runtime_error = ""
        self._saved_path = ""
        self._save_state = "idle"
        self._save_request_id = ""
        self._save_error = ""
        self._save_thread: Optional[threading.Thread] = None
        self._estimating = False
        self._estimate_extrema: Optional[list[float]] = None
        self._estimated_bounds: list[float] = []
        self._estimate_fingerprint = ""
        super().__init__(configDict)

    # -- configuration/control -------------------------------------------

    def configure(self, configDict):
        config_dict = dict(configDict or {})
        grid_command = config_dict.pop(RSM_GRID_NAMESPACE, None)
        with self._grid_lock:
            first_configure = not hasattr(self, "config")
            if first_configure or config_dict:
                if (
                    not first_configure
                    and self.session is not None
                    and self.session.state is GridSessionState.RUNNING
                ):
                    raise RuntimeError(
                        "Cannot reconfigure the RSM processor while a grid is running."
                    )
                super().configure(config_dict)
                self._configure_grid(config_dict)
            if grid_command is not None:
                self._handle_control(grid_command)

    def _configure_grid(self, config_dict: Mapping[str, Any]) -> None:
        grid_cfg = dict(self.config.get("RSM_GRID", {}) or {})
        grid_cfg.update(
            {
                key: value
                for key, value in config_dict.items()
                if key
                in {
                    "HMIN",
                    "HMAX",
                    "KMIN",
                    "KMAX",
                    "LMIN",
                    "LMAX",
                    "NX",
                    "NY",
                    "NZ",
                    "MONITOR",
                }
            }
        )
        self._grid_defaults = grid_cfg
        self._preview_interval = float(
            config_dict.get(
                "previewInterval", app_settings.RSM_GRID_PREVIEW_INTERVAL_SECONDS
            )
        )
        monitor = grid_cfg.get("MONITOR") or None
        self.binder = MetadataBinder(
            classify_hkl_channels(self.hkl_config, monitor_channel=monitor),
            monitor_channel=monitor,
            max_age_seconds=float(
                config_dict.get(
                    "metadataMaxAgeSeconds",
                    app_settings.RSM_GRID_METADATA_MAX_AGE_SECONDS,
                )
            ),
        )
        if self.session is None:
            self.session = LiveGridSession(
                output_dir=str(getattr(app_settings, "OUTPUT_PATH", "./outputs")),
                preview_budget_bytes=app_settings.RSM_GRID_PREVIEW_BUDGET_BYTES,
            )

    def _bounds_from(self, source: Mapping[str, Any]) -> GridBoundsSpec:
        def value(key: str, default: Any) -> Any:
            return source.get(key, self._grid_defaults.get(key, default))

        return GridBoundsSpec(
            hmin=float(value("HMIN", 0.0)),
            hmax=float(value("HMAX", 1.0)),
            kmin=float(value("KMIN", 0.0)),
            kmax=float(value("KMAX", 1.0)),
            lmin=float(value("LMIN", 0.0)),
            lmax=float(value("LMAX", 1.0)),
            nx=int(value("NX", 128)),
            ny=int(value("NY", 128)),
            nz=int(value("NZ", 128)),
        )

    def _handle_control(self, value: Any) -> None:
        request_id = ""
        command = ""
        try:
            envelope = parse_command_envelope(value)
            request_id = envelope["request_id"]
            command = envelope["command"]
            result = self._execute_command(
                command, envelope["payload"], request_id=request_id
            )
            self._command_error = ""
            if result.get("saved_path"):
                self._saved_path = str(result["saved_path"])
        except (GridTransportError, SessionError, ValueError, OSError) as exc:
            if isinstance(value, Mapping):
                request_id = request_id or str(value.get("request_id", ""))
                command = command or str(value.get("command", ""))
            self._command_error = str(exc)
        except Exception as exc:
            self._command_error = f"{type(exc).__name__}: {exc}"
        self._ack_request_id = request_id
        self._ack_command = command

    def _execute_command(
        self,
        command: str,
        payload: Mapping[str, Any],
        *,
        request_id: str = "",
    ) -> dict:
        assert self.session is not None
        assert self.binder is not None
        if self._save_state == "running":
            raise SessionError("A live-grid save is in progress; wait for completion.")
        if command == "estimate_start":
            if self.session.state is GridSessionState.RUNNING:
                raise SessionError("Stop the grid before estimating new bounds.")
            self.binder.reset(keep_static=True)
            self._estimating = True
            self._estimate_extrema = None
            self._estimated_bounds = []
            self._estimate_fingerprint = ""
            self._runtime_error = ""
            return self.session.get_state()
        if command == "estimate_finish":
            if self._estimate_extrema is None:
                raise SessionError(
                    "No valid frames were observed while estimating bounds."
                )
            self._estimating = False
            self._estimated_bounds = self._padded_extrema(self._estimate_extrema)
            return self.session.get_state()
        if command == "start":
            self._estimating = False
            self._runtime_error = ""
            if self.session.state is GridSessionState.IDLE:
                self.binder.reset(keep_static=True)
            current = self.session.get_state()
            fingerprint = (
                str(current.get("geometry_fingerprint", ""))
                if self.session.state is GridSessionState.STOPPED
                else self._estimate_fingerprint
            )
            return self.session.start(
                self._bounds_from(payload),
                fingerprint=fingerprint,
                monitor_name=self._grid_defaults.get("MONITOR") or None,
            )
        if command == "stop":
            return self.session.stop()
        if command == "clear":
            self.binder.reset(keep_static=True)
            self._estimating = False
            self._estimate_extrema = None
            self._estimated_bounds = []
            self._estimate_fingerprint = ""
            self._latest_preview = None
            self._saved_path = ""
            self._save_state = "idle"
            self._save_request_id = ""
            self._save_error = ""
            self._runtime_error = ""
            return self.session.clear()
        if command == "save":
            if self._save_state == "running":
                raise SessionError("A live-grid save is already in progress.")
            if self.session.state is not GridSessionState.STOPPED:
                raise SessionError(
                    "Save is only allowed after the accumulation has stopped."
                )
            filename = str(payload.get("filename", "live_grid.h5"))
            self._save_state = "running"
            self._save_request_id = request_id
            self._save_error = ""
            self._saved_path = ""
            self._save_thread = threading.Thread(
                target=self._save_worker,
                args=(filename, request_id),
                name="DashPVA-RSM-save",
                daemon=True,
            )
            self._save_thread.start()
            return self.session.get_state()
        raise GridTransportError(f"Unknown live-grid command {command!r}.")

    def _save_worker(self, filename: str, request_id: str) -> None:
        try:
            assert self.session is not None
            result = self.session.save(filename)
        except Exception as exc:
            with self._grid_lock:
                self._save_error = str(exc)
                self._save_state = "error"
            return
        with self._grid_lock:
            self._saved_path = str(result["saved_path"])
            self._save_request_id = request_id
            self._save_state = "complete"

    # -- frame path --------------------------------------------------------

    @staticmethod
    def _current_attributes(pv_object) -> dict[str, Any]:
        current: dict[str, Any] = {}
        for attribute in pv_object["attribute"]:
            try:
                current[str(attribute["name"])] = attribute["value"][0]["value"]
            except Exception:
                continue
        return current

    @staticmethod
    def _metadata_timestamps(attributes: Mapping[str, Any]) -> dict[str, float]:
        timestamps: dict[str, float] = {}
        for name, value in attributes.items():
            if not name.startswith(METADATA_TIMESTAMP_ATTRIBUTE_PREFIX):
                continue
            channel = name.removeprefix(METADATA_TIMESTAMP_ATTRIBUTE_PREFIX)
            try:
                timestamps[channel] = float(value)
            except (TypeError, ValueError):
                continue
        return timestamps

    @staticmethod
    def _frame_timestamp(pv_object) -> Optional[float]:
        try:
            stamp = pv_object["timeStamp"]
            return float(stamp["secondsPastEpoch"]) + float(stamp["nanoseconds"]) * 1e-9
        except Exception:
            return None

    @staticmethod
    def _frame_id(pv_object) -> Optional[int]:
        try:
            return int(pv_object["uniqueId"])
        except Exception:
            return None

    def _frame_geometry_fingerprint(
        self, values: Mapping[str, Any], shape: tuple[int, ...]
    ) -> str:
        assert self.binder is not None
        static_values = [(name, values.get(name)) for name in self.binder.static_names]
        canonical = self.config.get("IOC_RSM_PARAMETER", {}) or {}
        return geometry_fingerprint(static_values, canonical, shape)

    @staticmethod
    def _padded_extrema(extrema: list[float]) -> list[float]:
        result: list[float] = []
        for low, high in zip(extrema[::2], extrema[1::2]):
            span = high - low
            margin = span * 0.05 if span > 0 else 1e-6
            result.extend((low - margin, high + margin))
        return result

    def _observe_bounds(self, qxyz: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        extrema = [
            float(np.nanmin(qxyz[0])),
            float(np.nanmax(qxyz[0])),
            float(np.nanmin(qxyz[1])),
            float(np.nanmax(qxyz[1])),
            float(np.nanmin(qxyz[2])),
            float(np.nanmax(qxyz[2])),
        ]
        if self._estimate_extrema is None:
            self._estimate_extrema = extrema
            return
        for index in (0, 2, 4):
            self._estimate_extrema[index] = min(
                self._estimate_extrema[index], extrema[index]
            )
        for index in (1, 3, 5):
            self._estimate_extrema[index] = max(
                self._estimate_extrema[index], extrema[index]
            )

    def process(self, pvObject):
        started = time.time()
        with self._grid_lock:
            assert self.session is not None
            assert self.binder is not None
            running = self.session.state is GridSessionState.RUNNING
            active = running or self._estimating
            if not active:
                self.binder.observe_inactive_frame_id(self._frame_id(pvObject))
                self.updateOutputChannel(pvObject)
                return pvObject
            if running:
                self.session.note_frame_seen()

            try:
                dims = pvObject["dimension"]
                if not dims or "attribute" not in pvObject:
                    raise ValueError("Frame is missing dimensions or NDAttributes.")
                shape = tuple(int(dimension["size"]) for dimension in dims)
                current = self._current_attributes(pvObject)
                bound = self.binder.bind(
                    current,
                    frame_id=self._frame_id(pvObject),
                    timestamp=self._frame_timestamp(pvObject),
                    metadata_timestamps=self._metadata_timestamps(current),
                )
                if bound is None:
                    if running:
                        self.session.note_binding_rejection()
                    self.nFrameErrors += 1
                    return pvObject

                image = self.decompress_image(pvObject).reshape(shape)
                qxyz = self.create_rsm(dict(bound.values), shape)
                if qxyz is None or qxyz[0] is None:
                    raise ValueError("Angle-to-Q conversion returned no coordinates.")
                q_values = (qxyz[0], qxyz[1], qxyz[2])
                fingerprint = self._frame_geometry_fingerprint(bound.values, shape)

                if self._estimating:
                    if not self._estimate_fingerprint:
                        self._estimate_fingerprint = fingerprint
                    elif self._estimate_fingerprint != fingerprint:
                        self._estimating = False
                        raise ValueError(
                            "Static RSM geometry changed while estimating bounds. "
                            "Restart the estimate after the geometry is stable."
                        )
                    self._observe_bounds(q_values)
                else:
                    try:
                        self.session.validate_geometry(fingerprint)
                    except SessionError:
                        self.session.note_processing_rejection()
                        raise
                    accepted_before = int(
                        self.session.get_state().get("frames_accepted", 0)
                    )
                    self.session.add_frame(
                        *q_values,
                        image,
                        monitor=bound.monitor,
                        followed_gap=bound.followed_gap,
                    )
                    accepted_after = int(
                        self.session.get_state().get("frames_accepted", 0)
                    )
                    if accepted_after > accepted_before:
                        self.nFramesProcessed += 1
                        self._runtime_error = ""
                    else:
                        self.nFrameErrors += 1
                        self._runtime_error = str(
                            self.session.get_state().get("last_error", "")
                        )
                self._publish_preview_if_due()
            except SessionError as exc:
                self.nFrameErrors += 1
                self._runtime_error = str(exc)
                if hasattr(self, "logger"):
                    self.logger.warning("Live grid stopped: %s", exc)
            except Exception as exc:
                self.nFrameErrors += 1
                if running:
                    self.session.note_processing_rejection(str(exc))
                self._runtime_error = str(exc)
                if hasattr(self, "logger"):
                    self.logger.warning("Live grid skipped a frame: %s", exc)
            finally:
                self.processingTime += time.time() - started
                self.updateOutputChannel(pvObject)
        return pvObject

    def _publish_preview_if_due(self) -> None:
        now = time.time()
        if now - self._last_preview < self._preview_interval:
            return
        payload = self.session.preview() if self.session else None
        if payload is None or payload.frames_accepted == 0:
            return
        self._last_preview = now
        self._preview_publishes += 1
        self._latest_preview = payload

    # -- status ------------------------------------------------------------

    def _grid_status(self) -> dict[str, Any]:
        state = self.session.get_state() if self.session else {}
        binder = self.binder.counters.as_dict() if self.binder else {}
        preview = self._latest_preview
        status: dict[str, Any] = {
            "protocol_version": RSM_GRID_PROTOCOL_VERSION,
            "ack_request_id": self._ack_request_id,
            "ack_command": self._ack_command,
            "command_error": self._command_error,
            "state": state.get("state", "idle"),
            "estimate_state": "collecting" if self._estimating else "idle",
            "incomplete": int(bool(state.get("incomplete", False))),
            "last_error": state.get("last_error", "") or self._runtime_error,
            "saved_path": self._saved_path,
            "save_state": self._save_state,
            "save_request_id": self._save_request_id,
            "save_error": self._save_error,
            "geometry_fingerprint": state.get("geometry_fingerprint", ""),
            "grid_shape": state.get("grid_shape", []),
            "grid_bounds": state.get("grid_bounds", []),
            "estimated_bounds": self._estimated_bounds,
            "frames_seen_running": int(state.get("frames_seen_running", 0)),
            "frames_bound": int(binder.get("frames_bound", 0)),
            "frames_accepted": int(state.get("frames_accepted", 0)),
            "frames_rejected_binding": int(
                state.get("frames_rejected_binding", 0)
            ),
            "frames_rejected_processing": int(
                state.get("frames_rejected_processing", 0)
            ),
            "id_gap_events": int(binder.get("id_gap_events", 0)),
            "frames_missing_upstream": int(
                binder.get("frames_missing_upstream", 0)
            ),
            "ids_out_of_order": int(binder.get("ids_out_of_order", 0)),
            "points_binned": int(state.get("points_binned", 0)),
            "points_out_of_range": int(state.get("points_out_of_range", 0)),
            "points_nonfinite": int(state.get("points_nonfinite", 0)),
            "points_masked": int(state.get("points_masked", 0)),
            "accounting_consistent": int(
                bool(state.get("accounting_consistent", True))
            ),
            "memory_peak_bytes": int(state.get("memory_peak_bytes", 0)),
            "preview_publishes": self._preview_publishes,
            "preview_shape": [],
            "preview_origin": [],
            "preview_spacing": [],
            "preview_values": np.asarray([], dtype=np.float32),
            "intensity_range": [],
            "voxels_filled": 0,
        }
        if preview is not None:
            status.update(
                preview_shape=list(preview.shape),
                preview_origin=list(preview.origin),
                preview_spacing=list(preview.spacing),
                preview_values=preview.mean.flatten(order="F"),
                intensity_range=list(preview.intensity_range),
                voxels_filled=preview.voxels_filled,
            )
        return status

    def resetStats(self):
        # Grid accounting resets only via clear/new run, not pvapy's stats reset.
        super().resetStats()
        self._preview_publishes = 0

    def getStats(self):
        with self._grid_lock:
            stats = super().getStats()
            stats[RSM_GRID_NAMESPACE] = self._grid_status()
            return stats

    def getStatsPvaTypes(self):
        types = super().getStatsPvaTypes()
        types[RSM_GRID_NAMESPACE] = {
            "protocol_version": pva.UINT,
            "ack_request_id": pva.STRING,
            "ack_command": pva.STRING,
            "command_error": pva.STRING,
            "state": pva.STRING,
            "estimate_state": pva.STRING,
            "incomplete": pva.UINT,
            "last_error": pva.STRING,
            "saved_path": pva.STRING,
            "save_state": pva.STRING,
            "save_request_id": pva.STRING,
            "save_error": pva.STRING,
            "geometry_fingerprint": pva.STRING,
            "grid_shape": [pva.UINT],
            "grid_bounds": [pva.DOUBLE],
            "estimated_bounds": [pva.DOUBLE],
            "frames_seen_running": pva.ULONG,
            "frames_bound": pva.ULONG,
            "frames_accepted": pva.ULONG,
            "frames_rejected_binding": pva.ULONG,
            "frames_rejected_processing": pva.ULONG,
            "id_gap_events": pva.ULONG,
            "frames_missing_upstream": pva.ULONG,
            "ids_out_of_order": pva.ULONG,
            "points_binned": pva.ULONG,
            "points_out_of_range": pva.ULONG,
            "points_nonfinite": pva.ULONG,
            "points_masked": pva.ULONG,
            "accounting_consistent": pva.UINT,
            "memory_peak_bytes": pva.ULONG,
            "preview_publishes": pva.ULONG,
            "preview_shape": [pva.UINT],
            "preview_origin": [pva.DOUBLE],
            "preview_spacing": [pva.DOUBLE],
            "preview_values": [pva.FLOAT],
            "intensity_range": [pva.DOUBLE],
            "voxels_filled": pva.ULONG,
        }
        return types
