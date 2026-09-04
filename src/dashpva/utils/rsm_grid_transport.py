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

"""Control/status transport for the remote live RSM grid consumer."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

import dashpva.settings as app_settings

RSM_GRID_NAMESPACE = "rsm_grid"
RSM_GRID_PROTOCOL_VERSION = 1
RSM_GRID_COMMANDS = frozenset(
    {"estimate_start", "estimate_finish", "start", "stop", "clear", "save"}
)


class GridTransportError(RuntimeError):
    """The remote grid control/status contract could not be completed."""


def command_envelope(
    command: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    if command not in RSM_GRID_COMMANDS:
        raise GridTransportError(f"Unknown live-grid command {command!r}.")
    return {
        "version": RSM_GRID_PROTOCOL_VERSION,
        "request_id": request_id or uuid.uuid4().hex,
        "command": command,
        "payload": dict(payload or {}),
    }


def parse_command_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GridTransportError("rsm_grid control payload must be a mapping.")
    version = int(value.get("version", 0))
    if version != RSM_GRID_PROTOCOL_VERSION:
        raise GridTransportError(
            f"Unsupported rsm_grid protocol version {version}; "
            f"expected {RSM_GRID_PROTOCOL_VERSION}."
        )
    request_id = str(value.get("request_id", "")).strip()
    if not request_id:
        raise GridTransportError("rsm_grid request_id is required.")
    command = str(value.get("command", "")).strip()
    if command not in RSM_GRID_COMMANDS:
        raise GridTransportError(f"Unknown live-grid command {command!r}.")
    payload = value.get("payload", {})
    if not isinstance(payload, Mapping):
        raise GridTransportError("rsm_grid command payload must be a mapping.")
    return {
        "version": version,
        "request_id": request_id,
        "command": command,
        "payload": dict(payload),
    }


def preview_from_status(status: Mapping[str, Any]) -> Optional["RemotePreview"]:
    values = status.get("preview_values")
    shape = tuple(int(value) for value in status.get("preview_shape", ()))
    if values is None or len(shape) != 3 or any(value < 2 for value in shape):
        return None
    mean = np.asarray(values, dtype=np.float32)
    if mean.size != int(np.prod(shape)):
        raise GridTransportError(
            f"Preview has {mean.size} values but shape {shape} requires "
            f"{int(np.prod(shape))}."
        )
    return RemotePreview(
        mean=mean.reshape(shape, order="F"),
        shape=shape,
        origin=tuple(float(value) for value in status.get("preview_origin", ())),
        spacing=tuple(float(value) for value in status.get("preview_spacing", ())),
        intensity_range=[
            float(value) for value in status.get("intensity_range", (0.0, 1.0))
        ],
    )


@dataclass(frozen=True)
class RemotePreview:
    mean: np.ndarray
    shape: tuple[int, int, int]
    origin: tuple[float, ...]
    spacing: tuple[float, ...]
    intensity_range: list[float]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "toDict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise GridTransportError(
        f"PVA status value has unsupported type {type(value).__name__}."
    )


class GridControlClient:
    """Synchronous, injectable client over pvapy's control/status records.

    The GUI owns the timing/thread choice. Tests can inject a channel factory;
    production imports pvaccess lazily so non-GUI consumers never pull it in.
    """

    def __init__(
        self,
        control_channel: str,
        status_channel: str,
        *,
        channel_factory: Optional[Callable[[str], Any]] = None,
        control_value_factory: Optional[Callable[[str, str], Any]] = None,
        timeout_seconds: float = app_settings.RSM_GRID_CONTROL_TIMEOUT_SECONDS,
        save_timeout_seconds: float = app_settings.RSM_GRID_SAVE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = (
            app_settings.RSM_GRID_CONTROL_POLL_INTERVAL_SECONDS
        ),
    ) -> None:
        if not control_channel or not status_channel:
            raise GridTransportError(
                "Both analysis control and status channel names are required."
            )
        self.control_channel_name = control_channel
        self.status_channel_name = status_channel
        self.timeout_seconds = float(timeout_seconds)
        self.save_timeout_seconds = float(save_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        if channel_factory is None:
            import pvaccess as pva

            def channel_factory(name):
                return pva.Channel(name, pva.PVA)

            if control_value_factory is None:
                def control_value_factory(command, args):
                    return pva.PvObject(
                        {"command": pva.STRING, "args": pva.STRING},
                        {"command": command, "args": args},
                    )
        self._channel_factory = channel_factory
        self._control_value_factory = control_value_factory or (
            lambda command, args: {"command": command, "args": args}
        )
        self._control = channel_factory(control_channel)
        self._status = channel_factory(status_channel)

    def _put_control(self, command: str, args: str = "") -> None:
        try:
            self._control.put(self._control_value_factory(command, args))
        except Exception as exc:
            raise GridTransportError(
                f"Could not write analysis control channel "
                f"{self.control_channel_name!r}: {exc}"
            ) from exc

    def get_status(self) -> dict[str, Any]:
        try:
            outer = _as_dict(self._status.get())
        except GridTransportError:
            raise
        except Exception as exc:
            raise GridTransportError(
                f"Could not read analysis status channel "
                f"{self.status_channel_name!r}: {exc}"
            ) from exc
        user_stats = outer.get("userStats")
        if (
            not isinstance(user_stats, Mapping)
            or RSM_GRID_NAMESPACE not in user_stats
        ):
            raise GridTransportError(
                f"Analysis status channel {self.status_channel_name!r} does not "
                "expose userStats.rsm_grid. In Workflow > Analysis, select "
                "hpc_rsm_grid_consumer.py with HpcRsmGridProcessor and exactly "
                "one consumer."
            )
        grid = user_stats[RSM_GRID_NAMESPACE]
        if not isinstance(grid, Mapping):
            raise GridTransportError(
                f"Analysis status channel {self.status_channel_name!r} has an "
                "invalid userStats.rsm_grid value."
            )
        result = dict(grid)
        processor_stats = outer.get("processorStats", {})
        if isinstance(processor_stats, Mapping):
            result["frames_missed_pvapy"] = int(processor_stats.get("nMissed", 0))
        return result

    def refresh_status(self) -> dict[str, Any]:
        """Ask pvapy to publish current processor stats, then read them."""
        self._put_control("get_stats")
        time.sleep(self.poll_interval_seconds)
        return self.get_status()

    def command(
        self,
        command: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        envelope = command_envelope(command, payload)
        args = json.dumps({RSM_GRID_NAMESPACE: envelope}, separators=(",", ":"))
        self._put_control("configure", args)

        timeout = (
            self.save_timeout_seconds if command == "save" else self.timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while True:
            # pvapy schedules control commands on independent short-delay
            # timers, so one get_stats can run before configure. Repeating it
            # until the request id is acknowledged makes ordering explicit.
            status = self.refresh_status()
            if status.get("ack_request_id") == envelope["request_id"]:
                if status.get("command_error"):
                    raise GridTransportError(str(status["command_error"]))
                if command != "save":
                    return status
            if command == "save" and status.get("save_request_id") == envelope[
                "request_id"
            ]:
                if status.get("save_state") == "error":
                    raise GridTransportError(str(status.get("save_error", "")))
                if status.get("save_state") == "complete":
                    return status
            if time.monotonic() >= deadline:
                raise GridTransportError(
                    f"Timed out waiting for rsm_grid acknowledgement "
                    f"{envelope['request_id']} on {self.status_channel_name!r}."
                )
