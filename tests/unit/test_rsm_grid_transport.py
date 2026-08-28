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

"""The live-grid client uses pvapy's configured control/status records."""

import json

import numpy as np
import pytest

from dashpva.utils.rsm_grid_transport import (
    GridControlClient,
    GridTransportError,
    command_envelope,
    parse_command_envelope,
    preview_from_status,
)


class _FakeCluster:
    def __init__(self, *, save=False):
        self.pending = None
        self.refreshes = 0
        self.save = save
        self.grid = {}
        self.control = _FakeControl(self)
        self.status = _FakeStatus(self)

    def factory(self, name):
        return self.control if name == "control" else self.status

    def refresh(self):
        self.refreshes += 1
        # Reproduce pvapy's independent delayed configure/get_stats timers:
        # the first forced refresh completes before configure.
        if self.pending is None or self.refreshes < 2:
            return
        envelope = self.pending
        self.grid.update(
            ack_request_id=envelope["request_id"],
            ack_command=envelope["command"],
            command_error="",
            state="stopped" if envelope["command"] == "save" else "running",
        )
        if envelope["command"] == "save":
            self.grid.update(
                save_request_id=envelope["request_id"],
                save_state="running",
                save_error="",
            )
            if self.save and self.refreshes >= 3:
                self.grid["ack_request_id"] = "another-client-request"
            if self.refreshes >= 4:
                self.grid.update(save_state="complete", saved_path="/data/live.h5")


class _FakeControl:
    def __init__(self, cluster):
        self.cluster = cluster

    def put(self, value):
        if value["command"] == "configure":
            payload = json.loads(value["args"])
            self.cluster.pending = payload["rsm_grid"]
        elif value["command"] == "get_stats":
            self.cluster.refresh()


class _FakeStatus:
    def __init__(self, cluster):
        self.cluster = cluster

    def get(self):
        return {
            "userStats": {"rsm_grid": dict(self.cluster.grid)},
            "processorStats": {"nMissed": 7},
        }


def test_envelope_validation_is_namespaced_and_versioned():
    envelope = command_envelope("start", {"NX": 8}, request_id="request-1")
    assert parse_command_envelope(envelope) == envelope
    with pytest.raises(GridTransportError, match="protocol version"):
        parse_command_envelope({**envelope, "version": 99})


def test_client_repeats_get_stats_until_delayed_configure_is_acknowledged():
    cluster = _FakeCluster()
    client = GridControlClient(
        "control",
        "status",
        channel_factory=cluster.factory,
        timeout_seconds=0.2,
        poll_interval_seconds=0.0,
    )
    state = client.command("start", {"NX": 8})
    assert state["state"] == "running"
    assert state["frames_missed_pvapy"] == 7
    assert cluster.refreshes >= 2


def test_save_waits_for_completion_not_only_command_acknowledgement():
    cluster = _FakeCluster(save=True)
    client = GridControlClient(
        "control",
        "status",
        channel_factory=cluster.factory,
        timeout_seconds=0.1,
        save_timeout_seconds=0.2,
        poll_interval_seconds=0.0,
    )
    state = client.command("save", {"filename": "live.h5"})
    assert state["save_state"] == "complete"
    assert state["saved_path"] == "/data/live.h5"
    assert cluster.refreshes >= 4


def test_preview_reconstructs_fortran_order_without_full_grid_metadata():
    expected = np.arange(24, dtype=np.float32).reshape((2, 3, 4), order="F")
    preview = preview_from_status(
        {
            "preview_values": expected.flatten(order="F"),
            "preview_shape": [2, 3, 4],
            "preview_origin": [0.0, 1.0, 2.0],
            "preview_spacing": [0.1, 0.2, 0.3],
            "intensity_range": [0.0, 23.0],
        }
    )
    np.testing.assert_array_equal(preview.mean, expected)
