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

"""The RSM grid uses pvapy's real configure and userStats extension points."""

import json
import logging
import threading
from pathlib import Path

import numpy as np
import pvaccess as pva
from pvapy.hpc.dataConsumer import DataConsumer
from pvapy.hpc.dataProcessingController import DataProcessingController
from pvapy.hpc.systemController import SystemController

from dashpva.consumers.hpc.analysis.hpc_rsm_grid_consumer import (
    HpcRsmGridProcessor,
)
from dashpva.utils.metadata_binding import (
    METADATA_TIMESTAMP_ATTRIBUTE_PREFIX,
    ChannelClass,
    ChannelSpec,
    MetadataBinder,
)
from dashpva.utils.rsm_grid_session import GridSessionState, LiveGridSession
from dashpva.utils.rsm_grid_transport import (
    RSM_GRID_NAMESPACE,
    command_envelope,
)

PROFILE = Path(__file__).parents[2] / "pv_configs" / "sample_config.toml"


class _ControlHarness:
    CONTROLLER_TYPE = "consumer"

    def __init__(self, consumer):
        self.hpcObject = consumer
        self.hpcObjectId = 1
        self.logger = logging.getLogger("test-rsm-grid-control")
        self.controlPvObject = {}

    def stopScreen(self):
        pass


def _processor(tmp_path, *, save_fn=None):
    processor = HpcRsmGridProcessor({"path": str(PROFILE)})
    processor.binder = MetadataBinder(
        (
            ChannelSpec("angle", ChannelClass.REQUIRED_DYNAMIC),
            ChannelSpec("energy", ChannelClass.STATIC),
        )
    )
    kwargs = {} if save_fn is None else {"save_fn": save_fn}
    processor.session = LiveGridSession(
        output_dir=str(tmp_path / "out"),
        preview_budget_bytes=1024,
        **kwargs,
    )
    processor._preview_interval = 0.0
    processor.decompress_image = lambda _frame: np.ones(4, dtype=np.float32)

    def _create_rsm(_attributes, _shape):
        values = np.array([[0.2, 0.4], [0.6, 0.8]])
        return values, values, values

    processor.create_rsm = _create_rsm
    return processor


def _pipeline(processor):
    controller = DataProcessingController(
        {"skipInitialUpdates": 0, "objectIdOffset": 1},
        userDataProcessor=processor,
    )
    consumer = DataConsumer.__new__(DataConsumer)
    consumer.processingController = controller
    consumer.pvObjectQueue = None
    return controller, _ControlHarness(consumer)


def _configure(harness, command, payload=None, *, request_id=None):
    envelope = command_envelope(command, payload, request_id=request_id)
    args = json.dumps({RSM_GRID_NAMESPACE: envelope})
    SystemController.controlConfigure(harness, args)
    assert harness.controlPvObject["statusMessage"] == "Configuration successful"
    return envelope


def _frame(frame_id, *, angle=1.0, energy=None, metadata_timestamp=None):
    attributes = []
    if angle is not None:
        attributes.append({"name": "angle", "value": [{"value": angle}]})
        timestamp = frame_id if metadata_timestamp is None else metadata_timestamp
        attributes.append({
            "name": f"{METADATA_TIMESTAMP_ATTRIBUTE_PREFIX}angle",
            "value": [{"value": timestamp}],
        })
    if energy is not None:
        attributes.append({"name": "energy", "value": [{"value": energy}]})
    frame = {
        "uniqueId": frame_id,
        "dimension": [{"size": 2}, {"size": 2}],
        "attribute": attributes,
        "timeStamp": {"secondsPastEpoch": frame_id, "nanoseconds": 0},
    }
    return frame


def test_processor_uses_the_stationary_motor_timestamp_policy():
    processor = HpcRsmGridProcessor({"path": str(PROFILE)})

    assert np.isinf(processor.binder.max_age_seconds)


def test_stale_dynamic_metadata_is_rejected(tmp_path):
    processor = _processor(tmp_path)
    processor.binder.max_age_seconds = 0.5
    controller, harness = _pipeline(processor)
    _start(harness)

    controller.process(_frame(2, energy=10.0, metadata_timestamp=1.0))

    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["frames_accepted"] == 0
    assert status["frames_rejected_binding"] == 1
    assert status["frames_rejected_stale_timestamp"] == 1
    assert status["last_binding_rejection"] == "stale_timestamp"
    assert processor.binder.counters.rejections["stale_timestamp"] == 1


def _start(harness, *, request_id="start-1"):
    return _configure(
        harness,
        "start",
        {
            "HMIN": 0,
            "HMAX": 1,
            "KMIN": 0,
            "KMAX": 1,
            "LMIN": 0,
            "LMAX": 1,
            "NX": 2,
            "NY": 2,
            "NZ": 2,
        },
        request_id=request_id,
    )


def test_actual_pvapy_configure_process_and_nested_user_stats(tmp_path):
    processor = _processor(tmp_path)
    controller, harness = _pipeline(processor)
    _start(harness)

    first = _frame(1, energy=10.0)
    returned = controller.process(first)
    second = _frame(2, angle=None)
    controller.process(second)

    assert returned is first
    assert all(attribute["name"] != "RSM" for attribute in first["attribute"])
    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["ack_request_id"] == "start-1"
    assert status["frames_seen_running"] == 2
    assert status["frames_accepted"] == 1
    assert status["frames_rejected_binding"] == 1
    assert status["frames_rejected_missing_required"] == 1
    assert status["last_binding_rejection"] == "missing_required"
    assert status["accounting_consistent"] == 1
    assert status["preview_values"].nbytes <= 1024

    wire = pva.PvObject(
        controller.getUserStatsPvaTypes(),
        controller.getUserStats(),
    )
    wire_status = wire.toDict()[RSM_GRID_NAMESPACE]
    assert wire_status["frames_accepted"] == 1
    assert wire_status["frames_rejected_missing_required"] == 1


def test_detector_pixels_are_reconstructed_in_ntndarray_order(tmp_path):
    processor = _processor(tmp_path)
    processor.decompress_image = lambda _frame: np.arange(6, dtype=np.float32)
    processor.create_rsm = lambda _attributes, shape: tuple(
        np.ones(shape, dtype=float) * value for value in (0.2, 0.4, 0.6)
    )
    captured = []

    def capture_frame(qx, qy, qz, intensity, **kwargs):
        captured.append(intensity.copy())
        return intensity.size

    processor.session.add_frame = capture_frame
    controller, harness = _pipeline(processor)
    _start(harness)
    frame = _frame(1, energy=10.0)
    frame["dimension"] = [{"size": 2}, {"size": 3}]
    controller.process(frame)

    np.testing.assert_array_equal(
        captured[0],
        np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype=np.float32),
    )


def test_static_geometry_change_stops_before_mixing_frames(tmp_path):
    processor = _processor(tmp_path)
    controller, harness = _pipeline(processor)
    _start(harness)
    controller.process(_frame(1, energy=10.0))
    controller.process(_frame(2, energy=11.0))

    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert processor.session.state is GridSessionState.STOPPED
    assert status["frames_accepted"] == 1
    assert status["frames_rejected_processing"] == 1
    assert status["frames_seen_running"] == 2
    assert status["accounting_consistent"] == 1
    assert "geometry changed" in status["last_error"].lower()


def test_gap_counts_match_pvapy_without_reset_on_resume(tmp_path):
    processor = _processor(tmp_path)
    controller, harness = _pipeline(processor)
    _start(harness)
    controller.process(_frame(1, energy=10.0))
    controller.process(_frame(4))
    _configure(harness, "stop", request_id="stop-1")
    _start(harness, request_id="resume-1")
    controller.process(_frame(5))

    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["id_gap_events"] == 1
    assert status["frames_missing_upstream"] == 2
    assert controller.getProcessorStats()["nMissed"] == 2
    assert status["frames_accepted"] == 3
    assert status["frames_seen_running"] == 3


def test_frames_observed_while_stopped_do_not_become_false_upstream_gaps(tmp_path):
    processor = _processor(tmp_path)
    controller, harness = _pipeline(processor)
    _start(harness)
    controller.process(_frame(1, energy=10.0))
    _configure(harness, "stop", request_id="stop-1")
    controller.process(_frame(2))
    controller.process(_frame(3))
    _start(harness, request_id="resume-1")
    controller.process(_frame(4))

    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["id_gap_events"] == 0
    assert status["frames_missing_upstream"] == 0
    assert status["frames_seen_running"] == 2
    assert status["frames_accepted"] == 2


def test_save_runs_asynchronously_and_serializes_commands(tmp_path):
    save_started = threading.Event()
    release_save = threading.Event()

    def _slow_save(*_args, **_kwargs):
        save_started.set()
        assert release_save.wait(2)

    processor = _processor(tmp_path, save_fn=_slow_save)
    controller, harness = _pipeline(processor)
    _start(harness)
    controller.process(_frame(1, energy=10.0))
    _configure(harness, "stop", request_id="stop-1")
    _configure(
        harness,
        "save",
        {"filename": "live.h5"},
        request_id="save-1",
    )

    assert save_started.wait(1)
    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["save_state"] == "running"
    assert status["save_request_id"] == "save-1"

    _configure(harness, "clear", request_id="clear-during-save")
    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["ack_request_id"] == "clear-during-save"
    assert "save is in progress" in status["command_error"].lower()

    release_save.set()
    processor._save_thread.join(timeout=2)
    status = controller.getUserStats()[RSM_GRID_NAMESPACE]
    assert status["save_state"] == "complete"
    assert status["save_request_id"] == "save-1"
    assert status["saved_path"].endswith("live.h5")
