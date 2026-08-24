# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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
from dashpva.utils.metadata_binding import ChannelClass, ChannelSpec, MetadataBinder
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


def _frame(frame_id, *, angle=1.0, energy=None):
    attributes = []
    if angle is not None:
        attributes.append({"name": "angle", "value": [{"value": angle}]})
    if energy is not None:
        attributes.append({"name": "energy", "value": [{"value": energy}]})
    frame = {
        "uniqueId": frame_id,
        "dimension": [{"size": 2}, {"size": 2}],
        "attribute": attributes,
        "timeStamp": {"secondsPastEpoch": frame_id, "nanoseconds": 0},
    }
    return frame


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
    assert status["accounting_consistent"] == 1
    assert status["preview_values"].nbytes <= 1024

    wire = pva.PvObject(
        controller.getUserStatsPvaTypes(),
        controller.getUserStats(),
    )
    assert wire.toDict()[RSM_GRID_NAMESPACE]["frames_accepted"] == 1


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
