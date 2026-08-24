# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Viewer-side live-grid control stays asynchronous and invalidates stale Q."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import dashpva.viewer.hkl3d.hkl_3d_viewer as hkl_viewer
from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow
from dashpva.viewer.hkl3d.docks.grid_control import GridControlDock


class _Dock:
    def __init__(self):
        self.busy = []
        self.states = []

    def set_busy(self, value):
        self.busy.append(value)

    def update_status(self, value):
        self.states.append(value)


def test_remote_status_get_runs_on_the_injected_executor(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class _Client:
        def get_status(self):
            started.set()
            assert release.wait(2)
            return {"state": "running", "estimate_state": "idle"}

    dock = _Dock()
    executor = ThreadPoolExecutor(max_workers=1)
    callbacks = []
    window = SimpleNamespace(
        _grid_status_future=None,
        _grid_status_callback=None,
        _grid_status_busy=False,
        _grid_executor=executor,
        _ensure_grid_client=lambda: _Client(),
        _poll_grid_status=lambda: None,
        _grid_error=lambda error: callbacks.append(error),
        _grid_estimate_started=False,
        grid_dock=dock,
    )
    monkeypatch.setattr(
        hkl_viewer,
        "QTimer",
        SimpleNamespace(singleShot=lambda *_args: None),
    )

    before = time.monotonic()
    hkl_viewer.HKLImageWindow._submit_grid_status(
        window, callbacks.append, busy=True
    )
    elapsed = time.monotonic() - before
    assert elapsed < 0.2
    assert started.wait(1)
    assert dock.busy == [True]

    release.set()
    window._grid_status_future.result(timeout=1)
    hkl_viewer.HKLImageWindow._poll_grid_status(window)
    executor.shutdown()
    assert callbacks == [{"state": "running", "estimate_state": "idle"}]
    assert dock.busy == [True, False]


def test_estimate_reconciliation_preserves_an_active_remote_estimate():
    commands = []
    dock = _Dock()
    window = SimpleNamespace(
        _grid_estimate_started=False,
        _submit_grid_command=lambda command: commands.append(command),
        grid_dock=dock,
    )
    state = {"state": "idle", "estimate_state": "collecting"}
    hkl_viewer.HKLImageWindow._continue_grid_estimate(window, state)
    assert window._grid_estimate_started
    assert commands == []
    assert dock.states == [state]


def test_estimate_reconciliation_attaches_to_an_existing_grid():
    for remote_state in ("running", "stopped", "saving"):
        commands = []
        dock = _Dock()
        window = SimpleNamespace(
            _grid_estimate_started=False,
            _submit_grid_command=lambda command: commands.append(command),
            grid_dock=dock,
        )
        state = {"state": remote_state, "estimate_state": "idle"}
        hkl_viewer.HKLImageWindow._continue_grid_estimate(window, state)
        assert commands == []
        assert dock.states == [state]


class _EnableTarget:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, enabled):
        self.enabled = enabled


def _dock_state(state, *, has_accumulator=False):
    targets = [_EnableTarget() for _ in range(8)]
    dock = SimpleNamespace(
        _state=state,
        _running=state == "running",
        _busy=False,
        _remote_busy=state == "saving",
        _has_accumulator=has_accumulator,
        min_boxes={"H": targets[0]},
        max_boxes={"H": targets[1]},
        bin_boxes={"H": targets[2]},
        btn_estimate=targets[3],
        btn_start=targets[4],
        btn_stop=targets[5],
        btn_clear=targets[6],
        btn_save=targets[7],
    )
    GridControlDock._apply_running_state(dock)
    return dock


def test_grid_dock_locks_latched_configuration_until_clear():
    stopped = _dock_state("stopped", has_accumulator=True)
    assert not stopped.min_boxes["H"].enabled
    assert not stopped.btn_estimate.enabled
    assert stopped.btn_start.enabled
    assert stopped.btn_save.enabled

    idle = _dock_state("idle")
    assert idle.min_boxes["H"].enabled
    assert idle.btn_estimate.enabled
    assert idle.btn_start.enabled
    assert not idle.btn_save.enabled

    saving = _dock_state("saving", has_accumulator=True)
    assert not saving.btn_start.enabled
    assert not saving.btn_clear.enabled
    assert not saving.btn_save.enabled


class _Signal:
    def __init__(self):
        self.count = 0

    def emit(self, _value):
        self.count += 1


def test_static_hkl_update_invalidates_cached_geometry_and_q():
    signal = _Signal()
    window = SimpleNamespace(
        hkl_data={},
        _hkl_dynamic_channels={"angle"},
        _rsm_geometry_cache=object(),
        _rsm_geometry_cache_key="old",
        qx=object(),
        qy=object(),
        qz=object(),
        hkl_data_updated=signal,
    )
    DiffractionImageWindow.hkl_ca_callback(window, "direction", "z-")
    assert window._rsm_geometry_cache is None
    assert window._rsm_geometry_cache_key is None
    assert window.qx is None and window.qy is None and window.qz is None
    assert signal.count == 1


def test_disabling_hkl_clears_cached_geometry_and_q():
    window = SimpleNamespace(
        rsm_geometry_ready=True,
        _rsm_geometry_cache=object(),
        _rsm_geometry_cache_key="old",
        qx=object(),
        qy=object(),
        qz=object(),
    )
    DiffractionImageWindow._on_hkl_enabled_toggled(window, False)
    assert not window.rsm_geometry_ready
    assert window._rsm_geometry_cache is None
    assert window._rsm_geometry_cache_key is None
    assert window.qx is None and window.qy is None and window.qz is None
