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

"""Viewer-side live-grid control stays asynchronous and invalidates stale Q."""

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

import dashpva.viewer.hkl3d.hkl_3d_viewer as hkl_viewer
from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow
from dashpva.viewer.hkl3d.docks.grid_control import GridControlDock


class _Dock:
    def __init__(self):
        self.pending = []
        self.states = []
        self.errors = []
        self.unreachable = []
        self._local_error = ""

    def set_command_pending(self, value):
        self.pending.append(value)

    def update_status(self, value):
        self.states.append(value)

    def show_error(self, message):
        self.errors.append(message)

    def mark_unreachable(self, message):
        self.unreachable.append(message)


def _grid_window(client, executor, dock):
    window = SimpleNamespace(
        _grid_future=None,
        _grid_job=(None, None),
        _grid_queue=deque(),
        _grid_executor=executor,
        _ensure_grid_client=lambda: client,
        grid_dock=dock,
        seen=[],
    )
    window._on_grid_status = window.seen.append
    window._grid_tick = lambda: hkl_viewer.HKLImageWindow._grid_tick(window)
    window._grid_poll_future = lambda: None
    return window


@pytest.fixture
def no_qtimer(monkeypatch):
    monkeypatch.setattr(
        hkl_viewer, "QTimer", SimpleNamespace(singleShot=lambda *_args: None)
    )


def test_status_poll_is_asynchronous_and_feeds_one_handler(no_qtimer):
    started = threading.Event()
    release = threading.Event()

    class _Client:
        def refresh_status(self):
            started.set()
            assert release.wait(2)
            return {"state": "running"}

    dock = _Dock()
    executor = ThreadPoolExecutor(max_workers=1)
    window = _grid_window(_Client(), executor, dock)

    before = time.monotonic()
    hkl_viewer.HKLImageWindow._grid_tick(window)
    assert time.monotonic() - before < 0.2
    assert started.wait(1)
    hkl_viewer.HKLImageWindow._grid_tick(window)  # in flight: no second job

    release.set()
    window._grid_future.result(timeout=1)
    hkl_viewer.HKLImageWindow._grid_poll_future(window)
    executor.shutdown()
    assert window.seen == [{"state": "running"}]
    assert window._grid_future is None


def test_commands_queue_behind_a_status_poll_instead_of_being_dropped(no_qtimer):
    release = threading.Event()
    sent = []

    class _Client:
        def refresh_status(self):
            assert release.wait(2)
            return {"state": "idle"}

        def command(self, command, payload):
            sent.append(command)
            return {"state": "idle", "ack_command": command}

    dock = _Dock()
    executor = ThreadPoolExecutor(max_workers=1)
    window = _grid_window(_Client(), executor, dock)

    hkl_viewer.HKLImageWindow._grid_tick(window)
    hkl_viewer.HKLImageWindow._queue_grid_command(window, "clear")
    assert list(window._grid_queue)[0][0] == "clear"
    assert dock.pending == [True]

    release.set()
    window._grid_future.result(timeout=1)
    hkl_viewer.HKLImageWindow._grid_poll_future(window)
    window._grid_future.result(timeout=1)
    hkl_viewer.HKLImageWindow._grid_poll_future(window)
    executor.shutdown()

    assert sent == ["clear"]
    assert window.seen[-1]["ack_command"] == "clear"
    assert dock.pending[-1] is False
    assert dock.errors == [""]


def test_unreadable_status_unlocks_the_fields_instead_of_freezing_them(no_qtimer):
    class _Client:
        def refresh_status(self):
            raise RuntimeError("channel timed out")

    dock = _Dock()
    executor = ThreadPoolExecutor(max_workers=1)
    window = _grid_window(_Client(), executor, dock)
    hkl_viewer.HKLImageWindow._grid_tick(window)
    window._grid_future.exception(timeout=1)
    hkl_viewer.HKLImageWindow._grid_poll_future(window)
    executor.shutdown()

    assert "channel timed out" in dock.unreachable[0]
    assert window.seen == []


def test_grid_render_failure_is_reported_in_the_control_dock():
    dock = _Dock()

    class _Volume:
        def update(self, *_args, **_kwargs):
            raise RuntimeError("renderer unavailable")

    checked = SimpleNamespace(isChecked=lambda: True)
    value = SimpleNamespace(value=lambda: 1.0)
    window = SimpleNamespace(
        grid_volume=_Volume(),
        grid_dock=dock,
        stats_dock=SimpleNamespace(autoscale=checked),
        log_image=checked,
        sbox_min_opacity=value,
        sbox_max_opacity=value,
    )

    hkl_viewer.HKLImageWindow._render_grid_preview(window, object())

    assert "renderer unavailable" in dock.errors[0]


@pytest.fixture
def dock(monkeypatch, tmp_path):
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QAction, QApplication, QMainWindow

    import dashpva.viewer.hkl3d.docks.grid_control as grid_control

    app = QApplication.instance() or QApplication([])
    path = str(tmp_path / "viewer.ini")
    monkeypatch.setattr(grid_control, "_settings", lambda: QSettings(path, QSettings.IniFormat))

    class _Host(QMainWindow):
        def add_dock_toggle_action(self, dock, title, segment_name=None, **_kw):
            return QAction(title, self)

    host = _Host()
    widget = GridControlDock(main_window=host)
    widget.confirmations = []
    widget._confirm = lambda title, text: widget.confirmations.append(title) or widget.answer
    widget.answer = True
    yield widget
    host.deleteLater()
    del app


OBSERVED = [-0.05, -0.04, 0.001, 0.002, 1.89, 1.90]


def test_start_waits_for_observed_bounds_then_follows_them(dock):
    assert dock.btn_primary.text() == "Start Grid"
    assert not dock.btn_primary.isEnabled()
    assert "Waiting for the first valid HKL frame" in dock.hint_label.text()

    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    assert dock.btn_primary.isEnabled()
    assert dock.min_boxes["H"].value() == pytest.approx(-0.05)
    assert dock.max_boxes["L"].value() == pytest.approx(1.90)

    moved = [-0.06, -0.04, 0.001, 0.002, 1.89, 1.91]
    dock.update_status({"state": "idle", "observed_bounds": moved})
    assert dock.min_boxes["H"].value() == pytest.approx(-0.06)


def test_editing_a_bound_switches_to_manual_and_polling_never_overwrites_it(dock):
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    dock.min_boxes["H"].setValue(-0.2)

    assert dock.rb_manual.isChecked()
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    assert dock.min_boxes["H"].value() == pytest.approx(-0.2)
    assert not dock.btn_use_observed.isHidden()


def test_back_to_automatic_confirms_only_after_manual_edits(dock):
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    dock.rb_manual.setChecked(True)
    dock.rb_auto.setChecked(True)
    assert dock.confirmations == []

    dock.min_boxes["H"].setValue(-0.2)
    dock.answer = False
    dock.rb_auto.setChecked(True)
    assert dock.confirmations == ["Use automatic bounds?"]
    assert dock.rb_manual.isChecked()
    assert dock.min_boxes["H"].value() == pytest.approx(-0.2)


def test_manual_bounds_are_compared_with_the_observed_range(dock):
    dock.rb_manual.setChecked(True)
    dock.set_bounds(dock._bounds_dict([-0.045, 0.0, 0.0, 0.01, 1.8, 2.0]))
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    assert "outside the grid on H" in dock.notice_label.text()
    assert dock.notice_label.property("messageLevel") == "error"

    dock.set_bounds(dock._bounds_dict([-0.1, 0.0, 0.0, 0.01, 1.8, 2.0]))
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    assert dock.notice_label.property("messageLevel") == "success"


def test_primary_button_is_start_pause_resume_with_latched_payload(dock):
    started, paused = [], []
    dock.start_requested.connect(started.append)
    dock.stop_requested.connect(lambda: paused.append(True))
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    dock.btn_primary.click()
    assert started[-1]["HMIN"] == pytest.approx(-0.05)

    latched = [-0.0558475, -0.0463, 0.000692, 0.0023, 1.892974, 1.8982]
    running = {"state": "running", "grid_shape": [64, 64, 64], "grid_bounds": latched,
               "frames_accepted": 5}
    dock.update_status(running)
    assert dock.btn_primary.text() == "Pause Grid"
    assert not dock.min_boxes["H"].isEnabled()
    assert not dock.rb_manual.isEnabled()
    dock.btn_primary.click()
    assert paused == [True]

    dock.update_status(dict(running, state="stopped"))
    assert dock.btn_primary.text() == "Resume Grid"
    assert dock.btn_save.isEnabled()
    dock.btn_primary.click()
    assert started[-1]["HMIN"] == -0.0558475
    assert started[-1]["NX"] == 64


def test_new_grid_confirms_before_discarding_data(dock):
    cleared = []
    dock.clear_requested.connect(lambda: cleared.append(True))
    dock.btn_new.click()
    assert cleared == [True] and dock.confirmations == []

    dock.update_status({"state": "stopped", "grid_shape": [8, 8, 8],
                        "grid_bounds": [0, 1, 0, 1, 0, 1], "frames_accepted": 12})
    dock.answer = False
    dock.btn_new.click()
    assert cleared == [True]
    assert dock.confirmations == ["Start a new grid?"]


def test_pending_command_blocks_buttons_not_settings(dock):
    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    dock.set_command_pending(True)
    assert dock.min_boxes["H"].isEnabled()
    assert not dock.btn_primary.isEnabled()
    assert not dock.btn_new.isEnabled()


def test_unreachable_consumer_unlocks_settings(dock):
    dock.update_status({"state": "running", "grid_shape": [8, 8, 8],
                        "grid_bounds": [0, 1, 0, 1, 0, 1]})
    dock.mark_unreachable("status timed out")
    assert dock.min_boxes["H"].isEnabled()
    assert "status timed out" in dock.notice_label.text()


def test_notices_explain_binding_sparse_and_incomplete(dock):
    dock.update_status({"state": "running", "frames_accepted": 0,
                        "frames_rejected_binding": 3,
                        "last_binding_rejection": "missing_required"})
    assert "missing required" in dock.notice_label.text()

    dock.update_status({"state": "running", "frames_accepted": 50, "incomplete": 1,
                        "voxels_filled": 23, "preview_shape": [101, 101, 101]})
    assert "fills only 23" in dock.notice_label.text()

    dock.update_status({"state": "running", "frames_accepted": 50, "incomplete": 1,
                        "frames_missing_upstream": 201, "points_out_of_range": 70,
                        "points_binned": 30})
    assert "201 frames skipped" in dock.notice_label.text()
    assert "70.0% of points" in dock.notice_label.text()


def test_bounds_mode_and_bins_persist_per_profile(dock):
    from PyQt5.QtWidgets import QAction, QMainWindow

    dock.update_status({"state": "idle", "observed_bounds": OBSERVED})
    dock.min_boxes["H"].setValue(-0.3)
    dock.bin_boxes["K"].setValue(200)

    class _Host(QMainWindow):
        def add_dock_toggle_action(self, dock, title, segment_name=None, **_kw):
            return QAction(title, self)

    reopened = GridControlDock(main_window=_Host())
    assert reopened.rb_manual.isChecked()
    assert reopened.min_boxes["H"].value() == pytest.approx(-0.3)
    assert reopened.bin_boxes["K"].value() == 200


def test_clear_ignores_stale_previews_and_removes_the_volume():
    cleared = []
    window = SimpleNamespace(
        grid_dock=_Dock(),
        grid_volume=SimpleNamespace(actor=object()),
        _grid_clear_pending=True,
        _grid_preview_key=None,
        _reset_grid_view=lambda: cleared.append(True),
        _render_grid_preview=lambda _payload: pytest.fail("stale preview rendered"),
    )
    stale = {
        "state": "running", "preview_shape": [2, 2, 2],
        "preview_origin": [0, 0, 0], "preview_spacing": [1, 1, 1],
        "preview_values": [1.0] * 8, "intensity_range": [1.0, 1.0],
    }
    hkl_viewer.HKLImageWindow._on_grid_status(window, stale)
    assert cleared == [True]

    window._grid_clear_pending = False
    hkl_viewer.HKLImageWindow._on_grid_status(window, {"state": "idle"})
    assert cleared == [True, True]


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
