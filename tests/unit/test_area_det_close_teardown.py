"""Closing the area detector while it is live must leave nothing running.

The teardown methods are driven with a stand-in ``self``: building the real
``DiffractionImageWindow`` needs the whole Qt viewer and a live channel, while
what is under test is only *what gets stopped*.
"""

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow

from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow
from dashpva.viewer.core.base_window import BaseWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Deleted:
    """Stands in for a signal whose C++ owner is gone."""

    def emit(self, *args):
        raise RuntimeError(
            "wrapped C/C++ object of type DiffractionImageWindow has been deleted"
        )


class _Recorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _FakePV:
    def __init__(self, raises=False):
        self.cleared = False
        self.disconnected = False
        self._raises = raises

    def clear_callbacks(self):
        if self._raises:
            raise RuntimeError("dead PV")
        self.cleared = True

    def disconnect(self):
        self.disconnected = True


class _FakeChannel:
    def __init__(self):
        self.active = True

    def isMonitorActive(self):
        return self.active


class _FakeReader:
    def __init__(self):
        self.channel = _FakeChannel()
        self.stopped = False
        self.roi_monitors_cleared = False

    def _clear_roi_backup_monitor(self):
        self.roi_monitors_cleared = True

    def stop_channel_monitor(self):
        self.stopped = True


class _Stub:
    """Minimal stand-in for the viewer, carrying only what teardown touches."""

    _closing = False
    is_closing = DiffractionImageWindow.is_closing
    begin_close = DiffractionImageWindow.begin_close
    _emit_poller_status = DiffractionImageWindow._emit_poller_status
    _teardown_live_view = DiffractionImageWindow._teardown_live_view

    def __init__(self):
        self.stats_data = {}
        self.hkl_pvs = {}
        self.reader = _FakeReader()
        self.timers_stopped = False

    def stop_child_timers(self):
        self.timers_stopped = True


def test_emit_poller_status_skipped_once_closing():
    stub = _Stub()
    stub.pv_pollers_status = _Recorder()
    stub.begin_close()
    stub._emit_poller_status("Loading ROIs…", "info")
    assert stub.pv_pollers_status.calls == []


def test_emit_poller_status_swallows_deleted_window():
    """The window can be deleted between the flag check and the emit."""
    stub = _Stub()
    stub.pv_pollers_status = _Deleted()
    stub._emit_poller_status("PV poller error: boom", "error")


def test_emit_poller_status_passes_through_when_open():
    stub = _Stub()
    stub.pv_pollers_status = _Recorder()
    stub._emit_poller_status("ROIs and stats ready", "info")
    assert stub.pv_pollers_status.calls == [("ROIs and stats ready", "info")]


def test_teardown_clears_monitors_and_reader(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "dashpva.viewer.area_det.area_det_viewer.camonitor_clear",
        cleared.append,
    )
    stub = _Stub()
    stub.stats_data = {"13SIM1:Stats1:Total_RBV": 1.0, "13SIM1:Stats2:Total_RBV": 2.0}
    pv = _FakePV()
    stub.hkl_pvs = {"hkl:h": pv}

    stub._teardown_live_view()

    assert stub.timers_stopped
    assert sorted(cleared) == sorted(stub.stats_data)
    assert pv.cleared and pv.disconnected
    assert stub.hkl_pvs == {}
    assert stub.reader.roi_monitors_cleared
    assert stub.reader.stopped


def test_teardown_continues_past_a_failing_step(monkeypatch):
    """A step that raises must not strand the steps after it."""
    monkeypatch.setattr(
        "dashpva.viewer.area_det.area_det_viewer.camonitor_clear",
        lambda pv: (_ for _ in ()).throw(RuntimeError("no such monitor")),
    )
    stub = _Stub()
    stub.stats_data = {"13SIM1:Stats1:Total_RBV": 1.0}
    stub.hkl_pvs = {"hkl:h": _FakePV(raises=True)}

    stub._teardown_live_view()

    assert stub.hkl_pvs == {}
    assert stub.reader.stopped


def test_teardown_without_reader(monkeypatch):
    monkeypatch.setattr(
        "dashpva.viewer.area_det.area_det_viewer.camonitor_clear", lambda pv: None
    )
    stub = _Stub()
    stub.reader = None
    stub._teardown_live_view()
    assert stub.timers_stopped


def test_begin_close_sets_flag_readable_as_plain_attribute(qapp):
    window = BaseWindow()
    assert not window.is_closing
    window.begin_close()
    assert window.is_closing
    # Plain instance attribute: readable without going through the sip wrapper,
    # which is what makes it safe from a thread after the C++ side is gone.
    assert window.__dict__["_closing"] is True


def test_stop_child_timers_only_reaches_parented_timers(qapp):
    """Documents why the two plot timers had to become ``QTimer(self)``."""
    window = BaseWindow()
    parented = QTimer(window)
    parented.start(50)
    orphan = QTimer()
    orphan.start(50)

    window.stop_child_timers()

    assert not parented.isActive()
    assert orphan.isActive()
    orphan.stop()


def test_teardown_clears_roi_monitors_even_when_pva_monitor_idle(monkeypatch):
    """The ROI camonitors come from the poller sweep, not the PVA monitor, so
    they have to be cleared whether or not the channel is still streaming."""
    monkeypatch.setattr(
        "dashpva.viewer.area_det.area_det_viewer.camonitor_clear", lambda pv: None
    )
    stub = _Stub()
    stub.reader.channel.active = False

    stub._teardown_live_view()

    assert stub.reader.roi_monitors_cleared
    assert not stub.reader.stopped


def test_close_event_gates_before_tearing_the_live_view_down(qapp):
    """A cancelled close must leave a working window, not a gutted one."""
    order = []

    class _Probe(DiffractionImageWindow):
        def __init__(self):
            QMainWindow.__init__(self)  # skip the full viewer build
            self.mask_viewer = None
            self._closing = False

        def confirm_close(self, event):
            order.append("confirm_close")
            event.ignore()
            return False

        def _teardown_live_view(self):
            order.append("teardown")

    class _Event:
        accepted = True

        def ignore(self):
            self.accepted = False

        def accept(self):
            self.accepted = True

    probe, event = _Probe(), _Event()
    probe.closeEvent(event)

    assert order == ["confirm_close"]
    assert not event.accepted
    assert not probe.is_closing
