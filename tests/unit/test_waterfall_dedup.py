"""Regression test for the waterfall new-frame guard.

The waterfall is driven by the viewer's plot timer, which can fire faster than
frames arrive (or keep firing while acquisition is stopped). It must stack a row
only when a *new* frame has arrived — otherwise it piles duplicate rows of the
same frame. This test drives ``_on_tick`` directly and checks the buffer only
grows when the reader's ``frames_received`` changes.

Skipped when the GUI stack is unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pyqtgraph = pytest.importorskip("pyqtgraph")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from dashpva.viewer.area_det.docks.waterfall_dock import WaterfallDock  # noqa: E402


class _StubReader:
    def __init__(self):
        self.rois = {}
        self.frames_received = 0
        self.image = None


class _StubMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer_plot = QtCore.QTimer()
        self.reader = _StubReader()
        self._roi_overlays = {}
        self.image_view = pyqtgraph.ImageView()
        self.setCentralWidget(self.image_view)

    def add_dock_toggle_action(self, dock, title, segment_name=None):
        # BaseDock registers a Windows-menu toggle here; a bare action suffices.
        return QtWidgets.QAction(title, self)


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def test_waterfall_only_stacks_on_new_frame(app):
    mw = _StubMainWindow()
    dock = WaterfallDock(main_window=mw, show=True)
    mw.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    mw.show()
    dock.setVisible(True)
    app.processEvents()

    # A frame is on display.
    mw.image_view.setImage(np.ones((20, 20), dtype=np.float32))
    app.processEvents()

    if not dock.isVisible():
        pytest.skip("Dock visibility not realizable in this environment")

    # Manual ROI source (default) — create the ROI so a row can be extracted.
    dock.roi_combo.setCurrentText("Manual ROI")
    dock._ensure_manual_roi()
    app.processEvents()

    mw.reader.frames_received = 5
    dock._on_tick()
    after_first = len(dock._buffer)
    assert after_first >= 1, "first new frame should stack a row"

    # Same frame counter -> plot timer fires again -> must NOT stack.
    dock._on_tick()
    dock._on_tick()
    assert len(dock._buffer) == after_first, "no new frame -> no new row"

    # New frame arrives -> one more row.
    mw.reader.frames_received = 6
    dock._on_tick()
    assert len(dock._buffer) == after_first + 1, "new frame -> exactly one new row"

    # Idle again -> stays put.
    dock._on_tick()
    assert len(dock._buffer) == after_first + 1

    dock.deleteLater()
    mw.deleteLater()
    app.processEvents()
