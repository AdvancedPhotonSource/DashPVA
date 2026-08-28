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
