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

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget

from dashpva.gui import ui_path
from dashpva.viewer.core.docks.base_dock import BaseDock


class StatsDock(BaseDock):
    """HKL3D statistics dock: frame counters and intensity/opacity limits.

    The contents come from ``gui/hkl3d/docks/stats.ui``; every named widget in
    that file is re-exposed as an attribute here, so callers keep using
    ``stats_dock.sbox_min_intensity`` as before.

        dock = StatsDock(main_window=viewer)
        dock.sbox_max_opacity.setValue(0.5)
        dock.frames_received_val.setText("128")
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self._build()

    def _build(self):
        try:
            self._widget = QWidget(self)
            uic.loadUi(ui_path("hkl3d", "docks", "stats.ui"), self._widget)
            self.setWidget(self._widget)
            self._alias_children()
        except Exception as e:
            # Keep an empty widget so a bad .ui cannot take the viewer down.
            self._widget = QWidget(self)
            self.setWidget(self._widget)
            try:
                if hasattr(self.main_window, 'update_status'):
                    self.main_window.update_status(f"StatsDock UI load failed: {e}", 'error')
            except Exception:
                pass

    def _alias_children(self):
        """Expose the .ui's named widgets on the dock itself.

        Names shadowing a QDockWidget attribute are skipped, so a widget called
        e.g. "widget" cannot break the dock.
        """
        for child in self._widget.findChildren(QWidget):
            name = child.objectName()
            if name and not name.startswith('qt_') and not hasattr(BaseDock, name):
                setattr(self, name, child)
