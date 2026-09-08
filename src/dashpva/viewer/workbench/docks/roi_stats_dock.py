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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock


class ROIStatsDock(BaseDock):
    def __init__(self, main_window=None, segment_name="2d", dock_area=Qt.RightDockWidgetArea, show: bool = True):
        super().__init__(title="ROI", main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self.build_dock()

    def build_dock(self):
        """Build the ROI stats dock UI."""
        container = QWidget(self)

        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(6, 6, 6, 6)
        vlayout.setSpacing(6)

        controls_layout = QHBoxLayout()
        lbl_actions = QLabel("Actions for selected:")
        self.show_names_checkbox = QCheckBox("Show names above ROIs")
        controls_layout.addWidget(lbl_actions)
        controls_layout.addWidget(self.show_names_checkbox)
        controls_layout.addStretch(1)
        vlayout.addLayout(controls_layout)

        self.roi_stats_table = QTableWidget(0, 13, container)
        self.roi_stats_table.setHorizontalHeaderLabels([
            "", "Actions", "Name", "sum", "min", "max", "mean", "std", "count", "x", "y", "w", "h"
        ])
        vlayout.addWidget(self.roi_stats_table)

        self.setWidget(container)
