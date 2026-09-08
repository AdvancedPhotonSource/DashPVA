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
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"

_COLORED_ROIS = (1, 2, 3, 4)


def _total_label(object_name: str) -> QLabel:
    lbl = QLabel("0.0")
    lbl.setObjectName(object_name)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
    lbl.setProperty("valueLabel", True)
    return lbl


class RoiDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="ROI", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        totals = QFormLayout()
        totals.setVerticalSpacing(20)

        self.lbl_ROI1 = QLabel("ROI1 Total:")
        self.lbl_ROI1.setObjectName("lbl_ROI1")
        self.lbl_ROI2 = QLabel("ROI2 Total:")
        self.lbl_ROI2.setObjectName("lbl_ROI2")
        self.lbl_ROI3 = QLabel("ROI3 Total:")
        self.lbl_ROI3.setObjectName("lbl_ROI3")
        self.lbl_ROI4 = QLabel("ROI4 Total:")
        self.lbl_ROI4.setObjectName("lbl_ROI4")

        self.roi1_total_value   = _total_label("roi1_total_value")
        self.roi2_total_value   = _total_label("roi2_total_value")
        self.roi3_total_value   = _total_label("roi3_total_value")
        self.roi4_total_value   = _total_label("roi4_total_value")

        # Per-ROI show/hide checkboxes — paired with the colored label so the
        # user can drop a single ROI rectangle from the image overlay without
        # losing the whole set.  An ROI is drawn iff this is checked AND the
        # global "Show ROIs" checkbox in the image dock is checked.
        self.chk_show_roi1 = self._make_roi_checkbox(1)
        self.chk_show_roi2 = self._make_roi_checkbox(2)
        self.chk_show_roi3 = self._make_roi_checkbox(3)
        self.chk_show_roi4 = self._make_roi_checkbox(4)

        totals.addRow(self._roi_label_row(self.chk_show_roi1, self.lbl_ROI1), self.roi1_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi2, self.lbl_ROI2), self.roi2_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi3, self.lbl_ROI3), self.roi3_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi4, self.lbl_ROI4), self.roi4_total_value)
        outer.addLayout(totals)

        # Single entry point for the consolidated ROI stats + plots window
        # (EPICS ROI 1-4 + up to 5 manual ROIs). Replaces the old per-ROI
        # Stats/Plot buttons.
        self.btn_roi_panel = QPushButton("ROI Stats && Plots")
        self.btn_roi_panel.setObjectName("btn_roi_panel")
        self.btn_roi_panel.setToolTip(
            "Open the consolidated ROI stats + plots window (ROI 1-4 + manual ROIs)")
        outer.addWidget(self.btn_roi_panel)

        # Host for the embedded ROI Stats & Plots panel. The viewer places the
        # panel widget here; it can also be popped out to a standalone window.
        # Hidden until the panel is opened; scrolls when the dock is short.
        self.panel_area = QScrollArea()
        self.panel_area.setObjectName("roi_panel_area")
        self.panel_area.setWidgetResizable(True)
        self.panel_area.setFrameShape(QScrollArea.NoFrame)
        self.panel_area.setVisible(False)
        outer.addWidget(self.panel_area, stretch=1)

        self.setWidget(container)

    @staticmethod
    def _make_roi_checkbox(index: int) -> QCheckBox:
        chk = QCheckBox()
        chk.setObjectName(f"chk_show_roi{index}")
        chk.setChecked(True)
        chk.setToolTip(f"Show ROI{index} on the image")
        return chk

    @staticmethod
    def _roi_label_row(checkbox: QCheckBox, label: QLabel) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(checkbox)
        row.addWidget(label)
        row.addStretch()
        return wrapper
