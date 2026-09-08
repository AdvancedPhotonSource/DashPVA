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
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QSizePolicy,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "info"


def _value_label(default: str = "0") -> QLabel:
    lbl = QLabel(default)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lbl.setProperty("valueLabel", True)
    return lbl


class StatsDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QFormLayout(container)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.frames_received_val = _value_label("0")
        self.missed_frames_val   = _value_label("0")
        self.max_px_val          = _value_label("0.0")
        self.min_px_val          = _value_label("0.0")
        self.data_type_val       = _value_label("none")

        layout.addRow(QLabel("Frames Received:"), self.frames_received_val)
        layout.addRow(QLabel("Frames Missed:"),   self.missed_frames_val)
        layout.addRow(QLabel("Max [px value]:"),  self.max_px_val)
        layout.addRow(QLabel("Min [px value]:"),  self.min_px_val)
        layout.addRow(QLabel("Image Data Type:"), self.data_type_val)

        self.min_setting_val = QDoubleSpinBox()
        self.min_setting_val.setObjectName("min_setting_val")
        self.min_setting_val.setRange(-1e10, 9999999999.99)
        self.min_setting_val.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.min_setting_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.max_setting_val = QDoubleSpinBox()
        self.max_setting_val.setObjectName("max_setting_val")
        self.max_setting_val.setRange(-1e10, 9999999999.99)
        self.max_setting_val.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.max_setting_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addRow(QLabel("Set Min Intensity:"), self.min_setting_val)
        layout.addRow(QLabel("Set Max Intensity:"), self.max_setting_val)

        self.chk_autoscale = QCheckBox("Autoscale (5%-95% histogram)")
        layout.addRow(self.chk_autoscale)

        self.chk_threshold = QCheckBox("Auto threshold:")
        self.lbl_threshold_range = QLabel("0 to 0")
        layout.addRow(self.chk_threshold, self.lbl_threshold_range)

        self.setWidget(container)
