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
from PyQt5.QtWidgets import QFormLayout, QFrame, QLabel, QSizePolicy, QWidget

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "info"


def _value_label(default: str = "0") -> QLabel:
    lbl = QLabel(default)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lbl.setProperty("valueLabel", True)
    return lbl


class MousePosDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Mouse Position", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QFormLayout(container)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        self.mouse_x_val  = _value_label("0")
        self.mouse_y_val  = _value_label("0")
        self.mouse_px_val = _value_label("0.0")
        self.mouse_h      = _value_label("0.0")
        self.mouse_k      = _value_label("0.0")
        self.mouse_l      = _value_label("0.0")

        layout.addRow(QLabel("X position:"),  self.mouse_x_val)
        layout.addRow(QLabel("Y position:"),  self.mouse_y_val)
        layout.addRow(QLabel("Pixel Value:"), self.mouse_px_val)
        layout.addRow(QLabel("H:"),           self.mouse_h)
        layout.addRow(QLabel("K:"),           self.mouse_k)
        layout.addRow(QLabel("L:"),           self.mouse_l)

        self.setWidget(container)
