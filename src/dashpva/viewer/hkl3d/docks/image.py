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
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock


class ImageDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Image", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Pixel order row
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Image Pixel Order:"))
        self.rbtn_C = QRadioButton("C")
        self.rbtn_F = QRadioButton("Fortran")
        self.rbtn_F.setChecked(True)
        self._pixel_order_group = QButtonGroup(container)
        self._pixel_order_group.addButton(self.rbtn_C)
        self._pixel_order_group.addButton(self.rbtn_F)
        order_row.addWidget(self.rbtn_C)
        order_row.addWidget(self.rbtn_F)
        order_row.addStretch()
        layout.addLayout(order_row)

        # Log image + reset camera row
        tools_row = QHBoxLayout()
        self.log_image = QCheckBox("Log Image")
        self.btn_reset_camera = QPushButton("Reset Camera")
        tools_row.addWidget(self.log_image)
        tools_row.addWidget(self.btn_reset_camera)
        layout.addLayout(tools_row)

        # Action buttons
        self.btn_3d_slice_window = QPushButton("Open Slice 3D Window")

        self.btn_plot_cache = QPushButton("Plot Cache")
        self.btn_plot_cache.setProperty("role", "success")

        self.btn_save_h5 = QPushButton("Save Cache")
        self.btn_save_h5.setProperty("role", "info")
        self.btn_save_h5.setMinimumHeight(50)

        layout.addWidget(self.btn_3d_slice_window)
        layout.addWidget(self.btn_plot_cache)
        layout.addWidget(self.btn_save_h5)
        layout.addStretch()

        self.setWidget(container)
