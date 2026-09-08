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
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from dashpva.viewer.core.base_window import BaseWindow
from dashpva.viewer.core.docks.base_dock import BaseDock


class DashAI(BaseDock):
    """
    DashAI dockable window.
    """

    def __init__(self, title="DashAI", main_window: BaseWindow=None, segment_name="2d", dock_area=Qt.RightDockWidgetArea):
        # Call BaseDock with segment routing
        super().__init__(title, main_window, segment_name=segment_name, dock_area=dock_area)
        # Build the dock UI contents
        self.build_dock()

    def connect_all(self):
        self.btn_segment.clicked.connect(self.run_segmentation)

    def build_dock(self):
        self.gb_dash_sam = QGroupBox(self.title)
        layout = QVBoxLayout() # You need a layout to hold widgets

        # Segmentation setup
        # Use a QLabel for instructions
        self.prompt_label = QLabel(
            "<b>Instructions:</b><br>"
            "1. Click on the image to select points.<br>"
            "2. Press 'Segment' to run DashAI.<br>"
            "Add a prompt or message for DashAI to read"
        )
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)

        # 2. The Input Box (Where the user types)
        self.text_prompt_input = QLineEdit()
        self.text_prompt_input.setPlaceholderText("e.g., 'segment the large crystal'...")
        layout.addWidget(self.text_prompt_input)

        # 3. Action Button
        self.btn_segment = QPushButton("Run DashAI Segmentation")
        self.btn_segment.setProperty("role", "success")
        # Connect this button to your SAM function later
        # self.btn_segment.clicked.connect(self.run_segmentation)
        layout.addWidget(self.btn_segment)

        layout.addStretch() # Keeps everything at the top
        self.gb_dash_sam.setLayout(layout)
        self.setWidget(self.gb_dash_sam)

    def run_segmentation(self):
        print("Running segmentation called will be implemented soon")
