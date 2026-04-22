from viewer.core.docks.base_dock import BaseDock
from viewer.core.base_window import BaseWindow
from PyQt5.QtWidgets import QGroupBox, QMessageBox, QLabel, QVBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt


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
        self.btn_segment.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        # Connect this button to your SAM function later
        # self.btn_segment.clicked.connect(self.run_segmentation)
        layout.addWidget(self.btn_segment)

        layout.addStretch() # Keeps everything at the top
        self.gb_dash_sam.setLayout(layout)
        self.setWidget(self.gb_dash_sam)

    def run_segmentation(self):
        print("Running segmentation called will be implemented soon")
