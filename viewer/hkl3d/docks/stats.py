from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QLabel, QWidget,
)

from viewer.core.docks.base_dock import BaseDock


class StatsDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        layout = QFormLayout(container)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        def _val_label(default="0"):
            lbl = QLabel(default)
            lbl.setFrameShape(QLabel.Box)
            lbl.setFrameShadow(QLabel.Sunken)
            lbl.setMinimumHeight(25)
            lbl.setMaximumWidth(150)
            return lbl

        self.frames_received_val = _val_label("0")
        self.missed_frames_val   = _val_label("0")
        self.max_px_val          = _val_label("0.0")
        self.min_px_val          = _val_label("0.0")
        self.data_type_val       = _val_label("none")

        layout.addRow(QLabel("Frames Received:"),  self.frames_received_val)
        layout.addRow(QLabel("Frames Missed:"),    self.missed_frames_val)
        layout.addRow(QLabel("Max [px value]:"),   self.max_px_val)
        layout.addRow(QLabel("Min [px value]:"),   self.min_px_val)
        layout.addRow(QLabel("Image Data Type:"),  self.data_type_val)

        self.sbox_min_intensity = QDoubleSpinBox()
        self.sbox_min_intensity.setRange(-1e10, 1e10)
        self.sbox_min_intensity.setMinimumHeight(30)
        self.sbox_min_intensity.setMaximumWidth(150)

        self.sbox_max_intensity = QDoubleSpinBox()
        self.sbox_max_intensity.setRange(-1e10, 1e10)
        self.sbox_max_intensity.setMinimumHeight(30)
        self.sbox_max_intensity.setMaximumWidth(150)

        self.sbox_min_opacity = QDoubleSpinBox()
        self.sbox_min_opacity.setRange(0.0, 1.0)
        self.sbox_min_opacity.setSingleStep(0.1)
        self.sbox_min_opacity.setMaximumWidth(150)

        self.sbox_max_opacity = QDoubleSpinBox()
        self.sbox_max_opacity.setRange(0.0, 1.0)
        self.sbox_max_opacity.setSingleStep(0.1)
        self.sbox_max_opacity.setValue(1.0)
        self.sbox_max_opacity.setMaximumWidth(150)

        layout.addRow(QLabel("Set Min Intensity:"), self.sbox_min_intensity)
        layout.addRow(QLabel("Set Max Intensity:"), self.sbox_max_intensity)
        layout.addRow(QLabel("Set Min Opacity:"),   self.sbox_min_opacity)
        layout.addRow(QLabel("Set Max Opacity:"),   self.sbox_max_opacity)

        self.setWidget(container)
