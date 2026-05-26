from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "info"


def _value_label(default: str = "0") -> QLabel:
    lbl = QLabel(default)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setMinimumHeight(20)
    lbl.setMaximumWidth(150)
    return lbl


class StatsDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
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
        self.min_setting_val.setRange(-1e10, 9999999999.99)
        self.min_setting_val.setMaximumWidth(150)

        self.max_setting_val = QDoubleSpinBox()
        self.max_setting_val.setRange(-1e10, 9999999999.99)
        self.max_setting_val.setMaximumWidth(150)

        layout.addRow(QLabel("Set Min Intensity:"), self.min_setting_val)
        layout.addRow(QLabel("Set Max Intensity:"), self.max_setting_val)

        self.chk_autoscale = QCheckBox("Autoscale (5%-95% histogram)")
        self.chk_autoscale.setChecked(True)
        layout.addRow(self.chk_autoscale)

        self.chk_threshold = QCheckBox("Auto threshold:")
        self.lbl_threshold_range = QLabel("0 to 0")
        layout.addRow(self.chk_threshold, self.lbl_threshold_range)

        self.setWidget(container)
