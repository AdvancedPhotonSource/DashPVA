from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QSpinBox, QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"


def _value_label(default: str = "0") -> QLabel:
    lbl = QLabel(default)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setMinimumHeight(25)
    return lbl


class ImageDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Image", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        layout = QFormLayout(container)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        self.plot_call_id = _value_label("0")
        layout.addRow(QLabel("Plot Call ID:"), self.plot_call_id)

        self.plotting_frequency = QSpinBox()
        self.plotting_frequency.setRange(1, 999999999)
        self.plotting_frequency.setValue(5)
        layout.addRow(QLabel("Plotting rate (Hz):"), self.plotting_frequency)

        self.size_x_val = _value_label("0")
        self.size_y_val = _value_label("0")
        layout.addRow(QLabel("Size X [px]:"), self.size_x_val)
        layout.addRow(QLabel("Size Y [px]:"), self.size_y_val)

        self.log_image    = QCheckBox("Log Image")
        self.freeze_image = QCheckBox("Freeze Image")
        layout.addRow(self.log_image, self.freeze_image)

        self.chk_transpose = QCheckBox("Transpose Image")
        self.display_rois  = QCheckBox("Show Rois")
        self.display_rois.setChecked(True)
        layout.addRow(self.chk_transpose, self.display_rois)

        order_row = QHBoxLayout()
        self.rbtn_C = QRadioButton("C")
        self.rbtn_F = QRadioButton("Fortran")
        self.rbtn_F.setChecked(True)
        self._pixel_order_group = QButtonGroup(container)
        self._pixel_order_group.addButton(self.rbtn_C)
        self._pixel_order_group.addButton(self.rbtn_F)
        order_row.addWidget(self.rbtn_C)
        order_row.addWidget(self.rbtn_F)
        order_row.addStretch()
        layout.addRow(QLabel("Image Pixel Order:"), order_row)

        self.rotate90degCCW = QPushButton("Rotate 90° CCW")
        self.rotate90degCCW.setMinimumHeight(50)
        self.stop_hkl = QCheckBox("Stop HKL")
        layout.addRow(self.rotate90degCCW, self.stop_hkl)

        self.setWidget(container)
