from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QFrame, QLabel, QSizePolicy, QWidget

from dashpva.viewer.core.docks.base_dock import DOCK_MAX_WIDTH, BaseDock

_SEGMENT = "info"


def _value_label(default: str = "0") -> QLabel:
    lbl = QLabel(default)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setMinimumHeight(25)
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
        container.setMaximumWidth(DOCK_MAX_WIDTH)
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
