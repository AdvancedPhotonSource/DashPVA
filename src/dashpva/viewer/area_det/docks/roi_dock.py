from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
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

        outer.addStretch()
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
