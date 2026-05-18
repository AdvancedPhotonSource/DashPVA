from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"

_ROI_COLORS = {
    1: "rgb(255, 0, 0)",
    2: "rgb(0, 0, 255)",
    3: "rgb(76, 187, 23)",
    4: "rgb(255, 0, 255)",
    5: None,
}

_STATS_BTN_BASE = (
    "QPushButton { font: bold 12pt 'Sans Serif'; %s"
    "border: 1px solid #8f8f91; border-radius: 3px;"
    "background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #f6f7fa,stop:1 #dadbde); }"
    "QPushButton::pressed { background-color: qlineargradient("
    "x1:0,y1:0,x2:0,y2:1,stop:0 #dadbde,stop:1 #f6f7fa); }"
)

_PLOT_BTN_STYLE = (
    "QPushButton { font: 12pt 'Sans Serif'; border: 1px solid #8f8f91;"
    "border-radius: 3px; background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
    "stop:0 #e8f4e8,stop:1 #c8dcc8); }"
    "QPushButton::pressed { background-color: qlineargradient("
    "x1:0,y1:0,x2:0,y2:1,stop:0 #c8dcc8,stop:1 #e8f4e8); }"
)


def _total_label() -> QLabel:
    lbl = QLabel("0.0")
    lbl.setMinimumHeight(30)
    lbl.setFrameShape(QFrame.Box)
    lbl.setFrameShadow(QFrame.Sunken)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
    return lbl


class RoiDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="ROI", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        totals = QFormLayout()
        totals.setVerticalSpacing(20)

        self.lbl_ROI1 = QLabel("ROI1 Total:")
        self.lbl_ROI1.setStyleSheet(f"color: {_ROI_COLORS[1]};")
        self.lbl_ROI2 = QLabel("ROI2 Total:")
        self.lbl_ROI2.setStyleSheet(f"color: {_ROI_COLORS[2]};")
        self.lbl_ROI3 = QLabel("ROI3 Total:")
        self.lbl_ROI3.setStyleSheet(f"color: {_ROI_COLORS[3]};")
        self.lbl_ROI4 = QLabel("ROI4 Total:")
        self.lbl_ROI4.setStyleSheet(f"color: {_ROI_COLORS[4]};")
        self.lbl_image_total = QLabel("Stats5 Total:")

        self.roi1_total_value  = _total_label()
        self.roi1_total_value.setStyleSheet(f"color: {_ROI_COLORS[1]};")
        self.roi2_total_value  = _total_label()
        self.roi2_total_value.setStyleSheet(f"color: {_ROI_COLORS[2]};")
        self.roi3_total_value  = _total_label()
        self.roi3_total_value.setStyleSheet(f"color: {_ROI_COLORS[3]};")
        self.roi4_total_value  = _total_label()
        self.roi4_total_value.setStyleSheet(f"color: {_ROI_COLORS[4]};")
        self.stats5_total_value = _total_label()

        # Per-ROI show/hide checkboxes — paired with the colored label so the
        # user can drop a single ROI rectangle from the image overlay without
        # losing the whole set.  An ROI is drawn iff this is checked AND the
        # global "Show Rois" checkbox in the image dock is checked.
        self.chk_show_roi1 = self._make_roi_checkbox(1)
        self.chk_show_roi2 = self._make_roi_checkbox(2)
        self.chk_show_roi3 = self._make_roi_checkbox(3)
        self.chk_show_roi4 = self._make_roi_checkbox(4)

        totals.addRow(self._roi_label_row(self.chk_show_roi1, self.lbl_ROI1), self.roi1_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi2, self.lbl_ROI2), self.roi2_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi3, self.lbl_ROI3), self.roi3_total_value)
        totals.addRow(self._roi_label_row(self.chk_show_roi4, self.lbl_ROI4), self.roi4_total_value)
        totals.addRow(self.lbl_image_total, self.stats5_total_value)
        outer.addLayout(totals)

        self.lbl_roi_specifi_stats = QLabel("ROI Specific Stats")
        outer.addWidget(self.lbl_roi_specifi_stats)

        for i in range(1, 6):
            color = _ROI_COLORS.get(i)
            color_rule = f"color: {color};" if color else ""
            stat_btn = QPushButton(f"Stats{i}")
            stat_btn.setMinimumHeight(45)
            stat_btn.setStyleSheet(_STATS_BTN_BASE % color_rule)
            plot_btn = QPushButton(f"Plot Stats{i}")
            plot_btn.setMinimumHeight(45)
            plot_btn.setMaximumWidth(100)
            plot_btn.setStyleSheet(_PLOT_BTN_STYLE)

            setattr(self, f"btn_Stats{i}", stat_btn)
            setattr(self, f"btn_PlotStats{i}", plot_btn)

            row = QHBoxLayout()
            row.addWidget(stat_btn)
            row.addWidget(plot_btn)
            outer.addLayout(row)

        outer.addStretch()
        self.setWidget(container)

    @staticmethod
    def _make_roi_checkbox(index: int) -> QCheckBox:
        chk = QCheckBox()
        chk.setChecked(True)
        chk.setToolTip(f"Show ROI{index} on the image")
        color = _ROI_COLORS.get(index)
        if color:
            chk.setStyleSheet(f"QCheckBox::indicator:checked {{ background-color: {color}; }}")
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
