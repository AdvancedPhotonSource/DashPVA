from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock


class MaskDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Mask", main_window=main_window,
                         segment_name="other", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        layout = QFormLayout(container)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_mask_info = QLabel("No mask loaded")
        self.lbl_mask_info.setFrameShape(QFrame.Box)
        self.lbl_mask_info.setFrameShadow(QFrame.Sunken)
        self.lbl_mask_info.setWordWrap(True)
        self.lbl_mask_info.setProperty("valueLabel", True)
        layout.addRow("Mask:", self.lbl_mask_info)

        self.lbl_mask_pixel_count = QLabel("0")
        self.lbl_mask_pixel_count.setFrameShape(QFrame.Box)
        self.lbl_mask_pixel_count.setFrameShadow(QFrame.Sunken)
        self.lbl_mask_pixel_count.setProperty("valueLabel", True)
        layout.addRow("Masked pixels:", self.lbl_mask_pixel_count)

        load_show = QHBoxLayout()
        self.btn_load_mask = QPushButton("Load Mask")
        load_show.addWidget(self.btn_load_mask)
        self.btn_edit_mask = QPushButton("Edit Mask")
        self.btn_edit_mask.setToolTip("Open the mask editor (creates a blank mask if none exists)")
        load_show.addWidget(self.btn_edit_mask)
        layout.addRow(load_show)

        detect_clear = QHBoxLayout()
        self.btn_detect_dead = QPushButton("Detect Dead Px")
        detect_clear.addWidget(self.btn_detect_dead)
        self.btn_clear_mask = QPushButton("Clear Mask")
        detect_clear.addWidget(self.btn_clear_mask)
        layout.addRow(detect_clear)

        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.setToolTip("Export mask as EPICS NDPluginBadPixel JSON")
        layout.addRow(self.btn_export_json)

        self.chk_apply_mask = QCheckBox("Apply mask to display")
        self.chk_apply_mask.setChecked(True)
        layout.addRow(self.chk_apply_mask)

        self.setWidget(container)
