# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock


class MaskDock(BaseDock):
    """Area-detector mask dock: load, edit, clear and export the pixel mask.

        dock = MaskDock(main_window=viewer)
        dock.btn_load_mask.clicked.connect(viewer.load_mask_clicked)
        dock.lbl_mask_info.setText("mask.h5")
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Mask", main_window=main_window,
                         segment_name="other", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self.load_ui("area_det", "docks", "mask.ui")
