# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock


class ImageDock(BaseDock):
    """HKL3D image dock: pixel order, log scaling, and cache actions.

        dock = ImageDock(main_window=viewer)
        dock.log_image.setChecked(True)
        dock.btn_plot_cache.clicked.connect(viewer.plot_cache)
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Image", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self.load_ui("hkl3d", "docks", "image.ui")
