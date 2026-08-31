# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "controls"


class ImageDock(BaseDock):
    """Area-detector image dock: plot rate, size readout and display toggles.

        dock = ImageDock(main_window=viewer)
        dock.plotting_frequency.setValue(10)
        dock.log_image.setChecked(True)
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Image", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self.load_ui("area_det", "docks", "image.ui")
        # The persisted objectNames are prefixed to stay unambiguous against the
        # legacy imageshow.ui widgets; callers still use the short attribute.
        self.plotting_frequency = self.image_plotting_frequency
        self.log_image          = self.image_log_image
        self.freeze_image       = self.image_freeze_image
        self.chk_transpose      = self.image_chk_transpose
        self.display_rois       = self.image_display_rois
        self.stop_hkl           = self.image_stop_hkl
