# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "info"


class MousePosDock(BaseDock):
    """Area-detector cursor readout: pixel coordinates, value and HKL.

        dock = MousePosDock(main_window=viewer)
        dock.mouse_x_val.setText("512")
        dock.mouse_h.setText("1.004")
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Mouse Position", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self.load_ui("area_det", "docks", "mouse_pos.ui")
