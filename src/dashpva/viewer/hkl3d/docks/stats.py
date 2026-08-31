# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock


class StatsDock(BaseDock):
    """HKL3D statistics dock: frame counters and intensity/opacity limits.

        dock = StatsDock(main_window=viewer)
        dock.sbox_max_opacity.setValue(0.5)
        dock.frames_received_val.setText("128")
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self.load_ui("hkl3d", "docks", "stats.ui")
