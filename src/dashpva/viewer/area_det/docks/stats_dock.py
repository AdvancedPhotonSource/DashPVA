# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "info"


class StatsDock(BaseDock):
    """Area-detector statistics dock: frame counters and intensity limits.

        dock = StatsDock(main_window=viewer)
        dock.frames_received_val.setText("1024")
        dock.chk_autoscale.setChecked(True)
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Stats", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self.load_ui("area_det", "docks", "stats.ui")
        # Prefixed objectNames keep the persistence keys unique; short names are
        # what the viewer has always used.
        self.min_setting_val = self.stats_min_setting_val
        self.max_setting_val = self.stats_max_setting_val
        self.chk_autoscale   = self.stats_chk_autoscale
        self.chk_threshold   = self.stats_chk_threshold
