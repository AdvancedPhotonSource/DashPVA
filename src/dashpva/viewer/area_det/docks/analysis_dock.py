# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
from PyQt5.QtCore import Qt

from dashpva.viewer.core.docks.base_dock import BaseDock


class AnalysisDock(BaseDock):
    """Area-detector analysis dock: opens the standalone analysis window.

        dock = AnalysisDock(main_window=viewer)
        dock.btn_analysis_window.clicked.connect(viewer.open_analysis_window)
    """

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Analysis", main_window=main_window,
                         segment_name="other", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self.load_ui("area_det", "docks", "analysis.ui")
