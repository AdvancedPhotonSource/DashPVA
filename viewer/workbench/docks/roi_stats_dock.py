from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QTableWidget

from viewer.core.docks.base_dock import BaseDock


class ROIStatsDock(BaseDock):
    def __init__(self, main_window=None, segment_name="2d", dock_area=Qt.RightDockWidgetArea, show: bool = True):
        super().__init__(title="ROI", main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self.build_dock()

    def build_dock(self):
        """Build the ROI stats dock UI."""
        container = QWidget(self)

        vlayout = QVBoxLayout(container)
        vlayout.setContentsMargins(6, 6, 6, 6)
        vlayout.setSpacing(6)

        controls_layout = QHBoxLayout()
        lbl_actions = QLabel("Actions for selected:")
        self.show_names_checkbox = QCheckBox("Show names above ROIs")
        controls_layout.addWidget(lbl_actions)
        controls_layout.addWidget(self.show_names_checkbox)
        controls_layout.addStretch(1)
        vlayout.addLayout(controls_layout)

        self.roi_stats_table = QTableWidget(0, 13, container)
        self.roi_stats_table.setHorizontalHeaderLabels([
            "", "Actions", "Name", "sum", "min", "max", "mean", "std", "count", "x", "y", "w", "h"
        ])
        vlayout.addWidget(self.roi_stats_table)

        self.setWidget(container)
