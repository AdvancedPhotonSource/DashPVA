from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

from dashpva.viewer.core.docks.base_dock import BaseDock


class AnalysisDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Analysis", main_window=main_window,
                         segment_name="other", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)

        self.btn_analysis_window = QPushButton("Open Analysis Window")
        self.btn_analysis_window.setMinimumHeight(50)
        layout.addWidget(self.btn_analysis_window)
        layout.addStretch()

        self.setWidget(container)
