"""Live 1D metric plot for a single software ROI.

One point per detector frame, X/Y axes chosen from the ROI metrics. Data comes
from the manager's rolling history buffer (``roi_manager.history[id(roi)]``),
refreshed on every ``'frame'`` notification — no saved-file stack, no slider.

Reuses the metric vocabulary from the workbench ROI plot dock (coupling-free).
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.workbench.rois.roi_plot_dock import (
    AXIS_LABELS,
    METRIC_OPTIONS,
    _add_metric_item,
    _combo_key,
)


class AreaDetRoiPlotDock(QDockWidget):
    """1D live plot of a chosen ROI metric versus another (e.g. sum vs time)."""

    def __init__(self, main_window, roi):
        super().__init__("ROI Plot", main_window)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window
        self.roi = roi

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.x_select = QComboBox()
        self.y_select = QComboBox()
        for key in METRIC_OPTIONS:
            _add_metric_item(self.x_select, key)
            _add_metric_item(self.y_select, key)
        self.x_select.setCurrentIndex(METRIC_OPTIONS.index("time"))
        self.y_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
        controls.addWidget(QLabel("X:"))
        controls.addWidget(self.x_select)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Y:"))
        controls.addWidget(self.y_select)
        controls.addStretch()
        layout.addLayout(controls)

        self.plot_item = pg.PlotItem()
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        layout.addWidget(self.plot_widget)
        self.setWidget(container)

        self.x_select.currentIndexChanged.connect(lambda _: self._update_plot())
        self.y_select.currentIndexChanged.connect(lambda _: self._update_plot())
        self.main.roi_manager.add_listener(self._on_manager_event)

        self._update_title()
        self._update_plot()

    def _update_title(self):
        try:
            self.setWindowTitle(f"ROI Plot: {self.main.roi_manager.get_roi_name(self.roi)}")
        except Exception:
            self.setWindowTitle("ROI Plot")

    def _on_manager_event(self, event: str, roi) -> None:
        if event in ('deleted', 'cleared') and (roi is self.roi or event == 'cleared'):
            self.close()
            return
        if event == 'renamed' and roi is self.roi:
            self._update_title()
        if event in ('frame', 'renamed'):
            self._update_plot()

    def _series(self):
        samples = list(self.main.roi_manager.history.get(id(self.roi), []))
        return {m: np.asarray([s.get(m, 0.0) for s in samples], dtype=float) for m in METRIC_OPTIONS}

    def _update_plot(self):
        x_sel = _combo_key(self.x_select, 'time')
        y_sel = _combo_key(self.y_select, 'sum')
        series = self._series()
        x_data = series.get(x_sel, series['time'])
        y_data = series.get(y_sel, series['sum'])
        n = min(len(x_data), len(y_data))
        self.plot_item.setLabel('bottom', AXIS_LABELS.get(x_sel, x_sel))
        self.plot_item.setLabel('left', AXIS_LABELS.get(y_sel, y_sel))
        self.plot_item.clear()
        if n > 0:
            self.plot_item.plot(x_data[:n], y_data[:n], pen='y')

    def closeEvent(self, event):
        try:
            self.main.roi_manager.remove_listener(self._on_manager_event)
        except Exception:
            pass
        try:
            self.main._roi_plot_docks.pop(id(self.roi), None)
        except Exception:
            pass
        super().closeEvent(event)
