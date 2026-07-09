"""Live 2D X/Y/Z scatter for a single software ROI.

Each detector frame is one point; X and Y set the position, Z drives point color.
Data comes from the manager's rolling history buffer, refreshed on every
``'frame'`` notification. Reuses the colormap + metric helpers from the workbench
ROI 2D plot dock (coupling-free).
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.workbench.docks.rois.roi_2d_plot_dock import (
    _AVAILABLE_COLORMAPS,
    _build_colormap_fn,
)
from dashpva.viewer.workbench.rois.roi_plot_dock import (
    AXIS_LABELS,
    METRIC_OPTIONS,
    _add_metric_item,
    _combo_key,
)


class AreaDetRoi2DPlotDock(QDockWidget):
    """X/Y/Z scatter of ROI metrics, one point per frame, Z as color."""

    def __init__(self, main_window, roi):
        super().__init__("ROI 2D Plot", main_window)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window
        self.roi = roi
        self._z_auto = True

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        axis_row = QHBoxLayout()
        for lbl_text, attr in [("X:", "x_select"), ("Y:", "y_select"), ("Z (color):", "z_select")]:
            combo = QComboBox()
            for key in METRIC_OPTIONS:
                _add_metric_item(combo, key)
            setattr(self, attr, combo)
            axis_row.addWidget(QLabel(lbl_text))
            axis_row.addWidget(combo)
            axis_row.addSpacing(8)
        axis_row.addStretch()
        self.x_select.setCurrentIndex(METRIC_OPTIONS.index("time"))
        self.y_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
        self.z_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
        layout.addLayout(axis_row)

        cmap_row = QHBoxLayout()
        self.cmap_select = QComboBox()
        for cname in _AVAILABLE_COLORMAPS:
            self.cmap_select.addItem(cname, cname)
        default_cmap = 'viridis' if 'viridis' in _AVAILABLE_COLORMAPS else _AVAILABLE_COLORMAPS[0]
        idx = self.cmap_select.findData(default_cmap)
        if idx >= 0:
            self.cmap_select.setCurrentIndex(idx)
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_row.addWidget(self.cmap_select)
        cmap_row.addSpacing(8)
        self.z_min_spin = QDoubleSpinBox()
        self.z_max_spin = QDoubleSpinBox()
        for spin in (self.z_min_spin, self.z_max_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(4)
            spin.setFixedWidth(100)
            spin.setEnabled(False)
        self.z_auto_check = QCheckBox("Auto Z")
        self.z_auto_check.setChecked(True)
        cmap_row.addWidget(QLabel("Z min:"))
        cmap_row.addWidget(self.z_min_spin)
        cmap_row.addWidget(QLabel("Z max:"))
        cmap_row.addWidget(self.z_max_spin)
        cmap_row.addWidget(self.z_auto_check)
        cmap_row.addStretch()
        layout.addLayout(cmap_row)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        layout.addWidget(self.plot_widget)
        self.scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.plot_item.addItem(self.scatter)
        self.setWidget(container)

        self._colormap_fn = _build_colormap_fn(default_cmap)

        for combo in (self.x_select, self.y_select, self.z_select):
            combo.currentIndexChanged.connect(lambda _: self._update_plot())
        self.cmap_select.currentIndexChanged.connect(lambda _: self._on_cmap_changed())
        self.z_auto_check.toggled.connect(self._on_auto_toggled)
        self.z_min_spin.valueChanged.connect(lambda _: self._update_plot())
        self.z_max_spin.valueChanged.connect(lambda _: self._update_plot())
        self.main.roi_manager.add_listener(self._on_manager_event)

        self._update_title()
        self._update_plot()

    def _update_title(self):
        try:
            self.setWindowTitle(f"ROI 2D Plot: {self.main.roi_manager.get_roi_name(self.roi)}")
        except Exception:
            self.setWindowTitle("ROI 2D Plot")

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

    def _on_cmap_changed(self):
        name = self.cmap_select.currentData() or self.cmap_select.currentText()
        self._colormap_fn = _build_colormap_fn(name)
        self._update_plot()

    def _on_auto_toggled(self, checked: bool):
        self._z_auto = bool(checked)
        self.z_min_spin.setEnabled(not self._z_auto)
        self.z_max_spin.setEnabled(not self._z_auto)
        self._update_plot()

    def _colors(self, z_data, z_min, z_max):
        if len(z_data) == 0:
            return []
        if z_max == z_min:
            return [pg.mkBrush(128, 128, 255, 200)] * len(z_data)
        norm = np.clip((z_data - z_min) / (z_max - z_min), 0.0, 1.0)
        return self._colormap_fn(norm)

    def _update_plot(self):
        x_sel = _combo_key(self.x_select, 'time')
        y_sel = _combo_key(self.y_select, 'sum')
        z_sel = _combo_key(self.z_select, 'sum')
        series = self._series()
        x_data = series.get(x_sel, series['time'])
        y_data = series.get(y_sel, series['sum'])
        z_data = series.get(z_sel, series['sum'])
        n = min(len(x_data), len(y_data), len(z_data))
        if n == 0:
            self.scatter.clear()
            return
        x_data, y_data, z_data = x_data[:n], y_data[:n], z_data[:n]
        if self._z_auto:
            z_min, z_max = float(np.min(z_data)), float(np.max(z_data))
            self.z_min_spin.blockSignals(True)
            self.z_max_spin.blockSignals(True)
            self.z_min_spin.setValue(z_min)
            self.z_max_spin.setValue(z_max)
            self.z_min_spin.blockSignals(False)
            self.z_max_spin.blockSignals(False)
        else:
            z_min, z_max = float(self.z_min_spin.value()), float(self.z_max_spin.value())
            if z_min >= z_max:
                z_min, z_max = float(np.min(z_data)), float(np.max(z_data))
        self.scatter.setData(x=x_data.tolist(), y=y_data.tolist(),
                             brush=self._colors(z_data, z_min, z_max))
        self.plot_item.setLabel('bottom', AXIS_LABELS.get(x_sel, x_sel))
        self.plot_item.setLabel('left', AXIS_LABELS.get(y_sel, y_sel))

    def closeEvent(self, event):
        try:
            self.main.roi_manager.remove_listener(self._on_manager_event)
        except Exception:
            pass
        try:
            self.main._roi_2d_docks.pop(id(self.roi), None)
        except Exception:
            pass
        super().closeEvent(event)
