#!/usr/bin/env python3
"""
ROI Plot Dock for Workbench (Configurable X/Y Metrics)

Provides a QDockWidget that displays a 1D plot of an ROI metric across frames.
You can choose what the X and Y axes represent via dropdowns: time, sum, min,
max, com. Includes a slider and an interactive vertical line to scrub frames;
stays in sync with the Workbench's frame spinbox.
"""

import os

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QComboBox
)

METRIC_OPTIONS = ["time", "sum", "min", "max", "comx", "comy"]
AXIS_LABELS = {
    "time": "Time (Frame Index)",
    "sum": "ROI Sum",
    "min": "ROI Min",
    "max": "ROI Max",
    "comx": "ROI CoM X",
    "comy": "ROI CoM Y",
}

class ROIPlotDock(QDockWidget):
    def __init__(self, parent, title: str, main_window, roi):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window
        self.roi = roi
        # Ensure title starts with latest ROI name
        try:
            if hasattr(self.main, 'get_roi_name'):
                self._update_title()
        except Exception:
            pass

        # Container for controls + plot + slider
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Stats label above plot
        self.stats_label = QLabel("ROI Stats: -")
        try:
            self.stats_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        except Exception:
            pass
        layout.addWidget(self.stats_label)

        # Axis selection controls
        controls_row = QHBoxLayout()
        lbl_x = QLabel("X:"); lbl_y = QLabel("Y:")
        self.x_select = QComboBox(); self.x_select.addItems(METRIC_OPTIONS)
        self.y_select = QComboBox(); self.y_select.addItems(METRIC_OPTIONS)
        # Defaults
        try:
            self.x_select.setCurrentText("time")
        except Exception:
            pass
        try:
            self.y_select.setCurrentText("sum")
        except Exception:
            pass
        controls_row.addWidget(lbl_x)
        controls_row.addWidget(self.x_select)
        controls_row.addSpacing(12)
        controls_row.addWidget(lbl_y)
        controls_row.addWidget(self.y_select)
        layout.addLayout(controls_row)

        # Plot setup
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', AXIS_LABELS.get('time', 'Time'))
        self.plot_item.setLabel('left', AXIS_LABELS.get('sum', 'ROI Sum'))
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        layout.addWidget(self.plot_widget)

        # Vertical line to indicate current frame (positioned in X-space)
        self.frame_line = pg.InfiniteLine(angle=90, movable=True, pen='c')
        try:
            self.plot_item.addItem(self.frame_line)
        except Exception:
            pass

        # Slider for scrubbing frames
        self.slider = QSlider(Qt.Horizontal)
        layout.addWidget(self.slider)

        self.setWidget(container)

        # Storage for series metrics
        self.series = {m: np.array([0.0], dtype=float) for m in METRIC_OPTIONS}
        self.series['time'] = np.array([0], dtype=int)
        self.series['comx'] = np.array([0.0], dtype=float)
        self.series['comy'] = np.array([0.0], dtype=float)

        # Compute initial series and wire interactions
        self._compute_time_series()
        self._wire_interactions()
        # Initial labels
        self._update_axis_labels()

    def _get_roi_bounds(self):
        """Return integer ROI bounds (x0, y0, w, h) based on ROI position/size."""
        try:
            pos = self.roi.pos(); size = self.roi.size()
            x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
            w = max(1, int(size.x())); h = max(1, int(size.y()))
            return x0, y0, w, h
        except Exception:
            return 0, 0, 1, 1

    def _extract_roi_sub(self, frame, image_item):
        """Extract ROI subarray from frame, respecting transforms if possible."""
        sub = None
        # Try transform-aware extraction
        try:
            if image_item is not None:
                sub = self.roi.getArrayRegion(frame, image_item)
                if sub is not None and hasattr(sub, 'ndim') and sub.ndim > 2:
                    sub = np.squeeze(sub)
        except Exception:
            sub = None
        # Fallback to axis-aligned bounding box
        if sub is None or int(getattr(sub, 'size', 0)) == 0:
            x0, y0, w, h = self._get_roi_bounds()
            hgt, wid = frame.shape
            x1 = min(wid, x0 + w); y1 = min(hgt, y0 + h)
            if x0 < x1 and y0 < y1:
                sub = frame[y0:y1, x0:x1]
            else:
                sub = None
        return sub

    def _compute_time_series(self):
        """Compute per-frame ROI metrics: sum, min, max, std, and time (index)."""
        data = getattr(self.main, 'current_2d_data', None)
        if data is None or not isinstance(data, np.ndarray):
            # No data
            self.series = {m: np.array([0.0], dtype=float) for m in METRIC_OPTIONS}
            self.series['time'] = np.array([0], dtype=int)
            self.slider.setEnabled(False)
            self._update_stats_label()
            self._update_plot()
            return

        image_item = getattr(self.main.image_view, 'imageItem', None) if hasattr(self.main, 'image_view') else None

        if data.ndim == 3:
            num_frames = data.shape[0]
            sums, mins, maxs, comxs, comys = [], [], [], [], []
            for i in range(num_frames):
                frame = np.asarray(data[i], dtype=np.float32)
                sub = self._extract_roi_sub(frame, image_item)
                if sub is not None and int(getattr(sub, 'size', 0)) > 0:
                    s = float(np.sum(sub))
                    mn = float(np.min(sub))
                    mx = float(np.max(sub))
                    total = s if s != 0.0 else 1.0
                    cy = float((sub.sum(axis=0) @ np.arange(sub.shape[1])) / total)
                    cx = float((np.arange(sub.shape[0]) @ sub.sum(axis=1)) / total)
                else:
                    s = 0.0; mn = 0.0; mx = 0.0; cx = 0.0; cy = 0.0
                sums.append(s); mins.append(mn); maxs.append(mx)
                comxs.append(cx); comys.append(cy)
            self.series = {
                'time': np.arange(num_frames, dtype=int),
                'sum': np.asarray(sums, dtype=float),
                'min': np.asarray(mins, dtype=float),
                'max': np.asarray(maxs, dtype=float),
                'comx': np.asarray(comxs, dtype=float),
                'comy': np.asarray(comys, dtype=float),
            }
            self.slider.setEnabled(True)
            try:
                self.slider.setMinimum(0)
                self.slider.setMaximum(max(num_frames - 1, 0))
                cur = 0
                if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                    try:
                        cur = int(self.main.frame_spinbox.value())
                    except Exception:
                        cur = 0
                self.slider.setValue(cur)
            except Exception:
                pass
        else:
            # 2D image -> single point (frame 0)
            frame = np.asarray(data, dtype=np.float32)
            sub = self._extract_roi_sub(frame, image_item)
            if sub is not None and int(getattr(sub, 'size', 0)) > 0:
                s = float(np.sum(sub))
                mn = float(np.min(sub))
                mx = float(np.max(sub))
                total = s if s != 0.0 else 1.0
                cx = float((sub.sum(axis=0) @ np.arange(sub.shape[1])) / total)
                cy = float((np.arange(sub.shape[0]) @ sub.sum(axis=1)) / total)
            else:
                s = 0.0; mn = 0.0; mx = 0.0; cx = 0.0; cy = 0.0
            self.series = {
                'time': np.array([0], dtype=int),
                'sum': np.array([s], dtype=float),
                'min': np.array([mn], dtype=float),
                'max': np.array([mx], dtype=float),
                'comx': np.array([cx], dtype=float),
                'comy': np.array([cy], dtype=float),
            }
            self.slider.setEnabled(False)
        # Load motor positions and extend dropdowns/series
        motor_dict = self._load_motor_positions()
        self.series.update(motor_dict)
        self._refresh_motor_position_options(motor_dict)
        self._update_plot()

    def _load_motor_positions(self) -> dict:
        """Read axis-labeled motor position arrays from the loaded HDF5 file.

        Returns {axis_name: np.ndarray} for datasets whose names contain no ':'
        (i.e. already converted from PV names to human-readable labels like ETA, MU).
        """
        result = {}
        try:
            fp = getattr(self.main, 'current_file_path', None)
            if not fp or not os.path.exists(fp):
                return result
            motor_group_path = 'entry/data/metadata/motor_positions'
            with h5py.File(fp, 'r') as h5f:
                if motor_group_path not in h5f:
                    return result
                grp = h5f[motor_group_path]
                for key in grp.keys():
                    if ':' in key:
                        continue  # skip unconverted PV-named datasets
                    try:
                        arr = np.asarray(grp[key], dtype=float).ravel()
                        if arr.size > 1:  # skip scalars — constant across frames, useless as plot axis
                            result[key] = arr
                    except Exception:
                        pass
        except Exception:
            pass
        return result

    def _refresh_motor_position_options(self, motor_dict: dict):
        """Sync X/Y combo boxes: keep base metrics, append sorted motor position names."""
        motor_names = sorted(motor_dict.keys())
        for combo in (self.x_select, self.y_select):
            cur = combo.currentText()
            combo.blockSignals(True)
            # Trim any previously added motor options (beyond the base metrics)
            while combo.count() > len(METRIC_OPTIONS):
                combo.removeItem(combo.count() - 1)
            for name in motor_names:
                combo.addItem(name)
            # Restore selection; fall back to first item if gone
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _update_axis_labels(self):
        try:
            x_name = self.x_select.currentText()
        except Exception:
            x_name = 'time'
        try:
            y_name = self.y_select.currentText()
        except Exception:
            y_name = 'sum'
        try:
            self.plot_item.setLabel('bottom', AXIS_LABELS.get(x_name, x_name))
        except Exception:
            pass
        try:
            self.plot_item.setLabel('left', AXIS_LABELS.get(y_name, y_name))
        except Exception:
            pass

    def _update_stats_label(self):
        try:
            # Refresh dock title to keep in sync with ROI name changes
            try:
                self._update_title()
            except Exception:
                pass
            frame = None
            try:
                frame = self.main.get_current_frame_data()
            except Exception:
                frame = None
            stats = None
            if frame is not None and hasattr(self.main, 'roi_manager'):
                try:
                    stats = self.main.roi_manager.compute_roi_stats(frame, self.roi)
                except Exception:
                    stats = None
            if stats:
                text = (f"ROI [{stats['x']},{stats['y']} {stats['w']}x{stats['h']}] | "
                        f"sum={stats['sum']:.3f} min={stats['min']:.3f} max={stats['max']:.3f} "
                        f"mean={stats['mean']:.3f} std={stats['std']:.3f} count={stats['count']}")
            else:
                text = "ROI Stats: -"
            try:
                # Ensure label shows a consistent prefix
                if text.startswith("ROI ["):
                    self.stats_label.setText(text)
                else:
                    self.stats_label.setText(f"ROI Stats: {text}")
            except Exception:
                pass
        except Exception:
            pass

    def _update_plot(self):
        try:
            x_sel = self.x_select.currentText()
        except Exception:
            x_sel = 'time'
        try:
            y_sel = self.y_select.currentText()
        except Exception:
            y_sel = 'sum'

        # Always update axis labels first so they reflect the current selection
        # even if plotting subsequently fails
        self._update_axis_labels()

        try:
            x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
            y_data = np.asarray(self.series.get(y_sel, self.series.get('sum')), dtype=float)

            # Trim to matching length so mismatched motor/metric arrays never crash plot()
            min_len = min(len(x_data), len(y_data))
            if min_len > 0:
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]

            self.plot_item.clear()
            self.plot_item.plot(x_data, y_data, pen='y')
            # Re-add vertical line after clear
            try:
                self.plot_item.addItem(self.frame_line)
            except Exception:
                pass
            # Position frame line to current frame value in x-space
            cur = 0
            if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                try:
                    cur = int(self.main.frame_spinbox.value())
                except Exception:
                    cur = 0
            try:
                if x_sel == 'time':
                    self.frame_line.setPos(cur)
                else:
                    idx = np.clip(cur, 0, len(x_data) - 1)
                    self.frame_line.setPos(float(x_data[idx]))
            except Exception:
                pass
        except Exception:
            try:
                self.plot_widget.plot(self.series.get('time'), self.series.get('sum'), pen='y', clear=True)
            except Exception:
                pass

    def _update_title(self):
        try:
            name = None
            try:
                if hasattr(self.main, 'get_roi_name'):
                    name = self.main.get_roi_name(self.roi)
            except Exception:
                name = None
            if not name:
                name = 'ROI'
            try:
                self.setWindowTitle(f"ROI: {name}")
            except Exception:
                pass
        except Exception:
            pass

    def _wire_interactions(self):
        # Slider -> change Workbench frame
        try:
            self.slider.valueChanged.connect(self._on_slider_changed)
        except Exception:
            pass
        # Frame spinbox -> update slider and line
        try:
            if hasattr(self.main, 'frame_spinbox'):
                self.main.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        except Exception:
            pass
        # ROI changes -> recompute series
        try:
            if hasattr(self.roi, 'sigRegionChanged'):
                self.roi.sigRegionChanged.connect(lambda: (self._compute_time_series(), self._update_stats_label()))
            if hasattr(self.roi, 'sigRegionChangeFinished'):
                self.roi.sigRegionChangeFinished.connect(lambda: (self._compute_time_series(), self._update_stats_label()))
        except Exception:
            pass
        # Dragging the vertical line should also update frame
        try:
            self.frame_line.sigPositionChanged.connect(self._on_line_moved)
        except Exception:
            pass
        # Axis selection changes -> replot
        try:
            self.x_select.currentTextChanged.connect(lambda _: self._update_plot())
            self.y_select.currentTextChanged.connect(lambda _: self._update_plot())
        except Exception:
            pass

    def _on_slider_changed(self, value):
        # Update Workbench frame and vertical line
        try:
            if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                self.main.frame_spinbox.setValue(int(value))
        except Exception:
            pass
        try:
            # Update line position in current x-space
            x_sel = self.x_select.currentText() if hasattr(self, 'x_select') else 'time'
            if x_sel == 'time':
                self.frame_line.setPos(int(value))
            else:
                x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
                idx = np.clip(int(value), 0, len(x_data) - 1)
                self.frame_line.setPos(float(x_data[idx]))
        except Exception:
            pass
        try:
            self._update_stats_label()
        except Exception:
            pass

    def _on_frame_spinbox_changed(self, value):
        # Keep slider and line in sync with Workbench
        try:
            self.slider.blockSignals(True)
            self.slider.setValue(int(value))
        except Exception:
            pass
        try:
            # Update line position in current x-space
            x_sel = self.x_select.currentText() if hasattr(self, 'x_select') else 'time'
            if x_sel == 'time':
                self.frame_line.setPos(int(value))
            else:
                x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
                idx = np.clip(int(value), 0, len(x_data) - 1)
                self.frame_line.setPos(float(x_data[idx]))
        except Exception:
            pass
        try:
            self.slider.blockSignals(False)
        except Exception:
            pass
        try:
            self._update_stats_label()
        except Exception:
            pass

    def _on_line_moved(self):
        # When line is dragged, snap to nearest value and update frame
        try:
            pos_val = float(self.frame_line.value())
        except Exception:
            pos_val = 0.0
        # Determine new frame index from x-space
        try:
            x_sel = self.x_select.currentText() if hasattr(self, 'x_select') else 'time'
        except Exception:
            x_sel = 'time'
        try:
            if x_sel == 'time':
                pos = int(round(pos_val))
            else:
                x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
                # Find nearest frame index by metric value
                if len(x_data) == 0:
                    pos = 0
                else:
                    pos = int(np.argmin(np.abs(x_data - pos_val)))
            # Clamp
            if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                max_idx = int(self.slider.maximum()) if hasattr(self.slider, 'maximum') else len(self.series.get('time', [])) - 1
                pos = int(np.clip(pos, 0, max_idx))
        except Exception:
            pos = 0
        try:
            self.frame_line.blockSignals(True)
            # Reposition line to exact x-space of selected frame
            x_sel = self.x_select.currentText() if hasattr(self, 'x_select') else 'time'
            if x_sel == 'time':
                self.frame_line.setPos(int(pos))
            else:
                x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
                idx = np.clip(int(pos), 0, len(x_data) - 1)
                self.frame_line.setPos(float(x_data[idx]))
            self.frame_line.blockSignals(False)
        except Exception:
            pass
        self._on_slider_changed(pos)
        try:
            self._update_stats_label()
        except Exception:
            pass
