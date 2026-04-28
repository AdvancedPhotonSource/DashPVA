#!/usr/bin/env python3
"""
ROI 2D Plot Dock for Workbench (X/Y/Z Scatter Plot with Color Scale)

Displays a scatter plot where each frame is one point. X and Y set the axis
positions; Z drives the point color (color scale). All three axes are
configurable using the same metric options as the 1D ROI plot dock.

Controls:
  - X / Y / Z axis dropdowns (metrics + custom CA)
  - Colormap selector
  - Z min / Z max spinboxes with Auto checkbox
"""

import os

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDockWidget, QDoubleSpinBox, QHBoxLayout, QLabel, QComboBox,
    QVBoxLayout, QWidget,
)

from viewer.workbench.rois.roi_plot_dock import (
    METRIC_OPTIONS, AXIS_LABELS,
    _add_metric_item, _combo_key, _set_combo_key,
)

# ---------------------------------------------------------------------------
# Colormap registry — built once at import time from what pyqtgraph offers
# ---------------------------------------------------------------------------

_COLORMAP_CANDIDATES = [
    'viridis', 'plasma', 'inferno', 'magma', 'turbo',
    'CET-L9', 'CET-L1', 'CET-L4', 'CET-D1', 'CET-C1',
]

_AVAILABLE_COLORMAPS: list = []
for _cname in _COLORMAP_CANDIDATES:
    try:
        pg.colormap.get(_cname)
        _AVAILABLE_COLORMAPS.append(_cname)
    except Exception:
        pass

# Always include the built-in gradient as a guaranteed fallback
_BUILTIN_GRADIENT = 'blue→red (built-in)'
_AVAILABLE_COLORMAPS.append(_BUILTIN_GRADIENT)


def _build_colormap_fn(name: str):
    """Return a callable (norm_array → list[QBrush]) for the given colormap name."""
    if name != _BUILTIN_GRADIENT:
        try:
            cmap = pg.colormap.get(name)

            def _apply(t_arr):
                colors = []
                for t in t_arr:
                    rgba = cmap.map(float(np.clip(t, 0.0, 1.0)), mode='byte')
                    colors.append(pg.mkBrush(int(rgba[0]), int(rgba[1]), int(rgba[2]), 200))
                return colors

            return _apply
        except Exception:
            pass

    # Built-in blue → green → red gradient
    def _apply(t_arr):
        colors = []
        for t in t_arr:
            t = float(np.clip(t, 0.0, 1.0))
            r = int(255 * t)
            g = int(255 * (1.0 - abs(2.0 * t - 1.0)))
            b = int(255 * (1.0 - t))
            colors.append(pg.mkBrush(r, g, b, 200))
        return colors

    return _apply


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------

class ROI2DPlotDock(QDockWidget):
    def __init__(self, parent, title: str, main_window, roi):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window
        self.roi = roi
        self._user_closed = False
        self._z_auto = True  # when True, Z range is derived from data

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # -- Stats label --------------------------------------------------
        self.stats_label = QLabel("ROI Stats: -")
        try:
            self.stats_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
        except Exception:
            pass
        layout.addWidget(self.stats_label)

        # -- Axis dropdowns: X, Y, Z --------------------------------------
        axis_row = QHBoxLayout()
        axis_row.setContentsMargins(0, 2, 0, 2)
        for lbl_text, attr_name in [("X:", "x_select"), ("Y:", "y_select"), ("Z (color):", "z_select")]:
            lbl = QLabel(lbl_text)
            try:
                lbl.setStyleSheet("font-size: 11px;")
            except Exception:
                pass
            combo = QComboBox()
            for key in METRIC_OPTIONS:
                _add_metric_item(combo, key)
            setattr(self, attr_name, combo)
            axis_row.addWidget(lbl)
            axis_row.addWidget(combo)
            axis_row.addSpacing(8)
        axis_row.addStretch()

        try:
            self.x_select.setCurrentIndex(METRIC_OPTIONS.index("time"))
            self.y_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
            self.z_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
        except Exception:
            pass
        layout.addLayout(axis_row)

        # -- Colormap selector row ----------------------------------------
        cmap_row = QHBoxLayout()
        cmap_row.setContentsMargins(0, 0, 0, 0)
        cmap_lbl = QLabel("Colormap:")
        try:
            cmap_lbl.setStyleSheet("font-size: 11px;")
        except Exception:
            pass
        self.cmap_select = QComboBox()
        for cname in _AVAILABLE_COLORMAPS:
            self.cmap_select.addItem(cname, cname)
        # Default to viridis if available, else first entry
        _default_cmap = 'viridis' if 'viridis' in _AVAILABLE_COLORMAPS else _AVAILABLE_COLORMAPS[0]
        idx = self.cmap_select.findData(_default_cmap)
        if idx >= 0:
            self.cmap_select.setCurrentIndex(idx)
        cmap_row.addWidget(cmap_lbl)
        cmap_row.addWidget(self.cmap_select)
        cmap_row.addStretch()
        layout.addLayout(cmap_row)

        # -- Z range row: min / max spinboxes + Auto checkbox -------------
        z_range_row = QHBoxLayout()
        z_range_row.setContentsMargins(0, 0, 0, 0)

        z_min_lbl = QLabel("Z min:")
        z_max_lbl = QLabel("Z max:")
        try:
            for w in (z_min_lbl, z_max_lbl):
                w.setStyleSheet("font-size: 11px;")
        except Exception:
            pass

        self.z_min_spin = QDoubleSpinBox()
        self.z_max_spin = QDoubleSpinBox()
        for spin in (self.z_min_spin, self.z_max_spin):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setSingleStep(1.0)
            spin.setFixedWidth(100)
            spin.setEnabled(False)  # disabled when Auto is on

        self.z_auto_check = QCheckBox("Auto")
        self.z_auto_check.setChecked(True)
        try:
            self.z_auto_check.setStyleSheet("font-size: 11px;")
        except Exception:
            pass

        z_range_row.addWidget(z_min_lbl)
        z_range_row.addWidget(self.z_min_spin)
        z_range_row.addSpacing(8)
        z_range_row.addWidget(z_max_lbl)
        z_range_row.addWidget(self.z_max_spin)
        z_range_row.addSpacing(8)
        z_range_row.addWidget(self.z_auto_check)
        z_range_row.addStretch()
        layout.addLayout(z_range_row)

        # -- Z actual range display ----------------------------------------
        self.z_range_label = QLabel("Z: —")
        try:
            self.z_range_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        except Exception:
            pass
        layout.addWidget(self.z_range_label)

        # -- Scatter plot --------------------------------------------------
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        layout.addWidget(self.plot_widget)

        self.scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self.plot_item.addItem(self.scatter)

        # White ring to highlight the current frame
        self.highlight = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen('w', width=2), brush=pg.mkBrush(None)
        )
        self.plot_item.addItem(self.highlight)

        self.setWidget(container)

        # Build initial colormap function
        self._colormap_fn = _build_colormap_fn(_default_cmap)

        # Series storage
        self.series = {m: np.array([0.0], dtype=float) for m in METRIC_OPTIONS}
        self.series['time'] = np.array([0], dtype=int)
        self._last_custom_ca_dict: dict = {}

        self._compute_series()
        self._wire_interactions()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._user_closed = True
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # ROI geometry helpers (mirrors roi_plot_dock.py)
    # ------------------------------------------------------------------

    def _get_roi_bounds(self):
        try:
            pos = self.roi.pos()
            size = self.roi.size()
            return (
                max(0, int(pos.x())),
                max(0, int(pos.y())),
                max(1, int(size.x())),
                max(1, int(size.y())),
            )
        except Exception:
            return 0, 0, 1, 1

    def _extract_roi_sub(self, frame, image_item):
        sub = None
        try:
            if image_item is not None:
                sub = self.roi.getArrayRegion(frame, image_item)
                if sub is not None and hasattr(sub, 'ndim') and sub.ndim > 2:
                    sub = np.squeeze(sub)
        except Exception:
            sub = None
        if sub is None or int(getattr(sub, 'size', 0)) == 0:
            x0, y0, w, h = self._get_roi_bounds()
            hgt, wid = frame.shape
            x1 = min(wid, x0 + w)
            y1 = min(hgt, y0 + h)
            sub = frame[y0:y1, x0:x1] if x0 < x1 and y0 < y1 else None
        return sub

    # ------------------------------------------------------------------
    # Custom CA metadata (mirrors roi_plot_dock.py)
    # ------------------------------------------------------------------

    def _load_custom_ca_metadata(self) -> dict:
        result = {}
        try:
            fp = getattr(self.main, 'current_file_path', None)
            if not fp or not os.path.exists(fp):
                return result
            with h5py.File(fp, 'r') as h5f:
                def _collect_group(path):
                    if path not in h5f:
                        return
                    grp = h5f[path]
                    for key in grp.keys():
                        if key in result:
                            continue
                        try:
                            item = grp[key]
                            if not isinstance(item, h5py.Dataset):
                                continue
                            arr = np.asarray(item, dtype=float).ravel()
                            if arr.size >= 1:
                                result[key] = arr
                        except Exception:
                            pass

                _collect_group('entry/data/metadata/ca')
                _collect_group('entry/data/metadata/motor_positions')

                meta_path = 'entry/data/metadata'
                if meta_path in h5f:
                    for key in h5f[meta_path].keys():
                        if key in result:
                            continue
                        try:
                            item = h5f[meta_path][key]
                            if not isinstance(item, h5py.Dataset):
                                continue
                            arr = np.asarray(item, dtype=float).ravel()
                            if arr.size >= 1:
                                result[key] = arr
                        except Exception:
                            pass
        except Exception:
            pass
        return result

    def _refresh_extra_options(self, custom_ca_dict: dict):
        for combo in (self.x_select, self.y_select, self.z_select):
            cur_key = _combo_key(combo)
            combo.blockSignals(True)
            while combo.count() > len(METRIC_OPTIONS):
                combo.removeItem(combo.count() - 1)
            for name in sorted(custom_ca_dict.keys()):
                combo.addItem(name, name)
            _set_combo_key(combo, cur_key)
            if combo.currentIndex() < 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Series computation
    # ------------------------------------------------------------------

    def _compute_series(self):
        data = getattr(self.main, 'current_2d_data', None)
        if data is None or not isinstance(data, np.ndarray):
            self.series = {m: np.array([0.0], dtype=float) for m in METRIC_OPTIONS}
            self.series['time'] = np.array([0], dtype=int)
            self._update_plot()
            return

        image_item = (
            getattr(self.main.image_view, 'imageItem', None)
            if hasattr(self.main, 'image_view') else None
        )

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
                    s = mn = mx = cx = cy = 0.0
                sums.append(s); mins.append(mn); maxs.append(mx)
                comxs.append(cx); comys.append(cy)
            self.series = {
                'time': np.arange(num_frames, dtype=int),
                'sum':  np.asarray(sums,  dtype=float),
                'min':  np.asarray(mins,  dtype=float),
                'max':  np.asarray(maxs,  dtype=float),
                'comx': np.asarray(comxs, dtype=float),
                'comy': np.asarray(comys, dtype=float),
            }
        else:
            frame = np.asarray(data, dtype=np.float32)
            sub = self._extract_roi_sub(frame, image_item)
            if sub is not None and int(getattr(sub, 'size', 0)) > 0:
                s = float(np.sum(sub))
                mn = float(np.min(sub))
                mx = float(np.max(sub))
                total = s if s != 0.0 else 1.0
                cy = float((sub.sum(axis=0) @ np.arange(sub.shape[1])) / total)
                cx = float((np.arange(sub.shape[0]) @ sub.sum(axis=1)) / total)
            else:
                s = mn = mx = cx = cy = 0.0
            self.series = {
                'time': np.array([0], dtype=int),
                'sum':  np.array([s],  dtype=float),
                'min':  np.array([mn], dtype=float),
                'max':  np.array([mx], dtype=float),
                'comx': np.array([cx], dtype=float),
                'comy': np.array([cy], dtype=float),
            }

        custom_ca_dict = self._load_custom_ca_metadata()
        self._last_custom_ca_dict = custom_ca_dict
        self.series.update(custom_ca_dict)
        self._refresh_extra_options(custom_ca_dict)
        self._update_plot()

    # ------------------------------------------------------------------
    # Z range helpers
    # ------------------------------------------------------------------

    def _get_z_range(self, z_data: np.ndarray):
        """Return (z_min, z_max) from data (auto) or from spinboxes (manual)."""
        data_min = float(np.min(z_data)) if len(z_data) > 0 else 0.0
        data_max = float(np.max(z_data)) if len(z_data) > 0 else 1.0

        if self._z_auto:
            return data_min, data_max

        try:
            z_min = float(self.z_min_spin.value())
            z_max = float(self.z_max_spin.value())
            if z_min < z_max:
                return z_min, z_max
        except Exception:
            pass
        return data_min, data_max

    def _sync_spinboxes_to_data(self, z_data: np.ndarray):
        """Update spinbox display values to match the current data range (auto mode)."""
        if len(z_data) == 0:
            return
        try:
            self.z_min_spin.blockSignals(True)
            self.z_max_spin.blockSignals(True)
            self.z_min_spin.setValue(float(np.min(z_data)))
            self.z_max_spin.setValue(float(np.max(z_data)))
        except Exception:
            pass
        finally:
            self.z_min_spin.blockSignals(False)
            self.z_max_spin.blockSignals(False)

    # ------------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------------

    def _values_to_colors(self, z_data: np.ndarray, z_min: float, z_max: float):
        if len(z_data) == 0:
            return []
        if z_max == z_min:
            return [pg.mkBrush(128, 128, 255, 200)] * len(z_data)
        norm = np.clip((z_data - z_min) / (z_max - z_min), 0.0, 1.0)
        return self._colormap_fn(norm)

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------

    def _update_plot(self):
        try:
            x_sel = _combo_key(self.x_select, 'time')
            y_sel = _combo_key(self.y_select, 'sum')
            z_sel = _combo_key(self.z_select, 'sum')
        except Exception:
            return

        try:
            x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
            y_data = np.asarray(self.series.get(y_sel, self.series.get('sum')),  dtype=float)
            z_data = np.asarray(self.series.get(z_sel, self.series.get('sum')),  dtype=float)

            min_len = min(len(x_data), len(y_data), len(z_data))
            if min_len == 0:
                try:
                    self.scatter.clear()
                except Exception:
                    pass
                try:
                    self.highlight.clear()
                except Exception:
                    pass
                return

            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
            z_data = z_data[:min_len]

            # Sync spinbox display when in auto mode
            if self._z_auto:
                self._sync_spinboxes_to_data(z_data)

            z_min, z_max = self._get_z_range(z_data)
            colors = self._values_to_colors(z_data, z_min, z_max)
            self.scatter.setData(x=x_data.tolist(), y=y_data.tolist(), brush=colors)

            # Highlight current frame with a white ring
            cur = 0
            if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                try:
                    cur = int(np.clip(self.main.frame_spinbox.value(), 0, min_len - 1))
                except Exception:
                    cur = 0
            self.highlight.setData(x=[float(x_data[cur])], y=[float(y_data[cur])])

            # Z range info label
            try:
                mode = "auto" if self._z_auto else "manual"
                self.z_range_label.setText(
                    f"Z ({z_sel}): {z_min:.4g} … {z_max:.4g}  [{mode}]"
                )
            except Exception:
                pass

            # Axis labels
            try:
                self.plot_item.setLabel('bottom', AXIS_LABELS.get(x_sel, x_sel))
                self.plot_item.setLabel('left',   AXIS_LABELS.get(y_sel, y_sel))
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    def _update_title(self):
        try:
            name = None
            if hasattr(self.main, 'get_roi_name'):
                try:
                    name = self.main.get_roi_name(self.roi)
                except Exception:
                    name = None
            self.setWindowTitle(f"ROI 2D: {name or 'ROI'}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _on_cmap_changed(self):
        try:
            name = self.cmap_select.currentData() or self.cmap_select.currentText()
            self._colormap_fn = _build_colormap_fn(name)
        except Exception:
            pass
        self._update_plot()

    def _on_auto_toggled(self, checked: bool):
        self._z_auto = bool(checked)
        # Enable/disable spinboxes
        self.z_min_spin.setEnabled(not self._z_auto)
        self.z_max_spin.setEnabled(not self._z_auto)
        # When switching to manual, pre-fill spinboxes with current data range
        if not self._z_auto:
            try:
                z_sel = _combo_key(self.z_select, 'sum')
                z_data = np.asarray(self.series.get(z_sel, self.series.get('sum')), dtype=float)
                if len(z_data) > 0:
                    self._sync_spinboxes_to_data(z_data)
            except Exception:
                pass
        self._update_plot()

    def _wire_interactions(self):
        try:
            self.x_select.currentIndexChanged.connect(lambda _: self._update_plot())
            self.y_select.currentIndexChanged.connect(lambda _: self._update_plot())
            self.z_select.currentIndexChanged.connect(lambda _: self._update_plot())
        except Exception:
            pass
        try:
            self.cmap_select.currentIndexChanged.connect(lambda _: self._on_cmap_changed())
        except Exception:
            pass
        try:
            self.z_auto_check.toggled.connect(self._on_auto_toggled)
        except Exception:
            pass
        try:
            self.z_min_spin.valueChanged.connect(lambda _: self._update_plot())
            self.z_max_spin.valueChanged.connect(lambda _: self._update_plot())
        except Exception:
            pass
        try:
            if hasattr(self.roi, 'sigRegionChanged'):
                self.roi.sigRegionChanged.connect(self._on_roi_changed)
            if hasattr(self.roi, 'sigRegionChangeFinished'):
                self.roi.sigRegionChangeFinished.connect(self._on_roi_changed)
        except Exception:
            pass
        try:
            if hasattr(self.main, 'frame_spinbox'):
                self.main.frame_spinbox.valueChanged.connect(lambda _: self._update_plot())
        except Exception:
            pass

    def _on_roi_changed(self):
        self._compute_series()

    # ------------------------------------------------------------------
    # Public refresh API (called by workbench on dataset change)
    # ------------------------------------------------------------------

    def refresh_for_dataset_change(self):
        try:
            self._compute_series()
        except Exception:
            pass
