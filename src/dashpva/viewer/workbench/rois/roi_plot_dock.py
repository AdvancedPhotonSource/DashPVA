#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

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
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

METRIC_OPTIONS = ["time", "sum", "min", "max", "comx", "comy"]
SINGLE_FRAME_Y_OPTIONS = ["proj_x", "proj_y"]
# Human-readable display names shown in dropdowns (key → label)
DISPLAY_NAMES = {
    "time":   "Time",
    "sum":    "Sum",
    "min":    "Min",
    "max":    "Max",
    "comx":   "CoM X",
    "comy":   "CoM Y",
    "proj_x": "Proj X (sum↓cols)",
    "proj_y": "Proj Y (sum→rows)",
}
AXIS_LABELS = {
    "time":   "Time (Frame Index)",
    "sum":    "ROI Sum",
    "min":    "ROI Min",
    "max":    "ROI Max",
    "comx":   "ROI CoM X",
    "comy":   "ROI CoM Y",
    "proj_x": "Projection onto X (sum along Y)",
    "proj_y": "Projection onto Y (sum along X)",
    "pixel":  "Pixel Index",
}


def _add_metric_item(combo, key: str):
    """Add a metric key to a combo box using its display name, storing key as userData."""
    combo.addItem(DISPLAY_NAMES.get(key, key), key)


def normalize_series(y_data, norm):
    """Divide y_data by norm point for point, offline, from values already in the file.

    This is Workbench normalization by CA readings recorded in the scan HDF5 --
    not live Area Detector CA normalization. Both series come off disk and are
    expected to hold one reading per frame.

    A length mismatch is rejected rather than trimmed: a dropped CA sample
    shifts every later reading onto the wrong frame, and a silently truncated
    curve looks correct. The whole series comes back NaN instead, so the plot is
    visibly empty rather than quietly misaligned.

    A divisor must be finite and strictly positive. Zero, negative and non-finite
    readings give NaN at that frame, so the curve gaps instead of spiking or
    flipping sign.

    Example:
        >>> normalize_series([10.0, 20.0], [2.0, 4.0])
        array([5., 5.])
        >>> normalize_series([10.0, 20.0], [2.0])   # one reading missing
        array([nan, nan])
    """
    y = np.asarray(y_data, dtype=float).ravel()
    if norm is None:
        return y
    d = np.asarray(norm, dtype=float).ravel()
    if len(y) == 0:
        return y
    if len(d) != len(y):
        return np.full(len(y), np.nan, dtype=float)
    out = np.full(len(y), np.nan, dtype=float)
    good = np.isfinite(d) & (d > 0.0)
    out[good] = y[good] / d[good]
    return out


#: The only group a normalization divisor may come from. Motor positions and
#: other metadata are selectable as plot axes but are not per-frame flux
#: readings, so they must never divide a curve.
CA_METADATA_PATH = 'entry/data/metadata/ca'


def norm_key(combo) -> str:
    """Key of the divisor selected in a Norm combo, '' when normalization is off."""
    if combo is None:
        return ''
    data = combo.currentData()
    return '' if data is None else str(data)


def norm_array(combo, channels: dict):
    """The selected divisor's readings, or None when normalization is off.

    Example:
        >>> norm_array(dock.norm_select, {'I0': np.array([1.0, 2.0])})  # doctest: +SKIP
        array([1., 2.])
    """
    key = norm_key(combo)
    if not key:
        return None
    arr = channels.get(key)
    return None if arr is None else np.asarray(arr, dtype=float).ravel()


def load_ca_channels(file_path) -> dict:
    """Read the per-frame CA channels a divisor may be chosen from.

    Restricted to ``CA_METADATA_PATH`` so the Norm dropdown cannot offer a
    motor position or another metadata array. Shared by both ROI plot docks so
    the same channel means the same divisor in each.

    Example:
        >>> load_ca_channels('/data/scan_0007.h5')   # doctest: +SKIP
        {'I0': array([...]), 'I1': array([...])}
    """
    channels: dict = {}
    if not file_path or not os.path.exists(file_path):
        return channels
    try:
        with h5py.File(file_path, 'r') as h5f:
            group = h5f.get(CA_METADATA_PATH)
            if group is None:
                return channels
            for key in group.keys():
                item = group[key]
                if not isinstance(item, h5py.Dataset):
                    continue
                try:
                    arr = np.asarray(item, dtype=float).ravel()
                except Exception:
                    continue
                if arr.size > 1:
                    channels[key] = arr
    except Exception:
        pass
    return channels


def _combo_key(combo, default: str = 'time') -> str:
    """Return the internal key for the combo's current selection (userData if set, else text)."""
    data = combo.currentData()
    if data is not None:
        return str(data)
    txt = combo.currentText()
    return txt if txt else default


def _set_combo_key(combo, key: str):
    """Select the item whose userData equals key; fall back to text match."""
    idx = combo.findData(key)
    if idx < 0:
        idx = combo.findText(key)
    if idx >= 0:
        combo.setCurrentIndex(idx)

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
        self.stats_label.setObjectName("stats_label")
        layout.addWidget(self.stats_label)

        # Mode selection: Stack vs Single Frame
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 2, 0, 2)
        mode_lbl = QLabel("Mode:")
        mode_lbl.setObjectName("mode_label")
        self.radio_stack = QRadioButton("Stack")
        self.radio_single = QRadioButton("Single Frame")
        self.radio_stack.setChecked(True)
        self._mode_group = QButtonGroup(container)
        self._mode_group.addButton(self.radio_stack, 0)
        self._mode_group.addButton(self.radio_single, 1)
        mode_row.addWidget(mode_lbl)
        mode_row.addSpacing(4)
        mode_row.addWidget(self.radio_stack)
        mode_row.addSpacing(8)
        mode_row.addWidget(self.radio_single)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Axis selection controls
        controls_row = QHBoxLayout()
        self._lbl_x = QLabel("X:")
        lbl_y = QLabel("Y:")
        self.x_select = QComboBox()
        self.y_select = QComboBox()
        for key in METRIC_OPTIONS:
            _add_metric_item(self.x_select, key)
            _add_metric_item(self.y_select, key)
        # Defaults: X=time, Y=sum
        self.x_select.setCurrentIndex(METRIC_OPTIONS.index("time"))
        self.y_select.setCurrentIndex(METRIC_OPTIONS.index("sum"))
        controls_row.addWidget(self._lbl_x)
        controls_row.addWidget(self.x_select)
        controls_row.addSpacing(12)
        controls_row.addWidget(lbl_y)
        controls_row.addWidget(self.y_select)
        controls_row.addSpacing(12)
        self._lbl_norm = QLabel("Norm:")
        self.norm_select = QComboBox()
        self.norm_select.addItem("None", "")
        controls_row.addWidget(self._lbl_norm)
        controls_row.addWidget(self.norm_select)
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
        # Storage for single-frame projections
        self.proj_x = np.array([0.0], dtype=float)
        self.proj_y = np.array([0.0], dtype=float)
        self._last_custom_ca_dict: dict = {}

        # Compute initial series and wire interactions
        self._compute_time_series()
        self._wire_interactions()
        # Initial labels
        self._update_axis_labels()

    def _get_roi_bounds(self):
        """Return integer ROI bounds (x0, y0, w, h) based on ROI position/size."""
        try:
            pos = self.roi.pos()
            size = self.roi.size()
            x0 = max(0, int(pos.x()))
            y0 = max(0, int(pos.y()))
            w = max(1, int(size.x()))
            h = max(1, int(size.y()))
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
            x1 = min(wid, x0 + w)
            y1 = min(hgt, y0 + h)
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
                    s = 0.0
                    mn = 0.0
                    mx = 0.0
                    cx = 0.0
                    cy = 0.0
                sums.append(s)
                mins.append(mn)
                maxs.append(mx)
                comxs.append(cx)
                comys.append(cy)
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
                s = 0.0
                mn = 0.0
                mx = 0.0
                cx = 0.0
                cy = 0.0
            self.series = {
                'time': np.array([0], dtype=int),
                'sum': np.array([s], dtype=float),
                'min': np.array([mn], dtype=float),
                'max': np.array([mx], dtype=float),
                'comx': np.array([cx], dtype=float),
                'comy': np.array([cy], dtype=float),
            }
            self.slider.setEnabled(False)
        # Load custom CA metadata and extend dropdowns/series
        custom_ca_dict = self._load_custom_ca_metadata()
        self._last_custom_ca_dict = custom_ca_dict
        self.series.update(custom_ca_dict)
        self._refresh_extra_options(custom_ca_dict)
        self._update_plot()

    def _load_custom_ca_metadata(self) -> dict:
        """Read the scan's CA channels; see :func:`load_ca_channels`."""
        return load_ca_channels(getattr(self.main, 'current_file_path', None))

    def _refresh_extra_options(self, custom_ca_dict: dict):
        """Sync X/Y combo boxes: keep base metrics, then custom CA metadata names."""
        # Normalization list is "None" plus every CA channel, and applies in both
        # modes, so it is refreshed before the single-frame early return below.
        norm_combo = getattr(self, 'norm_select', None)
        if norm_combo is not None:
            cur_norm = self._norm_key()
            norm_combo.blockSignals(True)
            norm_combo.clear()
            norm_combo.addItem("None", "")
            for name in sorted(custom_ca_dict.keys()):
                norm_combo.addItem(name, name)
            # By key, not index: a reload that adds or drops channels must not
            # silently repoint the user's normalizer at a different one.
            idx = norm_combo.findData(cur_norm)
            norm_combo.setCurrentIndex(idx if idx >= 0 else 0)
            norm_combo.blockSignals(False)
        # Don't touch Y combo in single-frame mode — it uses proj options only
        if getattr(self, 'radio_single', None) and self.radio_single.isChecked():
            return
        for combo in (self.x_select, self.y_select):
            cur_key = _combo_key(combo)
            combo.blockSignals(True)
            # Trim any previously added extra options (beyond the base metrics)
            while combo.count() > len(METRIC_OPTIONS):
                combo.removeItem(combo.count() - 1)
            for name in sorted(custom_ca_dict.keys()):
                combo.addItem(name, name)  # userData=name for consistent key lookup
            # Restore selection; fall back to first item if gone
            _set_combo_key(combo, cur_key)
            if combo.currentIndex() < 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)

    def _on_mode_changed(self):
        single = self.radio_single.isChecked()
        # Toggle X controls and slider visibility
        self._lbl_x.setVisible(not single)
        self.x_select.setVisible(not single)
        self.slider.setVisible(not single)
        try:
            self.frame_line.setVisible(not single)
        except Exception:
            pass
        # Swap Y dropdown options
        self.y_select.blockSignals(True)
        self.y_select.clear()
        if single:
            for key in SINGLE_FRAME_Y_OPTIONS:
                _add_metric_item(self.y_select, key)
            self.y_select.setCurrentIndex(0)
        else:
            for key in METRIC_OPTIONS:
                _add_metric_item(self.y_select, key)
            for name in sorted(self._last_custom_ca_dict.keys()):
                self.y_select.addItem(name, name)
            _set_combo_key(self.y_select, "sum")
        self.y_select.blockSignals(False)
        # Recompute and replot for new mode
        if single:
            self._compute_and_plot_single_frame()
        else:
            self._compute_time_series()

    def _compute_and_plot_single_frame(self):
        """Sum ROI along each axis for the current frame and replot."""
        data = getattr(self.main, 'current_2d_data', None)
        if data is None:
            self.proj_x = np.array([0.0])
            self.proj_y = np.array([0.0])
            self._update_plot()
            return
        frame_idx = 0
        if data.ndim == 3 and hasattr(self.main, 'frame_spinbox'):
            try:
                frame_idx = int(self.main.frame_spinbox.value())
                frame_idx = int(np.clip(frame_idx, 0, data.shape[0] - 1))
            except Exception:
                frame_idx = 0
            frame = np.asarray(data[frame_idx], dtype=np.float32)
        else:
            frame = np.asarray(data, dtype=np.float32)
        image_item = getattr(self.main.image_view, 'imageItem', None) if hasattr(self.main, 'image_view') else None
        sub = self._extract_roi_sub(frame, image_item)
        if sub is not None and int(getattr(sub, 'size', 0)) > 0:
            # proj_x: sum along rows (axis=0) → 1D array over columns (X pixel)
            self.proj_x = sub.sum(axis=0).astype(float)
            # proj_y: sum along columns (axis=1) → 1D array over rows (Y pixel)
            self.proj_y = sub.sum(axis=1).astype(float)
        else:
            self.proj_x = np.array([0.0])
            self.proj_y = np.array([0.0])
        self._update_plot()

    def _norm_key(self) -> str:
        return norm_key(getattr(self, 'norm_select', None))

    def _norm_array(self):
        return norm_array(getattr(self, 'norm_select', None),
                          self._last_custom_ca_dict)

    def _normalize_series(self, y_data):
        """Time-series mode: one Y point per frame, one reading per frame."""
        return normalize_series(y_data, self._norm_array())

    def _normalize_frame(self, y_data):
        """Single-frame mode: divide the whole projection by this frame's reading.

        Returns (values, note). A missing, out-of-range, non-positive or
        non-finite reading returns the projection undivided with a note for the
        label, rather than plotting raw data under a "/ monitor" heading. Same
        rule as :func:`normalize_series`, so a frame that gaps in time-series
        mode is not quietly divided here.
        """
        y = np.asarray(y_data, dtype=float).ravel()
        arr = self._norm_array()
        if arr is None:
            return y, ''
        idx = 0
        if hasattr(self.main, 'frame_spinbox'):
            try:
                idx = int(self.main.frame_spinbox.value())
            except Exception:
                idx = 0
        if idx < 0 or idx >= len(arr):
            return y, ' (unavailable)'
        d = float(arr[idx])
        if not np.isfinite(d) or d <= 0.0:
            return y, ' (unavailable)'
        return y / d, ''

    def _update_axis_labels(self):
        try:
            x_name = _combo_key(self.x_select, 'time')
        except Exception:
            x_name = 'time'
        try:
            y_name = _combo_key(self.y_select, 'sum')
        except Exception:
            y_name = 'sum'
        try:
            self.plot_item.setLabel('bottom', AXIS_LABELS.get(x_name, x_name))
        except Exception:
            pass
        try:
            y_label = AXIS_LABELS.get(y_name, y_name)
            norm_key = self._norm_key()
            if norm_key:
                y_label = f"{y_label} / {norm_key}"
            self.plot_item.setLabel('left', y_label)
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
        # Single-frame projection mode
        if getattr(self, 'radio_single', None) and self.radio_single.isChecked():
            try:
                y_sel = _combo_key(self.y_select, 'proj_x')
                if y_sel == 'proj_x':
                    y_data = self.proj_x
                    x_label = "Column (X Pixel)"
                    y_label = AXIS_LABELS['proj_x']
                else:
                    y_data = self.proj_y
                    x_label = "Row (Y Pixel)"
                    y_label = AXIS_LABELS['proj_y']
                y_data, note = self._normalize_frame(y_data)
                norm_key = self._norm_key()
                if norm_key:
                    y_label = f"{y_label} / {norm_key}{note}"
                x_data = np.arange(len(y_data), dtype=float)
                self.plot_item.setLabel('bottom', x_label)
                self.plot_item.setLabel('left', y_label)
                self.plot_item.clear()
                self.plot_item.plot(x_data, y_data, pen='y')
            except Exception:
                pass
            return

        try:
            x_sel = _combo_key(self.x_select, 'time')
        except Exception:
            x_sel = 'time'
        try:
            y_sel = _combo_key(self.y_select, 'sum')
        except Exception:
            y_sel = 'sum'

        # Always update axis labels first so they reflect the current selection
        # even if plotting subsequently fails
        self._update_axis_labels()

        try:
            x_data = np.asarray(self.series.get(x_sel, self.series.get('time')), dtype=float)
            y_data = np.asarray(self.series.get(y_sel, self.series.get('sum')), dtype=float)

            # Normalize before the trim: dividing by a shorter channel shortens Y,
            # and X has to be cut to match it.
            y_data = self._normalize_series(y_data)

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
        # Mode radio buttons
        try:
            self.radio_stack.toggled.connect(lambda _: self._on_mode_changed())
            self.radio_single.toggled.connect(lambda _: self._on_mode_changed())
        except Exception:
            pass
        # ROI changes -> recompute series (mode-aware)
        try:
            if hasattr(self.roi, 'sigRegionChanged'):
                self.roi.sigRegionChanged.connect(self._on_roi_changed)
            if hasattr(self.roi, 'sigRegionChangeFinished'):
                self.roi.sigRegionChangeFinished.connect(self._on_roi_changed)
        except Exception:
            pass
        # Dragging the vertical line should also update frame
        try:
            self.frame_line.sigPositionChanged.connect(self._on_line_moved)
        except Exception:
            pass
        # Axis selection changes -> replot
        try:
            self.x_select.currentIndexChanged.connect(lambda _: self._update_plot())
            self.y_select.currentIndexChanged.connect(lambda _: self._update_plot())
            self.norm_select.currentIndexChanged.connect(lambda _: self._update_plot())
        except Exception:
            pass

    def refresh_for_dataset_change(self):
        """Called by the workbench when a new dataset/file is loaded.
        Re-reads custom CA metadata from the new file and recomputes the plot.
        """
        try:
            if getattr(self, 'radio_single', None) and self.radio_single.isChecked():
                # _compute_time_series is the only other place that loads the CA
                # metadata, and single-frame mode never calls it.
                custom_ca_dict = self._load_custom_ca_metadata()
                self._last_custom_ca_dict = custom_ca_dict
                self._refresh_extra_options(custom_ca_dict)
                self._compute_and_plot_single_frame()
            else:
                self._compute_time_series()
            self._update_stats_label()
        except Exception:
            pass

    def _on_roi_changed(self):
        if getattr(self, 'radio_single', None) and self.radio_single.isChecked():
            self._compute_and_plot_single_frame()
        else:
            self._compute_time_series()
        self._update_stats_label()

    def _on_slider_changed(self, value):
        # Update Workbench frame and vertical line
        try:
            if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                self.main.frame_spinbox.setValue(int(value))
        except Exception:
            pass
        try:
            # Update line position in current x-space
            x_sel = _combo_key(self.x_select, 'time') if hasattr(self, 'x_select') else 'time'
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
        # In single-frame mode, recompute projection for new frame
        if getattr(self, 'radio_single', None) and self.radio_single.isChecked():
            self._compute_and_plot_single_frame()
            return
        # Keep slider and line in sync with Workbench
        try:
            self.slider.blockSignals(True)
            self.slider.setValue(int(value))
        except Exception:
            pass
        try:
            # Update line position in current x-space
            x_sel = _combo_key(self.x_select, 'time') if hasattr(self, 'x_select') else 'time'
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
            x_sel = _combo_key(self.x_select, 'time') if hasattr(self, 'x_select') else 'time'
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
            x_sel = _combo_key(self.x_select, 'time') if hasattr(self, 'x_select') else 'time'
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
