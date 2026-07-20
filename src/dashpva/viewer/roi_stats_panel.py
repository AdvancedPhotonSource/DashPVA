"""Consolidated ROI stats + plots window for the area detector viewer.

Replaces the per-ROI Stats/Plot popups with one window that shows every
*available* ROI — EPICS ROI 1-4 (from ``viewer.stats_data``) plus up to 5
locally-computed manual ROIs — as a live stats table and a time-series plot with
two modes (per-curve: one ROI's 5 stats; overlaid: one stat across ROIs). The
panel is a pure view: all pixel work and the manual-ROI broadcast happen on the
viewer/reader side; here we only read cached ``stats_data`` into bounded deques.
"""
from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dashpva.gui.theme_colors import ROI_COLORS
from dashpva.utils.stats_analysis import calculate_1d_analysis

# Display name -> stats_data field suffix.
_STATS = [('Total', 'Total_RBV'), ('Min', 'MinValue_RBV'), ('Max', 'MaxValue_RBV'),
          ('Mean', 'MeanValue_RBV'), ('Sigma', 'Sigma_RBV')]
_STAT_NAMES = [n for n, _ in _STATS]
_STAT_FIELDS = [f for _, f in _STATS]
# COM columns are shown for manual ROIs (M1-M5) only; EPICS ROIs 1-4 show a dash.
_COM = [('COM X', 'ComX_RBV'), ('COM Y', 'ComY_RBV')]
_COM_NAMES = [n for n, _ in _COM]
_COM_FIELDS = [f for _, f in _COM]
# Fixed colors for the 5 stat curves in per-curve mode.
_STAT_CURVE_COLORS = [(31, 119, 180), (44, 160, 44), (214, 39, 40),
                      (255, 127, 14), (148, 103, 189)]
# Manual ROIs share amber, so tell them apart in the overlaid plot by marker
# symbol (dashed lines were hard to read) — star / circle / triangle / square / diamond.
_MANUAL_SYMBOLS = ['star', 'o', 't', 's', 'd']
_ANALYSIS = [('Peak', 'peak_intensity', 4), ('Peak frame', 'peak_pos', 0),
             ('COM frame', 'com_pos', 0), ('COM val', 'com_intensity', 4),
             ('FWHM', 'fwhm_value', 2), ('FWHM ctr', 'fwhm_center', 0)]


class RoiStatsPanel(QWidget):
    """One panel for all ROI stats + time-series plots (EPICS 1-4 + manual).

    Housed either embedded in the ROI dock or popped out into a standalone window
    (the viewer reparents this same widget between the two hosts)."""

    MAX_POINTS = 500

    def __init__(self, parent, timer):
        super().__init__(parent)
        self.viewer = parent
        self.timer = timer
        self._paused = False
        self._frame_index = 0
        self._last_frames = self._current_frames()
        self._max_points = self.MAX_POINTS
        self._series = {}     # roi_key -> {'frames': deque, field: deque, ...}
        self._curves = {}     # curve id -> pg.PlotDataItem
        self._avail_sig = None
        self._build_ui()
        self.timer.timeout.connect(self._update)

    # --------------------------------------------------------------- helpers
    def _current_frames(self) -> int:
        r = getattr(self.viewer, 'reader', None)
        return int(getattr(r, 'frames_received', 0) or 0)

    def _prefix(self) -> str:
        r = getattr(self.viewer, 'reader', None)
        return getattr(r, 'pva_prefix', '') if r is not None else ''

    def _available_rois(self) -> list:
        """EPICS ROI 1-4 that are connected + existing manual ROIs (trim the rest)."""
        prefix = self._prefix()
        rois = []
        for i in range(1, 5):
            if self.viewer.stats_data.get(f'{prefix}:Stats{i}:Total_RBV') is not None:
                rois.append({'key': f'Stats{i}', 'label': f'ROI{i}',
                             'color': ROI_COLORS[i - 1], 'symbol': None})
        for e in getattr(self.viewer, 'manual_rois', []):
            rois.append({'key': e['key'], 'label': f"M{e['n']}", 'color': ROI_COLORS[4],
                         'symbol': _MANUAL_SYMBOLS[(e['n'] - 1) % len(_MANUAL_SYMBOLS)]})
        return rois

    # -------------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel('Mode:'))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(['Per-curve', 'Overlaid'])
        self.cmb_mode.currentIndexChanged.connect(self._rebuild_curves)
        top.addWidget(self.cmb_mode)
        top.addWidget(QLabel('ROI:'))
        self.cmb_roi = QComboBox()
        self.cmb_roi.currentIndexChanged.connect(self._rebuild_curves)
        top.addWidget(self.cmb_roi)
        top.addWidget(QLabel('Stat:'))
        self.cmb_stat = QComboBox()
        self.cmb_stat.addItems(_STAT_NAMES)
        self.cmb_stat.currentIndexChanged.connect(self._rebuild_curves)
        top.addWidget(self.cmb_stat)
        top.addStretch()
        self.btn_detach = QPushButton('Detach ⧉')
        self.btn_detach.setToolTip('Pop the panel out into a standalone window')
        self.btn_detach.clicked.connect(self._request_detach)
        top.addWidget(self.btn_detach)
        layout.addLayout(top)

        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'Frame')
        self.plot_item.setLabel('left', 'Value')
        self.plot_item.addLegend(offset=(10, 10))
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        self.plot_widget.setMinimumHeight(160)   # stays usable when embedded/narrow
        # Manual zoom/pan disables auto-range so the user's view sticks.
        self.plot_item.getViewBox().sigRangeChangedManually.connect(self._on_manual_zoom)
        layout.addWidget(self.plot_widget, stretch=3)

        # Row 1: plot controls.
        ctrl1 = QHBoxLayout()
        self.btn_pause = QPushButton('Pause')
        self.btn_pause.clicked.connect(self._toggle_pause)
        ctrl1.addWidget(self.btn_pause)
        self.btn_clear = QPushButton('Clear')
        self.btn_clear.clicked.connect(self._clear)
        ctrl1.addWidget(self.btn_clear)
        self.chk_autoscale_x = QCheckBox('Auto X')
        self.chk_autoscale_x.setChecked(True)
        self.chk_autoscale_x.setToolTip('Follow data on X (auto-unchecks when you zoom/pan)')
        ctrl1.addWidget(self.chk_autoscale_x)
        self.chk_autoscale_y = QCheckBox('Auto Y')
        self.chk_autoscale_y.setChecked(True)
        ctrl1.addWidget(self.chk_autoscale_y)
        ctrl1.addWidget(QLabel('Window:'))
        self.spn_window = QSpinBox()
        self.spn_window.setRange(50, 10000)
        self.spn_window.setValue(self._max_points)
        self.spn_window.setSuffix(' pts')
        self.spn_window.valueChanged.connect(self._on_window)
        ctrl1.addWidget(self.spn_window)
        ctrl1.addStretch()
        layout.addLayout(ctrl1)

        # Row 2: manual-ROI controls (own row so the panel stays readable when
        # embedded in the narrow ROI dock).
        ctrl2 = QHBoxLayout()
        self.btn_add = QPushButton('+ Add manual ROI')
        self.btn_add.clicked.connect(self._add_manual)
        ctrl2.addWidget(self.btn_add)
        self.btn_remove = QPushButton('Remove manual')
        self.btn_remove.setToolTip('Remove the selected manual ROI (or the most recent one)')
        self.btn_remove.clicked.connect(self._remove_manual)
        ctrl2.addWidget(self.btn_remove)
        self.chk_broadcast = QCheckBox('Broadcast as PV')
        self.chk_broadcast.setToolTip('Publish all M1-M5 stats on one soft PV')
        self.chk_broadcast.toggled.connect(self._on_broadcast)
        ctrl2.addWidget(self.chk_broadcast)
        ctrl2.addStretch()
        layout.addLayout(ctrl2)

        self.table = QTableWidget(0, 1 + len(_STATS) + len(_COM))
        self.table.setHorizontalHeaderLabels(['ROI'] + _STAT_NAMES + _COM_NAMES)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        box = QGroupBox('Analysis (selected ROI, Total curve)')
        grid = QGridLayout(box)
        self._analysis_labels = {}
        for col, (text, key, _) in enumerate(_ANALYSIS):
            grid.addWidget(QLabel(f'<b>{text}</b>'), 0, col, alignment=Qt.AlignCenter)
            lbl = QLabel('--')
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._analysis_labels[key] = lbl
            grid.addWidget(lbl, 1, col)
        self.lbl_empty = QLabel('No ROIs available — start live view or add a manual ROI.')
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_empty)
        layout.addWidget(box)

        self._refresh_availability(force=True)
        self.setMinimumWidth(340)   # keep controls legible when docked narrow

    # ------------------------------------------------- detach / re-embed
    def _request_detach(self):
        toggler = getattr(self.viewer, 'toggle_roi_stats_detached', None)
        if callable(toggler):
            toggler()

    def set_detached(self, detached: bool) -> None:
        """Reflect where the panel currently lives on the detach/dock button."""
        self.btn_detach.setText('Dock ⧉' if detached else 'Detach ⧉')
        self.btn_detach.setToolTip(
            'Re-embed the panel in the ROI dock' if detached
            else 'Pop the panel out into a standalone window')

    # ------------------------------------------------------- availability
    def _refresh_availability(self, force: bool = False):
        rois = self._available_rois()
        sig = tuple((r['key'], r['label']) for r in rois)
        if not force and sig == self._avail_sig:
            return
        self._avail_sig = sig
        self._rois = rois
        # prune series for manual ROIs that no longer exist
        keep = {r['key'] for r in rois} | {f'Stats{i}' for i in range(1, 5)}
        for k in list(self._series):
            if k not in keep:
                del self._series[k]
        # ROI selector (preserve selection where possible)
        prev = self.cmb_roi.currentData()
        self.cmb_roi.blockSignals(True)
        self.cmb_roi.clear()
        for r in rois:
            self.cmb_roi.addItem(r['label'], r['key'])
        if prev is not None:
            idx = self.cmb_roi.findData(prev)
            if idx >= 0:
                self.cmb_roi.setCurrentIndex(idx)
        self.cmb_roi.blockSignals(False)
        # table rows
        self.table.setRowCount(len(rois))
        for row, r in enumerate(rois):
            item = QTableWidgetItem(r['label'])
            item.setForeground(pg.mkColor(r['color']))
            self.table.setItem(row, 0, item)
            for c in range(len(_STATS) + len(_COM)):
                self.table.setItem(row, c + 1, QTableWidgetItem('--'))
        self.lbl_empty.setVisible(not rois)
        manual_n = len(getattr(self.viewer, 'manual_rois', []))
        self.btn_add.setEnabled(manual_n < getattr(self.viewer, 'MAX_MANUAL_ROIS', 5))
        self.btn_remove.setEnabled(manual_n > 0)   # nothing to remove -> disabled
        self._rebuild_curves()

    def _rebuild_curves(self, *_):
        for cur in self._curves.values():
            self.plot_item.removeItem(cur)
        self._curves = {}
        per_curve = self.cmb_mode.currentIndex() == 0
        self.cmb_roi.setEnabled(True)          # drives per-curve plot + analysis
        self.cmb_stat.setEnabled(not per_curve)
        if per_curve:
            key = self.cmb_roi.currentData()
            if key is not None:
                for name, color in zip(_STAT_NAMES, _STAT_CURVE_COLORS):
                    self._curves[name] = self.plot_item.plot(
                        [], [], pen=pg.mkPen(color=color, width=2), name=name)
        else:
            for r in getattr(self, '_rois', []):
                self._curves[r['key']] = self.plot_item.plot(
                    [], [], pen=pg.mkPen(color=r['color'], width=2),
                    name=r['label'], symbol=r.get('symbol'),
                    symbolBrush=r['color'], symbolPen=None, symbolSize=7)
        self._redraw()

    # ------------------------------------------------------------- update
    def _update(self):
        if not self.isVisible():
            return
        self._refresh_availability()
        if self._paused:
            return
        frames = self._current_frames()
        if frames <= self._last_frames:
            self._update_table()
            return
        self._last_frames = frames
        prefix = self._prefix()
        for r in self._rois:
            s = self._series.setdefault(
                r['key'], {'frames': deque(maxlen=self._max_points),
                           **{f: deque(maxlen=self._max_points) for f in _STAT_FIELDS}})
            s['frames'].append(self._frame_index)
            for f in _STAT_FIELDS:
                s[f].append(float(self.viewer.stats_data.get(f"{prefix}:{r['key']}:{f}", 0.0)))
        self._frame_index += 1
        self._redraw()
        self._update_table()
        self._update_analysis()

    def _redraw(self):
        per_curve = self.cmb_mode.currentIndex() == 0
        if per_curve:
            key = self.cmb_roi.currentData()
            s = self._series.get(key)
            for name, field in _STATS:
                cur = self._curves.get(name)
                if cur is None:
                    continue
                if s:
                    cur.setData(list(s['frames']), list(s[field]))
                else:
                    cur.setData([], [])
        else:
            field = _STAT_FIELDS[self.cmb_stat.currentIndex()]
            for r in getattr(self, '_rois', []):
                cur = self._curves.get(r['key'])
                s = self._series.get(r['key'])
                if cur is None:
                    continue
                if s:
                    cur.setData(list(s['frames']), list(s[field]))
                else:
                    cur.setData([], [])
        if self.chk_autoscale_x.isChecked():
            self.plot_item.enableAutoRange(axis='x')
        if self.chk_autoscale_y.isChecked():
            self.plot_item.enableAutoRange(axis='y')

    def _update_table(self):
        prefix = self._prefix()
        for row, r in enumerate(getattr(self, '_rois', [])):
            for c, field in enumerate(_STAT_FIELDS):
                val = self.viewer.stats_data.get(f"{prefix}:{r['key']}:{field}")
                item = self.table.item(row, c + 1)
                if item is not None:
                    item.setText('--' if val is None else f'{float(val):.3g}')
            # COM columns: manual ROIs (M1-M5) only; EPICS ROIs 1-4 show an em-dash.
            is_manual = r['key'].startswith('Manual')
            for c, field in enumerate(_COM_FIELDS):
                item = self.table.item(row, 1 + len(_STAT_FIELDS) + c)
                if item is None:
                    continue
                if not is_manual:
                    item.setText('—')
                    continue
                val = self.viewer.stats_data.get(f"{prefix}:{r['key']}:{field}")
                if val is None:
                    item.setText('--')
                else:
                    # Pixel COM: fixed-point (0000.00), no scientific notation up to a
                    # 10k x 10k detector; only larger values fall back to scientific.
                    v = float(val)
                    item.setText(f'{v:.2f}' if abs(v) < 10000 else f'{v:.3g}')

    def _update_analysis(self):
        key = self.cmb_roi.currentData()
        s = self._series.get(key)
        if not s:
            return
        frames = list(s['frames'])
        totals = list(s['Total_RBV'])
        n = min(len(frames), len(totals))
        if n < 3:
            return
        result = calculate_1d_analysis(frames[:n], totals[:n])
        if not result:
            return
        for _text, key2, digits in _ANALYSIS:
            lbl = self._analysis_labels.get(key2)
            if lbl is not None:
                lbl.setText(f'{result.get(key2, 0.0):.{digits}f}')

    # ---------------------------------------------------------- actions
    def _add_manual(self):
        key = self.viewer.add_manual_roi()
        self._refresh_availability(force=True)
        if key:                                    # point the dropdown at the new ROI
            idx = self.cmb_roi.findData(key)
            if idx >= 0:
                self.cmb_roi.setCurrentIndex(idx)

    @staticmethod
    def _manual_remove_target(selected_key, manual_rois):
        """Key that 'Remove manual' should delete: the dropdown selection when it is
        a manual ROI, otherwise the most-recently-added manual ROI (highest slot),
        or None when there are none. Lets the button work even when the shared ROI
        dropdown is sitting on an EPICS ROI (the reason it 'did nothing' before)."""
        if selected_key and str(selected_key).startswith('Manual'):
            return selected_key
        if not manual_rois:
            return None
        return max(manual_rois, key=lambda e: e['n'])['key']

    def _remove_manual(self):
        key = self._manual_remove_target(
            self.cmb_roi.currentData(), getattr(self.viewer, 'manual_rois', []))
        if key:
            self.viewer.remove_manual_roi(key)
            self._refresh_availability(force=True)

    def _on_broadcast(self, checked):
        if not checked:
            self.viewer.stop_manual_broadcast()
            return
        if self.viewer.start_manual_broadcast():
            QMessageBox.information(
                self, 'Broadcasting Manual ROI Stats',
                'Manual ROI stats (M1-M5: Total, Min, Max, Mean, Sigma + active '
                'flags) are broadcasting on a single PV:\n\n'
                f'{self.viewer._manual_pv_channel}')
        else:
            self.chk_broadcast.blockSignals(True)
            self.chk_broadcast.setChecked(False)
            self.chk_broadcast.blockSignals(False)
            QMessageBox.warning(
                self, 'Broadcast Failed',
                'Could not start the PVA server (is pvaccess available?).')

    def _toggle_pause(self):
        self._paused = not self._paused
        self.btn_pause.setText('Resume' if self._paused else 'Pause')

    def _on_manual_zoom(self, *_):
        """User zoomed/panned the plot -> stop auto-ranging so their view sticks."""
        for chk in (self.chk_autoscale_x, self.chk_autoscale_y):
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)

    def _clear(self):
        self._series = {}
        self._frame_index = 0
        self._last_frames = self._current_frames()
        self._redraw()
        for lbl in self._analysis_labels.values():
            lbl.setText('--')

    def _on_window(self, value):
        self._max_points = value
        for s in self._series.values():
            for k in list(s):
                s[k] = deque(s[k], maxlen=value)
