from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QGroupBox, QGridLayout, QSpinBox
)
from PyQt5.QtCore import Qt
from utils.stats_analysis import calculate_1d_analysis


class RoiStatsPlotDialog(QDialog):
    """
    Real-time plotting dialog for ROI statistics.

    Displays time-series curves (Total, Min, Max, Mean, Sigma) that update
    as new frames arrive.  Also computes and displays Peak, COM, and FWHM
    from the Total curve.

    Connects to the parent viewer's timer_labels for update ticks and reads
    live values from parent.stats_data.
    """

    MAX_POINTS_DEFAULT = 500
    # Update analysis every N ticks to avoid computing every 10 ms
    ANALYSIS_INTERVAL = 20

    STAT_KEYS = ['TOTAL', 'MIN', 'MAX', 'MEAN', 'SIGMA']
    STAT_SUFFIXES = {
        'TOTAL': 'Total_RBV',
        'MIN': 'MinValue_RBV',
        'MAX': 'MaxValue_RBV',
        'MEAN': 'MeanValue_RBV',
        'SIGMA': 'Sigma_RBV',
    }
    STAT_COLORS = {
        'TOTAL': (31, 119, 180),    # blue
        'MIN': (44, 160, 44),       # green
        'MAX': (214, 39, 40),       # red
        'MEAN': (255, 127, 14),     # orange
        'SIGMA': (148, 103, 189),   # purple
    }

    def __init__(self, parent: 'QMainWindow', stats_text: str, timer: 'QTimer'):
        super().__init__()
        self.parent_viewer = parent
        self.stats_text = stats_text
        self.prefix = self.parent_viewer.reader.pva_prefix
        self.timer_labels = timer

        self.setWindowTitle(f'{stats_text} - Live Stats Plot')
        self.resize(800, 600)

        self._paused = False
        self._tick_count = 0
        self._frame_index = 0
        self._max_points = self.MAX_POINTS_DEFAULT

        # Data storage — one deque per stat
        self._frames = deque(maxlen=self._max_points)
        self._data = {key: deque(maxlen=self._max_points) for key in self.STAT_KEYS}

        self._build_ui()
        self.timer_labels.timeout.connect(self.update_plot)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Plot widget
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'Frame')
        self.plot_item.setLabel('left', 'Value')
        self.plot_item.addLegend(offset=(10, 10))
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        layout.addWidget(self.plot_widget, stretch=3)

        # Create curves
        self._curves = {}
        for key in self.STAT_KEYS:
            pen = pg.mkPen(color=self.STAT_COLORS[key], width=2)
            self._curves[key] = self.plot_item.plot([], [], pen=pen, name=key.capitalize())

        # Controls row
        ctrl_layout = QHBoxLayout()

        self.btn_refresh = QPushButton('Refresh / Clear')
        self.btn_refresh.clicked.connect(self.clear_data)
        ctrl_layout.addWidget(self.btn_refresh)

        self.btn_pause = QPushButton('Pause')
        self.btn_pause.clicked.connect(self.toggle_pause)
        ctrl_layout.addWidget(self.btn_pause)

        self.chk_autoscale = QCheckBox('Auto-scale Y')
        self.chk_autoscale.setChecked(True)
        self.chk_autoscale.stateChanged.connect(self._on_autoscale_changed)
        ctrl_layout.addWidget(self.chk_autoscale)

        ctrl_layout.addWidget(QLabel('Window:'))
        self.spn_window = QSpinBox()
        self.spn_window.setRange(50, 10000)
        self.spn_window.setValue(self._max_points)
        self.spn_window.setSuffix(' pts')
        self.spn_window.valueChanged.connect(self._on_window_changed)
        ctrl_layout.addWidget(self.spn_window)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Analysis stats table
        analysis_box = QGroupBox('Analysis (from Total curve)')
        grid = QGridLayout(analysis_box)
        self._analysis_labels = {}
        headers = [
            ('Peak Value', 'peak_intensity'),
            ('Peak Frame', 'peak_pos'),
            ('COM Frame', 'com_pos'),
            ('COM Value', 'com_intensity'),
            ('FWHM', 'fwhm_value'),
            ('FWHM Center', 'fwhm_center'),
        ]
        for col, (label_text, key) in enumerate(headers):
            grid.addWidget(QLabel(f'<b>{label_text}</b>'), 0, col, alignment=Qt.AlignCenter)
            val_label = QLabel('--')
            val_label.setAlignment(Qt.AlignCenter)
            self._analysis_labels[key] = val_label
            grid.addWidget(val_label, 1, col)

        layout.addWidget(analysis_box)

    # -------------------------------------------------------------- Slots
    def update_plot(self):
        if self._paused:
            return

        # Read latest values from parent
        has_new = False
        for key in self.STAT_KEYS:
            suffix = self.STAT_SUFFIXES[key]
            pv_name = f'{self.prefix}:{self.stats_text}:{suffix}'
            value = self.parent_viewer.stats_data.get(pv_name, None)
            if value is not None:
                self._data[key].append(float(value))
                has_new = True
            elif len(self._data[key]) > 0:
                # Repeat last value to keep curves aligned
                self._data[key].append(self._data[key][-1])

        if has_new:
            self._frames.append(self._frame_index)
            self._frame_index += 1

        # Update curves
        frames = list(self._frames)
        for key in self.STAT_KEYS:
            values = list(self._data[key])
            n = min(len(frames), len(values))
            if n > 0:
                self._curves[key].setData(frames[:n], values[:n])

        # Auto-scale
        if self.chk_autoscale.isChecked():
            self.plot_item.enableAutoRange(axis='y')

        # Periodic analysis update
        self._tick_count += 1
        if self._tick_count % self.ANALYSIS_INTERVAL == 0:
            self._update_analysis()

    def _update_analysis(self):
        frames = list(self._frames)
        total_values = list(self._data['TOTAL'])
        n = min(len(frames), len(total_values))
        if n < 3:
            return

        result = calculate_1d_analysis(frames[:n], total_values[:n])
        if result is None:
            return

        for key, label in self._analysis_labels.items():
            val = result.get(key, 0.0)
            if key in ('peak_pos', 'com_pos', 'fwhm_center'):
                label.setText(f'{val:.0f}')
            else:
                label.setText(f'{val:.4f}')

    def clear_data(self):
        self._frame_index = 0
        self._frames.clear()
        for key in self.STAT_KEYS:
            self._data[key].clear()
        for key in self.STAT_KEYS:
            self._curves[key].setData([], [])
        for label in self._analysis_labels.values():
            label.setText('--')

    def toggle_pause(self):
        self._paused = not self._paused
        self.btn_pause.setText('Resume' if self._paused else 'Pause')

    def _on_autoscale_changed(self, state):
        if state == Qt.Checked:
            self.plot_item.enableAutoRange(axis='y')
        else:
            self.plot_item.disableAutoRange(axis='y')

    def _on_window_changed(self, value):
        self._max_points = value
        self._frames = deque(self._frames, maxlen=value)
        for key in self.STAT_KEYS:
            self._data[key] = deque(self._data[key], maxlen=value)

    # ------------------------------------------------------------ Cleanup
    def closeEvent(self, event):
        try:
            self.timer_labels.timeout.disconnect(self.update_plot)
        except TypeError:
            pass
        self.parent_viewer.stats_plot_dialogs[self.stats_text] = None
        super().closeEvent(event)
