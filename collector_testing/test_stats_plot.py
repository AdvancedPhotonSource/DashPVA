#!/usr/bin/env python3
"""
Standalone test for RoiStatsPlotDialog.

Simulates stats data without requiring EPICS or PVA.
Run from the DashPVA root directory:
    python collector_testing/test_stats_plot.py
"""
import sys
import os
import math
import random

# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from viewer.roi_stats_plot import RoiStatsPlotDialog


class FakeReader:
    """Mimics the PVAReader interface for testing."""
    pva_prefix = 'SIM'


class FakeViewer(QMainWindow):
    """
    Minimal stand-in for DiffractionImageWindow.
    Generates synthetic stats data on a timer.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Stats Plot Test')
        self.reader = FakeReader()
        self.stats_data = {}
        self.stats_plot_dialogs = {}

        # Timer that simulates incoming stats at ~10 Hz
        self.timer_labels = QTimer()
        self.timer_labels.timeout.connect(self._generate_fake_data)

        self._frame = 0

        # Simple UI
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn = QPushButton('Open Stats1 Plot')
        btn.clicked.connect(self._open_plot)
        layout.addWidget(btn)

        btn_start = QPushButton('Start Fake Data')
        btn_start.clicked.connect(lambda: self.timer_labels.start(100))  # 10 Hz
        layout.addWidget(btn_start)

        btn_stop = QPushButton('Stop Fake Data')
        btn_stop.clicked.connect(self.timer_labels.stop)
        layout.addWidget(btn_stop)

        self.setCentralWidget(widget)

    def _generate_fake_data(self):
        """Generate synthetic stats that look like a scan peak."""
        t = self._frame
        # Gaussian-like peak centered around frame 150
        peak = 10000 * math.exp(-0.5 * ((t % 300 - 150) / 30) ** 2)
        noise = random.gauss(0, 200)

        total = peak + noise + 500
        mean_val = total / 1024  # pretend 1024 pixels in ROI
        max_val = total * 0.1 + random.gauss(0, 50)
        min_val = random.gauss(5, 2)
        sigma = abs(random.gauss(50, 10))

        prefix = self.reader.pva_prefix
        self.stats_data[f'{prefix}:Stats1:Total_RBV'] = total
        self.stats_data[f'{prefix}:Stats1:MinValue_RBV'] = min_val
        self.stats_data[f'{prefix}:Stats1:MaxValue_RBV'] = max_val
        self.stats_data[f'{prefix}:Stats1:MeanValue_RBV'] = mean_val
        self.stats_data[f'{prefix}:Stats1:Sigma_RBV'] = sigma

        self._frame += 1

    def _open_plot(self):
        existing = self.stats_plot_dialogs.get('Stats1')
        if existing is not None:
            existing.close()
        self.stats_plot_dialogs['Stats1'] = RoiStatsPlotDialog(
            parent=self,
            stats_text='Stats1',
            timer=self.timer_labels
        )
        self.stats_plot_dialogs['Stats1'].show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = FakeViewer()
    viewer.show()
    sys.exit(app.exec_())
