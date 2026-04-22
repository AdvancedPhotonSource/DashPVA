from PyQt5.QtWidgets import QWidget, QVBoxLayout, QRadioButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from viewer.core.docks.base_dock import BaseDock


class PlotModeDock(BaseDock):
    mode_changed    = pyqtSignal(str)  # 'post_scan' | 'realtime' | 'per_frame'
    plot_timer_fired = pyqtSignal(str) # fired at interval when a new frame is pending

    _PLOT_INTERVAL_MS = 200

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Plot Mode", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self._new_frame_pending = False
        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        self.rb_post_scan = QRadioButton("Post-scan (on complete)")
        self.rb_realtime  = QRadioButton("Realtime (cumulative)")
        self.rb_per_frame = QRadioButton("Per-frame (single frame)")
        self.rb_post_scan.setChecked(True)
        for rb in (self.rb_post_scan, self.rb_realtime, self.rb_per_frame):
            layout.addWidget(rb)
            rb.toggled.connect(self._on_radio_toggled)
        layout.addStretch()
        self.setWidget(container)

    def _setup_timer(self):
        self.timer_plot = QTimer()
        self.timer_plot.setInterval(self._PLOT_INTERVAL_MS)
        self.timer_plot.timeout.connect(self._on_timer_plot)

    def _on_radio_toggled(self, checked: bool):
        if checked:
            self.mode_changed.emit(self.current_mode)

    def _on_timer_plot(self):
        if not self._new_frame_pending:
            return
        self._new_frame_pending = False
        self.plot_timer_fired.emit(self.current_mode)

    @property
    def current_mode(self) -> str:
        if self.rb_realtime.isChecked():
            return 'realtime'
        if self.rb_per_frame.isChecked():
            return 'per_frame'
        return 'post_scan'

    @property
    def is_post_scan(self) -> bool:
        return self.rb_post_scan.isChecked()

    @property
    def is_realtime(self) -> bool:
        return self.rb_realtime.isChecked()

    @property
    def is_per_frame(self) -> bool:
        return self.rb_per_frame.isChecked()

    def notify_new_frame(self):
        self._new_frame_pending = True

    def start_plot_timer(self):
        self._new_frame_pending = False
        self.timer_plot.start()

    def stop_plot_timer(self):
        self.timer_plot.stop()
