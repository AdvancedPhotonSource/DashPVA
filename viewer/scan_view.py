import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import toml
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout
import pyqtgraph as pg

from utils import PVAReader, HDF5Handler
from utils.log_manager import LogMixin
from epics import caput

class ScanMonitorWindow(QMainWindow, LogMixin):
    signal_start_monitor = pyqtSignal()
    signal_trigger_save = pyqtSignal(bool, bool, bool)  # clear_caches, write_temp, write_output
    
    def __init__(self, channel: str = "", config_filepath: str = ""):
        super(ScanMonitorWindow, self).__init__()
        uic.loadUi('gui/scan_view.ui', self)
        # Title comes from UI; ensure consistent naming in code comments
        # Initialize structured logger via LogMixin (defaults to module.class name)
        try:
            self.set_log_manager()
        except Exception:
            pass

        self.channel = channel
        self.config_filepath = config_filepath
        self.scan_state = False
        
        # Track applied state for UI labels
        self.applied_channel = None
        self.applied_config = None
        self._last_frames_received = 0
        
        # Define Threads
        self.reader_thread = QThread()
        self.writer_thread = QThread()

        self.reader: PVAReader = None
        self.h5_handler: HDF5Handler = None
        
        # Timer for updating info display
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self._update_info_display)

        # Graph state
        self.graph_plot = None
        self.graph_curve = None
        self.graph_x = []
        self.graph_y = []
        self._frames_baseline = 0
        self.graph_window_seconds = 60  # sliding window length; newest point centered
        # Separate timeline for activity monitor (distinct from actual scan time)
        self.activity_start_time = None
        # Track how long the monitor has been actively listening
        self.listening_start_time = None
        
        # Scan timing variables
        self.scan_start_time = None
        self.scan_end_time = None
        self.last_scan_completion_time = None

        # Setup Initial UI State
        self._setup_ui_elements()
        self._setup_graph()

    def _setup_ui_elements(self):
        if hasattr(self, 'label_mode'):
            self.label_mode.setText("")
        if hasattr(self, 'checkbox_write_temp'):
            self.checkbox_write_temp.stateChanged.connect(self._update_save_warning)
        if hasattr(self, 'checkbox_write_output'):
            self.checkbox_write_output.stateChanged.connect(self._update_save_warning)
        self._update_save_warning()
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: off')
            self._apply_indicator_style()
        if hasattr(self, 'label_listening'):
            # Initialize listening elapsed time display as mm:ss
            self.label_listening.setText('00:00')
            self._apply_listening_style(False)

        self.lineedit_channel.setText(self.channel or "")
        self.lineedit_channel.textChanged.connect(self._on_channel_changed) 
        self.lineedit_config.setText(self.config_filepath or "")
        self.lineedit_config.textChanged.connect(self._on_config_path_changed) 

        self.btn_browse_config.clicked.connect(self._on_browse_config_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        
        self._update_info_display()

    def _setup_graph(self):
        """Initialize the PyQtGraph PlotWidget inside the placeholder widget."""
        try:
            if hasattr(self, 'widget_graph') and self.widget_graph is not None:
                # Create a layout for the placeholder widget if it doesn't have one
                layout = self.widget_graph.layout() if hasattr(self.widget_graph, 'layout') else None
                if layout is None:
                    layout = QVBoxLayout(self.widget_graph)
                    self.widget_graph.setLayout(layout)

                # Create and configure the plot
                self.graph_plot = pg.PlotWidget(background='w')
                self.graph_plot.showGrid(x=True, y=True)
                self.graph_plot.setLabel('bottom', 'Time', units='s')
                self.graph_plot.setLabel('left', 'Frames collected')
                # Keep Y auto-range; control X via sliding window
                try:
                    self.graph_plot.enableAutoRange(x=False, y=True)
                except Exception:
                    pass
                self.graph_curve = self.graph_plot.plot([], [], pen=pg.mkPen(color=(30, 144, 255), width=2))

                # Center indicator line; we'll place it at the center of the X range (latest time)
                try:
                    self.center_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(128, 128, 128), style=Qt.DashLine))
                    self.graph_plot.addItem(self.center_line)
                except Exception:
                    self.center_line = None

                layout.addWidget(self.graph_plot)
                self._reset_graph()
        except Exception as e:
            try:
                self.logger.error(f"Graph setup error: {e}")
            except Exception:
                pass

    # ================================================================================================
    # CORE LOGIC & THREADING
    # ================================================================================================

    def _on_apply_clicked(self) -> None:
        """Initializes Reader and Writer on separate threads."""
        if not self.channel or not self.config_filepath:
            return

        self._cleanup_existing_instances()

        # Point settings at selected TOML so the writer's converter can find it.
        # Wrapped in try/except so a settings failure cannot abort monitor startup.
        try:
            import settings
            settings.set_locator(self.config_filepath)
            settings.reload()
        except Exception as e:
            pass

        try:
            # 1. Create instances
            self.reader = PVAReader(
                input_channel=self.channel,
                config_filepath=self.config_filepath,
                viewer_type='image'
            )
            
            self.h5_handler = HDF5Handler(
                file_path="",
                pva_reader=self.reader
            )

            # 2. Move to specific worker threads
            self.reader.moveToThread(self.reader_thread)
            self.h5_handler.moveToThread(self.writer_thread)

            # 3. Connect Signals with QueuedConnection to bridge thread boundaries
            self.reader.reader_scan_complete.connect(self._on_reader_scan_complete, Qt.QueuedConnection)
            self.h5_handler.hdf5_writer_finished.connect(self._on_writer_finished, Qt.QueuedConnection)
            self.signal_start_monitor.connect(self.reader.start_channel_monitor, Qt.QueuedConnection)
            self.signal_trigger_save.connect(self.h5_handler.save_to_h5, Qt.QueuedConnection)

            if hasattr(self.reader, 'scan_state_changed'):
                self.reader.scan_state_changed.connect(self._on_scan_state_changed, Qt.QueuedConnection)
            
            # 4. Start Thread Event Loops
            self.reader_thread.start()
            self.writer_thread.start()

            # 5. Begin Monitoring
            self.signal_start_monitor.emit()

            # 6. Update UI Tracking
            self.applied_channel = self.channel
            self.applied_config = self.config_filepath
            if hasattr(self, 'label_listening'):
                self.label_listening.setText('True')
                self._apply_listening_style(True)
            
            self.info_timer.start(1000) 
            self._update_info_display()
            # For continuous monitoring, start a fresh graph timeline on apply
            self._reset_graph()
            # Initialize a timeline for the activity monitor (separate from scan time)
            self.activity_start_time = datetime.now()
            # Reset frames baseline to current reader count if available
            try:
                self._frames_baseline = int(getattr(self.reader, 'frames_received', 0) or 0)
            except Exception:
                self._frames_baseline = 0
            
        except Exception as e:
            try:
                self.logger.error(f"Apply Error: {e}")
            except Exception:
                pass
            self.reader = None
            self.h5_handler = None

    def _on_reader_scan_complete(self) -> None:
        """Slot executed when PVAReader emits the completion signal."""
        try:
            self.logger.info(f"reader_scan_complete received at {datetime.now()}")
        except Exception:
            pass
        self._trigger_automatic_save()

    def _trigger_automatic_save(self) -> None:
        """Triggers the HDF5Writer save process."""
        if self.h5_handler:
            # Read checkbox states; default to True if widgets missing
            write_temp = True
            write_output = True
            try:
                if hasattr(self, 'checkbox_write_temp') and self.checkbox_write_temp is not None:
                    write_temp = bool(self.checkbox_write_temp.isChecked())
                if hasattr(self, 'checkbox_write_output') and self.checkbox_write_output is not None:
                    write_output = bool(self.checkbox_write_output.isChecked())
            except Exception:
                pass

            try:
                self.logger.info("Triggering HDF5Handler.save_to_h5...")
            except Exception:
                pass
            self.signal_trigger_save.emit(True, write_temp, write_output)

    def _on_writer_finished(self, message: str) -> None:
        """Callback when the HDF5 file is finished writing."""
        try:
            self.logger.info(message)
        except Exception:
            pass
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: off')
            self._apply_indicator_style()
            self.scan_state = False
            self._update_button_states()

    def _on_stop_scan_clicked(self) -> None:
        if self.reader is None: return
        try:
            if getattr(self.reader, 'FLAG_PV', ''):
                caput(self.reader.FLAG_PV, self.reader.STOP_SCAN)
            else:
                self.reader.stop_channel_monitor()
                # If no flag PV triggers the reader, we trigger the complete sequence manually
                self._on_reader_scan_complete() 
        except Exception as e:
            try:
                self.logger.error(f"Manual Stop Error: {e}")
            except Exception:
                pass

    def _on_start_scan_clicked(self) -> None:
        if self.reader:
            self.signal_start_monitor.emit()

    def _cleanup_existing_instances(self) -> None:
        if self.reader is not None:
            try:
                self.reader.stop_channel_monitor()
                self.reader.reader_scan_complete.disconnect()
                self.h5_handler.hdf5_writer_finished.disconnect()
            except: pass
            try:
                self.signal_start_monitor.disconnect()
            except: pass
            try:
                self.signal_trigger_save.disconnect()
            except: pass

            self.reader_thread.quit()
            self.reader_thread.wait()
            self.writer_thread.quit()
            self.writer_thread.wait()
            
        self.reader = None
        self.h5_handler = None

    # ================================================================================================
    # UI STYLING & UPDATES
    # ================================================================================================

    def _on_channel_changed(self, text):
        self.channel = text
        self.applied_channel = None
        if hasattr(self, 'label_listening'):
            # Reset listening timer when channel changes
            self.listening_start_time = None
            self.label_listening.setText('0')
            self._apply_listening_style(False)

    def _on_config_path_changed(self, text):
        self.config_filepath = text
        self.applied_config = None
        if hasattr(self, 'label_listening'):
            # Reset listening timer when config changes
            self.listening_start_time = None
            self.label_listening.setText('0')
            self._apply_listening_style(False)

    def _on_browse_config_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Config', '', 'TOML (*.toml)')
        if fname: self.lineedit_config.setText(fname)

    def _on_scan_state_changed(self, is_on: bool) -> None:
        if is_on:
            self.scan_start_time = datetime.now()
            self.scan_end_time = None
            self._reset_graph()
        else:
            self.scan_end_time = datetime.now()
            self.last_scan_completion_time = self.scan_end_time
        
        self.scan_state = is_on
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: on' if is_on else 'scan: off')
            self._apply_indicator_style()
        self._update_button_states()

    def _apply_indicator_style(self):
        if hasattr(self, 'label_indicator'):
            color = "green" if "on" in self.label_indicator.text().lower() else "red"
            self.label_indicator.setStyleSheet(f'color: {color}; font-weight: bold;')

    def _apply_listening_style(self, state):
        if hasattr(self, 'label_listening'):
            color = "green" if state else "red"
            self.label_listening.setStyleSheet(f'color: {color}; font-weight: bold;')

    def _update_button_states(self):
        if hasattr(self, 'btn_start_scan'): self.btn_start_scan.setEnabled(not self.scan_state)
        if hasattr(self, 'btn_stop_scan'): self.btn_stop_scan.setEnabled(self.scan_state)

    def _update_save_warning(self):
        if not hasattr(self, 'label_mode'):
            return
        write_temp = hasattr(self, 'checkbox_write_temp') and self.checkbox_write_temp.isChecked()
        write_output = hasattr(self, 'checkbox_write_output') and self.checkbox_write_output.isChecked()
        if not write_temp and not write_output:
            self.label_mode.setText("Warning: No save targets selected — data will not be written")
            self.label_mode.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.label_mode.setText("")
            self.label_mode.setStyleSheet("")

    def _update_info_display(self):
        """Logic for periodically refreshing UI labels based on Reader state."""
        try:
            # Update Caching Mode
            caching_mode = "Not set"
            if self.config_filepath:
                try:
                    with open(self.config_filepath, 'r') as f:
                        cfg = toml.load(f)
                    caching_mode = cfg.get('CACHE_OPTIONS', {}).get('CACHING_MODE', 'Not set')
                except: pass
            if hasattr(self, 'label_caching_mode'): self.label_caching_mode.setText(str(caching_mode))
            
            # Update Flag PV
            flag_pv = "Not set"
            if self.reader and hasattr(self.reader, 'FLAG_PV'):
                flag_pv = str(self.reader.FLAG_PV) if self.reader.FLAG_PV else "Not set"
            if hasattr(self, 'label_flag_pv'): self.label_flag_pv.setText(flag_pv)

            # Update Monitor Activity
            channel_active = "No"
            is_listening = False
            if self.reader and hasattr(self.reader, 'channel'):
                is_active = bool(self.reader.channel.isMonitorActive())
                channel_active = "Yes" if is_active else "No"
                is_listening = is_active and (self.applied_channel == self.channel)

            if hasattr(self, 'label_channel_active'):
                self.label_channel_active.setText(channel_active)

            # Update Is Caching
            if hasattr(self, 'label_is_caching') and self.reader is not None:
                is_caching = bool(getattr(self.reader, 'is_caching', False))
                self.label_is_caching.setText('Yes' if is_caching else 'No')

            # Update Listening label to show elapsed listening time (positive integers)
            if hasattr(self, 'label_listening'):
                if is_listening:
                    if self.listening_start_time is None:
                        self.listening_start_time = datetime.now()
                    elapsed = int(max(0, (datetime.now() - self.listening_start_time).total_seconds()))
                    # Format as mm:ss
                    m, s = divmod(elapsed, 60)
                    self.label_listening.setText(f"{m:02d}:{s:02d}")
                else:
                    # Reset when not listening
                    self.listening_start_time = None
                    self.label_listening.setText('00:00')
                self._apply_listening_style(is_listening)

            # Update Timing
            if self.scan_start_time:
                duration = (self.scan_end_time if self.scan_end_time else datetime.now()) - self.scan_start_time
                s = int(duration.total_seconds())
                m, s = divmod(s, 60); h, m = divmod(m, 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}"
                if not self.scan_end_time: time_str += " (running)"
                if hasattr(self, 'label_scan_time'): self.label_scan_time.setText(time_str)

            if self.last_scan_completion_time and hasattr(self, 'label_last_scan_date'):
                self.label_last_scan_date.setText(self.last_scan_completion_time.strftime("%Y-%m-%d %H:%M:%S"))
            # Update graph after refreshing labels
            self._update_graph()
        except: pass

    def _reset_graph(self):
        self.graph_x = []
        self.graph_y = []
        if self.graph_curve:
            self.graph_curve.setData([], [])
        # Reset baseline when graph resets
        try:
            self._frames_baseline = int(getattr(self.reader, 'frames_received', 0) or 0)
        except Exception:
            self._frames_baseline = 0
        # Restart activity monitor timeline
        self.activity_start_time = datetime.now()

    def _update_graph(self):
        """Append the latest frame count against elapsed time and update the curve."""
        try:
            if not self.graph_curve or not self.graph_plot:
                return
            # Use activity monitor time (separate from actual scan time)
            if self.activity_start_time and self.reader is not None:
                # Only update when monitor is actively listening
                is_active = False
                is_listening = False
                try:
                    if hasattr(self.reader, 'channel') and self.reader.channel is not None:
                        is_active = bool(self.reader.channel.isMonitorActive())
                    # Listening indicates we applied the current channel successfully
                    is_listening = (self.applied_channel == self.channel) and is_active
                except Exception:
                    is_active = False
                    is_listening = False
                if not (is_active and is_listening):
                    return
                t = int(max(0, (datetime.now() - self.activity_start_time).total_seconds()))
                # Prefer frames collected during active caching; fallback to total frames_received
                frames_total = getattr(self.reader, 'frames_received', None)
                if frames_total is None:
                    return
                # Continuous monitor: always update regardless of scan_state
                self.graph_x.append(t)
                # Plot delta frames collected since baseline
                try:
                    delta = int(frames_total) - int(self._frames_baseline)
                    self.graph_y.append(max(0, int(delta)))
                except Exception:
                    self.graph_y.append(int(max(0, frames_total)))
                # Keep last N points to avoid excessive memory (e.g., last 600 seconds)
                max_points = 600
                if len(self.graph_x) > max_points:
                    self.graph_x = self.graph_x[-max_points:]
                    self.graph_y = self.graph_y[-max_points:]
                self.graph_curve.setData(self.graph_x, self.graph_y)

                # Sliding window: keep newest time centered; move the view range accordingly
                try:
                    half = self.graph_window_seconds / 2.0
                    # Clamp x_min to 0 to avoid negative time display
                    x_min = max(0.0, float(t) - half)
                    x_max = x_min + self.graph_window_seconds
                    self.graph_plot.setXRange(x_min, x_max, padding=0)
                    if self.center_line:
                        self.center_line.setPos(float(t))
                except Exception as _:
                    pass
        except Exception as e:
            try:
                self.logger.error(f"Graph update error: {e}")
            except Exception:
                pass

    def closeEvent(self, event):
        self.info_timer.stop()
        self._cleanup_existing_instances()
        super().closeEvent(event)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scan Monitor Window')
    parser.add_argument('--channel', default='', help='PVA channel name')
    parser.add_argument('--config', dest='config_path', default='', help='Path to TOML config file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = ScanMonitorWindow(channel=args.channel, config_filepath=args.config_path)
    window.show()
    sys.exit(app.exec_())