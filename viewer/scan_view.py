import sys
import toml
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from utils import PVAReader, HDF5Writer
from epics import caput

class ScanMonitorWindow(QMainWindow):
    signal_start_monitor = pyqtSignal()
    
    def __init__(self, channel: str = "", config_filepath: str = ""):
        super(ScanMonitorWindow, self).__init__()
        uic.loadUi('gui/scan_view.ui', self)
        # Title comes from UI; ensure consistent naming in code comments

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
        self.h5_handler: HDF5Writer = None
        
        # Timer for updating info display
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self._update_info_display)
        
        # Scan timing variables
        self.scan_start_time = None
        self.scan_end_time = None
        self.last_scan_completion_time = None

        # Setup Initial UI State
        self._setup_ui_elements()

    def _setup_ui_elements(self):
        if hasattr(self, 'label_mode'):
            self.label_mode.setText("")
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: off')
            self._apply_indicator_style()
        if hasattr(self, 'label_listening'):
            self.label_listening.setText('False')
            self._apply_listening_style(False)

        self.lineedit_channel.setText(self.channel or "")
        self.lineedit_channel.textChanged.connect(self._on_channel_changed) 
        self.lineedit_config.setText(self.config_filepath or "")
        self.lineedit_config.textChanged.connect(self._on_config_path_changed) 

        self.btn_browse_config.clicked.connect(self._on_browse_config_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        
        self._update_info_display()

    # ================================================================================================
    # CORE LOGIC & THREADING
    # ================================================================================================

    def _on_apply_clicked(self) -> None:
        """Initializes Reader and Writer on separate threads."""
        if not self.channel or not self.config_filepath:
            return

        self._cleanup_existing_instances()

        try:
            # 1. Create instances
            self.reader = PVAReader(
                input_channel=self.channel,
                config_filepath=self.config_filepath,
                viewer_type='image'
            )
            
            self.h5_handler = HDF5Writer(
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
            
        except Exception as e:
            print(f"Apply Error: {e}")
            self.reader = None
            self.h5_handler = None

    def _on_reader_scan_complete(self) -> None:
        """Slot executed when PVAReader emits the completion signal."""
        print(f"LOG: reader_scan_complete received by ScanMonitor at {datetime.now()}")
        self._trigger_automatic_save()

    def _trigger_automatic_save(self) -> None:
        """Triggers the HDF5Writer save process."""
        if self.h5_handler:
            print("LOG: Triggering HDF5Writer.save_caches_to_h5...")
            # This method runs in the writer_thread due to moveToThread earlier
            self.h5_handler.save_caches_to_h5(clear_caches=True, compress=True)

    def _on_writer_finished(self, message: str) -> None:
        """Callback when the HDF5 file is finished writing."""
        print(f"LOG: Writer finished - {message}")
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
            print(f"Manual Stop Error: {e}")

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
            self.label_listening.setText('False')
            self._apply_listening_style(False)

    def _on_config_path_changed(self, text):
        self.config_filepath = text
        self.applied_config = None
        if hasattr(self, 'label_listening'):
            self.label_listening.setText('False')
            self._apply_listening_style(False)

    def _on_browse_config_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Config', '', 'TOML (*.toml)')
        if fname: self.lineedit_config.setText(fname)

    def _on_scan_state_changed(self, is_on: bool) -> None:
        if is_on:
            self.scan_start_time = datetime.now()
            self.scan_end_time = None
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

    def _update_label_from_config(self):
        if hasattr(self, 'label_mode'): self.label_mode.setText("")

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
            listening = "False"
            if self.reader and hasattr(self.reader, 'channel'):
                is_active = self.reader.channel.isMonitorActive()
                channel_active = "Yes" if is_active else "No"
                if is_active and (self.applied_channel == self.channel):
                    listening = "True"
            
            if hasattr(self, 'label_channel_active'): self.label_channel_active.setText(channel_active)
            if hasattr(self, 'label_listening'): 
                self.label_listening.setText(listening)
                self._apply_listening_style(listening == "True")

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
        except: pass

    def closeEvent(self, event):
        self.info_timer.stop()
        self._cleanup_existing_instances()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScanMonitorWindow()
    window.show()
    sys.exit(app.exec_())