import sys
import toml
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton

from utils import PVAReader
from utils import HDF5Writer, HDF5Handler
from epics import caput

class ScanViewWindow(QMainWindow):
    signal_start_monitor = pyqtSignal()
    
    def __init__(self, channel: str = "", config_filepath: str = ""):
        super(ScanViewWindow, self).__init__()
        uic.loadUi('gui/scan_view.ui', self)
        self.setWindowTitle('Scan View')

        self.channel = channel
        self.config_filepath = config_filepath
        self.scan_state = False
        # Track applied state to determine if current inputs are being listened to
        self.applied_channel = None
        self.applied_config = None
        
        self.h5_handler_thread = QThread()
        self.reader_thread = QThread()

        self.reader: PVAReader = None
        #self.h5_handler: HDF5Writer = None
        self.h5_handler: HDF5Handler = None
        
        # Timer for updating info display
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self._update_info_display)
        
        # Scan timing
        self.scan_start_time = None
        self.scan_end_time = None
        self.last_scan_completion_time = None

        self.label_mode.setText("")
        self.label_indicator.setText('scan: off')
        self._apply_indicator_style()
        self._update_info_display()
        # Initialize listening label to False until Apply is pressed
        if hasattr(self, 'label_listening'):
            self.label_listening.setText('False')

        self.lineedit_channel.setText(self.channel or "")
        self.lineedit_channel.textChanged.connect(self._on_channel_changed) 
        
        self.lineedit_config.setText(self.config_filepath or "")
        self.lineedit_config.textChanged.connect(self._on_config_path_changed) 

        self.btn_browse_config.clicked.connect(self._on_browse_config_clicked)
        self.btn_apply.clicked.connect(self._on_apply_clicked)

    # ================================================================================================
    # BUTTON CLICK HANDLERS
    # ================================================================================================

    def _on_apply_clicked(self) -> None:
        """Create new reader and writer instances with proper signal connections."""
        if not self.channel or not self.config_filepath:
            return

        self._cleanup_existing_instances()

        try:
            self.reader = PVAReader(
                input_channel=self.channel,
                config_filepath=self.config_filepath,
                viewer_type='image'
            )
            
            # self.h5_handler = HDF5Writer(
            #     file_path="",
            #     pva_reader=self.reader
            # )
            self.h5_handler = HDF5Handler(pva_reader=self.reader, compress=True)
            
            self.reader.moveToThread(self.reader_thread)
            self.h5_handler.moveToThread(self.h5_handler_thread)
            
            self.reader.reader_scan_complete.connect(self._on_reader_scan_complete)
            self.h5_handler.hdf5_writer_finished.connect(self._on_writer_finished)
            self.signal_start_monitor.connect(self.reader.start_channel_monitor)

            # Add scan state tracking if available
            if hasattr(self.reader, 'scan_state_changed'):
                self.reader.scan_state_changed.connect(self._on_scan_state_changed)
            
            self.reader_thread.start()
            self.h5_handler_thread.start()
            
            # Start the channel monitor automatically
            self.signal_start_monitor.emit()
            # Mark current inputs as applied and listening
            self.applied_channel = self.channel
            self.applied_config = self.config_filepath
            if hasattr(self, 'label_listening'):
                self.label_listening.setText('True')
            
            # Start the info update timer
            self.info_timer.start(1000)  # Update every second
            
            # Update info labels immediately after applying changes
            self._update_info_display()
            
        except Exception:
            self.reader = None
            self.h5_handler = None
            # Update info display even on failure to show current state
            self._update_info_display()

    def _on_browse_config_clicked(self) -> None:
        """Handle browse config button click to open file dialog."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Select TOML config', '', 'TOML Files (*.toml);;All Files (*)')
        if fname:
            if hasattr(self, 'lineedit_config'):
                self.lineedit_config.setText(fname)
            else:
                self.config_filepath = fname
                self._update_label_from_config()

    def _on_reset_config_clicked(self) -> None:
        """Reset configuration by cleaning up existing instances."""
        self._cleanup_existing_instances()

    def _on_start_scan_clicked(self) -> None:
        """Handle start scan button click."""
        if self.reader is None or not self.channel or not self.config_filepath:
            return

        try:
            self.signal_start_monitor.emit()
        except Exception:
            pass

    def _on_stop_scan_clicked(self) -> None:
        """Handle stop scan button click."""
        if self.reader is None:
            return
            
        try:
            if getattr(self.reader, 'FLAG_PV', ''):
                caput(self.reader.FLAG_PV, self.reader.STOP_SCAN)
            else:
                self.reader.stop_channel_monitor() 
        except Exception:
            pass

    # ================================================================================================
    # INPUT CHANGE HANDLERS
    # ================================================================================================

    def _on_channel_changed(self, text: str) -> None:
        """Handle channel input text change."""
        self.channel = text
        # When inputs change, mark listening as False until Apply is pressed again
        self.applied_channel = None
        if hasattr(self, 'label_listening'):
            self.label_listening.setText('False')

    def _on_config_path_changed(self, text: str) -> None:
        """Handle config path input text change."""
        self.config_filepath = text
        self._update_label_from_config()
        # When inputs change, mark listening as False until Apply is pressed again
        self.applied_config = None
        if hasattr(self, 'label_listening'):
            self.label_listening.setText('False')

    # ================================================================================================
    # SCAN OPERATIONS & SIGNAL HANDLERS
    # ================================================================================================

    def _on_reader_scan_complete(self) -> None:
        """Handle reader scan complete signal and automatically save data."""
        self._trigger_automatic_save()

    def _on_scan_state_changed(self, is_on: bool) -> None:
        """Handle scan state change signal to update UI."""
        # Track scan timing
        if is_on:
            self.scan_start_time = datetime.now()
            self.scan_end_time = None
        else:
            self.scan_end_time = datetime.now()
            # Record the completion time for "Last Scan Date" display
            self.last_scan_completion_time = self.scan_end_time
        
        if hasattr(self, 'label_indicator'):
            text = 'scan: on' if is_on else 'scan: off'
            self.label_indicator.setText(text)
            
            if is_on:
                self.label_indicator.setStyleSheet('color: green; font-weight: bold;')
            else:
                self.label_indicator.setStyleSheet('color: red; font-weight: bold;')

            self.scan_state = is_on
            self._update_button_states()

    def _trigger_automatic_save(self, clear_caches: bool = True) -> None:
        """Trigger automatic save of cached data when scan completes."""
        if self.h5_handler is None:
            return
 
        # try:
        #     if not self.h5_handler_thread.isRunning():
        #         self.h5_handler_thread.start()
        #     self.h5_handler.save_caches_to_h5(clear_caches=clear_caches, compress=True)

        try:
            self.h5_handler.save_data(clear_caches=clear_caches, compress=True, is_scan=False)
        except Exception:
            pass

    def _on_writer_finished(self, message) -> None:
        """Handle writer finished signal."""
        try:
            # Keep monitor active after save; do not stop the channel monitor here
            if hasattr(self, 'label_indicator'):
                self.label_indicator.setText('scan: off')
                self.label_indicator.setStyleSheet('color: red; font-weight: bold;')
                self.scan_state = False
                self._update_button_states()
                
        except Exception:
            pass

    def _cleanup_existing_instances(self) -> None:
        """Clean up existing reader and writer instances and their threads."""
        if self.reader is not None:
            try:
                self.reader.stop_channel_monitor()
            except Exception:
                pass
            
            if self.reader_thread.isRunning():
                self.reader_thread.quit()
                self.reader_thread.wait()
            
            if self.h5_handler_thread.isRunning():
                self.h5_handler_thread.quit()
                self.h5_handler_thread.wait()

            try:
                self.reader.reader_scan_complete.disconnect()
                if self.h5_handler is not None:
                    self.h5_handler.hdf5_writer_finished.disconnect()
                self.signal_start_monitor.disconnect()
            except (TypeError, RuntimeError, AttributeError, Exception):
                pass
            
            self.reader = None
            self.h5_handler = None

    # ================================================================================================
    # WINDOW & APPLICATION LIFECYCLE
    # ================================================================================================

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop info timer and monitoring when the window is closing
        try:
            self.info_timer.stop()
        except Exception:
            pass
        if self.reader and hasattr(self.reader, 'channel'):
            try:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
            except Exception:
                pass

        if self.reader_thread.isRunning():
            self.reader_thread.quit()
            self.reader_thread.wait()
        if self.h5_handler_thread.isRunning():
            self.h5_handler_thread.quit()
            self.h5_handler_thread.wait()
        super().closeEvent(event)

    # ================================================================================================
    # HELPER METHODS & UI UTILITIES
    # ================================================================================================

    def _update_button_states(self) -> None:
        """Update button enabled/disabled states based on scan state."""
        try:
            on = self.scan_state
            if hasattr(self, 'btn_start_scan'):
                self.btn_start_scan.setEnabled(not on)
            if hasattr(self, 'btn_stop_scan'):
                self.btn_stop_scan.setEnabled(on)
        except Exception:
            pass

    def _apply_indicator_style(self) -> None:
        """Apply appropriate styling to the scan indicator label."""
        try:
            if hasattr(self, 'label_indicator'):
                txt = self.label_indicator.text().strip().lower()
                if txt == 'scan: on':
                    self.label_indicator.setStyleSheet('color: green; font-weight: bold;')
                else:
                    self.label_indicator.setStyleSheet('color: red; font-weight: bold;')
        except Exception:
            pass

    def _update_label_from_config(self) -> None:
        """Update mode label based on configuration file contents."""
        if hasattr(self, 'label_mode'):
            self.label_mode.setText("")

    def _update_info_display(self) -> None:
        """Update the info labels with current reader state information."""
        try:
            # Update caching mode from config
            caching_mode = "Not set"
            if self.config_filepath:
                try:
                    with open(self.config_filepath, 'r') as f:
                        cfg = toml.load(f)
                    caching_mode = cfg.get('CACHE_OPTIONS', {}).get('CACHING_MODE', 'Not set')
                except Exception:
                    pass
            
            if hasattr(self, 'label_caching_mode'):
                self.label_caching_mode.setText(str(caching_mode))
            
            # Update flag PV
            flag_pv = "Not set"
            if self.reader and hasattr(self.reader, 'FLAG_PV'):
                flag_pv = str(self.reader.FLAG_PV) if self.reader.FLAG_PV else "Not set"
            
            if hasattr(self, 'label_flag_pv'):
                self.label_flag_pv.setText(flag_pv)
            
            # Update channel active status
            channel_active = "No"
            if self.reader and hasattr(self.reader, 'channel'):
                try:
                    if self.reader.channel and hasattr(self.reader.channel, 'isMonitorActive'):
                        channel_active = "Yes" if self.reader.channel.isMonitorActive() else "No"
                except Exception:
                    pass
            
            if hasattr(self, 'label_channel_active'):
                self.label_channel_active.setText(channel_active)
            # Update listening status: True only if monitor is active AND current inputs match applied ones
            listening = "False"
            try:
                inputs_applied = (self.applied_channel == self.channel and self.applied_config == self.config_filepath)
                if self.reader and hasattr(self.reader, 'channel') and hasattr(self.reader.channel, 'isMonitorActive'):
                    if self.reader.channel.isMonitorActive() and inputs_applied:
                        listening = "True"
            except Exception:
                pass
            if hasattr(self, 'label_listening'):
                self.label_listening.setText(listening)
            
            # Update caching status
            is_caching = "No"
            if self.reader and hasattr(self.reader, 'is_caching'):
                try:
                    is_caching = "Yes" if self.reader.is_caching else "No"
                except Exception:
                    pass
            elif self.scan_state:
                is_caching = "Yes"
            
            if hasattr(self, 'label_is_caching'):
                self.label_is_caching.setText(is_caching)
            
            # Update scan time
            scan_time_text = "--"
            if self.scan_start_time:
                if self.scan_end_time:
                    # Scan completed - show duration
                    duration = self.scan_end_time - self.scan_start_time
                    total_seconds = int(duration.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if hours > 0:
                        scan_time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        scan_time_text = f"{minutes:02d}:{seconds:02d}"
                else:
                    # Scan in progress - show elapsed time
                    duration = datetime.now() - self.scan_start_time
                    total_seconds = int(duration.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if hours > 0:
                        scan_time_text = f"{hours:02d}:{minutes:02d}:{seconds:02d} (running)"
                    else:
                        scan_time_text = f"{minutes:02d}:{seconds:02d} (running)"
            
            if hasattr(self, 'label_scan_time'):
                self.label_scan_time.setText(scan_time_text)
            
            # Update last scan date
            last_scan_date_text = "--"
            if self.last_scan_completion_time:
                last_scan_date_text = self.last_scan_completion_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if hasattr(self, 'label_last_scan_date'):
                self.label_last_scan_date.setText(last_scan_date_text)
                
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    window = ScanViewWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
