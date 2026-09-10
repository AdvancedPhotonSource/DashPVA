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

import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from epics import caget, caput
from PyQt5.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog

import dashpva.settings as app_settings
from dashpva.gui import configure_app
from dashpva.gui.theme_colors import ERROR, SUCCESS
from dashpva.utils import HDF5Handler, PVAReader
from dashpva.viewer.core.base_window import BaseWindow


class ScanMonitorWindow(BaseWindow):
    #: No GPU work happens here, so the status bar does not carry the readout.
    show_gpu_stat = False

    signal_start_monitor = pyqtSignal()
    signal_trigger_save = pyqtSignal(bool, bool, bool, str)  # clear_caches, write_temp, write_output, output_override
    
    def __init__(self, channel: str = ""):
        super().__init__(ui_file_name='scan_view.ui', viewer_name='Scan Monitor',
                         visible_actions=['Documentation'])
        # An explicit --channel wins over anything the last session restores.
        self._cli_channel = channel
        self.channel = channel or app_settings.get_input_channel("")
        self.scan_state = False

        # Track applied state for UI labels
        self.applied_channel = None
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
        self.channel_plot = None
        self.channel_curve = None
        self.cache_plot = None
        self.cache_curve = None
        self.graph_markers = None
        self.channel_x = []
        self.channel_y = []
        self.cache_x = []
        self.cache_y = []
        #: Cache trace origin — reset per scan, so each scan starts at t=0.
        self.cache_start_time = None
        #: One entry per scan: start/stop time and frame count, for marker clicks.
        self._scan_sections = []
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
        #: Guards the Stop-Scan fallback from saving a scan the reader already saved.
        self._save_emitted_for_scan = False

        #: Plain heading for the channel plot; a clicked scan replaces it.
        self._channel_title = self.label_channel_title.text()

        # Setup Initial UI State
        self._setup_ui_elements()
        self._setup_graph()

    def on_session_restored(self) -> None:
        """Re-resolve the channel now the restored inputs are applied.

        An explicit --channel still wins; otherwise keep the channel restored
        from the last session, falling back to the configured input channel.
        """
        self.channel = (self._cli_channel or self.lineedit_channel.text().strip()
                        or app_settings.get_input_channel(""))
        self.lineedit_channel.setText(self.channel or "")

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
        if hasattr(self, 'label_flag_pv'):
            self.label_flag_pv.setCursor(Qt.PointingHandCursor)
            self.label_flag_pv.setToolTip('Click to copy')
            self.label_flag_pv.installEventFilter(self)

        self.lineedit_channel.setText(self.channel or "")
        self.lineedit_channel.textChanged.connect(self._on_channel_changed)

        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_start_scan.clicked.connect(self._on_start_scan_clicked)
        self.btn_stop_scan.clicked.connect(self._on_stop_scan_clicked)
        self.btn_browse_folder.clicked.connect(self._browse_save_folder)
        self.btn_override_clear.clicked.connect(self._on_override_clear_clicked)
        self.groupBox_override.toggled.connect(self._on_override_toggled)
        self.lineedit_save_folder.textChanged.connect(self._update_override_status)
        self.lineedit_save_filename.textChanged.connect(self._update_override_status)
        self.checkbox_auto_increment.toggled.connect(self._on_auto_increment_toggled)
        self.spinbox_start_index.valueChanged.connect(self._update_override_status)

        self._refresh_beamline_label()
        self._on_override_toggled(self.groupBox_override.isChecked())
        self._update_button_states()
        self._update_info_display()

    def _setup_graph(self):
        """Build the two activity plots: channel throughput and cache occupancy."""
        try:
            self.channel_plot, self.channel_curve, self.center_line, self.graph_markers = \
                self._build_plot(self.widget_graph_channel, 'Frames received',
                                 (30, 144, 255), markers=True)
            self.cache_plot, self.cache_curve, self.cache_line, _ = \
                self._build_plot(self.widget_graph_cache, 'Frames in cache',
                                 (46, 204, 113))
            self._reset_graph()
        except Exception as e:
            try:
                self.logger.error(f"Graph setup error: {e}")
            except Exception:
                pass

    def _build_plot(self, holder, y_label: str, colour: tuple, markers: bool = False):
        """Put a pyqtgraph plot into a .ui placeholder.

        Returns ``(plot, curve, centre_line, markers)``; ``markers`` is None
        unless asked for, and only the channel plot carries scan start/stop
        markers -- the cache trace already shows a scan as a rise and fall.
        """
        plot = pg.PlotWidget(background='w')
        plot.showGrid(x=True, y=True)
        plot.setLabel('bottom', 'Time', units='s')
        plot.setLabel('left', y_label)
        try:
            plot.enableAutoRange(x=False, y=True)
        except Exception:
            pass
        curve = plot.plot([], [], pen=pg.mkPen(color=colour, width=2))
        try:
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(color=(128, 128, 128), style=Qt.DashLine))
            plot.addItem(line)
        except Exception:
            line = None
        scatter = None
        if markers:
            try:
                scatter = pg.ScatterPlotItem(
                    size=app_settings.SCAN_MARKER_SIZE, hoverable=True,
                    hoverSize=app_settings.SCAN_MARKER_SIZE + 4)
                scatter.sigClicked.connect(self._on_marker_clicked)
                plot.addItem(scatter)
                plot.scene().sigMouseClicked.connect(self._on_channel_clicked)
            except Exception:
                scatter = None
        holder.layout().addWidget(plot)
        return plot, curve, line, scatter

    # ================================================================================================
    # CORE LOGIC & THREADING
    # ================================================================================================

    def _on_apply_clicked(self) -> None:
        """Initializes Reader and Writer on separate threads."""
        if not self.channel:
            return

        self._cleanup_existing_instances()

        try:
            app_settings.reload()
        except Exception:
            pass
        self._refresh_beamline_label()

        try:
            # 1. Create instances
            self.reader = PVAReader(
                input_channel=self.channel,
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
            self.signal_start_monitor.connect(self.reader.start_scan_monitor, Qt.QueuedConnection)
            self.signal_start_monitor.connect(self.reader.start_hkl_ca_monitor, Qt.QueuedConnection)
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
            if hasattr(self, 'label_listening'):
                self.label_listening.setText('True')
                self._apply_listening_style(True)
            
            self.info_timer.start(1000)
            self._update_button_states()
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
            self._update_button_states()

    def _on_reader_scan_complete(self) -> None:
        """Slot executed when PVAReader emits the completion signal."""
        try:
            self.logger.info(f"reader_scan_complete received at {datetime.now()}")
        except Exception:
            pass
        self._trigger_automatic_save()

    def _trigger_automatic_save(self) -> None:
        """Triggers the HDF5Writer save process."""
        cached = getattr(self.reader, 'cached_images', None)
        if cached is not None and len(cached) == 0:
            # The writer would raise "Caches cannot be empty", and the override
            # index would have been spent on a file that never appears.
            self.logger.warning("Nothing cached for this scan — save skipped")
            return
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
            output_override = self._next_override_path()
            self._save_emitted_for_scan = True
            self.signal_trigger_save.emit(True, write_temp, write_output, output_override)

    def _on_writer_finished(self, message: str) -> None:
        """Show where the scan landed, and reset the indicator."""
        # Logged only -- a save result does not belong in the window body.
        if message.startswith('Failed'):
            self.logger.error(message)
        else:
            self.logger.info(message)
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: off')
            self._apply_indicator_style()
            self.scan_state = False
            self._update_button_states()
        self._update_override_status()

    def _on_stop_scan_clicked(self) -> None:
        if self.reader is None:
            return
        try:
            if getattr(self.reader, 'FLAG_PV', ''):
                caput(self.reader.FLAG_PV, self.reader.STOP_SCAN)
                # The reader only emits reader_scan_complete if its CA monitor
                # also saw the scan start; otherwise Stop writes the flag and
                # nothing is ever saved. Check back and save if it did not.
                QTimer.singleShot(app_settings.SCAN_STOP_SAVE_GRACE_MS,
                                  self._save_if_reader_did_not)
            else:
                self.reader.stop_channel_monitor()
                # If no flag PV triggers the reader, we trigger the complete sequence manually
                self._on_reader_scan_complete()
                self._on_scan_state_changed(False)
        except Exception as e:
            try:
                self.logger.error(f"Manual Stop Error: {e}")
            except Exception:
                pass

    def _save_if_reader_did_not(self) -> None:
        """Save a stopped scan the reader never reported as complete."""
        if self.reader is None or self._save_emitted_for_scan:
            return
        if getattr(self.reader, 'is_caching', False):
            return
        try:
            self.logger.info("No scan-complete from the reader; saving from Stop Scan")
        except Exception:
            pass
        self._trigger_automatic_save()

    def _on_start_scan_clicked(self) -> None:
        """Drive the scan flag high, mirroring _on_stop_scan_clicked.

        Re-emitting signal_start_monitor is not a start: Apply already emitted
        it, so the monitors are up and nothing about the scan changes.
        """
        if self.reader is None:
            return
        try:
            if getattr(self.reader, 'FLAG_PV', ''):
                caput(self.reader.FLAG_PV, getattr(self.reader, 'START_SCAN', True))
            else:
                # Nothing will emit scan_state_changed without a flag PV, so
                # bring the monitors up and move the UI ourselves.
                self.signal_start_monitor.emit()
                self._on_scan_state_changed(True)
        except Exception as e:
            try:
                self.logger.error(f"Manual Start Error: {e}")
            except Exception:
                pass

    def _cleanup_existing_instances(self) -> None:
        if self.reader is not None:
            # Each disconnect gets its own guard: a failing stop_channel_monitor
            # must not skip them, or a re-Apply stacks a second connection and
            # every scan-complete saves twice.
            try:
                self.reader.stop_channel_monitor()
            except Exception:
                pass
            try:
                self.reader.reader_scan_complete.disconnect()
            except Exception:
                pass
            try:
                self.h5_handler.hdf5_writer_finished.disconnect()
            except Exception:
                pass
            try:
                self.signal_start_monitor.disconnect()
            except Exception:
                pass
            try:
                self.signal_trigger_save.disconnect()
            except Exception:
                pass

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

    def _on_scan_state_changed(self, is_on: bool) -> None:
        if is_on:
            self._save_emitted_for_scan = False
            self.scan_start_time = datetime.now()
            self.scan_end_time = None
            # The channel trace runs continuously; only the cache trace, which
            # describes this scan alone, starts over.
            self._reset_cache_graph()
            self._add_scan_marker(True)
        else:
            self.scan_end_time = datetime.now()
            self.last_scan_completion_time = self.scan_end_time
            self._add_scan_marker(False)
        
        self.scan_state = is_on
        if hasattr(self, 'label_indicator'):
            self.label_indicator.setText('scan: on' if is_on else 'scan: off')
            self._apply_indicator_style()
        self._update_button_states()

    def eventFilter(self, obj, event):
        """Copy the Flag PV on click — it is needed verbatim for a caput."""
        if (obj is getattr(self, 'label_flag_pv', None)
                and event.type() == QEvent.MouseButtonRelease):
            pv = self.label_flag_pv.text().strip()
            if pv and pv not in ('--', 'Not set'):
                QApplication.clipboard().setText(pv)
                self.update_status(f'Copied {pv}')
        return super().eventFilter(obj, event)

    def _apply_indicator_style(self):
        if hasattr(self, 'label_indicator'):
            color = SUCCESS if "on" in self.label_indicator.text().lower() else ERROR
            self.label_indicator.setStyleSheet(f'color: {color}; font-weight: bold;')

    def _apply_listening_style(self, state):
        if hasattr(self, 'label_listening'):
            color = SUCCESS if state else ERROR
            self.label_listening.setStyleSheet(f'color: {color}; font-weight: bold;')

    def _update_button_states(self):
        applied = self.reader is not None
        self.btn_start_scan.setEnabled(applied and not self.scan_state)
        self.btn_stop_scan.setEnabled(applied and self.scan_state)

    def _update_save_warning(self):
        if not hasattr(self, 'label_mode'):
            return
        write_temp = hasattr(self, 'checkbox_write_temp') and self.checkbox_write_temp.isChecked()
        write_output = hasattr(self, 'checkbox_write_output') and self.checkbox_write_output.isChecked()
        if not write_temp and not write_output:
            self._set_message(self.label_mode,
                              'Warning: No save targets selected — data will not be written',
                              'warning')
        else:
            self._set_message(self.label_mode, '', 'info')

    def _refresh_beamline_label(self) -> None:
        """Show the configured beamline. Static, so not on the 1 Hz refresh."""
        self.label_beamline.setText(app_settings.get_beamline_name() or '--')

    def _update_info_display(self):
        """Logic for periodically refreshing UI labels based on Reader state."""
        try:
            # Update Caching Mode
            try:
                caching_mode = app_settings.CACHE_OPTIONS.get('CACHING_MODE', 'Not set') or 'Not set'
            except Exception:
                caching_mode = 'Not set'
            if hasattr(self, 'label_caching_mode'):
                self.label_caching_mode.setText(str(caching_mode))
            
            # Update Flag PV
            flag_pv = "Not set"
            if self.reader and hasattr(self.reader, 'FLAG_PV'):
                flag_pv = str(self.reader.FLAG_PV) if self.reader.FLAG_PV else "Not set"
            if hasattr(self, 'label_flag_pv'):
                self.label_flag_pv.setText(flag_pv)

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
                m, s = divmod(s, 60)
                h, m = divmod(m, 60)
                time_str = f"{h:02d}:{m:02d}:{s:02d}"
                if not self.scan_end_time:
                    time_str += " (running)"
                if hasattr(self, 'label_scan_time'):
                    self.label_scan_time.setText(time_str)

            if self.last_scan_completion_time and hasattr(self, 'label_last_scan_date'):
                self.label_last_scan_date.setText(self.last_scan_completion_time.strftime("%Y-%m-%d %H:%M:%S"))

            rois = app_settings.ROI or {}
            self.label_detector_rois.setText(', '.join(sorted(rois)) if rois else '--')

            # Update File Output from PVs in settings
            if hasattr(self, 'label_file_output'):
                file_output = "--"
                try:
                    file_path_pv = app_settings.FILE_PATH_PV or ''
                    file_name_pv = app_settings.FILE_NAME_PV or ''
                    if file_path_pv or file_name_pv:
                        fp = caget(file_path_pv, timeout=0.3) if file_path_pv else ''
                        fn = caget(file_name_pv, timeout=0.3) if file_name_pv else ''
                        fp_str = str(fp or '').strip()
                        fn_str = str(fn or '').strip()
                        if fp_str and fn_str:
                            combined = fp_str.rstrip('/') + '/' + fn_str.lstrip('/')
                        else:
                            combined = fp_str or fn_str
                        file_output = combined if combined else "--"
                except Exception:
                    pass
                self.label_file_output.setText(file_output)

            # Update graph after refreshing labels
            self._update_graph()
        except Exception:
            pass

    def _on_override_toggled(self, enabled: bool) -> None:
        """React to the section being switched on/off; off means the config path.

        groupBox_override is checkable, so Qt greys its contents on its own --
        only the defaults and the status line need doing here.
        """
        if enabled:
            self._seed_override_defaults()
        # Qt re-enables every child on check, so re-apply the numbering gate.
        self._on_auto_increment_toggled(self.checkbox_auto_increment.isChecked())

    def _on_auto_increment_toggled(self, enabled: bool) -> None:
        """The start index only means anything while numbering is on."""
        self.label_start_index_text.setEnabled(enabled)
        self.spinbox_start_index.setEnabled(enabled)
        self._update_override_status()

    def _on_override_clear_clicked(self) -> None:
        """Empty the override inputs and restart the filename count."""
        self.lineedit_save_folder.clear()
        self.lineedit_save_filename.clear()
        self.spinbox_start_index.setValue(app_settings.SCAN_OVERRIDE_START_INDEX)
        self._update_override_status()

    def _override_stem(self) -> tuple:
        """Return ``(folder, stem)`` from the override inputs, both possibly ''."""
        folder = self.lineedit_save_folder.text().strip()
        stem = self.lineedit_save_filename.text().strip()
        if stem.endswith(app_settings.SCAN_OVERRIDE_SUFFIX):
            stem = stem[:-len(app_settings.SCAN_OVERRIDE_SUFFIX)]
        return folder, stem

    def _next_override_path(self, advance: bool = True) -> str:
        """Where the next save lands, e.g. ``kyle`` -> ``.../kyle1.h5``.

        With auto-increment on, the number comes from ``spinbox_start_index``,
        which steps up after each save and skips an index whose file already
        exists, so a re-launched session cannot overwrite an earlier scan. With it off the stem is used verbatim (``.../kyle.h5``),
        which each scan overwrites. Returns '' when the override is off or no
        filename is set.
        """
        if not self.groupBox_override.isChecked():
            return ''
        folder, stem = self._override_stem()
        if not stem:
            # A folder on its own is not a file path: handing it to the writer
            # made it try to create an h5 *at* the directory (Errno 21). No
            # filename means no override, which is what the status line says.
            return ''
        base = Path(folder).expanduser() if folder else Path(app_settings.OUTPUT_PATH)
        if not self.checkbox_auto_increment.isChecked():
            return str(base / f'{stem}{app_settings.SCAN_OVERRIDE_SUFFIX}')
        index = self.spinbox_start_index.value()
        while (base / f'{stem}{index}{app_settings.SCAN_OVERRIDE_SUFFIX}').exists():
            index += 1
        if advance:
            self.spinbox_start_index.setValue(index + 1)
        return str(base / f'{stem}{index}{app_settings.SCAN_OVERRIDE_SUFFIX}')

    def _update_override_status(self) -> None:
        """Say whether an override is in effect, and where the next save lands."""
        if not self.groupBox_override.isChecked():
            self._set_message(self.label_override_status,
                              'No override — saving to the configured output path', 'info')
            return
        preview = self._next_override_path(advance=False)
        if not preview:
            self._set_message(self.label_override_status,
                              'Override enabled but no filename set — '
                              'saving to the configured output path', 'warning')
        elif self.checkbox_auto_increment.isChecked():
            self._set_message(self.label_override_status,
                              f'Override active — next save: {preview}', 'warning')
        else:
            self._set_message(self.label_override_status,
                              f'Override active, auto-increment off — every scan overwrites {preview}',
                              'warning')

    def _set_message(self, label, text: str, level: str) -> None:
        """Set a label's text and its theme.qss messageLevel, then repolish it."""
        label.setText(text)
        label.setProperty('messageLevel', level)
        label.style().unpolish(label)
        label.style().polish(label)

    def _seed_override_defaults(self):
        """Fill any empty override field from PATHS > OUTPUTS > SCAN in settings.

        Only empties are filled, so a value the user typed (or one restored from
        the previous session) is never overwritten.
        """
        folder, filename = app_settings.get_scan_output_defaults()
        if not self.lineedit_save_folder.text().strip():
            self.lineedit_save_folder.setText(folder)
        if not self.lineedit_save_filename.text().strip():
            self.lineedit_save_filename.setText(filename)

    def _browse_save_folder(self):
        """Open a folder picker and set lineedit_save_folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder and hasattr(self, 'lineedit_save_folder'):
            self.lineedit_save_folder.setText(folder)

    def _frames_collected(self) -> int:
        """Frames taken in since the baseline, matching what the curve plots."""
        try:
            total = getattr(self.reader, 'frames_received', None)
            if total is not None:
                return max(0, int(total) - int(self._frames_baseline))
        except Exception:
            pass
        return self.channel_y[-1] if self.channel_y else 0

    def _add_scan_marker(self, started: bool) -> None:
        """Record a scan boundary and mark it on the channel plot.

        A stop also shades the whole scan as a region, so the span is visible
        without having to hit a marker precisely.
        """
        if self.graph_markers is None or self.activity_start_time is None:
            return
        now = datetime.now()
        frames = self._frames_collected()
        t = max(0.0, (now - self.activity_start_time).total_seconds())
        try:
            if started:
                self._scan_sections.append({'start_t': t, 'start': now,
                                            'start_frames': frames})
            elif self._scan_sections:
                self._scan_sections[-1].update({'stop_t': t, 'stop': now,
                                                'stop_frames': frames})
                self._shade_scan_region(self._scan_sections[-1])
            index = len(self._scan_sections) - 1
            self.graph_markers.addPoints([{
                'pos': (t, frames),
                'symbol': 't1' if started else 's',
                'brush': pg.mkBrush(SUCCESS if started else ERROR),
                'pen': None,
                'size': app_settings.SCAN_MARKER_SIZE,
                'data': index,
            }])
        except Exception as e:
            try:
                self.logger.error(f"Marker error: {e}")
            except Exception:
                pass

    def _shade_scan_region(self, section: dict) -> None:
        """Shade a completed scan's span on the channel plot."""
        try:
            region = pg.LinearRegionItem(
                values=(section['start_t'], section['stop_t']),
                movable=False, brush=pg.mkBrush(39, 174, 96, 40))
            region.setZValue(-10)
            self.channel_plot.addItem(region)
        except Exception:
            pass

    def _on_marker_clicked(self, _scatter, points) -> None:
        """A marker click selects its scan."""
        if not len(points):
            return
        try:
            self._show_scan_section(int(points[0].data()))
        except (TypeError, ValueError):
            pass

    def _on_channel_clicked(self, event) -> None:
        """Select whichever scan section was clicked on the channel plot."""
        try:
            x = self.channel_plot.getPlotItem().vb.mapSceneToView(event.scenePos()).x()
        except Exception:
            return
        for index, section in enumerate(self._scan_sections):
            stop = section.get('stop_t')
            if stop is None:
                stop = self.channel_x[-1] if self.channel_x else section['start_t']
            if section['start_t'] <= x <= stop:
                self._show_scan_section(index)
                return
        self._show_scan_section(None)

    def _show_scan_section(self, index) -> None:
        """Put a clicked scan's span in the channel plot's heading.

        One line, in place of the title, so the plot does not move when a
        section is selected.
        """
        if index is None:
            self.label_channel_title.setText(self._channel_title)
            return
        try:
            section = self._scan_sections[index]
        except (IndexError, TypeError):
            return
        started = section['start'].strftime('%H:%M:%S')
        if 'stop' in section:
            held = str(section['stop'] - section['start']).split('.')[0]
            frames = section['stop_frames'] - section['start_frames']
            detail = (f"{started} → {section['stop'].strftime('%H:%M:%S')}"
                      f"  ({held})  {frames} frames")
        else:
            detail = f'started {started} — running'
        self.label_channel_title.setText(f'Scan {index + 1}:  {detail}')

    def _reset_cache_graph(self):
        """Restart the cache trace at t=0 — it describes one scan, not the session."""
        self.cache_x, self.cache_y = [], []
        if self.cache_curve:
            self.cache_curve.setData([], [])
        self.cache_start_time = datetime.now()

    def _reset_graph(self):
        """Clear both timelines. Only Apply does this — a scan start does not."""
        self.channel_x, self.channel_y = [], []
        if self.channel_curve:
            self.channel_curve.setData([], [])
        self._reset_cache_graph()
        self._scan_sections = []
        self._show_scan_section(None)
        if self.graph_markers:
            self.graph_markers.clear()
        for item in list(getattr(self.channel_plot, 'items', lambda: [])()):
            if isinstance(item, pg.LinearRegionItem):
                self.channel_plot.removeItem(item)
        try:
            self._frames_baseline = int(getattr(self.reader, 'frames_received', 0) or 0)
        except Exception:
            self._frames_baseline = 0
        self.activity_start_time = datetime.now()

    def _cached_frame_count(self) -> int:
        """Frames currently held in the reader's cache, across bins if binned."""
        cached = getattr(self.reader, 'cached_images', None)
        if cached is None:
            return 0
        try:
            if len(cached) and isinstance(cached[0], (list, deque)):
                return sum(len(b) for b in cached)
            return len(cached)
        except Exception:
            return 0

    def _append_point(self, xs: list, ys: list, curve, plot, line, t: int, y: int) -> None:
        """Add one sample and slide the view so the newest point stays centred."""
        xs.append(t)
        ys.append(y)
        if len(xs) > app_settings.SCAN_GRAPH_MAX_POINTS:
            del xs[:-app_settings.SCAN_GRAPH_MAX_POINTS]
            del ys[:-app_settings.SCAN_GRAPH_MAX_POINTS]
        curve.setData(xs, ys)
        try:
            x_min = max(0.0, float(t) - self.graph_window_seconds / 2.0)
            plot.setXRange(x_min, x_min + self.graph_window_seconds, padding=0)
            if line:
                line.setPos(float(t))
        except Exception:
            pass

    def _update_graph(self):
        """Sample both traces: frames off the channel, and frames held in cache."""
        try:
            if not self.channel_curve or not self.cache_curve:
                return
            if not (self.activity_start_time and self.reader is not None):
                return
            # Only sample while the applied channel is actually being monitored.
            try:
                is_active = (self.reader.channel is not None
                             and bool(self.reader.channel.isMonitorActive()))
            except Exception:
                is_active = False
            if not (is_active and self.applied_channel == self.channel):
                return

            frames_total = getattr(self.reader, 'frames_received', None)
            if frames_total is None:
                return
            t = int(max(0, (datetime.now() - self.activity_start_time).total_seconds()))
            try:
                received = max(0, int(frames_total) - int(self._frames_baseline))
            except Exception:
                received = 0
            self._append_point(self.channel_x, self.channel_y, self.channel_curve,
                               self.channel_plot, self.center_line, t, received)
            origin = self.cache_start_time or self.activity_start_time
            t_cache = int(max(0, (datetime.now() - origin).total_seconds()))
            self._append_point(self.cache_x, self.cache_y, self.cache_curve,
                               self.cache_plot, self.cache_line, t_cache,
                               self._cached_frame_count())
        except Exception as e:
            try:
                self.logger.error(f"Graph update error: {e}")
            except Exception:
                pass

    def closeEvent(self, event):
        self.info_timer.stop()
        self._cleanup_existing_instances()
        # Geometry and inputs are persisted by BaseWindow.closeEvent.
        super().closeEvent(event)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Scan Monitor Window')
    parser.add_argument('--channel', default='', help='PVA channel name')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    configure_app(app)
    window = ScanMonitorWindow(channel=args.channel)
    window.show()
    sys.exit(app.exec_())