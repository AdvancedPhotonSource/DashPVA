import os
import sys
import time
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
from pyqtgraph.colormap import get as get_colormap
from pyqtgraph.colormap import listMaps
import xrayutilities as xu
from PyQt5 import uic
# from epics import caget
from epics import PV, pv
from epics import camonitor, caget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QDialog, QFileDialog, QSlider,
                             QLabel, QPushButton, QCheckBox, QHBoxLayout, QVBoxLayout)
# Custom imported classes
from roi_stats_dialog import RoiStatsDialog
from roi_stats_plot import RoiStatsPlotDialog
from mask_viewer import MaskViewerWindow
from analysis_window import AnalysisWindow
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import rotation_cycle
from utils import PVAReader, HDF5Writer
from utils.mask_manager import MaskManager
# from ..utils.size_manager import SizeManager


rot_gen = rotation_cycle(1,5)         


class ConfigDialog(QDialog):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
            config_path (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/pv_config.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ''
        self.config_path = ''
        # class can be prefilled with text
        self.init_ui()
        
        # Connecting signasl to 
        self.btn_clear.clicked.connect(self.clear_pv_setup)
        self.btn_browse.clicked.connect(self.browse_file_dialog)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_config.setText(self.le_config.text())

    def browse_file_dialog(self) -> None:
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_config.setText(self.pvs_path)

    def clear_pv_setup(self) -> None:
        """
        Clears line edit that tells image view where the config file is.
        """
        self.le_config.clear()

    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the DiffractionImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.config_path = self.le_config.text()
        if osp.isfile(self.config_path) or (self.config_path == ''):
            self.image_viewer = DiffractionImageWindow(input_channel=self.input_channel,
                                            file_path=self.config_path,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class DiffractionImageWindow(QMainWindow):
    hkl_data_updated = pyqtSignal(bool)

    def __init__(self, input_channel='s6lambda1:Pva1:Image', file_path=''): 
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
            file_path (str): The file path for loading configuration.
        """
        super(DiffractionImageWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle('DashPVA')
        self.show()
        # Initializing important variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        self.rot_num = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.rois: list[pg.ROI] = []
        self.stats_dialogs = {}
        self.stats_plot_dialogs = {}
        self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path

        # Mask management
        self.mask_manager = MaskManager()
        self.mask_viewer = None
        self._dead_px_frames = []
        self._dead_px_collecting = False
        self._dead_px_target = 50

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        self.file_writer_thread = QThread()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.timeout.connect(self.update_rois)

        # For testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50

        # HKL values
        self.is_hkl_ready = False
        self.hkl_config = None
        self.hkl_pvs = {}
        self.hkl_data = {}
        self.q_conv = None
        self.qx = None
        self.qy = None
        self.qz = None
        self.processes = {}
        
        # Initialize colormap once for better performance
        try:
            self.cet_colormap = get_colormap('viridis')
        except Exception:
            try:
                self.cet_colormap = get_colormap('viridis', source='colorcet')
            except Exception:
                self.cet_colormap = get_colormap('magma')  # fallback
        
        plot = pg.PlotItem()
        self.image_view = pg.ImageView(view=plot)
        # Set the default colormap when ImageView is created
        self.image_view.setColorMap(self.cet_colormap)
        # Remove ImageView's internal widgets to eliminate vertical margin mismatch
        self.image_view.ui.histogram.setParent(None)
        self.image_view.ui.roiBtn.setParent(None)
        self.image_view.ui.menuBtn.setParent(None)
        self.viewer_layout.addWidget(self.image_view,0,1)
        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')

        # Initialize crosshair lines (hidden initially)
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('red', width=2))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('red', width=2))
        self.image_view.addItem(self.crosshair_v, ignoreBounds=True)
        self.image_view.addItem(self.crosshair_h, ignoreBounds=True)
        self.crosshair_v.hide()
        self.crosshair_h.hide()
        # second is a separate plot to show the horiontal avg of peaks in the image
        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(200)
        self.horizontal_avg_plot.getAxis('bottom').setLabel(text='Horizontal Avg.')
        self.horizontal_avg_plot.hideAxis('left')
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 0,0)
        # Sync horizontal avg y-range to image view — use sigYRangeChanged
        # with padding=0 for precise alignment
        self.horizontal_avg_plot.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self.image_view.getView().getViewBox().sigYRangeChanged.connect(self._sync_havg_yrange)

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.btn_analysis_window.clicked.connect(self.open_analysis_window_clicked)
        self.btn_hkl_viewer.clicked.connect(self.start_hkl_viewer_clicked)
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.btn_Stats5.clicked.connect(self.stats_button_clicked)
        self.btn_PlotStats1.clicked.connect(self.stats_plot_button_clicked)
        self.btn_PlotStats2.clicked.connect(self.stats_plot_button_clicked)
        self.btn_PlotStats3.clicked.connect(self.stats_plot_button_clicked)
        self.btn_PlotStats4.clicked.connect(self.stats_plot_button_clicked)
        self.btn_PlotStats5.clicked.connect(self.stats_plot_button_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        # self.rotate90degCCW.clicked.connect(self.rotate_rois)
        self.log_image.clicked.connect(self.update_image)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.chk_transpose.stateChanged.connect(self.transpose_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.chk_autoscale.stateChanged.connect(self.autoscale_checked)
        self.chk_threshold.stateChanged.connect(self.threshold_checked)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.image_view.getView().mouseDoubleClickEvent = self.on_double_click
        self.hkl_data_updated.connect(self.handle_hkl_data_update)
        
        # Set checkboxes to checked by default
        self.chk_autoscale.setChecked(True)
        self.chk_threshold.setChecked(True)
        
        # Initialize threshold label - show it since threshold is checked by default
        self.update_threshold_label()
        self.lbl_threshold_range.show()
        
        self.analysis_window = None  # Initialize as None
        self._build_mask_controls()

    def _build_mask_controls(self):
        """Build mask management UI programmatically and insert at top of sidebar."""
        from PyQt5.QtWidgets import QGroupBox, QFormLayout, QSizePolicy

        group = QGroupBox('Mask')
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        form = QFormLayout(group)
        form.setContentsMargins(6, 6, 6, 6)
        form.setVerticalSpacing(4)

        # Row 0: mask status
        self.lbl_mask_info = QLabel('No mask loaded')
        self.lbl_mask_info.setFrameShape(QLabel.Box)
        self.lbl_mask_info.setFrameShadow(QLabel.Sunken)
        self.lbl_mask_info.setWordWrap(True)
        form.addRow('Mask:', self.lbl_mask_info)

        # Row 1: masked pixel count
        self.lbl_mask_pixel_count = QLabel('0')
        self.lbl_mask_pixel_count.setFrameShape(QLabel.Box)
        self.lbl_mask_pixel_count.setFrameShadow(QLabel.Sunken)
        form.addRow('Masked pixels:', self.lbl_mask_pixel_count)

        # Row 2: Load / Show buttons
        btn_row1 = QHBoxLayout()
        self.btn_load_mask = QPushButton('Load Mask')
        self.btn_load_mask.setMinimumHeight(35)
        self.btn_load_mask.clicked.connect(self.load_mask_clicked)
        btn_row1.addWidget(self.btn_load_mask)

        self.btn_show_mask = QPushButton('Show Mask')
        self.btn_show_mask.setMinimumHeight(35)
        self.btn_show_mask.clicked.connect(self.show_mask_clicked)
        btn_row1.addWidget(self.btn_show_mask)
        form.addRow(btn_row1)

        # Row 3: Detect Dead Px / Clear buttons
        btn_row2 = QHBoxLayout()
        self.btn_detect_dead = QPushButton('Detect Dead Px')
        self.btn_detect_dead.setMinimumHeight(35)
        self.btn_detect_dead.clicked.connect(self.detect_dead_pixels_clicked)
        btn_row2.addWidget(self.btn_detect_dead)

        self.btn_clear_mask = QPushButton('Clear Mask')
        self.btn_clear_mask.setMinimumHeight(35)
        self.btn_clear_mask.clicked.connect(self.clear_mask_clicked)
        btn_row2.addWidget(self.btn_clear_mask)
        form.addRow(btn_row2)

        # Row 4: Apply mask checkbox
        self.chk_apply_mask = QCheckBox('Apply mask to display')
        self.chk_apply_mask.setChecked(True)
        form.addRow(self.chk_apply_mask)

        # Insert at top of sidebar (index 0 = above stats group box)
        self.verticalLayout_3.insertWidget(0, group)

        # Update labels if mask was auto-loaded
        self._update_mask_labels()

    # ---- Mask handler methods ----

    def load_mask_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, 'Load Mask File', '',
            'Mask files (*.edf *.npy);;EDF files (*.edf);;NumPy files (*.npy)')
        if not filepath:
            return

        try:
            if filepath.lower().endswith('.edf'):
                new_mask = self.mask_manager.load_edf(filepath)
            else:
                new_mask = self.mask_manager.load_npy(filepath)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load mask:\n{e}')
            return

        # Ask whether to add or replace
        replace = True
        if self.mask_manager.mask is not None:
            reply = QMessageBox.question(
                self, 'Combine Mask',
                'A mask is already loaded.\n\n'
                'Click Yes to ADD (OR) with existing mask.\n'
                'Click No to REPLACE the existing mask.',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                return
            replace = (reply == QMessageBox.No)

        self.mask_manager.combine_masks(new_mask, replace=replace)
        self.mask_manager.mask_sources.append(filepath)
        self.mask_manager.save_active_mask()
        self._update_mask_labels()

    def show_mask_clicked(self):
        if self.mask_manager.mask is None:
            QMessageBox.information(self, 'No Mask', 'No mask is loaded.')
            return
        if self.mask_viewer is not None:
            self.mask_viewer.close()
        self.mask_viewer = MaskViewerWindow(
            mask=self.mask_manager.mask,
            mask_path=self.mask_manager.mask_path,
            parent=self)
        self.mask_viewer.mask_updated.connect(self._on_mask_edited)
        self.mask_viewer.show()

    def detect_dead_pixels_clicked(self):
        if self.reader is None:
            QMessageBox.warning(self, 'No Reader', 'Start live view first.')
            return
        if self._dead_px_collecting:
            QMessageBox.information(self, 'In Progress',
                                    'Dead pixel detection is already collecting frames.')
            return

        self._dead_px_frames = []
        self._dead_px_collecting = True
        self.btn_detect_dead.setText(f'Collecting 0/{self._dead_px_target}...')
        self.btn_detect_dead.setEnabled(False)

    def _collect_dead_pixel_frame(self):
        """Called from update_image to accumulate frames for dead pixel detection."""
        if not self._dead_px_collecting or self.reader is None:
            return
        if self.reader.image is not None and len(self.reader.shape) >= 2:
            self._dead_px_frames.append(self.reader.image.copy())
            self.btn_detect_dead.setText(
                f'Collecting {len(self._dead_px_frames)}/{self._dead_px_target}...')

            if len(self._dead_px_frames) >= self._dead_px_target:
                self._dead_px_collecting = False
                dead_mask = self.mask_manager.detect_dead_pixels(
                    self._dead_px_frames, variance_threshold=1.0)
                self._dead_px_frames = []
                self.btn_detect_dead.setText('Detect Dead Px')
                self.btn_detect_dead.setEnabled(True)

                if dead_mask is not None:
                    num_dead = int(np.sum(dead_mask))
                    self.mask_manager.combine_masks(dead_mask)
                    self.mask_manager.save_active_mask()
                    self._update_mask_labels()
                    QMessageBox.information(
                        self, 'Dead Pixel Detection',
                        f'Detected {num_dead} stuck pixels '
                        f'(from {self._dead_px_target} frames).\n'
                        f'Added to mask. Total masked: {self.mask_manager.num_masked_pixels}')

    def clear_mask_clicked(self):
        if self.mask_manager.mask is None:
            return
        reply = QMessageBox.question(
            self, 'Clear Mask',
            'Remove the active mask?',
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.mask_manager.clear_mask()
            self._update_mask_labels()

    def _on_mask_edited(self, mask):
        self.mask_manager.mask = mask.copy()
        self.mask_manager.save_active_mask()
        self._update_mask_labels()

    def _update_mask_labels(self):
        if self.mask_manager.mask is not None:
            path = self.mask_manager.mask_path or 'In memory'
            self.lbl_mask_info.setText(os.path.basename(path))
            self.lbl_mask_info.setToolTip(path)
            count = self.mask_manager.num_masked_pixels
            pct = self.mask_manager.mask_fraction * 100
            self.lbl_mask_pixel_count.setText(f'{count:,} ({pct:.1f}%)')
        else:
            self.lbl_mask_info.setText('No mask loaded')
            self.lbl_mask_info.setToolTip('')
            self.lbl_mask_pixel_count.setText('0')

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        if self.reader is not None and self.reader.channel.isMonitorActive():
            self.timer_labels.start(int(1000/100))
            self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_plot.stop()
        self.timer_labels.stop()

    def set_pixel_ordering(self) -> None:
        """
        Checks which pixel ordering is selected on startup
        """
        if self.reader is not None:
            if self.rbtn_C.isChecked():
                self.reader.pixel_ordering = 'C'
                self.reader.image_is_transposed = True 
            elif self.rbtn_F.isChecked():
                self.reader.pixel_ordering = 'F'
                self.image_is_transposed = False
                self.reader.image_is_transposed = False
                
    def trigger_save_caches(self) -> None:
        if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
        self.file_writer.save_caches_to_h5(clear_caches=True)

    def c_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to C style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'C'
            self.reader.image_is_transposed = True

    def f_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to Fortran style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'F'
            self.image_is_transposed = False
            self.reader.image_is_transposed = False

    def open_analysis_window_clicked(self) -> None:
        """
        Opens the analysis window if the reader and image are initialized.
        Also supports launching pyFAI analysis window as a separate independent process.
        """
        if self.reader is not None:
            if self.reader.image is not None:
                # Launch pyFAI analysis window as a separate independent process
                analysis_script = os.path.join(os.path.dirname(__file__), 'pyFAI_analysis.py')
                if os.path.exists(analysis_script):
                    # Use sys.executable to ensure we use the correct Python interpreter
                    import sys
                    # Launch as fully independent process:
                    # - start_new_session: Creates new process session (Unix) for independence
                    # - stdout/stderr: Redirect to devnull to avoid blocking parent
                    # - No reference kept: Process is fully detached
                    try:
                        # Pass PV address as command-line argument
                        pv_address = self._input_channel if self._input_channel else "pvapy:image"
                        
                        # Get threshold values if threshold is enabled
                        threshold_enabled = self.chk_threshold.isChecked() if hasattr(self, 'chk_threshold') else False
                        min_thresh, max_thresh = self.get_threshold_range() if self.reader is not None else (0, 0)
                        
                        # Build command arguments
                        cmd_args = [sys.executable, analysis_script, "--pv-address", pv_address]
                        # Note: min_thresh is typically 0, so we check max_thresh > 0 to ensure valid thresholds
                        if threshold_enabled and max_thresh > 0:
                            cmd_args.extend(["--threshold-min", str(min_thresh), "--threshold-max", str(max_thresh)])

                        # Pass mask file if active
                        if self.mask_manager.mask is not None and self.mask_manager.mask_path:
                            cmd_args.extend(["--mask-file", self.mask_manager.mask_path])

                        # Don't redirect stderr so errors are visible in terminal for debugging
                        # Redirect stdout to avoid clutter, but keep stderr visible
                        process = subprocess.Popen(
                            cmd_args, 
                            cwd=os.path.dirname(os.path.dirname(analysis_script)),
                            stdout=subprocess.DEVNULL,
                            # stderr=None means it goes to terminal - helpful for debugging
                            stderr=None,  # Let stderr go to terminal so we can see errors
                            start_new_session=True  # Creates new process group (Unix) or job (Windows)
                        )
                        output_pv = f"{pv_address}:pyFAI"
                        print(f"pyFAI_analysis.py launched successfully as independent process with PV: {pv_address}")
                        print(f"Broadcasting pyFAI results to: {output_pv}")
                        if threshold_enabled and max_thresh > 0:
                            print(f"Threshold values passed: min={min_thresh}, max={max_thresh}")
                        elif threshold_enabled:
                            print(f"Warning: Threshold enabled but invalid values (min={min_thresh}, max={max_thresh})")
                        print(f"Note: Check terminal/console for any error messages if window doesn't appear")
                    except Exception as e:
                        print(f"Error launching pyFAI_analysis.py: {e}")
                        # Fallback to regular analysis window
                        self.analysis_window = AnalysisWindow(parent=self)
                        self.analysis_window.show()
                else:
                    # Fallback to regular analysis window
                    self.analysis_window = AnalysisWindow(parent=self)
                    self.analysis_window.show()        

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.
        
        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            self.stop_timers()
            self.image_view.clear()
            self.reset_rsm_vars()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.file_writer = HDF5Writer(self.reader.OUTPUT_FILE_LOCATION, self.reader)
                self.file_writer.moveToThread(self.file_writer_thread)
            else:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
                if self.file_writer_thread.isRunning():
                    self.file_writer_thread.quit()
                    self.file_writer_thread.wait()
                for roi in self.rois:
                    self.image_view.getView().removeItem(roi)
                self.rois.clear()
                self.btn_save_caches.clicked.disconnect()
                # self.reader.reader_scan_complete.disconnect()
                self.file_writer.hdf5_writer_finished.disconnect()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.file_writer.pva_reader = self.reader
            # Reconnecting signals
            self.reader.reader_scan_complete.connect(self.trigger_save_caches)
            self.file_writer.hdf5_writer_finished.connect(self.on_writer_finished)
            self.btn_save_caches.clicked.connect(self.save_caches_clicked)

            if self.reader.CACHING_MODE == 'scan':
                self.file_writer_thread.start()
            elif self.reader.CACHING_MODE == 'bin':
                self.slider = QSlider()
                self.slider.setRange(0, self.reader.BIN_COUNT-1)
                self.slider.setValue(0)
                self.slider.setOrientation(Qt.Horizontal) 
                self.slider.setTickPosition(QSlider.TicksAbove)
                self.viewer_layout.addWidget(self.slider, 1, 1)
                
            self.set_pixel_ordering()
            self.transpose_image_checked()
            self.reader.start_channel_monitor()
        except Exception as e:
            print(f'Failed to Connect to {self._input_channel}: {e}')
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            self.reset_rsm_vars()
            del self.file_writer
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
            self.btn_save_caches.clicked.disconnect()
            self.file_writer_thread.terminate()
        
        try:
            if self.reader is not None:
                if not(self.reader.rois):
                    if 'ROI' in self.reader.config:
                        self.reader.start_roi_backup_monitor()
                # Start monitoring CA metadata PVs if configured
                if 'METADATA' in self.reader.config:
                    self.reader.start_metadata_ca_monitor()
                # HKL monitoring disabled - uncomment to enable
                # self.start_hkl_monitors()
                self.start_stats_monitors()
                self.add_rois()
                self.start_timers()
        except Exception as e:
            print(f'[Diffraction Image Viewer] Error Starting Image Viewer {e}')

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            if self.reader.channel.isMonitorActive():
                self.reader.stop_channel_monitor()
            self.stop_timers()
            for key in self.stats_dialogs:
                self.stats_dialogs[key] = None
            # for roi in self.rois:
            #     self.image_view.getView().removeItem(roi)
            for hkl_pv in self.hkl_pvs.values():
                hkl_pv.clear_callbacks()
                hkl_pv.disconnect()
            self.hkl_pvs = {}
            self.hkl_data = {}
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def start_hkl_viewer_clicked(self) -> None:
        try:
            if self.reader is not None and self.reader.HKL_IN_CONFIG:
                cmd = ['python', 'viewer/hkl_3d_viewer.py',]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    universal_newlines=True
                )
                self.processes[process.pid] = process

        except Exception as e:
            print(f'[Diffraction Image Viewer] Failed to load HKL Viewer:{e}')

    def save_caches_clicked(self) -> None:
        if not self.reader.channel.isMonitorActive():  
            if not self.file_writer_thread.isRunning():
                self.file_writer_thread.start()
            self.file_writer.save_caches_to_h5(clear_caches=True)
        else:
            QMessageBox.critical(None,
                                'Error',
                                'Stop Live View to Save Cache',
                                QMessageBox.Ok)

    def stats_button_clicked(self) -> None:
        """
        Creates a popup dialog for viewing the stats of a specific button.

        This method identifies the button pressed and opens the corresponding stats dialog.
        """
        if self.reader is not None:
            sending_button = self.sender()
            text = sending_button.text()
            self.stats_dialogs[text] = RoiStatsDialog(parent=self,
                                                     stats_text=text,
                                                     timer=self.timer_labels)
            self.stats_dialogs[text].show()

    def stats_plot_button_clicked(self) -> None:
        """
        Creates a live-updating plot dialog for the corresponding Stats device.
        Button text format: 'Plot Stats1' -> stats_text = 'Stats1'
        """
        if self.reader is not None:
            sending_button = self.sender()
            # Extract stats name: 'Plot Stats1' -> 'Stats1'
            stats_text = sending_button.text().replace('Plot ', '')
            # Close existing plot dialog for this stats if open
            existing = self.stats_plot_dialogs.get(stats_text)
            if existing is not None:
                existing.close()
            self.stats_plot_dialogs[stats_text] = RoiStatsPlotDialog(
                parent=self,
                stats_text=stats_text,
                timer=self.timer_labels
            )
            self.stats_plot_dialogs[stats_text].show()

    def start_stats_monitors(self)  -> None:
        """
        Initializes monitors for updating stats values. If a PV fails to read, it's skipped.
        If all Stats PVs fail, Stats feature is disabled (like no TOML config).

        This method uses `camonitor` to observe changes in the stats PVs and update
        them in the UI accordingly.
        """
        if not self.reader.stats:
            return
        
        for stat_num in self.reader.stats.keys():
            for stat in self.reader.stats[stat_num].keys():
                pv = f"{self.reader.stats[stat_num][stat]}"
                try:
                    # Use timeout to avoid blocking on slow PVs
                    pv_value = caget(pv, timeout=0.5)
                    if pv_value is not None:
                        self.stats_data[pv] = pv_value
                        camonitor(pvname=pv, callback=self.stats_ca_callback)
                except Exception as e:
                    # Silently skip failed PVs to avoid spam
                    pass

    def stats_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Updates the stats PV value based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the specific Stat PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.stats_data[pvname] = value
        
    def on_writer_finished(self, message) -> None:
        print(message)
        self.file_writer_thread.quit()
        self.file_writer_thread.wait()

    def show_rois_checked(self) -> None:
        """
        Toggles visibility of ROIs based on the checked state of the display checkbox.
        """
        if self.reader is not None:
            if self.display_rois.isChecked():
                for roi in self.rois:
                    roi.show()
            else:
                for roi in self.rois:
                    roi.hide()

    def freeze_image_checked(self) -> None:
        """
        Toggles freezing/unfreezing of the plot based on the checked state
        without stopping the collection of PVA objects.
        """
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.stop_timers()
            else:
                self.start_timers()
    
    def transpose_image_checked(self) -> None:
        """
        Toggles whether the image data is transposed based on the checkbox state.
        """
        if self.reader is not None:
            if self.chk_transpose.isChecked():
                self.image_is_transposed = True
            else: 
                self.image_is_transposed = False

    def reset_first_plot(self) -> None:
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def rotation_count(self) -> None:
        """
        Cycles the image rotation number between 1 and 4.
        """
        self.rot_num = next(rot_gen)

    def add_rois(self) -> None:
        """
        Adds ROIs to the image viewer and assigns them color codes.

        Color Codes:
            ROI1 -- Red (#ff0000)
            ROI2 -- Blue (#0000ff)
            ROI3 -- Green (#4CBB17)
            ROI4 -- Pink (#ff00ff)
        """
        try:
            roi_colors = ['#ff0000', '#0000ff', '#4CBB17', '#ff00ff']  
            # Track how many ROIs are too big for offset calculation
            too_big_count = 0
            # TODO: can just loop through values rather than lookup with keys
            for roi_num, roi in self.reader.rois.items():
                x = roi.get("MinX", 0) if not(self.image_is_transposed) else roi.get('MinY',0)
                y = roi.get("MinY", 0) if not(self.image_is_transposed) else roi.get('MinX',0)
                width = roi.get("SizeX", 0) if not(self.image_is_transposed) else roi.get('SizeY',0)
                height = roi.get("SizeY", 0) if not(self.image_is_transposed) else roi.get('SizeX',0)
                
                # Skip ROIs with invalid dimensions
                if width <= 0 or height <= 0:
                    continue
                
                # Extract ROI number from key (e.g., "ROI1" -> 1, "ROI2" -> 2)
                try:
                    # Try to extract number from ROI key (e.g., "ROI1" -> 1)
                    # Look for digits in the key
                    digits = ''.join(c for c in roi_num if c.isdigit())
                    if digits:
                        roi_index = int(digits) - 1
                    else:
                        # Fallback: use index in dictionary
                        roi_index = list(self.reader.rois.keys()).index(roi_num)
                except (ValueError, IndexError):
                    # Final fallback: use 0 (first color)
                    roi_index = 0
                
                # Ensure index is within bounds
                roi_index = max(0, min(roi_index, len(roi_colors) - 1))
                roi_color = roi_colors[roi_index]
                
                # Check if ROI is unreasonably large (> 10000 pixels) - always resize these
                MAX_REASONABLE_SIZE = 10000
                roi_too_big = False
                original_width = width
                original_height = height
                
                if width > MAX_REASONABLE_SIZE or height > MAX_REASONABLE_SIZE:
                    roi_too_big = True
                    # Get image dimensions if available, otherwise use default
                    image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
                    if image_shape:
                        img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                        img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                    else:
                        # Fallback: use reasonable default size
                        img_width = 2000
                        img_height = 2000
                    # Force resize: Draw ROI just slightly bigger than image (100 pixels bigger)
                    x = -50
                    y = -50
                    width = img_width + 100  # Force resize to image + 100px
                    height = img_height + 100  # Force resize to image + 100px
                else:
                    # Check if ROI is larger than image (for smaller but still too large ROIs)
                    image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
                    if image_shape:
                        img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                        img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                        if width > img_width or height > img_height:
                            roi_too_big = True
                            # Draw ROI just slightly bigger than image (100 pixels bigger)
                            x = -50
                            y = -50
                            width = img_width + 100  # Always image + 100px when too large
                            height = img_height + 100  # Always image + 100px when too large
                
                # Final safety check: if still too large, force resize to reasonable size
                if width > MAX_REASONABLE_SIZE:
                    width = 2100
                    roi_too_big = True
                if height > MAX_REASONABLE_SIZE:
                    height = 2100
                    roi_too_big = True
                if roi_too_big:
                    # Add horizontal offset so multiple "too large" ROIs don't overlay
                    x = -50 + (too_big_count * 250)  # Offset by 250px per too-big ROI
                    y = -50
                    too_big_count += 1
                
                # Create ROI with final dimensions (ensured to be reasonable)
                roi = pg.ROI(pos=[x,y],
                            size=[width, height],
                            movable=False,
                            pen=pg.mkPen(color=roi_color))
                self.rois.append(roi)
                self.image_view.addItem(roi)
                
                # Add label if ROI is too big
                if roi_too_big:
                    label = pg.TextItem(f'{roi_num} - ROI too large', color=roi_color, anchor=(0, 0))
                    label.setPos(x + 5, y + 5)
                    self.image_view.addItem(label)
                
                roi.sigRegionChanged.connect(self.update_roi_region)
        except Exception as e:
            print(f'[Diffraction Image Viewer] Failed to add ROIs:{e}')


    def start_hkl_monitors(self) -> None:
        """
        Initializes camonitors for HKL values and stores them in a dictionary.
        """
        try:
            if self.reader.HKL_IN_CONFIG:
                self.hkl_config = self.reader.config["HKL"]
                if not self.hkl_pvs:
                    for section, pv_dict in self.hkl_config.items():
                        for section_key, pv_name in pv_dict.items():
                            if pv_name not in self.hkl_pvs:
                                self.hkl_pvs[pv_name] = PV(pvname=pv_name)
                for pv_name, pv_obj in self.hkl_pvs.items():
                    self.hkl_data[pv_name] = pv_obj.get() # Get current value
                    pv_obj.add_callback(callback=self.hkl_ca_callback)
                self.hkl_data_updated.emit(True)
        except Exception as e:
            print(f"[Diffraction Image Viewer] Failed to initialize HKL monitors: {e}")

    def hkl_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Callback for updating HKL values based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.hkl_data[pvname] = value
        if self.qx is not None and self.qy is not None and self.qz is not None:
            self.hkl_data_updated.emit(True)

    def handle_hkl_data_update(self):
        # HKL functionality disabled - return early to prevent errors
        return
        # if self.reader is not None and not self.stop_hkl.isChecked() and self.hkl_data:
        #     self.hkl_setup()
        #     if self.q_conv is None:
        #         raise ValueError("QConversion object is not initialized.")
        #     self.update_rsm()
  
                
    def hkl_setup(self) -> None:
        if (self.hkl_config is not None) and (not self.stop_hkl.isChecked()):
            try:
                # Get everything for the sample circles
                sample_circle_keys = [pv_name for section, pv_dict in self.hkl_config.items() if section.startswith('SAMPLE_CIRCLE') for pv_name in pv_dict.values()]
                self.sample_circle_directions = []
                self.sample_circle_names = []
                self.sample_circle_positions = []
                for pv_key in sample_circle_keys:
                    if pv_key.endswith('DirectionAxis'):
                        direction = self.hkl_data.get(pv_key)
                        if direction is None:
                            raise ValueError(f"Missing sample circle direction PV data: {pv_key}")
                        self.sample_circle_directions.append(direction)
                    elif pv_key.endswith('SpecMotorName'):
                        name = self.hkl_data.get(pv_key)
                        if name is None:
                            raise ValueError(f"Missing sample circle motor name PV data: {pv_key}")
                        self.sample_circle_names.append(name)
                    elif pv_key.endswith('Position'):
                        position = self.hkl_data.get(pv_key)
                        if position is None:
                            raise ValueError(f"Missing sample circle position PV data: {pv_key}")
                        self.sample_circle_positions.append(position)
                
                # Get everything for the detector circles
                det_circle_keys = [pv_name for section, pv_dict in self.hkl_config.items() if section.startswith('DETECTOR_CIRCLE') for pv_name in pv_dict.values()]
                self.det_circle_directions = []
                self.det_circle_names = []
                self.det_circle_positions = []
                for pv_key in det_circle_keys:
                    if pv_key.endswith('DirectionAxis'):
                        direction = self.hkl_data.get(pv_key)
                        if direction is None:
                            raise ValueError(f"Missing detector circle direction PV data: {pv_key}")
                        self.det_circle_directions.append(direction)
                    elif pv_key.endswith('SpecMotorName'):
                        name = self.hkl_data.get(pv_key)
                        if name is None:
                            raise ValueError(f"Missing detector circle motor name PV data: {pv_key}")
                        self.det_circle_names.append(name)
                    elif pv_key.endswith('Position'):
                        position = self.hkl_data.get(pv_key)
                        if position is None:
                            raise ValueError(f"Missing detector circle position PV data: {pv_key}")
                        self.det_circle_positions.append(position)
                
                # Primary Beam Direction
                primary_beam_pvs = self.hkl_config.get('PRIMARY_BEAM_DIRECTION', {}).values()
                self.primary_beam_directions = [self.hkl_data.get(axis_number) for axis_number in primary_beam_pvs]
                if any(val is None for val in self.primary_beam_directions):
                    raise ValueError("Missing primary beam direction PV data")
                
                # Inplane Reference Direction
                inplane_ref_pvs = self.hkl_config.get('INPLANE_REFERENCE_DIRECITON', {}).values()
                self.inplane_reference_directions = [self.hkl_data.get(axis_number) for axis_number in inplane_ref_pvs]
                if any(val is None for val in self.inplane_reference_directions):
                    raise ValueError("Missing inplane reference direction PV data")

                # Sample Surface Normal Direction
                surface_normal_pvs = self.hkl_config.get('SAMPLE_SURFACE_NORMAL_DIRECITON', {}).values()
                self.sample_surface_normal_directions = [self.hkl_data.get(axis_number) for axis_number in surface_normal_pvs]
                if any(val is None for val in self.sample_surface_normal_directions):
                    raise ValueError("Missing sample surface normal direction PV data")

                # UB Matrix
                ub_matrix_pv = self.hkl_config['SPEC']['UB_MATRIX_VALUE']
                self.ub_matrix = self.hkl_data.get(ub_matrix_pv)
                if self.ub_matrix is None or not isinstance(self.ub_matrix, np.ndarray) or self.ub_matrix.size != 9:
                    raise ValueError("Invalid UB Matrix data")
                self.ub_matrix = np.reshape(self.ub_matrix,(3,3))

                # Energy
                energy_pv = self.hkl_config['SPEC']['ENERGY_VALUE']
                self.energy = self.hkl_data.get(energy_pv)
                if self.energy is None:
                    raise ValueError("Missing energy PV data")
                self.energy *= 1000

                # Make sure all values are setup correctly before instantiating QConversion
                if all([self.sample_circle_directions, self.det_circle_directions, self.primary_beam_directions]):
                    self.q_conv = xu.experiment.QConversion(self.sample_circle_directions, 
                                                            self.det_circle_directions, 
                                                            self.primary_beam_directions)
                else:
                    self.q_conv = None
                    raise ValueError("QConversion initialization failed due to missing PV data.")

            except Exception as e:
                print(f'[Diffraction Image Viewer] Error Setting up HKL: {e}')
                self.q_conv = None # Reset to None on failure to prevent invalid calculations
                
    def create_rsm(self) -> np.ndarray:
        """
        Creates a reciprocal space map (RSM) from the current detector image.

        This method uses the xrayutilities package to convert detector coordinates 
        to reciprocal space coordinates (Q-space). It requires:
        - Valid detector statistics data
        - Properly initialized HKL parameters
        - Detector setup parameters (pixel directions, center channel, size, etc.)

        Returns:
            numpy.ndarray:
            - The Q-space coordinates for the current detector image,
                         or None if required data is missing.
            - shape (N, 3) containing qx, qy, and qz values.

        The conversion uses the current sample and detector angles along with the UB matrix
        to transform from angular to reciprocal space coordinates.
        """
        if self.reader is not None and self.hkl_data and (not self.stop_hkl.isChecked()):
            try:
                hxrd = xu.HXRD(self.inplane_reference_directions,
                            self.sample_surface_normal_directions, 
                            en=self.energy, 
                            qconv=self.q_conv)

                roi = [0, self.reader.shape[0], 0, self.reader.shape[1]]
                cch1 = self.hkl_data['DetectorSetup:CenterChannelPixel'][0] # Center Channel Pixel 1
                cch2 = self.hkl_data['DetectorSetup:CenterChannelPixel'][1] # Center Channel Pixel 2
                distance = self.hkl_data['DetectorSetup:Distance'] # Distance
                pixel_dir1 = self.hkl_data['DetectorSetup:PixelDirection1'] # Pixel Direction 1
                pixel_dir2 = self.hkl_data['DetectorSetup:PixelDirection2'] # PIxel Direction 2
                nch1 = self.reader.shape[0] # Number of detector pixels along direction 1
                nch2 = self.reader.shape[1] # Number of detector pixels along direction 2
                pixel_width1 = self.hkl_data['DetectorSetup:Size'][0] / nch1 # width of a pixel along direction 1
                pixel_width2 = self.hkl_data['DetectorSetup:Size'][1] / nch2 # width of a pixel along direction 2

                hxrd.Ang2Q.init_area(pixel_dir1, pixel_dir2, cch1=cch1, cch2=cch2,
                                    Nch1=nch1, Nch2=nch2, pwidth1=pixel_width1, 
                                    pwidth2=pixel_width2, distance=distance, roi=roi)
                
                angles = [*self.sample_circle_positions, *self.det_circle_positions]
                
                return hxrd.Ang2Q.area(*angles, UB=self.ub_matrix)
            except Exception as e:
                print(f'[Diffration Image Viewer] Error Creating RSM: {e}')
                return
        else:
            return
        
    def reset_rsm_vars(self) -> None:
        self.hkl_data = {}
        self.rois.clear()
        self.qx = None
        self.qy = None
        self.qz = None


    def update_rois(self) -> None:
        """
        Updates the positions and sizes of ROIs based on changes from the EPICS software.
        Loops through the cached ROIs and adjusts their parameters accordingly.
        """
        MAX_REASONABLE_SIZE = 3000
        too_big_count = 0
        for roi, roi_dict in zip(self.rois, self.reader.rois.values()):
            x_pos = roi_dict.get("MinX",0) if not(self.image_is_transposed) else roi_dict.get('MinY',0)
            y_pos = roi_dict.get("MinY",0) if not(self.image_is_transposed) else roi_dict.get('MinX',0)
            width = roi_dict.get("SizeX",0) if not(self.image_is_transposed) else roi_dict.get('SizeY',0)
            height = roi_dict.get("SizeY",0) if not(self.image_is_transposed) else roi_dict.get('SizeX',0)
            
            # Enforce size limits when updating from callback
            if width > MAX_REASONABLE_SIZE or height > MAX_REASONABLE_SIZE:
                image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
                if image_shape:
                    img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                    img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                    width = img_width + 100
                    height = img_height + 100
                    x_pos = -50 + (too_big_count * 250)  # Offset by 250px per too-big ROI
                    y_pos = -50
                    too_big_count += 1
                else:
                    width = 2100
                    height = 2100
                    x_pos = -50 + (too_big_count * 250)  # Offset by 250px per too-big ROI
                    y_pos = -50
                    too_big_count += 1
            
            roi.setPos(pos=x_pos, y=y_pos)
            roi.setSize(size=(width, height))
        self.image_view.update()

    def update_roi_region(self) -> None:
        """
        Forces the image viewer to refresh when an ROI region changes.
        Also ensures ROIs don't exceed reasonable size limits.
        """
        # Ensure ROIs don't get resized to unreasonable sizes by callbacks
        MAX_REASONABLE_SIZE = 10000
        for roi in self.rois:
            current_size = roi.size()
            if current_size[0] > MAX_REASONABLE_SIZE or current_size[1] > MAX_REASONABLE_SIZE:
                # Get image dimensions if available
                image_shape = self.reader.shape if hasattr(self.reader, 'shape') and len(self.reader.shape) >= 2 else None
                if image_shape:
                    img_width = image_shape[1] if not self.image_is_transposed else image_shape[0]
                    img_height = image_shape[0] if not self.image_is_transposed else image_shape[1]
                    roi.setSize([img_width + 100, img_height + 100])
                    roi.setPos([-50, -50])
                else:
                    roi.setSize([2100, 2100])
                    roi.setPos([-50, -50])
        self.image_view.update()

    def update_pv_prefix(self) -> None:
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()
    
    def on_double_click(self, event) -> None:
        """
        Handle double-click events on the image view to place crosshairs.
        
        Args:
            event: Mouse event containing position information
        """
        if self.reader is not None and self.reader.image is not None:
            # Get the position in scene coordinates, then map to view coordinates
            scene_pos = event.scenePos()
            view_box = self.image_view.getView().getViewBox()
            view_pos = view_box.mapSceneToView(scene_pos)
            
            # Set crosshair positions
            self.crosshair_v.setPos(view_pos.x())
            self.crosshair_h.setPos(view_pos.y())
            
            # Show crosshairs
            self.crosshair_v.show()
            self.crosshair_h.show()
            
            print(f"Crosshairs placed at: ({view_pos.x():.1f}, {view_pos.y():.1f})")
    
    def _sync_havg_yrange(self, vb, y_range):
        """Keep horizontal avg plot y-range in sync with the image view."""
        self.horizontal_avg_plot.getPlotItem().getViewBox().setYRange(
            y_range[0], y_range[1], padding=0)

    def update_mouse_pos(self, pos) -> None:
        """
        Maps the mouse position in the Image Viewer to the corresponding pixel value.

        Args:
            pos (QPointF): Position event sent by the mouse moving.
        """
        if pos is not None:
            if self.reader is not None:
                img = self.image_view.getImageItem()
                q_pointer = img.mapFromScene(pos)
                self.mouse_x, self.mouse_y = int(q_pointer.x()), int(q_pointer.y())
                self.update_mouse_labels()
    
    def update_mouse_labels(self) -> None:
            if self.reader is not None:
                if self.image is not None:
                    if 0 <= self.mouse_x < self.image.shape[0] and 0 <= self.mouse_y < self.image.shape[1]:
                        self.mouse_x_val.setText(f"{self.mouse_x}")
                        self.mouse_y_val.setText(f"{self.mouse_y}")
                        self.mouse_px_val.setText(f'{self.image[self.mouse_x][self.mouse_y]}')
                        if self.qx is not None and len(self.qx) > 0:
                            self.mouse_h.setText(f'{self.qx[self.mouse_x][self.mouse_y]:.7f}')
                            self.mouse_k.setText(f'{self.qy[self.mouse_x][self.mouse_y]:.7f}')
                            self.mouse_l.setText(f'{self.qz[self.mouse_x][self.mouse_y]:.7f}')

    def update_labels(self) -> None:
        """
        Updates the UI labels with current connection and cached data.
        """
        if self.reader is not None:
            provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
            is_connected = 'Connected' if self.reader.channel.isMonitorActive() else 'Disconnected'
            self.provider_name.setText(provider_name)
            self.is_connected.setText(is_connected)
            self.missed_frames_val.setText(f'{self.reader.frames_missed:d}')
            self.frames_received_val.setText(f'{self.reader.frames_received:d}')
            self.plot_call_id.setText(f'{self.call_id_plot:d}')
            self.update_mouse_labels()
            if len(self.reader.shape):
                self.size_x_val.setText(f'{self.reader.shape[0]:d}')
                self.size_y_val.setText(f'{self.reader.shape[1]:d}')
            self.data_type_val.setText(self.reader.display_dtype)
            self.update_threshold_label()
            self.roi1_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats1:Total_RBV', '0.0')}")
            self.roi2_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats2:Total_RBV', '0.0')}")
            self.roi3_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats3:Total_RBV', '0.0')}")
            self.roi4_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats4:Total_RBV', '0.0')}")
            self.stats5_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats5:Total_RBV', '0.0')}")

    def update_rsm(self) -> None:
        if (self.reader is not None) and (not self.stop_hkl.isChecked()):
            if self.hkl_data:
                qxyz = self.create_rsm()
                self.qx = qxyz[0].T if self.image_is_transposed else qxyz[0]
                self.qy = qxyz[1].T if self.image_is_transposed else qxyz[1]
                self.qz = qxyz[2].T if self.image_is_transposed else qxyz[2]

    def update_image(self) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            if self.reader.CACHING_MODE in ['', 'alignment', 'scan']:
                self.image = self.reader.image
            elif self.reader.CACHING_MODE == 'bin':
                index = self.slider.value()
                self.image = np.reshape(np.mean(np.stack(self.reader.cached_images[index]), axis=0), self.reader.shape) # (self.reader.shape)

            if self.image is not None:
                # Collect frame for dead pixel detection if active
                self._collect_dead_pixel_frame()
                # Apply vectorized thresholding if enabled
                if self.chk_threshold.isChecked():
                    self.image = self.apply_threshold(self.image)
                # Apply mask if enabled (before transpose/rotation)
                if self.chk_apply_mask.isChecked() and self.mask_manager.mask is not None:
                    self.image = self.mask_manager.apply_to_image(self.image)
                self.image = np.transpose(self.image) if self.image_is_transposed else self.image
                self.image = np.rot90(m=self.image, k=self.rot_num)
                if len(self.image.shape) == 2:
                    min_level, max_level = np.min(self.image), np.max(self.image)
                    if self.log_image.isChecked():
                            # Ensure non negative values
                            self.image = np.maximum(self.image, 0)
                            epsilon = 1e-10
                            self.image = np.log10(self.image + 1)
                            min_level = np.log10(max(min_level, epsilon) + 1)
                            max_level = np.log10(max_level + 1)
                    if self.first_plot:
                        self.image_view.setImage(self.image, 
                                                autoRange=False, 
                                                autoLevels=False, 
                                                levels=(min_level, max_level),
                                                autoHistogramRange=False)
                        # Set colormap separately
                        self.image_view.setColorMap(self.cet_colormap)
                        # Auto sets the max value based on first incoming image
                        if not self.chk_autoscale.isChecked():
                            self.max_setting_val.setValue(max_level)
                            self.min_setting_val.setValue(min_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(self.image,
                                                autoRange=False, 
                                                autoLevels=False, 
                                                autoHistogramRange=False)
                    # Apply autoscale if checkbox is checked (after image is set)
                    if self.chk_autoscale.isChecked():
                        self.apply_autoscale()
                # Separate image update for horizontal average plot
                self.horizontal_avg_plot.plot(x=np.mean(self.image, axis=0), 
                                            y=np.arange(self.image.shape[1]), 
                                            clear=True)

                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    
    def update_min_max_setting(self) -> None:
        """
        Updates the min/max pixel levels in the Image Viewer based on UI settings.
        """
        # Only update if autoscale is not checked (to prevent feedback loop)
        if not self.chk_autoscale.isChecked():
            min = self.min_setting_val.value()
            max = self.max_setting_val.value()
            self.image_view.setLevels(min, max)
    
    def autoscale_checked(self) -> None:
        """
        Handles autoscale checkbox state changes.
        When checked, calculates and sets min/max based on 5th and 95th percentiles of intensity histogram.
        """
        if self.chk_autoscale.isChecked() and self.image is not None:
            self.apply_autoscale()
    
    def apply_autoscale(self) -> None:
        """
        Calculates and applies autoscale based on 5th and 95th percentiles of the intensity histogram.
        Sets min to 5th percentile and max to 95th percentile.
        """
        if self.image is not None:
            # Flatten the image to get all intensity values
            intensities = self.image.flatten()
            
            # Remove any NaN or infinite values
            intensities = intensities[np.isfinite(intensities)]
            
            if len(intensities) > 0:
                # Calculate 5th and 95th percentiles
                min_percentile = np.percentile(intensities, 5)
                max_percentile = np.percentile(intensities, 95)
                
                # Update the spinbox values (this will trigger update_min_max_setting)
                # Temporarily block signals to prevent feedback loop
                self.min_setting_val.blockSignals(True)
                self.max_setting_val.blockSignals(True)
                self.min_setting_val.setValue(min_percentile)
                self.max_setting_val.setValue(max_percentile)
                self.min_setting_val.blockSignals(False)
                self.max_setting_val.blockSignals(False)
                
                # Apply the levels directly
                self.image_view.setLevels(min_percentile, max_percentile)
    
    def get_threshold_range(self) -> tuple:
        """
        Determines the threshold range based on the data type.
        
        Returns:
            tuple: (min_threshold, max_threshold) based on data type
        """
        if self.reader is None or self.reader.display_dtype is None:
            return (0, 0)
        
        dtype_map = {
            'ubyteValue': (0, 2**8 - 2),      # uint8: 0 to 254 (accounting for zero-based indexing)
            'ushortValue': (0, 2**16 - 2),    # uint16: 0 to 65534 (accounting for zero-based indexing)
            'uintValue': (0, 2**32 - 2),      # uint32: 0 to 4294967294 (accounting for zero-based indexing)
            'floatValue': (0, 2**16 - 1),     # float32: 0 to 65535
            'doubleValue': (0, 2**16 - 1),    # float64: 0 to 65535
        }
        
        return dtype_map.get(self.reader.display_dtype, (0, 0))
    
    def update_threshold_label(self) -> None:
        """
        Updates the threshold range label based on current data type.
        """
        min_thresh, max_thresh = self.get_threshold_range()
        self.lbl_threshold_range.setText(f"{min_thresh} to {max_thresh}")
    
    def threshold_checked(self) -> None:
        """
        Handles threshold checkbox state changes.
        Updates the label visibility and applies thresholding if enabled.
        """
        if self.chk_threshold.isChecked():
            self.update_threshold_label()
            self.lbl_threshold_range.show()
        else:
            self.lbl_threshold_range.hide()
    
    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Applies vectorized thresholding to the image based on data type limits.
        Values below min_thresh are set to min_thresh, values above max_thresh are set to 0.
        
        Args:
            image: Input image array
            
        Returns:
            Thresholded image array
        """
        min_thresh, max_thresh = self.get_threshold_range()
        if min_thresh == 0 and max_thresh == 0:
            return image
        
        # Vectorized thresholding: 
        # - Values below min_thresh are set to min_thresh
        # - Values above max_thresh are set to 0
        result = image.copy()
        result[result < min_thresh] = min_thresh
        result[result > max_thresh] = 0
        return result
    
    def closeEvent(self, event):
        """
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        del self.stats_dialogs # otherwise dialogs stay in memory
        del self.stats_plot_dialogs
        if self.mask_viewer is not None:
            self.mask_viewer.close()
        if self.file_writer_thread.isRunning():
            self.file_writer_thread.quit()
            self.file_writer_thread
        super(DiffractionImageWindow,self).closeEvent(event)

def main():
    app = QApplication(sys.argv)
    # size_manager = SizeManager(app=app)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()