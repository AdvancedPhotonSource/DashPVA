import os
import sys
import time
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import xrayutilities as xu
from PyQt5 import uic
# from epics import caget
from epics import PV, pv
from epics import camonitor, caget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QFileDialog, QSlider
# Custom imported classes
from roi_stats_dialog import RoiStatsDialog
from analysis_window import AnalysisWindow 
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import rotation_cycle
from utils import PVAReader, HDF5Writer
# from ..utils.size_manager import SizeManager


rot_gen = rotation_cycle(1,5)         


class ConfigDialog(QDialog):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/pv_config.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ''
        # class can be prefilled with text
        self.init_ui()

        self.btn_accept_reject.accepted.connect(self.dialog_accepted)

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())

    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the DiffractionImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.image_viewer = DiffractionImageWindow(input_channel=self.input_channel)


class DiffractionImageWindow(QMainWindow, LogMixin):
    hkl_data_updated = pyqtSignal(bool)

    def __init__(self, input_channel=None):
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
        """
        super(DiffractionImageWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        try:
            self.set_log_manager(viewer_name="AreaDetViewer")
        except Exception:
            pass
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
        self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)

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
        
        plot = pg.PlotItem()        
        self.image_view = pg.ImageView(view=plot)
        self.viewer_layout.addWidget(self.image_view,0,1)
        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        # second is a separate plot to show the horiontal avg of peaks in the image
        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(200)
        self.horizontal_avg_plot.getAxis('bottom').setLabel(text='Horizontal Avg.')
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 0,0)

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
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.hkl_data_updated.connect(self.handle_hkl_data_update)

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
        """
        if self.reader is not None:
            if self.reader.image is not None:
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
                # print('[start_live_view] reader is None — creating new PVAReader')
                self.reader = PVAReader(input_channel=self._input_channel)
                # print(f'[start_live_view] reader created: input_channel={self.reader.input_channel} ROI_IN_CONFIG={self.reader.ROI_IN_CONFIG} HKL_IN_CONFIG={self.reader.HKL_IN_CONFIG} CACHING_MODE={self.reader.CACHING_MODE}')
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
                self.reader = PVAReader(input_channel=self._input_channel)
                # print(f'[start_live_view] new reader created: input_channel={self.reader.input_channel} ROI_IN_CONFIG={self.reader.ROI_IN_CONFIG} HKL_IN_CONFIG={self.reader.HKL_IN_CONFIG} CACHING_MODE={self.reader.CACHING_MODE}')
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
                    self.start_hkl_monitors()
                    self.start_stats_monitors()
                    self.add_rois()
                    self.start_timers()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.exception(f'Error starting image viewer: {e}')

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

    def start_stats_monitors(self)  -> None:
        """
        Initializes monitors for updating stats values.

        This method uses `camonitor` to observe changes in the stats PVs and update
        them in the UI accordingly.
        """
        try:
            if self.reader.stats:
                for stat_num in self.reader.stats.keys():
                    for stat in self.reader.stats[stat_num].keys():
                        pv = f"{self.reader.stats[stat_num][stat]}"
                        self.stats_data[pv] = caget(pv)
                        camonitor(pvname=pv, callback=self.stats_ca_callback)
        except Exception as e:
            print(f"[Diffraction Image Viewer] Failed to Connect to Stats CA Monitors: {e}")

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
            # TODO: can just loop through values rather than lookup with keys
            for roi_num, roi in self.reader.rois.items():
                x = roi.get("MinX", 0) if not(self.image_is_transposed) else roi.get('MinY',0)
                y = roi.get("MinY", 0) if not(self.image_is_transposed) else roi.get('MinX',0)
                width = roi.get("SizeX", 0) if not(self.image_is_transposed) else roi.get('SizeY',0)
                height = roi.get("SizeY", 0) if not(self.image_is_transposed) else roi.get('SizeX',0)
                roi_color = int(roi_num[-1]) - 1 
                roi = pg.ROI(pos=[x,y],
                            size=[width, height],
                            movable=False,
                            pen=pg.mkPen(color=roi_colors[roi_color]))
                self.rois.append(roi)
                self.image_view.addItem(roi)
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
        if self.reader is not None and not self.stop_hkl.isChecked() and self.hkl_data:
            self.hkl_setup()
            if self.q_conv is None:
                raise ValueError("QConversion object is not initialized.")
            self.update_rsm()
  
                
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
        for roi, roi_dict in zip(self.rois, self.reader.rois.values()):
            x_pos = roi_dict.get("MinX",0) if not(self.image_is_transposed) else roi_dict.get('MinY',0)
            y_pos = roi_dict.get("MinY",0) if not(self.image_is_transposed) else roi_dict.get('MinX',0)
            width = roi_dict.get("SizeX",0) if not(self.image_is_transposed) else roi_dict.get('SizeY',0)
            height = roi_dict.get("SizeY",0) if not(self.image_is_transposed) else roi_dict.get('SizeX',0)
            roi.setPos(pos=x_pos, y=y_pos)
            roi.setSize(size=(width, height))
        self.image_view.update()

    def update_roi_region(self) -> None:
        """
        Forces the image viewer to refresh when an ROI region changes.
        """
        self.image_view.update()

    def update_pv_prefix(self) -> None:
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()
    
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
                        # Auto sets the max value based on first incoming image
                        self.max_setting_val.setValue(max_level)
                        self.min_setting_val.setValue(min_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(self.image,
                                                autoRange=False, 
                                                autoLevels=False, 
                                                autoHistogramRange=False)
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
        min = self.min_setting_val.value()
        max = self.max_setting_val.value()
        self.image_view.setLevels(min, max)
    
    def closeEvent(self, event):
        """
        Custom close event to clean up resources, including stat dialogs.

        Args:
            event (QCloseEvent): The close event triggered when the main window is closed.
        """
        del self.stats_dialogs # otherwise dialogs stay in memory
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