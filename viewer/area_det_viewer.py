import os
import sys
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import xrayutilities as xu
from PyQt5 import uic
# from epics import caget
from epics import camonitor, caget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
# Custom imported classes
from generators import rotation_cycle
from pva_reader import PVAReader
from roi_stats_dialog import RoiStatsDialog
# from unused_files.pv_setup_dialog import PVSetupDialog
from analysis_window import AnalysisWindow 


rot_gen = rotation_cycle(1,5)         


class ConfigDialog(QDialog):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
            roi_config (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/pv_config.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ""
        self.roi_config =  ""
        # class can be prefilled with text
        self.init_ui()
        
        # Connecting signasl to 
        # self.btn_edit.clicked.connect(self.json_open_file_dialog)
        self.btn_browse.clicked.connect(self.browse_file_dialog)
        # self.btn_create.clicked.connect(self.new_pv_setup)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_roi_config.setText(self.le_roi_config.text())

    def browse_file_dialog(self) -> None:
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_roi_config.setText(self.pvs_path)

    # def new_pv_setup(self) -> None:
    #     """
    #     Opens a new window for setting up a new PV configuration within the UI.
    #     """
    #     self.new_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='w', path=None)
    
    # def edit_pv_setup(self) -> None:
    #     """
    #     Opens a window for editing an existing PV configuration.
    #     """
    #     if self.le_edit_file_path.text() != '':
    #         self.edit_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='r+', path=self.pvs_path)
    #     else:
    #         print('file path empty')

    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the DiffractionImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.roi_config = self.le_roi_config.text()
        if osp.isfile(self.roi_config) or (self.roi_config == ''):
            self.image_viewer = DiffractionImageWindow(input_channel=self.input_channel,
                                            file_path=self.roi_config,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class DiffractionImageWindow(QMainWindow):

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
        self.rois: list[pg.ROI] = []
        self.stats_dialog = {}
        self.stats_data = {}
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path
        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        # self.timer_rsm = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.timeout.connect(self.update_rois)
        # For testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50
        # HKL values
        self.hkl_config = None
        self.hkl_data = {}
        self.qx = None
        self.qy = None
        self.qz = None
        self.processes = {}
        
        # Adding widgets manually to have better control over them
        plot = pg.PlotItem()        
        self.image_view = pg.ImageView(view=plot)
        self.viewer_layout.addWidget(self.image_view,1,1)
        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        # second is a separate plot to show the horiontal avg of peaks in the image
        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(175)
        self.horizontal_avg_plot.setYLink(self.image_view.getView())
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 1,0)

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.btn_analysis_window.clicked.connect(self.open_analysis_window_clicked)
        self.btn_hkl_viewer.clicked.connect(self.start_hkl_viewer)
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.btn_Stats5.clicked.connect(self.stats_button_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        # self.rotate90degCCW.clicked.connect(self.rotate_rois)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.chk_transpose.stateChanged.connect(self.transpose_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.log_image.clicked.connect(self.update_image)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))
        self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_plot.stop()
        self.timer_labels.stop()

    #TODO: CHECK With 4id network camera to test if
    # start of X,Y and size of X,Y line up when transposed
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
            # A double check to make sure there isn't a connection already when starting
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.set_pixel_ordering()
                self.transpose_image_checked()
                self.reader.start_channel_monitor()
            else:
                self.stop_timers()
                self.reader.stop_channel_monitor()
                del self.reader
                for roi in self.rois:
                    self.image_view.getView().removeItem(roi)
                self.rois = []
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.set_pixel_ordering()
                self.transpose_image_checked()
                self.reader.start_channel_monitor()
        except:
            print(f'Failed to Connect to {self._input_channel}')
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
        if self.reader is not None:
            if not(self.reader.rois):
                    if ('ROI' in self.reader.config):
                        self.reader.start_roi_backup_monitor()
            self.start_stats_monitors()
            self.add_rois()
            self.start_timers()
            try:
                self.init_hkl()
                if self.hkl_data:
                    qxyz = self.create_rsm()
                    self.qx = qxyz[0].T if self.image_is_transposed else qxyz[0]
                    self.qy = qxyz[1].T if self.image_is_transposed else qxyz[1]
                    self.qz = qxyz[2].T if self.image_is_transposed else qxyz[2]
            except Exception as e:
                print('failed to create rsm: %s' % e)

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            for key in self.stats_dialog:
                self.stats_dialog[key] = None
            for roi in self.rois:
                self.image_view.getView().removeItem(roi)
            self.rois = []
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

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
        except:
            print("Failed to Connect to Stats CA Monitors")

    def stats_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Updates the stats PV value based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the specific Stat PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.stats_data[pvname] = value
        
    def stats_button_clicked(self) -> None:
        """
        Creates a popup dialog for viewing the stats of a specific button.

        This method identifies the button pressed and opens the corresponding stats dialog.
        """
        if self.reader is not None:
            sending_button = self.sender()
            text = sending_button.text()
            self.stats_dialog[text] = RoiStatsDialog(parent=self, 
                                                     stats_text=text, 
                                                     timer=self.timer_labels)
            self.stats_dialog[text].show()
    
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

    # def rotate_rois(self) -> None:
    #     img_center = img_center = np.array(self.image_view.image.shape[::-1]) / 2.0 # width, height

    #     for roi in self.rois:
    #         roi_center = np.array(roi.pos()) + np.array(roi.size()) / 2.0
    #         roi.setPos(pos=img_center)
    #         roi.rotate(-90)
    #         roi_pos = roi.pos()
    #         roi.setPos(pos=roi_pos[0]-img_center[0], y=roi_pos[1]-img_center[1]/2)

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
            print(f'Failed to add ROIs:{e}')

    def start_hkl_viewer(self) -> None:
        try:
            if self.reader is not None and self.reader.HKL_IN_CONFIG:
                # qx = self.qx.flatten()
                # qy = self.qy.flatten()
                # qz = self.qz.flatten()
                # intensity = self.reader.image.flatten()

                # np.save('qx.npy', qx)
                # np.save('qy.npy', qy)
                # np.save('qz.npy', qz)
                # np.save('intensity.npy', intensity)

                cmd = ['python', 'viewer/hkl_test.py',]
                #        '--qx-file', 'qx.npy',
                #        '--qy-file', 'qy.npy',
                #        '--qz-file', 'qz.npy',
                #        '--intensity-file', 'intensity.npy']

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    universal_newlines=True
                )

                self.processes[process.pid] = process

        except Exception as e:
            print(f'Failed to load HKL Viewer:{e}')


    def start_hkl_monitors(self) -> None:
        """
        Initializes camonitors for HKL values and stores them in a dictionary.
        """
        try:
            if "HKL" in self.reader.config:
                self.hkl_config = self.reader.config["HKL"]

                # Monitor each HKL parameter
                for section, pv_dict in self.hkl_config.items():
                    for key, pv_name in pv_dict.items():
                        self.hkl_data[pv_name] = caget(pv_name)
                        camonitor(pvname=pv_name, callback=self.hkl_ca_callback)
        except Exception as e:
            print(f"Failed to initialize HKL monitors: {e}")

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
            self.update_rsm()

    def init_hkl(self) -> None:
        """
        Initializes HKL parameters by setting up monitors for each HKL value.
        """
        self.start_hkl_monitors()
        self.hkl_setup()
        
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
                        self.sample_circle_directions.append(self.hkl_data[pv_key])
                    elif pv_key.endswith('SpecMotorName'):
                        self.sample_circle_names.append(self.hkl_data[pv_key])
                    elif pv_key.endswith('Position'):
                        self.sample_circle_positions.append(self.hkl_data[pv_key])
                # Get everything for the detector circles
                det_circle_keys = [pv_name for section, pv_dict in self.hkl_config.items() if section.startswith('DETECTOR_CIRCLE') for pv_name in pv_dict.values()]
                self.det_circle_directions = []
                self.det_circle_names = []
                self.det_circle_positions = []
                for pv_key in det_circle_keys:
                    if pv_key.endswith('DirectionAxis'):
                        self.det_circle_directions.append(self.hkl_data[pv_key])
                    elif pv_key.endswith('SpecMotorName'):
                        self.det_circle_names.append(self.hkl_data[pv_key])
                    elif pv_key.endswith('Position'):
                        self.det_circle_positions.append(self.hkl_data[pv_key])
                # Primary Beam Direction
                self.primary_beam_directions = [self.hkl_data[axis_number] for axis_number in self.hkl_config['PRIMARY_BEAM_DIRECTION'].values()]
                # Inplane Reference Direction
                self.inplane_reference_directions = [self.hkl_data[axis_number] for axis_number in self.hkl_config['INPLANE_REFERENCE_DIRECITON'].values()]
                # Sample Surface Normal Direction
                self.sample_surface_normal_directions = [self.hkl_data[axis_number] for axis_number in self.hkl_config['SAMPLE_SURFACE_NORMAL_DIRECITON'].values()]
                # UB Matrix
                self.ub_matrix = self.hkl_data[self.hkl_config['SPEC']['UB_MATRIX_VALUE']]
                self.ub_matrix  = np.reshape(self.ub_matrix,(3,3))
                # Energy
                self.energy = self.hkl_data[self.hkl_config['SPEC']['ENERGY_VALUE']] * 1000
                # Make sure all values are setup correctly before instantiating QConversion
                if self.sample_circle_directions and self.det_circle_directions and self.primary_beam_directions:
                    # Class for the conversion of angular coordinates to momentum space 
                    self.q_conv = xu.experiment.QConversion(self.sample_circle_directions, 
                                                            self.det_circle_directions, 
                                                            self.primary_beam_directions)
            except Exception as e:
                print(f'Error Setting up HKL: {e}')
                return
         

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
        if self.hkl_data and (not self.stop_hkl.isChecked()):
            try:
                hxrd = xu.HXRD(self.inplane_reference_directions,
                            self.sample_surface_normal_directions, 
                            en=self.energy, 
                            qconv=self.q_conv)

                if self.stats_data:
                    if f"{self.reader.pva_prefix}:Stats4:Total_RBV" in self.stats_data:
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
                print(f'Error Creating RSM: {e}')
                return
        else:
            return

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
                x, y = q_pointer.x(), q_pointer.y() 
                if self.image is not None:
                    if 0 <= x < self.image.shape[0] and 0 <= y < self.image.shape[1]:
                        self.mouse_x_val.setText(f"{x:.3f}")
                        self.mouse_y_val.setText(f"{y:.3f}")
                        self.mouse_px_val.setText(f'{self.image[int(x)][int(y)]}')
                        if self.hkl_data:
                            self.mouse_h.setText(f'{self.qx[int(x)][int(y)]}')
                            self.mouse_k.setText(f'{self.qy[int(x)][int(y)]}')
                            self.mouse_l.setText(f'{self.qz[int(x)][int(y)]}')

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
                self.hkl_setup()
                self.qx = self.create_rsm()[0].T if self.image_is_transposed else self.create_rsm()[0]
                self.qy = self.create_rsm()[1].T if self.image_is_transposed else self.create_rsm()[1]
                self.qz = self.create_rsm()[2].T if self.image_is_transposed else self.create_rsm()[2]

    def update_image(self) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            image = self.reader.image
            if image is not None:
                self.image = np.rot90(image, k=self.rot_num).T if self.image_is_transposed else np.rot90(image, k=self.rot_num)
                if len(self.image.shape) == 2:
                    min_level, max_level = np.min(self.image), np.max(self.image)
                    if self.log_image.isChecked():
                            self.image = np.log1p(self.image + 1)
                            min_level = np.log1p(min_level + 1)
                            max_level = np.log1p(max_level + 1)
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
                                              y=np.arange(0,self.image.shape[1]), 
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
        del self.stats_dialog # otherwise dialogs stay in memory
        super(DiffractionImageWindow,self).closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())