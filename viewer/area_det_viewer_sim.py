import sys
import toml
# import copy
import time
import numpy as np
import os.path as osp
import pvaccess as pva
import pyqtgraph as pg
import xrayutilities as xu
from PyQt5 import uic
# from epics import caget
from epics import camonitor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
# Custom imported classes
from generators import rotation_cycle
from roi_stats_dialog import RoiStatsDialog
from pv_setup_dialog import PVSetupDialog
from analysis_window import AnalysisWindow 


max_cache_size = 900 #TODO: Put this in the config file 
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
        self.btn_create.clicked.connect(self.new_pv_setup)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self):
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_roi_config.setText(self.le_roi_config.text())

    def browse_file_dialog(self):
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_roi_config.setText(self.pvs_path)

    def new_pv_setup(self):
        """
        Opens a new window for setting up a new PV configuration within the UI.
        """
        self.new_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='w', path=None)
    
    def edit_pv_setup(self):
        """
        Opens a window for editing an existing PV configuration.
        """
        if self.le_edit_file_path.text() != '':
            self.edit_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='r+', path=self.pvs_path)
        else:
            print('file path empty')

    def dialog_accepted(self):
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the ImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.roi_config = self.le_roi_config.text()
        if osp.isfile(self.roi_config) or (self.roi_config == ''):
            self.image_viewer = ImageWindow(input_channel=self.input_channel,
                                            file_path=self.roi_config,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class PVA_Reader:

    def __init__(self, input_channel='s6lambda1:Pva1:Image', provider=pva.PVA, config_filepath: str = 'pv_configs/metadata_pvs.toml'):
        """
        Initializes the PVA Reader for monitoring connections and handling image data.

        Args:
            input_channel (str): Input channel for the PVA connection.
            provider (protocol): The protocol for the PVA channel.
            config_filepath (str): File path to the configuration TOML file.
        """
        self.input_channel = input_channel        
        self.provider = provider
        self.config_filepath = config_filepath
        self.channel = pva.Channel(self.input_channel, self.provider)
        self.pva_prefix = "dp-ADSim"
        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.pixel_ordering = 'C'
        self.image_is_transposed = True
        self.attributes = []
        self.timestamp = None
        self.data_type = None
        # variables used for parsing analysis PV
        self.analysis_index = None
        self.analysis_exists = True
        self.analysis_attributes = {}
        # variables used for later logic
        self.last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.id_diff = 0
        # variables used for ROI and Stats PVs from config
        self.config = {}
        self.rois = {}
        self.stats = {}

        if self.config_filepath != '':
            with open(self.config_filepath, 'r') as toml_file:
                # loads the pvs in the toml file into a python dictionary
                self.config:dict = toml.load(toml_file)
                self.stats:dict = self.config["stats"]
                if self.config["ConsumerType"] == "spontaneous":
                    # TODO: change to dictionaries that store postions as keys and pv as value
                    self.analysis_cache_dict = {"Intensity": {},
                                                "ComX": {},
                                                "ComY": {},
                                                "Position": {}} 

    def pva_callbackSuccess(self, pv):
        """
        Callback for handling monitored PV changes.

        Args:
            pv (PvaObject): The PV object received by the channel monitor.
        """
        self.pva_object = pv
        self.parse_image_data_type()
        self.pva_to_image()
        self.parse_pva_attributes()
        self.parse_roi_pvs()
        if (self.analysis_index is None) and (self.analysis_exists): #go in with the assumption analysis exists, is changed to false otherwise
            self.analysis_index = self.locate_analysis_index()
            # print(self.analysis_index)
        if self.analysis_exists:
            self.analysis_attributes = self.attributes[self.analysis_index]
            if self.config["ConsumerType"] == "spontaneous":
                # turns axis1 and axis2 into a tuple
                incoming_coord = (self.analysis_attributes["value"][0]["value"].get("Axis1", 0.0), 
                                  self.analysis_attributes["value"][0]["value"].get("Axis2", 0.0))
                # use a tuple as a key so that we can check if there is a repeat position
                self.analysis_cache_dict["Intensity"].update({incoming_coord: self.analysis_cache_dict["Intensity"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("Intensity", 0.0)})
                self.analysis_cache_dict["ComX"].update({incoming_coord: self.analysis_cache_dict["ComX"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComX", 0.0)})
                self.analysis_cache_dict["ComY"].update({incoming_coord:self.analysis_cache_dict["ComY"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComY", 0.0)})
                # double storing of the postion, will find out if needed
                self.analysis_cache_dict["Position"][incoming_coord] = incoming_coord
                
    def parse_image_data_type(self):
        """
        Parses the PVA Object to determine the incoming data type.
        """
        if self.pva_object is not None:
            try:
                self.data_type = list(self.pva_object['value'][0].keys())[0]
            except:
                self.data_type = "could not detect"
    
    def parse_pva_attributes(self):
        """
        Converts the PVA object to a Python dictionary and extracts its attributes.
        """
        if self.pva_object is not None:
            self.attributes: list = self.pva_object.get().get("attribute", [])
    
    def locate_analysis_index(self):
        """
        Locates the index of the analysis attribute in the PVA attributes.

        Returns:
            int: The index of the analysis attribute or None if not found.
        """
        if self.attributes:
            for i in range(len(self.attributes)):
                attr_pv: dict = self.attributes[i]
                if attr_pv.get("name", "") == "Analysis":
                    return i
            else:
                self.analysis_exists = False
                return None
            
    def parse_roi_pvs(self):
        """
        Parses attributes to extract ROI-specific PV information.
        """
        if self.attributes:
            for i in range(len(self.attributes)):
                attr_pv: dict = self.attributes[i]
                attr_name:str = attr_pv.get("name", "")
                if "ROI" in attr_name:
                    name_components = attr_name.split(":")
                    prefix = name_components[0]
                    roi_key = name_components[1]
                    pv_key = name_components[2]
                    pv_value = attr_pv["value"][0]["value"]
                    # can't append simply by using 2 keys in a row, there must be a value to call to then add to
                    # then adds the key to the inner dictionary with update
                    self.rois.setdefault(roi_key, {}).update({pv_key: pv_value})
            
    def pva_to_image(self):
        """
        Converts the PVA Object to an image array and determines if a frame was missed.
        """
        try:
            if self.pva_object is not None:
                self.frames_received += 1
                # Parses dimensions and reshapes array into image
                if 'dimension' in self.pva_object:
                    self.shape = tuple([dim['size'] for dim in self.pva_object['dimension']])
                    self.image = np.array(self.pva_object['value'][0][self.data_type])
                    # reshapes but also transposes image so it is viewed correctly
                    self.image = np.reshape(self.image, self.shape, order=self.pixel_ordering).T if self.image_is_transposed else np.reshape(self.image, self.shape, order=self.pixel_ordering)
                else:
                    self.image = None
                # Check for missed frame starts here
                current_array_id = self.pva_object['uniqueId']
                if self.last_array_id is not None: 
                    self.id_diff = current_array_id - self.last_array_id - 1
                    if (self.id_diff > 0):
                        self.frames_missed += self.id_diff 
                    else:
                        self.id_diff = 0
                self.last_array_id = current_array_id
        except:
            print("failed process image")
            self.frames_missed += 1
            # return 1
            
    def start_channel_monitor(self):
        """
        Subscribes to the PVA channel with a callback function and starts monitoring for PV changes.
        """
        self.channel.subscribe('pva callback success', self.pva_callbackSuccess)
        self.channel.startMonitor()
        
    def stop_channel_monitor(self):
        """
        Stops all monitoring and callback functions.
        """
        self.channel.unsubscribe('pva callback success')
        self.channel.stopMonitor()

    def get_frames_missed(self):
        """
        Returns the number of frames missed.

        Returns:
            int: The number of missed frames.
        """
        return self.frames_missed

    def get_pva_image(self):
        """
        Returns the current PVA image.

        Returns:
            numpy.ndarray: The current image array.
        """
        return self.image
    
    def get_attributes_dict(self):
        """
        Returns the attributes of the current PVA object.

        Returns:
            list: The attributes of the current PVA object.
        """
        return self.attributes


class ImageWindow(QMainWindow):

    def __init__(self, input_channel='s6lambda1:Pva1:Image', file_path=''): 
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
            file_path (str): The file path for loading configuration.
        """
        super(ImageWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle('Image Viewer with PVAaccess')
        self.show()
        # Initializing important variables
        self.reader = None
        self.call_id_plot = 0
        self.first_plot = True
        self.rot_num = 0
        self.rois = []
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
        # self.timer_rsm.timeout.connect(self.update_rsm)
        # For testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50
        
        # Adding widgets manually to have better control over them
        plot = pg.PlotItem()        
        self.image_view = pg.ImageView(view=plot)
        self.viewer_layout.addWidget(self.image_view,1,1)
        self.image_view.view.getAxis('left').setLabel(text='Row [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='Columns [pixels]')
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
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.btn_Stats5.clicked.connect(self.stats_button_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.log_image.clicked.connect(self.update_image)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    def start_timers(self):
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))
        self.timer_plot.start(int(1000/self.plotting_frequency.value()))
        # if self.hkl_data:
        #     self.timer_rsm.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self):
        """
        Stops the updating of main window labels and plots.
        """
        self.timer_plot.stop()
        self.timer_labels.stop()
        # if self.hkl_data:
        #     self.timer_rsm.stop()

    def c_ordering_clicked(self):
        """
        Sets the pixel ordering to C style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'C'

    def f_ordering_clicked(self):
        """
        Sets the pixel ordering to Fortran style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'F'

    def open_analysis_window_clicked(self):
        """
        Opens the analysis window if the reader and image are initialized.
        """
        if self.reader is not None:
            if self.reader.image is not None:
                self.analysis_window = AnalysisWindow(parent=self)
                self.analysis_window.show()

    def start_live_view_clicked(self):
        """
        Initializes the connections to the PVA channel using the provided Channel Name.
        
        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            # A double check to make sure there isn't a connection already when starting
            if self.reader is None:
                self.reader = PVA_Reader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.reader.start_channel_monitor()

            else:
                self.stop_timers()
                self.reader.stop_channel_monitor()
                del self.reader
                self.reader = PVA_Reader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path)
                self.reader.start_channel_monitor()

            # Additional functions that aren't affected by whether the PVA reader is None or not
            if self.reader is not None:
                self.start_stats_monitors()
                self.add_rois()
                self.init_hkl()
                self.start_timers()
                self.qx, self.qy, self.qz = self.create_rsm() 


                # print(self.qx, self.qy, self.qz)
        except:
            print(f'Failed to Connect to {self._input_channel}')
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
    def stop_live_view_clicked(self):
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            for key in self.stats_dialog:
                self.stats_dialog[key] = None
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def start_stats_monitors(self):
        """
        Initializes monitors for updating stats values.

        This method uses `camonitor` to observe changes in the stats PVs and update
        them in the UI accordingly.
        """
        try:
            if self.reader.stats:
                for stat in self.reader.stats.keys():
                    for pv in self.reader.stats[stat].keys():
                        name = f"{self.reader.stats[stat][pv]}"
                        camonitor(pvname=name, callback=self.stats_ca_callback)
        except:
            print("Failed to Connect to Stats CA Monitors")

    def stats_ca_callback(self, pvname, value, **kwargs):
        """
        Updates the stats PV value based on changes observed by `camonitor`.

        Args:
            pvname (str): The name of the specific Stat PV that has been updated.
            value: The new value sent by the monitor for the PV.
            **kwargs: Additional keyword arguments sent by the monitor.
        """
        self.stats_data[pvname] = value
        
    def stats_button_clicked(self):
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
    
    def show_rois_checked(self):
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

    def freeze_image_checked(self):
        """
        Toggles freezing/unfreezing of the plot based on the checked state
        without stopping the collection of PVA objects.
        """
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.stop_timers()
            else:
                self.start_timers()
    
    def transpose_image_checked(self):
        """
        Toggles whether the image data is transposed based on the checkbox state.
        """
        if self.reader is not None:
            if self.chk_transpose.isChecked():
                self.reader.image_is_transposed = True
            else: 
                self.reader.image_is_transposed = False

    def reset_first_plot(self):
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def rotation_count(self):
        """
        Cycles the image rotation number between 1 and 4.
        """
        self.rot_num = next(rot_gen)

    def add_rois(self):
        """
        Adds ROIs to the image viewer and assigns them color codes.

        Color Codes:
            ROI1 -- Red (#ff0000)
            ROI2 -- Blue (#0000ff)
            ROI3 -- Green (#4CBB17)
            ROI4 -- Pink (#ff00ff)
        """
        roi_colors = ['ff0000', '0000ff', '4CBB17', 'ff00ff']
        for i, roi in enumerate(self.reader.rois.keys()):
            x = self.reader.rois[roi].get("MinX", 0)
            y = self.reader.rois[roi].get("MinY", 0)
            width = self.reader.rois[roi].get("SizeX", 0)
            height = self.reader.rois[roi].get("SizeY", 0)
            roi = pg.ROI(pos=[x,y],
                         size=[width, height],
                         movable=False,
                         pen=pg.mkPen(roi_colors[i]))
            self.rois.append(roi)
            self.image_view.addItem(roi)
            roi.sigRegionChanged.connect(self.update_roi_region)

    def init_hkl(self):
        if "HKL" in self.reader.config:
            self.hkl_data = self.reader.config["HKL"]
            # Get everything for the sample circles
            sample_circle_keys = [key for key in list(self.hkl_data.keys()) if key.startswith('SampleCircle')]
            self.sample_circle_directions = []
            self.sample_cirlce_names = []
            self.sample_circle_positions = []
            for sample_circle in sample_circle_keys:
                self.sample_circle_directions.append(self.hkl_data[sample_circle]['DirectionAxis'])
                self.sample_cirlce_names.append(self.hkl_data[sample_circle]['SpecMotorName'])
                # temporary simulated data until we get the beamline up and running
                self.sample_circle_positions.append(self.hkl_data[sample_circle]['Position'])
            # Get everything for the detector circles
            det_circle_keys = [key for key in list(self.hkl_data.keys()) if key.startswith('DetectorCircle')]
            self.det_circle_directions = []
            self.det_cirlce_names = []
            self.det_circle_positions = []
            for det_circle in det_circle_keys:
                self.det_circle_directions.append(self.hkl_data[det_circle]['DirectionAxis'])
                self.det_cirlce_names.append(self.hkl_data[det_circle]['SpecMotorName'])
                # temporary simulated data until we get the beamline up and running
                self.det_circle_positions.append(self.hkl_data[det_circle]['Position'])
            # Primary Beam Direction
            self.primary_beam_directions= [self.hkl_data['PrimaryBeamDirection'][axis] for axis in self.hkl_data['PrimaryBeamDirection'].keys()]
            # Inplane Reference Direction
            self.inplane_beam_direction = [self.hkl_data['InplaneReferenceDirection'][axis] for axis in self.hkl_data['InplaneReferenceDirection'].keys()]
            # Sample Surface Normal Direction
            self.sample_surface_normal_direction = [self.hkl_data['SampleSurfaceNormalDirection'][axis] for axis in self.hkl_data['SampleSurfaceNormalDirection'].keys()]
            # Class for the conversion of angular coordinates to momentum space 
            self.q_conv = xu.experiment.QConversion(self.sample_circle_directions, 
                                                    self.det_circle_directions, 
                                                    self.primary_beam_directions)
            print(self.det_circle_directions)
            # UB Matrix
            self.ub_matrix = self.hkl_data['UBMatrix']['Value']
            self.ub_matrix  = np.reshape(self.ub_matrix,(3,3))
            # Energy
            self.energy = self.hkl_data['Energy']['Value'] * 1000

    def create_rsm(self):
        hxrd  = xu.HXRD(self.inplane_beam_direction, self.sample_surface_normal_direction, en=self.energy, qconv=self.q_conv)
        if self.stats_data:
            if f"{self.reader.pva_prefix}:Stats4:Total_RBV" in self.stats_data and f"{self.reader.pva_prefix}:Stats4:MaxValue_RBV" in self.stats_data:
                roi = [0, self.reader.shape[0], 0, self.reader.shape[1]]
                pixel_dir1 = self.hkl_data['DetectorSetup']['PixelDirection1']
                pixel_dir2 = self.hkl_data['DetectorSetup']['PixelDirection2']
                cch1 = self.hkl_data['DetectorSetup']['CenterChannelPixel'][0]
                cch2 = self.hkl_data['DetectorSetup']['CenterChannelPixel'][1]
                nch1 = self.reader.shape[0]
                nch2 = self.reader.shape[1]
                pixel_width1 = self.hkl_data['DetectorSetup']['Size'][0] / nch1
                pixel_width2 = self.hkl_data['DetectorSetup']['Size'][1] / nch2
                distance = self.hkl_data['DetectorSetup']['Distance']

                hxrd.Ang2Q.init_area(pixel_dir1, pixel_dir2, cch1=cch1, cch2=cch2,
                                    Nch1=nch1, Nch2=nch2, pwidth1=pixel_width1, pwidth2=pixel_width2,
                                    distance=distance, roi=roi)
                #used temporarily until we have a way to read pvs from detector directly or add it to metadata
                angles = [*self.sample_circle_positions, *self.det_circle_positions]

                return hxrd.Ang2Q.area(*angles, UB=self.ub_matrix)

    def update_rois(self):
        """
        Updates the positions and sizes of ROIs based on changes from the EPICS software.

        Loops through the cached ROIs and adjusts their parameters accordingly.
        """
        for roi, roi_key in zip(self.rois, self.reader.rois.keys()):
            x_pos = self.reader.rois[roi_key].get("MinX",0)
            y_pos = self.reader.rois[roi_key].get("MinY",0)
            width = self.reader.rois[roi_key].get("SizeX",0)
            height = self.reader.rois[roi_key].get("SizeY",0)
            roi.setPos(pos=x_pos, y=y_pos)
            roi.setSize(size=(width, height))
    
    def update_roi_region(self):
        """
        Forces the image viewer to refresh when an ROI region changes.
        """
        self.image_view.update()

    def update_pv_prefix(self):
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()
    
    def update_mouse_pos(self, pos):
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
                self.mouse_x_val.setText(f"{x:.3f}")
                self.mouse_y_val.setText(f"{y:.3f}")
                img_data = self.reader.get_pva_image()
                if img_data is not None:
                    img_data = np.rot90(img_data, k = self.rot_num)
                    if 0 <= x < self.reader.shape[0] and 0 <= y < self.reader.shape[1]:
                       self.mouse_px_val.setText(f'{img_data[int(x)][int(y)]}')
                       if self.hkl_data:
                        self.mouse_h.setText(f'{self.qx[int(x)][int(y)]}')
                        self.mouse_k.setText(f'{self.qy[int(x)][int(y)]}')
                        self.mouse_l.setText(f'{self.qx[int(x)][int(y)]}')

    def update_labels(self):
        """
        Updates the UI labels with current connection and cached data.
        """
        provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
        is_connected = 'Connected' if self.reader.channel.isMonitorActive() else 'Disconnected'
        self.provider_name.setText(provider_name)
        self.is_connected.setText(is_connected)
        self.missed_frames_val.setText(f'{self.reader.frames_missed:d}')
        self.frames_received_val.setText(f'{self.reader.frames_received:d}')
        self.plot_call_id.setText(f'{self.call_id_plot:d}')
        self.size_x_val.setText(f'{self.reader.shape[0]:d}')
        self.size_y_val.setText(f'{self.reader.shape[1]:d}')
        self.data_type_val.setText(self.reader.data_type)
        self.roi1_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats1:Total_RBV', '0.0')}")
        self.roi2_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats2:Total_RBV', '0.0')}")
        self.roi3_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats3:Total_RBV', '0.0')}")
        self.roi4_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats4:Total_RBV', '0.0')}")
        self.stats5_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats5:Total_RBV', '0.0')}")

    def update_rsm(self):
        self.qx, self.qy, self.qz = self.create_rsm()

    def update_image(self):
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            image = self.reader.image
            if image is not None:
                image = np.rot90(image, k = self.rot_num)
                if len(image.shape) == 2:
                    min_level, max_level = np.min(image), np.max(image)
                    height, width = image.shape[:2]
                    # coordinates = pg.QtCore.QRectF(0, 0, width - 1, height - 1)
                    if self.log_image.isChecked():
                            image = np.log(image + 1)
                            min_level = np.log(min_level + 1)
                            max_level = np.log(max_level + 1)
                    if self.first_plot:
                        self.image_view.setImage(image, 
                                                 autoRange=False, 
                                                 autoLevels=False, 
                                                 levels=(min_level, max_level),
                                                 autoHistogramRange=False) 
                        # Auto sets the max value based on first incoming image
                        self.max_setting_val.setValue(max_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(image,
                                                autoRange=False, 
                                                autoLevels=False, 
                                                autoHistogramRange=False)
                # Separate image update for horizontal average plot
                self.horizontal_avg_plot.plot(x=np.mean(image, axis=0), 
                                              y=np.arange(0,self.reader.shape[1]), 
                                              clear=True)

                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    
    def update_min_max_setting(self):
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
        super(ImageWindow,self).closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigDialog()
    window.show()

    sys.exit(app.exec_())