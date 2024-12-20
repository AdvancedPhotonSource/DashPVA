import sys
import json
import copy
import time
import numpy as np
import os.path as osp
import pvaccess as pva
import pyqtgraph as pg
from PyQt5 import uic
from epics import caget
from epics import camonitor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QSizePolicy, QLabel, QFormLayout, QWidget, QFrame
# Custom imported classes
from generators import rotation_cycle
from roi_stats_dialog import RoiStatsDialog
from pv_setup_dialog import PVSetupDialog
from analysis_window import AnalysisWindow #, analysis_window_process
from scan_plan_dialog import ScanPlanDialog


max_cache_size = 900 #TODO: ask the user this important information before opening the analysis window. Ask for a scan plan json file
rot_gen = rotation_cycle(1,5)         
                
# timer_for_roi_change = time.time()
class ConfigDialog(QDialog):

    def __init__(self, prefix='', c_address='', cache_freq=10):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Keyword Args:
        prefix (str) -- used to populate the prefix for all pvas in PVAReader
        c_address (str) -- used to populate the address for the collector channel in PVAReader
        cache_freq (int) -- used to set the update frequency of ImageViewer
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/pv_config.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.prefix = prefix
        self.collector_address = c_address
        self.pvs_path =  ''
        self.cache_frequency = cache_freq
        # class can be prefilled with text
        self.init_ui()

        self.btn_load.clicked.connect(self.json_open_file_dialog)
        self.btn_edit.clicked.connect(self.json_open_file_dialog)
        self.btn_new_config.clicked.connect(self.new_pv_setup)
        self.btn_edit_accept_reject.accepted.connect(self.edit_pv_setup)
        self.btn_setup_accept_reject.accepted.connect(self.dialog_accepted) 

    
    def init_ui(self):
        """
        function called which prefills text in the Line Editors
        """
        self.le_pv_prefix.setText(self.prefix)
        self.le_collector.setText(self.collector_address)
        self.sb_collector_frequency.setValue(self.cache_frequency)


    def json_open_file_dialog(self):
        """
        Function called when you want to get the file path to a json file.
        is split between 2 buttons:
        - file path for the config you wan't to load and monitor
        - file path for a config you want to edit
        """
        btn_sender = self.sender()
        sender_name = btn_sender.objectName()
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select PV Json', 'pv_configs', '*.json (*.json')

        if sender_name.endswith('load'):
            self.le_load_file_path.setText(self.pvs_path)
        else:
            self.le_edit_file_path.setText(self.pvs_path)

    def new_pv_setup(self):
        """
        Pops up a new window for setting up a new config within the ui
        """
        self.new_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='w', path=None)
    
    def edit_pv_setup(self):
        """
        Pops up a new window for editting an already existing config
        """
        if self.le_edit_file_path.text() != '':
            self.edit_pv_setup_dialog = PVSetupDialog(parent=self, file_mode='r+', path=self.pvs_path)
        else:
            print('file path empty')

    def dialog_accepted(self):
        """
        Function called when the last Dialog Accept button is pressed.
        Starts the ImageWindow process passing all filled out information to it and initializing it.
        """
        self.prefix = self.le_pv_prefix.text()
        self.collector_address = self.le_collector.text()
        self.pvs_path = self.le_load_file_path.text()
        self.cache_frequency = self.sb_collector_frequency.value()
        if osp.isfile(self.pvs_path) ^ (self.pvs_path == ''):
            if self.cache_frequency > 0 and self.collector_address != '':
                self.image_viewer = ImageWindow(prefix=self.prefix,
                                                collector_address=self.collector_address,
                                                file_path=self.pvs_path,
                                                cache_frequency=self.cache_frequency) 
            else:
                self.image_viewer = ImageWindow(prefix=self.prefix,
                                                collector_address=self.collector_address,
                                                file_path=self.pvs_path,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog(prefix=self.prefix, c_address=self.collector_address, cache_freq=self.cache_frequency)
            self.new_dialog.show()    


class PVA_Reader:

    def __init__(self, pva_prefix='dp-ADSim', provider=pva.PVA, collector_address='', config_filepath: str = 'pv_configs/PVs.json'):
        """
        Variables needed for monitoring a connection.
        Provides connections and to broadcasted images and PVAs
        
        KeyWord Args:
        pva_prefix (str) -- The prefix of the specific detector that will be appended to the PVA channel name (default dp-ADSim)
        provider (protocol) -- The protocol that will be used when creating the channel (default pva.PVA)
        collector_address (str) -- address to the collector server
        config_filepath (str) -- file path to where the config json file is located
        """
        self.pva_prefix = pva_prefix        
        self.provider = provider
        self.collector_address = collector_address
        self.config_filepath = config_filepath
        self.pva_address = self.pva_prefix + ':Pva1:Image' if self.collector_address == "" else self.collector_address
        self.channel = pva.Channel(self.pva_address, self.provider)
        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.attributes = {}
        self.timestamp = None
        # self.pva_cache = []
        self.images_cache = None
        self.positions_cache = None
        self.__last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.data_type = None
        self.pvs = {}
        self.metadata = {}
        self.num_rois = 0
        self.cache_id = None
        self.cache_id_gen = rotation_cycle(0,max_cache_size)
        self.first_scan_detected = False

        if self.config_filepath != '':
            with open(self.config_filepath, 'r') as json_file:
                # loads the pvs in the json file into a python dictionary
                self.pvs = json.load(json_file)
        
    def pva_callbackSuccess(self, pv):
        """
        Function is called every time a PV change is monitored.
        Makes sure we only keep queue of 1000 PV objects in memory 
        and before caching then processes incoming pv. 
        
        KeyWord Args:
        pv (PVObject) -- Received by channel Monitor
        """
        self.pva_object = pv
        # Create a deep copy of pv before appending
        # pv_copy = copy.deepcopy(pv)
        # if len(self.pva_cache) < 100: 
        #     self.pva_cache.append(pv_copy)
        # else:
        #     self.pva_cache = self.pva_cache[1:]
        #     self.pva_cache.append(pv_copy)
        self.parse_pva_ndattributes()
        self.parse_image_data_type()
        self.pva_to_image()
        x_value = self.attributes.get('x')[0]['value']
        y_value = self.attributes.get('y')[0]['value']
        #need to avoid hard coding the initial scan position 
        if (x_value == 0) and (y_value == 0) and self.first_scan_detected == False:
            self.first_scan_detected = True
            print(f"First Scan detected...")
  
        #now start caching after the first scan is detected 
        # now scan pos recorded should match the prerecorded scan positions 
        # TODO: make sure the scan positions read from the numpy file is the same as those read through the collector 
        # and overwrite if not
        if self.first_scan_detected:
            #TODO: Debug why starting cache causes missed frames 
            if self.id_diff> 0:
                for i in range(self.id_diff):
                    self.cache_id = next(self.cache_id_gen)
                self.images_cache[self.cache_id-self.id_diff+1:self.cache_id+1,:,:] = 0
                self.positions_cache[self.cache_id-self.id_diff+1:self.cache_id+1,0] = np.NaN #TODO: wil be overwritten by predetermined scan positions in analysis window if called
                self.positions_cache[self.cache_id-self.id_diff+1:self.cache_id+1,1] = np.NaN #TODO: wil be overwritten by predetermined scan positions in analysis window if called
            else:
                self.cache_id = next(self.cache_id_gen)
                self.images_cache[self.cache_id,:,:] = copy.deepcopy(self.image)
                self.positions_cache[self.cache_id,0] = copy.deepcopy(x_value) #TODO: generalize for whatever scan positions we get
                self.positions_cache[self.cache_id,1] = copy.deepcopy(y_value) #TODO: generalize for whatever scan positions we get
            # print(self.cache_id)
            # print(self.positions_cache[self.cache_id])
            
            # TODO: Use thise actual positions to do calculations in analysis window

    def parse_image_data_type(self):
        """Parse through a PVA Object to store the incoming datatype."""
        if self.pva_object is not None:
            self.data_type = list(self.pva_object['value'][0].keys())[0]
    
    def parse_pva_ndattributes(self):
        """Convert a pva object to python dict and parses attributes into a separate dict."""
        if self.pva_object is not None:
            obj_dict = self.pva_object.get()
        else:
            return None
        
        attributes = {}
        for attr in obj_dict.get("attribute", []):
            name = attr['name']
            value = attr['value']
            attributes[name] = value

        for value in ["codec", "uniqueId", "uncompressedSize"]:
            if value in self.pva_object:
                attributes[value] = self.pva_object[value]

        self.attributes = attributes

    def pva_to_image(self):
        """
        Parses through the PVA Object to retrieve the size and use that to shape incoming image.
        Then immedately check if that PVA Object is next image or if we missed a frame in between.
        """
        try:
            if self.pva_object is not None:
                self.frames_received += 1
                # Parses dimensions and reshapes array into image
                if 'dimension' in self.pva_object:
                    self.shape = tuple([dim['size'] for dim in self.pva_object['dimension']])
                    self.image = np.array(self.pva_object['value'][0][self.data_type])
                    # reshapes but also transposes image so it is viewed correctly
                    self.image= np.reshape(self.image, self.shape).T
                else:
                    self.image = None
                # Initialize Image and Positions Cache
                if self.images_cache is None:
                    self.images_cache = np.zeros((max_cache_size, *self.shape))
                    self.positions_cache = np.zeros((max_cache_size,2)) # TODO: make useable for more metadata
                # Check for missed frame starts here
                current_array_id = self.pva_object['uniqueId']
                if self.__last_array_id is not None: 
                    self.id_diff = current_array_id - self.__last_array_id - 1
                    if (self.id_diff > 0):
                        self.frames_missed += self.id_diff 
                    else:
                        self.id_diff = 0
                self.__last_array_id = current_array_id
                # Check for missed frame starts here
                # current_array_id = self.pva_object['uniqueId']
                # if self.__last_array_id is not None: 
                #     id_diff = current_array_id - self.__last_array_id - 1
                #     self.frames_missed += id_diff if (id_diff > 0) else 0
                # self.__last_array_id = current_array_id

        except:
            self.frames_missed += 1
            # return 1
            
    def start_channel_monitor(self):
        """
        Calls the PVA subscribe function of the pvaccess module to 
        provide a callback function to process any incoming PV Objects. 

        After that it starts the channel Monitor and it goes through each 
        item stored in the metadata PVs dict to retrieve ROI information 
        using epics caget as monitoring them is not consistent at the start.
        (this can be changed with the collector running)
        """
        self.channel.subscribe('pva callback success', self.pva_callbackSuccess)
        self.channel.startMonitor()
        if self.pvs and self.pvs is not None:
            try:
                for key in self.pvs:
                    if f'{self.pvs[key]}'.startswith('ROI'):
                        metadata_name = f'{self.pva_prefix}:{self.pvs[key]}'
                        self.metadata[metadata_name] = caget(metadata_name)
                        if not(f'{self.pvs[key]}'.startswith(f'ROI{self.num_rois}')):
                            self.num_rois += 1                        
            except:
                print('Failed to connect to PV ROIs/Stats')
        
    def stop_channel_monitor(self):
        """Stops all monitorg and callback functions from continuing"""
        self.channel.unsubscribe('pva callback success')
        self.channel.stopMonitor()

    def get_pva_objects(self):
        """Returns entire cached list of PVA Objects"""
        return self.pva_cache

    def get_last_pva_object(self):
        return self.pva_cache[-1]

    def get_frames_missed(self):
        return self.frames_missed

    def get_pva_image(self):
        return self.image
    
    def get_attributes_dict(self):
        return self.attributes


class ImageWindow(QMainWindow):

    def __init__(self, prefix='dp-ADSim', collector_address='', file_path='', cache_frequency=100): 
        """
        This is the Main Window that first pops up and allows a user to type 
        a detector prefix in and connect to it. It does things like allow one 
        to view and manipulate the incoming image, change the color scheme, and 
        view ROIs. In addtion to this it can also show multiple stats about the 
        connection, the images shown, and specific ROIs.
        """
        super(ImageWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle('Image Viewer with PVAaccess')
        self.show()
        # Initializing important variables
        self.reader = None
        self.call_id_plot = 0
        self.first_plot = True
        self.caching_frequency = cache_frequency #TODO take this value from a texbox .. user has to put in the frequency of the detector or collector manually or it has to be acertained it matches with the collector
        self.rot_num = 0
        self.rois = []
        self.stats_dialog = {}
        self.stats_data = {}
        self._prefix = prefix
        self.pv_prefix.setText(self._prefix)
        self._collector_address = collector_address
        self._file_path = file_path
        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)
        #for testing ROIs being sent from analysis window
        self.roi_x = 100
        self.roi_y = 200
        self.roi_width = 50
        self.roi_height = 50
        # Adding widgets manually to have better control over them
        # First is a Image View with a plot to view incoming images with axes shown
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
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.btn_analysis_window.clicked.connect(self.open_analysis_window_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        self.log_image.clicked.connect(self.update_image)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.btn_Stats5.clicked.connect(self.stats_button_clicked)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.display_rois.stateChanged.connect(self.show_rois_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    def open_analysis_window_clicked(self):
        scd_dialog = ScanPlanDialog()
        if scd_dialog.exec_():
            self.analysis_window = AnalysisWindow(parent=self, xpos_path=scd_dialog.x_positions, ypos_path=scd_dialog.y_positions,save_path=scd_dialog.download_loc)
            self.analysis_window.show()
        
    def start_timers(self):
        """Timer speeds for updating labels and plotting"""
        self.timer_labels.start(int(1000/float(self.caching_frequency)))
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))

    def stop_timers(self):
        """Stops the updating of Main Window Labels"""
        self.timer_plot.stop()
        self.timer_labels.stop()

    def start_live_view_clicked(self):
        """
        Goes through and tries to initialize the connections to the PVA channel using
        the prefix which was typed in. 
        """
        try:
            prefix = self.pv_prefix.text()
            # a double check to make sure there isn't a connection already when starting
            if self.reader is None:
                self.reader = PVA_Reader(pva_prefix=prefix, collector_address=self._collector_address, config_filepath=self._file_path)
                self.reader.start_channel_monitor()
                if self.reader.channel.get():
                    self.start_timers()
            else:
                self.stop_timers()
                self.reader.stop_channel_monitor()
                del self.reader
                self.reader = PVA_Reader(pva_prefix=self._prefix, collector_address=self._collector_address, config_filepath=self._file_path)
                self.reader.start_channel_monitor()
                if self.reader.channel.get():
                    self.start_timers()
            
            # additional functions that aren't affected by whether the PVA reader is None or not
            self.start_stats_monitors()
            self.start_roi_monitors()
            self.add_rois()
        except:
            print('Failed to Connect')
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
    def stop_live_view_clicked(self):
        """Clears the connection for the PVA channel and any active monitors."""
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            for key in self.stats_dialog:
                self.stats_dialog[key] = None
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def start_roi_monitors(self):
        """Monitors used for changing the shape/location of ROIS Live."""
        try:
            if self.reader.pvs and self.reader.pvs is not None:
                for key in self.reader.pvs:
                    if f'{self.reader.pvs[key]}'.startswith('ROI'):
                            camonitor(pvname=f'{self.reader.pva_prefix}:{self.reader.pvs[key]}',
                                      callback=self.roi_ca_callback)
        except:
            print('Failed to Connect to ROI CA Monitors')

    def start_stats_monitors(self):
        """Monitors used to update Stats values."""
        try:
            if self.reader.pvs and self.reader.pvs is not None:
                for key in self.reader.pvs:
                    if f'{self.reader.pvs[key]}'.startswith('Stats'):
                            camonitor(pvname=f'{self.reader.pva_prefix}:{self.reader.pvs[key]}', 
                                      callback=self.stats_ca_callback)
        except:
            print("Failed to Connect to Stats CA Monitors")
    
    def roi_ca_callback(self, pvname, value, **kwargs):
        """
        Manipulates ROIs live whenever a change is made in the EPICS software
        then loops through the list of cached ROIs and updates their position/size.
        
        KeyWord Args:
        pvname -- The name of the specific ROI PV that's been updated
        value -- The new value sent by the monitor for the PV
        **kwargs -- a catch all for the other values sent by the monitor
        """
        self.reader.metadata[pvname] = value
        prefix = self.reader.pva_prefix
        for i, roi in enumerate(self.rois):
            roi.setPos(pos=self.reader.metadata[f'{prefix}:ROI{i+1}:MinX'], 
                       y=self.reader.metadata[f'{prefix}:ROI{i+1}:MinY'])
            roi.setSize(size=(self.reader.metadata[f'{prefix}:ROI{i+1}:SizeX'], 
                              self.reader.metadata[f'{prefix}:ROI{i+1}:SizeY']))

    def stats_ca_callback(self, pvname, value, **kwargs):
        """
        Updates the Stats PV Value.
        
        KeyWord Args:
        pvname -- The name of the specific Stat PV that's been updated.
        value -- The new value sent by the monitor for the PV.
        **kwargs -- a catch all for the other values sent by the monitor."""
        self.stats_data[pvname] = value
        
    def stats_button_clicked(self):
        """Creates a pop up dialog specifically for the stat that you want to view."""
        if self.reader is not None:
            sending_button = self.sender()
            text = sending_button.text()
            # TODO: Pass it a timer so it doesn't create a new one every time
            self.stats_dialog[text] = RoiStatsDialog(parent=self, stats_text=text, timer=self.timer_labels)
            self.stats_dialog[text].show()
    
    def show_rois_checked(self):
        """Shows/Hides ROIs depending on checked state."""
        if self.reader is not None:
            if self.display_rois.isChecked():
                for roi in self.rois:
                    roi.show()
            else:
                for roi in self.rois:
                    roi.hide()

    def freeze_image_checked(self):
        """
        Freezes/Unfreezes plot depending on checked state
        without stopping collection of PVA Objects.
        """
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.timer_labels.stop()
                self.timer_plot.stop()
            else:
                self.start_timers()

    def reset_first_plot(self):
        self.first_plot = True

    def rotation_count(self):
        """Used to cycle image rotation number between 1 - 4."""
        self.rot_num = next(rot_gen)

    def add_rois(self):
        """
        Takes the number of ROIs detected earlier from pvs dict and 
        adds them to the image viewer and color codes them.

        Color Codes:
        ROI1 -- Red (ff0000)
        ROI2 -- Blue (0000ff)
        ROI3 -- Green (4CBB17)
        ROI4 -- Pink (ff00ff)
        """
        self.roi_colors = ['ff0000', '0000ff', '4CBB17', 'ff00ff']
        for i in range(self.reader.num_rois):
            roi = pg.ROI(pos=[self.reader.metadata[f'{self.reader.pva_prefix}:ROI{i+1}:MinX'],
                               self.reader.metadata[f'{self.reader.pva_prefix}:ROI{i+1}:MinY']],
                         size=[self.reader.metadata[f'{self.reader.pva_prefix}:ROI{i+1}:SizeX'], 
                              self.reader.metadata[f'{self.reader.pva_prefix}:ROI{i+1}:SizeY']],
                         movable=False,
                         pen=pg.mkPen(self.roi_colors[i]))
            self.rois.append(roi)
            self.image_view.addItem(roi)
    
    def update_mouse_pos(self, pos):
        """
        Receives mouse position signal inside the ImageViewer and maps it 
        to a QPointer on the image to receive pixel value where the mouse is.

        KeyWord Args:
        pos -- position event sent by mouse moving
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

    def update_labels(self):
        """Updates labels based on connection and cached data"""
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

    def update_image(self):
        """
        Redraws plots based on rate entered in main window.
        Processes the images based on the different settings in the main window.
        And shows the min/max pixel value within the entire image
        """
        if self.reader is not None:
            self.call_id_plot +=1
            image = self.reader.image
            if image is not None:
                # print(self.reader.cache_id)
                # print(self.reader.positions_cache[self.reader.cache_id])
                image = np.rot90(image, k = self.rot_num)
                if len(image.shape) == 2:
                    min_level, max_level = np.min(image), np.max(image)
                    height, width = image.shape[:2]
                    # print(image.shape[:2])
                    coordinates = pg.QtCore.QRectF(0, 0, width - 1, height - 1)
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
        """Updates the levels for the pixel values you want to see in the ImageViewer"""
        min = float(self.min_setting_val.text())
        max = float(self.max_setting_val.text())
        self.image_view.setLevels(min, max)
    
    def closeEvent(self, event):
        """
        Altered close event to delete stat dialogs as well when main window closes
        
        Keyword Args:
        event -- close event sent by main window
        """
        del self.stats_dialog # otherwise memory piles up
        super(ImageWindow,self).closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConfigDialog(prefix='dp-ADSim', c_address='collector:1:output')
    window.show()

    sys.exit(app.exec_())