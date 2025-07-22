import toml
import time
import threading
import numpy as np
import pvaccess as pva
import h5py
import bitshuffle
import blosc2
import lz4.block
from collections import deque
from epics import camonitor, caget

class PVAReader:
    def __init__(self, input_channel='s6lambda1:Pva1:Image', provider=pva.PVA, config_filepath: str = 'pv_configs/metadata_pvs.toml', viewer_type='image'):
        """
        Initializes the PVA Reader for monitoring connections and handling image data.

        Args:
            input_channel (str): Input channel for the PVA connection.
            provider (protocol): The protocol for the PVA channel.
            config_filepath (str): File path to the configuration TOML file.
        """
        # Each PVA ScalarType is enumerated in C++ starting 1-10
        # This means we map them as numbers to a numpy datatype which we parse from pva codec parameters
        # Then use this to correctly decompress the image depending on the codec used
        self.NUMPY_DATA_TYPE_MAP = {
            pva.UBYTE   : np.dtype('uint8'),
            pva.BYTE    : np.dtype('int8'),
            pva.USHORT  : np.dtype('uint16'),
            pva.SHORT   : np.dtype('int16'),
            pva.UINT    : np.dtype('uint32'),
            pva.INT     : np.dtype('int32'),
            pva.ULONG   : np.dtype('uint64'),
            pva.LONG    : np.dtype('int64'),
            pva.FLOAT   : np.dtype('float32'),
            pva.DOUBLE  : np.dtype('float64')
        }

        # This also means we can parse the pva codec parameters to show the correct datatype in viewer
        # rather than using default compressed dtype
        self.NTNDA_DATA_TYPE_MAP = {
            pva.UBYTE   : 'ubyteValue',
            pva.BYTE    : 'byteValue',
            pva.USHORT  : 'ushortValue',
            pva.SHORT   : 'shortValue',
            pva.UINT    : 'uintValue',
            pva.INT     : 'intValue',
            pva.ULONG   : 'ulongValue',
            pva.LONG    : 'longValue',
            pva.FLOAT   : 'floatValue',
            pva.DOUBLE  : 'doubleValue',
        }

        self.NTNDA_NUMPY_MAP = {
            'ubyteValue'  : np.dtype('uint8'),
            'byteValue'   : np.dtype('int8'),
            'ushortValue' : np.dtype('uint16'),
            'shortValue'  : np.dtype('int16'),
            'uintValue'   : np.dtype('uint32'),
            'intValue'    : np.dtype('int32'),
            'ulongValue'  : np.dtype('uint64'),
            'longValue'   : np.dtype('int64'),
            'floatValue'  : np.dtype('float32'),
            'doubleValue' : np.dtype('float64')
        }

        self.VIEWER_TYPE_MAP = {
            'image': 'i',
            'analysis': 'a',
            'rsm': 'r' 
        }

        # variables related to monitoring connection
        self.input_channel = input_channel        
        self.provider = provider
        self.channel = pva.Channel(self.input_channel, self.provider)
        self.pva_prefix = input_channel.split(":")[0]

        # variables setup using config
        self.config = {}
        self.rois = {}
        self._roi_names = []
        self.stats = {}
        self.CONSUMER_MODE = ''
        self.OUTPUT_FILE_LOCATION = ''
        self.ROI_IN_CONFIG = False
        self.ANALYSIS_IN_CONFIG = False
        self.HKL_IN_CONFIG = False
        self.CACHE_OPTIONS = {}
        self.CACHING_MODE = ''
        self.MAX_CACHE_SIZE = 0
        self.is_caching = False
        self.is_scan_complete = False

        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.timestamp = None
        self.data_type = None
        self.display_dtype = None
        self.numpy_dtype = None
        self.attributes = []
        self.pv_attributes = {}

        # variables used for image manipulaiton
        self.pixel_ordering = 'F'
        self.viewer_type = self.VIEWER_TYPE_MAP.get(viewer_type, 'i')
        self.image_is_transposed = False
        
        # variables used for parsing analysis data
        self.analysis_index = None
        self.analysis_attributes = {}

        # variables used for parsing hkl data
        # self.rsm_index = None
        self.rsm_attributes = {}

        # variables used for frame count
        self.last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.id_diff = 0

        # variables for data caches
        self.caches_needed = False
        self.caches_initialized = False
        self.cache_attributes = None
        self.cache_images = None
        self.cache_qx = None
        self.cache_qy = None
        self.cache_qz = None
        self._on_scan_complete_callbacks = []

        self._configure(config_filepath)

############################# Configuration #############################
    def _configure(self, config_path: str) -> None:
        if config_path != '':
            with open(config_path, 'r') as toml_file:
                # loads toml config into a python dict
                self.config:dict = toml.load(toml_file)
        
        #TODO: make it so that file location can be parsed as a pv with a function
        # using something like caget or parse the pv attributes
        self.OUTPUT_FILE_LOCATION = self.config.get('OUTPUT_FILE_LOCATION','OUTPUT.h5')  
        self.stats:dict = self.config.get('STATS', {})
        self.ROI_IN_CONFIG = ('ROI' in self.config)
        self.ANALYSIS_IN_CONFIG = ('ANALYSIS' in self.config)
        self.HKL_IN_CONFIG = ('HKL' in self.config)

        if self.config.get('DETECTOR_PREFIX', ''):
            self.pva_prefix = self.config['DETECTOR_PREFIX']

        if self.ROI_IN_CONFIG:
            for roi in self.config['ROI']:
                self._roi_names.append(roi)

        # Configuring Cache settings
        self.CACHE_OPTIONS: dict = self.config.get('CACHE_OPTIONS', {})
        self.set_cache_options()
        if self.caches_needed != self.caches_initialized:
            self.init_caches()
        
        # Configuring Analysis Caches
        if self.ANALYSIS_IN_CONFIG:
            self.CONSUMER_MODE = self.config.get('CONSUMER_MODE', '')
            if self.CONSUMER_MODE == "continuous":
                self.analysis_cache_dict = {"Position": set(),
                                            "Intensity": {},
                                            "ComX": {},
                                            "ComY": {}}
    def set_cache_options(self) -> None:
        self.CACHING_MODE = self.CACHE_OPTIONS.get('CACHING_MODE', '')
        if self.CACHING_MODE != '':
            self.caches_needed = True
            if self.CACHING_MODE == 'alignment':
                self.MAX_CACHE_SIZE = self.CACHE_OPTIONS.setdefault('ALIGNMENT', {'MAX_CACHE_SIZE': 100}).get('MAX_CACHE_SIZE')
            elif self.CACHING_MODE == 'scan':
                self.FLAG_PV = self.CACHE_OPTIONS.setdefault('SCAN', {'FLAG_PV': ''}).get('FLAG_PV')
                self.START_SCAN = self.CACHE_OPTIONS.setdefault('SCAN', {'START_SCAN': True}).get('START_SCAN')
                self.STOP_SCAN = self.CACHE_OPTIONS.setdefault('SCAN', {'STOP_SCAN': False}).get('STOP_SCAN')
                self.MAX_CACHE_SIZE = self.CACHE_OPTIONS.setdefault('SCAN', {'MAX_CACHE_SIZE': 100}).get('MAX_CACHE_SIZE')
            elif self.CACHING_MODE == 'bin':
                self.BIN_COUNT = self.CACHE_OPTIONS.setdefault('BIN', {'COUNT': 10}).get('COUNT')
                self.BIN_SIZE = self.CACHE_OPTIONS.setdefault('BIN', {'SIZE': 16}).get('SIZE')

    def init_caches(self) -> None:
        if self.CACHING_MODE == 'alignment' or self.CACHING_MODE == 'scan':
            self.cache_images = deque(maxlen=self.MAX_CACHE_SIZE)
            self.cache_attributes = deque(maxlen=self.MAX_CACHE_SIZE)
            if self.HKL_IN_CONFIG:
                self.cache_qx = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cache_qy = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cache_qz = deque(maxlen=self.MAX_CACHE_SIZE)
        elif self.CACHING_MODE == 'bin':
            # TODO: when creating the h5 file, have one entry called data that is the average of each bin
            # and then an entry for each bin that lines up with the attributes and rsm attributes
            self.cache_images = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            self.cache_attributes = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            if self.HKL_IN_CONFIG:
                self.cache_qx = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cache_qy = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cache_qz = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]               

        self.caches_initialized = True

#################### Class and PVA Channel Callbacks ########################
    def add_on_scan_complete_callback(self, callback_func):
        if callable(callback_func):
            self._on_scan_complete_callbacks.append(callback_func)
        
    def pva_callbackSuccess(self, pv) -> None:
        """
        Callback for handling monitored PVA changes.

        Args:
            pv (PvObject): The PVA object received by the channel monitor.
        """
        try:
            self.frames_received += 1
            self.pva_object = pv

            # parse data required to manipulate pv image
            self.parse_image_data_type(pv)
            self.shape = self.parse_img_shape(pv)
            self.image = self.pva_to_image(pv)

            # update with latest pv metadata
            self.pv_attributes = self.parse_attributes(pv)
            
            # Check for any roi pvs in metadata
            if self.ROI_IN_CONFIG:
                self.parse_roi_pvs(self.pv_attributes) 

            # Check for rsm attributes in metadata
            if self.HKL_IN_CONFIG and 'RSM' in self.pv_attributes:
                self.parse_rsm_attributes(self.pv_attributes)

            if self.ANALYSIS_IN_CONFIG and 'Analysis' in self.pv_attributes:
                self.parse_analysis_attributes()
            
            if self.caches_initialized:
                self.cache_pv_attributes(self.pv_attributes, self.rsm_attributes)
                self.cache_image(np.ravel(self.image))

            #TODO: depreciated change the parsing to be closer to parsing RSM attributes with the new parse_attributes function
            if self.ANALYSIS_IN_CONFIG:
                self.analysis_index = self.locate_analysis_index()
                # Only runs if an analysis index was found
                if self.analysis_index is not None:
                    self.analysis_attributes = self.attributes[self.analysis_index]
                    if self.config["CONSUMER_MODE"] == "continuous":
                        # turns axis1 and axis2 into a tuple
                        incoming_coord = (self.analysis_attributes["value"][0]["value"].get("Axis1", 0.0), 
                                        self.analysis_attributes["value"][0]["value"].get("Axis2", 0.0))
                        # use a tuple as a key so that we can check if there is a repeat position
                        self.analysis_cache_dict["Intensity"].update({incoming_coord: self.analysis_cache_dict["Intensity"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("Intensity", 0.0)})
                        self.analysis_cache_dict["ComX"].update({incoming_coord: self.analysis_cache_dict["ComX"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComX", 0.0)})
                        self.analysis_cache_dict["ComY"].update({incoming_coord: self.analysis_cache_dict["ComY"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComY", 0.0)})
                        # double storing of the postion, will find out if needed
                        self.analysis_cache_dict["Position"][incoming_coord] = incoming_coord

            if self.is_scan_complete == True:
                print('Scan Complete')
                for callable in self._on_scan_complete_callbacks:
                    threading.Thread(target=callable,).start()
                    
        except Exception as e:
            print(f'Failed to execute callback: {e}')
            self.frames_received -= 1
            self.frames_missed += 1

    def roi_backup_callback(self, pvname, value, **kwargs) -> None:
        name_components = pvname.split(":")
        roi_key = name_components[1]
        pv_key = name_components[2]
        pv_value = value
        # can't append simply by using 2 keys in a row (self.rois[roi_key][pv_key]), there must be an inner dict to call
        # then adds the key to the inner dictionary with update
        self.rois.setdefault(roi_key, {}).update({pv_key: pv_value})
        
########################### PVA PARSING ##################################
    def locate_analysis_index(self) -> int|None:
        """
        Locates the index of the analysis attribute in the PVA attributes.

        Returns:
            int: The index of the analysis attribute or None if not found.
        """
        if self.pv_attributes:
            for i, attr_name in enumerate(self.pv_attributes.keys()): 
                if attr_name == "Analysis":
                    return i
            else:
                return None

    def parse_image_data_type(self, pva_object) -> None:
        """
        Parses the PVA Object to determine the incoming data type.
        """
        if pva_object is not None:
            try:
                self.data_type = list(pva_object['value'][0].keys())[0]
                self.display_dtype = self.data_type if pva_object['codec']['name'] == '' else self.NTNDA_DATA_TYPE_MAP.get(pva_object['codec']['parameters'][0]['value'])
                self.numpy_dtype = self.NTNDA_NUMPY_MAP.get(self.display_dtype, None)
            except:
                self.display_dtype = "could not detect"

    def parse_img_shape(self, pva_object) -> tuple:
        if 'dimension' in pva_object:
            return tuple([dim['size'] for dim in pva_object['dimension']])

    def parse_attributes(self, pva_object) -> dict:
        pv_attributes = {}
        if pva_object != None and 'attribute' in pva_object:
            pv_attributes['timeStamp-secondsPastEpoch'] = pva_object['timeStamp']['secondsPastEpoch']
            pv_attributes['timeStamp-nanoseconds'] = pva_object['timeStamp']['nanoseconds']
            attributes = pva_object['attribute']
            for attr in attributes:
                name = attr['name']
                value = attr['value'][0].get('value', None)
                if value is not None:
                    pv_attributes[name] = value
            return pv_attributes
        else:
            return {}

    def parse_analysis_attributes(self, pv_attributes: dict) -> None:
        pass    
    
    def parse_rsm_attributes(self, pv_attributes: dict) -> None:
        rsm_attributes: dict = pv_attributes['RSM']
        codec = rsm_attributes['codec'].get('name', '')
        if  codec !=  '':
            dtype = self.NUMPY_DATA_TYPE_MAP.get(rsm_attributes['codec']['parameters'])
            self.rsm_attributes = {'qx' : self.decompress_array(compressed_array=rsm_attributes['qx']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qx']['uncompressedSize'],
                                                                dtype=dtype),
                                   'qy' : self.decompress_array(compressed_array=rsm_attributes['qy']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qy']['uncompressedSize'],
                                                                dtype=dtype),
                                   'qz' : self.decompress_array(compressed_array=rsm_attributes['qz']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qz']['uncompressedSize'],
                                                                dtype=dtype)}
        else:
            self.rsm_attributes = {'qx' : rsm_attributes['qx']['value'], 
                                   'qy' : rsm_attributes['qy']['value'],
                                   'qz' : rsm_attributes['qz']['value']}
                          
    def parse_roi_pvs(self, pv_attributes: dict) -> None:
        """
        Parses attributes to extract ROI-specific PV information.
        """
        for roi in self._roi_names:
            for dimension in ['MinX', 'MinY', 'SizeX', 'SizeY']:
                pv_key = f'{self.pva_prefix}:{roi}:{dimension}'
                pv_value = pv_attributes.get(pv_key, None)
                # If a dictionary is empty, can't create an inner dictionary by using another [] at the end, there must be a dict to call to.
                # To make sure there is an inner dictionary, we use .setdefault() to initialize the inner dictionary
                if pv_value is not None:
                    self.rois.setdefault(roi, {}).update({dimension: pv_value})
            
    def pva_to_image(self, pva_object) -> np.ndarray:
        """
        Converts the PVA Object to an image array and determines if a frame was missed.
        Handles bslz4 and lz4 compressed image data.

        image is of type np.ndarray
        """
        try:
            if 'dimension' in pva_object:
                if pva_object['codec']['name'] != '':
                    image: np.ndarray = self.decompress_array(compressed_array=pva_object['value'][0][self.data_type],
                                                  codec=pva_object['codec']['name'],
                                                  uncompressed_size=pva_object['uncompressedSize'],
                                                  dtype=self.NUMPY_DATA_TYPE_MAP.get(pva_object['codec']['parameters'][0]['value']))
                else:
                    # Handle uncompressed data  
                    image: np.ndarray = pva_object['value'][0][self.data_type]

                # Check for missed frame starts here
                # TODO: can be it's own function
                current_array_id = pva_object['uniqueId']
                if self.last_array_id is not None: 
                    self.id_diff = current_array_id - self.last_array_id - 1
                    if (self.id_diff > 0):
                        self.frames_missed += self.id_diff 
                        # if self.HKL_IN_CONFIG:
                            # for i in range(self.id_diff):
                            #     if self.HKL_IN_CONFIG:
                            #         self.cache_images.append(self.empty_array)
                            #         self.cache_qx.append(self.empty_array)
                            #         self.cache_qy.append(self.empty_array)
                            #         self.cache_qz.append(self.empty_array)
                self.last_array_id = current_array_id
                self.id_diff = 0
                
                return image.reshape(self.shape, order=self.pixel_ordering).T if self.image_is_transposed else image.reshape(self.shape, order=self.pixel_ordering)
            else:
                self.image = None
                raise ValueError("Image data could not be processed.")
                
        except Exception as e:
            print(f"Failed to process image: {e}")
            self.frames_received -= 1
            self.frames_missed += 1
            
    def decompress_array(self, compressed_array: np.ndarray, codec: str, uncompressed_size, dtype: np.dtype) -> np.ndarray: 
        # Handle BSLZ4 compressed data
        if codec == 'bslz4':
            # uncompressed size has to be divided by the number of bytes needed to store the desired output dtype
            uncompressed_shape = (uncompressed_size // dtype.itemsize,)
            # Decompress numpy array to correct datatype
            return bitshuffle.decompress_lz4(compressed_array, uncompressed_shape, dtype)
        # Handle LZ4 compressed data
        elif codec == 'lz4':
            decompressed_bytes = lz4.block.decompress(compressed_array, uncompressed_size=uncompressed_size)
            # Convert bytes to numpy array with correct dtype
            return np.frombuffer(decompressed_bytes, dtype=dtype) # dtype makes sure we use the correct
        # handle BLOSC compressed data 
        elif codec == 'blosc':
            decompressed_bytes = blosc2.decompress(compressed_array)
            return np.frombuffer(decompressed_bytes, dtype=dtype)

################################## Caching ####################################
    def cache_pv_attributes(self, pv_attributes=None, rsm_attributes=None, analysis_attributes=None) -> None:
        if self.CACHING_MODE == 'alignment': 
            self.cache_attributes.append(pv_attributes)
            if rsm_attributes:
                self.cache_qx.append(rsm_attributes['qx'])
                self.cache_qy.append(rsm_attributes['qy'])
                self.cache_qz.append(rsm_attributes['qz'])
            elif not self.rsm_attributes and self.viewer_type == 'r':
                raise AttributeError('[PVA Reader] Could not find \'RSM\' attribute')
            return
        elif self.CACHING_MODE == 'scan':
            # TODO: create different scan functions for if a flag pv is bool/binary vs not
            # currently only works with binary/boolean flag pvs
            if self.FLAG_PV in pv_attributes:
                flag_value = pv_attributes[self.FLAG_PV]
                if flag_value == self.START_SCAN:
                    self.is_caching = True
                    self.is_scan_complete = False
                elif (flag_value == self.STOP_SCAN) and self.is_caching == True:
                    self.is_caching = False
                    self.is_scan_complete = True
                    
                if self.is_caching:
                    self.cache_attributes.append(pv_attributes)
                    if rsm_attributes:
                        self.cache_qx.append(rsm_attributes['qx'])
                        self.cache_qy.append(rsm_attributes['qy'])
                        self.cache_qz.append(rsm_attributes['qz'])
                    elif not rsm_attributes and self.viewer_type == 'r':
                        raise AttributeError('[PVA Reader] Could not find \'RSM\' attribute')
            else:
                raise AttributeError('[PVA Reader] Flag_PV not found') 
        elif self.CACHING_MODE == 'bin':
            bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
            self.cache_attributes[bin_index].append(pv_attributes) 

    def cache_image(self, image) -> None:
        if self.CACHING_MODE == 'alignment':
            self.cache_images.append(image)
            return
        elif self.CACHING_MODE == 'scan':
            if self.is_caching:
                self.cache_images.append(image)
                return
        elif self.CACHING_MODE == 'bin':
            if self.viewer_type == 'i':
                bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
                self.cache_images[bin_index].append(image)
                return     

########################### Start and Stop Channel Monitors ##########################    
    def start_channel_monitor(self) -> None:
        """
        Subscribes to the PVA channel with a callback function and starts monitoring for PV changes.
        """
        self.channel.subscribe('pva callback success', self.pva_callbackSuccess)
        self.channel.startMonitor()

    def stop_channel_monitor(self) -> None:
        """
        Stops all monitoring and callback functions.
        """
        self.channel.unsubscribe('pva callback success')
        self.channel.stopMonitor()

    def start_roi_backup_monitor(self) -> None:
        try:
            for roi_num, roi_dict in self.config['ROI'].items():
                for config_key, pv_name in roi_dict.items():
                    name_components = pv_name.split(":")

                    roi_key = name_components[1] # ROI1-ROI4
                    pv_key = name_components[2] # MinX, MinY, SizeX, SizeY

                    self.rois.setdefault(roi_key, {}).update({pv_key: caget(pv_name)})
                    camonitor(pvname=pv_name, callback=self.roi_backup_callback)
        except Exception as e:
            print(f'Failed to setup backup ROI monitor: {e}')

    def save_caches_to_h5(self) -> None:
        # TODO: add analysis
        """
        Saves available caches (images and HKL data) to an HDF5 file under a branch structure.
        The file structure is as follows:
            /entry/data         --> The image cache array
            /entry/rois/ROI1-4
            /entry/metadata/motor_positions
            /entry/analysis/intensity
            /entry/analysis/comx
            /entry/analysis/comy            
            /entry/HKL/qx        --> The qx cache array (if available)
            /entry/HKL/qy        --> The qy cache array (if available)
            /entry/HKL/qz        --> The qz cache array (if available)

        Args:
            filename (str): The output HDF5 file name.
        """
        try:
            if self.cache_images is None:
                raise ValueError("Caches cannot be empty.")
            n = len(self.cache_images)
            if not (len(self.cache_attributes) == n) or n == 0:
                raise ValueError("All caches must have the same number of elements.")
            print('attempting save')
            cache_metadata = self.cache_attributes
            merged_metadata = {}
            print('merging attributes')
            for attribute_dict in cache_metadata:
                for key, value in attribute_dict.items():
                    if key != 'RSM' and key != 'Analysis':
                        if key not in merged_metadata:
                            merged_metadata[key] = []
                            merged_metadata[key].append(value)
                        else:
                            merged_metadata[key].append(value)
            print('merging complete')
            
            with h5py.File(self.OUTPUT_FILE_LOCATION, 'w') as h5f:
                # Create the main "images" group
                print(f'creating file at: {self.OUTPUT_FILE_LOCATION}')
                images_grp = h5f.create_group("entry")
                data_grp = images_grp.create_group('data')
                data_grp.create_dataset("data", data=np.array([np.reshape(img,self.shape) for img in self.cache_images], dtype=self.numpy_dtype))
                print('images written')
                metadata_grp = data_grp.create_group("metadata")
                motor_pos_grp = metadata_grp.create_group('motor_positions')
                rois_grp = data_grp.create_group('rois')
                print('metadata, rois, and motorposistion groups created')
                for key, values in merged_metadata.items():
                    if all(isinstance(v, (int, float, np.number)) for v in values):
                        if 'ROI' in key:
                            parts = key.split(':')
                            roi = parts[1]
                            if roi not in rois_grp.keys():
                                rois_grp.create_group(name=roi)
                            rois_grp[roi].create_dataset(key, data=np.array(values))
                        elif 'Position' in key:
                            motor_pos_grp.create_dataset(key, data=np.array(values))
                        else:
                            metadata_grp.create_dataset(key, data=np.array(values))
                    elif all(isinstance(v, str) for v in values):
                        dt = h5py.string_dtype(encoding='utf-8')
                        metadata_grp.create_dataset(key, data=np.array(values, dtype=dt))
                print('metadata saved')

                # Create HKL subgroup under images if HKL caches exist
                if self.HKL_IN_CONFIG and self.caches_initialized:
                    if 'RSM' in self.pv_attributes:
                        if not (len(self.cache_qx) == len(self.cache_qy) == len(self.cache_qz) == n):
                            raise ValueError("qx, qy, and qz caches must have the same number of elements.")
                        hkl_grp = data_grp.create_group(name="hkl")
                        hkl_grp.create_dataset("qx", data=np.array([np.reshape(qx,self.shape) for qx in self.cache_qx]), dtype=np.float32)
                        hkl_grp.create_dataset("qy", data=np.array([np.reshape(qy,self.shape) for qy in self.cache_qy]), dtype=np.float32)
                        hkl_grp.create_dataset("qz", data=np.array([np.reshape(qz,self.shape) for qz in self.cache_qz]), dtype=np.float32)
                        print('qx, qy, qz written')
    
            print(f"Caches successfully saved in a branch structure to {self.OUTPUT_FILE_LOCATION}")
            #reset caches
            self.init_caches()
        except Exception as e:
            print(f"Failed to save caches to {self.OUTPUT_FILE_LOCATION}: {e}")

    def get_frames_missed(self) -> int:
        """
        Returns the number of frames missed.

        Returns:
            int: The number of missed frames.
        """
        return self.frames_missed

    def get_pva_image(self) -> np.ndarray:
        """
        Returns the current PVA image.

        Returns:
            numpy.ndarray: The current image array.
        """
        return self.image
    
    def get_attributes_dict(self) -> list[dict]:
        """
        Returns the attributes of the current PVA object.

        Returns:
            list: The attributes of the current PVA object.
        """
        return self.attributes
