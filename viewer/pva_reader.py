import toml
import numpy as np
import pvaccess as pva
import h5py
import bitshuffle
import blosc2
import lz4.block
from collections import deque
from epics import camonitor, caget

class PVAReader:
    def __init__(self, input_channel='s6lambda1:Pva1:Image', provider=pva.PVA, config_filepath: str = 'pv_configs/metadata_pvs.toml'):
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
        self. NTNDA_DATA_TYPE_MAP = {
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

        self.input_channel = input_channel        
        self.provider = provider
        self.channel = pva.Channel(self.input_channel, self.provider)
        self.pva_prefix = input_channel.split(":")[0]
        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.pixel_ordering = 'F'
        self.image_is_transposed = False
        self.attributes = []
        self.timestamp = None
        self.data_type = None
        self.display_dtype = None
        # variables used for parsing analysis data
        self.analysis_index = None
        self.analysis_attributes = {}
        # variables used for parsing hkl data
        self.rsm_index = None
        self.rsm_attributes = {}
        # self.qx = None
        # self.qy = None
        # self.qz = None
        # variables used for later logic
        self.last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.id_diff = 0
        # variables setup using config
        self.config = {}
        self.rois = {}
        self.stats = {}
        self.CONSUMER_MODE = ''
        self.MAX_CACHE_SIZE = 0
        # Cache variables that need to be initialized
        self.cache_images = None
        self.cache_qx = None
        self.cache_qy = None
        self.cache_qz = None


        self._configure(config_filepath)

    def _configure(self, config_path: str) -> None:
        if config_path != '':
            with open(config_path, 'r') as toml_file:
                # loads toml config into a python dict
                self.config:dict = toml.load(toml_file)

        self.stats:dict = self.config.get('STATS', {})
        self.CONSUMER_MODE = self.config.get('CONSUMER_MODE', '')
        self.MAX_CACHE_SIZE = self.config.get('MAX_CACHE_SIZE', 0)
        self.ANALYSIS_IN_CONFIG = ('ANALYSIS' in self.config)
        self._HKL_IN_CONFIG = ('HKL' in self.config)

        if self.config.get('DETECTOR_PREFIX', ''):
            self.pva_prefix = self.config['DETECTOR_PREFIX']

        if "MAX_CACHE_SIZE" in self.config and self.MAX_CACHE_SIZE > 0:
            # Create a 1D NumPy array whose length = MAX_CACHE_SIZE
            self.cache_images = deque(maxlen=self.MAX_CACHE_SIZE)
            if self._HKL_IN_CONFIG:
                self.cache_qx = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cache_qy = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cache_qz = deque(maxlen=self.MAX_CACHE_SIZE)

        else:
            self.cache_images = None

        if self.CONSUMER_MODE == "continuous":
            self.analysis_cache_dict = {"Intensity": {},
                                        "ComX": {},
                                        "ComY": {},
                                        "Position": {}}
        

    def pva_callbackSuccess(self, pv) -> None:
        """
        Callback for handling monitored PV changes.

        Args:
            pv (PvaObject): The PV object received by the channel monitor.
        """
        self.pva_object = pv
        self.parse_image_data_type()
        self.parse_pva_attributes()
         # Check for HKL attributes from all attributes
        if self._HKL_IN_CONFIG:
            self.rsm_index = self.locate_rsm_index()
            if self.rsm_index is not None:
                self.parse_rsm_attributes()
        self.parse_roi_pvs()
        self.pva_to_image()

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
       

    
    def roi_backup_callback(self, pvname, value, **kwargs) -> None:
        name_components = pvname.split(":")
        roi_key = name_components[1]
        pv_key = name_components[2]
        pv_value = value
        # can't append simply by using 2 keys in a row (self.rois[roi_key][pv_key]), there must be an inner dict to call
        # then adds the key to the inner dictionary with update
        self.rois.setdefault(roi_key, {}).update({pv_key: pv_value})
    
    def parse_image_data_type(self) -> None:
        """
        Parses the PVA Object to determine the incoming data type.
        """
        if self.pva_object is not None:
            try:
                self.data_type = list(self.pva_object['value'][0].keys())[0]
                self.display_dtype = self.data_type if self.pva_object['codec']['name'] == '' else self.NTNDA_DATA_TYPE_MAP.get(self.pva_object['codec']['parameters'][0]['value'])

            except:
                self.display_dtype = "could not detect"
    
    def parse_pva_attributes(self) -> None:
        """
        Converts the PVA object to a Python dictionary and extracts its attributes.
        """
        if self.pva_object is not None:
            self.attributes: list = self.pva_object.get().get("attribute", [])
    
    def locate_analysis_index(self) -> int|None:
        """
        Locates the index of the analysis attribute in the PVA attributes.

        Returns:
            int: The index of the analysis attribute or None if not found.
        """
        if self.attributes:
            # for i in range(len(self.attributes))
            for i, attr_pv in enumerate(self.attributes): # attr_pv : dict
                # attr_pv: dict = self.attributes[i]
                if attr_pv.get("name", "") == "Analysis":
                    return i
            else:
                return None
            
    def locate_rsm_index(self) -> int|None:
        if self.attributes:
            # for i in range(len(self.attributes))
            for i, attr_pv in enumerate(self.attributes): # attr_pv : dict
                # attr_pv: dict = self.attributes[i]
                if attr_pv.get("name", "") == "RSM":
                    return i
            else:
                return None
    
    def parse_rsm_attributes(self) -> None:
        rsm_attributes = self.attributes[self.rsm_index]
        self.rsm_attributes = rsm_attributes['value'][0].get('value', {})
        # if self.rsm_attributes:
        #     self.qx = self.rsm_attributes.get('qx', np.zeros(self.shape[0] * self.shape[1]))
        #     self.qy = self.rsm_attributes.get('qy', np.zeros(self.shape[0] * self.shape[1]))
        #     self.qz = self.rsm_attributes.get('qz', np.zeros(self.shape[0] * self.shape[1]))
                
    def parse_roi_pvs(self) -> None:
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
            
    def pva_to_image(self) -> None:
        """
        Converts the PVA Object to an image array and determines if a frame was missed.
        Handles bslz4 and lz4 compressed image data.
        """
        try:
            self.frames_received += 1

            if 'dimension' in self.pva_object:
                self.shape = tuple([dim['size'] for dim in self.pva_object['dimension']])

                # self.empty_array = np.zeros(self.shape[0]*self.shape[1])
                # print('empty shape:', self.empty_array.shape)

                if self.pva_object['codec']['name'] == 'bslz4':
                    # Handle BSLZ4 compressed data
                    dtype = self.NUMPY_DATA_TYPE_MAP.get(self.pva_object['codec']['parameters'][0]['value'])
                    uncompressed_size = self.pva_object['uncompressedSize'] // dtype.itemsize # size has to be divided by bytes needed to store dtype in bitshuffle
                    uncompressed_shape = (uncompressed_size,)
                    compressed_image = self.pva_object['value'][0][self.data_type]
                    # Decompress numpy array to correct datatype
                    self.image = bitshuffle.decompress_lz4(compressed_image, uncompressed_shape, dtype, 0)

                elif self.pva_object['codec']['name'] == 'lz4':
                    # Handle LZ4 compressed data
                    dtype = self.NUMPY_DATA_TYPE_MAP.get(self.pva_object['codec']['parameters'][0]['value'])
                    uncompressed_size = self.pva_object['uncompressedSize'] # raw size is used to decompress it into an lz4 buffer
                    compressed_image = self.pva_object['value'][0][self.data_type]
                    # Decompress using lz4.block
                    decompressed_bytes = lz4.block.decompress(compressed_image, uncompressed_size)
                    # Convert bytes to numpy array with correct dtype
                    self.image = np.frombuffer(decompressed_bytes, dtype=dtype) # dtype is used to convert from buffer to correct dtype from bytes

                elif self.pva_object['codec']['name'] == '':
                    # Handle uncompressed data
                    self.image = np.array(self.pva_object['value'][0][self.data_type]) 
                    # print('img:shape:', self.image.shape)                       

                # Check for missed frame starts here
                current_array_id = self.pva_object['uniqueId']
                if self.last_array_id is not None: 
                    self.id_diff = current_array_id - self.last_array_id - 1
                    if (self.id_diff > 0):
                        self.frames_missed += self.id_diff 
                        # if self._HKL_IN_CONFIG:
                            # for i in range(self.id_diff):
                            #     if self._HKL_IN_CONFIG:
                            #         self.cache_images.append(self.empty_array)
                            #         self.cache_qx.append(self.empty_array)
                            #         self.cache_qy.append(self.empty_array)
                            #         self.cache_qz.append(self.empty_array)                            
                    if self._HKL_IN_CONFIG:
                            self.cache_images.append(self.image)
                            self.cache_qx.append(self.rsm_attributes['qx'])
                            self.cache_qy.append(self.rsm_attributes['qy'])
                            self.cache_qz.append(self.rsm_attributes['qz'])
                        # print(self.cache_qx)
                self.image = self.image.reshape(self.shape, order=self.pixel_ordering).T if self.image_is_transposed else self.image.reshape(self.shape, order=self.pixel_ordering)
                self.last_array_id = current_array_id
                self.id_diff = 0

            else:
                self.image = None
                raise ValueError("Image data could not be processed.")
                
        except Exception as e:
            print(f"Failed to process image: {e}")
            self.frames_received -= 1
            self.frames_missed += 1
            
    def start_channel_monitor(self) -> None:
        """
        Subscribes to the PVA channel with a callback function and starts monitoring for PV changes.
        """
        self.channel.subscribe('pva callback success', self.pva_callbackSuccess)
        self.channel.startMonitor()

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

    def save_caches_to_h5(self, filename: str) -> None:
        # TODO:
        # add motor positions
        # add timestamps
        # add analysis and rest of metadata
        """
        Saves available caches (images and HKL data) to an HDF5 file under a branch structure.
        The file structure is as follows:
            /entry/data         --> The image cache array
            /entry/HKL/qx        --> The qx cache array (if available)
            /entry/HKL/qy        --> The qy cache array (if available)
            /entry/HKL/qz        --> The qz cache array (if available)

        Args:
            filename (str): The output HDF5 file name.
        """
        try:
            if self.cache_images is None or self.cache_qx is None or self.cache_qy is None or self.cache_qz is None:
                raise ValueError("Caches cannot be empty.")
            n = len(self.cache_images)
            if not (len(self.cache_qx) == len(self.cache_qy) == len(self.cache_qz) == n) or n == 0:
                raise ValueError("All four caches must have the same number of elements.")
            
            with h5py.File(filename, 'w') as h5f:
                # Create the main "images" group
                images_grp = h5f.create_group("entry")
                images_grp.create_dataset("data", data=np.array(self.cache_images))

                # Create HKL subgroup under images if HKL caches exist
                hkl_grp = images_grp.create_group("HKL")
                hkl_grp.create_dataset("qx", data=np.array(self.cache_qx))
                hkl_grp.create_dataset("qy", data=np.array(self.cache_qy))
                hkl_grp.create_dataset("qz", data=np.array(self.cache_qz))
                    
            print(f"Caches successfully saved in a branch structure to {filename}")
        except Exception as e:
            print(f"Failed to save caches to {filename}: {e}")

        
    def stop_channel_monitor(self) -> None:
        """
        Stops all monitoring and callback functions.
        """
        self.channel.unsubscribe('pva callback success')
        self.channel.stopMonitor()

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
        return self.attributesa
