import blosc2
import lz4.block
import bitshuffle
import numpy as np
import pvaccess as pva
from collections import deque
from epics import camonitor, camonitor_clear, caget
from PyQt5.QtCore import QObject, pyqtSignal
import settings as app_settings

class PVAReader(QObject):
    # Signals
    # signal_image_updated = pyqtSignal(np.ndarray)
    # signal_attributes_updated = pyqtSignal(dict)
    # signal_roi_updated = pyqtSignal(dict)
    # signal_rsm_updated = pyqtSignal(dict)
    # signal_analysis_updated = pyqtSignal(dict)
    reader_scan_complete = pyqtSignal()
    scan_state_changed = pyqtSignal(bool)
    reader_new_frame = pyqtSignal()
    
    def __init__(self,
                 input_channel=None,
                 provider=pva.PVA,
                 viewer_type:str='image'):
        """
        Initializes the PVA Reader for monitoring connections and handling image data.

        Args:
            input_channel (str): Input channel for the PVA connection.
            provider (protocol): The protocol for the PVA channel.
        """
        super(PVAReader, self).__init__()
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
        self.pva_prefix = self.input_channel.split(":")[0]

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
        self.metadata_ca = {}  # Store CA metadata PVs

        # variables used for image manipulaiton
        self.pixel_ordering = 'F'
        self.viewer_type = self.VIEWER_TYPE_MAP.get(viewer_type, 'i')
        self.image_is_transposed = False
        
        # variables used for parsing specific attribute data from pv
        self.analysis_index = None
        self.analysis_attributes = {}
        self.rsm_attributes = {}

        # variables used for frame count
        self.last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.id_diff = 0

        # variables for data caches
        self.caches_needed = False
        self.caches_initialized = False
        self.cached_attributes = None
        self.cached_images = None
        self.cached_qx = None
        self.cached_qy = None
        self.cached_qz = None
        # self._on_scan_complete_callbacks = []

        self._configure()

############################# Configuration #############################
    def _configure(self) -> None:
        self.config = app_settings.CONFIG
        self.OUTPUT_FILE_LOCATION = app_settings.OUTPUT_PATH
        self.stats: dict = app_settings.STATS
        self.ROI_IN_CONFIG = (app_settings.ROI != {})
        self.ANALYSIS_IN_CONFIG = (app_settings.ANALYSIS != {})
        self.HKL_IN_CONFIG = (app_settings.HKL != {})
        self.CONSUMER_MODE = app_settings.CONSUMER_MODE or ''
        self.CACHE_OPTIONS: dict = app_settings.CACHE_OPTIONS

        if app_settings.DETECTOR_PREFIX:
            self.pva_prefix = app_settings.DETECTOR_PREFIX

        if self.ROI_IN_CONFIG:
            for roi in app_settings.ROI:
                self._roi_names.append(roi)

        self.set_cache_options()
        if self.caches_needed != self.caches_initialized:
            self.init_caches()

        if self.ANALYSIS_IN_CONFIG and self.CONSUMER_MODE == "continuous":
            self.analysis_cache_dict = {"Position": set(),
                                        "Intensity": {},
                                        "ComX": {},
                                        "ComY": {}}
    def set_cache_options(self) -> None:
        self.CACHING_MODE = app_settings.CACHING_MODE or ''
        if self.CACHING_MODE:
            self.caches_needed = True
            if self.CACHING_MODE == 'alignment':
                self.MAX_CACHE_SIZE = app_settings.ALIGNMENT_MAX_CACHE_SIZE or 100
            elif self.CACHING_MODE == 'scan':
                self.FLAG_PV = app_settings.SCAN_FLAG_PV or ''
                self.START_SCAN = app_settings.SCAN_START_SCAN if app_settings.SCAN_START_SCAN is not None else True
                self.STOP_SCAN = app_settings.SCAN_STOP_SCAN if app_settings.SCAN_STOP_SCAN is not None else False
                self.MAX_CACHE_SIZE = app_settings.SCAN_MAX_CACHE_SIZE or 100
            elif self.CACHING_MODE == 'bin':
                self.BIN_COUNT = app_settings.BIN_COUNT or 10
                self.BIN_SIZE = app_settings.BIN_SIZE or 16

    def init_caches(self) -> None:
        if self.CACHING_MODE == 'alignment' or self.CACHING_MODE == 'scan':
            self.cached_images = deque(maxlen=self.MAX_CACHE_SIZE)
            self.cached_attributes = deque(maxlen=self.MAX_CACHE_SIZE)
            if self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                self.cached_qx = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cached_qy = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cached_qz = deque(maxlen=self.MAX_CACHE_SIZE)
        elif self.CACHING_MODE == 'bin':
            # TODO: when creating the h5 file, have one entry called data that is the average of each bin
            # and then an entry for each bin that lines up with the attributes and rsm attributes
            self.cached_images = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            self.cached_attributes = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            if self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                self.cached_qx = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cached_qy = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cached_qz = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]               
        self.caches_initialized = True

#################### Class and PVA Channel Callbacks ########################
    # def add_on_scan_complete_callback(self, callback_func):
    #     if callable(callback_func):
    #         self._on_scan_complete_callbacks.append(callback_func)
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
            if (self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']) and 'RSM' in self.pv_attributes:
                self.parse_rsm_attributes(self.pv_attributes)

            if self.ANALYSIS_IN_CONFIG and 'Analysis' in self.pv_attributes:
                self.parse_analysis_attributes()
            
            if self.caches_initialized:
                try:
                    if self.cache_attributes(self.pv_attributes, self.rsm_attributes):
                        self.cache_image(np.ravel(self.image))
                except Exception:
                    import traceback
                    traceback.print_exc()

            #TODO: depreciated change the parsing to be closer to parsing RSM attributes with the new parse_attributes function
            if self.ANALYSIS_IN_CONFIG:
                self.analysis_index = self.locate_analysis_index()
                # Only runs if an analysis index was found
                if self.analysis_index is not None:
                    self.analysis_attributes = self.attributes[self.analysis_index]
                    if self.CONSUMER_MODE == "continuous":
                        # turns axis1 and axis2 into a tuple
                        incoming_coord = (self.analysis_attributes["value"][0]["value"].get("Axis1", 0.0), 
                                        self.analysis_attributes["value"][0]["value"].get("Axis2", 0.0))
                        # use a tuple as a key so that we can check if there is a repeat position
                        self.analysis_cache_dict["Intensity"].update({incoming_coord: self.analysis_cache_dict["Intensity"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("Intensity", 0.0)})
                        self.analysis_cache_dict["ComX"].update({incoming_coord: self.analysis_cache_dict["ComX"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComX", 0.0)})
                        self.analysis_cache_dict["ComY"].update({incoming_coord: self.analysis_cache_dict["ComY"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComY", 0.0)})
                        # double storing of the postion, will find out if needed
                        self.analysis_cache_dict["Position"][incoming_coord] = incoming_coord

            self.reader_new_frame.emit()

            if self.is_scan_complete and not self.is_caching:
                self.is_scan_complete = False
                self.reader_scan_complete.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()

    def roi_backup_callback(self, pvname, value, **kwargs) -> None:
        name_components = pvname.split(":")
        # PV format: BL172:eiger4M:ROI1:MinX
        # name_components[0] = "BL172"
        # name_components[1] = "eiger4M"
        # name_components[2] = "ROI1" (this is the roi_key)
        # name_components[3] = "MinX" (this is the pv_key)
        if len(name_components) >= 4:
            roi_key = name_components[2]
            pv_key = name_components[3]
        else:
            # Fallback - try to extract from PV name
            roi_key = name_components[1] if len(name_components) > 1 else "ROI1"
            pv_key = name_components[2] if len(name_components) > 2 else "MinX"
        pv_value = value
        # can't append simply by using 2 keys in a row (self.rois[roi_key][pv_key]), there must be an inner dict to call
        # then adds the key to the inner dictionary with update
        self.rois.setdefault(roi_key, {}).update({pv_key: pv_value})
    
    def metadata_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Callback for CA metadata PV updates.
        Stores the value in self.metadata_ca and also updates pv_attributes.
        """
        self.metadata_ca[pvname] = value
        # Also update pv_attributes so it's available in the same way as PVA attributes
        self.pv_attributes[pvname] = value
        
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
        raise NotImplementedError
        # analysis_attributes: dict = pv_attributes['Analysis']
        # axis_pos = (analysis_attributes['Axis1'], analysis_attributes['Axis2'])
        # intensity = analysis_attributes['Intensity']
    
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
                self.last_array_id = current_array_id
                self.id_diff = 0
                
                return image.reshape(self.shape, order=self.pixel_ordering).T if self.image_is_transposed else image.reshape(self.shape, order=self.pixel_ordering)
            else:
                self.image = None
                raise ValueError("[PV Parsing] Image data could not be processed.")
                
        except Exception:
            pass
            
    def decompress_array(self, compressed_array: np.ndarray, codec: str, uncompressed_size: int, dtype: np.dtype) -> np.ndarray: 
        # Handle LZ4 compressed data
        if codec == 'lz4':
            decompressed_bytes = lz4.block.decompress(compressed_array, uncompressed_size=uncompressed_size)
            # Convert bytes to numpy array with correct dtype
            return np.frombuffer(decompressed_bytes, dtype=dtype) # dtype makes sure we use the correct
        # Handle BSLZ4 compressed data
        elif codec == 'bslz4':
            # uncompressed size has to be divided by the number of bytes needed to store the desired output dtype
            uncompressed_shape = (uncompressed_size // dtype.itemsize,)
            # Decompress numpy array to correct datatype
            return bitshuffle.decompress_lz4(compressed_array, uncompressed_shape, dtype)
        # handle BLOSC compressed data 
        elif codec == 'blosc':
            decompressed_bytes = blosc2.decompress(compressed_array)
            return np.frombuffer(decompressed_bytes, dtype=dtype)

################################## Caching ####################################
    def cache_attributes(self, pv_attributes=None, rsm_attributes=None, analysis_attributes=None) -> bool:
        """Returns True if this frame was cached (caller should also cache the image)."""
        if self.CACHING_MODE == 'alignment':
            self.cached_attributes.append(pv_attributes)
            if rsm_attributes:
                self.cached_qx.append(rsm_attributes['qx'])
                self.cached_qy.append(rsm_attributes['qy'])
                self.cached_qz.append(rsm_attributes['qz'])
            return True
        elif self.CACHING_MODE == 'scan':
            flag_found = self.FLAG_PV in pv_attributes
            if not flag_found:
                return False

            flag_value = pv_attributes[self.FLAG_PV]
            if flag_value == self.START_SCAN:
                if not self.is_caching:
                    self.is_caching = True
                    self.is_scan_complete = False
                    self.scan_state_changed.emit(True)
            elif flag_value == self.STOP_SCAN and self.is_caching:
                self.is_caching = False
                self.is_scan_complete = True
                self.scan_state_changed.emit(False)

            if not self.is_caching:
                return False

            if not rsm_attributes and self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                return False

            self.cached_attributes.append(pv_attributes)
            if rsm_attributes:
                self.cached_qx.append(rsm_attributes['qx'])
                self.cached_qy.append(rsm_attributes['qy'])
                self.cached_qz.append(rsm_attributes['qz'])
            return True
        elif self.CACHING_MODE == 'bin':
            bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
            self.cached_attributes[bin_index].append(pv_attributes)
            return True
        return False

    def cache_image(self, image) -> None:
        if self.CACHING_MODE == 'alignment':
            self.cached_images.append(image)
            return
        elif self.CACHING_MODE == 'scan':
            if self.is_caching:
                self.cached_images.append(image)
                return
        elif self.CACHING_MODE == 'bin':
            if self.viewer_type == 'i':
                bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
                self.cached_images[bin_index].append(image)
                return   
            
    def reset_caches(self) -> None:
        self.cached_images.clear()
        self.cached_attributes.clear()
        self.cached_qx.clear()
        self.cached_qy.clear()
        self.cached_qz.clear()

########################### Start and Stop Channel Monitors ##########################    
    def _flag_pv_ca_callback(self, pvname, value, **kwargs) -> None:
        """CA monitor callback for FLAG_PV — detects scan stop even when PVA frames stop arriving."""
        print(f'[DEBUG] CA flag callback: {pvname}={value!r}  is_caching={self.is_caching}')
        if value == self.STOP_SCAN and self.is_caching:
            self.is_caching = False
            self.scan_state_changed.emit(False)
            self.reader_scan_complete.emit()
            print(f'[DEBUG] CA flag: Scan STOPPED — emitting reader_scan_complete')
        elif value == self.START_SCAN and not self.is_caching:
            self.is_caching = True
            self.is_scan_complete = False
            self.scan_state_changed.emit(True)
            print(f'[DEBUG] CA flag: Scan STARTED')

    def start_channel_monitor(self, callback=None) -> None:
        """
        Subscribes to the PVA channel with a callback function and starts monitoring for PV changes.
        Args:
            callback (function, optional): A custom callback to use for the monitor.
                                           If None, defaults to self.pva_callbackSuccess.
        """
        monitor_callback = callback if callback is not None else self.pva_callbackSuccess
        self.channel.subscribe('pva_monitor', monitor_callback)
        self.channel.startMonitor()
        if self.CACHING_MODE == 'scan' and self.FLAG_PV:
            camonitor(pvname=self.FLAG_PV, callback=self._flag_pv_ca_callback)

    def stop_channel_monitor(self) -> None:
        """
        Stops all monitoring and callback functions.
        """
        self.channel.unsubscribe('pva_monitor')
        self.channel.stopMonitor()
        if self.CACHING_MODE == 'scan' and self.FLAG_PV:
            try:
                camonitor_clear(self.FLAG_PV)
            except Exception:
                pass

    def start_roi_backup_monitor(self) -> None:
        if not app_settings.ROI:
            return
        for roi_num, roi_dict in app_settings.ROI.items():
            for config_key, pv_name in roi_dict.items():
                try:
                    name_components = pv_name.split(":")
                    # PV format: BL172:eiger4M:ROI1:MinX
                    # name_components[0] = "BL172"
                    # name_components[1] = "eiger4M"
                    # name_components[2] = "ROI1" (this is the roi_key)
                    # name_components[3] = "MinX" (this is the pv_key)
                    if len(name_components) >= 4:
                        roi_key = name_components[2] # ROI1-ROI4
                        pv_key = name_components[3] # MinX, MinY, SizeX, SizeY
                    else:
                        # Fallback for different PV naming conventions
                        roi_key = roi_num  # Use config key (ROI1, ROI2, etc.)
                        # Map config keys to PV keys: MIN_X -> MinX, SIZE_X -> SizeX, etc.
                        pv_key_map = {'MIN_X': 'MinX', 'MIN_Y': 'MinY', 'SIZE_X': 'SizeX', 'SIZE_Y': 'SizeY'}
                        pv_key = pv_key_map.get(config_key, config_key)

                    # Use timeout to avoid blocking on slow PVs
                    pv_value = caget(pv_name, timeout=0.5)
                    if pv_value is not None:
                        self.rois.setdefault(roi_key, {}).update({pv_key: pv_value})
                        camonitor(pvname=pv_name, callback=self.roi_backup_callback)
                except Exception as e:
                    # Silently skip failed PVs to avoid spam
                    pass

    def start_metadata_ca_monitor(self) -> None:
        """
        Starts monitoring CA metadata PVs from the [METADATA.CA] section.
        If a PV fails to read, it's skipped. Values are stored in self.metadata_ca
        and also added to self.pv_attributes for consistency.
        """
        metadata_config = self.config.get('METADATA', {})
        if not metadata_config:
            return

        ca_config = metadata_config.get('CA', {})
        if not ca_config:
            return

        for config_key, pv_name in ca_config.items():
            try:
                # Use timeout to avoid blocking on slow PVs
                pv_value = caget(pv_name, timeout=0.5)
                if pv_value is not None:
                    self.metadata_ca[pv_name] = pv_value
                    # Also add to pv_attributes so it's available like PVA attributes
                    self.pv_attributes[pv_name] = pv_value
                    # Start monitoring for updates
                    camonitor(pvname=pv_name, callback=self.metadata_ca_callback)
            except Exception as e:
                # Silently skip failed PVs to avoid spam
                pass

    ################################# Getters ################################# 
    def get_cached_images(self) -> list[np.ndarray]:
        return list(self.cached_images)
    
    def get_cached_attributes(self) -> list[dict]:
        return list(self.cached_attributes)
    
    def get_cached_rsm(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        if len(self.cached_qx) == len(self.cached_qy) == len(self.cached_qz):
            return list(self.cached_qx), list(self.cached_qy), list(self.cached_qz)
        else:
            raise ValueError("[PVA Reader] Cached qx, qy, and qz must have the same length.")

    def get_all_caches(self, clear_caches: bool=False) -> dict:
        """
        Returns all cached data.

        Args:
            clear_caches (bool): Whether to clear the caches after returning the data.
        """
        images =  self.get_cached_images()
        attributes = self.get_cached_attributes()
        
        # Only get RSM data if HKL is configured or viewer is RSM type
        if (self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']) and self.viewer_type != 'i':
            rsm = self.get_cached_rsm()
            # Check lengths including RSM data
            if len(images) == len(attributes) == len(rsm[0]) == len(rsm[1]) == len(rsm[2]):
                data = {
                        'images': images,
                        'attributes': attributes,
                        'rsm': rsm
                        }
            else:
                raise ValueError("[PVA Reader] Cached data must have the same length.")
        else:
            # For image viewer type or when no HKL config, only check images and attributes
            rsm = ([], [], [])  # Empty RSM data
            if len(images) == len(attributes):
                data = {
                        'images': images,
                        'attributes': attributes,
                        'rsm': rsm
                        }
            else:
                raise ValueError("[PVA Reader] Cached data must have the same length.")
        
        
        if clear_caches:
            self.reset_caches()

        return data
    
    def get_output_file_location(self) -> dict:
        fp_pv_name = app_settings.METADATA_CA.get('FILE_PATH', '')
        fn_pv_name = app_settings.METADATA_CA.get('FILE_NAME', '')

        file_path_val = ''
        file_name_val = ''

        # Always caget the live PV values — the save location can change between
        # scan start and scan end, so cached frame attributes may be stale.
        if fp_pv_name:
            try:
                val = caget(fp_pv_name, timeout=1.0)
                if val is not None:
                    file_path_val = str(val).strip()
            except Exception:
                pass
        if fn_pv_name:
            try:
                val = caget(fn_pv_name, timeout=1.0)
                if val is not None:
                    file_name_val = str(val).strip()
            except Exception:
                pass

        if file_path_val and file_name_val:
            return {'FilePath': file_path_val, 'FileName': file_name_val}
        elif file_path_val:
            return {'FilePath': file_path_val}
        else:
            return {'FilePath': str(self.OUTPUT_FILE_LOCATION).strip()}
       
    
    def get_config_settings(self) -> dict:
        config_settings = {'OUTPUT_FILE_CONFIG' : self.get_output_file_location(),
                        'ROI_IN_CONFIG' : self.ROI_IN_CONFIG,
                        'ANALYSIS_IN_CONFIG' : self.ANALYSIS_IN_CONFIG,
                        'HKL_IN_CONFIG' : self.HKL_IN_CONFIG,
                        'CACHE_OPTIONS' : self.CACHE_OPTIONS,
                        'caches_initialized' : self.caches_initialized}
        
        return config_settings
    
    def get_frames_missed(self) -> int:
        """
        Returns the number of frames missed.

        Returns:
            int: The number of missed frames.
        """
        return self.frames_missed

    def get_latest_image(self) -> np.ndarray:
        """
        Returns the current PVA image.

        Returns:
            numpy.ndarray: The current image array.
        """
        return self.image
    
    def get_latest_attributes(self) -> list[dict]:
        """
        Returns the attributes of the current PVA object.

        Returns:
            list: The attributes of the current PVA object.
        """
        return self.attributes

    def get_shape(self) -> tuple[int]:
        return self.shape


