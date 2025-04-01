import toml
import numpy as np
import pvaccess as pva
import bitshuffle
import blosc2 as bls
import lz4.block
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
        # Each datatype is enumerated in C++ starting 1-10
        
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
        self.config_filepath = config_filepath
        self.channel = pva.Channel(self.input_channel, self.provider)
        self.pva_prefix = input_channel.split(":")[0]
        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.pixel_ordering = 'C'
        self.image_is_transposed = False
        self.attributes = []
        self.timestamp = None
        self.data_type = None
        self.display_dtype = None
        # variables used for parsing analysis PV
        self.analysis_index = None
        self.analysis_exists = False
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
                self.stats:dict = self.config["STATS"]
                if self.config["CONSUMER_TYPE"] == "spontaneous":
                    # TODO: change to dictionaries that store postions as keys and pv as value
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
        self.pva_to_image()
        self.parse_pva_attributes()
        self.parse_roi_pvs()
        if (self.analysis_index is None) and (not(self.analysis_exists)): #go in with the assumption analysis Doesn't Exist, is changed to True otherwise
            self.analysis_index = self.locate_analysis_index()
        # Only runs if an analysis index was found
        if self.analysis_exists:
            self.analysis_attributes = self.attributes[self.analysis_index]
            if self.config["CONSUMER_TYPE"] == "spontaneous":
                # turns axis1 and axis2 into a tuple
                incoming_coord = (self.analysis_attributes["value"][0]["value"].get("Axis1", 0.0), 
                                  self.analysis_attributes["value"][0]["value"].get("Axis2", 0.0))
                # use a tuple as a key so that we can check if there is a repeat position
                self.analysis_cache_dict["Intensity"].update({incoming_coord: self.analysis_cache_dict["Intensity"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("Intensity", 0.0)})
                self.analysis_cache_dict["ComX"].update({incoming_coord: self.analysis_cache_dict["ComX"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComX", 0.0)})
                self.analysis_cache_dict["ComY"].update({incoming_coord:self.analysis_cache_dict["ComY"].get(incoming_coord, 0) + self.analysis_attributes["value"][0]["value"].get("ComY", 0.0)})
                # double storing of the postion, will find out if needed
                self.analysis_cache_dict["Position"][incoming_coord] = incoming_coord
    
    def roi_backup_callback(self, pvname, value, **kwargs):
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
            for i in range(len(self.attributes)):
                attr_pv: dict = self.attributes[i]
                if attr_pv.get("name", "") == "Analysis":
                    self.analysis_exists = True
                    return i
            else:
                return None
            
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
            if 'dimension' in self.pva_object:
                self.shape = tuple([dim['size'] for dim in self.pva_object['dimension']])

                if self.pva_object['codec']['name'] == 'bslz4':
                    size = self.pva_object['uncompressedSize']
                    dtype = self.NUMPY_DATA_TYPE_MAP.get(self.pva_object['codec']['parameters'][0]['value'])
                    img_uncompressed_size = size // dtype.itemsize
                    uncompressed_shape = (img_uncompressed_size,)
                    # Handle compressed data
                    compressed_image = self.pva_object['value'][0][self.data_type]
                    decompressed_image = bitshuffle.decompress_lz4(compressed_image, uncompressed_shape, dtype, 0)
                    self.image = decompressed_image 
                
                elif self.pva_object['codec']['name'] == 'lz4':
                    # Handle LZ4 compressed data
                    size = self.pva_object['uncompressedSize']
                    dtype = self.NUMPY_DATA_TYPE_MAP.get(self.pva_object['codec']['parameters'][0]['value'])
                    compressed_image = self.pva_object['value'][0][self.data_type]
                    # Decompress using lz4.block
                    decompressed_bytes = lz4.block.decompress(compressed_image, size)
                    # Convert bytes to numpy array with correct dtype
                    self.image = np.frombuffer(decompressed_bytes, dtype=dtype)

                elif self.pva_object['codec']['name'] == '':
                    # Handle uncompressed data
                    self.image = np.array(self.pva_object['value'][0][self.data_type])
                        
                self.image = self.image.reshape(self.shape, order=self.pixel_ordering).T if self.image_is_transposed else self.image.reshape(self.shape, order=self.pixel_ordering)
                self.frames_received += 1
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
        except Exception as e:
            print(f"Failed to process image: {e}")
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
        return self.attributes
