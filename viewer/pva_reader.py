import toml
import numpy as np
import pvaccess as pva

class PVAReader:
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
                
    def parse_image_data_type(self) -> None:
        """
        Parses the PVA Object to determine the incoming data type.
        """
        if self.pva_object is not None:
            try:
                self.data_type = list(self.pva_object['value'][0].keys())[0]
            except:
                self.data_type = "could not detect"
    
    def parse_pva_attributes(self) -> None:
        """
        Converts the PVA object to a Python dictionary and extracts its attributes.
        """
        if self.pva_object is not None:
            self.attributes: list = self.pva_object.get().get("attribute", [])
    
    def locate_analysis_index(self) -> None:
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
