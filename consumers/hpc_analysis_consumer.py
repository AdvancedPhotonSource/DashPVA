import time
import copy
import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
#custom imports
from viewer.generators import rotation_cycle
# import logging

# Example AD Metadata Processor for the streaming framework
# Updates image attributes with values from metadata channels
class HpcAnalysisProcessor(AdImageProcessor):

    # Acceptable difference between image timestamp and metadata timestamp
    DEFAULT_TIMESTAMP_TOLERANCE = 0.001

    # Offset that will be applied to metadata timestamp before comparing it with
    # the image timestamp
    DEFAULT_METADATA_TIMESTAMP_OFFSET = .001

    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)
        # Configuration
        self.timestampTolerance = float(configDict.get('timestampTolerance', self.DEFAULT_TIMESTAMP_TOLERANCE))
        # self.logger.debug(f'Using timestamp tolerance: {self.timestampTolerance} seconds')
        self.metadataTimestampOffset = float(configDict.get('metadataTimestampOffset', self.DEFAULT_METADATA_TIMESTAMP_OFFSET))
        # self.logger.debug(f'Using metadata timestamp offset: {self.metadataTimestampOffset} seconds')

        # Statistics
        self.nFramesProcessed = 0 # Number of images associated with metadata
        self.nFrameErrors = 0 # Number of images that could not be associated with metadata
        self.nMetadataProcessed = 0 # Number of metadata values associated with images
        self.nMetadataDiscarded = 0 # Number of metadata values that were discarded
        self.processingTime = 0

        # PVObject vars needed for caching and analysis
        self.MAX_CACHE_SIZE = 900
        self.pvobject_cache = None    
        self.positions_cache = None
        self.image = None
        self.shape = (0,0)
        self.data_type = None
        self.attributes = {}
        self.cache_id = 0
        self.id_diff = 0
        self.cache_id_gen = rotation_cycle(0,self.MAX_CACHE_SIZE)  

        # The last object time
        self.lastFrameTimestamp = 0

        self.logger.debug(f'Created HpcAdMetadataProcessor')
        # self.logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG
      

    # Configure user processor
    def configure(self, configDict):
        self.logger.debug(f'Configuration update: {configDict}')
        if 'timestampTolerance' in configDict:
            self.timestampTolerance = float(configDict.get('timestampTolerance'))
            self.logger.debug(f'Updated timestamp tolerance: {self.timestampTolerance} seconds')
        if 'metadataTimestampOffset' in configDict:
            self.metadataTimestampOffset = float(configDict.get('metadataTimestampOffset'))
            self.logger.debug(f'Updated metadata timestamp offset: {self.metadataTimestampOffset} seconds')

    ########################################################################################################################
    ####################### Porting Code From Original Area Detector View and Analysis Views ###############################
    ########################################################################################################################
    def parse_image_data_type(self, pva_object):
        """Parse through a PVA Object to store the incoming datatype."""
        if pva_object is not None:
            self.data_type = list(pva_object['value'][0].keys())[0]
    
    def parse_pva_ndattributes(self, pva_object):
        """Convert a pva object to python dict and parses attributes into a separate dict."""
        if pva_object is not None:
            obj_dict = pva_object.get()
        else:
            return None
        
        attributes = {}
        for attr in obj_dict.get("attribute", []):
            name = attr['name']
            value = attr['value']
            attributes[name] = value

        for value in ["codec", "uniqueId", "uncompressedSize"]:
            if value in pva_object:
                attributes[value] = pva_object[value]
        
        self.attributes = attributes

    def pva_to_image(self, pva_object):
        """
        Parses through the PVA Object to retrieve the size and use that to shape incoming image.
        Then immedately check if that PVA Object is next image or if we missed a frame in between.
        """
        try:
            if pva_object is not None:
                # self.frames_received += 1
                # Parses dimensions and reshapes array into image
                if 'dimension' in pva_object:
                    self.shape = tuple([dim['size'] for dim in pva_object['dimension']])
                    self.image = np.array(pva_object['value'][0][self.data_type])
                    # reshapes but also transposes image so it is viewed correctly
                    self.image= np.reshape(self.image, self.shape).T
                else:
                    self.image = None
                # Initialize Image and Positions Cache
                if self.images_cache is None:
                    self.images_cache = np.zeros((self.MAX_CACHE_SIZE, *self.shape))
                    self.positions_cache = np.zeros((self.MAX_CACHE_SIZE,2)) # TODO: make useable for more metadata
                
        except:
            pass
            # self.frames_missed += 1

    def performAnalysis(self):
        pass

    # Process monitor update
    def process(self, pvObject):
        t0 = time.time()
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        # checking for required keys
        if not nDims:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
            return pvObject

        if 'timeStamp' not in pvObject:
            self.logger.error(f'Frame id {frameId} does not have field "timeStamp"')
            return pvObject
        
        self.parse_pva_ndattributes(pvObject)
        self.parse_image_data_type(pvObject)
        self.pva_to_image(pvObject)

        # Check for missed frame starts here:
        if self.last_frame_id is not None: 
            self.id_diff = frameId - self.last_frame_id - 1
        self.last_frame_id = frameId

        # TODO: Need to find a way to make it so that server side key can be changed
        # Look into --processor-args flag in the command line options
        x_value = self.attributes.get('x')[0]['value']
        y_value = self.attributes.get('y')[0]['value']
        
        # TODO: make it so that starting pv has a tolerance for when it's detected similar to check in analysis portions
        if (x_value == 0) and (y_value == 0) and self.first_scan_detected == False:
            self.first_scan_detected = True
            print(f"First Scan detected...")

        if self.first_scan_detected:
            if self.id_diff> 0:
                for i in range(self.id_diff):
                    self.cache_id = next(self.cache_id_gen)
                self.images_cache[self.cache_id-self.id_diff+1:self.cache_id+1,:,:] = 0
                self.positions_cache[self.cache_id-self.id_diff+1:self.cache_id+1,0] = np.NaN 
                self.positions_cache[self.cache_id-self.id_diff+1:self.cache_id+1,1] = np.NaN 
            else:
                self.cache_id = next(self.cache_id_gen)
                self.images_cache[self.cache_id,:,:] = copy.deepcopy(self.image)
                self.positions_cache[self.cache_id,0] = copy.deepcopy(x_value) #TODO: generalize for whatever scan positions we get
                self.positions_cache[self.cache_id,1] = copy.deepcopy(y_value) #TODO: generalize for whatever scan positions we get


        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.logger.debug(f'Frame id {frameId} timestamp: {frameTimestamp}')
                       
        # self.updateOutputChannel(pvObject) # check what this does in comparison to just returning
        self.lastFrameTimestamp = frameTimestamp
        t1 = time.time()
        self.processingTime += (t1-t0)
        return pvObject

    # Reset statistics for user processor
    def resetStats(self):
        self.nFramesProcessed = 0 
        self.nFrameErrors = 0 
        self.nMetadataProcessed = 0 
        self.nMetadataDiscarded = 0 
        self.processingTime = 0

    # Retrieve statistics for user processor
    def getStats(self):
        processedFrameRate = 0
        frameErrorRate = 0
        if self.processingTime > 0:
            processedFrameRate = self.nFramesProcessed/self.processingTime
            frameErrorRate = self.nFrameErrors/self.processingTime
        return { 
            'nFramesProcessed' : self.nFramesProcessed,
            'nFrameErrors' : self.nFrameErrors,
            'nMetadataProcessed' : self.nMetadataProcessed,
            'nMetadataDiscarded' : self.nMetadataDiscarded,
            'processingTime' : FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate' : FloatWithUnits(processedFrameRate, 'fps'),
            'frameErrorRate' : FloatWithUnits(frameErrorRate, 'fps')
        }

    # Define PVA types for different stats variables
    def getStatsPvaTypes(self):
        return { 
            'nFramesProcessed' : pva.UINT,
            'nFrameErrors' : pva.UINT,
            'nMetadataProcessed' : pva.UINT,
            'nMetadataDiscarded' : pva.UINT,
            'processingTime' : pva.DOUBLE,
            'processedFrameRate' : pva.DOUBLE,
            'frameErrorRate' : pva.DOUBLE
        }
