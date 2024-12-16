import time
import copy
import numpy as np
import pvaccess as pva
from pvaccess import PvObject, DOUBLE
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
#custom imports
import sys
sys.path.append('/home/beams0/JULIO.RODRIGUEZ/Desktop/Lab Software/area_det_PVA_viewer')
from viewer.generators import rotation_cycle

# import logging

# Example AD Metadata Processor for the streaming framework
# Updates image attributes with values from metadata channels
class HpcAnalysisProcessor(AdImageProcessor):

    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)

        # Statistics
        self.nFramesProcessed = 0 # Number of images associated with metadata
        self.nFrameErrors = 0 # Number of images that could not be associated with metadata
        self.nMetadataProcessed = 0 # Number of metadata values associated with images
        self.nMetadataDiscarded = 0 # Number of metadata values that were discarded
        self.processingTime = 0

        self.configure(configDict)

        # PVObject vars needed for caching
        self.images_cache = None    
        self.positions_cache = None
        self.image = None
        self.shape = (0,0)
        self.data_type = None
        self.attributes = {}
        self.cache_id = 0
        self.id_diff = 0
        self.cache_id_gen = rotation_cycle(0,self.MAX_CACHE_SIZE)
        self.first_scan_detected = False 
        self.last_frame_id = None 
        self.call_times = 0
        # Configure Processor Settings
        # The last object time
        self.lastFrameTimestamp = 0
        self.roi_x = 0
        self.roi_y = 0
        self.roi_height = 50
        self.roi_width = 50
        # self.logger.debug(f'Created HpcAnalysisProcessor')
        # self.logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG
      
    # Configure user processor
    def configure(self, configDict):
        # TODO: Use Configdict to get all these parameters.
        self.MAX_CACHE_SIZE = 900
        # Vars used for Analysis
        self.xpos_path = "/home/beams0/JULIO.RODRIGUEZ/Desktop/Lab Software/area_det_PVA_viewer/xpos.npy"
        self.ypos_path = "/home/beams0/JULIO.RODRIGUEZ/Desktop/Lab Software/area_det_PVA_viewer/ypos.npy"
        # self.save_path = "save_path"
        self.load_path()
        
    ####################### Porting Code From Original Area Detector View and Analysis Views ###############################
    
    def parse_image_data_type(self, pva_object):
        """Parse through a PVA Object to store the incoming datatype."""
        if pva_object is not None:
            self.data_type = list(pva_object['value'][0].keys())[0]
    
    def parse_pva_ndattributes(self, pva_object):
        """Convert a pva object to python dict and parses attributes into a separate dict."""
        if pva_object is None:
            return

        obj_dict = pva_object.get()
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

    def load_path(self):
        """
        This function loads the path information for the HDF5 file and uses it to populate other variables.
        """
        # TODO: These positions are references, use only when we miss frames.
        self.x_positions = np.load(self.xpos_path)
        self.y_positions = np.load(self.ypos_path)
    
        self.unique_x_positions = np.unique(self.x_positions) # Time Complexity = O(nlog(n))
        self.unique_y_positions = np.unique(self.y_positions) # Time Complexity = O(nlog(n))

        self.x_indices = np.searchsorted(self.unique_x_positions, self.x_positions) # Time Complexity = O(log(n))
        self.y_indices = np.searchsorted(self.unique_y_positions, self.y_positions) # Time Complexity = O(log(n))

    def process_analysis_objects(self, frameAttributes=None):
        """
        
        """
        # if self.cache_id is not None:
        #     xpos_det = self.positions_cache[self.cache_id, 0]
        #     xpos_plan = self.x_positions[self.cache_id]
        #     ypos_det = self.positions_cache[self.cache_id, 1]
        #     ypos_plan = self.y_positions[self.cache_id]

        #     # print(f'scan id: {self.cache_id}')
        #     # print(f'detector: {xpos_det},{ypos_det}')
        #     # print(f'scan plan: {xpos_plan},{ypos_plan}\n')

        #     x1, x2 = self.x_positions[0], self.x_positions[1]
        #     y1, y2 = self.y_positions[0], self.y_positions[30]


        # (np.abs(xpos_plan-xpos_det) < (np.abs(x2-x1) * 0.2)) and (np.abs(ypos_plan-ypos_det) < (np.abs(y2-y1) * 0.2)))
        if True: 
            
            self.call_times += 1
            image_rois = self.images_cache[:,
                                            self.roi_y:self.roi_y + self.roi_height,
                                            self.roi_x:self.roi_x + self.roi_width]# self.pv_dict.get('rois', [[]]) # Time Complexity = O(n)

            intensity_values = np.sum(image_rois, axis=(1, 2)) # Time Complexity = O(n)
            intensity_values_non_zeros = intensity_values # removed deep copy of intensity values as memory was cleared with every function call
            intensity_values_non_zeros[intensity_values_non_zeros==0] = 1E-6 # time complexity = O(1)

            y_coords, x_coords = np.indices((np.shape(image_rois)[1], np.shape(image_rois)[2])) # Time Complexity = 0(n)

            # Compute weighted sums
            weighted_sum_y = np.sum(image_rois[:, :, :] * y_coords[np.newaxis, :, :], axis=(1, 2)) # Time Complexity O(n)
            weighted_sum_x = np.sum(image_rois[:, :, :] * x_coords[np.newaxis, :, :], axis=(1, 2)) # Time Complexity O(n)
            # Calculate COM
            com_y = weighted_sum_y / intensity_values_non_zeros # time complexity = O(1)
            com_x = weighted_sum_x / intensity_values_non_zeros # time complexity = O(1)
            #filter out inf
            com_x[com_x==np.nan] = 0 # time complexity = O(1)
            com_y[com_y==np.nan] = 0 # time complexity = O(1)
            #Two lines below don't work if unique positions are messed by incomplete x y positions 
            self.intensity_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
            self.intensity_matrix[self.y_indices, self.x_indices] = intensity_values # Time Complexity = O(1)
            # gets the shape of the image to set the length of the axis
            self.com_x_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
            self.com_y_matrix = np.zeros((len(self.unique_y_positions), len(self.unique_x_positions))) # Time Complexity = O(n)
            # Populate the matrices using the indices
            self.com_x_matrix[self.y_indices, self.x_indices] = com_x # Time Complexity = O(1)
            self.com_y_matrix[self.y_indices, self.x_indices] = com_y # Time Complexity = O(1)

            # TODO: Create pv object out of the matrices and append them to the original pvobject
            analysis_object = PvObject({'value':{
                                            "Intensity": [DOUBLE],
                                            "ComX": [DOUBLE], 
                                            "ComY": [DOUBLE]}}, 
                                        {'value':{
                                            "Intensity":self.intensity_matrix.ravel(), 
                                            "ComX": self.com_x_matrix.ravel(), 
                                            "ComY": self.com_y_matrix.ravel()}})
            
            pvAttr = pva.NtAttribute('Analysis', analysis_object)

            frameAttributes.append(pvAttr)

    ########################################## Process monitor update ####################################################

    def process(self, pvObject):
        t0 = time.time()
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        # checking for required keys
        if not nDims:
            # self.logger.debug(f'Frame id {frameId} contains an empty image.')
            return pvObject

        if 'timeStamp' not in pvObject:
            # self.logger.error(f'Frame id {frameId} does not have field "timeStamp"')
            return pvObject
        
        self.parse_pva_ndattributes(pvObject)
        self.parse_image_data_type(pvObject)
        self.pva_to_image(pvObject)

        # Check for missed frame starts here:
        if self.last_frame_id is not None: 
            self.id_diff = frameId - self.last_frame_id - 1
        self.last_frame_id = frameId

        # TODO: USE --processor-args flag in the command line options to pass a config dict and get these values instead
        x_value = self.attributes.get('x')[0]['value']
        y_value = self.attributes.get('y')[0]['value']
        
        # TODO: make it so that starting pv has a tolerance for when it's detected similar to check in analysis portions
        if (x_value == 0) and (y_value == 0) and self.first_scan_detected == False:
            self.first_scan_detected = True
            print(f"First Scan detected...")

        if self.first_scan_detected:
            if self.id_diff > 0:
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
            
            # self.process_analysis_objects(pvObject=pvObject)
            frameAttributes = pvObject['attribute']
            self.process_analysis_objects(frameAttributes=frameAttributes)
            pvObject['attribute'] = frameAttributes

        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        #self.logger.debug(f'Frame id {frameId} timestamp: {frameTimestamp}')
                       
        # self.updateOutputChannel(pvObject) # check what this does in comparison to just returning
        self.updateOutputChannel(pvObject)
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
