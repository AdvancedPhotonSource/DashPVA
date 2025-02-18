import time
import numpy as np
import pvaccess as pva
from pvaccess import PvObject, DOUBLE
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import sys

import toml

# Example AD Metadata Processor for the streaming framework
# This updated version processes one frame at a time.
class HpcAnalysisProcessor(AdImageProcessor):

    def __init__(self, configDict={}):
        super(HpcAnalysisProcessor, self).__init__(configDict)

        # Statistics
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMetadataProcessed = 0
        self.nMetadataDiscarded = 0
        self.processingTime = 0

        # Configuration parameters
        self.configure(configDict)

        # Image and ROI settings
        # These can be changed to suit your analysis needs
        self.roi_x = 0
        self.roi_y = 0
        self.roi_width = 50
        self.roi_height = 50

        # The last processed frame's timestamp
        self.lastFrameTimestamp = 0

        # Dictionary to store attributes from the current frame
        self.attributes = {}

    def configure(self, configDict):
        """
        Configure user-defined settings from configDict if needed.
        """
        if 'path' in configDict:
            self.path = configDict['path']
            with open(self.path, 'r') as f:
                self.config: dict = toml.load(f)
        else:
            self.path = None

        self.axis1 = self.config.get('analysis', {}).get('Axis1', None)
        self.axis2 = self.config.get('analysis', {}).get('Axis2', None)



    def parse_image_data_type(self, pva_object):
        """
        Parse the PVA Object to determine the incoming datatype of the image.
        """
        if pva_object is not None:
            self.data_type = list(pva_object['value'][0].keys())[0]

    def parse_pva_ndattributes(self, pva_object):
        """
        Parse the NDAttributes from the PVA Object into a python dict.
        Store attributes in self.attributes for easy reference.
        """
        if pva_object is None:
            return
        obj_dict = pva_object.get()
        attributes = {}
        for attr in obj_dict.get("attribute", []):
            name = attr['name']
            value = attr['value']
            attributes[name] = value

        # Include additional values commonly found at top-level for completeness.
        for value_key in ["codec", "uniqueId", "uncompressedSize"]:
            if value_key in pva_object:
                attributes[value_key] = pva_object[value_key]

        self.attributes = attributes

    def pva_to_image(self, pva_object):
        """
        Convert the PVA Object to a NumPy array representing the image.
        Apply correct shaping and transpose if needed.
        """
        try:

            self.image = None
            if pva_object is not None and 'dimension' in pva_object:
                dims = pva_object['dimension']
                shape = tuple([dim['size'] for dim in dims])
                raw_data = np.array(pva_object['value'][0][self.data_type])
                # Reshape and transpose if necessary to get correct orientation
                self.image = np.reshape(raw_data, shape).T
        except:
            print("error parsing images")
                    

    def process(self, pvObject):
        """
        Process each incoming frame individually.
        Steps:
          1. Parse attributes and image data.
          2. Compute ROI-based intensity and center-of-mass (COM).
          3. Get current frame's X, Y from attributes.
          4. Append these analysis results as an NtAttribute to the pvObject.
        """
        t0 = time.time()

        # Retrieve frame id
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            # Frame has no image data
            return pvObject

        if 'timeStamp' not in pvObject:
            # No timestamp, just return the object
            return pvObject

        # Parse attributes and image type
        self.parse_pva_ndattributes(pvObject)
        self.parse_image_data_type(pvObject)
        self.pva_to_image(pvObject)

        if self.image is None:
            # If we cannot form the image, skip analysis
            return pvObject

        # Extract X, Y positions from attributes as they come in
        # The original code accessed x,y as: attributes.get('x')[0]['value']
        # Adjust as needed depending on attribute structure.
        if self.axis1 is not None and self.axis2 is not None:
            x_attr = self.attributes.get(self.axis1, None)
            y_attr = self.attributes.get(self.axis2, None)
            if x_attr is not None and y_attr is not None:
                x_value = x_attr[0]['value'] if isinstance(x_attr, tuple) else 0.0
                y_value = y_attr[0]['value'] if isinstance(y_attr, tuple) else 0.0
        else:
            # Default to 0 if attributes not found
            x_value = 0.0
            y_value = 0.0

        # Extract Region of Interest (ROI) from the image
        roi = self.image[self.roi_y:self.roi_y+self.roi_height,
                         self.roi_x:self.roi_x+self.roi_width]

        # Compute intensity (sum of ROI pixels)
        intensity = np.sum(roi)

        # Compute center-of-mass (COM)
        # To avoid division by zero, check intensity
        if intensity <= 0:
            com_x = 0.0
            com_y = 0.0
        else:
            y_coords, x_coords = np.indices(roi.shape)
            weighted_sum_x = np.sum(roi * x_coords)
            weighted_sum_y = np.sum(roi * y_coords)
            com_x = weighted_sum_x / intensity
            com_y = weighted_sum_y / intensity

        # Now create a PvObject with the analysis results
        # We will send out a single data point (X, Y, Intensity, ComX, ComY)
        analysis_object = PvObject({'value':{'Axis1': DOUBLE, 'Axis2': DOUBLE,
                                    'Intensity': DOUBLE, 'ComX': DOUBLE, 'ComY': DOUBLE}},
                                   {'value':{'Axis1': float(x_value),
                                    'Axis2': float(y_value),
                                    'Intensity': float(intensity),
                                    'ComX': float(com_x),
                                    'ComY': float(com_y)}})

        # Create an NtAttribute to hold this analysis data
        pvAttr = pva.NtAttribute('Analysis', analysis_object)

        # Append this attribute to the frame's attribute list
        frameAttributes = pvObject['attribute']
        frameAttributes.append(pvAttr)
        pvObject['attribute'] = frameAttributes

        # Update stats
        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.lastFrameTimestamp = frameTimestamp
        self.nFramesProcessed += 1

        # Update output channel if needed
        self.updateOutputChannel(pvObject)

        # Update processing time
        t1 = time.time()
        self.processingTime += (t1 - t0)

        return pvObject

    def resetStats(self):
        """
        Reset processor statistics.
        """
        self.nFramesProcessed = 0 
        self.nFrameErrors = 0 
        self.nMetadataProcessed = 0 
        self.nMetadataDiscarded = 0 
        self.processingTime = 0

    def getStats(self):
        """
        Get current statistics of processing.
        """
        processedFrameRate = 0
        frameErrorRate = 0
        if self.processingTime > 0:
            processedFrameRate = self.nFramesProcessed / self.processingTime
            frameErrorRate = self.nFrameErrors / self.processingTime
        return { 
            'nFramesProcessed' : self.nFramesProcessed,
            'nFrameErrors' : self.nFrameErrors,
            'nMetadataProcessed' : self.nMetadataProcessed,
            'nMetadataDiscarded' : self.nMetadataDiscarded,
            'processingTime' : FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate' : FloatWithUnits(processedFrameRate, 'fps'),
            'frameErrorRate' : FloatWithUnits(frameErrorRate, 'fps')
        }

    def getStatsPvaTypes(self):
        """
        Define PVA types for different stats variables.
        """
        return { 
            'nFramesProcessed' : pva.UINT,
            'nFrameErrors' : pva.UINT,
            'nMetadataProcessed' : pva.UINT,
            'nMetadataDiscarded' : pva.UINT,
            'processingTime' : pva.DOUBLE,
            'processedFrameRate' : pva.DOUBLE,
            'frameErrorRate' : pva.DOUBLE
        }
