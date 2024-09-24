import time
import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import logging

# Example AD Metadata Processor for the streaming framework
# Updates image attributes with values from metadata channels
class HpcAdMetadataProcessor(AdImageProcessor):

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

        # Current metadata map       
        self.currentMetadataMap = {}
        # self.currentframe_attributes = {}

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

    # Associate metadata
    # Returns true on success, false on definite failure, none on failure/try another
    def associateMetadata(self, mdChannel, frameId, frameTimestamp, frameAttributes):
        # self.logger.debug(f" current metadata map: {self.currentMetadataMap}") #modified since 3.8 env isn't working for me, works w/ 3.8
        if mdChannel not in self.currentMetadataMap:
            self.logger.error(f'Metadata channel {mdChannel} not found in current metadata map')
            return False

        mdObject = self.currentMetadataMap[mdChannel]

        # Check if metadata has a timestamp
        if 'timeStamp' in mdObject:
            mdTimestamp = TimeUtility.getTimeStampAsFloat(mdObject['timeStamp'])
            mdTimestamp2 = mdTimestamp + self.metadataTimestampOffset
        # else:
        #     mdTimestamp = frameTimestamp  # Use frame timestamp if no metadata timestamp
        #     mdTimestamp2 = mdTimestamp  # No offset in this case

        if 'value' not in mdObject:
            self.logger.error(f'Metadata object {mdObject} does not have field "value"')
            return False

        mdValue_str = str(mdObject['value'])  # Read value as a string
        self.logger.debug(f"Value from metadata object: {mdValue_str}")
        try:
            mdValue = float(mdValue_str)  # Convert mdValue to float
        except ValueError:
            self.logger.error(f"Failed to convert value '{mdValue_str}' to float")

            return False
        diff = abs(frameTimestamp - mdTimestamp2)
        self.logger.debug(f'Metadata {mdChannel} has value of {mdValue}, timestamp: {mdTimestamp} (with offset: {mdTimestamp2}), timestamp diff: {diff}')
        
        if diff <= self.timestampTolerance:
            # We can associate metadata with frame
            # Create an NtAttribute object with the desired metadata
            nt_attribute = {'name': mdChannel, 'value': pva.PvFloat(mdValue)}
            # Append the NtAttribute object to frameAttributes
            frameAttributes.append(nt_attribute)
            # frameAttributes.append(pva.NtAttribute(mdChannel, pva.PvFloat(mdValue)))
            # self.logger.debug(f'Associating frame id {frameId} with metadata {mdChannel} value of {mdValue}')
            self.nMetadataProcessed += 1 
            # del self.currentMetadataMap[mdChannel]
            return True
        elif frameTimestamp > mdTimestamp2:
            # This metadata is old, but we want it until new one is found (the logic above)
            nt_attribute = {'name': mdChannel, 'value': pva.PvFloat(mdValue)}
            # Append the NtAttribute object to frameAttributes
            frameAttributes.append(nt_attribute)
            # self.logger.debug(f'Keeping old metadata {mdChannel} value of {mdValue} with timestamp {mdTimestamp}')
            self.nMetadataProcessed += 1 
            # del self.currentMetadataMap[mdChannel]
            return True
        else:
            # This metadata is newer than the frame
            # Association failed, but keep metadata for the next frame
            associationFailed = True 
            # self.logger.debug(f'Keeping new metadata {mdChannel} value of {mdValue} with timestamp {mdTimestamp}')
            self.logger.debug('ERROR')
            return False
        # else:
        #     # Use metadata_dict approach if tags are not available
            # if diff <= self.timestampTolerance:
            #     metadata_dict = {
            #         'name': mdChannel,
            #         'value': str(mdValue),
            #         'tags': [],  
            #         'descriptor': '' 
            #     }
            #     frameAttributes.append(metadata_dict)
            #     self.logger.debug(f'Associating frame id {frameId} with metadata {mdChannel} value of {mdValue}')
            #     self.nMetadataProcessed += 1 
            #     del self.currentMetadataMap[mdChannel]
            #     return True
        #     elif frameTimestamp > mdTimestamp2:
        #         # This metadata is too old, discard it and try next one
        #         self.nMetadataDiscarded += 1 
        #         del self.currentMetadataMap[mdChannel]
        #         self.logger.debug(f'Discarding old metadata {mdChannel} value of {mdValue} with timestamp {mdTimestamp}')
        #         return None
        #     else:
        #         # This metadata is newer than the frame
        #         # Association failed, but keep metadata for the next frame
        #         associationFailed = True 
        #         self.logger.debug(f'Keeping new metadata {mdChannel} value of {mdValue} with timestamp {mdTimestamp}')
        #         return False
        
    # Process monitor update
    def process(self, pvObject):
        t0 = time.time()
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
            return pvObject

        frameAttributes = []
        
        if 'attribute' in pvObject:
            frameAttributes = pvObject['attribute']

        if 'timeStamp' not in pvObject:
            self.logger.error(f'Frame id {frameId} does not have field "timeStamp"')
            return pvObject

        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.logger.debug(f'Frame id {frameId} timestamp: {frameTimestamp}')

        # Log the entire pvObject for debugging
        # self.logger.debug(f'Processing pvObject: {pvObject.fram}')


        # TODO: CACHE QUEUE HERE for static metadata
        
        # self.metadataQueueMap will contain channel:pvObjectQueue map
        associationFailed = False
        for metadataChannel,metadataQueue in self.metadataQueueMap.items():
            # TODO: Check if where the metadataQueue.get_nowait() is placed has any effect on output frames issues we've been having
            # metadataQueue.get_nowait()
            while True:
                # if metadataChannel not in self.currentMetadataMap:
                try:
                    self.currentMetadataMap[metadataChannel] = metadataQueue.get(0) #might need to be replaced with metadataQueue.get_nowait()
                except pva.QueueEmpty as ex:
                    # No metadata in the queue, we failed
                    # associationFailed = True 
                    break
                    # pass
                    # result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
            result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
            if result is not None:
                if not result:
                    # Definite failure
                    associationFailed = True 
                    
        # Create a list of metadata channels that are in currentMetadataMap
        processedChannels = list(self.currentMetadataMap.keys())

        # Additional loop to check for any missed metadata channels
        for metadataChannel, metadataQueue in self.metadataQueueMap.items():
            if metadataChannel in processedChannels:
                # Remove the processed channel from the list
                processedChannels.remove(metadataChannel)
        self.logger.debug(f"Remaining channel to append to broacast; {processedChannels}")

        # If there are any remaining channels in processedChannels, process them
        for metadataChannel in processedChannels:
            while True:
                result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
                if result is not None:
                    if not result:
                        # Definite failure
                        associationFailed = True 
                    break
            
        # #debug 
        # if 'attribute' in pvObject:
        #     frameAttributes = pvObject['attribute']
        #     print(f"DEBUG !! Original frame attributes: {frameAttributes}")


        if associationFailed:
            self.nFrameErrors += 1 
        else:
            self.nFramesProcessed += 1 
        # self.logger.debug(f"{frameAttributes=}")
                       
        pvObject['attribute'] = frameAttributes 
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
