import time
import copy
import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import sys
import os
# Add the parent directory to the Python path to find utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

# COPIED FROM hpc_rsm_consumer.py - Compression libraries
import bitshuffle
import blosc2
import lz4.block
import toml

# Example AD Metadata Processor for the streaming framework
# Updates image attributes with values from metadata channels
class HpcAdMetadataProcessor(AdImageProcessor):

    # Acceptable difference between image timestamp and metadata timestamp
    DEFAULT_TIMESTAMP_TOLERANCE = 0.001
    MIN_COMPRESS_BYTES = 4098
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
        self.processor_id = configDict.get('collectorId') if 'collectorId' in configDict else configDict.get('metadataId', None)
        self.cd = None

        # Current metadata map       
        self.currentMetadataMap = {}
        # self.currentframe_attributes = {}

        # The last object time
        self.lastFrameTimestamp = 0

        # COPIED FROM hpc_rsm_consumer.py - Type mapping for compression
        self.CODEC_PARAMETERS_MAP = {
            np.dtype('uint8'): pva.UBYTE,
            np.dtype('int8'): pva.BYTE,
            np.dtype('uint16'): pva.USHORT,
            np.dtype('int16'): pva.SHORT,
            np.dtype('uint32'): pva.UINT,
            np.dtype('int32'): pva.INT,
            np.dtype('uint64'): pva.ULONG,
            np.dtype('int64'): pva.LONG,
            np.dtype('float32'): pva.FLOAT,
            np.dtype('float64'): pva.DOUBLE,
        }

        # COPIED FROM hpc_rsm_consumer.py - HKL parameters
        self.all_attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.hkl_config = None
        self.config = None
        self.old_hkl_attributes = None

        self.logger.debug(f'Created HpcAdMetadataProcessor')
        self.logger.setLevel(logging.DEBUG)  # Set the logger level to DEBUG

    # COPIED FROM hpc_rsm_consumer.py - Array compression method
    def compress_array(self, hkl_array: np.ndarray, codec_name: str) -> np.ndarray:
        
        if not isinstance(hkl_array, np.ndarray):
            raise TypeError("hkl_array must be a numpy array")
        if hkl_array.ndim != 1:
            raise ValueError("hkl_array must be a 1D numpy array")
        byte_data = hkl_array.tobytes()
        typesize = hkl_array.dtype.itemsize

        if codec_name == 'lz4':
            compressed = lz4.block.compress(byte_data, store_size=False)
        elif codec_name == 'bslz4':
            compressed = bitshuffle.compress_lz4(hkl_array)
        elif codec_name == 'blosc':
            compressed = blosc2.compress(
                byte_data,
                typesize=typesize
            )
        else:
            raise ValueError(f"Unsupported codec: {codec_name}")

        # Convert compressed bytes to a uint8 numpy array
        return np.frombuffer(compressed, dtype=np.uint8)

    # Configure user processor
    def configure(self, configDict):
        self.cd = configDict
        self.logger.debug(f'Configuration update: {configDict}')
        if 'timestampTolerance' in configDict:
            self.timestampTolerance = float(configDict.get('timestampTolerance'))
            self.logger.debug(f'Updated timestamp tolerance: {self.timestampTolerance} seconds')
        if 'metadataTimestampOffset' in configDict:
            self.metadataTimestampOffset = float(configDict.get('metadataTimestampOffset'))
            self.logger.debug(f'Updated metadata timestamp offset: {self.metadataTimestampOffset} seconds')
        
        # COPIED FROM hpc_rsm_consumer.py - HKL configuration setup
        if 'path' in configDict:
            self.path = configDict["path"]
            with open(self.path, "r") as config_file:
                self.config = toml.load(config_file)
                
            if 'HKL' in self.config:
                self.hkl_config : dict = self.config['HKL']
                for section in self.hkl_config.values(): # every section holds a dict
                    for channel in section.values(): # the values of each seciton is the pv name string
                        self.hkl_pv_channels.add(channel)
        
        with open('error_output.txt','w') as f:
            f.write(str(configDict))
        self.logger(configDict)
        #self.processor_id = configDict.get('collectorId') if 'collectorId' in configDict else configDict.get('metadataId', None)

    # Associate metadata
    # Returns true on success, false on definite failure, none on failure/try another
    def associateMetadata(self, mdChannel, frameId, frameTimestamp, frameAttributes):
        # self.logger.debug(f" current metadata map: {self.currentMetadataMap}") #modified since 3.8 env isn't working for me, works w/ 3.8
        if mdChannel not in self.currentMetadataMap:
            self.logger.error(f'Metadata channel {mdChannel} not found in current metadata map')
            print(f'Metadata channel {mdChannel} not found in current metadata map')

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

        mdValue = mdObject['value']  # Read value as a string
        self.logger.debug(f"Value from metadata object: {mdValue}")
        try:
            if isinstance(mdValue, (int, float)):
                mdValue = float(mdValue)  # Convert mdValue to float
                nt_attribute = {'name': mdChannel, 'value': pva.PvFloat(mdValue)}
            elif isinstance(mdValue, str):
                nt_attribute = {'name': mdChannel, 'value': pva.PvString(mdValue)}
            elif isinstance(mdValue, (np.ndarray)):
                pv = pva.PvScalarArray(pva.DOUBLE)
                pv.set(mdValue.tolist())
                nt_attribute = {'name': mdChannel, 'value': pv}
            elif isinstance(mdValue, bool):
                nt_attribute = {'name':mdChannel, 'value': pva.PvBoolean(mdValue)}
            else:
                raise ValueError(f'Failed to create metadata attribute: {mdChannel}: {mdValue}')

            frameAttributes.append(nt_attribute)
        except Exception as e:
            self.logger.error(f"[Metadata Associator] Error associatating metadata {e}")
            return False
        
        diff = abs(frameTimestamp - mdTimestamp2)
        self.logger.debug(f'Metadata {mdChannel} has value of {mdValue}, timestamp: {mdTimestamp} (with offset: {mdTimestamp2}), timestamp diff: {diff}')
        # Here is where any logic with time offsets would go
        # Attach Metadata no matter what
        self.nMetadataProcessed += 1
        return True
        
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
        
        # self.metadataQueueMap will contain channel:pvObjectQueue map
        associationFailed = False
        for metadataChannel,metadataQueue in self.metadataQueueMap.items():
            while True:
                try:
                    self.currentMetadataMap[metadataChannel] = metadataQueue.get(0) # might need to be replaced with metadataQueue.get_nowait()
                except pva.QueueEmpty as ex:
                    # No metadata in the queue, we failed
                    # associationFailed = True 
                    break
            result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
            if result is not None:
                if not result:
                    # Definite failure
                    associationFailed = True 
                    
        # Create a list of metadata channels that are in currentMetadataMap
        unprocessedChannels = list(self.currentMetadataMap.keys())
        # Additional loop to check for any missed/unprocessed metadata channels
        for metadataChannel, metadataQueue in self.metadataQueueMap.items():
            if metadataChannel in unprocessedChannels:
                # Remove the processed channel from the list
                unprocessedChannels.remove(metadataChannel)
        # self.logger.debug(f"Remaining channel to append to broacast; {processedChannels}")

        # If there are any remaining channels in unprocessedChannels, process them
        for metadataChannel in unprocessedChannels:
            while True:
                result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
                if result is not None:
                    if not result:
                        # Definite failure
                        associationFailed = True 
                    break
        # if 'attribute' in pvObject:
        #     frameAttributes = pvObject['attribute']
        #     print(f"DEBUG !! Original frame attributes: {frameAttributes}")

        if associationFailed:
            self.nFrameErrors += 1 
        else:
            self.nFramesProcessed += 1 
        
        #pvObject['attribute'] = frameAttributes 
        proc_time_start = pva.PvObject({'value': pva.DOUBLE})
        proc_time_start['value'] = t0  # seconds, or multiply by 1000.0 for ms
        frameAttributes.append({
            'name': f'procTimeStart_{self.__class__.__name__}{self.processor_id}' ,
            'value': proc_time_start
        })
        proc_time_end = pva.PvObject({'value': pva.DOUBLE})
        proc_time_end['value'] = time.time()  # seconds, or multiply by 1000.0 for ms
        frameAttributes.append({
            'name': f'procTimeEnd_{self.__class__.__name__}{self.processor_id}',
            'value': proc_time_end
        })
        proc_time = pva.PvObject({'value': pva.DOUBLE})
        proc_time['value'] = (time.time() - t0)  # seconds, or multiply by 1000.0 for ms
        frameAttributes.append({
            'name': f'procTime_{self.__class__.__name__}{self.processor_id}',
            'value': proc_time
        })

        
        self.compress_image(pvObject)

        pvObject['attribute'] = frameAttributes
        self.updateOutputChannel(pvObject)
        self.lastFrameTimestamp = frameTimestamp
        t1 = time.time()
        self.processingTime += (t1-t0)
        return pvObject
    
    def compress_image(self, pvObject) -> None:
        # If already compressed, do nothing
        try:
            codec_name = pvObject['codec']['name']
        except Exception:
            codec_name = ''
        if codec_name:
            return

        # Extract active union field and its array
        union_dict = pvObject['value'][0]
        field_name = next(iter(union_dict))
        pv_arr = union_dict[field_name]
        data_list = pv_arr.get() if hasattr(pv_arr, 'get') else pv_arr

        # Map union field to numpy dtype
        UNION_FIELD_TO_DTYPE = {
            'ubyteValue': np.uint8,
            'byteValue': np.int8,
            'ushortValue': np.uint16,
            'shortValue': np.int16,
            'uintValue': np.uint32,
            'intValue': np.int32,
            'ulongValue': np.uint64,
            'longValue': np.int64,
            'floatValue': np.float32,
            'doubleValue': np.float64,
        }
        dtype = UNION_FIELD_TO_DTYPE.get(field_name, None)
        arr = np.asarray(data_list, dtype=dtype) if dtype is not None else np.asarray(data_list)
        arr_c = np.ascontiguousarray(arr)
        raw = arr_c.tobytes()
        raw_len = arr_c.nbytes

        original_enum = self.CODEC_PARAMETERS_MAP.get(arr_c.dtype, None)

        # Compress and decide
        if raw_len >= self.MIN_COMPRESS_BYTES:
            comp = lz4.block.compress(raw, store_size=False)
            if len(comp) < raw_len:
                comp_data, codec = comp, 'lz4'
            else:
                comp_data, codec = raw, 'none'
        else:
            comp_data, codec = raw, 'none'

        if codec == 'lz4':
            # Compressed path: put bytes under UBYTE union branch
            arr_u8 = np.frombuffer(comp_data, dtype=np.uint8)
            # PvAccess expects a list for union array values
            pvObject['value'] = ({'ubyteValue': arr_u8.tolist()},)
            pvObject['codec']['name'] = 'lz4'
            pvObject['codec']['parameters'] = ({'value': int(original_enum)},) if original_enum is not None else ()
            pvObject['uncompressedSize'] = raw_len
        else:
            # Leave original branch and clear codec
            pvObject['codec']['name'] = ''
            pvObject['codec']['parameters'] = ({'value': int(original_enum)},) if original_enum is not None else ()
            pvObject['uncompressedSize'] = raw_len
        # Debug
        # print(f"Field: {field_name} Original bytes: {raw_len}, Compressed bytes: {len(comp_data)}, Codec: {codec}")


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
            'frameErrorRate' : FloatWithUnits(frameErrorRate, 'fps'),
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
        }
