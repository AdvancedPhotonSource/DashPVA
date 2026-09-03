# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import time

# COPIED FROM hpc_rsm_consumer.py - Compression libraries
import bitshuffle
import blosc2
import lz4.block
import numpy as np
import pvaccess as pva
import toml
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.core.base_meta_associator import BaseMetaAssociator
from dashpva.utils.config.hkl import semantic_hkl_channels
from dashpva.utils.config.resolver import resolve_profile_config
from dashpva.utils.metadata_binding import METADATA_TIMESTAMP_ATTRIBUTE_PREFIX


def _load_resolved_config(path):
    """Load TOML and derive canonical HKL records before metadata selection."""
    with open(path, "r") as config_file:
        return resolve_profile_config(toml.load(config_file))


# Example AD Metadata Processor for the streaming framework
# Updates image attributes with values from metadata channels
class HpcAdMetadataProcessor(BaseMetaAssociator):

    MIN_COMPRESS_BYTES = 4098

    def __init__(self, configDict={}):
        # The base supplies the log manager, the codec map, the generic frame
        # and metadata counters, and the tolerance/offset configuration.
        super().__init__(configDict)
        self.all_attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.hkl_config = None
        self.config = None
        self.old_hkl_attributes = None

        self.logger.debug('Created HpcAdMetadataProcessor')

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
        else:
            import dashpva.settings as _settings
            self.path = _settings.ensure_path()
            if self.path is None:
                raise RuntimeError(
                    "HpcAdMetadataProcessor: no 'path' in configDict and "
                    "no effective config path is available — configure a profile first."
                )

        self.config = _load_resolved_config(self.path)

        self.hkl_config = self.config.get('HKL') or {}
        self.hkl_pv_channels = set(semantic_hkl_channels(self.hkl_config))
        
        # Log configuration via central logger instead of writing to a file
        try:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Config dict: {configDict}")
        except Exception:
            pass
        #self.processor_id = configDict.get('collectorId') if 'collectorId' in configDict else configDict.get('metadataId', None)

    # Associate metadata
    # Returns true on success, false on definite failure, none on failure/try another
    def associateMetadata(self, mdChannel, frameId, frameTimestamp, frameAttributes):
        # self.logger.debug(f" current metadata map: {self.currentMetadataMap}") #modified since 3.8 env isn't working for me, works w/ 3.8
        if mdChannel not in self.currentMetadataMap:
            # Metadata for this channel has not arrived, so it cannot be
            # attached. A routine discard, not a fault: counted for the status
            # channel, not logged -- warning per channel per frame buried real
            # errors under thousands of lines.
            self.nMetadataDiscarded += 1
            return False

        mdObject = self.currentMetadataMap[mdChannel]

        # Check if metadata has a timestamp
        if 'timeStamp' in mdObject:
            mdTimestamp = TimeUtility.getTimeStampAsFloat(mdObject['timeStamp'])
            mdTimestamp2 = mdTimestamp + self.metadataTimestampOffset
        else:
            self.nMetadataDiscarded += 1
            return False

        if 'value' not in mdObject:
            self.logger.error(f'Metadata object {mdObject} does not have field "value"')
            return False

        diff = abs(frameTimestamp - mdTimestamp2)
        self.logger.debug(f'Metadata {mdChannel} has timestamp: {mdTimestamp} (with offset: {mdTimestamp2}), timestamp diff: {diff}')
        if diff > self.timestampTolerance:
            self.nMetadataDiscarded += 1
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
            frameAttributes.append({
                'name': f'{METADATA_TIMESTAMP_ATTRIBUTE_PREFIX}{mdChannel}',
                'value': pva.PvDouble(mdTimestamp2),
            })
        except Exception as e:
            self.logger.error(f"[Metadata Associator] Error associatating metadata {e}")
            return False
        
        self.nMetadataProcessed += 1
        return True
        
    # Process monitor update
    def process(self, pvObject):
        # Catch-all so any processing problem is logged (with traceback) via the
        # LogMixin logger instead of vanishing; the frame is still forwarded.
        try:
            return self._process_frame(pvObject)
        except Exception:
            self.nFrameErrors += 1
            try:
                _fid = pvObject['uniqueId']
            except Exception:
                _fid = '?'
            self.logger.exception(
                f'[Metadata Associator] process() failed for frame {_fid}')
            return pvObject

    def _process_frame(self, pvObject):
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
            while len(metadataQueue) > 0:
                self.currentMetadataMap[metadataChannel] = metadataQueue.get(0)
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
        # Original bytes: 2097152, Compressed bytes: 55418, Codec: lz4, Image size 1024x1024
        # Ratio 600 : 16
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
        #msg = f"Original bytes: {raw_len}, Compressed bytes: {len(comp_data)}, Codec: {codec}" * 10
        #print(msg)


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
