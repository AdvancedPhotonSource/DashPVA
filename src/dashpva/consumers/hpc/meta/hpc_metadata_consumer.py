# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
import logging
import time

import lz4.block
import numpy as np
import pvaccess as pva
import toml
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.hpc.meta.base_meta_associator import BaseMetaAssociator


class HpcAdMetadataProcessor(BaseMetaAssociator):

    MIN_COMPRESS_BYTES = 4098

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.all_attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.hkl_config = None
        self.config = None
        self.old_hkl_attributes = None
        self.logger.debug('Created HpcAdMetadataProcessor')
        self.logger.setLevel(logging.DEBUG)

    def configure(self, configDict):
        super().configure(configDict)
        self.logger.debug(f'Configuration update: {configDict}')

        if 'path' in configDict:
            self.path = configDict["path"]
        else:
            import dashpva.settings as _settings
            self.path = _settings.TOML_FILE
            if self.path is None:
                raise RuntimeError(
                    "HpcAdMetadataProcessor: no 'path' in configDict and "
                    "settings.TOML_FILE is not set — configure a TOML config first."
                )

        with open(self.path, "r") as config_file:
            self.config = toml.load(config_file)

        self.hkl_config = self.config.get('HKL') or {}
        self.hkl_pv_channels = set()
        for section in self.hkl_config.values():
            if isinstance(section, dict):
                for channel in section.values():
                    if channel:
                        self.hkl_pv_channels.add(channel)

        try:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Config dict: {configDict}")
        except Exception:
            pass

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

        associationFailed = False
        for metadataChannel, metadataQueue in self.metadataQueueMap.items():
            while True:
                try:
                    self.currentMetadataMap[metadataChannel] = metadataQueue.get(0)
                except pva.QueueEmpty:
                    break
            result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
            if result is not None:
                if not result:
                    associationFailed = True

        unprocessedChannels = list(self.currentMetadataMap.keys())
        for metadataChannel, metadataQueue in self.metadataQueueMap.items():
            if metadataChannel in unprocessedChannels:
                unprocessedChannels.remove(metadataChannel)

        for metadataChannel in unprocessedChannels:
            while True:
                result = self.associateMetadata(metadataChannel, frameId, frameTimestamp, frameAttributes)
                if result is not None:
                    if not result:
                        associationFailed = True
                    break

        if associationFailed:
            self.nFrameErrors += 1
        else:
            self.nFramesProcessed += 1

        proc_time_start = pva.PvObject({'value': pva.DOUBLE})
        proc_time_start['value'] = t0
        frameAttributes.append({
            'name': f'procTimeStart_{self.__class__.__name__}{self.processor_id}',
            'value': proc_time_start
        })
        proc_time_end = pva.PvObject({'value': pva.DOUBLE})
        proc_time_end['value'] = time.time()
        frameAttributes.append({
            'name': f'procTimeEnd_{self.__class__.__name__}{self.processor_id}',
            'value': proc_time_end
        })
        proc_time = pva.PvObject({'value': pva.DOUBLE})
        proc_time['value'] = (time.time() - t0)
        frameAttributes.append({
            'name': f'procTime_{self.__class__.__name__}{self.processor_id}',
            'value': proc_time
        })

        self.compress_image(pvObject)

        pvObject['attribute'] = frameAttributes
        self.updateOutputChannel(pvObject)
        self.lastFrameTimestamp = frameTimestamp
        t1 = time.time()
        self.processingTime += (t1 - t0)
        return pvObject

    def compress_image(self, pvObject) -> None:
        try:
            codec_name = pvObject['codec']['name']
        except Exception:
            codec_name = ''
        if codec_name:
            return

        union_dict = pvObject['value'][0]
        field_name = next(iter(union_dict))
        pv_arr = union_dict[field_name]
        data_list = pv_arr.get() if hasattr(pv_arr, 'get') else pv_arr

        dtype = self.UNION_FIELD_TO_DTYPE.get(field_name, None)
        arr = np.asarray(data_list, dtype=dtype) if dtype is not None else np.asarray(data_list)
        arr_c = np.ascontiguousarray(arr)
        raw = arr_c.tobytes()
        raw_len = arr_c.nbytes

        original_enum = self.CODEC_PARAMETERS_MAP.get(arr_c.dtype, None)

        if raw_len >= self.MIN_COMPRESS_BYTES:
            comp = lz4.block.compress(raw, store_size=False)
            if len(comp) < raw_len:
                comp_data, codec = comp, 'lz4'
            else:
                comp_data, codec = raw, 'none'
        else:
            comp_data, codec = raw, 'none'

        if codec == 'lz4':
            arr_u8 = np.frombuffer(comp_data, dtype=np.uint8)
            pvObject['value'] = ({'ubyteValue': arr_u8.tolist()},)
            pvObject['codec']['name'] = 'lz4'
            pvObject['codec']['parameters'] = ({'value': int(original_enum)},) if original_enum is not None else ()
            pvObject['uncompressedSize'] = raw_len
        else:
            pvObject['codec']['name'] = ''
            pvObject['codec']['parameters'] = ({'value': int(original_enum)},) if original_enum is not None else ()
            pvObject['uncompressedSize'] = raw_len
