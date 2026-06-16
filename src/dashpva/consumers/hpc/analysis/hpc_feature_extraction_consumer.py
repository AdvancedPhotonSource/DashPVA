"""
HPC consumer that extracts deterministic numerical features from each
detector frame and appends them as a FeatureVector NdAttribute.

Upstream requirements
---------------------
HpcBlobTrackingProcessor must be earlier in the pipeline — it populates
the BlobDetections and BlobDetCount attributes that this consumer reads.

Output attributes added to each frame
--------------------------------------
FeatureVector : PvString(JSON)
    JSON-encoded dict with 'n_blobs', 'blobs', and 'frame' keys.
    See dashpva.analysis.feature_extractor.FrameFeatureExtractor for schema.
"""

import json
import time

import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor

from dashpva.analysis.feature_extractor import FrameFeatureExtractor


class HpcFeatureExtractionProcessor(AdImageProcessor):
    """
    Wraps FrameFeatureExtractor as a pvapy HPC consumer.

    Reads the raw image and upstream BlobDetections, computes per-blob
    and global frame features, and appends FeatureVector to pvObject.
    """

    PVA_TO_NUMPY_DTYPE_MAP = {
        pva.UBYTE:  np.uint8,
        pva.BYTE:   np.int8,
        pva.USHORT: np.uint16,
        pva.SHORT:  np.int16,
        pva.UINT:   np.uint32,
        pva.INT:    np.int32,
        pva.ULONG:  np.uint64,
        pva.LONG:   np.int64,
        pva.FLOAT:  np.float32,
        pva.DOUBLE: np.float64,
    }

    UNION_FIELD_TO_DTYPE = {
        'ubyteValue':  np.uint8,
        'byteValue':   np.int8,
        'ushortValue': np.uint16,
        'shortValue':  np.int16,
        'uintValue':   np.uint32,
        'intValue':    np.int32,
        'ulongValue':  np.uint64,
        'longValue':   np.int64,
        'floatValue':  np.float32,
        'doubleValue': np.float64,
    }

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.extractor = FrameFeatureExtractor()
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.processingTime = 0.0

    def configure(self, configDict):
        pass

    # ------------------------------------------------------------------
    # Image decompression (mirrors HpcBlobTrackingProcessor)
    # ------------------------------------------------------------------

    def _decompress_image(self, pvObject) -> np.ndarray:
        import lz4.block
        codec_name = pvObject['codec']['name']
        if codec_name == 'lz4':
            u8_pv = pvObject['value'][0]['ubyteValue']
            u8_list = u8_pv.get() if hasattr(u8_pv, 'get') else u8_pv
            comp_bytes = np.asarray(u8_list, dtype=np.uint8).tobytes()
            out_bytes = lz4.block.decompress(
                comp_bytes, uncompressed_size=pvObject['uncompressedSize'])
            params = pvObject['codec']['parameters']
            enum = params[0]['value'] if (isinstance(params, tuple) and len(params) > 0) else pva.UBYTE
            dtype = self.PVA_TO_NUMPY_DTYPE_MAP.get(enum, np.uint8)
            return np.frombuffer(out_bytes, dtype=dtype)
        union_dict = pvObject['value'][0]
        field_name = next(iter(union_dict))
        pv_arr = union_dict[field_name]
        data_list = pv_arr.get() if hasattr(pv_arr, 'get') else pv_arr
        dtype = self.UNION_FIELD_TO_DTYPE.get(field_name)
        return np.asarray(data_list, dtype=dtype) if dtype is not None else np.asarray(data_list)

    # ------------------------------------------------------------------
    # Read blob detections appended by upstream consumer
    # ------------------------------------------------------------------

    def _read_blob_detections(self, pvObject) -> np.ndarray:
        # Use 'attribute' in pvObject + pvObject['attribute'] — the pvapy
        # PvObject.get() C++ binding doesn't accept Python keyword args and
        # would raise Boost.Python.ArgumentError if called as .get(key, default).
        try:
            if 'attribute' not in pvObject:
                return np.empty((0, 5))
            for attr in pvObject['attribute']:
                if attr['name'] == 'BlobDetections':
                    raw = attr['value'][0].get('value', '')
                    if not raw:
                        return np.empty((0, 5))
                    parsed = json.loads(raw)
                    if not parsed:
                        return np.empty((0, 5))
                    return np.array(parsed, dtype=np.float64).reshape(-1, 5)
        except Exception:
            pass
        return np.empty((0, 5))

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, pvObject):
        t0 = time.time()

        dims = pvObject['dimension']
        if not len(dims):
            self.updateOutputChannel(pvObject)
            return pvObject

        try:
            shape = tuple(dim['size'] for dim in dims)
            flat = self._decompress_image(pvObject)
            image = flat.reshape(shape) if flat.size == np.prod(shape) else flat.reshape(shape[:2])

            blob_dets = self._read_blob_detections(pvObject)
            features = self.extractor.extract(image, blob_dets)

            # Stamp the detector uniqueId and POSIX timestamp into the feature
            # dict so downstream consumers (SessionAnalyzer prompt labels,
            # historical-PV chat tools, HDF5 reader) can map each cached
            # feature to a specific detector frame instead of a list index.
            try:
                features['frame_id'] = int(pvObject['uniqueId'])
            except Exception:
                features['frame_id'] = -1
            try:
                ts = pvObject['timeStamp']
                features['timestamp'] = (
                    float(ts['secondsPastEpoch']) + float(ts['nanoseconds']) * 1e-9
                )
            except Exception:
                features['timestamp'] = 0.0

            new_attr = {
                'name': 'FeatureVector',
                'value': pva.PvString(json.dumps(features)),
            }
            existing = list(pvObject['attribute']) if 'attribute' in pvObject else []
            pvObject['attribute'] = existing + [new_attr]
            self.nFramesProcessed += 1

        except Exception as e:
            self.nFrameErrors += 1
            try:
                self.logger.exception("FeatureExtraction frame error", exc_info=e)
            except Exception:
                pass

        self.processingTime += time.time() - t0
        self.updateOutputChannel(pvObject)
        return pvObject

    # ------------------------------------------------------------------
    # pvapy statistics interface
    # ------------------------------------------------------------------

    def getStats(self):
        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFrameErrors':     self.nFrameErrors,
            'processingTime':   self.processingTime,
        }

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.processingTime = 0.0
