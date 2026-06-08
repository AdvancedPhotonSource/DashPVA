import time

import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.timeUtility import TimeUtility

from dashpva.utils.blob_detector import BlobDetector
from dashpva.utils.log_manager import LogMixin
from dashpva.utils.sort import KalmanBoxTracker, Sort


class HpcBlobTrackingProcessor(AdImageProcessor, LogMixin):
    """
    HPC consumer that runs real-time blob detection (SimpleBlobDetector) and
    SORT multi-object tracking on every PVA frame, appending results as
    NdAttributes on the pvObject so downstream consumers and the HDF5 writer
    can persist them.

    Output attributes added to each frame
    --------------------------------------
    BlobDetections : flat float64 array, length N*5 — rows are [x1,y1,x2,y2,score]
    BlobDetCount   : int scalar — number of detections in this frame
    BlobTracks     : flat float64 array, length M*5 — rows are [x1,y1,x2,y2,track_id]
    BlobTrackCount : int scalar — number of active tracks

    configDict keys
    ---------------
    Blob detector (map directly to SimpleBlobDetector_Params field names):
      minThreshold, maxThreshold, filterByArea, minArea, maxArea,
      filterByCircularity, minCircularity, maxCircularity,
      filterByConvexity, minConvexity, maxConvexity,
      filterByInertia, minInertiaRatio, maxInertiaRatio
    SORT tracker:
      sort_max_age, sort_min_hits, sort_iou_threshold
    """

    BLOB_PARAM_KEYS = {
        'minThreshold', 'maxThreshold',
        'filterByArea', 'minArea', 'maxArea',
        'filterByCircularity', 'minCircularity', 'maxCircularity',
        'filterByConvexity', 'minConvexity', 'maxConvexity',
        'filterByInertia', 'minInertiaRatio', 'maxInertiaRatio',
    }

    # Reverse mapping: PVA codec enum → numpy dtype (same as HpcRsmProcessor)
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
        super(HpcBlobTrackingProcessor, self).__init__(configDict)
        try:
            self.set_log_manager(viewer_name="HpcBlobTrackingProcessor")
        except Exception:
            pass

        self.blob_detector = BlobDetector()
        self._sort_params = dict(max_age=5, min_hits=3, iou_threshold=0.3)
        self.tracker = Sort(**self._sort_params)

        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.processingTime = 0.0

        self.configure(configDict)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, configDict):
        if not configDict:
            return
        blob_params = {k: v for k, v in configDict.items() if k in self.BLOB_PARAM_KEYS}
        if blob_params:
            self.blob_detector.update_params(**blob_params)

        changed = False
        for key, attr in (('sort_max_age', 'max_age'),
                           ('sort_min_hits', 'min_hits'),
                           ('sort_iou_threshold', 'iou_threshold')):
            if key in configDict:
                self._sort_params[attr] = configDict[key]
                changed = True
        if changed:
            KalmanBoxTracker.count = 0
            self.tracker = Sort(**self._sort_params)

    # ------------------------------------------------------------------
    # Image decompression (mirrors HpcRsmProcessor.decompress_image)
    # ------------------------------------------------------------------

    def _decompress_image(self, pvObject) -> np.ndarray:
        """Return flat 1-D numpy array of raw pixel values."""
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
    # Main processing loop
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

            detections = self.blob_detector.detect(image)   # (N, 5)
            tracks     = self.tracker.update(detections)     # (M, 5)

            det_pv = pva.PvScalarArray(pva.DOUBLE)
            det_pv.set(detections.flatten().tolist())

            trk_pv = pva.PvScalarArray(pva.DOUBLE)
            trk_pv.set(tracks.flatten().tolist())

            det_count_pv = pva.PvObject({'value': pva.INT})
            det_count_pv['value'] = int(len(detections))

            trk_count_pv = pva.PvObject({'value': pva.INT})
            trk_count_pv['value'] = int(len(tracks))

            new_attrs = [
                {'name': 'BlobDetections',  'value': det_pv},
                {'name': 'BlobDetCount',    'value': det_count_pv},
                {'name': 'BlobTracks',      'value': trk_pv},
                {'name': 'BlobTrackCount',  'value': trk_count_pv},
            ]

            existing = list(pvObject['attribute']) if 'attribute' in pvObject else []
            pvObject['attribute'] = existing + new_attrs

            self.nFramesProcessed += 1

        except Exception as e:
            self.nFrameErrors += 1
            try:
                if hasattr(self, 'logger'):
                    self.logger.exception("BlobTracking frame error", exc_info=e)
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
