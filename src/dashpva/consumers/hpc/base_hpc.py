import bitshuffle
import blosc2
import lz4.block
import numpy as np
import pvaccess as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits

from dashpva.utils.log_manager import LogMixin


class BaseHpcProcessor(AdImageProcessor, LogMixin):

    def __init__(self, configDict={}):
        super().__init__(configDict)
        try:
            self.set_log_manager()
        except Exception:
            pass

        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMetadataProcessed = 0
        self.nMetadataDiscarded = 0
        self.processingTime = 0
        self.lastFrameTimestamp = 0

        # numpy dtype → PVA codec enum
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

        # PVA codec enum → numpy dtype
        self.PVA_TO_NUMPY_DTYPE_MAP = {
            pva.UBYTE: np.uint8,
            pva.BYTE: np.int8,
            pva.USHORT: np.uint16,
            pva.SHORT: np.int16,
            pva.UINT: np.uint32,
            pva.INT: np.int32,
            pva.ULONG: np.uint64,
            pva.LONG: np.int64,
            pva.FLOAT: np.float32,
            pva.DOUBLE: np.float64,
        }

        # union field name → numpy dtype (uncompressed PVA payloads)
        self.UNION_FIELD_TO_DTYPE = {
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

    def compress_array(self, array: np.ndarray, codec_name: str) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy array")
        if array.ndim != 1:
            raise ValueError("array must be 1D")
        byte_data = array.tobytes()
        if codec_name == 'lz4':
            compressed = lz4.block.compress(byte_data, store_size=False)
        elif codec_name == 'bslz4':
            compressed = bitshuffle.compress_lz4(array)
        elif codec_name == 'blosc':
            compressed = blosc2.compress(byte_data, typesize=array.dtype.itemsize)
        else:
            raise ValueError(f"Unsupported codec: {codec_name}")
        return np.frombuffer(compressed, dtype=np.uint8)

    def decompress_image(self, pvObject) -> np.ndarray:
        codec_name = pvObject['codec']['name']
        if codec_name == 'lz4':
            u8_pv = pvObject['value'][0]['ubyteValue']
            u8_list = u8_pv.get() if hasattr(u8_pv, 'get') else u8_pv
            comp_bytes = np.asarray(u8_list, dtype=np.uint8).tobytes()
            out_bytes = lz4.block.decompress(comp_bytes, uncompressed_size=pvObject['uncompressedSize'])
            params = pvObject['codec']['parameters']
            enum = params[0]['value'] if (isinstance(params, tuple) and len(params) > 0) else pva.UBYTE
            dtype = self.PVA_TO_NUMPY_DTYPE_MAP.get(enum, np.uint8)
            return np.frombuffer(out_bytes, dtype=dtype)
        union_dict = pvObject['value'][0]
        field_name = next(iter(union_dict))
        pv_arr = union_dict[field_name]
        data_list = pv_arr.get() if hasattr(pv_arr, 'get') else pv_arr
        dtype = self.UNION_FIELD_TO_DTYPE.get(field_name, None)
        return np.asarray(data_list, dtype=dtype) if dtype is not None else np.asarray(data_list)

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMetadataProcessed = 0
        self.nMetadataDiscarded = 0
        self.processingTime = 0

    def getStats(self):
        processedFrameRate = 0
        frameErrorRate = 0
        if self.processingTime > 0:
            processedFrameRate = self.nFramesProcessed / self.processingTime
            frameErrorRate = self.nFrameErrors / self.processingTime
        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFrameErrors': self.nFrameErrors,
            'nMetadataProcessed': self.nMetadataProcessed,
            'nMetadataDiscarded': self.nMetadataDiscarded,
            'processingTime': FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate': FloatWithUnits(processedFrameRate, 'fps'),
            'frameErrorRate': FloatWithUnits(frameErrorRate, 'fps'),
        }

    def getStatsPvaTypes(self):
        return {
            'nFramesProcessed': pva.UINT,
            'nFrameErrors': pva.UINT,
            'nMetadataProcessed': pva.UINT,
            'nMetadataDiscarded': pva.UINT,
            'processingTime': pva.DOUBLE,
            'processedFrameRate': pva.DOUBLE,
            'frameErrorRate': pva.DOUBLE,
        }
