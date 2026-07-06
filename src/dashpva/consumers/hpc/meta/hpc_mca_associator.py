"""Associate SIS3820 MCA channel-access readings with a detector image stream.

Purpose-built metadata associator for the 12-ID-C PILATUS + SIS3820 setup: for
every incoming detector image (PVA NTNDArray, e.g. ``S12-PILATUS1:Pva1:Image``)
it attaches the current readings of N MCA scaler channels (EPICS CA waveforms,
e.g. ``12idc:3820:mca1.VAL`` … ``mca8.VAL``) matched by acquisition timestamp,
and republishes the image with ``mca1``…``mcaN`` added as NDAttributes.

Unlike the generic ``HpcAdMetadataProcessor`` (which reads metadata from the
framework's ``--metadata-channels`` queues), this processor monitors the MCA PVs
itself via Channel Access, so it needs no metadata-channel wiring — just select
it in the workflow "Metadata Associator" tab (processor class
``HpcMcaAssociator``) and set the input/output channels.

Matching policy: *hold-last within a window* — the most recent MCA reading is
attached to each image; a reading whose timestamp is farther than ``mcaWindow``
seconds from the image is counted as stale (still attached unless
``mcaDropStale`` is set), so images are never dropped.

Runs under pvapy's HPC streaming framework:
    python -m pvapy.cli.hpcConsumer \\
        --input-channel S12-PILATUS1:Pva1:Image \\
        --output-channel S12-PILATUS1:Associated \\
        --processor-file .../hpc_mca_associator.py \\
        --processor-class HpcMcaAssociator
"""

import logging
import threading
import time

import lz4.block
import numpy as np
import pvaccess as pva
from epics import PV as EpicsPV
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility

from dashpva.utils.log_manager import LogMixin


class HpcMcaAssociator(AdImageProcessor, LogMixin):

    # Default SIS3820 scaler channels at 12-ID-C.
    DEFAULT_MCA_BASE = '12idc:3820:mca'
    DEFAULT_MCA_COUNT = 8
    DEFAULT_MCA_SUFFIX = '.VAL'
    # First N array elements kept per channel (what ``caget -# 8`` returns).
    DEFAULT_MCA_ELEMENTS = 8
    # Max |image ts - mca ts| (s) to consider a reading fresh.
    DEFAULT_MCA_WINDOW = 0.5
    MIN_COMPRESS_BYTES = 4098

    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)
        try:
            self.set_log_manager(viewer_name="HpcMcaAssociator")
        except Exception:
            pass

        # -- configuration (all overridable via --processor-args) -------------
        self.mcaElements = int(configDict.get('mcaElements', self.DEFAULT_MCA_ELEMENTS))
        self.mcaWindow = float(configDict.get('mcaWindow', self.DEFAULT_MCA_WINDOW))
        self.mcaDropStale = bool(configDict.get('mcaDropStale', False))
        self.compress = bool(configDict.get('compress', True))
        self.startMonitors = bool(configDict.get('startMonitors', True))
        self.processor_id = configDict.get('collectorId', configDict.get('metadataId', ''))

        # PV list: explicit 'mcaPvs' comma string, else built from base+count.
        pvs_arg = configDict.get('mcaPvs')
        if pvs_arg:
            self.mcaPvs = [p.strip() for p in str(pvs_arg).split(',') if p.strip()]
        else:
            base = configDict.get('mcaBase', self.DEFAULT_MCA_BASE)
            count = int(configDict.get('mcaCount', self.DEFAULT_MCA_COUNT))
            suffix = configDict.get('mcaSuffix', self.DEFAULT_MCA_SUFFIX)
            self.mcaPvs = [f'{base}{i}{suffix}' for i in range(1, count + 1)]
        # Full PV name -> short attribute name (e.g. 12idc:3820:mca1.VAL -> mca1)
        self._pv_short = {pv: self._short_name(pv) for pv in self.mcaPvs}

        # -- runtime state ----------------------------------------------------
        self._lock = threading.Lock()
        self._mca_latest = {}          # short name -> (np.ndarray, unix_ts)
        self._pvs = []                 # live epics.PV handles
        self._ca_started = False

        # -- statistics -------------------------------------------------------
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMcaWithin = 0            # readings attached within the window
        self.nMcaStale = 0             # readings older than the window
        self.processingTime = 0
        self.lastFrameTimestamp = 0

        # Type map for the lz4 codec-parameters field (matches HpcAdMetadataProcessor).
        self.CODEC_PARAMETERS_MAP = {
            np.dtype('uint8'): pva.UBYTE, np.dtype('int8'): pva.BYTE,
            np.dtype('uint16'): pva.USHORT, np.dtype('int16'): pva.SHORT,
            np.dtype('uint32'): pva.UINT, np.dtype('int32'): pva.INT,
            np.dtype('uint64'): pva.ULONG, np.dtype('int64'): pva.LONG,
            np.dtype('float32'): pva.FLOAT, np.dtype('float64'): pva.DOUBLE,
        }
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(f'Created HpcMcaAssociator for {self.mcaPvs}')

    @staticmethod
    def _short_name(pv_name: str) -> str:
        """12idc:3820:mca1.VAL -> mca1"""
        return pv_name.split(':')[-1].split('.')[0]

    # ----------------------------------------------------------- CA monitors
    def _ensure_ca(self) -> None:
        """Start CA monitors once. Called from configure() and process() so the
        first one the framework invokes wins; a no-op when startMonitors=False
        (used by unit tests that inject readings directly)."""
        if self._ca_started or not self.startMonitors:
            return
        self._ca_started = True
        for pv_name in self.mcaPvs:
            try:
                pv = EpicsPV(pv_name, form='time', auto_monitor=True,
                             callback=self._mca_cb)
                self._pvs.append(pv)
            except Exception as e:
                self.logger.error(f'Failed to monitor MCA PV {pv_name}: {e}')

    def _mca_cb(self, pvname=None, value=None, timestamp=None, **kw) -> None:
        """CA monitor callback (runs on the pyepics thread)."""
        if value is None:
            return
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        ts = float(timestamp) if timestamp is not None else time.time()
        short = self._pv_short.get(pvname, self._short_name(pvname))
        with self._lock:
            self._mca_latest[short] = (arr, ts)

    def _stop_ca(self) -> None:
        for pv in self._pvs:
            try:
                pv.clear_callbacks()
                pv.disconnect()
            except Exception:
                pass
        self._pvs = []
        self._ca_started = False

    # --------------------------------------------------------------- matching
    def _match_mca(self, frame_ts: float) -> dict:
        """Pure, testable: {short: (values[:N], ts, within_window)} for the
        latest reading of every channel (hold-last-within-window)."""
        n = self.mcaElements
        with self._lock:
            items = list(self._mca_latest.items())
        out = {}
        for short, (vals, ts) in items:
            if vals is None:
                continue
            trunc = vals[:n] if n and n > 0 else vals
            within = abs(frame_ts - ts) <= self.mcaWindow
            out[short] = (trunc, ts, within)
        return out

    # ------------------------------------------------------- framework hooks
    def configure(self, configDict):
        self.logger.debug(f'Configuration update: {configDict}')
        if 'mcaWindow' in configDict:
            self.mcaWindow = float(configDict['mcaWindow'])
        if 'mcaElements' in configDict:
            self.mcaElements = int(configDict['mcaElements'])
        if 'mcaDropStale' in configDict:
            self.mcaDropStale = bool(configDict['mcaDropStale'])
        self._ensure_ca()

    def process(self, pvObject):
        self._ensure_ca()
        t0 = time.time()
        frameId = pvObject['uniqueId']
        if not len(pvObject['dimension']):
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
            return pvObject
        if 'timeStamp' not in pvObject:
            self.logger.error(f'Frame id {frameId} does not have field "timeStamp"')
            self.nFrameErrors += 1
            return pvObject

        frame_ts = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        frameAttributes = pvObject['attribute'] if 'attribute' in pvObject else []

        for short, (vals, ts, within) in self._match_mca(frame_ts).items():
            if within:
                self.nMcaWithin += 1
            else:
                self.nMcaStale += 1
                if self.mcaDropStale:
                    continue
            pv = pva.PvScalarArray(pva.DOUBLE)
            pv.set([float(v) for v in vals])
            frameAttributes.append({'name': short, 'value': pv})

        # Processing-time attributes (mirrors HpcAdMetadataProcessor).
        for name, val in (('procTimeStart', t0), ('procTimeEnd', time.time()),
                          ('procTime', time.time() - t0)):
            attr = pva.PvObject({'value': pva.DOUBLE})
            attr['value'] = val
            frameAttributes.append(
                {'name': f'{name}_{self.__class__.__name__}{self.processor_id}',
                 'value': attr})

        if self.compress:
            self.compress_image(pvObject)
        pvObject['attribute'] = frameAttributes
        self.updateOutputChannel(pvObject)

        self.nFramesProcessed += 1
        self.lastFrameTimestamp = frame_ts
        self.processingTime += (time.time() - t0)
        return pvObject

    def stop(self, *args, **kwargs):
        self._stop_ca()
        try:
            return super().stop(*args, **kwargs)
        except Exception:
            return None

    # --------------------------------------------------------------- image compress
    def compress_image(self, pvObject) -> None:
        """lz4-compress an uncompressed NTNDArray in place (copied convention
        from HpcAdMetadataProcessor.compress_image)."""
        try:
            if pvObject['codec']['name']:
                return
        except Exception:
            return
        union_dict = pvObject['value'][0]
        field_name = next(iter(union_dict))
        pv_arr = union_dict[field_name]
        data_list = pv_arr.get() if hasattr(pv_arr, 'get') else pv_arr

        UNION_FIELD_TO_DTYPE = {
            'ubyteValue': np.uint8, 'byteValue': np.int8,
            'ushortValue': np.uint16, 'shortValue': np.int16,
            'uintValue': np.uint32, 'intValue': np.int32,
            'ulongValue': np.uint64, 'longValue': np.int64,
            'floatValue': np.float32, 'doubleValue': np.float64,
        }
        dtype = UNION_FIELD_TO_DTYPE.get(field_name)
        arr = np.ascontiguousarray(
            np.asarray(data_list, dtype=dtype) if dtype is not None
            else np.asarray(data_list))
        raw = arr.tobytes()
        raw_len = arr.nbytes
        original_enum = self.CODEC_PARAMETERS_MAP.get(arr.dtype)

        codec = 'none'
        if raw_len >= self.MIN_COMPRESS_BYTES:
            comp = lz4.block.compress(raw, store_size=False)
            if len(comp) < raw_len:
                arr_u8 = np.frombuffer(comp, dtype=np.uint8)
                pvObject['value'] = ({'ubyteValue': arr_u8.tolist()},)
                codec = 'lz4'
        pvObject['codec']['name'] = 'lz4' if codec == 'lz4' else ''
        pvObject['codec']['parameters'] = (
            ({'value': int(original_enum)},) if original_enum is not None else ())
        pvObject['uncompressedSize'] = raw_len

    # --------------------------------------------------------------- stats
    def resetStats(self):
        self.nFramesProcessed = 0
        self.nFrameErrors = 0
        self.nMcaWithin = 0
        self.nMcaStale = 0
        self.processingTime = 0

    def getStats(self):
        rate = self.nFramesProcessed / self.processingTime if self.processingTime > 0 else 0
        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFrameErrors': self.nFrameErrors,
            'nMcaWithin': self.nMcaWithin,
            'nMcaStale': self.nMcaStale,
            'processingTime': FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate': FloatWithUnits(rate, 'fps'),
        }

    def getStatsPvaTypes(self):
        return {
            'nFramesProcessed': pva.UINT,
            'nFrameErrors': pva.UINT,
            'nMcaWithin': pva.UINT,
            'nMcaStale': pva.UINT,
            'processingTime': pva.DOUBLE,
            'processedFrameRate': pva.DOUBLE,
        }
