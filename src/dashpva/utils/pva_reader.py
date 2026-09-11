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

import threading
import time
from collections import deque

import bitshuffle
import blosc2
import lz4.block
import numpy as np
import pvaccess as pva
from epics import caget, camonitor, camonitor_clear
from PyQt5.QtCore import QObject, pyqtSignal

import dashpva.settings as app_settings
from dashpva.utils.config.hkl import semantic_hkl_channels
from dashpva.utils.frame_delivery import FramePacket, LatestFrame


class PVAReader(QObject):
    # Signals
    # signal_image_updated = pyqtSignal(np.ndarray)
    # signal_attributes_updated = pyqtSignal(dict)
    # signal_roi_updated = pyqtSignal(dict)
    # signal_rsm_updated = pyqtSignal(dict)
    # signal_analysis_updated = pyqtSignal(dict)
    reader_scan_complete = pyqtSignal()
    scan_state_changed = pyqtSignal(bool)
    reader_new_frame = pyqtSignal()
    
    def __init__(self,
                 input_channel=None,
                 provider=pva.PVA,
                 viewer_type:str='image',
                 pva_prefix:str=None,
                 delivery_mode: str = 'auto'):
        """
        Initializes the PVA Reader for monitoring connections and handling image data.

        Args:
            input_channel (str): Input channel for the PVA connection.
            provider (protocol): The protocol for the PVA channel.
            pva_prefix (str): Detector prefix used to construct ROI/Stats PV names.
                When provided, takes precedence over app_settings.DETECTOR_PREFIX.
                When omitted, falls back to splitting input_channel on ':'.
        """
        super(PVAReader, self).__init__()
        # Each PVA ScalarType is enumerated in C++ starting 1-10
        # This means we map them as numbers to a numpy datatype which we parse from pva codec parameters
        # Then use this to correctly decompress the image depending on the codec used
        self.NUMPY_DATA_TYPE_MAP = {
            pva.UBYTE   : np.dtype('uint8'),
            pva.BYTE    : np.dtype('int8'),
            pva.USHORT  : np.dtype('uint16'),
            pva.SHORT   : np.dtype('int16'),
            pva.UINT    : np.dtype('uint32'),
            pva.INT     : np.dtype('int32'),
            pva.ULONG   : np.dtype('uint64'),
            pva.LONG    : np.dtype('int64'),
            pva.FLOAT   : np.dtype('float32'),
            pva.DOUBLE  : np.dtype('float64')
        }

        # This also means we can parse the pva codec parameters to show the correct datatype in viewer
        # rather than using default compressed dtype
        self.NTNDA_DATA_TYPE_MAP = {
            pva.UBYTE   : 'ubyteValue',
            pva.BYTE    : 'byteValue',
            pva.USHORT  : 'ushortValue',
            pva.SHORT   : 'shortValue',
            pva.UINT    : 'uintValue',
            pva.INT     : 'intValue',
            pva.ULONG   : 'ulongValue',
            pva.LONG    : 'longValue',
            pva.FLOAT   : 'floatValue',
            pva.DOUBLE  : 'doubleValue',
        }

        self.NTNDA_NUMPY_MAP = {
            'ubyteValue'  : np.dtype('uint8'),
            'byteValue'   : np.dtype('int8'),
            'ushortValue' : np.dtype('uint16'),
            'shortValue'  : np.dtype('int16'),
            'uintValue'   : np.dtype('uint32'),
            'intValue'    : np.dtype('int32'),
            'ulongValue'  : np.dtype('uint64'),
            'longValue'   : np.dtype('int64'),
            'floatValue'  : np.dtype('float32'),
            'doubleValue' : np.dtype('float64')
        }

        self.VIEWER_TYPE_MAP = {
            'image': 'i',
            'analysis': 'a',
            'rsm': 'r' 
        }

        # variables related to monitoring connection
        self.input_channel = input_channel
        self.provider = provider
        self.channel = pva.Channel(self.input_channel, self.provider)
        self._explicit_prefix = pva_prefix
        self.pva_prefix = pva_prefix or self._prefix_from_channel(self.input_channel)

        # variables setup using config
        self.config = {}
        self.rois = {}
        self._roi_names = ['ROI1', 'ROI2', 'ROI3', 'ROI4']
        self._active_roi_pvs = []
        self.stats = {}
        self.CONSUMER_MODE = ''
        self.OUTPUT_FILE_LOCATION = ''
        self.ANALYSIS_IN_CONFIG = False
        self.HKL_IN_CONFIG = False
        self.CACHE_OPTIONS = {}
        self.CACHING_MODE = ''
        self.MAX_CACHE_SIZE = 0
        self.is_caching = False
        self.is_scan_complete = False

        # variables that will store pva data
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.timestamp = None
        self.data_type = None
        self.display_dtype = None
        self.numpy_dtype = None
        self.attributes = []
        self.pv_attributes = {}
        self.metadata_ca = {}  # Store CA metadata PVs
        self.cached_ca: dict = {}  # pv_name -> [values] captured during scan
        self.hkl_values: dict = {}  # HKL pv_name -> latest value, merged into frames

        # variables used for image manipulaiton
        self.pixel_ordering = 'F'
        self.viewer_type = self.VIEWER_TYPE_MAP.get(viewer_type, 'i')
        self.image_is_transposed = False
        
        # variables used for parsing specific attribute data from pv
        self.analysis_index = None
        self.analysis_attributes = {}
        self.rsm_attributes = {}

        # variables used for frame count
        self.last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.processing_errors = 0
        self.processing_error_counts = {}
        self.id_diff = 0

        # Producer/consumer buffering. pvapy's network thread pushes frames into
        # a bounded queue; a dedicated consumer thread drains and processes them.
        # When the consumer falls behind, the queue fills and pvapy drops the
        # newest frame rather than overrunning the monitor thread and crashing
        # the viewer. See start_channel_monitor / _consume_loop.
        self.QUEUE_SIZE = app_settings.PVA_MONITOR_QUEUE_SIZE
        self.MONITOR_REQUEST = app_settings.PVA_MONITOR_REQUEST
        self._queue = None
        self._consumer_thread = None
        self._consuming = False
        self._process_callback = None

        # variables for data caches
        self.caches_needed = False
        self.caches_initialized = False
        self.cached_attributes = None
        self.cached_images = None
        self.cached_qx = None
        self.cached_qy = None
        self.cached_qz = None
        # self._on_scan_complete_callbacks = []

        if delivery_mode not in ('auto', 'preview', 'ordered'):
            raise ValueError('delivery_mode must be auto, preview, or ordered')
        self.delivery_mode = delivery_mode
        self._preview_delivery = False
        self._preview = LatestFrame()
        self._frame_sequence = 0
        self._preview_policy = dict(app_settings.PREVIEW)
        self.preview_frames_superseded = 0
        self.preview_frames_rejected = 0
        self.last_decode_seconds = 0.0
        self.last_processing_seconds = 0.0
        self.last_dequeue_monotonic = None
        self._configure()

    @staticmethod
    def _prefix_from_channel(channel: str) -> str:
        """Detector prefix from the image channel.

        The image channel is ``<prefix>:Pva1:Image`` and the prefix itself may
        contain colons (e.g. ``1id:Eiger``), so strip the known image suffix
        rather than ``split(':')[0]`` — which would drop everything after the
        first colon and build ROI PV names off a truncated prefix.
        """
        ch = (channel or "").strip()
        low = ch.lower()
        for suffix in (":pva1:image", ":image"):
            if low.endswith(suffix):
                return ch[:len(ch) - len(suffix)]
        return ch.rsplit(":", 1)[0] if ":" in ch else ch

############################# Configuration #############################
    def _configure(self) -> None:
        self.config = app_settings.CONFIG
        self.OUTPUT_FILE_LOCATION = app_settings.OUTPUT_PATH
        self.ANALYSIS_IN_CONFIG = (app_settings.ANALYSIS != {})
        self.HKL_IN_CONFIG = (app_settings.HKL != {})
        self.CONSUMER_MODE = app_settings.CONSUMER_MODE or ''
        self.CACHE_OPTIONS: dict = app_settings.CACHE_OPTIONS

        if not self._explicit_prefix and app_settings.DETECTOR_PREFIX:
            self.pva_prefix = app_settings.DETECTOR_PREFIX

        self.set_cache_options()
        if self.caches_needed != self.caches_initialized:
            self.init_caches()

        if self.ANALYSIS_IN_CONFIG and self.CONSUMER_MODE == "continuous":
            self.analysis_cache_dict = {"Position": {},
                                        "Intensity": {},
                                        "ComX": {},
                                        "ComY": {}}
    def set_cache_options(self) -> None:
        self.CACHING_MODE = app_settings.CACHING_MODE or ''
        if self.CACHING_MODE:
            self.caches_needed = True
            if self.CACHING_MODE == 'alignment':
                self.MAX_CACHE_SIZE = app_settings.ALIGNMENT_MAX_CACHE_SIZE or 100
            elif self.CACHING_MODE == 'scan':
                self.FLAG_PV = app_settings.SCAN_FLAG_PV or ''
                self.START_SCAN = app_settings.SCAN_START_SCAN if app_settings.SCAN_START_SCAN is not None else True
                self.STOP_SCAN = app_settings.SCAN_STOP_SCAN if app_settings.SCAN_STOP_SCAN is not None else False
                self.MAX_CACHE_SIZE = app_settings.SCAN_MAX_CACHE_SIZE or 100
            elif self.CACHING_MODE == 'bin':
                self.BIN_COUNT = app_settings.BIN_COUNT or 10
                self.BIN_SIZE = app_settings.BIN_SIZE or 16

    def init_caches(self) -> None:
        if self.CACHING_MODE == 'alignment' or self.CACHING_MODE == 'scan':
            self.cached_images = deque(maxlen=self.MAX_CACHE_SIZE)
            self.cached_attributes = deque(maxlen=self.MAX_CACHE_SIZE)
            if self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                self.cached_qx = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cached_qy = deque(maxlen=self.MAX_CACHE_SIZE)
                self.cached_qz = deque(maxlen=self.MAX_CACHE_SIZE)
        elif self.CACHING_MODE == 'bin':
            # TODO: when creating the h5 file, have one entry called data that is the average of each bin
            # and then an entry for each bin that lines up with the attributes and rsm attributes
            self.cached_images = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            self.cached_attributes = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
            if self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                self.cached_qx = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cached_qy = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]
                self.cached_qz = [deque(maxlen=self.BIN_SIZE) for _ in range(self.BIN_COUNT)]               
        self.caches_initialized = True

#################### Class and PVA Channel Callbacks ########################
    # def add_on_scan_complete_callback(self, callback_func):
    #     if callable(callback_func):
    #         self._on_scan_complete_callbacks.append(callback_func)
    def pva_callbackSuccess(self, pv, *, stream_epoch=None) -> None:
        """
        Callback for handling monitored PVA changes.

        Args:
            pv (PvObject): The PVA object received by the channel monitor.
        """
        started = time.monotonic()
        epoch = self._preview.epoch if stream_epoch is None else stream_epoch
        try:
            self.frames_received += 1
            self.pva_object = pv
            self.rsm_attributes = {}
            self.analysis_attributes = {}
            self.analysis_index = None
            self.attributes = list(pv["attribute"]) if "attribute" in pv else []

            # parse data required to manipulate pv image
            self.parse_image_data_type(pv)
            self.shape = self.parse_img_shape(pv)
            decode_started = time.monotonic()
            self.image = self.pva_to_image(pv)
            self.last_decode_seconds = time.monotonic() - decode_started

            # update with latest pv metadata
            frame_attributes = self.parse_attributes(pv)
            self.pv_attributes = frame_attributes.copy()

            # Preserve legacy CA fallback, but label it separately in the preview packet.
            fallback_channels = []
            for pv_name, pv_value in list(self.hkl_values.items()):
                if pv_value is not None and pv_name not in frame_attributes:
                    fallback_channels.append(pv_name)
                    frame_attributes[pv_name] = pv_value
                    self.pv_attributes[pv_name] = pv_value

            # Check for any roi pvs in metadata
            self.parse_roi_pvs(self.pv_attributes)

            # Check for rsm attributes in metadata
            if (self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']) and 'RSM' in self.pv_attributes:
                self.parse_rsm_attributes(self.pv_attributes)

            if self.ANALYSIS_IN_CONFIG and 'Analysis' in self.pv_attributes:
                self.parse_analysis_attributes(self.pv_attributes)
            
            if self.caches_initialized:
                try:
                    if self.cache_attributes(self.pv_attributes, self.rsm_attributes):
                        self.cache_image(np.ravel(self.image))
                        if self.is_caching:
                            ca_config = self.config.get('METADATA', {}).get('CA', {})
                            for pv_name in ca_config.values():
                                val = self.pv_attributes.get(pv_name)
                                if val is not None:
                                    self.cached_ca.setdefault(pv_name, []).append(val)
                except Exception:
                    import traceback
                    traceback.print_exc()

            self._frame_sequence += 1
            try:
                seconds = frame_attributes.get('timeStamp-secondsPastEpoch')
                nanoseconds = frame_attributes.get('timeStamp-nanoseconds')
                source_timestamp = None
                if seconds is not None and nanoseconds is not None:
                    source_timestamp = float(seconds) + float(nanoseconds) * 1e-9
                packet = FramePacket.capture(
                    max_array_bytes=self._preview_policy['MAX_ARRAY_BYTES'],
                    stream_epoch=epoch,
                    sequence=self._frame_sequence,
                    unique_id=self.last_array_id,
                    source_timestamp=source_timestamp,
                    dequeued_monotonic=self.last_dequeue_monotonic or started,
                    published_monotonic=time.monotonic(),
                    image=self.image,
                    shape=tuple(self.shape),
                    pixel_ordering=self.pixel_ordering,
                    attributes={key: value for key, value in frame_attributes.items() if key != 'RSM'},
                    rsm_attributes=self.rsm_attributes,
                    fallback_channels=tuple(fallback_channels),
                    geometry_revision=None,
                )
            except ValueError as exc:
                self.preview_frames_rejected += 1
                self._count_processing_error(exc)
            else:
                if self._preview.publish(packet):
                    self.reader_new_frame.emit()

            if self.is_scan_complete and not self.is_caching:
                self.is_scan_complete = False
                self.reader_scan_complete.emit()

        except Exception as exc:
            self.processing_errors += 1
            self._count_processing_error(exc)
            self.image = None
            self.pv_attributes = {}
            self.rsm_attributes = {}
            self.analysis_attributes = {}
            self.analysis_index = None
            import traceback
            traceback.print_exc()
        finally:
            self.last_processing_seconds = time.monotonic() - started

    @property
    def latest_frame(self):
        return self._preview.peek()

    def take_latest_frame(self):
        return self._preview.take()

    def performance_snapshot(self):
        frame = self.latest_frame
        queue = self._queue
        return {
            'frame_identity': frame.identity if frame is not None else None,
            'frames_processed_attempted': self.frames_received,
            'observed_id_gaps_after_selection': self.frames_missed,
            'processing_errors': self.processing_errors,
            'processing_error_counts': dict(self.processing_error_counts),
            'preview_frames_superseded_before_decode': self.preview_frames_superseded,
            'preview_frames_superseded_before_gui': self._preview.superseded,
            'preview_frames_rejected': self.preview_frames_rejected,
            'last_decode_seconds': self.last_decode_seconds,
            'last_processing_seconds': self.last_processing_seconds,
            'client_queue_frames': len(queue) if queue is not None else 0,
            'client_queue_counters': dict(queue.getCounters()) if queue is not None else {},
        }

    def _count_processing_error(self, error: Exception) -> None:
        name = type(error).__name__
        self.processing_error_counts[name] = self.processing_error_counts.get(name, 0) + 1

    def roi_backup_callback(self, pvname, value, **kwargs) -> None:
        # PV format: {pva_prefix}:{roi}:{dimension}
        roi_key, pv_key = pvname.split(':')[-2:]
        self.rois.setdefault(roi_key, {}).update({pv_key: value})
    
    def metadata_ca_callback(self, pvname, value, **kwargs) -> None:
        """
        Callback for CA metadata PV updates.
        Stores the value in self.metadata_ca and also updates pv_attributes.
        """
        self.metadata_ca[pvname] = value
        # Also update pv_attributes so it's available in the same way as PVA attributes
        self.pv_attributes[pvname] = value
        
########################### PVA PARSING ##################################
    def locate_analysis_index(self) -> int|None:
        """
        Locates the index of the analysis attribute in the PVA attributes.

        Returns:
            int: The index of the analysis attribute or None if not found.
        """
        return next(
            (index for index, attribute in enumerate(self.attributes)
             if attribute['name'] == 'Analysis'),
            None,
        )

    def parse_image_data_type(self, pva_object) -> None:
        """
        Parses the PVA Object to determine the incoming data type.
        """
        if pva_object is not None:
            try:
                self.data_type = list(pva_object['value'][0].keys())[0]
                self.display_dtype = self.data_type if pva_object['codec']['name'] == '' else self.NTNDA_DATA_TYPE_MAP.get(pva_object['codec']['parameters'][0]['value'])
                self.numpy_dtype = self.NTNDA_NUMPY_MAP.get(self.display_dtype, None)
            except Exception:
                self.display_dtype = "could not detect"

    def parse_img_shape(self, pva_object) -> tuple:
        if 'dimension' in pva_object:
            return tuple([dim['size'] for dim in pva_object['dimension']])

    def parse_attributes(self, pva_object) -> dict:
        pv_attributes = {}
        if pva_object is not None and 'attribute' in pva_object:
            pv_attributes['timeStamp-secondsPastEpoch'] = pva_object['timeStamp']['secondsPastEpoch']
            pv_attributes['timeStamp-nanoseconds'] = pva_object['timeStamp']['nanoseconds']
            attributes = pva_object['attribute']
            for attr in attributes:
                name = attr['name']
                value = attr['value'][0].get('value', None)
                if value is not None:
                    pv_attributes[name] = value
            return pv_attributes
        else:
            return {}

    def parse_analysis_attributes(self, pv_attributes: dict) -> None:
        analysis = pv_attributes['Analysis']
        if not isinstance(analysis, dict):
            raise ValueError("Analysis must contain a structured result")
        fields = ['Intensity', 'ComX', 'ComY']
        if self.CONSUMER_MODE == 'continuous':
            fields += ['Axis1', 'Axis2']
        values = {name: np.asarray(analysis[name], dtype=np.float64) for name in fields}
        if self.CONSUMER_MODE == 'continuous':
            if any(value.ndim != 0 or not np.isfinite(value) for value in values.values()):
                raise ValueError("Continuous Analysis requires finite scalar results and axes")
            position = (float(values['Axis1']), float(values['Axis2']))
            for name in ('Intensity', 'ComX', 'ComY'):
                cache = self.analysis_cache_dict[name]
                cache[position] = cache.get(position, 0.0) + float(values[name])
            self.analysis_cache_dict['Position'][position] = position
        elif len({value.shape for value in values.values()}) != 1:
            raise ValueError("Analysis result arrays must have matching shapes")
        self.analysis_attributes = analysis
        self.analysis_index = self.locate_analysis_index()

    def parse_rsm_attributes(self, pv_attributes: dict) -> None:
        rsm_attributes: dict = pv_attributes['RSM']
        codec = rsm_attributes['codec'].get('name', '')
        if  codec !=  '':
            dtype = self.NUMPY_DATA_TYPE_MAP.get(rsm_attributes['codec']['parameters'])
            self.rsm_attributes = {'qx' : self.decompress_array(compressed_array=rsm_attributes['qx']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qx']['uncompressedSize'],
                                                                dtype=dtype),
                                   'qy' : self.decompress_array(compressed_array=rsm_attributes['qy']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qy']['uncompressedSize'],
                                                                dtype=dtype),
                                   'qz' : self.decompress_array(compressed_array=rsm_attributes['qz']['value'], 
                                                                codec=codec, 
                                                                uncompressed_size=rsm_attributes['qz']['uncompressedSize'],
                                                                dtype=dtype)}
        else:
            self.rsm_attributes = {'qx' : rsm_attributes['qx']['value'], 
                                   'qy' : rsm_attributes['qy']['value'],
                                   'qz' : rsm_attributes['qz']['value']}
        if any(np.asarray(values).size != self.image.size for values in self.rsm_attributes.values()):
            self.rsm_attributes = {}
            raise ValueError("RSM coordinate arrays must match the detector pixel count")

    def parse_roi_pvs(self, pv_attributes: dict) -> None:
        """Parse PVA attributes to extract ROI-specific PVs.

        Same all-or-nothing rule as ``start_roi_backup_monitor``: if any of
        the four corners is missing for an ROI, the whole ROI is skipped so
        the renderer never gets a partial dict.
        """
        dims = ['MinX', 'MinY', 'SizeX', 'SizeY']
        for roi in self._roi_names:
            collected = {}
            for dimension in dims:
                pv_value = pv_attributes.get(f'{self.pva_prefix}:{roi}:{dimension}')
                if pv_value is None:
                    collected = None
                    break
                collected[dimension] = pv_value
            if collected is not None:
                self.rois[roi] = collected
            
    def pva_to_image(self, pva_object) -> np.ndarray:
        """
        Converts the PVA Object to an image array and determines if a frame was missed.
        Handles bslz4 and lz4 compressed image data.

        image is of type np.ndarray
        """
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("Image dimensions must be nonempty and positive")
        if pva_object['codec']['name']:
            image = self.decompress_array(
                compressed_array=pva_object['value'][0][self.data_type],
                codec=pva_object['codec']['name'],
                uncompressed_size=pva_object['uncompressedSize'],
                dtype=self.NUMPY_DATA_TYPE_MAP.get(pva_object['codec']['parameters'][0]['value']),
            )
        else:
            image = pva_object['value'][0][self.data_type]
        image = np.asarray(image).reshape(self.shape, order=self.pixel_ordering)
        current_array_id = pva_object['uniqueId']
        if self.last_array_id is not None:
            self.frames_missed += max(0, current_array_id - self.last_array_id - 1)
        self.last_array_id = current_array_id
        self.id_diff = 0
        return image.T if self.image_is_transposed else image

    def decompress_array(self, compressed_array: np.ndarray, codec: str, uncompressed_size: int, dtype: np.dtype) -> np.ndarray: 
        if dtype is None:
            raise ValueError('Unknown compressed array dtype')
        # Handle LZ4 compressed data
        if codec == 'lz4':
            decompressed_bytes = lz4.block.decompress(compressed_array, uncompressed_size=uncompressed_size)
            # Convert bytes to numpy array with correct dtype
            return np.frombuffer(decompressed_bytes, dtype=dtype) # dtype makes sure we use the correct
        # Handle BSLZ4 compressed data
        elif codec == 'bslz4':
            # uncompressed size has to be divided by the number of bytes needed to store the desired output dtype
            uncompressed_shape = (uncompressed_size // dtype.itemsize,)
            # Decompress numpy array to correct datatype
            return bitshuffle.decompress_lz4(compressed_array, uncompressed_shape, dtype)
        # handle BLOSC compressed data 
        elif codec == 'blosc':
            decompressed_bytes = blosc2.decompress(compressed_array)
            return np.frombuffer(decompressed_bytes, dtype=dtype)

        raise ValueError(f"Unsupported array codec: {codec}")

################################## Caching ####################################
    def cache_attributes(self, pv_attributes=None, rsm_attributes=None, analysis_attributes=None) -> bool:
        """Returns True if this frame was cached (caller should also cache the image)."""
        if self.CACHING_MODE == 'alignment':
            self.cached_attributes.append(pv_attributes)
            if rsm_attributes:
                self.cached_qx.append(rsm_attributes['qx'])
                self.cached_qy.append(rsm_attributes['qy'])
                self.cached_qz.append(rsm_attributes['qz'])
            return True
        elif self.CACHING_MODE == 'scan':
            if not self.is_caching:
                return False
            if not rsm_attributes and self.viewer_type == self.VIEWER_TYPE_MAP['rsm']:
                return False
            self.cached_attributes.append(pv_attributes)
            if rsm_attributes:
                self.cached_qx.append(rsm_attributes['qx'])
                self.cached_qy.append(rsm_attributes['qy'])
                self.cached_qz.append(rsm_attributes['qz'])
            return True
        elif self.CACHING_MODE == 'bin':
            bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
            self.cached_attributes[bin_index].append(pv_attributes)
            return True
        return False

    def cache_image(self, image) -> None:
        if self.CACHING_MODE == 'alignment':
            self.cached_images.append(image)
            return
        elif self.CACHING_MODE == 'scan':
            if self.is_caching:
                self.cached_images.append(image)
                return
        elif self.CACHING_MODE == 'bin':
            if self.viewer_type == 'i':
                bin_index = (self.frames_received + self.frames_missed - 1) % self.BIN_COUNT
                self.cached_images[bin_index].append(image)
                return   
            
    def reset_caches(self) -> None:
        for name in ('cached_images', 'cached_attributes', 'cached_qx', 'cached_qy', 'cached_qz'):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()

########################### Start and Stop Channel Monitors ##########################    
    def _flag_pv_ca_callback(self, pvname, value, **kwargs) -> None:
        """CA monitor callback for FLAG_PV — detects scan stop even when PVA frames stop arriving."""
        print(f'[DEBUG] CA flag callback: {pvname}={value!r}  is_caching={self.is_caching}')
        if value == self.STOP_SCAN and self.is_caching:
            self.is_caching = False
            self.scan_state_changed.emit(False)
            self.reader_scan_complete.emit()
            print('[DEBUG] CA flag: Scan STOPPED — emitting reader_scan_complete')
        elif value == self.START_SCAN and not self.is_caching:
            self.is_caching = True
            self.is_scan_complete = False
            self.scan_state_changed.emit(True)
            self.cached_ca = {}
            print('[DEBUG] CA flag: Scan STARTED')

    def start_channel_monitor(self, callback=None) -> None:
        """
        Starts a queueing monitor on the PVA channel (producer/consumer).

        pvapy's network thread pushes each frame into a bounded queue; a
        dedicated consumer thread drains the queue and runs the per-frame
        processing, so receiving and processing no longer share one thread.

        Args:
            callback (function, optional): A custom per-frame processor.
                                           If None, defaults to self.pva_callbackSuccess.

        The scan FLAG_PV monitor is now set up by ``start_scan_monitor`` from
        the background PV-pollers thread, so a dead/slow FLAG_PV can't stall
        the GUI thread on Start Live View. See area_det_viewer._connect_pv_pollers.
        """
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            raise RuntimeError('The previous reader worker is still running')
        if self.delivery_mode == 'preview' and (self.CACHING_MODE or callback is not None):
            raise ValueError('Preview delivery cannot feed scientific caches or a custom processor')
        self._preview_delivery = self.delivery_mode != 'ordered' and not self.CACHING_MODE and callback is None
        self._preview_policy = dict(app_settings.PREVIEW)
        epoch = self._preview.reset()
        self._frame_sequence = 0
        self.last_array_id = None
        self.last_dequeue_monotonic = None
        self._monitor_caching_mode = self.CACHING_MODE
        self._process_callback = callback if callback is not None else (
            lambda pv: self.pva_callbackSuccess(pv, stream_epoch=epoch)
        )
        queue_size = self._preview_policy['QUEUE_FRAMES'] if self._preview_delivery else self.QUEUE_SIZE
        request = f'field() record[queueSize={queue_size}]' if self._preview_delivery else self.MONITOR_REQUEST
        self._queue = pva.PvObjectQueue(queue_size)
        self._consuming = True
        self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._consumer_thread.start()
        # qMonitor starts the monitor itself — no separate startMonitor() call.
        try:
            self.channel.qMonitor(self._queue, request)
        except Exception:
            self.stop_channel_monitor()
            raise

    def _consume_loop(self) -> None:
        """Drain the monitor queue and process one frame at a time.

        Runs at the machine's real processing speed: finish a frame, grab the
        next. Blocks briefly when the queue is empty so stop is responsive, and
        never lets a single bad frame kill the loop.
        """
        q = self._queue
        while self._consuming:
            if self.CACHING_MODE != self._monitor_caching_mode:
                self._consuming = False
                self._preview.reset()
                self.channel.stopMonitor()
                return
            try:
                pv = q.get()
            except pva.QueueEmpty:
                try:
                    q.waitForPut(0.5)
                except Exception:
                    pass
                continue
            if self._preview_delivery:
                for _ in range(min(len(q), self._preview_policy['QUEUE_FRAMES'])):
                    try:
                        pv = q.get()
                    except pva.QueueEmpty:
                        break
                    self.preview_frames_superseded += 1
            if not self._consuming:
                break
            try:
                self.last_dequeue_monotonic = time.monotonic()
                self._process_callback(pv)
            except Exception:
                import traceback
                traceback.print_exc()

    def start_scan_monitor(self) -> None:
        """Initial caget + CA monitor for the scan FLAG_PV. No-op outside scan mode.

        Designed to be called from a background thread (the PV pollers sweep).
        The caget has a 0.15s timeout so an unreachable FLAG_PV can't block
        more than a few hundred ms.
        """
        if self.CACHING_MODE != 'scan' or not self.FLAG_PV:
            return
        try:
            initial = caget(self.FLAG_PV, timeout=0.15)
        except Exception:
            initial = None
        if initial is not None:
            self.is_scan_complete = not bool(initial)
        camonitor(pvname=self.FLAG_PV, callback=self._flag_pv_ca_callback)

    def stop_channel_monitor(self) -> None:
        """
        Stops the queueing monitor, drains-loop consumer, and CA callbacks.
        """
        self._consuming = False
        self._preview.reset()
        try:
            self.channel.stopMonitor()
        except Exception:
            pass
        if self._queue is not None:
            try:
                self._queue.cancelWaitForPut()
            except Exception:
                pass
            self._queue = None
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=2.0)
            if self._consumer_thread.is_alive():
                raise RuntimeError("Reader worker is still stopping; retry before restarting")
            self._consumer_thread = None
        if self.CACHING_MODE == 'scan' and self.FLAG_PV:
            try:
                camonitor_clear(self.FLAG_PV)
            except Exception:
                pass
        for pv_name in self.hkl_values:
            try:
                camonitor_clear(pv_name)
            except Exception:
                pass
        for pv_name in list(self.metadata_ca):
            try:
                camonitor_clear(pv_name)
            except Exception:
                pass
        self._clear_roi_backup_monitor()
        self.reset_caches()
        if hasattr(self, 'cached_ca'):
            self.cached_ca.clear()
        self._process_callback = None
        self.pva_object = None
        self.image = None
        self.rsm_attributes = None
        self.pv_attributes = None

    def _clear_roi_backup_monitor(self) -> None:
        """Clear the CA camonitors ``start_roi_backup_monitor`` actually started.

        Uses the recorded PV names rather than re-deriving them from
        ``self.rois``/``self.pva_prefix`` — reconstruction drifts out of sync
        whenever the subscription naming convention changes.
        """
        for pv_name in self._active_roi_pvs:
            try:
                camonitor_clear(pv_name)
            except Exception:
                pass
        self._active_roi_pvs = []

    def start_roi_backup_monitor(self) -> None:
        """Connect to ROI PVs with a tight per-PV timeout.

        Each ROI requires all four corners (MinX/MinY/SizeX/SizeY) to be useful.
        Collect into a temp dict and only commit + start camonitors when all
        four are present — partial ROIs (e.g. MinX/MinY succeeded but SizeX
        failed) would otherwise render as a 0×0 rectangle at the origin and
        leave dangling CA monitors. Called from a background thread (see
        ``DiffractionImageWindow._connect_pv_pollers``).
        """
        dims = ['MinX', 'MinY', 'SizeX', 'SizeY']
        for roi in self._roi_names:
            collected = {}
            ok = True
            for dimension in dims:
                pv_key = f'{self.pva_prefix}:{roi}:{dimension}'
                try:
                    pv_value = caget(pv_key, timeout=0.15)
                except Exception:
                    ok = False
                    break
                if pv_value is None:
                    ok = False
                    break
                collected[dimension] = pv_value
            if not ok:
                continue
            self.rois[roi] = collected
            for dimension in dims:
                pv_name = f'{self.pva_prefix}:{roi}:{dimension}'
                camonitor(pvname=pv_name, callback=self.roi_backup_callback)
                self._active_roi_pvs.append(pv_name)

    def start_metadata_ca_monitor(self) -> None:
        """
        Starts monitoring CA metadata PVs from the [METADATA.CA] section.
        If a PV fails to read, it's skipped. Values are stored in self.metadata_ca
        and also added to self.pv_attributes for consistency.

        Runs on the background poller thread (see
        ``DiffractionImageWindow._connect_pv_pollers``). Timeout is short so a
        single dead PV doesn't add seconds to live-view startup.
        """
        metadata_config = self.config.get('METADATA', {})
        if not metadata_config:
            return

        ca_config = metadata_config.get('CA', {})
        if not ca_config:
            return

        for config_key, pv_name in ca_config.items():
            try:
                # 0.15 s is enough for a healthy network PV; dead ones get
                # skipped quickly so the poller isn't dominated by timeouts.
                pv_value = caget(pv_name, timeout=0.15)
                if pv_value is not None:
                    self.metadata_ca[pv_name] = pv_value
                    # Also add to pv_attributes so it's available like PVA attributes
                    self.pv_attributes[pv_name] = pv_value
                    # Start monitoring for updates
                    camonitor(pvname=pv_name, callback=self.metadata_ca_callback)
            except Exception:
                # Silently skip failed PVs to avoid spam
                pass

    def start_hkl_ca_monitor(self) -> None:
        """Caget + camonitor the [HKL] PVs so scan saves don't depend on the associator.

        The scan writer looks up each HKL value by raw PV name in the frame
        attributes; when the metadata associator doesn't attach them (static
        motor / stale timestamp) the HKL groups save empty. Capturing them here
        and merging in pva_callbackSuccess fills that gap. The 0.15s timeout
        bounds only the value fetch, not the connect, so an unreachable PV still
        waits pvapy's ~5s connection timeout; runs on the reader thread (last
        after the channel/scan monitors) so it never blocks the GUI.
        """
        if not self.HKL_IN_CONFIG:
            return
        hkl_config = self.config.get('HKL', {})
        for pv_name in semantic_hkl_channels(hkl_config):
            if pv_name in self.hkl_values:
                continue
            try:
                pv_value = caget(pv_name, timeout=0.15)
                if pv_value is not None:
                    self.hkl_values[pv_name] = pv_value
                camonitor(pvname=pv_name, callback=self.hkl_ca_callback)
            except Exception:
                pass

    def hkl_ca_callback(self, pvname, value, **kwargs) -> None:
        """Store the latest value for an HKL PV; merged into each frame."""
        self.hkl_values[pvname] = value

    ################################# Getters #################################
    def get_cached_images(self) -> list[np.ndarray]:
        return list(self.cached_images)
    
    def get_cached_attributes(self) -> list[dict]:
        return list(self.cached_attributes)
    
    def get_cached_rsm(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        if len(self.cached_qx) == len(self.cached_qy) == len(self.cached_qz):
            return list(self.cached_qx), list(self.cached_qy), list(self.cached_qz)
        else:
            raise ValueError("[PVA Reader] Cached qx, qy, and qz must have the same length.")

    def get_all_caches(self, clear_caches: bool=False) -> dict:
        """
        Returns all cached data.

        Args:
            clear_caches (bool): Whether to clear the caches after returning the data.
        """
        images =  self.get_cached_images()
        attributes = self.get_cached_attributes()
        
        # Only get RSM data if HKL is configured or viewer is RSM type
        if (self.HKL_IN_CONFIG or self.viewer_type == self.VIEWER_TYPE_MAP['rsm']) and self.viewer_type != 'i':
            rsm = self.get_cached_rsm()
            # Check lengths including RSM data
            if len(images) == len(attributes) == len(rsm[0]) == len(rsm[1]) == len(rsm[2]):
                data = {
                        'images': images,
                        'attributes': attributes,
                        'rsm': rsm,
                        'cached_ca': dict(self.cached_ca),
                        }
            else:
                raise ValueError("[PVA Reader] Cached data must have the same length.")
        else:
            # For image viewer type or when no HKL config, only check images and attributes
            rsm = ([], [], [])  # Empty RSM data
            if len(images) == len(attributes):
                data = {
                        'images': images,
                        'attributes': attributes,
                        'rsm': rsm,
                        'cached_ca': dict(self.cached_ca),
                        }
            else:
                raise ValueError("[PVA Reader] Cached data must have the same length.")
        
        
        if clear_caches:
            self.reset_caches()

        return data
    
    def get_output_file_location(self) -> dict:
        fp_pv_name = app_settings.FILE_PATH_PV or ''
        fn_pv_name = app_settings.FILE_NAME_PV or ''

        file_path_val = ''
        file_name_val = ''

        # Always caget the live PV values — the save location can change between
        # scan start and scan end, so cached frame attributes may be stale.
        if fp_pv_name:
            try:
                val = caget(fp_pv_name, timeout=1.0)
                if val is not None:
                    file_path_val = str(val).strip()
            except Exception:
                pass
        if fn_pv_name:
            try:
                val = caget(fn_pv_name, timeout=1.0)
                if val is not None:
                    file_name_val = str(val).strip()
            except Exception:
                pass

        if file_path_val and file_name_val:
            return {'FilePath': file_path_val, 'FileName': file_name_val}
        elif file_path_val:
            return {'FilePath': file_path_val}
        else:
            return {'FilePath': str(self.OUTPUT_FILE_LOCATION).strip()}
       
    
    def get_config_settings(self) -> dict:
        config_settings = {'OUTPUT_FILE_CONFIG' : self.get_output_file_location(),
                        'ANALYSIS_IN_CONFIG' : self.ANALYSIS_IN_CONFIG,
                        'HKL_IN_CONFIG' : self.HKL_IN_CONFIG,
                        'CACHE_OPTIONS' : self.CACHE_OPTIONS,
                        'caches_initialized' : self.caches_initialized}
        
        return config_settings
    
    def get_frames_missed(self) -> int:
        """
        Returns the number of frames missed.

        Returns:
            int: The number of missed frames.
        """
        return self.frames_missed

    def get_latest_image(self) -> np.ndarray:
        """
        Returns the current PVA image.

        Returns:
            numpy.ndarray: The current image array.
        """
        return self.image
    
    def get_latest_attributes(self) -> list[dict]:
        """
        Returns the attributes of the current PVA object.

        Returns:
            list: The attributes of the current PVA object.
        """
        return self.attributes

    def get_shape(self) -> tuple[int]:
        return self.shape
