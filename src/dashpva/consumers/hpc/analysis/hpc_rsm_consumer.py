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

import copy
import time

import numpy as np
import pvaccess as pva
from pvaccess import PvObject
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.core.base_analysis_processor import BaseAnalysisProcessor
from dashpva.utils.config.hkl import (
    axis_field_channels,
    get_hkl_section,
    required_rsm_channels,
    section_field_channels,
)
from dashpva.utils.rsm_geometry import (
    RotationAxis,
    RSMGeometry,
    build_hxrd,
    calculate_q,
    detector_model_from_setup,
    detector_setup_from_channels,
)
from dashpva.utils.units import to_eV


class HpcRsmProcessor(BaseAnalysisProcessor):

    def __init__(self, configDict={}):
        super().__init__(configDict)
        try:
            self.set_log_manager(viewer_name="HpcRsmProcessor")
        except Exception:
            pass

        # Config Variables
        self.hkl_config = {}

        # PV attributes
        self.shape : tuple = (0,0)
        self.type_dict = {
            'codec':{
                'name': pva.STRING,
                'parameters': pva.INT},
            'qx': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]},
            'qy': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]},
            'qz': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.DOUBLE]}
            }

        self.type_dict_compressed = {
            'codec':{
                'name': pva.STRING,
                'parameters': pva.INT},
            'qx': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.UBYTE,]},
            'qy': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.UBYTE,]},
            'qz': {
                'compressedSize': pva.LONG,
                'uncompressedSize': pva.LONG,
                'value':[pva.UBYTE,]}
            }

        # HKL parameters
        self.all_attributes = {}
        self.hkl_pv_channels = set()
        self.hkl_attributes = {}
        self.old_attrbutes : dict = None
        self.qx = None
        self.qy = None
        self.qz = None
        self.codec_name = None
        self.codec_parameters = -1
        self.original_dtype = np.dtype('float64')
        self.uncompressed_size = 0
        self.compressed_size_qx = 0
        self.compressed_size_qy = 0
        self.compressed_size_qz = 0

        self.configure(configDict)

    def configure(self, configDict):
        """Configure processor settings and initialize HKL parameters from DB config."""
        self.logger.debug(f'Configuration update: {configDict}')

        from dashpva.utils.config import ConfigSource, resolve_profile_config
        locator = configDict.get('profile_id') or configDict.get('path') or None
        config = resolve_profile_config(ConfigSource(locator).load())

        if not config:
            raise RuntimeError(
                "HpcRsmProcessor: no configuration found — ensure a profile is selected "
                "in the DB or pass 'profile_id' or 'path' in configDict."
            )

        self.config = config
        self.hkl_config = self.config.get('HKL') or {}
        self.hkl_pv_channels = required_rsm_channels(self.hkl_config)

    def parse_hkl_ndattributes(self, pva_object):
        """
        Parse the NDAttributes from the PVA Object into a python dict.
        Store attributes in self.all_attributes for easy reference.
        """
        if pva_object is None:
            return
        # obj_dict : dict = pva_object.get()
        attributes : list = pva_object['attribute']
        hkl_attributes = {}
        for attr in attributes:
            try:
                name = attr['name']
                value = attr['value'][0]['value']
                self.all_attributes[name] = value
                if name in self.hkl_pv_channels:
                    hkl_attributes[name] = value
            except Exception:
                pass
        return hkl_attributes

    def get_sample_and_detector_circles(self, hkl_attr: dict):
        missing = self.hkl_pv_channels.difference(hkl_attr)
        if missing:
            raise ValueError(f"missing required RSM attributes: {sorted(missing)}")

        def _values(role, field):
            return [
                hkl_attr[channel]
                for channel in axis_field_channels(self.hkl_config, role, field)
            ]

        return (
            _values('sample', 'DIRECTION_AXIS'),
            _values('sample', 'POSITION'),
            _values('detector', 'DIRECTION_AXIS'),
            _values('detector', 'POSITION'),
        )

    def get_axis_directions(self, hkl_attr: dict):
        """Get beam / reference / surface-normal direction triplets from hkl_attr.

        PV names come from the active HKL config (which includes any prefix the
        IOC publishes under), not hardcoded — so this works for `xidb:`,
        `6idb:`, or unprefixed schemas without code changes.
        """
        fields = ('AXIS_NUMBER_1', 'AXIS_NUMBER_2', 'AXIS_NUMBER_3')

        def _triplet(section_name):
            return [
                hkl_attr[channel]
                for channel in section_field_channels(
                    self.hkl_config,
                    section_name,
                    fields,
                )
            ]

        primary_beam_directions = _triplet('PRIMARY_BEAM_DIRECTION')
        inplane_beam_direction = _triplet('INPLANE_REFERENCE_DIRECTION')
        sample_surface_normal_direction = _triplet('SAMPLE_SURFACE_NORMAL_DIRECTION')
        return primary_beam_directions, inplane_beam_direction, sample_surface_normal_direction

    def get_ub_matrix(self, hkl_attr: dict):
        ub_matrix_key = get_hkl_section(
            self.hkl_config,
            'SPEC',
            required=True,
        )['UB_MATRIX_VALUE']

        return hkl_attr[ub_matrix_key]

    def get_energy(self, hkl_attr: dict):
        energy_key = get_hkl_section(
            self.hkl_config,
            'SPEC',
            required=True,
        )['ENERGY_VALUE']

        return hkl_attr[energy_key]

    def get_energy_units(self, hkl_attr: dict) -> str:
        """Units the energy channel publishes in.

        Defaults to keV, which is what the blind ``* 1000`` this replaced
        assumed, so a profile that does not declare ENERGY_UNITS keeps its
        current behavior instead of silently changing scale.
        """
        spec = get_hkl_section(self.hkl_config, 'SPEC', required=True)
        channel = spec.get('ENERGY_UNITS')
        if channel and hkl_attr.get(channel):
            return str(hkl_attr[channel])
        canonical = self.config.get('IOC_RSM_PARAMETER', {}) or {}
        return str(canonical.get('ENERGY_UNITS', 'keV'))

    def create_rsm(self, hkl_attr: dict, shape: tuple):
        """Calculate reciprocal space mapping"""
        try:
            # get Sample and Detection Circle positions and directions from hkl attributes
            sample_circle_directions, sample_circle_positions, det_circle_directions, det_circle_positions = self.get_sample_and_detector_circles(hkl_attr)
            # get all axis directions for primary beam, inplane beam, and sample surface normal from hkl attributes
            primary_beam_directions, inplane_beam_direction, sample_surface_normal_direction = self.get_axis_directions(hkl_attr)
            # get UB matrix and energy
            ub_matrix = self.get_ub_matrix(hkl_attr)
            ub_matrix = np.reshape(ub_matrix, (3,3))
            energy = to_eV(self.get_energy(hkl_attr), self.get_energy_units(hkl_attr),
                           'photon energy')

            # Resolve detector calibration through the active HKL channel map so
            # any prefix works, then convert declared units to mm/deg once.
            ds_cfg = get_hkl_section(self.hkl_config, 'DETECTOR_SETUP', required=True)
            canonical = self.config.get('IOC_RSM_PARAMETER', {}) or {}
            detector = detector_model_from_setup(
                detector_setup_from_channels(
                    ds_cfg, hkl_attr, canonical.get('DETECTOR_SETUP')
                ),
                shape,
            )
            model = RSMGeometry(
                sample_axes=tuple(
                    RotationAxis('sample', direction)
                    for direction in sample_circle_directions
                ),
                detector_axes=tuple(
                    RotationAxis('detector', direction)
                    for direction in det_circle_directions
                ),
                primary_beam_direction=primary_beam_directions,
                inplane_reference_direction=inplane_beam_direction,
                sample_surface_normal_direction=sample_surface_normal_direction,
                energy_eV=energy,
                ub_matrix=ub_matrix,
                detector=detector,
                sample_orientation=canonical.get('SAMPLE_ORIENTATION', 'det'),
            )
            return calculate_q(
                build_hxrd(model),
                sample_circle_positions,
                det_circle_positions,
            )
        except Exception as e:
            self.log_error("RSM creation failed", e)
            return None, None, None

    def attributes_diff(self, hkl_attr: dict, old_attr: dict) -> bool:
        if hkl_attr.keys() != old_attr.keys():
            return True
        for key, value in hkl_attr.items():
            old = old_attr[key]
            if isinstance(value, np.ndarray):
                if not np.array_equal(value, old):
                    return True
            elif old != value:
                return True
        return False

    def process(self, pvObject):
        t0 = time.time()

        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            # Frame has no image data
            return pvObject

        if 'timeStamp' not in pvObject:
            # No timestamp, just return the object
            return pvObject

        if 'attribute' not in pvObject:
            self.log_error('attributes not in pvObject')
            return pvObject

        # Optionally decode image data for local use, but do not modify pvObject['value']
        _ = self.decompress_image(pvObject)

        self.hkl_attributes = self.parse_hkl_ndattributes(pvObject)
        self.shape = tuple([dim['size'] for dim in dims])

        if self.old_attrbutes is not None:
            attributes_diff = self.attributes_diff(self.hkl_attributes, self.old_attrbutes)
        else:
            attributes_diff = True
        self.old_attrbutes = copy.deepcopy(self.hkl_attributes)

        if attributes_diff:
            # Only recalculate qxyz if there are new attributes
            qxyz = self.create_rsm(self.hkl_attributes, self.shape)
            if qxyz is None or qxyz[0] is None:
                self.nFrameErrors += 1
                if hasattr(self, 'logger'):
                    self.logger.warning(
                        "Skipping RSM for this frame: create_rsm returned None "
                        "(likely missing HKL attributes from associator)."
                    )
                self.updateOutputChannel(pvObject)
                self.processingTime += (time.time() - t0)
                return pvObject
            self.qx: np.ndarray = np.ravel(qxyz[0])
            self.qy: np.ndarray = np.ravel(qxyz[1])
            self.qz: np.ndarray = np.ravel(qxyz[2])
            self.codec_name = pvObject['codec']['name']
            self.original_dtype = self.qx.dtype if self.qx.dtype == self.qy.dtype == self.qz.dtype else np.dtype('float64')
            self.codec_parameters = int(self.CODEC_PARAMETERS_MAP.get(self.original_dtype, None)) if self.codec_name else -1
            self.uncompressed_size = self.qx.nbytes if self.qx.nbytes == self.qy.nbytes == self.qz.nbytes else np.prod(self.shape) * self.original_dtype.itemsize
            self.compressed_size_qx = self.uncompressed_size
            self.compressed_size_qy = self.uncompressed_size
            self.compressed_size_qz = self.uncompressed_size

            if self.codec_name != '':
                self.qx = self.compress_array(self.qx, self.codec_name)
                self.qy = self.compress_array(self.qy, self.codec_name)
                self.qz = self.compress_array(self.qz, self.codec_name)
                self.compressed_size_qx = self.qx.shape[0]
                self.compressed_size_qy = self.qy.shape[0]
                self.compressed_size_qz = self.qz.shape[0]

        if self.qx is None or self.codec_name is None:
            self.updateOutputChannel(pvObject)
            self.processingTime += (time.time() - t0)
            return pvObject

        try:
            # Create RSM data structure
            rsm_data = {
                        'codec':{
                            'name': self.codec_name,
                            'parameters': self.codec_parameters},
                        'qx': {
                            'compressedSize': int(self.compressed_size_qx),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qx},
                        'qy': {
                            'compressedSize': int(self.compressed_size_qy),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qy},
                        'qz': {
                            'compressedSize': int(self.compressed_size_qz),
                            'uncompressedSize': int(self.uncompressed_size),
                            'value':self.qz},
                        }

            # Create PV object to hold RSM attributes
            if self.codec_name != '':
                rsm_object = {'name': 'RSM', 'value': PvObject({'value': self.type_dict_compressed}, {'value': rsm_data})}
            else:
                rsm_object = {'name': 'RSM', 'value': PvObject({'value': self.type_dict}, {'value': rsm_data})}

            # Rebuild attribute list: all parsed metadata + RSM
            frameAttributes = []
            for name, value in self.all_attributes.items():
                try:
                    if isinstance(value, bool):
                        attr = {'name': name, 'value': pva.PvBoolean(value)}
                    elif isinstance(value, (int, float)):
                        attr = {'name': name, 'value': pva.PvFloat(float(value))}
                    elif isinstance(value, str):
                        attr = {'name': name, 'value': pva.PvString(value)}
                    elif isinstance(value, np.ndarray):
                        pv = pva.PvScalarArray(pva.DOUBLE)
                        pv.set(value.tolist())
                        attr = {'name': name, 'value': pv}
                    else:
                        continue
                    frameAttributes.append(attr)
                except Exception:
                    pass
            frameAttributes.append(rsm_object)

            # Update stats
            frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
            self.lastFrameTimestamp = frameTimestamp
            self.nFramesProcessed += 1

            proc_time_start = pva.PvObject({'value': pva.DOUBLE})
            proc_time_start['value'] = t0  # seconds, or multiply by 1000.0 for ms
            frameAttributes.append({
                'name': f'procTimeStart_{self.__class__.__name__}',
                'value': proc_time_start
            })
            proc_time_end = pva.PvObject({'value': pva.DOUBLE})
            proc_time_end['value'] = time.time()  # seconds, or multiply by 1000.0 for ms
            frameAttributes.append({
                'name': f'procTimeEnd_{self.__class__.__name__}',
                'value': proc_time_end
            })
            proc_time = pva.PvObject({'value': pva.DOUBLE})
            proc_time['value'] = (time.time() - t0)  # seconds, or multiply by 1000.0 for ms
            frameAttributes.append({
                'name': f'procTime_{self.__class__.__name__}',
                'value': proc_time
            })

            pvObject['attribute'] = frameAttributes

            self.updateOutputChannel(pvObject)

            # Update processing time
            t1 = time.time()
            self.processingTime += (t1 - t0)

            return pvObject

        except Exception as e:
            self.nFrameErrors += 1
            self.log_error("Frame processing error", e)
            return pvObject
