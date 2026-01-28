"""
HDF5 Handler that writes, reads in nexus standard file format
"""
import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
from utils.pva_reader import PVAReader
from pathlib import Path
import toml


HDF5_STRUCTURE = {
    "nexus": {
        "default": {
            "NX_class": "NXroot",
            "default": "entry",
            "entry": {
                "NX_class": "NXentry",
                "default": "data",
                
                # --- INSTRUMENT: The 'How' (Source + Detector) ---
                "instrument": {
                    "NX_class": "NXinstrument",
                    "source": {
                        "NX_class": "NXsource",
                        "target": "HKL/SPEC/ENERGY_VALUE",
                        "units": "keV"
                    },
                    "detector": {
                        "NX_class": "NXdetector",
                        "target": "HKL/DETECTOR_SETUP",
                        "data_link": "/entry/data/data" # Link to raw image stack
                    }
                },

                # --- SAMPLE: The 'What' (Motor Stacks + Environment) ---
                "sample": {
                    "NX_class": "NXsample",
                    "ub_matrix": {
                        "NX_class": "NXcollection", 
                        "target": "HKL/SPEC/UB_MATRIX_VALUE"
                    },
                    # Map your 4/6 circles here
                    "geometry": {
                        "NX_class": "NXtransformations",
                        "sample_phi": {"target": "HKL/SAMPLE_CIRCLE_AXIS_4", "type": "rotation"},
                        "sample_chi": {"target": "HKL/SAMPLE_CIRCLE_AXIS_3", "type": "rotation"},
                        "sample_eta": {"target": "HKL/SAMPLE_CIRCLE_AXIS_2", "type": "rotation"},
                        "sample_mu":  {"target": "HKL/SAMPLE_CIRCLE_AXIS_1", "type": "rotation"}
                    }
                },

                # --- DATA: The 'View' (Plotting Entry Point) ---
                "data": {
                    "NX_class": "NXdata",
                    "signal": "data",
                    "data": {"link": "/entry/data/data"}
                }
            }
        },
        "scans": {
            "NX_class": "NXroot",
            "default": "entry",
            "entry": {
                "name": "entry",
                "NX_class": "NXentry",
                "default": "data",
                # Nested Groups inside Entry
                "instrument": {
                    "name": "instrument",
                    "NX_class": "NXinstrument",
                    "detector": {
                        "name": "detector",
                        "NX_class": "NXdetector",
                        "field": "data",
                        # HKL.DETECTOR_SETUP
                        "distance": {"value": None, "units": "mm"},
                        "beam_center_x": {"value": None, "units": "pixel"},
                        "beam_center_y": {"value": None, "units": "pixel"},
                        "pixel_size": {"value": None, "units": "m"},
                        # HKL.DETECTOR_CIRCLE_AXIS
                        "transformations": {
                            "NX_class": "NXtransformations",
                            "axis_2": {"value": None, "type": "rotation", "vector": [0, 1, 0]}
                        }
                    },
                    # HKL.SPEC ENERGY
                    "source": {
                        "name": "source",
                        "NX_class": "NXsource",
                        "energy": {"value": None, "units": "keV"} 
                    },
                },
                "sample": {
                        "name": "sample",
                        "NX_class": "NXsample",
                        "field": "rotation_angle",
                        # HKL.SPEC UB_MATRIX
                        "ub_matrix": {"value": None, "units": "1/angstrom"},
                        "orientation_matrix": {"value": None},
                        # HKL Orientation Directions
                        "surface_normal": {"vector": [0, 0, 1]},
                        "inplane_reference": {"vector": [1, 0, 0]}
                    },
                "data": {
                    "name": "data",
                    "NX_class": "NXdata",
                    "signal": "data",
                    "axes": "rotation_angle"
                }
            }
        },

        "format": {
            "name":"nexus",
            "links":{
                "Nexus": "",
                "Scan Standard":"",
                "DashPVA":""
                }
        }
    }        
}

class HDF5Handler(QObject):
    hdf5_writer_finished = pyqtSignal(str)
    def __init__(self, file_path:str="", pva_reader:PVAReader=None, compress:bool=True):
        super(HDF5Handler, self).__init__()
        self.pva_reader = pva_reader
        self.file_path = file_path
        self.default_output = Path('~/DashPVA/outputs/scans/demo/nexus_standard_default_format.h5').expanduser()
        self.temp_output = Path('~/DashPVA/outputs/scans/demo/temp.h5').expanduser()
        self.compress = compress
        # Build reverse HKL map from the loaded TOML config (via PVAReader)
        self.hkl_reverse_map = {}
        if self.pva_reader is not None:
            try:
                self.hkl_reverse_map = self.parse_toml()
            except Exception as e:
                print(f"[HDF5Handler] Failed to parse TOML for HKL map: {e}")

    #Loading
    def load_data(self):
        pass

    @pyqtSlot()
    def save_data(self, compress=False, file_path=None, clear_caches=False, is_scan=False):
        if file_path is None:
            file_path = self.default_output

        self.file_path=file_path
        if self.pva_reader is not None:
            self.save_from_caches(compress, clear_caches=clear_caches, is_scan=is_scan)

    # Saving
    def save_from_caches(self, compress:bool=True, clear_caches:bool=True,is_scan=False):
        if is_scan:
            self.save_as_scan_format(compress, clear_caches)
        else:
            self.save_as_default_format(compress, clear_caches)

    def save_as_default_format(self, compress: bool = True, clear_caches: bool = True):
        """Save using the same unified structure as utils/metadata_converter.py.

        Layout:
          /entry/data/data                         -> image stack
          /entry/data/metadata                     -> base metadata
          /entry/data/metadata/motor_positions     -> position PVs
          /entry/data/metadata/HKL/...             -> hierarchical HKL per config
          /entry/data/hkl/qx,qy,qz                 -> optional caches when present
        """
        all_caches = self.pva_reader.get_all_caches(clear_caches=clear_caches)
        images = all_caches.get('images')
        attributes = all_caches.get('attributes')
        rsm = all_caches.get('rsm')
        shape = self.pva_reader.get_shape()

        # Align lengths
        len_images = len(images or [])
        len_attributes = len(attributes or [])
        if len_images != len_attributes:
            m = min(len_images, len_attributes)
            if m > 0:
                images = (images or [])[:m]
                attributes = (attributes or [])[:m]
                len_images = len(images)
                len_attributes = len(attributes)
        if images is None or len_images == 0:
            self.hdf5_writer_finished.emit("Failed to save: Empty image cache")
            return

        # Merge metadata across frames: key -> list of values
        merged_metadata = {}
        for attr in (attributes or []):
            for k, v in attr.items():
                if k in ('RSM', 'Analysis'):
                    continue
                merged_metadata.setdefault(k, []).append(v)

        # HKL config from reader
        hkl_cfg = getattr(self.pva_reader, 'config', {}).get('HKL', {})
        HKL_IN_CONFIG = bool(hkl_cfg)

        def is_position_pv(pv: str) -> bool:
            if not isinstance(pv, str):
                return False
            return (":Position" in pv) or (".RBV" in pv) or ("_RBV" in pv)

        ds_kwargs = (hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True) if compress else {})

        with h5py.File(self.file_path, 'w') as h5f:
            # /entry/data/data
            entry = h5f.create_group('entry')
            data_grp = entry.create_group('data')
            data_grp.create_dataset('data', data=np.array([np.reshape(img, shape) for img in images]), **ds_kwargs)

            # /entry/data/metadata and motor_positions
            metadata_grp = data_grp.create_group('metadata')
            motor_pos_grp = metadata_grp.create_group('motor_positions')
            for key, values in merged_metadata.items():
                try:
                    arr = np.array(values)
                    target = motor_pos_grp if is_position_pv(key) else metadata_grp
                    if arr.dtype.kind in ('i', 'u', 'f'):
                        target.create_dataset(key, data=arr)
                    elif arr.dtype.kind in ('U', 'S', 'O'):
                        dt = h5py.string_dtype(encoding='utf-8')
                        target.create_dataset(key, data=arr.astype(dt))
                    else:
                        dt = h5py.string_dtype(encoding='utf-8')
                        target.create_dataset(key, data=str(values), dtype=dt)
                except Exception:
                    pass

            # /entry/data/metadata/HKL ... per config
            if HKL_IN_CONFIG:
                hkl_root = metadata_grp.create_group('HKL')

                def _write_from_pv(group, name, pv_key):
                    vals = merged_metadata.get(pv_key)
                    if vals is None:
                        return
                    arr = np.array(vals)
                    if arr.dtype.kind in ('i', 'u', 'f'):
                        group.create_dataset(name, data=arr)
                    elif arr.dtype.kind in ('U', 'S', 'O'):
                        dt = h5py.string_dtype(encoding='utf-8')
                        group.create_dataset(name, data=arr.astype(dt))
                    else:
                        dt = h5py.string_dtype(encoding='utf-8')
                        group.create_dataset(name, data=str(vals), dtype=dt)

                for section_name in ['PRIMARY_BEAM_DIRECTION', 'INPLANE_REFERENCE_DIRECITON', 'SAMPLE_SURFACE_NORMAL_DIRECITON']:
                    sec = hkl_cfg.get(section_name, {})
                    if sec:
                        sec_grp = hkl_root.create_group(section_name)
                        for k, pv in sec.items():
                            _write_from_pv(sec_grp, k, pv)

                for base in ['SAMPLE_CIRCLE_AXIS_1', 'SAMPLE_CIRCLE_AXIS_2', 'SAMPLE_CIRCLE_AXIS_3', 'SAMPLE_CIRCLE_AXIS_4', 'DETECTOR_CIRCLE_AXIS_1', 'DETECTOR_CIRCLE_AXIS_2']:
                    sec = hkl_cfg.get(base, {})
                    if sec:
                        grp = hkl_root.create_group(base)
                        for k, pv in sec.items():
                            _write_from_pv(grp, k, pv)

                spec = hkl_cfg.get('SPEC', {})
                if spec:
                    spec_grp = hkl_root.create_group('SPEC')
                    ev_key = spec.get('ENERGY_VALUE')
                    if ev_key:
                        vals = merged_metadata.get(ev_key)
                        if vals is not None:
                            spec_grp.create_dataset('ENERGY_VALUE', data=np.array(vals))
                    ub_key = spec.get('UB_MATRIX_VALUE')
                    if ub_key:
                        vals = merged_metadata.get(ub_key)
                        if vals is not None:
                            arr = np.asarray(vals).ravel()
                            ub9 = arr[:9] if arr.size >= 9 else arr
                            spec_grp.create_dataset('UB_MATRIX_VALUE', data=ub9)

                detector = hkl_cfg.get('DETECTOR_SETUP', {})
                if detector:
                    det_grp = hkl_root.create_group('DETECTOR_SETUP')
                    for k, pv in detector.items():
                        _write_from_pv(det_grp, k, pv)

            # Optional HKL caches
            if HKL_IN_CONFIG and rsm:
                try:
                    if len(rsm[0]) == len_images:
                        hkl_grp = data_grp.create_group('hkl')
                        hkl_grp.create_dataset('qx', data=np.array([np.reshape(qx, shape) for qx in rsm[0]]), **ds_kwargs)
                        hkl_grp.create_dataset('qy', data=np.array([np.reshape(qy, shape) for qy in rsm[1]]), **ds_kwargs)
                        hkl_grp.create_dataset('qz', data=np.array([np.reshape(qz, shape) for qz in rsm[2]]), **ds_kwargs)
                except Exception:
                    pass

        self.hdf5_writer_finished.emit(f"Saved to: {self.file_path}\nFormat: unified-structure")

    def save_as_scan_format(self, compress:bool=True, clear_caches:bool=True):
        all_caches = self.pva_reader.get_all_caches(clear_caches=clear_caches)
        images = all_caches['images']
        attributes = all_caches['attributes']
        rsm = all_caches['rsm']
        shape = self.pva_reader.get_shape()

        nx_conf = HDF5_STRUCTURE['nexus']['scans']
        formatter = HDF5_STRUCTURE['nexus']['format']
        with h5py.File(self.file_path, 'w') as h5_file:
            # Set root
            h5_file.attrs['NX_class'] = nx_conf['NX_class']
            h5_file.attrs['default'] = nx_conf['default']

            # Set entry
            entry_cfg = nx_conf['entry']
            entry = h5_file.create_group(entry_cfg['name'])
            entry.attrs['NX_class'] = entry_cfg['NX_class']
            entry.attrs['default'] = entry_cfg['default']

            # Set instruments
            instr_cfg = entry_cfg['instrument']
            instr_grp = entry.create_group(instr_cfg['name'])
            instr_grp.attrs['NX_class'] = instr_cfg['NX_class']

            det_cfg = instr_cfg['detector']
            det_grp = instr_grp.create_group(det_cfg['name'])
            det_grp.attrs['NX_class'] = det_cfg['NX_class']

            det_grp.create_dataset(det_cfg['field'], 
                                   data=np.array([np.reshape(img, shape) for img in images]), 
                                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))

            # Write DETECTOR_SETUP attributes grouped per TOML under instrument/detector/DETECTOR_SETUP
            try:
                det_setup_cfg = getattr(self.pva_reader, 'config', {}).get('HKL', {}).get('DETECTOR_SETUP', {})
                if isinstance(det_setup_cfg, dict) and attributes:
                    setup_grp = det_grp.create_group('DETECTOR_SETUP')
                    for field_name, pv_key in det_setup_cfg.items():
                        series = [attr.get(pv_key, None) for attr in attributes]
                        # Choose dtype based on series content
                        if all((isinstance(v, (int, float, np.number)) or v is None) for v in series):
                            numeric_series = [float(v) if v is not None else np.nan for v in series]
                            setup_grp.create_dataset(str(field_name), data=np.array(numeric_series, dtype=np.float64))
                        elif all(isinstance(v, str) or v is None for v in series):
                            dt = h5py.string_dtype(encoding='utf-8')
                            str_series = [v if v is not None else '' for v in series]
                            setup_grp.create_dataset(str(field_name), data=np.array(str_series, dtype=dt))
                        else:
                            dt = h5py.string_dtype(encoding='utf-8')
                            str_series = [str(v) if v is not None else '' for v in series]
                            setup_grp.create_dataset(str(field_name), data=np.array(str_series, dtype=dt))
            except Exception as e:
                print(f"[HDF5Handler] Failed to write DETECTOR_SETUP attributes: {e}")
            
            # Source (Energy)
            src_cfg = instr_cfg['source']
            src_grp = instr_grp.create_group(src_cfg['name'])
            src_grp.attrs['NX_class'] = src_cfg['NX_class']
            if src_cfg['energy']['value'] is not None:
                en_ds = src_grp.create_dataset('energy', data=src_cfg['energy']['value'])
                en_ds.attrs['units'] = src_cfg['energy']['units']

            # Set sample -- Defines: ROI's, HKL
            sample_cfg = entry_cfg['sample']
            sample_grp = entry.create_group(sample_cfg['name'])
            sample_grp.attrs['NX_class'] = sample_cfg['NX_class']
            # Create rotation_angle dataset from motor position PVs
            primary_axis_values = []
            if attributes:
                pos_keys = [k for k in attributes[0].keys() if 'Position' in k]
                for attr in attributes:
                    v = None
                    for k in pos_keys:
                        val = attr.get(k)
                        if isinstance(val, (int, float, np.number)):
                            v = float(val)
                            break
                    primary_axis_values.append(0.0 if v is None else v)
            if primary_axis_values:
                rot_ds = sample_grp.create_dataset(sample_cfg['field'], data=np.array(primary_axis_values, dtype=np.float64))
                # Units could be degrees if known; skipping units attr due to lack of config

            # Set data -- Where images and motor_positions are added 
            data_cfg = entry_cfg['data']
            data_grp = entry.create_group(data_cfg['name'])
            data_grp.attrs['NX_class'] = data_cfg['NX_class']
            data_grp.attrs['signal'] = data_cfg['signal']
            data_grp.attrs['axes'] = data_cfg['axes']
            
            # Crucial for 2D Detectors: Map rotation_angle to the first dimension (index 0)
            data_grp.attrs[f"{data_cfg['axes']}_indices"] = 0

            # Write HKL grouped series under entry/hkl/<group>/<field>
            hkl_series = self.convert_to_nexus_format()
            if hkl_series:
                hkl_grp = entry.create_group('hkl')
                for grp_name, fields in hkl_series.items():
                    subgrp = hkl_grp.create_group(grp_name)
                    for field_name, series in fields.items():
                        if series and isinstance(series[0], str):
                            dt = h5py.string_dtype(encoding='utf-8')
                            subgrp.create_dataset(field_name, data=np.array(series, dtype=dt))
                        else:
                            subgrp.create_dataset(field_name, data=np.array(series, dtype=np.float64))

            data_grp['data'] = h5py.SoftLink(f'/{entry_cfg["name"]}/instrument/detector/data')
            data_grp['rotation_angle'] = h5py.SoftLink(f'/{entry_cfg["name"]}/sample/rotation_angle')
        self.hdf5_writer_finished.emit(f'\
                                    Saved to: {self.file_path}\n \
                                    Format: {formatter["name"]}\
                                    ')

    # Info
    def get_file_info(self):
        pass

    # Parse Toml dict
    def parse_toml(self):
        """
        Build a reverse HKL map from the loaded TOML configuration.
        Returns a dict mapping PV attribute keys -> (group_name, field_name), both lowercased.

        Example:
            'DetectorSetup:Distance' -> ('detector_setup', 'distance')
        """
        reverse_map: dict[str, tuple[str, str]] = {}
        try:
            cfg = getattr(self.pva_reader, 'config', {}) if self.pva_reader is not None else {}
            hkl_cfg: dict = cfg.get('HKL', {})
            for group_name, fields in hkl_cfg.items():
                if isinstance(fields, dict):
                    for field_name, pv_key in fields.items():
                        if isinstance(pv_key, str) and pv_key:
                            reverse_map[pv_key] = (str(group_name).lower(), str(field_name).lower())
            return reverse_map
        except Exception as e:
            print(f"[HDF5Handler] parse_toml failed: {e}")
            return {}

    def get_structured_attr(self, attr):
        """
        Group a single frame's attribute dict into structured sections.

        Returns a dict with keys:
            - 'hkl': {group: {field: value}}
            - 'rois': {roi_name: {dim: value}}
            - 'motor_positions': {pv_key: value}  # keys containing 'Position'
            - 'metadata': {other_key: value}
        """
        structured = {
            'hkl': {},
            'rois': {},
            'motor_positions': {},
            'sample_circle_axis_n': {},
            'detector_circle_axis_n': {},
            'spec':{},
        }
        if not isinstance(attr, dict):
            return structured

        # HKL grouping via reverse map
        for key, value in attr.items():
            if key in self.hkl_reverse_map:
                grp, field = self.hkl_reverse_map[key]
                structured['hkl'].setdefault(grp, {})[field] = value
                continue
            # ROI grouping
            if 'ROI' in key:
                parts = key.split(':')
                if len(parts) >= 3 and parts[1].startswith('ROI'):
                    roi = parts[1]
                    dim = parts[2]
                    structured['rois'].setdefault(roi, {})[dim] = value
                    continue
            # Motor positions (generic)
            if 'Position' in key:
                structured['motor_positions'][key] = value
                continue
            # Metadata fallback
            structured['metadata'][key] = value
        return structured
        
    # Convert to nexus standard
    def convert_to_nexus_format(self):
        """
        Prepare HKL grouped series from cached attribute list for writing under entry/hkl/<group>/<field>.
        Returns: {group: {field: [values...]}}
        """
        grouped_series: dict[str, dict[str, list]] = {}
        try:
            all_caches = self.pva_reader.get_all_caches(clear_caches=False) if self.pva_reader is not None else {'attributes': []}
            attributes_list = all_caches.get('attributes', [])
            for attr in attributes_list:
                # group one frame
                grouped = self.get_structured_attr(attr)
                for grp, fields in grouped['hkl'].items():
                    for field, val in fields.items():
                        grouped_series.setdefault(grp, {}).setdefault(field, []).append(val)
            return grouped_series
        except Exception as e:
            print(f"[HDF5Handler] convert_to_nexus_format failed: {e}")
            return {}
