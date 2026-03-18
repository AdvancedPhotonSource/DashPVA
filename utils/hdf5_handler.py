"""
HDF5 Handler that writes, reads in nexus standard file format
"""
import h5py
import numpy as np
import time
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
from utils.pva_reader import PVAReader
from utils.metadata_converter import (
    convert_files_or_dir,
    _build_axis_lookup,
    _derive_axis_from_pv,
    is_position_pv,
)
from utils.log_manager import LogMixin
import settings
import toml


class HDF5Handler(QObject, LogMixin):
    hdf5_writer_finished = pyqtSignal(str)
    def __init__(self, file_path:str="", pva_reader:PVAReader=None, compress:bool=True):
        super(HDF5Handler, self).__init__()
        self.pva_reader = pva_reader
        self.file_path = file_path
        # Default outputs under settings.OUTPUT_PATH/scans/demo
        try:
            base_out = Path(getattr(settings, 'OUTPUT_PATH', './outputs')).expanduser()
        except Exception:
            base_out = Path('./outputs')
        # Construct defaults and ensure parent directories exist
        self.default_output = base_out.joinpath('scans/demo/nexus_standard_default_format.h5')
        self.temp_output = base_out.joinpath('scans/demo/temp.h5')
        self.default_output.parent.mkdir(parents=True, exist_ok=True)
        self.temp_output.parent.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        # Build reverse HKL map from the loaded TOML config (via PVAReader)
        self.hkl_reverse_map = {}
        if self.pva_reader is not None:
            try:
                self.hkl_reverse_map = self.parse_toml()
            except Exception:
                pass
        try:
            self.set_log_manager()
        except Exception:
            pass

    #Loading
    def load_data(self):
        raise NotImplementedError

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
        """Save in NeXus-compliant HDF5 format.

        Layout:
          /entry/                                  (NXentry)
          /entry/data/data                         (NXdata) -> image stack
          /entry/data/metadata/motor_positions/    -> axis-labeled motor positions (ETA, MU, ...)
          /entry/data/metadata/HKL/...             -> hierarchical HKL per config
          /entry/instrument/                       (NXinstrument)
          /entry/instrument/detector/              (NXdetector) soft-linked to image data
          /entry/sample/                           (NXsample)
          /entry/sample/geometry/                  (NXtransformations) soft-linked to circle axes
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

        # Build axis lookup from TOML for motor position label resolution
        axis_lookup = {}
        try:
            toml_path = settings.ensure_path()
            if toml_path:
                mapping = toml.load(str(toml_path))
                axis_lookup = _build_axis_lookup(mapping)
        except Exception:
            pass

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
                    if is_position_pv(key):
                        # Only write motor positions with a resolved axis label — no flat PV names
                        axis_label = _derive_axis_from_pv(key, axis_lookup) if axis_lookup else None
                        if not axis_label:
                            continue
                        if arr.dtype.kind in ('i', 'u', 'f'):
                            ds = motor_pos_grp.create_dataset(axis_label, data=arr)
                            ds.attrs['units'] = 'deg'
                    else:
                        if arr.dtype.kind in ('i', 'u', 'f'):
                            metadata_grp.create_dataset(key, data=arr)
                        elif arr.dtype.kind in ('U', 'S', 'O'):
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=arr.astype(dt))
                        else:
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=str(values), dtype=dt)
                except Exception:
                    pass

            # /entry/data/metadata/HKL ... per config
            if HKL_IN_CONFIG:
                hkl_root = metadata_grp.create_group('HKL')

                for section_name in ['PRIMARY_BEAM_DIRECTION', 'INPLANE_REFERENCE_DIRECITON', 'SAMPLE_SURFACE_NORMAL_DIRECITON']:
                    sec = hkl_cfg.get(section_name, {})
                    if sec:
                        sec_grp = hkl_root.create_group(section_name)
                        for k, pv in sec.items():
                            self._write_pv_dataset(sec_grp, k, pv, merged_metadata)

                for base in ['SAMPLE_CIRCLE_AXIS_1', 'SAMPLE_CIRCLE_AXIS_2', 'SAMPLE_CIRCLE_AXIS_3', 'SAMPLE_CIRCLE_AXIS_4', 'DETECTOR_CIRCLE_AXIS_1', 'DETECTOR_CIRCLE_AXIS_2']:
                    sec = hkl_cfg.get(base, {})
                    if sec:
                        grp = hkl_root.create_group(base)
                        for k, pv in sec.items():
                            self._write_pv_dataset(grp, k, pv, merged_metadata)

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
                        self._write_pv_dataset(det_grp, k, pv, merged_metadata)

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

            # Apply NeXus NX_class attributes and structural groups
            self._apply_nx_structure(h5f, entry)

        # Auto-convert metadata structure per current TOML before emitting signal
        conversion_suffix = ""
        try:
            toml_path = settings.ensure_path()
            if toml_path:
                convert_files_or_dir(
                    toml_path=toml_path,
                    hdf5_path=str(self.file_path),
                    base_group="entry/data/metadata",
                    include=True,
                    in_place=True,
                    recursive=False,
                )
                conversion_suffix = " (converted)"
            else:
                conversion_suffix = " (conversion skipped: no TOML path)"
        except Exception as conv_err:
            conversion_suffix = f" (conversion failed: {conv_err})"

        self.hdf5_writer_finished.emit(f"Saved to: {self.file_path}\nFormat: nexus{conversion_suffix}")

    @staticmethod
    def _is_position_pv(key: str) -> bool:
        if not isinstance(key, str):
            return False
        return (":Position" in key) or (".RBV" in key) or ("_RBV" in key)

    def _write_pv_dataset(self, group, name: str, pv_key: str, merged_metadata: dict):
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

    def _apply_nx_structure(self, h5f: h5py.File, entry: h5py.Group, base_group: str = "entry/data/metadata"):
        """Apply NeXus NX_class attributes and create instrument/sample structural groups."""
        nx_def = settings.HDF5_STRUCTURE['nexus']['default']
        nx_entry = nx_def['entry']

        # Root-level attributes
        h5f.attrs['NX_class'] = nx_def['NX_class']
        h5f.attrs['default'] = nx_def['default']

        # Entry attributes
        entry.attrs['NX_class'] = nx_entry['NX_class']
        entry.attrs['default'] = nx_entry['default']

        # /entry/data attributes
        if 'data' in entry:
            nx_data = nx_entry['data']
            entry['data'].attrs['NX_class'] = nx_data['NX_class']
            entry['data'].attrs['signal'] = nx_data['signal']

        # /entry/instrument (NXinstrument)
        instr_cfg = nx_entry['instrument']
        instr_grp = entry.require_group('instrument')
        instr_grp.attrs['NX_class'] = instr_cfg['NX_class']

        src_grp = instr_grp.require_group('source')
        src_grp.attrs['NX_class'] = instr_cfg['source']['NX_class']

        det_grp = instr_grp.require_group('detector')
        det_grp.attrs['NX_class'] = instr_cfg['detector']['NX_class']
        if 'data' not in det_grp:
            det_grp['data'] = h5py.SoftLink(instr_cfg['detector']['data_link'])

        # /entry/sample (NXsample)
        sample_cfg = nx_entry['sample']
        sample_grp = entry.require_group('sample')
        sample_grp.attrs['NX_class'] = sample_cfg['NX_class']

        ub_grp = sample_grp.require_group('ub_matrix')
        ub_grp.attrs['NX_class'] = sample_cfg['ub_matrix']['NX_class']
        ub_src = f"{base_group}/HKL/SPEC/UB_MATRIX_VALUE"
        if ub_src in h5f and 'value' not in ub_grp:
            ub_grp['value'] = h5py.SoftLink(f'/{ub_src}')

        geo_grp = sample_grp.require_group('geometry')
        geo_grp.attrs['NX_class'] = sample_cfg['geometry']['NX_class']
        for field, axis_cfg in sample_cfg['geometry'].items():
            if field == 'NX_class' or not isinstance(axis_cfg, dict):
                continue
            target_path = f"{base_group}/HKL/{axis_cfg.get('target', '')}"
            if target_path in h5f and field not in geo_grp:
                geo_grp[field] = h5py.SoftLink(f'/{target_path}')

    @pyqtSlot(bool, bool, bool)
    def save_to_h5(self, clear_caches: bool = True, write_temp: bool = True, write_output: bool = True) -> None:
        """Gateway slot — delegates to HDF5Writer.save_caches_to_h5. No h5py logic here."""
        from utils.hdf5_writer import HDF5Writer
        try:
            writer = HDF5Writer(file_path="", pva_reader=self.pva_reader)
            writer.hdf5_writer_finished.connect(self.hdf5_writer_finished)
            writer.save_caches_to_h5(clear_caches, write_temp, write_output)
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save: {e}")

    def save_as_scan_format(self, compress: bool = True, clear_caches: bool = True):
        """Delegate scan-format write to HDF5Writer."""
        from utils.hdf5_writer import HDF5Writer
        try:
            writer = HDF5Writer(file_path="", pva_reader=self.pva_reader)
            writer.hdf5_writer_finished.connect(self.hdf5_writer_finished)
            writer.save_scan_to_h5(self.file_path, compress=compress, clear_caches=clear_caches)
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save scan: {e}")

    # Info
    def get_file_info(self):
        raise NotImplementedError

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
            try:
                self.logger.warning(f"parse_toml failed: {e}")
            except Exception:
                pass
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
            try:
                self.logger.warning(f"convert_to_nexus_format failed: {e}")
            except Exception:
                pass
            return {}
