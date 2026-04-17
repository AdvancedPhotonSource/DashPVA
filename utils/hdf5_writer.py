import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
import toml
from utils.metadata_converter import (
    _build_axis_lookup,
    _derive_axis_from_pv,
    is_position_pv,
)
import settings
from utils.log_manager import LogMixin
# removed traceback import
# from utils import PVAReader
import time

class HDF5Writer(QObject, LogMixin):
    hdf5_writer_finished = pyqtSignal(str)

    def __init__(self, file_path: str, pva_reader):
        super(HDF5Writer, self).__init__()
        self.file_path = file_path
        self.pva_reader = pva_reader
        # Default used when no OUTPUT_FILE_CONFIG is provided: filename under OUTPUT_PATH
        self.default_output_file_config = {'FilePath': 'SCAN_OUTPUT.h5'}
        # Initialize logging
        try:
            self.set_log_manager()
        except Exception:
            pass

    @pyqtSlot(bool, bool, bool, str)
    def save_caches_to_h5(self, clear_caches: bool = True, write_temp: bool = True, write_output: bool = True, output_override: str = '') -> None:
        # TODO: add analysis
        """
        Saves available caches (images and HKL data) to an HDF5 file under a branch structure.
        The file structure is as follows:
            /entry/data         --> The image cache array
            /entry/rois/ROI1-4
            /entry/metadata/motor_positions
            /entry/analysis/intensity
            /entry/analysis/comx
            /entry/analysis/comy
            /entry/HKL/qx        --> The qx cache array (if available)
            /entry/HKL/qy        --> The qy cache array (if available)
            /entry/HKL/qz        --> The qz cache array (if available)

        Args:
            filename (str): The output HDF5 file name.
        """
        OUTPUT_FILE_LOCATION = "UNKNOWN_FILE"  # Initialize to avoid UnboundLocalError
        try:
            data = dict()

            config = self.pva_reader.get_config_settings()
            OUTPUT_FILE_CONFIG= config.get('OUTPUT_FILE_CONFIG', {}) or {}
            # HKL_IN_CONFIG = config.get('HKL_IN_CONFIG', False)
            all_caches = self.pva_reader.get_all_caches(clear_caches=clear_caches)
            # images = all_caches['images']
            # attributes = all_caches['attributes']
            # rsm = all_caches['rsm']
            # shape = self.pva_reader.get_shape()
            data['images'] = all_caches['images']
            data['attributes'] = all_caches['attributes']
            data['rsm'] = all_caches['rsm']
            data['shape'] = self.pva_reader.get_shape()
            data['len_images'] = len(data['images'])
            data['len_attributes'] = len(data['attributes'])
            data['HKL_IN_CONFIG'] = config.get('HKL_IN_CONFIG', False)


            # Get OUTPUT path dir and create if it doesnt exist
            base_out_dir = Path(getattr(settings, 'OUTPUT_PATH', './outputs')).expanduser()
            base_out_dir.mkdir(parents=True, exist_ok=True)

            # Temp file always goes to OUTPUT_PATH
            TEMP_FILE_LOCATION = base_out_dir.joinpath('temp_hkl_3d.h5')
            # Default output location
            OUTPUT_FILE_LOCATION = base_out_dir.joinpath(f'OUTPUT_SCAN_{time.strftime("%Y%m%d_%H%M%S")}.h5')

            # Scan output override from UI takes priority over config
            if output_override:
                OUTPUT_FILE_LOCATION = Path(output_override).expanduser()
                OUTPUT_FILE_LOCATION.parent.mkdir(parents=True, exist_ok=True)
            # Resolve OUTPUT_FILE_LOCATION according to provided config keys
            elif 'FilePath' in OUTPUT_FILE_CONFIG and 'FileName' in OUTPUT_FILE_CONFIG:
                # Treat FilePath as a directory, FileName as filename
                file_dir = Path(str(OUTPUT_FILE_CONFIG.get('FilePath'))).expanduser()
                file_dir.mkdir(parents=True, exist_ok=True)
                file_name = Path(str(OUTPUT_FILE_CONFIG.get('FileName')))
                OUTPUT_FILE_LOCATION = file_dir.joinpath(file_name)
            elif 'FilePath' in OUTPUT_FILE_CONFIG:
                # Single FilePath can be a full path or just a filename
                fp = Path(str(OUTPUT_FILE_CONFIG.get('FilePath'))).expanduser()
                if str(fp.parent) in ('', '.', str(Path('.'))):
                    OUTPUT_FILE_LOCATION = base_out_dir.joinpath(fp.name)
                else:
                    OUTPUT_FILE_LOCATION = fp
                OUTPUT_FILE_LOCATION.parent.mkdir(parents=True, exist_ok=True)
            else:
                # No config provided -> default to OUTPUT_PATH/SCAN_OUTPUT.h5
                OUTPUT_FILE_LOCATION = base_out_dir.joinpath(self.default_output_file_config['FilePath'])
                OUTPUT_FILE_LOCATION.parent.mkdir(parents=True, exist_ok=True)


            if data['len_images'] != data['len_attributes']:
                min_length = min(data['len_images'], data['len_attributes'])
                if min_length > 0:
                    # Update the data dict consistently
                    data['images'] = data['images'][:min_length]
                    data['attributes'] = data['attributes'][:min_length]
                    data['len_images'] = min_length
                    data['len_attributes'] = min_length
                else:
                    raise ValueError(f"[Saving Caches] Cannot fix cache mismatch - both caches would be empty. Images: {data['len_images']}, Attributes: {data['len_attributes']}")
            # Guard against empty caches
            if not data['images'] or data['len_images'] == 0:
                raise ValueError("[Saving Caches] Caches cannot be empty.")

            # Merge metadata across frames: key -> list of values
            data['metadata'] = self.merge_metadata(data['attributes'])

            # Respect write target selections; if neither selected, skip.
            if not write_temp and not write_output:
                try:
                    self.logger.info("Skipped writing (no targets selected)")
                except Exception:
                    pass
                self.hdf5_writer_finished.emit("Skipped writing (no targets selected)")
                return

            # Write TEMP (uncompressed) when requested
            if write_temp:
                self.h5_save(TEMP_FILE_LOCATION, data, compress=False)

            # Write OUTPUT (compressed) when requested
            if write_output:
                self.h5_save(OUTPUT_FILE_LOCATION, data, compress=True)

            # Emit appropriate message (logging handled by the receiver)
            if write_temp and write_output:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {TEMP_FILE_LOCATION} and {OUTPUT_FILE_LOCATION}")
            elif write_output:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {OUTPUT_FILE_LOCATION}")
            else:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {TEMP_FILE_LOCATION} (temp only)")
        except Exception as e:
            try:
                self.logger.exception(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")
            except Exception:
                pass
            self.hdf5_writer_finished.emit(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")

    def h5_save(self, file_path: str, data: dict, compress: bool = False):
        """Write caches directly in NeXus structure — no flat intermediate, no post-conversion."""
        axis_lookup = {}
        try:
            toml_path = settings.ensure_path()
            if toml_path:
                axis_lookup = _build_axis_lookup(toml.load(str(toml_path)))
        except Exception:
            pass

        hkl_cfg = getattr(self.pva_reader, 'config', {}).get('HKL', {})
        HKL_IN_CONFIG = data.get('HKL_IN_CONFIG', False) or bool(hkl_cfg)
        merged_metadata = data['metadata']
        ds_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True) if compress else {}

        with h5py.File(file_path, 'w') as h5f:
            entry = h5f.create_group('entry')
            data_grp = entry.create_group('data')
            data_grp.create_dataset('data',
                                    data=np.array([np.reshape(img, data['shape']) for img in data['images']]),
                                    **ds_kwargs)

            metadata_grp = data_grp.create_group('metadata')

            # Write custom CA metadata to entry/data/metadata/ca_custom/
            custom_ca = {}
            try:
                custom_ca = settings.METADATA_CA.get('CUSTOM', {}) or {}
            except Exception:
                pass
            if custom_ca:
                ca_custom_grp = metadata_grp.create_group('ca')
                for friendly_name, pv_name in custom_ca.items():
                    try:
                        values = merged_metadata.get(pv_name)
                        if values is None:
                            continue
                        arr = np.array(values)
                        if arr.dtype.kind in ('i', 'u', 'f') and arr.size > 0:
                            ca_custom_grp.create_dataset(friendly_name, data=arr)
                    except Exception:
                        pass

            # Build axis-label -> per-frame values for HKL POSITION lookup
            motor_pos_values = {}
            for k, v in merged_metadata.items():
                if is_position_pv(k):
                    label = _derive_axis_from_pv(k, axis_lookup)
                    if label and label not in motor_pos_values:
                        motor_pos_values[label] = v

            if HKL_IN_CONFIG:
                hkl_root = metadata_grp.create_group('HKL')

                for section_name in ['PRIMARY_BEAM_DIRECTION', 'INPLANE_REFERENCE_DIRECITON', 'SAMPLE_SURFACE_NORMAL_DIRECITON']:
                    sec = hkl_cfg.get(section_name, {})
                    if sec:
                        sec_grp = hkl_root.create_group(section_name)
                        for k, pv in sec.items():
                            self._write_scan_pv_dataset(sec_grp, k, pv, merged_metadata)

                for base in ['SAMPLE_CIRCLE_AXIS_1', 'SAMPLE_CIRCLE_AXIS_2', 'SAMPLE_CIRCLE_AXIS_3', 'SAMPLE_CIRCLE_AXIS_4', 'DETECTOR_CIRCLE_AXIS_1', 'DETECTOR_CIRCLE_AXIS_2']:
                    sec = hkl_cfg.get(base, {})
                    if sec:
                        grp = hkl_root.create_group(base)
                        for k, pv in sec.items():
                            if k == 'POSITION':
                                axis_label = _derive_axis_from_pv(pv, axis_lookup)
                                vals = motor_pos_values.get(axis_label)
                                if vals is not None:
                                    arr = np.array(vals)
                                    if arr.dtype.kind in ('i', 'u', 'f'):
                                        grp.create_dataset(k, data=arr)
                            else:
                                self._write_scan_pv_dataset(grp, k, pv, merged_metadata)

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
                            spec_grp.create_dataset('UB_MATRIX_VALUE', data=arr[:9] if arr.size >= 9 else arr)

                detector = hkl_cfg.get('DETECTOR_SETUP', {})
                if detector:
                    det_grp = hkl_root.create_group('DETECTOR_SETUP')
                    for k, pv in detector.items():
                        self._write_scan_pv_dataset(det_grp, k, pv, merged_metadata)

            rsm = data.get('rsm')
            if HKL_IN_CONFIG and rsm:
                try:
                    if len(rsm[0]) == data['len_images']:
                        hkl_grp = data_grp.create_group('hkl')
                        hkl_grp.create_dataset('qx', data=np.array([np.reshape(qx, data['shape']) for qx in rsm[0]]), **ds_kwargs)
                        hkl_grp.create_dataset('qy', data=np.array([np.reshape(qy, data['shape']) for qy in rsm[1]]), **ds_kwargs)
                        hkl_grp.create_dataset('qz', data=np.array([np.reshape(qz, data['shape']) for qz in rsm[2]]), **ds_kwargs)
                except Exception:
                    pass

            self._apply_nx_structure(h5f, entry)

    def _apply_nx_structure(self, h5f: h5py.File, entry: h5py.Group, base_group: str = "entry/data/metadata"):
        """Apply NeXus NX_class attributes and create instrument/sample structural groups."""
        nx_def = settings.HDF5_STRUCTURE['nexus']['default']
        nx_entry = nx_def['entry']

        h5f.attrs['NX_class'] = nx_def['NX_class']
        h5f.attrs['default'] = nx_def['default']
        entry.attrs['NX_class'] = nx_entry['NX_class']
        entry.attrs['default'] = nx_entry['default']

        if 'data' in entry:
            nx_data = nx_entry['data']
            entry['data'].attrs['NX_class'] = nx_data['NX_class']
            entry['data'].attrs['signal'] = nx_data['signal']

        instr_cfg = nx_entry['instrument']
        instr_grp = entry.require_group('instrument')
        instr_grp.attrs['NX_class'] = instr_cfg['NX_class']
        src_grp = instr_grp.require_group('source')
        src_grp.attrs['NX_class'] = instr_cfg['source']['NX_class']
        det_grp = instr_grp.require_group('detector')
        det_grp.attrs['NX_class'] = instr_cfg['detector']['NX_class']
        if 'data' not in det_grp:
            det_grp['data'] = h5py.SoftLink(instr_cfg['detector']['data_link'])

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

    def merge_metadata(self, attributes):
        merged_metadata = {}
        for attribute_dict in attributes:
            for key, value in attribute_dict.items():
                if not (key == 'RSM' or key == 'Analysis'):
                    if key not in merged_metadata:
                        merged_metadata[key] = []
                        merged_metadata[key].append(value)
                    else:
                        merged_metadata[key].append(value)
        return merged_metadata

    # --- Scan format helpers (mirror of HDF5Handler methods) ---

    def _parse_toml_reverse_map(self) -> dict:
        """Build a reverse HKL map: PV key -> (group_name, field_name)."""
        reverse_map: dict = {}
        try:
            cfg = getattr(self.pva_reader, 'config', {}) or {}
            hkl_cfg = cfg.get('HKL', {})
            for group_name, fields in hkl_cfg.items():
                if isinstance(fields, dict):
                    for field_name, pv_key in fields.items():
                        if isinstance(pv_key, str) and pv_key:
                            reverse_map[pv_key] = (str(group_name).lower(), str(field_name).lower())
        except Exception:
            pass
        return reverse_map

    def _get_structured_attr(self, attr: dict, hkl_reverse_map: dict) -> dict:
        """Group a single frame's attribute dict into hkl/rois/motor_positions sections."""
        structured: dict = {'hkl': {}, 'rois': {}, 'motor_positions': {}, 'metadata': {}}
        if not isinstance(attr, dict):
            return structured
        for key, value in attr.items():
            if key in hkl_reverse_map:
                grp, field = hkl_reverse_map[key]
                structured['hkl'].setdefault(grp, {})[field] = value
                continue
            if 'ROI' in key:
                parts = key.split(':')
                if len(parts) >= 3 and parts[1].startswith('ROI'):
                    structured['rois'].setdefault(parts[1], {})[parts[2]] = value
                    continue
            if 'Position' in key:
                structured['motor_positions'][key] = value
                continue
            structured['metadata'][key] = value
        return structured

    def _build_hkl_series(self, attributes_list: list, hkl_reverse_map: dict) -> dict:
        """Aggregate per-frame HKL groups into series: {group: {field: [values...]}}."""
        grouped_series: dict = {}
        for attr in attributes_list:
            grouped = self._get_structured_attr(attr, hkl_reverse_map)
            for grp, fields in grouped['hkl'].items():
                for field, val in fields.items():
                    grouped_series.setdefault(grp, {}).setdefault(field, []).append(val)
        return grouped_series

    def _write_scan_pv_dataset(self, group, name: str, pv_key: str, merged_metadata: dict):
        """Write a single PV-keyed dataset into group under name."""
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

    def save_scan_to_h5(self, file_path: str, compress: bool = True, clear_caches: bool = True) -> None:
        """Write caches in NeXus scan format (NXdetector + NXsample + motor positions).

        Layout (entry/):
          instrument/detector/data   <- image stack
          instrument/detector/DETECTOR_SETUP/
          instrument/source/
          sample/rotation_angle      <- primary axis values
          hkl/<group>/<field>        <- HKL grouped series
          data/                      (NXdata, soft-linked to detector/data + rotation_angle)
        """
        all_caches = self.pva_reader.get_all_caches(clear_caches=clear_caches)
        images = all_caches.get('images') or []
        attributes = all_caches.get('attributes') or []
        shape = self.pva_reader.get_shape()

        nx_conf = settings.HDF5_STRUCTURE['nexus']['scans']
        formatter = settings.HDF5_STRUCTURE['nexus']['format']
        entry_cfg = nx_conf['entry']

        hkl_reverse_map = self._parse_toml_reverse_map()

        # Merge metadata for DETECTOR_SETUP writes
        merged_metadata = self.merge_metadata(attributes)

        with h5py.File(file_path, 'w') as h5_file:
            # Root
            h5_file.attrs['NX_class'] = nx_conf['NX_class']
            h5_file.attrs['default'] = nx_conf['default']

            # Entry
            entry = h5_file.create_group(entry_cfg['name'])
            entry.attrs['NX_class'] = entry_cfg['NX_class']
            entry.attrs['default'] = entry_cfg['default']

            # Instrument / Detector
            instr_cfg = entry_cfg['instrument']
            instr_grp = entry.create_group(instr_cfg['name'])
            instr_grp.attrs['NX_class'] = instr_cfg['NX_class']

            det_cfg = instr_cfg['detector']
            det_grp = instr_grp.create_group(det_cfg['name'])
            det_grp.attrs['NX_class'] = det_cfg['NX_class']
            ds_kwargs = hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True) if compress else {}
            det_grp.create_dataset(det_cfg['field'],
                                   data=np.array([np.reshape(img, shape) for img in images]),
                                   **ds_kwargs)

            # DETECTOR_SETUP from TOML HKL config
            try:
                det_setup_cfg = getattr(self.pva_reader, 'config', {}).get('HKL', {}).get('DETECTOR_SETUP', {})
                if isinstance(det_setup_cfg, dict) and attributes:
                    setup_grp = det_grp.create_group('DETECTOR_SETUP')
                    for field_name, pv_key in det_setup_cfg.items():
                        self._write_scan_pv_dataset(setup_grp, str(field_name), pv_key, merged_metadata)
            except Exception:
                pass

            # Source (Energy)
            src_cfg = instr_cfg['source']
            src_grp = instr_grp.create_group(src_cfg['name'])
            src_grp.attrs['NX_class'] = src_cfg['NX_class']
            if src_cfg['energy']['value'] is not None:
                en_ds = src_grp.create_dataset('energy', data=src_cfg['energy']['value'])
                en_ds.attrs['units'] = src_cfg['energy']['units']

            # Sample / rotation_angle (primary scan axis from first position PV)
            sample_cfg = entry_cfg['sample']
            sample_grp = entry.create_group(sample_cfg['name'])
            sample_grp.attrs['NX_class'] = sample_cfg['NX_class']
            primary_axis_values = []
            if attributes:
                pos_keys = [k for k in attributes[0].keys() if 'Position' in k]
                for attr in attributes:
                    v = next((float(attr[k]) for k in pos_keys
                              if isinstance(attr.get(k), (int, float, np.number))), 0.0)
                    primary_axis_values.append(v)
            if primary_axis_values:
                sample_grp.create_dataset(sample_cfg['field'],
                                          data=np.array(primary_axis_values, dtype=np.float64))

            # Data group (NXdata) with soft links
            data_cfg = entry_cfg['data']
            data_grp = entry.create_group(data_cfg['name'])
            data_grp.attrs['NX_class'] = data_cfg['NX_class']
            data_grp.attrs['signal'] = data_cfg['signal']
            data_grp.attrs['axes'] = data_cfg['axes']
            data_grp.attrs[f"{data_cfg['axes']}_indices"] = 0
            data_grp['data'] = h5py.SoftLink(f'/{entry_cfg["name"]}/instrument/detector/data')
            data_grp['rotation_angle'] = h5py.SoftLink(f'/{entry_cfg["name"]}/sample/rotation_angle')

            # HKL grouped series
            hkl_series = self._build_hkl_series(attributes, hkl_reverse_map)
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

        self.hdf5_writer_finished.emit(
            f"Saved to: {file_path}\nFormat: {formatter['name']}"
        )


