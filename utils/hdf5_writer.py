import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
import toml
from utils.metadata_converter import (
    _build_axis_lookup,
    _derive_axis_from_pv,
    _process_structure,
    _rename_motor_positions_and_link_hkl,
    is_position_pv,
    convert_files_or_dir,
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

    @pyqtSlot(bool, bool, bool)
    def save_caches_to_h5(self, clear_caches: bool = True, write_temp: bool = True, write_output: bool = True) -> None:
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

            # Always set a temp file path under OUTPUT_PATH for uncompressed copies when requested
            TEMP_FILE_LOCATION = base_out_dir.joinpath('temp_hkl_3d.h5')
            OUTPUT_FILE_LOCATION = base_out_dir.joinpath(f'OUTPUT_SCAN_{time.strftime("%Y%m%d_%H%M%S")}.h5')

            # Resolve OUTPUT_FILE_LOCATION according to provided config keys
            if 'FilePath' in OUTPUT_FILE_CONFIG and 'FileName' in OUTPUT_FILE_CONFIG:
                # Treat FilePath as a directory, FileName as filename
                file_dir = Path(str(OUTPUT_FILE_CONFIG.get('FilePath'))).expanduser()
                file_dir.mkdir(parents=True, exist_ok=True)
                file_name = Path(str(OUTPUT_FILE_CONFIG.get('FileName')))
                OUTPUT_FILE_LOCATION = file_dir.joinpath(file_name)
            elif 'FilePath' in OUTPUT_FILE_CONFIG:
                # Single FilePath can be a full path or just a filename
                fp = Path(str(OUTPUT_FILE_CONFIG.get('FilePath'))).expanduser()
                if str(fp.parent) in ('', '.', str(Path('.'))):
                    # Looks like only a filename; place it under OUTPUT_PATH
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

            # Auto-convert metadata structure per current TOML on the OUTPUT file only
            conversion_suffix = ""
            if write_output:
                try:
                    toml_path = settings.ensure_path()
                    if toml_path:
                        convert_files_or_dir(
                            toml_path=toml_path,
                            hdf5_path=str(OUTPUT_FILE_LOCATION),
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

            # Emit appropriate message (logging handled by the receiver)
            if write_temp and write_output:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {TEMP_FILE_LOCATION} and {OUTPUT_FILE_LOCATION}{conversion_suffix}")
            elif write_output:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {OUTPUT_FILE_LOCATION}{conversion_suffix}")
            else:
                self.hdf5_writer_finished.emit(f"{data['len_images']} successfully saved to {TEMP_FILE_LOCATION} (temp only)")
        except Exception as e:
            try:
                self.logger.exception(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")
            except Exception:
                pass
            self.hdf5_writer_finished.emit(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")

    def h5_save(self, file_path: str, data: dict, compress:bool=False):
        """_summary_

        Args:
            file_path (str): _description_
            data (dict): _description_
            compress (bool, optional): Defaults to False. If True, it will not save the HKL data

        Raises:
            ValueError: _description_
        """
        _mapping = {}
        _axis_lookup = {}
        try:
            _toml_path = settings.ensure_path()
            if _toml_path:
                _mapping = toml.load(str(_toml_path))
                _axis_lookup = _build_axis_lookup(_mapping)
        except Exception:
            pass

        with h5py.File(file_path, 'w') as h5f:
            # Create the main "images" group
            images_grp = h5f.create_group("entry")
            data_grp = images_grp.create_group('data')
            data_grp.create_dataset("data", data=np.array([np.reshape(img, data['shape']) for img in data['images']]),
                                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            metadata_grp = data_grp.create_group("metadata")
            motor_pos_grp = metadata_grp.create_group('motor_positions')
            rois_grp = data_grp.create_group('rois')
            for key, values in data['metadata'].items():
                if all(isinstance(v, (int, float, np.number)) for v in values):
                    if 'ROI' in key:
                        parts = key.split(':')
                        roi = parts[1]
                        if roi not in rois_grp.keys():
                            rois_grp.create_group(name=roi)
                        rois_grp[roi].create_dataset(key, data=np.array(values))
                    elif is_position_pv(key):
                        axis_label = _derive_axis_from_pv(key, _axis_lookup) if _axis_lookup else None
                        if not axis_label:
                            continue  # skip — no axis label, no flat PV names stored
                        ds = motor_pos_grp.create_dataset(axis_label, data=np.array(values))
                        ds.attrs['units'] = 'deg'
                    else:
                        metadata_grp.create_dataset(key, data=np.array(values))
                elif all(isinstance(v, str) for v in values):
                    dt = h5py.string_dtype(encoding='utf-8')
                    metadata_grp.create_dataset(key, data=np.array(values, dtype=dt))
                else:
                    metadata_grp.create_dataset(key, data=np.array(np.reshape(values, -1)))

            # Create HKL subgroup under images if HKL caches exist
            if not compress:
                shape = data['shape']
                rsm = data['rsm']
                HKL_IN_CONFIG = data['HKL_IN_CONFIG']
                if HKL_IN_CONFIG and rsm:
                    len_rsm = len(rsm[0])
                    if rsm:
                        if len_rsm != data['len_images']:
                            try:
                                self.logger.warning(
                                    f"RSM cache length ({len_rsm}) != image count ({data['len_images']}); skipping HKL write."
                                )
                            except Exception:
                                pass
                        else:
                            hkl_grp = data_grp.create_group(name="hkl")
                            hkl_grp.create_dataset("qx", data=np.array([np.reshape(qx, shape) for qx in rsm[0]]),
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                            hkl_grp.create_dataset("qy", data=np.array([np.reshape(qy, shape) for qy in rsm[1]]),
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                            hkl_grp.create_dataset("qz", data=np.array([np.reshape(qz, shape) for qz in rsm[2]]),
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))

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


