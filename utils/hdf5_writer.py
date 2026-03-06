import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
from utils.metadata_converter import convert_files_or_dir
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
            
            # if not compress:
            #     with h5py.File(TEMP_FILE_LOCATION, 'w') as h5f:
            #         # Create the main "images" group
            #         images_grp = h5f.create_group("entry")
            #         data_grp = images_grp.create_group('data')
            #         data_grp.create_dataset("data", data=np.array([np.reshape(img, shape) for img in images]), 
            #                             **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            #         metadata_grp = data_grp.create_group("metadata")
            #         motor_pos_grp = metadata_grp.create_group('motor_positions')
            #         rois_grp = data_grp.create_group('rois')
            #         for key, values in merged_metadata.items():
            #             if all(isinstance(v, (int, float, np.number)) for v in values):
            #                 if 'ROI' in key:
            #                     parts = key.split(':')
            #                     roi = parts[1]
            #                     if roi not in rois_grp.keys():
            #                         rois_grp.create_group(name=roi)
            #                     rois_grp[roi].create_dataset(key, data=np.array(values))
            #                 elif 'Position' in key:
            #                     motor_pos_grp.create_dataset(key, data=np.array(values))
            #                 else:
            #                     metadata_grp.create_dataset(key, data=np.array(values))      
            #             elif all(isinstance(v, str) for v in values):
            #                 dt = h5py.string_dtype(encoding='utf-8')
            #                 metadata_grp.create_dataset(key, data=np.array(values, dtype=dt))
            #             else:
            #                 metadata_grp.create_dataset(key, data=np.array(np.reshape(values, -1)))

            #         if HKL_IN_CONFIG and rsm:
            #             len_rsm = len(rsm[0])
            #             if rsm:
            #                 if not (len_rsm == len_images):
            #                     raise ValueError("[Saving Caches] qx, qy, and qz caches must have the same number of elements.")
            #                 hkl_grp = data_grp.create_group(name="hkl")
            #                 hkl_grp.create_dataset("qx", data=np.array([np.reshape(qx, shape) for qx in rsm[0]]), 
            #                                     **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            #                 hkl_grp.create_dataset("qy", data=np.array([np.reshape(qy, shape) for qy in rsm[1]]), 
            #                                     **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            #                 hkl_grp.create_dataset("qz", data=np.array([np.reshape(qz, shape) for qz in rsm[2]]), 
            #                                     **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            #                # removed debug prints: (TEMP) qx/qy/qz writes and finished HKL datasets
            
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

            # Log success and emit appropriate message
            try:
                if write_temp and write_output:
                    self.logger.info(f"Saved {data['len_images']} items to {TEMP_FILE_LOCATION} and {OUTPUT_FILE_LOCATION}{conversion_suffix}")
                elif write_output:
                    self.logger.info(f"Saved {data['len_images']} items to {OUTPUT_FILE_LOCATION}{conversion_suffix}")
                else:
                    self.logger.info(f"Saved {data['len_images']} items to {TEMP_FILE_LOCATION} (temp only)")
            except Exception:
                pass
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
                    elif 'Position' in key:
                        motor_pos_grp.create_dataset(key, data=np.array(values))
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
                    # if type(value) == list:
                    #     value = np.asarray(value)
                    if key not in merged_metadata:
                        merged_metadata[key] = []
                        merged_metadata[key].append(value)
                    else:
                        merged_metadata[key].append(value)
        return merged_metadata

        
