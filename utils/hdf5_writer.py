import h5py
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import hdf5plugin
# from utils import PVAReader

class HDF5Writer(QObject):
    hdf5_writer_finished = pyqtSignal(str)
    
    def __init__(self, file_path: str, pva_reader):
        super(HDF5Writer, self).__init__()
        self.file_path = file_path
        self.pva_reader = pva_reader
        self.default_output_file_config = {'FilePath': 'SCAN_OUTPUT.h5'}

    @pyqtSlot()
    def save_caches_to_h5(self, clear_caches:bool=True, compress=False) -> None:
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
            config = self.pva_reader.get_config_settings()
            OUTPUT_FILE_CONFIG= config.get('OUTPUT_FILE_CONFIG', self.default_output_file_config)
            HKL_IN_CONFIG = config.get('HKL_IN_CONFIG', False)
            all_caches = self.pva_reader.get_all_caches(clear_caches=clear_caches)
            images = all_caches['images']
            attributes = all_caches['attributes']
            rsm = all_caches['rsm']
            shape = self.pva_reader.get_shape()

            len_images = len(images)
            len_attributes = len(attributes)
            
            if len(OUTPUT_FILE_CONFIG) == 2:
                file_path = Path('~/DashPVA/test/').expanduser()
                if not file_path.exists():
                    file_path.mkdir(parents=True)
                file_name = Path('mytemp')
                TEMP_FILE_LOCATION = file_path.joinpath(file_name)
                
                file_path = Path(OUTPUT_FILE_CONFIG['FilePath']).expanduser()
                if not file_path.exists():
                    file_path.mkdir(parents=True)
                file_name = Path(OUTPUT_FILE_CONFIG['FileName'])
                OUTPUT_FILE_LOCATION = file_path.joinpath(file_name)
            else:
                OUTPUT_FILE_LOCATION = Path(OUTPUT_FILE_CONFIG['FilePath']).expanduser()
                if not OUTPUT_FILE_LOCATION.parent.exists():
                    OUTPUT_FILE_LOCATION.parent.mkdir(parents=True, exist_ok=True) # ensures directory exists before writing any files.
                    
            if len_images != len_attributes:
                
                min_length = min(len_images, len_attributes)
                if min_length > 0:
                    images = images[:min_length]
                    attributes = attributes[:min_length]
                    len_images = len(images)
                    len_attributes = len(attributes)
                else:
                    raise ValueError(f"[Saving Caches] Cannot fix cache mismatch - both caches would be empty. Images: {len_images}, Attributes: {len_attributes}")
            if images is None or len_images == 0:
                raise ValueError("[Saving Caches] Caches cannot be empty.")

            merged_metadata = {}
            for attribute_dict in attributes:
                for key, value in attribute_dict.items():
                    print(f"DICT ATTRS: {key} {value}")
                    if not (key == 'RSM' or key == 'Analysis'):
                        if key not in merged_metadata:
                            merged_metadata[key] = []
                            merged_metadata[key].append(value)
                        else:
                            merged_metadata[key].append(value)
            
            with h5py.File(OUTPUT_FILE_LOCATION, 'w') as h5f:
                # Create the main "images" group
                images_grp = h5f.create_group("entry")
                data_grp = images_grp.create_group('data')
                data_grp.create_dataset("data", data=np.array([np.reshape(img, shape) for img in images]), 
                                       **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                metadata_grp = data_grp.create_group("metadata")
                motor_pos_grp = metadata_grp.create_group('motor_positions')
                rois_grp = data_grp.create_group('rois')
                for key, values in merged_metadata.items():
                    print("Key",key,"Value",values)
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

            with h5py.File(TEMP_FILE_LOCATION, 'a') as h5f2:
                # Create HKL subgroup under images if HKL caches exist
                if not compress:
                    if HKL_IN_CONFIG and rsm:
                        len_rsm = len(rsm[0])
                        if rsm:
                            if not (len_rsm == len_images):
                                raise ValueError("[Saving Caches] qx, qy, and qz caches must have the same number of elements.")
                            hkl_grp = data_grp.create_group(name="hkl")
                            hkl_grp.create_dataset("qx", data=np.array([np.reshape(qx, shape) for qx in rsm[0]]), 
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                            hkl_grp.create_dataset("qy", data=np.array([np.reshape(qy, shape) for qy in rsm[1]]), 
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                            hkl_grp.create_dataset("qz", data=np.array([np.reshape(qz, shape) for qz in rsm[2]]), 
                                                **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
            print(f'HDF5 file save completed successfully: {OUTPUT_FILE_LOCATION}')
            self.hdf5_writer_finished.emit(f"Caches successfully saved to {OUTPUT_FILE_LOCATION}")
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")
