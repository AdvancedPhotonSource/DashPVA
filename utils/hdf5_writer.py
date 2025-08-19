import h5py
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
# from utils import PVAReader

class HDF5Writer(QObject):
    hdf5_writer_finished = pyqtSignal(str)
    
    def __init__(self, file_path: str, pva_reader):
        super(HDF5Writer, self).__init__()
        self.file_path = file_path
        self.pva_reader = pva_reader

    @pyqtSlot()
    def save_caches_to_h5(self) -> None:
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
        print('Calling HDF5Writer')
        try:
            config = self.pva_reader.get_config_settings()
            OUTPUT_FILE_LOCATION = config.get('OUTPUT_FILE_LOCATION', 'output.h5')
            HKL_IN_CONFIG = config.get('HKL_IN_CONFIG', False)
            all_caches = self.pva_reader.get_all_caches(clear_caches=True)
            images = all_caches['images']
            attributes = all_caches['attributes']
            rsm = all_caches['rsm']
            shape = self.pva_reader.get_shape()

            len_images = len(images)
            len_attributes = len(attributes)
            len_rsm = len(rsm[0])

            if len_images != len_attributes:
                raise ValueError("[Saving Caches] All caches must have the same number of elements.")
            if images is None or len_images == 0:
                raise ValueError("[Saving Caches] Caches cannot be empty.")

            print('attempting save')
            merged_metadata = {}
            print('merging attributes')
            for attribute_dict in attributes:
                for key, value in attribute_dict.items():
                    if not (key == 'RSM' or key == 'Analysis'):
                        if key not in merged_metadata:
                            merged_metadata[key] = []
                            merged_metadata[key].append(value)
                        else:
                            merged_metadata[key].append(value)
            print('merging complete')
            
            with h5py.File(OUTPUT_FILE_LOCATION, 'w') as h5f:
                # Create the main "images" group
                print(f'creating file at: {OUTPUT_FILE_LOCATION}')
                images_grp = h5f.create_group("entry")
                data_grp = images_grp.create_group('data')
                data_grp.create_dataset("data", data=np.array([np.reshape(img, shape) for img in images]))
                print('images written')
                metadata_grp = data_grp.create_group("metadata")
                motor_pos_grp = metadata_grp.create_group('motor_positions')
                rois_grp = data_grp.create_group('rois')
                print('metadata, rois, and motorposistion groups created')
                for key, values in merged_metadata.items():
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
                print('metadata saved')

                # Create HKL subgroup under images if HKL caches exist
                if HKL_IN_CONFIG:
                    print('attempting to save rsm attributes')
                    if rsm:
                        print('validating rsm data')
                        if not (len_rsm == len_images):
                            raise ValueError("[Saving Caches] qx, qy, and qz caches must have the same number of elements.")
                        print('saving rsm attributes')
                        hkl_grp = data_grp.create_group(name="hkl")
                        hkl_grp.create_dataset("qx", data=np.array([np.reshape(qx, shape) for qx in rsm[0]]))
                        hkl_grp.create_dataset("qy", data=np.array([np.reshape(qy, shape) for qy in rsm[1]]))
                        hkl_grp.create_dataset("qz", data=np.array([np.reshape(qz, shape) for qz in rsm[2]]))
            self.hdf5_writer_finished.emit(f"Caches successfully saved to {OUTPUT_FILE_LOCATION}")
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save caches to {OUTPUT_FILE_LOCATION}: {e}")