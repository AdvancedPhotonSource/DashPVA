import h5py
import numpy as np
from typing import Tuple, Optional, Union
import os
from pathlib import Path
import traceback
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

class HDF5Loader:
    """
    Utility class for loading and saving HDF5 files with 2D and 3D point cloud data
    """
    
    def __init__(self):
        """Initialize the HDF5 loader"""
        # ================ FILE HANDLING ======================= #
        self.current_file_path = None
        self.output_file_location = None
        
        # ================ DATA STORAGE ======================= #
        self.raw_data = None
        self.points_2d = None
        self.points_3d = None
        self.intensities = None
        self.images = None
        self.volume = None
        
        # ================ METADATA ======================= #
        self.num_images = 0
        self.image_shape = (0, 0)  # (height, width)
        self.volume_shape = (0, 0, 0)  # (depth, height, width)
        self.data_type = None  # '2d', '3d', 'volume', 'images'
        self.file_metadata = {}
        
        self.hdf5_structure = {
            'entry': 'entry',
            'data': 'entry/data',
            'images': 'entry/data/data',
            'metadata': 'entry/data/metadata',
            'motor_positions': 'entry/data/metadata/motor_positions',
            'rois': 'entry/data/rois',
            'hkl': 'entry/data/hkl',
            'qx': 'entry/data/hkl/qx',
            'qy': 'entry/data/hkl/qy',
            'qz': 'entry/data/hkl/qz',
            'analysis': 'entry/analysis',
            'intensity': 'entry/analysis/intensity',
            'comx': 'entry/analysis/comx',
            'comy': 'entry/analysis/comy'
        }
        # ================== ERROR ================= #
        self.debug_mode = True
        self.last_error = ''
        self.flatten_intensities = True
        self.auto_calculate_bounds = True
        self.validate_coordinates = True
        self.error_log = []
        self.log_file_path = 'hdf5_loader_errors.txt'
        self.coordinates_loaded = False
        self.intensities_loaded = False
        self.points_assembled = False
        
    # ================ BASE LOADING METHODS ======================= #
    def _load_hdf5_data(self, file_path: str) -> dict:
        """
        Base method to load HDF5 file and return raw data structure
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            dict: Raw data structure from HDF5 file
        """
        try:
            self.current_file_path = file_path
            raw_data = {}
            
            with h5py.File(file_path, 'r') as f:
                # Load all available datasets that we might need
                
                # Try to load coordinate data (for 3D)
                if 'entry/data/hkl/qx' in f:
                    raw_data['qx'] = f['entry/data/hkl/qx'][:]
                if 'entry/data/hkl/qy' in f:
                    raw_data['qy'] = f['entry/data/hkl/qy'][:]
                if 'entry/data/hkl/qz' in f:
                    raw_data['qz'] = f['entry/data/hkl/qz'][:]
                
                # Load image data (for both 2D and 3D)
                if 'entry/data/data' in f:
                    raw_data['images'] = f['entry/data/data'][:]
                    raw_data['num_images'] = f['entry/data/data'].shape[0]
                    raw_data['image_shape'] = (f['entry/data/data'].shape[1], 
                                            f['entry/data/data'].shape[2])
                
                # Load any metadata
                raw_data['metadata'] = {}
                for group_name in ['entry', 'entry/data', 'entry/data/metadata']:
                    if group_name in f:
                        for key, value in f[group_name].attrs.items():
                            raw_data['metadata'][f"{group_name}_{key}"] = value
            
            return raw_data
            
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return {}
    
    def _validate_hdf5_structure(self, data: dict) -> bool:
        """
        Validate that the HDF5 file has the expected structure
        
        Args:
            data (dict): Raw data from HDF5 file
            
        Returns:
            bool: True if structure is valid
        """
        try:
            # Check all required data exists
            required_keys = ['qx', 'qy', 'qz', 'images']
            for key in required_keys:
                if key not in data or data[key] is None:
                    return False
            
            # Check shapes match
            qx_shape = data['qx'].shape
            qy_shape = data['qy'].shape
            qz_shape = data['qz'].shape
            images_shape = data['images'].shape
            
            if not (qx_shape == qy_shape == qz_shape == images_shape):
                return False
            
            # Check not empty
            if qx_shape[0] == 0:
                return False
            
            # Store validated info
            self.num_images = data['num_images']
            self.original_shape = data['image_shape']
            
            return True
            
        except Exception as e:
            return False
        
    def validate_file(self, file_path: str) -> bool:
        """
        Basic file validation - works for any HDF5 file
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            bool: True if file is a valid HDF5 file
        """
        try:
            print('Validating File')
            # Basic file checks
            if not os.path.exists(file_path):
                self.last_error = f"File does not exist: {file_path}"
                return False
            
            if not h5py.is_hdf5(file_path):
                self.last_error = f"Not a valid HDF5 file: {file_path}"
                return False
            
            # Try to open and check for basic structure
            with h5py.File(file_path, 'r') as f:
                # Check for at least image data
                if 'entry/data/data' not in f:
                    self.last_error = "Missing required image data path: entry/data/data"
                    return False
            print('file valid')
            return True
            
        except Exception as e:
            self.last_error = f"File validation failed: {e}"
            print(self.last_error)
            return False
    
    
    def _validate_for_3d(self, data: dict) -> bool:
        """
        Validate that loaded data can be used for 3D operations
        
        Args:
            data (dict): Raw data from _load_hdf5_data
            
        Returns:
            bool: True if data is valid for 3D operations
        """
        try:
            # Check for 3D coordinate data
            required_3d = ['qx', 'qy', 'qz', 'images']
            for key in required_3d:
                if key not in data or data[key] is None:
                    if self.debug_mode:
                        print(f"Missing required 3D key: {key}")
                    return False
            
            # Check shapes match
            shapes = [data['qx'].shape, data['qy'].shape, data['qz'].shape, data['images'].shape]
            if not all(shape == shapes[0] for shape in shapes):
                if self.debug_mode:
                    print(f"Shape mismatch: qx{data['qx'].shape}, qy{data['qy'].shape}, qz{data['qz'].shape}, images{data['images'].shape}")
                return False
            
            # Check arrays are not empty
            if data['qx'].size == 0:
                if self.debug_mode:
                    print("Coordinate arrays are empty")
                return False
            
            if self.debug_mode:
                print(f"3D validation passed: {data['qx'].shape} points")
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"3D validation error: {e}")
            return False

    def _validate_for_2d(self, data: dict) -> bool:
        """
        Validate that loaded data can be used for 2D operations
        
        Args:
            data (dict): Raw data from _load_hdf5_data
            
        Returns:
            bool: True if data is valid for 2D operations
        """
        # For 2D, we just need image data
        if 'images' not in data or data['images'] is None:
            return False
        
        # Check images have proper dimensions
        if len(data['images'].shape) != 3:  # (num_images, height, width)
            return False
        
        return True
    
    # ================ 2D LOADING METHODS ======================= #
    def load_h5_to_2d(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        """
        Load HDF5 file to 2D points
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - points (np.ndarray): 2D points array (N, 2)
                - intensities (np.ndarray): Intensity values
                - num_images (int): Number of images
                - shape (Tuple[int, int]): Image dimensions (height, width)
        """
        pass
    
    def load_h5_images_2d(self, file_path: str) -> Tuple[np.ndarray, int, Tuple[int, int]]:
        """
        Load HDF5 file as 2D image stack
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - images (np.ndarray): Image stack (N, H, W)
                - num_images (int): Number of images
                - shape (Tuple[int, int]): Image dimensions (height, width)
        """
        pass
    
    # ================ 3D LOADING METHODS ======================= #
    def load_h5_to_3d(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        """
        Load HDF5 file to 3D points
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - points (np.ndarray): 3D points array (N, 3)
                - intensities (np.ndarray): Intensity values
                - num_images (int): Number of images
                - shape (Tuple[int, int]): Original image dimensions
        """
        print('Loading to ')
        try:
            if self.debug_mode:
                print(f"Loading 3D data from: {file_path}")
            
            # Validate file
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid file: {self.last_error}")
            
            # Load raw data
            raw_data = self._load_hdf5_data(file_path)
            if not raw_data:
                raise ValueError("Failed to load data from file")
            
            if self.debug_mode:
                print(f"Raw data keys: {list(raw_data.keys())}")
                if 'qx' in raw_data:
                    print(f"QX shape: {raw_data['qx'].shape}")
                if 'images' in raw_data:
                    print(f"Images shape: {raw_data['images'].shape}")
            
            # Validate for 3D operations
            if not self._validate_for_3d(raw_data):
                raise ValueError("File does not contain valid 3D coordinate data")
            
            # Process coordinate data
            qx_flat = self._flatten_coordinate_data(raw_data['qx'])
            qy_flat = self._flatten_coordinate_data(raw_data['qy'])
            qz_flat = self._flatten_coordinate_data(raw_data['qz'])
            
            if self.debug_mode:
                print(f"Flattened coordinates: qx={len(qx_flat)}, qy={len(qy_flat)}, qz={len(qz_flat)}")
            
            # Check if flattening worked
            if len(qx_flat) == 0 or len(qy_flat) == 0 or len(qz_flat) == 0:
                raise ValueError("Coordinate flattening resulted in empty arrays")
            
            # Create 3D points array
            points_3d = np.column_stack([qx_flat, qy_flat, qz_flat])
            
            if self.debug_mode:
                print(f"Created 3D points array: {points_3d.shape}")
            
            # Process intensity data
            if self.flatten_intensities:
                intensities = np.reshape(raw_data['images'], -1)
            else:
                intensities = raw_data['images']
            
            # Final validation
            if points_3d.size == 0:
                raise ValueError("Final 3D points array is empty")
            
            # Store in class variables
            self.points_3d = points_3d
            self.intensities = intensities
            self.num_images = raw_data['num_images']
            self.original_shape = raw_data['image_shape']
            
            return (points_3d, intensities, raw_data['num_images'], raw_data['image_shape'])
            
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return (np.array([]), np.array([]), 0, (0, 0))
    
    def load_h5_volume_3d(self, file_path: str) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Load HDF5 file as 3D volume
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - volume (np.ndarray): 3D volume data (D, H, W)
                - shape (Tuple[int, int, int]): Volume dimensions
        """
        try:
            # Validate and load
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid file: {self.last_error}")
            
            raw_data = self._load_hdf5_data(file_path)
            if not raw_data or 'images' not in raw_data:
                raise ValueError("No image data found")
            
            # Return images as volume (don't flatten)
            volume = raw_data['images']
            volume_shape = volume.shape  # (depth, height, width)
            
            # Store volume data
            self.volume = volume
            self.volume_shape = volume_shape
            
            return (volume, volume_shape)
            
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return (np.array([]), (0, 0, 0))
    
    def load_h5_with_coordinates(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load HDF5 file with coordinate transformations (qx, qy, qz)
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - points (np.ndarray): Transformed 3D points
                - intensities (np.ndarray): Intensity values
                - metadata (dict): Additional metadata from file
        """
        try:
            # Use the main 3D loader
            points, intensities, num_images, shape = self.load_h5_to_3d(file_path)
            
            # Load raw data again to get metadata
            raw_data = self._load_hdf5_data(file_path)
            
            # Prepare metadata dictionary
            metadata = {
                'num_images': num_images,
                'original_shape': shape,
                'coordinate_bounds': self.bounds_3d.copy(),
                'file_metadata': raw_data.get('metadata', {}),
                'coordinate_arrays': {
                    'qx_shape': raw_data['qx'].shape,
                    'qy_shape': raw_data['qy'].shape,
                    'qz_shape': raw_data['qz'].shape
                }
            }
            
            return (points, intensities, metadata)
            
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return (np.array([]), np.array([]), {})

    # ================ HELPER METHODS FOR 3D LOADING ======================= #
    def _flatten_coordinate_data(self, coord_array: np.ndarray) -> np.ndarray:
        """
        Flatten coordinate array from (num_images, height, width) to (N,)
        
        Args:
            coord_array (np.ndarray): Coordinate array to flatten
            
        Returns:
            np.ndarray: Flattened coordinate array
        """
        try:
            # Concatenate all images into single array
            flattened_list = []
            for i in range(coord_array.shape[0]):
                flattened_list.append(np.reshape(coord_array[i], -1))
            
            return np.concatenate(flattened_list)
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error flattening coordinate data: {e}")
            return np.array([])

    def _calculate_3d_bounds(self) -> None:
        """
        Calculate and store 3D bounds from loaded points
        """
        try:
            if self.points_3d is not None and len(self.points_3d) > 0:
                self.bounds_3d['x_min'] = np.min(self.points_3d[:, 0])
                self.bounds_3d['x_max'] = np.max(self.points_3d[:, 0])
                self.bounds_3d['y_min'] = np.min(self.points_3d[:, 1])
                self.bounds_3d['y_max'] = np.max(self.points_3d[:, 1])
                self.bounds_3d['z_min'] = np.min(self.points_3d[:, 2])
                self.bounds_3d['z_max'] = np.max(self.points_3d[:, 2])
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error calculating 3D bounds: {e}")
    
    
    # ================ SAVING METHODS ======================= #
    def save_3d_to_h5(self, file_path: str, points: np.ndarray, intensities: np.ndarray, 
                         metadata: Optional[dict] = None) -> bool:
        """
        Save point cloud data to HDF5 file
        
        Args:
            file_path (str): Output file path
            points (np.ndarray): Point coordinates
            intensities (np.ndarray): Intensity values
            metadata (dict, optional): Additional metadata to save
            
        Returns:
            bool: True if save successful
        """
        try:
            # Validate inputs
            if points is None or points.size == 0:
                raise ValueError("Points array cannot be empty")
            
            if intensities is None or intensities.size == 0:
                raise ValueError("Intensities array cannot be empty")
            
            if len(points) != len(intensities):
                raise ValueError("Points and intensities must have the same length")
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add default metadata
            default_metadata = {
                'num_points': len(points),
                'point_dimensions': points.shape[1],
                'data_type': '3d_point',
                'creation_timestamp': str(np.datetime64('now')),
                'source_file': getattr(self, 'current_file_path', 'unknown')
            }
            
            merged_metadata = {**default_metadata, **metadata}
            
            with h5py.File(file_path, 'w') as h5f:
                # Create main structure using hdf5_structure paths
                print(f'Creating file at: {file_path}')
                
                # Create entry group
                entry_grp = h5f.create_group(self.hdf5_structure['entry'])
                
                # Create data group
                data_grp = entry_grp.create_group(self.hdf5_structure['data'].split('/')[-1])
                
                # Save intensities using standard images path
                data_grp.create_dataset(
                    self.hdf5_structure['images'].split('/')[-1], 
                    data=intensities.reshape(-1, 1), 
                    dtype=np.float32
                )
                print('Intensity data written to standard images path')
                
                # Create HKL subgroup using standard structure
                hkl_grp = data_grp.create_group(self.hdf5_structure['hkl'].split('/')[-1])
                
                # Save coordinates using standard HKL paths
                hkl_grp.create_dataset(
                    self.hdf5_structure['qx'].split('/')[-1], 
                    data=points[:, 0], 
                    dtype=np.float32
                )
                hkl_grp.create_dataset(
                    self.hdf5_structure['qy'].split('/')[-1], 
                    data=points[:, 1], 
                    dtype=np.float32
                )
                hkl_grp.create_dataset(
                    self.hdf5_structure['qz'].split('/')[-1], 
                    data=points[:, 2], 
                    dtype=np.float32
                )
                print('HKL coordinates written to standard paths')
                
                # Create metadata group using standard structure
                metadata_grp = data_grp.create_group(self.hdf5_structure['metadata'].split('/')[-1])
                print('Metadata group created using standard path')
                
                # Save metadata
                for key, value in merged_metadata.items():
                    try:
                        if isinstance(value, (int, float, np.number)):
                            metadata_grp.create_dataset(key, data=value)
                        elif isinstance(value, str):
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=value, dtype=dt)
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            # Handle arrays/lists
                            if len(value) > 0:
                                if all(isinstance(v, (int, float, np.number)) for v in value):
                                    metadata_grp.create_dataset(key, data=np.array(value))
                                elif all(isinstance(v, str) for v in value):
                                    dt = h5py.string_dtype(encoding='utf-8')
                                    metadata_grp.create_dataset(key, data=np.array(value, dtype=dt))
                                else:
                                    # Mixed types, convert to string
                                    dt = h5py.string_dtype(encoding='utf-8')
                                    metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                        else:
                            # Convert to string for complex objects
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                    except Exception as e:
                        print(f"Warning: Could not save metadata key '{key}': {e}")
                
                print('Metadata saved')
                
                # Add file-level attributes
                entry_grp.attrs['data_type'] = '3d_point_cloud'
                entry_grp.attrs['num_points'] = len(points)
                entry_grp.attrs['point_dimensions'] = points.shape[1]
                
                # Add data group attributes
                data_grp.attrs['coordinate_system'] = 'hkl'
                data_grp.attrs['units'] = 'reciprocal_space'
            
            print(f"3D point cloud successfully saved using standard HDF5 structure")
            print(f"Saved {len(points)} points with 3D coordinates")
            print(f"Structure paths used:")
            print(f"  Entry: {self.hdf5_structure['entry']}")
            print(f"  Data: {self.hdf5_structure['data']}")
            print(f"  Images: {self.hdf5_structure['images']}")
            print(f"  HKL: {self.hdf5_structure['hkl']}")
            print(f"  QX: {self.hdf5_structure['qx']}")
            print(f"  QY: {self.hdf5_structure['qy']}")
            print(f"  QZ: {self.hdf5_structure['qz']}")
            print(f"  Metadata: {self.hdf5_structure['metadata']}")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to save point cloud to {file_path}: {e}"
            print(error_msg)
            self._handle_saving_error(e, file_path)
            return False
    
    
    def save_images_to_h5(self, file_path: str, images: np.ndarray, 
                         coordinates: Optional[dict] = None, metadata: Optional[dict] = None) -> bool:
        """
        Save image stack to HDF5 file
        
        Args:
            file_path (str): Output file path
            images (np.ndarray): Image stack
            coordinates (dict, optional): Coordinate transformation data (qx, qy, qz)
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: True if save successful
        """
        pass
    
    def save_volume_to_h5(self, file_path: str, volume: np.ndarray, 
                         metadata: Optional[dict] = None) -> bool:
        """
        Save 3D volume to HDF5 file
        
        Args:
            file_path (str): Output file path
            volume (np.ndarray): 3D volume data
            metadata (dict, optional): Additional metadata
            
        Returns:
            bool: True if save successful
        """
        pass
    
    # ================ UTILITY METHODS ======================= #
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about HDF5 file structure and contents
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            dict: File information including datasets, shapes, dtypes
        """
        pass
    
    def convert_2d_to_3d(self, points_2d: np.ndarray, intensities: np.ndarray, 
                        z_values: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D points to 3D by adding Z dimension
        
        Args:
            points_2d (np.ndarray): 2D points (N, 2)
            intensities (np.ndarray): Intensity values
            z_values (np.ndarray, optional): Z coordinates, defaults to zeros
            
        Returns:
            Tuple containing:
                - points_3d (np.ndarray): 3D points (N, 3)
                - intensities (np.ndarray): Intensity values
        """
        pass
    
    def extract_slice_from_3d(self, points_3d: np.ndarray, intensities: np.ndarray, 
                             slice_axis: str = 'z', slice_value: float = 0.0, 
                             tolerance: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D slice from 3D point cloud
        
        Args:
            points_3d (np.ndarray): 3D points (N, 3)
            intensities (np.ndarray): Intensity values
            slice_axis (str): Axis to slice along ('x', 'y', 'z')
            slice_value (float): Value along axis to slice at
            tolerance (float): Tolerance for slice selection
            
        Returns:
            Tuple containing:
                - points_2d (np.ndarray): 2D points from slice
                - intensities_2d (np.ndarray): Corresponding intensities
        """
        pass
    
    # ================ ERROR HANDLING ======================= #
    
    def _handle_loading_error(self, error: Exception, file_path: str) -> None:
        """
        Handle errors during loading operations
        
        Args:
            error (Exception): The exception that occurred
            file_path (str): Path to file that caused error
        """
        f = open('error_output.txt', 'w')
        f.write(f'Error Loading HDF5 File: {str(error)}\nHDF5 File Path: {file_path}\n\nTraceback:\n{traceback.format_exc()}')
        f.close()
        print(self.last_error)
        
    
    def _handle_saving_error(self, error: Exception, file_path: str) -> None:
        """
        Handle errors during saving operations
        
        Args:
            error (Exception): The exception that occurred
            file_path (str): Path to file that caused error
        """
        pass