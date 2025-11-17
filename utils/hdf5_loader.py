import h5py
import numpy as np
from typing import Tuple, Optional, Union
import os
from pathlib import Path
import traceback
import sys, pathlib
import hdf5plugin
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
                    data_ds = f['entry/data/data']
                    arr = data_ds[()]
                    raw_data['images'] = arr
                    if arr.ndim == 3:
                        raw_data['num_images'] = arr.shape[0]
                        raw_data['image_shape'] = (arr.shape[1], arr.shape[2])
                    elif arr.ndim == 2:
                        raw_data['num_images'] = 1
                        raw_data['image_shape'] = (arr.shape[0], arr.shape[1])
                    else:
                        raw_data['num_images'] = 0
                        raw_data['image_shape'] = (0, 0)
                
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
                    return False
            
            # Check shapes match
            shapes = [data['qx'].shape, data['qy'].shape, data['qz'].shape, data['images'].shape]
            if not all(shape == shapes[0] for shape in shapes):
                return False
            
            # Check arrays are not empty
            if data['qx'].size == 0:
                return False
            
            return True
            
        except Exception as e:
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
        try:
            
            # Validate file
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid file: {self.last_error}")
            
            # Load raw data
            raw_data = self._load_hdf5_data(file_path)
            if not raw_data:
                raise ValueError("Failed to load data from file")
            
            # Validate for 3D operations
            if not self._validate_for_3d(raw_data):
                raise ValueError("File does not contain valid 3D coordinate data")
            
            # Process coordinate data
            qx_flat = self._flatten_coordinate_data(raw_data['qx'])
            qy_flat = self._flatten_coordinate_data(raw_data['qy'])
            qz_flat = self._flatten_coordinate_data(raw_data['qz'])
            
            # Check if flattening worked
            if len(qx_flat) == 0 or len(qy_flat) == 0 or len(qz_flat) == 0:
                raise ValueError("Coordinate flattening resulted in empty arrays")
            
            # Create 3D points array
            points_3d = np.column_stack([qx_flat, qy_flat, qz_flat])
            
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
        Load HDF5 file as 3D volume (or 2D slice if saved that way) using the standard structure.
        Reads:
          - /entry/data/data as the array
          - /entry attrs (e.g., data_type)
          - /entry/data attrs (array_rank, array_shape)
          - /entry/data/metadata datasets (voxel_spacing, grid_origin, volume_shape, original_shape, etc.)
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            Tuple containing:
                - volume (np.ndarray): 3D volume data (D, H, W) or 2D slice (H, W)
                - shape (Tuple[int, int, int]): Volume dimensions (or 2D shape with leading 0)
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")
            if not h5py.is_hdf5(file_path):
                raise ValueError(f"Not a valid HDF5 file: {file_path}")
            
            meta: dict = {}
            with h5py.File(file_path, 'r') as f:
                # Basic path checks
                if 'entry' not in f or 'entry/data' not in f or 'entry/data/data' not in f:
                    raise ValueError("Required HDF5 paths missing (expected /entry/data/data)")
                
                entry_grp = f['entry']
                data_grp = f['entry/data']
                data_ds = data_grp['data']
                
                # Read array and shape
                volume = data_ds[()]  # numpy array
                vol_shape = volume.shape  # 3D for volume; 2D for slice
                
                # Read attributes for quick discovery
                try:
                    meta['data_type'] = entry_grp.attrs.get('data_type', '')
                except Exception:
                    meta['data_type'] = ''
                try:
                    meta['array_rank'] = int(data_grp.attrs.get('array_rank', volume.ndim))
                except Exception:
                    meta['array_rank'] = volume.ndim
                try:
                    arr_shape_attr = data_grp.attrs.get('array_shape', np.array(vol_shape, dtype=np.int64))
                    meta['array_shape'] = list(np.array(arr_shape_attr, dtype=np.int64).tolist())
                except Exception:
                    meta['array_shape'] = list(vol_shape)
                
                # Read metadata datasets if present
                voxel_spacing = None
                grid_origin = None
                original_shape = None
                volume_shape_ds = None
                if 'entry/data/metadata' in f:
                    md_grp = f['entry/data/metadata']
                    for key in md_grp.keys():
                        try:
                            ds = md_grp[key]
                            # Try to read as string safely, otherwise as numeric/array
                            if hasattr(ds, 'asstr'):
                                val = ds.asstr()[()]
                            else:
                                val = ds[()]
                            # Normalize numpy types to python types
                            if isinstance(val, np.ndarray):
                                meta[key] = val.tolist()
                            elif isinstance(val, (np.generic,)):
                                meta[key] = val.item()
                            else:
                                meta[key] = val
                        except Exception:
                            # Fall back to stringified content if any issue
                            try:
                                meta[key] = str(md_grp[key][()])
                            except Exception:
                                pass
                    voxel_spacing = meta.get('voxel_spacing', None)
                    grid_origin = meta.get('grid_origin', None)
                    original_shape = meta.get('original_shape', None)
                    volume_shape_ds = meta.get('volume_shape', None)
                
                # Store in instance for downstream consumers
                self.volume = volume
                # Always store a 3-tuple for volume_shape; if 2D, prefix with 0
                if volume.ndim == 3:
                    self.volume_shape = tuple(int(x) for x in vol_shape)
                elif volume.ndim == 2:
                    self.volume_shape = (0, int(vol_shape[0]), int(vol_shape[1]))
                else:
                    self.volume_shape = (0, 0, 0)
                # Keep file-level metadata for access by caller
                self.file_metadata = {
                    'data_type': (meta.get('data_type') or ''),
                    'array_rank': meta.get('array_rank'),
                    'array_shape': meta.get('array_shape'),
                    'voxel_spacing': voxel_spacing,
                    'grid_origin': grid_origin,
                    'original_shape': original_shape,
                    'volume_shape': volume_shape_ds,
                    'num_images': meta.get('num_images', None),
                    # Persisted grid reconstruction hints
                    'array_order': meta.get('array_order', None),
                    'grid_dimensions_cells': meta.get('grid_dimensions_cells', None),
                    'axes_labels': meta.get('axes_labels', None),
                    'intensity_range': meta.get('intensity_range', None),
                }
            
            # Return volume and a 3D shape; for 2D, return (0, H, W) so callers can detect
            return (volume, self.volume_shape)
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

    def load_vti_volume_3d(self, file_path: str, scalar_name: Optional[str] = None, prefer_cell_data: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Load a .vti (VTK XML ImageData) file and return a cell-centered 3D numpy volume (D, H, W).
        No saving performed. Populates self.volume, self.volume_shape, and self.file_metadata.

        Args:
            file_path: Path to the .vti file
            scalar_name: Optional name of the scalar to use; if None, uses active or first available
            prefer_cell_data: Prefer cell_data arrays (best for slicing); if False and only point_data exists, will convert

        Returns:
            (volume, shape): numpy array (D,H,W) and shape tuple
        """
        try:
            import pyvista as pv  # Lazy import to keep dependency optional
            self.current_file_path = file_path

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")

            grid = pv.read(file_path)

            # Select scalar from cell_data (preferred) or point_data
            chosen_name = None
            intens_1d = None
            target_obj_for_dims = grid  # object whose dimensions we use

            cell_keys = list(grid.cell_data.keys()) if hasattr(grid, "cell_data") else []
            point_keys = list(grid.point_data.keys()) if hasattr(grid, "point_data") else []

            def _pick_name(keys: list, active: Optional[str], requested: Optional[str]) -> Optional[str]:
                if requested and requested in keys:
                    return requested
                if active and active in keys:
                    return active
                return keys[0] if keys else None

            if prefer_cell_data and cell_keys:
                chosen_name = _pick_name(cell_keys, getattr(grid, "active_scalars_name", None), scalar_name)
                if not chosen_name:
                    raise ValueError("No scalar arrays found in VTI cell_data")
                intens_1d = np.asarray(grid.cell_data[chosen_name])
                target_obj_for_dims = grid
            else:
                # Try point_data first and convert to cell_data for consistent D×H×W slicing
                if point_keys:
                    chosen_name = _pick_name(point_keys, getattr(grid, "active_scalars_name", None), scalar_name)
                    if not chosen_name:
                        raise ValueError("No scalar arrays found in VTI point_data")
                    v_cd = grid.point_data_to_cell_data(pass_point_data=False)
                    intens_1d = np.asarray(v_cd.cell_data[chosen_name])
                    target_obj_for_dims = v_cd
                elif cell_keys:
                    # Fallback to cell_data if point_data missing
                    chosen_name = _pick_name(cell_keys, getattr(grid, "active_scalars_name", None), scalar_name)
                    intens_1d = np.asarray(grid.cell_data[chosen_name])
                    target_obj_for_dims = grid
                else:
                    raise ValueError("No scalar arrays found in VTI (point_data or cell_data)")

            # Derive cell-centered dimensions from points-based dimensions (cells = points - 1)
            dims_points = tuple(int(d) for d in getattr(target_obj_for_dims, "dimensions", (0, 0, 0)))
            dims_cells = tuple(max(d - 1, 0) for d in dims_points)

            expected = int(np.prod(dims_cells)) if all(d > 0 for d in dims_cells) else 0
            if expected <= 0:
                raise ValueError(f"Invalid VTI dimensions (points={dims_points}, cells={dims_cells})")
            if intens_1d.size != expected:
                # Attempt robust conversion via point_data_to_cell_data if mismatch
                try:
                    v_cd = target_obj_for_dims.point_data_to_cell_data(pass_point_data=False) if hasattr(target_obj_for_dims, "point_data_to_cell_data") else grid.point_data_to_cell_data(pass_point_data=False)
                    intens_1d = np.asarray(v_cd.cell_data[chosen_name])
                    dims_points = tuple(int(d) for d in getattr(v_cd, "dimensions", (0, 0, 0)))
                    dims_cells = tuple(max(d - 1, 0) for d in dims_points)
                    expected = int(np.prod(dims_cells)) if all(d > 0 for d in dims_cells) else 0
                except Exception:
                    pass
            if intens_1d.size != expected:
                raise ValueError(f"Scalar size {intens_1d.size} does not match expected cells product {expected}")

            # Reshape to D×H×W using Fortran order to match VTK/PyVista layout (x-fastest, z-slowest)
            volume = intens_1d.reshape(dims_cells, order="F").astype(np.float32)

            # Store in instance
            self.volume = volume
            if len(dims_cells) == 3:
                self.volume_shape = (int(dims_cells[2]), int(dims_cells[1]), int(dims_cells[0])) if False else tuple(int(x) for x in dims_cells)  # keep (D,H,W)
            elif len(dims_cells) == 2:
                self.volume_shape = (0, int(dims_cells[0]), int(dims_cells[1]))
            else:
                self.volume_shape = (0, 0, 0)

            spacing = getattr(grid, "spacing", (1.0, 1.0, 1.0))
            origin = getattr(grid, "origin", (0.0, 0.0, 0.0))
            intensity_range = [float(np.min(volume)), float(np.max(volume))] if volume.size > 0 else [0.0, 0.0]

            # Populate metadata to mirror HDF5 loader expectations
            self.file_metadata = {
                "data_type": "volume",
                "array_rank": int(volume.ndim),
                "array_shape": list(volume.shape),
                "voxel_spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
                "grid_origin": [float(origin[0]), float(origin[1]), float(origin[2])],
                "original_shape": [int(volume.shape[1]), int(volume.shape[2])] if volume.ndim == 3 else ([int(volume.shape[0]), int(volume.shape[1])] if volume.ndim == 2 else [0, 0]),
                "volume_shape": list(volume.shape) if volume.ndim == 3 else None,
                "num_images": 1,
                "array_order": "F",
                "grid_dimensions_cells": [int(x) for x in dims_cells],
                "axes_labels": ["H", "K", "L"],
                "intensity_range": intensity_range,
                "scalar_name": chosen_name,
            }

            return (volume, self.volume_shape)
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return (np.array([]), (0, 0, 0))

    def load_volume_auto(self, file_path: str, scalar_name: Optional[str] = None) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """
        Convenience loader that accepts .h5/.hdf5 or .vti and routes to the appropriate method.
        No saving performed.

        Args:
            file_path: Path to input file (.h5/.hdf5 or .vti)
            scalar_name: Optional scalar name for .vti inputs

        Returns:
            (volume, shape): numpy array and shape tuple
        """
        try:
            ext = str(Path(file_path).suffix).lower()
            if ext in (".h5", ".hdf5"):
                return self.load_h5_volume_3d(file_path)
            elif ext == ".vti":
                return self.load_vti_volume_3d(file_path, scalar_name=scalar_name)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        except Exception as e:
            self._handle_loading_error(e, file_path)
            return (np.array([]), (0, 0, 0))

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
    def save_vol_to_h5(self, file_path: str, points: np.ndarray, intensities: np.ndarray, 
                         metadata: Optional[dict] = None) -> bool:
        """
        Save volume data to HDF5 file
        
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
                'data_type': 'points',
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
                    dtype=np.float32,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
                )
                print('Intensity data written to standard images path')
                
                # Create HKL subgroup using standard structure
                hkl_grp = data_grp.create_group(self.hdf5_structure['hkl'].split('/')[-1])
                
                # Save coordinates using standard HKL paths
                hkl_grp.create_dataset(
                    self.hdf5_structure['qx'].split('/')[-1], 
                    data=points[:, 0], 
                    dtype=np.float32,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
                )
                hkl_grp.create_dataset(
                    self.hdf5_structure['qy'].split('/')[-1], 
                    data=points[:, 1], 
                    dtype=np.float32,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
                )
                hkl_grp.create_dataset(
                    self.hdf5_structure['qz'].split('/')[-1], 
                    data=points[:, 2], 
                    dtype=np.float32,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True)
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
    
    def save_vol_to_h5(self, file_path: str, volume: np.ndarray, 
                         metadata: Optional[dict] = None) -> bool:
        """
        Save 3D volume (or 2D slice) to HDF5 file using standard structure.
        Writes the array to /entry/data/data and metadata to /entry/data/metadata.
        
        Args:
            file_path (str): Output file path
            volume (np.ndarray): Volume array. Shape should be:
                                 - (D, H, W) for 3D volumes
                                 - (H, W) for 2D slices
            metadata (dict, optional): Additional metadata to save. Will be merged
                                       with defaults (including data_type).
        Returns:
            bool: True if save successful
        """
        try:
            if volume is None or volume.size == 0:
                raise ValueError("Volume array cannot be empty")
            if volume.ndim not in (2, 3):
                raise ValueError(f"Volume must be 2D or 3D, got ndim={volume.ndim}")
            
            # Prepare metadata and infer data_type if not provided
            meta = {} if metadata is None else dict(metadata)
            inferred_type = 'volume' if volume.ndim == 3 else 'slice'
            meta.setdefault('data_type', inferred_type)
            meta.setdefault('creation_timestamp', str(np.datetime64('now')))
            meta.setdefault('source_file', getattr(self, 'current_file_path', 'unknown'))
            if volume.ndim == 3:
                meta.setdefault('volume_shape', tuple(int(x) for x in volume.shape))
            else:
                meta.setdefault('slice_shape', tuple(int(x) for x in volume.shape))
            
            # Create HDF5 structure and write data
            with h5py.File(file_path, 'w') as h5f:
                # /entry
                entry_grp = h5f.create_group(self.hdf5_structure['entry'])
                # /entry/data
                data_grp = entry_grp.create_group(self.hdf5_structure['data'].split('/')[-1])
                # /entry/data/data -> write as float32 for consistency
                data_ds_name = self.hdf5_structure['images'].split('/')[-1]
                data_grp.create_dataset(data_ds_name, data=volume.astype(np.float32))
                
                # /entry/data/metadata
                metadata_grp = data_grp.create_group(self.hdf5_structure['metadata'].split('/')[-1])
                for key, value in meta.items():
                    try:
                        if isinstance(value, (int, float, np.number)):
                            metadata_grp.create_dataset(key, data=value)
                        elif isinstance(value, str):
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=value, dtype=dt)
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            if len(value) > 0:
                                if all(isinstance(v, (int, float, np.number)) for v in value):
                                    metadata_grp.create_dataset(key, data=np.array(value))
                                elif all(isinstance(v, str) for v in value):
                                    dt = h5py.string_dtype(encoding='utf-8')
                                    metadata_grp.create_dataset(key, data=np.array(value, dtype=dt))
                                else:
                                    dt = h5py.string_dtype(encoding='utf-8')
                                    metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                        else:
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                    except Exception as e:
                        print(f"Warning: Could not save metadata key '{key}': {e}")
                
                # Attributes for quick discovery
                entry_grp.attrs['data_type'] = meta.get('data_type', inferred_type)
                data_grp.attrs['array_rank'] = volume.ndim
                data_grp.attrs['array_shape'] = np.array(volume.shape, dtype=np.int64)
            
            print(f"{meta.get('data_type', inferred_type).capitalize()} successfully saved to {file_path}")
            print(f"Structure paths used:")
            print(f"  Entry: {self.hdf5_structure['entry']}")
            print(f"  Data: {self.hdf5_structure['data']}")
            print(f"  Images: {self.hdf5_structure['images']}")
            print(f"  Metadata: {self.hdf5_structure['metadata']}")
            return True
        except Exception as e:
            error_msg = f"Failed to save volume to {file_path}: {e}"
            print(error_msg)
            self._handle_saving_error(e, file_path)
            return False
    
    def extract_slice(self, file_path: str, points: np.ndarray, intensities: np.ndarray,
                      metadata: Optional[dict] = None, shape: Optional[Tuple[int, int]] = None) -> bool:
        """
        Save a 2D slice derived from scattered 3D points into an HDF5 file, keeping the structure
        consistent with HDF5Writer.save_caches_to_h5:
          - /entry/data/data -> 2D image (H, W)
          - /entry/data/hkl/qx,qy,qz -> 2D grids of coordinates (H, W)
          - /entry/data/metadata -> slice and provenance metadata

        Args:
            file_path: Output HDF5 path
            points: (N, 3) slice points in 3D
            intensities: (N,) intensity values
            metadata: dict containing at least 'slice_normal' and 'slice_origin' if available
            shape: desired 2D shape (H, W) for the slice image; if None, inferred

        Returns:
            True on success, False otherwise
        """
        try:
            if points is None or points.size == 0:
                raise ValueError("Slice points array cannot be empty")
            if intensities is None or intensities.size == 0:
                raise ValueError("Slice intensities array cannot be empty")
            if points.shape[0] != intensities.shape[0]:
                raise ValueError("Points and intensities must have the same number of elements")
            if points.shape[1] != 3:
                raise ValueError("Points must be 3D (N, 3)")

            meta = {} if metadata is None else dict(metadata)

            # Plane basis from metadata or fallback
            n = np.array(meta.get('slice_normal', [0.0, 0.0, 1.0]), dtype=float)
            n_norm = np.linalg.norm(n)
            if not np.isfinite(n_norm) or n_norm <= 0.0:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm
            o = np.array(meta.get('slice_origin', [0.0, 0.0, 0.0]), dtype=float)
            if not np.all(np.isfinite(o)):
                # Fallback to centroid of points
                o = np.mean(points, axis=0)

            # Choose an axis not parallel to n to build in-plane basis u,v
            world_axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            ref = world_axes[0]
            for ax in world_axes:
                if abs(float(np.dot(ax, n))) < 0.9:
                    ref = ax
                    break
            u = np.cross(n, ref)
            u_norm = np.linalg.norm(u)
            if not np.isfinite(u_norm) or u_norm <= 0.0:
                # Fallback if numerical degeneracy
                ref = np.array([0.0, 1.0, 0.0])
                u = np.cross(n, ref)
                u_norm = np.linalg.norm(u)
                if not np.isfinite(u_norm) or u_norm <= 0.0:
                    # Absolute fallback
                    u = np.array([1.0, 0.0, 0.0])
                    u_norm = 1.0
            u = u / u_norm
            v = np.cross(n, u)
            v_norm = np.linalg.norm(v)
            if not np.isfinite(v_norm) or v_norm <= 0.0:
                v = np.array([0.0, 1.0, 0.0])

            # Project points onto plane coordinates (U,V)
            rel = points - o[None, :]
            U = rel.dot(u)
            V = rel.dot(v)

            # Determine slice image shape (H, W)
            preferred_shape = None
            # Prefer provided shape
            if shape and isinstance(shape, (tuple, list)) and len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
                preferred_shape = (int(shape[0]), int(shape[1]))
            else:
                # Fallback to metadata original_shape if valid
                orig = meta.get('original_shape', None)
                if isinstance(orig, (tuple, list)) and len(orig) == 2 and int(orig[0]) > 0 and int(orig[1]) > 0:
                    preferred_shape = (int(orig[0]), int(orig[1]))
            # Final fallback to a reasonable default
            if preferred_shape is None:
                preferred_shape = (512, 512)
            H, W = preferred_shape

            # Compute bin edges over U,V extents
            U_min, U_max = float(np.min(U)), float(np.max(U))
            V_min, V_max = float(np.min(V)), float(np.max(V))
            # Avoid zero-size ranges
            if not np.isfinite(U_min) or not np.isfinite(U_max) or U_max == U_min:
                U_min, U_max = -0.5, 0.5
            if not np.isfinite(V_min) or not np.isfinite(V_max) or V_max == V_min:
                V_min, V_max = -0.5, 0.5

            # Bin points into HxW grid, averaging intensities per bin
            du = (U_max - U_min) / float(W)
            dv = (V_max - V_min) / float(H)
            # Guard against zero or NaN
            if not np.isfinite(du) or du <= 0.0:
                du = 1.0 / float(max(W, 1))
            if not np.isfinite(dv) or dv <= 0.0:
                dv = 1.0 / float(max(H, 1))
            iu = np.floor((U - U_min) / du).astype(int)
            iv = np.floor((V - V_min) / dv).astype(int)
            iu = np.clip(iu, 0, W - 1)
            iv = np.clip(iv, 0, H - 1)

            image_sum = np.zeros((H, W), dtype=np.float64)
            image_cnt = np.zeros((H, W), dtype=np.int64)
            # Accumulate
            for k in range(points.shape[0]):
                image_sum[iv[k], iu[k]] += float(intensities[k])
                image_cnt[iv[k], iu[k]] += 1
            # Average; fill empty bins with 0.0
            image = np.zeros((H, W), dtype=np.float32)
            nonzero = image_cnt > 0
            image[nonzero] = (image_sum[nonzero] / image_cnt[nonzero]).astype(np.float32)
            image[~nonzero] = 0.0

            # Build qx,qy,qz grids at bin centers
            Uc = U_min + (np.arange(W, dtype=np.float64) + 0.5) * du
            Vc = V_min + (np.arange(H, dtype=np.float64) + 0.5) * dv
            # Broadcast to HxW
            U_grid = np.broadcast_to(Uc[None, :], (H, W))
            V_grid = np.broadcast_to(Vc[:, None], (H, W))
            # q = o + U*u + V*v
            qx = (o[0] + U_grid * u[0] + V_grid * v[0]).astype(np.float32)
            qy = (o[1] + U_grid * u[1] + V_grid * v[1]).astype(np.float32)
            qz = (o[2] + U_grid * u[2] + V_grid * v[2]).astype(np.float32)

            # Prepare metadata consistent with writer
            intensity_range = [float(np.min(image)), float(np.max(image))]
            meta.setdefault('data_type', 'slice')
            meta.setdefault('extraction_timestamp', str(np.datetime64('now')))
            meta.setdefault('num_points', int(points.shape[0]))
            meta.setdefault('image_shape', [int(H), int(W)])
            meta.setdefault('slice_normal', [float(n[0]), float(n[1]), float(n[2])])
            meta.setdefault('slice_origin', [float(o[0]), float(o[1]), float(o[2])])
            meta.setdefault('u_axis', [float(u[0]), float(u[1]), float(u[2])])
            meta.setdefault('v_axis', [float(v[0]), float(v[1]), float(v[2])])
            meta.setdefault('u_range', [float(U_min), float(U_max)])
            meta.setdefault('v_range', [float(V_min), float(V_max)])
            meta.setdefault('intensity_range', intensity_range)

            # Write HDF5 file: /entry/data/data, /entry/data/hkl/{qx,qy,qz}, /entry/data/metadata
            with h5py.File(file_path, 'w') as h5f:
                entry_grp = h5f.create_group(self.hdf5_structure['entry'])
                data_grp = entry_grp.create_group(self.hdf5_structure['data'].split('/')[-1])
                # Image data
                data_ds_name = self.hdf5_structure['images'].split('/')[-1]
                data_grp.create_dataset(data_ds_name, data=image.astype(np.float32),
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                # HKL 2D grids
                hkl_grp = data_grp.create_group(self.hdf5_structure['hkl'].split('/')[-1])
                hkl_grp.create_dataset(self.hdf5_structure['qx'].split('/')[-1], data=qx,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                hkl_grp.create_dataset(self.hdf5_structure['qy'].split('/')[-1], data=qy,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                hkl_grp.create_dataset(self.hdf5_structure['qz'].split('/')[-1], data=qz,
                    **hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=True))
                # Metadata
                metadata_grp = data_grp.create_group(self.hdf5_structure['metadata'].split('/')[-1])
                for key, value in meta.items():
                    try:
                        if isinstance(value, (int, float, np.number)):
                            metadata_grp.create_dataset(key, data=value)
                        elif isinstance(value, str):
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=value, dtype=dt)
                        elif isinstance(value, (list, tuple, np.ndarray)):
                            arr = np.array(value)
                            if arr.dtype.kind in ('i', 'u', 'f'):
                                metadata_grp.create_dataset(key, data=arr)
                            elif arr.dtype.kind in ('U', 'S', 'O'):
                                dt = h5py.string_dtype(encoding='utf-8')
                                metadata_grp.create_dataset(key, data=arr.astype(dt))
                            else:
                                dt = h5py.string_dtype(encoding='utf-8')
                                metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                        else:
                            dt = h5py.string_dtype(encoding='utf-8')
                            metadata_grp.create_dataset(key, data=str(value), dtype=dt)
                    except Exception as e:
                        print(f"Warning: Could not save metadata key '{key}': {e}")

                # Attributes
                entry_grp.attrs['data_type'] = 'slice'
                data_grp.attrs['array_rank'] = 2
                data_grp.attrs['array_shape'] = np.array([H, W], dtype=np.int64)

            print(f"Slice successfully saved to {file_path} (shape {H}x{W})")
            print(f"Structure paths used:")
            print(f"  Entry: {self.hdf5_structure['entry']}")
            print(f"  Data: {self.hdf5_structure['data']}")
            print(f"  Images: {self.hdf5_structure['images']}")
            print(f"  HKL: {self.hdf5_structure['hkl']} -> qx,qy,qz grids")
            print(f"  Metadata: {self.hdf5_structure['metadata']}")
            return True
        except Exception as e:
            error_msg = f"Failed to save slice to {file_path}: {e}"
            print(error_msg)
            self._handle_saving_error(e, file_path)
            return False

    # ================ UTILITY METHODS ======================= #
    def get_file_info(self, file_path: str, *, style: str = "text", include_unknown: bool = True,
                       float_precision: int = 6, summarize_datasets: bool = True, raw: bool = False):
        """
        Inspect an HDF5 file and return a formatted summary.

        Args:
            file_path: Path to HDF5 file
            style: "text" (default) for human-readable string, or "dict" for a grouped dict.
            include_unknown: Include unrecognized metadata keys under 'other_metadata'.
            float_precision: Decimal places for float formatting in text output.
            summarize_datasets: If True, highlight /entry/data/data and summarize others; else list all.
            raw: If True, return the original raw dict (backward compatibility).

        Returns:
            - If raw=True: the original raw dict identical to prior implementation.
            - If style="text": a formatted multiline string.
            - If style="dict": a grouped dict with sections.
        """
        # Step 1: Build the raw info (same schema as before)
        info = {
            'valid': False,
            'data_type': '',
            'paths': [],
            'shapes': {},
            'dtypes': {},
            'entry_attrs': {},
            'data_attrs': {},
            'metadata': {}
        }
        try:
            if not os.path.exists(file_path) or not h5py.is_hdf5(file_path):
                return info if raw else self._format_file_info(info, style=style, include_unknown=include_unknown,
                                                              float_precision=float_precision, summarize_datasets=summarize_datasets)

            with h5py.File(file_path, 'r') as f:
                info['valid'] = True
                # entry attrs
                if 'entry' in f:
                    for k, v in f['entry'].attrs.items():
                        info['entry_attrs'][k] = v
                    info['data_type'] = str(info['entry_attrs'].get('data_type', ''))
                # data attrs
                if 'entry/data' in f:
                    for k, v in f['entry/data'].attrs.items():
                        info['data_attrs'][k] = v
                # metadata datasets
                if 'entry/data/metadata' in f:
                    md_grp = f['entry/data/metadata']
                    for key in md_grp.keys():
                        try:
                            ds = md_grp[key]
                            if hasattr(ds, 'asstr'):
                                val = ds.asstr()[()]
                            else:
                                val = ds[()]
                            if isinstance(val, np.ndarray):
                                info['metadata'][key] = val.tolist()
                            elif isinstance(val, (np.generic,)):
                                info['metadata'][key] = val.item()
                            else:
                                info['metadata'][key] = val
                        except Exception:
                            pass
                # datasets enumeration
                def visitor(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        p = '/' + name
                        info['paths'].append(p)
                        info['shapes'][p] = obj.shape
                        info['dtypes'][p] = str(obj.dtype)
                f.visititems(visitor)

            # Step 2: Return raw if requested
            if raw:
                return info

            # Step 3: Return formatted output
            return self._format_file_info(info, style=style, include_unknown=include_unknown,
                                          float_precision=float_precision, summarize_datasets=summarize_datasets)

        except Exception as e:
            # Also log to error file
            try:
                with open(self.log_file_path, 'a') as f:
                    f.write("==== HDF5 Inspect Error ====\n")
                    f.write(f"File Path: {file_path}\n")
                    f.write(f"Error: {repr(e)}\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
            except Exception:
                pass
            return info if raw else self._format_file_info(info, style=style, include_unknown=include_unknown,
                                                           float_precision=float_precision, summarize_datasets=summarize_datasets)

    # Internal helper to format file info; kept private within this class.
    def _format_file_info(self, info: dict, *, style: str = "text", include_unknown: bool = True,
                          float_precision: int = 6, summarize_datasets: bool = True):
        # Helper conversions
        def _to_native(x):
            try:
                if isinstance(x, bytes):
                    return x.decode('utf-8', errors='ignore')
                if isinstance(x, (np.generic,)):
                    return x.item()
                if isinstance(x, np.ndarray):
                    return [_to_native(v) for v in x.tolist()]
                if isinstance(x, (list, tuple)):
                    return [_to_native(v) for v in x]
                return x
            except Exception:
                return x

        def _fmt_num(v):
            try:
                fv = float(v)
                return f"{fv:.{int(float_precision)}f}"
            except Exception:
                return str(v)

        def _fmt_value(v):
            v = _to_native(v)
            if isinstance(v, float):
                return _fmt_num(v)
            if isinstance(v, (list, tuple)):
                return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
            return str(v)

        # Extract common fields
        md = {k: _to_native(v) for k, v in (info.get('metadata') or {}).items()}
        entry_attrs = {k: _to_native(v) for k, v in (info.get('entry_attrs') or {}).items()}
        data_attrs = {k: _to_native(v) for k, v in (info.get('data_attrs') or {}).items()}

        # Derive summary
        array_rank = data_attrs.get('array_rank', None)
        array_shape = data_attrs.get('array_shape', None)
        if isinstance(array_shape, np.ndarray):
            array_shape = array_shape.tolist()
        # grid/volume keys of interest
        voxel_spacing = md.get('voxel_spacing', None)
        grid_origin = md.get('grid_origin', None)
        grid_cells = md.get('grid_dimensions_cells', None)
        grid_points = None
        try:
            if isinstance(grid_cells, (list, tuple)) and len(grid_cells) == 3:
                grid_points = [int(grid_cells[0]) + 1, int(grid_cells[1]) + 1, int(grid_cells[2]) + 1]
        except Exception:
            grid_points = None
        array_order = md.get('array_order', None)
        axes_labels = md.get('axes_labels', None)
        volume_shape = md.get('volume_shape', None)
        original_shape = md.get('original_shape', None)
        intensity_range = md.get('intensity_range', None)
        num_images = md.get('num_images', None)

        # Build datasets summary
        datasets = {}
        paths = info.get('paths') or []
        shapes = info.get('shapes') or {}
        dtypes = info.get('dtypes') or {}
        primary_path = '/entry/data/data'
        if primary_path in paths:
            datasets[primary_path] = {
                'shape': shapes.get(primary_path),
                'dtype': dtypes.get(primary_path)
            }
        if not summarize_datasets:
            for p in paths:
                if p not in datasets:
                    datasets[p] = {'shape': shapes.get(p), 'dtype': dtypes.get(p)}
        else:
            # summarize count of others
            others = [p for p in paths if p != primary_path]
            if others:
                datasets['(other_datasets)'] = {'count': len(others)}

        # Determine recognized metadata keys to filter "other_metadata"
        recognized = {
            'voxel_spacing', 'grid_origin', 'grid_dimensions_cells', 'array_order',
            'axes_labels', 'volume_shape', 'original_shape', 'intensity_range', 'num_images',
            'array_rank', 'array_shape'
        }
        other_metadata = {}
        if include_unknown:
            for k, v in md.items():
                if k not in recognized:
                    other_metadata[k] = v

        # Grouped dict
        grouped = {
            'summary': {
                'valid': bool(info.get('valid', False)),
                'data_type': info.get('data_type', ''),
                'array_rank': array_rank,
                'array_shape': array_shape
            },
            'grid': {
                'voxel_spacing': voxel_spacing,
                'grid_origin': grid_origin,
                'grid_dimensions_cells': grid_cells,
                'grid_dimensions_points': grid_points,
                'array_order': array_order,
                'axes_labels': axes_labels
            },
            'volume': {
                'volume_shape': volume_shape,
                'original_shape': original_shape,
                'intensity_range': intensity_range,
                'num_images': num_images
            },
            'datasets': datasets,
            'entry_attrs': entry_attrs,
            'data_attrs': data_attrs,
            'other_metadata': other_metadata if include_unknown else {}
        }

        if style == "dict":
            return grouped

        # Default: style == "text"
        lines = []
        def _section(title):
            lines.append(f"{title}:")
        def _kv(label, value):
            if value is None or value == {} or value == []:
                return
            lines.append(f"  - {label}: {_fmt_value(value)}")

        _section("Summary")
        _kv("Valid", grouped['summary'].get('valid'))
        _kv("Data Type", grouped['summary'].get('data_type'))
        _kv("Array Rank", grouped['summary'].get('array_rank'))
        _kv("Array Shape", grouped['summary'].get('array_shape'))

        _section("Grid")
        _kv("Voxel Spacing (ΔH, ΔK, ΔL)", grouped['grid'].get('voxel_spacing'))
        _kv("Grid Origin (H0, K0, L0)", grouped['grid'].get('grid_origin'))
        _kv("Grid Dimensions (cells)", grouped['grid'].get('grid_dimensions_cells'))
        _kv("Grid Dimensions (points)", grouped['grid'].get('grid_dimensions_points'))
        _kv("Array Order", grouped['grid'].get('array_order'))
        _kv("Axes Labels", grouped['grid'].get('axes_labels'))

        _section("Volume/Image")
        _kv("Volume Shape (D,H,W)", grouped['volume'].get('volume_shape'))
        _kv("Original Shape (H,W)", grouped['volume'].get('original_shape'))
        _kv("Intensity Range", grouped['volume'].get('intensity_range'))
        _kv("Num Images", grouped['volume'].get('num_images'))

        _section("Datasets")
        if primary_path in datasets:
            ds = datasets[primary_path]
            _kv(f"{primary_path} shape", ds.get('shape'))
            _kv(f"{primary_path} dtype", ds.get('dtype'))
        if '(other_datasets)' in datasets:
            _kv("Other datasets count", datasets['(other_datasets)'].get('count'))
        elif not summarize_datasets:
            for p, ds in datasets.items():
                if p == primary_path:
                    continue
                _kv(f"{p} shape", ds.get('shape'))
                _kv(f"{p} dtype", ds.get('dtype'))

        _section("Entry Attributes")
        for k, v in grouped['entry_attrs'].items():
            _kv(k, v)

        _section("Data Attributes")
        for k, v in grouped['data_attrs'].items():
            _kv(k, v)

        if include_unknown and grouped.get('other_metadata'):
            _section("Other Metadata")
            for k, v in grouped['other_metadata'].items():
                _kv(k, v)

        return "\n".join(lines)
    
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
    
    def get_last_error(self) -> str:
        """Return the last recorded error message."""
        return self.last_error

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
        f = open('error_output.txt', 'w')
        f.write(f'Error Saving HDF5 File: {str(error)}\nHDF5 File Path: {file_path}\n\nTraceback:\n{traceback.format_exc()}')
        f.close()
        print(self.last_error)
