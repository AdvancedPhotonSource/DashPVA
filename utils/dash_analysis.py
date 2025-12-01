"""
Dash Analysis Module

Object-oriented analysis tools for HKL data processing, including slice extraction,
data manipulation, and visualization utilities for DashPVA.

This module provides comprehensive tools for loading, processing, and analyzing
HKL crystallographic data, including slice extraction, coordinate transformations,
and data filtering operations.
"""

import argparse
import os
from typing import Optional, Tuple, Union, List, Dict, Any
import numpy as np

# We try to keep PyVista optional; only import when building a grid or plotting
try:
    import pyvista as pv
except Exception:
    pv = None

try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.hdf5_loader import HDF5Loader
except Exception:
    try:
        from hdf5_loader import HDF5Loader
    except Exception:
        HDF5Loader=None

# Fallback reader using h5py for simple cases if HDF5Loader is unavailable or fails
try:
    import h5py
except Exception:
    h5py = None

# Optional Matplotlib for image plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import numpy as np
except Exception:
    np = None



# ============================================================================
# CLASSES
# ============================================================================

class Data:
    """
    Container class for 3D point data and intensities.
    
    This class encapsulates point cloud data along with intensity values
    for HKL crystallographic analysis.
    
    Attributes:
        points (np.ndarray): 3D point coordinates with shape (N, 3)
        intensities (np.ndarray): Intensity values with shape (N,)
    """
    
    def __init__(self, points: np.ndarray, intensities: np.ndarray, metadata: dict=None, num_images: int=0, shape: tuple=None):
        """
        Initialize Data object.
        
        Args:
            points: 3D point coordinates with shape (N, 3)
            intensities: Intensity values with shape (N,)
        """
        self.points = points
        self.intensities = intensities
        self.metadata = metadata
        self.num_images = num_images
        self.shape = shape


class LineCutData:
    """
    Container class for line cut analysis results.
    
    This class stores the results of line cut operations on slice data,
    including distance profiles, intensity values, and coordinate information.
    
    Attributes:
        distance (np.ndarray): Distance values along the line cut
        intensity (np.ndarray): Intensity values along the line cut
        H (np.ndarray): H coordinate values along the line cut
        K (np.ndarray): K coordinate values along the line cut
        endpoints (tuple): Start and end points of the line cut
        orientation (str): Orientation of the slice ('HK', 'KL', 'HL', etc.)
    """
    
    def __init__(self, distance: np.ndarray, intensity: np.ndarray, 
                 H: np.ndarray, K: np.ndarray, endpoints: tuple, orientation: str):
        """
        Initialize LineCutData object.
        
        Args:
            distance: Distance values along the line cut
            intensity: Intensity values along the line cut
            H: H coordinate values along the line cut
            K: K coordinate values along the line cut
            endpoints: Start and end points of the line cut
            orientation: Orientation of the slice
        """
        self.distance = distance
        self.intensity = intensity
        self.H = H
        self.K = K
        self.endpoints = endpoints
        self.orientation = orientation

    def get_peak(self):
        """
        Identify peak positions in the line cut data.
        
        Returns:
            Peak analysis results (to be implemented)
        """
        pass


class SliceData:
    """
    Container class for slice data and metadata.
    
    This class encapsulates slice data along with orientation information
    and provides methods for data manipulation and access.
    
    Attributes:
        data (Data): The slice data object
        orientation (int): Orientation identifier for the slice
    """
    
    def __init__(self):
        """Initialize SliceData object with default values."""
        self.data = Data(np.array([]), np.array([]))
        self.orientation = 0


class DashAnalysis:
    """
    Main analysis class for DashPVA HKL data processing.

    This class provides comprehensive tools for loading, processing, and analyzing
    HKL crystallographic data, including slice extraction, coordinate transformations,
    volume creation, and visualization utilities.

    The class is designed to be Jupyter-friendly and supports both volume and
    point cloud data formats with PyVista integration for 3D visualization.

    Attributes:
        _last_image (np.ndarray): Cached raster image from last slice operation
        _last_extent (list): Cached extent [U_min, U_max, V_min, V_max] from last slice
        _last_orientation (str): Cached orientation from last slice operation

    Usage:
        # Basic usage
        da = DashAnalysis()
        data = da.load_data('/path/to/file.h5')
        
        # Create and display volume
        vol = da.create_vol(data.points, data.intensities)
        da.show_vol(vol)
        
        # Create and analyze slices
        slice_mesh = da.slice_data(data, hkl='HK')
        da.show_slice(slice_mesh)
        
        # Perform line cuts
        line_data = da.line_cut('zero', param=(1.0, 'x'), vol=slice_mesh)
    """

    def __init__(self):
        """
        Initialize DashAnalysis with empty caches.
        
        Caches are used to store the last raster image and extent/orientation
        for efficient line cut operations without requiring volume regeneration.
        """
        # caches for last raster image and extent/orientation for line cuts
        self._last_image = None
        self._last_extent = None  # [U_min, U_max, V_min, V_max]
        self._last_orientation = None


# ============================================================================
# MAIN METHODS (Ordered by complexity/length - longest to shortest)
# ============================================================================

    def slice_data(self, data, hkl=None, normal=None, shape=(512,512), slab_thickness=None, 
                   clamp_to_bounds=True, spacing=(0.5, 0.5, 0.5), grid_origin=(0.0, 0.0, 0.0), 
                   show=True, axes=None, intensity_range=None, **kwargs):
        """
        Create a slice from either a volume or a point cloud with advanced processing options.

        This is the most comprehensive slice extraction method, supporting both volume and
        point cloud data with customizable HKL axes, integration parameters, and visualization
        options. Handles coordinate transformations, adaptive interpolation, and metadata preservation.

        Usage:
            # Traditional usage with presets
            slice_data = da.slice_data(data, hkl='HK')
            
            # Mixed HKL axes with custom orientation
            slice_data = da.slice_data(data, axes=((0.5, 0, 0), (0, 1, 0)))
            
            # Point cloud slicing with slab thickness
            slice_data = da.slice_data(point_data, hkl='KL', slab_thickness=0.2)

        Parameters:
            data: Volume or point cloud data. Supported formats:
                - Volume: pv.ImageData, np.ndarray (D,H,W), or (ndarray_volume, shape_tuple)
                - Points: (points, intensities) where points is (N,3) and intensities is (N,)
                - Data object with .points and .intensities attributes
                - Dict with 'points' and 'intensities' keys
            hkl: Slice origin or orientation preset. Options:
                - 3-vector: Used directly as slice origin coordinates
                - String preset: 'HK'/'XY', 'KL'/'YZ', 'HL'/'XZ' sets normal and uses dataset center
            normal (array-like): Slice plane normal vector (3,). Overridden by hkl presets
            shape (tuple): Resolution (rows, cols) for sampling the plane when slicing points
            slab_thickness (float): Thickness of selection slab around plane for point slicing
            clamp_to_bounds (bool): Clamp origin to dataset bounds
            spacing (tuple): Voxel spacing (ΔH, ΔK, ΔL) for grid construction from NumPy volumes
            grid_origin (tuple): Grid origin (H0, K0, L0) for grid construction from NumPy volumes
            show (bool): If True, displays the slice using show_slice
            axes: Optional HKL axis specification. Formats:
                - ((u_hkl, v_hkl),): Two 3-vectors defining in-plane axes in HKL coordinates
                - ((u_hkl, v_hkl), n_hkl): Two in-plane axes plus normal vector
                - None: Use hkl/normal parameters as before
            intensity_range (tuple): Optional (min, max) intensity bounds used to filter
                contributing data prior to slicing/interpolation. Use None for open bounds,
                e.g., (None, 500) or (100, None). Default None applies no filtering (full
                intensity range).
            **kwargs: Additional keyword arguments passed to show_slice when show=True.
                Common options include:
                - axis_display: 'hkl' (default) or 'uv' for axis label format
                - cmap: Colormap name for display
                - clim: (vmin, vmax) intensity display limits

        Returns:
            pv.PolyData: Slice mesh with field_data containing:
                - 'slice_normal': Normal vector of the slice plane
                - 'slice_origin': Origin point of the slice plane
                - 'slice_u_axis': U-axis vector in HKL coordinates (if custom axes used)
                - 'slice_v_axis': V-axis vector in HKL coordinates (if custom axes used)
                - 'slice_u_label': Formatted U-axis label (e.g., "H + K")
                - 'slice_v_label': Formatted V-axis label (e.g., "L/2")

        Raises:
            ImportError: If PyVista is not available
            TypeError: If data format is not supported
            ValueError: If slice parameters are invalid

        Examples:
            # Extract HK plane slice from volume
            hk_slice = da.slice_data(volume_data, hkl='HK')
            
            # Extract custom orientation slice from point cloud
            custom_slice = da.slice_data(
                point_data, 
                axes=((1, 1, 0), (0, 0, 1)), 
                slab_thickness=0.1
            )
            
            # High-resolution slice with specific spacing
            hr_slice = da.slice_data(
                data, hkl=(1.0, 0.5, 0.0), 
                shape=(1024, 1024), 
                spacing=(0.1, 0.1, 0.1)
            )
            
            # Slice with UV axis labels
            uv_slice = da.slice_data(
                data, hkl='HK', 
                axis_display='uv'
            )
        """
        if pv is None:
            raise ImportError("PyVista is required for slice_data()")

        import numpy as _np

        # Helper: normalize volume to pv.ImageData with cell_data['intensity']
        def _ensure_grid(_vol, _spacing=(1.0, 1.0, 1.0), _origin=(0.0, 0.0, 0.0)):
            if isinstance(_vol, pv.ImageData):
                _grid = _vol
                if ('intensity' not in _grid.cell_data) and ('intensity' in _grid.point_data):
                    _grid = _grid.point_data_to_cell_data(pass_point_data=False)
                if 'intensity' not in _grid.cell_data:
                    raise ValueError("ImageData must have cell_data['intensity'] for slicing.")
                return _grid
            if isinstance(_vol, (tuple, list)) and len(_vol) >= 1 and isinstance(_vol[0], _np.ndarray):
                _vol_np = _vol[0]
            elif isinstance(_vol, _np.ndarray):
                _vol_np = _vol
            else:
                raise TypeError("Volume must be pv.ImageData, a NumPy ndarray (D,H,W), or (ndarray_volume, shape) tuple.")
            if _vol_np.ndim != 3:
                raise ValueError("NumPy volume must be 3D shaped (D,H,W).")
            _dims_cells = _np.array(_vol_np.shape, dtype=int)
            _grid = pv.ImageData()
            _grid.dimensions = (_dims_cells + 1).tolist()
            _grid.spacing = tuple(float(x) for x in _spacing)
            _grid.origin = tuple(float(x) for x in _origin)
            _grid.cell_data['intensity'] = _np.asarray(_vol_np, dtype=_np.float32).flatten(order='F')
            return _grid

        # Helper: resolve normal and origin
        def _resolve_plane(_dataset_center, _bounds):
            # Resolve normal (from hkl preset or provided normal)
            _n = None
            if isinstance(hkl, str):
                s = hkl.strip().lower()
                if s in ('hk', 'xy'):
                    _n = _np.array([0.0, 0.0, 1.0], dtype=float)
                elif s in ('kl', 'yz'):
                    _n = _np.array([1.0, 0.0, 0.0], dtype=float)
                elif s in ('hl', 'xz'):
                    _n = _np.array([0.0, 1.0, 0.0], dtype=float)
            if _n is None:
                _n = _np.array(normal if (normal is not None) else [0.0, 0.0, 1.0], dtype=float)
            # Normalize
            nlen = float(_np.linalg.norm(_n))
            if not _np.isfinite(nlen) or nlen <= 0.0:
                _n = _np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                _n = _n / nlen

            # Resolve origin
            if isinstance(hkl, (tuple, list, _np.ndarray)) and len(hkl) == 3:
                _o = _np.array([float(hkl[0]), float(hkl[1]), float(hkl[2])], dtype=float)
            else:
                _o = _np.array(_dataset_center if _dataset_center is not None else [0.0, 0.0, 0.0], dtype=float)

            # Clamp origin to bounds
            if clamp_to_bounds and (_bounds is not None) and (len(_bounds) == 6):
                _o[0] = float(_np.clip(_o[0], _bounds[0], _bounds[1]))
                _o[1] = float(_np.clip(_o[1], _bounds[2], _bounds[3]))
                _o[2] = float(_np.clip(_o[2], _bounds[4], _bounds[5]))

            return _o, _n

        # Distinguish volume vs points
        is_volume_like = isinstance(data, pv.ImageData) or isinstance(data, np.ndarray) or (isinstance(data, (tuple, list)) and len(data) >= 1 and isinstance(data[0], np.ndarray) and data[0].ndim == 3)
        # If volume return the slice
        if is_volume_like:
            grid = _ensure_grid(data, _spacing=spacing, _origin=grid_origin)
            # Apply intensity range filter if provided (volume data)
            if intensity_range is not None:
                try:
                    if isinstance(intensity_range, (tuple, list)) and len(intensity_range) == 2:
                        _imin = None if intensity_range[0] is None else float(intensity_range[0])
                        _imax = None if intensity_range[1] is None else float(intensity_range[1])
                    else:
                        raise ValueError("intensity_range must be a (min, max) tuple")
                    
                    _arr = np.asarray(grid.cell_data['intensity']).astype(np.float32)
                    _mask = np.ones(_arr.shape, dtype=bool)
                    if _imin is not None:
                        _mask &= (_arr >= _imin)
                    if _imax is not None:
                        _mask &= (_arr <= _imax)
                    if not np.any(_mask):
                        import warnings as _warnings
                        _warnings.warn("slice_data: intensity_range excluded all voxels; leaving volume unfiltered.")
                    else:
                        _arr[~_mask] = 0.0
                        grid.cell_data['intensity'] = _arr
                except Exception:
                    # Be permissive; if anything goes wrong with filtering, continue unfiltered
                    pass
            center = getattr(grid, 'center', (0.0, 0.0, 0.0))
            origin_vec, n_vec = _resolve_plane(center, getattr(grid, 'bounds', None))
            sl = grid.slice(normal=n_vec, origin=origin_vec)
            sl.field_data['slice_normal'] = np.asarray(n_vec, dtype=float)
            sl.field_data['slice_origin'] = np.asarray(origin_vec, dtype=float)
            # Attach constant unit normals matching the slice normal
            try:
                normals_point = np.tile(np.asarray(n_vec, dtype=np.float32), (sl.n_points, 1))
                sl.point_data['Normals'] = normals_point
                try:
                    sl.point_data.set_active_normals('Normals')
                except Exception:
                    try:
                        sl.set_active_vectors('Normals')
                    except Exception:
                        pass
                if sl.n_cells > 0:
                    normals_cell = np.tile(np.asarray(n_vec, dtype=np.float32), (sl.n_cells, 1))
                    sl.cell_data['Normals'] = normals_cell
            except Exception:
                pass
            # Store a display clim on the slice similar to 3D viewer behavior
            try:
                vals = _np.asarray(sl['intensity'], dtype=float).reshape(-1)
            except Exception:
                vals = None
            disp_min = None
            disp_max = None
            try:
                if isinstance(intensity_range, (tuple, list)) and len(intensity_range) == 2:
                    imin, imax = intensity_range
                    disp_min = (float(imin) if (imin is not None) else (float(_np.nanmin(vals)) if (vals is not None and vals.size > 0) else None))
                    disp_max = (float(imax) if (imax is not None) else (float(_np.nanmax(vals)) if (vals is not None and vals.size > 0) else None))
                else:
                    if vals is not None and vals.size > 0:
                        disp_min = float(_np.nanmin(vals))
                        disp_max = float(_np.nanmax(vals))
            except Exception:
                disp_min = disp_min if disp_min is not None else None
                disp_max = disp_max if disp_max is not None else None
            try:
                if (disp_min is not None) and (disp_max is not None) and _np.isfinite(disp_min) and _np.isfinite(disp_max):
                    sl.field_data['slice_intensity_clim'] = _np.asarray([disp_min, disp_max], dtype=float)
            except Exception:
                pass
            # call show slice if requested
            if show:
                self.show_slice(sl, shape=shape, **kwargs)
            return sl

        # Treat as point cloud
        # Extract points and intensities
        points = None
        intensities = None
        # if data is input as data=(points,intensities)
        if isinstance(data, (tuple, list)) and len(data) >= 2:
            points = np.asarray(data[0], dtype=float)
            intensities = np.asarray(data[1], dtype=float).reshape(-1)
        # if input is input as data
        elif hasattr(data, 'points') and hasattr(data, 'intensities'):
            points = np.asarray(getattr(data, 'points'), dtype=float)
            intensities = np.asarray(getattr(data, 'intensities'), dtype=float).reshape(-1)
        elif isinstance(data, dict):
            points = np.asarray(data.get('points'), dtype=float)
            intensities = np.asarray(data.get('intensities'), dtype=float).reshape(-1)
        else:
            raise TypeError("Point data must be provided as (points, intensities) tuple/list, object with .points/.intensities, or {'points': ..., 'intensities': ...} dict.")

        if points is None or intensities is None or points.ndim != 2 or points.shape[1] != 3 or intensities.shape[0] != points.shape[0]:
            raise ValueError("Invalid point data: points must be (N,3) and intensities must be length N.")

        # Build cloud and bounds
        cloud = pv.PolyData(points)
        cloud['intensity'] = intensities.astype('float32')

        minb = points.min(axis=0)
        maxb = points.max(axis=0)
        bounds = (float(minb[0]), float(maxb[0]), float(minb[1]), float(maxb[1]), float(minb[2]), float(maxb[2]))
        center = ((minb + maxb) * 0.5).astype(float)

        origin_vec, n_vec = _resolve_plane(center, bounds)

        # Optional pre-filter: limit contributing points to a slab around the plane for interpolation
        rel = points - origin_vec[None, :]
        d_signed = rel.dot(n_vec)
        use_slab = slab_thickness is not None and np.isfinite(float(slab_thickness)) and float(slab_thickness) > 0.0
        if use_slab:
            tol = float(slab_thickness)
            mask_slab = np.abs(d_signed) <= tol
            # Fallback to all points if slab yields none
            if not np.any(mask_slab):
                mask_slab = np.ones(points.shape[0], dtype=bool)
        else:
            mask_slab = np.ones(points.shape[0], dtype=bool)

        # Optional intensity range filter
        if intensity_range is not None and isinstance(intensity_range, (tuple, list)) and len(intensity_range) == 2:
            try:
                _imin = None if intensity_range[0] is None else float(intensity_range[0])
                _imax = None if intensity_range[1] is None else float(intensity_range[1])
            except Exception:
                _imin = None; _imax = None
            mask_int = np.ones(intensities.shape[0], dtype=bool)
            if _imin is not None:
                mask_int &= (intensities >= _imin)
            if _imax is not None:
                mask_int &= (intensities <= _imax)
        else:
            mask_int = np.ones(intensities.shape[0], dtype=bool)

        mask_contrib = mask_slab & mask_int

        # For extent estimation, prefer slab mask (geometry) even if intensity filter removes all
        if np.any(mask_slab):
            pts_for_extent = points[mask_slab]
            vals_for_extent = intensities[mask_slab]
        else:
            pts_for_extent = points
            vals_for_extent = intensities

        # Resolve HKL axes or build default in-plane basis
        u_hkl = None
        v_hkl = None
        n_hkl = None
        
        if axes is not None:
            # Parse axes parameter: ((u_hkl, v_hkl),) or ((u_hkl, v_hkl), n_hkl)
            if isinstance(axes, (tuple, list)) and len(axes) >= 2:
                u_hkl = _np.asarray(axes[0], dtype=float)
                v_hkl = _np.asarray(axes[1], dtype=float)
                if len(axes) >= 3:
                    n_hkl = _np.asarray(axes[2], dtype=float)
                    # Normalize provided normal
                    n_len = float(_np.linalg.norm(n_hkl))
                    if _np.isfinite(n_len) and n_len > 0.0:
                        n_hkl = n_hkl / n_len
                        n_vec = n_hkl  # Override computed normal
                else:
                    # Compute normal from u_hkl × v_hkl
                    n_computed = _np.cross(u_hkl, v_hkl)
                    n_len = float(_np.linalg.norm(n_computed))
                    if _np.isfinite(n_len) and n_len > 0.0:
                        n_hkl = n_computed / n_len
                        n_vec = n_hkl  # Override computed normal
        
        if u_hkl is not None and v_hkl is not None:
            # Use provided HKL axes directly (preserving scale)
            u = u_hkl
            v = v_hkl
        else:
            # Build default orthonormal in-plane basis from normal
            world_axes = [
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            ]
            ref = world_axes[0]
            for ax in world_axes:
                if abs(float(np.dot(ax, n_vec))) < 0.9:
                    ref = ax
                    break
            u = np.cross(n_vec, ref)
            u_len = float(np.linalg.norm(u))
            if not np.isfinite(u_len) or u_len <= 0.0:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
                u = np.cross(n_vec, ref)
                u_len = float(np.linalg.norm(u))
                if not np.isfinite(u_len) or u_len <= 0.0:
                    u = np.array([1.0, 0.0, 0.0], dtype=float)
                    u_len = 1.0
            u = u / u_len
            v = np.cross(n_vec, u)
            v_len = float(np.linalg.norm(v))
            if not np.isfinite(v_len) or v_len <= 0.0:
                v = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                v = v / v_len

        # Project points used for extent, compute extents
        rel_ext = pts_for_extent - origin_vec[None, :]
        U = rel_ext.dot(u)
        V = rel_ext.dot(v)
        U_min, U_max = float(np.min(U)), float(np.max(U))
        V_min, V_max = float(np.min(V)), float(np.max(V))
        if not np.isfinite(U_min) or not np.isfinite(U_max) or U_max == U_min:
            U_min, U_max = -0.5, 0.5
        if not np.isfinite(V_min) or not np.isfinite(V_max) or V_max == V_min:
            V_min, V_max = -0.5, 0.5

        # Slight padding
        pad_u = (U_max - U_min) * 0.02
        pad_v = (V_max - V_min) * 0.02
        U_min -= pad_u
        U_max += pad_u
        V_min -= pad_v
        V_max += pad_v

        i_size = max(U_max - U_min, 1e-6)
        j_size = max(V_max - V_min, 1e-6)
        H, W = ((int(shape[0]), int(shape[1])) if (isinstance(shape, (tuple, list)) and len(shape) == 2) else tuple(getattr(data, 'metadata')['datasets']['/entry/data/data']['shape'][-2:]))
        H = max(int(H), 2)
        W = max(int(W), 2)

        # Create plane sized by extents and interpolate point data onto it
        plane = pv.Plane(center=origin_vec.tolist(), direction=n_vec.tolist(),
                         i_size=i_size, j_size=j_size, i_resolution=W-1, j_resolution=H-1)

        # Choose contributing cloud based on slab and intensity range
        if np.any(mask_contrib):
            cloud_contrib = pv.PolyData(points[mask_contrib])
            cloud_contrib['intensity'] = intensities[mask_contrib].astype('float32')
            no_contrib = False
        else:
            no_contrib = True

        # Use smart radius calculation to minimize gaps
        optimal_radius = self._calculate_smart_radius(
            pts_for_extent, 
            (U_min, U_max), 
            (V_min, V_max), 
            (H, W)
        )

        if not no_contrib:
            interp_plane = plane.interpolate(
                cloud_contrib,
                radius=optimal_radius,
                sharpness=1.5,
                null_value=0.0
            )
        else:
            # No contributing points: return a zero-intensity plane
            interp_plane = plane.copy()
            try:
                interp_plane['intensity'] = np.zeros(interp_plane.n_points, dtype=np.float32)
            except Exception:
                pass
            import warnings as _warnings
            _warnings.warn("slice_data: intensity_range and/or slab_thickness excluded all points; returning empty slice.")
        interp_plane.field_data['slice_normal'] = np.asarray(n_vec, dtype=float)
        interp_plane.field_data['slice_origin'] = np.asarray(origin_vec, dtype=float)
        # Attach constant unit normals matching the slice normal
        try:
            normals_point = np.tile(np.asarray(n_vec, dtype=np.float32), (interp_plane.n_points, 1))
            interp_plane.point_data['Normals'] = normals_point
            try:
                interp_plane.point_data.set_active_normals('Normals')
            except Exception:
                try:
                    interp_plane.set_active_vectors('Normals')
                except Exception:
                    pass
            if interp_plane.n_cells > 0:
                normals_cell = np.tile(np.asarray(n_vec, dtype=np.float32), (interp_plane.n_cells, 1))
                interp_plane.cell_data['Normals'] = normals_cell
        except Exception:
            pass
        # Store a display clim on the slice similar to 3D viewer behavior
        try:
            vals = _np.asarray(interp_plane['intensity'], dtype=float).reshape(-1)
        except Exception:
            vals = None
        disp_min = None
        disp_max = None
        try:
            if isinstance(intensity_range, (tuple, list)) and len(intensity_range) == 2:
                imin, imax = intensity_range
                disp_min = (float(imin) if (imin is not None) else (float(_np.nanmin(vals)) if (vals is not None and vals.size > 0) else None))
                disp_max = (float(imax) if (imax is not None) else (float(_np.nanmax(vals)) if (vals is not None and vals.size > 0) else None))
            else:
                if vals is not None and vals.size > 0:
                    disp_min = float(_np.nanmin(vals))
                    disp_max = float(_np.nanmax(vals))
        except Exception:
            disp_min = disp_min if disp_min is not None else None
            disp_max = disp_max if disp_max is not None else None
        try:
            if (disp_min is not None) and (disp_max is not None) and _np.isfinite(disp_min) and _np.isfinite(disp_max):
                interp_plane.field_data['slice_intensity_clim'] = _np.asarray([disp_min, disp_max], dtype=float)
        except Exception:
            pass
        
        # Store HKL axes for downstream use
        interp_plane.field_data['slice_u_axis'] = np.asarray(u, dtype=float)
        interp_plane.field_data['slice_v_axis'] = np.asarray(v, dtype=float)
        # Persist the slice resolution so downstream display/analysis can honor it
        interp_plane.field_data['slice_shape'] = _np.asarray([H, W], dtype=int)
        
        # Store HKL axis labels if available
        if u_hkl is not None:
            interp_plane.field_data['slice_u_label'] = format_hkl_axis(u_hkl)
        if v_hkl is not None:
            interp_plane.field_data['slice_v_label'] = format_hkl_axis(v_hkl)
        
        if show:
            self.show_slice(interp_plane, shape=shape, **kwargs)
        # sd = SliceData(data=Data())
        return interp_plane

    def line_cut(self, spec, param=None, vol=None, hkl='HK', origin=None, shape=(512, 512), 
                 n_samples=512, width_px=1, show=True, interactive=False):
        """
        Compute a line cut on a slice with comprehensive analysis options.

        This method performs line cuts on slice data supporting multiple specification formats,
        interactive editing, and comprehensive profile analysis. Supports both endpoint-based
        and preset-based line definitions with real-time visualization.

        Usage:
            # Horizontal line cut at V=1.0
            line_data = da.line_cut('zero', param=(1.0, 'x'), vol=slice_mesh)
            
            # Interactive line cut with draggable endpoints
            line_data = da.line_cut(((0, 0), (1, 1)), vol=slice_mesh, interactive=True)
            
            # Diagonal line cut with averaging
            line_data = da.line_cut('positive', vol=slice_mesh, width_px=3)

        Parameters:
            spec: Line specification. Options:
                - ((U1,V1),(U2,V2)): Explicit endpoints in physical slice coordinates
                - 'zero'/'horizontal': Horizontal line at fixed V value
                - 'infinite'/'vertical': Vertical line at fixed U value  
                - 'positive': Diagonal from (U_min,V_min) to (U_max,V_max)
                - 'negative': Diagonal from (U_min,V_max) to (U_max,V_min)
            param (tuple): Required for preset lines. Format (value, axis_letter):
                - For 'zero': (V_value, 'x') fixes V and traverses U_min→U_max
                - For 'infinite': (U_value, 'y') fixes U and traverses V_min→V_max
            vol: Optional volume or slice data. Formats:
                - pv.PolyData slice mesh
                - (img, extent) tuple from show_slice(..., return_image=True)
                - Volume data for fresh slice generation
                - None: Uses cached last image from previous show_slice call
            hkl (str): Orientation preset when generating slice from vol ('HK', 'KL', 'HL')
            origin (tuple): Slice origin (H,K,L) when generating slice from vol
            shape (tuple): Raster resolution (H, W) when generating slice from vol
            n_samples (int): Number of samples along the line cut
            width_px (int): Averaging strip width in pixels normal to line (1 = true line)
            show (bool): If True, overlays line on slice image and displays 1D profile
            interactive (bool): If True, enables draggable endpoints with live updates

        Returns:
            dict: Line cut analysis results containing:
                - 'distance': np.ndarray of distance values along line
                - 'intensity': np.ndarray of intensity values along line
                - 'U': np.ndarray of U coordinates along line
                - 'V': np.ndarray of V coordinates along line
                - 'endpoints': ((U1,V1),(U2,V2)) actual endpoints used
                - 'orientation': str orientation of the slice ('HK', 'KL', 'HL', etc.)

        Raises:
            ImportError: If PyVista or matplotlib are not available
            ValueError: If line specification is invalid or no slice data available

        Examples:
            # Horizontal line cut through peak
            horizontal = da.line_cut('zero', param=(0.5, 'x'), vol=slice_data)
            
            # Vertical line cut with wide averaging
            vertical = da.line_cut('infinite', param=(1.0, 'y'), width_px=5)
            
            # Custom endpoints with interactive editing
            custom = da.line_cut(((0.2, 0.3), (0.8, 0.7)), interactive=True)
            
            # Diagonal analysis across full extent
            diagonal = da.line_cut('positive', n_samples=1024, show=True)
        """
        import numpy as _np

        # Resolve image and extent
        H, W = None, None
        orientation = None
        U_min = U_max = V_min = V_max = None
        img = None

        try:
            if vol is not None:
                # Support passing a pre-rasterized image and its extent (as returned by show_slice(..., return_image=True))
                if isinstance(vol, (tuple, list)) and len(vol) >= 2:
                    try:
                        img_candidate = _np.asarray(vol[0])
                        ext_candidate = vol[1]
                        if img_candidate.ndim == 2 and isinstance(ext_candidate, (list, tuple)) and len(ext_candidate) == 4:
                            img = img_candidate.astype(_np.float32)
                            U_min, U_max, V_min, V_max = float(ext_candidate[0]), float(ext_candidate[1]), float(ext_candidate[2]), float(ext_candidate[3])
                            H, W = img.shape[:2]
                            orientation = self._last_orientation or "Auto"
                            # cache for subsequent calls
                            self._last_image = img
                            self._last_extent = [U_min, U_max, V_min, V_max]
                            self._last_orientation = orientation
                        else:
                            pass  # fall through
                    except Exception:
                        pass  # fall through

                if img is None:
                    if pv is None:
                        raise ImportError("PyVista is required to build a slice from 'vol'.")

                    # If vol is already a slice mesh
                    if isinstance(vol, pv.PolyData):
                        sl = vol
                        n_vec = _np.asarray(getattr(sl, 'field_data', {}).get('slice_normal', _np.array([0.0, 0.0, 1.0], dtype=float)), dtype=float)
                        o_vec = _np.asarray(getattr(sl, 'field_data', {}).get('slice_origin', _np.asarray(getattr(sl, 'center', (0.0, 0.0, 0.0)), dtype=float)), dtype=float)
                    else:
                        # If vol looks like a 3D volume, require an explicit slice beforehand.
                        # Call show_slice(..., return_image=True) and pass (img, extent) to line_cut.
                        is_3d_volume = isinstance(vol, pv.ImageData) or (isinstance(vol, _np.ndarray) and vol.ndim == 3) or (isinstance(vol, (tuple, list)) and len(vol) >= 1 and isinstance(vol[0], _np.ndarray) and getattr(vol[0], "ndim", None) == 3)
                        if is_3d_volume:
                            raise ValueError("line_cut expects slice data. Pass a pv.PolyData slice or (img, extent) from show_slice(..., return_image=True).")
                        # Otherwise attempt to slice via slice_data using defaults
                        sl = self.slice_data(vol, hkl='HK', shape=(512, 512), clamp_to_bounds=True)
                        n_vec = _np.asarray(getattr(sl, 'field_data', {}).get('slice_normal', _np.array([0.0, 0.0, 1.0], dtype=float)), dtype=float)
                        o_vec = _np.asarray(getattr(sl, 'field_data', {}).get('slice_origin', _np.asarray(getattr(sl, 'center', (0.0, 0.0, 0.0)), dtype=float)), dtype=float)

                pts = _np.asarray(getattr(sl, 'points', _np.empty((0, 3))), dtype=float)
                try:
                    vals = _np.asarray(sl['intensity'], dtype=float).reshape(-1)
                except Exception:
                    vals = _np.zeros((pts.shape[0],), dtype=float)

                H = max(int(shape[0] if (isinstance(shape, (tuple, list)) and len(shape) == 2) else 512), 2)
                W = max(int(shape[1] if (isinstance(shape, (tuple, list)) and len(shape) == 2) else 512), 2)

                def _infer_orientation_and_axes(normal_vec: _np.ndarray):
                    nn = _np.asarray(normal_vec, dtype=float)
                    nn_len = float(_np.linalg.norm(nn))
                    if not _np.isfinite(nn_len) or nn_len <= 0.0:
                        nn = _np.array([0.0, 0.0, 1.0], dtype=float)
                    else:
                        nn = nn / nn_len
                    X = _np.array([1.0, 0.0, 0.0], dtype=float)  # H
                    Y = _np.array([0.0, 1.0, 0.0], dtype=float)  # K
                    Z = _np.array([0.0, 0.0, 1.0], dtype=float)  # L
                    tol = 0.95
                    dX = abs(float(_np.dot(nn, X)))
                    dY = abs(float(_np.dot(nn, Y)))
                    dZ = abs(float(_np.dot(nn, Z)))
                    if dZ >= tol:
                        return "HK", (0, 1)
                    if dX >= tol:
                        return "KL", (1, 2)
                    if dY >= tol:
                        return "HL", (0, 2)
                    return "Custom", None

                orientation, uv_idxs = _infer_orientation_and_axes(n_vec)

                if pts.size == 0 or vals.size == 0 or pts.shape[0] != vals.shape[0]:
                    raise ValueError("Slice contains no valid points to rasterize")

                if uv_idxs is not None:
                    u_idx, v_idx = uv_idxs
                    U = pts[:, u_idx].astype(float)
                    V = pts[:, v_idx].astype(float)
                    U_min, U_max = float(_np.min(U)), float(_np.max(U))
                    V_min, V_max = float(_np.min(V)), float(_np.max(V))
                    if (not _np.isfinite(U_min)) or (not _np.isfinite(U_max)) or (U_max == U_min):
                        U_min, U_max = -0.5, 0.5
                    if (not _np.isfinite(V_min)) or (not _np.isfinite(V_max)) or (V_max == V_min):
                        V_min, V_max = -0.5, 0.5
                    sum_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]], weights=vals)
                    cnt_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]])
                    with _np.errstate(invalid="ignore", divide="ignore"):
                        img = _np.zeros_like(sum_img, dtype=_np.float32)
                        nz = cnt_img > 0
                        img[nz] = (sum_img[nz] / cnt_img[nz]).astype(_np.float32)
                        img[~nz] = 0.0
                else:
                    # Custom orientation: build in-plane basis from normal and origin
                    world_axes = [
                        _np.array([1.0, 0.0, 0.0], dtype=float),
                        _np.array([0.0, 1.0, 0.0], dtype=float),
                        _np.array([0.0, 0.0, 1.0], dtype=float),
                    ]
                    ref = world_axes[0]
                    for ax in world_axes:
                        if abs(float(_np.dot(ax, n_vec))) < 0.9:
                            ref = ax
                            break
                    u = _np.cross(n_vec, ref)
                    u_len = float(_np.linalg.norm(u))
                    if not _np.isfinite(u_len) or u_len <= 0.0:
                        ref = _np.array([0.0, 1.0, 0.0], dtype=float)
                        u = _np.cross(n_vec, ref)
                        u_len = float(_np.linalg.norm(u))
                        if not _np.isfinite(u_len) or u_len <= 0.0:
                            u = _np.array([1.0, 0.0, 0.0], dtype=float)
                            u_len = 1.0
                    u = u / u_len
                    v = _np.cross(n_vec, u)
                    v_len = float(_np.linalg.norm(v))
                    if not _np.isfinite(v_len) or v_len <= 0.0:
                        v = _np.array([0.0, 1.0, 0.0], dtype=float)
                    else:
                        v = v / v_len

                    rel = _np.asarray(pts - o_vec[None, :], dtype=float)
                    U = rel.dot(u)
                    V = rel.dot(v)

                    U_min, U_max = float(_np.min(U)), float(_np.max(U))
                    V_min, V_max = float(_np.min(V)), float(_np.max(V))
                    if not _np.isfinite(U_min) or not _np.isfinite(U_max) or (U_max == U_min):
                        U_min, U_max = -0.5, 0.5
                    if not _np.isfinite(V_min) or not _np.isfinite(V_max) or (V_max == V_min):
                        V_min, V_max = -0.5, 0.5

                    sum_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]], weights=vals)
                    cnt_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]])
                    with _np.errstate(invalid="ignore", divide="ignore"):
                        img = _np.zeros_like(sum_img, dtype=_np.float32)
                        nz = cnt_img > 0
                        img[nz] = (sum_img[nz] / cnt_img[nz]).astype(_np.float32)
                        img[~nz] = 0.0

                # cache
                self._last_image = img
                self._last_extent = [U_min, U_max, V_min, V_max]
                self._last_orientation = orientation
            else:
                img = self._last_image
                if img is None or self._last_extent is None:
                    raise ValueError("No slice image available; provide 'vol' or call show_slice(..., return_image=True) first.")
                U_min, U_max, V_min, V_max = self._last_extent
                H, W = img.shape[:2]
                orientation = self._last_orientation or "Auto"
        except Exception as e:
            raise

        # Endpoints from spec/preset
        def _endpoints_from_spec(_spec, _param):
            if isinstance(_spec, (tuple, list)) and len(_spec) == 2:
                return tuple(_spec[0]), tuple(_spec[1])
            s = str(_spec).strip().lower()
            if s in ("zero", "horizontal", "0"):
                if not (_param and len(_param) == 2):
                    raise ValueError("param=(value,'x') required for 'zero' preset")
                val, ax = _param
                ax = str(ax).lower()
                # V fixed at val; U spans full range
                return (U_min, float(val)), (U_max, float(val))
            if s in ("infinite", "vertical", "inf"):
                if not (_param and len(_param) == 2):
                    raise ValueError("param=(value,'y') required for 'infinite' preset")
                val, ax = _param
                ax = str(ax).lower()
                # U fixed at val; V spans full range
                return (float(val), V_min), (float(val), V_max)
            if s in ("positive", "pos"):
                return (U_min, V_min), (U_max, V_max)
            if s in ("negative", "neg"):
                return (U_min, V_max), (U_max, V_min)
            raise ValueError(f"Unknown spec '{_spec}'; pass endpoints ((U1,V1),(U2,V2)) or preset string")

        (U1, V1), (U2, V2) = _endpoints_from_spec(spec, param)

        # Convert endpoints to pixel coords
        def _uv_to_pixel(Uv, Vv):
            col = (float(Uv) - U_min) / (U_max - U_min if (U_max != U_min) else 1.0) * (W - 1)
            row = (float(Vv) - V_min) / (V_max - V_min if (V_max != V_min) else 1.0) * (H - 1)
            return col, row

        c1, r1 = _uv_to_pixel(U1, V1)
        c2, r2 = _uv_to_pixel(U2, V2)

        # Sampling points along the line in pixel space
        n_samples = max(int(n_samples), 2)
        ts = _np.linspace(0.0, 1.0, n_samples, dtype=float)
        cols = c1 + ts * (c2 - c1)
        rows = r1 + ts * (r2 - r1)

        # Bilinear interpolation
        def _bilinear(img_arr, cc, rr):
            h, w = img_arr.shape[:2]
            cc = _np.clip(cc, 0.0, w - 1.0)
            rr = _np.clip(rr, 0.0, h - 1.0)
            c0 = _np.floor(cc).astype(int)
            r0 = _np.floor(rr).astype(int)
            c1i = _np.clip(c0 + 1, 0, w - 1)
            r1i = _np.clip(r0 + 1, 0, h - 1)
            dc = cc - c0
            dr = rr - r0
            I00 = img_arr[r0, c0]
            I10 = img_arr[r0, c1i]
            I01 = img_arr[r1i, c0]
            I11 = img_arr[r1i, c1i]
            return (1 - dc) * (1 - dr) * I00 + dc * (1 - dr) * I10 + (1 - dc) * dr * I01 + dc * dr * I11

        # Width averaging across perpendicular offsets
        width_px = max(int(width_px), 1)
        if width_px == 1:
            prof = _bilinear(img, cols, rows)
        else:
            dcol = c2 - c1
            drow = r2 - r1
            length = float(_np.hypot(dcol, drow))
            if not _np.isfinite(length) or length <= 0.0:
                length = 1.0
            # Perpendicular unit vector (pixel space)
            u_perp = _np.array([-drow, dcol], dtype=float) / length
            half = (width_px - 1) / 2.0
            offsets = _np.linspace(-half, half, width_px, dtype=float)
            samples = []
            for off in offsets:
                cc = cols + off * u_perp[1]
                rr = rows + off * u_perp[0]
                samples.append(_bilinear(img, cc, rr))
            prof = _np.mean(_np.vstack(samples), axis=0)

        # Physical coordinates per sample and distance
        U_samples = U1 + ts * (U2 - U1)
        V_samples = V1 + ts * (V2 - V1)
        dist = _np.sqrt((U_samples - U1) ** 2 + (V_samples - V1) ** 2)

        lc = {
            "distance": dist.astype(_np.float32),
            "intensity": _np.asarray(prof, dtype=_np.float32),
            "U": _np.asarray(U_samples, dtype=_np.float32),
            "V": _np.asarray(V_samples, dtype=_np.float32),
            "endpoints": ((float(U1), float(V1)), (float(U2), float(V2))),
            "orientation": str(orientation),
        }

        if show and not interactive:
            # Overlay line and show profile
            if plt is not None:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                ax_img, ax_prof = axes
                extent = [U_min, U_max, V_min, V_max]
                ax_img.imshow(img, origin='lower', extent=extent, cmap='viridis', aspect='auto')
                ax_img.plot([U1, U2], [V1, V2], color='cyan', linewidth=2)
                ax_img.set_title(f"Slice ({orientation}) with line cut")
                ax_img.set_xlabel('U'); ax_img.set_ylabel('V')
                ax_prof.plot(dist, prof, color='magenta')
                ax_prof.set_xlabel('Distance')
                ax_prof.set_ylabel('Intensity')
                ax_prof.set_title("Line cut profile")
                plt.tight_layout()
                plt.show()

        # Interactive draggable endpoints with live profile updates
        if interactive:
            if plt is None:
                raise ImportError("matplotlib is required for interactive line_cut")
            # Check backend; warn and fallback to static overlay if non-interactive
            import matplotlib as _mpl
            _backend = str(getattr(_mpl, "get_backend", lambda: "")()).lower()
            if ("inline" in _backend) or ("agg" in _backend):
                try:
                    print(f"DashAnalysis.line_cut interactive=True requires an interactive Matplotlib backend. Detected backend: {_mpl.get_backend()}. Run '%matplotlib widget' (preferred, requires 'ipympl') or '%matplotlib notebook' in a notebook cell, then retry.")
                except Exception:
                    pass
                # Fallback: draw static overlay and return
                if show:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    ax_img, ax_prof = axes
                    extent = [U_min, U_max, V_min, V_max]
                    ax_img.imshow(img, origin='lower', extent=extent, cmap='viridis', aspect='auto')
                    ax_img.plot([U1, U2], [V1, V2], color='cyan', linewidth=2)
                    ax_img.set_title(f"Slice ({orientation}) with line cut (static, non-interactive backend)")
                    ax_img.set_xlabel('U'); ax_img.set_ylabel('V')
                    ax_prof.plot(dist, prof, color='magenta')
                    ax_prof.set_xlabel('Distance')
                    ax_prof.set_ylabel('Intensity')
                    ax_prof.set_title("Line cut profile")
                    plt.tight_layout()
                    plt.show()
                return lc
            try:
                from matplotlib.lines import Line2D
            except Exception:
                Line2D = None

            extent = [U_min, U_max, V_min, V_max]
            fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(10, 4))
            im = ax_img.imshow(img, origin='lower', extent=extent, cmap='viridis', aspect='auto')
            ax_img.set_title(f"Slice ({orientation}) — drag endpoints")
            # initial endpoints from spec
            p1 = [float(U1), float(V1)]
            p2 = [float(U2), float(V2)]

            # line + endpoint markers
            if Line2D is not None:
                line = Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', lw=2)
                ax_img.add_line(line)
            else:
                line_plot, = ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', lw=2)
            pt1 = ax_img.plot(p1[0], p1[1], 'o', color='cyan', ms=8, picker=5)[0]
            pt2 = ax_img.plot(p2[0], p2[1], 'o', color='cyan', ms=8, picker=5)[0]

            # initial profile
            prof_line, = ax_prof.plot(dist, prof, color='magenta')
            ax_prof.set_xlabel('Distance'); ax_prof.set_ylabel('Intensity')
            ax_prof.set_title('Line cut profile')

            state = {"drag": None}

            def update_profile():
                # recompute with current endpoints using cached image+extent
                lc_local = self.line_cut((tuple((float(pt1.get_xdata()[0]), float(pt1.get_ydata()[0]))),
                                          tuple((float(pt2.get_xdata()[0]), float(pt2.get_ydata()[0])))),
                                         vol=(img, extent),
                                         n_samples=n_samples,
                                         width_px=width_px,
                                         show=False)
                prof_line.set_data(lc_local["distance"], lc_local["intensity"])
                ax_prof.relim(); ax_prof.autoscale_view()
                fig.canvas.draw_idle()

            def on_press(event):
                if event.inaxes != ax_img:
                    return
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return
                # pick nearest endpoint
                d1 = float(np.hypot(x - pt1.get_xdata()[0], y - pt1.get_ydata()[0]))
                d2 = float(np.hypot(x - pt2.get_xdata()[0], y - pt2.get_ydata()[0]))
                state["drag"] = 0 if d1 <= d2 else 1

            def on_motion(event):
                if state["drag"] is None or event.inaxes != ax_img:
                    return
                x, y = event.xdata, event.ydata
                if x is None or y is None:
                    return
                # constrain to extents
                x = float(np.clip(x, U_min, U_max))
                y = float(np.clip(y, V_min, V_max))
                if state["drag"] == 0:
                    pt1.set_data([x], [y])
                else:
                    pt2.set_data([x], [y])
                if Line2D is not None:
                    line.set_data([pt1.get_xdata()[0], pt2.get_xdata()[0]],
                                  [pt1.get_ydata()[0], pt2.get_ydata()[0]])
                else:
                    line_plot.set_data([pt1.get_xdata()[0], pt2.get_xdata()[0]],
                                       [pt1.get_ydata()[0], pt2.get_ydata()[0]])
                update_profile()

            def on_release(event):
                state["drag"] = None

            cid1 = fig.canvas.mpl_connect('button_press_event', on_press)
            cid2 = fig.canvas.mpl_connect('motion_notify_event', on_motion)
            cid3 = fig.canvas.mpl_connect('button_release_event', on_release)

            plt.tight_layout()
            plt.show()

        return lc

    def show_slice(self, vol, hkl='HK', origin=None, shape=None, cmap='viridis',
                   spacing=(1.0, 1.0, 1.0), grid_origin=(0.0, 0.0, 0.0),
                   clim=None, min_intensity=None, max_intensity=None, axes=None, return_image=False):
        """
        Slice a 3D HKL volume or point cloud and display as 2D raster with interactive features.

        This method provides comprehensive slice visualization with automatic orientation detection,
        intensity filtering, interactive hover tooltips, and support for both volume and point cloud data.
        Includes caching for efficient line cut operations.

        Usage:
            # Display HK plane slice
            da.show_slice(volume_data, hkl='HK')
            
            # Custom slice with intensity filtering
            da.show_slice(data, origin=(1.0, 0.5, 0.0), min_intensity=100)
            
            # Return image data for further analysis
            img, extent = da.show_slice(data, return_image=True)

        Parameters:
            vol: Input data. Supported formats:
                - pv.ImageData with cell_data['intensity']
                - pv.PolyData slice mesh (rasterized directly)
                - NumPy ndarray (D,H,W) volume
                - (ndarray_volume, shape) tuple
                - Point cloud: (points, intensities), Data object, or dict
            hkl (str): Plane orientation preset:
                - 'HK'/'XY': normal (0,0,1) - HK plane
                - 'KL'/'YZ': normal (1,0,0) - KL plane  
                - 'HL'/'XZ': normal (0,1,0) - HL plane
            origin (tuple): (H,K,L) slice origin coordinates. Defaults to data center
            shape (tuple): (H, W) raster resolution. Defaults to (512, 512)
            cmap (str): Matplotlib colormap name for display
            spacing (tuple): Voxel spacing (ΔH, ΔK, ΔL) for NumPy volume construction
            grid_origin (tuple): Grid origin (H0, K0, L0) for NumPy volume construction
            clim (tuple): (vmin, vmax) intensity display limits. Overridden by min/max_intensity
            min_intensity (float): Minimum intensity threshold (filters and sets vmin)
            max_intensity (float): Maximum intensity threshold (filters and sets vmax)
            axes: Optional matplotlib Axes object for rendering. Creates new figure if None
            return_image (bool): If True, returns (img, extent) tuple instead of displaying

        Returns:
            tuple or None: If return_image=True, returns (img, extent) where:
                - img: np.ndarray of rasterized intensity values
                - extent: [U_min, U_max, V_min, V_max] physical coordinate bounds
            Otherwise returns None and displays the slice

        Raises:
            ImportError: If PyVista or matplotlib are not available
            ValueError: If input data format is not supported

        Examples:
            # Basic slice display with hover tooltips
            da.show_slice(volume, hkl='HK')
            
            # High-contrast slice with intensity filtering
            da.show_slice(data, min_intensity=50, max_intensity=500, cmap='hot')
            
            # Custom slice orientation with specific origin
            da.show_slice(data, origin=(1.5, 0.0, 0.5), shape=(1024, 1024))
            
            # Get image data for line cut analysis
            img, extent = da.show_slice(data, return_image=True)
            line_data = da.line_cut('zero', param=(0.5, 'x'), vol=(img, extent))
        """
        # Require PyVista for arbitrary oriented slicing
        if pv is None:
            raise ImportError("PyVista is required for show_slice()")

        import numpy as _np
        import matplotlib.pyplot as _plt

        # Helper: normalize volume to a PyVista ImageData with cell_data['intensity']
        def _ensure_grid(_vol, _spacing=(1.0, 1.0, 1.0), _origin=(0.0, 0.0, 0.0)):
            if isinstance(_vol, pv.ImageData):
                _grid = _vol
                if ('intensity' not in _grid.cell_data) and ('intensity' in _grid.point_data):
                    _grid = _grid.point_data_to_cell_data(pass_point_data=False)
                if 'intensity' not in _grid.cell_data:
                    raise ValueError("ImageData must have cell_data['intensity'] for slicing.")
                return _grid

            if isinstance(_vol, (tuple, list)) and len(_vol) >= 1 and isinstance(_vol[0], _np.ndarray):
                _vol_np = _vol[0]
            elif isinstance(_vol, _np.ndarray):
                _vol_np = _vol
            else:
                raise TypeError("vol must be pv.ImageData, a NumPy ndarray (D,H,W), or (ndarray_volume, shape) tuple.")

            if _vol_np.ndim != 3:
                raise ValueError("NumPy volume must be 3D shaped (D,H,W).")

            _dims_cells = _np.array(_vol_np.shape, dtype=int)
            _grid = pv.ImageData()
            _grid.dimensions = (_dims_cells + 1).tolist()
            _grid.spacing = tuple(float(x) for x in _spacing)
            _grid.origin = tuple(float(x) for x in _origin)
            # Flatten in Fortran order to match VTK/PyVista cell layout
            _grid.cell_data['intensity'] = _np.asarray(_vol_np, dtype=_np.float32).flatten(order='F')
            return _grid

        # Helper: map HKL preset to normal
        def _resolve_normal(_hkl):
            if isinstance(_hkl, str):
                s = _hkl.strip().lower()
                if s in ('hk', 'xy'):
                    return _np.array([0.0, 0.0, 1.0], dtype=float)
                if s in ('kl', 'yz'):
                    return _np.array([1.0, 0.0, 0.0], dtype=float)
                if s in ('hl', 'xz'):
                    return _np.array([0.0, 1.0, 0.0], dtype=float)
            return _np.array([0.0, 0.0, 1.0], dtype=float)

        # Helper: rasterize slice to image and report orientation & orthogonal axis value
        def _rasterize_slice(_slice_mesh, _normal, _origin, H=512, W=512, _min_intensity=None, _max_intensity=None):
            pts = _np.asarray(getattr(_slice_mesh, 'points', _np.empty((0, 3))), dtype=float)
            try:
                vals = _np.asarray(_slice_mesh['intensity'], dtype=float).reshape(-1)
            except Exception:
                vals = _np.zeros((len(pts),), dtype=float)

            # Optional pre-rasterization filter by intensity range
            if (_min_intensity is not None) or (_max_intensity is not None):
                m = _np.ones(vals.shape, dtype=bool)
                if _min_intensity is not None:
                    m &= (vals >= float(_min_intensity))
                if _max_intensity is not None:
                    m &= (vals <= float(_max_intensity))
                pts = pts[m]
                vals = vals[m]

            if pts.size == 0 or vals.size == 0 or pts.shape[0] != vals.shape[0]:
                # Return empty image with default ranges and Custom orientation
                return _np.zeros((H, W), dtype=_np.float32), -0.5, 0.5, -0.5, 0.5, "Custom", None, None

            n = _np.array(_normal, dtype=float)
            o = _np.array(_origin, dtype=float)

            # Normalize normal
            n_norm = float(_np.linalg.norm(n))
            if not _np.isfinite(n_norm) or n_norm <= 0.0:
                n = _np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                n = n / n_norm

            # Infer orientation from normal
            X = _np.array([1.0, 0.0, 0.0], dtype=float)  # H
            Y = _np.array([0.0, 1.0, 0.0], dtype=float)  # K
            Z = _np.array([0.0, 0.0, 1.0], dtype=float)  # L
            tol = 0.95
            dX = abs(float(_np.dot(n, X)))
            dY = abs(float(_np.dot(n, Y)))
            dZ = abs(float(_np.dot(n, Z)))

            if dZ >= tol:
                # HK plane: U=H, V=K; orth_label L
                U = pts[:, 0].astype(float)
                V = pts[:, 1].astype(float)
                orientation = "HK"
                orth_label = "L"
                orth_value = float(o[2])
            elif dX >= tol:
                # KL plane: U=K, V=L; orth_label H
                U = pts[:, 1].astype(float)
                V = pts[:, 2].astype(float)
                orientation = "KL"
                orth_label = "H"
                orth_value = float(o[0])
            elif dY >= tol:
                # HL plane: U=H, V=L; orth_label K
                U = pts[:, 0].astype(float)
                V = pts[:, 2].astype(float)
                orientation = "HL"
                orth_label = "K"
                orth_value = float(o[1])
            else:
                # Custom orientation: build in-plane basis u, v
                world_axes = [
                    _np.array([1.0, 0.0, 0.0], dtype=float),
                    _np.array([0.0, 1.0, 0.0], dtype=float),
                    _np.array([0.0, 0.0, 1.0], dtype=float),
                ]
                ref = world_axes[0]
                for ax in world_axes:
                    if abs(float(_np.dot(ax, n))) < 0.9:
                        ref = ax
                        break
                u = _np.cross(n, ref)
                u_norm = float(_np.linalg.norm(u))
                if not _np.isfinite(u_norm) or u_norm <= 0.0:
                    ref = _np.array([0.0, 1.0, 0.0], dtype=float)
                    u = _np.cross(n, ref)
                    u_norm = float(_np.linalg.norm(u))
                    if not _np.isfinite(u_norm) or u_norm <= 0.0:
                        u = _np.array([1.0, 0.0, 0.0], dtype=float)
                        u_norm = 1.0
                u = u / u_norm
                v = _np.cross(n, u)
                v_norm = float(_np.linalg.norm(v))
                if not _np.isfinite(v_norm) or v_norm <= 0.0:
                    v = _np.array([0.0, 1.0, 0.0], dtype=float)

                # Project points to plane (origin-relative)
                rel = pts - o[None, :]
                U = rel.dot(u)  # cols
                V = rel.dot(v)  # rows

                orientation = "Custom"
                orth_label = None
                try:
                    orth_value = float(_np.dot(n, o))
                except Exception:
                    orth_value = None

            # Extents and binning
            U_min, U_max = float(_np.min(U)), float(_np.max(U))
            V_min, V_max = float(_np.min(V)), float(_np.max(V))
            if not _np.isfinite(U_min) or not _np.isfinite(U_max) or (U_max == U_min):
                U_min, U_max = -0.5, 0.5
            if not _np.isfinite(V_min) or not _np.isfinite(V_max) or (V_max == V_min):
                V_min, V_max = -0.5, 0.5

            sum_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]], weights=vals)
            cnt_img, _, _ = _np.histogram2d(V, U, bins=[H, W], range=[[V_min, V_max], [U_min, U_max]])
            with _np.errstate(invalid='ignore', divide='ignore'):
                img = _np.zeros_like(sum_img, dtype=_np.float32)
                nz = cnt_img > 0
                img[nz] = (sum_img[nz] / cnt_img[nz]).astype(_np.float32)
                img[~nz] = 0.0

            return img, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value

        # Resolve normal and output image size
        normal = _resolve_normal(hkl)
        H, W = (shape if (isinstance(shape, (tuple, list)) and len(shape) == 2) else (512, 512))
        H = max(int(H), 1)
        W = max(int(W), 1)

        # Normalize normal vector
        n = _np.asarray(normal, dtype=float)
        n_norm = float(_np.linalg.norm(n))
        if not _np.isfinite(n_norm) or n_norm <= 0.0:
            n = _np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            n = n / n_norm

        # If a PolyData slice is provided, rasterize directly
        if isinstance(vol, pv.PolyData):
            normal_fd = getattr(vol, 'field_data', {}).get('slice_normal', None)
            origin_fd = getattr(vol, 'field_data', {}).get('slice_origin', None)
            u_axis_fd = getattr(vol, 'field_data', {}).get('slice_u_axis', None)
            v_axis_fd = getattr(vol, 'field_data', {}).get('slice_v_axis', None)
            u_label_fd = getattr(vol, 'field_data', {}).get('slice_u_label', None)
            v_label_fd = getattr(vol, 'field_data', {}).get('slice_v_label', None)
            
            normal_use = normal_fd if normal_fd is not None else n
            origin_use = origin_fd if origin_fd is not None else getattr(vol, 'center', (0.0, 0.0, 0.0))
            result = _rasterize_slice(
                vol, normal_use, _np.array(origin_use, dtype=float), H=H, W=W,
                _min_intensity=min_intensity, _max_intensity=max_intensity
            )
            if result is None:
                return None
            img, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value = result
            try:
                # cache for da.line_cut when called without vol
                self._last_image = img
                self._last_extent = [U_min, U_max, V_min, V_max]
                self._last_orientation = orientation
            except Exception:
                pass

            extent = [U_min, U_max, V_min, V_max]
            if axes is None:
                _plt.figure(figsize=(6, 5))
                ax = _plt.gca()
            else:
                ax = axes
            vmin = float(min_intensity) if (min_intensity is not None) else (clim[0] if clim else None)
            vmax = float(max_intensity) if (max_intensity is not None) else (clim[1] if clim else None)
            im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap,
                           vmin=vmin,
                           vmax=vmax,
                           aspect='auto')
            
            # Use HKL axis labels if available, otherwise fall back to orientation-based labels
            if u_label_fd is not None and v_label_fd is not None:
                ax.set_xlabel(str(u_label_fd))
                ax.set_ylabel(str(v_label_fd))
            elif u_axis_fd is not None and v_axis_fd is not None:
                ax.set_xlabel(format_hkl_axis(u_axis_fd))
                ax.set_ylabel(format_hkl_axis(v_axis_fd))
            elif orientation == "HK":
                ax.set_xlabel('H')
                ax.set_ylabel('K')
            elif orientation == "KL":
                ax.set_xlabel('K')
                ax.set_ylabel('L')
            elif orientation == "HL":
                ax.set_xlabel('H')
                ax.set_ylabel('L')
            else:
                ax.set_xlabel('U')
                ax.set_ylabel('V')

            title = None
            if orientation in ("HK", "KL", "HL") and (orth_label is not None) and (orth_value is not None) and _np.isfinite(orth_value):
                title = f'{orientation} plane ({orth_label} = {orth_value:.3f})'
            else:
                title = f'{hkl} slice'
            ax.set_title(title)

            _plt.colorbar(im, ax=ax, label='Intensity')
            
            # Add interactive hover tooltips for HKL and intensity
            def on_hover(event):
                if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                    # Convert mouse coordinates to image pixel coordinates
                    x_coord = event.xdata
                    y_coord = event.ydata
                    
                    # Convert to pixel indices
                    col = int((x_coord - extent[0]) / (extent[1] - extent[0]) * img.shape[1])
                    row = int((y_coord - extent[2]) / (extent[3] - extent[2]) * img.shape[0])
                    
                    # Clamp to image bounds
                    col = max(0, min(img.shape[1] - 1, col))
                    row = max(0, min(img.shape[0] - 1, row))
                    
                    # Get intensity value
                    intensity = img[row, col]
                    
                    # Convert back to HKL coordinates based on orientation
                    if u_label_fd is not None and v_label_fd is not None:
                        # Use custom HKL labels
                        tooltip_text = f"{u_label_fd}: {x_coord:.3f}, {v_label_fd}: {y_coord:.3f}\nIntensity: {intensity:.1f}"
                    elif orientation == "HK":
                        tooltip_text = f"H: {x_coord:.3f}, K: {y_coord:.3f}\nIntensity: {intensity:.1f}"
                    elif orientation == "KL":
                        tooltip_text = f"K: {x_coord:.3f}, L: {y_coord:.3f}\nIntensity: {intensity:.1f}"
                    elif orientation == "HL":
                        tooltip_text = f"H: {x_coord:.3f}, L: {y_coord:.3f}\nIntensity: {intensity:.1f}"
                    else:
                        tooltip_text = f"U: {x_coord:.3f}, V: {y_coord:.3f}\nIntensity: {intensity:.1f}"
                    
                    # Update or create annotation
                    if hasattr(ax, '_hover_annotation'):
                        ax._hover_annotation.set_text(tooltip_text)
                        ax._hover_annotation.xy = (x_coord, y_coord)
                        ax._hover_annotation.set_visible(True)
                    else:
                        ax._hover_annotation = ax.annotate(
                            tooltip_text,
                            xy=(x_coord, y_coord),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                            fontsize=9,
                            ha='left'
                        )
                    
                    _plt.draw()
                else:
                    # Hide annotation when not hovering over the plot
                    if hasattr(ax, '_hover_annotation'):
                        ax._hover_annotation.set_visible(False)
                        _plt.draw()
            
            # Connect the hover event
            if axes is None:  # Only add hover for standalone plots
                _plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)
                _plt.show()
            
            if return_image:
                return img, extent
            return None

        # Volume-like inputs: build grid and slice
        is_volume_like = isinstance(vol, pv.ImageData) or isinstance(vol, _np.ndarray) or (isinstance(vol, (tuple, list)) and len(vol) >= 1 and isinstance(vol[0], _np.ndarray) and vol[0].ndim == 3)
        if is_volume_like:
            grid = _ensure_grid(vol, _spacing=spacing, _origin=grid_origin)
            if origin is None:
                origin = getattr(grid, 'center', (0.0, 0.0, 0.0))
            # Clamp origin within grid bounds
            b = getattr(grid, "bounds", None)
            o = _np.array(origin, dtype=float)
            if b and len(b) == 6:
                o[0] = float(_np.clip(o[0], b[0], b[1]))
                o[1] = float(_np.clip(o[1], b[2], b[3]))
                o[2] = float(_np.clip(o[2], b[4], b[5]))
            origin = tuple(o.tolist())
            sl = grid.slice(normal=n, origin=_np.array(origin, dtype=float))
        else:
            # Point-cloud inputs: compute default origin from points center if not provided
            pts = None
            if isinstance(vol, (tuple, list)) and len(vol) >= 2:
                pts = _np.asarray(vol[0], dtype=float)
            elif hasattr(vol, 'points'):
                pts = _np.asarray(getattr(vol, 'points'), dtype=float)
            elif isinstance(vol, dict) and ('points' in vol):
                pts = _np.asarray(vol['points'], dtype=float)
            else:
                pts = None
            if origin is None:
                if pts is not None and pts.size >= 3:
                    mn = pts.min(axis=0)
                    mx = pts.max(axis=0)
                    origin = tuple(((mn + mx) * 0.5).astype(float).tolist())
                else:
                    origin = (0.0, 0.0, 0.0)
            # Create slice mesh via slice_data (handles points internally)
            sl = self.slice_data(vol, hkl=hkl, shape=(H, W), clamp_to_bounds=True)

        # Rasterize with orientation info
        result = _rasterize_slice(
            sl, n, _np.array(origin, dtype=float), H=H, W=W,
            _min_intensity=min_intensity, _max_intensity=max_intensity
        )
        if result is None:
            return None
        img, U_min, U_max, V_min, V_max, orientation, orth_label, orth_value = result
        try:
            # cache for da.line_cut when called without vol
            self._last_image = img
            self._last_extent = [U_min, U_max, V_min, V_max]
            self._last_orientation = orientation
        except Exception:
            pass

        # Display via matplotlib imshow with physical axis labels
        extent = [U_min, U_max, V_min, V_max]
        if axes is None:
            _plt.figure(figsize=(6, 5))
            ax = _plt.gca()
        else:
            ax = axes
        vmin = float(min_intensity) if (min_intensity is not None) else (clim[0] if clim else None)
        vmax = float(max_intensity) if (max_intensity is not None) else (clim[1] if clim else None)
        im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       aspect='auto')

        # Axis labels based on orientation
        if orientation == "HK":
            ax.set_xlabel('H')
            ax.set_ylabel('K')
        elif orientation == "KL":
            ax.set_xlabel('K')
            ax.set_ylabel('L')
        elif orientation == "HL":
            ax.set_xlabel('H')
            ax.set_ylabel('L')
        else:
            ax.set_xlabel('U')
            ax.set_ylabel('V')

        # Title includes orthogonal axis slice value when available
        title = None
        if orientation in ("HK", "KL", "HL") and (orth_label is not None) and (orth_value is not None) and _np.isfinite(orth_value):
            title = f'{orientation} plane ({orth_label} = {orth_value:.3f})'
        else:
            title = f'{hkl} slice at origin {tuple(_np.array(origin).round(3))}'
        ax.set_title(title)

        _plt.colorbar(im, ax=ax, label='Intensity')
        if axes is None:
            _plt.show()

        if return_image:
            return img, extent
        return None

    def show_point_cloud(self, cloud, intensities=None, *, notebook=True,
                         point_size=2.0, cmap='viridis', opacity=1.0,
                         render_points_as_spheres=False, axes_labels=('H','K','L'),
                         clim=None, show_bounds=True, hide_out_of_range=True, opacity_range=None):
        """
        Render a point cloud in HKL space with advanced visualization options.

        This method provides comprehensive 3D point cloud visualization with support for
        multiple data formats, opacity control, intensity filtering, and interactive features.
        Supports both notebook and standalone rendering with customizable appearance.

        Usage:
            # Basic point cloud rendering
            da.show_point_cloud(data.points, data.intensities)
            
            # Advanced rendering with opacity control
            da.show_point_cloud(cloud, clim=(100, 1000), opacity_range=(0.1, 1.0))
            
            # High-quality spherical rendering
            da.show_point_cloud(data, render_points_as_spheres=True, point_size=5.0)

        Parameters:
            cloud: Point cloud data. Supported formats:
                - (points, intensities) tuple/list
                - Data object with .points and .intensities attributes
                - Dict with 'points' and 'intensities' keys
                - pv.PolyData with optional 'intensity' array
                - np.ndarray of shape (N,3) for points (provide intensities separately)
            intensities (array-like): Optional 1D array of length N for intensity values
            notebook (bool): Use notebook plotter (True) or regular plotter (False)
            point_size (float): Point size for rendering
            cmap (str): Colormap name for intensity visualization
            opacity (float): Point opacity [0..1] (ignored if opacity_range is used)
            render_points_as_spheres (bool): True for spherical glyphs, False for points
            axes_labels (tuple): Axis labels, default ('H','K','L')
            clim (tuple): Optional (vmin, vmax) intensity display limits
            show_bounds (bool): Whether to show coordinate bounds with labels
            hide_out_of_range (bool): Hide intensities outside clim range using LookupTable
            opacity_range (tuple): Optional (min_opacity, max_opacity) for intensity-based opacity

        Returns:
            PyVista rendering result (displays inline in notebooks)

        Raises:
            ImportError: If PyVista is not available
            TypeError: If cloud format is not supported

        Examples:
            # Basic visualization
            da.show_point_cloud(point_data, intensity_data)
            
            # High-contrast visualization with filtering
            da.show_point_cloud(data, clim=(50, 500), hide_out_of_range=True)
            
            # Opacity-based intensity mapping
            da.show_point_cloud(data, opacity_range=(0.2, 1.0), cmap='plasma')
        """
        import numpy as np
        import pyvista as pv

        # Normalize inputs to pv.PolyData + 'intensity' if available
        pts = None
        ints = None
        poly = None

        if isinstance(cloud, pv.PolyData):
            poly = cloud
            if ('intensity' not in poly.array_names) and (intensities is not None):
                poly['intensity'] = np.asarray(intensities, dtype=np.float32)
        elif hasattr(cloud, 'points') and hasattr(cloud, 'intensities'):
            # Data object
            pts = np.asarray(cloud.points, dtype=float)
            ints = np.asarray(cloud.intensities, dtype=float)
            poly = pv.PolyData(pts)
            poly['intensity'] = ints.astype(np.float32)
        elif isinstance(cloud, (tuple, list)) and len(cloud) >= 2:
            pts = np.asarray(cloud[0], dtype=float)
            ints = np.asarray(cloud[1], dtype=float)
            poly = pv.PolyData(pts)
            poly['intensity'] = ints.astype(np.float32)
        elif isinstance(cloud, dict) and ('points' in cloud):
            pts = np.asarray(cloud['points'], dtype=float)
            ints = np.asarray(cloud.get('intensities', intensities), dtype=float) if ('intensities' in cloud or intensities is not None) else None
            poly = pv.PolyData(pts)
            if ints is not None and ints.shape[0] == pts.shape[0]:
                poly['intensity'] = ints.astype(np.float32)
        elif isinstance(cloud, np.ndarray) and cloud.ndim == 2 and cloud.shape[1] == 3:
            pts = np.asarray(cloud, dtype=float)
            poly = pv.PolyData(pts)
            if intensities is not None:
                poly['intensity'] = np.asarray(intensities, dtype=np.float32)
        else:
            raise TypeError("Unsupported cloud format. Provide (points, intensities), Data, dict, pv.PolyData, or Nx3 ndarray.")

        # Create plotter
        p = pv.Plotter(notebook=bool(notebook))
        p.add_axes(xlabel=str(axes_labels[0]), ylabel=str(axes_labels[1]), zlabel=str(axes_labels[2]))

        # Add points layer (with optional LUT gating)
        has_intensity = ('intensity' in poly.array_names)
        if has_intensity:
            # Determine scalar range
            if clim is not None and len(clim) == 2:
                vmin, vmax = float(clim[0]), float(clim[1])
            else:
                arr = np.asarray(poly['intensity'])
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))

            if bool(hide_out_of_range) or (opacity_range is not None):
                # Build a LookupTable that hides out-of-range values and sets an in-range opacity ramp
                lut = pv.LookupTable(cmap=cmap)
                lut.scalar_range = (vmin, vmax)
                lut.below_range_color = 'black'
                lut.above_range_color = 'black'
                lut.below_range_opacity = 0.0
                lut.above_range_opacity = 0.0
                if opacity_range is not None:
                    o0, o1 = float(opacity_range[0]), float(opacity_range[1])
                    lut.apply_opacity([o0, o1])
                # Important: do not pass a uniform opacity float; let LUT control per-intensity opacity
                p.add_mesh(poly,
                           scalars='intensity',
                           cmap=lut,
                           render_points_as_spheres=bool(render_points_as_spheres),
                           point_size=float(point_size),
                           name='points')
            else:
                # Original behavior: simple colormap and uniform opacity
                p.add_mesh(poly,
                           scalars='intensity',
                           cmap=cmap,
                           clim=(vmin, vmax),
                           render_points_as_spheres=bool(render_points_as_spheres),
                           point_size=float(point_size),
                           opacity=float(opacity),
                           name='points')
        else:
            p.add_mesh(poly,
                       color='white',
                       render_points_as_spheres=bool(render_points_as_spheres),
                       point_size=float(point_size),
                       opacity=float(opacity),
                       name='points')

        # Optional bounds
        if bool(show_bounds):
            try:
                p.show_bounds(mesh=poly,
                              xtitle=str(axes_labels[0]),
                              ytitle=str(axes_labels[1]),
                              ztitle=str(axes_labels[2]),
                              bounds=poly.bounds)
            except Exception:
                pass

        return p.show()

    def show_vol(self, vol, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), cmap='jet'):
        """
        Display a 3D HKL volume with comprehensive rendering options.

        This method provides volume rendering for 3D HKL data with automatic grid construction,
        proper coordinate handling, and interactive visualization features.

        Usage:
            # Display volume from ImageData
            da.show_vol(pyvista_volume)
            
            # Display volume from NumPy array
            da.show_vol(numpy_volume, spacing=(0.1, 0.1, 0.1))

        Parameters:
            vol: Volume data. Supported formats:
                - pv.ImageData with cell_data['intensity']
                - np.ndarray (D,H,W) of intensity values (cell-centered)
            spacing (tuple): Voxel spacing (ΔH, ΔK, ΔL) for NumPy arrays
            origin (tuple): Grid origin (H0, K0, L0) for NumPy arrays
            cmap (str): Colormap name for volume rendering

        Returns:
            PyVista rendering result (displays inline in notebooks)

        Raises:
            ValueError: If volume format is invalid or missing intensity data
            TypeError: If vol is not a supported format

        Examples:
            # Basic volume rendering
            da.show_vol(volume_data)
            
            # High-resolution volume with custom spacing
            da.show_vol(numpy_vol, spacing=(0.05, 0.05, 0.05), cmap='plasma')
        """
        # Normalize input to a PyVista ImageData with cell_data['intensity']
        if isinstance(vol, pv.ImageData):
            grid = vol
            # Ensure 'intensity' exists
            if 'intensity' not in grid.cell_data and 'intensity' in grid.point_data:
                # Convert to cell_data for consistent D×H×W handling
                grid = grid.point_data_to_cell_data(pass_point_data=False)
            if 'intensity' not in grid.cell_data:
                raise ValueError("ImageData must have cell_data['intensity'] for volume rendering.")
        elif isinstance(vol, np.ndarray):
            if vol.ndim != 3:
                raise ValueError("NumPy volume must be 3D shaped (D, H, W).")
            # Build grid: dimensions = cells + 1 (VTK requirement)
            dims_cells = np.array(vol.shape, dtype=int)
            grid = pv.ImageData()
            grid.dimensions = (dims_cells + 1).tolist()
            grid.spacing = tuple(float(x) for x in spacing)
            grid.origin = tuple(float(x) for x in origin)
            # For VTK/PyVista, flatten with Fortran order to match D×H×W cell-layout
            grid.cell_data['intensity'] = np.asarray(vol, dtype=np.float32).flatten(order='F')
        else:
            raise TypeError("vol must be a pyvista.ImageData or a NumPy ndarray (D,H,W).")

        # Compute display clim from data range if available
        try:
            data = np.asarray(grid.cell_data['intensity'])
            clim = (float(np.min(data)), float(np.max(data)))
        except Exception:
            clim = None

        # Render inline
        plotter = pv.Plotter(notebook=True)
        plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        plotter.add_volume(grid, scalars='intensity', cmap=cmap, clim=clim, name='cloud_volume', show_scalar_bar=True)
        try:
            plotter.show_bounds(mesh=grid, xtitle='H Axis', ytitle='K Axis', ztitle='L Axis', bounds=grid.bounds)
        except Exception:
            pass
        return plotter.show()

    def create_vol(self, points, intensities):
        """
        Create a 3D volume from point cloud data using adaptive interpolation.

        This method converts point cloud data into a structured 3D volume suitable for
        visualization and analysis, with automatic resolution selection based on data density.

        Usage:
            # Create volume from point cloud
            volume = da.create_vol(data.points, data.intensities)
            
            # Display the created volume
            da.show_vol(volume)

        Parameters:
            points (array-like): 3D point coordinates with shape (N, 3)
            intensities (array-like): Intensity values with shape (N,)

        Returns:
            pv.ImageData: Interpolated volume with cell_data['intensity']

        Examples:
            # Create and display volume
            vol = da.create_vol(point_data, intensity_data)
            da.show_vol(vol, cmap='viridis')
        """
        cloud = pv.PolyData(points)
        cloud['intensity'] = intensities.astype('float32')
        minb = cloud.points.min(axis=0)
        maxb = cloud.points.max(axis=0)
        data_range = maxb - minb
        padding = data_range * 0.10
        grid_min = minb - padding
        grid_max = maxb + padding
        grid_range = grid_max - grid_min

        # Resolution: adaptive cells per axis (mirror slicer thresholds)
        total_points = int(points.shape[0]) if hasattr(points, "shape") else len(points)
        if total_points >= 5_000_000:
            refine_cells = 250
        elif total_points >= 2_000_000:
            refine_cells = 275
        elif total_points >= 500_000:
            refine_cells = 300
        else:
            refine_cells = 300
        spacing = grid_range / refine_cells
        dimensions = np.ceil(grid_range / spacing).astype(int) + 1

        # Create grid
        grid = pv.ImageData()
        grid.origin = grid_min
        grid.spacing = spacing
        grid.dimensions = dimensions

        # Interpolate cloud into volume
        optimal_radius = float(np.mean(spacing) * 2.5)
        vol = grid.interpolate(cloud, radius=optimal_radius, sharpness=1.5, null_value=0.0)

        return vol

    def load_data(self, file_path: Optional[str] = None):
        """
        Load 3D point data and intensities from HDF5 file.

        This method uses the project's HDF5Loader for consistent data loading
        and returns a Data object containing points and intensities.

        Usage:
            # Load data from file
            data = da.load_data('/path/to/file.h5')
            print(f"Loaded {len(data.points)} points")

        Parameters:
            file_path (str, optional): Path to HDF5 file. If None, uses cached path

        Returns:
            Data: Object containing points (N,3) and intensities (N,) arrays

        Raises:
            FileNotFoundError: If no file path provided and none cached
            Exception: If file loading fails

        Examples:
            # Load and visualize data
            data = da.load_data('experiment_data.h5')
            da.show_point_cloud(data)
        """
        path = file_path or getattr(self, 'file_path', None)
        if not path:
            raise FileNotFoundError("No file path provided to load_data and none set in DashAnalysis.")

        try:
            loader = HDF5Loader()
            points_3d, intensities, num_images, shape = loader.load_h5_to_3d(path)
            return Data(points_3d, intensities)
        except Exception as e:
            raise Exception(f"Failed to load data using HDF5Loader: {e}")

    def show_meta(self, file_path, *, style="text", raw=False, include_unknown=True, 
                  float_precision=6, summarize_datasets=True):
        """
        Display metadata information from HDF5 file.

        This method provides comprehensive metadata inspection for HDF5 files
        using the project's HDF5Loader with customizable output formatting.

        Usage:
            # Display file metadata
            da.show_meta('/path/to/file.h5')
            
            # Raw metadata with high precision
            da.show_meta('data.h5', raw=True, float_precision=10)

        Parameters:
            file_path (str): Path to HDF5 file
            style (str): Output style ('text', 'json', etc.)
            raw (bool): Include raw metadata
            include_unknown (bool): Include unknown/unrecognized fields
            float_precision (int): Decimal precision for floating point values
            summarize_datasets (bool): Include dataset summaries

        Returns:
            Metadata information (format depends on style parameter)

        Raises:
            FileNotFoundError: If file path is invalid

        Examples:
            # Basic metadata display
            da.show_meta('data.h5')
            
            # Detailed JSON output
            da.show_meta('data.h5', style='json', raw=True)
        """
        if not file_path:
            raise FileNotFoundError("No file path provided to load_data and none set in DashAnalysis.")

        loader = HDF5Loader()
        return loader.get_file_info(file_path, style=style, raw=raw, include_unknown=include_unknown, 
                                   float_precision=float_precision, summarize_datasets=summarize_datasets)

    def _build_vol(self, data):
        """
        Build a PyVista ImageData grid from loaded volume data.

        This internal method constructs PyVista grids from various data formats,
        mirroring the viewer's approach for consistent handling.

        Parameters:
            data: Volume data in supported formats:
                - (volume_np, shape_tuple) tuple from HDF5Loader
                - {'volume': np.ndarray, 'metadata': {...}} dict with metadata

        Returns:
            pv.ImageData: Grid with cell_data['intensity'] populated

        Raises:
            ImportError: If PyVista is not available
            ValueError: If data format is unsupported or invalid
        """
        import numpy as np
        if pv is None:
            raise ImportError("PyVista is required to build the volume. Install pyvista and retry.")

        # Extract volume and optional metadata
        meta = {}
        if isinstance(data, tuple) and len(data) >= 1:
            volume = data[0]
        elif isinstance(data, dict):
            volume = data.get('volume')
            meta = data.get('metadata') or {}
        else:
            raise ValueError("Unsupported data format for _build_vol; pass (volume, shape) tuple or {'volume': ..., 'metadata': ...} dict")

        if volume is None or not hasattr(volume, 'shape'):
            raise ValueError("Invalid volume provided to _build_vol")

        # Determine cell-centered dimensions
        try:
            dims_cells_meta = meta.get('grid_dimensions_cells', None)
            if dims_cells_meta is not None:
                dims_cells = np.array(dims_cells_meta, dtype=int)
            else:
                dims_cells = np.array(volume.shape, dtype=int)
        except Exception:
            dims_cells = np.array(volume.shape, dtype=int)

        # Create grid with points-based dimensions (= cells + 1)
        grid = pv.ImageData()
        grid.dimensions = (dims_cells + 1).tolist()

        # Spacing and origin from metadata or defaults
        spacing = meta.get('voxel_spacing') or (1.0, 1.0, 1.0)
        origin = meta.get('grid_origin') or (0.0, 0.0, 0.0)
        try:
            grid.spacing = tuple(float(x) for x in spacing)
        except Exception:
            grid.spacing = (1.0, 1.0, 1.0)
        try:
            grid.origin = tuple(float(x) for x in origin)
        except Exception:
            grid.origin = (0.0, 0.0, 0.0)

        # Assign intensity scalars to cell_data using recorded array order
        arr_order = (meta.get('array_order') or 'F') if isinstance(meta, dict) else 'F'
        try:
            grid.cell_data["intensity"] = volume.flatten(order=arr_order)
        except Exception:
            grid.cell_data["intensity"] = volume.flatten(order="F")

        return grid

    def _calculate_smart_radius(self, points, u_extent, v_extent, grid_shape):
        """
        Calculate adaptive interpolation radius based on point density and grid resolution.
        
        This method computes an optimal interpolation radius to minimize gaps in slice
        interpolation while maintaining appropriate resolution for the given data density.
        
        Parameters:
            points (array-like): Points used for interpolation
            u_extent (tuple): (u_min, u_max) extent in U direction
            v_extent (tuple): (v_min, v_max) extent in V direction
            grid_shape (tuple): (height, width) of target grid
            
        Returns:
            float: Optimal interpolation radius
        """
        import numpy as _np
        
        if len(points) == 0:
            return 1e-6
            
        # Calculate area and point density
        u_range = max(u_extent[1] - u_extent[0], 1e-6)
        v_range = max(v_extent[1] - v_extent[0], 1e-6)
        area = u_range * v_range
        point_density = len(points) / area
        
        # Calculate average point spacing (approximate)
        avg_point_spacing = 1.0 / _np.sqrt(max(point_density, 1e-12))
        
        # Calculate average grid cell size
        avg_cell_u = u_range / max(grid_shape[1], 1)
        avg_cell_v = v_range / max(grid_shape[0], 1)
        avg_cell_size = (avg_cell_u + avg_cell_v) * 0.5
        
        # Define density thresholds (points per unit area)
        threshold_sparse = 0.5
        threshold_medium = 2.0
        
        # Adaptive radius multiplier based on point density
        if point_density < threshold_sparse:
            # radius_multiplier = 4.5  # Very aggressive for very sparse data (legacy)
            radius_multiplier = 2.0  # Tuned down for performance with large shapes
        elif point_density < threshold_medium:
            # radius_multiplier = 3.8  # Moderate for medium density (legacy)
            radius_multiplier = 1.8  # Tuned down to limit neighbors per sample
        else:
            # radius_multiplier = 3.2  # Conservative for dense data (legacy)
            radius_multiplier = 1.6  # Tuned down; keeps visuals similar while cutting work
        
        # Calculate radius ensuring it's at least as large as average point spacing
        # and scales appropriately with grid resolution
        radius_from_density = avg_point_spacing * 1.5
        radius_from_grid = radius_multiplier * avg_cell_size
        
        optimal_radius = max(radius_from_density, radius_from_grid, 1e-6)
        
        return float(optimal_radius)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_hkl_axis(hkl_vector, tolerance=1e-6, max_denominator=12):
    """
    Format a 3-vector in HKL coordinates as a readable string.
    
    This function converts HKL coordinate vectors into human-readable expressions
    with rational number approximation and proper mathematical formatting.
    
    Parameters:
        hkl_vector (array-like): 3-element vector with coefficients for [H, K, L]
        tolerance (float): Threshold for considering a coefficient as zero
        max_denominator (int): Maximum denominator for rational approximation
        
    Returns:
        str: Formatted expression like "H", "H/2", "H + K", "0.866H + 0.5K", etc.
        
    Examples:
        format_hkl_axis([1, 0, 0])      # Returns "H"
        format_hkl_axis([0.5, 1, 0])    # Returns "H/2 + K"
        format_hkl_axis([1, 1, 0])      # Returns "H + K"
        format_hkl_axis([0, 0, 0.5])    # Returns "L/2"
    """
    import numpy as np
    from fractions import Fraction
    
    h, k, l = np.asarray(hkl_vector, dtype=float)[:3]
    
    def rationalize(x):
        """Convert float to rational if close to a simple fraction."""
        if abs(x) < tolerance:
            return 0, 1
        try:
            frac = Fraction(x).limit_denominator(max_denominator)
            if abs(float(frac) - x) < tolerance:
                return frac.numerator, frac.denominator
        except:
            pass
        return x, 1
    
    def format_term(coeff, label):
        """Format a single term like '2H', 'H/3', '-K', etc."""
        if abs(coeff) < tolerance:
            return ""
        
        num, den = rationalize(coeff)
        if isinstance(num, (int, np.integer)) and isinstance(den, (int, np.integer)):
            if den == 1:
                if num == 1:
                    return label
                elif num == -1:
                    return f"-{label}"
                else:
                    return f"{num}{label}"
            else:
                if num == 1:
                    return f"{label}/{den}"
                elif num == -1:
                    return f"-{label}/{den}"
                else:
                    return f"{num}{label}/{den}"
        else:
            # Fallback to decimal
            if abs(coeff - 1.0) < tolerance:
                return label
            elif abs(coeff + 1.0) < tolerance:
                return f"-{label}"
            else:
                return f"{coeff:.3g}{label}"
    
    terms = []
    for coeff, label in [(h, 'H'), (k, 'K'), (l, 'L')]:
        term = format_term(coeff, label)
        if term:
            terms.append(term)
    
    if not terms:
        return "0"
    
    # Join terms with proper signs
    result = terms[0]
    for term in terms[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    
    return result


def show_slice(mesh, cmap='viridis'):
    """
    Convenience function for displaying slices.
    
    This function provides a simple interface for slice visualization
    by delegating to the DashAnalysis.show_slice method.
    
    Parameters:
        mesh: Slice mesh or data to display
        cmap (str): Colormap for visualization
        
    Returns:
        Result from DashAnalysis.show_slice
    """
    return DashAnalysis().show_slice(mesh, cmap=cmap)
