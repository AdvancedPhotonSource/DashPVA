# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""
Dash Analysis Module

Object-oriented analysis tools for HKL data processing, including slice extraction,
data manipulation, and visualization utilities for DashPVA.

This module provides comprehensive tools for loading, processing, and analyzing
HKL crystallographic data, including slice extraction, coordinate transformations,
and data filtering operations.
"""

import glob
import os
from typing import Optional

import numpy as np

import dashpva.settings as app_settings

# We try to keep PyVista optional; only import when building a grid or plotting
try:
    import pyvista as pv
except Exception:
    pv = None

try:
    from dashpva.utils.hdf5_loader import HDF5Loader
except Exception:
    HDF5Loader = None

try:
    from dashpva.utils.rsm_converter import RSMConverter
except Exception:
    RSMConverter = None

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

class Group:
    """HDF5 group with its members as attributes, so Jupyter can tab-complete them."""

    def __init__(self, members: dict):
        self.__dict__.update(members)

    def __repr__(self):
        return f"Group({', '.join(self.__dict__)})"


class Data:
    """
    Container class for 3D point data and intensities.
    
    This class encapsulates point cloud data along with intensity values
    for HKL crystallographic analysis.
    
    Attributes:
        points (np.ndarray): 3D point coordinates with shape (N, 3); None until load_3d
        intensities (np.ndarray): Intensity values with shape (N,)
        images (np.ndarray): Raw detector frames from /entry/data/data
        metadata (dict): Nested /entry/data/metadata tree (HKL, ca, rois)
        entry: The file's /entry tree with attribute access, e.g. data.entry.data.metadata.ca.eta
    """
    
    def __init__(self, points: np.ndarray=None, intensities: np.ndarray=None, metadata: dict=None, num_images: int=0, shape: tuple=None,
                 images: np.ndarray=None, entry: Group=None, file_path: str=None):
        """
        Initialize Data object.
        
        Args:
            points: 3D point coordinates with shape (N, 3)
            intensities: Intensity values with shape (N,)
            images: Raw detector frames (num_images, H, W)
            entry: The file's /entry tree as a Group
        """
        self.points = points
        self.intensities = intensities
        self.metadata = metadata
        self.num_images = num_images
        self.shape = shape
        self.images = images
        self.entry = entry
        self.file_path = file_path

    def __getitem__(self, index):
        """Return a raw detector frame by index."""
        if self.images is None:
            raise TypeError("This Data object does not contain detector images")
        return self.images[index]



class DataStack:
    """
    Several scans as one list you can slice; each scan loads (load_3d) when it is first used.

    stack[3] is that scan's Data; stack[2:8], stack[::2] and stack[[0, 4, 9]] are smaller
    DataStacks. Pass a DataStack to DashAnalysis.bin_stack or show_stack.

    Attributes:
        files (list): Scan file paths, in order
        names (list): File names without the folder
    """

    def __init__(self, files, loader, cache=False, _store=None):
        self.files = [str(f) for f in files]
        self._loader = loader
        self._cache = cache
        self._store = {} if _store is None else _store  # shared by slices of one stack

    @property
    def names(self):
        return [os.path.basename(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DataStack(self.files[idx], self._loader, self._cache, self._store)
        if isinstance(idx, (list, tuple, np.ndarray, range)):
            idx = np.asarray(idx)
            if idx.dtype == bool:
                idx = np.flatnonzero(idx)
            return DataStack([self.files[i] for i in idx], self._loader, self._cache, self._store)
        f = self.files[idx]
        if f in self._store:
            return self._store[f]
        data = self._loader(f)
        if self._cache:
            self._store[f] = data
        return data

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self):
        names = self.names
        shown = names if len(names) <= 6 else names[:3] + ['...'] + names[-2:]
        return f"DataStack({len(names)} scans: {', '.join(shown)})"


class SliceStack:
    """Slices sharing one live plane across a DataStack."""

    def __init__(self, names, loader):
        self.names = list(names)
        self._loader = loader

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return SliceStack([self.names[i] for i in indices], lambda j: self._loader(indices[j]))
        return self._loader(index)

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __repr__(self):
        return f"SliceStack({len(self)} slices)"


class LineCutData(Data):
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
        raise NotImplementedError


class SliceData:
    """
    Container class for slice data and metadata.
    
    This class encapsulates slice data along with orientation information
    and provides methods for data manipulation and access.
    
    Attributes:
        data (Data): The slice data object
        orientation (int): Orientation identifier for the slice
    """
    
    def __init__(self, data=None, orientation=0, shape=(512, 512)):
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
        self.last_slice = None
        self.slice_orientation = None
        self.stack_slices = None
        self.file_path = None
        self._loaded_stack = None


# ============================================================================
# MAIN METHODS (Ordered by complexity/length - longest to shortest)
# ============================================================================

    def slice_data(
        self,
        data: 'Data | tuple[np.ndarray, np.ndarray] | dict | pv.ImageData | np.ndarray',
        hkl: str | tuple[float, float, float] | None = None,
        normal: tuple[float, float, float] | None = None,
        shape: tuple[int, int] = (512, 512),
        slab_thickness: float | None = None,
        clamp_to_bounds: bool = True,
        spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
        grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        show: bool = True,
        axes: tuple | None = None,
        intensity_range: tuple[float | None, float | None] | None = None,
        orientation: dict | None = None,
        H: float | tuple[float, float] | tuple[float, float, str] | None = None,
        K: float | tuple[float, float] | tuple[float, float, str] | None = None,
        L: float | tuple[float, float] | tuple[float, float, str] | None = None,
        **kwargs,
    ) -> 'pv.PolyData':
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
                - None: Uses the 'HL' preset unless normal, axes, orientation, or H/K/L is provided
            normal (array-like): Slice plane normal vector (3,). Overridden by hkl presets
            shape (tuple): Resolution (rows, cols) for sampling the plane when slicing points
            slab_thickness (float): Thickness of selection slab around plane for point slicing
            clamp_to_bounds (bool): Clamp origin to dataset bounds
            orientation (dict): {'hkl': (H, K, L), 'normal': (h, k, l)}, e.g. da.slice_orientation
                saved from a 3D slice_plane; overrides hkl and normal
            H, K, L: Define the slice in plain H/K/L instead of hkl/normal/axes. Give one of them
                as a number (flat plane at that value) or (start, stop, along) (tilted: it goes
                from start to stop along the named axis), and the other two as (min, max) ranges
                (default: the data range). The plot axes are then those two, e.g.
                slice_data(data, L=1.5, H=(0, 2), K=(-1, 1)) or L=(1.0, 2.0, 'H').
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
        if orientation is not None:
            hkl, normal = tuple(orientation['hkl']), orientation['normal']
        if H is not None or K is not None or L is not None:
            return self._slice_hkl_ranges(data, (H, K, L), shape, slab_thickness, intensity_range, show, **kwargs)
        if hkl is None and normal is None and axes is None:
            hkl = 'HL'

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

        # Bounds
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
                _imin = None
                _imax = None
            mask_int = np.ones(intensities.shape[0], dtype=bool)
            if _imin is not None:
                mask_int &= (intensities >= _imin)
            if _imax is not None:
                mask_int &= (intensities <= _imax)
        else:
            mask_int = np.ones(intensities.shape[0], dtype=bool)

        mask_contrib = mask_slab & mask_int

        # For extent estimation, prefer slab mask (geometry) even if intensity filter removes all
        if use_slab and np.any(mask_slab):
            pts_for_extent = points[mask_slab]
        elif np.any(mask_slab):
            pts_for_extent = points  # no slab: every point counts, skip the copy
        else:
            pts_for_extent = points

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
            k = int(np.argmax(np.abs(n_vec)))
            if abs(float(n_vec[k])) >= 0.95:
                a, b = np.eye(3)[[(1, 0, 0)[k], (2, 2, 1)[k]]]
                u = a - a.dot(n_vec) * n_vec
                u = u / np.linalg.norm(u)
                v = b - b.dot(n_vec) * n_vec - b.dot(u) * u
                v = v / np.linalg.norm(v)
            u = u if u[np.argmax(np.abs(u))] > 0 else -u
            v = v if v[np.argmax(np.abs(v))] > 0 else -v

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
        # ORIGINAL (kept for reference):
        # plane = pv.Plane(center=origin_vec.tolist(), direction=n_vec.tolist(),
        #                  i_size=i_size, j_size=j_size, i_resolution=W, j_resolution=H)
        # Use W-1/H-1 so plane.n_points == H*W; reduces work and aligns with stored slice_shape
        plane = pv.Plane(center=origin_vec.tolist(), direction=n_vec.tolist(),
                         i_size=i_size, j_size=j_size, i_resolution=W-1, j_resolution=H-1)
        gu, gv = np.meshgrid(np.linspace(U_min, U_max, W), np.linspace(V_min, V_max, H))
        plane.points = origin_vec + np.outer(gu.ravel(), u / u.dot(u)) + np.outer(gv.ravel(), v / v.dot(v))

        # Use smart radius calculation to minimize gaps
        optimal_radius = self._calculate_smart_radius(
            pts_for_extent, 
            (U_min, U_max), 
            (V_min, V_max), 
            (H, W)
        )

        # Points farther than the radius from the plane get no kernel weight, so dropping them
        # leaves the result unchanged and keeps the interpolation from scanning the whole cloud
        mask_near = mask_contrib & (np.abs(d_signed) <= optimal_radius)

        # Choose contributing cloud based on slab and intensity range
        if np.any(mask_near):
            cloud_contrib = pv.PolyData(points[mask_near])
            cloud_contrib['intensity'] = intensities[mask_near].astype('float32')
            no_contrib = False
        else:
            no_contrib = True

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
            if not np.any(mask_contrib):
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

    def line_cut(self, spec, param=None, vol=None, hkl='HL', origin=None, shape=(512, 512),
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
                        sl = self.slice_data(vol, shape=(512, 512), clamp_to_bounds=True)
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
        except Exception:
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
                ax_img.set_xlabel('U')
                ax_img.set_ylabel('V')
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
                    ax_img.set_xlabel('U')
                    ax_img.set_ylabel('V')
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
            ax_img.imshow(img, origin='lower', extent=extent, cmap='viridis', aspect='auto')
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
            ax_prof.set_xlabel('Distance')
            ax_prof.set_ylabel('Intensity')
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
                ax_prof.relim()
                ax_prof.autoscale_view()
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

            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            fig.canvas.mpl_connect('button_release_event', on_release)

            plt.tight_layout()
            plt.show()

        return lc

    def show_slice(self, slice_mesh, shape=None, cmap='viridis',
                   clim=None, min_intensity=None, max_intensity=None, axes=None, return_image=False,
                   axis_display='hkl', show_grid=False, shape_data=True, *, navigate=False,
                   plane='HL', slab_thickness=None, return_slice=False,
                   show_info=True):
        """
        Display a pre-computed slice mesh as a 2D raster with interactive features.

        This method visualizes slice data that has already been created by slice_data(),
        providing rasterization, intensity range calculation, and interactive hover tooltips.

        Usage:
            # First create the slice
            slice_mesh = da.slice_data(data, hkl='HK', shape=(100, 100))
            
            # Then display it
            da.show_slice(slice_mesh)
            
            # Or display with custom settings
            da.show_slice(slice_mesh, cmap='hot', clim=(0, 1000))
            
            # Display with simple U/V labels
            da.show_slice(slice_mesh, axis_display='uv')

            # Navigate a new slice through one dataset
            moving_slice = da.show_slice(data, navigate=True, return_slice=True)  # HL by default

            # Navigate an existing slice through the scans previously given to load_data
            stack_slices = da.show_slice(slice_mesh, navigate=True, return_slice=True)
            da.show_stack(stack_slices, view='3d')

        Parameters:
            slice_mesh: Slice mesh from slice_data. With navigate=True, this is the Data or
                volume to slice. When stack is provided, this is the existing slice whose plane
                is applied across the stack.
            shape (tuple | None): Raster resolution as (rows, columns). None uses the stored
                slice shape or the default resolution.
            cmap (str): Matplotlib colormap used for slice intensities.
            clim (tuple | None): Display limits as (minimum, maximum). Either value may be None.
            min_intensity (float | None): Hide values below this threshold and use it as vmin.
            max_intensity (float | None): Hide values above this threshold and use it as vmax.
            axes (matplotlib.axes.Axes | None): Existing axes to draw into. None creates a figure.
            return_image (bool): Return the rasterized (image, extent) instead of only displaying it.
            axis_display (str): Axis label and coordinate format. Options:
                - 'hkl' (default): Shows formatted HKL expressions (e.g., "H + K", "L/2")
                - 'uv': Shows simple "U" and "V" labels
                - 'pixel': Axes count pixels (column, row); hover still reads H, K, L
            show_grid (bool): Draw a grid over the 2D axes.
            shape_data (bool): Preserve physical pixel size when shape changes.
            navigate (bool): Add navigation controls. For source data, move a slice through that
                dataset. For an existing slice, apply its plane across every scan remembered by
                load_data and add a scan selector.
            plane (str): Initial navigation plane: 'HL' (default), 'HK', or 'KL'.
            slab_thickness (float | None): Include points within this distance on either side of
                the plane while navigating.
            return_slice (bool): Return the live PolyData slice, or a lazy SliceStack when an
                existing slice navigates the scans remembered by load_data.
            show_info (bool): Show slice corner information beside a static plot. Navigation uses
                its own separate information panel.

        Returns:
            tuple or None: If return_image=True, returns (img, extent) where:
                - img: np.ndarray of rasterized intensity values
                - extent: [U_min, U_max, V_min, V_max] physical coordinate bounds
            Otherwise returns None and displays the slice

        Raises:
            ImportError: If matplotlib is not available
            TypeError: If slice_mesh is not a pv.PolyData
            ValueError: If slice_mesh lacks required data

        Examples:
            # Basic display
            slice_mesh = da.slice_data(volume, hkl='HK')
            da.show_slice(slice_mesh)
            
            # High-resolution display with filtering
            da.show_slice(slice_mesh, shape=(1024, 1024), min_intensity=50)
            
            # Get image data for line cut analysis
            img, extent = da.show_slice(slice_mesh, return_image=True)
            line_data = da.line_cut('zero', param=(0.5, 'x'), vol=(img, extent))
            
            # Display with simple U/V labels
            da.show_slice(slice_mesh, axis_display='uv')
        """
        if navigate:
            existing_slice = isinstance(slice_mesh, pv.PolyData) and 'slice_origin' in slice_mesh.field_data
            if existing_slice and self._loaded_stack is not None:
                source = self._loaded_stack
                start = slice_mesh
            else:
                source = slice_mesh
                start = None
            return self._show_slice_navigator(
                source,
                plane=plane,
                start=start,
                shape=(256, 256) if shape is None else shape,
                slab_thickness=slab_thickness,
                return_slice=return_slice,
                cmap=cmap,
                clim=clim,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
                axis_display=axis_display,
                show_grid=show_grid,
                shape_data=shape_data,
                show_info=False,
            )
        if plt is None:
            raise ImportError("matplotlib is required for show_slice()")
        
        if not isinstance(slice_mesh, pv.PolyData):
            raise TypeError("slice_mesh must be a pv.PolyData object from slice_data()")

        import numpy as _np

        # Get metadata from slice mesh
        normal_fd = getattr(slice_mesh, 'field_data', {}).get('slice_normal', None)
        origin_fd = getattr(slice_mesh, 'field_data', {}).get('slice_origin', None)
        u_axis_fd = getattr(slice_mesh, 'field_data', {}).get('slice_u_axis', None)
        v_axis_fd = getattr(slice_mesh, 'field_data', {}).get('slice_v_axis', None)
        u_label_fd = getattr(slice_mesh, 'field_data', {}).get('slice_u_label', None)
        v_label_fd = getattr(slice_mesh, 'field_data', {}).get('slice_v_label', None)
        
        # Resolve shape: prefer stored slice_shape from slice_data, else given shape, else 512x512
        stored_shape = getattr(slice_mesh, 'field_data', {}).get('slice_shape', None)
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            H, W = int(shape[0]), int(shape[1])
        elif stored_shape is not None and len(stored_shape) >= 2:
            H, W = int(stored_shape[0]), int(stored_shape[1])
        else:
            H, W = 512, 512
        H = max(int(H), 1)
        W = max(int(W), 1)
        
        # Rasterize the slice mesh
        pts = _np.asarray(slice_mesh.points, dtype=float)
        try:
            vals = _np.asarray(slice_mesh['intensity'], dtype=float).reshape(-1)
        except Exception:
            raise ValueError("slice_mesh must have 'intensity' point data array")
        if vals.size == slice_mesh.n_cells and vals.size != pts.shape[0]:
            pts = _np.asarray(slice_mesh.cell_centers().points, dtype=float)
        
        if pts.size == 0 or vals.size == 0:
            raise ValueError("slice_mesh contains no valid points to rasterize")
        
        # Get normal and origin from metadata
        normal = _np.asarray(normal_fd if normal_fd is not None else [0, 0, 1], dtype=float)
        origin = _np.asarray(origin_fd if origin_fd is not None else slice_mesh.center, dtype=float)
        
        # Normalize normal
        n_norm = float(_np.linalg.norm(normal))
        if n_norm > 0:
            normal = normal / n_norm
        else:
            normal = _np.array([0.0, 0.0, 1.0], dtype=float)
        
        # --- START Infer orientation from normal START --- #
        X = _np.array([1.0, 0.0, 0.0], dtype=float)  # H
        Y = _np.array([0.0, 1.0, 0.0], dtype=float)  # K
        Z = _np.array([0.0, 0.0, 1.0], dtype=float)  # L
        tolerance = 2.0 if 'slice_title' in getattr(slice_mesh, 'field_data', {}) else 0.95
        dX = abs(float(_np.dot(normal, X)))
        dY = abs(float(_np.dot(normal, Y)))
        dZ = abs(float(_np.dot(normal, Z)))
        
        if dZ >= tolerance:
            # HK plane
            U = pts[:, 0]
            V = pts[:, 1]
            orientation = "HK"
            orth_label = "L"
            orth_value = float(origin[2])
        elif dX >= tolerance:
            # KL plane
            U = pts[:, 1]
            V = pts[:, 2]
            orientation = "KL"
            orth_label = "H"
            orth_value = float(origin[0])
        elif dY >= tolerance:
            # HL plane
            U = pts[:, 0]
            V = pts[:, 2]
            orientation = "HL"
            orth_label = "K"
            orth_value = float(origin[1])
        else:
            # Custom orientation - use stored axes if available
            if u_axis_fd is not None and v_axis_fd is not None:
                u = _np.asarray(u_axis_fd, dtype=float)
                v = _np.asarray(v_axis_fd, dtype=float)
            else:
                # Build orthonormal basis
                world_axes = [X, Y, Z]
                ref = world_axes[0]
                for ax in world_axes:
                    if abs(float(_np.dot(ax, normal))) < 0.9:
                        ref = ax
                        break
                u = _np.cross(normal, ref)
                u_norm = float(_np.linalg.norm(u))
                if u_norm > 0:
                    u = u / u_norm
                else:
                    u = _np.array([1.0, 0.0, 0.0], dtype=float)
                v = _np.cross(normal, u)
                v_norm = float(_np.linalg.norm(v))
                if v_norm > 0:
                    v = v / v_norm
                else:
                    v = _np.array([0.0, 1.0, 0.0], dtype=float)
            
            # Project points
            U = pts.dot(u)
            V = pts.dot(v)
            orientation = "Custom"
            orth_label = None
            orth_value = None
        # --- END Infer orientation from normal END --- #
        # Apply intensity filtering if requested
        if (min_intensity is not None) or (max_intensity is not None):
            mask = _np.ones(vals.shape, dtype=bool)
            if min_intensity is not None:
                mask &= (vals >= float(min_intensity))
            if max_intensity is not None:
                mask &= (vals <= float(max_intensity))
            if _np.any(mask):
                U = U[mask]
                V = V[mask]
                vals = vals[mask]
            else:
                # No points pass filter
                U = _np.array([])
                V = _np.array([])
                vals = _np.array([])
        
        # Calculate extents
        if len(U) > 0:
            U_min, U_max = float(_np.min(U)), float(_np.max(U))
            V_min, V_max = float(_np.min(V)), float(_np.max(V))
        else:
            U_min, U_max = -0.5, 0.5
            V_min, V_max = -0.5, 0.5
        
        if U_max == U_min:
            U_min -= 0.5
            U_max += 0.5
        if V_max == V_min:
            V_min -= 0.5
            V_max += 0.5

        # If caller changed shape relative to stored slice_shape, expand/shrink HKL extents
        # to keep per-pixel physical size consistent. This makes axis ranges change with shape.
        try:
            stored_shape_fd = getattr(slice_mesh, 'field_data', {}).get('slice_shape', None)
            if shape_data and (stored_shape_fd is not None) and isinstance(shape, (tuple, list)) and (len(shape) == 2):
                orig_H = int(stored_shape_fd[0])
                orig_W = int(stored_shape_fd[1])
                new_H = int(H)
                new_W = int(W)
                if (orig_H > 0) and (orig_W > 0) and ((new_H != orig_H) or (new_W != orig_W)):
                    u_center = 0.5 * (U_min + U_max)
                    v_center = 0.5 * (V_min + V_max)
                    # Compute original per-pixel sizes; fallback to current ranges if degenerate
                    u_pp = (U_max - U_min) / float(orig_W) if (U_max != U_min and orig_W > 0) else (U_max - U_min)
                    v_pp = (V_max - V_min) / float(orig_H) if (V_max != V_min and orig_H > 0) else (V_max - V_min)
                    new_u_range = float(u_pp) * float(new_W)
                    new_v_range = float(v_pp) * float(new_H)
                    U_min = float(u_center) - 0.5 * float(new_u_range)
                    U_max = float(u_center) + 0.5 * float(new_u_range)
                    V_min = float(v_center) - 0.5 * float(new_v_range)
                    V_max = float(v_center) + 0.5 * float(new_v_range)
        except Exception:
            # Be permissive; if anything fails here, continue with original extents
            pass
        
        # Rasterize to image
        if len(vals) > 0:
            # Direct image placement for plane-generated slices (no histogram2d re-binning)
            is_plane_grid = False
            try:
                stored_shape_fd = getattr(slice_mesh, 'field_data', {}).get('slice_shape', None)
                if (stored_shape_fd is not None) and (int(stored_shape_fd[0]) * int(stored_shape_fd[1]) == int(pts.shape[0])):
                    is_plane_grid = True
            except Exception:
                is_plane_grid = False

            if is_plane_grid and (pts.shape[0] == (H * W)):
                img = _np.asarray(vals, dtype=_np.float32).reshape(H, W)
            else:
                sum_img, _, _ = _np.histogram2d(V, U, bins=[H, W],
                                                range=[[V_min, V_max], [U_min, U_max]],
                                                weights=vals)
                cnt_img, _, _ = _np.histogram2d(V, U, bins=[H, W],
                                                range=[[V_min, V_max], [U_min, U_max]])
                with _np.errstate(invalid="ignore", divide="ignore"):
                    img = _np.zeros_like(sum_img, dtype=_np.float32)
                    nz = cnt_img > 0
                    img[nz] = (sum_img[nz] / cnt_img[nz]).astype(_np.float32)
                    img[~nz] = 0.0

            valid_pixels = img[_np.isfinite(img)]
            if valid_pixels.size > 0:
                actual_min = float(_np.nanmin(valid_pixels))
                actual_max = float(_np.nanmax(valid_pixels))
            else:
                actual_min = 0.0
                actual_max = 0.0
        else:
            img = _np.zeros((H, W), dtype=_np.float32)
            actual_min = 0.0
            actual_max = 0.0
            try:
                # cache for da.line_cut when called without vol
                self._last_image = img
                self._last_extent = [U_min, U_max, V_min, V_max]
                self._last_orientation = orientation
            except Exception:
                pass

            extent = [U_min, U_max, V_min, V_max]
            if axes is None:
                fig, ax = plt.subplots(figsize=(6, 5))
            else:
                ax = axes
                fig = ax.figure
            # Resolve display limits: use actual rasterized data range
            vmin = float(min_intensity) if (min_intensity is not None) else actual_min
            vmax = float(max_intensity) if (max_intensity is not None) else actual_max
            
            # Override with clim if provided
            if clim:
                if clim[0] is not None:
                    vmin = clim[0]
                if clim[1] is not None:
                    vmax = clim[1]
            
            # Check for slice-stored clim as fallback
            if (vmin is None or vmax is None):
                try:
                    clim_fd = getattr(slice_mesh, 'field_data', {}).get('slice_intensity_clim', None)
                    if clim_fd is not None:
                        c = _np.asarray(clim_fd, dtype=float).reshape(-1)
                        if c.size >= 2 and _np.isfinite(c[0]) and _np.isfinite(c[1]):
                            if vmin is None:
                                vmin = float(c[0])
                            if vmax is None:
                                vmax = float(c[1])
                except Exception:
                    pass

            
                im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap,
                           vmin=vmin,
                           vmax=vmax,
                           aspect='auto')
            # Apply rectangular view when requested without reshaping data
            try:
                if (not shape_data) and isinstance(shape, (tuple, list)) and len(shape) == 2:
                    _H, _W = int(shape[0]), int(shape[1])
                    _ratio = float(_H) / float(_W) if (_W != 0) else 1.0
                    try:
                        ax.set_box_aspect(_ratio)
                    except Exception:
                        try:
                            # Fallback: adjust data aspect to approximate box aspect
                            ax.set_aspect(((extent[3] - extent[2]) / (extent[1] - extent[0])) * _ratio, adjustable='box')
                        except Exception:
                            pass
            except Exception:
                pass
            # Origin overlay textbox
            origin_text = f"Origin (center): H={origin[0]:.3f}, K={origin[1]:.3f}, L={origin[2]:.3f}"
            try:
                ax.text(
                    0.0,
                    -0.10,
                    origin_text,
                    transform=ax.transAxes,
                    ha='left',
                    va='top',
                    fontsize=9,
                    color='black',
                    clip_on=False
                )
            except Exception:
                pass
            
            # ----- Interactive hover label using annotate + mouse events
            fig = ax.figure
            # Interactive checkbox to toggle origin visibility
            ann = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="white", alpha=0.85)
            )
            ann.set_visible(False)

            def _label_text(x_coord, y_coord, intensity_val):
                # Hover label: show UV only when axis_display == 'uv'
                # Otherwise show H,K,L (all three), with orth axis from origin for canonical planes
                try:
                    if axis_display == 'uv':
                        return f"U={x_coord:.3f}, V={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                    if orientation == "HK":
                        return f"H={x_coord:.3f}, K={y_coord:.3f}, L={float(origin[2]):.3f}\nIntensity={float(intensity_val):.1f}"
                    if orientation == "KL":
                        return f"H={float(origin[0]):.3f}, K={x_coord:.3f}, L={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                    if orientation == "HL":
                        return f"H={x_coord:.3f}, K={float(origin[1]):.3f}, L={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                    # Custom orientation: project back to HKL; fix orth axis (closest to normal) to origin
                    if 'u' in locals() and 'v' in locals() and isinstance(u, _np.ndarray) and isinstance(v, _np.ndarray):
                        hkl = origin + (x_coord - origin.dot(u)) * u / u.dot(u) + (y_coord - origin.dot(v)) * v / v.dot(v)
                        return f"H={float(hkl[0]):.3f}, K={float(hkl[1]):.3f}, L={float(hkl[2]):.3f}\nIntensity={float(intensity_val):.1f}"
                except Exception:
                    pass
                # Fallback
                return f"U={x_coord:.3f}, V={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"

            def _on_move(event):
                if event.inaxes is not ax:
                    return
                x = event.xdata
                y = event.ydata
                if x is None or y is None:
                    return
                try:
                    col = int((x - extent[0]) / (extent[1] - extent[0]) * img.shape[1])
                    row = int((y - extent[2]) / (extent[3] - extent[2]) * img.shape[0])
                    col = max(0, min(img.shape[1] - 1, col))
                    row = max(0, min(img.shape[0] - 1, row))
                    intensity = img[row, col]
                except Exception:
                    return
                ann.xy = (x, y)
                ann.set_text(_label_text(x, y, intensity))
                if not ann.get_visible():
                    ann.set_visible(True)
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            def _on_leave(event):
                if ann.get_visible():
                    ann.set_visible(False)
                    try:
                        fig.canvas.draw_idle()
                    except Exception:
                        pass

            try:
                fig.canvas.mpl_connect("motion_notify_event", _on_move)
                fig.canvas.mpl_connect("axes_leave_event", _on_leave)
            except Exception:
                pass
            
            # Set axis labels based on axis_display parameter
            if axis_display == 'uv':
                # Simple U/V labels
                ax.set_xlabel('U')
                ax.set_ylabel('V')
            else:
                # HKL formatting (default)
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

            # Title: orth axis for canonical planes; plane normal for Custom
            title = None
            if orientation in ("HK", "KL", "HL") and (orth_label is not None) and (orth_value is not None) and _np.isfinite(orth_value):
                title = f'{orientation} plane ({orth_label} = {orth_value:.3f})'
            elif orientation == 'Custom':
                title = f'Custom slice (normal {format_hkl_axis(normal / normal[_np.argmax(_np.abs(normal))])})'
            else:
                title = f'{orientation} slice'
            ax.set_title(title)

            cbar = ax.figure.colorbar(im, ax=ax, label='Intensity')
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{float(vmin):.3g} min', f'{float(vmax):.3g} max'])
            
            # Add grid if requested
            if show_grid:
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Use Matplotlib status bar readout via format_coord (built-in hover)
            def _format_coord(x_coord, y_coord):
                try:
                    col = int((x_coord - extent[0]) / (extent[1] - extent[0]) * img.shape[1])
                    row = int((y_coord - extent[2]) / (extent[3] - extent[2]) * img.shape[0])
                    col = max(0, min(img.shape[1] - 1, col))
                    row = max(0, min(img.shape[0] - 1, row))
                    intensity = img[row, col]
                except Exception:
                    return ""
                # Status bar readout mirrors hover: UV for axis_display == 'uv'; else show H,K,L
                try:
                    if axis_display == 'uv':
                        return f"U: {x_coord:.3f}, V: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                    if orientation == "HK":
                        return f"H: {x_coord:.3f}, K: {y_coord:.3f}, L: {float(origin[2]):.3f}  Intensity: {float(intensity):.1f}"
                    if orientation == "KL":
                        return f"H: {float(origin[0]):.3f}, K: {x_coord:.3f}, L: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                    if orientation == "HL":
                        return f"H: {x_coord:.3f}, K: {float(origin[1]):.3f}, L: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                    if 'u' in locals() and 'v' in locals() and isinstance(u, _np.ndarray) and isinstance(v, _np.ndarray):
                        hkl = origin + (x_coord - origin.dot(u)) * u / u.dot(u) + (y_coord - origin.dot(v)) * v / v.dot(v)
                        return f"H: {float(hkl[0]):.3f}, K: {float(hkl[1]):.3f}, L: {float(hkl[2]):.3f}  Intensity: {float(intensity):.1f}"
                except Exception:
                    pass
                return f"U: {x_coord:.3f}, V: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
            ax.format_coord = _format_coord

            if axes is None:  # Only show if standalone
                plt.show()

            if return_image:
                return img, extent
            return None
        
        
        try:
            self._last_image = img
            self._last_extent = [U_min, U_max, V_min, V_max]
            self._last_orientation = orientation
        except Exception:
            pass

        # Display via matplotlib imshow with physical axis labels
        extent = [U_min, U_max, V_min, V_max]
        if axis_display == 'pixel':
            extent = [0, img.shape[1], 0, img.shape[0]]

        def _to_uv(x_coord, y_coord):
            if axis_display != 'pixel':
                return x_coord, y_coord
            return (U_min + x_coord / img.shape[1] * (U_max - U_min),
                    V_min + y_coord / img.shape[0] * (V_max - V_min))

        def _on_plane(hkl):
            w = _np.cross(u, v)
            return hkl + w * float((origin - hkl) @ normal) / float(w @ normal)
        if axes is None:
            fig, ax = plt.subplots(figsize=(9, 5) if show_info else (7, 5))
            if show_info:
                fig.subplots_adjust(right=0.6)
        else:
            ax = axes
            fig = ax.figure

        # Determine display limits
        vmin = float(min_intensity) if (min_intensity is not None) else actual_min
        vmax = float(max_intensity) if (max_intensity is not None) else actual_max
        if clim:
            if clim[0] is not None:
                vmin = clim[0]
            if clim[1] is not None:
                vmax = clim[1]
        if (vmin is None or vmax is None):
            try:
                clim_fd = getattr(slice_mesh, 'field_data', {}).get('slice_intensity_clim', None)
                if clim_fd is not None:
                    c = _np.asarray(clim_fd, dtype=float).reshape(-1)
                    if c.size >= 2 and _np.isfinite(c[0]) and _np.isfinite(c[1]):
                        if vmin is None:
                            vmin = float(c[0])
                        if vmax is None:
                            vmax = float(c[1])
            except Exception:
                pass

        im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        # Apply rectangular view when requested without reshaping data
        try:
            if (not shape_data) and isinstance(shape, (tuple, list)) and len(shape) == 2:
                _H, _W = int(shape[0]), int(shape[1])
                _ratio = float(_H) / float(_W) if (_W != 0) else 1.0
                try:
                    ax.set_box_aspect(_ratio)
                except Exception:
                    try:
                        ax.set_aspect(((extent[3] - extent[2]) / (extent[1] - extent[0])) * _ratio, adjustable='box')
                    except Exception:
                        pass
        except Exception:
            pass
        # Origin overlay textbox
        origin_text = f"Origin (center): H={origin[0]:.3f}, K={origin[1]:.3f}, L={origin[2]:.3f}"
        try:
            ax.text(
                0.0,
                -0.15,
                origin_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                color='black',
                clip_on=False
            )
        except Exception:
            pass
        # ----- START Interactive hover label using annotate + mouse events START ----- #
        fig = ax.figure
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="white", alpha=0.85)
        )
        ann.set_visible(False)

        def _label_text(x_coord, y_coord, intensity_val):
            # Hover: UV only when axis_display == 'uv'; else show H,K,L triple
            x_coord, y_coord = _to_uv(x_coord, y_coord)
            try:
                if axis_display == 'uv':
                    return f"U={x_coord:.3f}, V={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                if orientation == "HK":
                    return f"H={x_coord:.3f}, K={y_coord:.3f}, L={float(origin[2]):.3f}\nIntensity={float(intensity_val):.1f}"
                if orientation == "KL":
                    return f"H={float(origin[0]):.3f}, K={x_coord:.3f}, L={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                if orientation == "HL":
                    return f"H={x_coord:.3f}, K={float(origin[1]):.3f}, L={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"
                if 'u' in locals() and 'v' in locals() and isinstance(u, _np.ndarray) and isinstance(v, _np.ndarray):
                    hkl = _on_plane(origin + (x_coord - origin.dot(u)) * u / u.dot(u) + (y_coord - origin.dot(v)) * v / v.dot(v))
                    return f"H={float(hkl[0]):.3f}, K={float(hkl[1]):.3f}, L={float(hkl[2]):.3f}\nIntensity={float(intensity_val):.1f}"
            except Exception:
                pass
            return f"U={x_coord:.3f}, V={y_coord:.3f}\nIntensity={float(intensity_val):.1f}"

        def _on_move(event):
            if event.inaxes is not ax:
                return
            x = event.xdata
            y = event.ydata
            if x is None or y is None:
                return
            try:
                col = int((x - extent[0]) / (extent[1] - extent[0]) * img.shape[1])
                row = int((y - extent[2]) / (extent[3] - extent[2]) * img.shape[0])
                col = max(0, min(img.shape[1] - 1, col))
                row = max(0, min(img.shape[0] - 1, row))
                intensity = img[row, col]
            except Exception:
                return
            ann.xy = (x, y)
            ann.set_text(_label_text(x, y, intensity))
            if not ann.get_visible():
                ann.set_visible(True)
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

        def _on_leave(event):
            if ann.get_visible():
                ann.set_visible(False)
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

        try:
            fig.canvas.mpl_connect("motion_notify_event", _on_move)
            fig.canvas.mpl_connect("axes_leave_event", _on_leave)
        except Exception:
            pass
        # ----- END Interactive hover label using annotate + mouse events END ----- #

        # --- START Set axis labels based on axis_display parameter ---- START #
        if axis_display == 'uv':
            # Simple U/V labels
            ax.set_xlabel('U')
            ax.set_ylabel('V')
        elif axis_display == 'pixel':
            ax.set_xlabel('column (px)')
            ax.set_ylabel('row (px)')
        else:
            # HKL formatting (default)
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
        # --- END Set axis labels based on axis_display parameter ---- END #

        # Title: orth axis for canonical planes; plane normal for Custom
        pad = {}
        if 'slice_title' in slice_mesh.field_data:
            ax.set_title(str(slice_mesh.field_data['slice_title']), **pad)
        elif orientation in ("HK", "KL", "HL") and (orth_label is not None) and (orth_value is not None) and _np.isfinite(orth_value):
            ax.set_title(f'{orientation} plane ({orth_label} = {orth_value:.3f})', **pad)
        elif orientation == 'Custom':
            ax.set_title(f'Custom slice (normal {format_hkl_axis(normal / normal[_np.argmax(_np.abs(normal))])})', **pad)
        else:
            ax.set_title(f'{orientation} slice', **pad)
        cbar = ax.figure.colorbar(im, ax=ax, label='Intensity')
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f'{float(vmin):.3g} min', f'{float(vmax):.3g} max'])
        if show_info:
            try:
                corners = self.slice_info(slice_mesh)['corners']
                fmt = lambda c: '(' + ', '.join(f'{x:.3f}' for x in c) + ')'  # noqa: E731
                lines = [f"{name:<13}{fmt(corners[key])}" for name, key in
                         (('top-left', 'top_left'), ('top-right', 'top_right'),
                          ('bottom-left', 'bottom_left'), ('bottom-right', 'bottom_right'))]
                ax.annotate('Corners (H, K, L)\n' + '\n'.join(lines), xy=(1, 1), xycoords=cbar.ax,
                            xytext=(50, 0), textcoords='offset points', ha='left', va='top',
                            fontsize=8, family='monospace', annotation_clip=False)
            except Exception:
                pass

        # Add grid if requested
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Use Matplotlib status bar readout via format_coord (built-in hover)
        def _format_coord(x_coord, y_coord):
            try:
                col = int((x_coord - extent[0]) / (extent[1] - extent[0]) * img.shape[1])
                row = int((y_coord - extent[2]) / (extent[3] - extent[2]) * img.shape[0])
                col = max(0, min(img.shape[1] - 1, col))
                row = max(0, min(img.shape[0] - 1, row))
                intensity = img[row, col]
            except Exception:
                return ""
            x_coord, y_coord = _to_uv(x_coord, y_coord)
            try:
                if axis_display == 'uv':
                    return f"U: {x_coord:.3f}, V: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                if orientation == "HK":
                    return f"H: {x_coord:.3f}, K: {y_coord:.3f}, L: {float(origin[2]):.3f}  Intensity: {float(intensity):.1f}"
                if orientation == "KL":
                    return f"H: {float(origin[0]):.3f}, K: {x_coord:.3f}, L: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                if orientation == "HL":
                    return f"H: {x_coord:.3f}, K: {float(origin[1]):.3f}, L: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
                if 'u' in locals() and 'v' in locals() and isinstance(u, _np.ndarray) and isinstance(v, _np.ndarray):
                    hkl = _on_plane(origin + (x_coord - origin.dot(u)) * u / u.dot(u) + (y_coord - origin.dot(v)) * v / v.dot(v))
                    return f"H: {float(hkl[0]):.3f}, K: {float(hkl[1]):.3f}, L: {float(hkl[2]):.3f}  Intensity: {float(intensity):.1f}"
            except Exception:
                pass
            return f"U: {x_coord:.3f}, V: {y_coord:.3f}  Intensity: {float(intensity):.1f}"
        ax.format_coord = _format_coord

        if axes is None:
            plt.show()

        if return_image:
            return img, extent
        return None

    def slice_info(self, slice_mesh=None):
        """
        Readable summary of a slice: where it sits, its orientation and the H/K/L of its corners.

        Corners follow the 2D plot (origin lower-left). Values are rounded to
        ANALYSIS_SLICE_INFO_DECIMALS; 'hkl' and 'normal' can go straight back into
        slice_data(orientation=...).

        Usage:
            sl = da.slice_data(data, L=1.5, H=(0, 2), K=(-1, 1), show=False)
            info = da.slice_info(sl)          # or da.slice_info() for da.last_slice
            info['corners']['top_right']      # (2.0, 1.0, 1.5)

        Returns:
            dict: {'hkl', 'normal', 'shape', 'corners': {'bottom_left', 'bottom_right', 'top_left', 'top_right'}}
        """
        sl = self.last_slice if slice_mesh is None else slice_mesh
        fd = sl.field_data
        origin = np.asarray(fd['slice_origin'], dtype=float)
        normal = np.asarray(fd['slice_normal'], dtype=float)
        pts = np.asarray(sl.points, dtype=float)
        shape = tuple(int(x) for x in fd['slice_shape']) if 'slice_shape' in fd else None
        if shape is not None and shape[0] * shape[1] == len(pts):
            grid = pts.reshape(shape[0], shape[1], 3)
            corners = grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]
        else:
            k = int(np.argmax(np.abs(normal)))
            u, v = np.eye(3)[[(1, 0, 0)[k], (2, 2, 1)[k]]]
            if 'slice_u_axis' in fd:
                u, v = np.asarray(fd['slice_u_axis'], dtype=float), np.asarray(fd['slice_v_axis'], dtype=float)
            w = np.cross(u, v)
            U, V = pts @ u, pts @ v

            def back(a, b):
                p = origin + (a - origin @ u) * u / (u @ u) + (b - origin @ v) * v / (v @ v)
                return p + w * float((origin - p) @ normal) / float(w @ normal)

            corners = back(U.min(), V.min()), back(U.max(), V.min()), back(U.min(), V.max()), back(U.max(), V.max())
        r = lambda x: tuple(round(float(c), app_settings.ANALYSIS_SLICE_INFO_DECIMALS) + 0.0 for c in x)  # noqa: E731
        return {'hkl': r(origin), 'normal': r(normal), 'shape': shape,
                'corners': dict(zip(('bottom_left', 'bottom_right', 'top_left', 'top_right'), map(r, corners)))}

    def _slice_hkl_ranges(self, data, spec, shape, slab_thickness, intensity_range, show, **kwargs):
        """Build a slice from H/K/L keywords (see slice_data): one fixed or tilted axis, two ranges."""
        names = 'HKL'
        fixed = [i for i, r in enumerate(spec) if r is not None and (np.isscalar(r) or len(r) == 3)]
        if len(fixed) != 1:
            raise ValueError("Give exactly one of H/K/L as a number or (start, stop, along); "
                             "the other two as (min, max) ranges.")
        c = fixed[0]
        a, b = [i for i in range(3) if i != c]
        if isinstance(data, pv.ImageData):
            bounds = np.reshape(data.bounds, (3, 2))
        else:
            pts, ints = (data[0], data[1]) if isinstance(data, (tuple, list)) else \
                (data['points'], data['intensities']) if isinstance(data, dict) else (data.points, data.intensities)
            pts, ints = np.asarray(pts, dtype=float), np.asarray(ints, dtype=float).reshape(-1)
            bounds = np.column_stack([pts.min(axis=0), pts.max(axis=0)])
        ranges = [tuple(map(float, spec[i])) if i != c and spec[i] is not None else tuple(bounds[i]) for i in range(3)]
        if np.isscalar(spec[c]):
            c0 = c1 = float(spec[c])
            along = a
        else:
            c0, c1, along = float(spec[c][0]), float(spec[c][1]), names.index(str(spec[c][2]).upper())
            if along == c:
                raise ValueError(f"{names[c]} cannot tilt along itself; use {names[a]} or {names[b]}.")

        rows, cols = int(shape[0]), int(shape[1])
        A, B = np.meshgrid(np.linspace(*ranges[a], cols), np.linspace(*ranges[b], rows))
        t0, t1 = ranges[along]
        slope = (c1 - c0) / (t1 - t0)
        C = c0 + ((A if along == a else B) - t0) * slope
        grid_pts = np.empty((rows * cols, 3))
        grid_pts[:, a], grid_pts[:, b], grid_pts[:, c] = A.ravel(), B.ravel(), C.ravel()
        normal = np.eye(3)[c] - slope * np.eye(3)[along]
        normal /= np.linalg.norm(normal)
        origin = grid_pts[(rows // 2) * cols + cols // 2]

        plane = pv.Plane(i_resolution=cols - 1, j_resolution=rows - 1)
        plane.points = grid_pts
        if isinstance(data, pv.ImageData):
            sl = plane.sample(data)
        else:
            keep = np.ones(len(pts), dtype=bool)
            if intensity_range is not None:
                lo, hi = intensity_range
                keep &= (ints >= (-np.inf if lo is None else lo)) & (ints <= (np.inf if hi is None else hi))
            if slab_thickness:
                keep &= np.abs((pts - origin) @ normal) <= float(slab_thickness)
            cloud = pv.PolyData(pts[keep])
            cloud['intensity'] = ints[keep].astype('float32')
            radius = self._calculate_smart_radius(pts[keep], ranges[a], ranges[b], (rows, cols))
            sl = plane.interpolate(cloud, radius=radius, sharpness=1.5, null_value=0.0)
        sl.field_data['slice_origin'] = origin
        sl.field_data['slice_normal'] = normal
        sl.field_data['slice_u_axis'] = np.eye(3)[a]
        sl.field_data['slice_v_axis'] = np.eye(3)[b]
        sl.field_data['slice_u_label'] = names[a]
        sl.field_data['slice_v_label'] = names[b]
        sl.field_data['slice_shape'] = np.asarray([rows, cols], dtype=int)
        sl.field_data['slice_title'] = (f'{names[c]} = {c0:.3f}' if c0 == c1 else
                                        f'{names[c]} = {c0:.3f} to {c1:.3f} along {names[along]}')
        if show:
            self.show_slice(sl, shape=shape, **kwargs)
        return sl

    def show_point_cloud(self, data, intensities=None, *, notebook=True,
                         point_size=1.0, cmap='viridis', opacity=1.0,
                         render_points_as_spheres=False, axes_labels=('H','K','L'),
                         clim=None, show_bounds=True, opacity_range=None, slice_plane=None, slice_shape=(256, 256),
                         slice_snap=None,
                         camera='iso', zoom=1.0, controls=True, show_scalar_bar=True):
        """
        Render a point cloud in HKL space with advanced visualization options.

        This method provides comprehensive 3D point data visualization with support for
        multiple data formats, opacity control, intensity filtering, and interactive features.
        Supports both notebook and standalone rendering with customizable appearance.

        Usage:
            # Basic point cloud rendering
            da.show_point_cloud(data.points, data.intensities)
            
            # Advanced rendering with opacity control
            da.show_point_cloud(data, clim=(100, 1000), opacity_range=(0.1, 1.0))
            
            # High-quality spherical rendering
            da.show_point_cloud(data, render_points_as_spheres=False, point_size=5.0)

        Parameters:
            data: Point cloud data. Supported formats:
                - (points, intensities) tuple/list
                - Data object with .points and .intensities attributes
                - Dict with 'points' and 'intensities' keys
                - pv.PolyData with optional 'intensity' array
                - np.ndarray of shape (N,3) for points (provide intensities separately)
            intensities (array-like | None): Optional 1D intensity array when data contains only points.
            notebook (bool): Use a notebook plotter when True or a desktop plotter when False.
            point_size (float): Rendered point diameter.
            cmap (str): Colormap used for intensity values.
            opacity (float): Uniform point opacity from 0 to 1.
            render_points_as_spheres (bool): Render spherical glyphs instead of flat points.
            axes_labels (tuple): Labels for the three plotted axes; defaults to H, K, and L.
            clim (tuple | None): Intensity display limits as (minimum, maximum).
                Points outside it are hidden.
            show_bounds (bool): Show coordinate bounds and labels.
            opacity_range (tuple | None): Opacity range applied linearly across clim.
            slice_plane: True, an orientation dict ({'hkl': ..., 'normal': ...}), or a mesh from
                slice_data to start from, adds a draggable slice plane with an on/off checkbox,
                front/back, tilt and reset toolbar buttons and a right-side H/K/L information panel.
                The latest slice is kept in da.last_slice and its position in
                da.slice_orientation, which slice_data(..., orientation=...) accepts.
            slice_shape (tuple): Raster resolution of the interactive slice.
            slice_snap (float | None): Grid step in H/K/L; the slice position snaps to multiples of it
                and back/front move one step. The toolbar Snap button toggles it.
            camera: Starting view: 'iso', 'hk', 'hl', or 'kl'; an (azimuth, elevation)
                pair in degrees from the HL view (L up); or a PyVista camera_position.
            zoom (float): Starting zoom factor; values greater than 1 zoom in.
            controls (bool): Add HK / HL / KL / ISO and zoom +/- buttons to the viewer
                toolbar (renders with the trame backend). False keeps the notebook's backend.
            show_scalar_bar (bool): Show the intensity color bar beside the 3D plot.
            Note: In notebook mode, rendering caps at 5,000,000 points; if more are provided,
            the top 5,000,000 intensities are used to maintain interactive performance.
            clim is applied after that cap.

        Returns:
            PyVista rendering result (displays inline in notebooks)

        Raises:
            ImportError: If PyVista is not available
            TypeError: If cloud format is not supported
        Examples:
            # Basic visualization
            da.show_point_cloud(point_data, intensity_data)
            
            # High-contrast visualization with filtering
            da.show_point_cloud(data, clim=(50, 500))
            
            # Opacity-based intensity mapping
            da.show_point_cloud(data, opacity_range=(0.2, 1.0), cmap='plasma')

            # Drag/rotate a slice through the cloud; checkbox toggles it
            da.show_point_cloud(data, clim=(100, 50000), slice_plane=True)

            # Start looking down L, zoomed in
            da.show_point_cloud(data, camera='hk', zoom=1.5)
        """
        # Normalize inputs to pv.PolyData + 'intensity' if available
        pts = None
        ints = None
        poly = None    
        # Convert the data to polyData
        if isinstance(data, pv.PolyData):
            poly = data
            if ('intensity' not in poly.array_names) and (intensities is not None):
                poly['intensity'] = np.asarray(intensities, dtype=np.float32)
        elif hasattr(data, 'points') and hasattr(data, 'intensities'):
            pts = np.asarray(data.points, dtype=float)
            ints = np.asarray(data.intensities, dtype=float)
            poly = pv.PolyData(pts)
            poly['intensity'] = ints.astype(np.float32)
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            pts = np.asarray(data[0], dtype=float)
            ints = np.asarray(data[1], dtype=float)
            poly = pv.PolyData(pts)
            poly['intensity'] = ints.astype(np.float32)
        elif isinstance(data, dict) and ('points' in data):
            pts = np.asarray(data['points'], dtype=float)
            ints = np.asarray(data.get('intensities', intensities), dtype=float) if ('intensities' in data or intensities is not None) else None
            poly = pv.PolyData(pts)
            if ints is not None and ints.shape[0] == pts.shape[0]:
                poly['intensity'] = ints.astype(np.float32)
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
            pts = np.asarray(data, dtype=float)
            poly = pv.PolyData(pts)
            if intensities is not None:
                poly['intensity'] = np.asarray(intensities, dtype=np.float32)
        else:
            raise TypeError("Unsupported data format. Provide (points, intensities), Data, dict, pv.PolyData, or Nx3 ndarray.")
        print("Converted to poly data")

        # Cap points to top 5,000,000 intensities for performance in notebook mode
        MAX_POINTS = 5_000_000
        try:
            if bool(notebook) and poly is not None:
                N = int(poly.n_points)
                if N > MAX_POINTS:
                    if 'intensity' in poly.array_names:
                        ints_arr = np.asarray(poly['intensity'], dtype=np.float32).reshape(-1)
                        if ints_arr.shape[0] == N:
                            k = MAX_POINTS
                            idx = np.argpartition(ints_arr, N - k)[N - k:]
                            # Deterministic ordering by descending intensity
                            idx = idx[np.argsort(ints_arr[idx])[::-1]]
                            pts_arr = np.asarray(poly.points, dtype=float)
                            poly = pv.PolyData(pts_arr[idx])
                            poly['intensity'] = ints_arr[idx].astype(np.float32)
                            try:
                                print(f"Notebook mode: capped point cloud from {N} to {int(poly.n_points)} by top intensities.")
                            except Exception:
                                pass
                        else:
                            # Fallback: intensity length mismatch; random subsample
                            rng = np.random.default_rng()
                            idx = rng.choice(N, size=MAX_POINTS, replace=False)
                            poly = pv.PolyData(np.asarray(poly.points, dtype=float)[idx])
                            try:
                                print(f"Notebook mode: capped point cloud from {N} to {int(poly.n_points)} (random fallback due to intensity mismatch).")
                            except Exception:
                                pass
                    else:
                        # No intensities; random subsample to respect cap
                        rng = np.random.default_rng()
                        idx = rng.choice(N, size=MAX_POINTS, replace=False)
                        poly = pv.PolyData(np.asarray(poly.points, dtype=float)[idx])
                        try:
                            print(f"Notebook mode: capped point cloud from {N} to {int(poly.n_points)} (random fallback, no intensities).")
                        except Exception:
                            pass
        except Exception as _cap_err:
            # Be permissive; if capping fails, continue with original data
            try:
                print(f"Notebook mode capping failed: {_cap_err}")
            except Exception:
                pass

        # Create plotter
        p = pv.Plotter(notebook=bool(notebook))
        p.add_axes(xlabel=str(axes_labels[0]), ylabel=str(axes_labels[1]), zlabel=str(axes_labels[2]))
        # Simple LUT configuration; use given clim for color scaling (if provided)
        lut = pv.LookupTable(cmap=cmap)
        lut.above_range_color = 'white'
        lut.below_range_color = 'white'
        lut.above_range_opacity= 0
        lut.below_range_opacity= 0
        ints_arr = np.asarray(poly['intensity'])
        lut.scalar_range = clim if clim is not None else (float(ints_arr.min()), float(ints_arr.max()))
        if opacity_range is not None:
            lut.apply_opacity(opacity_range, kind='linear')

        p.add_mesh(poly,
                   scalars='intensity',
                    cmap=lut,
                    render_points_as_spheres=bool(render_points_as_spheres),
                    point_size=float(point_size),
                    opacity=float(opacity) if opacity_range is None else 1.0,
                    name='points',
                    show_scalar_bar=bool(show_scalar_bar))

        # Optional bounds
        bounds_kwargs = None
        if bool(show_bounds):
            bounds_kwargs = dict(mesh=poly, xtitle=str(axes_labels[0]), ytitle=str(axes_labels[1]),
                                 ztitle=str(axes_labels[2]), bounds=poly.bounds, grid='back', location='outer',
                                 ticks='both', fmt=app_settings.ANALYSIS_AXIS_NUMBER_FORMAT)
            try:
                p.show_bounds(**bounds_kwargs)
            except Exception:
                bounds_kwargs = None

        slice_buttons = None
        if slice_plane is not None:
            slice_buttons = self._add_slice_plane(p, (np.asarray(poly.points), ints_arr), slice_plane, poly.bounds,
                                                  cmap, lut.scalar_range, slice_shape, slice_snap)
            controls = True

        return _show_with_camera(p, camera, zoom, controls, slice_buttons, bounds_kwargs)

    def show_vol(self, vol, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), cmap='jet', slice_plane=None,
                 camera='iso', zoom=1.0, controls=True, slice_snap=None):
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
            spacing (tuple): Voxel spacing (ΔH, ΔK, ΔL) for NumPy arrays.
            origin (tuple): Grid origin (H0, K0, L0) for NumPy arrays.
            cmap (str): Colormap used for volume intensities.
            slice_plane: True, an orientation dict ({'hkl': ..., 'normal': ...}), or a mesh from
                slice_data to start from, adds a draggable slice plane with an on/off checkbox,
                front/back, tilt and reset toolbar buttons and a right-side H/K/L information panel.
                The latest slice is kept in da.last_slice and its position in
                da.slice_orientation, which slice_data(..., orientation=...) accepts.
            slice_snap (float | None): Grid step in H/K/L; the slice position snaps to multiples of it
                and back/front move one step. The toolbar Snap button toggles it.
            camera: Starting view: 'iso' (default), 'hk', 'hl', 'kl'; an (azimuth, elevation)
                pair in degrees from the HL view (L up); or a PyVista camera_position.
            zoom (float): Starting zoom factor; values greater than 1 zoom in.
            controls (bool): Add HK / HL / KL / ISO and zoom +/- buttons to the viewer
                toolbar (renders with the trame backend). False keeps the notebook's backend.

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

            # Drag/rotate a slice through the volume; checkbox toggles it
            da.show_vol(volume_data, slice_plane=True)
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
        bounds_kwargs = dict(mesh=grid, xtitle='H Axis', ytitle='K Axis', ztitle='L Axis', bounds=grid.bounds,
                             grid='back', location='outer', ticks='both',
                             fmt=app_settings.ANALYSIS_AXIS_NUMBER_FORMAT)
        try:
            plotter.show_bounds(**bounds_kwargs)
        except Exception:
            bounds_kwargs = None

        slice_buttons = None
        if slice_plane is not None:
            slice_buttons = self._add_slice_plane(plotter, grid, slice_plane, grid.bounds, cmap, clim, None, slice_snap)
            controls = True

        return _show_with_camera(plotter, camera, zoom, controls, slice_buttons, bounds_kwargs)

    def _add_slice_plane(self, p, data, slice_plane, bounds, cmap, clim, shape, snap=None):
        """
        Add a draggable slice plane, controls, and a right-side information panel to a 3D plotter.

        slice_plane is True (start at the data center, HL plane), an orientation dict
        {'hkl': ..., 'normal': ...}, or a slice mesh to start from. Every move re-slices into
        self.last_slice and records self.slice_orientation. snap is a grid step the position
        rounds to. Returns rows of (icon, tooltip, callback) buttons that step, tilt, snap,
        align and reset the plane, for _show_with_camera's Slice menu.
        """
        bounds = np.reshape(bounds, (3, 2))
        if slice_plane is True:
            start = (bounds.mean(axis=1), np.array([0.0, 1.0, 0.0]))
        elif isinstance(slice_plane, dict):
            start = (np.asarray(slice_plane['hkl'], dtype=float), np.asarray(slice_plane['normal'], dtype=float))
        else:
            start = (np.asarray(slice_plane.field_data['slice_origin'], dtype=float),
                     np.asarray(slice_plane.field_data['slice_normal'], dtype=float))
        self.slice_orientation = {
            'hkl': tuple(float(value) for value in start[0]),
            'normal': tuple(float(value) for value in start[1]),
        }
        slice_kwargs = {} if shape is None else {'shape': shape}
        state = {'snap': snap, 'widget': None}

        def update_slice(normal, origin):
            step = state['snap']
            if step:
                origin = np.clip(np.round(np.asarray(origin) / step) * step, bounds[:, 0], bounds[:, 1]).round(10)
                if state['widget'] is not None:
                    state['widget'].SetOrigin(*origin)
            self.last_slice = self.slice_data(data=data, hkl=tuple(origin), normal=tuple(normal), show=False,
                                              **slice_kwargs)
            p.add_mesh(self.last_slice, scalars='intensity', cmap=cmap, clim=clim, show_scalar_bar=False, name='slice')
            axes_actor = p.renderer.cube_axes_actor
            if axes_actor is not None:
                fmt = app_settings.ANALYSIS_AXIS_NUMBER_FORMAT
                axes_actor.x_label_format = axes_actor.y_label_format = axes_actor.z_label_format = fmt
            self.slice_orientation = {'hkl': tuple(float(x) for x in origin),
                                      'normal': tuple(float(x) for x in normal)}
            n = np.round(np.asarray(normal) / np.abs(normal).max(), 3) + 0.0
            details = (
                'SLICE\n'
                f'Origin (H, K, L)\n{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}\n\n'
                f'Normal\n{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}'
            )
            if step:
                details += f'\n\nSnap\n{step:g}'
            if 'slice_info' in p.actors:
                p.remove_actor('slice_info', render=False)
            p.add_text(details, position='upper_right', font_size=10, name='slice_info')

        widget = p.add_plane_widget(update_slice, normal=start[1], origin=start[0], bounds=bounds.ravel(),
                                    interaction_event='end')
        state['widget'] = widget
        widget.SetOrigin(*self.slice_orientation['hkl'])

        def toggle_slice(on):
            p.actors['slice'].SetVisibility(on)
            p.actors['slice_info'].SetVisibility(on)
            widget.SetEnabled(on)

        p.add_checkbox_button_widget(toggle_slice, value=True)

        def set_plane(origin, normal):
            widget.SetNormal(*normal)
            widget.SetOrigin(*np.clip(origin, bounds[:, 0], bounds[:, 1]))
            update_slice(np.asarray(widget.GetNormal()), np.asarray(widget.GetOrigin()))

        def move(sign):
            n = np.asarray(widget.GetNormal())
            step = state['snap'] or app_settings.ANALYSIS_SLICE_MOVE_FRACTION * float(np.abs(n) @ np.ptp(bounds, axis=1))
            return lambda: set_plane(np.asarray(widget.GetOrigin()) + sign * step * n, n)

        def tilt(about, sign):
            def run():
                n = np.asarray(widget.GetNormal())
                ref = np.eye(3)[int(np.argmin(np.abs(start[1])))]
                u = np.cross(n, ref)
                u /= np.linalg.norm(u)
                axis = u if about == 0 else np.cross(n, u)
                t = np.radians(sign * app_settings.ANALYSIS_SLICE_TILT_STEP_DEGREES)
                set_plane(np.asarray(widget.GetOrigin()), n * np.cos(t) + np.cross(axis, n) * np.sin(t))
            return run

        def toggle_snap():
            state['snap'] = None if state['snap'] else (snap or app_settings.ANALYSIS_SLICE_SNAP_STEP)
            set_plane(np.asarray(widget.GetOrigin()), np.asarray(widget.GetNormal()))

        def align():
            n = np.asarray(widget.GetNormal())
            k = int(np.argmax(np.abs(n)))
            set_plane(np.asarray(widget.GetOrigin()), np.eye(3)[k] * np.sign(n[k]))

        return [
            [('mdi-chevron-double-down', 'Slice back', lambda: move(-1)()),
             ('mdi-chevron-double-up', 'Slice front', lambda: move(1)())],
            [('mdi-arrow-left-bold', 'Tilt left', tilt(1, -1)),
             ('mdi-arrow-right-bold', 'Tilt right', tilt(1, 1)),
             ('mdi-arrow-up-bold', 'Tilt up', tilt(0, 1)),
             ('mdi-arrow-down-bold', 'Tilt down', tilt(0, -1))],
            [('mdi-grid', 'Snap to grid on/off', toggle_snap),
             ('mdi-axis-arrow', 'Align to nearest H/K/L axis', align),
             ('mdi-restore', 'Reset slice', lambda: set_plane(*start))],
        ]

    def _show_slice_navigator(self, data, plane='HL', start=None, shape=(256, 256), slab_thickness=None,
                              return_slice=False, **kwargs):
        """
        Move a slice through one scan or a stack using a position slider and arrow buttons.

        Runs over plain Jupyter comms, so it needs no trame server or forwarded port
        (works in VS Code over SSH). The slider shows the actual H/K/L position. The Step
        numeric up/down control sets how far the arrows move. Each change immediately redraws the
        existing 2D view. H, K, and L positions are remembered when the plane changes, and Reset
        returns all three to the data center. The latest slice is kept in da.last_slice.
        With return_slice=True it returns a slice mesh that keeps following the slider.

        Usage:
            da.show_slice(data, navigate=True, plane='HK', clim=(100, 5000), cmap='jet')
            da.show_slice(sl, navigate=True)                           # same slice through loaded scans
            sl = da.show_slice(data, navigate=True, return_slice=True)

        Parameters:
            data: Anything slice_data accepts, or a DataStack. A stack adds a scan selector and
                applies the same slice plane to the selected scan.
            plane (str): Starting preset plane, 'HK', 'KL' or 'HL', when start is None; default 'HL'
            start: A slice to move instead of the presets: a slice_data result, an orientation dict
                {'hkl', 'normal'} (e.g. da.slice_orientation), or H/K/L keywords as a dict. The
                slider moves it along its normal; for H/K/L keywords it shifts the fixed value.
            shape (tuple): Raster resolution of each slice
            slab_thickness (float): Optional ± thickness passed to slice_data
            return_slice (bool): Return a pv.PolyData slice that is updated in place to whatever
                the slider currently shows
            **kwargs: Passed to show_slice (e.g. clim, cmap, axis_display, show_grid)

        Returns:
            pv.PolyData or SliceStack if return_slice, else None
        """
        import ipywidgets as widgets
        from IPython.display import display
        from matplotlib.figure import Figure

        stack = data if isinstance(data, DataStack) else None
        if stack is not None and not len(stack):
            raise ValueError("show_slice: the stack is empty")

        def prepare(source):
            if isinstance(source, pv.ImageData):
                return source, np.reshape(source.bounds, (3, 2))
            points = np.asarray(source[0] if isinstance(source, (tuple, list)) else
                                source['points'] if isinstance(source, dict) else source.points, dtype=float)
            intensities = np.asarray(source[1] if isinstance(source, (tuple, list)) else
                                     source['intensities'] if isinstance(source, dict) else source.intensities,
                                     dtype=float).reshape(-1)
            return (points, intensities), np.column_stack([points.min(axis=0), points.max(axis=0)])

        prepared, bounds = prepare(stack[0] if stack is not None else data)

        fixed_axis = {'HK': 2, 'KL': 0, 'HL': 1}
        spec = start if isinstance(start, dict) and set(start) <= set('HKL') else None
        if spec is not None:
            name = next(a for a, v in spec.items() if np.isscalar(v) or len(v) == 3)
        elif isinstance(start, dict):
            base = (np.asarray(start['hkl'], dtype=float), np.asarray(start['normal'], dtype=float))
        elif start is not None:
            base = (np.asarray(start.field_data['slice_origin'], dtype=float),
                    np.asarray(start.field_data['slice_normal'], dtype=float))

        plane_box = widgets.Dropdown(options=list(fixed_axis), value=plane.upper(), description='Plane',
                                     layout=widgets.Layout(width='150px', display='' if start is None else 'none'))
        scan_box = None
        if stack is not None:
            scan_box = widgets.SelectionSlider(
                options=[(name, index) for index, name in enumerate(stack.names)],
                description='Scan', continuous_update=False, layout=widgets.Layout(width='570px'),
            )
        value_box = widgets.FloatSlider(continuous_update=True, readout_format='.3f',
                                        layout=widgets.Layout(width='420px'))
        step_box = widgets.BoundedFloatText(
            value=0.01, min=0.001, max=100.0, step=0.001, description='Step',
            layout=widgets.Layout(width='190px'),
        )
        previous_button = widgets.Button(icon='arrow-left', tooltip='Previous slice',
                                         layout=widgets.Layout(width='38px'))
        next_button = widgets.Button(icon='arrow-right', tooltip='Next slice',
                                     layout=widgets.Layout(width='38px'))
        reset_button = widgets.Button(icon='refresh', description='Reset', tooltip='Reset H, K, and L',
                                      layout=widgets.Layout(width='90px'))
        out = widgets.Output(layout=widgets.Layout(width='720px'))
        info = widgets.HTML(layout=widgets.Layout(
            width='300px', margin='0 0 0 16px', padding='0 0 0 16px', border_left='1px solid #888',
        ))
        backend = plt.get_backend().lower()
        live = 'ipympl' in backend or backend == 'widget'  # ipympl can redraw one figure in place
        state = {
            'fig': None,
            'cids': set(),
            'data': prepared,
            'bounds': bounds,
            'positions': bounds.mean(axis=1).astype(float),
        }
        current = pv.PolyData()

        def plane_basis():
            if start is None:
                k = fixed_axis[plane_box.value]
                return state['positions'].copy(), np.eye(3)[k]
            o, n = base
            return o, n / np.linalg.norm(n)

        def make_slice(value, source=None):
            source = state['data'] if source is None else source
            if spec is not None:
                v = spec[name]
                moved = dict(spec, **{name: value if np.isscalar(v) else (value, value + v[1] - v[0], v[2])})
                return self.slice_data(source, shape=shape, slab_thickness=slab_thickness,
                                       show=False, **moved)
            o, n = plane_basis()
            return self.slice_data(data=source, hkl=tuple(o + (value - o @ n) * n), normal=tuple(n), shape=shape,
                                   slab_thickness=slab_thickness, clamp_to_bounds=False, show=False)

        def redraw(*_):
            if start is None:
                state['positions'][fixed_axis[plane_box.value]] = value_box.value
            self.last_slice = make_slice(value_box.value)
            current.copy_from(self.last_slice)
            slice_info = self.slice_info(self.last_slice)
            origin = ', '.join(f'{value:.3f}' for value in slice_info['hkl'])
            normal = ', '.join(f'{value:.3f}' for value in slice_info['normal'])
            corners = slice_info['corners']
            corner_rows = ''.join(
                f'<div><b>{label}</b><br>{", ".join(f"{value:.3f}" for value in corners[key])}</div>'
                for label, key in (
                    ('Top left', 'top_left'),
                    ('Top right', 'top_right'),
                    ('Bottom left', 'bottom_left'),
                    ('Bottom right', 'bottom_right'),
                )
            )
            info.value = (
                '<div><h4>Slice</h4>'
                f'<div><b>Origin (H, K, L)</b><br>{origin}</div>'
                f'<div><b>Normal</b><br>{normal}</div>'
                f'<h4>Corners (H, K, L)</h4>{corner_rows}</div>'
            )
            # Figures stay out of pyplot so no plt.show() or end-of-cell flush shows them a second time
            if not live:
                fig = Figure(figsize=(7, 5))
                self.show_slice(self.last_slice, axes=fig.add_subplot(), **kwargs)
                with out:
                    out.clear_output(wait=False)
                    display(fig)
                return
            # Reuse one canvas: a new ipympl widget per redraw is slow and stacks up figures
            if state['fig'] is None:
                from ipympl.backend_nbagg import Canvas, FigureManager
                state['fig'] = Figure(figsize=(9, 5))
                FigureManager(Canvas(state['fig']), 0)
                with out:
                    out.clear_output(wait=False)
                    display(state['fig'].canvas)
            fig = state['fig']
            registry = fig.canvas.callbacks.callbacks
            for cid in state['cids']:
                fig.canvas.mpl_disconnect(cid)
            before = {cid for cids in registry.values() for cid in cids}
            fig.clf()
            self.show_slice(self.last_slice, axes=fig.add_subplot(), **kwargs)
            state['cids'] = {cid for cids in registry.values() for cid in cids} - before
            fig.canvas.draw_idle()

        def reset(*_):
            bounds = state['bounds']
            corners = np.array(np.meshgrid(*bounds)).reshape(3, -1).T
            if spec is not None:
                k = 'HKL'.index(name)
                v = spec[name]
                lo, hi, value = bounds[k, 0], bounds[k, 1], v if np.isscalar(v) else v[0]
                label = name if np.isscalar(v) else f'{name} start'
            else:
                o, n = plane_basis()
                proj = corners @ n
                lo, hi = proj.min(), proj.max()
                k = int(np.argmax(np.abs(n)))
                value = float(np.clip(state['positions'][k], lo, hi))
                state['positions'][k] = value
                label = 'HKL'[k] if abs(n[k]) > 0.999 else 'Position'
            value_box.unobserve(redraw, 'value')
            value_box.min, value_box.max = -1e12, 1e12
            value_box.min, value_box.max = float(min(lo, value)), float(max(hi, value))
            value_box.step = max(float((value_box.max - value_box.min) / 100), np.finfo(float).eps)
            value_box.value = float(value)
            value_box.description = label
            value_box.observe(redraw, 'value')
            redraw()

        def move_step(direction):
            value_box.value = float(np.clip(
                value_box.value + direction * step_box.value,
                value_box.min,
                value_box.max,
            ))

        def reset_positions(_=None):
            state['positions'] = state['bounds'].mean(axis=1).astype(float)
            reset()

        def select_scan(change):
            if change.get('name') != 'value':
                return
            state['data'], state['bounds'] = prepare(stack[change['new']])
            reset()

        plane_box.observe(reset, 'value')
        previous_button.on_click(lambda _: move_step(-1))
        next_button.on_click(lambda _: move_step(1))
        reset_button.on_click(reset_positions)
        if stack is not None:
            scan_box.observe(select_scan, 'value')
        controls = [scan_box] if stack is not None else []
        controls.append(widgets.HBox([
            plane_box, previous_button, value_box, next_button, step_box, reset_button,
        ]))
        content = widgets.HBox(
            [out, info],
            layout=widgets.Layout(align_items='flex-start'),
        )
        display(widgets.VBox([*controls, content]))
        reset()
        if return_slice:
            if stack is not None:
                return SliceStack(
                    stack.names,
                    lambda index: make_slice(value_box.value, prepare(stack[index])[0]),
                )
            return current

    def bin_stack(self, stack=None, bins=200, voxel=None, hkl_range=None, mode='mean', intensity_range=None,
                  output='volume'):
        """
        Merge several scans into one HKL grid by binning every point into a voxel.

        Each scan is loaded once and reduced to its occupied voxels, so the merged grid never
        needs all the scans in memory at the same time. Voxels hit by more than one scan are
        averaged ('mean'), added ('sum') or counted ('count').

        Usage:
            vol = da.bin_stack(stack[2:8], bins=200)
            da.show_vol(vol)
            da.slice_data(vol, hkl='HK')
            da.show_slice(vol, navigate=True, plane='HK')
            merged = da.bin_stack(stack, output='points')     # a Data for show_point_cloud

        Parameters:
            stack: A DataStack, SliceStack, or list of Data objects. None uses the scans
                remembered by load_data.
            bins (int | tuple): Voxels per H, K, L axis across hkl_range, or across the first
                scan's range when hkl_range is None. Ignored when voxel is given.
            voxel (float | tuple): Voxel size in H, K, L
            hkl_range: ((Hmin, Hmax), (Kmin, Kmax), (Lmin, Lmax)) to keep; default everything
            mode (str): 'mean' (default), 'sum' or 'count'
            intensity_range (tuple): Optional (min, max) to drop points before binning
            output (str): 'volume' (pv.ImageData with cell_data 'intensity' and 'count') or
                'points' (a Data of the occupied voxel centres)

        Returns:
            pv.ImageData or Data
        """
        stack = self._loaded_stack if stack is None else stack
        if stack is None:
            raise ValueError("bin_stack: call load_data with a file or folder first")
        if mode not in ('mean', 'sum', 'count'):
            raise ValueError("mode must be 'mean', 'sum' or 'count'")
        rng = None if hkl_range is None else np.asarray(hkl_range, dtype=float).reshape(3, 2)
        if voxel is not None:
            voxel = np.broadcast_to(np.asarray(voxel, dtype=float), (3,)).copy()
        elif rng is not None:
            voxel = (rng[:, 1] - rng[:, 0]) / np.broadcast_to(np.asarray(bins, dtype=float), (3,))

        keys, sums, counts = [], [], []
        for data in stack:
            pts = np.asarray(data.points, dtype=float)
            w = np.asarray(data.intensities, dtype=float).reshape(-1)
            keep = np.isfinite(pts).all(axis=1) & np.isfinite(w)
            if rng is not None:
                keep &= ((pts >= rng[:, 0]) & (pts <= rng[:, 1])).all(axis=1)
            if intensity_range is not None:
                lo, hi = intensity_range
                if lo is not None:
                    keep &= w >= lo
                if hi is not None:
                    keep &= w <= hi
            pts, w = pts[keep], w[keep]
            del data
            if not len(w):
                continue
            if voxel is None:
                voxel = np.ptp(pts, axis=0) / np.broadcast_to(np.asarray(bins, dtype=float), (3,))
                voxel[voxel <= 0] = 1e-3
            idx = np.floor(pts / voxel).astype(np.int64)
            lo = idx.min(axis=0)
            dims = idx.max(axis=0) - lo + 1
            flat = np.ravel_multi_index((idx - lo).T, dims)
            cnt = np.bincount(flat, minlength=int(np.prod(dims)))
            occupied = np.flatnonzero(cnt)
            keys.append(np.column_stack(np.unravel_index(occupied, dims)) + lo)
            sums.append(np.bincount(flat, weights=w, minlength=cnt.size)[occupied])
            counts.append(cnt[occupied])
        if not keys:
            raise ValueError("bin_stack: no points left to bin (check hkl_range / intensity_range)")

        keys = np.concatenate(keys)
        lo = keys.min(axis=0)
        dims = keys.max(axis=0) - lo + 1
        if np.prod(dims) > 500_000_000:
            raise MemoryError(f"bin_stack: merged grid would be {tuple(dims)} voxels; "
                              f"use fewer bins, a larger voxel, or an hkl_range")
        flat = np.ravel_multi_index((keys - lo).T, dims)
        total = np.bincount(flat, weights=np.concatenate(sums), minlength=int(np.prod(dims)))
        hits = np.bincount(flat, weights=np.concatenate(counts), minlength=total.size)
        if mode == 'count':
            values = hits
        elif mode == 'sum':
            values = total
        else:
            values = np.divide(total, hits, out=np.zeros_like(total), where=hits > 0)

        if output == 'points':
            occupied = np.flatnonzero(hits)
            centres = (np.column_stack(np.unravel_index(occupied, dims)) + lo + 0.5) * voxel
            return Data(points=centres, intensities=values[occupied])

        grid = pv.ImageData()
        grid.dimensions = (dims + 1).tolist()
        grid.spacing = tuple(float(x) for x in voxel)
        grid.origin = tuple(float(x) for x in lo * voxel)
        # ravel_multi_index is C order (L fastest); VTK cells run H fastest
        grid.cell_data['intensity'] = values.reshape(dims).ravel(order='F').astype(np.float32)
        grid.cell_data['count'] = hits.reshape(dims).ravel(order='F').astype(np.float32)
        return grid

    def show_stack(self, stack, view='2d', hkl='HL', shape=(256, 256), cmap='viridis', clim=None, ncols=4,
                   gap=None, opacity=1.0, hide_empty=True, camera='iso', zoom=1.0, controls=True,
                   **slice_kwargs):
        """
        Cut the same slice through every scan in a stack and show the slices together.

        view='2d' lays them out as panels sharing one colour scale; view='3d' layers them in
        3D, one sheet per scan, stepped along a 'scan' axis.

        Usage:
            da.show_stack(stack[2:8], clim=(100, 5000))  # HL by default
            da.show_stack(stack[::4], view='3d', L=0.7, H=(0.2, 0.6), K=(0.1, 0.4))
            da.show_stack(stack, view='3d', orientation=da.slice_orientation)

        Parameters:
            stack: A DataStack, or a list of Data objects
            view (str): '2d' panels or '3d' layers
            hkl: As in slice_data. Defaults to 'HL'. A preset ('HK', 'KL', 'HL') puts the plane through the
                centre of the first scan so every scan is cut at the same place; pass a point,
                orientation= or H/K/L keywords to choose it.
            shape (tuple): Raster resolution of each slice
            cmap (str): Colormap
            clim (tuple): Shared (min, max) colour range; default the range over all slices
            ncols (int): Panels per row for view='2d'
            gap (float): Distance between layers for view='3d'; default 1/4 of the slice width
            opacity (float): Layer opacity for view='3d'
            hide_empty (bool): view='3d' leaves pixels with no data (0) see-through
            camera, zoom, controls: Starting view for view='3d', as in show_point_cloud
            **slice_kwargs: Passed to slice_data (normal, orientation, slab_thickness,
                intensity_range, H, K, L, axes, ...)

        Returns:
            list of (name, img, extent), one per scan; the slices are also kept in da.stack_slices
        """
        sliced = isinstance(stack, SliceStack)
        names = stack.names if isinstance(stack, (DataStack, SliceStack)) else [f'scan {i}' for i in range(len(stack))]
        layers, self.stack_slices = [], []
        uv_labels = ('U', 'V')
        for i, item in enumerate(stack):
            if sliced:
                sl = item
            else:
                kw = dict(slice_kwargs)
                if not ({'H', 'K', 'L'} & set(kw)) and 'orientation' not in kw:
                    if isinstance(hkl, str) and 'normal' not in kw and 'axes' not in kw:
                        pts = np.asarray(item.points, dtype=float)
                        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
                        kw['normal'] = tuple(float(x) for x in np.eye(3)[{'HK': 2, 'KL': 0, 'HL': 1}[hkl.upper()]])
                        hkl = tuple(round(float(x), 4) for x in centre)
                        print(f"show_stack: plane through H, K, L = {hkl}, normal {kw['normal']}")
                        slice_kwargs['normal'] = kw['normal']
                    kw['hkl'] = hkl
                    kw.setdefault('clamp_to_bounds', False)
                sl = self.slice_data(item, shape=shape, show=False, **kw)
            self.stack_slices.append(sl)
            with plt.ioff():
                fig = plt.figure()
                img, extent = self.show_slice(sl, axes=fig.add_subplot(), return_image=True)
                plt.close(fig)
            layers.append((names[i], np.asarray(img, dtype=float), [float(e) for e in extent]))
            n = np.asarray(sl.field_data['slice_normal'], dtype=float)
            k = int(np.argmax(np.abs(n)))
            if abs(n[k]) > 0.95 * np.linalg.norm(n):
                uv_labels = {0: ('K', 'L'), 1: ('H', 'L'), 2: ('H', 'K')}[k]
        if not layers:
            raise ValueError("show_stack: the stack is empty")

        if clim is None:
            vals = np.concatenate([img[np.isfinite(img)].ravel() for _, img, _ in layers])
            clim = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)

        if view == '2d':
            ncols = max(1, min(ncols, len(layers)))
            nrows = -(-len(layers) // ncols)
            fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 3.2 * nrows), squeeze=False,
                                    constrained_layout=True)
            for ax in axs.flat[len(layers):]:
                ax.set_visible(False)
            for ax, (name, img, extent) in zip(axs.flat, layers):
                im = ax.imshow(img, origin='lower', extent=extent, cmap=cmap, vmin=clim[0], vmax=clim[1],
                               aspect='auto')
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(uv_labels[0])
                ax.set_ylabel(uv_labels[1])
            fig.colorbar(im, ax=axs, label='Intensity', shrink=0.8)
            plt.show()
        elif view == '3d':
            width = max(e[1] - e[0] for _, _, e in layers)
            gap = float(gap) if gap is not None else width / 4
            p = pv.Plotter(notebook=True)
            # z only orders the layers, so it gets no tick numbers; each layer is labelled 'i: name'
            for i, (name, img, extent) in enumerate(layers):
                rows, cols = img.shape
                sheet = pv.ImageData(dimensions=(cols + 1, rows + 1, 1),
                                     spacing=((extent[1] - extent[0]) / cols, (extent[3] - extent[2]) / rows, 1),
                                     origin=(extent[0], extent[2], i * gap))
                vals = img.ravel().astype(np.float32)  # row-major: U fastest, as VTK expects
                if hide_empty:
                    vals[vals == 0] = np.nan
                sheet.cell_data['intensity'] = vals
                p.add_mesh(sheet, scalars='intensity', cmap=cmap, clim=clim, opacity=opacity, nan_opacity=0.0,
                           show_scalar_bar=(i == 0), scalar_bar_args={'title': 'Intensity'})
                p.add_point_labels([(extent[0], extent[3], i * gap)], [f'{i}: {name}'], font_size=10, point_size=1,
                                   shape_opacity=0.3, always_visible=True)
            bounds_kwargs = dict(xtitle=uv_labels[0], ytitle=uv_labels[1], ztitle='scan', show_zlabels=False, grid='back',
                                 location='outer', ticks='both', fmt=app_settings.ANALYSIS_AXIS_NUMBER_FORMAT)
            p.show_bounds(**bounds_kwargs)
            _show_with_camera(p, camera, zoom, controls, None, bounds_kwargs)
        else:
            raise ValueError("view must be '2d' or '3d'")
        return layers

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

    def show_tree(self, tree, indent=1):
        """
        Print a nested dict (e.g. data.metadata) as an indented tree with array dtypes and shapes.

        Usage:
            da.show_tree(data_raw.metadata)
        """
        for key, val in tree.items():
            if isinstance(val, dict):
                print("  " * indent + f"{key}/")
                self.show_tree(val, indent + 1)
            else:
                arr = np.asarray(val)
                print("  " * indent + (f"{key}: {arr.dtype} {arr.shape}" if arr.ndim else f"{key}: {val}"))

    def plot_raw_roi(self, raw_data, roi=None, background_roi=None, frame_index=1,
                     file_index=0, x_axis=None, cmap="gray", log_scale=False,
                     cap=None, **kwargs):
        """Plot per-frame signal sums with optional background-mean subtraction.

        Parameters:
            raw_data: A DataStack/folder, Data object containing ``images``, a 2D frame,
                or a 3D frame stack. For a DataStack, one frame is selected from every file.
            roi: Signal ROI as ``(x, y, width, height)`` or a saved ROI reference.
                Pass None, True, or ``"draw"`` to draw it interactively.
            background_roi: Background ROI as ``(x, y, width, height)``. Pass True or
                ``"draw"`` to select it interactively after the signal ROI. A saved ROI
                can be selected with ``{source: roi_number}``, ``{"file": source,
                "roi": roi_number}``, or ``{"data": source, "roi": roi_number}``, where
                source is a Data object or scan file path. It uses the displayed
                ``frame_index`` unless ``"frame": index`` selects a different frame.
            frame_index (int): One-based frame number within each file. For example,
                ``frame_index=40`` selects array index 39. With a DataStack, this frame
                is selected from every file and used to build the scan trace.
            file_index (int): DataStack file displayed while drawing the ROIs. This matches
                the first index in ``folder_scans[file_index][frame_index]``.
            x_axis: Metadata name such as ``"eta"`` or ``"temp"``, or numeric values
                matching the trace length. When None, the numeric metadata field with the
                strongest relative variation is used. Falls back to frame or scan number.
            cmap (str): Matplotlib colormap used for the displayed detector frame.
            log_scale: False for linear scales; True or ``"both"`` for a logarithmic
                image and symmetric-log 1D plot; ``"image"`` or ``"plot"`` for one only.
            cap: Optional capacitance configuration. Pass a file path or a dictionary with
                ``file``/``data``, ``x_axis`` (for example ``"eta"`` or ``"voltage"``),
                ``axis`` (``"x"`` or ``"y"``), and ``unit`` (``"F"`` or ``"pF"``).
                Capacitance is read from ``entry/instruments/bluesky/streams/primary``.

        Returns:
            dict: Nested ``integrated_intensity`` and ``plot`` dictionaries.
        """
        if plt is None:
            raise ImportError("Matplotlib is required for plot_raw_roi()")

        if "scan_index" in kwargs:
            file_index = kwargs.pop("scan_index")
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"plot_raw_roi() got an unexpected keyword argument {unexpected!r}")

        from matplotlib.colors import LogNorm
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import RectangleSelector
        from scipy import ndimage

        if isinstance(log_scale, str):
            log_mode = log_scale.lower()
            if log_mode not in {"image", "plot", "both"}:
                raise ValueError("log_scale must be False, True, 'image', 'plot', or 'both'")
        else:
            log_mode = "both" if log_scale else "none"
        image_log = log_mode in {"image", "both"}
        plot_log = log_mode in {"plot", "both"}

        frame_number = int(frame_index)
        if frame_number < 1:
            raise ValueError("frame_index is one-based and must be at least 1")
        frame_offset = frame_number - 1
        file_index = int(file_index)
        is_folder = isinstance(raw_data, DataStack)
        scan_data = None
        if is_folder:
            if not len(raw_data):
                raise ValueError("raw_data folder does not contain scans")
            if not -len(raw_data) <= file_index < len(raw_data):
                raise IndexError(f"file_index {file_index} is outside the folder scan range")
            file_index %= len(raw_data)
            selected_frames = []
            scan_data = []
            for path in raw_data.files:
                loaded_scan = self._load_data_file(path)
                scan_data.append(loaded_scan)
                scan_images = np.asarray(loaded_scan.images)
                if frame_offset >= len(scan_images):
                    raise IndexError(
                        f"frame_index {frame_number} is outside the frame range for {path!r}"
                    )
                selected_frames.append(scan_images[frame_offset])
            images = np.asarray(selected_frames)
            display_index = file_index
            trace_labels = raw_data.names
            default_axis_label = "Scan"
            image_title = f"File {file_index + 1}, frame {frame_number}"
        else:
            images = raw_data.images if hasattr(raw_data, "images") else raw_data
            if images is None:
                raise ValueError("raw_data does not contain detector images")
            images = np.asarray(images)
            if images.ndim == 2:
                images = images[np.newaxis, ...]
            if images.ndim != 3:
                raise ValueError(
                    "Expected a DataStack, one frame (rows, cols), or a stack "
                    "(frames, rows, cols)"
                )
            if frame_offset >= len(images):
                raise IndexError(f"frame_index {frame_number} is outside the frame range")
            display_index = frame_offset
            trace_labels = None
            default_axis_label = "Frame"
            image_title = f"Frame {frame_number}"

        def flatten_numeric_metadata(tree, prefix=""):
            values = {}
            if isinstance(tree, Group):
                tree = vars(tree)
            if not isinstance(tree, dict):
                return values
            for key, value in tree.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, (dict, Group)):
                    values.update(flatten_numeric_metadata(value, path))
                    continue
                try:
                    array = np.asarray(value, dtype=float).ravel()
                except (TypeError, ValueError):
                    continue
                if array.size:
                    values[path] = array
            return values

        def metadata_axes():
            if is_folder:
                per_scan = [flatten_numeric_metadata(data.metadata) for data in scan_data]
                if not per_scan:
                    return {}
                common_names = set(per_scan[0]).intersection(*(set(item) for item in per_scan[1:]))
                axes = {}
                for name in common_names:
                    values = []
                    for item in per_scan:
                        array = item[name]
                        index = min(frame_offset, array.size - 1)
                        values.append(array[index])
                    axes[name] = np.asarray(values, dtype=float)
                return axes
            metadata = getattr(raw_data, "metadata", None)
            axes = flatten_numeric_metadata(metadata)
            return {name: values for name, values in axes.items() if values.size == len(images)}

        def resolve_x_axis():
            default_values = np.arange(1, len(images) + 1)
            if x_axis is not None and not isinstance(x_axis, str):
                values = np.asarray(x_axis, dtype=float).ravel()
                if values.size != len(images):
                    raise ValueError("x_axis values must match the ROI trace length")
                return values, "X"

            available = metadata_axes()
            if isinstance(x_axis, str):
                if x_axis.lower() in {"frame", "scan", "index"}:
                    return default_values, default_axis_label
                matches = [
                    name for name in available
                    if name.lower() == x_axis.lower()
                    or name.rsplit(".", 1)[-1].lower() == x_axis.lower()
                ]
                if not matches:
                    names = ", ".join(sorted(available)) or "none"
                    raise ValueError(f"No numeric metadata named {x_axis!r}; available: {names}")
                name = min(matches, key=len)
                return available[name], name

            varying = {}
            for name, values in available.items():
                if "roi" in name.lower():
                    continue
                finite = values[np.isfinite(values)]
                if finite.size < 2:
                    continue
                variation = float(np.ptp(finite))
                scale = max(float(np.max(np.abs(finite))), 1.0)
                if variation > np.finfo(float).eps * scale:
                    varying[name] = variation / scale
            if varying:
                name = max(varying, key=varying.get)
                return available[name], name
            return default_values, default_axis_label

        trace_positions, trace_axis_label = resolve_x_axis()

        def find_hdf5_dataset(group, field_name):
            target = field_name.lower()
            matches = []

            def visitor(name, item):
                if isinstance(item, h5py.Dataset) and name.rsplit("/", 1)[-1].lower() == target:
                    matches.append(item[()])

            group.visititems(visitor)
            return np.asarray(matches[0]).ravel() if matches else None

        def load_capacitance():
            if cap is None:
                return None
            cap_config = dict(cap) if isinstance(cap, dict) else {"file": cap}
            source = cap_config.get("file", cap_config.get("data", self.file_path))
            axis_name = cap_config.get("x_axis", x_axis)
            plot_axis = str(cap_config.get("axis", "y")).lower()
            unit = str(cap_config.get("unit", "F"))
            if plot_axis not in {"x", "y"}:
                raise ValueError("cap['axis'] must be 'x' or 'y'")
            if unit.lower() not in {"f", "pf"}:
                raise ValueError("cap['unit'] must be 'F' or 'pF'")
            if isinstance(source, Data):
                source_path = getattr(source, "file_path", None)
                if not source_path:
                    raise ValueError("CAP data requires an HDF5 file path")
            else:
                source_path = os.path.expanduser(str(source))
            if not source_path or not os.path.isfile(source_path):
                raise FileNotFoundError(f"No CAP data file at {source_path!r}")

            with h5py.File(source_path, "r") as handle:
                primary = None
                for path in (
                    "entry/instruments/bluesky/streams/primary",
                    "entry/instrument/bluesky/streams/primary",
                ):
                    if path in handle:
                        primary = handle[path]
                        break
                if primary is None:
                    raise ValueError(
                        "CAP file does not contain entry/instruments/bluesky/streams/primary"
                    )
                capacitance = find_hdf5_dataset(primary, "ah2700a_capacitance")
                if capacitance is None:
                    raise ValueError("CAP file does not contain ah2700a_capacitance")
                positions = (
                    find_hdf5_dataset(primary, axis_name) if axis_name is not None else None
                )

            if unit.lower() == "pf":
                capacitance = capacitance * 1e12
                unit = "pF"
            else:
                unit = "F"
            if positions is None:
                positions = np.arange(1, capacitance.size + 1, dtype=float)
                axis_name = "Point"
            if positions.size != capacitance.size:
                raise ValueError("CAP position and capacitance arrays must have matching lengths")

            valid = np.isfinite(positions) & np.isfinite(capacitance)
            positions = np.asarray(positions[valid], dtype=float)
            capacitance = np.asarray(capacitance[valid], dtype=float)
            unique_positions, inverse = np.unique(positions, return_inverse=True)
            sums = np.bincount(inverse, weights=capacitance)
            counts = np.bincount(inverse)
            averaged = sums / counts
            return {
                "capacitance": averaged,
                "position": unique_positions,
                "x_axis": axis_name,
                "axis": plot_axis,
                "unit": unit,
                "source": source_path,
                "samples_per_position": counts,
            }

        cap_result = load_capacitance()
        if cap_result is not None and cap_result["axis"] == "x":
            if cap_result["capacitance"].size != len(images):
                raise ValueError(
                    "CAP capacitance values must match the ROI trace length when cap axis is 'x'"
                )
            trace_positions = cap_result["capacitance"]
            trace_axis_label = f"Capacitance ({cap_result['unit']})"

        draw_background = background_roi is True or (
            isinstance(background_roi, str) and background_roi.lower() == "draw"
        )
        draw_signal = roi is None or roi is True or (
            isinstance(roi, str) and roi.lower() == "draw"
        )
        result = {
            "integrated_intensity": {
                "corrected": None,
                "signal_sum": None,
                "background_mean": None,
            },
            "CAP": cap_result,
            "plot": {
                "roi": None,
                "background_roi": None,
                "frame_index": frame_number,
                "file_index": file_index if is_folder else None,
                "x": trace_positions,
                "x_axis": trace_axis_label,
                "labels": trace_labels,
                "cmap": cmap,
                "log_scale": log_mode,
                "image_log_scale": image_log,
                "trace_log_scale": plot_log,
                "figure": None,
                "image_axis": None,
                "trace_axis": None,
                "cap_axis": None,
                "selector": None,
            },
        }
        figure, (image_axis, trace_axis) = plt.subplots(1, 2, figsize=(12, 4))
        result["plot"].update({
            "figure": figure,
            "image_axis": image_axis,
            "trace_axis": trace_axis,
        })
        display_image = images[display_index].T
        image_options = {"cmap": cmap, "origin": "upper"}
        if image_log:
            positive = display_image[np.isfinite(display_image) & (display_image > 0)]
            if positive.size:
                image_options["norm"] = LogNorm(
                    vmin=float(np.min(positive)),
                    vmax=float(np.max(positive)),
                )
        image_axis.imshow(display_image, **image_options)
        image_axis.set_xlabel("x pixel")
        image_axis.set_ylabel("y pixel")
        trace_axis.set_xlabel(trace_axis_label)
        trace_axis.set_ylabel("Signal ROI sum")
        patches = []

        def scalar_at_frame(value, saved_frame_index):
            values = np.asarray(value).ravel()
            if values.size == 0:
                raise ValueError("Saved ROI coordinate is empty")
            if values.size == 1:
                return float(values[0])
            if not -values.size <= saved_frame_index < values.size:
                raise IndexError(
                    f"Saved ROI frame {saved_frame_index} is outside its coordinate range"
                )
            index = saved_frame_index % values.size
            return float(values[index])

        def group_members(group):
            return vars(group) if isinstance(group, Group) else group

        def find_member(group, names):
            members = group_members(group)
            if not isinstance(members, dict):
                raise ValueError("Saved ROI metadata is not a group")
            lower_names = {str(name).lower(): name for name in members}
            for name in names:
                key = lower_names.get(name.lower())
                if key is not None:
                    return members[key]
            raise ValueError(f"Saved ROI is missing {names[0]}")

        def saved_roi(source, roi_number, saved_frame_index=None):
            saved_frame_number = (
                frame_number if saved_frame_index is None else int(saved_frame_index)
            )
            if saved_frame_number < 1:
                raise ValueError("Saved ROI frame is one-based and must be at least 1")
            saved_frame_offset = saved_frame_number - 1
            if isinstance(source, (str, os.PathLike)):
                source = self._load_data_file(source)
            entry = getattr(source, "entry", None)
            if entry is None:
                raise ValueError("Saved ROI source must be a Data object or scan file path")

            roi_groups = getattr(entry, "rois", None)
            if roi_groups is None:
                data_group = getattr(entry, "data", None)
                metadata = getattr(data_group, "metadata", None)
                roi_groups = getattr(metadata, "rois", None)
            if roi_groups is None:
                raise ValueError("Saved ROI source does not contain ROI metadata")

            members = group_members(roi_groups)
            roi_names = (f"ROI{int(roi_number)}", f"ROI_{int(roi_number)}")
            roi_group = find_member(members, roi_names)
            return (
                scalar_at_frame(
                    find_member(roi_group, ("MinX", "MIN_X", "min_x", "x")),
                    saved_frame_offset,
                ),
                scalar_at_frame(
                    find_member(roi_group, ("MinY", "MIN_Y", "min_y", "y")),
                    saved_frame_offset,
                ),
                scalar_at_frame(
                    find_member(roi_group, ("SizeX", "SIZE_X", "size_x", "width")),
                    saved_frame_offset,
                ),
                scalar_at_frame(
                    find_member(roi_group, ("SizeY", "SIZE_Y", "size_y", "height")),
                    saved_frame_offset,
                ),
            )

        def resolve_roi(value):
            if not isinstance(value, dict):
                return value
            if "roi" in value and ("file" in value or "data" in value):
                source = value.get("file", value.get("data"))
                saved_frame_index = value.get("frame", value.get("frame_index"))
                return saved_roi(source, value["roi"], saved_frame_index)
            if len(value) == 1:
                source, roi_number = next(iter(value.items()))
                return saved_roi(source, roi_number)
            raise ValueError(
                "Saved ROI must be {source: roi_number} or "
                "{'file': source, 'roi': roi_number}"
            )

        def normalize_roi(values):
            x, y, width, height = (float(value) for value in values)
            x0, x1 = sorted((int(np.floor(x)), int(np.ceil(x + width))))
            y0, y1 = sorted((int(np.floor(y)), int(np.ceil(y + height))))
            x0, x1 = np.clip((x0, x1), 0, images.shape[1])
            y0, y1 = np.clip((y0, y1), 0, images.shape[2])
            if x1 <= x0 or y1 <= y0:
                raise ValueError("ROI must contain at least one pixel")
            return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

        def selected_roi(click, release):
            if None in (click.xdata, click.ydata, release.xdata, release.ydata):
                return None
            return normalize_roi((
                click.xdata,
                click.ydata,
                release.xdata - click.xdata,
                release.ydata - click.ydata,
            ))

        def roi_pixels(bounds):
            x, y, width, height = bounds
            return images[:, x:x + width, y:y + height]

        def draw_overlays():
            for patch in patches:
                patch.remove()
            patches.clear()
            for bounds, color, label in (
                (result["plot"]["roi"], "tab:red", "signal"),
                (result["plot"]["background_roi"], "tab:cyan", "background"),
            ):
                if bounds is not None:
                    x, y, width, height = bounds
                    patch = Rectangle(
                        (x, y), width, height, fill=False,
                        edgecolor=color, linewidth=2, label=label,
                    )
                    image_axis.add_patch(patch)
                    patches.append(patch)
            if patches:
                image_axis.legend(loc="upper right")
            figure.canvas.draw_idle()

        def refresh_plot():
            if result["plot"]["roi"] is not None:
                signal = roi_pixels(result["plot"]["roi"])
                axes = (1, 2)
                signal_integrated = np.asarray([
                    ndimage.sum(np.nan_to_num(frame, nan=0.0)) for frame in signal
                ])
                background_mean = None
                trace = signal_integrated.copy()
                if result["plot"]["background_roi"] is not None:
                    background = np.nanmean(
                        roi_pixels(result["plot"]["background_roi"]), axis=axes
                    )
                    background_mean = background
                    trace = signal_integrated - background_mean

                result["integrated_intensity"].update({
                    "corrected": trace,
                    "signal_sum": signal_integrated,
                    "background_mean": background_mean,
                })
                trace_axis.clear()
                trace_axis.plot(
                    trace_positions,
                    trace,
                    marker="o" if len(trace) == 1 else None,
                )
                if plot_log:
                    trace_axis.set_yscale("symlog")
                if cap_result is not None and cap_result["axis"] == "y":
                    cap_axis = result["plot"]["cap_axis"]
                    if cap_axis is None:
                        cap_axis = trace_axis.twinx()
                        result["plot"]["cap_axis"] = cap_axis
                    cap_axis.clear()
                    cap_axis.plot(
                        cap_result["position"],
                        cap_result["capacitance"],
                        color="tab:orange",
                    )
                    cap_axis.set_xlabel(str(cap_result["x_axis"]))
                    cap_axis.set_ylabel(f"Capacitance ({cap_result['unit']})")
                trace_axis.set_xlabel(trace_axis_label)
                if (
                    trace_labels is not None
                    and trace_axis_label == default_axis_label
                    and len(trace_labels) <= 20
                ):
                    labels = [
                        f"{index + 1}: {name}" for index, name in enumerate(trace_labels)
                    ]
                    trace_axis.set_xticks(trace_positions, labels, rotation=45, ha="right")
                if result["plot"]["background_roi"] is not None:
                    trace_axis.set_ylabel("Signal ROI sum - background ROI mean")
                else:
                    trace_axis.set_ylabel("Signal ROI sum")
                trace_axis.grid(alpha=0.25)
            draw_overlays()

        def start_background_selection():
            image_axis.set_title(f"{image_title}: draw the BACKGROUND ROI")

            def on_background(click, release):
                bounds = selected_roi(click, release)
                if bounds is not None:
                    result["plot"]["background_roi"] = bounds
                    image_axis.set_title(f"{image_title}: signal and background ROIs")
                    refresh_plot()

            result["plot"]["selector"] = RectangleSelector(
                image_axis,
                on_background,
                useblit=True,
                button=[1],
                interactive=True,
                props={"edgecolor": "tab:cyan", "fill": False, "linewidth": 2},
            )

        if background_roi is not None and not draw_background:
            result["plot"]["background_roi"] = normalize_roi(
                resolve_roi(background_roi)
            )
            draw_overlays()

        if not draw_signal:
            result["plot"]["roi"] = normalize_roi(resolve_roi(roi))
            refresh_plot()
            if draw_background:
                start_background_selection()
        else:
            image_axis.set_title(f"{image_title}: draw the SIGNAL ROI")

            def on_signal(click, release):
                bounds = selected_roi(click, release)
                if bounds is None:
                    return
                result["plot"]["roi"] = bounds
                result["plot"]["selector"].set_active(False)
                refresh_plot()
                if draw_background:
                    start_background_selection()
                else:
                    image_axis.set_title(f"{image_title}: signal ROI")

            result["plot"]["selector"] = RectangleSelector(
                image_axis,
                on_signal,
                useblit=True,
                button=[1],
                interactive=True,
                props={"edgecolor": "tab:red", "fill": False, "linewidth": 2},
            )

        plt.tight_layout()
        plt.show()
        return result

    def plot_roi_trace(self, *args, **kwargs):
        """Compatibility alias for :meth:`plot_raw_roi`."""
        return self.plot_raw_roi(*args, **kwargs)

    def load_data(self, source=None, cache=False):
        """
        Load the first scan from a file, folder, glob, or list.

        The return value is always Data. The complete detected selection is retained internally
        so show_slice can navigate the same slice through every loaded scan.

        Usage:
            data = da.load_data('/path/to/file.h5')
            data = da.load_data('/path/to/scans/')

        Parameters:
            source: A file, folder, glob pattern, list of files, or None for the cached file
            cache (bool): Keep stack items in memory after their first load

        Returns:
            Data for the first detected file

        Raises:
            FileNotFoundError: If the source does not contain any scan files
            Exception: If file loading fails

        Examples:
            data = da.load_data('experiment_data.h5')
            da.show_point_cloud(data)
        """
        if source is None:
            source = getattr(self, 'file_path', None)
        if isinstance(source, (list, tuple, np.ndarray, range)):
            files = [os.path.expanduser(str(path)) for path in source]
            if not files:
                raise FileNotFoundError("No scan files were provided")
        else:
            path = os.path.expanduser(str(source)) if source else None
            if not path:
                raise FileNotFoundError("No scan file - set `source` to a scan file or folder.")
            if os.path.isdir(path):
                files = sorted(
                    os.path.join(path, name)
                    for name in os.listdir(path)
                    if name.lower().endswith(app_settings.ANALYSIS_SCAN_EXTENSIONS)
                )
            elif os.path.isfile(path):
                files = [path]
            else:
                files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No scan files found for {source!r}")

        self._loaded_stack = DataStack(files, self._load_3d_file, cache=cache)
        self.file_path = self._loaded_stack.files[0]
        return self._load_data_file(self.file_path)

    def _load_data_file(self, path):
        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No scan file at {path!r}")

        def read_group(grp):
            return {k: read_group(v) if isinstance(v, h5py.Group)
                    else (v.asstr()[()] if h5py.check_string_dtype(v.dtype) else v[()])
                    for k, v in grp.items()}

        def to_group(tree):
            return Group({k: to_group(v) if isinstance(v, dict) else v for k, v in tree.items()})

        with h5py.File(path, 'r') as f:
            tree = read_group(f['entry'])
        images = tree['data']['data']
        return Data(intensities=images.ravel(), metadata=tree['data'].get('metadata', {}), num_images=images.shape[0],
                    shape=images.shape[1:], images=images, entry=to_group(tree), file_path=path)

    def _load_3d_file(self, path):
        data = self._load_data_file(path)
        data.points = RSMConverter().get_q_points(path)
        return data

    def load_3d(self, file_path: Optional[str] = None):
        """
        Load a scan like load_data, then compute its (N, 3) HKL points from the
        NeXus geometry in /entry/data/metadata/HKL.

        Usage:
            data = da.load_3d('/path/to/file.h5')
            da.show_point_cloud(data)
        """
        path = file_path or getattr(self, 'file_path', None)
        if not path:
            raise FileNotFoundError("No scan file - call load_data with a scan file or folder first.")
        return self._load_3d_file(path)

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
            raise FileNotFoundError("No scan file - set `filename` in the loading cell to a real scan.")

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

def _show_with_camera(p, camera='iso', zoom=1.0, controls=True, slice_buttons=None, bounds_kwargs=None):
    """
    Set the starting view of a PyVista plotter and show it.

    camera is 'iso', 'hk', 'hl' or 'kl'; an (azimuth, elevation) pair in degrees measured
    from the HL view (L up, azimuth turns about L); or a full PyVista camera_position.
    controls=True renders with the trame backend and adds HK / HL / KL / ISO and zoom
    buttons to its toolbar; slice_buttons (rows of (icon, tooltip, callback)) go in a Slice
    dropdown after them. bounds_kwargs re-applies the H/K/L axes whenever the toolbar ruler
    toggle redraws them with PyVista's defaults.

    Usage:
        p = pv.Plotter(notebook=True)
        p.add_mesh(pv.Sphere())
        _show_with_camera(p, camera=(30, 20), zoom=1.5)
    """
    views = {'hk': p.view_xy, 'hl': p.view_xz, 'kl': p.view_yz, 'iso': p.view_isometric}
    if isinstance(camera, str):
        views[camera.lower()]()
    elif len(camera) == 2:
        p.view_xz()
        p.camera.azimuth, p.camera.elevation = camera
    else:
        p.camera_position = camera
    p.camera.zoom(zoom)
    if not controls:
        return p.show()

    from pyvista.trame.ui import get_viewer
    from trame.widgets import html, vuetify3

    def menu_items():
        viewer = get_viewer(p)

        def act(fn):
            def run():
                fn()
                viewer.update_camera()
            return run

        compact = dict(variant='text', density='compact', size='small', min_width=0, classes='px-1')
        for name, view in views.items():
            vuetify3.VBtn(name.upper(), click=act(lambda view=view: view(render=False)), **compact)
        step = app_settings.ANALYSIS_CAMERA_ZOOM_STEP
        vuetify3.VBtn(icon='mdi-magnify-plus-outline', click=act(lambda: p.camera.zoom(step)), **compact)
        vuetify3.VBtn(icon='mdi-magnify-minus-outline', click=act(lambda: p.camera.zoom(1 / step)), **compact)
        if bounds_kwargs:
            def restore_bounds(**state):
                if state[viewer.GRID]:
                    p.show_bounds(**bounds_kwargs)
                    viewer.update()
            viewer.server.state.change(viewer.GRID)(restore_bounds)
        if slice_buttons:
            with vuetify3.VMenu(close_on_content_click=False, location='bottom'):
                with vuetify3.Template(v_slot_activator=('{ props }',)):
                    vuetify3.VBtn('Slice', v_bind=('props',), append_icon='mdi-menu-down', **compact)
                with vuetify3.VCard(classes='pa-1'):
                    for row in slice_buttons:
                        with html.Div(style='display: flex; justify-content: center;'):
                            for icon, tip, fn in row:
                                vuetify3.VBtn(icon=icon, title=tip, click=lambda fn=fn: (fn(), viewer.update()),
                                              variant='text', density='comfortable')

    return p.show(jupyter_backend='trame', jupyter_kwargs=dict(add_menu_items=menu_items))


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
    from fractions import Fraction

    import numpy as np
    
    h, k, l_idx = np.asarray(hkl_vector, dtype=float)[:3]
    
    def rationalize(x):
        """Convert float to rational if close to a simple fraction."""
        if abs(x) < tolerance:
            return 0, 1
        try:
            frac = Fraction(x).limit_denominator(max_denominator)
            if abs(float(frac) - x) < tolerance:
                return frac.numerator, frac.denominator
        except Exception:
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
                return f"{coeff:.2g}{label}"
    
    terms = []
    for coeff, label in [(h, 'H'), (k, 'K'), (l_idx, 'L')]:
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
