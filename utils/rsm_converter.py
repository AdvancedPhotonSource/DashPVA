import toml
import numpy as np
import h5py
import hdf5plugin
import xrayutilities as xu
import logging
from typing import Tuple, Dict, Any, Optional
import pyvista as pv
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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

class RSMConverter:
    """
    Utility class for Reciprocal Space Mapping (RSM) conversion
    from HDF5 diffraction data.
    """

    def __init__(self, config_path: str = "pv_configs/s6lambda.toml"):
        self.hkl_pv_channels = set()
        self.hkl_config: Dict[str, Dict[str, str]] = {}

        config = toml.load(config_path)
        if "HKL" in config:
            self.hkl_config = config["HKL"]
            for section in self.hkl_config.values():
                self.hkl_pv_channels.update(section.values())

    # ------------------------------------------------------------------
    # HDF5 utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset_index(h5file: h5py.File) -> Dict[str, h5py.Dataset]:
        """
        Build a map of dataset full paths for fast lookup.
        """
        index = {}

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                index[name] = obj

        h5file.visititems(visitor)
        return index

    def get_data_by_key(self, h5index: Dict[str, h5py.Dataset], key: str):
        """
        Retrieve dataset whose path ends with `key`.
        """
        for path, ds in h5index.items():
            if path.endswith(key):
                if ds.dtype.kind in {"S", "O"}:
                    return ds.asstr()[:]
                return ds[:]
        raise KeyError(f"Dataset with key '{key}' not found.")

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------

    def get_sample_and_detector_circles(
        self, h5index, frame: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        sc_dir, sc_pos = [], []
        dc_dir, dc_pos = [], []

        for section, pv_dict in self.hkl_config.items():
            for pv in pv_dict.values():
                if section.startswith("SAMPLE_CIRCLE"):
                    if pv.endswith("DirectionAxis"):
                        sc_dir.append(self.get_data_by_key(h5index, pv))
                    elif pv.endswith("Position"):
                        sc_pos.append(self.get_data_by_key(h5index, pv))
                elif section.startswith("DETECTOR_CIRCLE"):
                    if pv.endswith("DirectionAxis"):
                        dc_dir.append(self.get_data_by_key(h5index, pv))
                    elif pv.endswith("Position"):
                        dc_pos.append(self.get_data_by_key(h5index, pv))

        sc_dir = np.asarray(sc_dir)[:, frame]
        sc_pos = np.asarray(sc_pos)[:, frame]
        dc_dir = np.asarray(dc_dir)[:, frame]
        dc_pos = np.asarray(dc_pos)[:, frame]

        return sc_dir, sc_pos, dc_dir, dc_pos

    def get_axis_directions(self, h5index):
        primary, inplane, surface = [], [], []

        for i in range(1, 4):
            primary.append(
                float(self.get_data_by_key(h5index, f"PrimaryBeamDirection:AxisNumber{i}")[0])
            )
            inplane.append(
                float(self.get_data_by_key(h5index, f"InplaneReferenceDirection:AxisNumber{i}")[0])
            )
            surface.append(
                float(self.get_data_by_key(h5index, f"SampleSurfaceNormalDirection:AxisNumber{i}")[0])
            )

        return primary, inplane, surface

    def get_ub_matrix(self, h5index) -> np.ndarray:
        key = self.hkl_config["SPEC"]["UB_MATRIX_VALUE"]
        ub = self.get_data_by_key(h5index, key)[:9]
        return np.asarray(ub).reshape(3, 3)

    def get_energy(self, h5index) -> float:
        key = self.hkl_config["SPEC"]["ENERGY_VALUE"]
        return float(self.get_data_by_key(h5index, key)[0]) * 1000.0  # keV → eV

    # ------------------------------------------------------------------
    # Detector parameters
    # ------------------------------------------------------------------

    def get_det_param(self, h5index, shape):
        roi = [0, shape[1], 0, shape[2]]

        pixel_dir1 = self.get_data_by_key(h5index, "DetectorSetup:PixelDirection1")[0]
        pixel_dir2 = self.get_data_by_key(h5index, "DetectorSetup:PixelDirection2")[0]
        cch1, cch2 = self.get_data_by_key(h5index, "DetectorSetup:CenterChannelPixel")[:2]

        nch1, nch2 = shape[1], shape[2]
        size = self.get_data_by_key(h5index, "DetectorSetup:Size")
        pixel_width1 = size[0] / nch1
        pixel_width2 = size[1] / nch2

        distance = self.get_data_by_key(h5index, "DetectorSetup:Distance")[0]

        return pixel_dir1, pixel_dir2, cch1, cch2, nch1, nch2, pixel_width1, pixel_width2, distance, roi

    # ------------------------------------------------------------------
    # RSM computation
    # ------------------------------------------------------------------

    def create_rsm(self, filename: str, frame: int):
        try:
            with h5py.File(filename, "r") as f:
                h5index = self._build_dataset_index(f)
                shape = f["entry/data/data"].shape

                sc_dir, sc_pos, dc_dir, dc_pos = \
                    self.get_sample_and_detector_circles(h5index, frame)

                primary, inplane, surface = self.get_axis_directions(h5index)
                ub = self.get_ub_matrix(h5index)
                energy = self.get_energy(h5index)

                qconv = xu.experiment.QConversion(tuple(sc_dir), tuple(dc_dir), primary)
                hxrd = xu.HXRD(inplane, surface, en=energy, qconv=qconv)

                pixel_dir1, pixel_dir2, cch1, cch2, nch1, nch2, pixel_width1, pixel_width2, distance,roi = self.get_det_param(h5index, shape)
                hxrd.Ang2Q.init_area(
                    pixel_dir1, pixel_dir2,
                    cch1=cch1, cch2=cch2,
                    Nch1=nch1, Nch2=nch2,
                    pwidth1=pixel_width1,
                    pwidth2=pixel_width2,
                    distance=distance,
                    roi=roi
                )


                angles = [*sc_pos, *dc_pos]
                return hxrd.Ang2Q.area(*angles, UB=ub)

        except Exception:
            logger.exception("RSM conversion failed")
            raise

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_q_points(self, filename: str) -> np.ndarray:
        with h5py.File(filename, "r") as f:
            n_frames = f["entry/data/data"].shape[0]

        qxyz = [self.create_rsm(filename, i) for i in range(n_frames)]
        qxyz = np.asarray(qxyz)

        return np.column_stack(
            (
                qxyz[:,0,:,:].ravel(),
                qxyz[:,1,:,:].ravel(),
                qxyz[:,2,:,:].ravel(),
            )
        )

    def get_intensity(self, filename: str) -> np.ndarray:
        with h5py.File(filename, "r") as f:
            return f["entry/data/data"][:].ravel()

    def load_h5_to_3d(self, filename: str):
        with h5py.File(filename, "r") as f:
            shape = f["entry/data/data"].shape
        qx = None
        qx = self.take_data_by_key(filename,'qx')
        if qx is not None:
            qy = self.take_data_by_key(filename,'qy')
            qz = self.take_data_by_key(filename,'qz')
            q = np.column_stack((np.ravel(qx), np.ravel(qy), np.ravel(qz)))
        else:
            q = self.get_q_points(filename)
        intensity = self.get_intensity(filename)

        return q, intensity, shape[0], shape[1:]

    def take_data_by_key(self, file_path, target_key):
        with h5py.File(file_path, 'r') as f:
            # This list will hold our result
            found_path = None
            
            # Internal function to check every object name
            def find_key(name, obj):
                nonlocal found_path
                # Check if the path ends with our target key
                if name.endswith(target_key):
                    found_path = name
                    return True # Stop searching once found

            # Walk the tree
            f.visititems(find_key)

            if found_path:
                #print(f"Found {target_key} at: {found_path}")
                # Access the data
                ds = f[found_path]
                
                # Handle string (object) vs numeric data
                if ds.dtype == 'O':
                    return ds.asstr()[:]
                else:
                    return ds[:]
            else:
                print(f"Key '{target_key}' not found in file.")
                return None
            
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

        points_3d, intensities, num_images, shape = self.load_h5_to_3d(path)
        return Data(points=points_3d, intensities=intensities,num_images=num_images,shape=shape)

    def slice_data(self, data, origin=None, normal=None, shape=(512, 512), slab_thickness=None, 
                   clamp_to_bounds=True, spacing=(0.5, 0.5, 0.5), grid_origin=(0.0, 0.0, 0.0), 
                   show=True, axes=None):
        """
        Create a slice from either a volume or a point cloud with advanced processing options.

        This is the most comprehensive slice extraction method, supporting both volume and
        point cloud data with customizable HKL axes, integration parameters, and visualization
        options. Handles coordinate transformations, adaptive interpolation, and metadata preservation.

        Usage:
            # Traditional usage with presets
            slice_data = da.slice_data(data, orign='HK')
            
            # Mixed HKL axes with custom orientation
            slice_data = da.slice_data(data, axes=((0.5, 0, 0), (0, 1, 0)))
            
            # Point cloud slicing with slab thickness
            slice_data = da.slice_data(point_data, origin='KL', slab_thickness=0.2)

        Parameters:
            data: Volume or point cloud data. Supported formats:
                - Volume: pv.ImageData, np.ndarray (D,H,W), or (ndarray_volume, shape_tuple)
                - Points: (points, intensities) where points is (N,3) and intensities is (N,)
                - Data object with .points and .intensities attributes
                - Dict with 'points' and 'intensities' keys
            origin: Slice origin or orientation preset. Options:
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
                - None: Use origin/normal parameters as before

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
            hk_slice = da.slice_data(volume_data, origin='HK')
            
            # Extract custom orientation slice from point cloud
            custom_slice = da.slice_data(
                point_data, 
                axes=((1, 1, 0), (0, 0, 1)), 
                slab_thickness=0.1
            )
            
            # High-resolution slice with specific spacing
            hr_slice = da.slice_data(
                data, origin=(1.0, 0.5, 0.0), 
                shape=(1024, 1024), 
                spacing=(0.1, 0.1, 0.1)
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
            # Resolve normal (from origin preset or provided normal)
            _n = None
            if isinstance(origin, str):
                s = origin.strip().lower()
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
            if isinstance(origin, (tuple, list, _np.ndarray)) and len(origin) == 3:
                _o = _np.array([float(origin[0]), float(origin[1]), float(origin[2])], dtype=float)
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
            center = getattr(grid, 'center', (0.0, 0.0, 0.0))
            origin_vec, n_vec = _resolve_plane(center, getattr(grid, 'bounds', None))
            sl = grid.slice(normal=n_vec, origin=origin_vec)
            sl.field_data['slice_normal'] = np.asarray(n_vec, dtype=float)
            sl.field_data['slice_origin'] = np.asarray(origin_vec, dtype=float)
            # call show slice
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
        #print("origin is", origin_vec)

        # Optional pre-filter: limit contributing points to a slab around the plane for interpolation
        rel = points - origin_vec[None, :]
        d_signed = rel.dot(n_vec)
        use_slab = slab_thickness is not None and np.isfinite(float(slab_thickness)) and float(slab_thickness) > 0.0
        if use_slab:
            tol = float(slab_thickness)
            mask = np.abs(d_signed) <= tol
            # Fallback to all points if slab yields none
            if not np.any(mask):
                mask = np.ones(points.shape[0], dtype=bool)
        else:
            mask = np.ones(points.shape[0], dtype=bool)

        pts_for_extent = points[mask]
        vals_for_extent = intensities[mask]

        # Resolve HKL axes or build default in-plane basis
        u_hkl = None
        v_hkl = None
        n_hkl = None
        
        if axes is not None:
            # Parse axes parameter: ((u_hkl, v_hkl),) or ((u_hkl, v_hkl), n_hkl)
            if isinstance(axes, (tuple, list)) and len(axes) >= 2:
                u_hkl = _np.asarray(axes[0], dtype=float)
                v_hkl = _np.asarray(axes[1], dtype=float)
                u_len = float(_np.linalg.norm(u_hkl))
                v_len = float(_np.linalg.norm(v_hkl))
                u_hkl = u_hkl / u_len
                v_hkl = v_hkl / v_len

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
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),                
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
            v = np.cross(u, n_vec)
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
        pad_u = (U_max - U_min) * (-0.01)
        pad_v = (V_max - V_min) * (-0.01)
        U_min -= pad_u
        U_max += pad_u
        V_min -= pad_v
        V_max += pad_v

            # grid parameters
    #    i_size = max(U_max - U_min, 1e-6)
    #    j_size = max(V_max - V_min, 1e-6)
        H, W = (shape if (isinstance(shape, (tuple, list)) and len(shape) == 2) else (512, 512))
        H = max(int(H), 2)
        W = max(int(W), 2)
        ui = np.linspace(U_min, U_max, H)
        vj = np.linspace(V_min, V_max, W)
    
        sl_points = []
        for a in ui:
            for b in vj:
                P = origin_vec + a*u + b*v
                sl_points.append(P)

        sl_points = np.array(sl_points)

        # Create plane sized by extents and interpolate point data onto it
#        plane = pv.Plane(center=origin_vec.tolist(), direction=n_vec.tolist(),
#                         i_size=i_size, j_size=j_size, i_resolution=W, j_resolution=H)
        plane = pv.PolyData(sl_points)


        # Choose contributing cloud (slab-filtered if applicable)
        if np.any(mask) and np.any(~mask) and use_slab:
            cloud_contrib = pv.PolyData(pts_for_extent)
            cloud_contrib['intensity'] = vals_for_extent.astype('float32')
        else:
            cloud_contrib = cloud  # full cloud built earlier

        # Use smart radius calculation to minimize gaps
        optimal_radius = self._calculate_smart_radius(
            pts_for_extent, 
            (U_min, U_max), 
            (V_min, V_max), 
            (H, W)
        )

        interp_plane = plane.interpolate(
            cloud_contrib,
            radius=optimal_radius,
            sharpness=1.5,
            null_value=0.0
        )
        interp_plane.field_data['slice_normal'] = np.asarray(n_vec, dtype=float)
        interp_plane.field_data['slice_origin'] = np.asarray(origin_vec, dtype=float)
        
        # Store HKL axes for downstream use
        interp_plane.field_data['slice_u_axis'] = np.asarray(u, dtype=float)
        interp_plane.field_data['slice_v_axis'] = np.asarray(v, dtype=float)
        interp_plane.field_data['axes_dim'] = np.asarray((H,W), dtype=int)
        interp_plane.field_data['axes_extent'] = np.asarray((V_min,V_max,U_min,U_max), dtype=float)
        
        # Store HKL axis labels if available
        if u_hkl is not None:
            interp_plane.field_data['slice_u_label'] = format_hkl_axis(u_hkl)
        if v_hkl is not None:
            interp_plane.field_data['slice_v_label'] = format_hkl_axis(v_hkl)
        
        if show:
            self.show_slice(interp_plane)
        return interp_plane
    

    def show_slice(self, vol, cmap='viridis', clim=None, return_image=False, aspect=1):
        """
        Slice a 3D HKL volume or point cloud and display as 2D raster with interactive features.

        This method provides comprehensive slice visualization with automatic orientation detection,
        intensity filtering, interactive hover tooltips, and support for both volume and point cloud data.
        Includes caching for efficient line cut operations.

        Usage:
        Parameters:
            vol: Input data. Supported formats:
                - pv.ImageData with cell_data['intensity']
                - pv.PolyData slice mesh (rasterized directly)
                - NumPy ndarray (D,H,W) volume
                - (ndarray_volume, shape) tuple
                - Point cloud: (points, intensities), Data object, or dict
            cmap (str): Matplotlib colormap name for display
            clim (tuple): (vmin, vmax) intensity display limits. Overridden by min/max_intensity
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
            da.show_slice(volume)
            
            # High-contrast slice with intensity filtering
            da.show_slice(data, cmap='hot')
                       
            # Get image data for line cut analysis
            img, extent = da.show_slice(data, return_image=True)
            line_data = da.line_cut('zero', param=(0.5, 'x'), vol=(img, extent))
        """
        # Require PyVista for arbitrary oriented slicing
        if pv is None:
            raise ImportError("PyVista is required for show_slice()")

        import numpy as _np
        import matplotlib.pyplot as _plt

        normal_1 = getattr(vol, 'field_data', {}).get('slice_normal', None)
        origin_1 = getattr(vol, 'field_data', {}).get('slice_origin', None)
        u_axis_1 = getattr(vol, 'field_data', {}).get('slice_u_axis', None)
        v_axis_1 = getattr(vol, 'field_data', {}).get('slice_v_axis', None)
        u_label_1 = getattr(vol, 'field_data', {}).get('slice_u_label', None)
        v_label_1 = getattr(vol, 'field_data', {}).get('slice_v_label', None)
        dim_1 = getattr(vol, 'field_data', {}).get('axes_dim', None)
        extent_1 = getattr(vol, 'field_data', {}).get('axes_extent', None)

        pts_1 = np.asarray(getattr(vol, 'points', np.empty((0, 3))), dtype=float)
        vals_1 = np.asarray(vol['intensity'], dtype=float)


        pts_1_2d = pts_1.reshape(dim_1[0], dim_1[1], 3)
        vals_1_2d = vals_1.reshape(dim_1[0],dim_1[1])

        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(vals_1_2d, cmap=cmap,extent=extent_1,vmax=clim,aspect=aspect)
                    
        # Use HKL axis labels if available, otherwise fall back to orientation-based labels
        if u_label_1 is not None and v_label_1 is not None:
            ax.set_xlabel(str(v_label_1))
            ax.set_ylabel(str(u_label_1))
        elif u_axis_1 is not None and v_axis_1 is not None:
            ax.set_xlabel(format_hkl_axis(v_axis_1))
            ax.set_ylabel(format_hkl_axis(u_axis_1))
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
        # if orientation in ("HK", "KL", "HL") and (orth_label is not None) and (orth_value is not None) and _np.isfinite(orth_value):
        #     title = f'{orientation} plane ({orth_label} = {orth_value:.3f})'
        # else:
        title = f'{origin_1} slice'
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label='Intensity')

        img = vals_1_2d
        extent = extent_1 
        
        if return_image:
            return img, extent
        return None


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

