import sys
import numpy as np
import os.path as osp
import pyvista as pyv
from pyvistaqt import QtInteractor
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QFileDialog, 
    QErrorMessage, 
    QMessageBox, 
    QProgressDialog,
    QDialog
    )
import h5py
import os
import time
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import SizeManager
from utils.hdf5_loader import HDF5Loader
from PyQt5.QtWidgets import QSizePolicy

class HKL3DSliceWindow(QMainWindow):
    def __init__(self, parent=None):
        # use parent if none
        """Initializes the viewing window for the 3D slicer"""

        # -------- Initialize the window
        super(HKL3DSliceWindow, self).__init__()
        self.parent = parent
        uic.loadUi('gui/hkl_3d_slice_window.ui', self)
        self.setWindowTitle('3D Slice')
        pyv.set_plot_theme('dark')

        # ------ Connecting the signals to the code that will be executed
        self.btnZoomIn.clicked.connect(self.zoom_in)
        self.btnZoomOut.clicked.connect(self.zoom_out)
        self.btnResetCamera.clicked.connect(self.reset_camera)
        self.btnSetCamPos.clicked.connect(self.set_camera_position)
        
        # Toggle
        self.cbToggleSlicePointer.clicked.connect(self.toggle_pointer)
        self.cbToggleCloudVolume.clicked.connect(self.toggle_cloud_vol)
        self.cbColorMapSelect.currentIndexChanged.connect(self.change_color_map)
        
        # Data alteration (deferred; apply on Render)
        if hasattr(self, 'btnRender'):
            self.btnRender.clicked.connect(self._apply_pending_changes)
        self.cbbReductionFactor.currentIndexChanged.connect(self._mark_reduction_changed)
        self.sbMinIntensity.editingFinished.connect(self._mark_intensity_changed)
        self.sbMaxIntensity.editingFinished.connect(self._mark_intensity_changed)
        
        #TODO: these functions
        self.actionLoadData.triggered.connect(self.load_data)
        self.actionExtractSlice.triggered.connect(self.extract_slice)
        self.actionSave.triggered.connect(self.save_data)
        
        # Determines if application is standalone mode
        if self.parent: 
            self.btnUseParentData.clicked.connect(self._load_parent_data)
        else:
            # Hide the button AND TEXT BOX for standalone mode
            self.fgbLoadData.setVisible(False)
            self.btnUseParentData.setVisible(False)
            self.setWindowTitle('3D Slice -- Standalone Mode')
        
        # Variables to be used
        self.grid = None
        self.cloud_copy = None
        self.orig_shape = (0,0)
        self.curr_shape = (0,0)
        self.num_images = 0

        # Pending-change tracking for batch render
        try:
            self._applied_reduction_text = self.cbbReductionFactor.currentText()
        except Exception:
            self._applied_reduction_text = None
        try:
            self._applied_min_intensity = self.sbMinIntensity.value()
            self._applied_max_intensity = self.sbMaxIntensity.value()
        except Exception:
            self._applied_min_intensity = None
            self._applied_max_intensity = None
        self._pending_reduction = False
        self._pending_intensity = False

        # ------- Look up table & Opacity
        self.lut = pyv.LookupTable(cmap='jet')  
        self.toggle_state = {"visible": False}

        # ------- Creating the plotter for the 3d slicer
        self.plotter = QtInteractor()
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plotter.setMinimumSize(500, 500)
        
        self.viewer_3d_slicer_layout.addWidget(self.plotter,1,1)
        
        # -------------------------------------------------------------------- #


    # ********* METHODS ******* #
    # ================ RESOLUTION PROCESSING METHODS ======================= #
    def lower_res(self, points, voxel_size=1.0) -> tuple:
        """
        Lower resolution by voxelizing point cloud            self._enable_all_widgets()

        Performance: O(n log n)
        
        Args:
            points (ndarray, optional): The points of the cloud. Defaults to None.
            voxel_size (float, optional): the size of each voxel. Defaults to 1.0.

        Returns:
            unique_centers (np.ndarray): The calculated unique center points
            averaged_intensities (np.ndarray): averaged intensities
        """
        # Add empty points catch case
        if points is None or len(points) == 0:
            return np.array([]), np.array([])
        
        coords = points[:, :3]
        intensities = points[:, 3]

        voxel_indices = np.round(coords / voxel_size) * voxel_size

        unique_centers, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        summed_intensities = np.bincount(inverse_indices, weights=intensities)
        point_counts = np.bincount(inverse_indices)

        averaged_intensities = summed_intensities / point_counts
        
        return unique_centers, averaged_intensities

    def calculate_voxel_size(self, cloud, reduction_factor=2):
        """
        Calculate voxel size based on desired reduction factor
        
        Args:
            cloud: Point cloud data
            reduction_factor: Factor by which to reduce resolution (2 = half points, 4 = quarter points, etc.)
        
        Returns:
            voxel_size: Calculated voxel size
        """
        if hasattr(cloud, 'points'):
            # It's a PyVista object, get the points
            cloud_points = cloud.points
        else:
            # It's already a numpy array
            cloud_points = cloud     
        
        # Get original image resolution
        original_res_x = self.orig_shape[0]
        original_res_y = self.orig_shape[1]
        
        # Calculate target resolution
        target_res_x = original_res_x // reduction_factor
        target_res_y = original_res_y // reduction_factor
        self.curr_shape = (target_res_x, target_res_y)
        
        # Get physical bounds of the data
        data_range = np.ptp(cloud_points, axis=0)  # [x_range, y_range, z_range]
        
        # Calculate voxel size needed to achieve target resolution
        voxel_size_x = data_range[0] / target_res_x
        voxel_size_y = data_range[1] / target_res_y
        
        # Use the larger voxel size to ensure we don't exceed target resolution
        suggested_voxel_size = max(voxel_size_x, voxel_size_y)
        return suggested_voxel_size

    def calculate_auto_reduction_factor(self, total_points, target_points=500_000, tolerance=0):
        """
        Calculate reduction factor so total point count is <= target_points (default 500k).
        If total_points <= target_points + tolerance, returns 1.0 (no reduction).
        Assumes reduction acts primarily along two in-plane axes so points scale ~ factor^2.
        """
        try:
            if total_points is None:
                return 1.0
            if total_points <= (target_points + tolerance):
                return 1.0
            # reduction along two dimensions -> points scale ~ factor^2
            factor = float(np.sqrt(float(total_points) / float(target_points)))
            # Guard against pathological values
            if not np.isfinite(factor) or factor < 1.0:
                return 1.0
            return factor
        except Exception:
            return 1.0

    def process_3d_to_lower_res(self, cloud, intensity, reduction_factor=2):
        """Function that lowers the resolutions"""
        # Extract points from PyVista object
        if hasattr(cloud, 'points'):
            cloud_points = cloud.points  # Get numpy array of points
        else:
            cloud_points = np.asarray(cloud)
        start_time = time.time()
        suggested_voxel_size = self.calculate_voxel_size(cloud_points, reduction_factor=reduction_factor)
        cloud_with_intensities = np.concatenate([cloud_points, intensity.reshape(-1, 1)], axis=1)
        voxel_centers, final_intensities = self.lower_res(points=cloud_with_intensities, voxel_size=suggested_voxel_size)
        self.lbRenderTimeNum.setText(f"{time.time() - start_time:.4f}")
        try:
            self._last_reduction_factor = reduction_factor
        except Exception:
            self._last_reduction_factor = None
        return voxel_centers, final_intensities




    def _compute_adaptive_cells(self, total_points: int) -> tuple:
        """
        Compute adaptive grid cell counts for single-pass rendering based on point count.
        Returns (preview_cells, refine_cells), but only refine_cells is used in single-pass mode.
        Aggressively lower grid resolution to speed up rendering (down to ~100s).
        """
        try:
            if total_points >= 5_000_000:
                return 80, 120
            elif total_points >= 2_000_000:
                return 100, 140
            elif total_points >= 500_000:
                return 120, 160
            else:
                return 160, 180
        except Exception:
            return 160, 180

    
    

    def reduction_factor(self):
        """Handle reduction factor changes"""
        try:
            reduction_factor_text = self.cbbReductionFactor.currentText()
            import re
            match = re.search(r'x(\d+(?:\.\d+)?)', reduction_factor_text)
            manual_factor = float(match.group(1)) if match else 1.0
            # Enforce auto cap at 500k points
            try:
                total_points = self.cloud_copy.n_points if hasattr(self.cloud_copy, 'n_points') else (len(self.cloud_copy) if hasattr(self.cloud_copy, '__len__') else 0)
            except Exception:
                total_points = 0
            auto_factor = self.calculate_auto_reduction_factor(total_points)
            factor = max(manual_factor, auto_factor)
            cloud, intensity = self.process_3d_to_lower_res(self.cloud_copy, self.cloud_copy['intensity'], reduction_factor=factor)
            self.create_3D(cloud=cloud, intensity=intensity)
            
        except Exception as e:
            import traceback
            with open('error_output2.txt','w') as f:
                f.write(f"Traceback:\n{traceback.format_exc()}\n\nError:\n{str(e)}")


    # ================ DATA SETUP METHODS ======================= #
    def setup_3d_cloud(self, cloud, intensity, shape):
        """Sets the cloud and intensity that will be used"""
        
        # Handle case where no cloud data is provided
        if cloud is None or (isinstance(cloud, np.ndarray) and cloud.size == 0):
            # Check if parent exists and has the required data
            if (self.parent and 
                hasattr(self.parent, 'cloud') and 
                self.parent.cloud is not None):
                self.cloud_copy = self.parent.cloud.copy(deep=True)
                self.cloud_copy['intensity'] = self.parent.cloud['intensity']
                
            else:
                # No parent data available and no cloud provided
                
                self.cloud_copy = None
        else:
            # We have valid cloud data, process it
            if isinstance(cloud, np.ndarray):
                self.cloud_copy = pyv.PolyData(cloud)
                self.cloud_copy['intensity'] = intensity
            else:
                # Already PyVista object
                self.cloud_copy = cloud.copy(deep=True) if hasattr(cloud, 'copy') else cloud
                self.cloud_copy['intensity'] = intensity
        self.orig_shape = shape      
        return True

    # ================ 3D VISUALIZATION METHODS ======================= #
    def create_3D(self, cloud=None, intensity=None):
        self.plotter.clear()
        # ------ Initialize the cloud ------ #
        
        self.cloud_mesh = pyv.PolyData(cloud)
        self.cloud_mesh['intensity'] = intensity
        
        # ----- Adding the point cloud data (commented out for performance) ----- #
        # self.plotter.add_mesh(
        #     self.cloud_mesh,
        #     scalars="intensity",
        #     cmap=self.lut,
        #     point_size=5.0,
        #     name="points",
        #     reset_camera=False,
        #     nan_color=None,
        #     nan_opacity=0.0,
        #     show_scalar_bar=True,
        # )
        
        # # Make points invisible by default for performance
        # self.plotter.renderer._actors["points"].SetVisibility(False)
        
        # -------- Reset Camera Focus & Plotter Bounds ----- #
        self.plotter.set_focus(self.cloud_mesh.center)
        self.plotter.reset_camera()

        # --------- Create grid  --------- #
        min_bounds = self.cloud_mesh.points.min(axis=0)
        max_bounds = self.cloud_mesh.points.max(axis=0)
        data_range = max_bounds - min_bounds
        padding_scale = 0.1
        padding = data_range * padding_scale  # Use data_range, not (max_bounds - min_bounds)
        grid_min = min_bounds - padding
        grid_max = max_bounds + padding
        grid_range = grid_max - grid_min

        # Adaptive grid based on point count (single-pass)
        total_points = int(len(cloud))
        _, refine_cells = self._compute_adaptive_cells(total_points)

        # Build full grid spacing/dimensions
        spacing_per_axis = grid_range / refine_cells
        dimensions = np.ceil(grid_range / spacing_per_axis).astype(int) + 1

        # create and set grid dims with NON-UNIFORM spacing
        self.grid = pyv.ImageData()
        self.grid.origin = grid_min
        self.grid.spacing = spacing_per_axis
        self.grid.dimensions = dimensions
        
        # Calculate optimal radius based on grid spacing
        grid_spacing = np.array(self.grid.spacing)
        optimal_radius = np.mean(grid_spacing) * 2.5  # 2.5x average spacing

        self.vol = self.grid.interpolate(
            self.cloud_mesh, 
            radius=optimal_radius,
            sharpness=1.5,
            null_value=0.0
        )
        
        self.plotter.add_volume(
            volume=self.vol,
            scalars="intensity",
            name="cloud_volume"
        )
        
        self.plotter.renderer._actors["cloud_volume"].SetVisibility(False)

        self.plotter.show_bounds(
            mesh=self.cloud_mesh, 
            xtitle='H Axis', 
            ytitle='K Axis', 
            ztitle='L Axis', 
            bounds=self.cloud_mesh.bounds
        )
        
        cube_axes_actor = self.plotter.renderer.cube_axes_actor
        if cube_axes_actor:
            # Set X axis (H) to red
            cube_axes_actor.GetXAxesLinesProperty().SetColor(1.0, 0.0, 0.0)
            cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
            cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
            
            # Set Y axis (K) to green
            cube_axes_actor.GetYAxesLinesProperty().SetColor(0.0, 1.0, 0.0)
            cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
            cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
            
            # Set Z axis (L) to blue
            cube_axes_actor.GetZAxesLinesProperty().SetColor(0.0, 0.0, 1.0)
            cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
            cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

        # ------- Adding the initial slice -------- #
        slice_normal = (0, 0, 1)
        slice_origin = self.cloud_mesh.center
        vol_slice = self.vol.slice(normal=slice_normal, origin=slice_origin)

        # Check if slice has data before adding
        if vol_slice.n_points > 0:
            self.plotter.add_mesh(vol_slice, scalars="intensity", name="slice", show_edges=False, reset_camera=False)
        else:
            pass

        # ---- Add the interactive slicing plane widget
        self.plotter.add_plane_widget(
            callback=self.update_slice_from_plane,
            normal=slice_normal,
            origin=slice_origin,
            bounds=self.vol.bounds,
            factor=1.0,
            implicit=True,
            assign_to_axis=None,
            tubing=False,
            origin_translation=True,
        )

        # -------- Adding the arrow -------------- #
        sphere = pyv.Sphere(radius=0.1, center=slice_origin)
        line = pyv.Line(
            pointa=slice_origin,
            pointb=np.array(slice_origin) + np.array(slice_normal) * (np.min(data_range))
        )
        self.plotter.add_mesh(sphere, color='red', name="origin_sphere", pickable=False)
        self.plotter.add_mesh(line, color='red', name="normal_line", pickable=False)

        # Hide them manually after adding
        self.plotter.renderer._actors["origin_sphere"].SetVisibility(False)
        self.plotter.renderer._actors["normal_line"].SetVisibility(False)

        # Store toggle visibility state
        self.toggle_state = {"visible": False}
        
        # --------- Adjust Labels -------------- #
        self.lbCurrentPointSizeNum.setText(str(len(cloud)))
        self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
        self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))
        intensity_range = int(np.min(intensity)), int(np.max(intensity))
        self.sbMinIntensity.setRange(*intensity_range)
        self.sbMinIntensity.setValue(int(np.min(intensity)))
        self.sbMaxIntensity.setRange(*intensity_range)
        self.sbMaxIntensity.setValue(int(np.max(intensity)))

    def update_intensity(self):
        """Updates the min/max intensity levels and scalar bar range"""
        self.min_intensity = self.sbMinIntensity.value()
        self.max_intensity = self.sbMaxIntensity.value()
        
        if self.min_intensity > self.max_intensity:
            self.min_intensity, self.max_intensity = self.max_intensity, self.min_intensity
            self.sbMinIntensity.setValue(self.min_intensity)
            self.sbMaxIntensity.setValue(self.max_intensity)
        
        # Define the new scalar range
        new_range = [self.min_intensity, self.max_intensity]

        # Update the volume actor by re-adding with new clim range
        if hasattr(self, 'vol') and self.vol is not None:
            if "cloud_volume" in self.plotter.actors:
                self.plotter.remove_actor("cloud_volume", reset_camera=False)
            
            # Re-add volume with the new range - this updates both volume and scalar bar
            self.plotter.add_volume(
                volume=self.vol,
                scalars="intensity",
                name="cloud_volume",
                clim=(new_range[0], new_range[1]),  # This sets both volume AND scalar bar range
                show_scalar_bar=True
            )
            
            # Maintain volume visibility state
            if hasattr(self, 'cbToggleCloudVolume') and self.cbToggleCloudVolume.isChecked():
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)
            else:
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)

        # Force update of all scalar bars with the new range
        if hasattr(self.plotter, 'scalar_bars'):
            for name, scalar_bar in self.plotter.scalar_bars.items():
                if scalar_bar:
                    scalar_bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                    scalar_bar.Modified()

        # Update slice actor scalar range if it exists
        if "slice" in self.plotter.actors:
            slice_actor = self.plotter.actors["slice"]
            if hasattr(slice_actor, 'mapper'):
                try:
                    slice_actor.mapper.scalar_range = (new_range[0], new_range[1])
                except Exception:
                    pass

        # Force a re-render to apply the changes
        self.plotter.render()
        # avoid the cloud volume reapearing after change intneisity limits
        self.toggle_cloud_vol()
                    
    # ================ RENDER APPLY METHODS ======================= #
    def _mark_reduction_changed(self):
        try:
            self._pending_reduction = True
        except Exception:
            self._pending_reduction = True

    def _mark_intensity_changed(self):
        try:
            self._pending_intensity = True
        except Exception:
            self._pending_intensity = True

    def _apply_pending_changes(self):
        """
        Apply pending UI changes in a single pass.
        Executes in order:
         - reduction_factor() if reduction changed
         - update_intensity() if intensity range changed
        """
        try:
            changed_any = False

            # Check reduction factor change
            # current_reduction_text = None
            # try:
            #     current_reduction_text = self.cbbReductionFactor.currentText()
            # except Exception:
            #     pass

            # if getattr(self, "_pending_reduction", False) or (current_reduction_text is not None and current_reduction_text != getattr(self, "_applied_reduction_text", None)):
            #     if getattr(self, "cloud_copy", None) is not None:
            #         self.reduction_factor()
            #         self._applied_reduction_text = current_reduction_text
            #         changed_any = True
            #     self._pending_reduction = False

            # Check intensity range change
            cur_min = None
            cur_max = None
            try:
                cur_min = self.sbMinIntensity.value()
                cur_max = self.sbMaxIntensity.value()
            except Exception:
                pass

            if getattr(self, "_pending_intensity", False) or (cur_min is not None and cur_max is not None and (cur_min != getattr(self, "_applied_min_intensity", None) or cur_max != getattr(self, "_applied_max_intensity", None))):
                if getattr(self, "vol", None) is not None:
                    self.update_intensity()
                    self._applied_min_intensity = cur_min
                    self._applied_max_intensity = cur_max
                    changed_any = True
                self._pending_intensity = False

            if changed_any and hasattr(self, "plotter") and self.plotter is not None:
                self.plotter.render()
        except Exception:
            pass

    # ================ CALLBACK METHODS ======================= #
    def update_slice_from_plane(self, normal, origin):
        """Callback for updating the slice and vector visuals"""
        # Update slice
        new_slice = self.vol.slice(normal=normal, origin=origin)
        focused_range = [self.min_intensity, self.max_intensity]
        # Add slice without modifying scalar ranges
        self.plotter.add_mesh(
            new_slice,
            scalars="intensity",
            cmap=self.lut,
            clim=focused_range,
            name="slice",
            show_edges=False,
            reset_camera=False,
            pickable=False
        )
        self.update_intensity() # Keeps the intensity limits
        # Remove previous origin and line
        self.plotter.remove_actor("origin_sphere", reset_camera=False)
        self.plotter.remove_actor("normal_line", reset_camera=False)

        # Create new visuals
        sphere_radius = (np.min(self.cloud_mesh.points.max(axis=0)-self.cloud_mesh.points.min(axis=0)))*0.002
        new_sphere = pyv.Sphere(radius=sphere_radius, center=origin)
        new_line = pyv.Line(pointa=origin, pointb=np.array(origin) + np.array(normal) * (np.min(self.cloud_mesh.points.max(axis=0)-self.cloud_mesh.points.min(axis=0))))

        self.plotter.add_mesh(new_sphere, color='red', name="origin_sphere", pickable=False)
        self.plotter.add_mesh(new_line, color='red', name="normal_line", pickable=False)

        # Set their visibility based on toggle state
        self.plotter.renderer._actors["origin_sphere"].SetVisibility(self.toggle_state["visible"])
        self.plotter.renderer._actors["normal_line"].SetVisibility(self.toggle_state["visible"])
        

    # ================ DATA LOADING ======================= #
    def render_changes(self):
        """
        Render the changes to the 
        """
        self.update_intensity()
        

    def _load_parent_data(self):
        """Check if parent has the required data available"""
        if hasattr(self.parent, 'cloud') and self.parent.cloud is not None and hasattr(self.parent, 'reader') and self.parent.reader is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.setEnabled(False)
            original_title = self.windowTitle()
            
            # Set loading state BEFORE processing events
            self.setEnabled(False)
            self.setWindowTitle(f"{original_title} ***** Loading...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            
            try:
                self.setup_3d_cloud(cloud=self.parent.cloud.copy(deep=True), intensity=self.parent.cloud['intensity'], shape=self.parent.reader.shape)
                # Auto reduction: target ~15M points if over threshold
                try:
                    total_points = self.cloud_copy.n_points if hasattr(self.cloud_copy, 'n_points') else (len(self.cloud_copy) if hasattr(self.cloud_copy, '__len__') else 0)
                except Exception:
                    total_points = 0
                auto_factor = self.calculate_auto_reduction_factor(total_points)
                reduction = auto_factor if auto_factor > 1.0 else 1.0
                cloud, intensity = self.process_3d_to_lower_res(self.cloud_copy, self.cloud_copy['intensity'], reduction_factor=reduction)
                self.create_3D(cloud=cloud, intensity=intensity)
                self.num_images = len(self.parent.reader.cache_images) if hasattr(self.parent.reader, 'cache_images') and self.parent.reader.cache_images is not None else 0
                self.groupBox3DViewer.setTitle(f'Viewing {len(self.parent.reader.cache_images)} Image(s)')
                self.lbOriginalPointSizeNum.setText(str(self.cloud_copy.n_points))
                self.lbOriginalResolutionX.setText(str(self.parent.reader.shape[0]))
                self.lbOriginalResolutionY.setText(str(self.parent.reader.shape[1]))
                intensity_range = int(np.min(intensity)), int(np.max(intensity))
                self.sbMinIntensity.setRange(*intensity_range)
                self.sbMinIntensity.setValue(int(np.min(intensity)))
                self.sbMaxIntensity.setRange(*intensity_range)
                self.sbMaxIntensity.setValue(int(np.max(intensity)))
            finally:
                # Always restore state
                QApplication.restoreOverrideCursor()
                self.setEnabled(True)
                self.setWindowTitle(original_title)
            
    
    def load_data(self):
        """Load data from HD5 file"""
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select an HD5 File','','HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not file_name:
            QMessageBox.warning(self, "File", "No Valid File Selected")
            return
        
        # Set the file path string when a file is selected
        self.leFilePathStr.setText(file_name)
        
        # Store original state
        original_title = self.windowTitle()
        
        # Set loading state BEFORE processing events
        self.setEnabled(False)
        self.setWindowTitle(f"{original_title} ***** Loading...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        
        try:
            from utils import HDF5Loader
            loader = HDF5Loader()
            # Detect file type via metadata
            info = loader.get_file_info(file_name)
            dt_raw = info.get('data_type', '')
            # Normalize possible bytes/numpy scalar types from HDF5 attrs
            if isinstance(dt_raw, (bytes, bytearray)):
                dt_norm = dt_raw.decode('utf-8', errors='ignore')
            elif isinstance(dt_raw, (np.generic,)):
                try:
                    dt_norm = str(dt_raw.item())
                except Exception:
                    dt_norm = str(dt_raw)
            else:
                dt_norm = str(dt_raw)
            # Fallback to metadata dataset 'data_type' if entry attr missing
            if not dt_norm or dt_norm == '':
                dt_meta = info.get('metadata', {}).get('data_type', '')
                if isinstance(dt_meta, (bytes, bytearray)):
                    dt_norm = dt_meta.decode('utf-8', errors='ignore')
                else:
                    dt_norm = str(dt_meta)
            dt = dt_norm.lower()
            if dt == 'volume':
                # Load saved volume and render directly without point cloud interpolation
                volume, vol_shape = loader.load_h5_volume_3d(file_name)
                if volume.size == 0:
                    QMessageBox.warning(self, "Loading Warning", f"No volume data found in HDF5 file\nError: {loader.last_error}")
                    return
                
                # Build an ImageData grid from the loaded volume
                grid = pyv.ImageData()
                dims_cells = np.array(volume.shape, dtype=int)  # (D, H, W)
                # For cell_data, dimensions must be number of points = cells + 1
                grid.dimensions = (dims_cells + 1).tolist()
                
                meta = getattr(loader, 'file_metadata', {})
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
                
                # Assign intensity scalars to cell_data
                grid.cell_data["intensity"] = volume.flatten(order="F")
                
                # Store and render
                self.grid = grid
                self.vol = grid
                self.plotter.clear()
                
                # Focus and camera
                try:
                    self.plotter.set_focus(self.grid.center)
                except Exception:
                    pass
                self.plotter.add_volume(volume=self.vol, scalars="intensity", name="cloud_volume")
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)
                self.plotter.reset_camera()
                
                # Show bounds like in point-cloud path
                try:
                    self.plotter.show_bounds(
                        mesh=self.grid,
                        xtitle='H Axis',
                        ytitle='K Axis',
                        ztitle='L Axis',
                        bounds=self.grid.bounds
                    )
                except Exception:
                    pass
                
                # Add initial slice and interactive plane widget, mirroring create_3D behavior
                try:
                    slice_normal = (0, 0, 1)
                    slice_origin = getattr(self.grid, 'center', (0.0, 0.0, 0.0))
                    vol_slice = self.vol.slice(normal=slice_normal, origin=slice_origin)
                    if vol_slice.n_points > 0:
                        self.plotter.add_mesh(
                            vol_slice,
                            scalars="intensity",
                            name="slice",
                            show_edges=False,
                            reset_camera=False
                        )
                    # Plane widget
                    self.plotter.add_plane_widget(
                        callback=self.update_slice_from_plane,
                        normal=slice_normal,
                        origin=slice_origin,
                        bounds=self.vol.bounds,
                        factor=1.0,
                        implicit=True,
                        assign_to_axis=None,
                        tubing=False,
                        origin_translation=True,
                    )
                    # Arrow visuals (sphere at origin, line along normal)
                    b = self.grid.bounds
                    min_bounds = np.array([b[0], b[2], b[4]])
                    max_bounds = np.array([b[1], b[3], b[5]])
                    data_range = max_bounds - min_bounds
                    sphere_radius = (np.min(data_range)) * 0.002
                    sphere = pyv.Sphere(radius=sphere_radius, center=slice_origin)
                    line = pyv.Line(
                        pointa=slice_origin,
                        pointb=np.array(slice_origin) + np.array(slice_normal) * (np.min(data_range))
                    )
                    self.plotter.add_mesh(sphere, color='red', name="origin_sphere", pickable=False)
                    self.plotter.add_mesh(line, color='red', name="normal_line", pickable=False)
                    # Hide arrow visuals by default
                    self.plotter.renderer._actors["origin_sphere"].SetVisibility(False)
                    self.plotter.renderer._actors["normal_line"].SetVisibility(False)
                    # Track toggle state consistently
                    self.toggle_state = {"visible": False}
                except Exception:
                    pass
                
                # Update UI based on metadata
                num_images_meta = meta.get('num_images', 1)
                orig_shape_meta = meta.get('original_shape', [0, 0])
                self.groupBox3DViewer.setTitle(f'Viewing {num_images_meta} Image(s)')
                self.lbOriginalPointSizeNum.setText(str(volume.size))
                try:
                    self.lbOriginalResolutionX.setText(str(int(orig_shape_meta[0])))
                    self.lbOriginalResolutionY.setText(str(int(orig_shape_meta[1])))
                except Exception:
                    pass
                # Current resolution from volume H,W
                try:
                    self.lbCurrentResolutionX.setText(str(int(dims_cells[2])))
                    self.lbCurrentResolutionY.setText(str(int(dims_cells[1])))
                except Exception:
                    pass
                # Intensity range
                try:
                    imin = int(float(np.min(volume)))
                    imax = int(float(np.max(volume)))
                    self.sbMinIntensity.setRange(imin, imax)
                    self.sbMinIntensity.setValue(imin)
                    self.sbMaxIntensity.setRange(imin, imax)
                    self.sbMaxIntensity.setValue(imax)
                except Exception:
                    pass
                
                # Done loading volume
                return
            
            # Fallback: load as 3D points (legacy files)
            points, intensities, num_images, shape = loader.load_h5_to_3d(file_name)
            
            # Check if we got valid data
            print(f"Loader returned: points.shape={points.shape}, intensities.shape={intensities.shape}")
            print(f"Points size check: {points.size}, empty check: {points.size == 0}")
            
            # Check if we got valid data
            if points.size == 0 or intensities.size == 0:
                print("ERROR: No valid data found")
                print(f"Points size: {points.size}, Intensities size: {intensities.size}")
                QMessageBox.warning(self, "Loading Warning", f"No valid data found in HDF5 file\nError: {loader.last_error}")
                return
                
            # Setup and process
            if self.setup_3d_cloud(cloud=points, intensity=intensities, shape=shape):
                self.orig_shape = shape
                self.num_images = num_images
                reduction_factor_text = self.cbbReductionFactor.currentText()
                import re
                match = re.search(r'x(\d+(?:\.\d+)?)', reduction_factor_text)
                manual_factor = float(match.group(1)) if match else 1.0
                # Auto reduction to <=500k points if over threshold
                # auto_factor = self.calculate_auto_reduction_factor(len(points))
                factor = max(manual_factor, 1.0) #auto_factor
                cloud, intensity = self.process_3d_to_lower_res(cloud=points, intensity=intensities, reduction_factor=factor)
                self.create_3D(cloud=cloud, intensity=intensity)
                
                # Update UI
                self.groupBox3DViewer.setTitle(f'Viewing {num_images} Image(s)')
                self.lbOriginalPointSizeNum.setText(str(len(points)))
                self.lbOriginalResolutionX.setText(str(shape[0]))
                self.lbOriginalResolutionY.setText(str(shape[1]))
                
                print(f"Successfully loaded {len(points)} points from {num_images} images")
            else:
                QMessageBox.warning(self, "Setup Error", "Failed to setup 3D cloud")
            
            # Update UI labels
            self.groupBox3DViewer.setTitle(f'Viewing {num_images} Image(s)')
            self.lbOriginalPointSizeNum.setText(str(len(points)))
            self.lbOriginalResolutionX.setText(str(shape[0]))
            self.lbOriginalResolutionY.setText(str(shape[1]))
            
        except ImportError:
            QMessageBox.critical(self, "Import Error", "Could not import HDF5Loader utility")
        except Exception as e:
            import traceback
            error_msg = f"Error loading data: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error Loading Data", f"Failed to load H5 file:\n{str(e)}\n\n\n{error_msg}")
        finally:
            # Always restore state
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self.setWindowTitle(original_title)
            
    def save_data(self):
        """Save the entire 3D data when parent is present"""
        print("Saving Volume")
        try:
            # In volume-only mode, allow saving without parent; require self.vol
            # Parent metadata will be included if available
            
            # Get save file path from user
            default_name = f"vol_data_{np.datetime64('now').astype('datetime64[s]').astype(str).replace(':', '-')}.h5"
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save 3D Data", 
                default_name,
                "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # Store original state and set loading state BEFORE processing events
            original_title = self.windowTitle()
            self.setEnabled(False)
            self.setWindowTitle(f"{original_title} ***** Saving...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            
            # Ensure we have a volume to save
            if not hasattr(self, 'vol') or self.vol is None:
                QMessageBox.warning(self, "No Volume", "No volume available to save")
                return
            
            # Extract volume scalars (intensity) and reshape to grid dimensions
            try:
                # Prefer named 'intensity' array
                try:
                    volume_scalars = np.array(self.vol['intensity'])
                except KeyError:
                    # Fallback to first point_data array if available
                    arr_names = list(getattr(self.vol, 'point_data', {}).keys()) if hasattr(self.vol, 'point_data') else []
                    volume_scalars = np.array(self.vol.point_data[arr_names[0]]) if arr_names else None
                if volume_scalars is None or volume_scalars.size == 0:
                    raise ValueError("Volume intensity scalars not found")
                
                # Determine dimensions from volume or original grid
                if hasattr(self.vol, 'dimensions'):
                    dims = tuple(int(x) for x in self.vol.dimensions)
                elif hasattr(self, 'grid') and self.grid is not None and hasattr(self.grid, 'dimensions'):
                    dims = tuple(int(x) for x in self.grid.dimensions)
                else:
                    raise ValueError("Grid dimensions not available")
                
                # Reshape using Fortran order to match VTK/PyVista point ordering
                vol_array = volume_scalars.reshape(dims, order='F').astype(np.float32)
            except Exception as e:
                QMessageBox.critical(self, "Volume Error", f"Failed to assemble volume array:\n{str(e)}")
                return
            
            # Prepare volume metadata
            # Use locally stored num_images/orig_shape to support standalone mode
            # Fallbacks: num_images -> 1, original_shape -> derived from volume array for 3D (H,W)
            metadata = {
                'data_type': 'volume',
                'num_images': int(self.num_images) if hasattr(self, 'num_images') and self.num_images else 1,
                'original_shape': list(self.orig_shape) if hasattr(self, 'orig_shape') and isinstance(self.orig_shape, (tuple, list)) and any(self.orig_shape) else (0, 0),
                'source': 'DashPVA 3D Slice Window',
                'parent_channel': getattr(self.parent.reader, 'input_channel', 'unknown') if hasattr(self.parent, 'reader') else 'unknown',
                'extraction_timestamp': str(np.datetime64('now')),
                'volume_shape': list(vol_array.shape),
                'voxel_spacing': list(map(float, self.vol.spacing if hasattr(self.vol, 'spacing') else (self.grid.spacing if hasattr(self, 'grid') and self.grid is not None else (1.0, 1.0, 1.0)))),
                'grid_origin': list(map(float, self.vol.origin if hasattr(self.vol, 'origin') else (self.grid.origin if hasattr(self, 'grid') and self.grid is not None else (0.0, 0.0, 0.0)))),
                'intensity_range': [float(np.min(vol_array)), float(np.max(vol_array))]
            }
            
            # Add additional metadata if available
            if hasattr(self.parent, 'reader') and hasattr(self.parent.reader, 'pv_attributes') and self.parent.reader.pv_attributes:
                metadata['pv_attributes_keys'] = list(self.parent.reader.pv_attributes.keys())
            if hasattr(self.parent, 'reader') and hasattr(self.parent.reader, 'frames_received'):
                metadata['frames_received'] = self.parent.reader.frames_received
                metadata['frames_missed'] = getattr(self.parent.reader, 'frames_missed', 0)
            
            # Use HDF5Loader to save volume
            loader = HDF5Loader()
            success = loader.save_vol_to_h5(
                file_path=file_path,
                volume=vol_array,
                metadata=metadata
            )
            
            if success:
                QMessageBox.information(self, "Success", f"Volume saved successfully!\nShape: {vol_array.shape}\nFile:\n{file_path}")
            else:
                QMessageBox.critical(self, "Error", f"Failed to save volume: {loader.last_error}")
                
        except Exception as e:
            import traceback
            error_msg = f"Error saving data: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Save Error", f"Error saving data:\n{str(e)}")
            # Write detailed error to file for debugging
            with open('save_error_output.txt', 'w') as f:
                f.write(error_msg)
        finally:
            # Always restore state
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self.setWindowTitle(original_title)
    
    def extract_slice_from_multiple_files(self):
        pass
    
    def extract_slice(self):
        """
        Extract the current slice data from the 3D viewer and save it using HDF5Loader
        """
        try:
            default_name = f"slice_extract_{np.datetime64('now').astype('datetime64[s]').astype(str).replace(':', '-')}.h5"
        
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                f"Save hkl Slice Data", 
                default_name,
                "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
            )
            # Check if we have a current slice
            if not hasattr(self, 'plotter') or 'slice' not in self.plotter.actors:
                QMessageBox.warning(self, "No Slice", "No slice data available to extract")
                return
            
            # Get the current slice mesh
            slice_mesh:pyv.Actor = self.plotter.actors['slice'].GetMapper().GetInput()
            
            # Extract slice points and intensities
            slice_points = np.array(slice_mesh.points) # 3D coordinates of slice
            slice_intensities = np.array(slice_mesh['intensity'])  # Intensity values
            
            # Get current plane widget information
            if self.plotter.plane_widgets:
                plane_widget = self.plotter.plane_widgets[0]
                slice_normal = plane_widget.GetNormal()
                slice_origin = plane_widget.GetOrigin()
            else:
                slice_normal = (0, 0, 1)
                slice_origin = (0, 0, 0)
            
            # Prepare slice metadata
            slice_metadata = {
                'data_type': 'slice',
                'slice_normal': slice_normal,
                'slice_origin': slice_origin,
                'num_points': len(slice_points),
                'original_file': getattr(self, 'current_file_path', 'unknown'),
                'original_shape': getattr(self, 'orig_shape', (0, 0)),
                'extraction_timestamp': str(np.datetime64('now')),
                'volume_bounds': self.vol.bounds if hasattr(self, 'vol') else None
            }
            # Use HDF5Loader to save the slice have a dialog for where to save
            # should also use the same save as h5 so that it can be used in 3d or 2d
            from utils.hdf5_loader import HDF5Loader
            loader = HDF5Loader()
            
            success = loader.save_3d_to_h5(
                file_path=file_path,
                points=slice_points,
                intensities=slice_intensities,
                metadata=slice_metadata
            )
            
            if success:
                QMessageBox.information(self, "Success", f"Slice extracted and saved successfully!\n{len(slice_points)} points saved.")
            else:
                QMessageBox.critical(self, "Error", f"Failed to save slice: {loader.get_last_error()}")
                
        except Exception as e:
            QMessageBox.critical(self, "Extract Error", f"Error extracting slice:\n{str(e)}")

            

    # ================ CAMERA CONTROL METHODS ======================= #
    def zoom_in(self):
        """Zoom camera in"""
        camera = self.plotter.camera
        camera.zoom(1.5)
        self.plotter.render()

    def zoom_out(self):
        """Zoom camera out"""
        camera = self.plotter.camera
        camera.zoom(0.5)
        self.plotter.render()

    def reset_camera(self):
        """Reset camera to default position"""
        self.plotter.reset_camera()
        self.plotter.render()

    def set_camera_position(self):
        """Set camera to predefined positions"""
        pos = self.cbSetCamPos.currentText().lower()
        if 'xy' in pos:
            self.plotter.view_xy()   
        elif 'yz' in pos:
            self.plotter.view_yz()   
        elif 'xz' in pos:
            self.plotter.view_xz()   
        self.plotter.render()

    # ================ VISIBILITY TOGGLE METHODS ======================= #
    def toggle_pointer(self):
        """Toggle visibility of the slice pointer (origin sphere and normal line)"""
        vis = self.cbToggleSlicePointer.isChecked()
        self.toggle_state["visible"] = vis
        try:
            self.plotter.renderer._actors["origin_sphere"].SetVisibility(vis)
        except Exception:
            pass
        try:
            self.plotter.renderer._actors["normal_line"].SetVisibility(vis)
        except Exception:
            pass
        self.plotter.render()
        
    def toggle_cloud_vol(self):
        """Toggle visibility of the cloud volume"""
        is_visible = self.cbToggleCloudVolume.isChecked()
        self.plotter.renderer._actors["cloud_volume"].SetVisibility(is_visible)

    def change_color_map(self):
        color_map_select = self.cbColorMapSelect.currentText()
        # Create new lookup table
        new_lut = pyv.LookupTable(cmap=color_map_select)
        self.lut = new_lut
        
        # Update all actors in plotter.actors (PyVista managed) - commented out for performance
        # for actor_name, actor in self.plotter.actors.items():
        #     if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'lookup_table'):
        #         actor.mapper.lookup_table = new_lut
        #         actor.mapper.update()
        
        # Update all actors in renderer._actors (VTK managed) - commented out for performance
        # for actor_name, actor in self.plotter.renderer._actors.items():
        #     # Check if it's a volume or other VTK actor with LUT
        #     if hasattr(actor, 'GetMapper') and actor.GetMapper():
        #         mapper = actor.GetMapper()
        #         if hasattr(mapper, 'SetLookupTable'):
        #             mapper.SetLookupTable(new_lut)
        #             mapper.Update()
        
        # Update scalar bar's lookup table directly
        if hasattr(self.plotter, 'scalar_bars'):
            for scalar_bar_name, scalar_bar_actor in self.plotter.scalar_bars.items():
                if scalar_bar_actor:
                    
                    scalar_bar_actor.SetLookupTable(new_lut)
                    scalar_bar_actor.Modified()
        # Force render
        self.plotter.render()

    def set_volume_scalar_bar_range(self, min_val, max_val):
        """Modify the existing volume scalar bar range"""
        # Use the existing UI controls and update method
        self.sbMinIntensity.setValue(min_val)
        self.sbMaxIntensity.setValue(max_val)
        self.update_intensity()  # This properly updates the volume scalar bar

    def set_volume_scalar_bar_range_direct(self, min_val, max_val):
        """Directly modify volume scalar bar by re-adding with clim"""
        if hasattr(self, 'vol') and self.vol is not None:
            # Remove existing volume
            if "cloud_volume" in self.plotter.actors:
                self.plotter.remove_actor("cloud_volume", reset_camera=False)
            
            # Re-add with custom range
            self.plotter.add_volume(
                volume=self.vol,
                scalars="intensity",
                name="cloud_volume",
                clim=(min_val, max_val)  # This sets the scalar bar range
            )
            
            # Maintain visibility state
            if hasattr(self, 'cbToggleCloudVolume') and self.cbToggleCloudVolume.isChecked():
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)
            else:
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)  # Default visible

    def add_custom_scalar_bar_below(self, min_val, max_val, title="Custom Range"):
        """Add a custom scalar bar positioned below the existing one"""
        
        # Create custom lookup table
        custom_lut = pyv.LookupTable(cmap='viridis')  # Match existing colormap
        custom_lut.scalar_range = (min_val, max_val)
        
        # Add scalar bar positioned below the existing one
        self.plotter.add_scalar_bar(
            title=title,
            position_x=0.85,    # Same X as existing (right side)
            position_y=0.02,    # Lower Y position (below existing)
            width=0.12,         # Same width as existing
            height=0.3,         # Shorter height to fit below
            n_labels=5,
            fmt="%.2f",
            title_font_size=14,
            label_font_size=12,
            color='white'       # Match dark theme
        )
        
        # Configure the new scalar bar
        if hasattr(self.plotter, 'scalar_bars'):
            # Get the newly added scalar bar (last one)
            scalar_bar_keys = list(self.plotter.scalar_bars.keys())
            if scalar_bar_keys:
                custom_scalar_bar = self.plotter.scalar_bars[scalar_bar_keys[-1]]
                custom_scalar_bar.SetLookupTable(custom_lut)
                custom_scalar_bar.SetRange(min_val, max_val)
                custom_scalar_bar.SetTitle(title)
                custom_scalar_bar.Modified()
        
        # Force render to show changes
        self.plotter.render()

    def setup_dual_scalar_bars(self, volume_min, volume_max, custom_min, custom_max):
        """Setup both volume scalar bar range and add custom scalar bar below"""
        
        # 1. Modify existing volume scalar bar range
        self.set_volume_scalar_bar_range(volume_min, volume_max)
        
        # 2. Add custom scalar bar below
        self.add_custom_scalar_bar_below(
            custom_min, 
            custom_max, 
            title=f"Custom [{custom_min}, {custom_max}]"
        )

    def remove_custom_scalar_bars(self):
        """Remove any custom scalar bars (keeping only the volume's original scalar bar)"""
        if hasattr(self.plotter, 'scalar_bars'):
            # Get all scalar bar keys
            scalar_bar_keys = list(self.plotter.scalar_bars.keys())
            
            # Remove all but the first one (assuming first is the volume's scalar bar)
            for i, key in enumerate(scalar_bar_keys):
                if i > 0:  # Keep the first scalar bar, remove others
                    try:
                        self.plotter.remove_scalar_bar(key)
                    except Exception:
                        pass
        
        self.plotter.render()

    def hide_all_scalar_bars(self):
        """Hide all scalar bars in the plotter"""
        if hasattr(self.plotter, 'scalar_bars'):
            for name, scalar_bar in self.plotter.scalar_bars.items():
                if scalar_bar:
                    scalar_bar.SetVisibility(False)
        self.plotter.render()

    def show_all_scalar_bars(self):
        """Show all scalar bars in the plotter"""
        if hasattr(self.plotter, 'scalar_bars'):
            for name, scalar_bar in self.plotter.scalar_bars.items():
                if scalar_bar:
                    scalar_bar.SetVisibility(True)
        self.plotter.render()

    def remove_all_scalar_bars(self):
        """Remove all scalar bars from the plotter"""
        if hasattr(self.plotter, 'scalar_bars'):
            for name in list(self.plotter.scalar_bars.keys()):
                try:
                    self.plotter.remove_scalar_bar(name)
                except Exception:
                    pass
        self.plotter.render()
            


    # ================ CLEANUP METHODS ======================= #
    def closeEvent(self, event):
        """Clean shutdown of 3D viewer without affecting parent"""
        # Clear the main troublemakers
        if hasattr(self, 'plotter'):
            self.plotter.clear()
            self.plotter.close()
            self.plotter = None
        
        # Clear data objects that hold memory
        for attr in ['cloud_copy', 'cloud_mesh', 'vol', 'grid']:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # Remove parent's reference to this window
        if hasattr(self.parent, 'slice_window'):
            self.parent.slice_window = None
        event.accept()
        
        
# add main 
if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = HKL3DSliceWindow()
        window.show()
        size_manager = SizeManager(app=app)
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)