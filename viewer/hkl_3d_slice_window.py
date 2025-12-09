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
    QDialog,
    QSizePolicy
    )
import h5py
import os
import time
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import SizeManager
from utils.hdf5_loader import HDF5Loader
from viewer.hkl_slice_2d_view import HKLSlice2DView
class HKL3DSliceWindow(QMainWindow):
    def __init__(self, parent=None):
        # use parent if none
        """Initializes the viewing window for the 3D slicer"""

        # -------- Initialize the window
        super(HKL3DSliceWindow, self).__init__()
        self.parent = parent
        uic.loadUi('gui/hkl_3d_slice_window.ui', self)
        # Initialize availability disabled until data loads
        try:
            self.actionSave.setEnabled(False)
            self.actionExtractSlice.setEnabled(False)
            self._set_slice_controls_enabled(False)
        except Exception:
            pass
        self.setWindowTitle('3D Slice')
        pyv.set_plot_theme('dark')

        # ------ Connecting menu actions to open Controls dialog
        try:
            if hasattr(self, 'actionSlice'):
                self.actionSlice.triggered.connect(lambda: self.open_controls_dialog(focus='slice'))
            # Tools -> View -> Slice
            if hasattr(self, 'actionViewSlice'):
                self.actionViewSlice.triggered.connect(self._on_view_slice_action)
        except Exception:
            pass
        
        # Toggle
        self.cbToggleSlicePointer.clicked.connect(self.toggle_pointer)
        self.cbToggleCloudVolume.clicked.connect(self.toggle_cloud_vol)
        self.cbColorMapSelect.currentIndexChanged.connect(self.change_color_map)
        
        # Slice mesh visibility toggle
        try:
            if hasattr(self, 'cbToggleSliceMesh'):
                self.cbToggleSliceMesh.clicked.connect(self.toggle_slice_mesh)
        except Exception:
            pass
        
        # Data alteration: re-enable reduction factor callback
        if hasattr(self, 'cbbReductionFactor'):
            self.cbbReductionFactor.currentIndexChanged.connect(self.reduction_factor)
        # Keep immediate intensity updates
        self.sbMinIntensity.editingFinished.connect(self.update_intensity)
        self.sbMaxIntensity.editingFinished.connect(self.update_intensity)
        
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

        # Slice control state
        self._slice_translate_step = 0.01
        self._slice_rotate_step_deg = 1.0
        # Camera control state (updated via Controls dialog)
        self._zoom_step = 1.5
        self._cam_pos_selection = None
        # Slice dialog-driven state
        self._slice_orientation_selection = None
        self._custom_normal = [0.0, 0.0, 1.0]

        # Pending-change tracking for batch render
        # Reduction factor deprecated
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
        # Setup Slice Lock overlay in-plot HUD
        try:
            self._setup_slice_lock_overlay()
        except Exception:
            pass

        # -------- Wire Slice/Camera controls defined in the .ui (no programmatic creation)
        # Slice auto-apply handlers
        if hasattr(self, 'sbSliceTranslateStep'):
            self.sbSliceTranslateStep.valueChanged.connect(self._on_translate_step_changed)
        if hasattr(self, 'sbSliceRotateStep'):
            self.sbSliceRotateStep.valueChanged.connect(self._on_rotate_step_changed)
        if hasattr(self, 'cbSliceOrientation'):
            self.cbSliceOrientation.currentIndexChanged.connect(self._on_orientation_changed)
        if hasattr(self, 'sbNormH'):
            self.sbNormH.editingFinished.connect(self._on_custom_normal_changed)
        if hasattr(self, 'sbNormK'):
            self.sbNormK.editingFinished.connect(self._on_custom_normal_changed)
        if hasattr(self, 'sbNormL'):
            self.sbNormL.editingFinished.connect(self._on_custom_normal_changed)

        # Slice translate buttons
        if hasattr(self, 'btnSliceUpNormal'):
            self.btnSliceUpNormal.clicked.connect(lambda: self.nudge_along_normal(+1))
        if hasattr(self, 'btnSliceDownNormal'):
            self.btnSliceDownNormal.clicked.connect(lambda: self.nudge_along_normal(-1))
        if hasattr(self, 'btnSlicePosH'):
            self.btnSlicePosH.clicked.connect(lambda: self.nudge_along_axis('H', +1))
        if hasattr(self, 'btnSliceNegH'):
            self.btnSliceNegH.clicked.connect(lambda: self.nudge_along_axis('H', -1))
        if hasattr(self, 'btnSlicePosK'):
            self.btnSlicePosK.clicked.connect(lambda: self.nudge_along_axis('K', +1))
        if hasattr(self, 'btnSliceNegK'):
            self.btnSliceNegK.clicked.connect(lambda: self.nudge_along_axis('K', -1))
        if hasattr(self, 'btnSlicePosL'):
            self.btnSlicePosL.clicked.connect(lambda: self.nudge_along_axis('L', +1))
        if hasattr(self, 'btnSliceNegL'):
            self.btnSliceNegL.clicked.connect(lambda: self.nudge_along_axis('L', -1))

        # Slice rotate buttons
        if hasattr(self, 'btnRotPlusH'):
            self.btnRotPlusH.clicked.connect(lambda: self.rotate_about_axis('H', +self._slice_rotate_step_deg))
        if hasattr(self, 'btnRotMinusH'):
            self.btnRotMinusH.clicked.connect(lambda: self.rotate_about_axis('H', -self._slice_rotate_step_deg))
        if hasattr(self, 'btnRotPlusK'):
            self.btnRotPlusK.clicked.connect(lambda: self.rotate_about_axis('K', +self._slice_rotate_step_deg))
        if hasattr(self, 'btnRotMinusK'):
            self.btnRotMinusK.clicked.connect(lambda: self.rotate_about_axis('K', -self._slice_rotate_step_deg))
        if hasattr(self, 'btnRotPlusL'):
            self.btnRotPlusL.clicked.connect(lambda: self.rotate_about_axis('L', +self._slice_rotate_step_deg))
        if hasattr(self, 'btnRotMinusL'):
            self.btnRotMinusL.clicked.connect(lambda: self.rotate_about_axis('L', -self._slice_rotate_step_deg))
        if hasattr(self, 'btnResetSlice'):
            self.btnResetSlice.clicked.connect(self._on_reset_slice)
        
        # -------------------------------------------------------------------- #


    # ********* METHODS ******* #
    def _ensure_slice_2d_view(self):
        """Ensure the 2D slice view exists as a popup window."""
        try:
            if getattr(self, "slice_2d_view", None) is None:
                self.slice_2d_view = HKLSlice2DView(self)
                try:
                    # Make sure it shows as a top-level popup
                    self.slice_2d_view.setWindowTitle("2D Slice View")
                    self.slice_2d_view.setWindowFlags(self.slice_2d_view.windowFlags() | Qt.Window)
                except Exception:
                    pass
            # Show and bring to front
            try:
                self.slice_2d_view.show()
                self.slice_2d_view.raise_()
                self.slice_2d_view.activateWindow()
            except Exception:
                pass
        except Exception:
            pass

    def _on_view_slice_action(self):
        """Menu Tools -> View -> Slice: open 2D view popup and sync."""
        try:
            # Ensure widget exists and open as popup
            self._ensure_slice_2d_view()
            # Perform comprehensive sync of all settings from parent
            try:
                if getattr(self, 'slice_2d_view', None):
                    self.slice_2d_view.sync_all_settings()
            except Exception:
                pass
            try:
                n, o = self.get_plane_state()
                # Schedule update to reflect current plane and slice; if no slice yet this will update once available
                if getattr(self, 'slice_2d_view', None) and "slice" in getattr(self.plotter, "actors", {}):
                    # Use current slice actor input if it exists
                    try:
                        current_slice = self.plotter.actors["slice"].GetMapper().GetInput()
                    except Exception:
                        current_slice = None
                    if current_slice is not None:
                        self.slice_2d_view.schedule_update(current_slice, self.normalize_vector(np.array(n, dtype=float)), np.array(o, dtype=float))
                else:
                    # Fallback: trigger a refresh path
                    self.update_slice_from_plane(n, o)
            except Exception:
                pass
        except Exception:
            pass

    def open_controls_dialog(self, focus=None):
        """Open or focus the modeless Controls dialog."""
        try:
            if hasattr(self, 'controls_dialog') and self.controls_dialog is not None and self.controls_dialog.isVisible():
                try:
                    self.controls_dialog.raise_()
                except Exception:
                    pass
                try:
                    self.controls_dialog.activateWindow()
                except Exception:
                    pass
                if focus == 'camera':
                    try:
                        self.controls_dialog.focus_camera_section()
                    except Exception:
                        pass
                elif focus == 'slice':
                    try:
                        self.controls_dialog.focus_slice_section()
                    except Exception:
                        pass
                return
        except Exception:
            pass
        try:
            from viewer.hkl_controls_dialog import HKLControlsDialog
            self.controls_dialog = HKLControlsDialog(self)
            if focus == 'camera':
                try:
                    self.controls_dialog.focus_camera_section()
                except Exception:
                    pass
            elif focus == 'slice':
                try:
                    self.controls_dialog.focus_slice_section()
                except Exception:
                    pass
            self.controls_dialog.show()
        except Exception:
            pass

    # ********* METHODS ******* #
    def _set_busy(self, busy: bool, message: str = None):
        """Enable/disable window and show wait cursor during long operations."""
        try:
            if busy:
                try:
                    self.setEnabled(False)
                except Exception:
                    pass
                try:
                    QApplication.setOverrideCursor(Qt.WaitCursor)
                except Exception:
                    pass
                try:
                    if message and hasattr(self, 'statusbar') and self.statusbar:
                        self.statusbar.showMessage(str(message))
                except Exception:
                    pass
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
            else:
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                try:
                    self.setEnabled(True)
                except Exception:
                    pass
                try:
                    if hasattr(self, 'statusbar') and self.statusbar:
                        self.statusbar.clearMessage()
                except Exception:
                    pass
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
        except Exception:
            pass
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




    def _compute_adaptive_cells(self, total_points: int) -> int:
        """
        Compute adaptive grid cell count for single-pass rendering based on point count.
        Returns refine_cells (cells per axis).
        Aggressively lower grid resolution to speed up rendering (down to ~100s).
        """
        try:
            if total_points >= 5_000_000:
                return 300
            elif total_points >= 2_000_000:
                return 300
            elif total_points >= 500_000:
                return 300
            else:
                return 300
        except Exception:
            return 300

    def _get_target_cells_override(self) -> int:
        """Return user-entered target cells per axis or None if not set/invalid."""
        try:
            txt = str(self.leTargetCells.text()).strip()
        except Exception:
            return None
        if not txt:
            return None
        try:
            val = int(float(txt))
        except Exception:
            return None
        if val <= 0:
            return None
        return val

    def reduction_factor(self):
        """Handle reduction factor changes"""
        # Guard: require a point cloud with intensity before applying reduction
        try:
            if getattr(self, "cloud_copy", None) is None or (hasattr(self.cloud_copy, "n_points") and int(self.cloud_copy.n_points) == 0) or ("intensity" not in getattr(self.cloud_copy, "array_names", [])):
                # Allow updating size labels in volume-only mode without applying reduction
                if hasattr(self, 'vol') and self.vol is not None and isinstance(getattr(self, 'orig_shape', None), (tuple, list)) and len(self.orig_shape) == 2:
                    try:
                        reduction_factor_text = self.cbbReductionFactor.currentText() if hasattr(self, 'cbbReductionFactor') else "None"
                        import re
                        if str(reduction_factor_text).lower() == "none":
                            factor = 1.0
                        else:
                            m = re.search(r'x(\d+(?:\.\d+)?)', str(reduction_factor_text))
                            factor = float(m.group(1)) if m else 1.0
                        ox, oy = int(self.orig_shape[0]), int(self.orig_shape[1])
                        if factor <= 1.0:
                            self.curr_shape = (ox, oy)
                        else:
                            self.curr_shape = (max(1, int(ox // factor)), max(1, int(oy // factor)))
                        # Update labels to reflect selection without changing volume
                        try:
                            if hasattr(self, 'lbCurrentResolutionX'):
                                self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
                            if hasattr(self, 'lbCurrentResolutionY'):
                                self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))
                            self._update_info_image_sizes()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return
                try:
                    QMessageBox.warning(self, "No Data", "No point cloud loaded to reduce")
                except Exception:
                    pass
                return
        except Exception:
            pass
        try:
            reduction_factor_text = self.cbbReductionFactor.currentText()
            import re
            
            # Handle "None" option - no reduction
            if reduction_factor_text.lower() == "none":
                manual_factor = 1.0
            else:
                match = re.search(r'x(\d+(?:\.\d+)?)', reduction_factor_text)
                manual_factor = float(match.group(1)) if match else 1.0
            
            # Determine reduction factor, honor "None" as no reduction
            try:
                total_points = self.cloud_copy.n_points if hasattr(self.cloud_copy, 'n_points') else (len(self.cloud_copy) if hasattr(self.cloud_copy, '__len__') else 0)
            except Exception:
                total_points = 0
            auto_factor = self.calculate_auto_reduction_factor(total_points)
            factor = max(manual_factor, auto_factor) if reduction_factor_text.lower() != "none" else 1.0

            if factor > 1.0:
                cloud, intensity = self.process_3d_to_lower_res(self.cloud_copy, self.cloud_copy['intensity'], reduction_factor=factor)
            else:
                # Use full resolution and reflect original shape in UI
                cloud = self.cloud_copy.points if hasattr(self.cloud_copy, 'points') else np.asarray(self.cloud_copy)
                intensity = self.cloud_copy['intensity']
                self.curr_shape = self.orig_shape
            self.create_3D(cloud=cloud, intensity=intensity)
            try:
                self._set_busy(False)
            except Exception:
                pass
            
        except Exception as e:
            import traceback
            with open('error_output2.txt','w') as f:
                f.write(f"Traceback:\n{traceback.format_exc()}\n\nError:\n{str(e)}")
            try:
                self._set_busy(False)
            except Exception:
                pass


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
        # Capture original unfiltered point/intensity arrays for HKL filtering
        try:
            pts = self.cloud_copy.points if hasattr(self.cloud_copy, 'points') else np.asarray(self.cloud_copy)
            intens = self.cloud_copy['intensity']
            self._original_cloud_data = {
                'points': np.asarray(pts).copy(),
                'intensity': np.asarray(intens).copy()
            }
        except Exception:
            self._original_cloud_data = None
        return True

    # ================ 3D VISUALIZATION METHODS ======================= #
    def create_3D(self, cloud=None, intensity=None):
        # Disable UI and show busy cursor during render
        try:
            self._set_busy(True, "Rendering 3D...")
        except Exception:
            pass
        # Avoid clearing entire plotter to keep axes stable
        try:
            if "cloud_volume" in self.plotter.actors:
                self.plotter.remove_actor("cloud_volume", reset_camera=False)
            if "slice" in self.plotter.actors:
                self.plotter.remove_actor("slice", reset_camera=False)
            if "origin_sphere" in self.plotter.actors:
                self.plotter.remove_actor("origin_sphere", reset_camera=False)
            if "normal_line" in self.plotter.actors:
                self.plotter.remove_actor("normal_line", reset_camera=False)
            if hasattr(self.plotter, "plane_widgets") and self.plotter.plane_widgets:
                for pw in list(self.plotter.plane_widgets):
                    try:
                        self.plotter.remove_plane_widget(pw)
                    except Exception:
                        pass
        except Exception:
            pass
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
        refine_cells = self._compute_adaptive_cells(total_points)
        override = self._get_target_cells_override()
        if override is not None:
            refine_cells = int(override)

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
            name="cloud_volume",
            reset_camera=False
        )
        
        self.plotter.renderer._actors["cloud_volume"].SetVisibility(False)

        self.plotter.show_bounds(
            mesh=self.cloud_mesh, 
            xtitle='H Axis', 
            ytitle='K Axis', 
            ztitle='L Axis', 
            bounds=self.cloud_mesh.bounds
        )
        
        # Preserve initial bounds for a stable axes actor that won't move on slice updates
        self.initial_bounds = self.cloud_mesh.bounds
        
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

            # Fix axes bounds so they do not change when adding pointer or slice meshes
            try:
                b = self.initial_bounds
                cube_axes_actor.SetBounds(b[0], b[1], b[2], b[3], b[4], b[5])
                cube_axes_actor.Modified()
            except Exception:
                pass

        # ------- Adding the initial slice -------- #
        slice_normal = (0, 0, 1)
        slice_origin = self.cloud_mesh.center
        vol_slice = self.vol.slice(normal=slice_normal, origin=slice_origin)

        # Check if slice has data before adding
        if vol_slice.n_points > 0:
            self.plotter.add_mesh(vol_slice, scalars="intensity", name="slice", show_edges=False, reset_camera=False)
            # Enforce slice visibility based on toggle control
            try:
                self._apply_slice_visibility()
            except Exception:
                pass
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
        # Ensure only one interactive plane widget exists (remove any extras)
        try:
            if hasattr(self.plotter, "plane_widgets") and len(self.plotter.plane_widgets) > 1:
                for pw in self.plotter.plane_widgets[:-1]:
                    try:
                        self.plotter.remove_plane_widget(pw)
                    except Exception:
                        pass
        except Exception:
            pass
        # Sync lock state with plane widget and UI
        try:
            self._sync_plane_interaction()
            self._sync_slice_controls()
            self._sync_lock_message()
        except Exception:
            pass
        # Ensure Slice Lock overlay persists
        try:
            self._ensure_slice_lock_overlay_present()
        except Exception:
            pass

        # -------- Adding the arrow -------------- #
        sphere = pyv.Sphere(radius=0.1, center=slice_origin)
        line = pyv.Line(
            pointa=slice_origin,
            pointb=np.array(slice_origin) + np.array(slice_normal) * (np.min(data_range))
        )
        # Capture pointer actors so we can reliably remove/update them later
        try:
            self._pointer_sphere = self.plotter.add_mesh(sphere, color='red', name="origin_sphere", pickable=False)
        except Exception:
            self._pointer_sphere = None
        try:
            self._pointer_line = self.plotter.add_mesh(line, color='red', name="normal_line", pickable=False)
        except Exception:
            self._pointer_line = None

        # Hide them manually after adding
        # Hide pointer visuals by default (use references if available)
        try:
            if getattr(self, "_pointer_sphere", None) is not None:
                self._pointer_sphere.SetVisibility(False)
            else:
                self.plotter.renderer._actors["origin_sphere"].SetVisibility(False)
        except Exception:
            pass
        try:
            if getattr(self, "_pointer_line", None) is not None:
                self._pointer_line.SetVisibility(False)
            else:
                self.plotter.renderer._actors["normal_line"].SetVisibility(False)
        except Exception:
            pass

        # Store toggle visibility state
        self.toggle_state = {"visible": False}
        
        # --------- Adjust Labels -------------- #
        try:
            if hasattr(self, 'lbCurrentPointSizeNum'):
                self.lbCurrentPointSizeNum.setText(str(len(cloud)))
        except Exception:
            pass
        try:
            if hasattr(self, 'lbCurrentResolutionX'):
                self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
        except Exception:
            pass
        try:
            if hasattr(self, 'lbCurrentResolutionY'):
                self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))
        except Exception:
            pass
        try:
            self._update_info_image_sizes()
        except Exception:
            pass
        intensity_range = int(np.min(intensity)), int(np.max(intensity))
        self.sbMinIntensity.setRange(*intensity_range)
        self.sbMinIntensity.setValue(int(np.min(intensity)))
        self.sbMaxIntensity.setRange(*intensity_range)
        self.sbMaxIntensity.setValue(int(np.max(intensity)))
        self.update_intensity()
        try:
            self.update_info_labels()
            self._refresh_availability()
        except Exception:
            pass

    def update_intensity(self):
        """Updates the min/max intensity levels and scalar bar range"""
        try:
            self._set_busy(True, "Updating intensity...")
        except Exception:
            pass
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
                self.plotter.renderer._actors["cloud_volume"].SetVisibility(False)

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
        # Ensure Slice Lock overlay persists
        try:
            self._ensure_slice_lock_overlay_present()
        except Exception:
            pass
        # Sync 2D slice view levels to inherit from parent
        try:
            if getattr(self, "slice_2d_view", None):
                self.slice_2d_view.sync_levels()
        except Exception:
            pass

        # Update Info labels and availability after intensity changes
        try:
            self.update_info_slice_labels()
            self._refresh_availability()
        except Exception:
            pass
        try:
            self._set_busy(False)
        except Exception:
            pass

    # ======= INFO/PRECHECK/AVAILABILITY HELPERS ======= #
    def _is_data_loaded(self) -> bool:
        try:
            return hasattr(self, 'vol') and (self.vol is not None)
        except Exception:
            return False

    def _slice_exists(self) -> bool:
        try:
            actor = getattr(self.plotter, 'actors', {}).get('slice')
            if not actor:
                return False
            mapper = actor.GetMapper() if hasattr(actor, 'GetMapper') else None
            inp = mapper.GetInput() if mapper and hasattr(mapper, 'GetInput') else None
            npts = int(inp.GetNumberOfPoints()) if inp and hasattr(inp, 'GetNumberOfPoints') else 0
            return npts > 0
        except Exception:
            return False

    def _set_slice_controls_enabled(self, enabled: bool):
        try:
            for wname in ('gbSteps', 'gbOrientation', 'gbTranslate', 'gbRotate'):
                w = getattr(self, wname, None)
                if w:
                    w.setEnabled(bool(enabled))
            # Also toggle the Slice tab if available
            if hasattr(self, 'tabsControls') and hasattr(self, 'tabSlice'):
                try:
                    idx = self.tabsControls.indexOf(self.tabSlice)
                    self.tabsControls.setTabEnabled(idx, bool(enabled))
                except Exception:
                    pass
        except Exception:
            pass

    def _refresh_availability(self):
        try:
            data_loaded = self._is_data_loaded()
            slice_exists = self._slice_exists()
            if hasattr(self, 'actionSave'):
                self.actionSave.setEnabled(bool(data_loaded))
            if hasattr(self, 'actionExtractSlice'):
                self.actionExtractSlice.setEnabled(bool(slice_exists))
            self._set_slice_controls_enabled(bool(data_loaded))
        except Exception:
            pass

    def _ensure_data_loaded_or_warn(self) -> bool:
        if self._is_data_loaded():
            return True
        try:
            QMessageBox.warning(self, "No Data", "No data is loaded. Load data before adjusting the slice.")
        except Exception:
            pass
        return False

    def update_info_slice_labels(self):
        """Update labels in Info group with current slice orientation/normal/origin."""
        try:
            # Orientation preset text
            orient_text = getattr(self, '_slice_orientation_selection', None)
            if not orient_text:
                try:
                    if hasattr(self, 'cbSliceOrientation') and self.cbSliceOrientation is not None:
                        orient_text = self.cbSliceOrientation.currentText()
                except Exception:
                    orient_text = "-"
            if orient_text is None or orient_text == "":
                orient_text = "-"
            # Plane state
            normal, origin = self.get_plane_state()
            n = self.normalize_vector(np.array(normal, dtype=float))
            o = np.array(origin, dtype=float)
            # Format strings
            n_str = f"[{n[0]:0.3f}, {n[1]:0.3f}, {n[2]:0.3f}]"
            o_str = f"[{o[0]:0.5f}, {o[1]:0.5f}, {o[2]:0.5f}]"
            # Apply to labels if present
            try:
                if hasattr(self, 'lbSliceOrientationVal'):
                    self.lbSliceOrientationVal.setText(str(orient_text))
            except Exception:
                pass
            try:
                if hasattr(self, 'lbSliceNormalVal'):
                    self.lbSliceNormalVal.setText(n_str)
            except Exception:
                pass
            try:
                if hasattr(self, 'lbSliceOriginVal'):
                    self.lbSliceOriginVal.setText(o_str)
            except Exception:
                pass
            # Compute and display Slice Position based on orientation
            try:
                pos_text = "-"
                orient_lower = (str(orient_text) or "").lower()
                # origin is o = [H, K, L]
                if ("hk" in orient_lower) or ("xy" in orient_lower):
                    pos_text = f"L = {o[2]:0.2f}"
                elif ("kl" in orient_lower) or ("yz" in orient_lower):
                    pos_text = f"H = {o[0]:0.2f}"
                elif ("hl" in orient_lower) or ("xz" in orient_lower):
                    pos_text = f"K = {o[1]:0.2f}"
                else:
                    # Custom: scalar position along the current normal
                    s = float(np.dot(n, o))
                    pos_text = f"n·origin = {s:0.3f}"
                if hasattr(self, 'lbSlicePositionVal'):
                    self.lbSlicePositionVal.setText(pos_text)
            except Exception:
                pass
            # Reflect availability of Extract action based on existence (no Info label)
            try:
                if hasattr(self, 'actionExtractSlice'):
                    self.actionExtractSlice.setEnabled(self._slice_exists())
            except Exception:
                pass
        except Exception:
            pass

    # ======= CENTRALIZED INFO LABEL UPDATE METHOD ======= #
    def update_info_labels(self):
        """Update all info labels with current data state - centralized method for consistency."""
        try:
            # Update point count labels
            if hasattr(self, 'cloud_mesh') and self.cloud_mesh is not None:
                current_points = len(self.cloud_mesh.points) if hasattr(self.cloud_mesh, 'points') else 0
                try:
                    self.lbCurrentPointSizeNum.setText(str(current_points))
                except Exception:
                    pass

            # Ensure Original point size label is populated centrally
            try:
                if hasattr(self, 'lbOriginalPointSizeNum'):
                    orig_count = None
                    try:
                        if hasattr(self, 'cloud_copy') and self.cloud_copy is not None:
                            if hasattr(self.cloud_copy, 'n_points'):
                                orig_count = int(self.cloud_copy.n_points)
                            elif hasattr(self.cloud_copy, '__len__'):
                                orig_count = int(len(self.cloud_copy))
                    except Exception:
                        orig_count = None
                    if orig_count is None:
                        try:
                            if getattr(self, '_original_cloud_data', None) is not None and 'points' in self._original_cloud_data:
                                orig_count = int(len(self._original_cloud_data['points']))
                        except Exception:
                            pass
                    if orig_count is None:
                        try:
                            orig_count = int(self.lbOriginalPointSizeNum.text())
                        except Exception:
                            orig_count = 0
                    self.lbOriginalPointSizeNum.setText(str(orig_count))
            except Exception:
                pass

            # Update resolution labels
            try:
                if hasattr(self, 'curr_shape') and self.curr_shape:
                    self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
                    self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))
            except Exception:
                pass

            # Update image sizes
            try:
                self._update_info_image_sizes()
            except Exception:
                pass

            # Update slice-related info
            try:
                self.update_info_slice_labels()
            except Exception:
                pass

            # Update Info group point counts
            try:
                # Original points
                if hasattr(self, 'lbInfoPointsOrigVal'):
                    orig_count = None
                    try:
                        if hasattr(self, 'cloud_copy') and self.cloud_copy is not None:
                            orig_count = int(self.cloud_copy.n_points) if hasattr(self.cloud_copy, 'n_points') else (len(self.cloud_copy) if hasattr(self.cloud_copy, '__len__') else None)
                    except Exception:
                        orig_count = None
                    if orig_count is None:
                        try:
                            orig_count = int(self.lbOriginalPointSizeNum.text())
                        except Exception:
                            orig_count = 0
                    self.lbInfoPointsOrigVal.setText(str(orig_count))
            except Exception:
                pass
            try:
                # Current points
                if hasattr(self, 'lbInfoPointsCurrVal'):
                    curr_count = None
                    try:
                        if hasattr(self, 'cloud_mesh') and self.cloud_mesh is not None and hasattr(self.cloud_mesh, 'points'):
                            curr_count = int(len(self.cloud_mesh.points))
                    except Exception:
                        curr_count = None
                    if curr_count is None:
                        try:
                            curr_count = int(self.lbCurrentPointSizeNum.text())
                        except Exception:
                            curr_count = 0
                    self.lbInfoPointsCurrVal.setText(str(curr_count))
            except Exception:
                pass

        except Exception as e:
            print(f"Error updating info labels: {e}")

    # ======= INFO IMAGE SIZE HELPERS ======= #
    def _update_info_image_sizes(self):
        """Update Info section image sizes (Original and Current) based on orig_shape/curr_shape or existing labels."""
        try:
            # Original size
            ox, oy = None, None
            try:
                os_shape = getattr(self, 'orig_shape', None)
                if isinstance(os_shape, (tuple, list)) and len(os_shape) == 2 and int(os_shape[0]) > 0 and int(os_shape[1]) > 0:
                    ox, oy = int(os_shape[0]), int(os_shape[1])
            except Exception:
                pass
            if ox is None or oy is None:
                try:
                    ox = int(self.lbOriginalResolutionX.text())
                    oy = int(self.lbOriginalResolutionY.text())
                except Exception:
                    ox, oy = None, None
            orig_text = f"{ox} x {oy}" if (ox is not None and oy is not None) else "0 x 0"
            try:
                if hasattr(self, 'lbInfoImageSizeOrigVal') and self.lbInfoImageSizeOrigVal is not None:
                    self.lbInfoImageSizeOrigVal.setText(orig_text)
            except Exception:
                pass

            # Current size
            cx, cy = None, None
            try:
                cs_shape = getattr(self, 'curr_shape', None)
                if isinstance(cs_shape, (tuple, list)) and len(cs_shape) == 2 and int(cs_shape[0]) > 0 and int(cs_shape[1]) > 0:
                    cx, cy = int(cs_shape[0]), int(cs_shape[1])
            except Exception:
                pass
            if cx is None or cy is None:
                try:
                    cx = int(self.lbCurrentResolutionX.text())
                    cy = int(self.lbCurrentResolutionY.text())
                except Exception:
                    cx, cy = None, None
            curr_text = f"{cx} x {cy}" if (cx is not None and cy is not None) else "0 x 0"
            try:
                if hasattr(self, 'lbInfoImageSizeCurrVal') and self.lbInfoImageSizeCurrVal is not None:
                    self.lbInfoImageSizeCurrVal.setText(curr_text)
            except Exception:
                pass
        except Exception:
            pass

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
        """Callback for updating the slice and vector visuals with bounds safety."""
        # Guard: volume must exist
        if not hasattr(self, 'vol') or self.vol is None:
            return
        # Ignore if slice is locked
        try:
            if bool(getattr(self, "_slice_locked", False)):
                return
        except Exception:
            pass

        # Normalize normal and clamp origin within bounds to avoid empty slice
        n = self.normalize_vector(np.array(normal, dtype=float))
        o = np.array(origin, dtype=float)
        try:
            clamped_origin, was_clamped = self.clamp_plane_origin_within_bounds(n, o)
        except Exception:
            clamped_origin, was_clamped = o, False

        # Sync plane widget if we adjusted origin
        try:
            if was_clamped and hasattr(self.plotter, 'plane_widgets') and self.plotter.plane_widgets:
                pw = self.plotter.plane_widgets[0]
                pw.SetNormal(n)
                pw.SetOrigin(clamped_origin)
        except Exception:
            pass

        # Prepare new slice
        new_slice = None
        try:
            new_slice = self.vol.slice(normal=n, origin=clamped_origin)
        except Exception:
            new_slice = None

        # If slice is empty, warn and do not alter existing actors
        try:
            if (new_slice is None) or (getattr(new_slice, 'n_points', 0) == 0):
                QMessageBox.warning(self, "Plane out of bounds",
                                    "Slice plane is out of bounds. Keeping previous slice and snapping plane just inside bounds.")
                return
        except Exception:
            # Silent fallback if QMessageBox unavailable
            return

        # Remove previous slice actor only after we have a valid new slice
        try:
            if "slice" in self.plotter.actors:
                self.plotter.remove_actor("slice", reset_camera=False)
        except Exception:
            pass

        # Add slice without modifying scalar ranges
        focused_range = [getattr(self, 'min_intensity', 0), getattr(self, 'max_intensity', 0)]
        try:
            self.plotter.add_mesh(
                new_slice,
                scalars="intensity",
                cmap=self.lut,
                clim=focused_range if focused_range[1] >= focused_range[0] else None,
                name="slice",
                show_edges=False,
                reset_camera=False,
                pickable=False
            )
        except Exception:
            return
        # Enforce slice visibility based on toggle control
        try:
            self._apply_slice_visibility()
        except Exception:
            pass
        self.update_intensity()
        # Mirror the slice into the lightweight 2D view only if it is already open (do not auto-open)
        try:
            if getattr(self, "slice_2d_view", None):
                self.slice_2d_view.schedule_update(new_slice, n, clamped_origin)
        except Exception:
            pass

        # Avoid re-adding volume/scalar bars during interactive plane moves to keep axes stable

        # Remove previous origin and line
        # Remove previous pointer actors using stored references if available
        try:
            if getattr(self, "_pointer_sphere", None) is not None:
                try:
                    self.plotter.remove_actor(self._pointer_sphere)
                except Exception:
                    pass
                self._pointer_sphere = None
            else:
                # Fallback by name
                self.plotter.remove_actor("origin_sphere", reset_camera=False)
        except Exception:
            pass
        try:
            if getattr(self, "_pointer_line", None) is not None:
                try:
                    self.plotter.remove_actor(self._pointer_line)
                except Exception:
                    pass
                self._pointer_line = None
            else:
                # Fallback by name
                self.plotter.remove_actor("normal_line", reset_camera=False)
        except Exception:
            pass

        # Create new visuals using clamped origin and normalized normal
        try:
            bounds_range = (self.cloud_mesh.points.max(axis=0) - self.cloud_mesh.points.min(axis=0))
            sphere_radius = (np.min(bounds_range)) * 0.002
        except Exception:
            sphere_radius = 0.1
        new_sphere = pyv.Sphere(radius=sphere_radius, center=clamped_origin)
        try:
            line_len = float(np.min(self.cloud_mesh.points.max(axis=0) - self.cloud_mesh.points.min(axis=0)))
        except Exception:
            line_len = 1.0
        new_line = pyv.Line(pointa=clamped_origin, pointb=np.array(clamped_origin) + np.array(n) * line_len)

        # Add new pointer actors and store references
        try:
            self._pointer_sphere = self.plotter.add_mesh(new_sphere, color='red', name="origin_sphere", pickable=False)
        except Exception:
            self._pointer_sphere = None
        try:
            self._pointer_line = self.plotter.add_mesh(new_line, color='red', name="normal_line", pickable=False)
        except Exception:
            self._pointer_line = None

        # Ensure axes bounds remain fixed (avoid axes moving with pointer updates)
        # try:
        #     cube_axes_actor = self.plotter.renderer.cube_axes_actor
        #     if cube_axes_actor and hasattr(self, 'initial_bounds') and self.initial_bounds:
        #         b = self.initial_bounds
        #         cube_axes_actor.SetBounds(b[0], b[1], b[2], b[3], b[4], b[5])
        #         cube_axes_actor.Modified()
        # except Exception:
        #     pass

        # Set pointer visibility based on toggle state
        try:
            if getattr(self, "_pointer_sphere", None) is not None:
                self._pointer_sphere.SetVisibility(self.toggle_state.get("visible", False))
            else:
                self.plotter.renderer._actors["origin_sphere"].SetVisibility(self.toggle_state.get("visible", False))
        except Exception:
            pass
        try:
            if getattr(self, "_pointer_line", None) is not None:
                self._pointer_line.SetVisibility(self.toggle_state.get("visible", False))
            else:
                self.plotter.renderer._actors["normal_line"].SetVisibility(self.toggle_state.get("visible", False))
        except Exception:
            pass

        # Keep plane widget synchronized (again) to final state
        try:
            if hasattr(self.plotter, 'plane_widgets') and self.plotter.plane_widgets:
                pw = self.plotter.plane_widgets[0]
                pw.SetNormal(n)
                pw.SetOrigin(clamped_origin)
        except Exception:
            pass

        # Update Info labels and availability
        try:
            self.update_info_slice_labels()
            self._refresh_availability()
        except Exception:
            pass
        # Render
        try:
            self.plotter.render()
        except Exception:
            pass

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
                # Reduction factor deprecated: use full-resolution data
                cloud = self.cloud_copy.points if hasattr(self.cloud_copy, 'points') else np.asarray(self.cloud_copy)
                intensity = self.cloud_copy['intensity']
                self.curr_shape = self.orig_shape
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
                try:
                    self.update_info_slice_labels()
                    self._refresh_availability()
                except Exception:
                    pass
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
        self.plotter.clear()
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
            info = loader.get_file_info(file_name, style="dict")
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
                try:
                    self._slice_lock_overlay_added = False
                except Exception:
                    pass
                
                # Focus and camera
                try:
                    self.plotter.set_focus(self.grid.center)
                except Exception:
                    pass
                self.plotter.add_volume(volume=self.vol, scalars="intensity", name="cloud_volume", reset_camera=False)
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
                    # Ensure only one interactive plane widget exists (remove any extras)
                    try:
                        if hasattr(self.plotter, "plane_widgets") and len(self.plotter.plane_widgets) > 1:
                            for pw in self.plotter.plane_widgets[:-1]:
                                try:
                                    self.plotter.remove_plane_widget(pw)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Sync lock state and recreate overlay after clear()
                    try:
                        self._setup_slice_lock_overlay()
                        self._sync_plane_interaction()
                        self._sync_slice_controls()
                        self._sync_lock_message()
                    except Exception:
                        pass
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
                
                # Update Info labels and availability for volume mode
                try:
                    self.update_info_slice_labels()
                    self._refresh_availability()
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
                # Reduction factor deprecated: use full-resolution data
                self.curr_shape = self.orig_shape
                self.create_3D(cloud=points, intensity=intensities)
                
                # Update UI
                self.groupBox3DViewer.setTitle(f'Viewing {num_images} Image(s)')
                self.lbOriginalPointSizeNum.setText(str(len(points)))
                self.lbOriginalResolutionX.setText(str(shape[0]))
                self.lbOriginalResolutionY.setText(str(shape[1]))
                
                print(f"Successfully loaded {len(points)} points from {num_images} images")
            try:
                self.update_info_slice_labels()
                self._refresh_availability()
            except Exception:
                pass
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
        # Precheck: require volume/data loaded before prompting save dialog
        if not self._is_data_loaded():
            QMessageBox.warning(self, "No Volume", "No volume available to save")
            return
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
                'parent_channel': getattr(self.parent.reader, 'input_channel', 'unknown') if hasattr(self, 'parent') and hasattr(self.parent, 'reader') else 'unknown',
                'extraction_timestamp': str(np.datetime64('now')),
                'volume_shape': list(vol_array.shape),  # (D,H,W) cell-centered
                'voxel_spacing': list(map(float, self.vol.spacing if hasattr(self.vol, 'spacing') else (self.grid.spacing if hasattr(self, 'grid') and self.grid is not None else (1.0, 1.0, 1.0)))),
                'grid_origin': list(map(float, self.vol.origin if hasattr(self.vol, 'origin') else (self.grid.origin if hasattr(self, 'grid') and self.grid is not None else (0.0, 0.0, 0.0)))),
                'intensity_range': [float(np.min(vol_array)), float(np.max(vol_array))],
                'grid_dimensions_cells': [int(x) for x in dims_cells],
                'array_order': 'F',
                'axes_labels': ['H', 'K', 'L']
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
            # Precheck: require current slice exists before prompting for save
            if not self._slice_exists():
                QMessageBox.warning(self, "No Slice", "No slice data available to extract")
                return

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

            # Prefer current viewer shape; fallback to original
            preferred_shape = None
            try:
                cs = getattr(self, 'curr_shape', None)
                if isinstance(cs, (tuple, list)) and len(cs) == 2 and int(cs[0]) > 0 and int(cs[1]) > 0:
                    preferred_shape = (int(cs[0]), int(cs[1]))
            except Exception:
                preferred_shape = None
            if preferred_shape is None:
                try:
                    os_shape = getattr(self, 'orig_shape', None)
                    if isinstance(os_shape, (tuple, list)) and len(os_shape) == 2 and int(os_shape[0]) > 0 and int(os_shape[1]) > 0:
                        preferred_shape = (int(os_shape[0]), int(os_shape[1]))
                except Exception:
                    preferred_shape = None

            success = loader.extract_slice(
                file_path=file_path,
                points=slice_points,
                intensities=slice_intensities,
                metadata=slice_metadata,
                shape=preferred_shape
            )
            
            if success:
                QMessageBox.information(self, "Success", f"Slice extracted and saved successfully!\n{len(slice_points)} points saved.")
            else:
                try:
                    last_err = loader.get_last_error()
                except Exception:
                    last_err = "Unknown error"
                QMessageBox.critical(self, "Error", f"Failed to save slice: {last_err}")
                
        except Exception as e:
            QMessageBox.critical(self, "Extract Error", f"Error extracting slice:\n{str(e)}")

            

    # ================ CAMERA CONTROL METHODS ======================= #
    def zoom_in(self):
        """Zoom camera in using step factor from main state (_zoom_step)"""
        camera = self.plotter.camera
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if not np.isfinite(step) or step <= 1.0:
                step = 1.5
        except Exception:
            step = 1.5
        camera.zoom(step)
        self.plotter.render()

    def zoom_out(self):
        """Zoom camera out using inverse of step factor from main state (_zoom_step)"""
        camera = self.plotter.camera
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if not np.isfinite(step) or step <= 1.0:
                step = 1.5
        except Exception:
            step = 1.5
        out_factor = 1.0 / step
        camera.zoom(out_factor)
        self.plotter.render()

    def reset_camera(self):
        """Reset camera to default position"""
        self.plotter.reset_camera()
        self.plotter.render()

    def set_camera_position(self):
        """Set camera to predefined positions (use dialog state or legacy UI combo)."""
        # Source text from dialog-updated state first
        pos_src = getattr(self, '_cam_pos_selection', None)
        if not pos_src:
            try:
                if hasattr(self, 'cbSetCamPos') and self.cbSetCamPos is not None:
                    pos_src = self.cbSetCamPos.currentText()
                elif hasattr(self, 'camSetPosCombo') and self.camSetPosCombo is not None:
                    pos_src = self.camSetPosCombo.currentText()
            except Exception:
                pass

        pos_text = (pos_src or '').strip().lower()
        p = self.plotter
        cam = getattr(p, 'camera', None)

        # Helper to center focus for consistent navigation
        def _set_focus_to_data_center():
            try:
                if hasattr(self, 'vol') and self.vol is not None and hasattr(self.vol, 'center'):
                    p.set_focus(self.vol.center)
                elif hasattr(self, 'cloud_mesh') and self.cloud_mesh is not None and hasattr(self.cloud_mesh, 'center'):
                    p.set_focus(self.cloud_mesh.center)
            except Exception:
                pass

        # Orthogonal planar presets (existing)
        if ('xy' in pos_text) or ('hk' in pos_text):
            _set_focus_to_data_center()
            p.view_xy()
        elif ('yz' in pos_text) or ('kl' in pos_text):
            _set_focus_to_data_center()
            p.view_yz()
        elif ('xz' in pos_text) or ('hl' in pos_text):
            _set_focus_to_data_center()
            p.view_xz()
        # Isometric
        elif 'iso' in pos_text:
            _set_focus_to_data_center()
            try:
                p.view_isometric()
            except Exception:
                # Fallback: a generic isometric-looking vector
                try:
                    p.view_vector((1.0, 1.0, 1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                except Exception:
                    pass
        else:
            # Axis-aligned (+/-H/K/L or +/-X/Y/Z)
            _set_focus_to_data_center()
            label = (pos_text or '')
            # Robust substring detection to handle composite labels like "H+ (X+)"
            try:
                if ('h+' in label) or ('x+' in label):
                    p.view_vector((1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif ('h-' in label) or ('x-' in label):
                    p.view_vector((-1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif ('k+' in label) or ('y+' in label):
                    p.view_vector((0.0, 1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif ('k-' in label) or ('y-' in label):
                    p.view_vector((0.0, -1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif ('l+' in label) or ('z+' in label):
                    p.view_vector((0.0, 0.0, 1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
                elif ('l-' in label) or ('z-' in label):
                    p.view_vector((0.0, 0.0, -1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
            except Exception:
                pass

        # Keep view_up orthogonal for safety
        try:
            if cam is not None and hasattr(cam, 'orthogonalize_view_up'):
                cam.orthogonalize_view_up()
        except Exception:
            pass

        # Render
        try:
            p.render()
        except Exception:
            pass

    def _apply_cam_preset_button(self, label: str):
        """
        Apply camera preset triggered by quick preset buttons.
        Selects the combo text and delegates to set_camera_position() for mapping and safety.
        """
        try:
            # Prefer dialog-driven state update
            self._cam_pos_selection = label
        except Exception:
            try:
                if hasattr(self, 'cbSetCamPos') and (self.cbSetCamPos is not None):
                    self.cbSetCamPos.setCurrentText(label)
            except Exception:
                pass
        try:
            self.set_camera_position()
        except Exception:
            # Fallback directly to orthogonal views
            try:
                if 'hk' in label.lower() or 'xy' in label.lower():
                    self.plotter.view_xy()
                elif 'kl' in label.lower() or 'yz' in label.lower():
                    self.plotter.view_yz()
                elif 'hl' in label.lower() or 'xz' in label.lower():
                    self.plotter.view_xz()
                self.plotter.render()
            except Exception:
                pass

    def view_slice_normal(self):
        """Align camera to look along the current slice normal at the slice origin."""
        try:
            # Get current slice plane normal and origin
            normal, origin = self.get_plane_state()
            normal = self.normalize_vector(np.array(normal, dtype=float))
            origin = np.array(origin, dtype=float)

            cam = getattr(self.plotter, 'camera', None)
            if cam is None:
                return

            # Determine a reasonable camera distance
            try:
                pos = np.array(getattr(cam, 'position', origin), dtype=float)
                focal = np.array(getattr(cam, 'focal_point', origin), dtype=float)
                distance = float(np.linalg.norm(pos - focal))
                if not np.isfinite(distance) or distance <= 0.0:
                    # fallback to data bounds
                    try:
                        rng = self.cloud_mesh.points.max(axis=0) - self.cloud_mesh.points.min(axis=0)
                        distance = float(np.linalg.norm(rng)) * 0.5
                    except Exception:
                        distance = 1.0
            except Exception:
                distance = 1.0

            # Set camera focal point and position along normal
            try:
                cam.focal_point = origin.tolist() if hasattr(origin, 'tolist') else origin
            except Exception:
                pass
            try:
                cam.position = (origin + normal * distance).tolist()
            except Exception:
                pass

            # Keep/upd view-up to avoid degeneracy with normal
            try:
                current_up = np.array(getattr(cam, 'view_up', [0.0, 1.0, 0.0]), dtype=float)
                up_norm = self.normalize_vector(current_up)
                # If up is nearly parallel to normal, choose an alternate axis
                if abs(float(np.dot(up_norm, normal))) > 0.99:
                    # pick Y unless normal's Y is dominant; else pick X
                    new_up = np.array([0.0, 1.0, 0.0], dtype=float) if abs(normal[1]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=float)
                    cam.view_up = new_up.tolist()
            except Exception:
                pass

            self.plotter.render()
        except Exception:
            pass

    # ================ VISIBILITY TOGGLE METHODS ======================= #
    def toggle_pointer(self):
        """Toggle visibility of the slice pointer (origin sphere and normal line)"""
        vis = self.cbToggleSlicePointer.isChecked()
        self.toggle_state["visible"] = vis
        try:
            if getattr(self, "_pointer_sphere", None) is not None:
                self._pointer_sphere.SetVisibility(vis)
            else:
                self.plotter.renderer._actors["origin_sphere"].SetVisibility(vis)
        except Exception:
            pass
        try:
            if getattr(self, "_pointer_line", None) is not None:
                self._pointer_line.SetVisibility(vis)
            else:
                self.plotter.renderer._actors["normal_line"].SetVisibility(vis)
        except Exception:
            pass
        self.plotter.render()
        
    def toggle_cloud_vol(self):
        """Toggle visibility of the cloud volume"""
        is_visible = self.cbToggleCloudVolume.isChecked()
        self.plotter.renderer._actors["cloud_volume"].SetVisibility(is_visible)

    def toggle_slice_mesh(self):
        """Toggle visibility of the slice mesh"""
        try:
            visible = True
            if hasattr(self, 'cbToggleSliceMesh'):
                visible = bool(self.cbToggleSliceMesh.isChecked())
            if "slice" in getattr(self.plotter, "actors", {}):
                actor = self.plotter.actors["slice"]
                try:
                    actor.SetVisibility(visible)
                except Exception:
                    try:
                        # Fallback via renderer actors mapping
                        self.plotter.renderer._actors["slice"].SetVisibility(visible)
                    except Exception:
                        pass
            # Render after toggling
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    def _apply_slice_visibility(self):
        """Apply visibility of slice actor based on the checkbox state."""
        try:
            if "slice" not in getattr(self.plotter, "actors", {}):
                return
            target = True
            if hasattr(self, 'cbToggleSliceMesh'):
                target = bool(self.cbToggleSliceMesh.isChecked())
            actor = self.plotter.actors["slice"]
            try:
                actor.SetVisibility(target)
            except Exception:
                try:
                    self.plotter.renderer._actors["slice"].SetVisibility(target)
                except Exception:
                    pass
        except Exception:
            pass

    def change_color_map(self):
        color_map_select = self.cbColorMapSelect.currentText()
        # Create new lookup table
        new_lut = pyv.LookupTable(cmap=color_map_select)
        self.lut = new_lut
        
        # Update scalar bar's lookup table directly
        if hasattr(self.plotter, 'scalar_bars'):
            for scalar_bar_name, scalar_bar_actor in self.plotter.scalar_bars.items():
                if scalar_bar_actor:
                    
                    scalar_bar_actor.SetLookupTable(new_lut)
                    scalar_bar_actor.Modified()
        # Force render
        self.plotter.render()
        # Sync 2D slice view colormap without adding controls
        try:
            if getattr(self, "slice_2d_view", None):
                self.slice_2d_view.sync_colormap()
        except Exception:
            pass
        normal, origin = self.get_plane_state()
        n = self.normalize_vector(np.array(normal, dtype=float))
        o = np.array(origin, dtype=float)
        self.update_slice_from_plane(n,o)

    # HUD overlay for slice lock
    def _setup_slice_lock_overlay(self):
        # Ensure minimal, labeled, top-left slice lock HUD
        try:
            # initialize state flag
            self._slice_locked = bool(getattr(self, "_slice_locked", False))
        except Exception:
            self._slice_locked = False

        # Always ensure the label exists and is positioned top-left
        try:
            if "slice_lock_label" not in getattr(self.plotter, "actors", {}):
                try:
                    # Prefer pixel positioning near the checkbox
                    self.plotter.add_text("Slice Lock", position=(40, 12), font_size=12, color="white", name="slice_lock_label")
                except Exception:
                    # Fallback to a named corner position if pixel positioning unsupported
                    self.plotter.add_text("Slice Lock", position="upper_left", font_size=12, color="white", name="slice_lock_label")
        except Exception:
            pass

        # Add the checkbox only once per plotter lifecycle unless it was cleared
        try:
            if not bool(getattr(self, "_slice_lock_overlay_added", False)):
                # Attempt to add a slightly smaller checkbox at top-left
                added = False
                try:
                    self._slice_lock_checkbox = self.plotter.add_checkbox_button_widget(
                        self._on_slice_lock_toggled,
                        value=bool(self._slice_locked),
                        position=(0.02, 0.96),  # normalized near top-left
                        size=18
                    )
                    added = True
                except Exception:
                    try:
                        # Fallback without size parameter if backend doesn't support it
                        self._slice_lock_checkbox = self.plotter.add_checkbox_button_widget(
                            self._on_slice_lock_toggled,
                            value=bool(self._slice_locked),
                            position=(0.02, 0.96)  # normalized near top-left
                        )
                        added = True
                    except Exception:
                        pass
                if added:
                    self._slice_lock_overlay_added = True
        except Exception:
            pass

        # Apply current state to plane and controls and message
        try:
            self._sync_plane_interaction()
            self._sync_slice_controls()
            self._sync_lock_message()
        except Exception:
            pass

    def _on_slice_lock_toggled(self, value: bool):
        try:
            self._slice_locked = bool(value)
        except Exception:
            self._slice_locked = bool(value)
        # Sync plane and UI state
        self._sync_plane_interaction()
        self._sync_slice_controls()
        self._sync_lock_message()

    def _sync_plane_interaction(self):
        try:
            widgets = getattr(self.plotter, "plane_widgets", [])
            for pw in widgets or []:
                try:
                    pw.SetEnabled(not bool(getattr(self, "_slice_locked", False)))
                except Exception:
                    try:
                        if bool(getattr(self, "_slice_locked", False)):
                            pw.EnabledOff()
                        else:
                            pw.EnabledOn()
                    except Exception:
                        pass
        except Exception:
            pass

    def _sync_slice_controls(self):
        try:
            idx = self.tabsControls.indexOf(self.tabSlice)
            self.tabsControls.setTabEnabled(idx, not bool(getattr(self, "_slice_locked", False)))
        except Exception:
            try:
                self.tabSlice.setEnabled(not bool(getattr(self, "_slice_locked", False)))
            except Exception:
                pass

    def _sync_lock_message(self):
        try:
            locked = bool(getattr(self, "_slice_locked", False))
            if locked:
                # Add text HUD if not present
                if not ("slice_lock_text" in getattr(self.plotter, "actors", {})):
                    try:
                        self.plotter.add_text("Slice Locked", position="upper_edge", font_size=16, color="red", name="slice_lock_text")
                    except Exception:
                        pass
                try:
                    if hasattr(self, "statusbar") and self.statusbar:
                        self.statusbar.showMessage("Slice is locked")
                except Exception:
                    pass
            else:
                # Remove HUD text if present
                try:
                    if "slice_lock_text" in getattr(self.plotter, "actors", {}):
                        self.plotter.remove_actor("slice_lock_text")
                except Exception:
                    pass
                try:
                    if hasattr(self, "statusbar") and self.statusbar:
                        self.statusbar.clearMessage()
                except Exception:
                    pass
            # Render to update HUD visibility
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_slice_lock_overlay_present(self):
        """Ensure Slice Lock overlay label and checkbox exist; re-add if missing."""
        try:
            # Ensure label
            if "slice_lock_label" not in getattr(self.plotter, "actors", {}):
                try:
                    self.plotter.add_text("Slice Lock", position=(40, 12), font_size=12, color="white", name="slice_lock_label")
                except Exception:
                    try:
                        self.plotter.add_text("Slice Lock", position="upper_left", font_size=12, color="white", name="slice_lock_label")
                    except Exception:
                        pass
            # Ensure checkbox
            if not bool(getattr(self, "_slice_lock_overlay_added", False)) or getattr(self, "_slice_lock_checkbox", None) is None:
                try:
                    self._slice_lock_checkbox = self.plotter.add_checkbox_button_widget(
                        self._on_slice_lock_toggled,
                        value=bool(getattr(self, "_slice_locked", False)),
                        position=(0.02, 0.96),
                        size=18
                    )
                except Exception:
                    try:
                        self._slice_lock_checkbox = self.plotter.add_checkbox_button_widget(
                            self._on_slice_lock_toggled,
                            value=bool(getattr(self, "_slice_locked", False)),
                            position=(0.02, 0.96)
                        )
                    except Exception:
                        pass
                self._slice_lock_overlay_added = True
            # Re-sync state with overlay
            try:
                self._sync_plane_interaction()
                self._sync_slice_controls()
                self._sync_lock_message()
            except Exception:
                pass
        except Exception:
            pass

    # ================ SLICE CONTROL METHODS ======================= #
    def _on_translate_step_changed(self, val: float):
        self._slice_translate_step = float(val)

    def _on_rotate_step_changed(self, val: float):
        self._slice_rotate_step_deg = float(val)

    def _on_orientation_changed(self, idx: int):
        if not self._ensure_data_loaded_or_warn():
            return
        preset = getattr(self, '_slice_orientation_selection', None)
        if not preset:
            try:
                preset = self.cbSliceOrientation.currentText()
            except Exception:
                preset = 'HK(xy)'
        self.set_plane_preset(preset)

    def _on_custom_normal_changed(self):
        if not self._ensure_data_loaded_or_warn():
            return
        # Only respond when Custom is selected
        preset = (str(getattr(self, '_slice_orientation_selection', '')) or '').lower()
        if preset.startswith('custom'):
            try:
                n_raw = np.array(getattr(self, '_custom_normal', [0.0, 0.0, 1.0]), dtype=float)
            except Exception:
                n_raw = np.array([0.0, 0.0, 1.0], dtype=float)
            n = self.normalize_vector(n_raw)
            # Apply immediately, keep origin
            _, origin = self.get_plane_state()
            self.update_slice_from_plane(n, origin)

    def _on_reset_slice(self):
        if not self._ensure_data_loaded_or_warn():
            return
        # Default to HK(xy)
        try:
            center = None
            if hasattr(self, 'vol') and self.vol is not None and hasattr(self.vol, 'center'):
                center = np.array(self.vol.center)
            elif hasattr(self, 'cloud_mesh') and self.cloud_mesh is not None and hasattr(self.cloud_mesh, 'center'):
                center = np.array(self.cloud_mesh.center)
            else:
                center = np.array([0.0, 0.0, 0.0], dtype=float)
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self.update_slice_from_plane(normal, center)
        except Exception:
            pass

    def get_plane_state(self):
        """Get current plane normal and origin; fall back to defaults."""
        try:
            if hasattr(self.plotter, 'plane_widgets') and self.plotter.plane_widgets:
                pw = self.plotter.plane_widgets[0]
                normal = np.array(pw.GetNormal(), dtype=float)
                origin = np.array(pw.GetOrigin(), dtype=float)
                return normal, origin
        except Exception:
            pass
        # Fallbacks
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        try:
            origin = np.array(self.cloud_mesh.center, dtype=float)
        except Exception:
            origin = np.array([0.0, 0.0, 0.0], dtype=float)
        return normal, origin

    def set_plane_state(self, normal, origin):
        """Programmatically set plane state and apply slice."""
        n = self.normalize_vector(np.array(normal, dtype=float))
        o = np.array(origin, dtype=float)
        self.update_slice_from_plane(n, o)

    def normalize_vector(self, v):
        v = np.array(v, dtype=float)
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return v / norm

    def _get_bounds_center_half(self):
        """Return (center, half_extents, margin) for current volume bounds."""
        try:
            b = self.vol.bounds
            minb = np.array([b[0], b[2], b[4]], dtype=float)
            maxb = np.array([b[1], b[3], b[5]], dtype=float)
            center = 0.5 * (minb + maxb)
            half = 0.5 * (maxb - minb)
            # Margin: 0.5% of smallest non-zero extent
            extent = np.maximum(maxb - minb, 1e-12)
            margin = 0.005 * float(np.min(extent))
            return center, half, margin
        except Exception:
            return np.zeros(3, dtype=float), np.ones(3, dtype=float), 1e-3

    def clamp_plane_origin_within_bounds(self, normal, origin):
        """
        Clamp plane origin so that the plane intersects the volume AABB with a safety margin.
        Uses AABB-plane intersection criterion: |n·(c - o)| <= dot(|n|, half_extents).
        """
        n = self.normalize_vector(np.array(normal, dtype=float))
        o = np.array(origin, dtype=float)
        c, h, margin = self._get_bounds_center_half()

        e = float(np.dot(np.abs(n), h))  # projected half extents along n
        s = float(np.dot(n, (c - o)))    # signed distance from plane origin to center along n

        # If within band (minus margin), keep origin
        if abs(s) <= max(e - margin, 0.0):
            return o, False

        # Snap to just inside bounds along normal direction
        s_target = np.sign(s) * max(e - margin, 0.0)
        o_new = c - n * s_target
        return o_new, True

    def set_plane_preset(self, preset_text: str):
        """Set plane normal to preset HK/KL/HL immediately."""
        preset = preset_text.lower()
        if 'xy' in preset or 'hk' in preset:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        elif 'yz' in preset or 'kl' in preset:
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif 'xz' in preset or 'hl' in preset:
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            # Custom uses dialog-updated state; avoid referencing dialog widgets directly
            try:
                n = np.array(getattr(self, '_custom_normal', [0.0, 0.0, 1.0]), dtype=float)
            except Exception:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            n = self.normalize_vector(n)
        _, origin = self.get_plane_state()
        self.set_plane_state(n, origin)

    def nudge_along_normal(self, sign: int):
        """Translate plane origin along its normal by translate_step."""
        if not self._ensure_data_loaded_or_warn():
            return
        normal, origin = self.get_plane_state()
        step = float(self._slice_translate_step)
        origin_new = origin + float(sign) * step * normal
        self.set_plane_state(normal, origin_new)

    def nudge_along_axis(self, axis: str, sign: int):
        """Translate plane origin along H/K/L axis by translate_step."""
        if not self._ensure_data_loaded_or_warn():
            return
        axis = axis.upper()
        if axis == 'H':
            d = np.array([1.0, 0.0, 0.0], dtype=float)
        elif axis == 'K':
            d = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            d = np.array([0.0, 0.0, 1.0], dtype=float)
        normal, origin = self.get_plane_state()
        step = float(self._slice_translate_step)
        origin_new = origin + float(sign) * step * d
        # Keep current normal; only translate origin along axis
        self.set_plane_state(normal, origin_new)

    def rotate_about_axis(self, axis: str, deg: float):
        """Rotate plane normal around H/K/L axis by deg (degrees)."""
        if not self._ensure_data_loaded_or_warn():
            return
        axis = axis.upper()
        if axis == 'H':
            u = np.array([1.0, 0.0, 0.0], dtype=float)
        elif axis == 'K':
            u = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            u = np.array([0.0, 0.0, 1.0], dtype=float)
        normal, origin = self.get_plane_state()
        theta = float(np.deg2rad(deg))
        ux, uy, uz = u
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
        ], dtype=float)
        new_normal = R @ normal
        new_normal = self.normalize_vector(new_normal)
        self.set_plane_state(new_normal, origin)

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
        # Close the lightweight 2D slice view if it exists
        try:
            if getattr(self, "slice_2d_view", None):
                self.slice_2d_view.close()
        except Exception:
            pass
        self.slice_2d_view = None
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