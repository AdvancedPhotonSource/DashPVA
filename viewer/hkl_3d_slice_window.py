import sys
import numpy as np
import os.path as osp
import pyvista as pyv
from pyvistaqt import QtInteractor
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt
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
        self.cbToggleCloud.clicked.connect(self.toggle_cloud)
        self.cbToggleCloudVolume.clicked.connect(self.toggle_cloud_vol)
        self.cbColorMapSelect.currentIndexChanged.connect(self.change_color_map)
        
        # Data alteration
        self.cbbReductionFactor.currentIndexChanged.connect(self.reduction_factor)
        self.sbMinIntensity.editingFinished.connect(self.update_intensity)
        self.sbMaxIntensity.editingFinished.connect(self.update_intensity)
        
        #TODO: these functions
        self.actionLoadData.triggered.connect(self.load_data)
        self.actionExtractSlice.triggered.connect(self.extract_slice)
        
        
        # Determines if application is standalone mode
        if self.parent: 
            self.btnUseParentData.clicked.connect(self._load_parent_data)
            self.actionSave.triggered.connect(self.save_data)
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
        Lower resolution by voxelizing point cloud
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
        return voxel_centers, final_intensities


    def reduction_factor(self):
        """Handle reduction factor changes"""
        try:
            reduction_factor = self.cbbReductionFactor.currentText()
            import re
            match = re.search(r'x(\d+(?:\.\d+)?)', reduction_factor)
            factor = float(match.group(1)) if match else 1.0
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
        
        # ----- Adding the point cloud data ----- #
        self.plotter.add_mesh(
            self.cloud_mesh,
            scalars="intensity",
            cmap=self.lut,
            point_size=5.0,
            name="points",
            reset_camera=False,
            nan_color=None,
            nan_opacity=0.0,
            show_scalar_bar=True,
        )
        
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

        target_cells = 300

        # FIX: Use axis-specific spacing instead of uniform spacing
        spacing_per_axis = grid_range / target_cells
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
            volume= self.vol,
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
        """Updates the min/max intensity levels and filters visible points"""
        self.min_intensity = self.sbMinIntensity.value()
        self.max_intensity = self.sbMaxIntensity.value()
        
        if self.min_intensity > self.max_intensity:
            self.min_intensity, self.max_intensity = self.max_intensity, self.min_intensity
            self.sbMinIntensity.setValue(self.min_intensity)
            self.sbMaxIntensity.setValue(self.max_intensity)
        
        # Apply threshold to filter points
        if hasattr(self, 'cloud_copy') and self.cloud_copy is not None:
            # Create filtered cloud
            filtered_cloud = self.cloud_copy.threshold(
                [self.min_intensity, self.max_intensity],
                scalars="intensity"
            )
            
            # Update the displayed mesh
            self.plotter.remove_actor("points", reset_camera=False)
            self.plotter.add_mesh(
                filtered_cloud,
                scalars="intensity",
                cmap=self.lut,
                point_size=5.0,
                name="points",
                reset_camera=False,
                show_scalar_bar=True
            )
            
            # Update the cloud_mesh reference for other operations
            self.cloud_mesh = filtered_cloud
            if hasattr(self, 'grid') and self.grid is not None:
                # Calculate optimal radius based on grid spacing
                grid_spacing = np.array(self.grid.spacing)
                optimal_radius = np.mean(grid_spacing) * 2.5
                
                # Create new interpolated volume
                self.vol = self.grid.interpolate(
                    filtered_cloud, 
                    radius=optimal_radius,
                    sharpness=1.5,
                    null_value=0.0
                )
                
                # Update the volume actor
                self.plotter.remove_actor("cloud_volume", reset_camera=False)
                self.plotter.add_volume(
                    volume=self.vol,
                    scalars="intensity",
                    name="cloud_volume"
                )
                
                # Maintain volume visibility state
                if hasattr(self, 'cbToggleCloudVolume') and self.cbToggleCloudVolume.isChecked():
                    self.plotter.renderer._actors["cloud_volume"].SetVisibility(True)
                else:
                    self.plotter.renderer._actors["cloud_volume"].SetVisibility(False)
                
                # Update the slice if plane widget exists
                if hasattr(self, 'plane_widget') and self.plane_widget:
                    normal = self.plane_widget.GetNormal()
                    origin = self.plane_widget.GetOrigin()
                    self.update_slice_from_plane(normal, origin)
                    
    # ================ CALLBACK METHODS ======================= #
    def update_slice_from_plane(self, normal, origin):
        """Callback for updating the slice and vector visuals"""
        # Update slice
        new_slice = self.vol.slice(normal=normal, origin=origin)
        
        self.plotter.add_mesh(new_slice, 
                            scalars="intensity", 
                            cmap=self.lut, 
                            name="slice", 
                            show_edges=False, 
                            reset_camera=False,
                            pickable=False
                            )

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
    def _load_parent_data(self):
        """Check if parent has the required data available"""
        if hasattr(self.parent, 'cloud') and self.parent.cloud is not None and hasattr(self.parent, 'reader') and self.parent.reader is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.setEnabled(False)
            original_title = self.windowTitle()
            self.setWindowTitle(f"{original_title} ***** Loading...")
            QApplication.processEvents()
            self.setup_3d_cloud(cloud=self.parent.cloud.copy(deep=True), intensity=self.parent.cloud['intensity'], shape=self.parent.reader.shape)
            cloud, intensity = self.process_3d_to_lower_res(self.cloud_copy, self.cloud_copy['intensity'])
            self.create_3D(cloud=cloud, intensity=intensity)
            self.groupBox3DViewer.setTitle(f'Viewing {len(self.parent.reader.cache_images)} Image(s)')
            self.lbOriginalPointSizeNum.setText(str(self.cloud_copy.n_points))
            self.lbOriginalResolutionX.setText(str(self.parent.reader.shape[0]))
            self.lbOriginalResolutionY.setText(str(self.parent.reader.shape[1]))
            intensity_range = int(np.min(intensity)), int(np.max(intensity))
            self.sbMinIntensity.setRange(*intensity_range)
            self.sbMinIntensity.setValue(int(np.min(intensity)))
            self.sbMaxIntensity.setRange(*intensity_range)
            self.sbMaxIntensity.setValue(int(np.max(intensity)))
            
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self.setWindowTitle(original_title)
            
    
    def load_data(self):
        """Load data from HD5 file"""
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select an HD5 File','','HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not file_name:
            QMessageBox.warning(self, "File", "No Valid File Selected")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.setEnabled(False)
        original_title = self.windowTitle()
        self.setWindowTitle(f"{original_title} ***** Loading...")
        QApplication.processEvents()
        
        try:
            from utils import HDF5Loader
            loader = HDF5Loader()
            points, intensities, num_images, shape = loader.load_h5_to_3d(file_name)
            
            #result = self.load_h5_to_3d(file_name)
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
            if self.setup_3d_cloud(cloud=points, intensity=intensities, shape=num_images):
                self.orig_shape = shape
                reduction_factor = self.cbbReductionFactor.currentText()
                import re
                match = re.search(r'x(\d+(?:\.\d+)?)', reduction_factor)
                factor = float(match.group(1)) if match else 1.0
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
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self.setWindowTitle(original_title)
            
    def save_data(self):
        """Only used for the parent"""
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
    def toggle_cloud(self):
        """Toggle visibility of the cloud mesh"""
        # Check if checkbox is checked
        is_visible = self.cbToggleCloud.isChecked()
        cloud_actor = self.plotter.actors["points"]
        cloud_actor.visibility = is_visible
        
    def toggle_cloud_vol(self):
        """Toggle visibility of the cloud volume"""
        is_visible = self.cbToggleCloudVolume.isChecked()
        self.plotter.renderer._actors["cloud_volume"].SetVisibility(is_visible)

    def change_color_map(self):
        color_map_select = self.cbColorMapSelect.currentText()
        # Create new lookup table
        new_lut = pyv.LookupTable(cmap=color_map_select)
        self.lut = new_lut
        
        # Update all actors in plotter.actors (PyVista managed)
        for actor_name, actor in self.plotter.actors.items():
            if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'lookup_table'):
                
                actor.mapper.lookup_table = new_lut
                actor.mapper.update()
        
        # Update all actors in renderer._actors (VTK managed)
        for actor_name, actor in self.plotter.renderer._actors.items():
            # Check if it's a volume or other VTK actor with LUT
            if hasattr(actor, 'GetMapper') and actor.GetMapper():
                mapper = actor.GetMapper()
                if hasattr(mapper, 'SetLookupTable'):
                    
                    mapper.SetLookupTable(new_lut)
                    mapper.Update()
            
            # Check for volume actors (different property structure)
            elif hasattr(actor, 'GetProperty') and hasattr(actor.GetProperty(), 'SetLookupTable'):
                
                actor.GetProperty().SetLookupTable(new_lut)
        
        # Update scalar bar's lookup table directly
        if hasattr(self.plotter, 'scalar_bars'):
            for scalar_bar_name, scalar_bar_actor in self.plotter.scalar_bars.items():
                if scalar_bar_actor:
                    
                    scalar_bar_actor.SetLookupTable(new_lut)
                    scalar_bar_actor.Modified()
        # Force render
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