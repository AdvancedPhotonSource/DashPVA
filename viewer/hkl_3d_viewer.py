from stat import filemode
import sys, pathlib
import subprocess
import numpy as np
import os.path as osp
import pyqtgraph as pg
import pyvista as pyv
import pyvistaqt as pyvqt
from pyvistaqt import QtInteractor, BackgroundPlotter
from PyQt5 import uic
# from epics import caget
from epics import camonitor, caget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
# Custom imported classes
from pva_reader import PVAReader
# Add the parent directory to the path so the font_scaling.py file can be imported
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils.size_manager import SizeManager

import h5py
import numpy as np
import os
import time

class ConfigDialog(QDialog):

    def __init__(self):
        """
        Class that does initial setup for getting the pva prefix, collector address,
        and the path to the json that stores the pvs that will be observed

        Attributes:
            input_channel (str): Input channel for PVA.
            config_path (str): Path to the ROI configuration file.
        """
        super(ConfigDialog,self).__init__()
        uic.loadUi('gui/hkl_viewer_setup.ui', self)
        self.setWindowTitle('PV Config')
        # initializing variables to pass to Image Viewer
        self.input_channel = ""
        self.config_path =  ""
        # class can be prefilled with text
        self.init_ui()
        
        # Connecting signasl to 
        self.btn_clear.clicked.connect(self.clear_pv_setup)
        self.btn_browse.clicked.connect(self.browse_file_dialog)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted) 

    def init_ui(self) -> None:
        """
        Prefills text in the Line Editors for the user.
        """
        self.le_input_channel.setText(self.le_input_channel.text())
        self.le_config.setText(self.le_config.text())

    def browse_file_dialog(self) -> None:
        """
        Opens a file dialog to select the path to a TOML configuration file.
        """
        self.pvs_path, _ = QFileDialog.getOpenFileName(self, 'Select TOML Config', 'pv_configs', '*.toml (*.toml)')

        self.le_config.setText(self.pvs_path)
    
    def clear_pv_setup(self) -> None:
        """
        Clears line edit that tells image view where the config file is.
        """
        self.le_config.clear()


    def dialog_accepted(self) -> None:
        """
        Handles the final step when the dialog's accept button is pressed.
        Starts the HKLImageWindow process with filled information.
        """
        self.input_channel = self.le_input_channel.text()
        self.config_path = self.le_config.text()
        if osp.isfile(self.config_path) or (self.config_path == ''):
            self.hkl_3d_viewer = HKLImageWindow(input_channel=self.input_channel,
                                            file_path=self.config_path,) 
        else:
            print('File Path Doesn\'t Exitst')  
            #TODO: ADD ERROR Dialog rather than print message so message is clearer
            self.new_dialog = ConfigDialog()
            self.new_dialog.show()    


class HKLImageWindow(QMainWindow):

    def __init__(self, input_channel='s6lambda1:Pva1:Image', file_path=''): 
        """
        Initializes the main window for real-time image visualization and manipulation.

        Args:
            input_channel (str): The PVA input channel for the detector.
            file_path (str): The file path for loading configuration.
        """
        super(HKLImageWindow, self).__init__()
        uic.loadUi('gui/hkl_viewer_window.ui', self)
        self.setWindowTitle('HKL Viewer')
        self.show()

        # Initializing Viewer variables
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)
        self._file_path = file_path

        # Initializing but not starting timers so they can be reached by different functions
        self.timer_labels = QTimer()
        # self.timer_plot = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        # self.timer_plot.timeout.connect(self.update_image)

        # HKL values
        self.hkl_config = None
        self.hkl_data = {}
        self.qx = None
        self.qy = None
        self.qz = None
        self.processes = {}
        
        # Adding widgets manually to have better control over them
        pyv.set_plot_theme('dark')
        self.plotter = QtInteractor(self)
        self.viewer_layout.addWidget(self.plotter,1,1)

        # pyvista vars
        self.actor = None
        self.lut = None
        self.cloud = None
        self.min_intensity = 0.0
        self.max_intensity = 0.0
        self.min_opacity = 0.0
        self.max_opacity = 1.0
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')

        # Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.btn_plot_cache.clicked.connect(self.update_image)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        # self.log_image.clicked.connect(self.reset_first_plot)
        # self.plotting_frequency.valueChanged.connect(self.start_timers)
        # self.log_image.clicked.connect(self.update_image)
        self.sbox_min_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_max_intensity.editingFinished.connect(self.update_intensity)
        self.sbox_min_opacity.editingFinished.connect(self.update_opacity)
        self.sbox_max_opacity.editingFinished.connect(self.update_opacity)
        # self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

        # Opening the 3d Slice 
        self.btn_3d_slice_window.clicked.connect(self.open_3d_slice_window)

    def start_timers(self) -> None:
        """
        Starts timers for updating labels and plotting at specified frequencies.
        """
        self.timer_labels.start(int(1000/100))
        # self.timer_plot.start(int(1000/self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        """
        Stops the updating of main window labels and plots.
        """
        # self.timer_plot.stop()
        self.timer_labels.stop()

    #TODO: CHECK With 4id network camera to test if
    # start of X,Y and size of X,Y line up when transposed
    def set_pixel_ordering(self) -> None:
        """
        Checks which pixel ordering is selected on startup
        """
        if self.reader is not None:
            if self.rbtn_C.isChecked():
                self.reader.pixel_ordering = 'C'
                self.reader.image_is_transposed = True 
            elif self.rbtn_F.isChecked():
                self.reader.pixel_ordering = 'F'
                self.image_is_transposed = False
                self.reader.image_is_transposed = False

    def c_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to C style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'C'
            self.reader.image_is_transposed = True

    def f_ordering_clicked(self) -> None:
        """
        Sets the pixel ordering to Fortran style.
        """
        if self.reader is not None:
            self.reader.pixel_ordering = 'F'
            self.image_is_transposed = False
            self.reader.image_is_transposed = False

    def start_live_view_clicked(self) -> None:
        """
        Initializes the connections to the PVA channel using the provided Channel Name.
        
        This method ensures that any existing connections are cleared and re-initialized.
        Also starts monitoring the stats and adds ROIs to the viewer.
        """
        try:
            # A double check to make sure there isn't a connection already when starting
            self.plotter.clear()
            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path,
                                         viewer_type='rsm')          
            else:
                self.stop_timers()
                self.btn_save_h5.clicked.disconnect()
                self.btn_plot_cache.clicked.disconnect()
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel, 
                                         config_filepath=self._file_path,
                                         viewer_type='rsm')
            self.reset_first_plot()
            self.btn_save_h5.clicked.connect(self.reader.save_caches_to_h5)
            self.btn_plot_cache.clicked.connect(self.update_image)
            if self.reader.CACHING_MODE == 'scan':
                self.reader.add_on_scan_complete_callback(self.update_image)
                self.reader.add_on_scan_complete_callback(self.reader.save_caches_to_h5)
        except:
            print(f'Failed to Connect to {self._input_channel}')
            del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')
        
        if self.reader is not None:
            self.set_pixel_ordering()
            self.reader.start_channel_monitor()
            self.start_timers()

    def stop_live_view_clicked(self) -> None:
        """
        Clears the connection for the PVA channel and stops all active monitors.

        This method also updates the UI to reflect the disconnected state.
        """
        if self.reader is not None:
            self.reader.stop_channel_monitor()
            self.stop_timers()
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    # def freeze_image_checked(self) -> None:
    #     """
    #     Toggles freezing/unfreezing of the plot based on the checked state
    #     without stopping the collection of PVA objects.
    #     """
    #     if self.reader is not None:
    #         if self.freeze_image.isChecked():
    #             self.stop_timers()
    #         else:
    #             self.start_timers()

    def reset_first_plot(self) -> None:
        """
        Resets the `first_plot` flag, ensuring the next plot behaves as the first one.
        """
        self.first_plot = True

    def update_pv_prefix(self) -> None:
        """
        Updates the input channel prefix based on the value entered in the prefix field.
        """
        self._input_channel = self.pv_prefix.text()

    def update_labels(self) -> None:
        """
        Updates the UI labels with current connection and cached data.
        """
        if self.reader is not None:
            provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
            is_connected = 'Connected' if self.reader.channel.isMonitorActive() else 'Disconnected'
            self.provider_name.setText(provider_name)
            self.is_connected.setText(is_connected)
            self.missed_frames_val.setText(f'{self.reader.frames_missed:d}')
            self.frames_received_val.setText(f'{self.reader.frames_received:d}')
            
    def update_image(self) -> None:
        """
        Redraws plots based on the configured update rate.

        Processes the image data according to main window settings, such as rotation
        and log transformations. Also sets initial min/max pixel values in the UI.
        """
        if self.reader is not None:
            self.call_id_plot +=1
            if self.reader.cache_images is not None and self.reader.cache_qx is not None:
                try:
                    num_images = len(self.reader.cache_images)
                    num_rsm = len(self.reader.cache_qx)
                    if num_images !=  num_rsm:
                        raise ValueError(f'Size of caches are uneven:\nimages:{num_images}\nqxyz: {num_rsm}')
                    # Collect all cached data
                    flat_intensity = np.concatenate(self.reader.cache_images, dtype=np.float32)
                    qx = np.concatenate(self.reader.cache_qx, dtype=np.float32)
                    qy = np.concatenate(self.reader.cache_qy, dtype=np.float32)
                    qz = np.concatenate(self.reader.cache_qz, dtype=np.float32)

                    points = np.column_stack((
                        qx, qy, qz
                    ))
                    # First-time setup
                    if self.first_plot:
                        self.min_intensity = np.min(flat_intensity)
                        self.max_intensity = np.max(flat_intensity)
                        self.sbox_min_intensity.setValue(self.min_intensity)
                        self.sbox_max_intensity.setValue(self.max_intensity)

                        self.cloud = pyv.PolyData(points)
                        self.cloud['intensity'] = flat_intensity 

                        self.lut = pyv.LookupTable(cmap='viridis')  
                        self.lut.below_range_color = 'black'
                        self.lut.above_range_color = 'black'
                        self.lut.below_range_opacity = 0
                        self.lut.above_range_opacity = 0 
                        self.update_opacity()
                        self.update_intensity()
                       
                        self.actor = self.plotter.add_mesh(
                            self.cloud,
                            scalars='intensity',
                            cmap=self.lut,
                            point_size=10
                        )
                        
                        self.first_plot = False
                    else:
                        print(f'Updating Image')
                        self.plotter.mesh.points = points
                        self.cloud['intensity'] = flat_intensity
                        self.update_intensity()
                        self.update_opacity()
                    self.plotter.show_bounds(xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
                except Exception as e:
                    print(f"[Viewer] Failed to update 3D plot: {e}")

    def update_opacity(self) -> None:
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        self.min_opacity = self.sbox_min_opacity.value()
        self.max_opacity = self.sbox_max_opacity.value()
        if self.min_opacity > self.max_opacity:
            self.min_opacity, self.max_opacity = self.max_opacity, self.min_opacity
            self.sbox_min_opacity.setValue(self.min_opacity)
            self.sbox_max_opacity.setValue(self.max_opacity)
        if self.lut is not None:
            self.lut.apply_opacity([self.min_opacity,self.max_opacity])

    def update_intensity(self) -> None:
        """
        Updates the min/max intensity levels in the HKL Viewer based on UI settings.
        """
        self.min_intensity = self.sbox_min_intensity.value()
        self.max_intensity = self.sbox_max_intensity.value()
        if self.min_intensity > self.max_intensity:
            self.min_intensity, self.max_intensity = self.max_intensity, self.min_intensity
            self.sbox_min_intensity.setValue(self.min_intensity)
            self.sbox_max_intensity.setValue(self.max_intensity)
        if self.lut is not None:
            self.lut.scalar_range = (self.min_intensity, self.max_intensity)
        if self.actor is not None:
            self.actor.mapper.scalar_range = (self.min_intensity,self.max_intensity)
    
    # def closeEvent(self, event):
    #     """pass
    #     Custom close event to clean up resources, including stat dialogs.

    #     Args:
    #         event (QCloseEvent): The close event triggered when the main window is closed.
    #     """
    #     super(HKLImageWindow,self).closeEvent(event)

    def open_3d_slice_window(self) -> None:
        try:
            self.slice_window = HKL3DSliceWindow(self) 
            self.slice_window.show()
        except Exception as e:
            import traceback
            with open('error_output2.txt','w') as f:
                f.write(f"Traceback:\n{traceback.format_exc()}\n\nError:\n{str(e)}")


class HKL3DSliceWindow(QMainWindow):
    # Todo: 
    # Fix wierd glitch with sphere in arrow after update
    # The interpolated plane mbecomes movable when data is loaded please fixs
    # Color map is a little wierd for the intensity -- make it a different color
    # HKL axis not bound to data -- bound to the bounds of the grid
    # Slice gives error and disappears when out of bounds
    # Lut and Opacity problem -- when they are applied interpolation disappears
    # The reason the 3DSlicer causes a segmentation fault is because the widgets inside them -- have to close saftley
    #   -- Reference AnalysisWindow.closeEvent()
    # (Done) Load the data from the an hd5 file 

    def __init__(self, parent):
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
        
        self.cbToggleCloud.clicked.connect(self.toggle_cloud)
        self.cbToggleCloudVolume.clicked.connect(self.toggle_cloud_vol)
        self.cbColorMapSelect.currentIndexChanged.connect(self.change_color_map)
        
        self.cbbReductionFactor.currentIndexChanged.connect(self.reduction_factor)
        
        self.btnLoadHD5.clicked.connect(self.load_data)
        self.btnUseParentData.clicked.connect(self._load_parent_data)
        
        # variables to be used
        self.grid = None
        self.cloud_copy = None
        self.orig_shape = ()
        self.curr_shape = ()

        # ------- Look up table & Opacity
        self.lut = pyv.LookupTable(cmap='magma')  

        # ------- Creating the plotter for the 3d slicer
        self.plotter = QtInteractor()
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        
        
        self.viewer_3d_slicer_layout.addWidget(self.plotter,1,1)
        # -------------------------------------------------------------------- #


    # ********* METHODS ******* #
    # ================ RESOLUTION PROCESSING METHODS ======================= #
    def lower_res(self, points, voxel_size=1.0) -> tuple:
        """
        Lower resolution by voxelizing point cloud
        
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
        original_res_x = self.parent.reader.shape[0]
        original_res_y = self.parent.reader.shape[1]
        
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
        
        suggested_voxel_size = self.calculate_voxel_size(cloud_points, reduction_factor=reduction_factor)
        cloud_with_intensities = np.concatenate([cloud_points, intensity.reshape(-1, 1)], axis=1)
        voxel_centers, final_intensities = self.lower_res(points=cloud_with_intensities, voxel_size=suggested_voxel_size)
        
        return voxel_centers, final_intensities

    def reduction_factor(self):
        """Handle reduction factor changes"""
        try:
            print('Reducing Res')
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
    def setup_3d_cloud(self, cloud, intensity):
        """Sets the cloud and intensity that will be used"""
        if not np.all(cloud):
            # need to create the 3d cloud in this case or call 
            # the update function in the parent
            self.cloud_copy = self.parent.cloud.copy(deep=True)
            self.cloud_copy['intensity'] = self.parent.cloud['intensity']
            
        # Check if cloud is numpy array, convert to PyVista if needed
        if isinstance(cloud, np.ndarray):
            self.cloud_copy = pyv.PolyData(cloud)
            self.cloud_copy['intensity'] = intensity
        else:
            # Already PyVista object
            self.cloud_copy = cloud
            self.cloud_copy['intensity'] = intensity

    # ================ 3D VISUALIZATION METHODS ======================= #
    def create_3D(self, cloud=None, intensity=None):
        self.plotter.clear()
        # ------ Initialize the cloud ------ #
        
        self.cloud_mesh = pyv.PolyData(cloud)
        self.cloud_mesh['intensity'] = intensity
        
        # ------- Look up table & Opacity ---- #
        # min_intensity = np.max(self.cloud_copy['intensity']) / 4
        # # May not need to do these portions
        # max_intensity = np.max(self.cloud_copy['intensity'])
        # masked_scalar = np.copy(self.cloud_copy['intensity']).astype(float)
        # masked_scalar[masked_scalar < min_intensity] = np.nan
        # masked_scalar[masked_scalar > max_intensity] = np.nan
        # self.cloud_copy['intensity'] = masked_scalar

        # self.lut.scalar_range = (
        #     min_intensity, 
        #     np.nanmax(self.cloud_copy['intensity'])
        # )

        # self.cloud_copy = self.cloud_copy.threshold([min_intensity,max_intensity])
        
        # ----- Adding the point cloud data ----- #
        self.plotter.add_mesh(
            self.cloud_mesh,
            scalars="intensity",
            cmap=self.lut,
            point_size=6.0,
            name="points",
            reset_camera=False,
            nan_color=None,
            nan_opacity=0.0
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
            bounds=self.cloud_mesh.bounds,
        )

        # ------- Adding the initial slice -------- #
        slice_normal = (0, 0, 1)
        slice_origin = self.cloud_mesh.center
        vol_slice = self.vol.slice(normal=slice_normal, origin=slice_origin)

        # Check if slice has data before adding
        if vol_slice.n_points > 0:
            self.plotter.add_mesh(vol_slice, scalars="intensity", name="slice", show_edges=False, reset_camera=False)
        else:
            print("Warning: Slice is empty!")

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
            origin_translation=True
        )

        # -------- Adding the arrow -------------- #
        sphere = pyv.Sphere(radius=0.3, center=slice_origin)
        line = pyv.Line(
            pointa=slice_origin,
            pointb=np.array(slice_origin) + np.array(slice_normal) * 2
        )
        self.plotter.add_mesh(sphere, color='red', name="origin_sphere", pickable=False)
        self.plotter.add_mesh(line, color='green', name="normal_line", pickable=False)

        # Hide them manually after adding
        self.plotter.renderer._actors["origin_sphere"].SetVisibility(False)
        self.plotter.renderer._actors["normal_line"].SetVisibility(False)

        # Store toggle visibility state
        self.toggle_state = {"visible": False}

        # ------- Point size slider ------------ #
        self.plotter.add_slider_widget(
            self.update_point_size,
            rng=[1, 20],
            value=5,
            title="Point Size",
            pointa=(0.01, .90),
            pointb=(0.21, .90)
        )
        
        # --------- Adjust Labels -------------- #
        self.lbCurrentPointSizeNum.setText(str(len(cloud)))
        self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
        self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))

    # ================ CALLBACK METHODS ======================= #
    def update_point_size(self, size):
        """Callback for updating the size of the points"""
        actor = self.plotter.renderer._actors["points"]
        actor.GetProperty().SetPointSize(size)

    def update_slice_from_plane(self, normal, origin):
        """Callback for updating the slice and vector visuals"""
        # Update slice
        new_slice = self.vol.slice(normal=normal, origin=origin)
        self.plotter.add_mesh(new_slice, scalars="intensity", cmap=self.lut, name="slice", show_edges=False, reset_camera=False)

        # Remove previous origin and line
        self.plotter.remove_actor("origin_sphere", reset_camera=False)
        self.plotter.remove_actor("normal_line", reset_camera=False)

        # Create new visuals
        new_sphere = pyv.Sphere(radius=0.3, center=origin)
        new_line = pyv.Line(pointa=origin, pointb=np.array(origin) + np.array(normal) * 2)

        self.plotter.add_mesh(new_sphere, color='red', name="origin_sphere", pickable=False)
        self.plotter.add_mesh(new_line, color='green', name="normal_line", pickable=False)

        # Set their visibility based on toggle state
        self.plotter.renderer._actors["origin_sphere"].SetVisibility(self.toggle_state["visible"])
        self.plotter.renderer._actors["normal_line"].SetVisibility(self.toggle_state["visible"])

    # ================ DATA LOADING ======================= #
    def _load_parent_data(self):
        """Check if parent has the required data available"""
        if hasattr(self.parent, 'cloud') and self.parent.cloud is not None and hasattr(self.parent, 'reader') and self.parent.reader is not None:
            self.setup_3d_cloud(cloud=self.parent.cloud.copy(deep=True), intensity=self.parent.cloud['intensity'])
            cloud, intensity = self.process_3d_to_lower_res(self.cloud_copy, self.cloud_copy['intensity'])
            self.create_3D(cloud=cloud, intensity=intensity)
            self.groupBox3DViewer.setTitle(f'Viewing {len(self.parent.reader.cache_images)} Image(s)')
            self.lbOriginalPointSizeNum.setText(str(self.cloud_copy.n_points))
            self.lbOriginalResolutionX.setText(str(self.parent.reader.shape[0]))
            self.lbOriginalResolutionY.setText(str(self.parent.reader.shape[1]))
            
    
    def load_data(self):
        """Load data from HD5 file"""
        # ToDo: When trying to load from the application it does not load
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select an HD5 File','','HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if file_name:
            self.lineEditFileLocation.setText(file_name)
        
        try:
            result = self.load_h5_to_3d(file_name)
            self.setup_3d_cloud(cloud=result[0], intensity=result[1])
            cloud, intensity = self.process_3d_to_lower_res(cloud=result[0], intensity=result[1])
            self.create_3D(cloud=cloud, intensity=intensity)
            
            self.groupBox3DViewer.setTitle(f'Viewing {result[2]} Image(s)')
            self.lbOriginalPointSizeNum.setText(str(len(result[0])))
        except Exception as e:
            import traceback
            with open('error_output1.txt', 'w') as f:
                f.write(f"Traceback:\n{traceback.format_exc()}")
                f.write(str(e))


    def load_h5_to_3d(self, file:str='') -> tuple:
        """
        Loads .h5 file to 3D points 

        Args: 
            file (str): String path to .h5 file

        Returns: 
            tuple: 
                points - the 3D points to be plotted
                flat_intensity - Intensity of the image
                num_of_images - Number of images in the .h5 file
            
        Example: 
            result = load_h5_to_3d('path/to/file')\n
            pd = pyvista.PolyData(result[0])
        """
        
        # Set the variables that will be returned
        num_of_images = 0
        points = None
        flat_intensity = None
        try:
            with h5py.File(file, 'r') as f:

                # q is for quaternion
                # Grab the data needed in order to create 3D data
                qx = f['entry/data/hkl/qx']
                qy = f['entry/data/hkl/qy']
                qz = f['entry/data/hkl/qz']
                images = f['/entry/data/data']
                num_of_images = f['entry/data/data'].shape[0]
                
                # Check that its not an empty images or data
                if not qx or not qy or not qz:
                    raise ValueError('No data/# of images is 0')
                

                # Turn the q's into a 1D array into 3D
                # shape that can be concatenated store
                # the data in a dict that can be accessed
                q_dict = {}
                for data in (qx, qy, qz):
                    q_data_list = []
                    
                    # Reshape to a 1D array and store it in an array list
                    for i in data:
                        q_d = np.reshape(i, -1)
                        q_data_list.append(q_d)
                    
                    # Create a numpy array of the list
                    q_array = np.array(q_data_list)

                    # Get the name of the vector being processed
                    name = data.name.split('/')[-1]
                    q_dict[name] = q_array 

                # Reshape the number of images to a one dimensional image 
                flat_intensity = np.reshape(images,-1)

                # Concatenate each images and column stack in order to create the image in 3D
                qx_con = np.concatenate(q_dict["qx"])
                qy_con = np.concatenate(q_dict["qy"])
                qz_con = np.concatenate(q_dict["qz"])
                points = np.column_stack([
                    qx_con, qy_con, qz_con
                ])
                return (points, flat_intensity, num_of_images)

        # Prevent crashing by checking file exists, not empty, etc
        # and return the empty points and 0 images
        except Exception as e:
            print('Error',e)
            return (points, flat_intensity, num_of_images)

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
        print(f"Changing color to {color_map_select}")
        
        try:
            # Create new lookup table
            new_lut = pyv.LookupTable(cmap=color_map_select)
            self.lut = new_lut
            
            # Update all actors in plotter.actors (PyVista managed)
            for actor_name, actor in self.plotter.actors.items():
                if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'lookup_table'):
                    print(f"Updating LUT for actor: {actor_name}")
                    actor.mapper.lookup_table = new_lut
                    actor.mapper.update()
            
            # Update all actors in renderer._actors (VTK managed)
            for actor_name, actor in self.plotter.renderer._actors.items():
                # Check if it's a volume or other VTK actor with LUT
                if hasattr(actor, 'GetMapper') and actor.GetMapper():
                    mapper = actor.GetMapper()
                    if hasattr(mapper, 'SetLookupTable'):
                        print(f"Updating VTK LUT for actor: {actor_name}")
                        mapper.SetLookupTable(new_lut)
                        mapper.Update()
                
                # Check for volume actors (different property structure)
                elif hasattr(actor, 'GetProperty') and hasattr(actor.GetProperty(), 'SetLookupTable'):
                    print(f"Updating Volume LUT for actor: {actor_name}")
                    actor.GetProperty().SetLookupTable(new_lut)
            
            # Force render
            self.plotter.render()
            
        except Exception as e:
            print(f"Error changing colormap: {e}")


    # ================ CLEANUP METHODS ======================= #
    def closeEvent(self, event):
        """Clean shutdown of 3D viewer without affecting parent"""
        try:
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
                
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        event.accept()



if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = ConfigDialog()
        size_manager = SizeManager(app=app)
        window.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        sys.exit(0)