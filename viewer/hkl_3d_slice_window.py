import sys
import numpy as np
import pyvista as pyv
from pyvistaqt import QtInteractor
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import settings
from utils import SizeManager, RSMConverter
from utils.hdf5_loader import HDF5Loader


class HKL3DSliceWindow(QMainWindow):
    """3D Slice window (point-only viewer).

    Point-only rendering with plane slicing:
      - No volume/grid interpolation; pure point-cloud rendering
      - Plane-based slice using a tolerance around the plane
      - Axes, intensity range sliders, camera presets, colormap selection
      - Data loading via RSMConverter.load_h5_to_3d
      - Extract slice saves slice points projected to a 2D slice dataset
    """

    def __init__(self, parent=None):
        super(HKL3DSliceWindow, self).__init__()
        self.parent = parent
        uic.loadUi('gui/hkl_3d_slice_window.ui', self)

        # Initial UI availability
        try:
            self.actionSave.setEnabled(False)  # volume-save removed
            self.actionExtractSlice.setEnabled(False)
            self._set_slice_controls_enabled(False)
        except Exception:
            pass
        self.setWindowTitle('3D Slice')
        pyv.set_plot_theme('dark')

        # Hook up controls
        try:
            if hasattr(self, 'actionSlice'):
                self.actionSlice.triggered.connect(lambda: self.open_controls_dialog(focus='slice'))
        except Exception:
            pass

        # Toggles and controls (align naming/behavior with viewer/hkl_3d.py)
        self.cbToggleSlicePointer.clicked.connect(self.toggle_pointer)
        if hasattr(self, 'cbTogglePoints'):
            try:
                self.cbTogglePoints.clicked.connect(self.toggle_cloud_vol)
            except Exception:
                pass
        if hasattr(self, 'cbLockSlice'):
            try:
                self.cbLockSlice.clicked.connect(self.toggle_slice_lock)
            except Exception:
                pass
        self.cbColorMapSelect.currentIndexChanged.connect(self.change_color_map)
        self.sbMinIntensity.editingFinished.connect(self.update_intensity)
        self.sbMaxIntensity.editingFinished.connect(self.update_intensity)

        # Actions
        self.actionLoadData.triggered.connect(self.load_data)
        self.actionExtractSlice.triggered.connect(self.extract_slice)

        # State
        self.cloud_mesh = None
        # Actor handles follow naming used in viewer/hkl_3d.py
        self.points_actor = None       # actor for the full cloud (name: "cloud_volume")
        self.slab = None               # extracted slice points
        self.slab_actor = None         # actor for slice points (name: "slab_points")
        self._plane_widget = None
        self._slice_locked = False
        self._plane_normal = None
        self._plane_origin = None
        self._slice_lock_text_actor = None
        self.orig_shape = (0, 0)
        self.curr_shape = (0, 0)
        self.num_images = 0
        self.current_file_path = None

        # Slice/camera state
        self._slice_translate_step = 0.01
        self._slice_rotate_step_deg = 1.0
        self._zoom_step = 1.5
        self._cam_pos_selection = None
        self._slice_orientation_selection = None
        self._custom_normal = [0.0, 0.0, 1.0]

        # LUTs
        self.lut = pyv.LookupTable(cmap='jet')
        self.lut.apply_opacity([0, 1])
        self.lut2 = pyv.LookupTable(cmap='jet')
        self.lut2.apply_opacity([0, 1])

        # Plotter
        self.plotter = QtInteractor()
        try:
            self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L', x_color='red', y_color='green', z_color='blue')
        except Exception:
            self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plotter.setMinimumSize(300, 300)
        self.viewer_3d_slicer_layout.addWidget(self.plotter, 1, 1)

        # Wire slice/camera controls from UI
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

    # Dialogs
    def open_controls_dialog(self, focus=None):
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

    # Availability
    def _is_data_loaded(self) -> bool:
        return bool(self.cloud_mesh is not None)

    def _slice_points_exist(self) -> bool:
        try:
            return (self.slab is not None) and (getattr(self.slab, 'n_points', 0) > 0)
        except Exception:
            return False

    def _set_slice_controls_enabled(self, enabled: bool):
        try:
            for wname in ('gbSteps', 'gbOrientation', 'gbTranslate', 'gbRotate'):
                w = getattr(self, wname, None)
                if w:
                    w.setEnabled(bool(enabled))
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
            slice_points_exist = self._slice_points_exist()
            if hasattr(self, 'actionExtractSlice'):
                self.actionExtractSlice.setEnabled(bool(slice_points_exist))
            self._set_slice_controls_enabled(bool(data_loaded))
        except Exception:
            pass

    # Data setup and create scene
    def setup_3d_cloud(self, cloud, intensity, shape):
        if cloud is None or (isinstance(cloud, np.ndarray) and cloud.size == 0):
            self.cloud_mesh = None
            return False
        if isinstance(cloud, np.ndarray):
            self.cloud_mesh = pyv.PolyData(cloud)
            self.cloud_mesh['intensity'] = intensity
        else:
            self.cloud_mesh = cloud.copy(deep=True) if hasattr(cloud, 'copy') else cloud
            self.cloud_mesh['intensity'] = intensity
        self.orig_shape = shape
        self.curr_shape = shape
        return True

    def create_3D(self, cloud=None, intensity=None):
        # Clear previous actors/widgets but keep axes stable
        try:
            self.plotter.clear()
            for name in ("cloud_volume", "slab_points", "origin_sphere", "normal_line"):
                if name in getattr(self.plotter, 'actors', {}):
                    self.plotter.remove_actor(name, reset_camera=False)
        except Exception:
            pass

        # Cloud
        self.cloud_mesh = pyv.PolyData(cloud)
        self.cloud_mesh['intensity'] = intensity
        self.lut.scalar_range = (float(np.min(intensity)), float(np.max(intensity)))
        self.lut2.scalar_range = (float(np.min(intensity)), float(np.max(intensity)))

        # Points actor (use naming "cloud_volume" like viewer/hkl_3d.py)
        self.points_actor = self.plotter.add_mesh(
            self.cloud_mesh,
            scalars='intensity',
            cmap=self.lut,
            point_size=5.0,
            name='cloud_volume',
            reset_camera=False,
            show_edges=False,
            show_scalar_bar=True,
        )

        # Bounds/axes
        try:
            self.plotter.show_bounds(
                mesh=self.cloud_mesh,
                xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                ticks='inside', minor_ticks=True,
                n_xlabels=7, n_ylabels=7, n_zlabels=7,
                x_color='red', y_color='green', z_color='blue',
                font_size=20,
            )
        except Exception:
            try:
                self.plotter.show_bounds(mesh=self.cloud_mesh, xtitle='H Axis', ytitle='K Axis', ztitle='L Axis')
            except Exception:
                pass

        # Plane widget
        slice_normal = (0, 0, 1)
        slice_origin = self.cloud_mesh.center
        self._plane_widget = self.plotter.add_plane_widget(
            callback=self.on_plane_update,
            normal=slice_normal,
            origin=slice_origin,
            bounds=self.cloud_mesh.bounds,
            factor=1.0,
            implicit=True,
            assign_to_axis=None,
            tubing=False,
            origin_translation=True,
            outline_opacity=0,
        )
        # Initialize stored plane state
        try:
            self._plane_normal = np.array(slice_normal, dtype=float)
            self._plane_origin = np.array(slice_origin, dtype=float)
        except Exception:
            self._plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self._plane_origin = np.array(self.cloud_mesh.center, dtype=float) if self.cloud_mesh is not None else np.array([0.0, 0.0, 0.0], dtype=float)

        # Ensure slice lock overlay exists and is hidden initially
        try:
            self._ensure_slice_lock_text_actor()
            if self._slice_lock_text_actor is not None:
                self._slice_lock_text_actor.SetVisibility(False)
        except Exception:
            pass

        # Labels and sliders
        try:
            self.lbCurrentPointSizeNum.setText(str(len(cloud)))
            self.lbCurrentResolutionX.setText(str(self.curr_shape[0]))
            self.lbCurrentResolutionY.setText(str(self.curr_shape[1]))
        except Exception:
            pass
        try:
            imin, imax = int(np.min(intensity)), int(np.max(intensity))
            self.sbMinIntensity.setRange(imin, imax)
            self.sbMinIntensity.setValue(imin)
            self.sbMaxIntensity.setRange(imin, imax)
            self.sbMaxIntensity.setValue(imax)
        except Exception:
            pass

        self.update_intensity()
        try:
            self.update_info_slice_labels()
            self._refresh_availability()
        except Exception:
            pass

        # Camera
        try:
            self.plotter.set_focus(self.cloud_mesh.center)
            self.plotter.reset_camera()
        except Exception:
            pass

    def _remove_plane_widget(self):
        """Safely remove existing plane widget (if any)."""
        try:
            if self._plane_widget is not None:
                try:
                    self._plane_widget.EnabledOff()
                except Exception:
                    pass
                try:
                    self.plotter.clear_plane_widgets()
                except Exception:
                    pass
                self._plane_widget = None
        except Exception:
            pass

    # Loading
    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select an HDF5 File', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not file_name:
            try:
                QMessageBox.warning(self, 'File', 'No Valid File Selected')
            except Exception:
                pass
            return
        self.current_file_path = file_name
        # reflect file path in UI if present
        try:
            if hasattr(self, 'leFilePathStr') and self.leFilePathStr is not None:
                self.leFilePathStr.setText(file_name)
        except Exception:
            pass

        original_title = self.windowTitle()
        self.setEnabled(False)
        self.setWindowTitle(f"{original_title} ***** Loading...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        # Hard reset interactive widgets and prior actors
        self._remove_plane_widget()
        try:
            for name in ("cloud_volume", "slab_points", "origin_sphere", "normal_line"):
                if name in getattr(self.plotter, 'actors', {}):
                    self.plotter.remove_actor(name, reset_camera=False)
        except Exception:
            pass
        try:
            conv = RSMConverter()
            if not conv.hkl_config:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'No Configuration', 'No configuration loaded. Please select a profile or TOML in the Workflow dialog.')
                return
            points, intensities, num_images, shape = conv.load_h5_to_3d(file_name)
            if points.size == 0 or intensities.size == 0:
                QMessageBox.warning(self, 'Loading Warning', 'No valid point data found in HDF5 file')
                return
            if self.setup_3d_cloud(points, intensities, shape):
                self.num_images = num_images
                self.create_3D(cloud=points, intensity=intensities)
                try:
                    self.groupBox3DViewer.setTitle(f'Viewing {num_images} Image(s)')
                    self.lbOriginalPointSizeNum.setText(str(len(points)))
                    self.lbOriginalResolutionX.setText(str(shape[0]))
                    self.lbOriginalResolutionY.setText(str(shape[1]))
                    # reflect current shape
                    self.curr_shape = shape
                except Exception:
                    pass
            try:
                self.update_info_slice_labels()
                self._refresh_availability()
            except Exception:
                pass
        except Exception as e:
            import traceback
            error_msg = f"Error loading data: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            try:
                QMessageBox.critical(self, 'Error Loading Data', error_msg)
            except Exception:
                pass
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self.setWindowTitle(original_title)

    # Slice points extraction
    def on_plane_update(self, normal, origin):
        # If slice is locked, immediately restore widget to stored state and do not update slice
        try:
            if bool(getattr(self, '_slice_locked', False)):
                self._restore_locked_plane_widget()
                return
        except Exception:
            pass
        if self.cloud_mesh is None:
            return
        normal = self.normalize_vector(np.array(normal, dtype=float))
        origin = np.array(origin, dtype=float)

        # Compute slice mask
        vec = self.cloud_mesh.points - origin
        dist = np.dot(vec, normal)
        thickness = 0.002
        mask = np.abs(dist) < thickness

        # Extract masked points
        try:
            self.slab = self.cloud_mesh.extract_points(mask)
        except Exception:
            self.slab = None

        # Remove previous slice points actor if present
        try:
            if 'slab_points' in getattr(self.plotter, 'actors', {}):
                self.plotter.remove_actor('slab_points', reset_camera=False)
        except Exception:
            pass

        # Add new slice points actor (name: slab_points)
        if self.slab is not None and getattr(self.slab, 'n_points', 0) > 0:
            self.slab_actor = self.plotter.add_mesh(
                self.slab,
                name='slab_points',
                render_points_as_spheres=True,
                point_size=10,
                scalars='intensity',
                cmap=self.lut2,
                show_scalar_bar=False,
            )
        else:
            self.slab_actor = None

        # Sync plane widget
        try:
            if self._plane_widget is not None:
                self._plane_widget.SetNormal(normal)
                self._plane_widget.SetOrigin(origin)
        except Exception:
            pass

        # Update labels and render
        try:
            self.update_info_slice_labels()
            if hasattr(self, 'lbInfoPointsCurrVal'):
                try:
                    n_curr = int(getattr(self.slab, 'n_points', 0) or 0)
                except Exception:
                    n_curr = 0
                self.lbInfoPointsCurrVal.setText(str(n_curr))
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass
        self._refresh_availability()

        # Update stored plane state
        try:
            self._plane_normal = np.array(normal, dtype=float)
            self._plane_origin = np.array(origin, dtype=float)
        except Exception:
            pass

    # Intensity and colormap updates
    def update_intensity(self):
        try:
            min_i = self.sbMinIntensity.value()
            max_i = self.sbMaxIntensity.value()
        except Exception:
            return
        if min_i > max_i:
            min_i, max_i = max_i, min_i
            try:
                self.sbMinIntensity.setValue(min_i)
                self.sbMaxIntensity.setValue(max_i)
            except Exception:
                pass
        new_range = (float(min_i), float(max_i))

        # Update points actor
        try:
            if self.points_actor is not None:
                self.points_actor.mapper.scalar_range = new_range
        except Exception:
            pass
        # Update slice points actor
        try:
            if self.slab_actor is not None:
                self.slab_actor.mapper.scalar_range = new_range
        except Exception:
            pass
        # Update scalar bars
        try:
            if hasattr(self.plotter, 'scalar_bars'):
                for _, sb in self.plotter.scalar_bars.items():
                    if sb:
                        sb.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                        sb.Modified()
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass

        # Maintain visibility based on toggles (match viewer/hkl_3d.py behavior)
        try:
            self.toggle_cloud_vol()
        except Exception:
            pass
        try:
            self.toggle_pointer()
        except Exception:
            pass

    def change_color_map(self):
        color_map_select = self.cbColorMapSelect.currentText()
        new_lut = pyv.LookupTable(cmap=color_map_select)
        new_lut2 = pyv.LookupTable(cmap=color_map_select)
        new_lut.apply_opacity([0, 1])
        new_lut2.apply_opacity([0, 1])
        self.lut = new_lut
        self.lut2 = new_lut2

        try:
            if self.points_actor is not None:
                self.points_actor.mapper.lookup_table = self.lut
            if self.slab_actor is not None:
                self.slab_actor.mapper.lookup_table = self.lut2
        except Exception:
            pass
        try:
            if hasattr(self.plotter, 'scalar_bars'):
                for _, sb in self.plotter.scalar_bars.items():
                    if sb:
                        sb.SetLookupTable(self.lut)
                        sb.Modified()
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass

    # Toggles
    def toggle_pointer(self):
        """Toggle visibility of the slice points and plane widget (like viewer/hkl_3d.py)."""
        vis = True
        try:
            vis = bool(self.cbToggleSlicePointer.isChecked())
        except Exception:
            pass
        try:
            if 'slab_points' in getattr(self.plotter, 'actors', {}):
                self.plotter.renderer._actors['slab_points'].SetVisibility(vis)
        except Exception:
            pass
        try:
            widgets = getattr(self.plotter, 'plane_widgets', [])
            for pw in widgets or []:
                try:
                    if vis:
                        pw.EnabledOn()
                    else:
                        pw.EnabledOff()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def toggle_cloud_vol(self):
        """Toggle visibility of the full cloud actor (named 'cloud_volume')."""
        vis = True
        try:
            # Use cbTogglePoints from this UI to drive cloud visibility
            vis = bool(self.cbTogglePoints.isChecked())
        except Exception:
            pass
        try:
            if 'cloud_volume' in getattr(self.plotter, 'actors', {}):
                self.plotter.renderer._actors['cloud_volume'].SetVisibility(vis)
        except Exception:
            pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def _ensure_slice_lock_text_actor(self):
        try:
            if self._slice_lock_text_actor is None:
                self._slice_lock_text_actor = self.plotter.add_text(
                    "Slice Locked",
                    position='upper_left',
                    font_size=16,
                    color='white'
                )
        except Exception:
            self._slice_lock_text_actor = None

    def toggle_slice_lock(self):
        # Update lock state
        try:
            self._slice_locked = bool(self.cbLockSlice.isChecked())
        except Exception:
            self._slice_locked = not bool(getattr(self, '_slice_locked', False))
        # Ensure overlay exists
        self._ensure_slice_lock_text_actor()
        # Set overlay visibility
        try:
            if self._slice_lock_text_actor is not None:
                self._slice_lock_text_actor.SetVisibility(bool(self._slice_locked))
        except Exception:
            pass
        # If we just locked, restore widget to stored plane state
        if bool(self._slice_locked):
            self._restore_locked_plane_widget()
        try:
            self.plotter.render()
        except Exception:
            pass

    def _restore_locked_plane_widget(self):
        try:
            if (self._plane_widget is not None) and (self._plane_normal is not None) and (self._plane_origin is not None):
                self._plane_widget.SetNormal(np.array(self._plane_normal, dtype=float))
                self._plane_widget.SetOrigin(np.array(self._plane_origin, dtype=float))
        except Exception:
            pass

    # (Removed) Volume toggle: no volume actor in point-only slice window

    # Extract slice (save slice points as 2D slice dataset)
    def extract_slice(self):
        if not self._slice_points_exist():
            try:
                QMessageBox.warning(self, 'No Slice', 'No slice points available to extract')
            except Exception:
                pass
            return
        default_name = f"slice_extract_{np.datetime64('now').astype('datetime64[s]').astype(str).replace(':', '-')}.h5"
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save hkl Slice Data', default_name, 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not file_path:
            return

        # Gather slice points and intensities
        try:
            # Use self.slab for consistency
            slice_points = np.array(self.slab.points)
            slice_intensities = np.array(self.slab['intensity'])
        except Exception:
            try:
                QMessageBox.critical(self, 'Extract Error', 'Failed to read slice points/intensity')
            except Exception:
                pass
            return

        # Plane state
        normal, origin = self.get_plane_state()

        # Metadata
        slice_metadata = {
            'data_type': 'slice',
            'slice_normal': list(map(float, self.normalize_vector(np.array(normal, dtype=float)))) ,
            'slice_origin': list(map(float, np.array(origin, dtype=float))),
            'num_points': int(len(slice_points)),
            'original_file': str(self.current_file_path or 'unknown'),
            'original_shape': list(map(int, self.orig_shape)) if isinstance(self.orig_shape, (tuple, list)) else [0, 0],
            'extraction_timestamp': str(np.datetime64('now')),
        }

        # Save via HDF5Loader
        try:
            loader = HDF5Loader()
            success = loader.extract_slice(
                file_path=file_path,
                points=slice_points,
                intensities=slice_intensities,
                metadata=slice_metadata,
                shape=self.orig_shape if isinstance(self.orig_shape, (tuple, list)) else None,
            )
            if success:
                try:
                    QMessageBox.information(self, 'Success', f'Slice extracted and saved successfully!\n{len(slice_points)} points saved.')
                except Exception:
                    pass
            else:
                try:
                    QMessageBox.critical(self, 'Error', f'Failed to save slice: {loader.get_last_error()}')
                except Exception:
                    pass
        except Exception as e:
            try:
                QMessageBox.critical(self, 'Extract Error', f'Error extracting slice: {str(e)}')
            except Exception:
                pass

    # Camera controls
    def zoom_in(self):
        cam = self.plotter.camera
        try:
            step = float(self._zoom_step)
            if not np.isfinite(step) or step <= 1.0:
                step = 1.5
        except Exception:
            step = 1.5
        cam.zoom(step)
        self.plotter.render()

    def zoom_out(self):
        cam = self.plotter.camera
        try:
            step = float(self._zoom_step)
            if not np.isfinite(step) or step <= 1.0:
                step = 1.5
        except Exception:
            step = 1.5
        cam.zoom(1.0 / step)
        self.plotter.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def set_camera_position(self):
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

        def _set_focus_to_data_center():
            try:
                if self.cloud_mesh is not None and hasattr(self.cloud_mesh, 'center'):
                    p.set_focus(self.cloud_mesh.center)
            except Exception:
                pass

        if ('xy' in pos_text) or ('hk' in pos_text):
            _set_focus_to_data_center(); p.view_xy()
        elif ('yz' in pos_text) or ('kl' in pos_text):
            _set_focus_to_data_center(); p.view_yz()
        elif ('xz' in pos_text) or ('hl' in pos_text):
            _set_focus_to_data_center(); p.view_xz()
        elif 'iso' in pos_text:
            _set_focus_to_data_center()
            try:
                p.view_isometric()
            except Exception:
                try:
                    p.view_vector((1.0, 1.0, 1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                except Exception:
                    pass
        else:
            _set_focus_to_data_center()
            label = (pos_text or '')
            try:
                if ('h+' in label) or ('x+' in label):
                    p.view_vector((1.0, 0.0, 0.0)); cam.view_up = (0.0, 0.0, 1.0)
                elif ('h-' in label) or ('x-' in label):
                    p.view_vector((-1.0, 0.0, 0.0)); cam.view_up = (0.0, 0.0, 1.0)
                elif ('k+' in label) or ('y+' in label):
                    p.view_vector((0.0, 1.0, 0.0)); cam.view_up = (0.0, 0.0, 1.0)
                elif ('k-' in label) or ('y-' in label):
                    p.view_vector((0.0, -1.0, 0.0)); cam.view_up = (0.0, 0.0, 1.0)
                elif ('l+' in label) or ('z+' in label):
                    p.view_vector((0.0, 0.0, 1.0)); cam.view_up = (0.0, 1.0, 0.0)
                elif ('l-' in label) or ('z-' in label):
                    p.view_vector((0.0, 0.0, -1.0)); cam.view_up = (0.0, 1.0, 0.0)
            except Exception:
                pass
        try:
            if cam is not None and hasattr(cam, 'orthogonalize_view_up'):
                cam.orthogonalize_view_up()
        except Exception:
            pass
        try:
            p.render()
        except Exception:
            pass

    def _apply_cam_preset_button(self, label: str):
        try:
            self._cam_pos_selection = label
        except Exception:
            pass
        try:
            self.set_camera_position()
        except Exception:
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
        try:
            normal, origin = self.get_plane_state()
            normal = self.normalize_vector(np.array(normal, dtype=float))
            origin = np.array(origin, dtype=float)
            cam = getattr(self.plotter, 'camera', None)
            if cam is None:
                return
            try:
                rng = self.cloud_mesh.points.max(axis=0) - self.cloud_mesh.points.min(axis=0)
                distance = float(np.linalg.norm(rng)) * 0.5
            except Exception:
                distance = 1.0
            try:
                cam.focal_point = origin.tolist()
                cam.position = (origin + normal * distance).tolist()
                up = np.array(getattr(cam, 'view_up', [0.0, 1.0, 0.0]), dtype=float)
                upn = self.normalize_vector(up)
                if abs(float(np.dot(upn, normal))) > 0.99:
                    new_up = np.array([0.0, 1.0, 0.0], dtype=float) if abs(normal[1]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=float)
                    cam.view_up = new_up.tolist()
            except Exception:
                pass
            self.plotter.render()
        except Exception:
            pass

    # Slice control helpers
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
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
        if not self._ensure_data_loaded_or_warn():
            return
        preset = (str(getattr(self, '_slice_orientation_selection', '')) or '').lower()
        if preset.startswith('custom'):
            try:
                n_raw = np.array(getattr(self, '_custom_normal', [0.0, 0.0, 1.0]), dtype=float)
            except Exception:
                n_raw = np.array([0.0, 0.0, 1.0], dtype=float)
            n = self.normalize_vector(n_raw)
            _, origin = self.get_plane_state()
            self.on_plane_update(n, origin)

    def _on_reset_slice(self):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
        if not self._ensure_data_loaded_or_warn():
            return
        try:
            center = self.cloud_mesh.center if (self.cloud_mesh is not None) else np.array([0.0, 0.0, 0.0])
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self.on_plane_update(normal, center)
        except Exception:
            pass

    # Plane helpers
    def get_plane_state(self):
        # Prefer stored state once initialized for consistency under lock
        try:
            if (self._plane_normal is not None) and (self._plane_origin is not None):
                return np.array(self._plane_normal, dtype=float), np.array(self._plane_origin, dtype=float)
        except Exception:
            pass
        try:
            if hasattr(self.plotter, 'plane_widgets') and self.plotter.plane_widgets:
                pw = self.plotter.plane_widgets[0]
                normal = np.array(pw.GetNormal(), dtype=float)
                origin = np.array(pw.GetOrigin(), dtype=float)
                return normal, origin
        except Exception:
            pass
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        try:
            origin = np.array(self.cloud_mesh.center, dtype=float)
        except Exception:
            origin = np.array([0.0, 0.0, 0.0], dtype=float)
        return normal, origin

    def set_plane_state(self, normal, origin):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
        n = self.normalize_vector(np.array(normal, dtype=float))
        o = np.array(origin, dtype=float)
        self.on_plane_update(n, o)

    def normalize_vector(self, v):
        v = np.array(v, dtype=float)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return v / n

    def set_plane_preset(self, preset_text: str):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
        preset = preset_text.lower()
        if 'xy' in preset or 'hk' in preset:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        elif 'yz' in preset or 'kl' in preset:
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif 'xz' in preset or 'hl' in preset:
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            try:
                n = np.array(getattr(self, '_custom_normal', [0.0, 0.0, 1.0]), dtype=float)
            except Exception:
                n = np.array([0.0, 0.0, 1.0], dtype=float)
            n = self.normalize_vector(n)
        _, origin = self.get_plane_state()
        self.set_plane_state(n, origin)

    def nudge_along_normal(self, sign: int):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
        if not self._ensure_data_loaded_or_warn():
            return
        normal, origin = self.get_plane_state()
        step = float(self._slice_translate_step)
        origin_new = origin + float(sign) * step * normal
        self.set_plane_state(normal, origin_new)

    def nudge_along_axis(self, axis: str, sign: int):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
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
        self.set_plane_state(normal, origin_new)

    def rotate_about_axis(self, axis: str, deg: float):
        # Guard: do nothing if slice is locked
        if bool(getattr(self, '_slice_locked', False)):
            self._restore_locked_plane_widget()
            return
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

    # Info labels
    def update_info_slice_labels(self):
        try:
            orient_text = getattr(self, '_slice_orientation_selection', None)
            if not orient_text:
                try:
                    if hasattr(self, 'cbSliceOrientation') and self.cbSliceOrientation is not None:
                        orient_text = self.cbSliceOrientation.currentText()
                except Exception:
                    orient_text = '-'
            if orient_text is None or orient_text == '':
                orient_text = '-'
            normal, origin = self.get_plane_state()
            n = self.normalize_vector(np.array(normal, dtype=float))
            o = np.array(origin, dtype=float)
            # Display floats with 5 decimal places
            n_str = f"[{n[0]:0.5f}, {n[1]:0.5f}, {n[2]:0.5f}]"
            o_str = f"[{o[0]:0.5f}, {o[1]:0.5f}, {o[2]:0.5f}]"
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
            try:
                pos_text = '-'
                orient_lower = (str(orient_text) or '').lower()
                if ('hk' in orient_lower) or ('xy' in orient_lower):
                    pos_text = f"L = {o[2]:0.5f}"
                elif ('kl' in orient_lower) or ('yz' in orient_lower):
                    pos_text = f"H = {o[0]:0.5f}"
                elif ('hl' in orient_lower) or ('xz' in orient_lower):
                    pos_text = f"K = {o[1]:0.5f}"
                else:
                    s = float(np.dot(n, o))
                    pos_text = f"n·origin = {s:0.5f}"
                if hasattr(self, 'lbSlicePositionVal'):
                    self.lbSlicePositionVal.setText(pos_text)
            except Exception:
                pass
            # Reflect availability of Extract action based on existence
            try:
                if hasattr(self, 'actionExtractSlice'):
                    self.actionExtractSlice.setEnabled(self._slice_points_exist())
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_data_loaded_or_warn(self) -> bool:
        try:
            if self.cloud_mesh is not None:
                return True
        except Exception:
            pass
        try:
            QMessageBox.warning(self, 'No Data', 'Load data before adjusting the slice.')
        except Exception:
            pass
        return False


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
