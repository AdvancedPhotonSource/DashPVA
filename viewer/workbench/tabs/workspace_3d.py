from typing import Optional
import os
from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QMessageBox, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt, QThread
import numpy as np
import pyvista as pv


# Import BaseTab using existing tabs package alias
from tabs.base_tab import BaseTab

# Import 3D visualization components
try:
    import pyvista as pyv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Worker for off-UI-thread 3D prep
from viewer.workbench.workers import Render3D
from utils.hdf5_loader import HDF5Loader, discover_hkl_axis_labels
from utils.rsm_converter import RSMConverter

class Workspace3D(BaseTab):
    """
    3D Tab encapsulating 3D viewer setup, loading, and plotting operations.
    Delegates UI widget access via main_window, but centralizes 3D actions here.
    """
    def __init__(self, parent=None, main_window=None, title="3D View"):
        pv.set_plot_theme('dark')
        try: 
            super().__init__(ui_file='gui/workbench/tabs/tab_3d.ui', parent=parent, main_window=main_window, title=title)
            self.title = title
            self.main_window = main_window 
            self.build()
            self.connect_all()
        except Exception as e:
            print(e)

    def connect_all(self):
        """Wire up 3D controls to main window handlers."""
        try:
            self.btn_load_3d_data.clicked.connect(self.load_data)
            self.cb_colormap_3d.currentTextChanged.connect(self.on_3d_colormap_changed)
            self.cb_show_points.toggled.connect(self.toggle_3d_points)
            self.cb_show_slice.toggled.connect(self.toggle_3d_slice)
            self.sb_min_intensity_3d.editingFinished.connect(self.update_intensity)
            self.sb_max_intensity_3d.editingFinished.connect(self.update_intensity)
        except Exception as e:
            try:
                self.main_window.update_status(f"Error setting up 3D connections: {e}")
            except Exception:
                pass
    
    def build(self):
        # Try to create VTK QtInteractor; fall back if unavailable
        try:
            self.plotter = QtInteractor(self)
        except Exception:
            self.plotter = None

        # HKL labels derived from metadata/HKL
        try:
            self.hkl_info_label = QLabel("HKL Motors: -")
            # Insert above plotter/placeholder
            self.container.insertWidget(0, self.hkl_info_label)
        except Exception:
            self.hkl_info_label = None

        if self.plotter is None:
            placeholder = QLabel("3D (VTK) unavailable in tunnel mode.")
            try:
                placeholder.setAlignment(Qt.AlignCenter)
            except Exception:
                pass
            try:
                placeholder.setWordWrap(True)
            except Exception:
                pass
            try:
                self.container.insertWidget(1, placeholder, stretch=1)
            except Exception:
                pass
            # Disable 3D controls that would require the plotter
            for w in [getattr(self, "btn_load_3d_data", None),
                      getattr(self, "cb_show_points", None),
                      getattr(self, "cb_show_slice", None),
                      getattr(self, "sb_min_intensity_3d", None),
                      getattr(self, "sb_max_intensity_3d", None)]:
                try:
                    if w is not None:
                        w.setEnabled(False)
                except Exception:
                    pass
            # Initialize defaults
            self.cloud_mesh_3d = None
            self.slab_actor = None
            self.plane_widget = None
            self.lut = None
            self.lut2 = None
            # Default target raster shape (HxW) for slice rasterization
            self.orig_shape = (0, 0)
            self.curr_shape = (0, 0)
            # Slice & Camera defaults
            self._slice_translate_step = 0.01
            self._slice_rotate_step_deg = 1.0
            self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self._zoom_step = 1.5
            return

        # If plotter exists, proceed to embed and configure
        self.container.insertWidget(1, self.plotter, stretch=1)
        try:
            self.scrollArea_3d_controls.setMinimumWidth(280)
        except Exception:
            pass
        try:
            self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L', x_color='red', y_color='green', z_color='blue')
        except Exception:
            pass

        # Color axes: H=red (X), K=green (Y), L=blue (Z)
        try:
            ca = getattr(self.plotter.renderer, 'cube_axes_actor', None)
            if ca:
                ca.GetXAxesLinesProperty().SetColor(1.0, 0.0, 0.0)
                ca.GetYAxesLinesProperty().SetColor(0.0, 1.0, 0.0)
                ca.GetZAxesLinesProperty().SetColor(0.0, 0.0, 1.0)
                ca.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
                ca.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
                ca.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
                ca.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)
                ca.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)
                ca.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)
        except Exception:
            pass
        self.cloud_mesh_3d = None
        self.slab_actor = None
        self.plane_widget = None
        self.lut = None
        self.lut2 = None
        # Default target raster shape (HxW) for slice rasterization
        self.orig_shape = (0, 0)
        self.curr_shape = (0, 0)
        # Slice & Camera defaults
        self._slice_translate_step = 0.01
        self._slice_rotate_step_deg = 1.0
        self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)
        self._zoom_step = 1.5

        
    def setup_plot_viewer(self):
        """
        Create and embed a PyVista QtInteractor into the 3D tab container.
        """
        mw = self.main_window
        try:
            if not PYVISTA_AVAILABLE:
                return
            pyv.set_plot_theme('dark')
            mw.plotter_3d = QtInteractor()
            mw.plotter_3d.add_axes(xlabel='H', ylabel='K', zlabel='L', x_color='red', y_color='green', z_color='blue')
            if hasattr(mw, 'layout3DPlotHost') and mw.layout3DPlotHost is not None:
                try:
                    # layout3DPlotHost may be a grid layout from the UI
                    mw.layout3DPlotHost.addWidget(mw.plotter_3d, 0, 0)
                except Exception:
                    mw.layout3DPlotHost.addWidget(mw.plotter_3d)
            else:
                print("Warning: layout3DPlotHost not found, 3D plot may not display correctly")
            # Info dock
            try:
                self._setup_info_dock()
            except Exception:
                pass
            # Clear initial state
            self.clear_plot()
        except Exception as e:
            try:
                mw.update_status(f"Error setting up 3D plot viewer: {e}")
            except Exception:
                pass

    def _setup_info_dock(self):
        """Create a small 3D Info dock to display render metrics (e.g., render time)."""
        mw = self.main_window
        try:
            mw.three_d_info_dock = QDockWidget("3D Info", mw)
            mw.three_d_info_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
            container = QWidget(mw.three_d_info_dock)
            layout = QVBoxLayout(container)
            mw.three_d_info_label = QLabel("Render time: - ms")
            layout.addWidget(mw.three_d_info_label)
            mw.three_d_info_dock.setWidget(container)
            mw.addDockWidget(Qt.RightDockWidgetArea, mw.three_d_info_dock)
            try:
                mw.add_dock_toggle_action(mw.three_d_info_dock, "3D Info", segment_name="3d")
            except Exception:
                pass
        except Exception as e:
            try:
                mw.update_status(f"Error setting up 3D info dock: {e}")
            except Exception:
                pass

    def toggle_3d_points(self, checked: bool):
        """Shows/Hides the main HKL point cloud."""
        if "points" in self.plotter.actors:
            self.plotter.actors["points"].SetVisibility(checked)
            self.plotter.render()

    def toggle_3d_slice(self, checked: bool):
        """Shows/Hides the interactive plane and the extracted slice points."""
        # Toggle the points extracted by the plane
        if "slab_points" in self.plotter.actors:
            self.plotter.actors["slab_points"].SetVisibility(checked)
        
        # Toggle the white plane widget tool
        if self.plane_widget:
            if checked:
                self.plane_widget.On()
            else:
                self.plane_widget.Off()
        
        self.plotter.render()

    def on_3d_colormap_changed(self):
        """Apply selected colormap to the points (and slab if available)."""
        try:
            cmap_name = getattr(self.cb_colormap_3d, 'currentText', lambda: 'viridis')()
        except Exception:
            cmap_name = 'viridis'
        self.lut = pv.LookupTable(cmap=cmap_name)
        self.lut.apply_opacity([0,1])


        # Update points colormap by re-adding the mesh with new cmap
        try:
            if "points" in self.plotter.actors and self.cloud_mesh_3d is not None:
                try:
                    self.plotter.remove_actor("points", reset_camera=False)
                except Exception:
                    pass
                self.points_actor = self.plotter.add_mesh(
                    self.cloud_mesh_3d,
                    scalars='intensity',
                    cmap=self.lut,
                    point_size=5.0,
                    name='points',
                    show_scalar_bar=True,
                    reset_camera=False,
                )
                # Reapply current intensity range
                try:
                    rng = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                    self.points_actor.mapper.scalar_range = rng
                except Exception:
                    pass
                # Respect checkbox visibility
                try:
                    self.points_actor.SetVisibility(bool(self.cb_show_points.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt to update slab colormap (best-effort)
        try:
            if self.slab_actor is not None:
                try:
                    import pyvista as pyv
                    lut = pyv.LookupTable(cmap=cmap_name)
                    # Some PyVista versions expose mapper.lookup_table, others require SetLookupTable
                    try:
                        self.slab_actor.mapper.lookup_table = lut
                    except Exception:
                        try:
                            self.slab_actor.GetMapper().SetLookupTable(lut)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Keep visibility consistent
                try:
                    self.slab_actor.SetVisibility(bool(self.cb_show_slice.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Render and ensure ranges/visibility remain in sync
        try:
            self.plotter.render()
        except Exception:
            pass
        try:
            self.update_intensity()
        except Exception:
            pass

    def update_info(self, render_ms: int):
        """Update the 3D info dock with render timing in milliseconds."""
        mw = self.main_window
        try:
            if hasattr(mw, 'three_d_info_label') and mw.three_d_info_label is not None:
                mw.three_d_info_label.setText(f"Render time: {int(render_ms)} ms")
        except Exception:
            pass

    # === Clear ===
    def clear_plot(self):
        try:
            if hasattr(self, 'plotter') and self.plotter is not None:
                self.plotter.clear()
                self.current_3d_data = None
                self.mesh = None
        except Exception as e:
            try:
                mw.update_status(f"Error clearing 3D plot: {e}")
            except Exception:
                pass

    # === Loading & Plotting ===
    def load_data(self):
        """Load dataset and render using the tab's local plotter."""
        print("Loading data into 3D viewer...")
        mw = self.main_window
        import time as _time
        start_all = _time.perf_counter()

        try:
            if not PYVISTA_AVAILABLE:
                QMessageBox.warning(self, "3D Viewer", "PyVista is not available.")
                return

            # 1. Get the file path
            file_path = getattr(mw, 'current_file_path', None) or getattr(mw, 'selected_dataset_path', None)
            if not file_path:
                file_name, _ = QFileDialog.getOpenFileName(
                    self, 'Select HDF5 or VTI File', '', 'HDF5 Files (*.h5 *.hdf5 *.vti);;All Files (*)'
                )
                if not file_name: return
                file_path = file_name
            # Update HKL label from metadata/HKL
            try:
                lbls = discover_hkl_axis_labels(file_path)
                txt = lbls.get('display_text', '') or "HKL Motors: -"
                if self.hkl_info_label is not None:
                    self.hkl_info_label.setText(txt)
            except Exception:
                pass
            conv = RSMConverter()
            # 2. Load the raw data
            # if the data is uncompressed
            data = conv.load_h5_to_3d(file_path)
            points, intensities, num_images, shape = data

            # 3. Define what happens when the worker finishes processing
            def _on_ready():
                try:
                    # IMPORTANT: Tell the worker to plot to THIS tab's plotter
                    # We pass 'self.plotter' instead of 'mw'
                    self._render3d_worker.plot_3d_points(self)
                    
                    # Switch to this tab automatically
                    if hasattr(mw, 'tabWidget_analysis'):
                        idx = mw.tabWidget_analysis.indexOf(self)
                        mw.tabWidget_analysis.setCurrentIndex(idx)
                    
                    self.main_window.update_status("3D Rendering Complete")
                except Exception as e:
                    print(f"Render Error: {e}")

            # 4. Threaded Execution
            self._render_thread = QThread(self)
            self._render3d_worker = Render3D(
                points=points, 
                intensities=intensities, 
                num_images=num_images, 
                shape=shape
            )
            
            self._render3d_worker.moveToThread(self._render_thread)
            
            # Connect signals
            self._render_thread.started.connect(self._render3d_worker.run)
            self._render3d_worker.render_ready.connect(_on_ready) # Use the local plotter
            
            # Cleanup
            self._render3d_worker.finished.connect(self._render_thread.quit)
            self._render3d_worker.finished.connect(self._render3d_worker.deleteLater)
            self._render_thread.finished.connect(self._render_thread.deleteLater)
            
            self._render_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "3D Viewer Error", f"Error: {str(e)}")
        finally:
            elapsed = int((_time.perf_counter() - start_all) * 1000)
            self.update_info(elapsed)

    def on_plane_update(self, normal, origin):
        """Extracts points near the plane to simulate a 3D slice."""
        if self.cloud_mesh_3d is None:
            return

        # Plane math: (Point - Origin) ⋅ Normal
        vec = self.cloud_mesh_3d.points - origin
        dist = np.dot(vec, normal)
        
        # Thickness of the slice in HKL units
        thickness = 0.005 
        mask = np.abs(dist) < thickness
        
        slab = self.cloud_mesh_3d.extract_points(mask)
        
        if slab.n_points > 0:
            self.slab_actor = self.plotter.add_mesh(
                slab, 
                name="slab_points", 
                render_points_as_spheres=True, 
                point_size=8, 
                scalars='intensity', 
                cmap=self.lut, 
                show_scalar_bar=False
            )
            # Ensure the new slab respects the current checkbox state
            self.slab_actor.SetVisibility(self.cb_show_slice.isChecked())
            
            # Match current intensity
            clim = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
            self.slab_actor.mapper.scalar_range = clim
            
        self.plotter.render()
        
        # Update the 3D Info dock with HKL slice information
        try:
            info_dock = getattr(self.main_window, 'info_3d_dock', None)
            if info_dock is not None:
                shape = None
                try:
                    shape = tuple(getattr(self, 'curr_shape', None) or ())
                    if not (isinstance(shape, tuple) and len(shape) == 2):
                        shape = (0, 0)
                except Exception:
                    shape = (0, 0)
                info_dock.update_from_slice(
                    slab,
                    np.asarray(normal, dtype=float),
                    np.asarray(origin, dtype=float),
                    target_shape=shape
                )
        except Exception:
            pass

    def update_intensity(self):
        """Updates the min/max intensity levels and scalar bar range"""
        if not self.plotter:
            return
        
        self.min_intensity = self.sb_min_intensity_3d.value()
        self.max_intensity = self.sb_max_intensity_3d.value()
        
        if self.min_intensity > self.max_intensity:
            self.min_intensity, self.max_intensity = self.max_intensity, self.min_intensity
            self.sb_min_intensity_3d.setValue(self.min_intensity)
            self.sb_max_intensity_3d.setValue(self.max_intensity)
        
        # Define the new scalar range
        new_range = [self.min_intensity, self.max_intensity]

        if "points" in self.plotter.actors:
            self.plotter.actors["points"].mapper.scalar_range = new_range

        if "slab_points" in self.plotter.actors:
            self.plotter.actors["slab_points"].mapper.scalar_range = new_range
            
        # Update the volume actor by re-adding with new clim range
        if hasattr(self.plotter, 'scalar_bars'):
            for bar in self.plotter.scalar_bars.values():
                bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])

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
        # Respect checkbox states after intensity update
        try:
            self.toggle_3d_points(self.cb_show_points.isChecked())
            self.toggle_3d_slice(self.cb_show_slice.isChecked())
        except Exception:
            pass

        # Update Info labels and availability after intensity changes
        try:
            self.update_info_slice_labels()
            self._refresh_availability()
        except Exception:
            pass

    # === Visibility & Colormap ===


    def reset_slice(self):
        """Reset slice to HK (xy) preset at the data center."""
        try:
            # Determine a reasonable center
            origin = None
            try:
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                    origin = np.array(self.cloud_mesh_3d.center, dtype=float)
                elif self.mesh is not None and hasattr(self.mesh, 'center'):
                    origin = np.array(self.mesh.center, dtype=float)
            except Exception:
                origin = None
            if origin is None:
                origin = np.array([0.0, 0.0, 0.0], dtype=float)
            # Normal along L for HK plane
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            self.set_plane_state(normal, origin)
        except Exception as e:
            try:
                self.main_window.update_status(f"Error resetting 3D slice: {e}")
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

    def toggle_pointer(self, checked: bool):
        """Enable/Disable the interactive plane widget and show/hide slab points."""
        try:
            # Plane widget visibility
            if self.plane_widget is not None:
                try:
                    if checked:
                        self.plane_widget.On()
                    else:
                        self.plane_widget.Off()
                except Exception:
                    pass
            else:
                # Fallback: use plotter.plane_widgets list if available
                widgets = getattr(self.plotter, "plane_widgets", [])
                for pw in widgets or []:
                    try:
                        if checked:
                            pw.EnabledOn()
                        else:
                            pw.EnabledOff()
                    except Exception:
                        pass
            # Slab points actor visibility
            if "slab_points" in self.plotter.actors:
                try:
                    self.plotter.actors["slab_points"].SetVisibility(bool(checked))
                except Exception:
                    try:
                        self.plotter.renderer._actors["slab_points"].SetVisibility(bool(checked))
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass

    # ===== Slice Plane helpers =====
    def get_plane_state(self):
        """Return (normal, origin) for current plane; defaults to Z-axis and mesh center."""
        try:
            if self.plane_widget is not None:
                try:
                    normal = np.array(self.plane_widget.GetNormal(), dtype=float)
                    origin = np.array(self.plane_widget.GetOrigin(), dtype=float)
                    return normal, origin
                except Exception:
                    pass
            # Fallback to first plane widget if present
            widgets = getattr(self.plotter, 'plane_widgets', [])
            if widgets:
                pw = widgets[0]
                try:
                    normal = np.array(pw.GetNormal(), dtype=float)
                    origin = np.array(pw.GetOrigin(), dtype=float)
                    return normal, origin
                except Exception:
                    pass
        except Exception:
            pass
        # Defaults
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
        try:
            if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                origin = np.array(self.cloud_mesh_3d.center, dtype=float)
            elif self.mesh is not None and hasattr(self.mesh, 'center'):
                origin = np.array(self.mesh.center, dtype=float)
            else:
                origin = np.array([0.0, 0.0, 0.0], dtype=float)
        except Exception:
            origin = np.array([0.0, 0.0, 0.0], dtype=float)
        return normal, origin

    def set_plane_state(self, normal, origin):
        """Programmatically set plane state and trigger slice update."""
        try:
            n = self.normalize_vector(np.array(normal, dtype=float))
            o = np.array(origin, dtype=float)
            # Update widget if available
            if self.plane_widget is not None:
                try:
                    self.plane_widget.SetNormal(n)
                    self.plane_widget.SetOrigin(o)
                except Exception:
                    pass
            else:
                widgets = getattr(self.plotter, 'plane_widgets', [])
                if widgets:
                    try:
                        widgets[0].SetNormal(n)
                        widgets[0].SetOrigin(o)
                    except Exception:
                        pass
            # Refresh slice
            try:
                self.on_plane_update(n, o)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def normalize_vector(v):
        try:
            v = np.array(v, dtype=float)
            norm = float(np.linalg.norm(v))
            if norm <= 0.0:
                return np.array([0.0, 0.0, 1.0], dtype=float)
            return v / norm
        except Exception:
            return np.array([0.0, 0.0, 1.0], dtype=float)

    def set_custom_normal(self, n):
        try:
            self._custom_normal = np.array(n, dtype=float)
        except Exception:
            self._custom_normal = np.array([0.0, 0.0, 1.0], dtype=float)

    def set_plane_preset(self, preset_text: str):
        """Set plane normal to preset HK/KL/HL or custom vector."""
        try:
            preset = (preset_text or '').lower()
        except Exception:
            preset = ''
        if ('xy' in preset) or ('hk' in preset):
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        elif ('yz' in preset) or ('kl' in preset):
            n = np.array([1.0, 0.0, 0.0], dtype=float)
        elif ('xz' in preset) or ('hl' in preset):
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            # Custom
            n = self.normalize_vector(getattr(self, '_custom_normal', np.array([0.0, 0.0, 1.0], dtype=float)))
        _, origin = self.get_plane_state()
        self.set_plane_state(n, origin)

    # ===== Translation =====
    def nudge_along_normal(self, sign: int):
        try:
            normal, origin = self.get_plane_state()
            step = float(getattr(self, '_slice_translate_step', 0.01))
            origin_new = origin + float(sign) * step * normal
            self.set_plane_state(normal, origin_new)
        except Exception:
            pass

    def nudge_along_axis(self, axis: str, sign: int):
        try:
            axis = (axis or 'H').upper()
            if axis == 'H':
                d = np.array([1.0, 0.0, 0.0], dtype=float)
            elif axis == 'K':
                d = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                d = np.array([0.0, 0.0, 1.0], dtype=float)
            normal, origin = self.get_plane_state()
            step = float(getattr(self, '_slice_translate_step', 0.01))
            origin_new = origin + float(sign) * step * d
            self.set_plane_state(normal, origin_new)
        except Exception:
            pass

    # ===== Rotation =====
    def rotate_about_axis(self, axis: str, deg: float):
        try:
            axis = (axis or 'H').upper()
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
        except Exception:
            pass

    # ===== Camera =====
    def zoom_in(self):
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if step <= 1.0:
                step = 1.5
            self.plotter.camera.zoom(step)
            self.plotter.render()
        except Exception:
            pass

    def zoom_out(self):
        try:
            step = float(getattr(self, '_zoom_step', 1.5))
            if step <= 1.0:
                step = 1.5
            self.plotter.camera.zoom(1.0 / step)
            self.plotter.render()
        except Exception:
            pass

    def reset_camera(self):
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def set_camera_position(self, preset: str):
        try:
            txt = (preset or '').strip().lower()
            p = self.plotter
            cam = getattr(p, 'camera', None)
            # center focus
            try:
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'center'):
                    p.set_focus(self.cloud_mesh_3d.center)
            except Exception:
                pass
            if txt in ('hk', 'xy'):
                p.view_xy()
            elif txt in ('kl', 'yz'):
                p.view_yz()
            elif txt in ('hl', 'xz'):
                p.view_xz()
            elif 'iso' in txt:
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
                # Axis-aligned
                if 'h+' in txt:
                    p.view_vector((1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'h-' in txt:
                    p.view_vector((-1.0, 0.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'k+' in txt:
                    p.view_vector((0.0, 1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'k-' in txt:
                    p.view_vector((0.0, -1.0, 0.0))
                    if cam is not None:
                        cam.view_up = (0.0, 0.0, 1.0)
                elif 'l+' in txt:
                    p.view_vector((0.0, 0.0, 1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
                elif 'l-' in txt:
                    p.view_vector((0.0, 0.0, -1.0))
                    if cam is not None:
                        cam.view_up = (0.0, 1.0, 0.0)
            try:
                if cam is not None and hasattr(cam, 'orthogonalize_view_up'):
                    cam.orthogonalize_view_up()
            except Exception:
                pass
            try:
                p.render()
            except Exception:
                pass
        except Exception:
            pass

    def view_slice_normal(self):
        try:
            normal, origin = self.get_plane_state()
            normal = self.normalize_vector(normal)
            origin = np.array(origin, dtype=float)
            cam = getattr(self.plotter, 'camera', None)
            if cam is None:
                return
            # distance heuristic
            try:
                rng = None
                if self.cloud_mesh_3d is not None and hasattr(self.cloud_mesh_3d, 'points'):
                    rng = self.cloud_mesh_3d.points.max(axis=0) - self.cloud_mesh_3d.points.min(axis=0)
                d = float(np.linalg.norm(rng)) * 0.5 if rng is not None else 1.0
            except Exception:
                d = 1.0
            try:
                cam.focal_point = origin.tolist()
            except Exception:
                pass
            try:
                cam.position = (origin + normal * d).tolist()
            except Exception:
                pass
            # adjust view up if parallel
            try:
                up = np.array(getattr(cam, 'view_up', [0.0, 1.0, 0.0]), dtype=float)
                upn = self.normalize_vector(up)
                if abs(float(np.dot(upn, normal))) > 0.99:
                    new_up = np.array([0.0, 1.0, 0.0], dtype=float) if abs(normal[1]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=float)
                    cam.view_up = new_up.tolist()
            except Exception:
                pass
            try:
                self.plotter.render()
            except Exception:
                pass
        except Exception:
            pass
