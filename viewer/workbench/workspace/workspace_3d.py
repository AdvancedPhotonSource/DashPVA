from typing import Optional
import os
from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QMessageBox, QFileDialog, QSizePolicy
from PyQt5.QtCore import Qt, QThread
import numpy as np
import pyvista as pv


# Import BaseTab using existing tabs package alias
from .base_tab import BaseTab

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
            super().__init__(ui_file='gui/workbench/workspace/workspace_3d.ui', parent=parent, main_window=main_window, title=title)
            self.title = title
            self.main_window = main_window 
            self.build()
            self.connect_all()
        except Exception as e:
            try:
                self.main_window.update_status(f"Error initializing 3D workspace: {e}")
            except Exception:
                pass

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
        # Initialize LUTs similar to viewer/hkl_3d.py
        try:
            self.lut = pv.LookupTable(cmap='jet')
            self.lut.apply_opacity([0, 1])
            self.lut2 = pv.LookupTable(cmap='jet')
            self.lut2.apply_opacity([0, 1])
            # Sync initial LUTs to the UI-selected colormap and current intensity controls
            try:
                self.on_3d_colormap_changed()
            except Exception:
                pass
        except Exception:
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
        # Cached true data intensity bounds (set on data load)
        self._data_intensity_min = None
        self._data_intensity_max = None

        
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
                try:
                    mw.update_status("Warning: layout3DPlotHost not found, 3D plot may not display correctly")
                except Exception:
                    pass
            # Clear initial state
            self.clear_plot()
        except Exception as e:
            try:
                mw.update_status(f"Error setting up 3D plot viewer: {e}")
            except Exception:
                pass

    def toggle_3d_points(self, checked: bool):
        """Shows/Hides the main HKL point cloud."""
        try:
            # Support either actor name used by different paths
            actor = None
            if "points" in getattr(self.plotter, 'actors', {}):
                actor = self.plotter.actors.get("points")
            elif "cloud_volume" in getattr(self.plotter, 'actors', {}):
                actor = self.plotter.actors.get("cloud_volume")
            if actor is not None:
                actor.SetVisibility(bool(checked))
                self.plotter.render()
        except Exception:
            pass

    def toggle_3d_slice(self, checked: bool):
        """Shows/Hides the interactive plane and the extracted slice points."""
        try:
            # Toggle the points extracted by the plane
            if "slab_points" in getattr(self.plotter, 'actors', {}):
                try:
                    self.plotter.actors["slab_points"].SetVisibility(bool(checked))
                except Exception:
                    try:
                        self.plotter.renderer._actors["slab_points"].SetVisibility(bool(checked))
                    except Exception:
                        pass

            # Toggle the interactive plane widget tool
            if self.plane_widget is not None:
                try:
                    if checked:
                        # Try both enable methods to support different versions
                        try:
                            self.plane_widget.EnabledOn()
                        except Exception:
                            self.plane_widget.On()
                    else:
                        try:
                            self.plane_widget.EnabledOff()
                        except Exception:
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

            self.plotter.render()
        except Exception:
            pass

    def on_3d_colormap_changed(self):
        """Apply selected colormap to the points (and slab if available)."""
        try:
            cmap_name = getattr(self.cb_colormap_3d, 'currentText', lambda: 'viridis')()
        except Exception:
            cmap_name = 'viridis'
        # Primary LUT used for the main cloud volume/points
        try:
            self.lut = pv.LookupTable(cmap=cmap_name)
            self.lut.apply_opacity([0, 1])
            self.lut2 = pv.LookupTable(cmap=cmap_name)
            self.lut2.apply_opacity([0, 1])
            # Keep LUT scalar ranges in sync with current controls or cached bounds
            try:
                # Prefer current UI intensity range
                vmin = float(self.sb_min_intensity_3d.value())
                vmax = float(self.sb_max_intensity_3d.value())
            except Exception:
                # Fallback to data bounds if UI unavailable
                vmin = getattr(self, '_data_intensity_min', None)
                vmax = getattr(self, '_data_intensity_max', None)
            try:
                if vmin is not None and vmax is not None:
                    # Ensure proper ordering and non-zero span
                    if vmin > vmax:
                        vmin, vmax = vmax, vmin
                    if vmin == vmax:
                        vmax = vmin + 1e-6
                    self.lut.scalar_range = (vmin, vmax)
                    self.lut2.scalar_range = (vmin, vmax)
            except Exception:
                pass
        except Exception:
            self.lut = None
            self.lut2 = None

        # Update points/cloud actor by changing the mapper's lookup table (no re-add)
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            tgt_name = 'points' if 'points' in actors else ('cloud_volume' if 'cloud_volume' in actors else None)
            if tgt_name and self.lut is not None:
                actor = actors.get(tgt_name)
                try:
                    # Prefer direct mapper property
                    actor.mapper.lookup_table = self.lut
                except Exception:
                    try:
                        actor.GetMapper().SetLookupTable(self.lut)
                    except Exception:
                        pass
                # Maintain scalar range and visibility
                try:
                    rng = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                    actor.mapper.scalar_range = rng
                except Exception:
                    pass
                try:
                    actor.SetVisibility(bool(self.cb_show_points.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Attempt to update slab colormap (best-effort) using a separate LUT
        try:
            # Use the cached actor if present otherwise lookup by name
            slab_actor = self.slab_actor
            if slab_actor is None:
                actors = getattr(self.plotter, 'actors', {}) or {}
                slab_actor = actors.get('slab_points')
            if slab_actor is not None:
                try:
                    # Apply LUT to mapper
                    try:
                        slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        slab_actor.GetMapper().SetLookupTable(self.lut2 or self.lut)
                except Exception:
                    pass
                # Apply scalar range from UI if available
                try:
                    rng = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                    slab_actor.mapper.scalar_range = rng
                except Exception:
                    pass
                # Keep visibility consistent
                try:
                    slab_actor.SetVisibility(bool(self.cb_show_slice.isChecked()))
                except Exception:
                    pass
        except Exception:
            pass

        # Render and ensure ranges/visibility remain in sync
        try:
            # Update existing scalar bars with the new LUT
            if hasattr(self.plotter, 'scalar_bars') and self.lut is not None:
                for _, scalar_bar in self.plotter.scalar_bars.items():
                    try:
                        scalar_bar.SetLookupTable(self.lut)
                        scalar_bar.Modified()
                    except Exception:
                        pass
            self.plotter.render()
        except Exception:
            pass
        try:
            self.update_intensity()
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
                self.main_window.update_status(f"Error clearing 3D plot: {e}")
            except Exception:
                pass

    # === Loading & Plotting ===
    def load_data(self):
        """Load dataset and render using the tab's local plotter."""
        mw = self.main_window
        try:
            mw.update_status("Loading data into 3D viewer...")
        except Exception:
            pass
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
                    # Cache a reference to the main points/cloud actor for fast updates
                    try:
                        if "points" in self.plotter.actors:
                            self.points_actor = self.plotter.actors.get("points")
                        elif "cloud_volume" in self.plotter.actors:
                            self.points_actor = self.plotter.actors.get("cloud_volume")
                    except Exception:
                        self.points_actor = None

                    # Cache true data intensity bounds and set LUT scalar ranges
                    try:
                        self._data_intensity_min = float(np.min(intensities))
                        self._data_intensity_max = float(np.max(intensities))
                        if self.lut is not None:
                            self.lut.scalar_range = (self._data_intensity_min, self._data_intensity_max)
                        if self.lut2 is not None:
                            self.lut2.scalar_range = (self._data_intensity_min, self._data_intensity_max)
                    except Exception:
                        pass
                    # Ensure the currently selected colormap is applied immediately on load
                    try:
                        self.on_3d_colormap_changed()
                    except Exception:
                        pass
                    # Apply LUTs to actors
                    try:
                        if self.points_actor is not None and self.lut is not None:
                            self.points_actor.mapper.lookup_table = self.lut
                    except Exception:
                        pass
                    try:
                        if self.slab_actor is not None and (self.lut2 or self.lut) is not None:
                            self.slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        pass
                    # Show bounds like in hkl_3d
                    try:
                        self.plotter.show_bounds(
                            mesh=self.points_actor.mapper.input if self.points_actor is not None else None,
                            xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                            ticks='inside', minor_ticks=True,
                            n_xlabels=7, n_ylabels=7, n_zlabels=7,
                            x_color='red', y_color='green', z_color='blue',
                            font_size=20
                        )
                    except Exception:
                        pass
                    # Sync scalar bars to primary LUT
                    try:
                        if hasattr(self.plotter, 'scalar_bars') and self.lut is not None:
                            for _, sb in self.plotter.scalar_bars.items():
                                try:
                                    sb.SetLookupTable(self.lut)
                                    sb.Modified()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Ensure visibility respects checkboxes
                    try:
                        self.toggle_3d_points(self.cb_show_points.isChecked())
                        self.toggle_3d_slice(self.cb_show_slice.isChecked())
                    except Exception:
                        pass

                    # Align current intensity range with data bounds and reflect in UI
                    try:
                        self.update_intensity()
                    except Exception:
                        pass

                    # Switch to this tab automatically
                    if hasattr(mw, 'tabWidget_analysis'):
                        idx = mw.tabWidget_analysis.indexOf(self)
                        mw.tabWidget_analysis.setCurrentIndex(idx)
                    
                    self.main_window.update_status("3D Rendering Complete")
                except Exception as e:
                    try:
                        self.main_window.update_status(f"Render Error: {e}")
                    except Exception:
                        pass

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
            pass

    def on_plane_update(self, normal, origin):
        """Extracts points near the plane to simulate a 3D slice."""
        if self.cloud_mesh_3d is None:
            return

        # Plane math: (Point - Origin) ⋅ Normal
        vec = self.cloud_mesh_3d.points - origin
        dist = np.dot(vec, normal)
        
        # Thickness of the slice in HKL units (align with HKL3D)
        thickness = 0.002 
        mask = np.abs(dist) < thickness
        
        slab = self.cloud_mesh_3d.extract_points(mask)

        # Clean up any existing slab actor before adding a new one to keep references current
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            if 'slab_points' in actors:
                try:
                    self.plotter.remove_actor('slab_points', reset_camera=False)
                except Exception:
                    try:
                        self.plotter.remove_actor(actors.get('slab_points'), reset_camera=False)
                    except Exception:
                        pass
        except Exception:
            pass
        self.slab_actor = None

        if slab.n_points > 0:
            # Add the slab without passing cmap; set mapper.lookup_table explicitly afterwards
            self.slab_actor = self.plotter.add_mesh(
                slab,
                name="slab_points",
                render_points_as_spheres=True,
                point_size=8,
                scalars='intensity',
                show_scalar_bar=False
            )
            # Apply current LUT and scalar range to the new slab actor
            try:
                if (self.lut2 or self.lut) is not None:
                    try:
                        self.slab_actor.mapper.lookup_table = (self.lut2 or self.lut)
                    except Exception:
                        self.slab_actor.GetMapper().SetLookupTable(self.lut2 or self.lut)
            except Exception:
                pass
            # Match current intensity/clim
            try:
                clim = [self.sb_min_intensity_3d.value(), self.sb_max_intensity_3d.value()]
                self.slab_actor.mapper.scalar_range = clim
            except Exception:
                pass
            # Ensure the new slab respects the current checkbox state
            try:
                self.slab_actor.SetVisibility(self.cb_show_slice.isChecked())
            except Exception:
                pass
        
        # Keep plane widget synchronized to final state
        try:
            widgets = getattr(self.plotter, 'plane_widgets', [])
            if self.plane_widget is not None:
                self.plane_widget.SetNormal(normal)
                self.plane_widget.SetOrigin(origin)
            elif widgets:
                widgets[0].SetNormal(normal)
                widgets[0].SetOrigin(origin)
        except Exception:
            pass
            
        self.plotter.render()
        # Respect slice toggle state after update
        try:
            self.toggle_3d_slice(self.cb_show_slice.isChecked())
        except Exception:
            pass
        
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

        # Read requested values from UI
        try:
            requested_min = float(self.sb_min_intensity_3d.value())
            requested_max = float(self.sb_max_intensity_3d.value())
        except Exception:
            # Fallback to current mapper range if spinboxes unavailable
            requested_min, requested_max = 0.0, 1.0

        # Clamp to true data range if available
        data_min = getattr(self, '_data_intensity_min', None)
        data_max = getattr(self, '_data_intensity_max', None)
        vmin = requested_min
        vmax = requested_max
        if data_min is not None and data_max is not None:
            vmin = max(requested_min, data_min)
            vmax = min(requested_max, data_max)

        # Enforce ordering and non-zero span
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmax = vmin + 1e-6

        # Reflect applied values back to the UI
        try:
            self.sb_min_intensity_3d.setValue(vmin)
            self.sb_max_intensity_3d.setValue(vmax)
        except Exception:
            pass

        # Define the new scalar range
        new_range = [vmin, vmax]

        # Update main cloud/points actor scalar range
        try:
            actors = getattr(self.plotter, 'actors', {}) or {}
            if "points" in actors:
                actors["points"].mapper.scalar_range = (new_range[0], new_range[1])
            if "cloud_volume" in actors:
                actors["cloud_volume"].mapper.scalar_range = (new_range[0], new_range[1])
        except Exception:
            pass

        if "slab_points" in self.plotter.actors:
            self.plotter.actors["slab_points"].mapper.scalar_range = (new_range[0], new_range[1])

        # Keep LUTs' internal ranges consistent as well
        try:
            if self.lut is not None:
                self.lut.scalar_range = (new_range[0], new_range[1])
            if self.lut2 is not None:
                self.lut2.scalar_range = (new_range[0], new_range[1])
        except Exception:
            pass
            
        # Update the volume actor by re-adding with new clim range
        if hasattr(self.plotter, 'scalar_bars'):
            for bar in self.plotter.scalar_bars.values():
                try:
                    bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                except Exception:
                    pass

        # Force update of all scalar bars with the new range
        if hasattr(self.plotter, 'scalar_bars'):
            for name, scalar_bar in self.plotter.scalar_bars.items():
                if scalar_bar:
                    try:
                        scalar_bar.GetLookupTable().SetTableRange(new_range[0], new_range[1])
                        scalar_bar.Modified()
                    except Exception:
                        pass

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

        # Update Info labels and availability after intensity changes (best-effort)
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
            # Use the same attribute name that is set by the Render3D worker
            if self.plane_widget is not None:
                try:
                    self.plane_widget.EnabledOff()
                except Exception:
                    pass

                try:
                    self.plotter.clear_plane_widgets()
                except Exception:
                    pass

                self.plane_widget = None
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

    # ===== Info/Availability (align with hkl_3d patterns) =====

    def _refresh_availability(self):
        """Enable/disable controls depending on plotter/data availability."""
        try:
            has_data = bool(self.cloud_mesh_3d is not None or getattr(self, 'points_actor', None) is not None)
            for w in [getattr(self, "cb_show_points", None),
                      getattr(self, "cb_show_slice", None),
                      getattr(self, "sb_min_intensity_3d", None),
                      getattr(self, "sb_max_intensity_3d", None)]:
                try:
                    if w is not None:
                        w.setEnabled(has_data)
                except Exception:
                    pass
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
