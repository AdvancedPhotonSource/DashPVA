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
from utils.hdf5_loader import HDF5Loader
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
        self.plotter = QtInteractor(self)
        self.container.insertWidget(0, self.plotter, stretch=1)
        self.scrollArea_3d_controls.setMinimumWidth(280)
        self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        self.cloud_mesh_3d = None
        self.slab_actor = None
        self.plane_widget = None
        self.lut = None
        self.lut2 = None
        
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
            mw.plotter_3d.add_axes(xlabel='H', ylabel='K', zlabel='L')
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
                    cmap=cmap_name,
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
        mw = self.main_window
        try:
            if hasattr(mw, 'cb_slice_orientation'):
                mw.cb_slice_orientation.setCurrentIndex(0)
            self.change_slice_orientation("HK (xy)")
        except Exception as e:
            mw.update_status(f"Error resetting 3D slice: {e}")

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

    def toggle_pointer(self):
        """Toggle visibility of the cloud volume"""
        is_visible_pointer = self.cbToggleSlicePointer.isChecked()
        self.plotter.renderer._actors["slab_points"].SetVisibility(is_visible_pointer)

        try:
            widgets = getattr(self.plotter, "plane_widgets", [])
            for pw in widgets or []:
                try:
                    if is_visible_pointer:
                        pw.EnabledOn()
                    else:
                        pw.EnabledOff()
                except Exception:
                    pass
        except Exception:
            pass
