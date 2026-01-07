from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np

class Render3D(QObject):
    finished = pyqtSignal()
    render_ready = pyqtSignal(object)

    def __init__(self, *, points=None, intensities=None, num_images=None, shape=None, parent=None):
        super().__init__(parent)
        self.points = points
        self.intensities = intensities
        self.num_images = int(num_images) if num_images is not None else 0
        self.shape = tuple(shape) if shape is not None else (0, 0)

    @pyqtSlot()
    def run(self):
        try:
            pts = np.asarray(self.points, dtype=float) if self.points is not None else np.empty((0, 3), dtype=float)
            ints = np.asarray(self.intensities, dtype=float).ravel() if self.intensities is not None else np.empty((0,), dtype=float)
            
            if pts.ndim != 2 or (pts.size > 0 and pts.shape[1] != 3):
                pts = np.empty((0, 3), dtype=float)
            
            if ints.size:
                high_mask = ints > 5e6
                if np.any(high_mask):
                    ints[high_mask] = 0.0
            
            self.points = pts
            self.intensities = ints
            self.render_ready.emit(self)
        finally:
            self.finished.emit()

    def plot_3d_points(self, target_tab):
        """
        target_tab: The Tab3D instance. 
        It provides access to self.plotter and the local UI widgets.
        """
        try:
            import pyvista as pyv
            pts = self.points
            ints = self.intensities
            
            # 1. Use the plotter local to the Tab
            plotter = target_tab.plotter 
            
            if pts.ndim != 2 or pts.shape[1] != 3 or ints.size != pts.shape[0]:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(target_tab, '3D Viewer', 'Invalid point cloud data.')
                return

            plotter.clear()
            plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')

            # --- Setup LUTs ---
            lut = pyv.LookupTable(cmap='viridis')
            lut.below_range_color = 'black'
            lut.above_range_color = (1.0, 1.0, 0.0)
            lut.below_range_opacity = 0
            lut.apply_opacity([0, 1])
            lut.above_range_opacity = 1

            # --- Create Mesh ---
            mesh = pyv.PolyData(pts)
            mesh['intensity'] = ints
            
            # Store references on the Tab instance for intensity updates later
            target_tab.cloud_mesh_3d = mesh
            target_tab.lut = lut

            # --- Add to Plotter ---
            plotter.add_mesh(
                mesh,
                scalars='intensity',
                cmap=lut,
                point_size=5.0,
                name='points',
                show_scalar_bar=True,
                nan_opacity=0.0,
                show_edges=False
            )

            plotter.show_bounds(
                mesh=mesh,
                xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                bounds=mesh.bounds,
            )
            plotter.reset_camera()

            # -- slice --
            slice_normal = (0, 0, 1)
            slice_origin = mesh.center

            target_tab.plane_widget = plotter.add_plane_widget(
            callback=target_tab.on_plane_update,
            normal=slice_normal,
            origin=slice_origin,
            bounds=mesh.bounds,
            factor=1.0,
            implicit=True,
            assign_to_axis=None,
            tubing=False,
            origin_translation=True,
            outline_opacity=0
        )

            # --- Update Local Tab UI Widgets ---
            # Note: We now look for names from tab_3d.ui
            ints_range = (int(np.min(ints)), int(np.max(ints)))
            
            if hasattr(target_tab, 'sb_min_intensity_3d'):
                target_tab.sb_min_intensity_3d.setRange(*ints_range) # Expand range
                target_tab.sb_min_intensity_3d.setValue(int(np.min(ints)))
            
            if hasattr(target_tab, 'sb_max_intensity_3d'):
                target_tab.sb_max_intensity_3d.setRange(*ints_range)
                target_tab.sb_max_intensity_3d.setValue(int(np.max(ints)))

            # Call intensity update logic local to the tab
            if hasattr(target_tab, 'update_intensity'):
                target_tab.update_intensity()
            

            plotter.render()
            
        except Exception as e:
            print(f"Error in plot_3d_points: {e}")

class DatasetLoader(QObject):
    loaded = pyqtSignal(object)  # numpy array
    failed = pyqtSignal(str)

    def __init__(self, file_path, dataset_path, max_frames=100):
        super().__init__()
        self.file_path = file_path
        self.dataset_path = dataset_path
        self.max_frames = max_frames

    @pyqtSlot()
    def run(self):
        try:
            import h5py
            with h5py.File(self.file_path, 'r') as h5file:
                if self.dataset_path not in h5file:
                    self.failed.emit("Dataset not found")
                    return
                dset = h5file[self.dataset_path]
                if not isinstance(dset, h5py.Dataset):
                    self.failed.emit("Selected item is not a dataset")
                    return

                # Efficient loading to avoid blocking on huge datasets
                if len(dset.shape) == 3:
                    max_frames = min(self.max_frames, dset.shape[0])
                    data = dset[:max_frames]
                else:
                    # Guard against extremely large 2D datasets by center cropping
                    try:
                        estimated_size = dset.size * dset.dtype.itemsize
                    except Exception:
                        estimated_size = 0
                    if len(dset.shape) == 2 and estimated_size > 512 * 1024 * 1024:  # >512MB
                        h, w = dset.shape
                        ch = min(h, 2048)
                        cw = min(w, 2048)
                        y0 = max(0, (h - ch) // 2)
                        x0 = max(0, (w - cw) // 2)
                        data = dset[y0:y0+ch, x0:x0+cw]
                    else:
                        data = dset[...]

                data = np.asarray(data, dtype=np.float32)
                # Clean high values
                high_mask = data > 5e6
                if np.any(high_mask):
                    data[high_mask] = 0

                # 1D handling: emit raw 1D data for dedicated 1D view
                self.loaded.emit(data)
        except Exception as e:
            self.failed.emit(f"Error loading dataset: {e}")
