#!/usr/bin/env python3
# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

"""
HKL 3D Plot Dock for Workbench (Minimal)

- No interpolation from points to volume
- No slice plane or extra controls
- Just plotting when 'Load Dataset' is clicked
- Supports HDF5 (points or volume) and VTI volumes
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    import pyvista as pyv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except Exception:
    PYVISTA_AVAILABLE = False

from dashpva.utils.hdf5_loader import HDF5Loader


class HKL3DPlotDock(QDockWidget):
    def __init__(self, parent, title: str, main_window):
        super().__init__(title, parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.main = main_window

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top row: Load button
        top = QHBoxLayout()
        self.btn_load = QPushButton("Load Dataset")
        self.btn_load.clicked.connect(self.load_data)
        top.addWidget(self.btn_load)
        layout.addLayout(top)


        # Plotter
        self.plotter = None
        if PYVISTA_AVAILABLE:
            try:
                pyv.set_plot_theme('dark')
            except Exception:
                pass
            try:
                self.plotter = QtInteractor(container)
                try:
                    self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
                except Exception:
                    pass
                layout.addWidget(self.plotter)
            except Exception:
                self.plotter = None
                self.btn_load.setEnabled(False)
                msg = QLabel("3D (VTK) unavailable in tunnel mode.")
                try:
                    msg.setAlignment(Qt.AlignCenter)
                except Exception:
                    pass
                try:
                    msg.setWordWrap(True)
                except Exception:
                    pass
                layout.addWidget(msg)
        else:
            # Fallback: button disabled and message
            self.btn_load.setEnabled(False)
            try:
                QMessageBox.warning(self, "3D Viewer", "PyVista not available. Install pyvista and pyvistaqt to enable 3D plotting.")
            except Exception:
                pass

        self.setWidget(container)

        # State
        self.current_file_path = None
        self.current_dataset_path = None
        self.cloud_mesh = None
        self.volume_grid = None
        self.h5loader = HDF5Loader()

    def _clear_plot(self):
        try:
            if self.plotter is not None:
                self.plotter.clear()
                self.plotter.add_axes(xlabel='H', ylabel='K', zlabel='L')
        except Exception:
            pass
        self.cloud_mesh = None
        self.volume_grid = None

    def _plot_points(self, points: np.ndarray, intensities: np.ndarray):
        if self.plotter is None:
            return
        self._clear_plot()
        # PolyData points with intensity scalars
        import pyvista as pyv
        self.cloud_mesh = pyv.PolyData(points)
        self.cloud_mesh['intensity'] = intensities
        self.plotter.add_mesh(
            self.cloud_mesh,
            scalars='intensity',
            cmap='jet',
            point_size=5.0,
            name='points',
            show_scalar_bar=True,
            reset_camera=True,
        )
        try:
            self.plotter.show_bounds(
                mesh=self.cloud_mesh,
                xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                bounds=self.cloud_mesh.bounds,
            )
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
        except Exception:
            pass
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def _plot_volume_array(self, volume: np.ndarray, metadata: dict = None):
        if self.plotter is None:
            return
        self._clear_plot()
        import pyvista as pyv
        grid = pyv.ImageData()
        dims_cells = np.array(volume.shape, dtype=int)
        grid.dimensions = (dims_cells + 1).tolist()
        spacing = (metadata or {}).get('voxel_spacing') or (1.0, 1.0, 1.0)
        origin = (metadata or {}).get('grid_origin') or (0.0, 0.0, 0.0)
        try:
            grid.spacing = tuple(float(x) for x in spacing)
        except Exception:
            grid.spacing = (1.0, 1.0, 1.0)
        try:
            grid.origin = tuple(float(x) for x in origin)
        except Exception:
            grid.origin = (0.0, 0.0, 0.0)
        try:
            arr_order = (metadata or {}).get('array_order', 'F') or 'F'
            grid.cell_data['intensity'] = volume.flatten(order=arr_order)
        except Exception:
            grid.cell_data['intensity'] = volume.flatten(order='F')
        self.volume_grid = grid
        self.plotter.add_volume(
            volume=self.volume_grid,
            scalars='intensity',
            name='cloud_volume',
            reset_camera=True,
            show_scalar_bar=True,
        )
        try:
            self.plotter.show_bounds(
                mesh=self.volume_grid,
                xtitle='H Axis', ytitle='K Axis', ztitle='L Axis',
                bounds=self.volume_grid.bounds,
            )
            # Color cube axes H/K/L
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
        except Exception:
            pass
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception:
            pass

    def load_data(self):
        """Load dataset and plot based on type. Prefer Workbench selection; fallback to file dialog."""
        if not PYVISTA_AVAILABLE:
            QMessageBox.warning(self, '3D Viewer', 'PyVista is not available.')
            return
        # Prefer current selection from Workbench
        file_path = getattr(self.main, 'current_file_path', None)
        dataset_path = getattr(self.main, 'selected_dataset_path', None)
        use_dialog = not (file_path and dataset_path)

        if use_dialog:
            from PyQt5.QtWidgets import QFileDialog
            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Select HDF5 or VTI File', '', 'HDF5 or VTI Files (*.h5 *.hdf5 *.vti);;All Files (*)'
            )
            if not file_name:
                QMessageBox.information(self, 'File', 'No file selected.')
                return
            self.current_file_path = file_name
            # Inspect via loader info
            loader = self.h5loader
            # VTI path
            try:
                from pathlib import Path as _Path
                if _Path(file_name).suffix.lower() == '.vti':
                    volume, vol_shape = loader.load_vti_volume_3d(file_name)
                    if volume is None or int(volume.size) == 0:
                        QMessageBox.warning(self, 'Load Error', f'No volume data found in VTI file.\nError: {loader.last_error}')
                        return
                    meta = getattr(loader, 'file_metadata', {}) or {}
                    self._plot_volume_array(volume, meta)
                    return
            except Exception:
                pass
            # HDF5 decide type
            try:
                info = loader.get_file_info(file_name, style='dict')
            except Exception:
                info = {}
            dt = str(info.get('data_type', '')).lower() or str(info.get('metadata', {}).get('data_type', '')).lower()
            if dt == 'volume':
                volume, vol_shape = loader.load_h5_volume_3d(file_name)
                if volume is None or int(volume.size) == 0:
                    QMessageBox.warning(self, 'Load Error', f'No volume data found in HDF5 file.\nError: {loader.last_error}')
                    return
                meta = getattr(loader, 'file_metadata', {}) or {}
                self._plot_volume_array(volume, meta)
            else:
                points, intensities, num_images, shape = loader.load_h5_to_3d(file_name)
                if int(points.size) == 0 or int(intensities.size) == 0:
                    QMessageBox.warning(self, 'Load Error', f'No valid 3D point data found.\nError: {loader.last_error}')
                    return
                self._plot_points(points, intensities)
            return

        # Use Workbench selection
        self.current_file_path = file_path
        self.current_dataset_path = dataset_path
        try:
            import h5py
            with h5py.File(file_path, 'r') as h5file:
                if dataset_path not in h5file:
                    QMessageBox.warning(self, 'Load Error', 'Selected dataset not found in file.')
                    return
                item = h5file[dataset_path]
                if not hasattr(item, 'shape'):
                    QMessageBox.warning(self, 'Load Error', 'Selected item is not a dataset.')
                    return
                data = np.asarray(item[...])
        except Exception as e:
            QMessageBox.critical(self, 'Error Loading Data', f'Failed to load dataset:\n{e}')
            return
        # Decide plotting
        if data.ndim == 3:
            self._plot_volume_array(data, metadata={'array_order': 'F'})
        elif data.ndim >= 2:
            # Flatten to points: H,K index grid with intensities from 2D
            h, k = data.shape[-2], data.shape[-1]
            X, Y = np.meshgrid(np.arange(k), np.arange(h))
            Z = np.zeros_like(X, dtype=float)
            points = np.column_stack([X.ravel().astype(float), Y.ravel().astype(float), Z.ravel()])
            intens = np.asarray(data[-1] if data.ndim == 3 else data, dtype=float).ravel()
            self._plot_points(points, intens)
        else:
            QMessageBox.information(self, 'Load', 'Dataset is not 2D/3D numeric; cannot plot.')
