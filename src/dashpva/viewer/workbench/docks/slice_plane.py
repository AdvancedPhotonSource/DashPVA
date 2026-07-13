import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QProgressDialog, QWidget

from dashpva.gui import ui_path
from dashpva.viewer.core.docks.base_dock import BaseDock


class SlicePlaneDock(BaseDock):
    """
    Slice Controls dock for manipulating the 3D slice plane and camera.
    Loads its UI from gui/workbench/docks/slice_plane.ui and wires signals
    into Workspace3D (tab_3d) methods.
    """
    def __init__(self, title: str = "Slice Controls", main_window=None, segment_name: str = "3d", dock_area: Qt.DockWidgetArea = Qt.LeftDockWidgetArea, show: bool = False):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self._widget = None
        self._build()
        self._wire()

    def setup(self):
        # BaseDock handles docking and Windows->segment toggle registration
        super().setup()

    def _build(self):
        try:
            self._widget = QWidget(self)
            uic.loadUi(ui_path("workbench", "docks", "slice_plane.ui"), self._widget)
            self.setWidget(self._widget)
        except Exception as e:
            # If UI fails to load, keep an empty widget to avoid crashing
            self._widget = QWidget(self)
            self.setWidget(self._widget)
            try:
                if hasattr(self.main_window, 'update_status'):
                    self.main_window.update_status(f"SlicePlaneDock UI load failed: {e}")
            except Exception:
                pass

    def _wire(self):
        mw = self.main_window
        if mw is None:
            return
        tab = getattr(mw, 'tab_3d', None)
        if tab is None:
            return
        w = self._widget
        try:
            # Steps
            if hasattr(w, 'sb_slice_translate_step'):
                w.sb_slice_translate_step.setValue(0.01)
                w.sb_slice_translate_step.valueChanged.connect(lambda v: setattr(tab, '_slice_translate_step', float(v)))
            if hasattr(w, 'sb_slice_rotate_step_deg'):
                w.sb_slice_rotate_step_deg.setValue(1.0)
                w.sb_slice_rotate_step_deg.valueChanged.connect(lambda v: setattr(tab, '_slice_rotate_step_deg', float(v)))
            if hasattr(w, 'sb_slice_thickness'):
                setattr(tab, '_slice_thickness', float(w.sb_slice_thickness.value()))
                w.sb_slice_thickness.valueChanged.connect(lambda v: setattr(tab, '_slice_thickness', float(v)))

            # Orientation preset
            if hasattr(w, 'cb_slice_orientation'):
                w.cb_slice_orientation.currentTextChanged.connect(lambda txt: tab.set_plane_preset(str(txt)))

            # Custom normal spinboxes
            def _apply_custom_normal():
                try:
                    h = float(w.sb_norm_h.value()) if hasattr(w, 'sb_norm_h') else 0.0
                    k = float(w.sb_norm_k.value()) if hasattr(w, 'sb_norm_k') else 0.0
                    l_val = float(w.sb_norm_l.value()) if hasattr(w, 'sb_norm_l') else 1.0
                    if hasattr(tab, 'set_custom_normal'):
                        tab.set_custom_normal([h, k, l_val])
                    # If Custom preset selected, apply immediately
                    cur = str(w.cb_slice_orientation.currentText()) if hasattr(w, 'cb_slice_orientation') else ''
                    if cur.lower().startswith('custom'):
                        tab.set_plane_preset('Custom')
                except Exception:
                    pass
            for name in ('sb_norm_h', 'sb_norm_k', 'sb_norm_l'):
                spin = getattr(w, name, None)
                if spin is not None:
                    try:
                        spin.editingFinished.connect(_apply_custom_normal)
                    except Exception:
                        pass

            # Translate buttons
            if hasattr(w, 'btn_up_normal'):
                w.btn_up_normal.clicked.connect(lambda: tab.nudge_along_normal(+1))
            if hasattr(w, 'btn_down_normal'):
                w.btn_down_normal.clicked.connect(lambda: tab.nudge_along_normal(-1))
            if hasattr(w, 'btn_pos_h'):
                w.btn_pos_h.clicked.connect(lambda: tab.nudge_along_axis('H', +1))
            if hasattr(w, 'btn_neg_h'):
                w.btn_neg_h.clicked.connect(lambda: tab.nudge_along_axis('H', -1))
            if hasattr(w, 'btn_pos_k'):
                w.btn_pos_k.clicked.connect(lambda: tab.nudge_along_axis('K', +1))
            if hasattr(w, 'btn_neg_k'):
                w.btn_neg_k.clicked.connect(lambda: tab.nudge_along_axis('K', -1))
            if hasattr(w, 'btn_pos_l'):
                w.btn_pos_l.clicked.connect(lambda: tab.nudge_along_axis('L', +1))
            if hasattr(w, 'btn_neg_l'):
                w.btn_neg_l.clicked.connect(lambda: tab.nudge_along_axis('L', -1))

            # Rotate buttons use current rotate-step from tab
            if hasattr(w, 'btn_rot_plus_h'):
                w.btn_rot_plus_h.clicked.connect(lambda: tab.rotate_about_axis('H', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_h'):
                w.btn_rot_minus_h.clicked.connect(lambda: tab.rotate_about_axis('H', -getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_plus_k'):
                w.btn_rot_plus_k.clicked.connect(lambda: tab.rotate_about_axis('K', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_k'):
                w.btn_rot_minus_k.clicked.connect(lambda: tab.rotate_about_axis('K', -getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_plus_l'):
                w.btn_rot_plus_l.clicked.connect(lambda: tab.rotate_about_axis('L', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_l'):
                w.btn_rot_minus_l.clicked.connect(lambda: tab.rotate_about_axis('L', -getattr(tab, '_slice_rotate_step_deg', 1.0)))

            # Reset
            if hasattr(w, 'btn_reset_slice'):
                w.btn_reset_slice.clicked.connect(tab.reset_slice)

            # Visibility
            if hasattr(w, 'cb_show_slice'):
                w.cb_show_slice.toggled.connect(lambda checked: tab.toggle_3d_slice(bool(checked)))
            # Show Points (main cloud)
            if hasattr(w, 'cb_show_points'):
                w.cb_show_points.toggled.connect(lambda checked: tab.toggle_3d_points(bool(checked)))

            # Camera
            if hasattr(w, 'cb_cam_preset'):
                w.cb_cam_preset.currentTextChanged.connect(lambda txt: tab.set_camera_position(str(txt)))
            if hasattr(w, 'btn_zoom_in'):
                w.btn_zoom_in.clicked.connect(tab.zoom_in)
            if hasattr(w, 'btn_zoom_out'):
                w.btn_zoom_out.clicked.connect(tab.zoom_out)
            if hasattr(w, 'btn_reset_camera'):
                w.btn_reset_camera.clicked.connect(tab.reset_camera)
            if hasattr(w, 'btn_view_slice_normal'):
                w.btn_view_slice_normal.clicked.connect(tab.view_slice_normal)

            # Slice → 2D actions (implemented on this dock). Wire both the dock's
            # own buttons and the 3D-workspace buttons to these methods.
            if hasattr(w, 'btn_goto_slice_2d'):
                w.btn_goto_slice_2d.clicked.connect(self.show_slice_2d_tab)
            if hasattr(w, 'btn_save_slice'):
                w.btn_save_slice.clicked.connect(self.save_slice)
            if hasattr(tab, 'btn_show_slice_2d'):
                tab.btn_show_slice_2d.clicked.connect(self.show_slice_2d_tab)
            if hasattr(tab, 'btn_save_slice'):
                tab.btn_save_slice.clicked.connect(self.save_slice)
            if hasattr(tab, 'btn_load_slice'):
                tab.btn_load_slice.clicked.connect(self.load_slice)
            if hasattr(tab, 'btn_multi_slice'):
                tab.btn_multi_slice.clicked.connect(self.multi_slice)

            # Initialize defaults
            try:
                if hasattr(w, 'cb_slice_orientation'):
                    w.cb_slice_orientation.setCurrentText('HK (xy)')
                if hasattr(w, 'cb_show_slice'):
                    # Mirror current Tab state if available by checking actor existence
                    w.cb_show_slice.setChecked(True)
                if hasattr(w, 'cb_show_points'):
                    w.cb_show_points.setChecked(True)
            except Exception:
                pass
        except Exception:
            pass

    # ── Slice → 2D actions ────────────────────────────────────────────────────
    def show_slice_2d_tab(self):
        """Show and raise the Slice 2D dock, refreshed with the current controls."""
        try:
            tab = getattr(self.main_window, 'tab_3d', None)
            # Sync intensity range / colormap before showing so the extracted slice
            # matches what is currently set in the 3D view.
            if tab is not None:
                tab.refresh_slice_2d()
            dock = getattr(self.main_window, 'slice_2d_dock', None)
            if dock is not None:
                dock.show()
                dock.raise_()
        except Exception:
            pass

    def save_slice(self):
        """Save the current 2D slice (image + per-pixel HKL) to HDF5.

        Uses the last slice rendered in the Slice 2D tab, which is already masked
        to the current intensity range. Prompts to write a new standalone file or
        append into the source file.
        """
        from PyQt5.QtWidgets import QMessageBox

        mw = self.main_window
        tab = getattr(mw, 'tab_3d', None)
        tab_2d = getattr(mw, 'tab_slice_2d', None)
        result = tab_2d.get_last_slice() if tab_2d is not None else None
        if not result:
            QMessageBox.warning(
                self, 'No Slice',
                'No slice available to save. Move the slice plane and make sure '
                'points fall within the current intensity range.'
            )
            return

        clim = result.get('clim', (0.0, 0.0))
        meta = {
            'data_type': 'slice',
            'slice_normal': [float(x) for x in getattr(tab, '_last_normal', np.array([0.0, 0.0, 1.0]))],
            'slice_origin': [float(x) for x in getattr(tab, '_last_origin', np.array([0.0, 0.0, 0.0]))],
            'u_axis': [float(x) for x in result['u_axis']],
            'v_axis': [float(x) for x in result['v_axis']],
            'u_range': [float(result['u_range'][0]), float(result['u_range'][1])],
            'v_range': [float(result['v_range'][0]), float(result['v_range'][1])],
            'orientation': result['orientation'],
            'orth_label': str(result.get('orth_label') or ''),
            'intensity_min': float(clim[0]),
            'intensity_max': float(clim[1]),
            'num_points': int(result.get('num_points', 0)),
            'original_file': str(getattr(mw, 'current_file_path', None) or 'unknown'),
            'extraction_timestamp': str(np.datetime64('now')),
        }
        if result.get('orth_value') is not None:
            meta['orth_value'] = float(result['orth_value'])

        # Ask: new file or append to source.
        box = QMessageBox(self)
        box.setWindowTitle('Save Slice')
        box.setText('Where should the extracted 2D slice be saved?')
        btn_new = box.addButton('New file…', QMessageBox.AcceptRole)
        btn_src = box.addButton('Append to source', QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        source_path = getattr(mw, 'current_file_path', None)
        btn_src.setEnabled(bool(source_path))
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (btn_new, btn_src):
            return

        from dashpva.utils.hdf5_loader import HDF5Loader
        loader = HDF5Loader()

        if clicked is btn_new:
            default_name = f"slice_extract_{str(np.datetime64('now', 's')).replace(':', '-')}.h5"
            file_path, _ = QFileDialog.getSaveFileName(
                self, 'Save Slice Data', default_name, 'HDF5 Files (*.h5 *.hdf5);;All Files (*)'
            )
            if not file_path:
                return
            ok = loader.save_slice_arrays(
                file_path, result['image'], result['qx'], result['qy'], result['qz'],
                metadata=meta, append=False,
            )
            target = file_path
        else:
            ok = loader.save_slice_arrays(
                source_path, result['image'], result['qx'], result['qy'], result['qz'],
                metadata=meta, append=True,
            )
            target = source_path

        if ok:
            try:
                mw.update_status(f"Slice saved to {target}")
            except Exception:
                pass
            QMessageBox.information(self, 'Success', f'Slice saved successfully to:\n{target}')
        else:
            QMessageBox.critical(self, 'Error', f'Failed to save slice:\n{loader.get_last_error()}')

    def load_slice(self):
        """Load a saved 2D slice (image + HKL + orientation) into the Slice 2D dock."""
        from PyQt5.QtWidgets import QMessageBox

        mw = self.main_window
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Slice', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)'
        )
        if not file_path:
            return

        from dashpva.utils.hdf5_loader import HDF5Loader
        loader = HDF5Loader()
        result = loader.load_slice(file_path)
        if not result:
            QMessageBox.warning(
                self, 'Load Slice',
                f'Could not load a slice from:\n{file_path}\n\n{loader.get_last_error()}'
            )
            return

        tab_2d = getattr(mw, 'tab_slice_2d', None)
        if tab_2d is not None:
            tab_2d.display_loaded(result)
        dock = getattr(mw, 'slice_2d_dock', None)
        if dock is not None:
            dock.show()
            dock.raise_()
        try:
            mw.update_status(f"Loaded slice from {file_path}")
        except Exception:
            pass

    def multi_slice(self):
        """Apply one slice plane to many datasets and save them to one HDF5.

        The plane comes from either the current slice or a saved slice (user's
        choice). Sources are a folder of .h5 files or a hand-picked set. Each
        extracted slice is written to its own group in a new multi-slice file,
        tagged with the source path.
        """
        import os
        from pathlib import Path

        from PyQt5.QtWidgets import QMessageBox

        from dashpva.utils.hdf5_loader import HDF5Loader
        from dashpva.utils.rsm_converter import RSMConverter
        from dashpva.utils.slice_raster import rasterize_slab

        mw = self.main_window
        tab = getattr(mw, 'tab_3d', None)
        loader = HDF5Loader()

        # 1. Plane source: current slice or a saved slice.
        box = QMessageBox(self)
        box.setWindowTitle('Multi-slice')
        box.setText('Which slice plane should be applied to the datasets?')
        btn_current = box.addButton('Current slice', QMessageBox.AcceptRole)
        btn_saved = box.addButton('From saved slice…', QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (btn_current, btn_saved):
            return

        if clicked is btn_current:
            normal = getattr(tab, '_last_normal', None)
            origin = getattr(tab, '_last_origin', None)
            if normal is None or origin is None:
                QMessageBox.warning(
                    self, 'No Slice Plane',
                    'There is no current slice plane. Load a 3D dataset and set up '
                    'a slice plane first, or choose "From saved slice".')
                return
            normal = np.asarray(normal, dtype=float)
            origin = np.asarray(origin, dtype=float)
        else:
            plane_path, _ = QFileDialog.getOpenFileName(
                self, 'Saved slice for plane', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
            if not plane_path:
                return
            saved = loader.load_slice(plane_path)
            if not saved or saved.get('slice_normal') is None or saved.get('slice_origin') is None:
                QMessageBox.warning(
                    self, 'Multi-slice', f'Could not read a slice plane from:\n{plane_path}')
                return
            normal = np.asarray(saved['slice_normal'], dtype=float)
            origin = np.asarray(saved['slice_origin'], dtype=float)

        n_norm = float(np.linalg.norm(normal))
        if not np.isfinite(n_norm) or n_norm <= 0.0:
            QMessageBox.warning(self, 'Multi-slice', 'Invalid slice normal.')
            return
        normal = normal / n_norm

        # 2. Sources: a folder of h5 files or a hand-picked set.
        box = QMessageBox(self)
        box.setWindowTitle('Multi-slice')
        box.setText('Apply the slice to which datasets?')
        btn_folder = box.addButton('Folder…', QMessageBox.AcceptRole)
        btn_files = box.addButton('Files…', QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (btn_folder, btn_files):
            return

        source_folder = None
        if clicked is btn_folder:
            folder = QFileDialog.getExistingDirectory(self, 'Folder of datasets')
            if not folder:
                return
            files = sorted(str(p) for p in Path(folder).iterdir()
                           if p.suffix.lower() in ('.h5', '.hdf5'))
            source_folder = folder
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self, 'Datasets to slice', '', 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not files:
            QMessageBox.warning(self, 'Multi-slice', 'No .h5/.hdf5 files selected.')
            return

        # 3. Output file.
        default_name = f"multislice_{str(np.datetime64('now', 's')).replace(':', '-')}.h5"
        out_path, _ = QFileDialog.getSaveFileName(
            self, 'Save multi-slice file', default_name, 'HDF5 Files (*.h5 *.hdf5);;All Files (*)')
        if not out_path:
            return

        # 4. Extract the plane from each source.
        thickness = getattr(tab, '_slice_thickness', 0.002)
        conv = RSMConverter()
        entries, failures = [], []
        progress = QProgressDialog('Extracting slices…', 'Cancel', 0, len(files), self)
        progress.setWindowTitle('Multi-slice')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        for i, path in enumerate(files):
            progress.setValue(i)
            progress.setLabelText(f'Slicing {os.path.basename(path)} ({i + 1}/{len(files)})')
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                points, intensities, _num, shape = conv.load_h5_to_3d(path)
                pts = np.asarray(points, dtype=float)
                vals = np.asarray(intensities, dtype=float).reshape(-1)
                dist = (pts - origin) @ normal
                m = np.abs(dist) < thickness
                pts_m, vals_m = pts[m], vals[m]
                result = rasterize_slab(pts_m, vals_m, normal, origin, shape=shape)
                if result is None or pts_m.shape[0] == 0:
                    failures.append(path)
                    continue
                meta = {
                    'data_type': 'slice',
                    'source_file': str(path),
                    'slice_normal': [float(x) for x in normal],
                    'slice_origin': [float(x) for x in origin],
                    'orientation': result['orientation'],
                    'orth_label': str(result.get('orth_label') or ''),
                    'u_axis': [float(x) for x in result['u_axis']],
                    'v_axis': [float(x) for x in result['v_axis']],
                    'u_range': [float(result['u_range'][0]), float(result['u_range'][1])],
                    'v_range': [float(result['v_range'][0]), float(result['v_range'][1])],
                    'num_points': int(pts_m.shape[0]),
                }
                if source_folder:
                    meta['source_folder'] = str(source_folder)
                if result.get('orth_value') is not None:
                    meta['orth_value'] = float(result['orth_value'])
                entries.append({
                    'name': Path(path).stem, 'image': result['image'],
                    'qx': result['qx'], 'qy': result['qy'], 'qz': result['qz'], 'meta': meta,
                })
            except Exception:
                failures.append(path)
        progress.setValue(len(files))

        if not entries:
            QMessageBox.warning(self, 'Multi-slice', 'No slices could be extracted.')
            return

        ok = loader.save_slices_to_file(out_path, entries)
        if ok:
            msg = f'Saved {len(entries)} slice(s) to:\n{out_path}'
            if failures:
                msg += f'\n\n{len(failures)} file(s) skipped (no data near plane or load error).'
            try:
                mw.update_status(f"Multi-slice saved to {out_path} ({len(entries)} slices)")
            except Exception:
                pass
            QMessageBox.information(self, 'Multi-slice', msg)
        else:
            QMessageBox.critical(self, 'Multi-slice', f'Failed to save:\n{loader.get_last_error()}')
