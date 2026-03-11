"""
ROI Manager for Workbench
Centralizes ROI lifecycle, docks, stats computation, and interactions to shorten WorkbenchWindow.
"""

from typing import List, Optional, Callable, Set
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QMenu,
    QAction,
    QInputDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QToolButton,
    QStyle,
    QFileDialog,
    QMessageBox,
)
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
import h5py
import os


class ContextRectROI(pg.RectROI):
    """Rectangular ROI with right-click context menu that delegates actions to main window handlers."""
    def __init__(self, parent_window, pos, size, pen=None):
        super().__init__(pos, size, pen=pen)
        self.parent_window = parent_window
        try:
            self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        except Exception:
            pass
        # Make ROI rotatable: add a rotate handle at the top-right, rotating about center
        try:
            self.addRotateHandle([1, 0], [0.5, 0.5])
        except Exception:
            pass

    def mouseClickEvent(self, ev):
        try:
            if ev.button() == Qt.RightButton:
                menu = QMenu()
                action_stats = QAction("Show ROI Stats", menu)
                action_rename = QAction("Rename ROI", menu)
                action_set_active = QAction("Set Active ROI", menu)
                action_plot = QAction("Open ROI Plot", menu)
                action_hide = QAction("Hide ROI", menu)
                action_delete = QAction("Delete ROI", menu)

                action_stats.triggered.connect(lambda: self.parent_window.roi_manager.show_roi_stats_for_roi(self))
                action_rename.triggered.connect(lambda: self.parent_window.roi_manager.rename_roi(self))
                action_set_active.triggered.connect(lambda: self.parent_window.roi_manager.set_active_roi(self))
                action_plot.triggered.connect(lambda: self.parent_window.open_roi_plot_dock(self))
                action_hide.triggered.connect(lambda: self.parent_window.roi_manager.set_roi_visibility(self, False))
                action_delete.triggered.connect(lambda: self.parent_window.roi_manager.delete_roi(self))

                # Add actions and separator before Save
                menu.addAction(action_stats)
                menu.addAction(action_rename)
                menu.addAction(action_set_active)
                menu.addAction(action_plot)
                menu.addAction(action_hide)
                menu.addAction(action_delete)
                menu.addSeparator()
                # Save ROI action
                action_save = QAction("Save ROI", menu)
                try:
                    action_save.triggered.connect(lambda: self.parent_window.roi_manager.save_roi(self))
                except Exception:
                    pass
                menu.addAction(action_save)
                try:
                    menu.exec_(QCursor.pos())
                except Exception:
                    menu.exec_(QCursor.pos())
                ev.accept()
                return
        except Exception:
            pass
        # default behavior
        try:
            super().mouseClickEvent(ev)
        except Exception:
            pass


class ROIManager:
    def __init__(self, main_window):
        self.main = main_window
        # ROI collections/state
        self.rois: List[pg.ROI] = []
        self.current_roi: Optional[pg.ROI] = None
        self.roi_by_item = {}
        self.item_by_roi_id = {}
        self.roi_names = {}
        self.stats_row_by_roi_id = {}
        # Mapping helpers for stats table and overlay labels
        self.roi_by_stats_row = {}
        self.roi_label_by_id = {}
        self.show_names_checkbox = None
        self.hidden_roi_ids = set()
        # Simple listener callbacks for external widgets (e.g., ROICalcDock)
        # Each listener is called as cb(event: str, roi: Optional[pg.ROI])
        self._listeners: Set[Callable[[str, Optional[pg.ROI]], None]] = set()
        # Track ROI source (file/dataset) for naming and scoping
        # roi_source_by_id[roi_id] = { 'file_path': str|None, 'dataset_path': str|None }
        self.roi_source_by_id = {}
        # Guard to suppress itemChanged recursion when programmatically updating table cells
        self._suppress_table_item_changed = False
        # Track lock state for ROIs (default locked)
        self.locked_roi_ids: Set[int] = set()

    # ----- External listeners -----
    def add_listener(self, callback: Callable[[str, Optional[pg.ROI]], None]) -> None:
        try:
            if callable(callback):
                self._listeners.add(callback)
        except Exception:
            pass

    def remove_listener(self, callback: Callable[[str, Optional[pg.ROI]], None]) -> None:
        try:
            self._listeners.discard(callback)
        except Exception:
            pass

    def _notify_listeners(self, event: str, roi: Optional[pg.ROI] = None) -> None:
        for cb in list(self._listeners):
            try:
                cb(str(event), roi)
            except Exception:
                continue

    # ----- Setup -----
    def setup_docks(self) -> None:
        """Create/attach ROI list dock and ROI stats dock to the main window."""
        try:
            # ROI list dock removed per request
            pass
        except Exception as e:
            self.main.update_status(f"Error setting up ROI dock: {e}", level='error')

        try:
            from viewer.workbench.docks.roi_stats_dock import ROIStatsDock
            dock = ROIStatsDock(main_window=self.main, segment_name="2d", dock_area=Qt.RightDockWidgetArea)
            self.main.roi_stats_dock = dock
            self.show_names_checkbox = dock.show_names_checkbox
            self.main.roi_stats_table = dock.roi_stats_table
            try:
                self.show_names_checkbox.toggled.connect(lambda _: self.update_all_roi_labels())
            except Exception:
                pass
            try:
                dock.visibilityChanged.connect(self.on_roi_stats_dock_visibility_changed)
            except Exception:
                pass
            try:
                self.main.roi_stats_table.itemChanged.connect(self.on_roi_stats_item_changed)
            except Exception:
                pass
        except Exception as e:
            self.main.update_status(f"Error setting up ROI stats dock: {e}", level='error')

    # ----- ROI lifecycle -----
    def create_and_add_roi(self) -> None:
        """Create a new ROI and add it to the image view and docks."""
        try:
            if not hasattr(self.main, 'image_view') or not hasattr(self.main, 'current_2d_data') or self.main.current_2d_data is None:
                self.main.update_status("Please load image data first", level='warning')
                return

            # cycle through a set of distinct colors
            roi_colors = [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
                (255, 0, 255, 255),
                (0, 255, 255, 255),
            ]
            pen = roi_colors[len(self.rois) % len(roi_colors)]
            roi = ContextRectROI(self.main, [50, 50], [100, 100], pen=pen)
            self.main.image_view.addItem(roi)
            self.rois.append(roi)
            # keep main.rois in sync for compatibility
            try:
                if hasattr(self.main, 'rois'):
                    self.main.rois.append(roi)
            except Exception:
                pass
            # track in dock
            try:
                self.add_roi_to_dock(roi)
            except Exception:
                pass
            # current_roi for compatibility
            self.current_roi = roi
            # Record source context for this in-memory ROI
            try:
                src_file = getattr(self.main, 'current_file_path', None)
                src_dset = getattr(self.main, 'selected_dataset_path', None)
                self.roi_source_by_id[id(roi)] = {
                    'file_path': src_file if isinstance(src_file, str) else None,
                    'dataset_path': src_dset if isinstance(src_dset, str) else None,
                }
            except Exception:
                pass
            try:
                roi.sigRegionChanged.connect(lambda r=roi: (self.show_roi_stats_for_roi(r), self.update_roi_item(r), self.refresh_label_for_roi(r)))
                # Also update stats when drag/resize finishes (some pyqtgraph versions emit this)
                if hasattr(roi, 'sigRegionChangeFinished'):
                    roi.sigRegionChangeFinished.connect(lambda r=roi: (self.show_roi_stats_for_roi(r), self.update_roi_item(r), self.refresh_label_for_roi(r)))
            except Exception:
                pass

            # Default lock ON: prevent drag/resize until explicitly unlocked
            try:
                self.lock_roi(roi)
            except Exception:
                pass

            # Populate stats immediately for the new ROI
            try:
                self.show_roi_stats_for_roi(roi)
            except Exception:
                pass

            self.main.update_status("ROI added - drag to position and resize as needed")
            # Notify listeners a new ROI was added
            try:
                self._notify_listeners('added', roi)
            except Exception:
                pass
        except Exception as e:
            self.main.update_status(f"Error drawing ROI: {e}", level='error')

    def set_active_roi(self, roi) -> None:
        try:
            self.current_roi = roi
            self.main.current_roi = roi  # keep main in sync
            self.main.update_status("Active ROI set")
        except Exception as e:
            self.main.update_status(f"Error setting active ROI: {e}", level='error')
    
    def rename_roi(self, roi) -> None:
        """Prompt user to rename an ROI and update docks/stats accordingly."""
        try:
            current_name = self.get_roi_name(roi)
            text, ok = QInputDialog.getText(self.main, "Rename ROI", "Enter ROI name:", text=current_name)
            if ok and str(text).strip():
                new_name = str(text).strip()
                self.roi_names[id(roi)] = new_name
                # update dock list item text
                self.update_roi_item(roi)
                # update stats table name cell if exists (column 1)
                row = self.stats_row_by_roi_id.get(id(roi))
                if row is not None and hasattr(self.main, 'roi_stats_table'):
                    try:
                        self.main.roi_stats_table.setItem(row, 1, QTableWidgetItem(new_name))
                    except Exception:
                        pass
                # update overlay label text if visible
                try:
                    self.refresh_label_for_roi(roi)
                except Exception:
                    pass
                # update dockable ROI plot title if open
                try:
                    if hasattr(self.main, 'update_roi_plot_dock_title'):
                        self.main.update_roi_plot_dock_title(roi)
                except Exception:
                    pass
                self.main.update_status(f"Renamed ROI to '{new_name}'")
                # Notify listeners about rename
                try:
                    self._notify_listeners('renamed', roi)
                except Exception:
                    pass
        except Exception as e:
            self.main.update_status(f"Error renaming ROI: {e}", level='error')

    def _remove_roi_memory(self, roi) -> None:
        """Remove the ROI from UI/memory without touching disk (used by delete/hide flows)."""
        try:
            # remove overlay label if present
            try:
                self.remove_label_for_roi(roi)
            except Exception:
                pass
            if hasattr(self.main, 'image_view'):
                try:
                    self.main.image_view.removeItem(roi)
                except Exception:
                    pass
            if roi in self.rois:
                self.rois.remove(roi)
            # keep main.rois in sync
            try:
                if hasattr(self.main, 'rois') and roi in self.main.rois:
                    self.main.rois.remove(roi)
            except Exception:
                pass
            if getattr(self.main, 'current_roi', None) is roi:
                self.main.current_roi = None
            if self.current_roi is roi:
                self.current_roi = None
            # Update dock list
            try:
                item = self.item_by_roi_id.pop(id(roi), None)
                if item is not None and hasattr(self.main, 'roi_list'):
                    row = self.main.roi_list.row(item)
                    self.main.roi_list.takeItem(row)
                if item in self.roi_by_item:
                    self.roi_by_item.pop(item, None)
            except Exception:
                pass
            # Rebuild ROI stats dock for remaining ROIs
            try:
                if hasattr(self.main, 'roi_stats_table') and self.main.roi_stats_table is not None:
                    self.main.roi_stats_table.setRowCount(0)
                    self.stats_row_by_roi_id = {}
                    self.roi_by_stats_row = {}
                    frame = self.get_current_frame_data()
                    for r in self.rois:
                        s = self.compute_roi_stats(frame, r)
                        if s:
                            self.update_stats_table_for_roi(r, s)
            except Exception:
                pass
            # Notify listeners about deletion
            try:
                self._notify_listeners('deleted', roi)
            except Exception:
                pass
            # Drop source mapping
            try:
                self.roi_source_by_id.pop(id(roi), None)
            except Exception:
                pass
        except Exception:
            pass

    def delete_roi_from_disk(self, roi) -> None:
        """Delete the ROI dataset from disk under /entry/data/rois. Ignores if not present."""
        try:
            # Resolve file path: prefer recorded source mapping, fallback to current file
            src = dict(self.roi_source_by_id.get(id(roi), {}))
            file_path = src.get('file_path') or getattr(self.main, 'current_file_path', None)
            if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
                return

            # Build candidate dataset paths for deletion
            candidates = []
            ds_path = src.get('dataset_path')
            if isinstance(ds_path, str) and ds_path.startswith('/entry/data/rois'):
                candidates.append(ds_path)
            # Fallback using ROI name: raw and sanitized underscores
            try:
                name = self.get_roi_name(roi)
            except Exception:
                name = None
            if isinstance(name, str) and name.strip():
                raw = f"/entry/data/rois/{name}"
                sani = f"/entry/data/rois/{name.replace(' ', '_')}"
                # Avoid duplicates
                for p in (raw, sani):
                    if p not in candidates:
                        candidates.append(p)

            # Delete the first existing candidate
            try:
                with h5py.File(file_path, 'a') as h5f:
                    for p in candidates:
                        try:
                            if p in h5f and isinstance(h5f[p], h5py.Dataset):
                                del h5f[p]
                                self.main.update_status(f"ROI dataset deleted from disk: {p}")
                                break
                        except Exception:
                            # continue to next candidate
                            continue
            except Exception:
                # Silently ignore disk errors per spec
                pass
        except Exception:
            pass

    def delete_roi(self, roi, prompt: bool = True) -> None:
        """Delete an ROI. With prompt=True, shows a confirmation dialog offering Hide/Delete/No.
        - Hide: hides the ROI from view (no disk changes)
        - Delete: removes ROI dataset from disk (when resolvable) and from UI/memory
        - No: cancels
        With prompt=False, removes from UI/memory only (used for internal clear operations).
        """
        try:
            if not prompt:
                self._remove_roi_memory(roi)
                self.main.update_status("ROI removed from workspace")
                return

            # Confirmation dialog
            msg = QMessageBox(self.main)
            msg.setWindowTitle("Delete ROI")
            try:
                msg.setIcon(QMessageBox.Warning)
            except Exception:
                pass
            msg.setText("You are deleting your ROI from disk for this file.")
            try:
                msg.setInformativeText("This action is irreversible. Proceed?")
            except Exception:
                pass

            btn_hide = msg.addButton("Hide", QMessageBox.ActionRole)
            btn_delete = msg.addButton("Delete", QMessageBox.DestructiveRole)
            btn_no = msg.addButton("No", QMessageBox.RejectRole)
            # Style the Delete button in red
            try:
                btn_delete.setStyleSheet("color: white; background-color: #d9534f;")
            except Exception:
                pass

            msg.exec_()
            clicked = msg.clickedButton()

            if clicked is btn_hide:
                try:
                    self.set_roi_visibility(roi, False)
                except Exception:
                    pass
                self.main.update_status("ROI hidden")
                return
            if clicked is btn_delete:
                # Attempt disk deletion first
                try:
                    self.delete_roi_from_disk(roi)
                except Exception:
                    pass
                # Remove from UI/memory
                self._remove_roi_memory(roi)
                self.main.update_status("ROI deleted")
                return
            # No: do nothing
            return
        except Exception as e:
            self.main.update_status(f"Error deleting ROI: {e}", level='error')

    def save_roi(self, roi) -> None:
        """Save the selected ROI to the current HDF5 file under /entry/data/rois with same frame structure."""
        try:
            # Ensure we have a current HDF5 file path
            file_path = getattr(self.main, "current_file_path", None)
            if not file_path or not isinstance(file_path, str):
                self.main.update_status("No current HDF5 file loaded", level='warning')
                return

            # Access current data and image item (for transform-aware extraction)
            data = getattr(self.main, "current_2d_data", None)
            if data is None:
                frame = self.get_current_frame_data()
                if frame is None:
                    self.main.update_status("No image data to save ROI from", level='warning')
                    return
                data = frame
            image_item = getattr(self.main.image_view, 'imageItem', None) if hasattr(self.main, 'image_view') else None

            # Helper: extract ROI subarray from a frame
            def extract_sub(frame):
                sub = None
                try:
                    if image_item is not None:
                        sub = roi.getArrayRegion(frame, image_item)
                        if sub is not None and hasattr(sub, 'ndim') and sub.ndim > 2:
                            sub = np.squeeze(sub)
                except Exception:
                    sub = None
                if sub is None or int(getattr(sub, 'size', 0)) == 0:
                    pos = roi.pos(); size = roi.size()
                    x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
                    w = max(1, int(size.x())); h = max(1, int(size.y()))
                    hgt, wid = frame.shape
                    x1 = min(wid, x0 + w); y1 = min(hgt, y0 + h)
                    if x0 < x1 and y0 < y1:
                        sub = frame[y0:y1, x0:x1]
                return sub

            # Build ROI stack across frames (or single frame for 2D data)
            # Build ROI-only stack: shape is (num_frames, h, w) for 3D data, or (h, w) for 2D
            if isinstance(data, np.ndarray) and data.ndim == 3:
                num_frames = int(data.shape[0])
                samples = []
                for i in range(num_frames):
                    frame = np.asarray(data[i], dtype=np.float32)
                    sub = extract_sub(frame)
                    if sub is None or int(getattr(sub, 'size', 0)) == 0:
                        # Fallback to zero array using current ROI box size
                        size = roi.size(); w = max(1, int(size.x())); h = max(1, int(size.y()))
                        samples.append(np.zeros((h, w), dtype=np.float32))
                    else:
                        samples.append(np.asarray(sub, dtype=np.float32))
                # Ensure consistent shape across frames by trimming to smallest h,w
                min_h = min(s.shape[0] for s in samples)
                min_w = min(s.shape[1] for s in samples)
                roi_stack = np.stack([s[:min_h, :min_w] for s in samples], axis=0)
            else:
                frame = np.asarray(data, dtype=np.float32)
                sub = extract_sub(frame)
                if sub is None or int(getattr(sub, 'size', 0)) == 0:
                    self.main.update_status("ROI appears empty; nothing to save", level='warning')
                    return
                roi_stack = np.asarray(sub, dtype=np.float32)

            # Write to HDF5 under /entry/data/rois
            try:
                with h5py.File(file_path, 'a') as h5f:
                    entry = h5f.require_group('entry')
                    data_grp = entry.get('data')
                    if data_grp is None or not isinstance(data_grp, h5py.Group):
                        data_grp = entry.require_group('data')
                    rois_grp = data_grp.require_group('rois')

                    # Dataset name based on ROI name
                    name = self.get_roi_name(roi)
                    ds_name = str(name).replace(' ', '_')
                    # Replace existing dataset if present
                    if ds_name in rois_grp:
                        try:
                            del rois_grp[ds_name]
                        except Exception:
                            pass
                    dset = rois_grp.create_dataset(ds_name, data=roi_stack, dtype=np.float32)
                    # Attach ROI metadata as dataset attributes: position/size and source dataset path
                    try:
                        pos = roi.pos(); size = roi.size()
                        x = max(0, int(pos.x())); y = max(0, int(pos.y()))
                        w = max(1, int(size.x())); h = max(1, int(size.y()))
                        dset.attrs['x'] = int(x)
                        dset.attrs['y'] = int(y)
                        dset.attrs['w'] = int(w)
                        dset.attrs['h'] = int(h)
                        src_path = getattr(self.main, 'selected_dataset_path', None) or '/entry/data/data'
                        dset.attrs['source_path'] = str(src_path)
                    except Exception:
                        pass

                    # Info group: original file name and frames used (blank for now)
                    info_grp = rois_grp.require_group('info')
                    try:
                        dt = h5py.string_dtype(encoding='utf-8')
                        if 'original_file_name' in info_grp:
                            del info_grp['original_file_name']
                        info_grp.create_dataset('original_file_name', data=np.array(os.path.basename(file_path), dtype=dt))
                    except Exception:
                        pass
                    try:
                        if 'frames' in info_grp:
                            del info_grp['frames']
                        info_grp.create_dataset('frames', data=np.array([], dtype=np.int32))
                    except Exception:
                        pass

                self.main.update_status(f"ROI saved to HDF5 at /entry/data/rois/{ds_name}")
                # Record source mapping for robust deletion later
                try:
                    self.roi_source_by_id[id(roi)] = {
                        'file_path': file_path,
                        'dataset_path': f"/entry/data/rois/{ds_name}",
                    }
                except Exception:
                    pass
            except Exception as e:
                self.main.update_status(f"Error writing ROI to HDF5: {e}", level='error')
        except Exception as e:
            self.main.update_status(f"Error in save_roi: {e}", level='error')

    def clear_all_rois(self) -> None:
        try:
            for r in list(self.rois):
                # Internal clear: no dialogs and no disk edits during dataset switches
                self.delete_roi(r, prompt=False)
            # Notify listeners that all ROIs were cleared
            try:
                self._notify_listeners('cleared', None)
            except Exception:
                pass
        except Exception:
            pass

    def render_rois_for_dataset(self, file_path: str, dataset_path: str) -> None:
        """Render ROI boxes associated with the given dataset by reading HDF5 ROI metadata."""
        try:
            if not file_path or not os.path.exists(file_path):
                return
            with h5py.File(file_path, 'r') as h5f:
                rois_grp = h5f.get('/entry/data/rois')
                if rois_grp is None:
                    return
                # Iterate ROI datasets
                for name in rois_grp.keys():
                    item = rois_grp.get(name)
                    if not isinstance(item, h5py.Dataset):
                        continue
                    src = item.attrs.get('source_path', None)
                    if src is None:
                        continue
                    if str(src) != str(dataset_path):
                        continue
                    # Read xywh
                    x = int(item.attrs.get('x', 0))
                    y = int(item.attrs.get('y', 0))
                    w = int(item.attrs.get('w', max(1, item.shape[-1] if len(item.shape) >= 2 else 1)))
                    h = int(item.attrs.get('h', max(1, item.shape[-2] if len(item.shape) >= 2 else 1)))
                    # Create ROI
                    pen = (255, 0, 0, 255)
                    roi = ContextRectROI(self.main, [x, y], [w, h], pen=pen)
                    try:
                        self.main.image_view.addItem(roi)
                    except Exception:
                        continue
                    # Track in manager structures
                    self.rois.append(roi)
                    try:
                        if hasattr(self.main, 'rois'):
                            self.main.rois.append(roi)
                    except Exception:
                        pass
                    # Name mapping: use dataset name
                    try:
                        self.roi_names[id(roi)] = str(name)
                    except Exception:
                        pass
                    # Record source context for this file-backed ROI
                    try:
                        self.roi_source_by_id[id(roi)] = {
                            'file_path': file_path,
                            'dataset_path': dataset_path,
                        }
                    except Exception:
                        pass
                    # Wire signals and populate stats
                    try:
                        roi.sigRegionChanged.connect(lambda r=roi: (self.show_roi_stats_for_roi(r), self.update_roi_item(r), self.refresh_label_for_roi(r)))
                        if hasattr(roi, 'sigRegionChangeFinished'):
                            roi.sigRegionChangeFinished.connect(lambda r=roi: (self.show_roi_stats_for_roi(r), self.update_roi_item(r), self.refresh_label_for_roi(r)))
                        self.show_roi_stats_for_roi(roi)
                    except Exception:
                        pass
                    # Default lock ON for file-rendered ROIs
                    try:
                        self.lock_roi(roi)
                    except Exception:
                        pass
                # Notify listeners each time an ROI is rendered from file metadata
                try:
                    self._notify_listeners('added', roi)
                except Exception:
                    pass
        except Exception:
            pass

    # ----- Source info API -----
    def get_roi_source(self, roi) -> dict:
        """Return {'file_path': str|None, 'dataset_path': str|None} for a given ROI."""
        try:
            return dict(self.roi_source_by_id.get(id(roi), {}))
        except Exception:
            return {'file_path': None, 'dataset_path': None}

    # ----- ROI stats -----
    def get_current_frame_data(self):
        """Return the image currently displayed in the ImageView.
        Falls back to the underlying current_2d_data if the ImageView has no image yet.
        This ensures ROI stats reflect what the user sees (including frame/log/levels changes).
        """
        try:
            # Prefer the image currently displayed in the ImageView
            img = None
            try:
                if hasattr(self.main, 'image_view'):
                    # Try ImageView.getImage() first (includes display transforms)
                    if hasattr(self.main.image_view, 'getImage'):
                        try:
                            img = self.main.image_view.getImage()
                            if img is not None:
                                arr = np.asarray(img, dtype=np.float32)
                                if isinstance(arr, tuple) and len(arr) > 0:
                                    arr = np.asarray(arr[0], dtype=np.float32)
                                if arr.ndim == 3:
                                    arr = np.asarray(arr[0], dtype=np.float32)
                                if arr.ndim == 2 and arr.size > 0:

                                    return arr
                        except Exception:
                            pass
                    # Fallback to imageItem.image
                    if hasattr(self.main.image_view, 'imageItem') and self.main.image_view.imageItem is not None:
                        img = getattr(self.main.image_view.imageItem, 'image', None)
            except Exception:
                img = None

            if img is not None:
                arr = np.asarray(img, dtype=np.float32)
                # Some versions store a tuple (data, ...); ensure we pick array
                if isinstance(arr, tuple) and len(arr) > 0:
                    arr = np.asarray(arr[0], dtype=np.float32)
                # Ensure 2D slice
                if arr.ndim == 3:
                    # Use first frame if a 3D stack somehow made it to imageItem
                    arr = np.asarray(arr[0], dtype=np.float32)
                if arr.ndim == 2 and arr.size > 0:

                    return arr

            # Fallback: use the underlying data model
            if not hasattr(self.main, 'current_2d_data') or self.main.current_2d_data is None:
                return None
            if self.main.current_2d_data.ndim == 3:
                frame_index = 0
                if hasattr(self.main, 'frame_spinbox') and self.main.frame_spinbox.isEnabled():
                    frame_index = self.main.frame_spinbox.value()
                if frame_index < 0 or frame_index >= self.main.current_2d_data.shape[0]:
                    frame_index = 0
                arr = np.asarray(self.main.current_2d_data[frame_index], dtype=np.float32)

                return arr
            else:
                arr = np.asarray(self.main.current_2d_data, dtype=np.float32)

                return arr
        except Exception:
            return None

    def compute_roi_stats(self, frame_data, roi):
        """Compute stats for ROI using pyqtgraph's array-extraction helpers to honor image/item transforms.
        Falls back to bounding-box slicing if needed. Returns None if ROI is empty/out-of-bounds.
        """
        try:
            if frame_data is None or roi is None:
                return None

            # Try to extract ROI region via pyqtgraph (handles scale/transform/orientation)
            image_item = getattr(self.main.image_view, 'imageItem', None) if hasattr(self.main, 'image_view') else None
            sub = None
            try:
                if image_item is not None:
                    sub = roi.getArrayRegion(frame_data, image_item)
                    if sub is not None and hasattr(sub, 'ndim'):
                        # Ensure 2D (some returns may add an extra dim)
                        if sub.ndim > 2:
                            sub = np.squeeze(sub)
            except Exception:
                sub = None

            # Compute xywh using getArraySlice if possible (pixel-space), else fallback to ROI pos/size
            x0 = y0 = w = h = None
            try:
                if image_item is not None:
                    slc_info = roi.getArraySlice(frame_data, image_item)
                    # slc_info returns (slices, transform). slices is typically a tuple of (rows, cols)
                    slices = slc_info[0] if (isinstance(slc_info, (tuple, list)) and len(slc_info) > 0) else None
                    if isinstance(slices, (tuple, list)) and len(slices) >= 2:
                        rs, cs = slices[0], slices[1]
                        # Handle slice or numpy index arrays
                        def _bounds_from_index(idx, maxdim):
                            try:
                                if isinstance(idx, slice):
                                    start = int(0 if idx.start is None else idx.start)
                                    stop = int(maxdim if idx.stop is None else idx.stop)
                                    return start, stop
                                idx_arr = np.asarray(idx)
                                if idx_arr.size > 0:
                                    return int(np.min(idx_arr)), int(np.max(idx_arr) + 1)
                            except Exception:
                                pass
                            return 0, maxdim
                        y0_, y1_ = _bounds_from_index(rs, frame_data.shape[0])
                        x0_, x1_ = _bounds_from_index(cs, frame_data.shape[1])
                        x0, y0 = max(0, x0_), max(0, y0_)
                        w = max(0, x1_ - x0_)
                        h = max(0, y1_ - y0_)
            except Exception:
                pass

            # If array-region failed or slices produced invalid region, fallback to bounding box from ROI pos/size
            try:
                if sub is None or (hasattr(sub, 'size') and int(sub.size) == 0) or any(v is None for v in (x0, y0, w, h)):
                    height, width = frame_data.shape
                    pos = roi.pos(); size = roi.size()
                    x0 = max(0, int(pos.x())); y0 = max(0, int(pos.y()))
                    w = max(1, int(size.x())); h = max(1, int(size.y()))
                    x1 = min(width, x0 + w); y1 = min(height, y0 + h)
                    if x0 >= x1 or y0 >= y1:
                        return None
                    sub = frame_data[y0:y1, x0:x1]
                    w = x1 - x0; h = y1 - y0
            except Exception:
                return None

            # Final safety: ensure we have a valid sub-region
            if sub is None or int(sub.size) == 0:
                return None

            stats = {
                'x': int(x0) if x0 is not None else 0,
                'y': int(y0) if y0 is not None else 0,
                'w': int(w) if w is not None else int(sub.shape[1]) if sub.ndim == 2 else 0,
                'h': int(h) if h is not None else int(sub.shape[0]) if sub.ndim == 2 else 0,
                'sum': float(np.sum(sub)),
                'min': float(np.min(sub)),
                'max': float(np.max(sub)),
                'mean': float(np.mean(sub)),
                'std': float(np.std(sub)),
                'count': int(sub.size),
            }

            return stats
        except Exception:
            return None

    def show_roi_stats_for_roi(self, roi) -> None:
        try:
            frame = self.get_current_frame_data()
            stats = self.compute_roi_stats(frame, roi)
            if stats is None:
                self.main.update_status("ROI stats unavailable", level='warning')
                return
            text = (f"ROI [{stats['x']},{stats['y']} {stats['w']}x{stats['h']}] | "
                    f"sum={stats['sum']:.3f} min={stats['min']:.3f} max={stats['max']:.3f} "
                    f"mean={stats['mean']:.3f} std={stats['std']:.3f} count={stats['count']}")
            if hasattr(self.main, 'roi_stats_label') and self.main.roi_stats_label is not None:
                try:
                    self.main.roi_stats_label.setText(f"ROI Stats: {text}")
                except Exception:
                    pass
            try:
                self.update_stats_table_for_roi(roi, stats)
            except Exception:
                pass
            self.main.update_status("ROI stats computed")
        except Exception as e:
            self.main.update_status(f"Error showing ROI stats: {e}", level='error')

    # ----- Batch stats refresh -----
    def update_all_roi_stats(self):
        try:
            frame = self.get_current_frame_data()
            if frame is None:
                return
            for r in list(self.rois):
                s = self.compute_roi_stats(frame, r)
                if s:
                    self.update_stats_table_for_roi(r, s)
        except Exception:
            pass

    # ----- Dock/list helpers -----
    def format_roi_text(self, roi):
        try:
            pos = roi.pos(); size = roi.size()
            x = int(pos.x()); y = int(pos.y())
            w = int(size.x()); h = int(size.y())
            name = self.get_roi_name(roi)
            return f"{name}: x={x}, y={y}, w={w}, h={h}"
        except Exception:
            return "ROI"

    def get_roi_name(self, roi):
        try:
            name = self.roi_names.get(id(roi))
            if name:
                return name
            idx = 1
            if roi in self.rois:
                idx = self.rois.index(roi) + 1
            name = f"ROI {idx}"
            self.roi_names[id(roi)] = name
            return name
        except Exception:
            return "ROI"

    def add_roi_to_dock(self, roi):
        try:
            if not hasattr(self.main, 'roi_list') or self.main.roi_list is None:
                return
            text = self.format_roi_text(roi)
            item = QListWidgetItem(text)
            self.main.roi_list.addItem(item)
            self.roi_by_item[item] = roi
            self.item_by_roi_id[id(roi)] = item
        except Exception as e:
            self.main.update_status(f"Error adding ROI to dock: {e}", level='error')

    def update_roi_item(self, roi):
        try:
            item = self.item_by_roi_id.get(id(roi))
            if item is not None:
                item.setText(self.format_roi_text(roi))
        except Exception:
            pass

    def on_roi_list_item_clicked(self, item):
        try:
            roi = self.roi_by_item.get(item)
            if roi:
                self.set_active_roi(roi)
        except Exception as e:
            self.main.update_status(f"Error selecting ROI from dock: {e}", level='error')

    def on_roi_list_item_double_clicked(self, item):
        try:
            roi = self.roi_by_item.get(item)
            if roi:
                self.show_roi_stats_for_roi(roi)
        except Exception as e:
            self.main.update_status(f"Error showing ROI stats from dock: {e}", level='error')

    def on_roi_stats_item_changed(self, item):
        """Respond to selection checkbox toggles and name edits in the ROI stats table."""
        try:
            # Avoid recursion when we programmatically update cells
            if getattr(self, '_suppress_table_item_changed', False):
                return
            if item is None:
                return
            row = item.row()
            col = item.column()
            roi = self.roi_by_stats_row.get(row)
            if not roi:
                return
            if col == 0:
                # Selection checkbox toggled
                self.update_label_visibility_for_roi(roi)
            elif col == 2:
                # Name edited; update internal mapping and overlay label
                try:
                    new_name = item.text() if hasattr(item, 'text') else None
                    if new_name:
                        self.roi_names[id(roi)] = str(new_name)
                        # update overlay label text if visible
                        self.refresh_label_for_roi(roi)
                        # update dockable ROI plot title if open
                        try:
                            if hasattr(self.main, 'update_roi_plot_dock_title'):
                                self.main.update_roi_plot_dock_title(roi)
                        except Exception:
                            pass
                        # Update any dock list item if present
                        try:
                            self.update_roi_item(roi)
                        except Exception:
                            pass
                except Exception:
                    pass
            elif col in (9, 10, 11, 12):
                # x, y, w, h edits: apply to the ROI, clamp to image bounds, and refresh stats/label
                # If locked, ignore XYWH edits
                try:
                    if self.is_locked(roi):
                        return
                except Exception:
                    pass
                try:
                    frame = self.get_current_frame_data()
                    if frame is None or frame.ndim != 2:
                        return
                    height, width = int(frame.shape[0]), int(frame.shape[1])
                    # Current ROI box
                    pos = roi.pos(); size = roi.size()
                    cur_x = max(0, int(pos.x()))
                    cur_y = max(0, int(pos.y()))
                    cur_w = max(1, int(size.x()))
                    cur_h = max(1, int(size.y()))
                    # Parse new value
                    try:
                        new_val = int(item.text())
                    except Exception:
                        new_val = None
                    if new_val is None:
                        return
                    new_x, new_y, new_w, new_h = cur_x, cur_y, cur_w, cur_h
                    if col == 9:  # x
                        new_x = max(0, min(int(new_val), max(0, width - 1)))
                        # ensure width fits
                        if new_x + new_w > width:
                            new_w = max(1, width - new_x)
                    elif col == 10:  # y
                        new_y = max(0, min(int(new_val), max(0, height - 1)))
                        if new_y + new_h > height:
                            new_h = max(1, height - new_y)
                    elif col == 11:  # w
                        new_w = max(1, int(new_val))
                        if new_x + new_w > width:
                            new_w = max(1, width - new_x)
                    elif col == 12:  # h
                        new_h = max(1, int(new_val))
                        if new_y + new_h > height:
                            new_h = max(1, height - new_y)
                    # Apply to ROI
                    try:
                        roi.setPos((float(new_x), float(new_y)))
                    except Exception:
                        pass
                    try:
                        roi.setSize((float(new_w), float(new_h)))
                    except Exception:
                        pass
                    # Recompute and refresh stats row items without retriggering
                    stats = self.compute_roi_stats(frame, roi)
                    if stats:
                        prev_guard = self._suppress_table_item_changed
                        self._suppress_table_item_changed = True
                        try:
                            # Update editable xywh and numeric cells
                            self.main.roi_stats_table.setItem(row, 3, QTableWidgetItem(f"{stats['sum']:.3f}"))
                            self.main.roi_stats_table.setItem(row, 4, QTableWidgetItem(f"{stats['min']:.3f}"))
                            self.main.roi_stats_table.setItem(row, 5, QTableWidgetItem(f"{stats['max']:.3f}"))
                            self.main.roi_stats_table.setItem(row, 6, QTableWidgetItem(f"{stats['mean']:.3f}"))
                            self.main.roi_stats_table.setItem(row, 7, QTableWidgetItem(f"{stats['std']:.3f}"))
                            self.main.roi_stats_table.setItem(row, 8, QTableWidgetItem(str(stats['count'])))
                            self.main.roi_stats_table.setItem(row, 9, QTableWidgetItem(str(stats['x'])))
                            self.main.roi_stats_table.setItem(row, 10, QTableWidgetItem(str(stats['y'])))
                            self.main.roi_stats_table.setItem(row, 11, QTableWidgetItem(str(stats['w'])))
                            self.main.roi_stats_table.setItem(row, 12, QTableWidgetItem(str(stats['h'])))
                        except Exception:
                            pass
                        self._suppress_table_item_changed = prev_guard
                    # Refresh overlay label position
                    try:
                        self.refresh_label_for_roi(roi)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def on_rois_dock_visibility_changed(self, visible):
        try:
            if hasattr(self.main, 'action_show_rois_dock'):
                self.main.action_show_rois_dock.setChecked(bool(visible))
        except Exception:
            pass

    def on_roi_stats_dock_visibility_changed(self, visible):
        try:
            if hasattr(self.main, 'action_show_roi_stats_dock'):
                self.main.action_show_roi_stats_dock.setChecked(bool(visible))
        except Exception:
            pass

    # ----- Overlay label helpers -----
    def update_all_roi_labels(self):
        try:
            for roi in list(self.rois):
                self.update_label_visibility_for_roi(roi)
        except Exception:
            pass

    def update_label_visibility_for_roi(self, roi):
        """Show/hide ROI name label above ROI depending on selection and checkbox."""
        try:
            # Determine selection state from stats table
            row = self.stats_row_by_roi_id.get(id(roi))
            selected = False
            if row is not None and hasattr(self.main, 'roi_stats_table'):
                try:
                    sel_item = self.main.roi_stats_table.item(row, 0)
                    selected = bool(sel_item) and sel_item.checkState() == Qt.Checked
                except Exception:
                    selected = False
            show_names = bool(self.show_names_checkbox and self.show_names_checkbox.isChecked())
            if selected and show_names:
                # ensure label exists and update position/text
                self.create_label_for_roi(roi)
                self.refresh_label_for_roi(roi)
            else:
                # hide/remove label for this ROI
                self.remove_label_for_roi(roi)
        except Exception:
            pass

    def create_label_for_roi(self, roi):
        try:
            if id(roi) in self.roi_label_by_id:
                return
            if not hasattr(self.main, 'image_view'):
                return
            name = self.get_roi_name(roi)
            label = pg.TextItem(text=name, color='w')
            try:
                label.setAnchor((0, 1))  # bottom-left anchor
            except Exception:
                pass
            self.main.image_view.addItem(label)
            self.roi_label_by_id[id(roi)] = label
        except Exception:
            pass

    def refresh_label_for_roi(self, roi):
        try:
            label = self.roi_label_by_id.get(id(roi))
            if not label:
                return
            name = self.get_roi_name(roi)
            try:
                label.setText(name)
            except Exception:
                pass
            pos = roi.pos()
            x = float(getattr(pos, 'x', lambda: 0)())
            y = float(getattr(pos, 'y', lambda: 0)())
            # place just above the ROI box
            y = max(0.0, y - 5.0)
            try:
                label.setPos(x, y)
            except Exception:
                pass
        except Exception:
            pass

    def remove_label_for_roi(self, roi):
        try:
            label = self.roi_label_by_id.pop(id(roi), None)
            if label and hasattr(self.main, 'image_view'):
                try:
                    self.main.image_view.removeItem(label)
                except Exception:
                    pass
        except Exception:
            pass

    def set_roi_visibility(self, roi, visible: bool):
        """Hide/show the ROI graphics and related overlay label using QtAwesome controls."""
        try:
            if not hasattr(self.main, 'image_view'):
                return
            try:
                roi.setVisible(bool(visible))
            except Exception:
                pass
            if visible:
                try:
                    self.hidden_roi_ids.discard(id(roi))
                except Exception:
                    pass
                self.update_label_visibility_for_roi(roi)
            else:
                try:
                    self.hidden_roi_ids.add(id(roi))
                except Exception:
                    pass
                self.remove_label_for_roi(roi)
        except Exception:
            pass

    # ----- Stats table helpers -----
    def ensure_stats_row_for_roi(self, roi):
        try:
            if id(roi) in self.stats_row_by_roi_id:
                return self.stats_row_by_roi_id[id(roi)]
            if not hasattr(self.main, 'roi_stats_table') or self.main.roi_stats_table is None:
                return None
            # Suppress recursive itemChanged while inserting a new row and items
            prev_guard = self._suppress_table_item_changed
            self._suppress_table_item_changed = True
            row = self.main.roi_stats_table.rowCount()
            self.main.roi_stats_table.insertRow(row)
            self.stats_row_by_roi_id[id(roi)] = row
            self.roi_by_stats_row[row] = roi
            # selection checkbox in column 0
            try:
                select_item = QTableWidgetItem()
                select_item.setFlags(select_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                select_item.setCheckState(Qt.Unchecked)
                self.main.roi_stats_table.setItem(row, 0, select_item)
            except Exception:
                pass
            # actions widget in column 1: hide/show and delete using QtAwesome icons
            try:
                actions_widget = QWidget()
                h = QHBoxLayout(actions_widget)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(2)
                icon_visible = qta.icon('fa.eye', color='black')
                icon_hidden = qta.icon('fa.eye-slash', color='black')
                icon_trash = qta.icon('fa.trash', color='black')
                icon_lock = qta.icon('fa.lock', color='black')
                icon_unlock = qta.icon('fa.unlock', color='black')
                # Match checkbox indicator size
                try:
                    style = getattr(self.main, 'style', lambda: None)()
                    indicator_w = style.pixelMetric(QStyle.PM_IndicatorWidth) if style else 16
                    indicator_h = style.pixelMetric(QStyle.PM_IndicatorHeight) if style else 16
                except Exception:
                    indicator_w, indicator_h = 16, 16
                icon_size = QSize(indicator_w, indicator_h)
                btn_eye = QToolButton(actions_widget)
                btn_eye.setAutoRaise(True)
                btn_eye.setCheckable(True)
                visible = id(roi) not in self.hidden_roi_ids
                btn_eye.setChecked(visible)
                btn_eye.setIcon(icon_visible if visible else icon_hidden)
                btn_eye.setIconSize(icon_size)
                try:
                    btn_eye.setFixedSize(icon_size.width()+4, icon_size.height()+4)
                except Exception:
                    pass
                btn_eye.setToolTip("Hide/Show ROI")
                btn_trash = QToolButton(actions_widget)
                btn_trash.setAutoRaise(True)
                btn_trash.setIcon(icon_trash)
                btn_trash.setIconSize(icon_size)
                try:
                    btn_trash.setFixedSize(icon_size.width()+4, icon_size.height()+4)
                except Exception:
                    pass
                btn_trash.setToolTip("Delete ROI")
                # wire actions
                def on_eye_toggled(checked, r=roi, b=btn_eye):
                    try:
                        self.set_roi_visibility(r, bool(checked))
                        b.setIcon(icon_visible if bool(checked) else icon_hidden)
                    except Exception:
                        pass
                btn_eye.toggled.connect(on_eye_toggled)
                # Lock/unlock toggle
                btn_lock = QToolButton(actions_widget)
                btn_lock.setAutoRaise(True)
                locked = self.is_locked(roi)
                btn_lock.setCheckable(True)
                btn_lock.setChecked(locked)
                btn_lock.setIcon(icon_lock if locked else icon_unlock)
                btn_lock.setIconSize(icon_size)
                try:
                    btn_lock.setFixedSize(icon_size.width()+4, icon_size.height()+4)
                except Exception:
                    pass
                btn_lock.setToolTip("Lock/Unlock ROI (affects drag/resize and XYWH editing)")
                def on_lock_toggled(checked, r=roi, b=btn_lock):
                    try:
                        if bool(checked):
                            self.lock_roi(r)
                            b.setIcon(icon_lock)
                        else:
                            self.unlock_roi(r)
                            b.setIcon(icon_unlock)
                    except Exception:
                        pass
                btn_lock.toggled.connect(on_lock_toggled)
                # "Lock all except this" convenience
                btn_lock_others = QToolButton(actions_widget)
                btn_lock_others.setAutoRaise(True)
                btn_lock_others.setIcon(icon_lock)
                btn_lock_others.setIconSize(icon_size)
                try:
                    btn_lock_others.setFixedSize(icon_size.width()+4, icon_size.height()+4)
                except Exception:
                    pass
                btn_lock_others.setToolTip("Lock all except this ROI")
                btn_lock_others.clicked.connect(lambda _: self.lock_all_except(roi))
                btn_trash.clicked.connect(lambda _, r=roi: self.delete_roi(r))
                h.addWidget(btn_eye)
                h.addWidget(btn_lock)
                h.addWidget(btn_lock_others)
                h.addWidget(btn_trash)
                h.addStretch(1)
                self.main.roi_stats_table.setCellWidget(row, 1, actions_widget)
                try:
                    self.main.roi_stats_table.setRowHeight(row, icon_size.height()+6)
                except Exception:
                    pass
            except Exception:
                pass
            # set name cell (column 2)
            name = self.get_roi_name(roi)
            self.main.roi_stats_table.setItem(row, 2, QTableWidgetItem(name))
            # Restore guard
            self._suppress_table_item_changed = prev_guard
            return row
        except Exception:
            return None

    def update_stats_table_for_roi(self, roi, stats):
        try:
            row = self.ensure_stats_row_for_roi(roi)
            if row is None:
                return
            # Suppress itemChanged while we programmatically update cells
            prev_guard = self._suppress_table_item_changed
            self._suppress_table_item_changed = True
            # keep name cell in sync (column 2)
            self.main.roi_stats_table.setItem(row, 2, QTableWidgetItem(self.get_roi_name(roi)))
            # fill numeric cells with xywh at the end starting column 3
            self.main.roi_stats_table.setItem(row, 3, QTableWidgetItem(f"{stats['sum']:.3f}"))
            self.main.roi_stats_table.setItem(row, 4, QTableWidgetItem(f"{stats['min']:.3f}"))
            self.main.roi_stats_table.setItem(row, 5, QTableWidgetItem(f"{stats['max']:.3f}"))
            self.main.roi_stats_table.setItem(row, 6, QTableWidgetItem(f"{stats['mean']:.3f}"))
            self.main.roi_stats_table.setItem(row, 7, QTableWidgetItem(f"{stats['std']:.3f}"))
            self.main.roi_stats_table.setItem(row, 8, QTableWidgetItem(str(stats['count'])))
            # Make xywh editable by default
            self.main.roi_stats_table.setItem(row, 9, QTableWidgetItem(str(stats['x'])))
            self.main.roi_stats_table.setItem(row, 10, QTableWidgetItem(str(stats['y'])))
            self.main.roi_stats_table.setItem(row, 11, QTableWidgetItem(str(stats['w'])))
            self.main.roi_stats_table.setItem(row, 12, QTableWidgetItem(str(stats['h'])))
            # Apply editability based on lock state
            try:
                self.update_xywh_editability_for_roi(roi)
            except Exception:
                pass
            # Restore guard
            self._suppress_table_item_changed = prev_guard
        except Exception:
            pass

    # ----- Locking API -----
    def is_locked(self, roi) -> bool:
        try:
            return id(roi) in self.locked_roi_ids
        except Exception:
            return False

    def lock_roi(self, roi) -> None:
        try:
            self.locked_roi_ids.add(id(roi))
            # Block left-drag/resize
            try:
                roi.setMovable(False)
            except Exception:
                pass
            try:
                roi.setAcceptedMouseButtons(Qt.RightButton)
            except Exception:
                pass
            # Update stats table editability
            try:
                self.update_xywh_editability_for_roi(roi)
            except Exception:
                pass
            # Notify listeners so other UI (e.g., ROICalcDock) can reflect state
            try:
                self._notify_listeners('lock-changed', roi)
            except Exception:
                pass
        except Exception:
            pass

    def unlock_roi(self, roi) -> None:
        try:
            self.locked_roi_ids.discard(id(roi))
            # Restore left/right mouse for drag/resize
            try:
                roi.setMovable(True)
            except Exception:
                pass
            try:
                roi.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
            except Exception:
                pass
            # Update stats table editability
            try:
                self.update_xywh_editability_for_roi(roi)
            except Exception:
                pass
            # Notify listeners
            try:
                self._notify_listeners('lock-changed', roi)
            except Exception:
                pass
        except Exception:
            pass

    def lock_all_rois(self) -> None:
        try:
            for r in list(self.rois):
                self.lock_roi(r)
        except Exception:
            pass

    def unlock_all_rois(self) -> None:
        try:
            for r in list(self.rois):
                self.unlock_roi(r)
        except Exception:
            pass

    def lock_all_except(self, roi) -> None:
        try:
            for r in list(self.rois):
                if r is roi:
                    self.unlock_roi(r)
                else:
                    self.lock_roi(r)
        except Exception:
            pass

    def update_xywh_editability_for_roi(self, roi) -> None:
        """Set XYWH cells in stats table to editable when unlocked, non-editable when locked."""
        try:
            row = self.stats_row_by_roi_id.get(id(roi))
            if row is None or not hasattr(self.main, 'roi_stats_table') or self.main.roi_stats_table is None:
                return
            locked = self.is_locked(roi)
            for col in (9, 10, 11, 12):
                item = self.main.roi_stats_table.item(row, col)
                if item is None:
                    continue
                flags = item.flags()
                if locked:
                    try:
                        item.setFlags(flags & ~Qt.ItemIsEditable)
                    except Exception:
                        pass
                else:
                    try:
                        item.setFlags(flags | Qt.ItemIsEditable)
                    except Exception:
                        pass
        except Exception:
            pass
