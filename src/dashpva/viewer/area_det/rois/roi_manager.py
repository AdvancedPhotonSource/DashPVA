"""Software-ROI manager for the area-detector live viewer.

Owns user-drawn :class:`ContextRectROI` items that live *alongside* (and never
touch) the viewer's read-only PV-driven ROIs. Stats (sum/mean/min/max/std/count
and center-of-mass) are computed locally from each live frame; a rolling
per-ROI history feeds the live 1D/2D plot docks.

Adapted from the workbench ``ROIManager`` with all saved-file/stack coupling
removed — it works on the single most-recent frame (``viewer.image``).

Example:
    mgr = AreaDetRoiManager(viewer)
    mgr.create_and_add_roi()        # adds a draggable rectangle
    mgr.on_new_frame()              # called each timer tick; refreshes stats
"""

import collections
import json
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog, QFileDialog, QInputDialog, QMessageBox

import dashpva.settings as settings
from dashpva.gui.theme_colors import ROI_COLORS
from dashpva.viewer.area_det.rois.context_roi import ContextLineROI, ContextRectROI


class AreaDetRoiManager:
    """Manages the viewer's interactive software ROIs and their live stats."""

    def __init__(self, main_window):
        self.main = main_window
        self.rois: list = []
        # User-drawn line cuts (ContextLineROI). Managed alongside the rectangular
        # ROIs — same names/colors/stats table — but sampled as a 1D intensity
        # profile (see compute_cut_stats) and plotted in the Line Cuts dock.
        self.cuts: list = []
        self.cut_ids: set = set()
        # id(cut) -> latest 1D intensity profile (np.ndarray) for the Line Cuts dock.
        self.cut_profiles: dict = {}
        self.current_roi: Optional[pg.ROI] = None
        self.roi_names: dict = {}
        self.hidden_roi_ids: set = set()
        self.history: dict = {}
        self._listeners: set = set()
        self._last_frame_count = -1
        # Read-only PV-driven ROIs (ROI1-4) mirrored from the viewer's overlays,
        # kept name-ordered. Stats are computed locally like software ROIs, but
        # geometry is owned by EPICS so they can't be renamed/resized/deleted.
        self.pv_rois: dict = {}
        self.pv_roi_ids: set = set()
        # Custom per-ROI colors (id -> hex); falls back to the ROI_COLORS palette.
        self.roi_colors: dict = {}
        # Set by SoftwareRoiDock so numeric cells can be refreshed each frame.
        self.stats_dock = None

    # ----- Listener bus -----
    def add_listener(self, callback: Callable[[str, Optional[pg.ROI]], None]) -> None:
        if callable(callback):
            self._listeners.add(callback)

    def remove_listener(self, callback: Callable[[str, Optional[pg.ROI]], None]) -> None:
        self._listeners.discard(callback)

    def _notify(self, event: str, roi: Optional[pg.ROI] = None) -> None:
        for cb in list(self._listeners):
            try:
                cb(str(event), roi)
            except Exception:
                continue

    # ----- Lifecycle -----
    def create_and_add_roi(self, pos=None, size=None, name=None) -> Optional[pg.ROI]:
        try:
            color = ROI_COLORS[len(self.rois) % len(ROI_COLORS)]
            roi = ContextRectROI(self.main, list(pos or [50, 50]),
                                 list(size or [100, 100]),
                                 pen=pg.mkPen(color, width=2))
            self.main.image_view.addItem(roi)
            self.rois.append(roi)
            if name:
                self.roi_names[id(roi)] = str(name)
            self.history[id(roi)] = collections.deque(maxlen=settings.ROI_HISTORY_MAXLEN)
            self.current_roi = roi
            roi.sigRegionChanged.connect(lambda r=roi: self._on_region_changed(r))
            if hasattr(roi, 'sigRegionChangeFinished'):
                roi.sigRegionChangeFinished.connect(lambda r=roi: self._on_region_changed(r))
            self._notify('added', roi)
            self.show_roi_stats_for_roi(roi)
            self.main.update_status(f"Added {self.get_roi_name(roi)} — drag to position and resize")
            return roi
        except Exception as e:
            self.main.update_status(f"Error adding ROI: {e}", level='error')
            return None

    def add_roi_from_geom(self, name, x, y, w, h) -> Optional[pg.ROI]:
        """Recreate an ROI at explicit geometry (used to restore from QSettings)."""
        return self.create_and_add_roi(pos=[x, y], size=[max(1, w), max(1, h)], name=name)

    def create_and_add_cut(self, positions=None, name=None) -> Optional[pg.ROI]:
        """Add a draggable two-point line cut; profile is drawn in the Line Cuts dock."""
        try:
            color = ROI_COLORS[len(self.cuts) % len(ROI_COLORS)]
            pts = [list(p) for p in (positions or settings.LINE_CUT_DEFAULT)]
            cut = ContextLineROI(self.main, pts,
                                 pen=pg.mkPen(color, width=settings.LINE_CUT_PEN_WIDTH))
            self.main.image_view.addItem(cut)
            self.cuts.append(cut)
            self.cut_ids.add(id(cut))
            if name:
                self.roi_names[id(cut)] = str(name)
            self.history[id(cut)] = collections.deque(maxlen=settings.ROI_HISTORY_MAXLEN)
            self.current_roi = cut
            cut.sigRegionChanged.connect(lambda r=cut: self._on_region_changed(r))
            if hasattr(cut, 'sigRegionChangeFinished'):
                cut.sigRegionChangeFinished.connect(lambda r=cut: self._on_region_changed(r))
            self._notify('added', cut)
            self.show_roi_stats_for_roi(cut)
            self.main.update_status(f"Added {self.get_roi_name(cut)} — drag endpoints to position")
            return cut
        except Exception as e:
            self.main.update_status(f"Error adding line cut: {e}", level='error')
            return None

    def set_active_roi(self, roi) -> None:
        self.current_roi = roi
        self.main.update_status(f"Active ROI: {self.get_roi_name(roi)}")

    def rename_roi(self, roi) -> None:
        try:
            current = self.get_roi_name(roi)
            text, ok = QInputDialog.getText(self.main, "Rename ROI", "Enter ROI name:", text=current)
            if ok and str(text).strip():
                self.roi_names[id(roi)] = str(text).strip()
                self._notify('renamed', roi)
                self.main.update_status(f"Renamed ROI to '{self.roi_names[id(roi)]}'")
        except Exception as e:
            self.main.update_status(f"Error renaming ROI: {e}", level='error')

    def set_roi_visibility(self, roi, visible: bool) -> None:
        try:
            roi.setVisible(bool(visible))
            if visible:
                self.hidden_roi_ids.discard(id(roi))
            else:
                self.hidden_roi_ids.add(id(roi))
        except Exception:
            pass

    def delete_roi(self, roi, prompt: bool = True) -> None:
        try:
            if prompt:
                reply = QMessageBox.question(
                    self.main, "Delete ROI",
                    f"Delete {self.get_roi_name(roi)}?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
            try:
                self.main.image_view.removeItem(roi)
            except Exception:
                pass
            if roi in self.rois:
                self.rois.remove(roi)
            if roi in self.cuts:
                self.cuts.remove(roi)
            self.cut_ids.discard(id(roi))
            self.cut_profiles.pop(id(roi), None)
            self.roi_names.pop(id(roi), None)
            self.roi_colors.pop(id(roi), None)
            self.history.pop(id(roi), None)
            self.hidden_roi_ids.discard(id(roi))
            if self.current_roi is roi:
                self.current_roi = None
            self._notify('deleted', roi)
        except Exception as e:
            self.main.update_status(f"Error deleting ROI: {e}", level='error')

    def clear_all_rois(self) -> None:
        for r in list(self.rois) + list(self.cuts):
            self.delete_roi(r, prompt=False)
        self._notify('cleared', None)

    # ----- Read-only PV ROIs -----
    def all_rois(self) -> list:
        """Table/plot order: user-drawn ROIs, then line cuts, then read-only PV ROIs."""
        return list(self.rois) + list(self.cuts) + list(self.pv_rois.values())

    def is_readonly(self, roi) -> bool:
        return id(roi) in self.pv_roi_ids

    def is_cut(self, roi) -> bool:
        return id(roi) in self.cut_ids

    def set_pv_rois(self, overlays: dict) -> None:
        """Mirror the viewer's PV overlay dict ({name: pg.ROI}) as read-only ROIs.

        Called after the viewer (re)builds its EPICS ROI overlays. Tears down any
        stale PV entries (and their open plot docks) first so ids don't leak."""
        self.clear_pv_rois(notify=False)
        for name, roi in (overlays or {}).items():
            self.pv_rois[name] = roi
            self.pv_roi_ids.add(id(roi))
            self.roi_names[id(roi)] = str(name)
            self.history[id(roi)] = collections.deque(maxlen=settings.ROI_HISTORY_MAXLEN)
        self._notify('structure', None)

    def clear_pv_rois(self, notify: bool = True) -> None:
        for roi in list(self.pv_rois.values()):
            rid = id(roi)
            for reg in (getattr(self.main, '_roi_plot_docks', {}),
                        getattr(self.main, '_roi_2d_docks', {})):
                dock = reg.get(rid)
                if dock is not None:
                    dock.close()
            self.history.pop(rid, None)
            self.roi_names.pop(rid, None)
            self.roi_colors.pop(rid, None)
        self.pv_rois.clear()
        self.pv_roi_ids.clear()
        if notify:
            self._notify('structure', None)

    def get_roi_name(self, roi) -> str:
        try:
            name = self.roi_names.get(id(roi))
            if name:
                return name
            if roi in self.cuts:
                name = f"Cut {self.cuts.index(roi) + 1}"
            else:
                idx = self.rois.index(roi) + 1 if roi in self.rois else len(self.rois) + 1
                # Distinct from the detector's read-only ROI1-4 to avoid confusion.
                name = f"Region {idx}"
            self.roi_names[id(roi)] = name
            return name
        except Exception:
            return "ROI"

    # ----- Colors -----
    def _default_color(self, roi) -> str:
        """Palette color: PV ROIs by their name index (ROI1→ROI_COLORS[0]…),
        software ROIs by their position — matching the existing ROI dock."""
        if id(roi) in self.pv_roi_ids:
            digits = ''.join(c for c in self.roi_names.get(id(roi), '') if c.isdigit())
            idx = (int(digits) - 1) if digits else 0
        elif roi in self.cuts:
            idx = self.cuts.index(roi)
        else:
            idx = self.rois.index(roi) if roi in self.rois else len(self.rois)
        return ROI_COLORS[idx % len(ROI_COLORS)]

    def get_roi_color(self, roi) -> str:
        return self.roi_colors.get(id(roi)) or self._default_color(roi)

    def set_roi_color(self, roi, color: str) -> None:
        self.roi_colors[id(roi)] = color
        try:
            width = settings.LINE_CUT_PEN_WIDTH if self.is_cut(roi) else 2
            roi.setPen(pg.mkPen(color, width=width))
        except Exception:
            pass
        self._notify('recolor', roi)

    def pick_roi_color(self, roi) -> None:
        """Open a color dialog to recolor a user-drawn ROI (workbench-style)."""
        if self.is_readonly(roi):
            return
        chosen = QColorDialog.getColor(QColor(self.get_roi_color(roi)), self.main, "Pick ROI Color")
        if chosen.isValid():
            self.set_roi_color(roi, chosen.name())

    # ----- JSON export -----
    def _roi_to_dict(self, roi) -> dict:
        """Serialize an ROI/cut to a plain dict (name, color, geometry)."""
        entry = {'name': self.get_roi_name(roi), 'color': self.get_roi_color(roi)}
        if self.is_cut(roi):
            entry['type'] = 'cut'
            entry['points'] = self._cut_endpoints(roi)
        else:
            pos, size = roi.pos(), roi.size()
            entry.update({'type': 'rect', 'x': float(pos.x()), 'y': float(pos.y()),
                          'w': float(size.x()), 'h': float(size.y())})
        return entry

    def save_rois_to_json(self, rois, path: str = None) -> None:
        """Save one or more ROIs/line cuts to a JSON file."""
        rois = [r for r in (rois or []) if r is not None]
        if not rois:
            self.main.update_status("No ROIs selected to save", level='warning')
            return
        if path is None:
            default = (f"{self.get_roi_name(rois[0]).replace(' ', '_')}.json"
                       if len(rois) == 1 else "rois.json")
            path, _ = QFileDialog.getSaveFileName(self.main, "Save ROIs to JSON", default,
                                                  "JSON files (*.json)")
            if not path:
                return
        try:
            with open(path, 'w') as f:
                json.dump([self._roi_to_dict(r) for r in rois], f, indent=2)
            self.main.update_status(f"Saved {len(rois)} item(s) to {path}")
        except Exception as e:
            self.main.update_status(f"Error saving ROI JSON: {e}", level='error')

    def save_roi_to_json(self, roi, path: str = None) -> None:
        """Save a single ROI or line cut to a JSON file (the selected one)."""
        self.save_rois_to_json([roi], path)

    def save_cuts_to_json(self, path: str = None) -> None:
        """Save all line cuts to a JSON file."""
        if not self.cuts:
            self.main.update_status("No line cuts to save", level='warning')
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self.main, "Save Line Cuts to JSON",
                                                  "line_cuts.json", "JSON files (*.json)")
            if not path:
                return
        try:
            with open(path, 'w') as f:
                json.dump([self._roi_to_dict(c) for c in self.cuts], f, indent=2)
            self.main.update_status(f"Saved {len(self.cuts)} line cut(s) to {path}")
        except Exception as e:
            self.main.update_status(f"Error saving cuts JSON: {e}", level='error')

    def _roi_from_dict(self, entry: dict):
        """Recreate an ROI/cut from a :meth:`_roi_to_dict` entry, applying its
        saved color. Returns the new item, or None if the entry is malformed."""
        name = entry.get('name')
        color = entry.get('color')
        if entry.get('type') == 'cut':
            item = self.create_and_add_cut(positions=entry.get('points'), name=name)
        else:
            item = self.create_and_add_roi(
                pos=[entry['x'], entry['y']],
                size=[max(1, entry['w']), max(1, entry['h'])], name=name)
        if item is not None and color:
            self.set_roi_color(item, color)
        return item

    def load_from_json(self, path: str) -> int:
        """Load ROIs/line cuts from a JSON preset written by
        :meth:`save_rois_to_json`, replacing the current user-drawn set.
        Returns the number of items loaded."""
        with open(path) as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            raise ValueError("ROI JSON must be a list of entries")
        self.clear_all_rois()
        loaded = 0
        for entry in entries:
            try:
                if self._roi_from_dict(entry) is not None:
                    loaded += 1
            except (KeyError, TypeError, ValueError):
                continue
        return loaded

    # ----- Stats -----
    def get_current_frame_data(self):
        """Return the processed 2D frame currently displayed (matches ROI coords)."""
        img = getattr(self.main, 'image', None)
        if img is None:
            return None
        arr = np.asarray(img)
        return arr if arr.ndim == 2 and arr.size > 0 else None

    def compute_roi_stats(self, frame_data, roi):
        """Stats for ``roi`` over ``frame_data`` via pyqtgraph transform-aware
        extraction (bbox fallback). Returns None if empty/out-of-bounds."""
        try:
            if frame_data is None or roi is None:
                return None
            if self.is_cut(roi):
                return self.compute_cut_stats(frame_data, roi)
            image_item = getattr(self.main.image_view, 'imageItem', None)
            sub = None
            try:
                if image_item is not None:
                    sub = roi.getArrayRegion(frame_data, image_item)
                    if sub is not None and hasattr(sub, 'ndim') and sub.ndim > 2:
                        sub = np.squeeze(sub)
            except Exception:
                sub = None

            x0 = y0 = w = h = None
            try:
                if image_item is not None:
                    slc_info = roi.getArraySlice(frame_data, image_item)
                    slices = slc_info[0] if isinstance(slc_info, (tuple, list)) and slc_info else None
                    if isinstance(slices, (tuple, list)) and len(slices) >= 2:
                        def _bounds(idx, maxdim):
                            if isinstance(idx, slice):
                                return (int(0 if idx.start is None else idx.start),
                                        int(maxdim if idx.stop is None else idx.stop))
                            a = np.asarray(idx)
                            if a.size > 0:
                                return int(np.min(a)), int(np.max(a) + 1)
                            return 0, maxdim
                        y0_, y1_ = _bounds(slices[0], frame_data.shape[0])
                        x0_, x1_ = _bounds(slices[1], frame_data.shape[1])
                        x0, y0 = max(0, x0_), max(0, y0_)
                        w, h = max(0, x1_ - x0_), max(0, y1_ - y0_)
            except Exception:
                pass

            if sub is None or int(getattr(sub, 'size', 0)) == 0 or any(v is None for v in (x0, y0, w, h)):
                height, width = frame_data.shape
                pos, size = roi.pos(), roi.size()
                x0, y0 = max(0, int(pos.x())), max(0, int(pos.y()))
                w, h = max(1, int(size.x())), max(1, int(size.y()))
                x1, y1 = min(width, x0 + w), min(height, y0 + h)
                if x0 >= x1 or y0 >= y1:
                    return None
                sub = frame_data[y0:y1, x0:x1]
                w, h = x1 - x0, y1 - y0

            if sub is None or int(sub.size) == 0:
                return None

            s = float(np.sum(sub))
            total = s if s != 0.0 else 1.0
            comx = float((sub.sum(axis=0) @ np.arange(sub.shape[1])) / total)
            comy = float((sub.sum(axis=1) @ np.arange(sub.shape[0])) / total)
            return {
                'x': int(x0), 'y': int(y0), 'w': int(w), 'h': int(h),
                'sum': s, 'min': float(np.min(sub)), 'max': float(np.max(sub)),
                'mean': float(np.mean(sub)), 'std': float(np.std(sub)),
                'count': int(sub.size), 'comx': comx, 'comy': comy,
            }
        except Exception:
            return None

    def _cut_endpoints(self, cut):
        """The cut's two endpoints [(x1, y1), (x2, y2)] in image-pixel coordinates."""
        image_item = getattr(self.main.image_view, 'imageItem', None)
        coords = []
        for h in cut.handles:
            p = cut.mapToScene(h['item'].pos())
            if image_item is not None:
                p = image_item.mapFromScene(p)
            coords.append((float(p.x()), float(p.y())))
        return coords

    def compute_cut_stats(self, frame_data, cut):
        """Sample the 1D intensity profile along ``cut`` and reduce it to the same
        stats schema as the rectangular ROIs (geometry cols become the segment
        bounding box; CoM is the centroid position along the line). The raw profile
        is cached in ``cut_profiles`` for the Line Cuts dock."""
        try:
            image_item = getattr(self.main.image_view, 'imageItem', None)
            if image_item is None:
                return None
            profile = cut.getArrayRegion(frame_data, image_item)
            if profile is None:
                return None
            profile = np.asarray(profile, dtype=float)
            if profile.ndim > 1:
                profile = np.squeeze(profile)
            profile = profile.ravel()
            if profile.size == 0:
                return None
            self.cut_profiles[id(cut)] = profile

            pts = self._cut_endpoints(cut)
            (x1, y1), (x2, y2) = (pts[0], pts[1]) if len(pts) >= 2 else ((0.0, 0.0), (0.0, 0.0))
            s = float(np.sum(profile))
            total = s if s != 0.0 else 1.0
            frac = float((profile @ np.arange(profile.size)) / total) / max(1, profile.size - 1)
            frac = min(max(frac, 0.0), 1.0)
            return {
                'x': int(min(x1, x2)), 'y': int(min(y1, y2)),
                'w': int(round(abs(x2 - x1))), 'h': int(round(abs(y2 - y1))),
                'sum': s, 'min': float(np.min(profile)), 'max': float(np.max(profile)),
                'mean': float(np.mean(profile)), 'std': float(np.std(profile)),
                'count': int(profile.size),
                'comx': x1 + frac * (x2 - x1), 'comy': y1 + frac * (y2 - y1),
            }
        except Exception:
            return None

    def _on_region_changed(self, roi) -> None:
        """ROI dragged/resized: refresh its stats row and reposition its label."""
        self.show_roi_stats_for_roi(roi)
        self._notify('moved', roi)

    def show_roi_stats_for_roi(self, roi) -> None:
        frame = self.get_current_frame_data()
        stats = self.compute_roi_stats(frame, roi)
        if stats is None:
            return
        if self.stats_dock is not None:
            self.stats_dock.update_row(roi, stats)

    def update_all_roi_stats(self) -> None:
        frame = self.get_current_frame_data()
        if frame is None:
            return
        for r in self.all_rois():
            s = self.compute_roi_stats(frame, r)
            if s and self.stats_dock is not None:
                self.stats_dock.update_row(r, s)

    # ----- Live per-frame hook -----
    def on_new_frame(self) -> None:
        """Called each timer tick; appends one history sample per ROI when a new
        detector frame has arrived, then redraws stats + open plot docks."""
        reader = getattr(self.main, 'reader', None)
        if reader is None or not self.all_rois():
            return
        count = int(getattr(reader, 'frames_received', 0))
        if count == self._last_frame_count:
            return
        self._last_frame_count = count
        frame = self.get_current_frame_data()
        if frame is None:
            return
        for r in self.all_rois():
            s = self.compute_roi_stats(frame, r)
            if not s:
                continue
            self.history.setdefault(id(r), collections.deque(maxlen=settings.ROI_HISTORY_MAXLEN))
            self.history[id(r)].append({'time': count, **s})
            if self.stats_dock is not None:
                self.stats_dock.update_row(r, s)
        self._notify('frame', None)
