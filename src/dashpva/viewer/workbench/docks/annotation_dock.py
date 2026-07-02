"""Annotation Dock — displays all dataset annotations and provides add/edit/delete/save controls."""

from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAction,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dashpva.utils.annotations import Placement
from dashpva.viewer.core.docks.base_dock import BaseDock
from dashpva.viewer.workbench.managers.annotation_manager import Annotation

# Per-type accent colours used in the list and on the canvas overlay
ANN_COLORS = {
    'coordinate': '#00BCD4',  # teal
    'roi':        '#FF9800',  # orange
    'dataset':    '#9E9E9E',  # grey  (list only — no canvas overlay)
}


class _AnnotationDialog(QDialog):
    """Modeless editor used for both creating and editing annotations.

    Stays open between operations so the user can add several annotations
    without reopening.  Emits ``annotation_saved(edit_index, annotation)``
    where ``edit_index == -1`` means a new annotation, ``>= 0`` means an
    in-place update at that list position.
    """

    annotation_saved = pyqtSignal(int, object)   # (index, Annotation)

    def __init__(self, main_window):
        super().__init__(main_window, Qt.Tool)
        self.setWindowTitle("Annotation Editor")
        self.setMinimumWidth(420)
        self._mw = main_window
        self._edit_index: int = -1
        self._picking: bool = False
        self._roi_map: dict = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Title ---
        title_form = QFormLayout()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Short name for this annotation (falls back to first tag)")
        title_form.addRow("Title:", self.title_edit)
        layout.addLayout(title_form)

        # --- Text box ---
        text_box = QGroupBox("Annotation text")
        text_layout = QVBoxLayout(text_box)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "Notes, observations, or leave blank for a tag-only annotation…"
        )
        self.text_edit.setFixedHeight(80)
        text_layout.addWidget(self.text_edit)
        layout.addWidget(text_box)

        # --- Tags ---
        tag_form = QFormLayout()
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("artifact, peak, bad-frame  (comma-separated)")
        tag_form.addRow("Tags:", self.tags_edit)
        layout.addLayout(tag_form)

        # --- Type ---
        type_box = QGroupBox("Type")
        type_layout = QHBoxLayout(type_box)
        self._type_group = QButtonGroup(self)
        self.rb_dataset    = QRadioButton("Dataset")
        self.rb_coordinate = QRadioButton("Coordinate")
        self.rb_roi        = QRadioButton("ROI")
        self._type_group.addButton(self.rb_dataset,    0)
        self._type_group.addButton(self.rb_coordinate, 1)
        self._type_group.addButton(self.rb_roi,        2)
        self.rb_dataset.setChecked(True)
        type_layout.addWidget(self.rb_dataset)
        type_layout.addWidget(self.rb_coordinate)
        type_layout.addWidget(self.rb_roi)
        layout.addWidget(type_box)

        # --- Placements section (coordinate/roi; one point/ROI per frame) ---
        self.placement_box = QGroupBox("Placements  (one per frame)")
        pl_layout = QVBoxLayout(self.placement_box)
        self.placement_list = QListWidget()
        self.placement_list.setMaximumHeight(110)
        pl_layout.addWidget(self.placement_list)
        pl_btn_row = QHBoxLayout()
        self.btn_add_point = QPushButton("Add point (click image)")
        self.roi_combo = QComboBox()
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_remove_placement = QPushButton("Remove selected")
        pl_btn_row.addWidget(self.btn_add_point)
        pl_btn_row.addWidget(self.roi_combo, 1)
        pl_btn_row.addWidget(self.btn_add_roi)
        pl_btn_row.addWidget(self.btn_remove_placement)
        pl_layout.addLayout(pl_btn_row)
        self.placement_box.hide()
        layout.addWidget(self.placement_box)

        # --- Frame list (dataset notes only) ---
        self.frame_box = QGroupBox("Frames  (empty = current frame)")
        frame_layout = QVBoxLayout(self.frame_box)
        self.frame_list = QListWidget()
        self.frame_list.setMaximumHeight(100)
        frame_layout.addWidget(self.frame_list)
        frame_btn_row = QHBoxLayout()
        self.btn_add_frame    = QPushButton("Add current frame")
        self.btn_remove_frame = QPushButton("Remove selected")
        frame_btn_row.addWidget(self.btn_add_frame)
        frame_btn_row.addWidget(self.btn_remove_frame)
        frame_btn_row.addStretch()
        frame_layout.addLayout(frame_btn_row)
        layout.addWidget(self.frame_box)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        self.btn_save  = QPushButton("Add Annotation")
        self.btn_clear = QPushButton("Clear")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Connections
        self._type_group.buttonClicked.connect(self._on_type_changed)
        self.btn_add_point.clicked.connect(self._on_add_point)
        self.btn_add_roi.clicked.connect(self._on_add_roi)
        self.btn_remove_placement.clicked.connect(self._on_remove_placement)
        self.btn_add_frame.clicked.connect(self._on_add_frame)
        self.btn_remove_frame.clicked.connect(self._on_remove_frame)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_clear.clicked.connect(self._reset_to_add_mode)

    # ------------------------------------------------------------------
    # Public helpers called from the dock
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Switch to add-new mode, reset fields, refresh ROI combo."""
        self._edit_index = -1
        self.btn_save.setText("Add Annotation")
        self._reset_fields()
        self._refresh_roi_combo()
        if self._picking:
            self._cancel_pick()

    def open_for_edit(self, index: int, ann: Annotation) -> None:
        """Switch to edit mode pre-filled with an existing annotation."""
        self._edit_index = index
        self.btn_save.setText("Update Annotation")
        self._refresh_roi_combo()
        self._fill_from(ann)
        if self._picking:
            self._cancel_pick()

    def set_frame(self, frame_idx: int) -> None:
        """Switch to add-new dataset-type mode with a single frame preselected."""
        self.prepare()
        self.rb_dataset.setChecked(True)
        self._on_type_changed(None)
        item = QListWidgetItem(f"Frame {frame_idx}")
        item.setData(Qt.UserRole, int(frame_idx))
        self.frame_list.addItem(item)

    def set_roi(self, roi) -> None:
        """Switch to ROI type and seed a placement for the clicked ROI — from right-click on ROI."""
        self.prepare()
        self.rb_roi.setChecked(True)
        self._on_type_changed(None)
        rm = getattr(self._mw, 'roi_manager', None)
        if rm is not None:
            name = rm.get_roi_name(roi)
            idx = self.roi_combo.findText(name)
            if idx >= 0:
                self.roi_combo.setCurrentIndex(idx)
                self._on_add_roi()

    # ------------------------------------------------------------------
    # Fill from existing annotation
    # ------------------------------------------------------------------

    def _fill_from(self, ann: Annotation) -> None:
        self.title_edit.setText(ann.title)
        self.text_edit.setPlainText(ann.text)
        self.tags_edit.setText(", ".join(ann.tags))

        self.frame_list.clear()
        self.placement_list.clear()

        if ann.ann_type == 'coordinate':
            self.rb_coordinate.setChecked(True)
        elif ann.ann_type == 'roi':
            self.rb_roi.setChecked(True)
        else:
            self.rb_dataset.setChecked(True)
            for f in ann.frames:
                item = QListWidgetItem(f"Frame {f}")
                item.setData(Qt.UserRole, f)
                self.frame_list.addItem(item)

        for p in ann.placements:
            self._add_placement_item(p)

        self._on_type_changed(None)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_type_changed(self, _button) -> None:
        tid = self._type_group.checkedId()
        is_coord, is_roi, is_dataset = tid == 1, tid == 2, tid == 0
        self.placement_box.setVisible(is_coord or is_roi)
        self.btn_add_point.setVisible(is_coord)
        self.roi_combo.setVisible(is_roi)
        self.btn_add_roi.setVisible(is_roi)
        self.frame_box.setVisible(is_dataset)
        if not is_coord and self._picking:
            self._cancel_pick()
        self.adjustSize()

    # --- Placements ---------------------------------------------------

    def _add_placement_item(self, p: Placement) -> None:
        if p.w is not None:
            text = f"frame {p.frame}:  {p.roi_name or 'ROI'}  ({p.x},{p.y}  {p.w}×{p.h})"
        else:
            text = f"frame {p.frame}:  ({p.x}, {p.y})"
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, p)
        self.placement_list.addItem(item)

    def _placements(self) -> list:
        return [self.placement_list.item(i).data(Qt.UserRole) for i in range(self.placement_list.count())]

    def _on_add_point(self) -> None:
        if self._picking:
            self._cancel_pick()
            return
        scene = self._get_scene()
        if scene is None:
            return
        self._picking = True
        self.btn_add_point.setText("Cancel  (click image now…)")
        try:
            scene.sigMouseClicked.connect(self._on_scene_clicked)
        except Exception:
            self._picking = False
            self.btn_add_point.setText("Add point (click image)")

    def _on_scene_clicked(self, ev) -> None:
        try:
            if ev.button() != Qt.LeftButton:
                return
            vb = self._get_vb()
            if vb is None:
                return
            pt = vb.mapSceneToView(ev.scenePos())
            self._add_placement_item(Placement(
                frame=self._current_frame(),
                x=int(round(float(pt.x()))),
                y=int(round(float(pt.y()))),
            ))
        except Exception:
            pass
        finally:
            self._cancel_pick()
        self.raise_()
        self.activateWindow()

    def _cancel_pick(self) -> None:
        self._picking = False
        self.btn_add_point.setText("Add point (click image)")
        scene = self._get_scene()
        if scene is not None:
            try:
                scene.sigMouseClicked.disconnect(self._on_scene_clicked)
            except Exception:
                pass

    def _on_add_roi(self) -> None:
        name = self.roi_combo.currentText()
        roi_obj = self._roi_map.get(name)
        if roi_obj is None:
            return
        try:
            pos, size = roi_obj.pos(), roi_obj.size()
            self._add_placement_item(Placement(
                frame=self._current_frame(),
                x=max(0, int(pos.x())), y=max(0, int(pos.y())),
                w=max(1, int(size.x())), h=max(1, int(size.y())), roi_name=name,
            ))
        except Exception:
            pass

    def _on_remove_placement(self) -> None:
        row = self.placement_list.currentRow()
        if row >= 0:
            self.placement_list.takeItem(row)

    def _on_add_frame(self) -> None:
        frame_idx = 0
        try:
            if hasattr(self._mw, 'frame_spinbox') and self._mw.frame_spinbox.isEnabled():
                frame_idx = int(self._mw.frame_spinbox.value())
        except Exception:
            pass
        for i in range(self.frame_list.count()):
            if self.frame_list.item(i).data(Qt.UserRole) == frame_idx:
                return
        item = QListWidgetItem(f"Frame {frame_idx}")
        item.setData(Qt.UserRole, frame_idx)
        self.frame_list.addItem(item)

    def _on_remove_frame(self) -> None:
        row = self.frame_list.currentRow()
        if row >= 0:
            self.frame_list.takeItem(row)

    def _on_save(self) -> None:
        ann = self._build_annotation()
        self.annotation_saved.emit(self._edit_index, ann)
        self._reset_to_add_mode()

    def _reset_to_add_mode(self) -> None:
        self._edit_index = -1
        self.btn_save.setText("Add Annotation")
        self._reset_fields()

    def _reset_fields(self) -> None:
        self.title_edit.clear()
        self.text_edit.clear()
        self.tags_edit.clear()
        self.frame_list.clear()
        self.placement_list.clear()

    # ------------------------------------------------------------------
    # Build annotation from current widget state
    # ------------------------------------------------------------------

    def _current_frame(self) -> int:
        try:
            if hasattr(self._mw, 'frame_spinbox') and self._mw.frame_spinbox.isEnabled():
                return int(self._mw.frame_spinbox.value())
        except Exception:
            pass
        return 0

    def _build_annotation(self) -> Annotation:
        title = self.title_edit.text().strip()
        text = self.text_edit.toPlainText().strip()
        tags = [t.strip().lstrip("#") for t in self.tags_edit.text().split(",") if t.strip()]
        tid = self._type_group.checkedId()
        if tid == 1:
            return Annotation(text=text, tags=tags, title=title,
                              ann_type="coordinate", placements=self._placements())
        if tid == 2:
            return Annotation(text=text, tags=tags, title=title,
                              ann_type="roi", placements=self._placements())
        # dataset note
        frames = sorted(
            self.frame_list.item(i).data(Qt.UserRole)
            for i in range(self.frame_list.count())
        )
        if not frames:
            frames = [self._current_frame()]
        return Annotation(text=text, tags=tags, title=title, ann_type="dataset", frames=frames)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_roi_combo(self) -> None:
        self.roi_combo.clear()
        self._roi_map = {}
        rm = getattr(self._mw, 'roi_manager', None)
        if rm is None:
            return
        for roi in getattr(rm, 'rois', []):
            name = rm.get_roi_name(roi)
            self.roi_combo.addItem(name)
            self._roi_map[name] = roi

    def _get_scene(self):
        try:
            vb = getattr(self._mw.plot_item, 'vb', None)
            return vb.scene() if vb is not None else None
        except Exception:
            return None

    def _get_vb(self):
        try:
            return getattr(self._mw.plot_item, 'vb', None)
        except Exception:
            return None

    def closeEvent(self, event) -> None:
        if self._picking:
            self._cancel_pick()
        super().closeEvent(event)


class AnnotationDock(BaseDock):
    """Dock showing all annotations for the loaded dataset.

    Usage::

        dock = AnnotationDock(main_window=self, dock_area=Qt.RightDockWidgetArea)
        dock.refresh()
    """

    def __init__(self, main_window=None, dock_area=Qt.RightDockWidgetArea, segment_name="other", show=True):
        super().__init__(
            title="Annotations",
            main_window=main_window,
            segment_name=segment_name,
            dock_area=dock_area,
            show=show,
        )
        self._dialog: Optional[_AnnotationDialog] = None
        self._overlay_by_frame: dict = {}   # frame -> [pg items]; built on data change, toggled on scrub
        self._overlays_visible: bool = True
        self._populating: bool = False   # guards itemChanged during refresh()
        # Debounce HDF5 autosave so rapid edits/toggles coalesce into one write
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._do_save)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.annotation_list = QTreeWidget()
        self.annotation_list.setColumnCount(4)
        self.annotation_list.setHeaderLabels(["Frame(s)", "Title", "Tags", "Usage"])
        self.annotation_list.setRootIsDecorated(False)
        self.annotation_list.setAlternatingRowColors(True)
        self.annotation_list.setContextMenuPolicy(Qt.CustomContextMenu)
        header = self.annotation_list.header()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.annotation_list)

        btn_row = QHBoxLayout()
        self.btn_add    = QPushButton("Add Annotation")
        self.btn_delete = QPushButton("Delete")
        self.btn_toggle = QPushButton("Hide all overlays")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_toggle)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setWidget(container)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_toggle.clicked.connect(self._on_toggle_overlays)
        self.annotation_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.annotation_list.itemChanged.connect(self._on_item_check_changed)
        self.annotation_list.customContextMenuRequested.connect(self._show_context_menu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate the list and redraw canvas overlays."""
        mgr = self._manager()
        if mgr is None:
            return
        anns = mgr.all_annotations()
        self._populating = True
        try:
            self.annotation_list.clear()
            for ann in anns:
                tags_cell = str(len(ann.tags))
                usage_cell = str(ann.usage_count())
                item = QTreeWidgetItem([ann.frame_label(), ann.display_title(), tags_cell, usage_cell])
                color = QColor(ANN_COLORS.get(ann.ann_type, '#9E9E9E'))
                for col in range(4):
                    item.setForeground(col, color)
                # Column-0 checkbox doubles as the per-annotation show/hide toggle
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if ann.visible else Qt.Unchecked)
                item.setToolTip(1, "Double-click to jump to this frame/position; checkbox shows/hides its overlay")
                self.annotation_list.addTopLevelItem(item)
        finally:
            self._populating = False
        self._build_overlays()
        self.update_frame_label()

    def open_for_roi(self, roi) -> None:
        """Open the editor pre-set to the given in-memory ROI."""
        dlg = self._ensure_dialog()
        dlg.set_roi(roi)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def open_for_frame(self, frame_idx: int) -> None:
        """Open the editor to add a note for a single frame."""
        dlg = self._ensure_dialog()
        dlg.set_frame(frame_idx)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------
    # Slots — list interactions
    # ------------------------------------------------------------------

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int = 0) -> None:
        self._navigate_to(self.annotation_list.indexOfTopLevelItem(item))

    def _navigate_to(self, row: int) -> None:
        """Jump the 2D view to the annotation's frame and pan to its position."""
        mgr = self._manager()
        if mgr is None:
            return
        anns = mgr.all_annotations()
        if row < 0 or row >= len(anns):
            return
        ann = anns[row]
        mw = self.main_window
        placement = ann.placements[0] if ann.placements else None
        target_frame = placement.frame if placement is not None else (ann.frames[0] if ann.frames else None)
        # Jump to the target frame
        try:
            if target_frame is not None and hasattr(mw, 'frame_spinbox') and mw.frame_spinbox.isEnabled():
                lo, hi = mw.frame_spinbox.minimum(), mw.frame_spinbox.maximum()
                mw.frame_spinbox.setValue(max(lo, min(hi, int(target_frame))))
        except Exception:
            pass
        # Pan (keep zoom) to the placement's position
        try:
            if placement is not None:
                cx = placement.x + (placement.w / 2.0 if placement.w else 0)
                cy = placement.y + (placement.h / 2.0 if placement.h else 0)
                vb = mw.plot_item.getViewBox()
                (x0, x1), (y0, y1) = vb.viewRange()
                half_w, half_h = (x1 - x0) / 2.0, (y1 - y0) / 2.0
                vb.setRange(xRange=(cx - half_w, cx + half_w),
                            yRange=(cy - half_h, cy + half_h), padding=0)
        except Exception:
            pass

    def on_frame_changed(self) -> None:
        """Show the current frame's overlays and note label — called by the 2D view on scrub/load.

        Only toggles cached item visibility; no pyqtgraph items are created/destroyed.
        """
        self._apply_overlay_visibility()
        self.update_frame_label()

    def update_frame_label(self, frame_idx: int = None) -> None:
        """Show the note(s) attached to the current (or given) frame in the 2D view's inline label."""
        mw = self.main_window
        label = getattr(mw, 'annotation_note_label', None)
        if label is None:
            return
        if frame_idx is None:
            frame_idx = self._current_frame()
        mgr = self._manager()
        if mgr is None:
            label.setText("No annotation for this frame")
            return
        try:
            texts = [ann.text.strip() for ann in mgr.notes_for_frame(int(frame_idx)) if ann.text.strip()]
            label.setText("  •  ".join(texts) if texts else "No annotation for this frame")
        except Exception:
            label.setText("No annotation for this frame")

    def _on_item_check_changed(self, item: QTreeWidgetItem, column: int = 0) -> None:
        """Show/hide a single annotation's overlay from its checkbox, and persist."""
        if self._populating or column != 0:
            return
        mgr = self._manager()
        if mgr is None:
            return
        row = self.annotation_list.indexOfTopLevelItem(item)
        anns = mgr.all_annotations()
        if row < 0 or row >= len(anns):
            return
        ann = anns[row]
        ann.visible = item.checkState(0) == Qt.Checked
        mgr.update(row, ann)
        self._build_overlays()
        self._autosave()

    def _show_context_menu(self, pos) -> None:
        row = self._current_row()
        if row < 0:
            return
        menu = QMenu(self)
        action_edit   = QAction("Edit",   menu)
        action_delete = QAction("Delete", menu)
        action_edit.triggered.connect(lambda: self._open_edit(row))
        action_delete.triggered.connect(self._on_delete)
        menu.addAction(action_edit)
        menu.addAction(action_delete)
        menu.exec_(self.annotation_list.viewport().mapToGlobal(pos))

    def _current_row(self) -> int:
        return self.annotation_list.indexOfTopLevelItem(self.annotation_list.currentItem())

    def _open_edit(self, row: int) -> None:
        mgr = self._manager()
        if mgr is None:
            return
        anns = mgr.all_annotations()
        if row < 0 or row >= len(anns):
            return
        dlg = self._ensure_dialog()
        dlg.open_for_edit(row, anns[row])
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------
    # Slots — buttons
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        dlg = self._ensure_dialog()
        dlg.prepare()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_annotation_saved(self, index: int, ann: Annotation) -> None:
        mgr = self._manager()
        if mgr is None:
            return
        if index < 0:
            mgr.add(ann)
        else:
            mgr.update(index, ann)
        self.refresh()
        self._sync_workbench_label()
        self._autosave()

    def _on_delete(self) -> None:
        mgr = self._manager()
        if mgr is None:
            return
        row = self._current_row()
        if row < 0:
            return
        mgr.remove(row)
        self.refresh()
        self._sync_workbench_label()
        self._autosave()

    def _autosave(self) -> None:
        """Request a debounced HDF5 save; rapid edits/toggles coalesce into one write."""
        self._save_timer.start()

    def _do_save(self) -> None:
        """Persist annotations to the loaded HDF5 file (fired by the debounce timer)."""
        mw = self.main_window
        mgr = self._manager()
        if mgr is None:
            return
        file_path = getattr(mw, 'current_file_path', None)
        if not file_path:
            return
        mgr.save_to_hdf5(file_path)

    # ------------------------------------------------------------------
    # Canvas overlays
    # ------------------------------------------------------------------

    def _on_toggle_overlays(self) -> None:
        self._overlays_visible = not self._overlays_visible
        self.btn_toggle.setText("Show all overlays" if not self._overlays_visible else "Hide all overlays")
        if self._overlays_visible:
            self._build_overlays()
        else:
            self._clear_overlays()

    def _make_placement_items(self, ann_type: str, p, color, tip) -> list:
        """pyqtgraph items for a single placement (marker/box + label)."""
        if ann_type == 'coordinate':
            scatter = pg.ScatterPlotItem(
                x=[p.x], y=[p.y], symbol='t', size=14,
                pen=pg.mkPen(color=color, width=2), brush=pg.mkBrush(color),
            )
            label = pg.TextItem(text=tip, color=color, anchor=(0, 1))
            label.setPos(p.x + 3, p.y + 3)
            return [scatter, label]
        if ann_type == 'roi' and p.w is not None:
            xs = [p.x, p.x + p.w, p.x + p.w, p.x,        p.x]
            ys = [p.y, p.y,        p.y + p.h, p.y + p.h, p.y]
            rect = pg.PlotCurveItem(
                x=xs, y=ys, pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
            )
            label = pg.TextItem(text=tip, color=color, anchor=(0, 1))
            label.setPos(p.x, p.y)
            return [rect, label]
        return []

    def _clear_overlays(self) -> None:
        """Remove all cached overlay items from the canvas."""
        plot_item = getattr(self.main_window, 'plot_item', None)
        if plot_item is not None:
            for items in self._overlay_by_frame.values():
                for it in items:
                    try:
                        plot_item.removeItem(it)
                    except Exception:
                        pass
        self._overlay_by_frame = {}

    def _build_overlays(self) -> None:
        """Create overlay items for every placement once, bucketed by frame.

        Called only when annotation data changes. Frame scrubbing toggles the
        cached items' visibility via :meth:`_apply_overlay_visibility` — no items
        are created or destroyed per frame. Skipped entirely while overlays are hidden.
        """
        self._clear_overlays()
        if not self._overlays_visible:
            return
        plot_item = getattr(self.main_window, 'plot_item', None)
        mgr = self._manager()
        if plot_item is None or mgr is None:
            return
        cur_frame = self._current_frame()
        for ann in mgr.all_annotations():
            if ann.ann_type == 'dataset' or not ann.visible:
                continue
            color = ANN_COLORS[ann.ann_type]
            tip = ann.display_title()
            for p in ann.placements:
                items = self._make_placement_items(ann.ann_type, p, color, tip)
                for it in items:
                    plot_item.addItem(it)
                    it.setVisible(p.frame == cur_frame)
                    self._overlay_by_frame.setdefault(p.frame, []).append(it)

    def _apply_overlay_visibility(self) -> None:
        """Show only the current frame's cached overlays (fast path for scrubbing)."""
        if not self._overlays_visible:
            return
        cur_frame = self._current_frame()
        for frame, items in self._overlay_by_frame.items():
            vis = frame == cur_frame
            for it in items:
                try:
                    it.setVisible(vis)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dialog(self) -> _AnnotationDialog:
        if self._dialog is None:
            self._dialog = _AnnotationDialog(self.main_window)
            self._dialog.annotation_saved.connect(self._on_annotation_saved)
        return self._dialog

    def _sync_workbench_label(self) -> None:
        """Refresh the 2D view's per-frame note label after an add/edit/delete."""
        try:
            self.update_frame_label()
        except Exception:
            pass

    def _current_frame(self) -> int:
        mw = self.main_window
        try:
            if hasattr(mw, 'frame_spinbox') and mw.frame_spinbox.isEnabled():
                return int(mw.frame_spinbox.value())
        except Exception:
            pass
        return 0

    def _manager(self):
        return getattr(self.main_window, 'annotation_manager', None)
