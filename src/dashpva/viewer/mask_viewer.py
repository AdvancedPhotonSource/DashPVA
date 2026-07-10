import os

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


class MaskViewerWindow(QDialog):
    """
    Displays and optionally edits a boolean detector mask.

    self.mask starts in detector-native orientation (matching raw frames,
    EDF masks, and PONI geometry). The Transpose/Rotate buttons modify
    self.mask directly — this is intentional so users can correct a mask
    that was loaded in the wrong orientation.

    Display transforms (_is_transposed, _rot_num) are set once from the
    parent viewer on open so the mask visually matches the diffraction
    pattern. These flags are NOT changed by the Transpose/Rotate buttons;
    they only affect rendering via _get_display_mask().
    """

    mask_updated = pyqtSignal(object)

    def __init__(self, mask, mask_path=None, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        # ALWAYS detector-native orientation
        self.mask = mask.copy().astype(bool)
        self.mask_path = mask_path
        self._editing = False
        self._show_image = False
        self._alpha = 0.5
        # Line tool: first click stores the start point (native coords) until the
        # second click completes the segment.
        self._line_start = None
        # Drag state: snapshot of the mask at drag-start (for rect/line preview),
        # the drag origin, and the last brush point (for gap-free strokes).
        self._stroke_snapshot = None
        self._drag_start = None
        self._last_brush = None

        # Display-only orientation — initialized from parent viewer
        # so the mask appears the same way as the diffraction pattern
        self._is_transposed = getattr(parent, 'image_is_transposed', False) if parent else False
        self._rot_num = getattr(parent, 'rot_num', 0) if parent else 0

        num_masked = int(np.sum(self.mask))
        h, w = self.mask.shape
        path_display = mask_path or 'Unsaved mask'
        self.setWindowTitle(f"Mask -- {path_display} ({w}x{h}, {num_masked} masked)")
        self.resize(800, 700)

        self._build_ui()
        self._refresh_display()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Image view for mask display
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'X [pixels]')
        self.plot_item.setLabel('left', 'Y [pixels]')
        self.image_view = pg.ImageView(view=self.plot_item)
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, stretch=3)

        # Mask drawn as a cheap red overlay on top of the base image. Updating
        # only this layer (a 2-entry LUT applied to the bool mask) keeps live
        # drawing/dragging fast — the base image is not recomputed per edit.
        self.mask_overlay = pg.ImageItem()
        self.mask_overlay.setZValue(10)
        lut = np.zeros((2, 4), dtype=np.ubyte)
        lut[1] = (255, 50, 50, 255)  # masked pixels → red, unmasked → transparent
        self.mask_overlay.setLookupTable(lut)
        self.mask_overlay.setLevels((0, 1))
        self.plot_item.addItem(self.mask_overlay)

        # Take over left-button drags for painting; anything else (pan/zoom, or
        # not in edit mode) falls through to the viewbox's normal handler.
        self._orig_mouse_drag = self.plot_item.vb.mouseDragEvent
        self.plot_item.vb.mouseDragEvent = self._mouse_drag

        # Info label
        self.lbl_info = QLabel(self._info_text())
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # Controls row 1: overlay + alpha
        overlay_row = QHBoxLayout()

        self.chk_show_image = QCheckBox('Show Diffraction Image')
        self.chk_show_image.stateChanged.connect(self._toggle_image_overlay)
        overlay_row.addWidget(self.chk_show_image)

        overlay_row.addWidget(QLabel('Mask opacity:'))
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(50)
        self.sld_alpha.setMaximumWidth(120)
        self.sld_alpha.valueChanged.connect(self._alpha_changed)
        overlay_row.addWidget(self.sld_alpha)
        self.lbl_alpha = QLabel('50%')
        self.lbl_alpha.setMinimumWidth(35)
        overlay_row.addWidget(self.lbl_alpha)

        overlay_row.addStretch()
        layout.addLayout(overlay_row)

        # Controls row 2: actions
        ctrl = QHBoxLayout()

        self.btn_save = QPushButton('Save Mask')
        self.btn_save.clicked.connect(self._save_mask)
        ctrl.addWidget(self.btn_save)

        self.btn_invert = QPushButton('Invert Mask')
        self.btn_invert.clicked.connect(self._invert_mask)
        ctrl.addWidget(self.btn_invert)

        self.btn_export_json = QPushButton('Export JSON')
        self.btn_export_json.setToolTip('Export as EPICS NDPluginBadPixel JSON')
        self.btn_export_json.clicked.connect(self._export_json)
        ctrl.addWidget(self.btn_export_json)

        self.btn_transpose = QPushButton('Transpose')
        self.btn_transpose.clicked.connect(self._toggle_transpose)
        ctrl.addWidget(self.btn_transpose)

        self.btn_rotate = QPushButton('Rotate 90')
        self.btn_rotate.clicked.connect(self._rotate)
        ctrl.addWidget(self.btn_rotate)

        self.chk_edit = QCheckBox('Edit Mode')
        self.chk_edit.stateChanged.connect(self._toggle_edit)
        ctrl.addWidget(self.chk_edit)

        ctrl.addWidget(QLabel('Tool:'))
        self.cmb_tool = QComboBox()
        self.cmb_tool.addItems(['Brush', 'Rectangle', 'Line'])
        self.cmb_tool.currentTextChanged.connect(self._tool_changed)
        ctrl.addWidget(self.cmb_tool)

        ctrl.addWidget(QLabel('Size:'))
        self.spn_brush = QSpinBox()
        self.spn_brush.setToolTip('Brush radius / square side, in pixels')
        self.spn_brush.setRange(1, 500)
        self.spn_brush.setValue(3)
        self.spn_brush.setSuffix(' px')
        ctrl.addWidget(self.spn_brush)

        ctrl.addWidget(QLabel('Thickness:'))
        self.spn_thickness = QSpinBox()
        self.spn_thickness.setToolTip('Line thickness, in pixels')
        self.spn_thickness.setRange(1, 200)
        self.spn_thickness.setValue(3)
        self.spn_thickness.setSuffix(' px')
        ctrl.addWidget(self.spn_thickness)

        self.chk_erase = QCheckBox('Erase')
        self.chk_erase.setToolTip('Draw removes pixels from the mask instead of adding them')
        ctrl.addWidget(self.chk_erase)

        ctrl.addStretch()
        layout.addLayout(ctrl)

    def _info_text(self):
        num_masked = int(np.sum(self.mask))
        total = self.mask.size
        pct = 100 * num_masked / total if total > 0 else 0
        return f"Masked: {num_masked:,} / {total:,} pixels ({pct:.1f}%)"

    # ------------------------------------------------------------------
    # Display transform helpers
    # ------------------------------------------------------------------

    def _get_display_mask(self):
        """Apply display-only transforms to a COPY of the native mask."""
        display = self.mask.copy()
        if self._is_transposed:
            display = display.T
        if self._rot_num:
            display = np.rot90(display, k=self._rot_num)
        return display

    def _transform_data_for_display(self, data):
        """Apply the same display transforms to any 2D data (e.g. diffraction image)."""
        if self._is_transposed:
            if data.ndim == 2:
                data = np.transpose(data)
            else:
                data = np.transpose(data, axes=(1, 0, 2))
        if self._rot_num:
            data = np.rot90(data, k=self._rot_num)
        return data

    def _display_to_native(self, disp_i, disp_j):
        """Reverse-map display coordinates back to detector-native indices.

        Uses a probe array to exactly invert the forward transform chain
        (transpose then rot90) so edit-mode clicks modify the correct pixel.
        """
        display_mask = self._get_display_mask()
        probe = np.zeros(display_mask.shape, dtype=bool)
        probe[disp_i, disp_j] = True

        # Undo rot90(k) = apply rot90(4-k)
        if self._rot_num:
            probe = np.rot90(probe, k=(4 - self._rot_num))
        # Undo transpose
        if self._is_transposed:
            probe = probe.T

        idx = np.argwhere(probe)
        if len(idx) == 0:
            return -1, -1
        return int(idx[0, 0]), int(idx[0, 1])

    def _get_current_image(self):
        """Get current diffraction image from parent viewer if available."""
        if self.parent_viewer is not None and hasattr(self.parent_viewer, 'reader'):
            reader = self.parent_viewer.reader
            if reader is not None and reader.image is not None:
                return reader.image.copy()
        return None

    # ------------------------------------------------------------------
    # Orientation controls — modify self.mask data directly.
    # Display flags (_is_transposed, _rot_num) stay fixed from parent
    # so the auto-display-matching on open is preserved.
    # ------------------------------------------------------------------

    def _toggle_transpose(self):
        self.mask = self.mask.T.copy()
        self._refresh_display()
        self.mask_updated.emit(self.mask)

    def _rotate(self):
        self.mask = np.rot90(self.mask, k=1).copy()
        self._refresh_display()
        self.mask_updated.emit(self.mask)

    # ------------------------------------------------------------------
    # Display rendering
    # ------------------------------------------------------------------

    def _get_parent_display_settings(self):
        """Get display settings (log, levels, colormap) from parent viewer."""
        log_on = False
        levels = None
        colormap = None
        if self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'log_image'):
                log_on = self.parent_viewer.log_image.isChecked()
            if hasattr(self.parent_viewer, 'image_view'):
                try:
                    levels = self.parent_viewer.image_view.getLevels()
                except Exception:
                    pass
            if hasattr(self.parent_viewer, 'cet_colormap'):
                colormap = self.parent_viewer.cet_colormap
        return log_on, levels, colormap

    def _refresh_display(self):
        """Full redraw: grayscale base layer (diffraction image or black) plus
        the red mask overlay. Use _refresh_overlay for cheap mask-only updates
        during interactive drawing."""
        display_mask = self._get_display_mask()
        base = None
        if self._show_image:
            img = self._get_current_image()
            if img is not None:
                log_on, parent_levels, _ = self._get_parent_display_settings()
                img = self._transform_data_for_display(img)
                img_float = img.astype(np.float64)
                if log_on:
                    img_float = np.maximum(img_float, 0)
                    img_float = np.log10(img_float + 1)
                if parent_levels is not None:
                    img_min, img_max = parent_levels
                else:
                    img_min, img_max = img_float.min(), img_float.max()
                rng = img_max - img_min
                base = (np.clip((img_float - img_min) / rng, 0, 1)
                        if rng > 0 else np.zeros_like(img_float))

        if base is None:
            base = np.zeros(display_mask.shape, dtype=np.float32)

        self.image_view.setColorMap(pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]))
        self.image_view.setImage(base.astype(np.float32), autoRange=False,
                                 autoLevels=False, levels=(0, 1))
        # Scale the overlay to the base extent so masks of a different size still
        # register correctly (normally they match the detector frame exactly).
        self.mask_overlay.setRect(QRectF(0, 0, base.shape[0], base.shape[1]))
        self._refresh_overlay()

    def _refresh_overlay(self):
        """Cheap update of just the red mask layer — no base-image recompute."""
        display_mask = self._get_display_mask()
        self.mask_overlay.setImage(display_mask.astype(np.float32),
                                   autoLevels=False, levels=(0, 1))
        self.mask_overlay.setOpacity(self._alpha if self._show_image else 1.0)
        self.lbl_info.setText(self._info_text())

    def _toggle_image_overlay(self, state):
        self._show_image = (state == Qt.Checked)
        self._refresh_display()

    def _alpha_changed(self, value):
        self._alpha = value / 100.0
        self.lbl_alpha.setText(f'{value}%')
        self.mask_overlay.setOpacity(self._alpha if self._show_image else 1.0)

    # ------------------------------------------------------------------
    # Mask operations (always in detector-native orientation)
    # ------------------------------------------------------------------

    def _save_mask(self):
        # If no path set, get default from parent's mask_manager
        if not self.mask_path:
            if self.parent_viewer and hasattr(self.parent_viewer, 'mask_manager'):
                mm = self.parent_viewer.mask_manager
                self.mask_path = os.path.join(mm.masks_dir, mm.DEFAULT_MASK_FILENAME)
            else:
                self.lbl_info.setText("Error: No save path available")
                return

        # self.mask is always detector-native — safe to save directly
        np.save(self.mask_path, self.mask)
        self.mask_updated.emit(self.mask)
        num_masked = int(np.sum(self.mask))
        h, w = self.mask.shape
        self.setWindowTitle(f"Mask -- {self.mask_path} ({w}x{h}, {num_masked} masked)")
        self.lbl_info.setText(f"Saved! {self._info_text()}")

    def _invert_mask(self):
        self.mask = ~self.mask
        self._refresh_display()
        self.mask_updated.emit(self.mask)

    def _export_json(self):
        default_dir = ''
        if self.parent_viewer and hasattr(self.parent_viewer, 'mask_manager'):
            default_dir = self.parent_viewer.mask_manager.masks_dir
        default_path = os.path.join(default_dir, 'bad_pixels.json')
        filepath, _ = QFileDialog.getSaveFileName(
            self, 'Export JSON BadPixel File', default_path,
            'JSON files (*.json);;All files (*)')
        if not filepath:
            return
        try:
            import json
            bad_pixels = []
            rows, cols = np.where(self.mask)
            for row, col in zip(rows, cols):
                bad_pixels.append({"Pixel": [int(col), int(row)], "Set": 0})
            mask_rows, mask_cols = self.mask.shape
            data = {
                "Detector size": [int(mask_cols), int(mask_rows)],
                "Bad pixels": bad_pixels
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            num = len(bad_pixels)
            self.lbl_info.setText(f"Exported {num} bad pixels to JSON")
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to export JSON:\n{e}')

    # ------------------------------------------------------------------
    # Edit mode — clicks in display coords, edits in native coords
    # ------------------------------------------------------------------

    def _toggle_edit(self, state):
        self._editing = (state == Qt.Checked)
        self._line_start = None
        if self._editing:
            self.image_view.getView().scene().sigMouseClicked.connect(self._on_click)
        else:
            try:
                self.image_view.getView().scene().sigMouseClicked.disconnect(self._on_click)
            except TypeError:
                pass

    def _tool_changed(self, *_):
        """Reset any in-progress line when the active tool changes."""
        self._line_start = None
        self.lbl_info.setText(self._info_text())

    def _event_native(self, scene_pos):
        """Map a scene position to a detector-native (row, col), or None if the
        point is outside the mask."""
        mouse_point = self.plot_item.vb.mapSceneToView(scene_pos)
        # pyqtgraph pixel (i,j) occupies [i, i+1) x [j, j+1) — use floor
        vx = int(np.floor(mouse_point.x()))
        vy = int(np.floor(mouse_point.y()))
        display_mask = self._get_display_mask()
        if not (0 <= vx < display_mask.shape[0] and 0 <= vy < display_mask.shape[1]):
            return None
        row, col = self._display_to_native(vx, vy)
        if not (0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]):
            return None
        return row, col

    def _on_click(self, event):
        """Single-click editing: stamp a dot/square, or set a line endpoint."""
        if not self._editing:
            return
        pt = self._event_native(event.scenePos())
        if pt is None:
            return
        row, col = pt
        value = not self.chk_erase.isChecked()
        tool = self.cmb_tool.currentText()
        if tool == 'Rectangle':
            self._paint_square(row, col, self.spn_brush.value(), value)
        elif tool == 'Line':
            if self._line_start is None:
                self._line_start = (row, col)
                self.lbl_info.setText('Line: click the end point… (or drag)')
                return
            self._paint_line(self._line_start, (row, col), self.spn_thickness.value(), value)
            self._line_start = None
        else:  # Brush
            self._paint_disk(row, col, self.spn_brush.value(), value)
        self._refresh_overlay()
        self.mask_updated.emit(self.mask)

    def _mouse_drag(self, ev, axis=None):
        """Left-drag painting: brush strokes continuously, rectangle/line preview
        from the drag origin. Only the mask overlay is redrawn per move, and the
        mask is saved once on release."""
        if not self._editing or ev.button() != Qt.LeftButton:
            return self._orig_mouse_drag(ev, axis=axis)
        ev.accept()
        value = not self.chk_erase.isChecked()
        tool = self.cmb_tool.currentText()
        pt = self._event_native(ev.scenePos())

        if ev.isStart():
            self._line_start = None  # a drag cancels any pending two-click line
            self._drag_start = pt
            self._last_brush = pt
            self._stroke_snapshot = (self.mask.copy()
                                     if tool in ('Rectangle', 'Line') else None)
            if pt is not None and tool == 'Brush':
                self._paint_disk(pt[0], pt[1], self.spn_brush.value(), value)
                self._refresh_overlay()
            return

        if pt is not None:
            if tool == 'Brush':
                start = self._last_brush or pt
                self._stamp_along(start, pt, max(1, self.spn_brush.value()), value)
                self._last_brush = pt
            elif self._stroke_snapshot is not None and self._drag_start is not None:
                self.mask[:] = self._stroke_snapshot
                if tool == 'Rectangle':
                    self._paint_rect(self._drag_start, pt, value)
                else:  # Line
                    self._paint_line(self._drag_start, pt, self.spn_thickness.value(), value)
            self._refresh_overlay()

        if ev.isFinish():
            self._stroke_snapshot = None
            self._drag_start = None
            self._last_brush = None
            self.mask_updated.emit(self.mask)

    def _paint_disk(self, row, col, radius, value):
        """Set a circular region of pixels (native coords) to ``value``."""
        radius = max(1, int(radius))
        h, w = self.mask.shape
        r0, r1 = max(0, row - radius + 1), min(h, row + radius)
        c0, c1 = max(0, col - radius + 1), min(w, col + radius)
        if r0 >= r1 or c0 >= c1:
            return
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disk = (rr - row) ** 2 + (cc - col) ** 2 < radius * radius
        self.mask[r0:r1, c0:c1][disk] = value

    def _paint_square(self, row, col, side, value):
        """Set a square region of side ``side`` centered at (row, col)."""
        side = max(1, int(side))
        half = side // 2
        h, w = self.mask.shape
        r0, r1 = max(0, row - half), min(h, row - half + side)
        c0, c1 = max(0, col - half), min(w, col - half + side)
        if r0 < r1 and c0 < c1:
            self.mask[r0:r1, c0:c1] = value

    def _paint_rect(self, corner0, corner1, value):
        """Set the filled rectangle spanning two native-coord corners."""
        (r0, c0), (r1, c1) = corner0, corner1
        rlo, rhi = sorted((int(r0), int(r1)))
        clo, chi = sorted((int(c0), int(c1)))
        self.mask[rlo:rhi + 1, clo:chi + 1] = value

    def _stamp_along(self, start, end, radius, value):
        """Stamp disks of ``radius`` along the segment start→end (native coords)."""
        (r0, c0), (r1, c1) = start, end
        steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rows = np.linspace(r0, r1, steps).round().astype(int)
        cols = np.linspace(c0, c1, steps).round().astype(int)
        for r, c in zip(rows, cols):
            self._paint_disk(int(r), int(c), radius, value)

    def _paint_line(self, start, end, thickness, value):
        """Set a thick line between two native-coord points to ``value``."""
        # _paint_disk(radius=R) paints a ~(2R-1)px-wide dot, so R=(T+1)/2.
        radius = max(1, int(round((thickness + 1) / 2)))
        self._stamp_along(start, end, radius, value)
