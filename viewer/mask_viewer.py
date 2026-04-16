import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QSpinBox, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal


class MaskViewerWindow(QDialog):
    """
    Displays and optionally edits a boolean detector mask.

    Shows the mask as a binary image with optional diffraction image
    overlay. Supports pixel toggling in edit mode with configurable
    brush size. Has own rotate/transpose controls initialized from
    the parent viewer's settings.
    """

    mask_updated = pyqtSignal(object)

    def __init__(self, mask, mask_path=None, parent=None):
        super().__init__(parent)
        self.parent_viewer = parent
        self.mask = mask.copy().astype(bool)
        self.mask_path = mask_path
        self._editing = False
        self._show_image = False
        self._alpha = 0.5

        # Local orientation controls, initialized from parent viewer
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

        self.btn_transpose = QPushButton('Transpose')
        self.btn_transpose.clicked.connect(self._toggle_transpose)
        ctrl.addWidget(self.btn_transpose)

        self.btn_rotate = QPushButton('Rotate 90')
        self.btn_rotate.clicked.connect(self._rotate)
        ctrl.addWidget(self.btn_rotate)

        self.chk_edit = QCheckBox('Edit Mode')
        self.chk_edit.stateChanged.connect(self._toggle_edit)
        ctrl.addWidget(self.chk_edit)

        ctrl.addWidget(QLabel('Brush:'))
        self.spn_brush = QSpinBox()
        self.spn_brush.setRange(1, 10)
        self.spn_brush.setValue(1)
        self.spn_brush.setSuffix(' px')
        ctrl.addWidget(self.spn_brush)

        ctrl.addStretch()
        layout.addLayout(ctrl)

    def _info_text(self):
        num_masked = int(np.sum(self.mask))
        total = self.mask.size
        pct = 100 * num_masked / total if total > 0 else 0
        return f"Masked: {num_masked:,} / {total:,} pixels ({pct:.1f}%)"

    def _get_current_image(self):
        """Get current diffraction image from parent viewer if available."""
        if self.parent_viewer is not None and hasattr(self.parent_viewer, 'reader'):
            reader = self.parent_viewer.reader
            if reader is not None and reader.image is not None:
                return reader.image.copy()
        return None

    def _transform_image_to_match_mask(self, data):
        """Transform the raw diffraction image to match the mask's current orientation."""
        if self._is_transposed:
            if data.ndim == 2:
                data = np.transpose(data)
            else:
                data = np.transpose(data, axes=(1, 0, 2))
        if self._rot_num:
            data = np.rot90(data, k=self._rot_num)
        return data

    def _click_to_mask(self, view_x, view_y):
        """Map view click coordinates to mask indices.

        Transpose/rotate modify self.mask directly, so the displayed
        array IS self.mask. pyqtgraph renders data[i,j] at (x=i, y=j),
        so click (vx, vy) → mask[vx, vy].
        """
        if 0 <= view_x < self.mask.shape[0] and 0 <= view_y < self.mask.shape[1]:
            return view_x, view_y
        return None

    def _toggle_transpose(self):
        self.mask = self.mask.T.copy()
        self._is_transposed = not self._is_transposed
        self._refresh_display()
        self.mask_updated.emit(self.mask)

    def _rotate(self):
        self.mask = np.rot90(self.mask, k=1).copy()
        self._rot_num = (self._rot_num + 1) % 4
        self._refresh_display()
        self.mask_updated.emit(self.mask)

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
        if self._show_image:
            img = self._get_current_image()
            if img is not None:
                log_on, parent_levels, colormap = self._get_parent_display_settings()

                # Transform raw diffraction image to match mask orientation
                img = self._transform_image_to_match_mask(img)

                img_float = img.astype(np.float64)
                if log_on:
                    img_float = np.maximum(img_float, 0)
                    img_float = np.log10(img_float + 1)

                if parent_levels is not None:
                    img_min, img_max = parent_levels
                else:
                    img_min, img_max = img_float.min(), img_float.max()

                rng = img_max - img_min
                if rng > 0:
                    img_norm = np.clip((img_float - img_min) / rng, 0, 1)
                else:
                    img_norm = np.zeros_like(img_float)

                # Build RGBA: grayscale image + mask in red with alpha
                h, w = img_norm.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.float32)
                rgba[..., 0] = img_norm
                rgba[..., 1] = img_norm
                rgba[..., 2] = img_norm
                rgba[..., 3] = 1.0

                mask_resized = self.mask
                if mask_resized.shape != img_norm.shape[:2]:
                    from skimage.transform import resize
                    mask_resized = resize(self.mask.astype(np.uint8),
                                          img_norm.shape[:2], order=0,
                                          preserve_range=True).astype(bool)
                rgba[mask_resized, 0] = 1.0 * self._alpha + img_norm[mask_resized] * (1 - self._alpha)
                rgba[mask_resized, 1] = 0.0 * self._alpha + img_norm[mask_resized] * (1 - self._alpha)
                rgba[mask_resized, 2] = 0.0 * self._alpha + img_norm[mask_resized] * (1 - self._alpha)

                self.image_view.setImage(rgba, autoRange=False, autoLevels=False,
                                         levels=(0, 1))
            else:
                self._show_mask_only()
        else:
            self._show_mask_only()
        self.lbl_info.setText(self._info_text())

    def _show_mask_only(self):
        """Display mask directly — transpose/rotate already modify self.mask."""
        cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 50, 50)])
        self.image_view.setColorMap(cmap)
        self.image_view.setImage(self.mask.astype(np.float32),
                                 autoRange=False, autoLevels=False,
                                 levels=(0, 1))

    def _toggle_image_overlay(self, state):
        self._show_image = (state == Qt.Checked)
        self._refresh_display()

    def _alpha_changed(self, value):
        self._alpha = value / 100.0
        self.lbl_alpha.setText(f'{value}%')
        if self._show_image:
            self._refresh_display()

    def _save_mask(self):
        # If no path set, get default from parent's mask_manager
        if not self.mask_path:
            if self.parent_viewer and hasattr(self.parent_viewer, 'mask_manager'):
                mm = self.parent_viewer.mask_manager
                self.mask_path = os.path.join(mm.masks_dir, mm.DEFAULT_MASK_FILENAME)
            else:
                self.lbl_info.setText("Error: No save path available")
                return

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

    def _toggle_edit(self, state):
        self._editing = (state == Qt.Checked)
        if self._editing:
            self.image_view.getView().scene().sigMouseClicked.connect(self._on_click)
        else:
            try:
                self.image_view.getView().scene().sigMouseClicked.disconnect(self._on_click)
            except TypeError:
                pass

    def _on_click(self, event):
        if not self._editing:
            return
        pos = event.scenePos()
        mouse_point = self.plot_item.vb.mapSceneToView(pos)
        vx = int(round(mouse_point.x()))
        vy = int(round(mouse_point.y()))

        result = self._click_to_mask(vx, vy)
        if result is not None:
            raw_row, raw_col = result
            radius = self.spn_brush.value()
            self._toggle_region(raw_row, raw_col, radius)
            self._refresh_display()
            self.mask_updated.emit(self.mask)

    def _toggle_region(self, row, col, radius):
        """Toggle a circular region of pixels."""
        h, w = self.mask.shape
        new_val = not self.mask[row, col]
        for dr in range(-radius + 1, radius):
            for dc in range(-radius + 1, radius):
                if dr * dr + dc * dc < radius * radius:
                    r, c = row + dr, col + dc
                    if 0 <= r < h and 0 <= c < w:
                        self.mask[r, c] = new_val
