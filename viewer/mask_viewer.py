import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal


class MaskViewerWindow(QDialog):
    """
    Displays and optionally edits a boolean detector mask.

    Shows the mask as a binary image (black=good, white=masked) using
    pyqtgraph ImageView. Supports pixel toggling in edit mode with
    configurable brush size.
    """

    mask_updated = pyqtSignal(object)

    def __init__(self, mask, mask_path=None, parent=None):
        super().__init__(parent)
        self.mask = mask.copy().astype(bool)
        self.mask_path = mask_path
        self._editing = False
        self._mouse_pressed = False

        num_masked = int(np.sum(self.mask))
        h, w = self.mask.shape
        path_display = mask_path or 'Unsaved mask'
        self.setWindowTitle(f"Mask -- {path_display} ({w}x{h}, {num_masked} masked)")
        self.resize(800, 700)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Image view for mask display
        self.plot_item = pg.PlotItem()
        self.plot_item.setLabel('bottom', 'X [pixels]')
        self.plot_item.setLabel('left', 'Y [pixels]')
        self.image_view = pg.ImageView(view=self.plot_item)
        # Binary colormap: black (good) / red (masked)
        cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 50, 50)])
        self.image_view.setColorMap(cmap)
        self.image_view.setImage(self.mask.astype(np.float32).T,
                                 autoRange=True, autoLevels=False,
                                 levels=(0, 1))
        # Hide histogram for binary image
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, stretch=3)

        # Info label
        self.lbl_info = QLabel(self._info_text())
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # Controls row
        ctrl = QHBoxLayout()

        self.btn_save = QPushButton('Save Mask')
        self.btn_save.clicked.connect(self._save_mask)
        ctrl.addWidget(self.btn_save)

        self.btn_invert = QPushButton('Invert Mask')
        self.btn_invert.clicked.connect(self._invert_mask)
        ctrl.addWidget(self.btn_invert)

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

    def _refresh_display(self):
        self.image_view.setImage(self.mask.astype(np.float32).T,
                                 autoRange=False, autoLevels=False,
                                 levels=(0, 1))
        self.lbl_info.setText(self._info_text())

    def _save_mask(self):
        if self.mask_path:
            np.save(self.mask_path, self.mask)
            self.mask_updated.emit(self.mask)
            num_masked = int(np.sum(self.mask))
            h, w = self.mask.shape
            self.setWindowTitle(f"Mask -- {self.mask_path} ({w}x{h}, {num_masked} masked)")

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
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        h, w = self.mask.shape
        # Note: image is displayed transposed, so x maps to col, y maps to row
        if 0 <= x < w and 0 <= y < h:
            radius = self.spn_brush.value()
            self._toggle_region(y, x, radius)
            self._refresh_display()
            self.mask_updated.emit(self.mask)

    def _toggle_region(self, row, col, radius):
        """Toggle a circular region of pixels."""
        h, w = self.mask.shape
        # Determine new value from center pixel
        new_val = not self.mask[row, col]
        for dr in range(-radius + 1, radius):
            for dc in range(-radius + 1, radius):
                if dr * dr + dc * dc < radius * radius:
                    r, c = row + dr, col + dc
                    if 0 <= r < h and 0 <= c < w:
                        self.mask[r, c] = new_val
