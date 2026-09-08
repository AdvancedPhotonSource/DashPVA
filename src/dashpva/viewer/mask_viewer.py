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

import os

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt5.QtCore import QEvent, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

import dashpva.settings as settings


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
        self._original_mask = self.mask.copy()  # snapshot at open, for the close-time save/discard prompt
        self.mask_path = mask_path
        self._original_path = mask_path
        self._editing = False
        self._show_image = False
        self._alpha = 0.5
        self._undo_stack = []
        self._redo_stack = []
        self._history_bytes = 0
        # Line tool: first click/drag start stores the start point until the
        # second interaction completes the segment.
        self._line_start = None
        # Active-stroke state (set at press, cleared at release).
        self._edit_press_pt = None   # native (row, col) at mouse press
        self._edit_dragging = False  # True once the mouse has moved
        self._stroke_bounds = None
        self._stroke_before = None
        self._last_brush = None      # last-stamped native pt (gap-free brush)
        self._view_ready = False     # True after first setRange so zoom is preserved
        self._resume_plotting = False  # timer_plot was active before an edit-mode pause
        self._canvas_shape = None    # set by _refresh_display(): shown image's own resolution

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
        # Prevent the ViewBox from auto-ranging when setImage is called (e.g. when
        # the diffraction image is toggled on/off). Only our explicit setRange call
        # (on first open) should control the zoom level.
        self.plot_item.vb.enableAutoRange(enable=False)

        # Cache the viewport once — gv.viewport() creates a new Python wrapper on
        # every call, so `obj is gv.viewport()` in eventFilter would always be False.
        self._gv_viewport = self.image_view.ui.graphicsView.viewport()
        self._gv_viewport.installEventFilter(self)

        # Info label
        self.lbl_info = QLabel(self._info_text())
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # Shown only when the mask and the live diffraction image are different
        # resolutions — edits still land in the right place (scaled), but the
        # scaling can distort fine detail in the mask (e.g. thin lines widening
        # or thinning under up/downsampling).
        self.lbl_shape_warning = QLabel()
        self.lbl_shape_warning.setObjectName('lbl_mask_shape_warning')
        self.lbl_shape_warning.setAlignment(Qt.AlignCenter)
        self.lbl_shape_warning.setVisible(False)
        layout.addWidget(self.lbl_shape_warning)

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

        # Controls row 2: history + mask operations
        row_ops = QHBoxLayout()

        self.btn_undo = QPushButton()
        self.btn_undo.setIcon(qta.icon('fa5s.undo'))
        self.btn_undo.setToolTip('Undo (Ctrl+Z)')
        self.btn_undo.setFixedWidth(32)
        self.btn_undo.clicked.connect(self.undo)
        row_ops.addWidget(self.btn_undo)

        self.btn_redo = QPushButton()
        self.btn_redo.setIcon(qta.icon('fa5s.redo'))
        redo_keys = QKeySequence(QKeySequence.Redo).toString(QKeySequence.NativeText)
        self.btn_redo.setToolTip(f'Redo ({redo_keys})')
        self.btn_redo.setFixedWidth(32)
        self.btn_redo.clicked.connect(self.redo)
        row_ops.addWidget(self.btn_redo)

        self.btn_save = QPushButton('Save')
        self.btn_save.clicked.connect(self._save_mask)
        row_ops.addWidget(self.btn_save)

        self.btn_clear = QPushButton('Clear')
        self.btn_clear.setToolTip('Remove all masked pixels (undoable)')
        self.btn_clear.clicked.connect(self._clear_drawing)
        row_ops.addWidget(self.btn_clear)

        self.btn_invert = QPushButton('Invert')
        self.btn_invert.clicked.connect(self._invert_mask)
        row_ops.addWidget(self.btn_invert)

        self.btn_export_json = QPushButton('Export JSON')
        self.btn_export_json.setToolTip('Export as EPICS NDPluginBadPixel JSON')
        self.btn_export_json.clicked.connect(self._export_json)
        row_ops.addWidget(self.btn_export_json)

        row_ops.addStretch()
        layout.addLayout(row_ops)

        # Controls row 3: orientation + drawing tools
        row_draw = QHBoxLayout()

        self.btn_transpose = QPushButton('Transpose')
        self.btn_transpose.clicked.connect(self._toggle_transpose)
        row_draw.addWidget(self.btn_transpose)

        self.btn_rotate = QPushButton('Rotate 90')
        self.btn_rotate.clicked.connect(self._rotate)
        row_draw.addWidget(self.btn_rotate)

        self.chk_edit = QCheckBox('Edit Mode')
        self.chk_edit.stateChanged.connect(self._toggle_edit)
        row_draw.addWidget(self.chk_edit)

        row_draw.addWidget(QLabel('Tool:'))
        self.cmb_tool = QComboBox()
        self.cmb_tool.addItems(['Brush', 'Rectangle', 'Line'])
        self.cmb_tool.currentTextChanged.connect(self._tool_changed)
        row_draw.addWidget(self.cmb_tool)

        row_draw.addWidget(QLabel('Size:'))
        self.spn_thickness = QSpinBox()
        self.spn_thickness.setToolTip('Brush radius / square side / line thickness, in pixels')
        self.spn_thickness.setRange(1, 500)
        self.spn_thickness.setValue(3)
        self.spn_thickness.setSuffix(' px')
        row_draw.addWidget(self.spn_thickness)

        self.chk_erase = QCheckBox('Erase')
        self.chk_erase.setToolTip('Draw removes pixels from the mask instead of adding them')
        row_draw.addWidget(self.chk_erase)

        row_draw.addStretch()
        layout.addLayout(row_draw)

        QShortcut(QKeySequence.Undo, self, self.undo)
        QShortcut(QKeySequence.Redo, self, self.redo)
        self._update_undo_buttons()

    def closeEvent(self, event):
        if not np.array_equal(self.mask, self._original_mask):
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'This mask has changes since it was opened. Save before closing?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Discard:
                self.mask = self._original_mask.copy()
            elif reply == QMessageBox.Save:
                if not self._save_mask():
                    event.ignore()
                    return
        self._pause_live_plotting(False)
        event.accept()

    def reject(self):
        self.close()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._view_ready:
            # Defer to let Qt finish laying out the just-shown dialog first —
            # fitting the range against not-yet-final geometry is what makes
            # the initial view come up zoomed in far too much.
            QTimer.singleShot(0, self._apply_initial_range)

    def _apply_initial_range(self):
        if self._view_ready:
            return
        # autoRange() fits the view to the items already on the scene while
        # respecting the ImageView's locked (square-pixel) aspect ratio.
        # Setting xRange/yRange explicitly is over-constrained under a locked
        # aspect — pyqtgraph resolves the conflict by blowing one axis's
        # range up far past the mask, which is what produced the huge zoom.
        self.plot_item.vb.autoRange(padding=0.02)
        self._view_ready = True

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    @staticmethod
    def _history_size(entry):
        return 128 + sum(
            value.nbytes for value in entry.values()
            if isinstance(value, np.ndarray)
        )

    def _push_history(self, entry):
        self._history_bytes -= sum(self._history_size(item) for item in self._redo_stack)
        self._redo_stack.clear()
        size = self._history_size(entry)
        if size > settings.MASK_UNDO_MAX_BYTES:
            self._update_undo_buttons()
            return
        self._undo_stack.append(entry)
        self._history_bytes += size
        while self._history_bytes > settings.MASK_UNDO_MAX_BYTES and self._undo_stack:
            self._history_bytes -= self._history_size(self._undo_stack.pop(0))
        self._update_undo_buttons()

    def _apply_history(self, entry, undo):
        kind = entry["kind"]
        if kind == "region":
            r0, r1, c0, c1 = entry["bounds"]
            self.mask[r0:r1, c0:c1] = entry["before" if undo else "after"]
        elif kind == "transpose":
            self.mask = self.mask.T.copy()
        elif kind == "rotate":
            self.mask = np.rot90(self.mask, -1 if undo else 1).copy()
        elif kind == "invert":
            self.mask = ~self.mask
        elif kind == "clear":
            if undo:
                restored = np.unpackbits(entry["before"], count=self.mask.size)
                self.mask = restored.reshape(entry["shape"]).astype(bool)
            else:
                self.mask[:] = False

    def undo(self):
        if not self._undo_stack:
            return
        prev_shape = self.mask.shape
        entry = self._undo_stack.pop()
        self._apply_history(entry, True)
        self._redo_stack.append(entry)
        self._refresh_shape_change(prev_shape)
        self._update_undo_buttons()

    def redo(self):
        if not self._redo_stack:
            return
        prev_shape = self.mask.shape
        entry = self._redo_stack.pop()
        self._apply_history(entry, False)
        self._undo_stack.append(entry)
        self._refresh_shape_change(prev_shape)
        self._update_undo_buttons()

    def _refresh_shape_change(self, prev_shape):
        """Full redraw if the mask's shape changed (e.g. undoing a transpose/rotate),
        otherwise the cheap overlay-only refresh."""
        if self.mask.shape != prev_shape:
            self._view_ready = False
            self._refresh_display()
        else:
            self._refresh_overlay()

    def _clear_drawing(self):
        entry = {
            "kind": "clear",
            "shape": self.mask.shape,
            "before": np.packbits(self.mask.reshape(-1)),
        }
        self.mask[:] = False
        self._push_history(entry)
        self._refresh_overlay()

    def _update_undo_buttons(self):
        self.btn_undo.setEnabled(bool(self._undo_stack))
        self.btn_redo.setEnabled(bool(self._redo_stack))

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

    def _display_shape(self):
        """Shape _get_display_mask() would produce, without building a copy."""
        h, w = self.mask.shape
        m, n = (w, h) if self._is_transposed else (h, w)  # shape just before rot90
        return (n, m) if self._rot_num % 2 else (m, n)

    def _display_to_native(self, disp_i, disp_j):
        """Reverse-map display coordinates back to detector-native indices.

        Closed-form inverse of the forward transform chain (transpose then
        rot90): both are index permutations, so this is direct arithmetic
        instead of building a probe array and scanning it with argwhere —
        this runs on every mouse-move while drawing.
        """
        h, w = self.mask.shape
        m, n = (w, h) if self._is_transposed else (h, w)  # shape just before rot90
        k = self._rot_num % 4
        if k == 0:
            ti, tj = disp_i, disp_j
        elif k == 1:
            ti, tj = disp_j, n - 1 - disp_i
        elif k == 2:
            ti, tj = m - 1 - disp_i, n - 1 - disp_j
        else:  # k == 3
            ti, tj = m - 1 - disp_j, disp_i
        return (tj, ti) if self._is_transposed else (ti, tj)

    @staticmethod
    def _scale_canvas_index(index, canvas_size, mask_size):
        return min(mask_size - 1, max(0, int(index) * mask_size // canvas_size))

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
        self._push_history({"kind": "transpose"})
        self.mask = self.mask.T.copy()
        self._view_ready = False
        self._refresh_display()

    def _rotate(self):
        self._push_history({"kind": "rotate"})
        self.mask = np.rot90(self.mask, k=1).copy()
        self._view_ready = False
        self._refresh_display()

    # ------------------------------------------------------------------
    # Display rendering
    # ------------------------------------------------------------------

    def _get_parent_display_settings(self):
        """Get display settings (log, levels) from parent viewer."""
        log_on = False
        levels = None
        if self.parent_viewer is not None:
            if hasattr(self.parent_viewer, 'log_image'):
                log_on = self.parent_viewer.log_image.isChecked()
            if hasattr(self.parent_viewer, 'image_view'):
                try:
                    levels = self.parent_viewer.image_view.getLevels()
                except Exception:
                    pass
        return log_on, levels

    def _refresh_display(self):
        """Full redraw: grayscale base layer (diffraction image or black) plus
        the red mask overlay. Use _refresh_overlay for cheap mask-only updates
        during interactive drawing."""
        mask_disp_shape = self._display_shape()
        base = None
        if self._show_image:
            img = self._get_current_image()
            if img is not None:
                log_on, parent_levels = self._get_parent_display_settings()
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
            base = np.zeros(mask_disp_shape, dtype=np.float32)

        # The canvas is the diffraction image's own resolution while it's shown
        # (a previously-saved mask can be a different resolution than the live
        # detector — different ROI/binning) — never warp the image to the
        # mask's shape. The mask overlay is resampled to match in
        # _refresh_overlay(), and click coordinates are scaled back to the
        # mask's own array in _event_native(), so editing still lands on the
        # right pixels.
        self._canvas_shape = base.shape[:2]
        shape_mismatch = self._show_image and self._canvas_shape != mask_disp_shape
        self.lbl_shape_warning.setVisible(shape_mismatch)
        if shape_mismatch:
            self.lbl_shape_warning.setText(
                f"Mask ({mask_disp_shape[0]}x{mask_disp_shape[1]}) and image "
                f"({self._canvas_shape[0]}x{self._canvas_shape[1]}) don't match — "
                f"there may be distortion."
            )

        self.image_view.setColorMap(pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)]))
        self.image_view.setImage(base.astype(np.float32), autoRange=False,
                                 autoLevels=False, levels=(0, 1))
        # setImage() must run before setRect(): pg.ImageItem computes setRect's
        # scale against whatever image size it has *at that moment* — calling
        # setRect on a still-imageless ImageItem bakes in a scale against a
        # placeholder 1x1 size, and a later setImage() never recomputes it,
        # leaving the item's real (scene-mapped) bounds squared (e.g. a
        # 100x400 mask reporting bounds of 10000x160000).
        self._refresh_overlay()
        self.image_view.getImageItem().setRect(QRectF(0, 0, *self._canvas_shape))
        self.mask_overlay.setRect(QRectF(0, 0, *self._canvas_shape))
        # Only fit the range here for shape changes while already visible (e.g.
        # transpose/rotate mid-session) — the very first fit is deferred to
        # showEvent's _apply_initial_range, once the dialog has real geometry.
        if not self._view_ready and self.isVisible():
            self.plot_item.vb.autoRange(padding=0.02)
            self._view_ready = True

    @staticmethod
    def _nearest_resize(arr, out_shape):
        """Nearest-neighbor resize of a 2D array to out_shape via index arrays —
        used to render the mask overlay at the diffraction image's own
        resolution when it differs from the mask's."""
        in_h, in_w = arr.shape
        out_h, out_w = out_shape
        row_idx = np.arange(out_h) * in_h // out_h
        col_idx = np.arange(out_w) * in_w // out_w
        return arr[row_idx][:, col_idx]

    def _refresh_overlay(self, update_info=True):
        """Cheap update of just the red mask layer — no base-image recompute.

        ``update_info`` is False during interactive dragging: _info_text()
        does a full-mask np.sum, too slow to run on every mouse-move.
        """
        display_mask = self._get_display_mask()
        if self._canvas_shape != display_mask.shape:
            display_mask = self._nearest_resize(display_mask, self._canvas_shape)
        self.mask_overlay.setImage(display_mask.astype(np.float32),
                                   autoLevels=False, levels=(0, 1))
        self.mask_overlay.setOpacity(self._alpha if self._show_image else 1.0)
        if update_info:
            self.lbl_info.setText(self._info_text())

    def _refresh_native_region(self, bounds, update_info=True):
        r0, r1, c0, c1 = bounds
        image = self.mask_overlay.image
        if image is None or image.shape != self._canvas_shape:
            self._refresh_overlay(update_info=update_info)
            return
        patch = self.mask[r0:r1, c0:c1]
        display_h, display_w = self._display_shape()
        pre_h, pre_w = (
            (self.mask.shape[1], self.mask.shape[0])
            if self._is_transposed
            else self.mask.shape
        )
        if self._is_transposed:
            patch = patch.T
            dr0, dr1, dc0, dc1 = c0, c1, r0, r1
        else:
            dr0, dr1, dc0, dc1 = r0, r1, c0, c1
        rotation = self._rot_num % 4
        if rotation == 1:
            patch = np.rot90(patch, 1)
            dr0, dr1, dc0, dc1 = pre_w - dc1, pre_w - dc0, dr0, dr1
        elif rotation == 2:
            patch = np.rot90(patch, 2)
            dr0, dr1, dc0, dc1 = (
                pre_h - dr1, pre_h - dr0, pre_w - dc1, pre_w - dc0
            )
        elif rotation == 3:
            patch = np.rot90(patch, 3)
            dr0, dr1, dc0, dc1 = dc0, dc1, pre_h - dr1, pre_h - dr0

        canvas_h, canvas_w = self._canvas_shape
        cr0 = (dr0 * canvas_h + display_h - 1) // display_h
        cr1 = (dr1 * canvas_h + display_h - 1) // display_h
        cc0 = (dc0 * canvas_w + display_w - 1) // display_w
        cc1 = (dc1 * canvas_w + display_w - 1) // display_w
        row_idx = np.arange(cr0, cr1) * display_h // canvas_h - dr0
        col_idx = np.arange(cc0, cc1) * display_w // canvas_w - dc0
        image[cr0:cr1, cc0:cc1] = patch[row_idx][:, col_idx]
        self.mask_overlay.update()
        if update_info:
            self.lbl_info.setText(self._info_text())

    def _toggle_image_overlay(self, state):
        self._show_image = (state == Qt.Checked)
        self._refresh_display()
        self.plot_item.vb.autoRange(padding=0.02)

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
                return False

        try:
            np.save(self.mask_path, self.mask)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return False
        self._original_mask = self.mask.copy()
        self._original_path = self.mask_path
        if self.parent_viewer and hasattr(self.parent_viewer, 'mask_manager'):
            self.parent_viewer.mask_manager.mask_path = self.mask_path
        self.mask_updated.emit(self.mask)
        num_masked = int(np.sum(self.mask))
        h, w = self.mask.shape
        self.setWindowTitle(f"Mask -- {self.mask_path} ({w}x{h}, {num_masked} masked)")
        self.lbl_info.setText(f"Saved! {self._info_text()}")
        return True

    def _invert_mask(self):
        self._push_history({"kind": "invert"})
        self.mask = ~self.mask
        self._refresh_display()

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
            mask = self.mask.T if self._is_transposed else self.mask  # -> raw detector orientation
            rows, cols = np.where(mask)
            for row, col in zip(rows, cols):
                bad_pixels.append({"Pixel": [int(col), int(row)], "Set": 0})
            mask_rows, mask_cols = mask.shape
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
    # Edit mode — viewport event filter handles all mouse editing so that
    # left-button press is consumed before pyqtgraph's ViewBox sees it
    # (preventing pan/zoom while drawing).
    # ------------------------------------------------------------------

    def _toggle_edit(self, state):
        self._editing = (state == Qt.Checked)
        # Disable ViewBox pan/zoom while drawing so mouse events are ours alone.
        self.plot_item.vb.setMouseEnabled(x=not self._editing, y=not self._editing)
        self._pause_live_plotting(self._editing)
        self._line_start = None
        self._edit_press_pt = None
        self._edit_dragging = False
        self._stroke_bounds = None
        self._stroke_before = None
        self._last_brush = None

    def _pause_live_plotting(self, pause):
        """Stop the parent viewer's live-plot timer while drawing on large detectors.

        On large detectors, the per-frame redraw (mask apply, transpose,
        rotate, log, autoscale) can consume most or all of the timer_plot
        period, starving this dialog's own repaint so drawn strokes don't
        visibly appear until streaming stops. Below settings.MASK_EDITOR_PAUSE_MIN_PIXELS
        that contention doesn't happen, so leave the live view running.
        """
        if self.mask.size < settings.MASK_EDITOR_PAUSE_MIN_PIXELS:
            return
        timer = getattr(self.parent_viewer, 'timer_plot', None)
        if timer is None:
            return
        if pause:
            self._resume_plotting = timer.isActive()
            timer.stop()
        elif self._resume_plotting:
            timer.start()
            self._resume_plotting = False

    def _tool_changed(self, *_):
        """Reset any in-progress line when the active tool changes."""
        self._line_start = None
        self.lbl_info.setText(self._info_text())

    def _brush_bounds(self, points, width):
        rows = [point[0] for point in points]
        cols = [point[1] for point in points]
        offset = (max(1, int(width)) - 1) // 2
        extent = max(1, int(width)) - offset
        return (
            max(0, min(rows) - offset),
            min(self.mask.shape[0], max(rows) + extent),
            max(0, min(cols) - offset),
            min(self.mask.shape[1], max(cols) + extent),
        )

    def _capture_stroke_region(self, bounds):
        if self._stroke_bounds is None:
            self._stroke_bounds = bounds
            r0, r1, c0, c1 = bounds
            self._stroke_before = self.mask[r0:r1, c0:c1].copy()
            return
        old = self._stroke_bounds
        merged = (
            min(old[0], bounds[0]), max(old[1], bounds[1]),
            min(old[2], bounds[2]), max(old[3], bounds[3]),
        )
        if merged == old:
            return
        r0, r1, c0, c1 = merged
        before = self.mask[r0:r1, c0:c1].copy()
        or0, or1, oc0, oc1 = old
        before[or0-r0:or1-r0, oc0-c0:oc1-c0] = self._stroke_before
        self._stroke_bounds = merged
        self._stroke_before = before

    def _restore_stroke_region(self):
        if self._stroke_bounds is None:
            return
        r0, r1, c0, c1 = self._stroke_bounds
        self.mask[r0:r1, c0:c1] = self._stroke_before

    def _finish_stroke(self):
        if self._stroke_bounds is None:
            return
        r0, r1, c0, c1 = self._stroke_bounds
        after = self.mask[r0:r1, c0:c1].copy()
        if not np.array_equal(self._stroke_before, after):
            self._push_history({
                "kind": "region",
                "bounds": self._stroke_bounds,
                "before": self._stroke_before,
                "after": after,
            })

    def _event_native(self, scene_pos, clamp=False):
        """Map a scene position to a detector-native (row, col). Returns None if
        the point is outside the mask, unless ``clamp`` is set, in which case the
        point is pinned to the nearest edge so a stroke that leaves the image
        still fills up to the boundary instead of vanishing."""
        mouse_point = self.plot_item.vb.mapSceneToView(scene_pos)
        # pyqtgraph pixel (i,j) occupies [i, i+1) x [j, j+1) — use floor
        vx = int(np.floor(mouse_point.x()))
        vy = int(np.floor(mouse_point.y()))
        disp_shape = self._display_shape()
        canvas_shape = self._canvas_shape or disp_shape
        if clamp:
            vx = min(max(vx, 0), canvas_shape[0] - 1)
            vy = min(max(vy, 0), canvas_shape[1] - 1)
        elif not (0 <= vx < canvas_shape[0] and 0 <= vy < canvas_shape[1]):
            return None
        vx = self._scale_canvas_index(vx, canvas_shape[0], disp_shape[0])
        vy = self._scale_canvas_index(vy, canvas_shape[1], disp_shape[1])
        row, col = self._display_to_native(vx, vy)
        if clamp:
            row = min(max(row, 0), self.mask.shape[0] - 1)
            col = min(max(col, 0), self.mask.shape[1] - 1)
        elif not (0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]):
            return None
        return row, col

    def eventFilter(self, obj, event):
        if not self._editing:
            return super().eventFilter(obj, event)
        event_type = event.type()
        if event_type not in (
            QEvent.MouseButtonPress,
            QEvent.MouseButtonDblClick,
            QEvent.MouseMove,
            QEvent.MouseButtonRelease,
        ):
            return super().eventFilter(obj, event)

        graphics_view = self.image_view.ui.graphicsView
        tool = self.cmb_tool.currentText()
        value = not self.chk_erase.isChecked()
        width = self.spn_thickness.value()

        if event_type in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            if event.button() != Qt.LeftButton:
                return super().eventFilter(obj, event)
            point = self._event_native(graphics_view.mapToScene(event.pos()))
            if point is None:
                return False
            self._edit_press_pt = point
            self._edit_dragging = False
            self._stroke_bounds = None
            self._stroke_before = None
            self._last_brush = point
            if tool == 'Brush':
                self._capture_stroke_region(self._brush_bounds((point,), width))
                self._paint_disk(point[0], point[1], width, value)
                self._refresh_native_region(self._stroke_bounds, update_info=False)
            return True

        if event_type == QEvent.MouseMove and event.buttons() & Qt.LeftButton:
            if self._edit_press_pt is None:
                return False
            point = self._event_native(graphics_view.mapToScene(event.pos()), clamp=True)
            self._edit_dragging = True
            if tool == 'Brush':
                start = self._last_brush or point
                self._capture_stroke_region(self._brush_bounds((start, point), width))
                self._stamp_along(start, point, width, value)
                self._last_brush = point
            else:
                start = (
                    self._line_start
                    if tool == 'Line' and self._line_start is not None
                    else self._edit_press_pt
                )
                bounds = self._brush_bounds(
                    (start, point), width if tool == 'Line' else 1
                )
                self._capture_stroke_region(bounds)
                self._restore_stroke_region()
                if tool == 'Rectangle':
                    self._paint_rect(start, point, value)
                else:
                    self._paint_line(start, point, width, value)
            self._refresh_native_region(self._stroke_bounds, update_info=False)
            return True

        if event_type == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._edit_press_pt is None:
                return False
            point = (
                self._event_native(graphics_view.mapToScene(event.pos()), clamp=True)
                or self._edit_press_pt
            )
            if not self._edit_dragging and tool == 'Rectangle':
                self._capture_stroke_region(self._brush_bounds((point,), width))
                self._paint_square(point[0], point[1], width, value)
            elif not self._edit_dragging and tool == 'Line':
                if self._line_start is None:
                    self._line_start = point
                    self.lbl_info.setText('Line: click the end point… (or drag)')
                    self._reset_stroke()
                    return True
                self._capture_stroke_region(
                    self._brush_bounds((self._line_start, point), width)
                )
                self._paint_line(self._line_start, point, width, value)
                self._line_start = None
            elif self._edit_dragging and tool != 'Brush':
                self._restore_stroke_region()
                start = (
                    self._line_start
                    if tool == 'Line' and self._line_start is not None
                    else self._edit_press_pt
                )
                if tool == 'Rectangle':
                    self._paint_rect(start, point, value)
                else:
                    self._paint_line(start, point, width, value)
                    self._line_start = None
            self._finish_stroke()
            if self._stroke_bounds is not None:
                self._refresh_native_region(self._stroke_bounds)
            self._reset_stroke()
            return True

        return super().eventFilter(obj, event)

    def _reset_stroke(self):
        self._edit_press_pt = None
        self._edit_dragging = False
        self._stroke_bounds = None
        self._stroke_before = None
        self._last_brush = None

    # ------------------------------------------------------------------
    # Paint helpers (always operate in detector-native coordinates)
    # ------------------------------------------------------------------

    def _paint_disk(self, row, col, width, value):
        """Set a circular brush footprint with an exact native-pixel width."""
        width = max(1, int(width))
        h, w = self.mask.shape
        offset = (width - 1) // 2
        r0, r1 = max(0, row - offset), min(h, row - offset + width)
        c0, c1 = max(0, col - offset), min(w, col - offset + width)
        if r0 >= r1 or c0 >= c1:
            return
        rr, cc = np.ogrid[r0:r1, c0:c1]
        center_row = row + (0.5 if width % 2 == 0 else 0.0)
        center_col = col + (0.5 if width % 2 == 0 else 0.0)
        disk = ((rr - center_row) ** 2 + (cc - center_col) ** 2
                <= (width / 2.0) ** 2)
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

    def _stamp_along(self, start, end, width, value):
        """Stamp exact-width brush footprints along a native-coordinate segment."""
        (r0, c0), (r1, c1) = start, end
        steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rows = np.linspace(r0, r1, steps).round().astype(int)
        cols = np.linspace(c0, c1, steps).round().astype(int)
        for r, c in zip(rows, cols):
            self._paint_disk(int(r), int(c), width, value)

    def _paint_line(self, start, end, thickness, value):
        """Set a thick line between two native-coord points to ``value``."""
        self._stamp_along(start, end, thickness, value)
