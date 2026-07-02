from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QWidget,
)

from dashpva.viewer.workbench.docks.information_dock_base import InformationDockBase


class Info2DDock(InformationDockBase):
    """Information dock specialized for 2D viewing state.

    Shows point counts, intensity range, axis labels, the mouse readout, and the
    per-frame HKL computed from the frame index via ``RSMConverter`` (off the GUI
    thread, with a busy progress bar). A Center/Hover toggle picks which detector
    pixel the frame's HKL corresponds to.
    """

    def __init__(
        self,
        main_window=None,
        title: str = "2D Info",
        segment_name: Optional[str] = "2d",
        dock_area: Qt.DockWidgetArea = Qt.RightDockWidgetArea,
        show: bool = True,
    ):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        # Per-frame HKL state
        self._hkl_cache: dict = {}
        self._hkl_key = None
        self._hkl_unavailable = False
        self._hkl_thread = None
        self._hkl_worker = None
        self._hkl_busy = False
        self._hkl_debounce = QTimer(self._widget)
        self._hkl_debounce.setSingleShot(True)
        self._hkl_debounce.setInterval(150)
        self._hkl_debounce.timeout.connect(self._compute_hkl_for_current)
        self._build_hkl_ui()

    # ------------------------------------------------------------------
    # UI (built in code — the base information_dock.ui is shared with Info3DDock)
    # ------------------------------------------------------------------
    def _build_hkl_ui(self) -> None:
        form = self._widget.layout() if self._widget is not None else None
        if form is None:
            return
        try:
            self.lbl_frame_H = QLabel("—")
            self.lbl_frame_K = QLabel("—")
            self.lbl_frame_L = QLabel("—")
            form.addRow(QLabel("Frame HKL"), QLabel(""))
            form.addRow("H:", self.lbl_frame_H)
            form.addRow("K:", self.lbl_frame_K)
            form.addRow("L:", self.lbl_frame_L)

            self.rb_pixel_center = QRadioButton("Center")
            self.rb_pixel_hover = QRadioButton("Hover")
            self.rb_pixel_center.setChecked(True)
            self._pixel_group = QButtonGroup(self._widget)
            self._pixel_group.addButton(self.rb_pixel_center)
            self._pixel_group.addButton(self.rb_pixel_hover)
            pixel_row = QWidget()
            prl = QHBoxLayout(pixel_row)
            prl.setContentsMargins(0, 0, 0, 0)
            prl.addWidget(self.rb_pixel_center)
            prl.addWidget(self.rb_pixel_hover)
            prl.addStretch(1)
            form.addRow("Pixel:", pixel_row)
            self.rb_pixel_center.toggled.connect(lambda _checked: self._update_hkl_labels())

            self.hkl_progress = QProgressBar()
            self.hkl_progress.setRange(0, 0)  # indeterminate / busy
            self.hkl_progress.setTextVisible(False)
            self.hkl_progress.setMaximumHeight(6)
            self.hkl_progress.setVisible(False)
            form.addRow("", self.hkl_progress)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-frame HKL computation (RSMConverter on a worker thread)
    # ------------------------------------------------------------------
    def _dataset_key(self):
        mw = self.main_window
        return (getattr(mw, 'current_file_path', None), getattr(mw, 'selected_dataset_path', None))

    def _current_frame(self) -> int:
        mw = self.main_window
        try:
            fs = getattr(mw, 'frame_spinbox', None)
            if fs is not None and fs.isEnabled():
                return int(fs.value())
        except Exception:
            pass
        return 0

    def _schedule_hkl_update(self) -> None:
        """Invalidate on dataset change, show what we have, then (debounced) compute."""
        key = self._dataset_key()
        if key != self._hkl_key:
            self._hkl_key = key
            self._hkl_cache.clear()
            self._hkl_unavailable = False
        self._update_hkl_labels()
        self._hkl_debounce.start()

    def _compute_hkl_for_current(self) -> None:
        mw = self.main_window
        file_path = getattr(mw, 'current_file_path', None)
        if not file_path or self._hkl_unavailable:
            self._update_hkl_labels()
            return
        frame = self._current_frame()
        if frame in self._hkl_cache:
            self._update_hkl_labels()
            return
        if self._hkl_busy:
            return  # will re-check when the running worker finishes
        self._start_hkl_worker(file_path, frame)

    def _start_hkl_worker(self, file_path, frame) -> None:
        from dashpva.viewer.workbench.workers import HKLFrameWorker
        self._hkl_busy = True
        if getattr(self, 'hkl_progress', None) is not None:
            self.hkl_progress.setVisible(True)
        self._hkl_thread = QThread(self._widget)
        self._hkl_worker = HKLFrameWorker(file_path, frame)
        self._hkl_worker.moveToThread(self._hkl_thread)
        self._hkl_thread.started.connect(self._hkl_worker.run)
        self._hkl_worker.computed.connect(self._on_hkl_computed)
        self._hkl_worker.failed.connect(self._on_hkl_failed)
        self._hkl_worker.finished.connect(self._on_hkl_finished)
        self._hkl_thread.start()

    @pyqtSlot(int, object)
    def _on_hkl_computed(self, frame, hkl) -> None:
        self._hkl_cache[int(frame)] = hkl
        if int(frame) == self._current_frame():
            self._update_hkl_labels()

    @pyqtSlot(int, str)
    def _on_hkl_failed(self, frame, msg) -> None:
        # No HKL geometry in this dataset — stop trying for it
        self._hkl_unavailable = True
        self._update_hkl_labels()

    @pyqtSlot()
    def _on_hkl_finished(self) -> None:
        try:
            if self._hkl_thread is not None:
                self._hkl_thread.quit()
                self._hkl_thread.wait()
        except Exception:
            pass
        self._hkl_thread = None
        self._hkl_worker = None
        self._hkl_busy = False
        if getattr(self, 'hkl_progress', None) is not None:
            self.hkl_progress.setVisible(False)
        # Catch up if the user scrubbed to another frame while this one computed
        if not self._hkl_unavailable:
            self._compute_hkl_for_current()

    def _chosen_pixel(self, h: int, w: int):
        """Return (y, x) for the toggle: hover pixel if selected & valid, else center."""
        if getattr(self, 'rb_pixel_hover', None) is not None and self.rb_pixel_hover.isChecked():
            xy = getattr(self.main_window, '_last_hover_xy', None)
            if xy is not None:
                try:
                    x, y = int(xy[0]), int(xy[1])
                    if 0 <= x < w and 0 <= y < h:
                        return y, x
                except Exception:
                    pass
        return h // 2, w // 2

    def _update_hkl_labels(self) -> None:
        if not hasattr(self, 'lbl_frame_H'):
            return

        def setv(H, K, L):
            self.lbl_frame_H.setText(H)
            self.lbl_frame_K.setText(K)
            self.lbl_frame_L.setText(L)

        if self._hkl_unavailable:
            setv("N/A", "N/A", "N/A")
            return
        grid = self._hkl_cache.get(self._current_frame())
        if grid is None:
            setv("…", "…", "…")
            return
        try:
            Hg, Kg, Lg = grid[0], grid[1], grid[2]
            h, w = int(Hg.shape[-2]), int(Hg.shape[-1])
            y, x = self._chosen_pixel(h, w)
            y = min(max(y, 0), h - 1)
            x = min(max(x, 0), w - 1)
            setv(f"{float(Hg[y, x]):.4f}", f"{float(Kg[y, x]):.4f}", f"{float(Lg[y, x]):.4f}")
        except Exception:
            setv("N/A", "N/A", "N/A")

    def refresh(self) -> None:
        """Refresh the displayed information based on the main window's 2D state."""
        mw = getattr(self, 'main_window', None)
        if mw is None:
            return
        # Try to keep mouse info consistent when refresh occurs
        try:
            xy = getattr(mw, '_last_hover_xy', None)
            frame = mw.get_current_frame_data() if hasattr(mw, 'get_current_frame_data') else None
            intensity = None
            H_val = K_val = L_val = None
            pos = None
            # Populate Mouse HKL even if no hover yet by falling back to center pixel
            if frame is not None and frame.ndim == 2:
                h, w = frame.shape
                # Validate hover position
                if xy is not None:
                    try:
                        x_hover, y_hover = int(xy[0]), int(xy[1])
                        if 0 <= x_hover < w and 0 <= y_hover < h:
                            x, y = x_hover, y_hover
                            pos = (x, y)
                        else:
                            x, y = w // 2, h // 2
                            pos = (x, y)
                    except Exception:
                        x, y = w // 2, h // 2
                        pos = (x, y)
                else:
                    x, y = w // 2, h // 2
                    pos = (x, y)
                # Intensity at chosen position
                try:
                    intensity = float(frame[y, x])
                except Exception:
                    intensity = None
                # HKL from cached q-grids if present
                try:
                    qxg = getattr(mw, '_qx_grid', None)
                    qyg = getattr(mw, '_qy_grid', None)
                    qzg = getattr(mw, '_qz_grid', None)
                    if qxg is not None and qyg is not None and qzg is not None:
                        if qxg.ndim == 3:
                            idx = int(mw.frame_spinbox.value()) if hasattr(mw, 'frame_spinbox') and mw.frame_spinbox.isEnabled() else 0
                            if 0 <= idx < qxg.shape[0]:
                                H_val = float(qxg[idx, y, x])
                                K_val = float(qyg[idx, y, x])
                                L_val = float(qzg[idx, y, x])
                        elif qxg.ndim == 2:
                            H_val = float(qxg[y, x])
                            K_val = float(qyg[y, x])
                            L_val = float(qzg[y, x])
                except Exception:
                    H_val = K_val = L_val = None
                # Update Mouse section in the dock
                self.set_mouse_info(pos, intensity, H_val, K_val, L_val)
        except Exception:
            pass
        # Points: show total points across data dimensions (include product, e.g., FxHxW = N)
        points_str = None
        low_val = None
        high_val = None
        try:
            data = getattr(mw, 'current_2d_data', None)
            if isinstance(data, np.ndarray):
                # total points
                total = int(data.size)
                points_str = f"{total:,}"
                # intensity low/high across dataset
                try:
                    low_val = float(np.min(data))
                except Exception:
                    low_val = None
                try:
                    high_val = float(np.max(data))
                except Exception:
                    high_val = None
        except Exception:
            points_str = None
        self.set_points(points_str)
        try:
            self.set_intensity(low_val, high_val)
        except Exception:
            pass
        # Axes: from WorkbenchWindow axis variables; annotate default source axes
        try:
            xlab = getattr(mw, 'axis_2d_x', None)
            ylab = getattr(mw, 'axis_2d_y', None)
            dx = xlab
            dy = ylab
            try:
                if isinstance(xlab, str) and xlab.strip().lower() in ("columns", "column"):
                    dx = f"{xlab}(Source)"
                if isinstance(ylab, str) and ylab.strip().lower() in ("row", "rows"):
                    dy = f"{ylab}(Source)"
            except Exception:
                pass
        except Exception:
            dx = None
            dy = None
        self.set_axes(dx, dy)
        # Per-frame HKL (RSMConverter, threaded) — schedule after other info is set
        self._schedule_hkl_update()
