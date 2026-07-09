"""Live beam-profile fitting dock for the area detector viewer.

On each plotting tick *for which a new detector frame has arrived*, this dock
extracts a rotatable rectangular ROI from the current (processed/masked) image,
sums the last N frames of that ROI (frame binning), collapses the binned 2D
region to a horizontal and a vertical 1D profile (SUM along each local axis),
and fits each profile with a selectable peak model (Gaussian / Lorentzian /
Laplacian). It reports FWHM, amplitude, position offset (0 = ROI center), and a
Poisson-weighted reduced ("regularized") chi-square.

Order of operations is **extract -> bin -> collapse**: only small ROI-sized
arrays are accumulated (not full frames), which is cheaper when the ROI is much
smaller than the frame; summing is linear, so this equals full-frame binning
while the ROI is stationary. Like the waterfall dock, the bin is reset whenever
the ROI geometry changes so binned subarrays always share one region/shape.

The ROI is a red rotatable rectangle (distinct from the waterfall's gold ROI)
with a red horizontal + blue vertical crosshair showing which axis maps to which
colored 1D profile. Pixel extraction is delegated to the shared, transform-aware
``dashpva.utils.roi_ops._extract_roi_subarray``; fitting to the edition-agnostic
``dashpva.utils.peak_fit.fit_profile`` run off the UI thread in ``BeamFitWorker``.
"""

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsLineItem,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dashpva.gui.theme_colors import ROI_COLORS, status_style
from dashpva.utils.peak_fit import MODELS, fit_profile
from dashpva.utils.roi_ops import _extract_roi_subarray
from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "other"

_DEFAULT_BIN = 5
_MIN_BIN = 1
_MAX_BIN = 50

# Red = horizontal (local-x) profile; Blue = vertical (local-y) profile.
_RED = ROI_COLORS[0]
_BLUE = ROI_COLORS[1]

# Sentinel for "no frame processed yet" (distinct from 0 and None).
_UNSET = object()

# Monospace, fixed-width readout so the numbers don't shift as values change.
_BLANK_READOUT = ("FWHM      —  px   pos      —  px\n"
                  "amp       —       χ²ᵣ    —    R²    —")


class BeamFitWorker(QThread):
    """Fit both 1D profiles off the UI thread and emit one result."""

    # frame_id, PeakFit(x/red), PeakFit(y/blue), elapsed_s, error
    done = pyqtSignal(int, object, object, float, str)

    def __init__(self, x_red, y_red, x_blue, y_blue, model, frame_id, parent=None):
        super().__init__(parent)
        self._x_red = x_red
        self._y_red = y_red
        self._x_blue = x_blue
        self._y_blue = y_blue
        self._model = model
        self._frame_id = frame_id

    def run(self):
        t0 = time.perf_counter()
        try:
            fit_r = fit_profile(self._x_red, self._y_red, self._model)
            fit_b = fit_profile(self._x_blue, self._y_blue, self._model)
            self.done.emit(self._frame_id, fit_r, fit_b, time.perf_counter() - t0, "")
        except Exception as e:  # defense in depth; fit_profile swallows its own errors
            self.done.emit(self._frame_id, None, None, time.perf_counter() - t0, str(e))


class BeamFitDock(BaseDock):
    """Dockable live ROI beam-profile fitter (H/V FWHM, position, reduced chi-square)."""

    def __init__(self, main_window=None, show: bool = False):
        super().__init__(title="Beam Profiler", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        # Rolling bin of ROI subarrays (small) + new-frame guard.
        self._bin: deque[np.ndarray] = deque(maxlen=_DEFAULT_BIN)
        self._last_frame_id = _UNSET
        self._manual_roi = None
        self._roi_geom = None  # (pos, size, angle) kept across tab hide/show
        self._guide_x = None
        self._guide_y = None
        # Single-flight fit state.
        self._busy = False
        self._pending = None
        self._worker = None
        # Soft-PV broadcast of fit results (server created on demand).
        self._pva_server = None
        self._pv_obj = None
        self._pv_channel = None
        self._build()
        try:
            self.main_window.timer_plot.timeout.connect(self._on_tick)
        except Exception:
            pass
        self.visibilityChanged.connect(self._on_visibility_changed)

    # ------------------------------------------------------------------ build
    def _build(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        controls = QFormLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.addItems([m.capitalize() for m in MODELS])
        self.model_combo.setToolTip("Peak model fit to each 1D profile")
        controls.addRow(QLabel("Model:"), self.model_combo)

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(_MIN_BIN, _MAX_BIN)
        self.bin_spin.setValue(_DEFAULT_BIN)
        self.bin_spin.setToolTip("Number of frames summed before fitting")
        self.bin_spin.valueChanged.connect(self._on_bin_changed)
        controls.addRow(QLabel("Frame bin:"), self.bin_spin)

        self.chk_autofit = QCheckBox("Auto-fit")
        self.chk_autofit.setChecked(True)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear)
        controls.addRow(self.chk_autofit, self.btn_clear)

        self.chk_broadcast = QCheckBox("Broadcast fit as PVs")
        self.chk_broadcast.setToolTip(
            "Serve the fit results as a single soft PVA structure")
        self.chk_broadcast.toggled.connect(self._on_broadcast_toggled)
        controls.addRow(self.chk_broadcast)

        outer.addLayout(controls)

        # Horizontal (red) profile plot + readout.
        self.lbl_x = QLabel("Horizontal averaging · red")
        self.lbl_x.setStyleSheet(status_style(_RED, bold=True))
        outer.addWidget(self.lbl_x)
        self.val_x = self._make_value_label()
        outer.addWidget(self.val_x)
        self.plot_x, self.curve_x_data, self.curve_x_fit, self.marker_x = \
            self._make_profile_plot(_RED, "offset [px]")
        outer.addWidget(self.plot_x, stretch=1)

        # Vertical (blue) profile plot + readout.
        self.lbl_y = QLabel("Vertical averaging · blue")
        self.lbl_y.setStyleSheet(status_style(_BLUE, bold=True))
        outer.addWidget(self.lbl_y)
        self.val_y = self._make_value_label()
        outer.addWidget(self.val_y)
        self.plot_y, self.curve_y_data, self.curve_y_fit, self.marker_y = \
            self._make_profile_plot(_BLUE, "offset [px]")
        outer.addWidget(self.plot_y, stretch=1)

        self.lbl_status = QLabel("Draw the red box over the beam.")
        self.lbl_status.setStyleSheet(status_style("#7A8394"))
        outer.addWidget(self.lbl_status)

        self.setWidget(container)

    def _make_profile_plot(self, color, xlabel):
        plot = pg.PlotWidget()
        plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot.setMinimumHeight(130)
        plot.getAxis("bottom").setLabel(text=xlabel)
        plot.getAxis("left").setLabel(text="Σ counts")
        plot.showGrid(x=True, y=True, alpha=0.2)
        data = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4,
                               symbolBrush=color, symbolPen=None)
        fit = pg.PlotDataItem(pen=pg.mkPen(color, width=2))
        marker = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen(color, width=1, style=Qt.DashLine))
        marker.hide()
        plot.addItem(data)
        plot.addItem(fit)
        plot.addItem(marker)
        return plot, data, fit, marker

    @staticmethod
    def _make_value_label() -> QLabel:
        lbl = QLabel(_BLANK_READOUT)
        lbl.setStyleSheet(
            "font-family: 'Consolas','Courier New',monospace; font-size: 9pt;")
        return lbl

    @staticmethod
    def _format_readout(fit) -> str:
        # Fixed-width fields (monospace) keep columns stable as values change.
        return (f"FWHM {fit.fwhm:8.2f} px   pos {fit.center:+8.2f} px\n"
                f"amp  {fit.amplitude:10.3e}   χ²ᵣ {fit.redchi:6.2f}   R² {fit.r_squared:6.3f}")

    # ---------------------------------------------------------------- helpers
    def _current_model(self) -> str:
        return self.model_combo.currentText().lower()

    @staticmethod
    def _centered_axis(n: int) -> np.ndarray:
        """Pixel axis shifted so the ROI center maps to 0."""
        return np.arange(n, dtype=float) - (n - 1) / 2.0

    def _image_item(self):
        view = getattr(self.main_window, "image_view", None)
        if view is None:
            return None
        try:
            return view.getImageItem()
        except Exception:
            return None

    def _current_frame_id(self):
        reader = getattr(self.main_window, "reader", None)
        if reader is None:
            return None
        return getattr(reader, "frames_received", None)

    def _on_bin_changed(self, value: int) -> None:
        self._bin = deque(self._bin, maxlen=int(value))

    def _on_clear(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._bin.clear()
        self._last_frame_id = _UNSET
        self._pending = None
        for curve in (self.curve_x_data, self.curve_x_fit,
                      self.curve_y_data, self.curve_y_fit):
            curve.clear()
        self.marker_x.hide()
        self.marker_y.hide()
        self.val_x.setText(_BLANK_READOUT)
        self.val_y.setText(_BLANK_READOUT)

    # ------------------------------------------------------------- manual ROI
    def _ensure_manual_roi(self) -> None:
        if self._manual_roi is not None:
            return
        image_item = self._image_item()
        if image_item is None or image_item.image is None:
            return
        if self._roi_geom is not None:
            pos, size, angle = self._roi_geom
        else:
            shape = image_item.image.shape
            iw = int(shape[1]) if len(shape) >= 2 else 100
            ih = int(shape[0]) if len(shape) >= 2 else 100
            w = max(4, iw // 4)
            h = max(4, ih // 4)
            pos, size, angle = [(iw - w) // 2, (ih - h) // 2], [w, h], 0.0
        roi = pg.ROI(pos, size, angle=angle, pen=pg.mkPen(_RED, width=2),
                     movable=True, rotatable=True)
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addRotateHandle([1, 0], [0.5, 0.5])
        roi.sigRegionChanged.connect(self._on_roi_changed)
        roi.sigRegionChangeFinished.connect(self._on_roi_released)
        self.main_window.image_view.getView().addItem(roi)
        self._manual_roi = roi
        self._add_axis_guides(roi)
        self._update_axis_guides()

    def _add_axis_guides(self, roi) -> None:
        """Red horizontal + blue vertical crosshair (children of the ROI, so they
        follow its move/rotation) that map each axis to its profile color."""
        self._guide_x = self._make_guide(roi, _RED)
        self._guide_y = self._make_guide(roi, _BLUE)

    @staticmethod
    def _make_guide(roi, color) -> QGraphicsLineItem:
        item = QGraphicsLineItem(roi)
        pen = pg.mkPen(color, width=2)
        pen.setCosmetic(True)  # constant on-screen width regardless of zoom
        item.setPen(pen)
        item.setAcceptedMouseButtons(Qt.NoButton)  # never steal ROI drags
        item.setZValue(20)
        return item

    def _update_axis_guides(self) -> None:
        roi = self._manual_roi
        if roi is None or self._guide_x is None:
            return
        size = roi.size()
        sx, sy = float(size[0]), float(size[1])
        self._guide_x.setLine(0.0, sy / 2.0, sx, sy / 2.0)  # along local-x (red)
        self._guide_y.setLine(sx / 2.0, 0.0, sx / 2.0, sy)  # along local-y (blue)

    def _remove_manual_roi(self) -> None:
        if self._manual_roi is None:
            return
        # Remember geometry so the box returns unchanged when the tab is shown
        # again (visibilityChanged tears the ROI down on every tab switch).
        self._roi_geom = (self._manual_roi.pos(), self._manual_roi.size(),
                          self._manual_roi.angle())
        try:
            self.main_window.image_view.getView().removeItem(self._manual_roi)
        except Exception:
            pass
        # Guides are children of the ROI, destroyed with it.
        self._guide_x = None
        self._guide_y = None
        self._manual_roi = None

    def _on_roi_changed(self, *_args) -> None:
        # Geometry changed -> stale binned subarrays; drop them and refresh guides.
        self._bin.clear()
        self._last_frame_id = _UNSET
        self._update_axis_guides()

    def _on_roi_released(self, *_args) -> None:
        # Mouse released after a move/resize: fit the current frame once, even if
        # acquisition is paused (no new frames arrive to drive _on_tick).
        self._last_frame_id = _UNSET
        self._on_tick()

    # --------------------------------------------------------------- lifecycle
    def _on_visibility_changed(self, visible: bool) -> None:
        if visible:
            self._ensure_manual_roi()
            # Refresh the profiles/fit for the restored ROI on the current frame.
            self._last_frame_id = _UNSET
            self._on_tick()
        else:
            self._remove_manual_roi()

    def on_channel_changed(self) -> None:
        """Called by the host when the PVA channel switches."""
        self._remove_manual_roi()
        self._roi_geom = None  # new image size -> re-center a fresh ROI
        self._reset()
        # Re-point the broadcast at the new image channel if it's active.
        if self.chk_broadcast.isChecked():
            self._start_broadcast()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(500)
        self._stop_broadcast()
        super().closeEvent(event)

    # --------------------------------------------------------------- broadcast
    def _broadcast_channel(self) -> str:
        base = getattr(self.main_window, "_input_channel", "") or "pvapy:image"
        return f"{base}:BeamProfiler"

    def _on_broadcast_toggled(self, checked: bool) -> None:
        if checked:
            if self._start_broadcast():
                self._show_broadcast_dialog()
            else:  # start failed -> revert the checkbox without re-triggering
                self.chk_broadcast.blockSignals(True)
                self.chk_broadcast.setChecked(False)
                self.chk_broadcast.blockSignals(False)
        else:
            self._stop_broadcast()
            self.lbl_status.setText("PV broadcast stopped.")

    def _start_broadcast(self) -> bool:
        try:
            import pvaccess as pva
        except Exception as e:
            self.lbl_status.setText(f"PVA unavailable: {e}")
            return False
        self._stop_broadcast()
        struct = {
            'model': pva.STRING,
            'fwhm_h': pva.DOUBLE, 'fwhm_v': pva.DOUBLE,
            'amplitude_h': pva.DOUBLE, 'amplitude_v': pva.DOUBLE,
            'position_h': pva.DOUBLE, 'position_v': pva.DOUBLE,
            'redchi_h': pva.DOUBLE, 'redchi_v': pva.DOUBLE,
            'r2_h': pva.DOUBLE, 'r2_v': pva.DOUBLE,
            'frameId': pva.INT,
            'timeStamp': {'secondsPastEpoch': pva.LONG, 'nanoseconds': pva.INT},
        }
        try:
            self._pv_channel = self._broadcast_channel()
            self._pv_obj = pva.PvObject(struct)
            self._pva_server = pva.PvaServer()
            self._pva_server.addRecord(self._pv_channel, self._pv_obj, None)
            self._pva_server.start()
        except Exception as e:
            self.lbl_status.setText(f"broadcast failed: {e}")
            self._stop_broadcast()
            return False
        self.lbl_status.setText(f"broadcasting {self._pv_channel}")
        return True

    def _stop_broadcast(self) -> None:
        if self._pva_server is not None:
            try:
                self._pva_server.stop()
            except Exception:
                pass
        self._pva_server = None
        self._pv_obj = None
        self._pv_channel = None

    def _publish_fit(self, fit_r, fit_b, frame_id) -> None:
        if self._pva_server is None or self._pv_obj is None:
            return

        def v(fit, attr):
            return float(getattr(fit, attr)) if fit is not None else float("nan")

        obj = self._pv_obj
        obj['model'] = self._current_model()
        obj['fwhm_h'] = v(fit_r, 'fwhm')
        obj['fwhm_v'] = v(fit_b, 'fwhm')
        obj['amplitude_h'] = v(fit_r, 'amplitude')
        obj['amplitude_v'] = v(fit_b, 'amplitude')
        obj['position_h'] = v(fit_r, 'center')
        obj['position_v'] = v(fit_b, 'center')
        obj['redchi_h'] = v(fit_r, 'redchi')
        obj['redchi_v'] = v(fit_b, 'redchi')
        obj['r2_h'] = v(fit_r, 'r_squared')
        obj['r2_v'] = v(fit_b, 'r_squared')
        obj['frameId'] = int(frame_id) if frame_id is not None else 0
        obj['timeStamp'] = {'secondsPastEpoch': int(time.time()), 'nanoseconds': 0}
        try:
            self._pva_server.update(self._pv_channel, obj)
        except Exception:
            pass

    def _show_broadcast_dialog(self) -> None:
        ch = self._pv_channel
        QMessageBox.information(
            self, "Beam Profiler — broadcasting fit PVs",
            "Fit results are served as a single soft PVA structure:\n\n"
            f"    {ch}\n\n"
            "Fields (h = horizontal/red, v = vertical/blue):\n"
            "    model, fwhm_h, fwhm_v, amplitude_h, amplitude_v,\n"
            "    position_h, position_v, redchi_h, redchi_v,\n"
            "    r2_h, r2_v, frameId, timeStamp\n\n"
            "Listen with:\n"
            f"    pvget {ch}\n"
            f"    pvmonitor {ch}\n\n"
            "Updates once per fit while Auto-fit is on.")

    # ------------------------------------------------------------------ update
    def _on_tick(self) -> None:
        if not self.isVisible():
            return
        image_item = self._image_item()
        if image_item is None or image_item.image is None:
            return
        # Draw the ROI once a frame is displayed (deferred if dock opened first).
        if self._manual_roi is None:
            self._ensure_manual_roi()
            if self._manual_roi is None:
                return
        frame_id = self._current_frame_id()
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id

        # Extract ROI from the current frame, then bin small subarrays.
        frame = np.asarray(image_item.image)
        sub = _extract_roi_subarray(frame, self._manual_roi, image_item)
        if sub is None or sub.ndim != 2 or sub.size == 0:
            return
        if self._bin and sub.shape != self._bin[0].shape:
            self._bin.clear()
        self._bin.append(np.array(sub, dtype=np.float32))

        binned = np.nansum(np.stack(self._bin), axis=0)
        profile_x = np.nansum(binned, axis=0)  # vs local-x (RED)
        profile_y = np.nansum(binned, axis=1)  # vs local-y (BLUE)
        x_red = self._centered_axis(profile_x.size)
        x_blue = self._centered_axis(profile_y.size)

        # Always refresh the data curves (cheap, UI thread).
        self.curve_x_data.setData(x_red, profile_x)
        self.curve_y_data.setData(x_blue, profile_y)

        if not self.chk_autofit.isChecked():
            return
        payload = (x_red, profile_x, x_blue, profile_y, self._current_model(), frame_id)
        if self._busy:
            self._pending = payload  # keep only the latest
        else:
            self._start_fit(payload)

    def _start_fit(self, payload) -> None:
        # Ensure the previous worker has fully stopped before we drop its
        # reference. Rapid ROI moves dispatch fits back-to-back; without this the
        # old QThread can be garbage-collected while still finishing, aborting
        # with "QThread: Destroyed while thread is still running".
        prev = self._worker
        if prev is not None and prev.isRunning():
            prev.wait(2000)
        self._busy = True
        self._pending = None
        self._worker = BeamFitWorker(*payload)
        self._worker.done.connect(self._on_fit_done)
        self._worker.start()

    def _on_fit_done(self, frame_id, fit_r, fit_b, elapsed, error) -> None:
        self._busy = False
        if error:
            self.lbl_status.setText(f"fit error: {error}")
        else:
            self._render_fit(self.curve_x_fit, self.marker_x, self.val_x, fit_r)
            self._render_fit(self.curve_y_fit, self.marker_y, self.val_y, fit_b)
            self.lbl_status.setText(f"frame {frame_id} · fit {elapsed * 1000:.1f} ms "
                                    f"· bin {len(self._bin)}")
            self._publish_fit(fit_r, fit_b, frame_id)
        if self._pending is not None:
            self._start_fit(self._pending)

    def _render_fit(self, curve_fit, marker, val_lbl, fit) -> None:
        if fit is None or not fit.success or fit.y_fit.size == 0:
            curve_fit.clear()
            marker.hide()
            val_lbl.setText(_BLANK_READOUT)
            return
        x = self._centered_axis(fit.y_fit.size)
        curve_fit.setData(x, fit.y_fit)
        marker.setValue(fit.center)
        marker.show()
        val_lbl.setText(self._format_readout(fit))
