"""Live waterfall-plot dock for the area detector viewer.

On each plotting tick *for which a new detector frame has arrived*, the dock
averages a rectangular region of the current (processed/masked) image — exactly
as shown in the viewer — down to a 1D profile and stacks those profiles over time
into a scrolling waterfall (frame/time on the vertical axis, ROI position on the
horizontal axis). The new-frame guard means the stack does not keep growing while
acquisition is idle even though the plot timer keeps firing.

The averaged region comes from one of two sources, selected in the dock:
  * an EPICS detector ROI (ROI1..ROI4) when those PVs are available, reusing the
    on-image overlay the host already positions and keeps transform-correct;
  * a "Manual ROI" -- an interactive pyqtgraph rectangle the user can move,
    resize, and *rotate* (detector ROIs cannot rotate). The manual rectangle is
    drawn on the image only while the dock is visible and Manual is selected,
    and is removed on dock hide, source change, or channel change so it never
    lingers.

Pixel extraction is delegated to the shared, transform-aware helper
``dashpva.utils.roi_ops._extract_roi_subarray`` (pyqtgraph ``getArrayRegion``
with a pixel-slice fallback), the same routine the workbench ROI tools use.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainterPath, QStandardItemModel
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsPathItem,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.colormap import get as get_colormap

from dashpva.utils.roi_ops import _extract_roi_subarray
from dashpva.viewer.core.docks.base_dock import BaseDock

_SEGMENT = "other"

# Sentinel combo entry for the user-drawn rectangle.
_MANUAL = "Manual ROI"

# Ring-buffer depth (number of stacked 1D profiles). Mirrors pyFAI's waterfall.
_DEFAULT_DEPTH = 300
_MIN_DEPTH = 10
_MAX_DEPTH = 10000

# Sentinel for "no frame stacked yet" — distinct from any real frame counter
# (including 0 and None) so the first tick always appends.
_UNSET = object()


class WaterfallDock(BaseDock):
    """Dockable live waterfall of an ROI-averaged 1D profile over time."""

    def __init__(self, main_window=None, show: bool = False):
        super().__init__(title="Waterfall", main_window=main_window,
                         segment_name=_SEGMENT, dock_area=Qt.RightDockWidgetArea,
                         show=show)
        # Ring buffer of 1D profiles plus the current profile length so we can
        # detect when the ROI/averaging change resizes the row and reset.
        self._buffer: deque[np.ndarray] = deque(maxlen=_DEFAULT_DEPTH)
        self._row_len = None
        # Frame counter of the most recently stacked row. The plot timer can fire
        # faster than frames arrive, so we only append when this changes — this
        # stops the waterfall from piling duplicate rows when no new frame is in.
        self._last_frame_id = _UNSET
        # The interactive manual rectangle (created lazily, removed on teardown)
        # plus its child arrow that shows the averaging (collapse) direction.
        self._manual_roi = None
        self._roi_geom = None  # (pos, size, angle) kept across tab hide/show
        self._avg_arrow = None
        self._build()
        # Drive updates from the host's plot timer (same cadence as the image),
        # but _on_tick only stacks a row when a new frame has arrived since the
        # last one, so a fast plot timer can't pile duplicate rows when no new
        # frame is coming in.
        try:
            self.main_window.timer_plot.timeout.connect(self._on_tick)
        except Exception:
            pass
        self.visibilityChanged.connect(self._on_visibility_changed)
        self.refresh_roi_sources()

    # ------------------------------------------------------------------ build
    def _build(self):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        controls = QFormLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)

        self.roi_combo = QComboBox()
        self.roi_combo.currentIndexChanged.connect(self._on_source_changed)
        controls.addRow(QLabel("ROI source:"), self.roi_combo)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Horizontal", "Vertical"])
        self.direction_combo.setToolTip(
            "Direction to average the 2D ROI into a 1D profile")
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        controls.addRow(QLabel("Average:"), self.direction_combo)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(_MIN_DEPTH, _MAX_DEPTH)
        self.depth_spin.setValue(_DEFAULT_DEPTH)
        self.depth_spin.setToolTip("Number of stacked frames kept in the waterfall")
        self.depth_spin.valueChanged.connect(self._on_depth_changed)
        controls.addRow(QLabel("Max frames:"), self.depth_spin)

        self.chk_log = QCheckBox("Log intensity")
        self.chk_log.setChecked(False)
        self.chk_autoscale = QCheckBox("Autoscale")
        self.chk_autoscale.setChecked(True)
        controls.addRow(self.chk_log, self.chk_autoscale)

        outer.addLayout(controls)

        # Waterfall view: pyqtgraph ImageItem in a PlotWidget (row-major so the
        # array indexes as [frame, position] == [y, x]).
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setMinimumHeight(160)
        self.plot_widget.getAxis('bottom').setLabel(text='ROI position [px]')
        self.plot_widget.getAxis('left').setLabel(text='Frame (time →)')
        self.waterfall_img = pg.ImageItem()
        self.waterfall_img.setOpts(axisOrder='row-major')
        try:
            self.waterfall_img.setLookupTable(get_colormap('viridis').getLookupTable())
        except Exception:
            pass
        self.plot_widget.addItem(self.waterfall_img)
        outer.addWidget(self.plot_widget, stretch=1)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear)
        outer.addWidget(self.btn_clear)

        self.setWidget(container)

    # -------------------------------------------------------------- ROI source
    def refresh_roi_sources(self) -> None:
        """Rebuild the source dropdown from currently available EPICS ROIs.

        Always offers "Manual ROI". EPICS ROIs are listed and enabled when the
        reader exposes them; otherwise a single disabled, tooltip'd placeholder
        is shown so the user understands why none can be picked.
        """
        prev = self.roi_combo.currentText() if self.roi_combo.count() else _MANUAL
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItem(_MANUAL)

        rois = self._available_roi_names()
        if rois:
            for name in rois:
                self.roi_combo.addItem(name)
        else:
            self.roi_combo.addItem("EPICS ROIs (not available)")
            self._disable_combo_item(
                self.roi_combo.count() - 1,
                "No EPICS ROIs available on this channel")

        idx = self.roi_combo.findText(prev)
        self.roi_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.roi_combo.blockSignals(False)
        # Apply the (possibly unchanged) selection so overlays/state stay correct.
        self._on_source_changed()

    def _available_roi_names(self) -> list:
        reader = getattr(self.main_window, "reader", None)
        rois = getattr(reader, "rois", None) if reader is not None else None
        if not rois:
            return []
        # Keep a stable, human order (ROI1, ROI2, ...).
        return sorted(rois.keys())

    def _disable_combo_item(self, index: int, tooltip: str) -> None:
        model = self.roi_combo.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(index)
            if item is not None:
                item.setEnabled(False)
                item.setToolTip(tooltip)

    # ------------------------------------------------------------- selection
    def _selected_source(self) -> str:
        return self.roi_combo.currentText()

    def _on_source_changed(self, *_args) -> None:
        if self._selected_source() == _MANUAL and self.isVisible():
            self._ensure_manual_roi()
        else:
            # EPICS ROI selected (detector ROI overrides) or nothing selectable.
            self._remove_manual_roi()
        self._reset_buffer()

    def _on_direction_changed(self, *_args) -> None:
        self._reset_buffer()
        self._update_avg_indicator()

    def _on_depth_changed(self, value: int) -> None:
        # Preserve the most recent rows when resizing the ring buffer.
        self._buffer = deque(self._buffer, maxlen=int(value))

    def _on_clear(self) -> None:
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self._buffer.clear()
        self._row_len = None
        self._last_frame_id = _UNSET
        self.waterfall_img.clear()

    # ------------------------------------------------------------- manual ROI
    def _ensure_manual_roi(self) -> None:
        if self._manual_roi is not None:
            return
        image_item = self._image_item()
        # Only draw the manual rectangle once a frame is actually displayed.
        # Otherwise (e.g. the dock restored visible from saved state before the
        # user clicks Start Live View) it would float, hugely magnified, in an
        # empty view. _on_tick re-attempts this on the first streamed frame.
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
        roi = pg.ROI(pos, size, angle=angle, pen=pg.mkPen((255, 215, 0), width=2),
                     movable=True, rotatable=True)
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addRotateHandle([1, 0], [0.5, 0.5])
        roi.sigRegionChanged.connect(self._on_manual_roi_changed)
        self.main_window.image_view.getView().addItem(roi)
        self._manual_roi = roi
        self._update_avg_indicator()

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
        # The arrow is a child of the ROI, so it is destroyed with it.
        self._avg_arrow = None
        self._manual_roi = None

    def _on_manual_roi_changed(self, *_args) -> None:
        # Resizing changes the profile length; drop stale rows so the stack stays
        # rectangular. _append_row also guards this, but clearing keeps it clean.
        self._reset_buffer()
        self._update_avg_indicator()

    def _update_avg_indicator(self) -> None:
        """Draw/refresh a double-headed arrow on the manual ROI showing the axis
        the 1D profile spans (the kept axis, perpendicular to the collapsed one).

        The arrow is a child of the ROI, so it follows the rectangle's position
        and rotation automatically. "Horizontal" collapses the ROI's local-y
        (height / rows, axis 0), leaving a profile along local-x -> arrow points
        along width; "Vertical" collapses local-x (width / cols, axis 1), leaving
        a profile along local-y -> arrow points along height.
        """
        roi = self._manual_roi
        if roi is None:
            return
        if self._avg_arrow is None:
            self._avg_arrow = QGraphicsPathItem(roi)
            pen = pg.mkPen((0, 220, 255), width=2)
            pen.setCosmetic(True)  # constant on-screen width regardless of zoom
            self._avg_arrow.setPen(pen)
            # Purely decorative: never intercept mouse so ROI drag still works.
            self._avg_arrow.setAcceptedMouseButtons(Qt.NoButton)
            self._avg_arrow.setZValue(20)
        size = roi.size()
        horizontal = self.direction_combo.currentIndex() == 0
        self._avg_arrow.setPath(
            self._arrow_path(float(size[0]), float(size[1]), horizontal))

    @staticmethod
    def _arrow_path(sx: float, sy: float, horizontal: bool) -> QPainterPath:
        """Double-headed arrow in ROI-local coordinates (0..sx, 0..sy)."""
        path = QPainterPath()
        if sx <= 0 or sy <= 0:
            return path

        def seg(x1, y1, x2, y2):
            path.moveTo(x1, y1)
            path.lineTo(x2, y2)

        if horizontal:
            cy = sy / 2.0
            a, b = 0.15 * sx, 0.85 * sx
            head = max(1.0, min(0.12 * (b - a), 0.35 * sy))
            seg(a, cy, b, cy)                  # shaft
            seg(a, cy, a + head, cy - head)    # left head
            seg(a, cy, a + head, cy + head)
            seg(b, cy, b - head, cy - head)    # right head
            seg(b, cy, b - head, cy + head)
        else:
            cx = sx / 2.0
            a, b = 0.15 * sy, 0.85 * sy
            head = max(1.0, min(0.12 * (b - a), 0.35 * sx))
            seg(cx, a, cx, b)                  # shaft
            seg(cx, a, cx - head, a + head)    # top head
            seg(cx, a, cx + head, a + head)
            seg(cx, b, cx - head, b - head)    # bottom head
            seg(cx, b, cx + head, b - head)
        return path

    # --------------------------------------------------------------- lifecycle
    def _on_visibility_changed(self, visible: bool) -> None:
        if visible:
            if self._selected_source() == _MANUAL:
                self._ensure_manual_roi()
        else:
            self._remove_manual_roi()

    def on_channel_changed(self) -> None:
        """Called by the host when the PVA channel switches.

        Tears down the manual overlay and clears the buffer, then refreshes the
        EPICS ROI list (availability may have changed with the new channel).
        """
        self._remove_manual_roi()
        self._roi_geom = None  # new image size -> re-center a fresh ROI
        self._reset_buffer()
        self.refresh_roi_sources()

    # ------------------------------------------------------------------ update
    def _image_item(self):
        view = getattr(self.main_window, "image_view", None)
        if view is None:
            return None
        try:
            return view.getImageItem()
        except Exception:
            return None

    def _active_roi(self):
        """Return the pyqtgraph ROI object for the current source, or None."""
        source = self._selected_source()
        if source == _MANUAL:
            return self._manual_roi
        overlays = getattr(self.main_window, "_roi_overlays", None)
        if overlays:
            return overlays.get(source)
        return None

    def _on_tick(self) -> None:
        # Cheap no-op when hidden or not yet streaming.
        if not self.isVisible():
            return
        image_item = self._image_item()
        if image_item is None or image_item.image is None:
            return
        # A frame is now displayed: draw the manual rectangle if it was deferred
        # because the dock opened before streaming began.
        if self._selected_source() == _MANUAL and self._manual_roi is None:
            self._ensure_manual_roi()
        # Only stack a row when a genuinely new frame has arrived. The plot timer
        # may fire faster than frames stream in (or keep firing while acquisition
        # is stopped); without this the waterfall would pile copies of the same
        # frame. When frames arrive faster than the plot rate we just sample the
        # latest one at the plot rate.
        frame_id = self._current_frame_id()
        if frame_id == self._last_frame_id:
            return
        roi = self._active_roi()
        if roi is None:
            return
        # Use the processed/masked image exactly as shown in the viewer, so the
        # waterfall matches the display (no separate re-processing per frame).
        frame = np.asarray(image_item.image)
        sub = _extract_roi_subarray(frame, roi, image_item)
        if sub is None or sub.ndim != 2 or sub.size == 0:
            return
        # Horizontal -> collapse rows (axis 0); Vertical -> collapse cols (axis 1).
        axis = 0 if self.direction_combo.currentIndex() == 0 else 1
        row = np.nanmean(sub, axis=axis)
        if row.size == 0:
            return
        self._last_frame_id = frame_id
        self._append_row(np.asarray(row, dtype=np.float32))

    def _current_frame_id(self):
        """Monotonic count of frames received, used to detect a new frame.

        Returns None when no reader is connected; combined with the ``_UNSET``
        sentinel this still appends once for a static/test image but then stops.
        """
        reader = getattr(self.main_window, "reader", None)
        if reader is None:
            return None
        return getattr(reader, "frames_received", None)

    def _append_row(self, row: np.ndarray) -> None:
        if self._row_len is None:
            self._row_len = int(row.size)
        elif int(row.size) != self._row_len:
            # ROI resized/rotated to a new output length -> restart the stack.
            self._buffer.clear()
            self._row_len = int(row.size)
        self._buffer.append(row)
        self._render()

    def _render(self) -> None:
        if not self._buffer:
            return
        stack = np.vstack(self._buffer)  # shape (n_frames, n_bins) == (y, x)
        if self.chk_log.isChecked():
            stack = np.log10(np.clip(stack, 1e-10, None))
        autoscale = self.chk_autoscale.isChecked()
        self.waterfall_img.setImage(stack, autoLevels=autoscale)
        if autoscale:
            # Re-assert auto-range every render so the view keeps fitting as the
            # buffer fills (30 -> max frames) and so an accidental zoom or the
            # right-click menu can't permanently freeze auto-scaling. Uncheck
            # Autoscale to pan/zoom freely.
            self.plot_widget.getViewBox().enableAutoRange(enable=True)
