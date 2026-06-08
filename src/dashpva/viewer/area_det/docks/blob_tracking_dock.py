from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.viewer.core.docks.base_dock import BaseDock


class BlobTrackingDock(BaseDock):
    """Side panel for real-time blob detection and SORT tracking controls."""

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Blob Tracking", main_window=main_window,
                         segment_name="other", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._build()
        self._load_from_config()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── Enable / overlay row ──────────────────────────────────────────
        top_row = QHBoxLayout()
        self.chk_enable_blob = QCheckBox("Enable")
        self.chk_enable_blob.setToolTip(
            "When checked, blob detection and SORT tracking run on every incoming frame.")
        self.chk_show_overlay = QCheckBox("Show overlay")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.setToolTip(
            "Draw bounding-box and track-ID overlays on the live image.")
        top_row.addWidget(self.chk_enable_blob)
        top_row.addWidget(self.chk_show_overlay)
        top_row.addStretch()
        outer.addLayout(top_row)

        # ── Status label ─────────────────────────────────────────────────
        self.lbl_blob_status = QLabel("—")
        self.lbl_blob_status.setAlignment(Qt.AlignCenter)
        outer.addWidget(self.lbl_blob_status)

        # ── Detector parameters ───────────────────────────────────────────
        det_box = QGroupBox("SimpleBlobDetector params")
        det_layout = QFormLayout(det_box)
        det_layout.setHorizontalSpacing(8)
        det_layout.setVerticalSpacing(3)
        det_layout.setContentsMargins(6, 6, 6, 6)

        self.sb_min_threshold = self._dspin(1, 65534, 10)
        self.sb_min_threshold.setToolTip(
            "Pixel intensity at which binarisation starts.")
        self.sb_max_threshold = self._dspin(2, 65535, 200)
        self.sb_max_threshold.setToolTip(
            "Pixel intensity at which binarisation stops. Set to ~= peak blob intensity.")
        self.sb_threshold_step = self._dspin(1, 65535, 10, step=5, decimals=0)
        self.sb_threshold_step.setToolTip(
            "Step between threshold levels. Smaller = finer detection but slower.")
        det_layout.addRow("Min threshold:", self.sb_min_threshold)
        det_layout.addRow("Max threshold:", self.sb_max_threshold)
        det_layout.addRow("Threshold step:", self.sb_threshold_step)

        # filterByColor — CRITICAL for bright vs dark blobs.
        # cv2 default is blobColor=0 (dark blobs). For bright peaks on dark
        # background (e.g. diffraction spots), set blobColor=255.
        self.chk_filter_color = QCheckBox("Filter by color")
        self.chk_filter_color.setChecked(True)
        self.chk_filter_color.setToolTip(
            "When checked, only detect blobs of the chosen color (bright/dark).")
        from PyQt5.QtWidgets import QComboBox
        self.cb_blob_color = QComboBox()
        self.cb_blob_color.addItem("Bright (255)", 255)
        self.cb_blob_color.addItem("Dark (0)", 0)
        self.cb_blob_color.setCurrentIndex(0)   # default: bright blobs
        self.cb_blob_color.setToolTip(
            "Bright: detect light spots on dark background (diffraction peaks).\n"
            "Dark: detect dark spots on bright background.")
        color_row = QHBoxLayout()
        color_row.addWidget(self.chk_filter_color)
        color_row.addWidget(self.cb_blob_color)
        det_layout.addRow(color_row)

        # minDistBetweenBlobs merges detections closer than this distance.
        # Too large → nearby real blobs merge into one.
        # Too small → one physical blob gets detected multiple times at different threshold levels.
        # Rule of thumb: set to roughly 1–1.5× the blob radius.
        self.sb_min_dist = self._dspin(1, 4096, 20, step=5, decimals=0)
        self.sb_min_dist.setToolTip(
            "Merge detections closer than this distance (pixels). "
            "~1× blob radius prevents duplicates without merging nearby real blobs.")
        det_layout.addRow("Min dist between blobs:", self.sb_min_dist)

        self.sb_min_repeatability = QSpinBox()
        self.sb_min_repeatability.setRange(1, 20)
        self.sb_min_repeatability.setValue(2)
        self.sb_min_repeatability.setToolTip(
            "Threshold levels a blob must appear at to be reported. "
            "Lower = more detections; 2 is a good starting point.")
        det_layout.addRow("Min repeatability:", self.sb_min_repeatability)

        self.chk_filter_area = QCheckBox("Filter by area")
        self.chk_filter_area.setChecked(True)
        det_layout.addRow(self.chk_filter_area)
        self.sb_min_area = self._dspin(1, 1e8, 100)
        self.sb_min_area.setToolTip("Minimum blob contour area in pixels.")
        self.sb_max_area = self._dspin(2, 1e8, 1e7)
        self.sb_max_area.setToolTip(
            "Maximum blob contour area in pixels. Set high to avoid rejecting large blobs.")
        det_layout.addRow("  Min area:", self.sb_min_area)
        det_layout.addRow("  Max area:", self.sb_max_area)

        self.chk_filter_circularity = QCheckBox("Filter by circularity")
        det_layout.addRow(self.chk_filter_circularity)
        self.sb_min_circularity = self._dspin(0, 1, 0.1, step=0.05, decimals=3)
        self.sb_max_circularity = self._dspin(0, 1, 1.0, step=0.05, decimals=3)
        det_layout.addRow("  Min:", self.sb_min_circularity)
        det_layout.addRow("  Max:", self.sb_max_circularity)

        self.chk_filter_convexity = QCheckBox("Filter by convexity")
        det_layout.addRow(self.chk_filter_convexity)
        self.sb_min_convexity = self._dspin(0, 1, 0.5, step=0.05, decimals=3)
        self.sb_max_convexity = self._dspin(0, 1, 1.0, step=0.05, decimals=3)
        det_layout.addRow("  Min:", self.sb_min_convexity)
        det_layout.addRow("  Max:", self.sb_max_convexity)

        self.chk_filter_inertia = QCheckBox("Filter by inertia ratio")
        det_layout.addRow(self.chk_filter_inertia)
        self.sb_min_inertia = self._dspin(0, 1, 0.1, step=0.05, decimals=3)
        self.sb_max_inertia = self._dspin(0, 1, 1.0, step=0.05, decimals=3)
        det_layout.addRow("  Min:", self.sb_min_inertia)
        det_layout.addRow("  Max:", self.sb_max_inertia)

        outer.addWidget(det_box)

        # ── SORT tracker parameters ───────────────────────────────────────
        trk_box = QGroupBox("SORT tracker params")
        trk_layout = QFormLayout(trk_box)
        trk_layout.setHorizontalSpacing(8)
        trk_layout.setVerticalSpacing(3)
        trk_layout.setContentsMargins(6, 6, 6, 6)

        self.sb_sort_max_age = QSpinBox()
        self.sb_sort_max_age.setRange(1, 200)
        self.sb_sort_max_age.setValue(5)
        self.sb_sort_max_age.setToolTip("Frames a track survives without a matched detection.")

        self.sb_sort_min_hits = QSpinBox()
        self.sb_sort_min_hits.setRange(1, 50)
        self.sb_sort_min_hits.setValue(3)
        self.sb_sort_min_hits.setToolTip("Consecutive hits required before a track is reported.")

        self.sb_sort_iou = self._dspin(0, 1, 0.3, step=0.05, decimals=3)
        self.sb_sort_iou.setToolTip("IoU threshold for matching detections to existing tracks.")

        trk_layout.addRow("Max age (frames):", self.sb_sort_max_age)
        trk_layout.addRow("Min hits:", self.sb_sort_min_hits)
        trk_layout.addRow("IoU threshold:", self.sb_sort_iou)

        outer.addWidget(trk_box)

        # ── Reset button ──────────────────────────────────────────────────
        self.btn_reset_tracker = QPushButton("Reset Tracker")
        self.btn_reset_tracker.setMinimumHeight(26)
        self.btn_reset_tracker.setToolTip(
            "Discard all active tracks, clear cached results, and restart track IDs from 1.")
        outer.addWidget(self.btn_reset_tracker)

        outer.addStretch()
        self.setWidget(container)

    # ── Config loading ────────────────────────────────────────────────────

    def _load_from_config(self) -> None:
        """
        Populate spinboxes from the [BLOB_TRACKING] section of the active
        DashPVA profile config (loaded via app_settings.CONFIG).  Falls back
        to widget defaults if the section is absent (e.g. non-blob profiles).
        After applying, writes ~/.dashpva_blob_config.json so the consumer
        gets the profile values immediately on startup.
        """
        cfg = app_settings.CONFIG or {}
        bt = cfg.get('BLOB_TRACKING', {})
        det = bt.get('DETECTOR', {})
        srt = bt.get('SORT', {})

        if not bt:
            # No BLOB_TRACKING section in active profile — write the widget
            # defaults to the JSON file so the consumer always starts with
            # valid, known-good params rather than OpenCV's internal defaults.
            self._write_json()
            return

        def _set(widget, key, mapping):
            if key in mapping:
                try:
                    widget.setValue(mapping[key])
                except Exception:
                    pass

        def _set_chk(widget, key, mapping):
            if key in mapping:
                widget.setChecked(bool(mapping[key]))

        _set(self.sb_min_threshold,       'MIN_THRESHOLD',        det)
        _set(self.sb_max_threshold,       'MAX_THRESHOLD',        det)
        _set(self.sb_threshold_step,      'THRESHOLD_STEP',       det)
        _set_chk(self.chk_filter_color,   'FILTER_BY_COLOR',      det)
        if 'BLOB_COLOR' in det:
            idx = self.cb_blob_color.findData(int(det['BLOB_COLOR']))
            if idx >= 0:
                self.cb_blob_color.setCurrentIndex(idx)
        _set(self.sb_min_dist,            'MIN_DIST_BETWEEN_BLOBS', det)
        _set(self.sb_min_repeatability,   'MIN_REPEATABILITY',    det)
        _set_chk(self.chk_filter_area,    'FILTER_BY_AREA',      det)
        _set(self.sb_min_area,            'MIN_AREA',            det)
        _set(self.sb_max_area,            'MAX_AREA',            det)
        _set_chk(self.chk_filter_circularity, 'FILTER_BY_CIRCULARITY', det)
        _set(self.sb_min_circularity,  'MIN_CIRCULARITY',  det)
        _set(self.sb_max_circularity,  'MAX_CIRCULARITY',  det)
        _set_chk(self.chk_filter_convexity, 'FILTER_BY_CONVEXITY', det)
        _set(self.sb_min_convexity,    'MIN_CONVEXITY',    det)
        _set(self.sb_max_convexity,    'MAX_CONVEXITY',    det)
        _set_chk(self.chk_filter_inertia, 'FILTER_BY_INERTIA', det)
        _set(self.sb_min_inertia,      'MIN_INERTIA_RATIO', det)
        _set(self.sb_max_inertia,      'MAX_INERTIA_RATIO', det)

        _set(self.sb_sort_max_age,  'MAX_AGE',       srt)
        _set(self.sb_sort_min_hits, 'MIN_HITS',      srt)
        _set(self.sb_sort_iou,      'IOU_THRESHOLD', srt)

        self._write_json()

    def _write_json(self) -> None:
        """Write current widget values to ~/.dashpva_blob_config.json."""
        import json, pathlib
        config = {
            'minThreshold':          self.sb_min_threshold.value(),
            'maxThreshold':          self.sb_max_threshold.value(),
            'thresholdStep':         self.sb_threshold_step.value(),
            'filterByColor':         self.chk_filter_color.isChecked(),
            'blobColor':             int(self.cb_blob_color.currentData()),
            'minDistBetweenBlobs':   self.sb_min_dist.value(),
            'minRepeatability':      int(self.sb_min_repeatability.value()),
            'filterByArea':          self.chk_filter_area.isChecked(),
            'minArea':               self.sb_min_area.value(),
            'maxArea':               self.sb_max_area.value(),
            'filterByCircularity':self.chk_filter_circularity.isChecked(),
            'minCircularity':     self.sb_min_circularity.value(),
            'maxCircularity':     self.sb_max_circularity.value(),
            'filterByConvexity':  self.chk_filter_convexity.isChecked(),
            'minConvexity':       self.sb_min_convexity.value(),
            'maxConvexity':       self.sb_max_convexity.value(),
            'filterByInertia':    self.chk_filter_inertia.isChecked(),
            'minInertiaRatio':    self.sb_min_inertia.value(),
            'maxInertiaRatio':    self.sb_max_inertia.value(),
            'sort_max_age':       self.sb_sort_max_age.value(),
            'sort_min_hits':      self.sb_sort_min_hits.value(),
            'sort_iou_threshold': self.sb_sort_iou.value(),
        }
        try:
            pathlib.Path.home().joinpath('.dashpva_blob_config.json').write_text(
                json.dumps(config, indent=2))
        except Exception:
            pass

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _dspin(lo, hi, default, step=1.0, decimals=1):
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(default)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        return sb
