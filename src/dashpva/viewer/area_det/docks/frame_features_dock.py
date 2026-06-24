"""
Dock that shows live deterministic feature statistics extracted by
HpcFeatureExtractionProcessor.  No LLM dependency.

Feature labels update on every timer tick from the PVAReader's latest
pv_attributes.  A "VLM Sampler" group box (off by default) controls the
optional background moondream sampler (BackgroundVlmSampler) for catching
rings and artifacts that blob detection misses.
"""

import json

import dashpva.settings as app_settings
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock


class FrameFeaturesDock(BaseDock):
    """Live feature statistics from the deterministic HPC pipeline."""

    def __init__(self, main_window=None, show: bool = False):
        super().__init__(title="Frame Features", main_window=main_window,
                         segment_name="analysis", dock_area=Qt.RightDockWidgetArea,
                         show=show)
        self._last_fv_raw: str = ''  # skip JSON parse when FeatureVector hasn't changed
        self._build()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # ── Feature summary ───────────────────────────────────────────
        feat_box = QGroupBox("Frame Features")
        feat_layout = QFormLayout(feat_box)
        feat_layout.setHorizontalSpacing(8)
        feat_layout.setVerticalSpacing(3)
        feat_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_n_blobs      = QLabel("—")
        self.lbl_snr          = QLabel("—")
        self.lbl_total_int    = QLabel("—")
        self.lbl_background   = QLabel("—")
        self.lbl_active_frac  = QLabel("—")
        self.lbl_peak         = QLabel("—")
        self.lbl_bg_texture   = QLabel("—")
        self.lbl_radial_peaks = QLabel("—")
        self.lbl_radial_peaks.setWordWrap(True)
        # Prevent the value column from ever forcing the dock wider than its
        # container by capping horizontal growth and letting the label wrap.
        self.lbl_radial_peaks.setMaximumWidth(200)
        self.lbl_radial_peaks.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred)

        feat_layout.addRow("Blobs detected:",          self.lbl_n_blobs)
        feat_layout.addRow("Peak/median:",             self.lbl_snr)
        feat_layout.addRow("Total intensity:",         self.lbl_total_int)
        feat_layout.addRow("Background (median):",     self.lbl_background)
        feat_layout.addRow("Active area fraction:",    self.lbl_active_frac)
        feat_layout.addRow("Peak location (x, y):",   self.lbl_peak)
        feat_layout.addRow("Background texture (σ):", self.lbl_bg_texture)
        feat_layout.addRow("Radial ring peaks:",       self.lbl_radial_peaks)
        outer.addWidget(feat_box)

        # ── VLM sampler controls (optional) ──────────────────────────
        vlm_box = QGroupBox("VLM Sampler (optional)")
        vlm_layout = QVBoxLayout(vlm_box)
        vlm_layout.setContentsMargins(6, 6, 6, 6)
        vlm_layout.setSpacing(4)

        self.chk_vlm_enabled = QCheckBox("Enable periodic moondream sampling")
        self.chk_vlm_enabled.setToolTip(
            "Samples frames every N seconds and sends them to the local moondream "
            "model to capture rings, artifacts, and texture the blob detector misses.")
        self.chk_vlm_enabled.setChecked(False)
        vlm_layout.addWidget(self.chk_vlm_enabled)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (s):"))
        self.sb_vlm_interval = QSpinBox()
        self.sb_vlm_interval.setRange(5, 600)
        self.sb_vlm_interval.setValue(30)
        self.sb_vlm_interval.setSuffix(" s")
        interval_row.addWidget(self.sb_vlm_interval)
        interval_row.addStretch()
        vlm_layout.addLayout(interval_row)

        self.lbl_vlm_status = QLabel("VLM sampler off")
        self.lbl_vlm_status.setAlignment(Qt.AlignCenter)
        vlm_layout.addWidget(self.lbl_vlm_status)

        outer.addWidget(vlm_box)

        # ── Session-level analysis launcher ───────────────────────────
        # Spawns SessionAnalysisWindow as a top-level window — the response
        # area in this narrow dock is too small for multi-paragraph LLM text.
        self.btn_session_analysis = QPushButton("Open Session Analysis…")
        self.btn_session_analysis.setMinimumHeight(32)
        self.btn_session_analysis.setToolTip(
            "Open a separate window that sends the cached features, VLM "
            "observations, and PV history to an LLM for a session-wide "
            "scientific interpretation."
        )
        outer.addWidget(self.btn_session_analysis)

        outer.addStretch()
        self.setWidget(container)

    # ------------------------------------------------------------------
    # Live update (called from viewer timer on every frame)
    # ------------------------------------------------------------------

    def update_features(self, pv_attributes: dict):
        """Refresh feature summary labels from the latest pv_attributes dict."""
        raw = pv_attributes.get('FeatureVector', '')
        if not raw or raw == self._last_fv_raw:
            return  # no new frame — skip redundant JSON parse
        self._last_fv_raw = raw
        try:
            fv = json.loads(raw)
        except Exception:
            return

        frame = fv.get('frame', {})
        n = fv.get('n_blobs', '—')

        self.lbl_n_blobs.setText(str(n))
        # 'peak_to_median' is the current key; fall back to legacy 'snr'.
        self.lbl_snr.setText(str(frame.get('peak_to_median', frame.get('snr', '—'))))

        total = frame.get('total_intensity')
        self.lbl_total_int.setText(f"{total:,.0f}" if isinstance(total, (int, float)) else '—')

        self.lbl_background.setText(str(frame.get('background', '—')))

        af = frame.get('active_fraction')
        self.lbl_active_frac.setText(f"{af:.4f}" if isinstance(af, float) else '—')
        self.lbl_peak.setText(f"({frame.get('peak_x', '?')}, {frame.get('peak_y', '?')})")

        tex = frame.get('background_texture')
        self.lbl_bg_texture.setText(f"{tex:.2f}" if isinstance(tex, float) else '—')

        peaks = frame.get('radial_profile_peaks', [])
        if peaks:
            peak_str = ', '.join(f"r={p['r_px']}px" for p in peaks[:3])
            self.lbl_radial_peaks.setText(peak_str)
        else:
            self.lbl_radial_peaks.setText("none")

    def update_vlm_status(self, text: str):
        """Called by BackgroundVlmSampler to report sampling status."""
        self.lbl_vlm_status.setText(text)
