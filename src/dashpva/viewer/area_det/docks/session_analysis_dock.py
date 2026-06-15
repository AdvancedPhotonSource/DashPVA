"""
SessionAnalysisDock — manual "Analyze Session" panel that wires the cached
feature vectors / VLM observations / PV history to a chosen LLM backend.

Spawns a worker QThread per analyze click so the GUI stays responsive while
the LLM is generating.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.analysis.experiment_context import ExperimentContext
from dashpva.analysis.llm_backend import make_backend
from dashpva.analysis.session_analyzer import SessionAnalyzer
from dashpva.viewer.core.docks.base_dock import BaseDock


class _AnalyzeWorker(QThread):
    """Runs SessionAnalyzer.analyze() off the GUI thread."""
    finished_ok = pyqtSignal(str, float)         # result text, elapsed seconds
    finished_err = pyqtSignal(str)               # error message

    def __init__(self, analyzer: SessionAnalyzer, parent=None):
        super().__init__(parent)
        self._analyzer = analyzer

    def run(self) -> None:
        t0 = time.time()
        try:
            text = self._analyzer.analyze()
            self.finished_ok.emit(text, time.time() - t0)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class SessionAnalysisDock(BaseDock):

    def __init__(self, main_window=None, show: bool = False):
        super().__init__(
            title="Session Analysis",
            main_window=main_window,
            segment_name="analysis",
            dock_area=Qt.RightDockWidgetArea,
            show=show,
        )
        self._worker: Optional[_AnalyzeWorker] = None
        self._build()
        self._apply_session_config()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build(self):
        container = QWidget()
        container.setMinimumWidth(380)
        container.setMaximumWidth(560)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # ── Backend selection ────────────────────────────────────────
        backend_box = QGroupBox("Backend")
        backend_layout = QVBoxLayout(backend_box)
        backend_layout.setContentsMargins(6, 6, 6, 6)
        backend_layout.setSpacing(4)

        self.rb_ollama = QRadioButton("Local ollama")
        self.rb_argo = QRadioButton("Argo API")
        self.rb_ollama.setChecked(True)
        self._backend_group = QButtonGroup(backend_box)
        self._backend_group.addButton(self.rb_ollama)
        self._backend_group.addButton(self.rb_argo)
        backend_layout.addWidget(self.rb_ollama)
        backend_layout.addWidget(self.rb_argo)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.le_model = QLineEdit()
        self.le_model.setPlaceholderText("e.g. llama3.2 or gpt-4o")
        model_row.addWidget(self.le_model)
        backend_layout.addLayout(model_row)

        snap_row = QHBoxLayout()
        snap_row.addWidget(QLabel("Snapshots in prompt:"))
        self.sb_snapshots = QSpinBox()
        self.sb_snapshots.setRange(1, 100)
        self.sb_snapshots.setValue(10)
        snap_row.addWidget(self.sb_snapshots)
        snap_row.addStretch()
        backend_layout.addLayout(snap_row)
        outer.addWidget(backend_box)

        # ── Experiment context ───────────────────────────────────────
        ctx_box = QGroupBox("Experiment Context (optional)")
        ctx_layout = QHBoxLayout(ctx_box)
        ctx_layout.setContentsMargins(6, 6, 6, 6)
        self.le_context = QLineEdit()
        self.le_context.setPlaceholderText("path to .toml / .json / .txt")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setMaximumWidth(80)
        self.btn_browse.clicked.connect(self._browse_context)
        ctx_layout.addWidget(self.le_context)
        ctx_layout.addWidget(self.btn_browse)
        outer.addWidget(ctx_box)

        # ── Action row ───────────────────────────────────────────────
        action_row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze Session")
        self.btn_analyze.setMinimumHeight(32)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        action_row.addWidget(self.btn_analyze)

        self.btn_preview = QPushButton("Preview Prompt")
        self.btn_preview.setMinimumHeight(32)
        self.btn_preview.setToolTip(
            "Build the prompt from current caches and show it in the output area "
            "without contacting the LLM. Useful for verifying what would be sent.")
        self.btn_preview.clicked.connect(self._on_preview_clicked)
        action_row.addWidget(self.btn_preview)
        outer.addLayout(action_row)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setAlignment(Qt.AlignLeft)
        outer.addWidget(self.lbl_status)

        # ── Result display ───────────────────────────────────────────
        self.txt_result = QTextBrowser()
        self.txt_result.setOpenExternalLinks(True)
        self.txt_result.setPlaceholderText(
            "Click Analyze Session to send cached features + observations to "
            "the configured LLM backend."
        )
        outer.addWidget(self.txt_result, stretch=1)

        self.setWidget(container)

    def _apply_session_config(self) -> None:
        """Seed UI from app_settings.SESSION_ANALYSIS so the user sees the
        same defaults the TOML profile defined."""
        cfg = getattr(app_settings, 'SESSION_ANALYSIS', {}) or {}
        backend = (cfg.get('BACKEND') or 'ollama').lower()
        if backend == 'argo':
            self.rb_argo.setChecked(True)
            self.le_model.setText(str(cfg.get('ARGO_MODEL') or 'gpt-4o'))
        else:
            self.rb_ollama.setChecked(True)
            self.le_model.setText(str(cfg.get('OLLAMA_MODEL') or 'llama3.2'))

        if cfg.get('N_SNAPSHOTS'):
            try:
                self.sb_snapshots.setValue(int(cfg['N_SNAPSHOTS']))
            except Exception:
                pass

        prior = cfg.get('PRIOR_FILE')
        if prior:
            self.le_context.setText(str(prior))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_context(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose experiment context file",
            self.le_context.text() or "",
            "Context files (*.toml *.json *.txt);;All files (*)",
        )
        if path:
            self.le_context.setText(path)

    def _build_analyzer(self) -> SessionAnalyzer:
        reader = getattr(self.main_window, 'reader', None)
        if reader is None:
            raise RuntimeError("No PVA reader — start Live View first.")

        # Build a per-click config so manual UI overrides take precedence over
        # the TOML defaults pulled at construction.
        cfg = dict(getattr(app_settings, 'SESSION_ANALYSIS', {}) or {})
        if self.rb_argo.isChecked():
            cfg['BACKEND'] = 'argo'
            cfg['ARGO_MODEL'] = self.le_model.text().strip() or cfg.get('ARGO_MODEL', 'gpt-4o')
        else:
            cfg['BACKEND'] = 'ollama'
            cfg['OLLAMA_MODEL'] = self.le_model.text().strip() or cfg.get('OLLAMA_MODEL', 'llama3.2')

        backend = make_backend(cfg)
        context = ExperimentContext(self.le_context.text().strip() or None)
        return SessionAnalyzer(
            pva_reader=reader,
            backend=backend,
            context=context,
            n_snapshots=int(self.sb_snapshots.value()),
        )

    def _on_preview_clicked(self) -> None:
        try:
            analyzer = self._build_analyzer()
            self.txt_result.setPlainText(analyzer.build_prompt())
            self.lbl_status.setText("Prompt preview (no LLM call)")
        except Exception as e:
            self._show_error(f"Preview failed: {e}")

    def _on_analyze_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            analyzer = self._build_analyzer()
        except Exception as e:
            self._show_error(f"Failed to build analyzer: {e}")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.lbl_status.setText(f"Running ({analyzer.backend.name})…")
        self.txt_result.setPlainText("")

        self._worker = _AnalyzeWorker(analyzer, parent=self)
        self._worker.finished_ok.connect(self._on_analysis_ok)
        self._worker.finished_err.connect(self._on_analysis_err)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    def _on_analysis_ok(self, text: str, elapsed_s: float) -> None:
        self.lbl_status.setText(f"Done ({elapsed_s:.1f}s)")
        self.txt_result.setPlainText(text)

    def _on_analysis_err(self, msg: str) -> None:
        self._show_error(msg)

    def _cleanup_worker(self) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self._worker = None

    def _show_error(self, msg: str) -> None:
        self.lbl_status.setText("Error")
        # Plain HTML so red doesn't bleed into the next display via markdown.
        self.txt_result.setHtml(
            f'<pre style="color:#cc4444; white-space:pre-wrap;">{msg}</pre>'
        )