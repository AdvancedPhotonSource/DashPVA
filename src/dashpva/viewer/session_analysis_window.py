"""
SessionAnalysisWindow — standalone top-level window for assembling cached
features + observations + PV history and sending them to a chosen LLM
backend for scientific interpretation.

Lives outside the area detector dock layout because LLM responses are
typically multi-paragraph and don't fit comfortably in a ~380px-wide dock.
The window is parented to the launching window so it's automatically
cleaned up on app exit, but rendered as a top-level Qt.Window.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.analysis.experiment_context import ExperimentContext
from dashpva.analysis.llm_backend import make_backend
from dashpva.analysis.session_analyzer import SessionAnalyzer


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


class SessionAnalysisWindow(QMainWindow):
    """Top-level Session Analysis window.

    Pass the active ``PVAReader`` so the analyzer can pull the cached feature
    vectors / VLM descriptions / PV history. The window builds and tears down
    a fresh backend per Analyze click, so changes to the model / backend
    selector take effect immediately.
    """

    def __init__(self, pva_reader, parent=None):
        super().__init__(parent, Qt.Window)
        self.reader = pva_reader
        self._worker: Optional[_AnalyzeWorker] = None

        self.setWindowTitle("DashPVA — Session Analysis")
        self.resize(900, 700)

        self._build_ui()
        self._apply_session_config()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # ── Backend + model row ──────────────────────────────────────
        backend_box = QGroupBox("Backend")
        backend_grid = QGridLayout(backend_box)
        backend_grid.setContentsMargins(8, 8, 8, 8)
        backend_grid.setHorizontalSpacing(10)
        backend_grid.setVerticalSpacing(6)

        self.rb_ollama = QRadioButton("Local ollama")
        self.rb_argo = QRadioButton("Argo API")
        self.rb_ollama.setChecked(True)
        self._backend_group = QButtonGroup(backend_box)
        self._backend_group.addButton(self.rb_ollama)
        self._backend_group.addButton(self.rb_argo)
        backend_grid.addWidget(self.rb_ollama, 0, 0)
        backend_grid.addWidget(self.rb_argo, 0, 1)

        backend_grid.addWidget(QLabel("Model:"), 1, 0)
        self.le_model = QLineEdit()
        self.le_model.setPlaceholderText("e.g. llama3.2, claudesonnet46, gpt5")
        backend_grid.addWidget(self.le_model, 1, 1, 1, 3)

        backend_grid.addWidget(QLabel("Snapshots in prompt:"), 2, 0)
        self.sb_snapshots = QSpinBox()
        self.sb_snapshots.setRange(1, 100)
        self.sb_snapshots.setValue(10)
        backend_grid.addWidget(self.sb_snapshots, 2, 1)
        backend_grid.setColumnStretch(3, 1)
        outer.addWidget(backend_box)

        # ── Experiment context ───────────────────────────────────────
        ctx_box = QGroupBox("Experiment Context (optional)")
        ctx_row = QHBoxLayout(ctx_box)
        ctx_row.setContentsMargins(8, 8, 8, 8)
        self.le_context = QLineEdit()
        self.le_context.setPlaceholderText("path to .toml / .json / .txt")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setMaximumWidth(90)
        self.btn_browse.clicked.connect(self._browse_context)
        ctx_row.addWidget(self.le_context)
        ctx_row.addWidget(self.btn_browse)
        outer.addWidget(ctx_box)

        # ── Action buttons ───────────────────────────────────────────
        action_row = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyze Session")
        self.btn_analyze.setMinimumHeight(36)
        self.btn_analyze.setDefault(True)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        action_row.addWidget(self.btn_analyze, stretch=2)

        self.btn_preview = QPushButton("Preview Prompt")
        self.btn_preview.setMinimumHeight(36)
        self.btn_preview.setToolTip(
            "Assemble the prompt from current caches and show it below "
            "without contacting the LLM. Useful for checking what would be sent."
        )
        self.btn_preview.clicked.connect(self._on_preview_clicked)
        action_row.addWidget(self.btn_preview, stretch=2)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setMinimumHeight(36)
        self.btn_copy.setMaximumWidth(90)
        self.btn_copy.clicked.connect(self._copy_result)
        action_row.addWidget(self.btn_copy)

        self.btn_save = QPushButton("Save…")
        self.btn_save.setMinimumHeight(36)
        self.btn_save.setMaximumWidth(90)
        self.btn_save.clicked.connect(self._save_result)
        action_row.addWidget(self.btn_save)

        outer.addLayout(action_row)

        # ── Result display (takes the rest of the window) ────────────
        self.txt_result = QTextBrowser()
        self.txt_result.setOpenExternalLinks(True)
        self.txt_result.setPlaceholderText(
            "Click Analyze Session to send cached features + observations "
            "to the configured LLM backend. Click Preview Prompt to see "
            "exactly what would be sent without spending tokens."
        )
        font = QFont(self.txt_result.font())
        # Use a slightly larger and proportional font for narrative LLM text.
        font.setPointSize(font.pointSize() + 1)
        self.txt_result.setFont(font)
        outer.addWidget(self.txt_result, stretch=1)

        self.setCentralWidget(central)

        # ── Status bar ───────────────────────────────────────────────
        self.status_bar = QStatusBar(self)
        self.lbl_status = QLabel("Idle")
        self.status_bar.addWidget(self.lbl_status, 1)
        self.setStatusBar(self.status_bar)

    def _apply_session_config(self) -> None:
        """Seed the UI from SESSION_ANALYSIS so the user sees the TOML defaults."""
        cfg = getattr(app_settings, 'SESSION_ANALYSIS', {}) or {}
        backend = (cfg.get('BACKEND') or 'ollama').lower()
        if backend == 'argo':
            self.rb_argo.setChecked(True)
            self.le_model.setText(str(cfg.get('ARGO_MODEL') or 'claudesonnet46'))
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
        if self.reader is None:
            raise RuntimeError("No PVA reader — start Live View first.")

        cfg = dict(getattr(app_settings, 'SESSION_ANALYSIS', {}) or {})
        if self.rb_argo.isChecked():
            cfg['BACKEND'] = 'argo'
            cfg['ARGO_MODEL'] = (
                self.le_model.text().strip()
                or cfg.get('ARGO_MODEL', 'claudesonnet46')
            )
        else:
            cfg['BACKEND'] = 'ollama'
            cfg['OLLAMA_MODEL'] = (
                self.le_model.text().strip()
                or cfg.get('OLLAMA_MODEL', 'llama3.2')
            )

        backend = make_backend(cfg)
        context = ExperimentContext(self.le_context.text().strip() or None)
        return SessionAnalyzer(
            pva_reader=self.reader,
            backend=backend,
            context=context,
            n_snapshots=int(self.sb_snapshots.value()),
        )

    def _on_preview_clicked(self) -> None:
        try:
            analyzer = self._build_analyzer()
            self.txt_result.setPlainText(analyzer.build_prompt())
            self.lbl_status.setText(
                f"Prompt preview ({analyzer.backend.name}, no LLM call)"
            )
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
        self.lbl_status.setText(f"Done in {elapsed_s:.1f}s")
        self.txt_result.setPlainText(text)

    def _on_analysis_err(self, msg: str) -> None:
        self._show_error(msg)

    def _cleanup_worker(self) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self._worker = None

    def _show_error(self, msg: str) -> None:
        self.lbl_status.setText("Error")
        # html-escape via setPlainText would lose color; use a styled <pre>
        # block instead and rely on the message being safe (we built it).
        self.txt_result.setHtml(
            f'<pre style="color:#cc4444; white-space:pre-wrap;">{msg}</pre>'
        )

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _copy_result(self) -> None:
        text = self.txt_result.toPlainText()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.lbl_status.setText("Copied to clipboard")

    def _save_result(self) -> None:
        text = self.txt_result.toPlainText()
        if not text:
            QMessageBox.information(
                self, "Nothing to save",
                "Run an analysis (or preview a prompt) first."
            )
            return
        default = f"session_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis", default,
            "Text files (*.txt);;Markdown (*.md);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.lbl_status.setText(f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_reader(self, pva_reader) -> None:
        """Hot-swap the reader (the launcher calls this when Live View is
        (re)started so the window always points at the current reader)."""
        self.reader = pva_reader

    def closeEvent(self, event) -> None:
        # Don't block close on a running worker — Qt disconnects signals
        # automatically when the receiver is destroyed, so the worker will
        # run to completion and its result simply gets dropped.
        super().closeEvent(event)
