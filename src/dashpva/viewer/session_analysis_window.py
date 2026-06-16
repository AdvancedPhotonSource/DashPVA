"""
SessionAnalysisWindow — standalone top-level chat window for interrogating a
live (or saved) session with an LLM.

The LLM can call tools (read EPICS PVs live or historically by frame/timestamp,
inspect cached session features) on demand, so the conversation is no longer a
single fixed prompt. The old one-shot "Analyze" button survives as an
always-available "Summarize session" shortcut that injects the existing
six-section prompt as the next chat message.

Lives outside the area detector dock layout because LLM responses are typically
multi-paragraph. Parented to the launching window for automatic cleanup, but
rendered as a top-level Qt.Window.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Callable, Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.analysis.chat_controller import ChatController, ControllerEvent
from dashpva.analysis.experiment_context import ExperimentContext
from dashpva.analysis.llm_backend import make_backend
from dashpva.analysis.session_analyzer import SessionAnalyzer
from dashpva.analysis.tools.base import ToolRegistry
from dashpva.analysis.tools.pv_tools import PvTools
from dashpva.analysis.tools.session_tools import SessionTools

# ======================================================================
# Worker — runs one chat turn off the GUI thread
# ======================================================================


class _ChatTurnWorker(QThread):
    """Runs one ChatController turn off the GUI thread.

    The controller invokes our ``_emit`` callback (on this worker thread) for
    each :class:`ControllerEvent`; we re-emit each as a Qt signal, which Qt
    auto-marshals to the GUI thread.
    """

    assistant_text = pyqtSignal(str)
    tool_requested = pyqtSignal(str, dict)      # name, arguments
    tool_completed = pyqtSignal(str, dict)      # name, result
    turn_error = pyqtSignal(str)
    turn_done = pyqtSignal(int, float)          # rounds_used, elapsed_s

    def __init__(self, runner: Callable[[Callable[[ControllerEvent], None]], None],
                 parent=None):
        super().__init__(parent)
        self._runner = runner
        self._t0 = 0.0

    def run(self) -> None:
        self._t0 = time.time()
        try:
            self._runner(self._emit)
        except Exception as e:
            self.turn_error.emit(f"{type(e).__name__}: {e}")

    def _emit(self, ev: ControllerEvent) -> None:
        if ev.kind == 'assistant_text':
            self.assistant_text.emit(ev.text)
        elif ev.kind == 'tool_call_requested':
            self.tool_requested.emit(ev.tool_name, ev.tool_arguments or {})
        elif ev.kind == 'tool_call_result':
            self.tool_completed.emit(ev.tool_name, ev.tool_result or {})
        elif ev.kind == 'error':
            self.turn_error.emit(ev.text)
        elif ev.kind == 'done':
            self.turn_done.emit(ev.rounds_used, time.time() - self._t0)


# ======================================================================
# Conversation widgets
# ======================================================================


class _ChatInput(QPlainTextEdit):
    """Multi-line input that sends on Enter, newlines on Shift+Enter."""

    submitted = pyqtSignal()

    def keyPressEvent(self, e) -> None:
        if (e.key() in (Qt.Key_Return, Qt.Key_Enter)
                and not (e.modifiers() & Qt.ShiftModifier)):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class _ToolBubble(QFrame):
    """Collapsible row showing one tool call: header line + hidden details."""

    def __init__(self, name: str, arguments: dict, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "_ToolBubble { background:#f0f0f3; border:1px solid #d8d8de; "
            "border-radius:6px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        arg_summary = ', '.join(f"{k}={v}" for k, v in arguments.items())
        if len(arg_summary) > 80:
            arg_summary = arg_summary[:80] + '…'

        self._toggle = QToolButton()
        self._toggle.setStyleSheet("QToolButton { border:none; color:#555; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toggle.setCheckable(True)
        self._toggle.setText(f"▸  tool · {name}({arg_summary})")
        self._toggle.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._details.setStyleSheet("color:#444; font-family:monospace;")
        self._details.setVisible(False)
        lay.addWidget(self._details)

        self._name = name
        self._arguments = arguments
        self._result: dict | None = None
        self._render_details()

    def set_result(self, result: dict) -> None:
        self._result = result
        # Surface failures in the header without needing to expand.
        if isinstance(result, dict) and 'error' in result:
            self._toggle.setStyleSheet("QToolButton { border:none; color:#b04040; }")
            self._toggle.setText(self._toggle.text() + "  — error")
        self._render_details()

    def _on_toggle(self, checked: bool) -> None:
        self._details.setVisible(checked)
        self._toggle.setText(
            ('▾' if checked else '▸') + self._toggle.text()[1:]
        )

    def _render_details(self) -> None:
        parts = [f"args: {json.dumps(self._arguments, default=str)}"]
        if self._result is not None:
            parts.append(f"result: {json.dumps(self._result, default=str, indent=2)}")
        self._details.setText('\n'.join(parts))

    def transcript_line(self) -> str:
        res = json.dumps(self._result, default=str) if self._result is not None else '(pending)'
        return f"[tool {self._name}] args={json.dumps(self._arguments, default=str)} -> {res}"


class ConversationView(QScrollArea):
    """Vertical stack of chat bubbles (user / assistant / tool / error).

    Standard Qt chat pattern: a QScrollArea wrapping a QVBoxLayout, auto-scrolled
    to the bottom on each addition. Keeps a parallel ``_entries`` list of
    ``(role, text_or_bubble)`` for transcript export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._vbox = QVBoxLayout(self._host)
        self._vbox.setContentsMargins(8, 8, 8, 8)
        self._vbox.setSpacing(8)
        self._vbox.addStretch(1)   # keep bubbles top-aligned
        self.setWidget(self._host)
        self._entries: list = []   # (role, payload) where payload is str or _ToolBubble

    # -- public API --

    def add_user(self, text: str) -> None:
        self._add_text_bubble('You', text, bg='#e3f0ff', align_right=True)
        self._entries.append(('user', text))

    def add_assistant(self, text: str) -> None:
        self._add_text_bubble('Assistant', text, bg='#f3f3f0', align_right=False)
        self._entries.append(('assistant', text))

    def add_error(self, text: str) -> None:
        self._add_text_bubble('Error', text, bg='#fbe6e6', align_right=False,
                              title_color='#b04040')
        self._entries.append(('error', text))

    def add_tool(self, name: str, arguments: dict) -> _ToolBubble:
        bubble = _ToolBubble(name, arguments)
        self._insert_widget(bubble)
        self._entries.append(('tool', bubble))
        return bubble

    def clear(self) -> None:
        while self._vbox.count() > 1:   # keep the trailing stretch
            item = self._vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._entries.clear()

    def to_plain_text(self) -> str:
        lines = []
        for role, payload in self._entries:
            if role == 'tool':
                lines.append(payload.transcript_line())
            else:
                label = {'user': 'You', 'assistant': 'Assistant', 'error': 'Error'}.get(role, role)
                lines.append(f"{label}: {payload}")
        return '\n\n'.join(lines)

    # -- internals --

    def _add_text_bubble(self, title: str, text: str, *, bg: str,
                         align_right: bool, title_color: str = '#666') -> None:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(
            f"QFrame {{ background:{bg}; border:1px solid #d8d8de; border-radius:8px; }}"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        hdr = QLabel(title)
        hf = QFont(hdr.font())
        hf.setBold(True)
        hf.setPointSize(max(1, hf.pointSize() - 1))
        hdr.setFont(hf)
        hdr.setStyleSheet(f"color:{title_color}; border:none;")
        lay.addWidget(hdr)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setStyleSheet("border:none;")
        lay.addWidget(body)

        self._insert_widget(frame, align_right=align_right)

    def _insert_widget(self, widget: QWidget, align_right: bool = False) -> None:
        # Insert just before the trailing stretch.
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if align_right:
            row.addStretch(1)
            row.addWidget(widget, 4)
        else:
            row.addWidget(widget, 4)
            row.addStretch(1)
        container = QWidget()
        container.setLayout(row)
        self._vbox.insertWidget(self._vbox.count() - 1, container)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        # Defer to the next event-loop tick so the new bubble's height is in the
        # scrollbar range. Must NOT use processEvents() here — calling it inside
        # a signal handler reenters the loop and delivers later-queued chat
        # events out of order (tool/assistant bubbles get scrambled).
        def _do():
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        QTimer.singleShot(0, _do)


# ======================================================================
# Main window
# ======================================================================


class SessionAnalysisWindow(QMainWindow):
    """Top-level Session Analysis chat window.

    Pass the active ``PVAReader``. The window lazily builds a
    :class:`ChatController` (with PV + session tools) on the first turn and
    refreshes the backend from the UI before each turn, so model/backend
    changes take effect immediately while conversation history persists.
    """

    def __init__(self, pva_reader, parent=None):
        super().__init__(parent, Qt.Window)
        self.reader = pva_reader
        self._worker: Optional[_ChatTurnWorker] = None

        # Lazily-built chat machinery.
        self.controller: Optional[ChatController] = None
        self.pv_tools: Optional[PvTools] = None
        self.session_tools: Optional[SessionTools] = None
        self._analyzer: Optional[SessionAnalyzer] = None
        self._pending_tools: deque[_ToolBubble] = deque()

        self.setWindowTitle("DashPVA — Session Analysis")
        self.resize(900, 760)

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

        # ── Backend / model / reset row ──────────────────────────────
        backend_box = QGroupBox("Backend")
        grid = QGridLayout(backend_box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        self.rb_ollama = QRadioButton("Local ollama")
        self.rb_argo = QRadioButton("Argo API")
        self.rb_ollama.setChecked(True)
        self._backend_group = QButtonGroup(backend_box)
        self._backend_group.addButton(self.rb_ollama)
        self._backend_group.addButton(self.rb_argo)
        grid.addWidget(self.rb_ollama, 0, 0)
        grid.addWidget(self.rb_argo, 0, 1)

        self.btn_reset = QPushButton("Reset chat")
        self.btn_reset.setMaximumWidth(110)
        self.btn_reset.clicked.connect(self._on_reset)
        grid.addWidget(self.btn_reset, 0, 3)

        grid.addWidget(QLabel("Model:"), 1, 0)
        self.le_model = QLineEdit()
        self.le_model.setPlaceholderText("e.g. llama3.2, qwen2.5, claudesonnet46, gpt5")
        grid.addWidget(self.le_model, 1, 1, 1, 3)

        grid.addWidget(QLabel("Snapshots in summary:"), 2, 0)
        self.sb_snapshots = QSpinBox()
        self.sb_snapshots.setRange(1, 100)
        self.sb_snapshots.setValue(10)
        self.sb_snapshots.setToolTip(
            "Number of evenly-spaced frame snapshots in the 'Summarize session' "
            "prompt. Does not affect regular chat messages."
        )
        grid.addWidget(self.sb_snapshots, 2, 1)
        grid.setColumnStretch(2, 1)
        outer.addWidget(backend_box)

        # ── Context + history-file pickers ───────────────────────────
        files_box = QGroupBox("Context & history (optional)")
        files_grid = QGridLayout(files_box)
        files_grid.setContentsMargins(8, 8, 8, 8)
        files_grid.setHorizontalSpacing(8)

        files_grid.addWidget(QLabel("Context:"), 0, 0)
        self.le_context = QLineEdit()
        self.le_context.setPlaceholderText("prior-knowledge .toml / .json / .txt")
        files_grid.addWidget(self.le_context, 0, 1)
        self.btn_browse_ctx = QPushButton("Browse…")
        self.btn_browse_ctx.setMaximumWidth(90)
        self.btn_browse_ctx.clicked.connect(self._browse_context)
        files_grid.addWidget(self.btn_browse_ctx, 0, 2)

        files_grid.addWidget(QLabel("History file:"), 1, 0)
        self.le_history = QLineEdit()
        self.le_history.setPlaceholderText(
            "saved scan .h5 — enables source='h5' historical lookups")
        files_grid.addWidget(self.le_history, 1, 1)
        self.btn_browse_hist = QPushButton("Browse…")
        self.btn_browse_hist.setMaximumWidth(90)
        self.btn_browse_hist.clicked.connect(self._browse_history)
        files_grid.addWidget(self.btn_browse_hist, 1, 2)
        files_grid.setColumnStretch(1, 1)
        outer.addWidget(files_box)

        # ── Shortcut row ─────────────────────────────────────────────
        shortcut_row = QHBoxLayout()
        self.btn_summarize = QPushButton("Summarize session")
        self.btn_summarize.setMinimumHeight(32)
        self.btn_summarize.setToolTip(
            "Send the deterministic six-section session summary as a chat "
            "message. Works at any point in the conversation."
        )
        self.btn_summarize.clicked.connect(self._on_summarize_clicked)
        shortcut_row.addWidget(self.btn_summarize, stretch=2)

        self.btn_preview = QPushButton("Preview summary prompt")
        self.btn_preview.setMinimumHeight(32)
        self.btn_preview.setToolTip(
            "Show exactly what 'Summarize session' would send, without "
            "contacting the LLM.")
        self.btn_preview.clicked.connect(self._on_preview_clicked)
        shortcut_row.addWidget(self.btn_preview, stretch=2)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setMinimumHeight(32)
        self.btn_copy.setMaximumWidth(80)
        self.btn_copy.clicked.connect(self._copy_transcript)
        shortcut_row.addWidget(self.btn_copy)

        self.btn_save = QPushButton("Save…")
        self.btn_save.setMinimumHeight(32)
        self.btn_save.setMaximumWidth(80)
        self.btn_save.clicked.connect(self._save_transcript)
        shortcut_row.addWidget(self.btn_save)
        outer.addLayout(shortcut_row)

        # ── Conversation ─────────────────────────────────────────────
        self.conversation = ConversationView()
        outer.addWidget(self.conversation, stretch=1)

        # ── Input row ────────────────────────────────────────────────
        input_row = QHBoxLayout()
        self.input_text = _ChatInput()
        self.input_text.setPlaceholderText(
            "Ask about the session… (Enter to send, Shift+Enter for newline)")
        self.input_text.setMaximumHeight(90)
        self.input_text.submitted.connect(self._on_send_clicked)
        input_row.addWidget(self.input_text, stretch=1)

        self.btn_send = QPushButton("Send")
        self.btn_send.setMinimumHeight(48)
        self.btn_send.setMinimumWidth(90)
        self.btn_send.clicked.connect(self._on_send_clicked)
        input_row.addWidget(self.btn_send)
        outer.addLayout(input_row)

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
    # File pickers
    # ------------------------------------------------------------------

    def _browse_context(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose experiment context file",
            self.le_context.text() or "",
            "Context files (*.toml *.json *.txt);;All files (*)",
        )
        if path:
            self.le_context.setText(path)

    def _browse_history(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose saved scan HDF5 file",
            self.le_history.text() or "",
            "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)",
        )
        if path:
            self.le_history.setText(path)
            self._sync_history_file()

    def _sync_history_file(self) -> None:
        if self.pv_tools is not None:
            self.pv_tools.set_history_file(self.le_history.text().strip() or None)

    # ------------------------------------------------------------------
    # Backend / controller construction
    # ------------------------------------------------------------------

    def _build_backend(self):
        cfg = dict(getattr(app_settings, 'SESSION_ANALYSIS', {}) or {})
        if self.rb_argo.isChecked():
            cfg['BACKEND'] = 'argo'
            cfg['ARGO_MODEL'] = (self.le_model.text().strip()
                                 or cfg.get('ARGO_MODEL', 'claudesonnet46'))
        else:
            cfg['BACKEND'] = 'ollama'
            cfg['OLLAMA_MODEL'] = (self.le_model.text().strip()
                                   or cfg.get('OLLAMA_MODEL', 'llama3.2'))
        return make_backend(cfg)

    def _build_analyzer(self, backend) -> SessionAnalyzer:
        context = ExperimentContext(self.le_context.text().strip() or None)
        return SessionAnalyzer(
            pva_reader=self.reader, backend=backend, context=context,
            n_snapshots=int(self.sb_snapshots.value()),
        )

    def _ensure_controller(self, backend) -> None:
        if self.controller is not None:
            return
        self.pv_tools = PvTools(self.reader, app_settings)
        self._sync_history_file()
        self._analyzer = self._build_analyzer(backend)
        self.session_tools = SessionTools(self.reader, self._analyzer)
        registry = ToolRegistry([self.pv_tools, self.session_tools])
        rounds = int((getattr(app_settings, 'CHAT_TOOLS', {}) or {}).get('MAX_TOOL_ROUNDS', 5))
        self.controller = ChatController(
            pva_reader=self.reader, backend=backend,
            tool_registry=registry, max_tool_rounds=rounds,
        )

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def _on_send_clicked(self) -> None:
        text = self.input_text.toPlainText().strip()
        if not text:
            return
        if not self._begin_turn():
            return
        self.input_text.clear()
        self.conversation.add_user(text)
        controller = self.controller
        self._start_worker(lambda on_event: controller.send_user_message(text, on_event))

    def _on_summarize_clicked(self) -> None:
        if not self._begin_turn():
            return
        self.conversation.add_user("[Summarize session]")
        controller, analyzer = self.controller, self._analyzer
        self._start_worker(
            lambda on_event: controller.inject_session_analysis_prompt(analyzer, on_event))

    def _begin_turn(self) -> bool:
        """Validate + (re)build backend/analyzer. Returns False if a turn can't start."""
        if self._worker is not None and self._worker.isRunning():
            self.lbl_status.setText("Busy — wait for the current turn to finish.")
            return False
        if self.reader is None:
            self.conversation.add_error("Start Live View before chatting.")
            return False
        try:
            backend = self._build_backend()
        except Exception as e:
            self.conversation.add_error(f"Backend not ready: {e}")
            return False
        self._ensure_controller(backend)
        self.controller.set_backend(backend)
        self._analyzer = self._build_analyzer(backend)
        self.session_tools.set_analyzer(self._analyzer)
        self._sync_history_file()
        return True

    def _start_worker(self, runner) -> None:
        self._set_busy(True)
        self.lbl_status.setText(f"Calling {self.controller.backend.name}…")
        self._worker = _ChatTurnWorker(runner, parent=self)
        self._worker.assistant_text.connect(self._on_assistant_text)
        self._worker.tool_requested.connect(self._on_tool_requested)
        self._worker.tool_completed.connect(self._on_tool_completed)
        self._worker.turn_error.connect(self._on_turn_error)
        self._worker.turn_done.connect(self._on_turn_done)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.start()

    # -- event slots (GUI thread) --

    def _on_assistant_text(self, text: str) -> None:
        self.conversation.add_assistant(text)

    def _on_tool_requested(self, name: str, arguments: dict) -> None:
        bubble = self.conversation.add_tool(name, arguments)
        self._pending_tools.append(bubble)
        self.lbl_status.setText(f"Running tool: {name}…")

    def _on_tool_completed(self, name: str, result: dict) -> None:
        if self._pending_tools:
            self._pending_tools.popleft().set_result(result)

    def _on_turn_error(self, msg: str) -> None:
        self.conversation.add_error(msg)
        self.lbl_status.setText("Error")

    def _on_turn_done(self, rounds_used: int, elapsed_s: float) -> None:
        tail = f" · {rounds_used} tool round(s)" if rounds_used else ""
        self.lbl_status.setText(f"Done in {elapsed_s:.1f}s{tail}")

    def _cleanup_worker(self) -> None:
        self._set_busy(False)
        self._pending_tools.clear()
        self._worker = None

    def _set_busy(self, busy: bool) -> None:
        self.btn_send.setEnabled(not busy)
        self.btn_summarize.setEnabled(not busy)
        self.input_text.setReadOnly(busy)

    # ------------------------------------------------------------------
    # Reset / preview / transcript
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        if self.controller is not None:
            self.controller.reset()
        self.conversation.clear()
        self._pending_tools.clear()
        self.lbl_status.setText("Conversation reset")

    def _on_preview_clicked(self) -> None:
        if self.reader is None:
            self.conversation.add_error("Start Live View before previewing.")
            return
        try:
            # build_prompt() never touches the backend, so a None backend is
            # fine here — rebuild against the current snapshot count / context.
            analyzer = SessionAnalyzer(
                pva_reader=self.reader, backend=None,
                context=ExperimentContext(self.le_context.text().strip() or None),
                n_snapshots=int(self.sb_snapshots.value()),
            )
            prompt = analyzer.build_prompt()
        except Exception as e:
            self.conversation.add_error(f"Preview failed: {e}")
            return
        self._show_prompt_dialog(prompt)

    def _show_prompt_dialog(self, prompt: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Summary prompt preview")
        dlg.resize(720, 560)
        lay = QVBoxLayout(dlg)
        view = QTextEdit()
        view.setReadOnly(True)
        view.setPlainText(prompt)
        view.setFont(QFont("monospace"))
        lay.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec_()

    def _copy_transcript(self) -> None:
        text = self.conversation.to_plain_text()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.lbl_status.setText("Transcript copied to clipboard")

    def _save_transcript(self) -> None:
        text = self.conversation.to_plain_text()
        if not text:
            QMessageBox.information(self, "Nothing to save",
                                   "Have a conversation first.")
            return
        default = f"session_chat_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", default,
            "Markdown (*.md);;Text files (*.txt);;All files (*)",
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
        """Hot-swap the reader (launcher calls this when Live View is
        (re)started). Forwards to the controller and the tools, which each
        hold their own reader reference."""
        self.reader = pva_reader
        if self.pv_tools is not None:
            self.pv_tools.set_reader(pva_reader)
        if self.session_tools is not None:
            self.session_tools.set_reader(pva_reader)
        if self.controller is not None:
            self.controller.set_reader(pva_reader)

    def closeEvent(self, event) -> None:
        # Don't block close on a running worker — Qt drops queued signals to a
        # destroyed receiver, so the turn runs to completion and is discarded.
        super().closeEvent(event)
