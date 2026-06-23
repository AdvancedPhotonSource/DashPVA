"""Standalone PyQt5 window for the beamline-analysis agent.

A top-level chat window that drives :class:`~dashpva.agent.sdk_agent.SdkAgent`
over a loaded saved scan. The chat widgets (``ConversationView`` + the tool /
reasoning bubbles + the Enter-to-send input) are adapted from
``viewer/session_analysis_window.py`` — copied here so the embedded window stays
untouched while the agent package decouples from the viewer.

Async/Qt threading model (the careful part): the Claude Agent SDK is async, so a
single long-lived :class:`_AgentThread` owns an asyncio event loop for the
window's lifetime. It connects one ``SdkAgent`` (preserving conversation memory
across turns), then services questions posted from the GUI thread via
``loop.call_soon_threadsafe``. Each normalized ``AgentEvent`` is re-emitted as a
Qt signal, which Qt marshals back to the GUI thread for rendering. The
``argo-proxy`` sidecar is started inside the thread (off the GUI thread) and
stopped by the window on close.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Optional

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import dashpva.settings as app_settings
from dashpva.agent.proxy_manager import ProxyError, ProxyManager
from dashpva.agent.saved_scan_reader import SavedScanReader
from dashpva.agent.sdk_agent import DEFAULT_MODEL, SdkAgent

# ======================================================================
# Async agent worker — one persistent asyncio loop for the window lifetime
# ======================================================================


class _AgentThread(QThread):
    """Owns an asyncio loop running one connected :class:`SdkAgent`.

    Questions are posted from the GUI thread via :meth:`submit`; each
    :class:`AgentEvent` comes back as ``sig_event``. The agent connection (and
    thus conversation history) persists until :meth:`shutdown`.
    """

    sig_status = pyqtSignal(str)
    sig_ready = pyqtSignal()
    sig_event = pyqtSignal(object)   # AgentEvent
    sig_turn_done = pyqtSignal()
    sig_error = pyqtSignal(str)
    sig_permission = pyqtSignal(str, dict, object)  # tool_name, input, Future[bool]

    def __init__(self, reader, settings, proxy: ProxyManager, *,
                 model: str, vision: bool, enable_builtin_tools: bool, parent=None):
        super().__init__(parent)
        self.reader = reader
        self.settings = settings
        self.proxy = proxy
        self.model = model
        self.vision = vision
        self.enable_builtin_tools = enable_builtin_tools
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._inbox: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    # -- thread body --

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(self._serve())
        try:
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass  # clean stop requested via request_stop()
        except Exception as e:
            self.sig_error.emit(f"{type(e).__name__}: {e}")
        finally:
            try:
                self._loop.close()
            finally:
                self._loop = None
                self._task = None

    async def _serve(self) -> None:
        self._inbox = asyncio.Queue()
        self.sig_status.emit("Starting argo-proxy / connecting to Argo…")
        try:
            base_url = await asyncio.to_thread(self.proxy.ensure_running)
        except ProxyError as e:
            self.sig_error.emit(str(e))
            return
        # The SdkAgent connection persists across turns; the loop is stopped by
        # cancelling this task (request_stop), which unwinds through __aexit__.
        async with SdkAgent(
            self.reader, self.settings, base_url=base_url,
            model=self.model, vision_enabled=self.vision,
            enable_builtin_tools=self.enable_builtin_tools,
            can_use_tool=self._permission,
            auth_token=self.proxy.user,
        ) as agent:
            self.sig_ready.emit()
            while True:
                question = await self._inbox.get()
                try:
                    async for event in agent.ask(question):
                        self.sig_event.emit(event)
                except asyncio.CancelledError:
                    raise  # stop requested mid-turn — unwind cleanly
                except Exception as e:
                    self.sig_error.emit(f"{type(e).__name__}: {e}")
                finally:
                    self.sig_turn_done.emit()

    async def _permission(self, tool_name: str, tool_input: dict, context):
        """SDK permission callback (runs in this thread's loop). Bridges to a GUI
        confirmation dialog via a thread-safe Future and awaits the user's answer."""
        fut: Future = Future()
        self.sig_permission.emit(tool_name, dict(tool_input or {}), fut)
        try:
            allowed = await asyncio.wrap_future(fut)
        except Exception:
            allowed = False
        if allowed:
            return PermissionResultAllow()
        return PermissionResultDeny(message=f"User declined to run {tool_name}.")

    # -- cross-thread control (called on the GUI thread) --

    def submit(self, question: str) -> None:
        loop, inbox = self._loop, self._inbox
        if loop is not None and inbox is not None:
            loop.call_soon_threadsafe(inbox.put_nowait, question)

    def request_stop(self) -> None:
        """Cleanly stop the loop by cancelling the serve task — works whether the
        thread is idle (awaiting a question) or mid-turn (streaming a response).
        Cancellation unwinds through SdkAgent.__aexit__ so the connection closes."""
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)


# ======================================================================
# Conversation widgets (adapted from viewer/session_analysis_window.py)
# ======================================================================


def _truncate(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + "…"


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

        arg_summary = ", ".join(f"{k}={v}" for k, v in arguments.items())
        if len(arg_summary) > 80:
            arg_summary = arg_summary[:80] + "…"

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
        if isinstance(result, dict) and "error" in result:
            self._toggle.setStyleSheet("QToolButton { border:none; color:#b04040; }")
            self._toggle.setText(self._toggle.text() + "  — error")
        self._render_details()

    def _on_toggle(self, checked: bool) -> None:
        self._details.setVisible(checked)
        self._toggle.setText(("▾" if checked else "▸") + self._toggle.text()[1:])

    def _render_details(self) -> None:
        parts = [f"args: {json.dumps(self._arguments, default=str)}"]
        if self._result is not None:
            parts.append(f"result: {json.dumps(self._result, default=str, indent=2)}")
        self._details.setText("\n".join(parts))

    def transcript_line(self) -> str:
        res = json.dumps(self._result, default=str) if self._result is not None else "(pending)"
        return f"[tool {self._name}] args={json.dumps(self._arguments, default=str)} -> {res}"


class _ThinkingBubble(QFrame):
    """Collapsible grey row for one native extended-thinking block from Claude.

    Reasoning is the SDK's job now (no plan/finding/hypothesis scaffolding) — this
    just surfaces the model's thinking, collapsed by default."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = str(text)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "_ThinkingBubble { background:#fbfbfd; border:1px solid #e0e0e8; "
            "border-radius:6px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        self._toggle = QToolButton()
        self._toggle.setStyleSheet("QToolButton { border:none; color:#777777; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toggle.setCheckable(True)
        self._toggle.setText(f"▸  thinking · {_truncate(self._text.replace(chr(10), ' '), 70)}")
        self._toggle.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle)

        self._details = QLabel(self._text)
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._details.setStyleSheet("color:#444; font-family:monospace;")
        self._details.setVisible(False)  # collapsed by default
        lay.addWidget(self._details)

    def _on_toggle(self, checked: bool) -> None:
        self._details.setVisible(checked)
        self._toggle.setText(("▾" if checked else "▸") + self._toggle.text()[1:])

    def transcript_line(self) -> str:
        return f"[thinking] {self._text.replace(chr(10), ' | ')}"


class ConversationView(QScrollArea):
    """Vertical stack of chat bubbles, auto-scrolled to the bottom on additions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._vbox = QVBoxLayout(self._host)
        self._vbox.setContentsMargins(8, 8, 8, 8)
        self._vbox.setSpacing(8)
        self._vbox.addStretch(1)
        self.setWidget(self._host)
        self._entries: list = []

    def add_user(self, text: str) -> None:
        self._add_text_bubble("You", text, bg="#e3f0ff", align_right=True)
        self._entries.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self._add_text_bubble("Assistant", text, bg="#f3f3f0", align_right=False)
        self._entries.append(("assistant", text))

    def add_error(self, text: str) -> None:
        self._add_text_bubble("Error", text, bg="#fbe6e6", align_right=False,
                              title_color="#b04040")
        self._entries.append(("error", text))

    def add_tool(self, name: str, arguments: dict) -> _ToolBubble:
        bubble = _ToolBubble(name, arguments)
        self._insert_widget(bubble)
        self._entries.append(("tool", bubble))
        return bubble

    def add_thinking(self, text: str) -> None:
        bubble = _ThinkingBubble(text)
        self._insert_widget(bubble)
        self._entries.append(("reasoning", bubble))

    def clear(self) -> None:
        while self._vbox.count() > 1:
            item = self._vbox.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._entries.clear()

    def to_plain_text(self) -> str:
        lines = []
        for role, payload in self._entries:
            if role in ("tool", "reasoning"):
                lines.append(payload.transcript_line())
            else:
                label = {"user": "You", "assistant": "Assistant",
                         "error": "Error"}.get(role, role)
                lines.append(f"{label}: {payload}")
        return "\n\n".join(lines)

    def _add_text_bubble(self, title: str, text: str, *, bg: str,
                         align_right: bool, title_color: str = "#666") -> None:
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
        def _do():
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        QTimer.singleShot(0, _do)


# ======================================================================
# Main window
# ======================================================================


class AgentWindow(QMainWindow):
    """Standalone beamline-analysis agent window over a saved scan.

    Load a ``.h5``; the window starts the proxy + a persistent agent connection,
    then each message runs one investigation turn through the Claude Agent SDK.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, vision: bool = False,
                 full_tools: bool = True, settings=app_settings, parent=None):
        super().__init__(parent, Qt.Window)
        self.settings = settings
        self.reader = None
        self.proxy = ProxyManager()
        self._thread: Optional[_AgentThread] = None
        self._pending: dict[str, _ToolBubble] = {}   # tool_id -> bubble
        self._turn_t0 = 0.0

        self.setWindowTitle("DashPVA — Beamline Analysis Agent")
        self.resize(900, 780)
        self._build_ui(model=model, vision=vision, full_tools=full_tools)
        self._set_chat_enabled(False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, *, model: str, vision: bool, full_tools: bool) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Top control row.
        top = QHBoxLayout()
        self.btn_load = QPushButton("Load scan (.h5)…")
        self.btn_load.clicked.connect(self._on_load_clicked)
        top.addWidget(self.btn_load)

        self.lbl_scan = QLabel("No scan loaded")
        self.lbl_scan.setStyleSheet("color:#555;")
        top.addWidget(self.lbl_scan, stretch=1)

        top.addWidget(QLabel("Model:"))
        self.le_model = QLineEdit(model)
        self.le_model.setMaximumWidth(160)
        self.le_model.setToolTip("Argo model id (applied when a scan is loaded).")
        top.addWidget(self.le_model)

        self.cb_full_tools = QCheckBox("Full tools")
        self.cb_full_tools.setChecked(full_tools)
        self.cb_full_tools.setToolTip(
            "Give the agent Claude Code's built-in tools (Bash, file read/write, "
            "web) in addition to the domain tools. Bash/Write/Edit/etc. still ask "
            "for your confirmation each time. Unchecked = read-only domain tools.")
        top.addWidget(self.cb_full_tools)

        self.cb_vision = QCheckBox("Vision")
        self.cb_vision.setChecked(vision)
        self.cb_vision.setToolTip("Allow the describe_frame vision tool.")
        top.addWidget(self.cb_vision)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setMaximumWidth(80)
        self.btn_reset.setToolTip("Clear the conversation and reconnect a fresh agent.")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        top.addWidget(self.btn_reset)
        outer.addLayout(top)

        # Conversation.
        self.conversation = ConversationView()
        outer.addWidget(self.conversation, stretch=1)

        # Input row.
        input_row = QHBoxLayout()
        self.input_text = _ChatInput()
        self.input_text.setPlaceholderText(
            "Load a scan, then ask… (Enter to send, Shift+Enter for newline)")
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

        self.status_bar = QStatusBar(self)
        self.lbl_status = QLabel("Load a scan to begin.")
        self.status_bar.addWidget(self.lbl_status, 1)
        self.setStatusBar(self.status_bar)

    # ------------------------------------------------------------------
    # Scan loading / agent lifecycle
    # ------------------------------------------------------------------

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a saved scan HDF5 file", "",
            "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)")
        if path:
            self.load_scan(path)

    def load_scan(self, path: str) -> None:
        """Open a scan file and (re)start the agent connection over it."""
        try:
            reader = SavedScanReader(path, settings=self.settings)
        except Exception as e:
            QMessageBox.critical(self, "Could not open scan", f"{type(e).__name__}: {e}")
            return
        self.reader = reader
        name = Path(path).name
        self.lbl_scan.setText(
            f"{name} — {reader.frames_received} frames, shape {reader.shape}, "
            f"{len(reader.feature_vector_cache)} features, "
            f"PVs: {', '.join(sorted(reader.cached_ca)) or 'none'}")
        self.conversation.clear()
        self._start_agent()

    def _start_agent(self) -> None:
        self._stop_agent()  # tear down any prior connection (fresh conversation)
        self._set_chat_enabled(False)
        self._thread = _AgentThread(
            self.reader, self.settings, self.proxy,
            model=self.le_model.text().strip() or DEFAULT_MODEL,
            vision=self.cb_vision.isChecked(),
            enable_builtin_tools=self.cb_full_tools.isChecked(), parent=self)
        self._thread.sig_status.connect(self.lbl_status.setText)
        self._thread.sig_ready.connect(self._on_agent_ready)
        self._thread.sig_event.connect(self._on_event)
        self._thread.sig_turn_done.connect(self._on_turn_done)
        self._thread.sig_error.connect(self._on_error)
        self._thread.sig_permission.connect(self._on_permission)
        self._thread.start()

    def _stop_agent(self) -> None:
        """Fully join the agent thread before dropping it. A QThread destroyed
        while still running aborts the whole process, so we escalate until it is
        truly finished: clean cancel → wait → terminate as a last resort."""
        th = self._thread
        self._thread = None
        if th is not None:
            th.request_stop()
            if not th.wait(10000):
                th.terminate()       # last resort; never leave it running
                th.wait(3000)
        self._pending.clear()

    def _on_agent_ready(self) -> None:
        self._set_chat_enabled(True)
        self.lbl_status.setText("Ready. Ask about the scan.")
        self.input_text.setFocus()

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def _on_send_clicked(self) -> None:
        text = self.input_text.toPlainText().strip()
        if not text or self._thread is None:
            return
        if not self.btn_send.isEnabled():
            return
        self.input_text.clear()
        self.conversation.add_user(text)
        self._pending.clear()
        self._turn_t0 = time.time()
        self._set_busy(True)
        self.lbl_status.setText("Investigating…")
        self._thread.submit(text)

    def _on_event(self, ev) -> None:
        kind = ev.kind
        if kind == "assistant_text":
            self.conversation.add_assistant(ev.text)
        elif kind == "thinking":
            self.conversation.add_thinking(ev.text)
        elif kind == "tool_call":
            self._on_tool_call(ev)
        elif kind == "tool_result":
            self._on_tool_result(ev)
        elif kind == "result":
            self._on_result(ev)

    def _on_tool_call(self, ev) -> None:
        bubble = self.conversation.add_tool(ev.tool_name, ev.tool_input or {})
        if ev.tool_id:
            self._pending[ev.tool_id] = bubble
        self.lbl_status.setText(f"Running tool: {ev.tool_name}…")

    def _on_tool_result(self, ev) -> None:
        result = _parse_result(ev.tool_result)
        bubble = self._pending.pop(ev.tool_id, None)
        if bubble is not None:
            bubble.set_result(result)

    def _on_result(self, ev) -> None:
        info = ev.info or {}
        bits = []
        if info.get("num_turns") is not None:
            bits.append(f"{info['num_turns']} turns")
        if info.get("total_cost_usd") is not None:
            bits.append(f"${info['total_cost_usd']:.4f}")
        elapsed = time.time() - self._turn_t0 if self._turn_t0 else None
        if elapsed is not None:
            bits.append(f"{elapsed:.1f}s")
        suffix = f" ({', '.join(bits)})" if bits else ""
        self.lbl_status.setText(("Error" if ev.is_error else "Done") + suffix)

    def _on_turn_done(self) -> None:
        self._set_busy(False)
        self.input_text.setFocus()

    def _on_permission(self, tool_name: str, tool_input: dict, fut) -> None:
        """Confirm a Claude Code built-in tool call (Bash/Write/Edit/…). Runs on
        the GUI thread; resolves the agent thread's Future with the user's answer."""
        try:
            detail = json.dumps(tool_input, indent=2, default=str)
        except Exception:
            detail = str(tool_input)
        if len(detail) > 1500:
            detail = detail[:1500] + "\n…"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Confirm tool")
        box.setText(f"Allow the agent to run <b>{tool_name}</b>?")
        box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        approved = box.exec_() == QMessageBox.Yes
        if not fut.done():
            fut.set_result(approved)
        self.lbl_status.setText(
            f"{'Approved' if approved else 'Declined'} {tool_name}")

    def _on_error(self, msg: str) -> None:
        self.conversation.add_error(msg)
        self.lbl_status.setText("Error")
        self._set_busy(False)

    def _on_reset_clicked(self) -> None:
        if self.reader is None:
            return
        self.conversation.clear()
        self._start_agent()

    # ------------------------------------------------------------------
    # Helpers / lifecycle
    # ------------------------------------------------------------------

    def _set_chat_enabled(self, enabled: bool) -> None:
        self.input_text.setEnabled(enabled)
        self.btn_send.setEnabled(enabled)

    def _set_busy(self, busy: bool) -> None:
        self.btn_send.setEnabled(not busy)
        self.input_text.setReadOnly(busy)
        self.btn_load.setEnabled(not busy)
        self.btn_reset.setEnabled(not busy)

    def closeEvent(self, event) -> None:
        self._stop_agent()
        try:
            self.proxy.stop()
        except Exception:
            pass
        super().closeEvent(event)


def _parse_result(text: str):
    """Tool-result content is a JSON string; parse it for display, else wrap raw."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


# Convenience for ``python -m dashpva.agent`` / app.py.
def launch(scan: str | None = None, *, model: str = DEFAULT_MODEL,
           vision: bool = False, full_tools: bool = True) -> int:
    app = QApplication.instance() or QApplication([])
    win = AgentWindow(model=model, vision=vision, full_tools=full_tools)
    if scan:
        win.load_scan(scan)
    win.show()
    return app.exec_()