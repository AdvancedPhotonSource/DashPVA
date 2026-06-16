"""
ChatController — the LLM-call → tool-execute → LLM-call loop.

The window drives one user turn at a time::

    controller.send_user_message(text, on_event)

where ``on_event(ControllerEvent)`` is called for each step (assistant text,
a tool call requested, a tool result, an error, done). The controller is
GUI-agnostic — the window adapts these events to Qt signals via a worker
thread. This mirrors EAA's serial tool executor (one tool call at a time, in
order) without taking EAA as a dependency; see
:mod:`dashpva.analysis.tools.base` for the tool layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from dashpva.analysis.llm_backend import LLMBackend
from dashpva.analysis.tools.base import ToolRegistry

CHAT_SYSTEM_PROMPT = (
    "You are an expert beamline scientist embedded in a live synchrotron "
    "data-acquisition viewer. You can call tools to read EPICS process "
    "variables (live, or historical by frame id or timestamp) and to inspect "
    "cached session features. Prefer calling tools over guessing. If you are "
    "unsure what PVs are available, call list_known_pvs first. Historical "
    "lookups target the live in-memory cache (source='live') by default; only "
    "use source='h5' if the user has loaded a history file. When a historical "
    "value is not an exact match (exact=false), say so and report the "
    "matched_frame_id you actually used. Keep responses concise and cite exact "
    "numbers."
)

EventKind = Literal[
    'assistant_text', 'tool_call_requested', 'tool_call_result', 'error', 'done'
]


@dataclass
class ControllerEvent:
    kind: EventKind
    text: str = ''                          # assistant_text / error
    tool_name: str = ''                     # tool_*
    tool_call_id: str = ''
    tool_arguments: Optional[dict] = None   # tool_call_requested
    tool_result: Optional[dict] = None      # tool_call_result
    rounds_used: int = 0                    # done


OnEvent = Callable[[ControllerEvent], None]


class ChatController:
    """Owns chat history and the tool-calling loop for one conversation."""

    def __init__(self, *, pva_reader, backend: LLMBackend,
                 tool_registry: ToolRegistry,
                 system_prompt: str = CHAT_SYSTEM_PROMPT,
                 max_tool_rounds: int = 5):
        self.reader = pva_reader
        self.backend = backend
        self.tools = tool_registry
        self.system_prompt = system_prompt
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.messages: list[dict] = []
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history but keep backend/tools/system prompt."""
        self.messages = (
            [{'role': 'system', 'content': self.system_prompt}]
            if self.system_prompt else []
        )

    def set_backend(self, backend: LLMBackend) -> None:
        self.backend = backend

    def set_reader(self, pva_reader) -> None:
        self.reader = pva_reader

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def inject_session_analysis_prompt(self, analyzer, on_event: OnEvent) -> None:
        """Send the existing six-section summary prompt as the next user turn.

        Usable at any point in the conversation — this is the "Summarize
        session" shortcut (the artist formerly known as the Analyze button).
        """
        try:
            text = analyzer.build_prompt()
        except Exception as e:
            on_event(ControllerEvent('error', text=f'Failed to build summary prompt: {e}'))
            return
        self.send_user_message(text, on_event)

    def send_user_message(self, text: str, on_event: OnEvent) -> None:
        """Run one user turn: append the message, then loop LLM↔tools until the
        model returns a plain text answer or max_tool_rounds is hit.

        ``on_event`` is invoked synchronously for each step, on the calling
        thread. The window runs this inside a worker QThread.
        """
        self.messages.append({'role': 'user', 'content': text})
        tool_schemas = self.tools.openai_schemas() if self.tools else []

        rounds = 0
        while rounds < self.max_tool_rounds:
            try:
                result = self.backend.chat(self.messages, tools=tool_schemas or None)
            except Exception as e:
                on_event(ControllerEvent('error', text=f'{type(e).__name__}: {e}'))
                return

            calls = result.get('tool_calls') or []
            # Append the assistant turn verbatim — OpenAI/Argo require the
            # assistant tool-call message to precede its tool results.
            self.messages.append({
                k: v for k, v in result.items()
                if k in ('role', 'content', 'tool_calls')
            })

            if result.get('content'):
                on_event(ControllerEvent('assistant_text', text=result['content']))

            if not calls:
                on_event(ControllerEvent('done', rounds_used=rounds))
                return

            for call in calls:
                cid = call.get('id', '')
                cname = call.get('name', '')
                cargs = call.get('arguments') or {}
                on_event(ControllerEvent(
                    'tool_call_requested',
                    tool_name=cname, tool_call_id=cid, tool_arguments=cargs))
                tool_result = self.tools.call(cname, cargs)
                on_event(ControllerEvent(
                    'tool_call_result',
                    tool_name=cname, tool_call_id=cid, tool_result=tool_result))
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': cid,
                    'name': cname,
                    'content': json.dumps(tool_result, default=str),
                })
            rounds += 1

        on_event(ControllerEvent(
            'error',
            text=f'Reached max_tool_rounds ({self.max_tool_rounds}) without a '
                 f'final answer. The model may be stuck in a tool loop.'))
