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

# Deliberate mode: a rigorous plan -> investigate -> answer protocol with an
# enforced final-answer contract. Backend-agnostic — it shapes behavior even on
# models without native reasoning, so quality improves on ollama/GPT as well as
# on Claude (where it composes with native extended thinking).
DELIBERATE_SYSTEM_PROMPT = (
    "You are an expert beamline scientist embedded in a live synchrotron "
    "data-acquisition viewer, assisting a researcher who needs RIGOROUS, "
    "evidence-backed analysis. Work like a careful investigator, not a "
    "describer.\n\n"
    "PROTOCOL — follow it every turn:\n"
    "1. PLAN: call record_plan once with 1-3 concrete steps before other tools.\n"
    "2. INVESTIGATE: gather evidence with tools. NEVER state a number you did "
    "not read from a tool result this turn. To inspect how something evolves, "
    "use get_feature_timeseries / get_feature_statistics / detect_anomalies; to "
    "relate detector behavior to beamline conditions, use correlate_series; to "
    "examine a specific frame, use compute_radial_profile / fit_peak / "
    "get_roi_statistics / check_saturation; when the numbers are ambiguous and a "
    "vision tool is available, look at the frame with describe_frame. Record "
    "confirmed facts with record_finding (cite the exact tool/frame), and record "
    "anything unconfirmed with note_hypothesis (state the test that would settle "
    "it).\n"
    "3. If a tool errors or returns no data, say so explicitly — do not paper "
    "over gaps. Be honest about what you do not know.\n"
    "4. ANSWER: end every substantive reply with exactly this structure:\n"
    "## Answer\n"
    "<direct answer; cite exact numbers and the tool/frame each came from>\n"
    "## Confidence\n"
    "<high|medium|low> - <one sentence why>\n"
    "## What I did not verify / would need to check\n"
    "<bullets: missing data, assumptions, PVs/frames not inspected>\n\n"
    "Historical lookups default to the live in-memory cache (source='live'); use "
    "source='h5' only if a history file is loaded. When a historical value is not "
    "an exact match (exact=false), say so and report the matched_frame_id used."
)

EventKind = Literal[
    'assistant_text', 'tool_call_requested', 'tool_call_result', 'error', 'done',
    # additive reasoning-surface events (older UIs silently drop unknown kinds):
    'plan', 'thinking', 'finding', 'hypothesis', 'critique',
]

# Scratchpad tool name -> the typed ControllerEvent kind it surfaces.
REASONING_TOOL_KINDS = {
    'record_plan': 'plan',
    'record_finding': 'finding',
    'note_hypothesis': 'hypothesis',
}


@dataclass
class ControllerEvent:
    kind: EventKind
    text: str = ''                          # assistant_text / error / thinking / critique
    tool_name: str = ''                     # tool_*
    tool_call_id: str = ''
    tool_arguments: Optional[dict] = None   # tool_call_requested
    tool_result: Optional[dict] = None      # tool_call_result
    rounds_used: int = 0                    # done
    detail: Optional[dict] = None           # plan / finding / hypothesis payload


OnEvent = Callable[[ControllerEvent], None]


class ChatController:
    """Owns chat history and the tool-calling loop for one conversation."""

    def __init__(self, *, pva_reader, backend: LLMBackend,
                 tool_registry: ToolRegistry,
                 system_prompt: str = CHAT_SYSTEM_PROMPT,
                 max_tool_rounds: int = 5,
                 mode: str = 'standard',
                 max_tool_rounds_deliberate: int = 12,
                 enable_self_critique: bool = False):
        self.reader = pva_reader
        self.backend = backend
        self.tools = tool_registry
        self.mode = mode if mode in ('standard', 'deliberate') else 'standard'
        # Track whether the caller supplied a custom prompt; if not, deliberate
        # mode swaps in DELIBERATE_SYSTEM_PROMPT.
        self._custom_prompt = system_prompt not in (CHAT_SYSTEM_PROMPT, '', None)
        self.system_prompt = self._prompt_for_mode(system_prompt)
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.max_tool_rounds_deliberate = max(1, int(max_tool_rounds_deliberate))
        self.enable_self_critique = bool(enable_self_critique)
        self.messages: list[dict] = []
        if self.system_prompt:
            self.messages.append({'role': 'system', 'content': self.system_prompt})

    def _prompt_for_mode(self, base_prompt: str) -> str:
        if self._custom_prompt:
            return base_prompt
        return DELIBERATE_SYSTEM_PROMPT if self.mode == 'deliberate' else CHAT_SYSTEM_PROMPT

    @property
    def _effective_max_rounds(self) -> int:
        return (self.max_tool_rounds_deliberate
                if self.mode == 'deliberate' else self.max_tool_rounds)

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

    def set_mode(self, mode: str) -> None:
        """Switch standard/deliberate. The system prompt is only swapped when the
        conversation has not started yet (history is empty apart from the system
        message); mid-conversation we keep the existing prompt and only the round
        cap changes, to avoid rewriting an in-flight system message."""
        if mode not in ('standard', 'deliberate') or mode == self.mode:
            return
        self.mode = mode
        if self._custom_prompt:
            return
        history_started = any(m.get('role') != 'system' for m in self.messages)
        if not history_started:
            self.system_prompt = self._prompt_for_mode(self.system_prompt)
            self.messages = (
                [{'role': 'system', 'content': self.system_prompt}]
                if self.system_prompt else []
            )

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
        self._reset_turn_budgets()

        max_rounds = self._effective_max_rounds
        rounds = 0
        while rounds < max_rounds:
            try:
                result = self.backend.chat(self.messages, tools=tool_schemas or None)
            except Exception as e:
                on_event(ControllerEvent('error', text=f'{type(e).__name__}: {e}'))
                return

            calls = result.get('tool_calls') or []
            # Append the assistant turn (in the backend's normalized
            # ``{id, name, arguments: dict}`` tool-call shape) before its tool
            # results — every provider requires the tool-call message to precede
            # them. Each backend re-serializes these tool_calls into its own wire
            # format at send time (OpenAI/Argo want JSON-string arguments, ollama
            # wants a dict), so we store the normalized shape verbatim here.
            # ``_provider_blocks`` (when a backend supplies it, e.g. Anthropic
            # native thinking) is carried verbatim so thinking blocks round-trip
            # across tool rounds; other backends ignore it.
            self.messages.append({
                k: v for k, v in result.items()
                if k in ('role', 'content', 'tool_calls', '_provider_blocks')
            })

            # Surface native reasoning ("thinking") before the visible answer.
            if result.get('thinking'):
                on_event(ControllerEvent('thinking', text=str(result['thinking'])))

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
                # Scratchpad tools also surface as typed reasoning events so the
                # plan/findings/hypotheses appear as first-class artifacts.
                kind = REASONING_TOOL_KINDS.get(cname)
                if kind and isinstance(tool_result, dict) and 'error' not in tool_result:
                    on_event(ControllerEvent(
                        kind, detail=tool_result,
                        tool_name=cname, tool_call_id=cid))
                self.messages.append({
                    'role': 'tool',
                    'tool_call_id': cid,
                    'name': cname,
                    'content': json.dumps(tool_result, default=str),
                })
            rounds += 1

        # Round cap reached. In deliberate mode, close out gracefully: ask once
        # for a final answer (no tools) so the scientist gets a rigorous summary
        # of what was found rather than a bare failure. Standard mode keeps the
        # original error for backward compatibility.
        if self.mode == 'deliberate':
            self.messages.append({
                'role': 'user',
                'content': ('You have used all available investigation rounds. '
                            'Stop calling tools and give your final answer now, '
                            'following the required answer contract, based only on '
                            'evidence already gathered. State clearly what remains '
                            'unverified.'),
            })
            try:
                final = self.backend.chat(self.messages, tools=None)
            except Exception as e:
                on_event(ControllerEvent('error', text=f'{type(e).__name__}: {e}'))
                return
            self.messages.append({
                k: v for k, v in final.items()
                if k in ('role', 'content', '_provider_blocks')
            })
            if final.get('thinking'):
                on_event(ControllerEvent('thinking', text=str(final['thinking'])))
            on_event(ControllerEvent(
                'assistant_text', text=final.get('content') or '(no final answer)'))
            on_event(ControllerEvent('done', rounds_used=rounds))
            return

        on_event(ControllerEvent(
            'error',
            text=f'Reached max_tool_rounds ({max_rounds}) without a '
                 f'final answer. The model may be stuck in a tool loop.'))

    def _reset_turn_budgets(self) -> None:
        """Give tool instances a chance to reset per-turn budgets (e.g. rate
        limits on expensive frame/vision tools). Duck-typed so the registry and
        controller stay decoupled from any specific tool class."""
        if not self.tools:
            return
        for inst in self.tools.instances():
            reset = getattr(inst, 'reset_turn_budgets', None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    pass
