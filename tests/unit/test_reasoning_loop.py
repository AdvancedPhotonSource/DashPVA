"""Tests for deliberate mode, graceful close-out, scratchpad events, and native
thinking surfacing in ChatController."""

from __future__ import annotations

from dashpva.analysis.chat_controller import (
    CHAT_SYSTEM_PROMPT,
    DELIBERATE_SYSTEM_PROMPT,
    ChatController,
    ControllerEvent,
)
from dashpva.analysis.llm_backend import LLMBackend
from dashpva.analysis.tools.base import BaseTool, ToolRegistry, tool
from dashpva.analysis.tools.reasoning_tools import ReasoningTools


class _ScriptedBackend(LLMBackend):
    name = 'fake/scripted'

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({'messages': [dict(m) for m in messages], 'tools': tools})
        if self._responses:
            return self._responses.pop(0)
        return {'role': 'assistant', 'content': 'fallback final answer'}


class _EchoTool(BaseTool):
    @tool(description="Echo.")
    def echo(self, value: str = 'x') -> dict:
        return {'echoed': value}


def _collect():
    events: list[ControllerEvent] = []
    return events, events.append


class TestDeliberateMode:

    def test_deliberate_uses_deliberate_prompt(self):
        ctrl = ChatController(pva_reader=None, backend=_ScriptedBackend([]),
                              tool_registry=ToolRegistry([]), mode='deliberate')
        assert ctrl.messages[0]['content'] == DELIBERATE_SYSTEM_PROMPT

    def test_standard_default_unchanged(self):
        ctrl = ChatController(pva_reader=None, backend=_ScriptedBackend([]),
                              tool_registry=ToolRegistry([]))
        assert ctrl.messages[0]['content'] == CHAT_SYSTEM_PROMPT

    def test_custom_prompt_overrides_mode(self):
        ctrl = ChatController(pva_reader=None, backend=_ScriptedBackend([]),
                              tool_registry=ToolRegistry([]),
                              system_prompt='CUSTOM', mode='deliberate')
        assert ctrl.messages[0]['content'] == 'CUSTOM'

    def test_set_mode_before_history_swaps_prompt(self):
        ctrl = ChatController(pva_reader=None, backend=_ScriptedBackend([]),
                              tool_registry=ToolRegistry([]))
        ctrl.set_mode('deliberate')
        assert ctrl.messages[0]['content'] == DELIBERATE_SYSTEM_PROMPT


class TestGracefulCloseout:

    def test_deliberate_cap_emits_final_answer_not_error(self):
        loop = {'role': 'assistant', 'content': '',
                'tool_calls': [{'id': 'c', 'name': 'echo', 'arguments': {}}]}
        backend = _ScriptedBackend([loop, loop])  # always wants tools
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([_EchoTool()]),
                              mode='deliberate', max_tool_rounds_deliberate=2)
        events, on_event = _collect()
        ctrl.send_user_message('investigate', on_event)
        kinds = [e.kind for e in events]
        assert kinds[-2:] == ['assistant_text', 'done']
        assert 'error' not in kinds
        assert events[-2].text == 'fallback final answer'

    def test_standard_cap_still_errors(self):
        loop = {'role': 'assistant', 'content': '',
                'tool_calls': [{'id': 'c', 'name': 'echo', 'arguments': {}}]}
        backend = _ScriptedBackend([loop] * 10)
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([_EchoTool()]),
                              max_tool_rounds=2)
        events, on_event = _collect()
        ctrl.send_user_message('loop', on_event)
        assert events[-1].kind == 'error'
        assert 'max_tool_rounds' in events[-1].text


class TestScratchpadEvents:

    def test_record_plan_emits_plan_event(self):
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'p', 'name': 'record_plan',
                             'arguments': {'steps': ['a', 'b']}}]},
            {'role': 'assistant', 'content': 'done'},
        ])
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([ReasoningTools()]),
                              mode='deliberate')
        events, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        plans = [e for e in events if e.kind == 'plan']
        assert len(plans) == 1
        assert plans[0].detail == {'kind': 'plan', 'steps': ['a', 'b']}
        assert any(e.kind == 'tool_call_result' for e in events)

    def test_finding_and_hypothesis_events(self):
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [
                 {'id': 'f', 'name': 'record_finding',
                  'arguments': {'statement': 'SNR is 50', 'evidence': 'tool X'}},
                 {'id': 'h', 'name': 'note_hypothesis',
                  'arguments': {'hypothesis': 'drift', 'test': 'correlate'}},
             ]},
            {'role': 'assistant', 'content': 'done'},
        ])
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([ReasoningTools()]),
                              mode='deliberate')
        events, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        kinds = [e.kind for e in events]
        assert 'finding' in kinds
        assert 'hypothesis' in kinds


class TestThinkingSurface:

    def test_thinking_emitted_before_text(self):
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': 'the answer',
             'thinking': 'let me reason...'},
        ])
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([]))
        events, on_event = _collect()
        ctrl.send_user_message('q', on_event)
        kinds = [e.kind for e in events]
        assert kinds == ['thinking', 'assistant_text', 'done']
        assert events[0].text == 'let me reason...'

    def test_provider_blocks_carried_into_history(self):
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c', 'name': 'echo', 'arguments': {}}],
             '_provider_blocks': [{'type': 'thinking', 'thinking': 't', 'signature': 's'},
                                  {'type': 'tool_use', 'id': 'c', 'name': 'echo', 'input': {}}]},
            {'role': 'assistant', 'content': 'final'},
        ])
        ctrl = ChatController(pva_reader=None, backend=backend,
                              tool_registry=ToolRegistry([_EchoTool()]))
        _, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        assistant_msgs = [m for m in ctrl.messages if m['role'] == 'assistant']
        assert '_provider_blocks' in assistant_msgs[0]
        assert assistant_msgs[0]['_provider_blocks'][0]['type'] == 'thinking'


class TestBudgetReset:

    def test_reset_turn_budgets_called(self):
        class _Budgeted(BaseTool):
            def __init__(self):
                self.resets = 0

            def reset_turn_budgets(self):
                self.resets += 1

            @tool(description="noop")
            def noop(self) -> dict:
                return {'ok': True}

        bt = _Budgeted()
        ctrl = ChatController(pva_reader=None,
                              backend=_ScriptedBackend([{'role': 'assistant', 'content': 'hi'}]),
                              tool_registry=ToolRegistry([bt]))
        _, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        assert bt.resets == 1