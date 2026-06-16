"""Tests for ChatController — the LLM↔tools loop, history growth, max rounds, reset."""

from __future__ import annotations

from dashpva.analysis.chat_controller import ChatController, ControllerEvent
from dashpva.analysis.llm_backend import LLMBackend
from dashpva.analysis.tools.base import BaseTool, ToolRegistry, tool


class _ScriptedBackend(LLMBackend):
    """Returns a queued sequence of chat() responses, one per call."""

    name = 'fake/scripted'

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        # Record a shallow copy of the conversation at each call.
        self.calls.append([dict(m) for m in messages])
        if self._responses:
            return self._responses.pop(0)
        return {'role': 'assistant', 'content': 'fallback'}


class _EchoTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.invocations = []

    @tool(description="Echo a value back.")
    def echo(self, value: str) -> dict:
        """Echo.

        Args:
            value: anything.
        """
        self.invocations.append(value)
        return {'echoed': value}


def _collect():
    events: list[ControllerEvent] = []
    return events, events.append


def _controller(backend, tools=None):
    reg = ToolRegistry(tools or [])
    return ChatController(pva_reader=None, backend=backend, tool_registry=reg,
                          system_prompt='SYS', max_tool_rounds=3)


class TestPlainTextTurn:

    def test_single_text_response(self):
        backend = _ScriptedBackend([{'role': 'assistant', 'content': 'hello'}])
        ctrl = _controller(backend)
        events, on_event = _collect()
        ctrl.send_user_message('hi', on_event)

        kinds = [e.kind for e in events]
        assert kinds == ['assistant_text', 'done']
        assert events[0].text == 'hello'
        assert events[1].rounds_used == 0

    def test_history_has_system_user_assistant(self):
        backend = _ScriptedBackend([{'role': 'assistant', 'content': 'hello'}])
        ctrl = _controller(backend)
        _, on_event = _collect()
        ctrl.send_user_message('hi', on_event)
        roles = [m['role'] for m in ctrl.messages]
        assert roles == ['system', 'user', 'assistant']


class TestToolLoop:

    def test_tool_call_then_final_answer(self):
        echo = _EchoTool()
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'name': 'echo', 'arguments': {'value': 'x'}}]},
            {'role': 'assistant', 'content': 'done with x'},
        ])
        ctrl = _controller(backend, [echo])
        events, on_event = _collect()
        ctrl.send_user_message('please echo', on_event)

        kinds = [e.kind for e in events]
        assert kinds == [
            'tool_call_requested', 'tool_call_result', 'assistant_text', 'done',
        ]
        assert echo.invocations == ['x']
        # tool result event carries the dict
        assert events[1].tool_result == {'echoed': 'x'}
        # done after one tool round
        assert events[-1].rounds_used == 1

    def test_tool_result_appended_as_tool_message(self):
        echo = _EchoTool()
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'name': 'echo', 'arguments': {'value': 'y'}}]},
            {'role': 'assistant', 'content': 'final'},
        ])
        ctrl = _controller(backend, [echo])
        _, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        roles = [m['role'] for m in ctrl.messages]
        # system, user, assistant(tool_calls), tool, assistant(final)
        assert roles == ['system', 'user', 'assistant', 'tool', 'assistant']
        tool_msg = ctrl.messages[3]
        assert tool_msg['tool_call_id'] == 'c1'
        assert tool_msg['name'] == 'echo'
        assert 'echoed' in tool_msg['content']

    def test_unknown_tool_returns_error_dict_to_model(self):
        backend = _ScriptedBackend([
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'name': 'nope', 'arguments': {}}]},
            {'role': 'assistant', 'content': 'ok'},
        ])
        ctrl = _controller(backend, [_EchoTool()])
        events, on_event = _collect()
        ctrl.send_user_message('go', on_event)
        result_event = [e for e in events if e.kind == 'tool_call_result'][0]
        assert 'error' in result_event.tool_result


class TestMaxRounds:

    def test_max_rounds_triggers_error(self):
        # Backend always asks for a tool, never finishes.
        loop_resp = {'role': 'assistant', 'content': '',
                     'tool_calls': [{'id': 'c', 'name': 'echo', 'arguments': {'value': 'z'}}]}
        backend = _ScriptedBackend([loop_resp] * 10)
        ctrl = _controller(backend, [_EchoTool()])  # max_tool_rounds=3
        events, on_event = _collect()
        ctrl.send_user_message('loop forever', on_event)
        assert events[-1].kind == 'error'
        assert 'max_tool_rounds' in events[-1].text


class TestBackendError:

    def test_backend_exception_becomes_error_event(self):
        class _Boom(LLMBackend):
            name = 'boom'
            def chat(self, messages, tools=None):
                raise ConnectionError('no server')

        ctrl = _controller(_Boom())
        events, on_event = _collect()
        ctrl.send_user_message('hi', on_event)
        assert len(events) == 1
        assert events[0].kind == 'error'
        assert 'ConnectionError' in events[0].text


class TestResetAndInject:

    def test_reset_keeps_only_system(self):
        backend = _ScriptedBackend([{'role': 'assistant', 'content': 'hi'}])
        ctrl = _controller(backend)
        _, on_event = _collect()
        ctrl.send_user_message('hello', on_event)
        assert len(ctrl.messages) > 1
        ctrl.reset()
        assert ctrl.messages == [{'role': 'system', 'content': 'SYS'}]

    def test_inject_session_analysis_prompt_sends_built_prompt(self):
        backend = _ScriptedBackend([{'role': 'assistant', 'content': 'summary'}])
        ctrl = _controller(backend)
        events, on_event = _collect()

        class _Analyzer:
            def build_prompt(self):
                return 'THE SIX SECTION PROMPT'

        ctrl.inject_session_analysis_prompt(_Analyzer(), on_event)
        # user message should be the built prompt
        user_msgs = [m for m in ctrl.messages if m['role'] == 'user']
        assert user_msgs[0]['content'] == 'THE SIX SECTION PROMPT'
        assert events[-1].kind == 'done'

    def test_inject_handles_build_failure(self):
        backend = _ScriptedBackend([])
        ctrl = _controller(backend)
        events, on_event = _collect()

        class _BadAnalyzer:
            def build_prompt(self):
                raise ValueError('cache empty')

        ctrl.inject_session_analysis_prompt(_BadAnalyzer(), on_event)
        assert events[-1].kind == 'error'
