"""Tests for OllamaBackend.chat() and ArgoBackend.chat() — wire format + tool-call parsing."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from dashpva.analysis.backends.argo_backend import ArgoBackend
from dashpva.analysis.backends.ollama_backend import OllamaBackend


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class TestOllamaChat:

    def test_chat_no_tools_returns_content(self):
        b = OllamaBackend({'OLLAMA_MODEL': 'llama3.2'})
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            return_value=_FakeResponse({'message': {'content': 'hello world'}}),
        ) as mocked:
            out = b.chat([{'role': 'user', 'content': 'hi'}])
        assert out == {'role': 'assistant', 'content': 'hello world'}
        body = mocked.call_args.kwargs['json']
        assert body['model'] == 'llama3.2'
        assert body['stream'] is False
        assert body['messages'] == [{'role': 'user', 'content': 'hi'}]
        assert 'tools' not in body

    def test_chat_with_tools_sends_tools_field(self):
        b = OllamaBackend({'OLLAMA_MODEL': 'qwen2.5'})
        schemas = [{'type': 'function', 'function': {'name': 'read_pv', 'parameters': {}}}]
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            return_value=_FakeResponse({'message': {'content': 'ok'}}),
        ) as mocked:
            b.chat([{'role': 'user', 'content': 'go'}], tools=schemas)
        body = mocked.call_args.kwargs['json']
        assert body['tools'] == schemas

    def test_chat_parses_tool_call_response_with_dict_args(self):
        b = OllamaBackend({})
        payload = {
            'message': {
                'content': '',
                'tool_calls': [{
                    'function': {'name': 'read_pv', 'arguments': {'pv_name': '6idb1:m1'}},
                }],
            }
        }
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            return_value=_FakeResponse(payload),
        ):
            out = b.chat([{'role': 'user', 'content': 'what'}], tools=[{'x': 1}])
        assert out['role'] == 'assistant'
        assert len(out['tool_calls']) == 1
        call = out['tool_calls'][0]
        assert call['name'] == 'read_pv'
        assert call['arguments'] == {'pv_name': '6idb1:m1'}
        assert call['id']  # synthesized but present

    def test_chat_parses_tool_call_response_with_string_args(self):
        b = OllamaBackend({})
        payload = {
            'message': {
                'content': '',
                'tool_calls': [{
                    'function': {
                        'name': 'history',
                        'arguments': '{"pv_name": "6idb1:m1", "start": 10, "end": 50}',
                    },
                }],
            }
        }
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            return_value=_FakeResponse(payload),
        ):
            out = b.chat([{'role': 'user', 'content': 'q'}], tools=[{'x': 1}])
        assert out['tool_calls'][0]['arguments'] == {
            'pv_name': '6idb1:m1', 'start': 10, 'end': 50,
        }

    def test_chat_raises_on_connection_error(self):
        b = OllamaBackend({})
        import requests as _req
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            side_effect=_req.exceptions.ConnectionError('boom'),
        ):
            with pytest.raises(ConnectionError):
                b.chat([{'role': 'user', 'content': 'x'}])

    def test_complete_default_wraps_chat(self):
        b = OllamaBackend({})
        with patch(
            'dashpva.analysis.backends.ollama_backend.requests.post',
            return_value=_FakeResponse({'message': {'content': '  hi  '}}),
        ):
            text = b.complete('hello', system='be brief')
        assert text == 'hi'


# ---------------------------------------------------------------------------
# Argo
# ---------------------------------------------------------------------------


def _make_argo() -> ArgoBackend:
    return ArgoBackend({'ARGO_USER': 'tester', 'ARGO_MODEL': 'claudesonnet46'})


class TestArgoChat:

    def test_chat_no_tools_returns_content(self):
        b = _make_argo()
        payload = {'choices': [{'message': {'content': 'hello'}}]}
        with patch(
            'dashpva.analysis.backends.argo_backend.requests.post',
            return_value=_FakeResponse(payload),
        ) as mocked:
            out = b.chat([{'role': 'user', 'content': 'hi'}])
        assert out == {'role': 'assistant', 'content': 'hello'}
        body = mocked.call_args.kwargs['json']
        assert body['model'] == 'claudesonnet46'
        assert 'tools' not in body
        assert 'tool_choice' not in body
        # Default sampling: temperature only, no top_p.
        assert 'top_p' not in body

    def test_chat_with_tools_adds_tools_and_tool_choice(self):
        b = _make_argo()
        schemas = [{'type': 'function', 'function': {'name': 'read_pv', 'parameters': {}}}]
        payload = {'choices': [{'message': {'content': 'ok'}}]}
        with patch(
            'dashpva.analysis.backends.argo_backend.requests.post',
            return_value=_FakeResponse(payload),
        ) as mocked:
            b.chat([{'role': 'user', 'content': 'q'}], tools=schemas)
        body = mocked.call_args.kwargs['json']
        assert body['tools'] == schemas
        assert body['tool_choice'] == 'auto'

    def test_chat_parses_openai_tool_call_string_args(self):
        b = _make_argo()
        payload = {
            'choices': [{
                'message': {
                    'content': '',
                    'tool_calls': [{
                        'id': 'call_abc',
                        'function': {
                            'name': 'read_pv',
                            'arguments': '{"pv_name": "6idb1:m1"}',
                        },
                    }],
                }
            }]
        }
        with patch(
            'dashpva.analysis.backends.argo_backend.requests.post',
            return_value=_FakeResponse(payload),
        ):
            out = b.chat([{'role': 'user', 'content': 'x'}], tools=[{'x': 1}])
        call = out['tool_calls'][0]
        assert call['id'] == 'call_abc'
        assert call['name'] == 'read_pv'
        assert call['arguments'] == {'pv_name': '6idb1:m1'}

    def test_chat_raises_on_401(self):
        b = _make_argo()
        with patch(
            'dashpva.analysis.backends.argo_backend.requests.post',
            return_value=_FakeResponse({}, status=401),
        ):
            with pytest.raises(RuntimeError, match='auth'):
                b.chat([{'role': 'user', 'content': 'x'}])

    def test_complete_default_wraps_chat(self):
        b = _make_argo()
        payload = {'choices': [{'message': {'content': 'hi'}}]}
        with patch(
            'dashpva.analysis.backends.argo_backend.requests.post',
            return_value=_FakeResponse(payload),
        ):
            text = b.complete('say hi', system='be brief')
        assert text == 'hi'