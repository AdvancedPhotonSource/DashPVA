"""Tests for the native Anthropic backend's translation layer.

The pure helpers (history <-> content blocks, response parsing, tool schema)
need no SDK or network, so they are tested directly. We also check the
missing-SDK path raises a clear, actionable error.
"""

from __future__ import annotations

import types

import pytest

from dashpva.analysis.backends import anthropic_backend as ab


class TestToAnthropicMessages:

    def test_system_hoisted_and_roles_alternate(self):
        msgs = [
            {'role': 'system', 'content': 'SYS'},
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi there'},
        ]
        system, out = ab._to_anthropic_messages(msgs)
        assert system == 'SYS'
        assert out[0]['role'] == 'user'
        assert out[0]['content'][0] == {'type': 'text', 'text': 'hello'}
        assert out[1]['role'] == 'assistant'

    def test_tool_calls_become_tool_use_blocks(self):
        msgs = [
            {'role': 'assistant', 'content': 'let me check',
             'tool_calls': [{'id': 'c1', 'name': 'read_pv',
                             'arguments': {'pv_name': 'sim:x'}}]},
        ]
        _, out = ab._to_anthropic_messages(msgs)
        blocks = out[0]['content']
        assert blocks[0] == {'type': 'text', 'text': 'let me check'}
        assert blocks[1] == {'type': 'tool_use', 'id': 'c1',
                             'name': 'read_pv', 'input': {'pv_name': 'sim:x'}}

    def test_tool_results_merge_into_single_user_turn(self):
        # assistant(tool_use) then two tool results then a user nudge must collapse
        # into one assistant turn followed by ONE user turn (alternating roles).
        msgs = [
            {'role': 'assistant', 'content': '',
             'tool_calls': [{'id': 'c1', 'name': 't', 'arguments': {}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'name': 't', 'content': '{"v":1}'},
            {'role': 'user', 'content': 'now answer'},
        ]
        _, out = ab._to_anthropic_messages(msgs)
        assert [m['role'] for m in out] == ['assistant', 'user']
        user_blocks = out[1]['content']
        assert user_blocks[0]['type'] == 'tool_result'
        assert user_blocks[0]['tool_use_id'] == 'c1'
        assert user_blocks[1] == {'type': 'text', 'text': 'now answer'}

    def test_provider_blocks_used_verbatim(self):
        raw = [{'type': 'thinking', 'thinking': 't', 'signature': 's'},
               {'type': 'tool_use', 'id': 'c1', 'name': 't', 'input': {}}]
        msgs = [{'role': 'assistant', 'content': 'x', 'tool_calls': [],
                 '_provider_blocks': raw}]
        _, out = ab._to_anthropic_messages(msgs)
        assert out[0]['content'] is raw  # verbatim round-trip preserves signature

    def test_empty_assistant_gets_placeholder(self):
        msgs = [{'role': 'assistant', 'content': ''}]
        _, out = ab._to_anthropic_messages(msgs)
        assert out[0]['content'] == [{'type': 'text', 'text': '(no content)'}]


class TestToAnthropicTool:

    def test_openai_schema_translated(self):
        openai = {'type': 'function', 'function': {
            'name': 'read_pv', 'description': 'read a pv',
            'parameters': {'type': 'object', 'properties': {'pv_name': {'type': 'string'}}}}}
        out = ab._to_anthropic_tool(openai)
        assert out['name'] == 'read_pv'
        assert out['description'] == 'read a pv'
        assert out['input_schema']['properties']['pv_name']['type'] == 'string'


class TestParseResponse:

    def test_parses_thinking_text_and_tool_use(self):
        blocks = [
            types.SimpleNamespace(type='thinking', thinking='reasoning...'),
            types.SimpleNamespace(type='text', text='the answer'),
            types.SimpleNamespace(type='tool_use', id='c1', name='read_pv',
                                  input={'pv_name': 'sim:x'}),
        ]
        resp = types.SimpleNamespace(content=blocks)
        out = ab._parse_response(resp)
        assert out['content'] == 'the answer'
        assert out['thinking'] == 'reasoning...'
        assert out['tool_calls'] == [{'id': 'c1', 'name': 'read_pv',
                                      'arguments': {'pv_name': 'sim:x'}}]
        assert len(out['_provider_blocks']) == 3

    def test_block_to_dict_prefers_model_dump(self):
        class _Block:
            def model_dump(self):
                return {'type': 'thinking', 'thinking': 't', 'signature': 's'}
        assert ab._block_to_dict(_Block())['signature'] == 's'


class TestMissingSDK:

    def test_clear_error_when_anthropic_missing(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'anthropic':
                raise ImportError('no anthropic')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(RuntimeError) as exc:
            ab.AnthropicBackend({'ARGO_USER': 'tester', 'ARGO_MODEL': 'claudesonnet46'})
        assert 'anthropic' in str(exc.value).lower()