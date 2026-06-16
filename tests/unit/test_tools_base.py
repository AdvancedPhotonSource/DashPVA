"""Tests for the tool decorator / registry / OpenAI-schema generator."""

from __future__ import annotations

import json
from typing import Optional

import pytest

from dashpva.analysis.tools.base import (
    BaseTool,
    ToolRegistry,
    discover_tools,
    tool,
    tool_to_openai_schema,
)


class _SampleTool(BaseTool):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []

    @tool(description="Read a PV.")
    def read_pv(self, pv_name: str) -> dict:
        """Read a PV.

        Args:
            pv_name: Full EPICS PV name.
        """
        self.calls.append(('read_pv', {'pv_name': pv_name}))
        return {'name': pv_name, 'value': 1.0}

    @tool()
    def history(self, pv_name: str, start: int, end: int = 100) -> dict:
        """Return PV history.

        Args:
            pv_name: Name of the PV to look up.
            start: Frame to start at.
            end: Frame to stop at.
        """
        self.calls.append(('history', {'pv_name': pv_name, 'start': start, 'end': end}))
        return {'values': [start, end]}

    @tool()
    def with_optional(self, label: Optional[str] = None, n: int = 5) -> dict:
        """Demo optional-typed parameter handling."""
        return {'label': label, 'n': n}

    @tool()
    def takes_list(self, names: list[str]) -> dict:
        """Demo list-of-str parameter."""
        return {'count': len(names)}


class _ChildTool(_SampleTool):
    @tool(description="Overridden read_pv from child.")
    def read_pv(self, pv_name: str) -> dict:  # noqa: D401 - intentional override
        return {'overridden': True, 'name': pv_name}


def test_discover_tools_picks_decorated_methods_only():
    inst = _SampleTool()
    specs = discover_tools(inst)
    names = sorted(s.name for s in specs)
    assert names == ['history', 'read_pv', 'takes_list', 'with_optional']


def test_discover_tools_uses_first_paragraph_when_description_missing():
    inst = _SampleTool()
    specs = {s.name: s for s in discover_tools(inst)}
    assert specs['history'].description.startswith('Return PV history')


def test_discover_tools_respects_subclass_override():
    inst = _ChildTool()
    specs = {s.name: s for s in discover_tools(inst)}
    # The child's @tool decorator wins; child description is used.
    assert specs['read_pv'].description == 'Overridden read_pv from child.'
    # And calling it goes to the child impl.
    assert specs['read_pv'].func('foo') == {'overridden': True, 'name': 'foo'}


def test_schema_required_vs_optional():
    inst = _SampleTool()
    specs = {s.name: s for s in discover_tools(inst)}

    # history has start (required) and end (has default → not required)
    hist = specs['history'].parameters
    assert hist['type'] == 'object'
    assert hist['required'] == ['pv_name', 'start']
    assert hist['properties']['start']['type'] == 'integer'
    assert hist['properties']['end']['type'] == 'integer'

    # Optional[str] in with_optional → not required, schema is for str
    opt = specs['with_optional'].parameters
    assert 'label' in opt['properties']
    assert opt['properties']['label'] == {
        'type': 'string',
        'description': '',
    } or opt['properties']['label']['type'] == 'string'
    assert 'label' not in opt.get('required', [])


def test_schema_list_param():
    inst = _SampleTool()
    specs = {s.name: s for s in discover_tools(inst)}
    schema = specs['takes_list'].parameters
    assert schema['properties']['names']['type'] == 'array'
    assert schema['properties']['names']['items'] == {'type': 'string'}


def test_arg_docs_carry_into_schema():
    inst = _SampleTool()
    specs = {s.name: s for s in discover_tools(inst)}
    schema = specs['read_pv'].parameters
    assert schema['properties']['pv_name']['description'] == 'Full EPICS PV name.'


def test_tool_to_openai_schema_shape():
    inst = _SampleTool()
    spec = discover_tools(inst)[0]
    schema = tool_to_openai_schema(spec)
    assert schema['type'] == 'function'
    assert schema['function']['name'] == spec.name
    assert schema['function']['parameters'] == spec.parameters


def test_registry_aggregates_and_calls():
    reg = ToolRegistry([_SampleTool()])
    assert 'read_pv' in reg.names()
    result = reg.call('read_pv', {'pv_name': '6idb1:m1'})
    assert result == {'name': '6idb1:m1', 'value': 1.0}


def test_registry_unknown_tool_returns_error_not_raises():
    reg = ToolRegistry([_SampleTool()])
    result = reg.call('does_not_exist', {})
    assert 'error' in result
    assert 'unknown tool' in result['error']


def test_registry_wraps_exception_as_error_dict():
    class _Broken(BaseTool):
        @tool()
        def boom(self) -> dict:
            """Always raises."""
            raise ValueError("nope")

    reg = ToolRegistry([_Broken()])
    result = reg.call('boom', {})
    assert result == {'error': 'ValueError: nope'}


def test_registry_wraps_typeerror_for_bad_args():
    reg = ToolRegistry([_SampleTool()])
    result = reg.call('read_pv', {'wrong_param': 'x'})
    assert 'error' in result
    assert 'TypeError' in result['error']


def test_registry_openai_schemas_is_json_serializable():
    reg = ToolRegistry([_SampleTool()])
    schemas = reg.openai_schemas()
    assert len(schemas) >= 1
    # Should round-trip through JSON without errors.
    json.dumps(schemas)