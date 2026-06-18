"""Tool decorator, registry, and OpenAI-schema generator.

The shape here is intentionally aligned with the APS EAA framework
(https://github.com/AdvancedPhotonSource/EAA): subclass :class:`BaseTool`,
decorate methods with :func:`tool`, then call :func:`discover_tools` on an
instance to get back a list of :class:`ToolSpec`.

Migrating to EAA later means renaming the marker attribute
``__dashpva_tool__`` → ``__eaa_tool__`` and swapping this module's
``discover_tools`` / executor for EAA's; nothing in the tool implementations
(e.g. ``PvTools``, ``SessionTools``) changes.
"""

from __future__ import annotations

import inspect
import json
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

_TOOL_MARKER = '__dashpva_tool__'


@dataclass(frozen=True)
class ToolSpec:
    """A discovered tool — name, description, JSON-Schema parameters, and bound callable."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict
    require_approval: bool = False
    owner_cls: type | None = None


def tool(
    name: str | None = None,
    description: str | None = None,
    require_approval: bool = False,
) -> Callable:
    """Tag a method on a :class:`BaseTool` subclass as an exposable tool.

    Parameters
    ----------
    name : str, optional
        Tool name advertised to the model. Defaults to ``func.__name__``.
    description : str, optional
        Description sent in the OpenAI schema. Defaults to the first paragraph
        of the function's docstring.
    require_approval : bool
        Reserved for future write-tool gating; current executor does not act on it.
    """

    def decorator(func: Callable) -> Callable:
        meta = {
            'name': name or func.__name__,
            'description': description,  # may be None; resolved at schema time
            'require_approval': bool(require_approval),
        }
        setattr(func, _TOOL_MARKER, meta)
        return func

    return decorator


class BaseTool:
    """Subclass and decorate methods with :func:`tool`. Instance state (PVAReader
    handles, history stores, etc.) goes in ``__init__``; bound methods are what
    :func:`discover_tools` picks up."""

    def __init__(self) -> None:  # noqa: D401 - explicit no-op base
        pass


def discover_tools(instance: BaseTool) -> list[ToolSpec]:
    """Return one :class:`ToolSpec` per decorated method visible on *instance*.

    Walks ``type(instance).__mro__`` so subclasses can override a parent's tool
    by re-decorating a method of the same name (subclass wins).
    """
    seen: dict[str, ToolSpec] = {}
    for cls in type(instance).__mro__:
        if cls is object:
            continue
        for attr_name, attr in cls.__dict__.items():
            if not callable(attr):
                continue
            meta = getattr(attr, _TOOL_MARKER, None)
            if meta is None:
                continue
            if meta['name'] in seen:
                # Subclass walked first via MRO — already recorded the override.
                continue
            bound = getattr(instance, attr_name)
            description = meta['description'] or _first_paragraph(attr.__doc__) or meta['name']
            parameters = _signature_to_schema(attr)
            seen[meta['name']] = ToolSpec(
                name=meta['name'],
                description=description,
                func=bound,
                parameters=parameters,
                require_approval=meta['require_approval'],
                owner_cls=cls,
            )
    return list(seen.values())


def tool_to_openai_schema(spec: ToolSpec) -> dict:
    """Render a :class:`ToolSpec` as the OpenAI function-call tool schema."""
    return {
        'type': 'function',
        'function': {
            'name': spec.name,
            'description': spec.description,
            'parameters': spec.parameters,
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """Aggregates tools from one or more :class:`BaseTool` instances."""

    _by_name: dict[str, ToolSpec] = field(default_factory=dict)

    def __init__(self, instances: Iterable[BaseTool] | None = None):
        self._by_name = {}
        self._instances: list[BaseTool] = []
        if instances:
            for inst in instances:
                self.add(inst)

    def add(self, instance: BaseTool) -> None:
        if instance not in self._instances:
            self._instances.append(instance)
        for spec in discover_tools(instance):
            self._by_name[spec.name] = spec

    def instances(self) -> list[BaseTool]:
        """The tool-owning instances, in registration order. Lets callers
        invoke cross-cutting hooks (e.g. ``reset_turn_budgets``) without the
        registry knowing about any specific tool class."""
        return list(self._instances)

    def remove(self, name: str) -> None:
        """Unregister a tool by name (no-op if absent). Used to gate optional
        tools, e.g. hiding ``describe_frame`` when vision is toggled off."""
        self._by_name.pop(name, None)

    def specs(self) -> list[ToolSpec]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def openai_schemas(self) -> list[dict]:
        return [tool_to_openai_schema(s) for s in self._by_name.values()]

    def call(self, name: str, arguments: dict | None) -> dict:
        """Invoke a tool by name, returning a JSON-serialisable result.

        Exceptions are caught and returned as ``{'error': '<type>: <msg>'}`` so
        the model can read the failure and recover rather than the entire chat
        turn aborting.
        """
        spec = self._by_name.get(name)
        if spec is None:
            return {'error': f'unknown tool {name!r}; available: {sorted(self._by_name)}'}
        args = dict(arguments or {})
        try:
            result = spec.func(**args)
        except TypeError as e:
            return {'error': f'TypeError calling {name!r} with {args!r}: {e}'}
        except Exception as e:
            return {'error': f'{type(e).__name__}: {e}'}
        if isinstance(result, dict):
            return result
        # Wrap non-dict results so tool messages always have a stable shape.
        try:
            json.dumps(result, default=str)
            return {'result': result}
        except TypeError:
            return {'result': repr(result)}


# ---------------------------------------------------------------------------
# Schema generation helpers
# ---------------------------------------------------------------------------


_PRIMITIVE_TYPES: dict[type, str] = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
}


def _signature_to_schema(func: Callable) -> dict:
    """Build a JSON Schema ``object`` matching *func*'s signature.

    - Drops ``self``.
    - Treats parameters with defaults as optional (not in ``required``).
    - ``Optional[T]`` / ``T | None`` → schema for ``T`` and parameter is optional.
    - Unannotated parameters → ``{"type": "string"}`` (permissive default).
    """
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    properties: dict[str, dict] = {}
    required: list[str] = []
    arg_docs = _parse_arg_docs(func.__doc__)

    for pname, param in sig.parameters.items():
        if pname == 'self':
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, param.annotation)
        prop, optional = _annotation_to_schema(annotation)
        if arg_docs.get(pname):
            prop['description'] = arg_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty and not optional:
            required.append(pname)

    schema: dict = {'type': 'object', 'properties': properties}
    if required:
        schema['required'] = required
    return schema


def _annotation_to_schema(annotation: Any) -> tuple[dict, bool]:
    """Return ``(json_schema, is_optional)`` for *annotation*."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {'type': 'string'}, False

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[T] / T | None
    if origin is typing.Union or _is_uniontype(origin):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema, _ = _annotation_to_schema(non_none[0])
            return schema, True
        # Union of multiple primitives → JSON Schema accepts a type list.
        types = []
        for a in non_none:
            t = _PRIMITIVE_TYPES.get(a)
            if t and t not in types:
                types.append(t)
        if types:
            return ({'type': types[0]} if len(types) == 1 else {'type': types}), True
        return {'type': 'string'}, True

    # list[T] / List[T]
    if origin in (list, tuple):
        item_schema = {'type': 'string'}
        if args:
            item_schema, _ = _annotation_to_schema(args[0])
        return {'type': 'array', 'items': item_schema}, False

    # dict[...] / Dict[...]
    if origin is dict:
        return {'type': 'object'}, False

    # Plain primitive
    t = _PRIMITIVE_TYPES.get(annotation)
    if t:
        return {'type': t}, False

    if annotation is list:
        return {'type': 'array', 'items': {'type': 'string'}}, False
    if annotation is dict:
        return {'type': 'object'}, False

    return {'type': 'string'}, False


def _is_uniontype(origin: Any) -> bool:
    """Detect ``X | Y`` (PEP 604) without importing types.UnionType conditionally."""
    try:
        import types as _types
        return origin is getattr(_types, 'UnionType', None)
    except Exception:
        return False


def _first_paragraph(docstring: str | None) -> str:
    if not docstring:
        return ''
    text = inspect.cleandoc(docstring).strip()
    para = text.split('\n\n', 1)[0]
    return ' '.join(line.strip() for line in para.splitlines() if line.strip())


def _parse_arg_docs(docstring: str | None) -> dict[str, str]:
    """Pull per-parameter descriptions from a Google-style ``Args:`` block.

    Example::

        Args:
            pv_name: The full EPICS PV name (or friendly key from METADATA.CA).
            timeout: Seconds to wait before giving up.
    """
    if not docstring:
        return {}
    text = inspect.cleandoc(docstring)
    lines = text.splitlines()
    out: dict[str, str] = {}
    in_args = False
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if not in_args:
            if stripped in ('Args:', 'Arguments:', 'Parameters:'):
                in_args = True
            continue
        if not line.startswith((' ', '\t')) and stripped:
            # Left margin re-entered a new section.
            break
        if not stripped:
            continue
        if ':' in stripped and not stripped.startswith(' '):
            name, _, desc = stripped.partition(':')
            current = name.strip()
            out[current] = desc.strip()
        elif current:
            out[current] = (out[current] + ' ' + stripped).strip()
    return out