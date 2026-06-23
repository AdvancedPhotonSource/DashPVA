"""Adapt the existing DashPVA tool registry into an in-process SDK MCP server.

The agent app *owns* the tools but does not rebuild them: it instantiates the
existing :mod:`dashpva.analysis.tools` classes over a
:class:`~dashpva.agent.reader_interface.ScanReader`, collects them in the
existing :class:`~dashpva.analysis.tools.base.ToolRegistry`, then wraps each
discovered :class:`ToolSpec` as a Claude Agent SDK tool via :func:`tool` +
:func:`create_sdk_mcp_server` — **one in-process MCP server, private to the
agent** (no separate server process).

Bridging details:

  * Each existing tool is a **sync** method returning a JSON-serializable dict
    and never raising (:meth:`ToolRegistry.call` wraps errors as
    ``{'error': ...}``). The SDK wants an **async** handler returning
    ``{"content": [{"type": "text", "text": ...}]}`` — so we run the sync tool
    in a worker thread (keeps the event loop responsive) and ``json.dumps`` the
    result into the content shape.
  * The handler closes over ``registry`` + the tool name (a factory avoids the
    classic late-binding-in-a-loop bug); it does NOT close over a bound method,
    so error wrapping stays centralized in :meth:`ToolRegistry.call`.
  * We reuse ``spec.parameters`` (already a full JSON Schema object) as the SDK
    input schema verbatim — the SDK accepts a JSON-Schema dict.

The SDK addresses these tools as ``mcp__<server_name>__<tool_name>``; we return
that fully-qualified list so :mod:`sdk_agent` can pass it as ``allowed_tools``.
"""

from __future__ import annotations

import json

import anyio
from claude_agent_sdk import create_sdk_mcp_server, tool

from dashpva.analysis.session_analyzer import SessionAnalyzer
from dashpva.analysis.tools.analysis_tools import AnalysisTools
from dashpva.analysis.tools.base import ToolRegistry, ToolSpec
from dashpva.analysis.tools.pv_tools import PvTools
from dashpva.analysis.tools.session_tools import SessionTools

SERVER_NAME = "dashpva"


def build_tool_registry(
    reader,
    settings,
    *,
    analyzer: SessionAnalyzer | None = None,
    vision_enabled: bool = False,
) -> ToolRegistry:
    """Instantiate the four existing tool classes over *reader* and register them.

    ``PvTools`` gates EPICS access, ``AnalysisTools`` borrows it for
    ``correlate_series('pv:…')``, and ``SessionTools`` reuses a ``SessionAnalyzer``
    for change-event detection (no LLM backend needed for that). When vision is
    off, ``describe_frame`` is removed entirely.

    The old ``ReasoningTools`` scratchpad (``record_plan``/``record_finding``/
    ``note_hypothesis``) is intentionally **not** registered: the Claude Agent SDK
    plans and reasons natively (extended thinking), so those scaffolding tools are
    redundant here. They remain available to the embedded agent in ``analysis/``.
    """
    pv_tools = PvTools(reader, settings)
    if analyzer is None:
        # SessionTools only calls analyzer._detect_change_events (pure, no backend).
        analyzer = SessionAnalyzer(reader, backend=None)
    session_tools = SessionTools(reader, analyzer)
    analysis_tools = AnalysisTools(
        reader, settings, pv_tools=pv_tools, vision_enabled=vision_enabled
    )
    registry = ToolRegistry([pv_tools, session_tools, analysis_tools])
    if not vision_enabled:
        registry.remove("describe_frame")
    return registry


def build_mcp_server(registry: ToolRegistry, *, server_name: str = SERVER_NAME):
    """Wrap every spec in *registry* as an SDK tool and build an MCP server.

    Returns ``(server_config, qualified_tool_names)`` where the names are
    ``mcp__<server_name>__<tool>`` for use as ``allowed_tools``.
    """
    specs = registry.specs()
    sdk_tools = [_adapt_spec(registry, spec) for spec in specs]
    server = create_sdk_mcp_server(name=server_name, version="1.0.0", tools=sdk_tools)
    qualified = [f"mcp__{server_name}__{spec.name}" for spec in specs]
    return server, qualified


def reset_turn_budgets(registry: ToolRegistry) -> None:
    """Reset per-turn rate limits on expensive frame/vision tools.

    The SDK has no built-in "turn start" hook, so :mod:`sdk_agent` calls this
    before sending each user message — mirroring
    ``ChatController._reset_turn_budgets``. Duck-typed so the registry stays
    decoupled from any specific tool class.
    """
    for inst in registry.instances():
        reset = getattr(inst, "reset_turn_budgets", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _adapt_spec(registry: ToolRegistry, spec: ToolSpec):
    """Build one SDK tool that defers to ``registry.call(spec.name, args)``.

    A dedicated factory function captures *spec* per-iteration (no late binding).
    The sync tool runs in a worker thread so a blocking tool (numpy / lazy h5
    frame reads) doesn't stall the asyncio loop.
    """
    name = spec.name
    description = spec.description
    schema = spec.parameters or {"type": "object", "properties": {}}

    async def _handler(args):
        call_args = dict(args or {})
        result = await anyio.to_thread.run_sync(lambda: registry.call(name, call_args))
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    return tool(name, description, schema)(_handler)