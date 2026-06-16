"""Tool abstraction for LLM tool-calling.

Subclass :class:`BaseTool`, decorate methods with :func:`tool`, then aggregate
instances via :class:`ToolRegistry` to feed an :class:`~dashpva.analysis.llm_backend.LLMBackend`'s
``chat()`` call.

Shape mirrors the APS EAA framework's ``BaseTool`` / ``@tool`` so a future
migration is a swap of the executor and marker attribute, not a rewrite.
"""

from dashpva.analysis.tools.base import (
    BaseTool,
    ToolRegistry,
    ToolSpec,
    discover_tools,
    tool,
    tool_to_openai_schema,
)

__all__ = [
    'BaseTool',
    'ToolRegistry',
    'ToolSpec',
    'discover_tools',
    'tool',
    'tool_to_openai_schema',
]