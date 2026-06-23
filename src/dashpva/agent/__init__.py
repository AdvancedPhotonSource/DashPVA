"""Standalone, general beamline-analysis agent built on the Claude Agent SDK.

This package turns the embedded :mod:`dashpva.analysis` chat agent into a
standalone app that runs *alongside* DashPVA. It owns its own data layer (a
:class:`~dashpva.agent.reader_interface.ScanReader`) and reuses the existing
:mod:`dashpva.analysis.tools` logic verbatim — those tools are the reusable
asset. The hand-rolled ``ChatController`` and custom LLM backends are replaced
by the Claude Agent SDK loop (pointed at ANL's Argo gateway via an
``argo-proxy`` sidecar).

Milestone 1 (this package, additive — nothing in ``analysis/`` or the viewer
changes):

  * :mod:`reader_interface` — the ``ScanReader`` Protocol the tools depend on.
  * :mod:`saved_scan_reader` — ``SavedScanReader`` over a recorded ``.h5``.
  * :mod:`tool_bridge` — adapts the existing ``ToolRegistry`` into SDK MCP tools.
  * :mod:`proxy_manager` — launches/health-checks the ``argo-proxy`` sidecar.
  * :mod:`sdk_agent` — the ``ClaudeSDKClient`` harness, streaming events out.
  * :mod:`run` — a headless runner (``python -m dashpva.agent.run``).
"""

from __future__ import annotations

__all__ = ["ScanReader"]


def __getattr__(name: str):
    # Lazy re-export so importing the package never drags in the SDK (only the
    # Protocol is cheap). ``from dashpva.agent import ScanReader`` works without
    # claude-agent-sdk installed.
    if name == "ScanReader":
        from dashpva.agent.reader_interface import ScanReader

        return ScanReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
