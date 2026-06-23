"""Tests for the tool bridge — registry → SDK MCP tool adaptation, no network.

Verifies the existing ``ToolRegistry`` is built over a reader, that specs adapt
into SDK tools addressed as ``mcp__dashpva__<name>``, and that the async handler
round-trips a tool's dict result (including the ``{'error': ...}`` wrapping) into
the SDK ``{"content": [{"type": "text", "text": ...}]}`` shape.
"""

from __future__ import annotations

import json
import types
from collections import deque

import anyio
import pytest

from dashpva.agent.tool_bridge import (
    _adapt_spec,
    build_mcp_server,
    build_tool_registry,
)


class _Reader:
    """Minimal ScanReader-shaped stub (no h5 / no frames)."""

    def __init__(self):
        self.cached_images = deque()
        self.cached_frame_ids = deque([100, 101])
        self.cached_timestamps = deque([0.0, 1.0])
        self.feature_vector_cache = []
        self.blob_detections_cache = []
        self.sampled_descriptions = []
        self.cached_ca = {}
        self.cached_ca_frame_ids = {}
        self.cached_ca_timestamps = {}
        self.cached_attributes = None
        self.shape = (0, 0)
        self.image_is_transposed = False
        self.frames_received = 2
        self.config = {"METADATA": {"CA": {}}}
        self.CACHING_MODE = "saved"


def _settings():
    return types.SimpleNamespace(
        CONFIG={},
        METADATA_CA={},
        IOC_PREFIX="6idb1:",
        DETECTOR_PREFIX="6idb1:",
        CHAT_TOOLS={"HISTORY_MAX_POINTS": 500, "FRAME_TOOL_MAX_CALLS_PER_TURN": 20},
        SESSION_ANALYSIS={},
    )


def _spec(registry, name):
    return next(s for s in registry.specs() if s.name == name)


# ----------------------------------------------------------------------
# Registry construction
# ----------------------------------------------------------------------

class TestRegistryBuild:

    def test_expected_tools_present(self):
        registry = build_tool_registry(_Reader(), _settings())
        names = set(registry.names())
        for expected in ("read_pv", "get_latest_features",
                         "get_feature_timeseries", "compute_radial_profile",
                         "list_available_frames", "get_scan_info"):
            assert expected in names

    def test_reasoning_scaffolding_not_registered(self):
        # Planning/reasoning is the SDK's job (native thinking) — the old
        # record_plan/finding/hypothesis scratchpad tools are intentionally absent.
        registry = build_tool_registry(_Reader(), _settings())
        names = set(registry.names())
        for absent in ("record_plan", "record_finding", "note_hypothesis"):
            assert absent not in names

    def test_vision_tool_removed_by_default(self):
        registry = build_tool_registry(_Reader(), _settings())
        assert "describe_frame" not in registry.names()

    def test_vision_tool_present_when_enabled(self):
        registry = build_tool_registry(_Reader(), _settings(), vision_enabled=True)
        assert "describe_frame" in registry.names()


# ----------------------------------------------------------------------
# SDK server / qualified names
# ----------------------------------------------------------------------

class TestMcpServer:

    def test_qualified_names_prefixed(self):
        registry = build_tool_registry(_Reader(), _settings())
        server, qualified = build_mcp_server(registry)
        assert qualified
        assert all(q.startswith("mcp__dashpva__") for q in qualified)
        # one qualified name per registered spec
        assert len(qualified) == len(registry.specs())

    def test_server_config_shape(self):
        registry = build_tool_registry(_Reader(), _settings())
        server, _ = build_mcp_server(registry)
        # create_sdk_mcp_server returns a dict config {type, name, instance}.
        assert isinstance(server, dict)
        assert server.get("name") == "dashpva"
        assert server.get("type") == "sdk"


# ----------------------------------------------------------------------
# Handler round-trip
# ----------------------------------------------------------------------

class TestHandlerRoundTrip:

    def test_handler_returns_content_shape(self):
        registry = build_tool_registry(_Reader(), _settings())
        sdk_tool = _adapt_spec(registry, _spec(registry, "list_available_frames"))
        out = anyio.run(sdk_tool.handler, {})
        assert "content" in out
        block = out["content"][0]
        assert block["type"] == "text"
        payload = json.loads(block["text"])
        assert payload["n_frames"] == 2
        assert payload["caching_mode"] == "saved"

    def test_handler_passes_through_tool_error(self):
        registry = build_tool_registry(_Reader(), _settings())
        # frame 999 isn't cached → tool returns an {'error': ...} dict.
        sdk_tool = _adapt_spec(registry, _spec(registry, "compute_radial_profile"))
        out = anyio.run(sdk_tool.handler, {"frame_id": 999})
        payload = json.loads(out["content"][0]["text"])
        assert "error" in payload
        assert "999" in payload["error"]

    def test_input_schema_is_passed_through(self):
        registry = build_tool_registry(_Reader(), _settings())
        spec = _spec(registry, "compute_radial_profile")
        sdk_tool = _adapt_spec(registry, spec)
        # The SDK tool reuses the spec's JSON Schema verbatim.
        assert sdk_tool.input_schema is spec.parameters
        assert sdk_tool.input_schema["properties"]["frame_id"]["type"] == "integer"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])