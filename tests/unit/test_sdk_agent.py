"""Tests for SdkAgent option wiring + the permission policy (no network).

Constructing an SdkAgent builds the MCP server + ClaudeAgentOptions but does NOT
connect, so these run offline. They lock in the two postures:

  * full-power  — Claude Code built-ins available, mutating/exec ones gated by a
    confirmation callback (no bypassPermissions); domain + safe-read tools auto-run.
  * read-only   — built-ins banned, only the safe domain tools (bypass is fine).
"""

from __future__ import annotations

import types
from collections import deque

import anyio
import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from dashpva.agent.sdk_agent import (
    AGENT_SYSTEM_PROMPT,
    AUTO_ALLOW_BUILTINS,
    BUILTIN_TOOLS,
    CONFIRM_BUILTINS,
    SdkAgent,
    stdin_permission_handler,
)


class _Reader:
    def __init__(self):
        self.cached_images = deque()
        self.cached_frame_ids = deque()
        self.cached_timestamps = deque()
        self.feature_vector_cache = []
        self.blob_detections_cache = []
        self.sampled_descriptions = []
        self.cached_ca = {}
        self.cached_ca_frame_ids = {}
        self.cached_ca_timestamps = {}
        self.cached_attributes = None
        self.shape = (0, 0)
        self.image_is_transposed = False
        self.frames_received = 0
        self.config = {}
        self.CACHING_MODE = "saved"


def _settings():
    return types.SimpleNamespace(
        CONFIG={}, METADATA_CA={}, IOC_PREFIX="x:", DETECTOR_PREFIX="x:",
        CHAT_TOOLS={}, SESSION_ANALYSIS={},
    )


def _agent(**kw):
    return SdkAgent(_Reader(), _settings(), base_url="http://localhost:1", **kw)


class TestOptionWiring:

    def test_full_power_defaults(self):
        o = _agent().options
        assert o.permission_mode == "default"          # NOT bypassPermissions
        assert o.can_use_tool is not None              # confirmation callback present
        assert o.disallowed_tools == []                # built-ins available
        # safe reads + domain tools auto-approved; mutating/exec ones are not
        for t in AUTO_ALLOW_BUILTINS:
            assert t in o.allowed_tools
        for t in CONFIRM_BUILTINS:
            assert t not in o.allowed_tools
        assert any(x.startswith("mcp__dashpva__") for x in o.allowed_tools)
        # rides on Claude Code's own system prompt + a beamline append
        assert isinstance(o.system_prompt, dict)
        assert o.system_prompt["preset"] == "claude_code"
        assert "beamline" in o.system_prompt["append"].lower()
        assert o.max_thinking_tokens == 6000           # native thinking on

    def test_read_only(self):
        o = _agent(enable_builtin_tools=False).options
        assert o.permission_mode == "bypassPermissions"
        assert o.can_use_tool is None
        assert set(BUILTIN_TOOLS) <= set(o.disallowed_tools)
        assert o.system_prompt == AGENT_SYSTEM_PROMPT

    def test_thinking_disabled_when_zero(self):
        s = _settings()
        s.CHAT_TOOLS = {"THINKING_BUDGET_TOKENS": 0}
        # 0 -> no explicit budget (None); the model still reasons by default.
        assert _agent(thinking_tokens=None).options.max_thinking_tokens == 6000
        assert SdkAgent(_Reader(), s, base_url="http://x").options.max_thinking_tokens is None


class TestStdinPermission:

    def test_auto_allows_safe_reads(self):
        res = anyio.run(stdin_permission_handler, "Read", {"file": "a"}, None)
        assert isinstance(res, PermissionResultAllow)

    def test_denies_exec_without_tty(self, monkeypatch):
        # Force "no TTY" so the handler denies deterministically (never blocks on input()).
        fake = types.SimpleNamespace(isatty=lambda: False)
        monkeypatch.setattr("dashpva.agent.sdk_agent.sys.stdin", fake)
        res = anyio.run(stdin_permission_handler, "Bash", {"command": "ls"}, None)
        assert isinstance(res, PermissionResultDeny)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])