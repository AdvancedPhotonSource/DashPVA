"""The agent harness — a ``ClaudeSDKClient`` wired to the DashPVA tools + Argo.

This replaces the hand-rolled ``ChatController`` loop and the custom
``ollama``/``argo``/``anthropic`` backends with the Claude Agent SDK's own
agentic loop. It:

  * builds the in-process MCP tool server from :mod:`tool_bridge`,
  * restricts the model to *those* tools (``allowed_tools``) and bans the
    built-in coding tools (``disallowed_tools``) — the agent is read-only,
  * points the SDK at the ``argo-proxy`` sidecar via
    ``env={ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN}`` and pins the CLI via
    ``cli_path`` (the SDK spawns the ``claude`` CLI; version drift breaks it),
  * streams normalized :class:`AgentEvent`s out for a headless printer or a UI.

Connection lifecycle is an ``async with`` (the SDK warns: don't break out of
``receive_response()`` early or asyncio cleanup hangs — so consumers iterate
:meth:`ask` to completion).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
)

from dashpva.agent.tool_bridge import (
    SERVER_NAME,
    build_mcp_server,
    build_tool_registry,
    reset_turn_budgets,
)

# Verified-good defaults (HANDOFF_PHASE4 §2). Pin the CLI; use Opus 4.8 on Argo.
DEFAULT_CLI_PATH = "~/.local/share/claude/versions/2.1.177"
DEFAULT_MODEL = "claudeopus48"

# Claude Code's standard built-in tools. In full-power mode the agent is "Claude
# Code + the domain tools"; in read-only mode they are all banned.
#
# Read-only built-ins are auto-approved (like Claude Code's normal behavior);
# mutating / exec / network built-ins require the user's confirmation each call
# via the permission callback. (WebSearch/WebFetch are server-side and may be
# unavailable through Argo; they still go through confirmation if attempted.)
AUTO_ALLOW_BUILTINS = ["Read", "Glob", "Grep", "TodoWrite"]
CONFIRM_BUILTINS = ["Bash", "Write", "Edit", "NotebookEdit", "WebSearch", "WebFetch", "Task"]
BUILTIN_TOOLS = AUTO_ALLOW_BUILTINS + CONFIRM_BUILTINS

# Domain guidance appended to Claude Code's own (preset) system prompt in
# full-power mode — adds the beamline-scientist role + the scan tools on top of
# Claude Code's general tool-use scaffolding.
BEAMLINE_APPEND = (
    "\n\n## Beamline analysis role\n"
    "You are also an expert synchrotron beamline scientist. The user has a beamline "
    "session loaded — a live stream or a saved scan from a file — and you have extra "
    "domain tools (the mcp__dashpva__* tools) on top of your standard toolset:\n"
    "- orient: get_scan_info / list_available_frames / list_known_pvs\n"
    "- evolution: get_feature_timeseries / get_feature_statistics / detect_anomalies\n"
    "- PV relations: correlate_series (a feature vs a 'pv:<name>')\n"
    "- single frame: compute_radial_profile / fit_peak / get_roi_statistics / "
    "check_saturation / get_frame_image_summary\n"
    "Prefer these domain tools for questions about the loaded scan; use your general "
    "tools (Bash, file read/write, etc.) for everything else (scripting, plotting, "
    "reading related files). NEVER state a measured number you did not read from a "
    "tool result. PV history exists only if the scan was recorded in 'scan' caching "
    "mode; if PV tools report no data, say so. End substantive analyses with: "
    "## Answer (cite exact numbers + the tool/frame each came from) / ## Confidence "
    "(high|medium|low + one sentence why) / ## What I did not verify."
)

# Self-contained prompt used only in read-only mode (when Claude Code's built-in
# tools — and thus its preset prompt's tool scaffolding — are disabled).
AGENT_SYSTEM_PROMPT = (
    "You are an expert beamline scientist analyzing data from a synchrotron "
    "X-ray experiment, assisting a researcher who needs rigorous, evidence-backed "
    "analysis of a beamline session — a live stream or a saved scan loaded from a "
    "file. Think through what to check, then use the tools to gather evidence "
    "before you answer. Work like a careful investigator, not a describer.\n\n"
    "Tool discipline:\n"
    "- If you don't know what's available, call get_scan_info / "
    "list_available_frames / list_known_pvs first.\n"
    "- NEVER state a number you did not read from a tool result. To see how "
    "something evolves use get_feature_timeseries / get_feature_statistics / "
    "detect_anomalies; to relate detector behavior to beamline conditions use "
    "correlate_series; to examine a specific frame use compute_radial_profile / "
    "fit_peak / get_roi_statistics / check_saturation / get_frame_image_summary.\n"
    "- If a tool errors or returns no data, say so explicitly — do not paper over "
    "gaps. PV history exists only if the scan was recorded in 'scan' caching mode; "
    "if PV tools report no data, say so.\n\n"
    "End every substantive reply with exactly this structure:\n"
    "## Answer\n"
    "<direct answer; cite exact numbers and the tool/frame each came from>\n"
    "## Confidence\n"
    "<high|medium|low> - <one sentence why>\n"
    "## What I did not verify / would need to check\n"
    "<bullets: missing data, assumptions, PVs/frames not inspected>"
)

# Default extended-thinking budget (tokens). Native reasoning replaces the old
# scratchpad scaffolding; override via [CHAT_TOOLS].THINKING_BUDGET_TOKENS or the
# constructor. Set to 0 / None to disable.
DEFAULT_THINKING_TOKENS = 6000


@dataclass
class AgentEvent:
    """A normalized event streamed out of the SDK loop (UI / printer agnostic)."""

    kind: str  # assistant_text | thinking | tool_call | tool_result | result | error
    text: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_input: dict | None = None
    tool_result: str = ""
    is_error: bool = False
    info: dict = field(default_factory=dict)  # result metadata (cost, turns, ...)
    raw: Any = None  # the original SDK message/block, for advanced consumers


class SdkAgent:
    """A connected Claude Agent SDK client bound to a DashPVA ``ScanReader``.

    Use as an async context manager::

        async with SdkAgent(reader, settings, base_url=url) as agent:
            async for ev in agent.ask("Characterize this scan."):
                print(ev)
    """

    def __init__(
        self,
        reader,
        settings,
        *,
        base_url: str,
        model: str = DEFAULT_MODEL,
        auth_token: str | None = None,
        cli_path: str | os.PathLike = DEFAULT_CLI_PATH,
        system_prompt: str | dict | None = None,
        vision_enabled: bool = False,
        enable_builtin_tools: bool = True,
        can_use_tool=None,
        cwd: str | os.PathLike | None = None,
        max_turns: int = 24,
        thinking_tokens: int | None = None,
    ):
        self.registry = build_tool_registry(reader, settings, vision_enabled=vision_enabled)
        self.server, mcp_allowed = build_mcp_server(self.registry)
        token = (
            auth_token
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ARGO_USER")
            or _login_name()
        )

        if enable_builtin_tools:
            # "Claude Code + domain tools." Safe read-only built-ins + our domain
            # tools auto-run; mutating/exec/network built-ins are NOT pre-approved,
            # so each one routes to the permission callback for the user to confirm.
            # No bypassPermissions — that would auto-approve everything.
            allowed = mcp_allowed + list(AUTO_ALLOW_BUILTINS)
            disallowed: list[str] = []
            permission_mode = "default"
            handler = can_use_tool or stdin_permission_handler
            sp = system_prompt if system_prompt is not None else {
                "type": "preset", "preset": "claude_code", "append": BEAMLINE_APPEND,
            }
        else:
            # Read-only domain agent: built-ins banned, only safe domain tools — so
            # auto-approving them is fine and frictionless.
            allowed = mcp_allowed
            disallowed = list(BUILTIN_TOOLS)
            permission_mode = "bypassPermissions"
            handler = None
            sp = system_prompt if system_prompt is not None else AGENT_SYSTEM_PROMPT

        budget = _resolve_thinking_tokens(thinking_tokens, settings)
        self.options = ClaudeAgentOptions(
            model=model,
            cli_path=str(Path(cli_path).expanduser()),
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            mcp_servers={SERVER_NAME: self.server},
            allowed_tools=allowed,
            disallowed_tools=disallowed,
            system_prompt=sp,
            permission_mode=permission_mode,
            can_use_tool=handler,
            setting_sources=[],  # ignore user/project/local Claude settings — clean agent
            max_turns=max_turns,
            # Native extended thinking — Claude plans/reasons here, not via tools.
            max_thinking_tokens=budget,
            # SDK merges this over os.environ (subprocess_cli.py), so PATH/HOME survive.
            env={"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": token},
        )
        self._client: ClaudeSDKClient | None = None
        self._tool_names: dict[str, str] = {}  # tool_use_id -> tool name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SdkAgent":
        self._client = ClaudeSDKClient(options=self.options)
        await self._client.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    async def ask(self, question: str, *, reset_budgets: bool = True) -> AsyncIterator[AgentEvent]:
        """Run one question through the SDK loop, yielding normalized events.

        Iterate to completion — do not ``break`` early (the SDK's
        ``receive_response`` needs to drain or asyncio cleanup hangs).
        """
        if self._client is None:
            raise RuntimeError("SdkAgent must be used as 'async with SdkAgent(...)'")
        if reset_budgets:
            reset_turn_budgets(self.registry)
        await self._client.query(question)
        async for msg in self._client.receive_response():
            for event in self._translate(msg):
                yield event

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _translate(self, msg) -> list[AgentEvent]:
        cls = type(msg).__name__
        if cls == "AssistantMessage":
            return self._translate_assistant(msg)
        if cls == "UserMessage":
            return self._translate_user(msg)
        if cls == "ResultMessage":
            return [self._translate_result(msg)]
        return []  # SystemMessage / StreamEvent / RateLimitEvent — ignored

    def _translate_assistant(self, msg) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        for block in msg.content or []:
            bc = type(block).__name__
            if bc == "TextBlock":
                if block.text:
                    events.append(AgentEvent("assistant_text", text=block.text, raw=block))
            elif bc == "ThinkingBlock":
                thinking = getattr(block, "thinking", "")
                if thinking:
                    events.append(AgentEvent("thinking", text=thinking, raw=block))
            elif bc == "ToolUseBlock":
                self._tool_names[block.id] = block.name
                events.append(AgentEvent(
                    "tool_call",
                    tool_name=_short_name(block.name),
                    tool_id=block.id,
                    tool_input=dict(block.input or {}),
                    raw=block,
                ))
        return events

    def _translate_user(self, msg) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        content = msg.content
        if not isinstance(content, list):
            return events
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            tid = getattr(block, "tool_use_id", "")
            events.append(AgentEvent(
                "tool_result",
                tool_name=_short_name(self._tool_names.get(tid, "")),
                tool_id=tid,
                tool_result=_block_text(block.content),
                is_error=bool(getattr(block, "is_error", False)),
                raw=block,
            ))
        return events

    def _translate_result(self, msg) -> AgentEvent:
        return AgentEvent(
            "result",
            text=getattr(msg, "result", "") or "",
            is_error=bool(getattr(msg, "is_error", False)),
            info={
                "num_turns": getattr(msg, "num_turns", None),
                "total_cost_usd": getattr(msg, "total_cost_usd", None),
                "duration_ms": getattr(msg, "duration_ms", None),
                "subtype": getattr(msg, "subtype", None),
            },
            raw=msg,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_name(name: str) -> str:
    """Strip the ``mcp__<server>__`` prefix for display (keeps the bare tool name)."""
    if name.startswith("mcp__"):
        return name.split("__", 2)[-1]
    return name


def _block_text(content) -> str:
    """Normalize a ToolResultBlock's content (str | list[dict|block]) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") if item.get("type") == "text"
                             else json.dumps(item, default=str))
            else:
                parts.append(getattr(item, "text", str(item)))
        return "\n".join(p for p in parts if p)
    return str(content)


async def stdin_permission_handler(tool_name: str, tool_input: dict, context):
    """Default permission callback: confirm a tool call on the terminal (y/N).

    Used by the headless runner. Auto-approves the safe read-only built-ins (so it
    only ever asks about the mutating/exec ones), and denies when there is no TTY
    to ask at (safe default). The UI supplies its own dialog-based handler instead.
    """
    if tool_name in AUTO_ALLOW_BUILTINS:
        return PermissionResultAllow()
    summary = json.dumps(tool_input, default=str)
    if len(summary) > 300:
        summary = summary[:300] + " …"
    if not (sys.stdin and sys.stdin.isatty()):
        return PermissionResultDeny(
            message=f"{tool_name} not run: no TTY available to confirm.")
    answer = (await asyncio.to_thread(
        input, f"\n[confirm] Allow {tool_name}? {summary}\n  [y/N] ")).strip().lower()
    if answer in ("y", "yes"):
        return PermissionResultAllow()
    return PermissionResultDeny(message=f"User declined to run {tool_name}.")


def _resolve_thinking_tokens(explicit: int | None, settings) -> int | None:
    """Pick the extended-thinking budget: explicit arg → ``[CHAT_TOOLS]
    .THINKING_BUDGET_TOKENS`` → :data:`DEFAULT_THINKING_TOKENS`. A falsy value
    (0/None) disables thinking (returns None)."""
    if explicit is not None:
        return int(explicit) or None
    chat_cfg = getattr(settings, "CHAT_TOOLS", {}) or {}
    cfg_val = chat_cfg.get("THINKING_BUDGET_TOKENS")
    if cfg_val is not None:
        return int(cfg_val) or None
    return DEFAULT_THINKING_TOKENS or None


def _login_name() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return "user"