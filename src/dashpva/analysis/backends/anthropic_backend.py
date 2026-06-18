"""Native Anthropic Messages backend (via ANL's Argo proxy).

Unlike :class:`~dashpva.analysis.backends.argo_backend.ArgoBackend` (which uses
Argo's OpenAI-compatible ``/v1/chat/completions`` path), this backend talks to
Argo's *native* Anthropic Messages endpoint (``/v1/messages``) using the
``anthropic`` SDK with a ``base_url`` override — the configuration Argo's docs
explicitly endorse. That unlocks Claude's genuine **extended thinking**: the
model reasons internally (and, with interleaved thinking, *between* tool calls)
before answering, which is materially stronger for multi-step quantitative
investigation than prompt-induced "think step by step".

This composes with the backend-agnostic deliberate-mode protocol in
:class:`~dashpva.analysis.chat_controller.ChatController`: that protocol shapes
the *output* (evidence + confidence + uncertainty) on every backend, while
native thinking deepens the *reasoning* on Claude.

Wire format
-----------
The controller stores history in a normalized OpenAI-ish shape. This backend
translates it to Anthropic content blocks on the way out, and translates the
response back to the normalized shape on the way in. Crucially, it returns the
raw response content blocks under ``_provider_blocks`` so that thinking blocks
(with their signatures) round-trip faithfully across tool rounds — Anthropic
requires the thinking block to lead an assistant turn that contains tool_use.

Authentication is identical to the Argo OpenAI path: the ANL domain username is
passed as the ``api_key`` (it is NOT a secret). Requires a Claude model
(``claudesonnet46``, ``claudeopus47``, ...).
"""

from __future__ import annotations

import os

from dashpva.analysis.llm_backend import LLMBackend

_DEFAULT_BASE_URL = "https://apps.inside.anl.gov/argoapi"
# Interleaved thinking lets Claude think between tool calls. Sent as a beta
# header; if Argo rejects it we retry once without it (see chat()).
_INTERLEAVED_BETA = "interleaved-thinking-2025-05-14"


class AnthropicBackend(LLMBackend):
    """Backend that posts to Argo's native Anthropic Messages endpoint with
    extended thinking enabled."""

    def __init__(self, config: dict):
        self.base_url = (config.get('ARGO_BASE_URL') or _DEFAULT_BASE_URL).rstrip('/')
        self.model = (config.get('ANTHROPIC_MODEL')
                      or config.get('ARGO_MODEL') or 'claudesonnet46')
        self.user = (
            config.get('ARGO_USER') or os.environ.get('ARGO_USER', '') or ''
        ).strip()
        self.timeout = int(config.get('ARGO_TIMEOUT', 120))
        self.max_tokens = int(config.get('ANTHROPIC_MAX_TOKENS')
                              or config.get('ARGO_MAX_TOKENS', 4096))
        budget = config.get('THINKING_BUDGET_TOKENS')
        self.thinking_budget = int(budget) if budget else 0
        self.interleaved = bool(config.get('INTERLEAVED_THINKING', True))

        if not self.user:
            raise RuntimeError(
                "Argo user not configured. Set the ARGO_USER environment "
                "variable (or SESSION_ANALYSIS.ARGO_USER in the active TOML "
                "profile) to your bare ANL domain username."
            )
        if '@' in self.user or '"' in self.user or "'" in self.user:
            raise RuntimeError(
                f"ARGO_USER must be your bare ANL domain username, not an email "
                f"or quoted string. Got: {self.user!r}"
            )
        # If a thinking budget is requested, max_tokens must exceed it.
        if self.thinking_budget:
            self.max_tokens = max(self.max_tokens, self.thinking_budget + 1024)

        self._client = self._make_client()

    def _make_client(self):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required for the native Anthropic "
                "thinking backend. Install it (e.g. `uv sync --extra full`) or "
                "switch SESSION_ANALYSIS.BACKEND to 'argo' (OpenAI-compatible) "
                "or 'ollama'."
            ) from e
        return Anthropic(api_key=self.user, base_url=self.base_url,
                         timeout=self.timeout)

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        system_text, anthropic_messages = _to_anthropic_messages(messages)
        params: dict = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': anthropic_messages,
        }
        if system_text:
            params['system'] = system_text
        if tools:
            params['tools'] = [_to_anthropic_tool(t) for t in tools]
        if self.thinking_budget:
            params['thinking'] = {'type': 'enabled',
                                  'budget_tokens': self.thinking_budget}

        resp = self._create_with_fallback(params)
        return _parse_response(resp)

    def _create_with_fallback(self, params: dict):
        """Call messages.create, degrading gracefully when Argo's proxy does not
        support an optional capability (interleaved-thinking beta, or thinking
        itself). An unsupported extra must never break the chat."""
        from anthropic import APIError, APIStatusError

        attempts = []
        if self.thinking_budget and self.interleaved:
            attempts.append({'extra_headers': {'anthropic-beta': _INTERLEAVED_BETA}})
        attempts.append({})                       # plain (still with thinking)
        if 'thinking' in params:
            attempts.append({'_drop_thinking': True})   # last resort: no thinking

        last_exc: Exception | None = None
        for opt in attempts:
            call = dict(params)
            extra_headers = opt.get('extra_headers')
            if opt.get('_drop_thinking'):
                call.pop('thinking', None)
            try:
                if extra_headers:
                    return self._client.messages.create(**call, extra_headers=extra_headers)
                return self._client.messages.create(**call)
            except (APIStatusError, APIError) as e:
                last_exc = e
                continue
            except Exception as e:  # network etc. — don't keep retrying blindly
                raise RuntimeError(f"Argo (Anthropic endpoint) request failed: {e}") from e
        raise RuntimeError(
            f"Argo (Anthropic endpoint) rejected the request: {last_exc}")


# ----------------------------------------------------------------------
# Translation helpers
# ----------------------------------------------------------------------

def _to_anthropic_tool(openai_tool: dict) -> dict:
    fn = openai_tool.get('function') or {}
    return {
        'name': fn.get('name', ''),
        'description': fn.get('description', ''),
        'input_schema': fn.get('parameters') or {'type': 'object', 'properties': {}},
    }


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Translate normalized history into (system_text, anthropic_messages).

    Consecutive user/tool messages are merged into a single user turn so the
    conversation strictly alternates user/assistant, as Anthropic requires
    (tool_result and text blocks may share one user turn).
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending_user: list[dict] = []

    def flush_user():
        if pending_user:
            out.append({'role': 'user', 'content': list(pending_user)})
            pending_user.clear()

    for m in messages:
        role = m.get('role')
        if role == 'system':
            if m.get('content'):
                system_parts.append(str(m['content']))
        elif role == 'user':
            if m.get('content'):
                pending_user.append({'type': 'text', 'text': str(m['content'])})
        elif role == 'tool':
            pending_user.append({
                'type': 'tool_result',
                'tool_use_id': m.get('tool_call_id', ''),
                'content': str(m.get('content', '')),
            })
        elif role == 'assistant':
            flush_user()
            out.append({'role': 'assistant', 'content': _assistant_content(m)})

    flush_user()
    return '\n\n'.join(system_parts), out


def _assistant_content(m: dict) -> list[dict]:
    """Build an assistant turn's content blocks. Prefer the verbatim provider
    blocks (so thinking-block signatures round-trip); otherwise reconstruct from
    the normalized text + tool_calls."""
    blocks = m.get('_provider_blocks')
    if blocks:
        return blocks
    rebuilt: list[dict] = []
    if m.get('content'):
        rebuilt.append({'type': 'text', 'text': str(m['content'])})
    for call in (m.get('tool_calls') or []):
        rebuilt.append({
            'type': 'tool_use',
            'id': call.get('id', ''),
            'name': call.get('name', ''),
            'input': call.get('arguments') or {},
        })
    if not rebuilt:
        # Anthropic rejects empty assistant content.
        rebuilt.append({'type': 'text', 'text': '(no content)'})
    return rebuilt


def _parse_response(resp) -> dict:
    """Translate an Anthropic Messages response into the normalized dict, keeping
    raw blocks under ``_provider_blocks`` for faithful round-tripping."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    provider_blocks: list[dict] = []

    for block in (getattr(resp, 'content', None) or []):
        btype = getattr(block, 'type', None)
        provider_blocks.append(_block_to_dict(block))
        if btype == 'text':
            text_parts.append(getattr(block, 'text', '') or '')
        elif btype == 'thinking':
            thinking_parts.append(getattr(block, 'thinking', '') or '')
        elif btype == 'tool_use':
            tool_calls.append({
                'id': getattr(block, 'id', '') or '',
                'name': getattr(block, 'name', '') or '',
                'arguments': getattr(block, 'input', None) or {},
            })

    out: dict = {
        'role': 'assistant',
        'content': '\n'.join(p for p in text_parts if p).strip(),
        '_provider_blocks': provider_blocks,
    }
    if thinking_parts:
        out['thinking'] = '\n'.join(p for p in thinking_parts if p).strip()
    if tool_calls:
        out['tool_calls'] = tool_calls
    return out


def _block_to_dict(block) -> dict:
    """Best-effort conversion of an SDK content block to a plain dict suitable
    for re-sending (preserves thinking signatures)."""
    for attr in ('model_dump', 'dict', 'to_dict'):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(block, dict):
        return block
    # Fallback: reconstruct the minimum by type.
    btype = getattr(block, 'type', None)
    if btype == 'text':
        return {'type': 'text', 'text': getattr(block, 'text', '')}
    if btype == 'tool_use':
        return {'type': 'tool_use', 'id': getattr(block, 'id', ''),
                'name': getattr(block, 'name', ''), 'input': getattr(block, 'input', {})}
    return {'type': btype or 'text', 'text': str(block)}