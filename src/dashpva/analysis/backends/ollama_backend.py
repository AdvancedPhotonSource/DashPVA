"""
Local ollama backend for LLM chat / completion.

Uses the ollama HTTP API (defaults to http://localhost:11434). Provides an
optional helper to start the ollama server in the background if it isn't
already running on the configured URL.

Tool calling
------------
Ollama supports OpenAI-shape tool calling on a subset of models (``llama3.2``,
``llama3.1``, ``qwen2.5``, ``mistral-nemo`` work; ``moondream`` and older
``phi3`` do not). If the model doesn't support tools, ollama silently ignores
the ``tools`` field and returns a text-only reply — we surface that as a
no-op turn (``content`` present, no ``tool_calls``) and let the
:class:`~dashpva.analysis.chat_controller.ChatController` flag it.
"""

from __future__ import annotations

import json
import subprocess
import time

import requests

from dashpva.analysis.llm_backend import LLMBackend


class OllamaBackend(LLMBackend):
    """Backend that talks to a local ollama server over HTTP."""

    def __init__(self, config: dict):
        self.url = (config.get('OLLAMA_URL') or 'http://localhost:11434').rstrip('/')
        self.model = config.get('OLLAMA_MODEL') or 'llama3.2'
        self.timeout = int(config.get('OLLAMA_TIMEOUT', 120))

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body: dict = {
            'model': self.model,
            'messages': _to_ollama_messages(messages),
            'stream': False,
        }
        if tools:
            body['tools'] = tools

        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json=body,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach ollama at {self.url}. "
                f"Is the server running? ({e})"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Ollama returned non-JSON: {resp.text[:200]}") from e

        message = data.get('message') or {}
        content = (message.get('content') or '').strip()
        if not content:
            # ollama can also stream-chunk responses where content lives elsewhere
            content = (data.get('response') or '').strip()

        out: dict = {'role': 'assistant', 'content': content}

        raw_calls = message.get('tool_calls') or []
        if raw_calls:
            out['tool_calls'] = [_normalize_tool_call(c, i) for i, c in enumerate(raw_calls)]
        return out


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    """Serialize chat history into ollama's expected wire format.

    The :class:`~dashpva.analysis.chat_controller.ChatController` stores assistant
    tool calls in our internal normalized shape (``{id, name, arguments: dict}``).
    Ollama renders tool calls through the model's chat template, which reads
    ``tool_call.function.name`` / ``tool_call.function.arguments`` — so the call
    must be nested under ``function``. Unlike OpenAI/Argo, ollama expects
    ``arguments`` to remain a JSON *object* (dict), not a JSON-encoded string.

    Only assistant messages carrying ``tool_calls`` are rewritten; everything
    else passes through (ollama ignores fields like ``tool_call_id`` it doesn't use).
    """
    out: list[dict] = []
    for m in messages:
        calls = m.get('tool_calls')
        if m.get('role') == 'assistant' and calls:
            out.append({
                'role': 'assistant',
                'content': m.get('content') or '',
                'tool_calls': [_to_ollama_tool_call(c) for c in calls],
            })
        else:
            out.append(m)
    return out


def _to_ollama_tool_call(call: dict) -> dict:
    """Render one normalized ``{id, name, arguments}`` call as ollama's nested shape."""
    fn = call.get('function')
    if isinstance(fn, dict) and isinstance(fn.get('arguments'), dict):
        return call  # already ollama-shaped; pass through defensively
    args = call.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    elif not isinstance(args, dict):
        args = {}
    name = call.get('name') or (fn or {}).get('name', '')
    return {'function': {'name': name, 'arguments': args}}


def _normalize_tool_call(call: dict, idx: int) -> dict:
    """Translate ollama's tool-call shape into our internal ``{id, name, arguments}``."""
    fn = call.get('function') or {}
    args = fn.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {'_raw': args}
    elif not isinstance(args, dict):
        args = {}
    # Ollama omits an id; synthesize one so subsequent role='tool' messages can
    # reference it (Argo and OpenAI both require tool_call_id).
    cid = call.get('id') or f"call_{idx}_{int(time.time() * 1000)}"
    return {'id': cid, 'name': fn.get('name', ''), 'arguments': args}


def ensure_server(url: str = 'http://localhost:11434',
                  max_wait_s: float = 10.0) -> subprocess.Popen | None:
    """
    Ensure an ollama server is reachable at *url*. Returns the spawned process
    if we started one (caller should terminate it on shutdown), or None if a
    server was already running.

    Polls /api/tags every 0.5s for up to *max_wait_s* after spawning before
    giving up.
    """
    base = url.rstrip('/')
    if _ping(base):
        return None

    try:
        proc = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ollama binary not found in PATH; install it from https://ollama.com "
            "or set SESSION_ANALYSIS.BACKEND to a different value."
        ) from e

    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if _ping(base):
            return proc
        time.sleep(0.5)

    proc.terminate()
    raise ConnectionError(
        f"Started ollama serve but {base} did not become reachable within "
        f"{max_wait_s}s."
    )


def _ping(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=1.0)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False