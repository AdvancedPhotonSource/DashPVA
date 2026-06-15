"""
Local ollama backend for LLM completion.

Uses the ollama HTTP API (defaults to http://localhost:11434). Provides an
optional helper to start the ollama server in the background if it isn't
already running on the configured URL.
"""

from __future__ import annotations

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

    def complete(self, prompt: str, system: str = '') -> str:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    'model': self.model,
                    'messages': messages,
                    'stream': False,
                },
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
        content = message.get('content', '')
        if not content:
            # ollama can also stream-chunk responses where content lives elsewhere
            content = data.get('response', '')
        return content.strip()


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