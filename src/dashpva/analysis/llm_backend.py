"""
Abstract LLM backend interface and factory.

Usage
-----
    from dashpva.analysis.llm_backend import make_backend
    import dashpva.settings as app_settings

    backend = make_backend(app_settings.SESSION_ANALYSIS)

    # Single-shot (back-compat)
    text = backend.complete(prompt="...", system="You are a scientist...")

    # Multi-turn / tool-calling
    result = backend.chat(
        messages=[{'role': 'system', 'content': '...'},
                  {'role': 'user',   'content': '...'}],
        tools=[<openai function schema>, ...],
    )
    # result is one of:
    #   {'role': 'assistant', 'content': '<text>'}
    #   {'role': 'assistant', 'content': '' | None,
    #    'tool_calls': [{'id': str, 'name': str, 'arguments': dict}, ...]}

Supported backends
------------------
  "ollama" — local ollama server (llama3.2, qwen2.5, phi3, etc.)
  "argo"   — ANL on-site Argo Gateway proxy (gpt5, claudesonnet46, etc.)

The backend is selected by the BACKEND key in the SESSION_ANALYSIS config dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Minimal interface every backend must implement.

    Implementers only need to provide :meth:`chat` and :attr:`name`;
    :meth:`complete` is a concrete default that wraps :meth:`chat` for callers
    that only want single-shot text completion (the original Phase-2 surface).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. ``'ollama/llama3.2'``."""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a multi-turn message list and return the model's response.

        Parameters
        ----------
        messages : list[dict]
            Chat history in our normalized shape::

                [{'role': 'system'|'user', 'content': str},
                 {'role': 'assistant', 'content': str | None,
                  # only when the assistant requested tools — SAME shape this
                  # method returns, NOT the OpenAI wire shape. Each backend
                  # serializes it to its own wire format before sending.
                  'tool_calls': [{'id': str, 'name': str, 'arguments': dict}, ...]},
                 {'role': 'tool', 'tool_call_id': str, 'name': str, 'content': str},
                 ...]
        tools : list[dict] | None
            List of OpenAI function-call tool schemas (see
            :func:`dashpva.analysis.tools.base.tool_to_openai_schema`). When
            ``None``, no tools are advertised and the call behaves like a plain
            chat completion.

        Returns
        -------
        dict
            Either ``{'role': 'assistant', 'content': str}`` for a final text
            answer, or ``{'role': 'assistant', 'content': str | None,
            'tool_calls': [{'id': str, 'name': str, 'arguments': dict}, ...]}``
            when the model wants to call tools.

        Raises
        ------
        ConnectionError
            Network / DNS / VPN failure.
        RuntimeError
            HTTP error, auth rejection, or malformed response from the server.
        """

    def complete(self, prompt: str, system: str = '') -> str:
        """Single-shot convenience wrapper over :meth:`chat`.

        Parameters
        ----------
        prompt : str
            The full user message (may be multi-paragraph).
        system : str
            Optional system / instruction message.

        Returns
        -------
        str
            Model response text. If the model returned tool calls instead of
            text, returns an empty string (callers that need tools should use
            :meth:`chat` directly).
        """
        messages: list[dict] = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        result = self.chat(messages, tools=None)
        return (result.get('content') or '').strip()


def make_backend(config: dict) -> LLMBackend:
    """
    Instantiate the backend named in *config['BACKEND']*.

    Parameters
    ----------
    config : dict
        Typically ``app_settings.SESSION_ANALYSIS``.

    Raises
    ------
    ValueError  — unknown BACKEND value.
    """
    kind = (config.get('BACKEND') or 'ollama').lower()
    if kind == 'ollama':
        from dashpva.analysis.backends.ollama_backend import OllamaBackend
        return OllamaBackend(config)
    if kind == 'argo':
        from dashpva.analysis.backends.argo_backend import ArgoBackend
        return ArgoBackend(config)
    raise ValueError(
        f"Unknown LLM backend {kind!r}. "
        "Set SESSION_ANALYSIS.BACKEND to 'ollama' or 'argo'."
    )