"""
Abstract LLM backend interface and factory.

Usage
-----
    from dashpva.analysis.llm_backend import make_backend
    import dashpva.settings as app_settings

    backend = make_backend(app_settings.SESSION_ANALYSIS)
    response = backend.complete(prompt="...", system="You are a scientist...")

Supported backends
------------------
  "ollama" — local ollama server (llama3.2, phi3, etc.)
  "argo"   — ANL on-site Argo API proxy (gpt-4o, etc.)

The backend is selected by the BACKEND key in the SESSION_ANALYSIS config dict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Minimal interface every backend must implement."""

    @abstractmethod
    def complete(self, prompt: str, system: str = '') -> str:
        """
        Send a single user prompt and return the model's text response.

        Parameters
        ----------
        prompt : str
            The full user message (may be multi-paragraph).
        system : str
            Optional system/instruction message.  Not all backends honour this
            (e.g. vision-only models) — implementations silently ignore it when
            unsupported.

        Returns
        -------
        str
            Model response text.  Raises on connection / auth errors.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'ollama/llama3.2'."""


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