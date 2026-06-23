"""Lifecycle for the ``argo-proxy`` sidecar the Agent SDK talks to.

The Claude Agent SDK / CLI cannot talk to ANL's Argo gateway directly — Claude
Code sends the system prompt as a ``role:"system"`` message, which Argo's
``/v1/messages`` rejects (``400 Unexpected role "system"``). ANL's ``argo-proxy``
exposes a native ``/v1/messages`` and translates the request correctly, so the
SDK points at ``ANTHROPIC_BASE_URL=http://localhost:<port>`` (see
HANDOFF_PHASE4 §2).

This module owns that process the way DashPVA owns its viewer subprocesses
(``cli.py`` uses ``subprocess``): it reuses an already-running proxy if one
answers ``/health``, otherwise spawns ``argo-proxy serve`` and waits until
healthy. The standalone agent process starts it before the first turn and stops
it on shutdown.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 44497
DEFAULT_ARGO_BASE_URL = "https://apps.inside.anl.gov/argoapi"
DEFAULT_CONFIG_PATH = Path("~/.config/argoproxy/config.yaml").expanduser()


class ProxyError(RuntimeError):
    """Raised when the proxy cannot be reached or started."""


class ProxyManager:
    """Start (or reuse) and health-check an ``argo-proxy`` sidecar.

    Args:
        config_path: argo-proxy YAML config (default ``~/.config/argoproxy/config.yaml``).
            Read for host/port/user if present; written from defaults if absent.
        host / port: override the bind address (else taken from config, else defaults).
        user: Argo username for a freshly written config (else ``$ARGO_USER`` or the
            OS login name).
        argo_base_url: upstream Argo endpoint for a freshly written config.
        startup_timeout: seconds to wait for ``/health`` after spawning.
    """

    def __init__(
        self,
        *,
        config_path: str | os.PathLike | None = None,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        argo_base_url: str | None = None,
        startup_timeout: float = 30.0,
    ):
        self.config_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
        cfg = self._read_config(self.config_path)
        self.host = host or cfg.get("host") or DEFAULT_HOST
        self.port = int(port or cfg.get("port") or DEFAULT_PORT)
        self.user = user or cfg.get("user") or os.environ.get("ARGO_USER") or _login_name()
        self.argo_base_url = argo_base_url or cfg.get("argo_base_url") or DEFAULT_ARGO_BASE_URL
        self.startup_timeout = float(startup_timeout)

        self._proc: subprocess.Popen | None = None
        self._owns_process = False  # True only if *we* spawned it

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """True if something answers ``GET /health`` (2xx) at the bind address."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def ensure_running(self) -> str:
        """Reuse a healthy proxy if present, else spawn one and wait. Returns base_url."""
        if self.is_healthy():
            self._owns_process = False
            return self.base_url
        self._write_config_if_absent()
        self._spawn()
        self._wait_healthy()
        return self.base_url

    def stop(self) -> None:
        """Terminate the proxy only if we started it (never kill a reused one)."""
        if not (self._owns_process and self._proc):
            return
        proc = self._proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._proc = None
        self._owns_process = False

    def __enter__(self) -> "ProxyManager":
        self.ensure_running()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spawn(self) -> None:
        exe = self._argo_proxy_executable()
        cmd = [exe, "serve", str(self.config_path), "--no-banner",
               "--host", self.host, "--port", str(self.port)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as e:
            raise ProxyError(f"failed to launch argo-proxy ({cmd!r}): {e}") from e
        self._owns_process = True

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                self._owns_process = False
                raise ProxyError(
                    f"argo-proxy exited early (code {self._proc.returncode}); "
                    f"check its config at {self.config_path}"
                )
            if self.is_healthy():
                return
            time.sleep(0.5)
        self.stop()
        raise ProxyError(
            f"argo-proxy did not become healthy at {self.base_url}/health "
            f"within {self.startup_timeout:.0f}s"
        )

    @staticmethod
    def _argo_proxy_executable() -> str:
        """Prefer the argo-proxy next to the running interpreter (the venv), then PATH."""
        candidate = Path(sys.executable).parent / "argo-proxy"
        if candidate.exists():
            return str(candidate)
        found = shutil.which("argo-proxy")
        if found:
            return found
        raise ProxyError(
            "argo-proxy not found. Install the agent extra: `uv sync --extra agent`."
        )

    @staticmethod
    def _read_config(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            import yaml

            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _write_config_if_absent(self) -> None:
        if self.config_path.exists():
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            'config_version: "3"\n'
            f'user: "{self.user}"\n'
            f"host: {self.host}\n"
            f"port: {self.port}\n"
            f'argo_base_url: "{self.argo_base_url}"\n'
        )
        self.config_path.write_text(content)


def _login_name() -> str:
    try:
        import getpass

        return getpass.getuser()
    except Exception:
        return "user"