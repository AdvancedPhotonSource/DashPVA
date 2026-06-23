"""Headless runner for the standalone agent over a saved scan.

Wires the whole Milestone-1 pipeline end-to-end::

    proxy_manager  →  SavedScanReader  →  tool_bridge  →  sdk_agent

and prints the streamed answer + tool trace. This is the go/no-go on utility:
point it at a real recorded ``.h5`` and judge the reasoning.

Example::

    python -m dashpva.agent.run --scan OUTPUT_SCAN.h5 \\
        --question "Characterize this scan: drift? fading rings? cite frame ids."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anyio

import dashpva.settings as settings
from dashpva.agent.proxy_manager import ProxyError, ProxyManager
from dashpva.agent.saved_scan_reader import SavedScanReader
from dashpva.agent.sdk_agent import DEFAULT_MODEL, SdkAgent

# ANSI dim/colors (skipped automatically when stdout isn't a TTY).
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _print_event(ev) -> None:
    if ev.kind == "thinking":
        print(_c("2;3", f"[thinking] {ev.text.strip()}"))
    elif ev.kind == "assistant_text":
        print(ev.text.rstrip())
    elif ev.kind == "tool_call":
        args = ev.tool_input or {}
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(_c("36", f"  🔧 {ev.tool_name}({rendered})"))
    elif ev.kind == "tool_result":
        body = (ev.tool_result or "").strip().replace("\n", " ")
        if len(body) > 400:
            body = body[:400] + " …"
        tag = _c("31", "error") if ev.is_error else _c("2", "ok")
        print(_c("2", f"     ↳ [{tag}] {body}"))
    elif ev.kind == "result":
        info = ev.info or {}
        bits = []
        if info.get("num_turns") is not None:
            bits.append(f"{info['num_turns']} turns")
        if info.get("total_cost_usd") is not None:
            bits.append(f"${info['total_cost_usd']:.4f}")
        if info.get("duration_ms") is not None:
            bits.append(f"{info['duration_ms'] / 1000:.1f}s")
        print(_c("2", f"\n— done ({', '.join(bits)}) —"))


async def _run(reader, question, base_url, model, vision, max_turns, user,
               enable_builtin_tools) -> int:
    async with SdkAgent(
        reader, settings, base_url=base_url, model=model,
        vision_enabled=vision, max_turns=max_turns, auth_token=user,
        enable_builtin_tools=enable_builtin_tools,
    ) as agent:
        saw_error = False
        async for ev in agent.ask(question):
            _print_event(ev)
            if ev.kind == "result" and ev.is_error:
                saw_error = True
        return 1 if saw_error else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashpva.agent.run",
        description="Run the standalone beamline-analysis agent over a saved scan .h5.",
    )
    parser.add_argument("--scan", required=True, help="Path to a recorded scan .h5.")
    parser.add_argument("--question", "-q", required=True, help="The question to ask.")
    parser.add_argument("--config", default=None,
                        help="Config locator (TOML path or DB profile) to load into settings.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Argo model id.")
    parser.add_argument("--user", default=None,
                        help="Argo username (ANTHROPIC_AUTH_TOKEN). Defaults to env/login.")
    parser.add_argument("--vision", action="store_true",
                        help="Enable the describe_frame vision tool (needs ARGO_USER).")
    parser.add_argument("--read-only", action="store_true",
                        help="Domain tools only (disable Claude Code's built-in "
                             "Bash/file/web tools). Default: full tools, with "
                             "mutating/exec commands requiring y/N confirmation.")
    parser.add_argument("--max-turns", type=int, default=24, help="Max agent turns.")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Assume an argo-proxy is already running; do not launch one.")
    parser.add_argument("--port", type=int, default=None, help="argo-proxy port override.")
    args = parser.parse_args(argv)

    scan_path = Path(args.scan).expanduser()
    if not scan_path.exists():
        print(f"error: scan file not found: {scan_path}", file=sys.stderr)
        return 2

    if args.config:
        try:
            settings.set_locator(args.config)
            settings.reload()
        except Exception as e:
            print(f"error: failed to load config {args.config!r}: {e}", file=sys.stderr)
            return 2

    # 1) Reader (validates the file up front).
    try:
        reader = SavedScanReader(scan_path, settings=settings)
    except Exception as e:
        print(f"error: could not open scan: {e}", file=sys.stderr)
        return 2
    print(_c("1", f"Loaded {scan_path.name}: {reader.frames_received} frames, "
                  f"shape {reader.shape}, {len(reader.feature_vector_cache)} feature vectors, "
                  f"PV series: {sorted(reader.cached_ca) or 'none'}"))

    # 2) Proxy sidecar.
    proxy = ProxyManager(port=args.port, user=args.user)
    try:
        if args.no_proxy:
            if not proxy.is_healthy():
                print(f"error: --no-proxy set but nothing healthy at {proxy.base_url}/health",
                      file=sys.stderr)
                return 3
            base_url = proxy.base_url
        else:
            print(_c("2", f"Starting / checking argo-proxy at {proxy.base_url} …"))
            base_url = proxy.ensure_running()
        print(_c("1", f"\nQ: {args.question}\n"))

        # 3) Agent loop.
        return anyio.run(
            _run, reader, args.question,
            base_url, args.model, args.vision, args.max_turns, args.user,
            not args.read_only,
        )
    except ProxyError as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: on an air-gapped host, request a BIS_Argo_Access conduit and run "
              "`argo-proxy serve` manually, then pass --no-proxy.", file=sys.stderr)
        return 3
    finally:
        if not args.no_proxy:
            proxy.stop()


if __name__ == "__main__":
    raise SystemExit(main())