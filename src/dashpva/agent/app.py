"""Entry point for the standalone agent UI.

``python -m dashpva.agent`` and ``DashPVA agent`` (with no headless ``-q``) land
here: build the ``QApplication``, optionally preload a scan / config, show the
window, and run the Qt loop.
"""

from __future__ import annotations

import argparse
import sys

import dashpva.settings as settings
from dashpva.agent.sdk_agent import DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashpva.agent",
        description="Standalone beamline-analysis agent UI (Claude Agent SDK).",
    )
    parser.add_argument("--scan", default=None, help="Optional .h5 scan to preload.")
    parser.add_argument("--config", default=None,
                        help="Config locator (TOML path or DB profile) to load into settings.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Argo model id.")
    parser.add_argument("--vision", action="store_true",
                        help="Enable the describe_frame vision tool.")
    parser.add_argument("--read-only", action="store_true",
                        help="Start with domain tools only (Claude Code's built-in "
                             "Bash/file/web tools off; toggle 'Full tools' in the UI).")
    args = parser.parse_args(argv)

    if args.config:
        try:
            settings.set_locator(args.config)
            settings.reload()
        except Exception as e:
            print(f"error: failed to load config {args.config!r}: {e}", file=sys.stderr)
            return 2

    # Import the UI lazily so argparse / --help don't require a Qt display.
    from dashpva.agent.ui import launch

    return launch(scan=args.scan, model=args.model, vision=args.vision,
                  full_tools=not args.read_only)


if __name__ == "__main__":
    raise SystemExit(main())