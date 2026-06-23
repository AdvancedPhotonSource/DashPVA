"""``python -m dashpva.agent`` → the standalone agent UI.

For the headless one-shot runner, use ``python -m dashpva.agent.run`` instead.
"""

from __future__ import annotations

from dashpva.agent.app import main

if __name__ == "__main__":
    raise SystemExit(main())