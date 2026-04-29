#!/usr/bin/env python3
"""DashPVA installer — sets up the Python environment and writes the edition file."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
EDITION_FILE = REPO / '.dashpva_edition'


def _version():
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            cwd=str(REPO), capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().lstrip('v')
    except Exception:
        pass
    return '?'


def _print_banner():
    try:
        import pyfiglet
        print(pyfiglet.figlet_format('DashPVA', font='slant'))
    except ImportError:
        print('\n  DashPVA\n')
    print(f'  v{_version()}')
    print()


def _detect_uv():
    return shutil.which('uv') is not None


def _run(cmd):
    print(f'  $ {" ".join(str(c) for c in cmd)}')
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        print(f'\n  ERROR: command exited with code {result.returncode}')
        sys.exit(result.returncode)


def _write_edition(edition: str):
    EDITION_FILE.write_text(edition)
    print(f'\n  Written: .dashpva_edition = {edition}')


def _prompt_edition() -> str:
    if sys.platform != 'linux':
        print('  Only Standalone edition is available on this platform.\n')
        return 'standalone'
    print('  Which edition would you like to install?\n')
    print('    [1] Full        — live streaming + pvaccess/EPICS (Linux only) (recommended)')
    print('    [2] Standalone  — post-analysis tools only (any OS)\n')
    while True:
        choice = input('  Enter 1 or 2 [default: 1]: ').strip() or '1'
        if choice == '1':
            return 'full'
        if choice == '2':
            return 'standalone'
        print('  Please enter 1 or 2.')


def _require_linux_for_full():
    if sys.platform != 'linux':
        print('  ERROR: Full edition (pvaccess/pyepics) is only supported on Linux.')
        print(f'         Detected platform: {sys.platform}')
        print('         Use --standalone or run on Linux.')
        sys.exit(1)


def _print_next_steps():
    print()
    print('  ─────────────────────────────────────────────')
    print('  Setup complete!  Launch DashPVA with:')
    print()
    print('    python dashpva.py run')
    print()
    print('  To update later:')
    print()
    print('    python install.py --update')
    print('  ─────────────────────────────────────────────')
    print()


def main():
    parser = argparse.ArgumentParser(description='DashPVA installer')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='Install Full edition (pvaccess/EPICS)')
    group.add_argument('--standalone', action='store_true', help='Install Standalone edition')
    parser.add_argument('--update', action='store_true', help='Update existing installation')
    args = parser.parse_args()

    _print_banner()

    if args.update and EDITION_FILE.exists():
        edition = EDITION_FILE.read_text().strip()
        print(f'  Updating existing {edition} installation…\n')
    elif args.full:
        _require_linux_for_full()
        edition = 'full'
    elif args.standalone:
        edition = 'standalone'
    else:
        edition = _prompt_edition()
        print()

    # Block downgrade: full → standalone removes pvaccess packages the user relied on
    if edition == 'standalone' and EDITION_FILE.exists() and EDITION_FILE.read_text().strip() == 'full':
        print('  ERROR: Cannot downgrade from Full to Standalone edition.')
        print('         pvaccess/pyepics packages cannot be automatically removed.')
        print('         To switch, set up a fresh environment manually.')
        sys.exit(1)

    has_uv = _detect_uv()

    if has_uv:
        print(f'  Using uv  ({shutil.which("uv")})\n')
        if edition == 'full':
            _require_linux_for_full()
            _run(['uv', 'sync', '--extra', 'full'])
        else:
            _run(['uv', 'sync'])
    else:
        print('  uv not found on PATH.  Falling back to pip.')
        print('  (For faster installs, run:  pip install uv)\n')
        if edition == 'full':
            _require_linux_for_full()
            _run([sys.executable, '-m', 'pip', 'install', '-e', '.[full]'])
        else:
            _run([sys.executable, '-m', 'pip', 'install', '-e', '.'])

    _write_edition(edition)
    _print_next_steps()


if __name__ == '__main__':
    main()
