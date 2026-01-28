import sys

# Define view entries to be rendered as buttons in the Launcher.
# To add a new view, append another dict with the same keys.
VIEWS = [
    {
        'key': 'scan_monitors',
        'label': 'Scan Monitors',
        'cmd': [sys.executable, 'dashpva.py', 'view', 'scan'],
        'running_text': 'Scan Monitors — Running…',
        'tooltip': 'Open Scan Monitors (CLI: dashpva.py view scan)'
    },
]
