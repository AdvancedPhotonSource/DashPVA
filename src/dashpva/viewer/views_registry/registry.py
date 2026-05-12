import sys

VIEWS = [
    {
        'key': 'scan_monitors',
        'label': 'Scan Monitors',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'monitor', 'scan'],
        'running_text': 'Scan Monitors — Running…',
        'tooltip': 'Open Scan Monitors (CLI: DashPVA monitor scan)'
    },
]
