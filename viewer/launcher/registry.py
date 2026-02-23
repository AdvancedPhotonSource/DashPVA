import sys

"""
Define view entries to be rendered as buttons in the Launcher.
Each entry must have a 'section' key that determines which group it appears under.

Fields per entry:
  - key: unique key for the module
  - label: button label
  - section: grouping header
  - cmd: list command to execute
  - running_text: button text while running
  - tooltip: short help text
"""
# Sections are rendered in the order they first appear in this list.
# To add a new view, append another dict with the same keys.
VIEWS = [
    # setup
    {
        'key':'setup',
        'label': 'Setup',
        'section': 'Setup',
        'cmd': [sys.executable, 'dashpva.py', 'setup'],
        'running_text': 'Setup — Running…',
        'tooltip': 'Open Setup (CLI: dashpva.py setup)'
    },
    # streaming
    {
        'key': 'area_det',
        'label': 'Area Detector 2D',
        'section': 'Streaming',
        'cmd': [sys.executable, 'dashpva.py', 'hkl3d'],
        'running_text': 'HKL 3D — Running…',
        'tooltip': 'Open HKL 3D (CLI: dashpva.py hkl3d)'
    },
    {
        'key': 'hkl3d',
        'label': 'HKL 3D',
        'section': 'Streaming',
        'cmd': [sys.executable, 'dashpva.py', 'hkl3d'],
        'running_text': 'HKL 3D — Running…',
        'tooltip': 'Open HKL 3D (CLI: dashpva.py hkl3d)'
    },
    {
        'key': 'monitor_scan',
        'label': 'Scan',
        'section': 'Monitor',
        'cmd': [sys.executable, 'dashpva.py', 'monitor', 'scan'],
        'running_text': 'Scan Monitors — Running…',
        'tooltip': 'Open Scan monitor (CLI: dashpva.py monitor scan)'
    },
    # post analysis
    {
        'key': 'slice3d',
        'label': 'HKL 3D Slicer',
        'section': 'Post Analysis',
        'cmd': [sys.executable, 'dashpva.py', 'slice3d'],
        'running_text': 'HKL 3D Slicer — Running…',
        'tooltip': 'Open HKL 3D Slicer (CLI: dashpva.py slice3d)'
    },
    {
        'key': 'workbench',
        'label': 'Workbench',
        'section': 'Post Analysis',
        'cmd': [sys.executable, 'dashpva.py', 'workbench'],
        'running_text': 'Workbench — Running…',
        'tooltip': 'Open Workbench (CLI: dashpva.py workbench)'
    },
    {
        'key': 'metadata_converter',
        'label': 'Metadata Converter',
        'section': 'Tools',
        'cmd': [sys.executable, 'viewer/tools/metadata_converter_gui.py'],
        'running_text': 'Metadata Converter — Running…',
        'tooltip': 'Open the Metadata Converter tool'
    },
    {
        'key': 'file_convert',
        'label': 'File Convert',
        'section': 'Tools',
        'cmd': [sys.executable, 'viewer/tools/file_convert.py'],
        'running_text': 'File Convert — Running…',
        'tooltip': 'Convert folder(s) to HDF5 in standard structure'
    },
]
