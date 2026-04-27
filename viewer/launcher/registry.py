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
  - edition: 'full' | 'standalone' | 'both'
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
        'tooltip': 'Open Setup (CLI: dashpva.py setup)',
        'edition': 'full',
    },
    # streaming
    {
        'key': 'area_det',
        'label': 'Area Detector 2D',
        'section': 'Streaming',
        'cmd': [sys.executable, 'dashpva.py', 'detector'],
        'running_text': 'Area Detector — Running…',
        'tooltip': 'Open Area detector (CLI: dashpva.py detector)',
        'edition': 'full',
    },
    {
        'key': 'hkl3d',
        'label': 'HKL 3D',
        'section': 'Streaming',
        'cmd': [sys.executable, 'dashpva.py', 'hkl3d'],
        'running_text': 'HKL 3D — Running…',
        'tooltip': 'Open HKL 3D (CLI: dashpva.py hkl3d)',
        'edition': 'full',
    },
    {
        'key': 'ioc_rsm_parameter',
        'label': 'IOC RSM Parameter',
        'section': 'Streaming',
        'cmd': [sys.executable, 'consumers/ioc_rsm_parameter.py'],
        'running_text': 'IOC RSM Parameter — Running…',
        'tooltip': 'Launch IOC for RSM conversion parameters (motor PVs, energy, detector setup)',
        'edition': 'full',
    },
    {
        'key': 'monitor_scan',
        'label': 'Scan',
        'section': 'Monitor',
        'cmd': [sys.executable, 'dashpva.py', 'monitor', 'scan'],
        'running_text': 'Scan Monitors — Running…',
        'tooltip': 'Open Scan monitor (CLI: dashpva.py monitor scan)',
        'edition': 'full',
    },
    # post analysis
    {
        'key': 'workbench',
        'label': 'Workbench',
        'section': 'Post Analysis',
        'cmd': [sys.executable, 'dashpva.py', 'workbench'],
        'running_text': 'Workbench — Running…',
        'tooltip': 'Open Workbench (CLI: dashpva.py workbench)',
        'edition': 'both',
    },
    {
        'key': 'metadata_converter',
        'label': 'Metadata Converter',
        'section': 'Tools',
        'cmd': [sys.executable, 'viewer/tools/metadata_converter_gui.py'],
        'running_text': 'Metadata Converter — Running…',
        'tooltip': 'Open the Metadata Converter tool',
        'edition': 'both',
    },
    {
        'key': 'file_convert',
        'label': 'File Convert',
        'section': 'Tools',
        'cmd': [sys.executable, 'viewer/tools/file_convert.py'],
        'running_text': 'File Convert — Running…',
        'tooltip': 'Convert folder(s) to HDF5 in standard structure',
        'edition': 'both',
    },
    # bayesian
    {
        'key': 'bayesian_scan',
        'label': 'Bayesian Scan',
        'section': 'Bayesian',
        'cmd': [sys.executable, 'dashpva.py', 'bayesian'],
        'running_text': 'Bayesian Scan — Running…',
        'tooltip': 'Open Bayesian 2-D Scan Viewer (CLI: dashpva.py bayesian)',
        'edition': 'both',
    },
]


def get_views() -> list:
    """Return VIEWS filtered to the installed edition."""
    try:
        import settings
        edition_file = settings.PROJECT_ROOT / '.dashpva_edition'
        edition = edition_file.read_text().strip() if edition_file.exists() else 'full'
    except Exception:
        edition = 'both'

    if edition == 'full':
        return VIEWS
    return [v for v in VIEWS if v.get('edition', 'both') in ('standalone', 'both')]
