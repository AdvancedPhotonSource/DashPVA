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
        'label': 'PVA Workflow Setup',
        'section': 'Setup',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'setup'],
        'running_text': 'PVA Workflow Setup — Running…',
        'tooltip': 'Open PVA Workflow Setup (CLI: DashPVA setup)',
        'edition': 'full',
    },
    {
        'key': 'ioc_rsm_parameter',
        'label': 'HKL Setup (IOC RSM parameter)',
        'section': 'Setup',
        'cmd': [sys.executable, '-m', 'dashpva.consumers.ioc_rsm_parameter'],
        'running_text': 'IOC RSM Parameter — Running…',
        'tooltip': 'Launch IOC for RSM conversion parameters (motor PVs, energy, detector setup)',
        'edition': 'full',
    },
    # stream live
    {
        'key': 'area_det',
        'label': 'Area Detector 2D',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'detector'],
        'running_text': 'Area Detector — Running…',
        'tooltip': 'Open Area detector (CLI: DashPVA detector)',
        'edition': 'full',
    },
    {
        'key': 'hkl3d',
        'label': 'HKL 3D',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'hkl3d'],
        'running_text': 'HKL 3D — Running…',
        'tooltip': 'Open HKL 3D (CLI: DashPVA hkl3d)',
        'edition': 'full',
    },
    {
        'key': 'pyfai',
        'label': 'pyFAI 1D Reduction',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'pyfai'],
        'running_text': 'pyFAI 1D Reduction — Running…',
        'tooltip': 'Live azimuthal integration (CLI: DashPVA pyfai)',
        'edition': 'full',
    },
    {
        'key': 'phase_fitter_live',
        'label': 'XRD Phase Fitter',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'phasefitter'],
        'running_text': 'XRD Phase Fitter — Running…',
        'tooltip': 'Live XRD phase fitting (CLI: DashPVA phasefitter)',
        'edition': 'both',
    },
    {
        'key': 'monitor_scan',
        'label': 'Scan Monitor',
        'section': 'Monitor',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'monitor', 'scan'],
        'running_text': 'Scan Monitors — Running…',
        'tooltip': 'Open Scan monitor (CLI: DashPVA monitor scan)',
        'edition': 'full',
    },
    {
        'key': 'scan_viz',
        'label': '2D Scan Visualization',
        'section': 'Monitor',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'monitor', 'scan'],
        'running_text': '2D Scan Visualization — Running…',
        'tooltip': 'Live 2D scan data collection and visualization',
        'edition': 'full',
    },
    # post analysis
    {
        'key': 'workbench',
        'label': 'Workbench',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'workbench'],
        'running_text': 'Workbench — Running…',
        'tooltip': 'Open Workbench (CLI: DashPVA workbench)',
        'edition': 'both',
    },
    {
        'key': 'h5viewer',
        'label': 'HDF5 Viewer',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'h5viewer'],
        'running_text': 'HDF5 Viewer — Running…',
        'tooltip': 'Interactive HDF5 file browser and image viewer (CLI: DashPVA h5viewer)',
        'edition': 'both',
    },
    {
        'key': 'phase_fitter',
        'label': 'XRD Phase Fitter',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'phasefitter'],
        'running_text': 'XRD Phase Fitter — Running…',
        'tooltip': 'XRD phase fitting — file or live mode (CLI: DashPVA phasefitter)',
        'edition': 'both',
    },
    {
        'key': 'metadata_converter',
        'label': 'Metadata Converter',
        'section': 'Tools',
        'cmd': [sys.executable, '-m', 'dashpva.viewer.tools.metadata_converter_gui'],
        'running_text': 'Metadata Converter — Running…',
        'tooltip': 'Open the Metadata Converter tool',
        'edition': 'both',
    },
    {
        'key': 'file_convert',
        'label': 'HDF5 Converter',
        'section': 'Tools',
        'cmd': [sys.executable, '-m', 'dashpva.viewer.tools.file_convert'],
        'running_text': 'HDF5 Converter — Running…',
        'tooltip': 'Convert folder(s) to HDF5 in standard structure',
        'edition': 'both',
    },
    # bayesian
    {
        'key': 'bayesian_scan',
        'label': 'Bayesian Scan',
        'section': 'Bayesian',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'bayesian'],
        'running_text': 'Bayesian Scan — Running…',
        'tooltip': 'Open Bayesian 2-D Scan Viewer (CLI: DashPVA bayesian)',
        'edition': 'both',
    },
]


def get_views() -> list:
    """Return VIEWS filtered to the installed edition."""
    try:
        import dashpva.settings as settings
        edition_file = settings.PROJECT_ROOT / '.dashpva_edition'
        edition = edition_file.read_text().strip() if edition_file.exists() else 'full'
    except Exception:
        edition = 'both'

    return [v for v in VIEWS if v.get('edition', 'both') in (edition, 'both')]
