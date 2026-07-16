import sys

"""
Define view entries to be rendered as buttons in the Launcher.
Each entry must have a 'section' key that determines which group it appears under.

Fields per entry:
  - key: unique key for the module
  - label: button label
  - section: grouping header
  - cmd: list command to execute
  - tooltip: short help text
  - requires: tuple of layer names that must be active for this entry to appear,
               e.g. ('area-det',) or ('area-det', 'standalone'). Empty tuple = core (always shown).
"""
# Sections are rendered in the order they first appear in this list.
# To add a new view, append another dict with the same keys.
VIEWS = [
    # setup
    {
        'key': 'setup',
        'label': 'PVA Workflow Setup',
        'section': 'Setup',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'setup'],
        'tooltip': 'Open PVA Workflow Setup (CLI: DashPVA setup)',
        'requires': ('area-det',),
    },
    {
        'key': 'ioc_rsm_parameter',
        'label': 'HKL Setup (IOC RSM parameter)',
        'section': 'Setup',
        'cmd': [sys.executable, '-m', 'dashpva.consumers.ioc_rsm_parameter'],
        'tooltip': 'Launch IOC for RSM conversion parameters (motor PVs, energy, detector setup)',
        'requires': ('area-det',),
    },
    # stream live
    {
        'key': 'area_det',
        'label': 'Area Detector 2D',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'detector'],
        'tooltip': 'Open Area detector (CLI: DashPVA detector)',
        'requires': ('area-det',),
    },
    {
        'key': 'hkl3d',
        'label': 'HKL 3D',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'hkl3d'],
        'tooltip': 'Open HKL 3D (CLI: DashPVA hkl3d)',
        'requires': ('standalone',),
    },
    {
        'key': 'pyfai',
        'label': 'pyFAI 1D Reduction',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'pyfai'],
        'tooltip': 'Live azimuthal integration (CLI: DashPVA pyfai)',
        'requires': ('area-det',),
    },
    {
        'key': 'phase_fitter_live',
        'label': 'XRD Phase Fitter',
        'section': 'Stream Live',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'phasefitter'],
        'tooltip': 'Live XRD phase fitting (CLI: DashPVA phasefitter)',
        'requires': ('area-det', 'standalone'),
    },
    {
        'key': 'monitor_scan',
        'label': 'Scan Monitor',
        'section': 'Monitor',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'monitor', 'scan'],
        'tooltip': 'Open Scan monitor (CLI: DashPVA monitor scan)',
        'requires': ('area-det',),
    },
    {
        'key': 'scan_viz',
        'label': '2D Scan Visualization',
        'section': 'Monitor',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'monitor', 'scan'],
        'tooltip': 'Live 2D scan data collection and visualization',
        'requires': ('area-det',),
    },
    # post analysis
    {
        'key': 'workbench',
        'label': 'Workbench',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'workbench'],
        'tooltip': 'Open Workbench (CLI: DashPVA workbench)',
        'requires': ('standalone',),
    },
    {
        'key': 'h5viewer',
        'label': 'HDF5 Viewer',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'h5viewer'],
        'tooltip': 'Interactive HDF5 file browser and image viewer (CLI: DashPVA h5viewer)',
        'requires': (),
    },
    {
        'key': 'phase_fitter',
        'label': 'XRD Phase Fitter',
        'section': 'Post Analysis',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'phasefitter'],
        'tooltip': 'XRD phase fitting — file or live mode (CLI: DashPVA phasefitter)',
        'requires': ('standalone',),
    },
    {
        'key': 'metadata_converter',
        'label': 'Metadata Converter',
        'section': 'Tools',
        'cmd': [sys.executable, '-m', 'dashpva.viewer.tools.metadata_converter_gui'],
        'tooltip': 'Open the Metadata Converter tool',
        'requires': (),
    },
    {
        'key': 'file_convert',
        'label': 'HDF5 Converter',
        'section': 'Tools',
        'cmd': [sys.executable, '-m', 'dashpva.viewer.tools.file_convert'],
        'tooltip': 'Convert folder(s) to HDF5 in standard structure',
        'requires': (),
    },
    # bayesian
    {
        'key': 'bayesian_scan',
        'label': 'Bayesian Scan',
        'section': 'Bayesian',
        'cmd': [sys.executable, '-m', 'dashpva.cli', 'bayesian'],
        'tooltip': 'Open Bayesian 2-D Scan Viewer (CLI: DashPVA bayesian)',
        'requires': ('bayesian',),
    },
]

EDITION_LAYERS = {
    'area-det':   frozenset({'area-det'}),
    'standalone': frozenset({'standalone'}),
    'bayesian':   frozenset({'area-det', 'bayesian'}),
    'full':       frozenset({'area-det', 'standalone', 'bayesian'}),
}


def get_views(edition=None) -> list:
    """Return VIEWS filtered to the installed edition."""
    if edition is None:
        try:
            import dashpva.settings as settings
            f = settings.PROJECT_ROOT / '.dashpva_edition'
            edition = f.read_text().strip() if f.exists() else 'full'
        except Exception:
            edition = 'full'
    layers = EDITION_LAYERS.get(edition, EDITION_LAYERS['full'])
    return [v for v in VIEWS if set(v.get('requires', ())) <= layers]
