"""
Centralized settings module for DashPVA.

Exports constants resolved from the currently selected configuration source
(TOML file or DB profile), with helper functions to control the locator and
refresh values.

The source is determined by ConfigSource in utils.config.source, which
auto-detects the backend from the locator:
  - TOML path  → reads from file
  - DB profile → reads from database
  - None       → uses minimal hard-coded defaults below

Usage:
  - Programmatic selection:
      import settings
      settings.set_locator('/path/to/config.toml')  # or int profile_id, or "profile:<name>"
      settings.reload()

  - Optional env var override:
      export DASPVA_CONFIG_LOCATOR=profile:my_profile
      # or export DASPVA_CONFIG_LOCATOR=/abs/path/config.toml
      # or export DASPVA_CONFIG_LOCATOR=42  (profile id)

  - Diagnostics:
      settings.SOURCE_TYPE -> "toml", "db", or None
      settings.LOCATOR     -> the current locator (str path, "profile:<name>", or int id)
      settings.CONFIG      -> full configuration dictionary
      settings.ensure_path() -> a TOML path (original path or temp file when using DB)
"""

from typing import Any, Dict, Optional, Union
import os
from pathlib import Path

try:
    from utils.config.source import ConfigSource
except Exception:
    ConfigSource = None  # type: ignore[assignment]


# NeXus / HDF5 structure definition (static — not config-driven)
HDF5_STRUCTURE = {
    "nexus": {
        "default": {
            "NX_class": "NXroot",
            "default": "entry",
            "entry": {
                "NX_class": "NXentry",
                "default": "data",

                # --- INSTRUMENT: The 'How' (Source + Detector) ---
                "instrument": {
                    "NX_class": "NXinstrument",
                    "source": {
                        "NX_class": "NXsource",
                        "target": "HKL/SPEC/ENERGY_VALUE",
                        "units": "keV"
                    },
                    "detector": {
                        "NX_class": "NXdetector",
                        "target": "HKL/DETECTOR_SETUP",
                        "data_link": "/entry/data/data"
                    }
                },

                # --- SAMPLE: The 'What' (Motor Stacks + Environment) ---
                "sample": {
                    "NX_class": "NXsample",
                    "ub_matrix": {
                        "NX_class": "NXcollection",
                        "target": "HKL/SPEC/UB_MATRIX_VALUE"
                    },
                    "geometry": {
                        "NX_class": "NXtransformations",
                        "sample_phi": {"target": "HKL/SAMPLE_CIRCLE_AXIS_4", "type": "rotation"},
                        "sample_chi": {"target": "HKL/SAMPLE_CIRCLE_AXIS_3", "type": "rotation"},
                        "sample_eta": {"target": "HKL/SAMPLE_CIRCLE_AXIS_2", "type": "rotation"},
                        "sample_mu":  {"target": "HKL/SAMPLE_CIRCLE_AXIS_1", "type": "rotation"}
                    }
                },

                # --- DATA: The 'View' (Plotting Entry Point) ---
                "data": {
                    "NX_class": "NXdata",
                    "signal": "data",
                    "data": {"link": "/entry/data/data"}
                }
            }
        },
        "scans": {
            "NX_class": "NXroot",
            "default": "entry",
            "entry": {
                "name": "entry",
                "NX_class": "NXentry",
                "default": "data",
                "instrument": {
                    "name": "instrument",
                    "NX_class": "NXinstrument",
                    "detector": {
                        "name": "detector",
                        "NX_class": "NXdetector",
                        "field": "data",
                        "distance": {"value": None, "units": "mm"},
                        "beam_center_x": {"value": None, "units": "pixel"},
                        "beam_center_y": {"value": None, "units": "pixel"},
                        "pixel_size": {"value": None, "units": "m"},
                        "transformations": {
                            "NX_class": "NXtransformations",
                            "axis_2": {"value": None, "type": "rotation", "vector": [0, 1, 0]}
                        }
                    },
                    "source": {
                        "name": "source",
                        "NX_class": "NXsource",
                        "energy": {"value": None, "units": "keV"}
                    },
                },
                "sample": {
                    "name": "sample",
                    "NX_class": "NXsample",
                    "field": "rotation_angle",
                    "ub_matrix": {"value": None, "units": "1/angstrom"},
                    "orientation_matrix": {"value": None},
                    "surface_normal": {"vector": [0, 0, 1]},
                    "inplane_reference": {"vector": [1, 0, 0]}
                },
                "data": {
                    "name": "data",
                    "NX_class": "NXdata",
                    "signal": "data",
                    "axes": "rotation_angle"
                }
            }
        },
        "format": {
            "name": "nexus",
            "links": {
                "Nexus": "",
                "Scan Standard": "",
                "DashPVA": ""
            }
        }
    }
}

__VERSION__ = 2.1

# User variables
BEAMLINE_NAME: Optional[str] = None

# Core
PROJECT_ROOT = Path(__file__).resolve().parent
DETECTOR_PREFIX: Optional[str] = None
OUTPUT_FILE_LOCATION: Optional[str] = None
CONSUMER_MODE: Optional[str] = None

# Cache + convenience
CACHING_MODE: Optional[str] = None
CACHE_OPTIONS: Dict[str, Any] = {}
ALIGNMENT_MAX_CACHE_SIZE: Optional[int] = None
SCAN_FLAG_PV: Optional[str] = None
SCAN_START_SCAN: Optional[bool] = None
SCAN_STOP_SCAN: Optional[bool] = None
SCAN_THRESHOLD: Optional[float] = None
SCAN_MAX_CACHE_SIZE: Optional[int] = None
BIN_COUNT: Optional[int] = None
BIN_SIZE: Optional[int] = None

# Sections
METADATA_CA: Dict[str, Any] = {}
METADATA_PVA: Dict[str, Any] = {}
ROI: Dict[str, Any] = {}
STATS: Dict[str, Any] = {}
HKL: Dict[str, Any] = {}
ANALYSIS: Dict[str, Any] = {}

# AppSettings
LOG_PATH: Optional[str] = str(PROJECT_ROOT / "logs")
OUTPUT_PATH: Optional[str] = str(PROJECT_ROOT / "outputs")
CONFIG_PATH: Optional[str] = str(PROJECT_ROOT / "pv_configs")
CONSUMERS_PATH: Optional[str] = None

# Diagnostics
CONFIG: Dict[str, Any] = {}
SOURCE_TYPE: Optional[str] = None
LOCATOR: Optional[Union[int, str]] = None

# Active TOML config path — set by the Settings dialog or resolved from the locator.
# Resolved TOML path for components that need a direct file path.
TOML_FILE: Optional[str] = None

# Internal state
_locator_internal: Optional[Union[int, str]] = None


def set_locator(locator: Union[int, str]) -> None:
    """Set the configuration locator (TOML path, "profile:<name>", or int profile_id)."""
    global _locator_internal
    _locator_internal = locator


def ensure_path() -> Optional[str]:
    """Return a TOML path: original path for TOML sources, temp file for DB sources."""
    eff = _get_effective_locator()
    if ConfigSource is None:
        return None
    return ConfigSource(eff).ensure_path()


def reload() -> None:
    """Re-resolve current LOCATOR and repopulate all exported constants from the configuration source."""
    global CONFIG, SOURCE_TYPE, LOCATOR, TOML_FILE
    global DETECTOR_PREFIX, OUTPUT_FILE_LOCATION, CONSUMER_MODE
    global CACHING_MODE, CACHE_OPTIONS, ALIGNMENT_MAX_CACHE_SIZE
    global SCAN_FLAG_PV, SCAN_START_SCAN, SCAN_STOP_SCAN, SCAN_THRESHOLD, SCAN_MAX_CACHE_SIZE
    global BIN_COUNT, BIN_SIZE
    global METADATA_CA, METADATA_PVA, ROI, STATS, HKL, ANALYSIS
    global LOG_PATH, OUTPUT_PATH, CONFIG_PATH, CONSUMERS_PATH

    eff = _get_effective_locator()
    LOCATOR = eff

    src = ConfigSource(eff) if ConfigSource else None
    cfg = src.load() if src else {}
    CONFIG = cfg
    SOURCE_TYPE = src.source_type if (src and eff is not None) else None

    try:
        TOML_FILE = ensure_path()
    except Exception:
        TOML_FILE = None

    # Core
    DETECTOR_PREFIX = cfg.get('DETECTOR_PREFIX')
    OUTPUT_FILE_LOCATION = cfg.get('OUTPUT_FILE_LOCATION')
    CONSUMER_MODE = cfg.get('CONSUMER_MODE')

    # Cache and convenience
    CACHE_OPTIONS = cfg.get('CACHE_OPTIONS', {}) or {}
    CACHING_MODE = CACHE_OPTIONS.get('CACHING_MODE')

    # ALIGNMENT
    ALIGNMENT_MAX_CACHE_SIZE = None
    try:
        ALIGNMENT_MAX_CACHE_SIZE = int(CACHE_OPTIONS.get('ALIGNMENT', {}).get('MAX_CACHE_SIZE'))
    except Exception:
        pass

    # SCAN
    scan = CACHE_OPTIONS.get('SCAN', {}) or {}
    SCAN_FLAG_PV = (cfg.get('METADATA', {}).get('CA', {}) or {}).get('FLAG_PV')
    try:
        SCAN_START_SCAN = bool(scan.get('START_SCAN')) if scan.get('START_SCAN') is not None else None
    except Exception:
        SCAN_START_SCAN = None
    try:
        SCAN_STOP_SCAN = bool(scan.get('STOP_SCAN')) if scan.get('STOP_SCAN') is not None else None
    except Exception:
        SCAN_STOP_SCAN = None
    try:
        SCAN_THRESHOLD = float(scan.get('THRESHOLD')) if scan.get('THRESHOLD') is not None else None
    except Exception:
        SCAN_THRESHOLD = None
    try:
        SCAN_MAX_CACHE_SIZE = int(scan.get('MAX_CACHE_SIZE')) if scan.get('MAX_CACHE_SIZE') is not None else None
    except Exception:
        SCAN_MAX_CACHE_SIZE = None

    # BIN
    bin_opts = CACHE_OPTIONS.get('BIN', {}) or {}
    try:
        BIN_COUNT = int(bin_opts.get('COUNT')) if bin_opts.get('COUNT') is not None else None
    except Exception:
        BIN_COUNT = None
    try:
        BIN_SIZE = int(bin_opts.get('SIZE')) if bin_opts.get('SIZE') is not None else None
    except Exception:
        BIN_SIZE = None

    # Sections
    metadata = cfg.get('METADATA', {}) or {}
    METADATA_CA = metadata.get('CA', {}) or {}
    METADATA_PVA = metadata.get('PVA', {}) or {}

    ROI = cfg.get('ROI', {}) or {}
    STATS = cfg.get('STATS', {}) or {}
    HKL = cfg.get('HKL', {}) or {}
    ANALYSIS = cfg.get('ANALYSIS', {}) or {}

    # AppSettings: paths (expand ~ if provided). Defaults to ./logs and ./outputs when absent.
    try:
        lp = cfg.get('LOG_PATH')
        LOG_PATH = str(Path(lp).expanduser()) if isinstance(lp, str) and lp.strip() else './logs'
    except Exception:
        LOG_PATH = './logs'
    try:
        op = cfg.get('OUTPUT_PATH')
        OUTPUT_PATH = str(Path(op).expanduser()) if isinstance(op, str) and op.strip() else './outputs'
    except Exception:
        OUTPUT_PATH = './outputs'
    CONFIG_PATH = cfg.get('CONFIG_PATH')
    CONSUMERS_PATH = cfg.get('CONSUMERS_PATH')


def _get_effective_locator() -> Union[int, str, None]:
    """Determine the effective locator: set_locator → env var → None.

    When None is returned, ConfigSource handles the selected-DB-profile
    fallback internally, so settings.py doesn't need to know about it.
    """
    # 1) Programmatic locator via set_locator
    if _locator_internal is not None:
        return _locator_internal

    # 2) Optional override via environment variable
    env_locator = os.getenv('DASPVA_CONFIG_LOCATOR')
    if env_locator and env_locator.strip():
        loc = env_locator.strip()
        if loc.isdigit():
            return int(loc)
        return loc

    return None


# Initialize on import
reload()


class Settings:
    """
    Object-oriented settings container.

    Can be constructed from:
      - TOML path (str)
      - DB profile id (int)
      - "profile:<name>" (str)
      - A custom source object with .load() and .save()

    Examples:
      s1 = Settings.from_toml('pv_configs/sample_config.toml')
      s2 = Settings.from_profile_id(42)
      s3 = Settings.from_profile_name('my_profile')
      s4 = Settings.from_source(ConfigSource('/path/to.toml'))
    """

    def __init__(
        self,
        locator: Optional[Union[int, str]] = None,
    ) -> None:
        self.source_type: Optional[str] = None
        self.locator: Optional[Union[int, str]] = None
        self._source: Optional[Any] = None  # only set for custom source objects
        self.CONFIG: Dict[str, Any] = {}
        self.PROJECT_ROOT: Path = Path(__file__).resolve().parent

        # Public attributes mirroring module-level constants
        self.BEAMLINE_NAME: Optional[str] = None
        self.DETECTOR_PREFIX: Optional[str] = None
        self.OUTPUT_FILE_LOCATION: Optional[str] = None
        self.CONSUMER_MODE: Optional[str] = None

        self.CACHING_MODE: Optional[str] = None
        self.CACHE_OPTIONS: Dict[str, Any] = {}
        self.ALIGNMENT_MAX_CACHE_SIZE: Optional[int] = None
        self.SCAN_FLAG_PV: Optional[str] = None
        self.SCAN_START_SCAN: Optional[bool] = None
        self.SCAN_STOP_SCAN: Optional[bool] = None
        self.SCAN_THRESHOLD: Optional[float] = None
        self.SCAN_MAX_CACHE_SIZE: Optional[int] = None
        self.BIN_COUNT: Optional[int] = None
        self.BIN_SIZE: Optional[int] = None

        self.METADATA_CA: Dict[str, Any] = {}
        self.METADATA_PVA: Dict[str, Any] = {}
        self.ROI: Dict[str, Any] = {}
        self.STATS: Dict[str, Any] = {}
        self.HKL: Dict[str, Any] = {}
        self.ANALYSIS: Dict[str, Any] = {}

        self.LOG_PATH: Optional[str] = None
        self.OUTPUT_PATH: Optional[str] = None
        self.CONFIG_PATH: Optional[str] = None
        self.CONSUMERS_PATH: Optional[str] = None

        if locator is None:
            self.locator = _get_effective_locator()
        else:
            self.set_locator(locator)
        self.reload()

    # Convenience constructors
    @classmethod
    def from_locator(cls, locator: Union[int, str]) -> "Settings":
        return cls(locator=locator)

    @classmethod
    def from_toml(cls, path: str) -> "Settings":
        return cls(locator=path)

    @classmethod
    def from_profile_id(cls, profile_id: int) -> "Settings":
        return cls(locator=profile_id)

    @classmethod
    def from_profile_name(cls, name: str) -> "Settings":
        return cls(locator=f"profile:{name}")

    @classmethod
    def from_source(cls, source: Any) -> "Settings":
        return cls(locator=source)

    def set_locator(self, locator: Any) -> None:
        """
        Accepts int (profile id), str (TOML path or "profile:<name>"),
        or a custom source object with .load() and .save().
        """
        if hasattr(locator, "load") and hasattr(locator, "save"):
            self._source = locator
            self.locator = None
        else:
            self.locator = locator
            self._source = None

    def _resolve_source(self) -> Any:
        """Return the active source object (custom or ConfigSource)."""
        if self._source is not None:
            return self._source
        if ConfigSource is not None:
            return ConfigSource(self.locator)
        return None

    def ensure_path(self) -> Optional[str]:
        """
        Return a TOML file path for the current source:
          - TOML: original path
          - DB: temp TOML file
        """
        src = self._resolve_source()
        if src is None:
            return None
        if hasattr(src, 'ensure_path'):
            return src.ensure_path()
        return None

    def reload(self) -> None:
        """Load and parse configuration into object attributes."""
        src = self._resolve_source()
        self.source_type = getattr(src, 'source_type', None) if src else None
        cfg: Dict[str, Any] = {}
        try:
            cfg = src.load() if src else {}
        except Exception:
            cfg = {}
        self.CONFIG = cfg

        # Core
        self.PROJECT_ROOT = PROJECT_ROOT
        self.DETECTOR_PREFIX = cfg.get('DETECTOR_PREFIX')
        self.OUTPUT_FILE_LOCATION = cfg.get('OUTPUT_FILE_LOCATION')
        self.CONSUMER_MODE = cfg.get('CONSUMER_MODE')

        # Cache and convenience
        self.CACHE_OPTIONS = cfg.get('CACHE_OPTIONS', {}) or {}
        self.CACHING_MODE = self.CACHE_OPTIONS.get('CACHING_MODE')

        # ALIGNMENT
        self.ALIGNMENT_MAX_CACHE_SIZE = None
        try:
            self.ALIGNMENT_MAX_CACHE_SIZE = int(self.CACHE_OPTIONS.get('ALIGNMENT', {}).get('MAX_CACHE_SIZE'))
        except Exception:
            pass

        # SCAN
        scan = self.CACHE_OPTIONS.get('SCAN', {}) or {}
        self.SCAN_FLAG_PV = (cfg.get('METADATA', {}).get('CA', {}) or {}).get('FLAG_PV')
        try:
            self.SCAN_START_SCAN = bool(scan.get('START_SCAN')) if scan.get('START_SCAN') is not None else None
        except Exception:
            self.SCAN_START_SCAN = None
        try:
            self.SCAN_STOP_SCAN = bool(scan.get('STOP_SCAN')) if scan.get('STOP_SCAN') is not None else None
        except Exception:
            self.SCAN_STOP_SCAN = None
        try:
            self.SCAN_THRESHOLD = float(scan.get('THRESHOLD')) if scan.get('THRESHOLD') is not None else None
        except Exception:
            self.SCAN_THRESHOLD = None
        try:
            self.SCAN_MAX_CACHE_SIZE = int(scan.get('MAX_CACHE_SIZE')) if scan.get('MAX_CACHE_SIZE') is not None else None
        except Exception:
            self.SCAN_MAX_CACHE_SIZE = None

        # BIN
        bin_opts = self.CACHE_OPTIONS.get('BIN', {}) or {}
        try:
            self.BIN_COUNT = int(bin_opts.get('COUNT')) if bin_opts.get('COUNT') is not None else None
        except Exception:
            self.BIN_COUNT = None
        try:
            self.BIN_SIZE = int(bin_opts.get('SIZE')) if bin_opts.get('SIZE') is not None else None
        except Exception:
            self.BIN_SIZE = None

        # Sections
        metadata = cfg.get('METADATA', {}) or {}
        self.METADATA_CA = metadata.get('CA', {}) or {}
        self.METADATA_PVA = metadata.get('PVA', {}) or {}

        self.ROI = cfg.get('ROI', {}) or {}
        self.STATS = cfg.get('STATS', {}) or {}
        self.HKL = cfg.get('HKL', {}) or {}
        self.ANALYSIS = cfg.get('ANALYSIS', {}) or {}

        # AppSettings
        try:
            lp = cfg.get('LOG_PATH')
            self.LOG_PATH = str(Path(lp).expanduser()) if isinstance(lp, str) and lp.strip() else './logs'
        except Exception:
            self.LOG_PATH = './logs'
        try:
            op = cfg.get('OUTPUT_PATH')
            self.OUTPUT_PATH = str(Path(op).expanduser()) if isinstance(op, str) and op.strip() else './outputs'
        except Exception:
            self.OUTPUT_PATH = './outputs'
        self.CONFIG_PATH = cfg.get('CONFIG_PATH')
        self.CONSUMERS_PATH = cfg.get('CONSUMERS_PATH')

# Export a default instance using the same precedence as the module-level globals
SETTINGS = Settings()
