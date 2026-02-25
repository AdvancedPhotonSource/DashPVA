"""
Centralized settings module for DashPVA.

Exports constants resolved from the currently selected configuration source
(TOML file or DB profile), with helper functions to control the locator and
refresh values. By default it uses the database-selected profile in dashpva.db.
If a TOML path is provided (via set_locator or the DASPVA_CONFIG_LOCATOR env var),
it will use that TOML instead.

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
      settings.SOURCE_TYPE -> "toml" or "db"
      settings.LOCATOR     -> the current locator (str path, "profile:<name>", or int id)
      settings.CONFIG      -> full configuration dictionary
      settings.ensure_path() -> a TOML path (original path or temp file when using DB)
"""

from typing import Any, Dict, Optional, Union
import os
import tempfile
import toml
from pathlib import Path

from utils.config.repository import ConfigRepository
from utils.config.interfaces import ConfigSource
try:
    from database import DatabaseInterface
except Exception:
    DatabaseInterface = None  # type: ignore[assignment]


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

# Internal state
_repo = ConfigRepository(DatabaseInterface() if DatabaseInterface else None)
_default_toml = 'pv_configs/sample_config.toml'
_locator_internal: Optional[Union[int, str]] = None


def set_locator(locator: Union[int, str]) -> None:
    """Set the configuration locator (TOML path, "profile:<name>", or int profile_id)."""
    global _locator_internal
    _locator_internal = locator


def ensure_path() -> Optional[str]:
    """Return a TOML path (real path for TOML; temp file for DB) via ConfigRepository.ensure_path."""
    eff = _get_effective_locator()
    return _repo.ensure_path(eff)


def reload() -> None:
    """Re-resolve current LOCATOR and repopulate all exported constants from the configuration source."""
    global CONFIG, SOURCE_TYPE, LOCATOR
    global DETECTOR_PREFIX, OUTPUT_FILE_LOCATION, CONSUMER_MODE
    global CACHING_MODE, CACHE_OPTIONS, ALIGNMENT_MAX_CACHE_SIZE
    global SCAN_FLAG_PV, SCAN_START_SCAN, SCAN_STOP_SCAN, SCAN_THRESHOLD, SCAN_MAX_CACHE_SIZE
    global BIN_COUNT, BIN_SIZE
    global METADATA_CA, METADATA_PVA, ROI, STATS, HKL, ANALYSIS
    global LOG_PATH, OUTPUT_PATH, CONFIG_PATH, CONSUMERS_PATH

    eff = _get_effective_locator()
    LOCATOR = eff
    cfg = _repo.load(eff) if eff is not None else {}
    CONFIG = cfg

    # Source type
    src = _repo.resolve(eff)
    SOURCE_TYPE = getattr(src, 'source_type', None) if src else None

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
    # Flag PV comes from METADATA.CA.FLAG_PV
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
    # Pass through any optional paths if present
    CONFIG_PATH = cfg.get('CONFIG_PATH')
    CONSUMERS_PATH = cfg.get('CONSUMERS_PATH')


def _get_effective_locator() -> Union[int, str, None]:
    """Determine the effective locator based on precedence: set_locator -> env var -> fallback TOML."""
    # 1) Programmatic locator via set_locator (primary when Browse runs)
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


class DictConfigSource(ConfigSource):
    """
    In-memory configuration source backed by a Python dict.
    Useful for constructing Settings from an already-built configuration object.
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = dict(data)
        self.source_type = "dict"

    def load(self) -> Dict[str, Any]:
        return dict(self._data)

    def save(self, update: Dict[str, Any]) -> bool:
        try:
            self._data.clear()
            self._data.update(update or {})
            return True
        except Exception:
            return False


class Settings:
    """
    Object-oriented settings container.

    Can be constructed from:
      - TOML path (str)
      - DB profile id (int)
      - "profile:<name>" (str)
      - ConfigSource instance (TomlConfigSource, DbProfileConfigSource, etc.)
      - Plain dict via Settings.from_dict()

    Examples:
      s1 = Settings.from_toml('pv_configs/sample_config.toml')
      s2 = Settings.from_profile_id(42)
      s3 = Settings.from_profile_name('my_profile')
      s4 = Settings.from_source(TomlConfigSource('path/to.toml'))
      s5 = Settings.from_dict({'DETECTOR_PREFIX': 'sim'})
    """

    def __init__(
        self,
        locator: Optional[Union[int, str, ConfigSource, Dict[str, Any]]] = None,
        repo: Optional[ConfigRepository] = None,
    ) -> None:
        self.repo = repo or ConfigRepository(DatabaseInterface() if DatabaseInterface else None)
        self.source_type: Optional[str] = None
        self.locator: Optional[Union[int, str]] = None
        self._source: Optional[ConfigSource] = None
        self.CONFIG: Dict[str, Any] = {}
        # Absolute project root directory for path resolution
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

        # Initialize
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
    def from_source(cls, source: ConfigSource) -> "Settings":
        return cls(locator=source)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        return cls(locator=DictConfigSource(data))

    def set_locator(self, locator: Union[int, str, ConfigSource, Dict[str, Any]]) -> None:
        """
        Accepts int (profile id), str (TOML path or "profile:<name>"), a ConfigSource-like object, or dict.
        """
        if isinstance(locator, dict):
            self._source = DictConfigSource(locator)
            self.locator = None
        elif hasattr(locator, "load") and hasattr(locator, "save"):
            # Treat as a ConfigSource-like object
            self._source = locator  # type: ignore[assignment]
            self.locator = None
        else:
            self.locator = locator  # type: ignore[assignment]
            self._source = None

    def _resolve_source(self) -> Optional[ConfigSource]:
        if self._source is not None:
            return self._source
        return self.repo.resolve(self.locator)

    def ensure_path(self) -> Optional[str]:
        """
        Return a TOML path for the current source.
          - TOML: original path
          - DB: temp TOML path
          - dict/other: writes temp TOML and returns its path
        """
        src = self._resolve_source()
        if src is None:
            return None

        # TomlConfigSource exposes .path
        if getattr(src, "source_type", None) == "toml" and hasattr(src, "path"):
            return getattr(src, "path")

        # For DB or dict, create a temp TOML
        try:
            data = src.load()
            fd, tmp = tempfile.mkstemp(suffix=".toml", prefix="dashpva_")
            os.close(fd)
            with open(tmp, "w") as f:
                toml.dump(data, f)
            return tmp
        except Exception:
            return None

    def reload(self) -> None:
        """
        Load and parse configuration into object attributes.
        """
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

        # AppSettings: paths (expand ~ if provided). Defaults to ./logs and ./outputs when absent.
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

    @property
    def db(self) -> DatabaseInterface:
        """Expose the underlying DatabaseInterface so callers can use database settings APIs."""
        return self.repo.db


# Export a default object instance using the same precedence as the module-level globals
SETTINGS = Settings()
