# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

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
      settings.RAW_CONFIG  -> exact persisted configuration dictionary
      settings.CONFIG      -> effective runtime configuration dictionary
      settings.ensure_path() -> a TOML path containing the effective configuration
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

_logger = logging.getLogger(__name__)

try:
    from dashpva.utils.config.resolver import resolve_profile_config
    from dashpva.utils.config.source import ConfigSource
except Exception:
    ConfigSource = None  # type: ignore[assignment]

    def resolve_profile_config(raw):
        return dict(raw or {})


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
                        "NX_class": "NXtransformations"
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

__VERSION__ = __import__("dashpva").__version__

# User variables
BEAMLINE_NAME: Optional[str] = None

# Core
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHPVA_ROOT = Path(__file__).resolve().parent
DETECTOR_PREFIX: Optional[str] = None
IOC_PREFIX: Optional[str] = None
INPUT_CHANNEL: Optional[str] = None
INPUT_CHANNEL_HKL3D: Optional[str] = None
OUTPUT_FILE_LOCATION: Optional[str] = None
CONSUMER_MODE: Optional[str] = None

# Hardcoded PV name suffixes — combined with IOC_PREFIX at reload time to
# produce SCAN_FLAG_PV / FILE_PATH_PV / FILE_NAME_PV. These are the sole
# source of truth; METADATA.CA is reserved for user-custom CA PVs only.
_FLAG_PV_SUFFIX = "ScanOn:Value"
_FILE_PATH_SUFFIX = "FilePath:Value"
_FILE_NAME_SUFFIX = "FileName:Value"

# Whether closing a window that has unsaved edits prompts Save / Discard /
# Cancel instead of closing silently (static — not config-driven). This is the
# global default; an individual window overrides it by setting
# BaseWindow.confirm_unsaved_changes on its class or instance.
CONFIRM_UNSAVED_CHANGES_ON_CLOSE: bool = True

# Simulation servers the workflow Sim Server tab can launch, label -> path
# relative to PROJECT_ROOT (static -- not config-driven). The label order is the
# dropdown order; the first entry is the default.
#: ``options`` lists only the flags that server's parser accepts -- passing any
#: other makes argparse exit before the server starts. The RSM data server takes
#: none at all.
SIM_SERVER_TYPES: dict[str, dict] = {
    "Area detector": {
        "path": "src/dashpva/consumers/caIOC_servers/ad_sim_server_modified.py",
        "options": ("cn", "nx", "ny", "fps", "dt", "nf", "rt", "rp", "mpv"),
    },
    "Probe beam": {
        "path": "src/dashpva/consumers/caIOC_servers/probe_sim_server.py",
        "options": ("cn", "nx", "ny", "fps", "dt", "rt", "rp", "shape"),
    },
    "RSM data": {
        "path": "src/dashpva/consumers/caIOC_servers/sim_rsm_data.py",
        "options": (),
    },
}

# Beam shapes the probe simulation server accepts (its -shape choices). Only
# meaningful when SIM_SERVER_TYPES["Probe beam"] is the selected simulation.
SIM_PROBE_SHAPES: tuple[str, ...] = ("gaussian", "laplacian", "lorentzian", "zone-plate")

# Bounded pvapy monitor queue depth for the PVA reader (static — not
# config-driven). The network thread enqueues frames here and a consumer thread
# drains them; when the consumer falls behind the queue fills and pvapy drops
# the newest frame (counted as nRejected) rather than overrunning the monitor
# thread and crashing the viewer. Memory is bounded at this depth × frame size.
PVA_MONITOR_QUEUE_SIZE: int = 128

# Server-side pvAccess monitor queue depth for this subscription. This is the
# buffer where high-rate frames are dropped (uniqueId gaps) when the client
# can't drain fast enough; it is separate from the client-side
# PVA_MONITOR_QUEUE_SIZE above. Deeper = absorbs larger bursts, but the server
# holds this many full frames in memory (depth × frame size).
PVA_MONITOR_SERVER_QUEUE_SIZE: int = 64

# pvapy monitor request descriptor for the PVA reader (static — not
# config-driven). 'field()' selects the full NTNDArray structure (value, codec,
# dimension, uniqueId, uncompressedSize, attribute, ...) which the reader needs
# to decode frames. A value-only selector strips those fields on servers that
# honor the request (e.g. the pvapy sim server), leaving the image undecodable.
# 'record[queueSize=N]' enlarges the server monitor queue (see above).
PVA_MONITOR_REQUEST: str = f'field() record[queueSize={PVA_MONITOR_SERVER_QUEUE_SIZE}]'

# RSM Volume Builder memory policy. Batching bounds the coordinate arrays, but the
# dense Gridder3D result scales with nx * ny * nz, so reject unsafe peaks up front.
RSM_GRID_BATCH_MEMORY_BYTES: int = 256 * 1024 * 1024
RSM_GRID_MAX_MEMORY_FRACTION: float = 0.70
RSM_GRID_WORKING_BYTES_PER_PIXEL: int = 96
RSM_GRID_ENERGY_RELATIVE_TOLERANCE: float = 1e-4
RSM_GRID_UB_ABSOLUTE_TOLERANCE: float = 1e-4
RSM_GRID_PREVIEW_BUDGET_BYTES: int = 4 * 1024 * 1024
RSM_GRID_PREVIEW_INTERVAL_SECONDS: float = 1.0
# Motors publish only on value changes; infinity still requires the associator
# source timestamp, while a finite age limit remains an explicit override.
RSM_GRID_METADATA_MAX_AGE_SECONDS: float = float("inf")
RSM_GRID_CONTROL_TIMEOUT_SECONDS: float = 5.0
RSM_GRID_SAVE_TIMEOUT_SECONDS: float = 300.0
RSM_GRID_CONTROL_POLL_INTERVAL_SECONDS: float = 0.05
RSM_GRID_DEFAULT_RESOLUTION: int = 200
# Grid lines the box preview will draw per axis before it stops subdividing --
# past this the lines merge into a solid block and stop conveying anything.
RSM_GRID_PREVIEW_MAX_DIVISIONS: int = 24
RSM_STATIC_METADATA_RELATIVE_TOLERANCE: float = 1e-6
RSM_STATIC_METADATA_ABSOLUTE_TOLERANCE: float = 1e-9
RSM_IOC_POLL_INTERVAL_SECONDS: float = 0.01
RSM_IOC_SNAPSHOT_EVERY: int = 5
METADATA_ASSOCIATOR_STALENESS_CHECK_MS: int = 2_000

ANALYSIS_PROCESSOR_CLASSES: dict[str, str] = {
    "hpc_rsm_consumer.py": "HpcRsmProcessor",
    "hpc_rsm_grid_consumer.py": "HpcRsmGridProcessor",
    "hpc_spontaneous_analysis_consumer.py": "HpcAnalysisProcessor",
    "hpc_vectorized_analysis_consumer.py": "HpcAnalysisProcessor",
}
RSM_GRID_PROCESSOR_CLASS: str = "HpcRsmGridProcessor"

# Combined byte budget for mask-editor undo and redo data.
MASK_UNDO_MAX_BYTES: int = 32 * 1024 * 1024

# Mask editor only pauses the parent viewer's live-plot timer while drawing on
# detectors at or above this many pixels (static — not config-driven). Below
# this, timer_plot's per-frame cost doesn't come close to starving the editor's
# own repaint, so pausing the live view would just be a needless interruption.
MASK_EDITOR_PAUSE_MIN_PIXELS: int = 1_000_000

# Cache + convenience
CACHING_MODE: Optional[str] = None
CACHE_OPTIONS: Dict[str, Any] = {}
ALIGNMENT_MAX_CACHE_SIZE: Optional[int] = None
SCAN_FLAG_PV: Optional[str] = None
FILE_PATH_PV: Optional[str] = None
FILE_NAME_PV: Optional[str] = None
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
# Ordered, role-split views of HKL's circle groups, derived on reload. A thin
# convenience over the same discovery rsm_geometry uses -- call sites that only
# need "the sample circles, in order" should not have to know the group-naming
# or numeric-sort rules. Each entry is (group_name, field_map).
HKL_SAMPLE_CIRCLES: "list[tuple[str, Dict[str, Any]]]" = []
HKL_DETECTOR_CIRCLES: "list[tuple[str, Dict[str, Any]]]" = []
ANALYSIS: Dict[str, Any] = {}

# AppSettings
LOG_PATH: Optional[str] = str(PROJECT_ROOT / "logs")
OUTPUT_PATH: Optional[str] = str(PROJECT_ROOT / "outputs")
CONFIG_PATH: Optional[str] = str(PROJECT_ROOT / "pv_configs")
CONSUMERS_PATH: Optional[str] = None

# Diagnostics
RAW_CONFIG: Dict[str, Any] = {}
CONFIG: Dict[str, Any] = {}
SOURCE_TYPE: Optional[str] = None
LOCATOR: Optional[Union[int, str]] = None
# Set when resolve_profile_config() rejects the active profile (e.g. a
# corrupted IOC_RSM_PARAMETER section); CONFIG then falls back to the raw,
# unresolved config so the package stays importable. Cleared on a clean load.
CONFIG_ERROR: Optional[str] = None

# Active TOML config path — set by the Settings dialog or resolved from the locator.
# Resolved TOML path for components that need a direct file path.
TOML_FILE: Optional[str] = None

# Default file browser directory (test_data has sample PONI, CIF, mask files)
_DEFAULT_BROWSE_DIR: str = str(PROJECT_ROOT / 'tests' / 'test_data')
LAST_PONI_DIR: str = _DEFAULT_BROWSE_DIR
LAST_CIF_DIR: str = _DEFAULT_BROWSE_DIR
LAST_TOML_DIR: str = str(PROJECT_ROOT / 'pv_configs')

# Internal state
_locator_internal: Optional[Union[int, str]] = None
_STATE_FILE: Path = PROJECT_ROOT / '.dashpva_locator'


def set_locator(locator: Optional[Union[int, str]]) -> None:
    """Set the configuration locator (TOML path, "profile:<name>", or int profile_id).

    Persists to a state file so sibling subprocesses can find the active config.
    Passing None clears the locator instead of writing the literal string
    'None' -- writing that string used to make the next read treat it as a
    real, unresolvable locator instead of "unset".
    """
    global _locator_internal
    _locator_internal = locator
    if locator is None:
        os.environ.pop('DASPVA_CONFIG_LOCATOR', None)
        try:
            _STATE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return
    os.environ['DASPVA_CONFIG_LOCATOR'] = str(locator)
    try:
        _STATE_FILE.write_text(str(locator))
    except Exception:
        pass


def ensure_path() -> Optional[str]:
    """Return a TOML path containing the effective runtime configuration."""
    eff = _get_effective_locator()
    if ConfigSource is None:
        return None
    src = ConfigSource(eff)
    raw = src.load()
    effective = resolve_profile_config(raw)
    return src.ensure_path(effective if effective != raw else None)


def _circles_by_role(hkl: Dict[str, Any], role: str) -> "list":
    """Ordered (group_name, fields) pairs for one circle role.

    Uses the same resolution rsm_geometry does -- numbered groups sorted by
    integer suffix, legacy named groups as a per-role fallback -- so this view
    can never disagree with the geometry the Q conversion actually builds.
    """
    try:
        from dashpva.utils.hkl_axes import resolved_axis_groups
    except Exception:
        return []
    try:
        return [
            (name, hkl.get(name, {}) or {})
            for name in resolved_axis_groups((hkl or {}).keys(), role)
        ]
    except Exception:
        return []


def reload() -> None:
    """Re-resolve current LOCATOR and repopulate all exported constants from the configuration source."""
    global RAW_CONFIG, CONFIG, SOURCE_TYPE, LOCATOR, TOML_FILE, CONFIG_ERROR
    global DETECTOR_PREFIX, IOC_PREFIX, INPUT_CHANNEL, INPUT_CHANNEL_HKL3D, OUTPUT_FILE_LOCATION, CONSUMER_MODE
    global CACHING_MODE, CACHE_OPTIONS, ALIGNMENT_MAX_CACHE_SIZE
    global SCAN_FLAG_PV, FILE_PATH_PV, FILE_NAME_PV
    global SCAN_START_SCAN, SCAN_STOP_SCAN, SCAN_THRESHOLD, SCAN_MAX_CACHE_SIZE
    global BIN_COUNT, BIN_SIZE
    global METADATA_CA, METADATA_PVA, ROI, STATS, HKL, ANALYSIS
    global HKL_SAMPLE_CIRCLES, HKL_DETECTOR_CIRCLES
    global LOG_PATH, OUTPUT_PATH, CONFIG_PATH, CONSUMERS_PATH

    eff = _get_effective_locator()
    LOCATOR = eff

    src = ConfigSource(eff) if ConfigSource else None
    raw_cfg = src.load() if src else {}
    try:
        cfg = resolve_profile_config(raw_cfg)
        CONFIG_ERROR = None
    except Exception as exc:
        _logger.error("resolve_profile_config failed, using raw config as-is: %s", exc)
        CONFIG_ERROR = str(exc)
        cfg = dict(raw_cfg or {})
    RAW_CONFIG = raw_cfg
    CONFIG = cfg
    SOURCE_TYPE = src.source_type if (src and eff is not None) else None

    try:
        TOML_FILE = src.ensure_path() if src else None
    except Exception:
        TOML_FILE = None

    # Core
    # IOC_PREFIX is per-profile — read it from the active profile/TOML only.
    # Backward-compat: also accept legacy 'DETECTOR_PREFIX' from older TOMLs.
    IOC_PREFIX = cfg.get('IOC_PREFIX') or cfg.get('DETECTOR_PREFIX') or ''
    DETECTOR_PREFIX = cfg.get('DETECTOR_PREFIX')
    # Match the IOC convention (consumers/ioc_rsm_parameter.py enforces this on
    # apply): a non-empty prefix always ends with ':'. Saves users from a
    # silent caget/camonitor miss when they enter "6idb1" instead of "6idb1:".
    if IOC_PREFIX and not IOC_PREFIX.endswith(':'):
        IOC_PREFIX += ':'
    INPUT_CHANNEL = cfg.get('INPUT_CHANNEL')
    INPUT_CHANNEL_HKL3D = cfg.get('INPUT_CHANNEL_HKL3D')
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
    SCAN_FLAG_PV = f"{IOC_PREFIX}{_FLAG_PV_SUFFIX}" if IOC_PREFIX else _FLAG_PV_SUFFIX
    FILE_PATH_PV = f"{IOC_PREFIX}{_FILE_PATH_SUFFIX}" if IOC_PREFIX else _FILE_PATH_SUFFIX
    FILE_NAME_PV = f"{IOC_PREFIX}{_FILE_NAME_SUFFIX}" if IOC_PREFIX else _FILE_NAME_SUFFIX
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
    HKL_SAMPLE_CIRCLES = _circles_by_role(HKL, 'sample')
    HKL_DETECTOR_CIRCLES = _circles_by_role(HKL, 'detector')
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


def _parse_locator(value: Optional[str]) -> Union[int, str, None]:
    """Parse a raw locator string from the env var or state file.

    A blank string, or the literal 'None', is treated as unset rather than as
    a real locator -- set_locator(None) used to write that exact string to
    both, poisoning the next read into treating "unset" as a bogus locator.
    This also lets an already-poisoned state file recover on the next call.
    """
    if value is None:
        return None
    loc = value.strip()
    if not loc or loc == 'None':
        return None
    if loc.isdigit():
        return int(loc)
    return loc


def _get_effective_locator() -> Union[int, str, None]:
    """Determine the effective locator: set_locator → env var → state file → None.

    When None is returned, ConfigSource handles the selected-DB-profile
    fallback internally, so settings.py doesn't need to know about it.
    """
    # 1) Programmatic locator via set_locator
    if _locator_internal is not None:
        return _locator_internal

    # 2) Optional override via environment variable
    parsed = _parse_locator(os.getenv('DASPVA_CONFIG_LOCATOR'))
    if parsed is not None:
        return parsed

    # 3) State file written by set_locator in another process
    try:
        parsed = _parse_locator(_STATE_FILE.read_text())
        if parsed is not None:
            return parsed
    except Exception:
        pass

    return None


def save_detector_prefix(prefix: str) -> bool:
    """Persist *prefix* to the active config source, update the module global,
    and rewrite ROI/STATS PV names to use the new prefix."""
    global DETECTOR_PREFIX, ROI, STATS
    old_prefix = DETECTOR_PREFIX
    DETECTOR_PREFIX = prefix
    update: dict = {'DETECTOR_PREFIX': prefix}
    if old_prefix and old_prefix != prefix:
        ROI = _reprefix(ROI, old_prefix, prefix)
        STATS = _reprefix(STATS, old_prefix, prefix)
        update['ROI'] = ROI
        update['STATS'] = STATS
    eff = _get_effective_locator()
    if eff is None or ConfigSource is None:
        return False
    try:
        src = ConfigSource(eff)
        return src.save(update)
    except Exception:
        return False


def _reprefix(section: dict, old: str, new: str) -> dict:
    """Replace the detector prefix in all PV name values within a ROI/STATS section."""
    rebuilt = {}
    for group_key, group_dict in section.items():
        if isinstance(group_dict, dict):
            rebuilt[group_key] = {
                k: v.replace(old, new, 1) if isinstance(v, str) else v
                for k, v in group_dict.items()
            }
        else:
            rebuilt[group_key] = group_dict
    return rebuilt


def get_input_channel(fallback: str = "pvapy:image") -> str:
    """Return INPUT_CHANNEL if explicitly set, else derive from DETECTOR_PREFIX."""
    if INPUT_CHANNEL:
        return INPUT_CHANNEL
    if DETECTOR_PREFIX:
        return f"{DETECTOR_PREFIX}:Pva1:Image"
    return fallback


def save_input_channel(channel: str) -> bool:
    """Persist *channel* to the active config source and update the module global."""
    global INPUT_CHANNEL
    INPUT_CHANNEL = channel
    eff = _get_effective_locator()
    if eff is None or ConfigSource is None:
        return False
    try:
        src = ConfigSource(eff)
        return src.save({'INPUT_CHANNEL': channel})
    except Exception:
        return False


def get_input_channel_hkl3d(fallback: str = "pvapy:image") -> str:
    """Return HKL3D-specific INPUT_CHANNEL, independent of the Area Detector channel."""
    if INPUT_CHANNEL_HKL3D:
        return INPUT_CHANNEL_HKL3D
    return fallback


def save_input_channel_hkl3d(channel: str) -> bool:
    """Persist HKL3D input channel separately from the Area Detector channel."""
    global INPUT_CHANNEL_HKL3D
    INPUT_CHANNEL_HKL3D = channel
    eff = _get_effective_locator()
    if eff is None or ConfigSource is None:
        return False
    try:
        src = ConfigSource(eff)
        return src.save({'INPUT_CHANNEL_HKL3D': channel})
    except Exception:
        return False


def get_analysis_transport_channels(database=None) -> tuple[str, str]:
    """Return the one live analysis consumer's control and status channels.

    Workflow persists these names under ``APP_DATA/workflow/analysis/last``.
    Profile ``[ANALYSIS]`` values are accepted for legacy/TOML-only setups.
    The live grid is stateful and deliberately supports one consumer only.
    """
    profile_analysis = ANALYSIS if isinstance(ANALYSIS, dict) else {}
    config = {
        "control_channel": profile_analysis.get("CONTROL_CHANNEL", ""),
        "status_channel": profile_analysis.get("STATUS_CHANNEL", ""),
        "n_consumers": profile_analysis.get("N_CONSUMERS", 1),
        "consumer_id": profile_analysis.get("CONSUMER_ID", 1),
    }
    try:
        if database is None:
            from dashpva.database import DatabaseInterface

            database = DatabaseInterface()
        setting = database.get_setting_by_path(
            ["APP_DATA", "workflow", "analysis"]
        )
        saved = database.get_setting_value(setting.id, "last") if setting else None
        if isinstance(saved, dict):
            config.update(saved)
    except Exception as exc:
        _logger.debug("Could not read analysis workflow channels: %s", exc)

    try:
        n_consumers = int(config.get("n_consumers", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Analysis n_consumers must be an integer.") from exc
    if n_consumers != 1:
        raise RuntimeError(
            "Live RSM gridding requires exactly one stateful analysis consumer; "
            f"workflow is configured for {n_consumers}."
        )

    try:
        consumer_id = int(config.get("consumer_id", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Analysis consumer_id must be an integer.") from exc
    control = str(config.get("control_channel", "")).strip().replace(
        "*", str(consumer_id)
    )
    status = str(config.get("status_channel", "")).strip().replace(
        "*", str(consumer_id)
    )
    if not control or not status:
        raise RuntimeError(
            "Analysis control/status channels are not configured. Open Workflow, "
            "set both Analysis Consumer channels, and save the configuration."
        )
    return control, status


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
        self.RAW_CONFIG: Dict[str, Any] = {}
        self.CONFIG: Dict[str, Any] = {}
        self.CONFIG_ERROR: Optional[str] = None
        self.PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

        # Public attributes mirroring module-level constants
        self.BEAMLINE_NAME: Optional[str] = None
        self.DETECTOR_PREFIX: Optional[str] = None
        self.IOC_PREFIX: Optional[str] = None
        self.INPUT_CHANNEL: Optional[str] = None
        self.OUTPUT_FILE_LOCATION: Optional[str] = None
        self.CONSUMER_MODE: Optional[str] = None

        self.CACHING_MODE: Optional[str] = None
        self.CACHE_OPTIONS: Dict[str, Any] = {}
        self.ALIGNMENT_MAX_CACHE_SIZE: Optional[int] = None
        self.SCAN_FLAG_PV: Optional[str] = None
        self.FILE_PATH_PV: Optional[str] = None
        self.FILE_NAME_PV: Optional[str] = None
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
            raw = src.load()
            effective = resolve_profile_config(raw)
            if effective != raw:
                try:
                    return src.ensure_path(effective)
                except TypeError:
                    pass
            return src.ensure_path()
        return None

    def reload(self) -> None:
        """Load and parse configuration into object attributes."""
        src = self._resolve_source()
        self.source_type = getattr(src, 'source_type', None) if src else None
        cfg: Dict[str, Any] = {}
        try:
            raw_cfg = src.load() if src else {}
        except Exception:
            raw_cfg = {}
        try:
            cfg = resolve_profile_config(raw_cfg)
            self.CONFIG_ERROR = None
        except Exception as exc:
            _logger.error("resolve_profile_config failed, using raw config as-is: %s", exc)
            self.CONFIG_ERROR = str(exc)
            cfg = dict(raw_cfg or {})
        self.RAW_CONFIG = raw_cfg
        self.CONFIG = cfg

        # Core
        self.PROJECT_ROOT = PROJECT_ROOT
        # IOC_PREFIX is per-profile — active profile/TOML only.
        # Backward-compat: also accept legacy 'DETECTOR_PREFIX' from older TOMLs.
        self.IOC_PREFIX = cfg.get('IOC_PREFIX') or cfg.get('DETECTOR_PREFIX') or ''
        self.DETECTOR_PREFIX = cfg.get('DETECTOR_PREFIX')
        # See module-level reload(): match the IOC convention that the prefix
        # always ends with ':'.
        if self.IOC_PREFIX and not self.IOC_PREFIX.endswith(':'):
            self.IOC_PREFIX += ':'
        self.INPUT_CHANNEL = cfg.get('INPUT_CHANNEL')
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
        self.SCAN_FLAG_PV = f"{self.IOC_PREFIX}{_FLAG_PV_SUFFIX}" if self.IOC_PREFIX else _FLAG_PV_SUFFIX
        self.FILE_PATH_PV = f"{self.IOC_PREFIX}{_FILE_PATH_SUFFIX}" if self.IOC_PREFIX else _FILE_PATH_SUFFIX
        self.FILE_NAME_PV = f"{self.IOC_PREFIX}{_FILE_NAME_SUFFIX}" if self.IOC_PREFIX else _FILE_NAME_SUFFIX
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
