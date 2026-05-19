#!/usr/bin/env python3
"""
Seed the default per-profile config (the JSON blob in profile_configs.__data__)
using raw SQL.

Mirrors :mod:`seed_settings_defaults_sql` in style, but operates on the
*per-profile* JSON blob rather than the app-level ``settings`` table.

Source of truth
---------------
``DEFAULT_PROFILE_DATA`` below is the single canonical default for a
newly-added profile. Previously this lived in ``pv_configs/sample_config.toml``
and was loaded at runtime — but that meant the GUI had a hidden filesystem
dependency that broke when the TOML moved or was edited inconsistently.
Keeping the defaults as a Python literal removes that fragility and lets the
seed run from any context (installed package, test harness, headless script).

ROI / STATS are intentionally NOT included — area_det_viewer builds them at
runtime from ``{IOC_PREFIX}:ROI{N}:*`` / ``{IOC_PREFIX}:Stats{N}:*`` (per
``b7d57f3``), so seeding them would just be dead JSON.
"""
import copy
import json
import sqlite3
from typing import Any, Dict

import dashpva.settings as _settings

_DB_FILE = _settings.PROJECT_ROOT / "dashpva.db"
DB_PATH = str(_DB_FILE)


# ── Canonical default profile data ─────────────────────────────────────────── #
# Edit values here (not in a TOML file). Mirror the structure of any
# beamline-specific TOML in ``pv_configs/`` so users can compare side-by-side.
DEFAULT_PROFILE_DATA: Dict[str, Any] = {
    "DETECTOR_PREFIX": "prefixs",
    "OUTPUT_FILE_LOCATION": "OUTPUT.h5",
    "CONSUMER_MODE": "continuous",
    "IOC_PREFIX": "xidb",
    "CACHE_OPTIONS": {
        "CACHING_MODE": "alignment",
        "ALIGNMENT": {"MAX_CACHE_SIZE": 1000},
        "SCAN": {
            "START_SCAN": True,
            "STOP_SCAN": False,
            "THRESHOLD": 0.05,
            "MAX_CACHE_SIZE": 1000,
        },
        "BIN": {"COUNT": 10, "SIZE": 16},
    },
    "METADATA": {
        "CA": {
            "TEMPERATURE": "Temperature:Value",
            "PRESSURE": "Pressure:Value",
            "POSITION": "Position:Value",
            "VOLTAGE": "Voltage:Value",
        },
        "PVA": {},
    },
    "ANALYSIS": {
        "AXIS1": "x",
        "AXIS2": "y",
    },
    "HKL": {
        "SAMPLE_CIRCLE_AXIS_1": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "SAMPLE_CIRCLE_AXIS_2": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "SAMPLE_CIRCLE_AXIS_3": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "SAMPLE_CIRCLE_AXIS_4": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "DETECTOR_CIRCLE_AXIS_1": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "DETECTOR_CIRCLE_AXIS_2": {
            "AXIS_NUMBER": "detector:motor:AxisNumber",
            "DIRECTION_AXIS": "detector:motor:DirectionAxis",
            "POSITION": "detector:motor:Position",
        },
        "SPEC": {
            "ENERGY_VALUE": "detector:spec:Energy:Value",
            "UB_MATRIX_VALUE": "detector:spec:UB_matrix:Value",
        },
        "PRIMARY_BEAM_DIRECTION": {
            "AXIS_NUMBER_1": "PrimaryBeamDirection:AxisNumber1",
            "AXIS_NUMBER_2": "PrimaryBeamDirection:AxisNumber2",
            "AXIS_NUMBER_3": "PrimaryBeamDirection:AxisNumber3",
        },
        "INPLANE_REFERENCE_DIRECITON": {
            "AXIS_NUMBER_1": "InplaneReferenceDirection:AxisNumber1",
            "AXIS_NUMBER_2": "InplaneReferenceDirection:AxisNumber2",
            "AXIS_NUMBER_3": "InplaneReferenceDirection:AxisNumber3",
        },
        "SAMPLE_SURFACE_NORMAL_DIRECITON": {
            "AXIS_NUMBER_1": "SampleSurfaceNormalDirection:AxisNumber1",
            "AXIS_NUMBER_2": "SampleSurfaceNormalDirection:AxisNumber2",
            "AXIS_NUMBER_3": "SampleSurfaceNormalDirection:AxisNumber3",
        },
        "DETECTOR_SETUP": {
            "CENTER_CHANNEL_PIXEL": "DetectorSetup:CenterChannelPixel",
            "DISTANCE": "DetectorSetup:Distance",
            "PIXEL_DIRECTION_1": "DetectorSetup:PixelDirection1",
            "PIXEL_DIRECTION_2": "DetectorSetup:PixelDirection2",
            "SIZE": "DetectorSetup:Size",
            "UNITS": "DetectorSetup:Units",
        },
    },
}


def get_default_profile_data() -> Dict[str, Any]:
    """Return a fresh deep-copy of the default profile data.

    Deep-copied so callers can mutate the returned dict (e.g. inject a
    profile-specific IOC_PREFIX) without polluting the canonical template.
    """
    return copy.deepcopy(DEFAULT_PROFILE_DATA)


def seed_profile_defaults(profile_id: int) -> bool:
    """Write ``DEFAULT_PROFILE_DATA`` into the given profile's ``__data__`` row.

    Replaces any existing ``__data__`` blob for that profile (matching the
    semantics of ``import_toml_to_profile`` in the profile manager). Returns
    True on success, False if the DB write failed.
    """
    if not _DB_FILE.exists():
        return False
    payload = json.dumps(DEFAULT_PROFILE_DATA)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        # Remove any existing JSON blob for this profile, then insert.
        cur.execute(
            "DELETE FROM profile_configs "
            "WHERE profile_id=? AND config_type='__toml__' AND config_key='__data__'",
            (profile_id,),
        )
        cur.execute(
            "INSERT INTO profile_configs "
            "(profile_id, config_type, config_section, config_key, config_value) "
            "VALUES (?, '__toml__', NULL, '__data__', ?)",
            (profile_id, payload),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    # CLI helper: `python -m dashpva.scripts.seed_profile_defaults_sql <id>`
    import sys

    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print("usage: python -m dashpva.scripts.seed_profile_defaults_sql <profile_id>", file=sys.stderr)
        raise SystemExit(2)
    ok = seed_profile_defaults(int(sys.argv[1]))
    print("ok" if ok else "failed")
    raise SystemExit(0 if ok else 1)
