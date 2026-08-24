#!/usr/bin/env python3
# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
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

ROI / STATS PVs use the detector prefix (``DETECTOR_PREFIX``) — placeholder
``xlambda`` values are seeded so the GUI shows the canonical four ROIs and
five Stats sections; users edit the prefix per profile.

SCAN_FLAG_PV is intentionally NOT seeded into ``METADATA.CA``. ``settings.py``
derives it at reload time as ``f"{IOC_PREFIX}{_FLAG_PV_SUFFIX}"`` (suffix
``ScanOn:Value``). The same applies to ``FILE_PATH_PV`` / ``FILE_NAME_PV``.
Putting ``FLAG_PV`` / ``FILE_PATH`` / ``FILE_NAME`` in ``METADATA.CA`` only
makes sense for non-standard IOCs where the suffix doesn't match.

METADATA.CA shape (kept empty here, populated per-profile by the user)
---------------------------------------------------------------------
A user adding a profile would extend ``METADATA.CA`` with their custom PVs,
e.g.::

    "CA": {
        "TEMPERATURE": "Temperature:Value",
        "PRESSURE":    "Pressure:Value",
        "POSITION":    "Position:Value",
        "VOLTAGE":     "Voltage:Value",
    }
"""
import copy
import json
import sqlite3
from typing import Any, Dict

import dashpva.settings as _settings

_DB_FILE = _settings.PROJECT_ROOT / "dashpva.db"
DB_PATH = str(_DB_FILE)


# ── Canonical default profile data ─────────────────────────────────────────── #
# Edit values here (not in a TOML file). Mirrors pv_configs/sample_config.toml.
DEFAULT_PROFILE_DATA: Dict[str, Any] = {
    "DETECTOR_PREFIX": "xlambda",
    "IOC_PREFIX": "xidb",
    "IOC_RSM_PARAMETER": {
        "SCHEMA_VERSION": 1,
        "SAMPLE_AXES": [
            {
                "LABEL": "Mu", "RECORD_NAME": "Mu",
                "SOURCE_PV": "6idb1:m28.RBV", "DIRECTION": "x+",
                "ANGLE_UNITS": "deg",
            },
            {
                "LABEL": "Eta", "RECORD_NAME": "Eta",
                "SOURCE_PV": "6idb1:m17.RBV", "DIRECTION": "z-",
                "ANGLE_UNITS": "deg",
            },
            {
                "LABEL": "Chi", "RECORD_NAME": "Chi",
                "SOURCE_PV": "6idb1:m19.RBV", "DIRECTION": "y+",
                "ANGLE_UNITS": "deg",
            },
            {
                "LABEL": "Phi", "RECORD_NAME": "Phi",
                "SOURCE_PV": "6idb1:m20.RBV", "DIRECTION": "z-",
                "ANGLE_UNITS": "deg",
            },
        ],
        "DETECTOR_AXES": [
            {
                "LABEL": "Nu", "RECORD_NAME": "Nu",
                "SOURCE_PV": "6idb1:m29.RBV", "DIRECTION": "x+",
                "ANGLE_UNITS": "deg",
            },
            {
                "LABEL": "Delta", "RECORD_NAME": "Delta",
                "SOURCE_PV": "6idb1:m18.RBV", "DIRECTION": "z-",
                "ANGLE_UNITS": "deg",
            },
        ],
        "ENERGY_SOURCE_PV": "6idb:spec:Energy",
        "ENERGY_UNITS": "keV",
        "SAMPLE_ORIENTATION": "det",
        "UB_MATRIX": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "PRIMARY_BEAM_DIRECTION": [0.0, 1.0, 0.0],
        "INPLANE_REFERENCE_DIRECTION": [0.0, 1.0, 0.0],
        "SAMPLE_SURFACE_NORMAL_DIRECTION": [0.0, 0.0, 1.0],
        "DETECTOR_SETUP": {
            "PIXEL_DIRECTION_1": "z-",
            "PIXEL_DIRECTION_2": "x-",
            "CENTER_CHANNEL_PIXEL": [300.0, 300.0],
            "SIZE": [28.38, 28.38],
            "DISTANCE": 400.644,
            "UNITS": "mm",
        },
    },
    "OUTPUT_FILE_LOCATION": "OUTPUT.h5",
    "CONSUMER_MODE": "continuous",
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
        "CA": {},
        "PVA": {},
    },
    "ANALYSIS": {
        "AXIS1": "x",
        "AXIS2": "y",
    },
    "ROI": {
        "ROI1": {
            "MIN_X": "xlambda:ROI1:MinX",
            "MIN_Y": "xlambda:ROI1:MinY",
            "SIZE_X": "xlambda:ROI1:SizeX",
            "SIZE_Y": "xlambda:ROI1:SizeY",
        },
        "ROI2": {
            "MIN_X": "xlambda:ROI2:MinX",
            "MIN_Y": "xlambda:ROI2:MinY",
            "SIZE_X": "xlambda:ROI2:SizeX",
            "SIZE_Y": "xlambda:ROI2:SizeY",
        },
        "ROI3": {
            "MIN_X": "xlambda:ROI3:MinX",
            "MIN_Y": "xlambda:ROI3:MinY",
            "SIZE_X": "xlambda:ROI3:SizeX",
            "SIZE_Y": "xlambda:ROI3:SizeY",
        },
        "ROI4": {
            "MIN_X": "xlambda:ROI4:MinX",
            "MIN_Y": "xlambda:ROI4:MinY",
            "SIZE_X": "xlambda:ROI4:SizeX",
            "SIZE_Y": "xlambda:ROI4:SizeY",
        },
    },
    "STATS": {
        "STATS1": {
            "TOTAL": "xlambda:Stats1:Total_RBV",
            "MIN": "xlambda:Stats1:MinValue_RBV",
            "MAX": "xlambda:Stats1:MaxValue_RBV",
            "SIGMA": "xlambda:Stats1:Sigma_RBV",
            "MEAN": "xlambda:Stats1:MeanValue_RBV",
        },
        "STATS2": {
            "TOTAL": "xlambda:Stats2:Total_RBV",
            "MIN": "xlambda:Stats2:MinValue_RBV",
            "MAX": "xlambda:Stats2:MaxValue_RBV",
            "SIGMA": "xlambda:Stats2:Sigma_RBV",
            "MEAN": "xlambda:Stats2:MeanValue_RBV",
        },
        "STATS3": {
            "TOTAL": "xlambda:Stats3:Total_RBV",
            "MIN": "xlambda:Stats3:MinValue_RBV",
            "MAX": "xlambda:Stats3:MaxValue_RBV",
            "SIGMA": "xlambda:Stats3:Sigma_RBV",
            "MEAN": "xlambda:Stats3:MeanValue_RBV",
        },
        "STATS4": {
            "TOTAL": "xlambda:Stats4:Total_RBV",
            "MIN": "xlambda:Stats4:MinValue_RBV",
            "MAX": "xlambda:Stats4:MaxValue_RBV",
            "SIGMA": "xlambda:Stats4:Sigma_RBV",
            "MEAN": "xlambda:Stats4:MeanValue_RBV",
        },
        "STATS5": {
            "TOTAL": "xlambda:Stats5:Total_RBV",
            "MIN": "xlambda:Stats5:MinValue_RBV",
            "MAX": "xlambda:Stats5:MaxValue_RBV",
            "SIGMA": "xlambda:Stats5:Sigma_RBV",
            "MEAN": "xlambda:Stats5:MeanValue_RBV",
        },
    },
    "HKL": {},
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
    payload = json.dumps(get_default_profile_data())
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
