#!/usr/bin/env python3
"""
Seed default hierarchical settings for DashPVA using raw SQL.

Safe to call on an existing database — every node is checked before insertion
so no duplicates are created.

Creates (if missing):
- BEAMLINE_NAME (string)
    BEAMLINE_NAME = ""          — human-readable beamline identifier
- PATHS (root)
    LOG     (BASE=logs)         — log file directory
    CONFIGS (BASE=configs)      — PV config directory
    OUTPUTS (BASE=outputs)
        SCAN   (BASE=scans)
        SLICES (BASE=slices)
- PATHS (root) continued:
    CONSUMERS (path)
        BASE = consumers        — consumer working directory
        IOC  = caIOC_servers    — IOC directory name
        hpc (section)
            meta     = meta     — metadata consumer name
            analysis = analysis — analysis consumer name
- APP_DATA (root)
    workflow (section)
        meta       (section)    — associator consumer last-used / named configs
        collector  (section)    — collector last-used / named configs
        analysis   (section)    — analysis consumer last-used / named configs
"""
import sqlite3
import settings as _settings

# Issue 3: resolve the DB path from settings.PROJECT_ROOT — same anchor used by
# database.db — instead of computing it relative to __file__ which breaks when the
# scripts/ directory moves or the package is installed elsewhere.
_DB_FILE = _settings.PROJECT_ROOT / "dashpva.db"
DB_PATH = str(_DB_FILE)


def get_or_create_setting(cur, name: str, type_: str, desc: str = "", parent_id=None) -> int:
    if parent_id is None:
        cur.execute(
            "SELECT id FROM settings WHERE name=? AND parent_id IS NULL",
            (name,),
        )
    else:
        cur.execute(
            "SELECT id FROM settings WHERE name=? AND parent_id=?",
            (name, parent_id),
        )
    row = cur.fetchone()
    if row and row[0]:
        return int(row[0])
    cur.execute(
        "INSERT INTO settings (name, type, desc, parent_id) VALUES (?, ?, ?, ?)",
        (name, type_, desc, parent_id),
    )
    return int(cur.lastrowid)


def add_value_if_missing(cur, setting_id: int, key: str, value: str, value_type: str = "string") -> None:
    cur.execute(
        "SELECT id FROM setting_values WHERE setting_id=? AND key=?",
        (setting_id, key),
    )
    if cur.fetchone():
        return
    cur.execute(
        "INSERT INTO setting_values (setting_id, key, value, value_type) VALUES (?, ?, ?, ?)",
        (setting_id, key, value, value_type),
    )


def seed_defaults() -> None:
    if not _DB_FILE.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # ── BEAMLINE_NAME ─────────────────────────────────────────────────── #
        beamline_id = get_or_create_setting(cur, "BEAMLINE_NAME", "string", "Beamline identification", None)
        add_value_if_missing(cur, beamline_id, "BEAMLINE_NAME", "")

        # ── PATHS ─────────────────────────────────────────────────────────── #
        paths_id = get_or_create_setting(cur, "PATHS", "root", "Application paths", None)
        log_id = get_or_create_setting(cur, "LOG", "path", "Logs directory", paths_id)
        add_value_if_missing(cur, log_id, "BASE", "logs")
        configs_id = get_or_create_setting(cur, "CONFIGS", "path", "PV configs directory", paths_id)
        add_value_if_missing(cur, configs_id, "BASE", "configs")
        outputs_id = get_or_create_setting(cur, "OUTPUTS", "path", "Outputs directory", paths_id)
        add_value_if_missing(cur, outputs_id, "BASE", "outputs")
        scan_id = get_or_create_setting(cur, "SCAN", "path", "Scans directory", outputs_id)
        add_value_if_missing(cur, scan_id, "BASE", "scans")
        slices_id = get_or_create_setting(cur, "SLICES", "path", "Slices directory", outputs_id)
        add_value_if_missing(cur, slices_id, "BASE", "slices")

        # ── CONSUMERS (under PATHS) ───────────────────────────────────────── #
        consumers_id = get_or_create_setting(cur, "CONSUMERS", "path", "Consumer directories", paths_id)
        add_value_if_missing(cur, consumers_id, "BASE", "consumers")
        add_value_if_missing(cur, consumers_id, "IOC", "caIOC_servers")
        hpc_id = get_or_create_setting(cur, "hpc", "section", "HPC consumer names", consumers_id)
        add_value_if_missing(cur, hpc_id, "BASE", "hpc")
        add_value_if_missing(cur, hpc_id, "meta", "meta")
        add_value_if_missing(cur, hpc_id, "analysis", "analysis")

        # ── APP_DATA > workflow > {meta, collector, analysis} ─────────────── #
        app_data_id = get_or_create_setting(cur, "APP_DATA", "root", "Application runtime data", None)
        workflow_id = get_or_create_setting(cur, "workflow", "section", "Workflow tab state", app_data_id)
        get_or_create_setting(cur, "meta",      "section", "Associator consumer configs", workflow_id)
        get_or_create_setting(cur, "collector", "section", "Collector configs",           workflow_id)
        get_or_create_setting(cur, "analysis",  "section", "Analysis consumer configs",   workflow_id)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    seed_defaults()
