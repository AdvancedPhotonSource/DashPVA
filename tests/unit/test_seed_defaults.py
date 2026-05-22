"""Tests for dashpva.scripts.seed_settings_defaults_sql — seeding & idempotency.

Covers the seed script's two responsibilities:
1. Seeding app-level rows in `settings` / `setting_values` (idempotent).
2. Backfilling root-level `IOC_PREFIX` into each profile's `__data__` JSON,
   derived from `METADATA.CA.FLAG_PV` (prefixed form) or `DETECTOR_PREFIX`.
"""

import json
import sqlite3

import pytest


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """Create a fresh sqlite DB with the schema that seed_defaults() expects.

    Returns (db_path, run_seed) where run_seed() invokes the production
    seed_defaults() against this isolated DB.
    """
    db_file = tmp_path / "seed_test.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        """
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(100) NOT NULL,
            "desc" TEXT,
            parent_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES settings(id)
        );
        CREATE TABLE setting_values (
            id INTEGER PRIMARY KEY,
            setting_id INTEGER NOT NULL,
            "key" VARCHAR(255) NOT NULL,
            value TEXT NOT NULL,
            value_type VARCHAR(20) NOT NULL,
            FOREIGN KEY(setting_id) REFERENCES settings(id)
        );
        CREATE TABLE profiles (
            id INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE
        );
        CREATE TABLE profile_configs (
            id INTEGER PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            config_type VARCHAR(100) NOT NULL,
            config_section VARCHAR(255),
            config_key VARCHAR(255) NOT NULL,
            config_value TEXT NOT NULL,
            FOREIGN KEY(profile_id) REFERENCES profiles(id)
        );
        """
    )
    conn.commit()
    conn.close()

    # Repoint the seed script at the test DB.
    import dashpva.scripts.seed_settings_defaults_sql as seed_mod

    monkeypatch.setattr(seed_mod, "_DB_FILE", db_file)
    monkeypatch.setattr(seed_mod, "DB_PATH", str(db_file))

    def run_seed():
        seed_mod.seed_defaults()

    return db_file, run_seed


def _insert_profile(db_file, profile_id, name, data_dict):
    """Helper: insert a profile + its __toml__/__data__ JSON blob."""
    conn = sqlite3.connect(db_file)
    conn.execute("INSERT INTO profiles (id, name) VALUES (?, ?)", (profile_id, name))
    conn.execute(
        "INSERT INTO profile_configs (profile_id, config_type, config_section, config_key, config_value) "
        "VALUES (?, '__toml__', NULL, '__data__', ?)",
        (profile_id, json.dumps(data_dict)),
    )
    conn.commit()
    conn.close()


def _read_profile_data(db_file, profile_id):
    """Helper: read the parsed __data__ JSON for a profile."""
    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT config_value FROM profile_configs "
        "WHERE profile_id=? AND config_type='__toml__' AND config_key='__data__'",
        (profile_id,),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


class TestSeedDefaultsAppRows:
    """The script seeds BEAMLINE_NAME, PATHS, CONSUMERS, APP_DATA. Should be idempotent."""

    def test_seeds_expected_top_level_settings(self, seeded_db):
        db_file, run_seed = seeded_db
        run_seed()
        conn = sqlite3.connect(db_file)
        names = {row[0] for row in conn.execute("SELECT name FROM settings WHERE parent_id IS NULL")}
        conn.close()
        assert {"BEAMLINE_NAME", "PATHS", "APP_DATA"} <= names

    def test_idempotent_no_duplicate_rows(self, seeded_db):
        db_file, run_seed = seeded_db
        run_seed()
        run_seed()
        run_seed()
        conn = sqlite3.connect(db_file)
        # No duplicates: each (name, parent_id) pair should appear at most once.
        dupes = conn.execute(
            "SELECT name, parent_id, COUNT(*) FROM settings GROUP BY name, parent_id HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert dupes == []


class TestSeedDefaultsIocPrefixBackfill:
    """The script also backfills root-level IOC_PREFIX into profile __data__ JSONs."""

    def test_derives_from_prefixed_flag_pv(self, seeded_db):
        """FLAG_PV like '6idb1:ScanOn:Value' → IOC_PREFIX = '6idb1'."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 1, "p1", {
            "DETECTOR_PREFIX": "s6lambda1",
            "METADATA": {"CA": {"FLAG_PV": "6idb1:ScanOn:Value"}},
        })
        run_seed()
        data = _read_profile_data(db_file, 1)
        assert data["IOC_PREFIX"] == "6idb1"
        # DETECTOR_PREFIX preserved — not overwritten.
        assert data["DETECTOR_PREFIX"] == "s6lambda1"

    def test_falls_back_to_detector_prefix_when_flag_pv_unprefixed(self, seeded_db):
        """FLAG_PV like 'ScanOn:Value' (no prefix) → falls back to DETECTOR_PREFIX."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 2, "p2", {
            "DETECTOR_PREFIX": "xidb",
            "METADATA": {"CA": {"FLAG_PV": "ScanOn:Value"}},
        })
        run_seed()
        data = _read_profile_data(db_file, 2)
        assert data["IOC_PREFIX"] == "xidb"

    def test_falls_back_to_detector_prefix_when_no_metadata(self, seeded_db):
        """No METADATA at all → falls back to DETECTOR_PREFIX."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 3, "p3", {"DETECTOR_PREFIX": "fallback"})
        run_seed()
        data = _read_profile_data(db_file, 3)
        assert data["IOC_PREFIX"] == "fallback"

    def test_strips_trailing_colon_from_detector_prefix(self, seeded_db):
        """DETECTOR_PREFIX='13SIM1:' → IOC_PREFIX='13SIM1' (no trailing colon at root)."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 4, "p4", {"DETECTOR_PREFIX": "13SIM1:"})
        run_seed()
        data = _read_profile_data(db_file, 4)
        assert data["IOC_PREFIX"] == "13SIM1"

    def test_preserves_explicit_ioc_prefix(self, seeded_db):
        """An existing IOC_PREFIX is never overwritten, even if a different one could be derived."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 5, "p5", {
            "IOC_PREFIX": "explicit_value",
            "DETECTOR_PREFIX": "would_overwrite",
            "METADATA": {"CA": {"FLAG_PV": "would_also_overwrite:ScanOn:Value"}},
        })
        run_seed()
        data = _read_profile_data(db_file, 5)
        assert data["IOC_PREFIX"] == "explicit_value"

    def test_handles_string_metadata(self, seeded_db):
        """METADATA == '' (string instead of dict) must not crash — common in partial profiles."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 6, "p6", {
            "DETECTOR_PREFIX": "ok",
            "METADATA": "",  # malformed
        })
        run_seed()  # must not raise
        data = _read_profile_data(db_file, 6)
        assert data["IOC_PREFIX"] == "ok"

    def test_handles_string_metadata_ca(self, seeded_db):
        """METADATA.CA == None — same deal, must not crash."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 7, "p7", {
            "DETECTOR_PREFIX": "ok",
            "METADATA": {"CA": None},
        })
        run_seed()
        data = _read_profile_data(db_file, 7)
        assert data["IOC_PREFIX"] == "ok"

    def test_skips_profile_with_no_prefix_signals(self, seeded_db):
        """No FLAG_PV, no DETECTOR_PREFIX → leave IOC_PREFIX unset (don't fabricate one)."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 8, "p8", {"some_other_key": "value"})
        run_seed()
        data = _read_profile_data(db_file, 8)
        assert "IOC_PREFIX" not in data

    def test_idempotent_does_not_re_derive(self, seeded_db):
        """Running twice: second run should be a no-op for already-backfilled profiles."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 9, "p9", {
            "DETECTOR_PREFIX": "first",
            "METADATA": {"CA": {"FLAG_PV": "first:ScanOn:Value"}},
        })
        run_seed()
        first = _read_profile_data(db_file, 9)
        assert first["IOC_PREFIX"] == "first"

        # User edits the profile, changing what the derive logic *would* pick.
        # Idempotent seed must not stomp the user's IOC_PREFIX choice.
        conn = sqlite3.connect(db_file)
        first["IOC_PREFIX"] = "user_chose_this"
        conn.execute(
            "UPDATE profile_configs SET config_value=? WHERE profile_id=9 AND config_key='__data__'",
            (json.dumps(first),),
        )
        conn.commit()
        conn.close()

        run_seed()
        assert _read_profile_data(db_file, 9)["IOC_PREFIX"] == "user_chose_this"

    def test_handles_multiple_profiles_in_one_pass(self, seeded_db):
        """Several profiles in one seed pass — each derives independently."""
        db_file, run_seed = seeded_db
        _insert_profile(db_file, 10, "p10", {"DETECTOR_PREFIX": "alpha"})
        _insert_profile(db_file, 11, "p11", {
            "DETECTOR_PREFIX": "beta",
            "METADATA": {"CA": {"FLAG_PV": "gamma:ScanOn:Value"}},
        })
        _insert_profile(db_file, 12, "p12", {"IOC_PREFIX": "delta"})
        run_seed()
        assert _read_profile_data(db_file, 10)["IOC_PREFIX"] == "alpha"
        assert _read_profile_data(db_file, 11)["IOC_PREFIX"] == "gamma"
        assert _read_profile_data(db_file, 12)["IOC_PREFIX"] == "delta"
