"""Tests for dashpva.scripts.seed_profile_defaults_sql — default profile JSON seeding.

Covers the script's two surfaces:
1. The ``DEFAULT_PROFILE_DATA`` literal (structure + invariants).
2. The ``seed_profile_defaults(profile_id)`` write — insert + replace semantics.
"""

import json
import sqlite3

import pytest


@pytest.fixture()
def profile_db(tmp_path, monkeypatch):
    """Create a sqlite DB with the profile_configs schema, repointed for tests.

    Returns the path of the test DB. The seed script is monkey-patched so its
    raw-SQL writes hit this DB instead of the production ``dashpva.db``.
    """
    db_file = tmp_path / "profile_seed_test.db"
    conn = sqlite3.connect(db_file)
    conn.executescript(
        """
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
    # Pre-create a couple of profile rows so seed_profile_defaults has IDs to target.
    conn.execute("INSERT INTO profiles (id, name) VALUES (1, 'p1')")
    conn.execute("INSERT INTO profiles (id, name) VALUES (2, 'p2')")
    conn.commit()
    conn.close()

    import dashpva.scripts.seed_profile_defaults_sql as seed_mod

    monkeypatch.setattr(seed_mod, "_DB_FILE", db_file)
    monkeypatch.setattr(seed_mod, "DB_PATH", str(db_file))
    return db_file


def _read_data_row(db_file, profile_id):
    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT config_value FROM profile_configs "
        "WHERE profile_id=? AND config_type='__toml__' AND config_key='__data__'",
        (profile_id,),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _count_data_rows(db_file, profile_id):
    conn = sqlite3.connect(db_file)
    row = conn.execute(
        "SELECT COUNT(*) FROM profile_configs "
        "WHERE profile_id=? AND config_type='__toml__' AND config_key='__data__'",
        (profile_id,),
    ).fetchone()
    conn.close()
    return row[0]


class TestDefaultProfileDataLiteral:
    """Invariants on the canonical default dict — no DB writes."""

    def test_top_level_keys(self):
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        expected = {
            "DETECTOR_PREFIX", "OUTPUT_FILE_LOCATION", "CONSUMER_MODE",
            "IOC_PREFIX", "CACHE_OPTIONS", "METADATA", "ANALYSIS", "HKL",
        }
        assert set(DEFAULT_PROFILE_DATA.keys()) == expected

    def test_no_roi_or_stats_at_root(self):
        """ROI/STATS removed — area_det_viewer builds them at runtime."""
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        assert "ROI" not in DEFAULT_PROFILE_DATA
        assert "STATS" not in DEFAULT_PROFILE_DATA

    def test_ioc_prefix_at_root(self):
        """IOC_PREFIX must be a root-level string (not nested under METADATA)."""
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        assert isinstance(DEFAULT_PROFILE_DATA["IOC_PREFIX"], str)
        assert DEFAULT_PROFILE_DATA["IOC_PREFIX"]  # non-empty

    def test_cache_options_has_expected_modes(self):
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        cache = DEFAULT_PROFILE_DATA["CACHE_OPTIONS"]
        assert set(cache.keys()) >= {"CACHING_MODE", "ALIGNMENT", "SCAN", "BIN"}

    def test_metadata_has_ca_and_pva(self):
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        metadata = DEFAULT_PROFILE_DATA["METADATA"]
        assert "CA" in metadata
        assert "PVA" in metadata

    def test_json_serializable(self):
        """Must survive a JSON round-trip — that's how it's stored in the DB."""
        from dashpva.scripts.seed_profile_defaults_sql import DEFAULT_PROFILE_DATA

        restored = json.loads(json.dumps(DEFAULT_PROFILE_DATA))
        assert restored == DEFAULT_PROFILE_DATA


class TestGetDefaultProfileData:
    """The accessor returns a deep copy to protect the template from mutation."""

    def test_returns_equal_to_constant(self):
        from dashpva.scripts.seed_profile_defaults_sql import (
            DEFAULT_PROFILE_DATA, get_default_profile_data,
        )

        assert get_default_profile_data() == DEFAULT_PROFILE_DATA

    def test_returns_deep_copy_not_alias(self):
        """Mutating the returned dict must not pollute the canonical template."""
        from dashpva.scripts.seed_profile_defaults_sql import (
            DEFAULT_PROFILE_DATA, get_default_profile_data,
        )

        copy_a = get_default_profile_data()
        copy_a["IOC_PREFIX"] = "mutated"
        copy_a["CACHE_OPTIONS"]["ALIGNMENT"]["MAX_CACHE_SIZE"] = 999

        # Original untouched
        assert DEFAULT_PROFILE_DATA["IOC_PREFIX"] != "mutated"
        assert DEFAULT_PROFILE_DATA["CACHE_OPTIONS"]["ALIGNMENT"]["MAX_CACHE_SIZE"] != 999

        # Fresh call gives a clean copy too
        copy_b = get_default_profile_data()
        assert copy_b["IOC_PREFIX"] != "mutated"


class TestSeedProfileDefaultsWrite:
    """seed_profile_defaults(profile_id) — inserts or replaces the __data__ row."""

    def test_inserts_when_no_data_row_exists(self, profile_db):
        from dashpva.scripts.seed_profile_defaults_sql import (
            DEFAULT_PROFILE_DATA, seed_profile_defaults,
        )

        assert _count_data_rows(profile_db, 1) == 0
        result = seed_profile_defaults(1)
        assert result is True
        assert _count_data_rows(profile_db, 1) == 1
        assert _read_data_row(profile_db, 1) == DEFAULT_PROFILE_DATA

    def test_replaces_existing_data_row(self, profile_db):
        """If a __data__ row already exists, it's replaced (not duplicated)."""
        from dashpva.scripts.seed_profile_defaults_sql import (
            DEFAULT_PROFILE_DATA, seed_profile_defaults,
        )

        # Pre-seed with a stale value
        conn = sqlite3.connect(profile_db)
        conn.execute(
            "INSERT INTO profile_configs "
            "(profile_id, config_type, config_section, config_key, config_value) "
            "VALUES (1, '__toml__', NULL, '__data__', ?)",
            (json.dumps({"DETECTOR_PREFIX": "stale_value"}),),
        )
        conn.commit()
        conn.close()
        assert _count_data_rows(profile_db, 1) == 1

        result = seed_profile_defaults(1)
        assert result is True
        # Still exactly one row (replaced, not duplicated)
        assert _count_data_rows(profile_db, 1) == 1
        # New content is the defaults
        assert _read_data_row(profile_db, 1) == DEFAULT_PROFILE_DATA

    def test_does_not_touch_other_profiles(self, profile_db):
        """Seeding profile 1 must not affect profile 2's data row."""
        from dashpva.scripts.seed_profile_defaults_sql import seed_profile_defaults

        # Pre-seed profile 2 with custom data
        custom = {"DETECTOR_PREFIX": "p2_custom"}
        conn = sqlite3.connect(profile_db)
        conn.execute(
            "INSERT INTO profile_configs "
            "(profile_id, config_type, config_section, config_key, config_value) "
            "VALUES (2, '__toml__', NULL, '__data__', ?)",
            (json.dumps(custom),),
        )
        conn.commit()
        conn.close()

        seed_profile_defaults(1)

        # Profile 2 should be unchanged
        assert _read_data_row(profile_db, 2) == custom

    def test_returns_false_when_db_missing(self, tmp_path, monkeypatch):
        """No DB file → return False gracefully (don't raise)."""
        import dashpva.scripts.seed_profile_defaults_sql as seed_mod

        missing = tmp_path / "does_not_exist.db"
        monkeypatch.setattr(seed_mod, "_DB_FILE", missing)
        monkeypatch.setattr(seed_mod, "DB_PATH", str(missing))

        assert seed_mod.seed_profile_defaults(1) is False

    def test_seeded_blob_round_trips_through_json(self, profile_db):
        """What we wrote can be read back and parsed without loss."""
        from dashpva.scripts.seed_profile_defaults_sql import (
            DEFAULT_PROFILE_DATA, seed_profile_defaults,
        )

        seed_profile_defaults(1)
        read_back = _read_data_row(profile_db, 1)
        assert read_back == DEFAULT_PROFILE_DATA
        # Sanity: nested structure preserved (not flattened or stringified)
        assert isinstance(read_back["CACHE_OPTIONS"]["ALIGNMENT"], dict)
        assert isinstance(read_back["HKL"]["SAMPLE_CIRCLE_AXIS_1"], dict)
