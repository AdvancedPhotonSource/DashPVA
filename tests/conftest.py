import os

import pytest

# Run Qt-based GUI tests headless so they don't fatally abort on machines without
# a display (CI / headless servers). Set before any test module imports Qt.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def isolated_settings(monkeypatch, tmp_path):
    """Ensure settings uses no external config during tests."""
    import dashpva.settings as settings

    monkeypatch.setattr(settings, "_locator_internal", None)
    monkeypatch.delenv("DASPVA_CONFIG_LOCATOR", raising=False)
    monkeypatch.setattr(settings, "_STATE_FILE", tmp_path / ".dashpva_locator_isolated")
    settings.reload()
    yield settings


@pytest.fixture()
def sample_config_dict():
    return {
        "BEAMLINE_NAME": "Test Beamline",
        "DETECTOR_PREFIX": "13SIM1:",
        "OUTPUT_FILE_LOCATION": "/tmp/dashpva_test",
        "CONSUMER_MODE": "standalone",
        "CACHE_OPTIONS": {
            "CACHING_MODE": "disk",
            "ALIGNMENT": {"MAX_CACHE_SIZE": 100},
            "SCAN": {
                "START_SCAN": True,
                "STOP_SCAN": False,
                "THRESHOLD": 0.5,
                "MAX_CACHE_SIZE": 50,
            },
            "BIN": {"COUNT": 10, "SIZE": 256},
        },
        "METADATA": {
            "CA": {"ENERGY": "test:energy"},
            "PVA": {"CHANNEL": "test:pva"},
        },
        "ROI": {"X": 0, "Y": 0, "W": 100, "H": 100},
        "STATS": {"ENABLED": True},
        "HKL": {"LATTICE": "cubic"},
        "ANALYSIS": {"MODE": "auto"},
        "LOG_PATH": "/tmp/dashpva_test_logs",
        "OUTPUT_PATH": "/tmp/dashpva_test_output",
    }


@pytest.fixture()
def tmp_toml(tmp_path, sample_config_dict):
    """Create a temporary TOML config file."""
    import toml

    toml_path = tmp_path / "test_config.toml"
    toml_path.write_text(toml.dumps(sample_config_dict))
    return str(toml_path)


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Create a temporary SQLite database for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import dashpva.database.db as db_mod

    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    monkeypatch.setattr(db_mod, "DB_FILE", db_file)
    monkeypatch.setattr(db_mod, "DATABASE_URL", db_url)
    monkeypatch.setattr(db_mod, "_engine", engine)
    monkeypatch.setattr(db_mod, "_Session", session_factory)
    monkeypatch.setattr(db_mod, "_init_done", False)

    db_mod.Base.metadata.create_all(engine)

    from dashpva.database import DatabaseInterface

    return DatabaseInterface()
