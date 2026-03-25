# """
# SQLAlchemy models for DashPVA profile management
# """
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import settings

log = logging.getLogger(__name__)

Base = declarative_base()

# Database configuration using absolute path
# Get project root from the centralized settings module
PROJECT_ROOT = settings.PROJECT_ROOT
DB_FILE = PROJECT_ROOT / "dashpva.db"
DATABASE_URL = f"sqlite:///{DB_FILE.as_posix()}"

# Issue 5: module-level engine and session factory (not created per-call)
_engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)
_Session = sessionmaker(bind=_engine, expire_on_commit=False)


def get_engine():
    """Return the shared database engine."""
    return _engine


def get_session():
    """Return a new session from the shared session factory."""
    return _Session()


def create_tables():
    """Create all tables in the database (idempotent via create_all)."""
    # Late-import models so they register themselves with Base.metadata
    import database.models.setting_value  # noqa: F401
    import database.models.settings  # noqa: F401
    import database.models.profile   # noqa: F401
    Base.metadata.create_all(_engine)
    migrate_database()


def migrate_database():
    """Apply incremental column migrations to existing tables."""
    import sqlite3
    if not DB_FILE.exists():
        return
    conn = sqlite3.connect(str(DB_FILE))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='profiles'")
        if cursor.fetchone():
            cursor.execute("PRAGMA table_info(profiles)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            if 'is_default' not in existing_cols:
                cursor.execute("ALTER TABLE profiles ADD COLUMN is_default BOOLEAN NOT NULL DEFAULT 0")
            if 'is_selected' not in existing_cols:
                cursor.execute("ALTER TABLE profiles ADD COLUMN is_selected BOOLEAN NOT NULL DEFAULT 0")
        conn.commit()
    finally:
        conn.close()


_init_done = False


def init_database():
    """Initialize the database (tables + seed). No-op after the first successful call."""
    global _init_done
    if _init_done:
        return
    # Issue 1: always create tables so new models get their tables on existing DBs
    is_new_db = not DB_FILE.exists()
    create_tables()
    if is_new_db:
        # Issue 7: log failures instead of silently swallowing them
        try:
            from scripts.seed_settings_defaults_sql import seed_defaults
            seed_defaults()
        except Exception as exc:
            log.warning("seed_defaults() failed on new database: %s", exc)
    # Issue 2: set flag only after all operations succeed
    _init_done = True
    log.debug("Database initialized (new_db=%s)", is_new_db)
