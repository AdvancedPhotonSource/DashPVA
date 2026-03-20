# """
# SQLAlchemy models for DashPVA profile management
# """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import settings

Base = declarative_base()

# Database configuration using absolute path
# Get project root from the centralized settings module
PROJECT_ROOT = settings.PROJECT_ROOT
DB_FILE = PROJECT_ROOT / "dashpva.db"
DATABASE_URL = f"sqlite:///{DB_FILE.as_posix()}"

def get_engine():
    """Create and return database engine"""
    return create_engine(DATABASE_URL, echo=False)

def get_session():
    """Create and return database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def create_tables():
    """Create all tables in the database"""
    # Late-import models so they register themselves with Base.metadata
    import database.models.setting_value  # noqa: F401
    import database.models.settings  # noqa: F401
    import database.models.profile   # noqa: F401
    engine = get_engine()
    Base.metadata.create_all(engine)
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


def init_database():
    """Initialize the database with tables"""
    if not DB_FILE.exists():
        create_tables()
        # Seed default settings only on first creation using raw SQL script
        try:
            from scripts.seed_settings_defaults_sql import seed_defaults
            seed_defaults()
        except Exception:
            # Seed script may be absent; ignore errors per original behavior
            pass
        print("Database initialized successfully")
    else:
        print("Database already exists")
