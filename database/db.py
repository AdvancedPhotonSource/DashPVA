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
    engine = get_engine()
    Base.metadata.create_all(engine)

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
