"""Database connection management."""

import logging
import os
from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./billiards_trainer.db")

# For SQLite in-memory testing
TEST_DATABASE_URL = "sqlite:///:memory:"


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, database_url: str = None, echo: bool = False):
        """Initialize database manager."""
        self.database_url = database_url or DATABASE_URL
        self.echo = echo
        self._engine = None
        self._session_local = None

    def get_engine(self) -> Engine:
        """Get database engine."""
        if self._engine is None:
            # Special handling for SQLite
            if self.database_url.startswith("sqlite"):
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
            else:
                # PostgreSQL or other databases
                self._engine = create_engine(
                    self.database_url,
                    echo=self.echo,
                    pool_pre_ping=True,
                    pool_recycle=300,
                )

            # Enable foreign key constraints for SQLite
            if self.database_url.startswith("sqlite"):

                @event.listens_for(self._engine, "connect")
                def set_sqlite_pragma(dbapi_connection, _connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()

        return self._engine

    def get_session_local(self) -> sessionmaker:
        """Get session local factory."""
        if self._session_local is None:
            self._session_local = sessionmaker(
                autocommit=False, autoflush=False, bind=self.get_engine()
            )
        return self._session_local

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.get_engine())
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.get_engine())
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise

    def get_session(self) -> Session:
        """Get a new database session."""
        session_local = self.get_session_local()
        return session_local()

    def init_db(self) -> None:
        """Initialize database with tables and default data."""
        self.create_tables()
        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default admin user if no users exist."""
        from ..utils.security import UserRole
        from .repositories import UserRepository

        session = self.get_session()
        try:
            user_repo = UserRepository(session)

            # Check if any users exist
            user_count = user_repo.count()
            if user_count > 0:
                logger.info("Users already exist, skipping default user creation")
                return

            # Create default admin user
            admin_user = user_repo.create(
                username="admin",
                email="admin@billiards-trainer.local",
                password="admin123!",
                role=UserRole.ADMIN,
                first_name="System",
                last_name="Administrator",
                is_verified=True,
            )

            # Create default operator user
            operator_user = user_repo.create(
                username="operator",
                email="operator@billiards-trainer.local",
                password="operator123!",
                role=UserRole.OPERATOR,
                first_name="System",
                last_name="Operator",
                is_verified=True,
            )

            # Create default viewer user
            viewer_user = user_repo.create(
                username="viewer",
                email="viewer@billiards-trainer.local",
                password="viewer123!",
                role=UserRole.VIEWER,
                first_name="System",
                last_name="Viewer",
                is_verified=True,
            )

            session.commit()
            logger.info("Default users created successfully")
            logger.info(f"Admin user ID: {admin_user.id}")
            logger.info(f"Operator user ID: {operator_user.id}")
            logger.info(f"Viewer user ID: {viewer_user.id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error creating default users: {e}")
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    session = db_manager.get_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    """Initialize database for the application."""
    try:
        db_manager.init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_test_db_manager() -> DatabaseManager:
    """Get a test database manager with in-memory SQLite."""
    return DatabaseManager(database_url=TEST_DATABASE_URL, echo=False)


def setup_test_database() -> DatabaseManager:
    """Setup a test database with tables."""
    test_db = get_test_db_manager()
    test_db.create_tables()
    return test_db
