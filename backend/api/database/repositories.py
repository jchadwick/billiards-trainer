"""Database repositories for common operations."""

from sqlalchemy.orm import Session


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session
