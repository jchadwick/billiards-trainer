"""Session storage implementations for different backends."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aiosqlite

    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from .enhanced_session import SessionData, SessionStatus, SessionStorage

logger = logging.getLogger(__name__)


class RedisSessionStorage(SessionStorage):
    """Redis-based session storage with persistence and clustering support."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379", key_prefix: str = "session:"
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisSessionStorage")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None

    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _session_key(self, jti: str) -> str:
        """Generate Redis key for session."""
        return f"{self.key_prefix}{jti}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user sessions set."""
        return f"{self.key_prefix}user:{user_id}"

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in Redis."""
        try:
            redis_client = await self._get_redis()
            session_key = self._session_key(session_data.jti)
            user_sessions_key = self._user_sessions_key(session_data.user_id)

            # Store session data with expiration
            session_json = json.dumps(session_data.to_dict())
            ttl = int(
                (
                    session_data.absolute_expires_at - datetime.now(timezone.utc)
                ).total_seconds()
            )

            if ttl > 0:
                await redis_client.setex(session_key, ttl, session_json)
                # Add to user sessions set
                await redis_client.sadd(user_sessions_key, session_data.jti)
                await redis_client.expire(user_sessions_key, ttl)
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti} in Redis: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data from Redis."""
        try:
            redis_client = await self._get_redis()
            session_key = self._session_key(jti)

            session_json = await redis_client.get(session_key)
            if session_json:
                session_dict = json.loads(session_json)
                return SessionData.from_dict(session_dict)

            return None
        except Exception as e:
            logger.error(f"Failed to get session {jti} from Redis: {e}")
            return None

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data in Redis."""
        # For Redis, update is the same as store
        return await self.store_session(session_data)

    async def delete_session(self, jti: str) -> bool:
        """Delete session data from Redis."""
        try:
            redis_client = await self._get_redis()
            session_key = self._session_key(jti)

            # Get session to find user_id
            session = await self.get_session(jti)
            if session:
                user_sessions_key = self._user_sessions_key(session.user_id)
                # Remove from user sessions set
                await redis_client.srem(user_sessions_key, jti)

            # Delete session
            result = await redis_client.delete(session_key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete session {jti} from Redis: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user from Redis."""
        try:
            redis_client = await self._get_redis()
            user_sessions_key = self._user_sessions_key(user_id)

            session_ids = await redis_client.smembers(user_sessions_key)
            sessions = []

            for jti in session_ids:
                session = await self.get_session(jti)
                if session:
                    sessions.append(session)
                else:
                    # Clean up stale reference
                    await redis_client.srem(user_sessions_key, jti)

            return sessions
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id} from Redis: {e}")
            return []

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions from Redis."""
        try:
            redis_client = await self._get_redis()

            # Scan for all session keys
            sessions = []
            async for key in redis_client.scan_iter(match=f"{self.key_prefix}*"):
                if not key.startswith(f"{self.key_prefix}user:"):
                    jti = key[len(self.key_prefix) :]
                    session = await self.get_session(jti)
                    if session:
                        sessions.append(session)

            return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions from Redis: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from Redis."""
        # Redis automatically expires keys, but we should clean up user session sets
        try:
            redis_client = await self._get_redis()
            count = 0

            # Scan for user session sets
            async for key in redis_client.scan_iter(match=f"{self.key_prefix}user:*"):
                key[len(f"{self.key_prefix}user:") :]
                session_ids = await redis_client.smembers(key)

                for jti in session_ids:
                    if not await redis_client.exists(self._session_key(jti)):
                        await redis_client.srem(key, jti)
                        count += 1

            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions from Redis: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class DatabaseSessionStorage(SessionStorage):
    """SQLite database session storage for persistence."""

    def __init__(self, db_path: str = "sessions.db"):
        if not SQLITE_AVAILABLE:
            raise ImportError(
                "aiosqlite package is required for DatabaseSessionStorage"
            )

        self.db_path = db_path
        self._initialized = False

    async def _init_db(self):
        """Initialize database schema."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    jti TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    absolute_expires_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL
                )
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_id ON sessions(user_id)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_expires_at ON sessions(expires_at)
            """
            )

            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status ON sessions(status)
            """
            )

            await db.commit()

        self._initialized = True

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                session_json = json.dumps(session_data.to_dict())

                await db.execute(
                    """
                    INSERT OR REPLACE INTO sessions
                    (jti, user_id, session_data, created_at, expires_at, absolute_expires_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_data.jti,
                        session_data.user_id,
                        session_json,
                        session_data.created_at.isoformat(),
                        session_data.expires_at.isoformat(),
                        session_data.absolute_expires_at.isoformat(),
                        session_data.status.value,
                    ),
                )

                await db.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti} in database: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data from database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT session_data FROM sessions WHERE jti = ?", (jti,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        session_dict = json.loads(row["session_data"])
                        return SessionData.from_dict(session_dict)

            return None
        except Exception as e:
            logger.error(f"Failed to get session {jti} from database: {e}")
            return None

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data in database."""
        return await self.store_session(session_data)

    async def delete_session(self, jti: str) -> bool:
        """Delete session data from database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("DELETE FROM sessions WHERE jti = ?", (jti,))
                await db.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete session {jti} from database: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user from database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT session_data FROM sessions WHERE user_id = ?", (user_id,)
                ) as cursor:
                    sessions = []
                    async for row in cursor:
                        session_dict = json.loads(row["session_data"])
                        sessions.append(SessionData.from_dict(session_dict))
                    return sessions
        except Exception as e:
            logger.error(
                f"Failed to get user sessions for {user_id} from database: {e}"
            )
            return []

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions from database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("SELECT session_data FROM sessions") as cursor:
                    sessions = []
                    async for row in cursor:
                        session_dict = json.loads(row["session_data"])
                        sessions.append(SessionData.from_dict(session_dict))
                    return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions from database: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from database."""
        try:
            await self._init_db()

            async with aiosqlite.connect(self.db_path) as db:
                now = datetime.now(timezone.utc).isoformat()
                cursor = await db.execute(
                    """
                    DELETE FROM sessions
                    WHERE expires_at <= ? OR absolute_expires_at <= ? OR status IN ('expired', 'invalidated')
                """,
                    (now, now),
                )
                await db.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions from database: {e}")
            return 0


class FileSessionStorage(SessionStorage):
    """File-based session storage for simple persistence."""

    def __init__(self, storage_dir: str = "session_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()

    def _session_file(self, jti: str) -> Path:
        """Get file path for session."""
        return self.storage_dir / f"{jti}.json"

    def _user_sessions_file(self, user_id: str) -> Path:
        """Get file path for user sessions index."""
        return self.storage_dir / f"user_{user_id}.json"

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in file."""
        try:
            async with self._lock:
                session_file = self._session_file(session_data.jti)
                user_sessions_file = self._user_sessions_file(session_data.user_id)

                # Store session data
                session_file.write_text(json.dumps(session_data.to_dict(), indent=2))

                # Update user sessions index
                user_sessions = []
                if user_sessions_file.exists():
                    user_sessions = json.loads(user_sessions_file.read_text())

                if session_data.jti not in user_sessions:
                    user_sessions.append(session_data.jti)
                    user_sessions_file.write_text(json.dumps(user_sessions, indent=2))

                return True
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti} in file: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data from file."""
        try:
            session_file = self._session_file(jti)
            if session_file.exists():
                session_dict = json.loads(session_file.read_text())
                return SessionData.from_dict(session_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to get session {jti} from file: {e}")
            return None

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data in file."""
        return await self.store_session(session_data)

    async def delete_session(self, jti: str) -> bool:
        """Delete session data from file."""
        try:
            async with self._lock:
                session_file = self._session_file(jti)

                # Get session to find user_id
                session = await self.get_session(jti)
                if session:
                    user_sessions_file = self._user_sessions_file(session.user_id)

                    # Update user sessions index
                    if user_sessions_file.exists():
                        user_sessions = json.loads(user_sessions_file.read_text())
                        if jti in user_sessions:
                            user_sessions.remove(jti)
                            if user_sessions:
                                user_sessions_file.write_text(
                                    json.dumps(user_sessions, indent=2)
                                )
                            else:
                                user_sessions_file.unlink(missing_ok=True)

                # Delete session file
                session_file.unlink(missing_ok=True)
                return True
        except Exception as e:
            logger.error(f"Failed to delete session {jti} from file: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user from files."""
        try:
            user_sessions_file = self._user_sessions_file(user_id)
            if not user_sessions_file.exists():
                return []

            user_sessions = json.loads(user_sessions_file.read_text())
            sessions = []

            for jti in user_sessions[:]:  # Copy list to allow modification
                session = await self.get_session(jti)
                if session:
                    sessions.append(session)
                else:
                    # Clean up stale reference
                    user_sessions.remove(jti)

            # Update user sessions file if we removed stale references
            if len(user_sessions) != len(json.loads(user_sessions_file.read_text())):
                async with self._lock:
                    if user_sessions:
                        user_sessions_file.write_text(
                            json.dumps(user_sessions, indent=2)
                        )
                    else:
                        user_sessions_file.unlink(missing_ok=True)

            return sessions
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id} from files: {e}")
            return []

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions from files."""
        try:
            sessions = []
            for session_file in self.storage_dir.glob("*.json"):
                if not session_file.name.startswith("user_"):
                    jti = session_file.stem
                    session = await self.get_session(jti)
                    if session:
                        sessions.append(session)
            return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions from files: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from files."""
        try:
            count = 0
            now = datetime.now(timezone.utc)

            for session_file in self.storage_dir.glob("*.json"):
                if not session_file.name.startswith("user_"):
                    jti = session_file.stem
                    session = await self.get_session(jti)
                    if (
                        session
                        and (
                            session.status
                            in [SessionStatus.EXPIRED, SessionStatus.INVALIDATED]
                            or session.expires_at <= now
                            or session.absolute_expires_at <= now
                        )
                        and await self.delete_session(jti)
                    ):
                        count += 1

            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions from files: {e}")
            return 0


def create_session_storage(backend_type: str, **kwargs) -> SessionStorage:
    """Factory function to create session storage backends."""
    if backend_type.lower() == "redis":
        return RedisSessionStorage(**kwargs)
    elif backend_type.lower() == "database":
        return DatabaseSessionStorage(**kwargs)
    elif backend_type.lower() == "file":
        return FileSessionStorage(**kwargs)
    else:
        # Default to memory storage
        from .enhanced_session import MemorySessionStorage

        return MemorySessionStorage()
