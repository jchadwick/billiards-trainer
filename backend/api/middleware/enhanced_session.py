"""Enhanced session management system with advanced features."""

import asyncio
import json
import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Optional

try:
    from cryptography.fernet import Fernet

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

    # Create a mock Fernet for testing
    class MockFernet:
        @staticmethod
        def generate_key():
            return b"mock_key_1234567890123456789012345678"

        def __init__(self, key):
            self.key = key

        def encrypt(self, data):
            return data

        def decrypt(self, data):
            return data

    Fernet = MockFernet

from pydantic import BaseModel, Field

from ..utils.security import SecurityEvent, SecurityEventType, UserRole, security_logger

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status values."""

    ACTIVE = "active"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    LOCKED = "locked"
    SUSPENDED = "suspended"


class SessionStorageBackend(str, Enum):
    """Session storage backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"
    FILE = "file"


class SessionPolicy(BaseModel):
    """Configurable session policy."""

    # Timeout settings
    idle_timeout_minutes: int = Field(default=30, ge=1, le=1440)  # 30 min to 24 hours
    absolute_timeout_hours: int = Field(default=8, ge=1, le=72)  # 1 to 72 hours
    remember_me_timeout_days: int = Field(default=30, ge=1, le=365)  # 1 to 365 days

    # Concurrent session limits
    max_concurrent_sessions: int = Field(default=5, ge=1, le=50)
    allow_concurrent_same_device: bool = True

    # Security settings
    require_ip_validation: bool = True
    allow_ip_change: bool = False
    require_user_agent_validation: bool = True
    enable_session_hijacking_protection: bool = True

    # Monitoring and cleanup
    cleanup_interval_minutes: int = Field(default=15, ge=1, le=60)
    session_activity_tracking: bool = True
    detailed_audit_logging: bool = True

    # Encryption
    encrypt_session_data: bool = True
    rotate_encryption_key: bool = False
    key_rotation_days: int = Field(default=30, ge=1, le=365)


@dataclass
class SessionData:
    """Enhanced session data structure."""

    # Basic session info
    jti: str
    user_id: str
    role: str
    status: SessionStatus

    # Timestamps
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    absolute_expires_at: datetime

    # Security tracking
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None

    # Session metadata
    remember_me: bool = False
    is_mobile: bool = False
    session_type: str = "web"  # web, api, mobile

    # Activity tracking
    request_count: int = 0
    last_request_path: Optional[str] = None
    geographic_location: Optional[str] = None

    # Security flags
    ip_locked: bool = False
    requires_reauth: bool = False
    suspicious_activity_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, SessionStatus):
                data[key] = value.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = [
            "created_at",
            "last_activity",
            "expires_at",
            "absolute_expires_at",
        ]
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = SessionStatus(data["status"])

        return cls(**data)


class SessionAnalytics(BaseModel):
    """Session analytics and metrics."""

    total_sessions: int = 0
    active_sessions: int = 0
    expired_sessions: int = 0
    concurrent_sessions_by_user: dict[str, int] = {}
    sessions_by_role: dict[str, int] = {}
    sessions_by_location: dict[str, int] = {}
    average_session_duration: float = 0.0
    peak_concurrent_sessions: int = 0
    suspicious_activity_count: int = 0


class SessionStorage:
    """Abstract session storage interface."""

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data."""
        raise NotImplementedError

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data."""
        raise NotImplementedError

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data."""
        raise NotImplementedError

    async def delete_session(self, jti: str) -> bool:
        """Delete session data."""
        raise NotImplementedError

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user."""
        raise NotImplementedError

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions."""
        raise NotImplementedError

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        raise NotImplementedError


class MemorySessionStorage(SessionStorage):
    """In-memory session storage implementation."""

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._user_sessions: dict[str, set[str]] = {}
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data."""
        try:
            self._sessions[session_data.jti] = session_data

            # Track user sessions
            if session_data.user_id not in self._user_sessions:
                self._user_sessions[session_data.user_id] = set()
            self._user_sessions[session_data.user_id].add(session_data.jti)

            return True
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti}: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data."""
        return self._sessions.get(jti)

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data."""
        if jti in self._sessions:
            self._sessions[jti] = session_data
            return True
        return False

    async def delete_session(self, jti: str) -> bool:
        """Delete session data."""
        session = self._sessions.pop(jti, None)
        if session:
            # Remove from user sessions
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(jti)
                if not self._user_sessions[session.user_id]:
                    del self._user_sessions[session.user_id]
            return True
        return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user."""
        sessions = []
        if user_id in self._user_sessions:
            for jti in self._user_sessions[user_id]:
                session = self._sessions.get(jti)
                if session:
                    sessions.append(session)
        return sessions

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions."""
        return list(self._sessions.values())

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired_sessions = []

        for jti, session in self._sessions.items():
            if (
                session.status == SessionStatus.EXPIRED
                or session.expires_at <= now
                or session.absolute_expires_at <= now
            ):
                expired_sessions.append(jti)

        count = 0
        for jti in expired_sessions:
            if await self.delete_session(jti):
                count += 1

        return count


class RedisSessionStorage(SessionStorage):
    """Redis-based session storage implementation."""

    def __init__(
        self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "session:"
    ):
        self.key_prefix = key_prefix
        self._redis = None
        self._redis_url = redis_url
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

    async def _get_redis(self):
        """Get Redis connection with lazy initialization."""
        if self._redis is None:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(self._redis_url, decode_responses=True)
                # Test connection
                await self._redis.ping()
            except ImportError:
                logger.error(
                    "redis package not installed, falling back to memory storage"
                )
                raise NotImplementedError("Redis backend requires 'redis' package")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis

    def _serialize_session(self, session_data: SessionData) -> str:
        """Serialize and encrypt session data."""
        session_dict = {
            "jti": session_data.jti,
            "user_id": session_data.user_id,
            "username": session_data.username,
            "role": session_data.role,
            "permissions": list(session_data.permissions),
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat(),
            "expires_at": session_data.expires_at.isoformat(),
            "absolute_expires_at": session_data.absolute_expires_at.isoformat(),
            "ip_address": session_data.ip_address,
            "user_agent": session_data.user_agent,
            "metadata": session_data.metadata,
            "status": session_data.status.value,
        }
        json_str = json.dumps(session_dict)
        encrypted = self._cipher.encrypt(json_str.encode())
        return encrypted.decode()

    def _deserialize_session(self, encrypted_data: str) -> SessionData:
        """Decrypt and deserialize session data."""
        decrypted = self._cipher.decrypt(encrypted_data.encode())
        session_dict = json.loads(decrypted.decode())

        return SessionData(
            jti=session_dict["jti"],
            user_id=session_dict["user_id"],
            username=session_dict["username"],
            role=session_dict["role"],
            permissions=set(session_dict["permissions"]),
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_activity=datetime.fromisoformat(session_dict["last_activity"]),
            expires_at=datetime.fromisoformat(session_dict["expires_at"]),
            absolute_expires_at=datetime.fromisoformat(
                session_dict["absolute_expires_at"]
            ),
            ip_address=session_dict["ip_address"],
            user_agent=session_dict["user_agent"],
            metadata=session_dict["metadata"],
            status=SessionStatus(session_dict["status"]),
        )

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in Redis."""
        try:
            redis = await self._get_redis()
            key = f"{self.key_prefix}{session_data.jti}"
            user_key = f"{self.key_prefix}user:{session_data.user_id}"

            # Store session data
            serialized = self._serialize_session(session_data)

            # Calculate TTL in seconds
            ttl = int(
                (
                    session_data.absolute_expires_at - datetime.now(timezone.utc)
                ).total_seconds()
            )
            if ttl <= 0:
                return False

            # Store session with expiration
            await redis.setex(key, ttl, serialized)

            # Add to user sessions set
            await redis.sadd(user_key, session_data.jti)
            await redis.expire(user_key, ttl)

            return True
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti}: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data from Redis."""
        try:
            redis = await self._get_redis()
            key = f"{self.key_prefix}{jti}"
            encrypted_data = await redis.get(key)

            if not encrypted_data:
                return None

            return self._deserialize_session(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to retrieve session {jti}: {e}")
            return None

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data in Redis."""
        try:
            redis = await self._get_redis()
            key = f"{self.key_prefix}{jti}"

            # Check if session exists
            exists = await redis.exists(key)
            if not exists:
                return False

            # Update session
            return await self.store_session(session_data)
        except Exception as e:
            logger.error(f"Failed to update session {jti}: {e}")
            return False

    async def delete_session(self, jti: str) -> bool:
        """Delete session data from Redis."""
        try:
            redis = await self._get_redis()
            key = f"{self.key_prefix}{jti}"

            # Get session to find user_id
            session = await self.get_session(jti)
            if session:
                user_key = f"{self.key_prefix}user:{session.user_id}"
                await redis.srem(user_key, jti)

            # Delete session
            result = await redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete session {jti}: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user from Redis."""
        try:
            redis = await self._get_redis()
            user_key = f"{self.key_prefix}user:{user_id}"

            session_ids = await redis.smembers(user_key)
            sessions = []

            for jti in session_ids:
                session = await self.get_session(jti)
                if session:
                    sessions.append(session)

            return sessions
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions from Redis."""
        try:
            redis = await self._get_redis()
            pattern = f"{self.key_prefix}*"

            # Get all session keys (exclude user: keys)
            keys = []
            async for key in redis.scan_iter(match=pattern):
                if not key.startswith(f"{self.key_prefix}user:"):
                    keys.append(key)

            sessions = []
            for key in keys:
                jti = key.replace(self.key_prefix, "")
                session = await self.get_session(jti)
                if session:
                    sessions.append(session)

            return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions in Redis."""
        # Redis handles TTL automatically, but we clean up user mappings
        try:
            redis = await self._get_redis()
            count = 0

            # Get all user keys
            user_pattern = f"{self.key_prefix}user:*"
            async for user_key in redis.scan_iter(match=user_pattern):
                # Get all session IDs for this user
                session_ids = await redis.smembers(user_key)

                # Check which sessions still exist
                valid_sessions = []
                for jti in session_ids:
                    session_key = f"{self.key_prefix}{jti}"
                    if await redis.exists(session_key):
                        valid_sessions.append(jti)
                    else:
                        count += 1

                # Update user sessions set
                if valid_sessions != list(session_ids):
                    await redis.delete(user_key)
                    if valid_sessions:
                        await redis.sadd(user_key, *valid_sessions)

            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0


class DatabaseSessionStorage(SessionStorage):
    """Database-based session storage implementation."""

    def __init__(self, database_url: str = "sqlite:///sessions.db"):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None
        self._encryption_key = Fernet.generate_key()
        self._cipher = Fernet(self._encryption_key)

    async def _get_db_session(self):
        """Get database session with lazy initialization."""
        if self._engine is None:
            try:
                from sqlalchemy import Column, DateTime, MetaData, String, Table, Text
                from sqlalchemy.ext.asyncio import (
                    async_sessionmaker,
                    create_async_engine,
                )
                from sqlalchemy.ext.declarative import declarative_base

                # Modify database URL for async if needed
                if self.database_url.startswith("sqlite:"):
                    db_url = self.database_url.replace("sqlite:", "sqlite+aiosqlite:")
                elif self.database_url.startswith("postgresql:"):
                    db_url = self.database_url.replace(
                        "postgresql:", "postgresql+asyncpg:"
                    )
                else:
                    db_url = self.database_url

                self._engine = create_async_engine(db_url, echo=False)
                self._session_factory = async_sessionmaker(self._engine)

                # Create tables if they don't exist
                await self._create_tables()

            except ImportError as e:
                logger.error(f"Required database packages not installed: {e}")
                raise NotImplementedError(
                    "Database backend requires SQLAlchemy and async database driver"
                )
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise

        return self._session_factory()

    async def _create_tables(self):
        """Create session tables if they don't exist."""
        from sqlalchemy import text

        create_sessions_table = text(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                jti TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                encrypted_data TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        create_index = text(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
        """
        )

        create_expires_index = text(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)
        """
        )

        async with self._engine.begin() as conn:
            await conn.execute(create_sessions_table)
            await conn.execute(create_index)
            await conn.execute(create_expires_index)

    def _serialize_session(self, session_data: SessionData) -> str:
        """Serialize and encrypt session data."""
        session_dict = {
            "jti": session_data.jti,
            "user_id": session_data.user_id,
            "username": session_data.username,
            "role": session_data.role,
            "permissions": list(session_data.permissions),
            "created_at": session_data.created_at.isoformat(),
            "last_activity": session_data.last_activity.isoformat(),
            "expires_at": session_data.expires_at.isoformat(),
            "absolute_expires_at": session_data.absolute_expires_at.isoformat(),
            "ip_address": session_data.ip_address,
            "user_agent": session_data.user_agent,
            "metadata": session_data.metadata,
            "status": session_data.status.value,
        }
        json_str = json.dumps(session_dict)
        encrypted = self._cipher.encrypt(json_str.encode())
        return encrypted.decode()

    def _deserialize_session(self, encrypted_data: str) -> SessionData:
        """Decrypt and deserialize session data."""
        decrypted = self._cipher.decrypt(encrypted_data.encode())
        session_dict = json.loads(decrypted.decode())

        return SessionData(
            jti=session_dict["jti"],
            user_id=session_dict["user_id"],
            username=session_dict["username"],
            role=session_dict["role"],
            permissions=set(session_dict["permissions"]),
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            last_activity=datetime.fromisoformat(session_dict["last_activity"]),
            expires_at=datetime.fromisoformat(session_dict["expires_at"]),
            absolute_expires_at=datetime.fromisoformat(
                session_dict["absolute_expires_at"]
            ),
            ip_address=session_dict["ip_address"],
            user_agent=session_dict["user_agent"],
            metadata=session_dict["metadata"],
            status=SessionStatus(session_dict["status"]),
        )

    async def store_session(self, session_data: SessionData) -> bool:
        """Store session data in database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                encrypted_data = self._serialize_session(session_data)

                query = text(
                    """
                    INSERT OR REPLACE INTO sessions (jti, user_id, encrypted_data, expires_at)
                    VALUES (:jti, :user_id, :encrypted_data, :expires_at)
                """
                )

                await session.execute(
                    query,
                    {
                        "jti": session_data.jti,
                        "user_id": session_data.user_id,
                        "encrypted_data": encrypted_data,
                        "expires_at": session_data.absolute_expires_at,
                    },
                )
                await session.commit()

                return True
        except Exception as e:
            logger.error(f"Failed to store session {session_data.jti}: {e}")
            return False

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Retrieve session data from database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                query = text(
                    """
                    SELECT encrypted_data FROM sessions
                    WHERE jti = :jti AND expires_at > CURRENT_TIMESTAMP
                """
                )

                result = await session.execute(query, {"jti": jti})
                row = result.fetchone()

                if not row:
                    return None

                return self._deserialize_session(row[0])
        except Exception as e:
            logger.error(f"Failed to retrieve session {jti}: {e}")
            return None

    async def update_session(self, jti: str, session_data: SessionData) -> bool:
        """Update session data in database."""
        return await self.store_session(session_data)

    async def delete_session(self, jti: str) -> bool:
        """Delete session data from database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                query = text("DELETE FROM sessions WHERE jti = :jti")
                result = await session.execute(query, {"jti": jti})
                await session.commit()

                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete session {jti}: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user from database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                query = text(
                    """
                    SELECT encrypted_data FROM sessions
                    WHERE user_id = :user_id AND expires_at > CURRENT_TIMESTAMP
                """
                )

                result = await session.execute(query, {"user_id": user_id})
                sessions = []

                for row in result.fetchall():
                    try:
                        session_data = self._deserialize_session(row[0])
                        sessions.append(session_data)
                    except Exception as e:
                        logger.warning(
                            f"Failed to deserialize session for user {user_id}: {e}"
                        )

                return sessions
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []

    async def get_all_sessions(self) -> list[SessionData]:
        """Get all sessions from database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                query = text(
                    """
                    SELECT encrypted_data FROM sessions
                    WHERE expires_at > CURRENT_TIMESTAMP
                """
                )

                result = await session.execute(query)
                sessions = []

                for row in result.fetchall():
                    try:
                        session_data = self._deserialize_session(row[0])
                        sessions.append(session_data)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize session: {e}")

                return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from database."""
        try:
            from sqlalchemy import text

            async with await self._get_db_session() as session:
                query = text(
                    "DELETE FROM sessions WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                result = await session.execute(query)
                await session.commit()

                return result.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0


class EnhancedSessionManager:
    """Enhanced session manager with advanced features."""

    def __init__(
        self,
        policy: Optional[SessionPolicy] = None,
        storage_backend: SessionStorageBackend = SessionStorageBackend.MEMORY,
    ):
        self.policy = policy or SessionPolicy()
        self._storage = self._create_storage_backend(storage_backend)
        self._blacklisted_tokens: set[str] = set()
        self._analytics = SessionAnalytics()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_started = False

    def _create_storage_backend(self, backend: SessionStorageBackend) -> SessionStorage:
        """Create appropriate storage backend."""
        if backend == SessionStorageBackend.MEMORY:
            return MemorySessionStorage()
        elif backend == SessionStorageBackend.REDIS:
            try:
                return RedisSessionStorage()
            except NotImplementedError as e:
                logger.error(f"Redis backend not available: {e}")
                logger.warning("Falling back to memory storage")
                return MemorySessionStorage()
            except Exception as e:
                logger.error(f"Failed to initialize Redis backend: {e}")
                logger.warning("Falling back to memory storage")
                return MemorySessionStorage()
        elif backend == SessionStorageBackend.DATABASE:
            try:
                return DatabaseSessionStorage()
            except NotImplementedError as e:
                logger.error(f"Database backend not available: {e}")
                logger.warning("Falling back to memory storage")
                return MemorySessionStorage()
            except Exception as e:
                logger.error(f"Failed to initialize Database backend: {e}")
                logger.warning("Falling back to memory storage")
                return MemorySessionStorage()
        else:
            logger.warning(f"Unknown storage backend {backend}, using memory")
            return MemorySessionStorage()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_started:
            return

        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()

            async def cleanup_loop():
                while True:
                    try:
                        await asyncio.sleep(self.policy.cleanup_interval_minutes * 60)
                        await self.cleanup_expired_sessions()
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in cleanup task: {e}")

            # Only create task if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(cleanup_loop())
                self._cleanup_started = True
            except RuntimeError:
                # No running event loop, will start later when needed
                logger.debug(
                    "No running event loop, cleanup task will start when first used"
                )
        except Exception as e:
            logger.warning(f"Could not start cleanup task: {e}")

    def _calculate_session_timeouts(
        self, remember_me: bool = False
    ) -> tuple[datetime, datetime]:
        """Calculate session timeout timestamps."""
        now = datetime.now(timezone.utc)

        if remember_me:
            idle_timeout = now + timedelta(days=self.policy.remember_me_timeout_days)
            absolute_timeout = now + timedelta(
                days=self.policy.remember_me_timeout_days
            )
        else:
            idle_timeout = now + timedelta(minutes=self.policy.idle_timeout_minutes)
            absolute_timeout = now + timedelta(hours=self.policy.absolute_timeout_hours)

        return idle_timeout, absolute_timeout

    def _generate_device_fingerprint(self, user_agent: str, ip_address: str) -> str:
        """Generate a device fingerprint."""
        data = f"{user_agent}:{ip_address}:{secrets.token_hex(8)}"
        return sha256(data.encode()).hexdigest()[:16]

    async def create_session(
        self,
        user_id: str,
        jti: str,
        role: UserRole,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        remember_me: bool = False,
        session_type: str = "web",
    ) -> Optional[SessionData]:
        """Create a new session with enhanced features."""
        # Start cleanup task if not already started
        if not self._cleanup_started:
            self._start_cleanup_task()

        # Check concurrent session limits
        if await self._check_concurrent_session_limit(user_id):
            logger.warning(f"Concurrent session limit exceeded for user {user_id}")
            if self.policy.detailed_audit_logging:
                security_logger.log_event(
                    SecurityEvent(
                        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                        timestamp=datetime.now(timezone.utc),
                        user_id=user_id,
                        ip_address=ip_address,
                        details={"reason": "concurrent_session_limit_exceeded"},
                        success=False,
                    )
                )
            return None

        # Calculate timeouts
        expires_at, absolute_expires_at = self._calculate_session_timeouts(remember_me)

        # Create session data
        session_data = SessionData(
            jti=jti,
            user_id=user_id,
            role=role.value,
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            expires_at=expires_at,
            absolute_expires_at=absolute_expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=self._generate_device_fingerprint(
                user_agent or "", ip_address or ""
            ),
            remember_me=remember_me,
            is_mobile=self._detect_mobile_device(user_agent),
            session_type=session_type,
        )

        # Store session
        if await self._storage.store_session(session_data):
            await self._update_analytics()

            if self.policy.detailed_audit_logging:
                security_logger.log_event(
                    SecurityEvent(
                        event_type=SecurityEventType.LOGIN_SUCCESS,
                        timestamp=datetime.now(timezone.utc),
                        user_id=user_id,
                        ip_address=ip_address,
                        details={
                            "jti": jti,
                            "session_type": session_type,
                            "remember_me": remember_me,
                            "device_fingerprint": session_data.device_fingerprint,
                        },
                        success=True,
                    )
                )

            return session_data

        return None

    async def _check_concurrent_session_limit(self, user_id: str) -> bool:
        """Check if user has exceeded concurrent session limit."""
        user_sessions = await self._storage.get_user_sessions(user_id)
        active_sessions = [s for s in user_sessions if s.status == SessionStatus.ACTIVE]
        return len(active_sessions) >= self.policy.max_concurrent_sessions

    def _detect_mobile_device(self, user_agent: Optional[str]) -> bool:
        """Detect if request is from mobile device."""
        if not user_agent:
            return False

        mobile_indicators = [
            "Mobile",
            "Android",
            "iPhone",
            "iPad",
            "BlackBerry",
            "Windows Phone",
        ]
        return any(indicator in user_agent for indicator in mobile_indicators)

    async def get_session(self, jti: str) -> Optional[SessionData]:
        """Get session data."""
        return await self._storage.get_session(jti)

    async def update_session_activity(
        self,
        jti: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_path: Optional[str] = None,
    ) -> bool:
        """Update session activity with enhanced tracking."""
        session = await self._storage.get_session(jti)
        if not session:
            return False

        now = datetime.now(timezone.utc)

        # Check for suspicious activity
        if await self._detect_suspicious_activity(session, ip_address, user_agent):
            session.suspicious_activity_count += 1
            if session.suspicious_activity_count > 3:
                session.status = SessionStatus.LOCKED
                await self._storage.update_session(jti, session)
                return False

        # Update activity
        session.last_activity = now
        session.request_count += 1
        if request_path:
            session.last_request_path = request_path

        # Extend idle timeout
        if session.remember_me:
            session.expires_at = now + timedelta(
                days=self.policy.remember_me_timeout_days
            )
        else:
            session.expires_at = now + timedelta(
                minutes=self.policy.idle_timeout_minutes
            )

        return await self._storage.update_session(jti, session)

    async def _detect_suspicious_activity(
        self,
        session: SessionData,
        current_ip: Optional[str],
        current_user_agent: Optional[str],
    ) -> bool:
        """Detect suspicious session activity."""
        if not self.policy.enable_session_hijacking_protection:
            return False

        suspicious = False

        # IP address validation
        if (
            self.policy.require_ip_validation
            and not self.policy.allow_ip_change
            and current_ip
            and session.ip_address
            and current_ip != session.ip_address
        ):
            logger.warning(
                f"IP address change detected for session {session.jti}: {session.ip_address} -> {current_ip}"
            )
            suspicious = True

        # User agent validation
        if (
            self.policy.require_user_agent_validation
            and current_user_agent
            and session.user_agent
            and current_user_agent != session.user_agent
        ):
            logger.warning(f"User agent change detected for session {session.jti}")
            suspicious = True

        if suspicious and self.policy.detailed_audit_logging:
            security_logger.log_event(
                SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    timestamp=datetime.now(timezone.utc),
                    user_id=session.user_id,
                    ip_address=current_ip,
                    details={
                        "jti": session.jti,
                        "original_ip": session.ip_address,
                        "current_ip": current_ip,
                        "original_user_agent": session.user_agent,
                        "current_user_agent": current_user_agent,
                    },
                    success=False,
                )
            )

        return suspicious

    async def is_session_active(self, jti: str) -> bool:
        """Check if session is active and valid."""
        if jti in self._blacklisted_tokens:
            return False

        session = await self._storage.get_session(jti)
        if not session:
            return False

        if session.status != SessionStatus.ACTIVE:
            return False

        now = datetime.now(timezone.utc)

        # Check timeouts
        if session.expires_at <= now or session.absolute_expires_at <= now:
            session.status = SessionStatus.EXPIRED
            await self._storage.update_session(jti, session)
            return False

        return True

    async def invalidate_session(self, jti: str, reason: str = "manual") -> bool:
        """Invalidate a specific session."""
        session = await self._storage.get_session(jti)
        if not session:
            return False

        session.status = SessionStatus.INVALIDATED
        success = await self._storage.update_session(jti, session)

        if success and self.policy.detailed_audit_logging:
            security_logger.log_event(
                SecurityEvent(
                    event_type=SecurityEventType.LOGOUT,
                    timestamp=datetime.now(timezone.utc),
                    user_id=session.user_id,
                    details={
                        "jti": jti,
                        "reason": reason,
                        "session_duration": (
                            datetime.now(timezone.utc) - session.created_at
                        ).total_seconds(),
                    },
                    success=True,
                )
            )

        return success

    async def invalidate_user_sessions(
        self, user_id: str, exclude_jti: Optional[str] = None
    ) -> int:
        """Invalidate all sessions for a user."""
        user_sessions = await self._storage.get_user_sessions(user_id)
        count = 0

        for session in user_sessions:
            if session.jti != exclude_jti and session.status == SessionStatus.ACTIVE:
                if await self.invalidate_session(session.jti, "user_logout_all"):
                    count += 1

        return count

    def blacklist_token(self, jti: str) -> None:
        """Add token to blacklist."""
        self._blacklisted_tokens.add(jti)

    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        return jti in self._blacklisted_tokens

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed."""
        removed_count = await self._storage.cleanup_expired_sessions()

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired sessions")
            await self._update_analytics()

        return removed_count

    async def get_session_analytics(self) -> SessionAnalytics:
        """Get session analytics and metrics."""
        await self._update_analytics()
        return self._analytics

    async def _update_analytics(self):
        """Update analytics data."""
        all_sessions = await self._storage.get_all_sessions()

        self._analytics.total_sessions = len(all_sessions)
        self._analytics.active_sessions = len(
            [s for s in all_sessions if s.status == SessionStatus.ACTIVE]
        )
        self._analytics.expired_sessions = len(
            [s for s in all_sessions if s.status == SessionStatus.EXPIRED]
        )

        # Update concurrent sessions by user
        self._analytics.concurrent_sessions_by_user = {}
        for session in all_sessions:
            if session.status == SessionStatus.ACTIVE:
                user_id = session.user_id
                self._analytics.concurrent_sessions_by_user[user_id] = (
                    self._analytics.concurrent_sessions_by_user.get(user_id, 0) + 1
                )

        # Update sessions by role
        self._analytics.sessions_by_role = {}
        for session in all_sessions:
            if session.status == SessionStatus.ACTIVE:
                role = session.role
                self._analytics.sessions_by_role[role] = (
                    self._analytics.sessions_by_role.get(role, 0) + 1
                )

        # Calculate average session duration for completed sessions
        completed_sessions = [
            s
            for s in all_sessions
            if s.status in [SessionStatus.EXPIRED, SessionStatus.INVALIDATED]
        ]
        if completed_sessions:
            total_duration = sum(
                (s.last_activity - s.created_at).total_seconds()
                for s in completed_sessions
            )
            self._analytics.average_session_duration = total_duration / len(
                completed_sessions
            )

        # Update peak concurrent sessions
        current_active = self._analytics.active_sessions
        if current_active > self._analytics.peak_concurrent_sessions:
            self._analytics.peak_concurrent_sessions = current_active

        # Update suspicious activity count
        self._analytics.suspicious_activity_count = sum(
            s.suspicious_activity_count for s in all_sessions
        )

    async def get_user_sessions(self, user_id: str) -> list[SessionData]:
        """Get all sessions for a user."""
        return await self._storage.get_user_sessions(user_id)

    async def suspend_session(self, jti: str, reason: str = "security") -> bool:
        """Suspend a session temporarily."""
        session = await self._storage.get_session(jti)
        if not session:
            return False

        session.status = SessionStatus.SUSPENDED
        session.requires_reauth = True

        success = await self._storage.update_session(jti, session)

        if success and self.policy.detailed_audit_logging:
            security_logger.log_event(
                SecurityEvent(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    timestamp=datetime.now(timezone.utc),
                    user_id=session.user_id,
                    details={
                        "jti": jti,
                        "action": "suspended",
                        "reason": reason,
                    },
                    success=True,
                )
            )

        return success

    async def reactivate_session(self, jti: str) -> bool:
        """Reactivate a suspended session."""
        session = await self._storage.get_session(jti)
        if not session or session.status != SessionStatus.SUSPENDED:
            return False

        session.status = SessionStatus.ACTIVE
        session.requires_reauth = False
        session.last_activity = datetime.now(timezone.utc)

        return await self._storage.update_session(jti, session)

    def update_policy(self, new_policy: SessionPolicy):
        """Update session policy."""
        self.policy = new_policy
        # Restart cleanup task with new interval
        self._start_cleanup_task()

    def __del__(self):
        """Cleanup on destruction."""
        if self._cleanup_task:
            self._cleanup_task.cancel()


# Global enhanced session manager instance
enhanced_session_manager = EnhancedSessionManager()
