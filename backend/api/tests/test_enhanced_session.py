"""Comprehensive tests for enhanced session management system."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from ..middleware.enhanced_auth_integration import EnhancedAuthenticationManager
from ..middleware.enhanced_session import (
    EnhancedSessionManager,
    MemorySessionStorage,
    SessionData,
    SessionPolicy,
    SessionStatus,
)
from ..middleware.session_monitor import MonitoringLevel, SessionEvent, SessionMonitor
from ..middleware.session_security import SecurityThreatLevel, SessionSecurityValidator
from ..utils.security import UserRole


class TestSessionData:
    """Test SessionData class."""

    def test_session_data_creation(self):
        """Test creating session data."""
        now = datetime.now(timezone.utc)
        session = SessionData(
            jti="test_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert session.jti == "test_jti"
        assert session.user_id == "test_user"
        assert session.role == "viewer"
        assert session.status == SessionStatus.ACTIVE
        assert session.ip_address == "192.168.1.1"

    def test_session_data_serialization(self):
        """Test session data serialization and deserialization."""
        now = datetime.now(timezone.utc)
        original = SessionData(
            jti="test_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
        )

        # Convert to dict and back
        data_dict = original.to_dict()
        restored = SessionData.from_dict(data_dict)

        assert restored.jti == original.jti
        assert restored.user_id == original.user_id
        assert restored.status == original.status
        assert restored.created_at == original.created_at


class TestSessionPolicy:
    """Test SessionPolicy configuration."""

    def test_default_policy(self):
        """Test default session policy."""
        policy = SessionPolicy()

        assert policy.idle_timeout_minutes == 30
        assert policy.absolute_timeout_hours == 8
        assert policy.max_concurrent_sessions == 5
        assert policy.require_ip_validation is True
        assert policy.enable_session_hijacking_protection is True

    def test_custom_policy(self):
        """Test custom session policy."""
        policy = SessionPolicy(
            idle_timeout_minutes=60,
            max_concurrent_sessions=10,
            require_ip_validation=False,
        )

        assert policy.idle_timeout_minutes == 60
        assert policy.max_concurrent_sessions == 10
        assert policy.require_ip_validation is False


class TestMemorySessionStorage:
    """Test memory session storage implementation."""

    @pytest.fixture()
    def storage(self):
        """Create storage instance for testing."""
        return MemorySessionStorage()

    @pytest.fixture()
    def sample_session(self):
        """Create sample session data."""
        now = datetime.now(timezone.utc)
        return SessionData(
            jti="test_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
        )

    async def test_store_and_retrieve_session(self, storage, sample_session):
        """Test storing and retrieving a session."""
        # Store session
        success = await storage.store_session(sample_session)
        assert success is True

        # Retrieve session
        retrieved = await storage.get_session(sample_session.jti)
        assert retrieved is not None
        assert retrieved.jti == sample_session.jti
        assert retrieved.user_id == sample_session.user_id

    async def test_update_session(self, storage, sample_session):
        """Test updating session data."""
        # Store initial session
        await storage.store_session(sample_session)

        # Update session
        sample_session.request_count = 10
        success = await storage.update_session(sample_session.jti, sample_session)
        assert success is True

        # Verify update
        retrieved = await storage.get_session(sample_session.jti)
        assert retrieved.request_count == 10

    async def test_delete_session(self, storage, sample_session):
        """Test deleting a session."""
        # Store session
        await storage.store_session(sample_session)

        # Delete session
        success = await storage.delete_session(sample_session.jti)
        assert success is True

        # Verify deletion
        retrieved = await storage.get_session(sample_session.jti)
        assert retrieved is None

    async def test_get_user_sessions(self, storage):
        """Test getting all sessions for a user."""
        # Create multiple sessions for same user
        now = datetime.now(timezone.utc)
        sessions = []
        for i in range(3):
            session = SessionData(
                jti=f"jti_{i}",
                user_id="test_user",
                role="viewer",
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_activity=now,
                expires_at=now + timedelta(minutes=30),
                absolute_expires_at=now + timedelta(hours=8),
            )
            sessions.append(session)
            await storage.store_session(session)

        # Get user sessions
        user_sessions = await storage.get_user_sessions("test_user")
        assert len(user_sessions) == 3

        # Verify all sessions belong to user
        for session in user_sessions:
            assert session.user_id == "test_user"

    async def test_cleanup_expired_sessions(self, storage):
        """Test cleaning up expired sessions."""
        now = datetime.now(timezone.utc)

        # Create expired session
        expired_session = SessionData(
            jti="expired_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.EXPIRED,
            created_at=now - timedelta(hours=2),
            last_activity=now - timedelta(hours=1),
            expires_at=now - timedelta(minutes=30),
            absolute_expires_at=now - timedelta(hours=1),
        )
        await storage.store_session(expired_session)

        # Create active session
        active_session = SessionData(
            jti="active_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
        )
        await storage.store_session(active_session)

        # Cleanup expired sessions
        removed_count = await storage.cleanup_expired_sessions()
        assert removed_count == 1

        # Verify only active session remains
        expired_retrieved = await storage.get_session("expired_jti")
        assert expired_retrieved is None

        active_retrieved = await storage.get_session("active_jti")
        assert active_retrieved is not None


class TestEnhancedSessionManager:
    """Test enhanced session manager."""

    @pytest.fixture()
    def session_manager(self):
        """Create session manager for testing."""
        policy = SessionPolicy(
            idle_timeout_minutes=30,
            absolute_timeout_hours=8,
            max_concurrent_sessions=3,
        )
        return EnhancedSessionManager(policy=policy)

    async def test_create_session(self, session_manager):
        """Test creating a session."""
        session = await session_manager.create_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert session is not None
        assert session.user_id == "test_user"
        assert session.jti == "test_jti"
        assert session.role == UserRole.VIEWER.value
        assert session.status == SessionStatus.ACTIVE

    async def test_concurrent_session_limit(self, session_manager):
        """Test concurrent session limit enforcement."""
        # Create sessions up to limit
        for i in range(3):
            session = await session_manager.create_session(
                user_id="test_user",
                jti=f"jti_{i}",
                role=UserRole.VIEWER,
            )
            assert session is not None

        # Try to create one more (should fail)
        session = await session_manager.create_session(
            user_id="test_user",
            jti="jti_4",
            role=UserRole.VIEWER,
        )
        assert session is None

    async def test_session_activity_update(self, session_manager):
        """Test updating session activity."""
        # Create session
        session = await session_manager.create_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
        )

        original_activity = session.last_activity

        # Update activity
        await asyncio.sleep(0.1)  # Small delay to ensure timestamp difference
        success = await session_manager.update_session_activity(
            "test_jti", request_path="/api/test"
        )

        assert success is True

        # Verify activity was updated
        updated_session = await session_manager.get_session("test_jti")
        assert updated_session.last_activity > original_activity
        assert updated_session.last_request_path == "/api/test"
        assert updated_session.request_count == 1

    async def test_session_invalidation(self, session_manager):
        """Test invalidating a session."""
        # Create session
        await session_manager.create_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
        )

        # Verify session is active
        assert await session_manager.is_session_active("test_jti") is True

        # Invalidate session
        success = await session_manager.invalidate_session("test_jti")
        assert success is True

        # Verify session is no longer active
        assert await session_manager.is_session_active("test_jti") is False

        # Verify session status is updated
        invalidated_session = await session_manager.get_session("test_jti")
        assert invalidated_session.status == SessionStatus.INVALIDATED

    async def test_session_timeout(self, session_manager):
        """Test session timeout behavior."""
        # Create session with very short timeout
        policy = SessionPolicy(idle_timeout_minutes=0.01)  # Very short timeout
        session_manager.policy = policy

        await session_manager.create_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
        )

        # Initially active
        assert await session_manager.is_session_active("test_jti") is True

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Should now be expired
        assert await session_manager.is_session_active("test_jti") is False

    async def test_invalidate_user_sessions(self, session_manager):
        """Test invalidating all sessions for a user."""
        # Create multiple sessions for user
        sessions = []
        for i in range(3):
            session = await session_manager.create_session(
                user_id="test_user",
                jti=f"jti_{i}",
                role=UserRole.VIEWER,
            )
            sessions.append(session)

        # Invalidate all user sessions
        count = await session_manager.invalidate_user_sessions("test_user")
        assert count == 3

        # Verify all sessions are invalidated
        for session in sessions:
            assert await session_manager.is_session_active(session.jti) is False


class TestSessionMonitor:
    """Test session monitoring system."""

    @pytest.fixture()
    def monitor(self):
        """Create session monitor for testing."""
        return SessionMonitor(MonitoringLevel.DETAILED)

    async def test_record_session_created(self, monitor):
        """Test recording session creation event."""
        now = datetime.now(timezone.utc)
        session_data = SessionData(
            jti="test_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        await monitor.record_session_created(session_data)

        # Verify event was recorded
        assert len(monitor._events) > 0
        latest_event = monitor._events[-1]
        assert latest_event.event_type == "session_created"
        assert latest_event.user_id == "test_user"
        assert latest_event.session_id == "test_jti"

    async def test_get_current_metrics(self, monitor):
        """Test getting current session metrics."""
        # Create sample sessions
        now = datetime.now(timezone.utc)
        sessions = []
        for i in range(5):
            session = SessionData(
                jti=f"jti_{i}",
                user_id=f"user_{i}",
                role="viewer",
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_activity=now,
                expires_at=now + timedelta(minutes=30),
                absolute_expires_at=now + timedelta(hours=8),
                is_mobile=(i % 2 == 0),
                user_agent=(
                    "Mozilla/5.0 Chrome" if i % 2 == 0 else "Mozilla/5.0 Firefox"
                ),
            )
            sessions.append(session)

        # Get metrics
        metrics = await monitor.get_current_metrics(sessions)

        assert metrics.active_sessions == 5
        assert "Chrome" in metrics.sessions_by_browser
        assert "Firefox" in metrics.sessions_by_browser

    async def test_user_activity_summary(self, monitor):
        """Test getting user activity summary."""
        # Record some events for a user
        await monitor.record_event(
            SessionEvent(
                timestamp=datetime.now(timezone.utc),
                event_type="session_created",
                session_id="test_jti",
                user_id="test_user",
                ip_address="192.168.1.1",
            )
        )

        # Get activity summary
        summary = await monitor.get_user_activity_summary("test_user", days=7)

        assert summary["user_id"] == "test_user"
        assert summary["total_events"] >= 1
        assert "session_created" in summary["events_by_type"]


class TestSessionSecurityValidator:
    """Test session security validation."""

    @pytest.fixture()
    def validator(self):
        """Create security validator for testing."""
        return SessionSecurityValidator()

    @pytest.fixture()
    def sample_session(self):
        """Create sample session for testing."""
        now = datetime.now(timezone.utc)
        return SessionData(
            jti="test_jti",
            user_id="test_user",
            role="viewer",
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

    async def test_validate_same_ip(self, validator, sample_session):
        """Test validation with same IP address."""
        result = await validator.validate_session_security(
            sample_session,
            "192.168.1.1",  # Same IP
            "Mozilla/5.0",  # Same user agent
        )

        assert result.is_valid is True
        assert result.threat_level == SecurityThreatLevel.NONE

    async def test_validate_ip_change(self, validator, sample_session):
        """Test validation with IP address change."""
        result = await validator.validate_session_security(
            sample_session,
            "192.168.2.1",  # Different IP
            "Mozilla/5.0",  # Same user agent
        )

        # Should detect IP change as suspicious
        assert result.is_valid is False
        assert any("IP address changed" in reason for reason in result.reasons)

    async def test_validate_user_agent_change(self, validator, sample_session):
        """Test validation with user agent change."""
        result = await validator.validate_session_security(
            sample_session,
            "192.168.1.1",  # Same IP
            "Chrome/90.0",  # Different user agent
        )

        # Should detect user agent change as suspicious
        assert result.is_valid is False
        assert any("User agent changed" in reason for reason in result.reasons)

    async def test_concurrent_session_limits(self, validator):
        """Test concurrent session limit checking."""
        # Create multiple sessions for same user
        sessions = []
        for i in range(10):  # More than the default limit
            session = SessionData(
                jti=f"jti_{i}",
                user_id="test_user",
                role="viewer",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
                absolute_expires_at=datetime.now(timezone.utc) + timedelta(hours=8),
            )
            sessions.append(session)

        result = await validator.check_concurrent_session_limits(
            "test_user", "192.168.1.1", sessions
        )

        assert result.is_valid is False
        assert any("concurrent sessions" in reason for reason in result.reasons)


class TestEnhancedAuthenticationManager:
    """Test enhanced authentication manager integration."""

    @pytest.fixture()
    def auth_manager(self):
        """Create enhanced authentication manager for testing."""
        policy = SessionPolicy(
            idle_timeout_minutes=30,
            max_concurrent_sessions=3,
            require_ip_validation=True,
            enable_session_hijacking_protection=True,
        )
        return EnhancedAuthenticationManager(policy=policy)

    @pytest.fixture()
    def mock_request(self):
        """Create mock request object."""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "Mozilla/5.0"}
        return request

    async def test_create_enhanced_session(self, auth_manager, mock_request):
        """Test creating enhanced session through manager."""
        session = await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            request=mock_request,
        )

        assert session is not None
        assert session.user_id == "test_user"
        assert session.ip_address == "192.168.1.1"

    async def test_validate_session_request(self, auth_manager, mock_request):
        """Test validating session requests."""
        # Create session first
        await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            request=mock_request,
        )

        # Validate request
        is_valid = await auth_manager.validate_session_request(
            "test_jti", mock_request, "/api/test"
        )

        assert is_valid is True

    async def test_get_session_info(self, auth_manager, mock_request):
        """Test getting comprehensive session information."""
        # Create session
        await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            request=mock_request,
        )

        # Get session info
        info = await auth_manager.get_session_info("test_jti")

        assert info is not None
        assert info["jti"] == "test_jti"
        assert info["user_id"] == "test_user"
        assert "created_at" in info
        assert "last_activity" in info

    async def test_get_system_metrics(self, auth_manager):
        """Test getting comprehensive system metrics."""
        metrics = await auth_manager.get_system_metrics()

        assert "session_analytics" in metrics
        assert "current_metrics" in metrics
        assert "security_alerts" in metrics
        assert "policy_settings" in metrics

    async def test_update_session_policy(self, auth_manager):
        """Test updating session policy."""
        success = await auth_manager.update_session_policy(
            {
                "idle_timeout_minutes": 60,
                "max_concurrent_sessions": 10,
            }
        )

        assert success is True
        assert auth_manager.policy.idle_timeout_minutes == 60
        assert auth_manager.policy.max_concurrent_sessions == 10


# Integration tests
class TestSessionManagementIntegration:
    """Integration tests for the complete session management system."""

    @pytest.fixture()
    def auth_manager(self):
        """Create configured authentication manager."""
        policy = SessionPolicy(
            idle_timeout_minutes=5,  # Short timeout for testing
            max_concurrent_sessions=2,
            require_ip_validation=True,
        )
        return EnhancedAuthenticationManager(policy=policy)

    @pytest.fixture()
    def mock_request(self):
        """Create mock request."""
        request = Mock()
        request.client.host = "192.168.1.1"
        request.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "accept-language": "en-US,en;q=0.9",
        }
        return request

    async def test_complete_session_lifecycle(self, auth_manager, mock_request):
        """Test complete session lifecycle."""
        # 1. Create session
        session = await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            request=mock_request,
        )
        assert session is not None

        # 2. Validate session requests
        is_valid = await auth_manager.validate_session_request(
            "test_jti", mock_request, "/api/endpoint1"
        )
        assert is_valid is True

        # 3. Update activity
        is_valid = await auth_manager.validate_session_request(
            "test_jti", mock_request, "/api/endpoint2"
        )
        assert is_valid is True

        # 4. Get session info
        info = await auth_manager.get_session_info("test_jti")
        assert info["request_count"] > 0

        # 5. Invalidate session
        success = await auth_manager.invalidate_session("test_jti")
        assert success is True

        # 6. Verify session is invalid
        is_valid = await auth_manager.validate_session_request(
            "test_jti", mock_request, "/api/endpoint3"
        )
        assert is_valid is False

    async def test_security_threat_detection(self, auth_manager, mock_request):
        """Test security threat detection and response."""
        # Create session
        await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="test_jti",
            role=UserRole.VIEWER,
            request=mock_request,
        )

        # Simulate suspicious request (different IP)
        suspicious_request = Mock()
        suspicious_request.client.host = "10.0.0.1"  # Different IP
        suspicious_request.headers = mock_request.headers

        # Should detect IP change as suspicious
        await auth_manager.validate_session_request(
            "test_jti", suspicious_request, "/api/suspicious"
        )

        # Session might be suspended or flagged
        session_info = await auth_manager.get_session_info("test_jti")
        assert session_info["suspicious_activity_count"] > 0

    async def test_concurrent_session_management(self, auth_manager, mock_request):
        """Test concurrent session limit enforcement."""
        # Create sessions up to limit
        sessions = []
        for i in range(2):  # Policy limit is 2
            session = await auth_manager.create_enhanced_session(
                user_id="test_user",
                jti=f"jti_{i}",
                role=UserRole.VIEWER,
                request=mock_request,
            )
            sessions.append(session)
            assert session is not None

        # Try to create one more (should handle gracefully)
        await auth_manager.create_enhanced_session(
            user_id="test_user",
            jti="jti_extra",
            role=UserRole.VIEWER,
            request=mock_request,
        )

        # Should either create session (after terminating oldest) or handle limit
        user_sessions = await auth_manager.get_user_sessions_info("test_user")
        assert len([s for s in user_sessions if s["status"] == "active"]) <= 2


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])
