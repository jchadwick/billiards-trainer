"""Integration tests for the authentication system."""

import pytest
from sqlalchemy.orm import Session

from ..database.connection import setup_test_database
from ..services.auth_service import AuthenticationService
from ..utils.security import UserRole


@pytest.fixture()
def test_db():
    """Create test database."""
    db_manager = setup_test_database()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def auth_service(test_db: Session):
    """Create authentication service with test database."""
    return AuthenticationService(test_db)


@pytest.fixture()
def test_user(auth_service: AuthenticationService):
    """Create a test user."""
    user = auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="TestPassword123!",
        role=UserRole.VIEWER,
        first_name="Test",
        last_name="User",
        is_verified=True,
    )
    return user


class TestAuthenticationService:
    """Test authentication service functionality."""

    def test_create_user(self, auth_service: AuthenticationService):
        """Test user creation."""
        user = auth_service.create_user(
            username="newuser",
            email="newuser@example.com",
            password="NewPassword123!",
            role=UserRole.OPERATOR,
            first_name="New",
            last_name="User",
        )

        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.role == UserRole.OPERATOR
        assert user.first_name == "New"
        assert user.last_name == "User"
        assert user.is_active
        assert not user.is_verified  # Not verified by default
        assert user.verify_password("NewPassword123!")

    def test_duplicate_username(self, auth_service: AuthenticationService, test_user):
        """Test creating user with duplicate username."""
        with pytest.raises(Exception):  # Should raise HTTPException but we catch all
            auth_service.create_user(
                username=test_user.username,
                email="different@example.com",
                password="Password123!",
            )

    def test_duplicate_email(self, auth_service: AuthenticationService, test_user):
        """Test creating user with duplicate email."""
        with pytest.raises(Exception):  # Should raise HTTPException but we catch all
            auth_service.create_user(
                username="differentuser",
                email=test_user.email,
                password="Password123!",
            )

    def test_authenticate_user(self, auth_service: AuthenticationService, test_user):
        """Test user authentication."""
        # Test successful authentication
        user = auth_service.authenticate_user(
            username=test_user.username,
            password="TestPassword123!",
        )
        assert user is not None
        assert user.id == test_user.id

        # Test authentication with email
        user = auth_service.authenticate_user(
            username=test_user.email,
            password="TestPassword123!",
        )
        assert user is not None
        assert user.id == test_user.id

        # Test failed authentication
        user = auth_service.authenticate_user(
            username=test_user.username,
            password="WrongPassword",
        )
        assert user is None

    def test_update_user(self, auth_service: AuthenticationService, test_user):
        """Test user updates."""
        updated_user = auth_service.update_user(
            user_id=str(test_user.id),
            first_name="Updated",
            last_name="Name",
            email="updated@example.com",
        )

        assert updated_user.first_name == "Updated"
        assert updated_user.last_name == "Name"
        assert updated_user.email == "updated@example.com"
        assert updated_user.updated_at > test_user.updated_at

    def test_list_users(self, auth_service: AuthenticationService, test_user):
        """Test listing users."""
        # Create additional users
        auth_service.create_user(
            username="admin_user",
            email="admin@example.com",
            password="AdminPassword123!",
            role=UserRole.ADMIN,
        )

        auth_service.create_user(
            username="operator_user",
            email="operator@example.com",
            password="OperatorPassword123!",
            role=UserRole.OPERATOR,
        )

        # Test listing all users
        users, total = auth_service.list_users(page=1, per_page=10)
        assert total >= 3  # At least our test users
        assert len(users) >= 3

        # Test filtering by role
        admin_users, admin_total = auth_service.list_users(role=UserRole.ADMIN)
        assert admin_total >= 1
        assert all(user.role == UserRole.ADMIN for user in admin_users)

        # Test search functionality
        search_users, search_total = auth_service.list_users(search="test")
        assert search_total >= 1
        assert any("test" in user.username.lower() for user in search_users)

    def test_session_management(self, auth_service: AuthenticationService, test_user):
        """Test session creation and management."""
        # Create session
        session = auth_service.create_session(
            user=test_user,
            jti="test_jti_123",
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0",
        )

        assert session.user_id == test_user.id
        assert session.jti == "test_jti_123"
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "TestAgent/1.0"
        assert session.is_active

        # Validate session
        validated_session = auth_service.validate_session("test_jti_123")
        assert validated_session is not None
        assert validated_session.id == session.id

        # Get user sessions
        user_sessions = auth_service.get_user_sessions(str(test_user.id))
        assert len(user_sessions) == 1
        assert user_sessions[0].id == session.id

        # Invalidate session
        success = auth_service.invalidate_session("test_jti_123")
        assert success

        # Session should no longer be valid
        validated_session = auth_service.validate_session("test_jti_123")
        assert validated_session is None

    def test_api_key_management(self, auth_service: AuthenticationService, test_user):
        """Test API key creation and management."""
        # Create API key
        api_key_record, api_key = auth_service.create_api_key(
            user_id=str(test_user.id),
            name="Test API Key",
            role=UserRole.VIEWER,
            expires_days=30,
        )

        assert api_key_record.user_id == test_user.id
        assert api_key_record.name == "Test API Key"
        assert api_key_record.role == UserRole.VIEWER
        assert api_key_record.is_active
        assert api_key.startswith("bt_")  # Default prefix

        # Authenticate with API key
        authenticated_key = auth_service.authenticate_api_key(api_key)
        assert authenticated_key is not None
        assert authenticated_key.id == api_key_record.id

        # List API keys
        api_keys, total = auth_service.list_api_keys(user_id=str(test_user.id))
        assert total == 1
        assert api_keys[0].id == api_key_record.id

        # Revoke API key
        success = auth_service.revoke_api_key(
            api_key_id=str(api_key_record.id),
            revoked_by_user_id=str(test_user.id),
        )
        assert success

        # API key should no longer authenticate
        authenticated_key = auth_service.authenticate_api_key(api_key)
        assert authenticated_key is None

    def test_password_reset(self, auth_service: AuthenticationService, test_user):
        """Test password reset functionality."""
        # Request password reset
        token = auth_service.request_password_reset(
            email=test_user.email,
            ip_address="127.0.0.1",
        )
        assert token is not None

        # Reset password with token
        success = auth_service.reset_password(
            token=token,
            new_password="NewTestPassword123!",
            ip_address="127.0.0.1",
        )
        assert success

        # Old password should not work
        user = auth_service.authenticate_user(
            username=test_user.username,
            password="TestPassword123!",
        )
        assert user is None

        # New password should work
        user = auth_service.authenticate_user(
            username=test_user.username,
            password="NewTestPassword123!",
        )
        assert user is not None

        # Token should not work again
        success = auth_service.reset_password(
            token=token,
            new_password="AnotherPassword123!",
        )
        assert not success

    def test_security_event_logging(self, auth_service: AuthenticationService):
        """Test security event logging."""
        # Log a security event
        auth_service.log_security_event(
            event_type="test_event",
            success=True,
            user_id="test_user_id",
            ip_address="127.0.0.1",
            details={"test": "data"},
        )

        # Get security events
        events, total = auth_service.get_security_events(page=1, per_page=10)
        assert total >= 1

        # Find our test event
        test_event = next(
            (event for event in events if event.event_type == "test_event"), None
        )
        assert test_event is not None
        assert test_event.success
        assert test_event.user_id == "test_user_id"
        assert test_event.ip_address == "127.0.0.1"
        assert test_event.details["test"] == "data"

    def test_cleanup_expired_data(self, auth_service: AuthenticationService):
        """Test cleanup of expired data."""
        results = auth_service.cleanup_expired_data()
        assert isinstance(results, dict)
        assert "expired_sessions" in results
        assert "expired_reset_tokens" in results
        assert "old_security_events" in results

    def test_auth_status(self, auth_service: AuthenticationService):
        """Test authentication status."""
        status = auth_service.get_auth_status()
        assert isinstance(status, dict)
        assert status["service"] == "authentication"
        assert "database_connected" in status
        assert "total_users" in status
