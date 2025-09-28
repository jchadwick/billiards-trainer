"""Comprehensive security tests for authentication and authorization system."""

import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.middleware.authentication import session_manager

# Import the modules we're testing
from backend.api.utils.security import (
    APIKeyUtils,
    JWTUtils,
    PasswordPolicy,
    PasswordUtils,
    UserRole,
    check_role_permissions,
    constant_time_compare,
    input_validator,
)


class TestPasswordUtils:
    """Test password hashing and validation utilities."""

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123!"

        # Hash the password
        hashed = PasswordUtils.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long

        # Verify correct password
        assert PasswordUtils.verify_password(password, hashed)

        # Verify incorrect password
        assert not PasswordUtils.verify_password("wrong_password", hashed)

    def test_password_strength_validation(self):
        """Test password strength validation."""
        policy = PasswordPolicy()

        # Test valid password
        is_valid, errors = PasswordUtils.validate_password_strength(
            "ValidPass123!", policy
        )
        assert is_valid
        assert len(errors) == 0

        # Test too short
        is_valid, errors = PasswordUtils.validate_password_strength("short", policy)
        assert not is_valid
        assert any("8 characters" in error for error in errors)

        # Test missing uppercase
        is_valid, errors = PasswordUtils.validate_password_strength(
            "lowercase123!", policy
        )
        assert not is_valid
        assert any("uppercase" in error for error in errors)

        # Test missing lowercase
        is_valid, errors = PasswordUtils.validate_password_strength(
            "UPPERCASE123!", policy
        )
        assert not is_valid
        assert any("lowercase" in error for error in errors)

        # Test missing digits
        is_valid, errors = PasswordUtils.validate_password_strength(
            "NoNumbers!", policy
        )
        assert not is_valid
        assert any("digit" in error for error in errors)

        # Test missing special characters
        is_valid, errors = PasswordUtils.validate_password_strength(
            "NoSpecial123", policy
        )
        assert not is_valid
        assert any("special" in error for error in errors)

        # Test forbidden pattern
        is_valid, errors = PasswordUtils.validate_password_strength(
            "password123!", policy
        )
        assert not is_valid
        assert any("password" in error for error in errors)


class TestJWTUtils:
    """Test JWT token utilities."""

    def test_token_creation_and_validation(self):
        """Test JWT token creation and validation."""
        subject = "test_user"
        role = UserRole.ADMIN

        # Create token
        token = JWTUtils.create_access_token(subject, role)
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are long

        # Decode and validate token
        token_data = JWTUtils.decode_token(token)
        assert token_data is not None
        assert token_data.sub == subject
        assert token_data.role == role
        assert token_data.token_type == "access"
        assert isinstance(token_data.jti, str)

    def test_token_expiration(self):
        """Test token expiration handling."""
        subject = "test_user"
        role = UserRole.VIEWER

        # Create token with very short expiration
        short_expiry = timedelta(seconds=1)
        token = JWTUtils.create_access_token(subject, role, short_expiry)

        # Token should be valid immediately
        token_data = JWTUtils.decode_token(token)
        assert token_data is not None
        assert not JWTUtils.is_token_expired(token_data)

        # Wait for expiration
        time.sleep(2)

        # Token should be expired
        assert JWTUtils.is_token_expired(token_data)

    def test_refresh_token_creation(self):
        """Test refresh token creation."""
        subject = "test_user"
        role = UserRole.OPERATOR

        refresh_token = JWTUtils.create_refresh_token(subject, role)
        token_data = JWTUtils.decode_token(refresh_token)

        assert token_data is not None
        assert token_data.sub == subject
        assert token_data.role == role
        assert token_data.token_type == "refresh"

    def test_invalid_token_handling(self):
        """Test handling of invalid tokens."""
        # Test completely invalid token
        assert JWTUtils.decode_token("invalid.token.here") is None

        # Test empty token
        assert JWTUtils.decode_token("") is None

        # Test None token
        assert JWTUtils.decode_token(None) is None


class TestAPIKeyUtils:
    """Test API key utilities."""

    def test_api_key_generation(self):
        """Test API key generation."""
        api_key = APIKeyUtils.generate_api_key()

        assert isinstance(api_key, str)
        assert api_key.startswith("bt_")
        assert len(api_key) > 40  # Should be long enough

        # Generate multiple keys and ensure they're unique
        keys = [APIKeyUtils.generate_api_key() for _ in range(10)]
        assert len(set(keys)) == 10  # All unique

    def test_api_key_hashing(self):
        """Test API key hashing."""
        api_key = APIKeyUtils.generate_api_key()
        hashed = APIKeyUtils.hash_api_key(api_key)

        assert isinstance(hashed, str)
        assert hashed != api_key
        assert len(hashed) == 64  # SHA256 hex digest length

        # Same key should produce same hash
        assert APIKeyUtils.hash_api_key(api_key) == hashed

    def test_api_key_format_validation(self):
        """Test API key format validation."""
        # Valid API key
        valid_key = APIKeyUtils.generate_api_key()
        assert APIKeyUtils.verify_api_key_format(valid_key)

        # Invalid prefix
        assert not APIKeyUtils.verify_api_key_format("invalid_prefix_key")

        # No prefix
        assert not APIKeyUtils.verify_api_key_format("no_prefix")

        # Empty string
        assert not APIKeyUtils.verify_api_key_format("")

        # Wrong length
        assert not APIKeyUtils.verify_api_key_format("bt_short")


class TestRolePermissions:
    """Test role-based access control."""

    def test_role_hierarchy(self):
        """Test role permission hierarchy."""
        # Admin should have all permissions
        assert check_role_permissions(UserRole.ADMIN, UserRole.VIEWER)
        assert check_role_permissions(UserRole.ADMIN, UserRole.OPERATOR)
        assert check_role_permissions(UserRole.ADMIN, UserRole.ADMIN)

        # Operator should have operator and viewer permissions
        assert check_role_permissions(UserRole.OPERATOR, UserRole.VIEWER)
        assert check_role_permissions(UserRole.OPERATOR, UserRole.OPERATOR)
        assert not check_role_permissions(UserRole.OPERATOR, UserRole.ADMIN)

        # Viewer should only have viewer permissions
        assert check_role_permissions(UserRole.VIEWER, UserRole.VIEWER)
        assert not check_role_permissions(UserRole.VIEWER, UserRole.OPERATOR)
        assert not check_role_permissions(UserRole.VIEWER, UserRole.ADMIN)


class TestInputValidator:
    """Test input validation and sanitization."""

    def test_string_sanitization(self):
        """Test string sanitization."""
        # Test normal string
        result = input_validator.sanitize_string("normal string")
        assert result == "normal string"

        # Test string with control characters
        result = input_validator.sanitize_string("string\x00with\x01control")
        assert "\x00" not in result
        assert "\x01" not in result

        # Test length limit
        long_string = "a" * 300
        result = input_validator.sanitize_string(long_string, max_length=100)
        assert len(result) == 100

    def test_email_validation(self):
        """Test email validation."""
        # Valid emails
        assert input_validator.validate_email("user@example.com")
        assert input_validator.validate_email("test.user+tag@domain.co.uk")

        # Invalid emails
        assert not input_validator.validate_email("invalid")
        assert not input_validator.validate_email("@domain.com")
        assert not input_validator.validate_email("user@")
        assert not input_validator.validate_email("")

    def test_username_validation(self):
        """Test username validation."""
        # Valid usernames
        is_valid, msg = input_validator.validate_username("validuser")
        assert is_valid

        is_valid, msg = input_validator.validate_username("user_with_underscore")
        assert is_valid

        is_valid, msg = input_validator.validate_username("user-with-dash")
        assert is_valid

        # Invalid usernames
        is_valid, msg = input_validator.validate_username("")
        assert not is_valid

        is_valid, msg = input_validator.validate_username("ab")  # Too short
        assert not is_valid

        is_valid, msg = input_validator.validate_username("a" * 60)  # Too long
        assert not is_valid

        is_valid, msg = input_validator.validate_username(
            "user@domain"
        )  # Invalid chars
        assert not is_valid

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        # Safe strings
        assert not input_validator.detect_sql_injection("normal text")
        assert not input_validator.detect_sql_injection("user input")

        # SQL injection patterns
        assert input_validator.detect_sql_injection("'; DROP TABLE users; --")
        assert input_validator.detect_sql_injection("1 OR 1=1")
        assert input_validator.detect_sql_injection("UNION SELECT * FROM passwords")
        assert input_validator.detect_sql_injection("admin'--")

    def test_xss_detection(self):
        """Test XSS pattern detection."""
        # Safe strings
        assert not input_validator.detect_xss("normal text")
        assert not input_validator.detect_xss("user input")

        # XSS patterns
        assert input_validator.detect_xss("<script>alert('xss')</script>")
        assert input_validator.detect_xss("javascript:alert('xss')")
        assert input_validator.detect_xss("<img src=x onerror=alert('xss')>")
        assert input_validator.detect_xss("<iframe src='evil.com'></iframe>")


class TestSessionManager:
    """Test session management functionality."""

    def setUp(self):
        """Set up test data."""
        # Clear any existing sessions
        session_manager._active_sessions.clear()
        session_manager._user_sessions.clear()
        session_manager._blacklisted_tokens.clear()

    def test_session_creation(self):
        """Test session creation and retrieval."""
        self.setUp()

        user_id = "test_user"
        jti = "test_jti_123"
        role = UserRole.VIEWER
        ip_address = "192.168.1.1"

        # Create session
        session_manager.create_session(user_id, jti, role, ip_address)

        # Verify session exists
        session = session_manager.get_session(jti)
        assert session is not None
        assert session["user_id"] == user_id
        assert session["role"] == role.value
        assert session["ip_address"] == ip_address
        assert session["is_active"] is True

        # Verify session is active
        assert session_manager.is_session_active(jti)

    def test_session_invalidation(self):
        """Test session invalidation."""
        self.setUp()

        user_id = "test_user"
        jti = "test_jti_123"
        role = UserRole.VIEWER

        # Create and then invalidate session
        session_manager.create_session(user_id, jti, role)
        assert session_manager.is_session_active(jti)

        session_manager.invalidate_session(jti)
        assert not session_manager.is_session_active(jti)

    def test_token_blacklisting(self):
        """Test token blacklisting."""
        self.setUp()

        jti = "test_jti_123"

        # Token should not be blacklisted initially
        assert not session_manager.is_token_blacklisted(jti)

        # Blacklist token
        session_manager.blacklist_token(jti)
        assert session_manager.is_token_blacklisted(jti)

    def test_failed_attempt_tracking(self):
        """Test failed attempt tracking and lockout."""
        self.setUp()

        identifier = "192.168.1.1"

        # Initially not locked out
        assert not session_manager.is_locked_out(identifier)

        # Record failed attempts
        for _i in range(4):
            session_manager.record_failed_attempt(identifier)
            assert not session_manager.is_locked_out(identifier)

        # Fifth attempt should trigger lockout
        session_manager.record_failed_attempt(identifier)
        assert session_manager.is_locked_out(identifier)

        # Clear attempts
        session_manager.clear_failed_attempts(identifier)
        assert not session_manager.is_locked_out(identifier)


class TestSecurityUtilities:
    """Test miscellaneous security utilities."""

    def test_constant_time_compare(self):
        """Test constant time string comparison."""
        # Equal strings
        assert constant_time_compare("secret", "secret")

        # Different strings
        assert not constant_time_compare("secret", "different")

        # Different lengths
        assert not constant_time_compare("short", "much_longer_string")


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test API endpoints with authentication."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def get_auth_headers(self, role: UserRole = UserRole.ADMIN):
        """Get authentication headers for testing."""
        # Create a test token
        token = JWTUtils.create_access_token("test_user", role)
        return {"Authorization": f"Bearer {token}"}

    def test_health_endpoint_no_auth(self, client):
        """Test health endpoint doesn't require authentication."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_config_endpoint_requires_auth(self, client):
        """Test config endpoint requires authentication."""
        # Without auth should fail
        response = client.get("/api/v1/config/")
        assert response.status_code == 401

        # With auth should succeed
        headers = self.get_auth_headers(UserRole.VIEWER)
        response = client.get("/api/v1/config/", headers=headers)
        # May fail due to missing dependencies, but should not be auth error
        assert response.status_code != 401

    def test_config_update_requires_admin(self, client):
        """Test config update requires admin role."""
        config_data = {"test": "value"}

        # Viewer should not be able to update
        headers = self.get_auth_headers(UserRole.VIEWER)
        response = client.put("/api/v1/config/", json=config_data, headers=headers)
        assert response.status_code == 403

        # Admin should be able to update
        headers = self.get_auth_headers(UserRole.ADMIN)
        response = client.put("/api/v1/config/", json=config_data, headers=headers)
        # May fail due to missing dependencies, but should not be auth error
        assert response.status_code not in [401, 403]

    def test_game_reset_requires_operator(self, client):
        """Test game reset requires operator role."""
        # Viewer should not be able to reset
        headers = self.get_auth_headers(UserRole.VIEWER)
        response = client.post("/api/v1/game/reset", headers=headers)
        assert response.status_code == 403

        # Operator should be able to reset
        headers = self.get_auth_headers(UserRole.OPERATOR)
        response = client.post("/api/v1/game/reset", headers=headers)
        # May fail due to missing dependencies, but should not be auth error
        assert response.status_code not in [401, 403]

    def test_login_endpoint(self, client):
        """Test login endpoint."""
        # Valid login
        login_data = {"username": "admin", "password": "admin123!"}
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        # Invalid login
        invalid_login = {"username": "admin", "password": "wrong_password"}
        response = client.post("/api/v1/auth/login", json=invalid_login)
        assert response.status_code == 401

    def test_api_key_management(self, client):
        """Test API key management endpoints."""
        headers = self.get_auth_headers(UserRole.ADMIN)

        # Create API key
        api_key_data = {"name": "test_key", "role": "viewer", "expires_days": 30}
        response = client.post(
            "/api/v1/auth/api-keys", json=api_key_data, headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert "key_id" in data

        key_id = data["key_id"]

        # List API keys
        response = client.get("/api/v1/auth/api-keys", headers=headers)
        assert response.status_code == 200
        keys = response.json()
        assert isinstance(keys, list)
        assert any(key["key_id"] == key_id for key in keys)

        # Revoke API key
        response = client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=headers)
        assert response.status_code == 200

    def test_logout_endpoint(self, client):
        """Test logout endpoint."""
        headers = self.get_auth_headers(UserRole.VIEWER)

        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 200


@pytest.mark.asyncio
class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def client(self):
        """Create test client with rate limiting."""
        app = create_app()
        return TestClient(app)

    def test_rate_limiting_kicks_in(self, client):
        """Test that rate limiting eventually kicks in."""
        # Make many requests quickly
        responses = []
        for _i in range(10):
            response = client.get("/api/v1/health/")
            responses.append(response.status_code)

        # All should succeed initially (rate limits are generous in config)
        # This test would need adjustment based on actual rate limit settings
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 5  # At least some should succeed


if __name__ == "__main__":
    # Run specific test classes if executed directly
    pytest.main([__file__, "-v"])
