"""Integration layer for enhanced session management with existing authentication."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, Request, status

from ..utils.security import UserRole
from .enhanced_session import (
    EnhancedSessionManager,
    SessionData,
    SessionPolicy,
    SessionStatus,
    SessionStorageBackend,
)
from .session_monitor import session_monitor
from .session_security import session_security_validator
from .session_storage import create_session_storage

logger = logging.getLogger(__name__)


class EnhancedAuthenticationManager:
    """Enhanced authentication manager that integrates all session management features."""

    def __init__(
        self,
        policy: Optional[SessionPolicy] = None,
        storage_backend: str = "memory",
        storage_config: dict[str, Any] = None,
    ):
        """Initialize the enhanced authentication manager."""
        # Create session policy with enhanced settings
        self.policy = policy or SessionPolicy(
            idle_timeout_minutes=30,
            absolute_timeout_hours=8,
            remember_me_timeout_days=30,
            max_concurrent_sessions=5,
            require_ip_validation=True,
            allow_ip_change=False,
            require_user_agent_validation=True,
            enable_session_hijacking_protection=True,
            cleanup_interval_minutes=15,
            session_activity_tracking=True,
            detailed_audit_logging=True,
            encrypt_session_data=True,
        )

        # Create storage backend
        storage = create_session_storage(storage_backend, **(storage_config or {}))

        # Initialize components
        self.session_manager = EnhancedSessionManager(
            policy=self.policy,
            storage_backend=(
                SessionStorageBackend.MEMORY
                if storage_backend == "memory"
                else SessionStorageBackend.REDIS
            ),
        )
        self.session_manager._storage = storage  # Override with our storage

        self.monitor = session_monitor
        self.security_validator = session_security_validator

        # Integration with legacy session manager
        self._legacy_sessions: dict[str, SessionData] = {}

    async def create_enhanced_session(
        self,
        user_id: str,
        jti: str,
        role: UserRole,
        request: Request,
        remember_me: bool = False,
        session_type: str = "web",
    ) -> Optional[SessionData]:
        """Create an enhanced session with full security validation."""
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        # Security pre-checks
        if ip_address:
            # Check if IP is blocked or suspicious
            ip_validation = await self.security_validator._validate_ip_address(
                SessionData(
                    jti="temp",
                    user_id=user_id,
                    role=role.value,
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.now(timezone.utc),
                    last_activity=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc),
                    absolute_expires_at=datetime.now(timezone.utc),
                ),
                ip_address,
            )

            if not ip_validation.is_valid and ip_validation.score > 0.7:
                logger.warning(
                    f"Blocking session creation from suspicious IP {ip_address}: {ip_validation.reasons}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Session creation blocked due to security concerns",
                )

        # Check concurrent session limits
        all_sessions = await self.session_manager._storage.get_all_sessions()
        concurrent_check = (
            await self.security_validator.check_concurrent_session_limits(
                user_id, ip_address or "unknown", all_sessions
            )
        )

        if not concurrent_check.is_valid:
            logger.warning(f"Concurrent session limit exceeded for user {user_id}")
            # Optionally terminate oldest sessions
            user_sessions = await self.session_manager._storage.get_user_sessions(
                user_id
            )
            active_sessions = [
                s for s in user_sessions if s.status == SessionStatus.ACTIVE
            ]

            if len(active_sessions) >= self.policy.max_concurrent_sessions:
                # Terminate oldest session
                oldest_session = min(active_sessions, key=lambda s: s.created_at)
                await self.session_manager.invalidate_session(
                    oldest_session.jti, "concurrent_limit_exceeded"
                )

        # Create the session
        session_data = await self.session_manager.create_session(
            user_id=user_id,
            jti=jti,
            role=role,
            ip_address=ip_address,
            user_agent=user_agent,
            remember_me=remember_me,
            session_type=session_type,
        )

        if session_data:
            # Record session creation event
            await self.monitor.record_session_created(session_data)

            # Update user behavior profile
            self.security_validator.update_user_behavior_profile(user_id, session_data)

            logger.info(f"Enhanced session created for user {user_id}: {jti}")

        return session_data

    async def validate_session_request(
        self, jti: str, request: Request, request_path: str = None
    ) -> bool:
        """Validate a session request with comprehensive security checks."""
        session = await self.session_manager.get_session(jti)
        if not session:
            return False

        # Basic session validity
        if not await self.session_manager.is_session_active(jti):
            return False

        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        additional_headers = dict(request.headers)

        # Comprehensive security validation
        validation_result = await self.security_validator.validate_session_security(
            session, ip_address or "", user_agent or "", additional_headers
        )

        # Handle validation results
        if not validation_result.is_valid:
            logger.warning(
                f"Session {jti} failed security validation: {validation_result.reasons}"
            )

            # Record suspicious activity
            await self.monitor.record_suspicious_activity(
                session,
                "security_validation_failed",
                {
                    "threat_level": validation_result.threat_level.value,
                    "score": validation_result.score,
                    "reasons": validation_result.reasons,
                },
            )

            # Take action based on threat level
            if validation_result.threat_level.value in ["critical", "high"]:
                # Suspend or invalidate session
                await self.session_manager.suspend_session(jti, "security_threat")
                return False
            elif validation_result.threat_level.value == "medium":
                # Require additional verification (placeholder)
                session.requires_reauth = True
                await self.session_manager._storage.update_session(jti, session)

        # Update session activity
        success = await self.session_manager.update_session_activity(
            jti, ip_address, user_agent, request_path
        )

        if success and request_path:
            # Record activity for monitoring
            await self.monitor.record_session_activity(session, request_path)

        return success

    async def invalidate_session(self, jti: str, reason: str = "user_logout") -> bool:
        """Invalidate a session with proper cleanup and logging."""
        session = await self.session_manager.get_session(jti)
        if session:
            # Record session end
            await self.monitor.record_session_ended(session, reason)

        # Invalidate the session
        success = await self.session_manager.invalidate_session(jti, reason)

        # Blacklist the token
        if success:
            self.session_manager.blacklist_token(jti)

        return success

    async def invalidate_user_sessions(
        self, user_id: str, exclude_jti: Optional[str] = None
    ) -> int:
        """Invalidate all sessions for a user."""
        user_sessions = await self.session_manager.get_user_sessions(user_id)
        count = 0

        for session in user_sessions:
            if session.jti != exclude_jti and session.status == SessionStatus.ACTIVE:
                if await self.invalidate_session(session.jti, "user_logout_all"):
                    count += 1

        return count

    async def get_session_info(self, jti: str) -> Optional[dict[str, Any]]:
        """Get comprehensive session information."""
        session = await self.session_manager.get_session(jti)
        if not session:
            return None

        # Get additional security information
        all_sessions = await self.session_manager._storage.get_all_sessions()
        concurrent_check = (
            await self.security_validator.check_concurrent_session_limits(
                session.user_id, session.ip_address or "unknown", all_sessions
            )
        )

        return {
            "jti": session.jti,
            "user_id": session.user_id,
            "role": session.role,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "absolute_expires_at": session.absolute_expires_at.isoformat(),
            "ip_address": session.ip_address,
            "user_agent": session.user_agent,
            "device_fingerprint": session.device_fingerprint,
            "is_mobile": session.is_mobile,
            "session_type": session.session_type,
            "request_count": session.request_count,
            "last_request_path": session.last_request_path,
            "remember_me": session.remember_me,
            "requires_reauth": session.requires_reauth,
            "suspicious_activity_count": session.suspicious_activity_count,
            "concurrent_sessions_warning": not concurrent_check.is_valid,
        }

    async def get_user_sessions_info(self, user_id: str) -> list[dict[str, Any]]:
        """Get information about all user sessions."""
        user_sessions = await self.session_manager.get_user_sessions(user_id)
        sessions_info = []

        for session in user_sessions:
            session_info = await self.get_session_info(session.jti)
            if session_info:
                sessions_info.append(session_info)

        return sessions_info

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get comprehensive system metrics."""
        all_sessions = await self.session_manager._storage.get_all_sessions()
        analytics = await self.session_manager.get_session_analytics()
        metrics = await self.monitor.get_current_metrics(all_sessions)
        alerts = await self.monitor.get_alerts(unresolved_only=True)

        return {
            "session_analytics": {
                "total_sessions": analytics.total_sessions,
                "active_sessions": analytics.active_sessions,
                "expired_sessions": analytics.expired_sessions,
                "concurrent_sessions_by_user": analytics.concurrent_sessions_by_user,
                "sessions_by_role": analytics.sessions_by_role,
                "average_session_duration": analytics.average_session_duration,
                "peak_concurrent_sessions": analytics.peak_concurrent_sessions,
                "suspicious_activity_count": analytics.suspicious_activity_count,
            },
            "current_metrics": {
                "active_sessions": metrics.active_sessions,
                "total_sessions_today": metrics.total_sessions_today,
                "failed_logins_today": metrics.failed_logins_today,
                "suspicious_activities_today": metrics.suspicious_activities_today,
                "sessions_by_device_type": metrics.sessions_by_device_type,
                "sessions_by_browser": metrics.sessions_by_browser,
                "most_active_users": metrics.most_active_users[:5],
                "login_times_distribution": metrics.login_times_distribution,
            },
            "security_alerts": {
                "total_unresolved": len(alerts),
                "critical": len([a for a in alerts if a.severity.value == "critical"]),
                "high": len([a for a in alerts if a.severity.value == "high"]),
                "medium": len([a for a in alerts if a.severity.value == "medium"]),
                "low": len([a for a in alerts if a.severity.value == "low"]),
            },
            "policy_settings": {
                "idle_timeout_minutes": self.policy.idle_timeout_minutes,
                "absolute_timeout_hours": self.policy.absolute_timeout_hours,
                "max_concurrent_sessions": self.policy.max_concurrent_sessions,
                "require_ip_validation": self.policy.require_ip_validation,
                "enable_session_hijacking_protection": self.policy.enable_session_hijacking_protection,
                "session_activity_tracking": self.policy.session_activity_tracking,
            },
        }

    async def get_security_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent security alerts."""
        alerts = await self.monitor.get_alerts(unresolved_only=False)
        recent_alerts = alerts[:limit]

        return [alert.to_dict() for alert in recent_alerts]

    async def resolve_security_alert(self, alert_id: str) -> bool:
        """Resolve a security alert."""
        return await self.monitor.resolve_alert(alert_id)

    async def update_session_policy(self, policy_updates: dict[str, Any]) -> bool:
        """Update session policy settings."""
        try:
            # Validate and update policy
            current_policy_dict = self.policy.dict()
            current_policy_dict.update(policy_updates)

            new_policy = SessionPolicy(**current_policy_dict)
            self.session_manager.update_policy(new_policy)
            self.policy = new_policy

            logger.info(f"Session policy updated: {policy_updates}")
            return True

        except Exception as e:
            logger.error(f"Failed to update session policy: {e}")
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Manually trigger session cleanup."""
        return await self.session_manager.cleanup_expired_sessions()

    async def get_user_activity_summary(
        self, user_id: str, days: int = 7
    ) -> dict[str, Any]:
        """Get detailed user activity summary."""
        return await self.monitor.get_user_activity_summary(user_id, days)

    def is_legacy_session_active(self, jti: str) -> bool:
        """Check if legacy session is active (for backward compatibility)."""
        # This method provides compatibility with the original SessionManager
        session = self._legacy_sessions.get(jti)
        if not session:
            return False

        return session.status == SessionStatus.ACTIVE

    async def migrate_legacy_session(
        self, jti: str, user_id: str, role: UserRole
    ) -> bool:
        """Migrate a legacy session to enhanced session management."""
        try:
            # Create enhanced session data for existing JWT
            session_data = SessionData(
                jti=jti,
                user_id=user_id,
                role=role.value,
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc)
                + timedelta(minutes=self.policy.idle_timeout_minutes),
                absolute_expires_at=datetime.now(timezone.utc)
                + timedelta(hours=self.policy.absolute_timeout_hours),
            )

            # Store in enhanced session manager
            success = await self.session_manager._storage.store_session(session_data)

            if success:
                logger.info(
                    f"Migrated legacy session {jti} to enhanced session management"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to migrate legacy session {jti}: {e}")
            return False


# Global enhanced authentication manager instance
enhanced_auth_manager = EnhancedAuthenticationManager()
