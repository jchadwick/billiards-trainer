"""Session monitoring, analytics, and reporting system."""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from ..utils.security import SecurityEvent, SecurityEventType, security_logger
from .enhanced_session import SessionData, SessionStatus

logger = logging.getLogger(__name__)


class MonitoringLevel(str, Enum):
    """Monitoring detail levels."""

    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SessionEvent:
    """Session event for monitoring."""

    timestamp: datetime
    event_type: str
    session_id: str
    user_id: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class SecurityAlert:
    """Security alert for suspicious activity."""

    id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: dict[str, Any] = None
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["severity"] = self.severity.value
        return data


@dataclass
class SessionMetrics:
    """Session metrics for monitoring."""

    # Current state
    active_sessions: int = 0
    total_sessions_today: int = 0
    failed_logins_today: int = 0

    # Performance metrics
    average_session_duration: float = 0.0
    median_session_duration: float = 0.0
    peak_concurrent_sessions: int = 0

    # Security metrics
    suspicious_activities_today: int = 0
    blocked_attempts_today: int = 0
    concurrent_limit_violations: int = 0

    # Geographic distribution
    sessions_by_country: dict[str, int] = None
    sessions_by_city: dict[str, int] = None

    # Device distribution
    sessions_by_device_type: dict[str, int] = None
    sessions_by_browser: dict[str, int] = None

    # User behavior
    most_active_users: list[tuple[str, int]] = None
    login_times_distribution: dict[int, int] = None  # hour -> count

    def __post_init__(self):
        if self.sessions_by_country is None:
            self.sessions_by_country = {}
        if self.sessions_by_city is None:
            self.sessions_by_city = {}
        if self.sessions_by_device_type is None:
            self.sessions_by_device_type = {}
        if self.sessions_by_browser is None:
            self.sessions_by_browser = {}
        if self.most_active_users is None:
            self.most_active_users = []
        if self.login_times_distribution is None:
            self.login_times_distribution = {}


class SessionMonitor:
    """Advanced session monitoring and analytics system."""

    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED):
        self.monitoring_level = monitoring_level
        self._events: deque = deque(maxlen=10000)  # Keep last 10k events
        self._alerts: list[SecurityAlert] = []
        self._metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self._user_activity: dict[str, list[datetime]] = defaultdict(list)
        self._ip_activity: dict[str, list[datetime]] = defaultdict(list)
        self._failed_attempts: dict[str, list[datetime]] = defaultdict(list)
        self._geolocation_cache: dict[str, dict[str, str]] = {}

        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins_per_minute": 10,
            "concurrent_sessions_per_user": 10,
            "suspicious_activities_per_hour": 5,
            "session_duration_anomaly_factor": 3.0,  # > 3x normal duration
            "rapid_login_threshold_seconds": 5,  # Multiple logins in 5 seconds
        }

        # Start background monitoring tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_started = False

    def _start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_started:
            return

        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()

            async def monitoring_loop():
                while True:
                    try:
                        await asyncio.sleep(60)  # Run every minute
                        await self._analyze_patterns()
                        await self._cleanup_old_data()
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error in monitoring loop: {e}")

            # Only create task if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                self._monitoring_task = loop.create_task(monitoring_loop())
                self._monitoring_started = True
            except RuntimeError:
                # No running event loop, will start later when needed
                logger.debug(
                    "No running event loop, monitoring task will start when first used"
                )
        except Exception as e:
            logger.warning(f"Could not start monitoring task: {e}")

    async def record_event(self, event: SessionEvent):
        """Record a session event for monitoring."""
        # Start monitoring if not already started
        if not self._monitoring_started:
            self._start_monitoring()

        self._events.append(event)

        # Immediate analysis for critical events
        if event.event_type in [
            "login_failed",
            "suspicious_activity",
            "session_hijack",
        ]:
            await self._analyze_immediate_threat(event)

        # Update activity tracking
        if event.user_id:
            self._user_activity[event.user_id].append(event.timestamp)

        if event.ip_address:
            self._ip_activity[event.ip_address].append(event.timestamp)

        if event.event_type == "login_failed":
            key = event.ip_address or event.user_id or "unknown"
            self._failed_attempts[key].append(event.timestamp)

    async def record_session_created(self, session_data: SessionData):
        """Record session creation event."""
        event = SessionEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="session_created",
            session_id=session_data.jti,
            user_id=session_data.user_id,
            ip_address=session_data.ip_address,
            user_agent=session_data.user_agent,
            details={
                "role": session_data.role,
                "remember_me": session_data.remember_me,
                "session_type": session_data.session_type,
                "device_fingerprint": session_data.device_fingerprint,
            },
        )
        await self.record_event(event)

    async def record_session_activity(
        self, session_data: SessionData, request_path: str
    ):
        """Record session activity event."""
        if self.monitoring_level in [
            MonitoringLevel.DETAILED,
            MonitoringLevel.COMPREHENSIVE,
        ]:
            event = SessionEvent(
                timestamp=datetime.now(timezone.utc),
                event_type="session_activity",
                session_id=session_data.jti,
                user_id=session_data.user_id,
                ip_address=session_data.ip_address,
                details={
                    "request_path": request_path,
                    "request_count": session_data.request_count,
                },
            )
            await self.record_event(event)

    async def record_suspicious_activity(
        self, session_data: SessionData, reason: str, details: dict[str, Any]
    ):
        """Record suspicious activity event."""
        event = SessionEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="suspicious_activity",
            session_id=session_data.jti,
            user_id=session_data.user_id,
            ip_address=session_data.ip_address,
            details={
                "reason": reason,
                "suspicion_details": details,
            },
        )
        await self.record_event(event)

    async def record_session_ended(self, session_data: SessionData, reason: str):
        """Record session end event."""
        duration = (
            datetime.now(timezone.utc) - session_data.created_at
        ).total_seconds()

        event = SessionEvent(
            timestamp=datetime.now(timezone.utc),
            event_type="session_ended",
            session_id=session_data.jti,
            user_id=session_data.user_id,
            ip_address=session_data.ip_address,
            details={
                "reason": reason,
                "duration_seconds": duration,
                "request_count": session_data.request_count,
                "status": session_data.status.value,
            },
        )
        await self.record_event(event)

    async def _analyze_immediate_threat(self, event: SessionEvent):
        """Analyze events for immediate security threats."""
        now = datetime.now(timezone.utc)

        # Check for rapid failed login attempts
        if event.event_type == "login_failed":
            identifier = event.ip_address or event.user_id or "unknown"
            recent_attempts = [
                ts
                for ts in self._failed_attempts[identifier]
                if (now - ts).total_seconds() < 60
            ]

            if (
                len(recent_attempts)
                >= self.alert_thresholds["failed_logins_per_minute"]
            ):
                await self._create_alert(
                    AlertSeverity.HIGH,
                    "rapid_failed_logins",
                    f"Rapid failed login attempts detected from {identifier}",
                    event.session_id,
                    event.user_id,
                    event.ip_address,
                    {"attempts_count": len(recent_attempts), "timeframe": "1 minute"},
                )

        # Check for rapid login pattern (potential brute force)
        if event.event_type == "session_created" and event.user_id:
            recent_logins = [
                ts
                for ts in self._user_activity[event.user_id]
                if (now - ts).total_seconds()
                < self.alert_thresholds["rapid_login_threshold_seconds"]
            ]

            if len(recent_logins) > 3:  # More than 3 logins in threshold time
                await self._create_alert(
                    AlertSeverity.MEDIUM,
                    "rapid_login_pattern",
                    f"Rapid login pattern detected for user {event.user_id}",
                    event.session_id,
                    event.user_id,
                    event.ip_address,
                    {"logins_count": len(recent_logins)},
                )

    async def _analyze_patterns(self):
        """Analyze patterns and generate alerts."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)

        # Analyze suspicious activities in the last hour
        recent_suspicious = [
            event
            for event in self._events
            if event.event_type == "suspicious_activity" and event.timestamp > hour_ago
        ]

        if (
            len(recent_suspicious)
            >= self.alert_thresholds["suspicious_activities_per_hour"]
        ):
            await self._create_alert(
                AlertSeverity.MEDIUM,
                "high_suspicious_activity",
                "High volume of suspicious activities detected",
                None,
                None,
                None,
                {"count": len(recent_suspicious), "timeframe": "1 hour"},
            )

        # Analyze concurrent session violations
        user_session_counts = defaultdict(int)
        for event in self._events:
            if event.event_type == "session_created" and event.timestamp > hour_ago:
                user_session_counts[event.user_id] += 1

        for user_id, count in user_session_counts.items():
            if count >= self.alert_thresholds["concurrent_sessions_per_user"]:
                await self._create_alert(
                    AlertSeverity.MEDIUM,
                    "concurrent_session_violation",
                    f"User {user_id} exceeded concurrent session limit",
                    None,
                    user_id,
                    None,
                    {"session_count": count},
                )

    async def _create_alert(
        self,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        session_id: Optional[str],
        user_id: Optional[str],
        ip_address: Optional[str],
        details: dict[str, Any],
    ):
        """Create a security alert."""
        alert = SecurityAlert(
            id=f"alert_{len(self._alerts) + 1:06d}",
            severity=severity,
            alert_type=alert_type,
            message=message,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )

        self._alerts.append(alert)

        # Log to security logger
        security_logger.log_event(
            SecurityEvent(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                timestamp=alert.timestamp,
                user_id=user_id,
                ip_address=ip_address,
                details={
                    "alert_id": alert.id,
                    "alert_type": alert_type,
                    "severity": severity.value,
                    "message": message,
                    **details,
                },
                success=False,
            )
        )

        # Send notifications for high/critical alerts
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            await self._send_alert_notification(alert)

    async def _send_alert_notification(self, alert: SecurityAlert):
        """Send alert notification (placeholder for notification system)."""
        logger.warning(
            f"SECURITY ALERT [{alert.severity.value.upper()}]: {alert.message}"
        )
        # TODO: Implement actual notification system (email, Slack, etc.)

    async def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)

        # Clean up old activity data
        for user_id in list(self._user_activity.keys()):
            self._user_activity[user_id] = [
                ts for ts in self._user_activity[user_id] if ts > cutoff_time
            ]
            if not self._user_activity[user_id]:
                del self._user_activity[user_id]

        for ip in list(self._ip_activity.keys()):
            self._ip_activity[ip] = [
                ts for ts in self._ip_activity[ip] if ts > cutoff_time
            ]
            if not self._ip_activity[ip]:
                del self._ip_activity[ip]

        for key in list(self._failed_attempts.keys()):
            self._failed_attempts[key] = [
                ts for ts in self._failed_attempts[key] if ts > cutoff_time
            ]
            if not self._failed_attempts[key]:
                del self._failed_attempts[key]

    async def get_current_metrics(self, sessions: list[SessionData]) -> SessionMetrics:
        """Calculate current session metrics."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Basic counts
        active_sessions = len([s for s in sessions if s.status == SessionStatus.ACTIVE])

        # Today's events
        today_events = [e for e in self._events if e.timestamp >= today_start]
        total_sessions_today = len(
            [e for e in today_events if e.event_type == "session_created"]
        )
        failed_logins_today = len(
            [e for e in today_events if e.event_type == "login_failed"]
        )
        suspicious_activities_today = len(
            [e for e in today_events if e.event_type == "suspicious_activity"]
        )

        # Session duration metrics
        completed_sessions = [
            s
            for s in sessions
            if s.status in [SessionStatus.EXPIRED, SessionStatus.INVALIDATED]
        ]
        durations = []
        for session in completed_sessions:
            duration = (session.last_activity - session.created_at).total_seconds()
            durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0
        median_duration = sorted(durations)[len(durations) // 2] if durations else 0

        # Device and browser distribution
        device_types = defaultdict(int)
        browsers = defaultdict(int)

        for session in sessions:
            if session.status == SessionStatus.ACTIVE:
                # Simple device detection
                if session.is_mobile:
                    device_types["Mobile"] += 1
                else:
                    device_types["Desktop"] += 1

                # Simple browser detection from user agent
                if session.user_agent:
                    ua = session.user_agent.lower()
                    if "chrome" in ua:
                        browsers["Chrome"] += 1
                    elif "firefox" in ua:
                        browsers["Firefox"] += 1
                    elif "safari" in ua:
                        browsers["Safari"] += 1
                    elif "edge" in ua:
                        browsers["Edge"] += 1
                    else:
                        browsers["Other"] += 1

        # User activity ranking
        user_activity_counts = defaultdict(int)
        for event in today_events:
            if event.user_id:
                user_activity_counts[event.user_id] += 1

        most_active_users = sorted(
            user_activity_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Login time distribution
        login_times = defaultdict(int)
        for event in today_events:
            if event.event_type == "session_created":
                hour = event.timestamp.hour
                login_times[hour] += 1

        return SessionMetrics(
            active_sessions=active_sessions,
            total_sessions_today=total_sessions_today,
            failed_logins_today=failed_logins_today,
            average_session_duration=avg_duration,
            median_session_duration=median_duration,
            peak_concurrent_sessions=max(
                active_sessions, getattr(self, "_peak_sessions", 0)
            ),
            suspicious_activities_today=suspicious_activities_today,
            blocked_attempts_today=0,  # TODO: Implement blocked attempts tracking
            concurrent_limit_violations=0,  # TODO: Implement from alerts
            sessions_by_device_type=dict(device_types),
            sessions_by_browser=dict(browsers),
            most_active_users=most_active_users,
            login_times_distribution=dict(login_times),
        )

    async def get_alerts(
        self, severity: Optional[AlertSeverity] = None, unresolved_only: bool = True
    ) -> list[SecurityAlert]:
        """Get security alerts."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False

    async def get_user_activity_summary(
        self, user_id: str, days: int = 7
    ) -> dict[str, Any]:
        """Get detailed activity summary for a user."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        user_events = [
            event
            for event in self._events
            if event.user_id == user_id and event.timestamp > cutoff_time
        ]

        # Group events by type
        events_by_type = defaultdict(list)
        for event in user_events:
            events_by_type[event.event_type].append(event)

        # Calculate session statistics
        session_events = events_by_type.get("session_created", [])
        total_sessions = len(session_events)

        # IP addresses used
        ip_addresses = list(
            {event.ip_address for event in user_events if event.ip_address}
        )

        # User agents used
        user_agents = list(
            {event.user_agent for event in user_events if event.user_agent}
        )

        # Activity timeline
        daily_activity = defaultdict(int)
        for event in user_events:
            day = event.timestamp.date().isoformat()
            daily_activity[day] += 1

        return {
            "user_id": user_id,
            "period_days": days,
            "total_events": len(user_events),
            "total_sessions": total_sessions,
            "events_by_type": {k: len(v) for k, v in events_by_type.items()},
            "unique_ip_addresses": len(ip_addresses),
            "ip_addresses": ip_addresses[:10],  # Limit to 10 most recent
            "unique_user_agents": len(user_agents),
            "daily_activity": dict(daily_activity),
            "suspicious_activities": len(events_by_type.get("suspicious_activity", [])),
        }

    def update_alert_thresholds(self, thresholds: dict[str, Any]):
        """Update alert thresholds."""
        self.alert_thresholds.update(thresholds)

    def __del__(self):
        """Cleanup on destruction."""
        if self._monitoring_task:
            self._monitoring_task.cancel()


# Global session monitor instance
session_monitor = SessionMonitor()
