"""Advanced session security features including hijacking protection and validation."""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

try:
    import geoip2.database
    import geoip2.errors

    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False

    # Create mock classes
    class MockGeoIP:
        def __init__(self, *args, **kwargs):
            pass

        def city(self, ip):
            # Return mock response
            class MockResponse:
                class Country:
                    iso_code = "US"

                class City:
                    name = "Unknown"

                class Subdivisions:
                    class MostSpecific:
                        name = "Unknown"

                    most_specific = MostSpecific()

                class Location:
                    time_zone = "UTC"
                    latitude = 0.0
                    longitude = 0.0

                country = Country()
                city = City()
                subdivisions = Subdivisions()
                location = Location()

            return MockResponse()

    class MockGeoIPErrors:
        class AddressNotFoundError(Exception):
            pass

    geoip2 = type(
        "geoip2",
        (),
        {
            "database": type("database", (), {"Reader": MockGeoIP}),
            "errors": MockGeoIPErrors,
        },
    )

try:
    from user_agents import parse as parse_user_agent

    USER_AGENTS_AVAILABLE = True
except ImportError:
    USER_AGENTS_AVAILABLE = False

    # Create mock user agent parser
    def parse_user_agent(user_agent_string):
        class MockUA:
            class Browser:
                family = "Unknown"
                version_string = "Unknown"

            class OS:
                family = "Unknown"
                version_string = "Unknown"

            class Device:
                family = "Unknown"

            browser = Browser()
            os = OS()
            device = Device()
            is_mobile = False
            is_tablet = False
            is_pc = True
            is_bot = False

        return MockUA()


from .enhanced_session import SessionData, SessionStatus

logger = logging.getLogger(__name__)


class SecurityThreatLevel(str, Enum):
    """Security threat levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityValidationResult:
    """Result of security validation."""

    is_valid: bool
    threat_level: SecurityThreatLevel
    reasons: list[str]
    recommendations: list[str]
    score: float  # 0.0 = completely secure, 1.0 = maximum threat


@dataclass
class IPInfo:
    """IP address information."""

    ip: str
    country: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    timezone: Optional[str] = None
    isp: Optional[str] = None
    is_proxy: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    risk_score: float = 0.0


@dataclass
class DeviceFingerprint:
    """Enhanced device fingerprint."""

    browser_family: str
    browser_version: str
    os_family: str
    os_version: str
    device_family: str
    is_mobile: bool
    is_tablet: bool
    is_pc: bool
    is_bot: bool
    screen_resolution: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    fingerprint_hash: Optional[str] = None


class SessionSecurityValidator:
    """Advanced session security validation and protection."""

    def __init__(self, geoip_db_path: Optional[str] = None):
        self.geoip_db_path = geoip_db_path
        self._geoip_reader = None
        self._known_proxies: set[str] = set()
        self._known_tor_exits: set[str] = set()
        self._datacenter_ranges: list[tuple[str, str]] = []

        # Security thresholds
        self.security_config = {
            "max_ip_changes_per_session": 2,
            "max_user_agent_changes_per_session": 1,
            "max_concurrent_sessions_per_user": 5,
            "max_concurrent_sessions_per_ip": 10,
            "session_hijack_detection_enabled": True,
            "geolocation_validation_enabled": True,
            "device_fingerprint_validation_enabled": True,
            "suspicious_ip_blocking_enabled": True,
            "max_login_attempts_per_ip_per_hour": 20,
            "max_failed_sessions_per_ip_per_day": 50,
        }

        # Behavioral analysis
        self._user_behavior_profiles: dict[str, dict[str, Any]] = {}
        self._ip_reputation_cache: dict[str, IPInfo] = {}
        self._device_fingerprint_cache: dict[str, DeviceFingerprint] = {}

        # Load threat intelligence data
        self._load_threat_intelligence()

    def _load_threat_intelligence(self):
        """Load threat intelligence data."""
        try:
            if self.geoip_db_path:
                self._geoip_reader = geoip2.database.Reader(self.geoip_db_path)
        except Exception as e:
            logger.warning(f"Could not load GeoIP database: {e}")

        # Load known proxy/VPN/Tor lists (placeholder - in production load from threat feeds)
        self._known_proxies = {
            # Add known proxy/VPN IP ranges
        }

        self._known_tor_exits = {
            # Add known Tor exit nodes
        }

        self._datacenter_ranges = [
            # Add datacenter IP ranges
        ]

    async def validate_session_security(
        self,
        session: SessionData,
        current_ip: str,
        current_user_agent: str,
        additional_headers: dict[str, str] = None,
    ) -> SecurityValidationResult:
        """Comprehensive security validation for a session."""
        reasons = []
        recommendations = []
        threat_score = 0.0

        # IP address validation
        ip_validation = await self._validate_ip_address(session, current_ip)
        if not ip_validation.is_valid:
            reasons.extend(ip_validation.reasons)
            recommendations.extend(ip_validation.recommendations)
            threat_score += ip_validation.score

        # User agent validation
        ua_validation = await self._validate_user_agent(session, current_user_agent)
        if not ua_validation.is_valid:
            reasons.extend(ua_validation.reasons)
            recommendations.extend(ua_validation.recommendations)
            threat_score += ua_validation.score

        # Device fingerprint validation
        if self.security_config["device_fingerprint_validation_enabled"]:
            fingerprint_validation = await self._validate_device_fingerprint(
                session, current_user_agent, additional_headers or {}
            )
            if not fingerprint_validation.is_valid:
                reasons.extend(fingerprint_validation.reasons)
                recommendations.extend(fingerprint_validation.recommendations)
                threat_score += fingerprint_validation.score

        # Geolocation validation
        if self.security_config["geolocation_validation_enabled"]:
            geo_validation = await self._validate_geolocation(session, current_ip)
            if not geo_validation.is_valid:
                reasons.extend(geo_validation.reasons)
                recommendations.extend(geo_validation.recommendations)
                threat_score += geo_validation.score

        # Behavioral analysis
        behavior_validation = await self._validate_user_behavior(session, current_ip)
        if not behavior_validation.is_valid:
            reasons.extend(behavior_validation.reasons)
            recommendations.extend(behavior_validation.recommendations)
            threat_score += behavior_validation.score

        # Session hijacking detection
        if self.security_config["session_hijack_detection_enabled"]:
            hijack_validation = await self._detect_session_hijacking(
                session, current_ip, current_user_agent
            )
            if not hijack_validation.is_valid:
                reasons.extend(hijack_validation.reasons)
                recommendations.extend(hijack_validation.recommendations)
                threat_score += hijack_validation.score

        # Determine overall threat level
        if threat_score >= 0.8:
            threat_level = SecurityThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = SecurityThreatLevel.HIGH
        elif threat_score >= 0.4:
            threat_level = SecurityThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            threat_level = SecurityThreatLevel.LOW
        else:
            threat_level = SecurityThreatLevel.NONE

        is_valid = threat_level in [SecurityThreatLevel.NONE, SecurityThreatLevel.LOW]

        return SecurityValidationResult(
            is_valid=is_valid,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=min(threat_score, 1.0),
        )

    async def _validate_ip_address(
        self, session: SessionData, current_ip: str
    ) -> SecurityValidationResult:
        """Validate IP address changes and reputation."""
        reasons = []
        recommendations = []
        score = 0.0

        # Check for IP address changes
        if session.ip_address and current_ip != session.ip_address:
            reasons.append(
                f"IP address changed from {session.ip_address} to {current_ip}"
            )
            score += 0.3

            # Check if this is an allowed change pattern
            if not await self._is_allowed_ip_change(session.ip_address, current_ip):
                reasons.append("IP address change appears suspicious")
                score += 0.2
                recommendations.append("Consider requiring re-authentication")

        # IP reputation check
        ip_info = await self._get_ip_info(current_ip)
        if ip_info.risk_score > 0.5:
            reasons.append(
                f"IP address has poor reputation (score: {ip_info.risk_score})"
            )
            score += ip_info.risk_score * 0.4

        if ip_info.is_proxy or ip_info.is_tor:
            reasons.append("IP address is from proxy/VPN/Tor network")
            score += 0.3
            recommendations.append(
                "Consider additional verification for proxy/VPN users"
            )

        if ip_info.is_datacenter:
            reasons.append("IP address belongs to a datacenter")
            score += 0.2

        threat_level = (
            SecurityThreatLevel.HIGH
            if score >= 0.6
            else SecurityThreatLevel.MEDIUM
            if score >= 0.3
            else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.5,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _validate_user_agent(
        self, session: SessionData, current_user_agent: str
    ) -> SecurityValidationResult:
        """Validate user agent changes and consistency."""
        reasons = []
        recommendations = []
        score = 0.0

        if not current_user_agent:
            reasons.append("Missing user agent")
            score += 0.3

        if session.user_agent and current_user_agent != session.user_agent:
            reasons.append("User agent changed during session")
            score += 0.4

            # Analyze the change
            original_ua = parse_user_agent(session.user_agent)
            current_ua = parse_user_agent(current_user_agent)

            if original_ua.browser.family != current_ua.browser.family:
                reasons.append("Browser family changed")
                score += 0.3

            if original_ua.os.family != current_ua.os.family:
                reasons.append("Operating system changed")
                score += 0.5
                recommendations.append("OS change during session is highly suspicious")

        # Check for suspicious user agent patterns
        if current_user_agent and self._is_suspicious_user_agent(current_user_agent):
            reasons.append("User agent appears to be automated/suspicious")
            score += 0.4

        threat_level = (
            SecurityThreatLevel.HIGH
            if score >= 0.6
            else SecurityThreatLevel.MEDIUM
            if score >= 0.3
            else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.5,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _validate_device_fingerprint(
        self, session: SessionData, user_agent: str, headers: dict[str, str]
    ) -> SecurityValidationResult:
        """Validate device fingerprint consistency."""
        reasons = []
        recommendations = []
        score = 0.0

        current_fingerprint = self._generate_device_fingerprint(user_agent, headers)

        if (
            session.device_fingerprint
            and current_fingerprint.fingerprint_hash != session.device_fingerprint
        ):
            reasons.append("Device fingerprint changed")
            score += 0.5

            # Analyze what changed
            if hasattr(session, "_original_fingerprint"):
                original = session._original_fingerprint
                if original.browser_family != current_fingerprint.browser_family:
                    reasons.append("Browser changed")
                    score += 0.2

                if original.os_family != current_fingerprint.os_family:
                    reasons.append("Operating system changed")
                    score += 0.4

        # Check for bot patterns
        if current_fingerprint.is_bot:
            reasons.append("Request appears to be from a bot")
            score += 0.6
            recommendations.append("Consider implementing CAPTCHA")

        threat_level = (
            SecurityThreatLevel.HIGH
            if score >= 0.6
            else SecurityThreatLevel.MEDIUM
            if score >= 0.3
            else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.5,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _validate_geolocation(
        self, session: SessionData, current_ip: str
    ) -> SecurityValidationResult:
        """Validate geolocation consistency."""
        reasons = []
        recommendations = []
        score = 0.0

        if not self._geoip_reader:
            return SecurityValidationResult(
                is_valid=True,
                threat_level=SecurityThreatLevel.NONE,
                reasons=[],
                recommendations=[],
                score=0.0,
            )

        try:
            current_location = self._geoip_reader.city(current_ip)
            current_country = current_location.country.iso_code

            # If we have previous location data
            if hasattr(session, "geographic_location") and session.geographic_location:
                original_country = session.geographic_location

                if current_country != original_country:
                    reasons.append(
                        f"Country changed from {original_country} to {current_country}"
                    )
                    score += 0.4

                    # Check if this is a realistic change (adjacent countries = less suspicious)
                    if not self._are_adjacent_countries(
                        original_country, current_country
                    ):
                        reasons.append("Geolocation change appears unrealistic")
                        score += 0.3
                        recommendations.append(
                            "Require additional verification for international access"
                        )

        except geoip2.errors.AddressNotFoundError:
            reasons.append("Could not determine geolocation")
            score += 0.1

        threat_level = (
            SecurityThreatLevel.HIGH
            if score >= 0.6
            else SecurityThreatLevel.MEDIUM
            if score >= 0.3
            else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.4,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _validate_user_behavior(
        self, session: SessionData, current_ip: str
    ) -> SecurityValidationResult:
        """Validate user behavior patterns."""
        reasons = []
        recommendations = []
        score = 0.0

        user_id = session.user_id
        profile = self._user_behavior_profiles.get(user_id, {})

        # Check request frequency
        if session.request_count > 0:
            session_duration = (
                datetime.now(timezone.utc) - session.created_at
            ).total_seconds()
            requests_per_minute = (session.request_count / session_duration) * 60

            # Compare with user's normal pattern
            normal_rpm = profile.get("average_requests_per_minute", 10)
            if requests_per_minute > normal_rpm * 3:
                reasons.append(
                    f"Unusually high request frequency: {requests_per_minute:.1f} req/min"
                )
                score += 0.3

        # Check session timing patterns
        current_hour = datetime.now(timezone.utc).hour
        normal_hours = profile.get("normal_activity_hours", set(range(6, 24)))

        if current_hour not in normal_hours:
            reasons.append("Activity outside normal hours")
            score += 0.2

        # Check for concurrent sessions
        # This would require access to all user sessions - placeholder for now
        concurrent_sessions = profile.get("typical_concurrent_sessions", 1)
        if concurrent_sessions > 3:
            reasons.append("Unusually high number of concurrent sessions")
            score += 0.2

        threat_level = (
            SecurityThreatLevel.MEDIUM if score >= 0.3 else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.4,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _detect_session_hijacking(
        self, session: SessionData, current_ip: str, current_user_agent: str
    ) -> SecurityValidationResult:
        """Advanced session hijacking detection."""
        reasons = []
        recommendations = []
        score = 0.0

        # Multiple simultaneous changes (high risk)
        changes = []
        if session.ip_address and current_ip != session.ip_address:
            changes.append("IP")
        if session.user_agent and current_user_agent != session.user_agent:
            changes.append("User-Agent")

        if len(changes) >= 2:
            reasons.append(
                f"Multiple session parameters changed simultaneously: {', '.join(changes)}"
            )
            score += 0.7
            recommendations.append("Immediately require re-authentication")

        # Check for session replay attacks
        if self._detect_session_replay(session):
            reasons.append("Potential session replay attack detected")
            score += 0.8
            recommendations.append("Invalidate session immediately")

        # Check for impossible travel
        if await self._detect_impossible_travel(session, current_ip):
            reasons.append("Impossible travel detected between locations")
            score += 0.9
            recommendations.append("Session likely compromised - terminate immediately")

        # Check session timing anomalies
        if self._detect_timing_anomalies(session):
            reasons.append("Unusual session timing patterns detected")
            score += 0.3

        threat_level = (
            SecurityThreatLevel.CRITICAL
            if score >= 0.8
            else (
                SecurityThreatLevel.HIGH
                if score >= 0.6
                else (
                    SecurityThreatLevel.MEDIUM
                    if score >= 0.3
                    else SecurityThreatLevel.LOW
                )
            )
        )

        return SecurityValidationResult(
            is_valid=score < 0.6,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    async def _get_ip_info(self, ip: str) -> IPInfo:
        """Get comprehensive IP information."""
        if ip in self._ip_reputation_cache:
            return self._ip_reputation_cache[ip]

        ip_info = IPInfo(ip=ip)

        try:
            if self._geoip_reader:
                response = self._geoip_reader.city(ip)
                ip_info.country = response.country.iso_code
                ip_info.city = response.city.name
                ip_info.region = response.subdivisions.most_specific.name
                ip_info.timezone = str(response.location.time_zone)

        except geoip2.errors.AddressNotFoundError:
            pass

        # Check against threat lists
        ip_info.is_proxy = ip in self._known_proxies
        ip_info.is_tor = ip in self._known_tor_exits
        ip_info.is_datacenter = self._is_datacenter_ip(ip)

        # Calculate risk score
        risk_score = 0.0
        if ip_info.is_tor:
            risk_score += 0.8
        elif ip_info.is_proxy:
            risk_score += 0.6
        elif ip_info.is_datacenter:
            risk_score += 0.4

        ip_info.risk_score = min(risk_score, 1.0)

        # Cache the result
        self._ip_reputation_cache[ip] = ip_info
        return ip_info

    def _generate_device_fingerprint(
        self, user_agent: str, headers: dict[str, str]
    ) -> DeviceFingerprint:
        """Generate enhanced device fingerprint."""
        ua = parse_user_agent(user_agent) if user_agent else None

        fingerprint = DeviceFingerprint(
            browser_family=ua.browser.family if ua else "Unknown",
            browser_version=ua.browser.version_string if ua else "Unknown",
            os_family=ua.os.family if ua else "Unknown",
            os_version=ua.os.version_string if ua else "Unknown",
            device_family=ua.device.family if ua else "Unknown",
            is_mobile=ua.is_mobile if ua else False,
            is_tablet=ua.is_tablet if ua else False,
            is_pc=ua.is_pc if ua else False,
            is_bot=ua.is_bot if ua else False,
        )

        # Extract additional information from headers
        fingerprint.language = headers.get("Accept-Language", "")
        fingerprint.timezone = headers.get("X-Timezone", "")

        # Generate hash of fingerprint components
        fingerprint_data = f"{fingerprint.browser_family}:{fingerprint.browser_version}:{fingerprint.os_family}:{fingerprint.os_version}:{fingerprint.device_family}:{fingerprint.language}"
        fingerprint.fingerprint_hash = hashlib.sha256(
            fingerprint_data.encode()
        ).hexdigest()[:16]

        return fingerprint

    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent appears suspicious."""
        suspicious_patterns = [
            r"curl|wget|python|java|php|perl|ruby|go-http-client",
            r"bot|crawler|spider|scraper",
            r"scanner|probe|test",
            r"^$",  # Empty user agent
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                return True

        return False

    def _is_datacenter_ip(self, ip: str) -> bool:
        """Check if IP belongs to a datacenter."""
        # This is a simplified check - in production, use comprehensive datacenter IP lists
        datacenter_patterns = [
            r"^69\.46\.",  # Example datacenter range
            r"^173\.252\.",  # Example datacenter range
        ]

        return any(re.match(pattern, ip) for pattern in datacenter_patterns)

    async def _is_allowed_ip_change(self, original_ip: str, new_ip: str) -> bool:
        """Check if IP change is within allowed patterns."""
        # Check if IPs are in same subnet
        try:
            orig_parts = original_ip.split(".")
            new_parts = new_ip.split(".")

            # Same /24 subnet
            if orig_parts[:3] == new_parts[:3]:
                return True

            # Same /16 subnet (more lenient for mobile users)
            if orig_parts[:2] == new_parts[:2]:
                return True

        except (ValueError, IndexError):
            pass

        return False

    def _are_adjacent_countries(self, country1: str, country2: str) -> bool:
        """Check if countries are adjacent (simplified)."""
        # This is a simplified implementation - in production use comprehensive country adjacency data
        adjacent_countries = {
            "US": ["CA", "MX"],
            "CA": ["US"],
            "MX": ["US"],
            "GB": ["IE", "FR"],
            "FR": ["GB", "DE", "ES", "IT"],
            "DE": ["FR", "NL", "BE", "PL", "AT", "CH"],
            # Add more as needed
        }

        return country2 in adjacent_countries.get(country1, [])

    def _detect_session_replay(self, session: SessionData) -> bool:
        """Detect potential session replay attacks."""
        # This is a placeholder - implement based on request patterns
        # Look for identical request sequences, timestamps, etc.
        return False

    async def _detect_impossible_travel(
        self, session: SessionData, current_ip: str
    ) -> bool:
        """Detect impossible travel between locations."""
        if not self._geoip_reader or not session.ip_address:
            return False

        try:
            original_location = self._geoip_reader.city(session.ip_address)
            current_location = self._geoip_reader.city(current_ip)

            # Calculate distance (simplified)
            # In production, use proper geospatial distance calculation
            orig_lat = float(original_location.location.latitude or 0)
            orig_lon = float(original_location.location.longitude or 0)
            curr_lat = float(current_location.location.latitude or 0)
            curr_lon = float(current_location.location.longitude or 0)

            # Simple distance calculation (not accurate, use proper geospatial library in production)
            distance = ((curr_lat - orig_lat) ** 2 + (curr_lon - orig_lon) ** 2) ** 0.5

            # Time since session started
            time_diff = (
                datetime.now(timezone.utc) - session.created_at
            ).total_seconds() / 3600  # hours

            # Maximum realistic travel speed (including flights)
            max_speed_km_h = 1000  # km/h

            # Very rough calculation - in production use proper implementation
            if (
                distance > 10 and time_diff > 0
            ):  # Only check if significant distance and time
                required_speed = (
                    distance * 111
                ) / time_diff  # 111 km per degree (rough)
                return required_speed > max_speed_km_h

        except (geoip2.errors.AddressNotFoundError, ValueError, TypeError):
            pass

        return False

    def _detect_timing_anomalies(self, session: SessionData) -> bool:
        """Detect timing anomalies in session behavior."""
        # Check for unusually fast request patterns
        if session.request_count > 10:
            session_duration = (
                datetime.now(timezone.utc) - session.created_at
            ).total_seconds()
            if (
                session_duration < 60 and session.request_count > 100
            ):  # 100+ requests in under a minute
                return True

        return False

    async def check_concurrent_session_limits(
        self, user_id: str, ip_address: str, all_sessions: list[SessionData]
    ) -> SecurityValidationResult:
        """Check concurrent session limits."""
        reasons = []
        recommendations = []
        score = 0.0

        # Count user sessions
        user_sessions = [
            s
            for s in all_sessions
            if s.user_id == user_id and s.status == SessionStatus.ACTIVE
        ]
        if (
            len(user_sessions)
            >= self.security_config["max_concurrent_sessions_per_user"]
        ):
            reasons.append(
                f"User has {len(user_sessions)} concurrent sessions (limit: {self.security_config['max_concurrent_sessions_per_user']})"
            )
            score += 0.6
            recommendations.append(
                "Terminate oldest sessions or require additional authentication"
            )

        # Count IP sessions
        ip_sessions = [
            s
            for s in all_sessions
            if s.ip_address == ip_address and s.status == SessionStatus.ACTIVE
        ]
        if len(ip_sessions) >= self.security_config["max_concurrent_sessions_per_ip"]:
            reasons.append(
                f"IP has {len(ip_sessions)} concurrent sessions (limit: {self.security_config['max_concurrent_sessions_per_ip']})"
            )
            score += 0.7
            recommendations.append("Block additional sessions from this IP")

        threat_level = (
            SecurityThreatLevel.HIGH
            if score >= 0.6
            else SecurityThreatLevel.MEDIUM
            if score >= 0.3
            else SecurityThreatLevel.LOW
        )

        return SecurityValidationResult(
            is_valid=score < 0.5,
            threat_level=threat_level,
            reasons=reasons,
            recommendations=recommendations,
            score=score,
        )

    def update_user_behavior_profile(self, user_id: str, session_data: SessionData):
        """Update user behavior profile for better detection."""
        if user_id not in self._user_behavior_profiles:
            self._user_behavior_profiles[user_id] = {
                "normal_activity_hours": set(),
                "typical_concurrent_sessions": 1,
                "average_requests_per_minute": 5,
                "common_ip_ranges": set(),
                "typical_session_duration": 1800,  # 30 minutes
            }

        profile = self._user_behavior_profiles[user_id]

        # Update activity hours
        current_hour = datetime.now(timezone.utc).hour
        profile["normal_activity_hours"].add(current_hour)

        # Update IP ranges
        if session_data.ip_address:
            ip_range = ".".join(session_data.ip_address.split(".")[:3]) + ".*"
            profile["common_ip_ranges"].add(ip_range)

        # Update request patterns
        if session_data.request_count > 0:
            session_duration = (
                datetime.now(timezone.utc) - session_data.created_at
            ).total_seconds()
            rpm = (session_data.request_count / session_duration) * 60
            profile["average_requests_per_minute"] = (
                profile["average_requests_per_minute"] + rpm
            ) / 2

    def get_security_recommendations(
        self, validation_result: SecurityValidationResult
    ) -> list[str]:
        """Get actionable security recommendations."""
        recommendations = validation_result.recommendations.copy()

        if validation_result.threat_level == SecurityThreatLevel.CRITICAL:
            recommendations.extend(
                [
                    "Immediately terminate the session",
                    "Force user re-authentication with MFA",
                    "Block the source IP temporarily",
                    "Alert security team",
                ]
            )
        elif validation_result.threat_level == SecurityThreatLevel.HIGH:
            recommendations.extend(
                [
                    "Require additional authentication",
                    "Limit session privileges",
                    "Increase monitoring for this session",
                ]
            )
        elif validation_result.threat_level == SecurityThreatLevel.MEDIUM:
            recommendations.extend(
                [
                    "Request additional verification",
                    "Monitor session closely",
                    "Consider shorter session timeout",
                ]
            )

        return list(set(recommendations))  # Remove duplicates


# Global security validator instance
session_security_validator = SessionSecurityValidator()
