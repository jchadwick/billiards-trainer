"""UDP broadcaster for sending game state to projector application."""

import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UDPBroadcaster:
    """UDP broadcaster for sending game state to projector application."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        """Initialize UDP broadcaster.

        Args:
            host: Target host for UDP packets (projector machine). Defaults to env var or 192.168.1.31
            port: Target port. Defaults to env var or 9999
            enabled: Whether broadcasting is enabled. Defaults to env var or True
        """
        self.host = host or os.getenv("UDP_PROJECTOR_HOST", "192.168.1.31")
        self.port = int(port or os.getenv("UDP_PROJECTOR_PORT", "9999"))
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("UDP_PROJECTOR_ENABLED", "true").lower() == "true"
        )

        self.socket: Optional[socket.socket] = None
        self.sequence = 0
        self.stats = {"messages_sent": 0, "bytes_sent": 0, "errors": 0}

        if self.enabled:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                logger.info(f"UDP broadcaster initialized: {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to create UDP socket: {e}")
                self.enabled = False
        else:
            logger.info("UDP broadcaster disabled via configuration")

    def _get_next_sequence(self) -> int:
        """Get next sequence number."""
        self.sequence += 1
        return self.sequence

    def _convert_ball_format(self, ball: dict) -> dict:
        """Convert backend ball format to projector format.

        Backend format: {"position": {"x": 100, "y": 200}}
        Projector format: {"position": [100, 200]}
        """
        converted = {
            "id": ball.get("id"),
            "is_moving": ball.get("is_moving", False),
        }

        # Convert position dict to array if needed
        if "position" in ball:
            pos = ball["position"]
            if isinstance(pos, dict):
                converted["position"] = [pos.get("x", 0), pos.get("y", 0)]
            else:
                converted["position"] = pos

        # Convert velocity dict to array if needed
        if "velocity" in ball:
            vel = ball["velocity"]
            if isinstance(vel, dict):
                converted["velocity"] = [vel.get("x", 0), vel.get("y", 0)]
            else:
                converted["velocity"] = vel

        # Optional fields
        for field in ["number", "is_cue_ball", "type"]:
            if field in ball:
                converted[field] = ball[field]

        return converted

    def send_message(self, message_type: str, data: dict[str, Any]) -> bool:
        """Send UDP message to projector.

        Args:
            message_type: Message type (state, motion, trajectory, alert, config)
            data: Message data payload

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or not self.socket:
            return False

        try:
            message = {
                "type": message_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": self._get_next_sequence(),
                "data": data,
            }

            # Encode to JSON
            json_str = json.dumps(message)
            json_bytes = json_str.encode("utf-8")

            # Check packet size (UDP max 65507 bytes)
            if len(json_bytes) > 65507:
                logger.warning(f"Message too large for UDP: {len(json_bytes)} bytes")
                return False

            # Send UDP packet
            self.socket.sendto(json_bytes, (self.host, self.port))

            # Update stats
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(json_bytes)

            logger.debug(
                f"Sent UDP {message_type} to {self.host}:{self.port} ({len(json_bytes)} bytes)"
            )
            return True

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"UDP send error: {e}")
            return False

    def send_game_state(
        self,
        balls: list[dict],
        cue: Optional[dict] = None,
        table: Optional[dict] = None,
    ) -> bool:
        """Send game state update.

        Args:
            balls: List of ball objects
            cue: Optional cue stick data
            table: Optional table data

        Returns:
            True if sent successfully
        """
        # Convert ball format
        converted_balls = [self._convert_ball_format(ball) for ball in balls]

        data = {"balls": converted_balls}

        if cue:
            data["cue"] = cue
        if table:
            data["table"] = table

        return self.send_message("state", data)

    def send_trajectory(
        self, lines: list[dict], collisions: Optional[list[dict]] = None
    ) -> bool:
        """Send trajectory update.

        Args:
            lines: List of trajectory line segments
            collisions: Optional collision points

        Returns:
            True if sent successfully
        """
        return self.send_message(
            "trajectory", {"lines": lines, "collisions": collisions or []}
        )

    def send_aim_line(
        self, start: dict, end: dict, confidence: Optional[float] = None
    ) -> bool:
        """Send aim line update.

        Args:
            start: Start point {"x": float, "y": float}
            end: End point {"x": float, "y": float}
            confidence: Optional confidence score 0-1

        Returns:
            True if sent successfully
        """
        data = {"start": start, "end": end}
        if confidence is not None:
            data["confidence"] = confidence

        return self.send_message("aim_line", data)

    def send_ghost_ball(
        self, position: dict, target_ball_id: Optional[str] = None
    ) -> bool:
        """Send ghost ball position.

        Args:
            position: Ghost ball position {"x": float, "y": float}
            target_ball_id: Optional target ball ID

        Returns:
            True if sent successfully
        """
        data = {"position": position}
        if target_ball_id:
            data["target_ball_id"] = target_ball_id

        return self.send_message("ghost_ball", data)

    def send_alert(
        self, level: str, message: str, details: Optional[dict] = None
    ) -> bool:
        """Send alert message.

        Args:
            level: Alert level (info, warning, error)
            message: Alert message text
            details: Optional additional details

        Returns:
            True if sent successfully
        """
        return self.send_message(
            "alert", {"level": level, "message": message, "details": details or {}}
        )

    def send_collision(
        self,
        ball1_id: str,
        ball2_id: Optional[str],
        position: dict,
        impact_force: Optional[float] = None,
    ) -> bool:
        """Send collision event.

        Args:
            ball1_id: First ball ID
            ball2_id: Second ball ID (None for cushion collision)
            position: Collision position {"x": float, "y": float}
            impact_force: Optional impact force magnitude

        Returns:
            True if sent successfully
        """
        data = {"ball1_id": ball1_id, "position": position}

        if ball2_id:
            data["ball2_id"] = ball2_id
            data["collision_type"] = "ball"
        else:
            data["collision_type"] = "cushion"

        if impact_force is not None:
            data["impact_force"] = impact_force

        return self.send_message("collision", data)

    def get_stats(self) -> dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            **self.stats,
            "target": f"{self.host}:{self.port}",
            "sequence": self.sequence,
            "enabled": self.enabled,
        }

    def close(self):
        """Close UDP socket."""
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("UDP broadcaster closed")


# Global instance - will be initialized with environment variables
udp_broadcaster = UDPBroadcaster()
