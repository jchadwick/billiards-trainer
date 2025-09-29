"""FastAPI WebSocket endpoints with advanced features and rate limiting."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.routing import APIRouter

from .broadcaster import message_broadcaster
from .handler import websocket_handler
from .manager import websocket_manager
from .monitoring import connection_monitor
from .schemas import VALID_STREAM_TYPES, SubscribeRequest, UnsubscribeRequest
from .subscriptions import subscription_manager

logger = logging.getLogger(__name__)

# Create router for WebSocket endpoints
websocket_router = APIRouter(tags=["websocket"])


# Rate limiting for WebSocket connections
class ConnectionRateLimiter:
    """Rate limiter for WebSocket connections per IP."""

    def __init__(self):
        self.connection_attempts = {}  # IP -> [timestamps]
        self.max_attempts = 10  # Max connections per IP per minute
        self.window_seconds = 60

    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if IP is within rate limits."""
        import time

        current_time = time.time()

        if client_ip not in self.connection_attempts:
            self.connection_attempts[client_ip] = []

        # Clean old attempts
        attempts = self.connection_attempts[client_ip]
        self.connection_attempts[client_ip] = [
            timestamp
            for timestamp in attempts
            if current_time - timestamp < self.window_seconds
        ]

        # Check limit
        if len(self.connection_attempts[client_ip]) >= self.max_attempts:
            return False

        # Add current attempt
        self.connection_attempts[client_ip].append(current_time)
        return True


rate_limiter = ConnectionRateLimiter()


# Dependency for extracting client IP
async def get_client_ip(websocket: WebSocket) -> str:
    """Extract client IP address from WebSocket."""
    # Try to get real IP from headers (if behind proxy)
    real_ip = websocket.headers.get("x-real-ip")
    if real_ip:
        return real_ip

    forwarded_for = websocket.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    # Fallback to client host
    return websocket.client.host if websocket.client else "unknown"


# Dependency for extracting authentication token
async def extract_websocket_token(
    websocket: WebSocket, token: Optional[str] = Query(None)
) -> Optional[str]:
    """Extract authentication token from WebSocket connection.

    Supports token extraction from:
    1. Query parameter: ?token=<jwt_token>
    2. Authorization header: Authorization: Bearer <jwt_token>
    3. Custom header: X-Auth-Token: <jwt_token>
    """
    # First priority: Query parameter (for backward compatibility)
    if token:
        return token

    # Second priority: Authorization header
    auth_header = websocket.headers.get("authorization")
    if auth_header:
        # Handle "Bearer <token>" format
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        # Handle raw token in authorization header
        return auth_header

    # Third priority: Custom X-Auth-Token header
    auth_token_header = websocket.headers.get("x-auth-token")
    if auth_token_header:
        return auth_token_header

    return None


# Authentication dependency with unauthenticated mode support
async def authenticate_websocket(
    token: Optional[str] = None, websocket: Optional[WebSocket] = None
) -> tuple[Optional[str], dict[str, Any]]:
    """Authenticate WebSocket connection and return user_id and user_info."""
    from ..dependencies import _get_unauthenticated_user, _is_auth_enabled
    from ..middleware.authentication import verify_jwt_token

    # Check if authentication is enabled
    if not _is_auth_enabled():
        # Return unauthenticated user with admin privileges
        user_info = _get_unauthenticated_user("admin")
        return user_info["user_id"], user_info

    # Authentication is enabled, validate token
    if not token:
        logger.warning(
            "WebSocket connection attempted without token when auth is enabled"
        )
        return None, {}

    try:
        # Validate JWT token
        payload = verify_jwt_token(token)
        user_info = {
            "user_id": payload.get("sub"),
            "username": payload.get("username"),
            "role": payload.get("role", "viewer"),
            "auth_type": "jwt",
            "permissions": payload.get("permissions", []),
        }
        logger.info(
            f"WebSocket authenticated user: {user_info['username']} ({user_info['role']})"
        )
        return user_info["user_id"], user_info

    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {str(e)}")
        return None, {}


@websocket_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_ip: str = Depends(get_client_ip),
    token: Optional[str] = Depends(extract_websocket_token),
):
    """Main WebSocket endpoint for real-time data streaming.

    Supports:
    - Authentication via:
      * Query parameter: ?token=<jwt_token>
      * Authorization header: Authorization: Bearer <jwt_token>
      * Custom header: X-Auth-Token: <jwt_token>
    - Multiple concurrent connections
    - Selective stream subscriptions
    - Real-time health monitoring
    - Automatic reconnection handling
    """
    # Rate limiting check
    if not rate_limiter.check_rate_limit(client_ip):
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Rate limit exceeded"
        )
        return

    client_id = None
    try:
        # Authenticate connection
        user_id, user_info = await authenticate_websocket(token, websocket)

        # Check if authentication is required but failed
        from ..dependencies import _is_auth_enabled

        if _is_auth_enabled() and not user_id:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION, reason="Authentication required"
            )
            return

        # Establish WebSocket connection
        client_id = await websocket_handler.connect(websocket, token)

        # Determine permissions based on user role
        if user_info.get("role") == "admin":
            permissions = {"stream:*", "control:*", "config:*"}
        elif user_info.get("role") == "operator":
            permissions = {"stream:*", "control:basic"}
        else:
            permissions = {"stream:frame", "stream:state"}

        # Register client in manager
        await websocket_manager.register_client(
            client_id=client_id,
            user_id=user_id,
            permissions=permissions,
            metadata={
                "ip_address": client_ip,
                "user_agent": websocket.headers.get("user-agent"),
                "auth_type": user_info.get("auth_type", "none"),
                "role": user_info.get("role", "viewer"),
            },
        )

        # Start monitoring for this client
        connection_monitor.client_metrics[
            client_id
        ] = connection_monitor.client_metrics.get(
            client_id,
            connection_monitor.client_metrics.__class__.__dict__[
                "__dataclass_fields__"
            ]["client_id"].default_factory(),
        )

        logger.info(f"WebSocket connection established: {client_id} from {client_ip}")

        # Message handling loop
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                connection_monitor.record_message_received(
                    client_id, len(message.encode())
                )

                # Handle the message
                await websocket_handler.handle_message(client_id, message)

            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected normally")
                break

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from {client_id}: {e}")
                await websocket_handler.send_to_client(
                    client_id,
                    {
                        "type": "error",
                        "data": {
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON format",
                        },
                    },
                )

            except Exception as e:
                logger.error(f"Error handling WebSocket message from {client_id}: {e}")
                connection_monitor.record_error(client_id, "protocol_error")

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        if client_id:
            connection_monitor.record_error(client_id, "disconnection")

    finally:
        # Clean up connection
        if client_id:
            await websocket_handler.disconnect(client_id)
            await websocket_manager.unregister_client(client_id)
            await subscription_manager.remove_all_subscriptions(client_id)

            logger.info(f"WebSocket connection cleaned up: {client_id}")


# REST endpoints for WebSocket management
@websocket_router.get("/connections", response_model=dict[str, Any])
async def get_active_connections():
    """Get information about all active WebSocket connections."""
    try:
        stats = websocket_handler.get_connection_stats()
        sessions = await websocket_manager.get_all_sessions()

        return {
            "handler_stats": stats,
            "session_info": sessions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting connections: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve connection information"
        )


@websocket_router.get("/connections/{client_id}", response_model=dict[str, Any])
async def get_connection_info(client_id: str):
    """Get detailed information about a specific connection."""
    try:
        session_info = await websocket_manager.get_session_info(client_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Connection not found")

        health_info = connection_monitor.get_client_health(client_id)
        subscription_info = await subscription_manager.get_subscription_info(client_id)

        return {
            "session": session_info,
            "health": health_info,
            "subscriptions": subscription_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting connection info for {client_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve connection information"
        )


@websocket_router.post("/connections/{client_id}/subscribe")
async def subscribe_client_to_streams(client_id: str, request: SubscribeRequest):
    """Subscribe a client to specific data streams."""
    try:
        if client_id not in websocket_handler.connections:
            raise HTTPException(status_code=404, detail="Connection not found")

        results = []
        for stream_type in request.streams:
            if stream_type not in VALID_STREAM_TYPES:
                results.append(
                    {
                        "stream": stream_type,
                        "success": False,
                        "error": "Invalid stream type",
                    }
                )
                continue

            success = await websocket_manager.subscribe_to_stream(
                client_id, stream_type
            )
            results.append({"stream": stream_type, "success": success})

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update subscriptions")


@websocket_router.post("/connections/{client_id}/unsubscribe")
async def unsubscribe_client_from_streams(client_id: str, request: UnsubscribeRequest):
    """Unsubscribe a client from specific data streams."""
    try:
        if client_id not in websocket_handler.connections:
            raise HTTPException(status_code=404, detail="Connection not found")

        results = []
        for stream_type in request.streams:
            success = await websocket_manager.unsubscribe_from_stream(
                client_id, stream_type
            )
            results.append({"stream": stream_type, "success": success})

        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update subscriptions")


@websocket_router.post("/connections/{client_id}/disconnect")
async def disconnect_client(client_id: str):
    """Forcefully disconnect a specific client."""
    try:
        if client_id not in websocket_handler.connections:
            raise HTTPException(status_code=404, detail="Connection not found")

        await websocket_handler.disconnect(client_id)
        return {"message": f"Client {client_id} disconnected successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting client {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to disconnect client")


@websocket_router.get("/health", response_model=dict[str, Any])
async def get_websocket_health():
    """Get overall WebSocket system health."""
    try:
        system_health = connection_monitor.get_system_health()
        broadcaster_stats = message_broadcaster.get_broadcast_stats()
        subscription_stats = subscription_manager.get_performance_stats()

        return {
            "system_health": system_health,
            "broadcaster_stats": broadcaster_stats,
            "subscription_stats": subscription_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve health information"
        )


@websocket_router.get("/health/summary", response_model=dict[str, Any])
async def get_health_summary():
    """Get a summary of health status for all connections."""
    try:
        return connection_monitor.get_health_summary()
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve health summary")


@websocket_router.post("/broadcast/frame")
async def broadcast_test_frame(
    width: int = 1920, height: int = 1080, quality: int = 85
):
    """Broadcast a test frame to all frame subscribers."""
    try:
        # Generate a simple test image (black with white text)
        import cv2
        import numpy as np

        # Create a black image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add timestamp text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img, f"Test Frame - {timestamp}", (50, 100), font, 2, (255, 255, 255), 3
        )

        # Broadcast the frame
        await message_broadcaster.broadcast_frame(
            image_data=img, width=width, height=height, quality=quality
        )

        return {
            "message": "Test frame broadcasted successfully",
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.error(f"Error broadcasting test frame: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast test frame")


@websocket_router.post("/broadcast/alert")
async def broadcast_test_alert(
    level: str = "info", message: str = "Test alert message", code: str = "TEST_001"
):
    """Broadcast a test alert to all alert subscribers."""
    try:
        await message_broadcaster.broadcast_alert(
            level=level,
            message=message,
            code=code,
            details={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        return {"message": "Test alert broadcasted successfully"}

    except Exception as e:
        logger.error(f"Error broadcasting test alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast test alert")


@websocket_router.post("/system/start")
async def start_websocket_system():
    """Start the WebSocket system services."""
    try:
        # Start all WebSocket services
        await message_broadcaster.start_streaming()
        await websocket_handler.start_monitoring()
        await connection_monitor.start_monitoring()

        return {"message": "WebSocket system started successfully"}

    except Exception as e:
        logger.error(f"Error starting WebSocket system: {e}")
        raise HTTPException(status_code=500, detail="Failed to start WebSocket system")


@websocket_router.post("/system/stop")
async def stop_websocket_system():
    """Stop the WebSocket system services."""
    try:
        # Stop all WebSocket services
        await message_broadcaster.stop_streaming()
        await websocket_handler.stop_monitoring()
        await connection_monitor.stop_monitoring()

        # Disconnect all clients
        active_connections = list(websocket_handler.connections.keys())
        for client_id in active_connections:
            await websocket_handler.disconnect(client_id)

        return {"message": "WebSocket system stopped successfully"}

    except Exception as e:
        logger.error(f"Error stopping WebSocket system: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop WebSocket system")


@websocket_router.get("/metrics", response_model=dict[str, Any])
async def get_websocket_metrics():
    """Get detailed WebSocket performance metrics."""
    try:
        # Get comprehensive metrics from both broadcaster and websocket handler
        metrics_data = {
            "broadcast_stats": {
                "messages_sent": message_broadcaster.broadcast_stats.messages_sent,
                "bytes_sent": message_broadcaster.broadcast_stats.bytes_sent,
                "failed_sends": message_broadcaster.broadcast_stats.failed_sends,
                "average_latency": message_broadcaster.broadcast_stats.average_latency,
                "peak_latency": message_broadcaster.broadcast_stats.peak_latency,
                "compression_enabled": message_broadcaster.broadcast_stats.compression_enabled,
            },
            "frame_metrics": {
                "frames_sent": message_broadcaster.broadcast_stats.frame_metrics.frames_sent,
                "bytes_sent": message_broadcaster.broadcast_stats.frame_metrics.bytes_sent,
                "compression_ratio": (
                    message_broadcaster.broadcast_stats.frame_metrics.compression_ratio
                ),
                "average_latency": (
                    message_broadcaster.broadcast_stats.frame_metrics.average_latency
                ),
                "dropped_frames": message_broadcaster.broadcast_stats.frame_metrics.dropped_frames,
                "target_fps": message_broadcaster.broadcast_stats.frame_metrics.target_fps,
                "actual_fps": message_broadcaster.broadcast_stats.frame_metrics.actual_fps,
            },
            "connection_stats": websocket_handler.get_connection_stats(),
            "broadcaster_stats": message_broadcaster.get_broadcast_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return metrics_data

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Event handlers for monitoring alerts
async def handle_health_alert(alert: dict[str, Any]):
    """Handle health monitoring alerts."""
    logger.warning(f"Health alert: {alert}")

    # Broadcast critical alerts to all connected clients
    if alert.get("type") == "critical_connections":
        await message_broadcaster.broadcast_alert(
            level="warning",
            message=alert["message"],
            code="HEALTH_ALERT",
            details=alert,
        )


# Initialize monitoring
connection_monitor.add_alert_handler(handle_health_alert)


# Export the router
__all__ = ["websocket_router"]
