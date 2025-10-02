"""WebSocket connection handler with message processing."""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, status

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Represents a single WebSocket connection with metadata."""

    def __init__(
        self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None
    ):
        self.websocket = websocket
        self.client_id = client_id
        self.user_id = user_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.subscriptions: set[str] = set()
        self.message_count = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.quality_score = 1.0  # 0.0 to 1.0, higher is better
        self.is_alive = True

    def update_ping(self):
        """Update last ping timestamp."""
        self.last_ping = time.time()

    def update_pong(self):
        """Update last pong timestamp and recalculate quality."""
        self.last_pong = time.time()
        # Simple quality calculation based on ping-pong latency
        latency = self.last_pong - self.last_ping
        if latency < 0.05:  # < 50ms
            self.quality_score = 1.0
        elif latency < 0.1:  # < 100ms
            self.quality_score = 0.8
        elif latency < 0.2:  # < 200ms
            self.quality_score = 0.6
        else:
            self.quality_score = 0.3

    def add_subscription(self, stream_type: str):
        """Subscribe to a data stream."""
        self.subscriptions.add(stream_type)

    def remove_subscription(self, stream_type: str):
        """Unsubscribe from a data stream."""
        self.subscriptions.discard(stream_type)

    def is_subscribed(self, stream_type: str) -> bool:
        """Check if subscribed to a stream type."""
        return stream_type in self.subscriptions

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information for monitoring."""
        return {
            "client_id": self.client_id,
            "user_id": self.user_id,
            "connected_at": self.connected_at.isoformat(),
            "uptime": (datetime.now(timezone.utc) - self.connected_at).total_seconds(),
            "subscriptions": list(self.subscriptions),
            "message_count": self.message_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "quality_score": self.quality_score,
            "last_ping_latency": self.last_pong - self.last_ping,
            "is_alive": self.is_alive,
        }


class WebSocketHandler:
    """Advanced WebSocket connection handler with subscriptions and monitoring."""

    def __init__(self):
        self.connections: dict[str, WebSocketConnection] = {}
        self.user_connections: dict[str, list[str]] = {}  # user_id -> [client_ids]
        self.ping_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        self.max_message_rate = 100  # messages per minute per connection
        self.rate_limit_windows: dict[str, list[float]] = {}
        self._monitoring_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_connections())

    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

            self._monitoring_task = None

    async def authenticate_connection(
        self, websocket: WebSocket, token: Optional[str] = None
    ) -> Optional[str]:
        """Accept WebSocket connection without authentication."""
        # No authentication - return a default user_id
        logger.info("WebSocket connection accepted (no authentication)")
        return "unauthenticated_user"

    async def connect(self, websocket: WebSocket, token: Optional[str] = None) -> str:
        """Accept new WebSocket connection."""
        try:
            # Authenticate the connection
            user_id = await self.authenticate_connection(websocket, token)

            # Accept the WebSocket connection
            await websocket.accept()

            # Generate unique client ID
            client_id = str(uuid.uuid4())

            # Create connection object
            connection = WebSocketConnection(websocket, client_id, user_id)

            # Store the connection
            self.connections[client_id] = connection

            # Track user connections
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = []
                self.user_connections[user_id].append(client_id)

            # Initialize rate limiting
            self.rate_limit_windows[client_id] = []

            logger.info(f"WebSocket connected: {client_id} (user: {user_id})")

            # Start monitoring if not already running
            await self.start_monitoring()

            # Send welcome message
            await self.send_to_client(
                client_id,
                {
                    "type": "connection",
                    "data": {
                        "client_id": client_id,
                        "status": "connected",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                },
            )

            return client_id

        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(
                status_code=403, detail="Connection authentication failed"
            )

    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection."""
        if client_id not in self.connections:
            return

        connection = self.connections[client_id]
        connection.is_alive = False

        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            user_client_ids = self.user_connections[connection.user_id]
            if client_id in user_client_ids:
                user_client_ids.remove(client_id)
            if not user_client_ids:
                del self.user_connections[connection.user_id]

        # Clean up rate limiting
        if client_id in self.rate_limit_windows:
            del self.rate_limit_windows[client_id]

        # Remove connection
        del self.connections[client_id]

        logger.info(f"WebSocket disconnected: {client_id}")

        # Stop monitoring if no connections remain
        if not self.connections:
            await self.stop_monitoring()

    async def handle_message(self, client_id: str, message: str):
        """Handle incoming WebSocket message from client."""
        if client_id not in self.connections:
            return

        connection = self.connections[client_id]

        # Rate limiting check
        if not self._check_rate_limit(client_id):
            await self.send_to_client(
                client_id,
                {
                    "type": "error",
                    "data": {
                        "code": "RATE_LIMIT",
                        "message": "Message rate limit exceeded",
                    },
                },
            )
            return

        try:
            # Parse message
            data = json.loads(message)
            connection.bytes_received += len(message.encode())
            connection.message_count += 1

            # Handle different message types
            message_type = data.get("type")

            if message_type == "ping":
                await self._handle_ping(client_id, data)
            elif message_type == "subscribe":
                await self._handle_subscribe(client_id, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(client_id, data)
            elif message_type == "get_status":
                await self._handle_get_status(client_id, data)
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                await self.send_to_client(
                    client_id,
                    {
                        "type": "error",
                        "data": {
                            "code": "UNKNOWN_MESSAGE_TYPE",
                            "message": f"Unknown message type: {message_type}",
                        },
                    },
                )

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {client_id}: {message}")
            await self.send_to_client(
                client_id,
                {
                    "type": "error",
                    "data": {"code": "INVALID_JSON", "message": "Invalid JSON format"},
                },
            )
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(
                client_id,
                {
                    "type": "error",
                    "data": {
                        "code": "MESSAGE_HANDLER_ERROR",
                        "message": "Internal message handling error",
                    },
                },
            )

    async def send_to_client(self, client_id: str, message: dict[str, Any]) -> bool:
        """Send message to specific client."""
        if client_id not in self.connections:
            return False

        connection = self.connections[client_id]

        try:
            # Add timestamp and sequence if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(timezone.utc).isoformat()
            if "sequence" not in message:
                message["sequence"] = connection.message_count

            message_str = json.dumps(message)
            await connection.websocket.send_text(message_str)

            connection.bytes_sent += len(message_str.encode())
            return True

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
            return False
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            return False

    async def broadcast_to_subscribers(self, stream_type: str, message: dict[str, Any]):
        """Broadcast message to all clients subscribed to a stream type."""
        if not self.connections:
            return

        subscribers = [
            client_id
            for client_id, conn in self.connections.items()
            if conn.is_subscribed(stream_type) and conn.is_alive
        ]

        if not subscribers:
            return

        # Send to all subscribers concurrently
        tasks = [self.send_to_client(client_id, message) for client_id in subscribers]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        failed_count = sum(
            1 for result in results if not result or isinstance(result, Exception)
        )
        if failed_count > 0:
            logger.warning(
                f"Failed to send to {failed_count}/{len(subscribers)} subscribers for {stream_type}"
            )

    async def broadcast_message(self, message: dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.connections:
            return

        # Send to all connected clients concurrently
        tasks = [
            self.send_to_client(client_id, message) for client_id in self.connections
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        failed_count = sum(
            1 for result in results if not result or isinstance(result, Exception)
        )
        if failed_count > 0:
            logger.warning(
                f"Failed to broadcast to {failed_count}/{len(tasks)} clients"
            )

    def get_connection_stats(self) -> dict[str, Any]:
        """Get overall connection statistics."""
        if not self.connections:
            return {
                "total_connections": 0,
                "authenticated_connections": 0,
                "average_quality": 0.0,
                "total_messages": 0,
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
            }

        authenticated_count = sum(
            1 for conn in self.connections.values() if conn.user_id
        )
        avg_quality = sum(
            conn.quality_score for conn in self.connections.values()
        ) / len(self.connections)
        total_messages = sum(conn.message_count for conn in self.connections.values())
        total_bytes_sent = sum(conn.bytes_sent for conn in self.connections.values())
        total_bytes_received = sum(
            conn.bytes_received for conn in self.connections.values()
        )

        return {
            "total_connections": len(self.connections),
            "authenticated_connections": authenticated_count,
            "average_quality": round(avg_quality, 3),
            "total_messages": total_messages,
            "total_bytes_sent": total_bytes_sent,
            "total_bytes_received": total_bytes_received,
            "connections": [
                conn.get_connection_info() for conn in self.connections.values()
            ],
        }

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        window = self.rate_limit_windows.get(client_id, [])

        # Remove old entries (older than 1 minute)
        window = [timestamp for timestamp in window if now - timestamp < 60]

        # Check if under limit
        if len(window) >= self.max_message_rate:
            return False

        # Add current timestamp
        window.append(now)
        self.rate_limit_windows[client_id] = window

        return True

    async def _handle_ping(self, client_id: str, data: dict[str, Any]):
        """Handle ping message."""
        connection = self.connections[client_id]
        connection.update_ping()

        await self.send_to_client(
            client_id,
            {
                "type": "pong",
                "data": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "quality_score": connection.quality_score,
                },
            },
        )

        connection.update_pong()

    async def _handle_subscribe(self, client_id: str, data: dict[str, Any]):
        """Handle subscription request."""
        stream_types = data.get("data", {}).get("streams", [])

        if not isinstance(stream_types, list):
            await self.send_to_client(
                client_id,
                {
                    "type": "error",
                    "data": {
                        "code": "INVALID_SUBSCRIBE_FORMAT",
                        "message": "streams must be an array",
                    },
                },
            )
            return

        connection = self.connections[client_id]
        valid_streams = ["frame", "state", "trajectory", "alert", "config"]

        added_streams = []
        for stream_type in stream_types:
            if stream_type in valid_streams:
                connection.add_subscription(stream_type)
                added_streams.append(stream_type)
            else:
                logger.warning(
                    f"Invalid stream type requested by {client_id}: {stream_type}"
                )

        await self.send_to_client(
            client_id,
            {
                "type": "subscribed",
                "data": {
                    "streams": added_streams,
                    "all_subscriptions": list(connection.subscriptions),
                },
            },
        )

    async def _handle_unsubscribe(self, client_id: str, data: dict[str, Any]):
        """Handle unsubscription request."""
        stream_types = data.get("data", {}).get("streams", [])

        if not isinstance(stream_types, list):
            await self.send_to_client(
                client_id,
                {
                    "type": "error",
                    "data": {
                        "code": "INVALID_UNSUBSCRIBE_FORMAT",
                        "message": "streams must be an array",
                    },
                },
            )
            return

        connection = self.connections[client_id]

        removed_streams = []
        for stream_type in stream_types:
            if connection.is_subscribed(stream_type):
                connection.remove_subscription(stream_type)
                removed_streams.append(stream_type)

        await self.send_to_client(
            client_id,
            {
                "type": "unsubscribed",
                "data": {
                    "streams": removed_streams,
                    "all_subscriptions": list(connection.subscriptions),
                },
            },
        )

    async def _handle_get_status(self, client_id: str, data: dict[str, Any]):
        """Handle status request."""
        connection = self.connections[client_id]

        await self.send_to_client(
            client_id, {"type": "status", "data": connection.get_connection_info()}
        )

    async def _monitor_connections(self):
        """Background task to monitor connection health."""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)

                if not self.connections:
                    continue

                current_time = time.time()
                disconnected_clients = []

                # Check for stale connections
                for client_id, connection in self.connections.items():
                    time_since_pong = current_time - connection.last_pong

                    if time_since_pong > self.connection_timeout:
                        logger.warning(
                            f"Connection {client_id} timed out (no pong for {time_since_pong:.1f}s)"
                        )
                        disconnected_clients.append(client_id)
                    elif time_since_pong > self.ping_interval:
                        # Send ping to check if connection is alive
                        await self.send_to_client(
                            client_id,
                            {
                                "type": "ping",
                                "data": {
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                },
                            },
                        )

                # Disconnect stale connections
                for client_id in disconnected_clients:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")


# Global handler instance
websocket_handler = WebSocketHandler()
