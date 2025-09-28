"""WebSocket client manager for advanced connection lifecycle and subscription management."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union

from .handler import websocket_handler

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class StreamType(Enum):
    """Available data stream types."""

    FRAME = "frame"
    STATE = "state"
    TRAJECTORY = "trajectory"
    ALERT = "alert"
    CONFIG = "config"


@dataclass
class SubscriptionFilter:
    """Filter configuration for stream subscriptions."""

    stream_type: StreamType
    min_fps: Optional[float] = None  # For frame streams
    max_fps: Optional[float] = None  # For frame streams
    quality_level: str = "auto"  # low, medium, high, auto
    include_fields: Optional[list[str]] = None  # Specific fields to include
    exclude_fields: Optional[list[str]] = None  # Specific fields to exclude
    conditions: dict[str, Any] = field(default_factory=dict)  # Custom conditions


@dataclass
class ClientSession:
    """Extended client session information."""

    client_id: str
    user_id: Optional[str]
    connection_state: ConnectionState
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0  # seconds
    subscription_filters: dict[StreamType, SubscriptionFilter] = field(
        default_factory=dict
    )
    permissions: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WebSocketManager:
    """Advanced WebSocket client manager with lifecycle management and smart subscriptions."""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.user_sessions: dict[str, list[str]] = {}  # user_id -> [client_ids]
        self.stream_subscribers: dict[StreamType, set[str]] = {
            stream_type: set() for stream_type in StreamType
        }
        self.event_handlers: dict[str, list[Callable]] = {}
        self.auto_reconnect_enabled = True
        self.reconnect_tasks: dict[str, asyncio.Task] = {}

    async def register_client(
        self,
        client_id: str,
        user_id: Optional[str] = None,
        permissions: Optional[set[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ClientSession:
        """Register a new client session."""
        session = ClientSession(
            client_id=client_id,
            user_id=user_id,
            connection_state=ConnectionState.CONNECTED,
            permissions=permissions or set(),
            metadata=metadata or {},
        )

        self.sessions[client_id] = session

        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(client_id)

        logger.info(f"Client session registered: {client_id} (user: {user_id})")
        await self._emit_event(
            "client_registered",
            {"client_id": client_id, "user_id": user_id, "session": session},
        )

        return session

    async def unregister_client(self, client_id: str):
        """Unregister a client session."""
        if client_id not in self.sessions:
            return

        session = self.sessions[client_id]
        session.connection_state = ConnectionState.DISCONNECTED

        # Remove from stream subscriptions
        for stream_type in StreamType:
            self.stream_subscribers[stream_type].discard(client_id)

        # Remove from user sessions
        if session.user_id and session.user_id in self.user_sessions:
            user_client_ids = self.user_sessions[session.user_id]
            if client_id in user_client_ids:
                user_client_ids.remove(client_id)
            if not user_client_ids:
                del self.user_sessions[session.user_id]

        # Cancel any reconnect tasks
        if client_id in self.reconnect_tasks:
            self.reconnect_tasks[client_id].cancel()
            del self.reconnect_tasks[client_id]

        del self.sessions[client_id]

        logger.info(f"Client session unregistered: {client_id}")
        await self._emit_event(
            "client_unregistered", {"client_id": client_id, "session": session}
        )

    async def subscribe_to_stream(
        self,
        client_id: str,
        stream_type: Union[StreamType, str],
        filter_config: Optional[SubscriptionFilter] = None,
    ) -> bool:
        """Subscribe client to a data stream with optional filtering."""
        if client_id not in self.sessions:
            logger.warning(
                f"Attempted to subscribe unknown client {client_id} to {stream_type}"
            )
            return False

        if isinstance(stream_type, str):
            try:
                stream_type = StreamType(stream_type.lower())
            except ValueError:
                logger.error(f"Invalid stream type: {stream_type}")
                return False

        session = self.sessions[client_id]

        # Check permissions
        required_permission = f"stream:{stream_type.value}"
        if (
            required_permission not in session.permissions
            and "stream:*" not in session.permissions
        ):
            logger.warning(f"Client {client_id} lacks permission for {stream_type}")
            return False

        # Add to subscribers
        self.stream_subscribers[stream_type].add(client_id)

        # Store filter configuration
        if filter_config:
            session.subscription_filters[stream_type] = filter_config

        # Subscribe in the handler as well
        if client_id in websocket_handler.connections:
            connection = websocket_handler.connections[client_id]
            connection.add_subscription(stream_type.value)

        session.last_activity = datetime.now(timezone.utc)

        logger.info(f"Client {client_id} subscribed to {stream_type.value}")
        await self._emit_event(
            "stream_subscribed",
            {
                "client_id": client_id,
                "stream_type": stream_type,
                "filter": filter_config,
            },
        )

        return True

    async def unsubscribe_from_stream(
        self, client_id: str, stream_type: Union[StreamType, str]
    ) -> bool:
        """Unsubscribe client from a data stream."""
        if client_id not in self.sessions:
            return False

        if isinstance(stream_type, str):
            try:
                stream_type = StreamType(stream_type.lower())
            except ValueError:
                return False

        session = self.sessions[client_id]

        # Remove from subscribers
        self.stream_subscribers[stream_type].discard(client_id)

        # Remove filter configuration
        session.subscription_filters.pop(stream_type, None)

        # Unsubscribe in the handler as well
        if client_id in websocket_handler.connections:
            connection = websocket_handler.connections[client_id]
            connection.remove_subscription(stream_type.value)

        session.last_activity = datetime.now(timezone.utc)

        logger.info(f"Client {client_id} unsubscribed from {stream_type.value}")
        await self._emit_event(
            "stream_unsubscribed", {"client_id": client_id, "stream_type": stream_type}
        )

        return True

    async def broadcast_to_stream(
        self,
        stream_type: Union[StreamType, str],
        data: dict[str, Any],
        apply_filters: bool = True,
    ):
        """Broadcast data to all subscribers of a stream with optional filtering."""
        if isinstance(stream_type, str):
            try:
                stream_type = StreamType(stream_type.lower())
            except ValueError:
                logger.error(f"Invalid stream type for broadcast: {stream_type}")
                return

        subscribers = self.stream_subscribers[stream_type].copy()

        if not subscribers:
            return

        # Create base message
        message = {
            "type": stream_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }

        # Send to subscribers, applying filters if requested
        tasks = []
        for client_id in subscribers:
            if client_id not in self.sessions:
                continue

            session = self.sessions[client_id]
            session.last_activity = datetime.now(timezone.utc)

            # Apply filters if configured and requested
            filtered_message = message
            if apply_filters and stream_type in session.subscription_filters:
                filtered_message = await self._apply_message_filter(
                    message, session.subscription_filters[stream_type]
                )
                if filtered_message is None:
                    continue  # Message filtered out

            # Send message
            task = websocket_handler.send_to_client(client_id, filtered_message)
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_count = sum(
                1 for result in results if not result or isinstance(result, Exception)
            )
            if failed_count > 0:
                logger.warning(
                    f"Failed to send to {failed_count}/{len(tasks)} subscribers for {stream_type.value}"
                )

    async def send_to_user(self, user_id: str, message: dict[str, Any]) -> int:
        """Send message to all sessions of a specific user."""
        if user_id not in self.user_sessions:
            return 0

        client_ids = self.user_sessions[user_id].copy()
        tasks = [
            websocket_handler.send_to_client(client_id, message)
            for client_id in client_ids
            if client_id in self.sessions
        ]

        if not tasks:
            return 0

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(
            1 for result in results if result and not isinstance(result, Exception)
        )

        return success_count

    async def send_alert(
        self,
        level: str,
        message: str,
        code: str,
        details: Optional[dict[str, Any]] = None,
        target_clients: Optional[list[str]] = None,
        target_users: Optional[list[str]] = None,
    ):
        """Send alert message to specified clients/users or all subscribers."""
        alert_data = {
            "level": level,
            "message": message,
            "code": code,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        alert_message = {"type": "alert", "data": alert_data}

        if target_clients:
            # Send to specific clients
            tasks = [
                websocket_handler.send_to_client(client_id, alert_message)
                for client_id in target_clients
                if client_id in self.sessions
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        elif target_users:
            # Send to specific users
            for user_id in target_users:
                await self.send_to_user(user_id, alert_message)
        else:
            # Broadcast to all alert subscribers
            await self.broadcast_to_stream(
                StreamType.ALERT, alert_data, apply_filters=False
            )

    async def get_session_info(self, client_id: str) -> Optional[dict[str, Any]]:
        """Get detailed session information."""
        if client_id not in self.sessions:
            return None

        session = self.sessions[client_id]
        connection_info = {}

        # Get connection info from handler if available
        if client_id in websocket_handler.connections:
            connection_info = websocket_handler.connections[
                client_id
            ].get_connection_info()

        return {
            "client_id": session.client_id,
            "user_id": session.user_id,
            "connection_state": session.connection_state.value,
            "reconnect_attempts": session.reconnect_attempts,
            "subscriptions": [s.value for s in session.subscription_filters],
            "permissions": list(session.permissions),
            "metadata": session.metadata,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "uptime": (datetime.now(timezone.utc) - session.created_at).total_seconds(),
            **connection_info,
        }

    async def get_all_sessions(self) -> dict[str, Any]:
        """Get information about all active sessions."""
        sessions_info = []
        for client_id in self.sessions:
            session_info = await self.get_session_info(client_id)
            if session_info:
                sessions_info.append(session_info)

        return {
            "total_sessions": len(self.sessions),
            "total_users": len(self.user_sessions),
            "stream_subscribers": {
                stream_type.value: len(subscribers)
                for stream_type, subscribers in self.stream_subscribers.items()
            },
            "sessions": sessions_info,
        }

    async def handle_connection_lost(self, client_id: str):
        """Handle unexpected connection loss."""
        if client_id not in self.sessions:
            return

        session = self.sessions[client_id]
        session.connection_state = ConnectionState.DISCONNECTED

        logger.warning(f"Connection lost for client {client_id}")

        # Attempt auto-reconnection if enabled
        if (
            self.auto_reconnect_enabled
            and session.reconnect_attempts < session.max_reconnect_attempts
        ):
            await self._schedule_reconnect(client_id)
        else:
            # Give up on reconnection
            await self.unregister_client(client_id)

        await self._emit_event(
            "connection_lost", {"client_id": client_id, "session": session}
        )

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for WebSocket events."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler."""
        if (
            event_type in self.event_handlers
            and handler in self.event_handlers[event_type]
        ):
            self.event_handlers[event_type].remove(handler)

    async def _emit_event(self, event_type: str, data: dict[str, Any]):
        """Emit event to all registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")

    async def _apply_message_filter(
        self, message: dict[str, Any], filter_config: SubscriptionFilter
    ) -> Optional[dict[str, Any]]:
        """Apply filter configuration to a message."""
        # Clone the message to avoid modifying the original
        filtered_message = message.copy()
        data = filtered_message.get("data", {})

        # Apply field inclusion/exclusion
        if filter_config.include_fields:
            data = {k: v for k, v in data.items() if k in filter_config.include_fields}
        elif filter_config.exclude_fields:
            data = {
                k: v for k, v in data.items() if k not in filter_config.exclude_fields
            }

        # Apply custom conditions
        for condition_key, condition_value in filter_config.conditions.items():
            if condition_key in data and data[condition_key] != condition_value:
                return None  # Filter out this message

        # Apply quality level adjustments for frame streams
        if filter_config.stream_type == StreamType.FRAME:
            data = await self._apply_frame_quality_filter(data, filter_config)

        filtered_message["data"] = data
        return filtered_message

    async def _apply_frame_quality_filter(
        self, frame_data: dict[str, Any], filter_config: SubscriptionFilter
    ) -> dict[str, Any]:
        """Apply quality filtering to frame data."""
        # This is a placeholder for frame quality reduction
        # In a real implementation, you would resize/compress the image based on quality_level

        if filter_config.quality_level == "low":
            # Reduce quality for low-end clients
            frame_data["quality"] = "low"
        elif filter_config.quality_level == "medium":
            frame_data["quality"] = "medium"
        elif filter_config.quality_level == "high":
            frame_data["quality"] = "high"
        # "auto" would detect client capabilities

        return frame_data

    async def _schedule_reconnect(self, client_id: str):
        """Schedule automatic reconnection attempt."""
        if client_id not in self.sessions:
            return

        session = self.sessions[client_id]
        session.connection_state = ConnectionState.RECONNECTING
        session.reconnect_attempts += 1

        # Calculate exponential backoff delay
        delay = session.reconnect_delay * (2 ** (session.reconnect_attempts - 1))
        delay = min(delay, 30)  # Cap at 30 seconds

        logger.info(
            f"Scheduling reconnect for {client_id} in {delay:.1f}s (attempt {session.reconnect_attempts})"
        )

        async def reconnect_task():
            try:
                await asyncio.sleep(delay)

                if client_id not in self.sessions:
                    return  # Session was cleaned up

                # In a real implementation, you would trigger reconnection logic here
                # For now, we just update the state
                session.connection_state = ConnectionState.CONNECTING

                await self._emit_event(
                    "reconnect_attempted",
                    {
                        "client_id": client_id,
                        "attempt": session.reconnect_attempts,
                        "delay": delay,
                    },
                )

            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in reconnect task for {client_id}: {e}")
            finally:
                if client_id in self.reconnect_tasks:
                    del self.reconnect_tasks[client_id]

        # Store the task so we can cancel it if needed
        self.reconnect_tasks[client_id] = asyncio.create_task(reconnect_task())


# Global manager instance
websocket_manager = WebSocketManager()
