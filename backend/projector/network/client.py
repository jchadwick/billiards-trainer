"""WebSocket client for projector module to communicate with backend API server."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MessageType(str, Enum):
    """WebSocket message types (from API schema)."""

    # Incoming data streams
    FRAME = "frame"
    STATE = "state"
    TRAJECTORY = "trajectory"
    ALERT = "alert"
    CONFIG = "config"
    METRICS = "metrics"

    # Connection management
    CONNECTION = "connection"
    PING = "ping"
    PONG = "pong"

    # Subscription management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # Status and control
    STATUS = "status"
    ERROR = "error"


@dataclass
class ClientStats:
    """WebSocket client connection and performance statistics."""

    # Connection metrics
    connection_time: Optional[float] = None
    last_ping_time: Optional[float] = None
    last_pong_time: Optional[float] = None
    latency_ms: float = 0.0
    uptime_seconds: float = 0.0

    # Message metrics
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    # Stream metrics
    frames_received: int = 0
    states_received: int = 0
    trajectories_received: int = 0
    alerts_received: int = 0
    config_updates_received: int = 0

    # Error metrics
    connection_errors: int = 0
    message_errors: int = 0
    reconnection_attempts: int = 0

    # Performance tracking
    latency_history: list[float] = field(default_factory=list)
    average_latency_ms: float = 0.0
    message_rate_per_second: float = 0.0

    def update_latency(self, latency_ms: float):
        """Update latency measurements."""
        self.latency_ms = latency_ms
        self.latency_history.append(latency_ms)

        # Keep only last 100 measurements
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]

        # Calculate average
        if self.latency_history:
            self.average_latency_ms = sum(self.latency_history) / len(
                self.latency_history
            )

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "connection": {
                "uptime_seconds": self.uptime_seconds,
                "latency_ms": self.latency_ms,
                "average_latency_ms": self.average_latency_ms,
                "connection_errors": self.connection_errors,
                "reconnection_attempts": self.reconnection_attempts,
            },
            "messages": {
                "sent": self.messages_sent,
                "received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "rate_per_second": self.message_rate_per_second,
                "errors": self.message_errors,
            },
            "streams": {
                "frames": self.frames_received,
                "states": self.states_received,
                "trajectories": self.trajectories_received,
                "alerts": self.alerts_received,
                "config_updates": self.config_updates_received,
            },
        }


class WebSocketClient:
    """Advanced WebSocket client for projector-to-backend communication.

    Features:
    - Automatic reconnection with exponential backoff
    - Message queuing and ordering
    - Subscription management
    - Comprehensive error handling
    - Performance monitoring
    - Real-time data streaming
    """

    def __init__(
        self,
        api_url: str = "ws://localhost:8000/ws",
        client_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        reconnect_enabled: bool = True,
        max_reconnect_attempts: int = -1,  # -1 = infinite
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ping_interval: float = 30.0,
        message_queue_size: int = 1000,
    ):
        """Initialize WebSocket client.

        Args:
            api_url: WebSocket server URL
            client_id: Unique client identifier (auto-generated if None)
            auth_token: Authentication token for secure connections
            reconnect_enabled: Enable automatic reconnection
            max_reconnect_attempts: Maximum reconnection attempts (-1 = infinite)
            initial_reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
            ping_interval: Ping interval in seconds
            message_queue_size: Maximum message queue size
        """
        # Connection configuration
        self.api_url = api_url
        self.client_id = client_id or f"projector_{int(time.time())}"
        self.auth_token = auth_token

        # Build WebSocket URL with authentication
        self._ws_url = self._build_websocket_url()

        # Reconnection configuration
        self.reconnect_enabled = reconnect_enabled
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.current_reconnect_delay = initial_reconnect_delay

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False

        # Statistics and monitoring
        self.stats = ClientStats()
        self.ping_interval = ping_interval

        # Message handling
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=message_queue_size)
        self.subscriptions: set[str] = set()
        self.sequence_number = 0

        # Event handlers
        self.message_handlers: dict[str, list[Callable]] = {}
        self.connection_handlers: list[Callable] = []
        self.error_handlers: list[Callable] = []

        # Background tasks
        self._tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        logger.info(f"WebSocket client initialized for {self.api_url}")

    def _build_websocket_url(self) -> str:
        """Build WebSocket URL with authentication parameters."""
        url = self.api_url
        params = []

        if self.auth_token:
            params.append(f"token={self.auth_token}")

        if params:
            url += "?" + "&".join(params)

        return url

    async def connect(self) -> bool:
        """Connect to the WebSocket server.

        Returns:
            True if connection successful, False otherwise
        """
        if self.connected:
            logger.warning("Already connected")
            return True

        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to {self.api_url}...")

            # Establish WebSocket connection
            self.websocket = await websockets.connect(
                self._ws_url,
                ping_interval=None,  # We handle ping manually
                ping_timeout=10,
                close_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                compression=None,  # Disable compression for real-time data
            )

            # Update connection state
            self.connected = True
            self.state = ConnectionState.CONNECTED
            self.stats.connection_time = time.time()
            self.current_reconnect_delay = self.initial_reconnect_delay

            # Start background tasks
            await self._start_background_tasks()

            # Notify connection handlers
            await self._notify_connection_handlers(True)

            logger.info(f"Successfully connected to {self.api_url}")
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.stats.connection_errors += 1
            logger.error(f"Failed to connect to {self.api_url}: {e}")
            await self._notify_error_handlers(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        """Gracefully disconnect from the WebSocket server."""
        logger.info("Disconnecting from WebSocket server...")

        # Signal shutdown
        self._shutdown_event.set()

        # Update state
        self.connected = False
        self.state = ConnectionState.DISCONNECTED

        # Cancel background tasks
        await self._stop_background_tasks()

        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

        # Notify connection handlers
        await self._notify_connection_handlers(False)

        logger.info("Disconnected from WebSocket server")

    async def subscribe(self, streams: list[str]) -> bool:
        """Subscribe to data streams.

        Args:
            streams: List of stream types to subscribe to

        Returns:
            True if subscription request sent successfully
        """
        if not self.connected:
            logger.error("Cannot subscribe: not connected")
            return False

        message = {"type": MessageType.SUBSCRIBE, "data": {"streams": streams}}

        success = await self._send_message(message)
        if success:
            self.subscriptions.update(streams)
            logger.info(f"Subscribed to streams: {streams}")

        return success

    async def unsubscribe(self, streams: list[str]) -> bool:
        """Unsubscribe from data streams.

        Args:
            streams: List of stream types to unsubscribe from

        Returns:
            True if unsubscription request sent successfully
        """
        if not self.connected:
            logger.error("Cannot unsubscribe: not connected")
            return False

        message = {"type": MessageType.UNSUBSCRIBE, "data": {"streams": streams}}

        success = await self._send_message(message)
        if success:
            self.subscriptions.difference_update(streams)
            logger.info(f"Unsubscribed from streams: {streams}")

        return success

    async def ping(self) -> bool:
        """Send ping to server.

        Returns:
            True if ping sent successfully
        """
        if not self.connected:
            return False

        self.stats.last_ping_time = time.time()
        message = {
            "type": MessageType.PING,
            "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }

        return await self._send_message(message)

    async def request_status(self) -> bool:
        """Request connection status from server.

        Returns:
            True if status request sent successfully
        """
        if not self.connected:
            return False

        message = {"type": "get_status", "data": {"include_details": True}}

        return await self._send_message(message)

    def add_message_handler(
        self, message_type: str, handler: Callable[[dict[str, Any]], None]
    ):
        """Add message handler for specific message type.

        Args:
            message_type: Message type to handle
            handler: Handler function that accepts message data
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []

        self.message_handlers[message_type].append(handler)
        logger.debug(f"Added handler for {message_type}")

    def add_connection_handler(self, handler: Callable[[bool], None]):
        """Add connection state change handler.

        Args:
            handler: Handler function that accepts connection state (bool)
        """
        self.connection_handlers.append(handler)
        logger.debug("Added connection handler")

    def add_error_handler(self, handler: Callable[[str], None]):
        """Add error handler.

        Args:
            handler: Handler function that accepts error message
        """
        self.error_handlers.append(handler)
        logger.debug("Added error handler")

    def get_connection_info(self) -> dict[str, Any]:
        """Get current connection information.

        Returns:
            Dictionary with connection details
        """
        uptime = 0.0
        if self.stats.connection_time:
            uptime = time.time() - self.stats.connection_time

        self.stats.uptime_seconds = uptime

        return {
            "client_id": self.client_id,
            "state": self.state.value,
            "connected": self.connected,
            "api_url": self.api_url,
            "subscriptions": list(self.subscriptions),
            "stats": self.stats.get_summary(),
            "uptime_seconds": uptime,
        }

    async def _send_message(self, message: dict[str, Any]) -> bool:
        """Send message to server.

        Args:
            message: Message dictionary to send

        Returns:
            True if message sent successfully
        """
        if not self.connected or not self.websocket:
            return False

        try:
            # Add sequence number
            message["sequence"] = self.sequence_number
            self.sequence_number += 1

            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now(timezone.utc).isoformat()

            # Serialize and send
            message_str = json.dumps(message)
            await self.websocket.send(message_str)

            # Update statistics
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message_str.encode())

            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats.message_errors += 1
            await self._notify_error_handlers(f"Send error: {e}")
            return False

    async def _receive_loop(self):
        """Main message receiving loop."""
        logger.debug("Starting message receive loop")

        while self.connected and not self._shutdown_event.is_set():
            try:
                if not self.websocket:
                    break

                # Receive message with timeout
                message_str = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)

                # Update statistics
                self.stats.messages_received += 1
                self.stats.bytes_received += len(message_str.encode())

                # Parse and handle message
                try:
                    message = json.loads(message_str)
                    await self._handle_message(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    self.stats.message_errors += 1

            except asyncio.TimeoutError:
                # Normal timeout, continue
                continue

            except ConnectionClosed:
                logger.warning("WebSocket connection closed by server")
                break

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                self.stats.message_errors += 1
                await self._notify_error_handlers(f"Receive error: {e}")
                break

        # Connection lost, attempt reconnection if enabled
        if self.connected:
            self.connected = False
            self.state = ConnectionState.DISCONNECTED
            await self._notify_connection_handlers(False)

            if self.reconnect_enabled and not self._shutdown_event.is_set():
                await self._reconnect()

    async def _handle_message(self, message: dict[str, Any]):
        """Handle incoming message from server.

        Args:
            message: Parsed message dictionary
        """
        message_type = message.get("type")
        data = message.get("data", {})

        # Handle special message types
        if message_type == MessageType.PONG:
            await self._handle_pong(data)
        elif message_type == MessageType.ERROR:
            await self._handle_error(data)
        elif message_type == MessageType.SUBSCRIBED:
            await self._handle_subscribed(data)
        elif message_type == MessageType.UNSUBSCRIBED:
            await self._handle_unsubscribed(data)

        # Update stream-specific statistics
        if message_type == MessageType.FRAME:
            self.stats.frames_received += 1
        elif message_type == MessageType.STATE:
            self.stats.states_received += 1
        elif message_type == MessageType.TRAJECTORY:
            self.stats.trajectories_received += 1
        elif message_type == MessageType.ALERT:
            self.stats.alerts_received += 1
        elif message_type == MessageType.CONFIG:
            self.stats.config_updates_received += 1

        # Call registered handlers
        if message_type in self.message_handlers:
            for handler in self.message_handlers[message_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in message handler for {message_type}: {e}")

    async def _handle_pong(self, data: dict[str, Any]):
        """Handle pong response."""
        self.stats.last_pong_time = time.time()

        if self.stats.last_ping_time:
            latency_ms = (self.stats.last_pong_time - self.stats.last_ping_time) * 1000
            self.stats.update_latency(latency_ms)

    async def _handle_error(self, data: dict[str, Any]):
        """Handle error message from server."""
        error_code = data.get("code", "UNKNOWN")
        error_message = data.get("message", "Unknown error")

        logger.error(f"Server error {error_code}: {error_message}")
        await self._notify_error_handlers(f"Server error {error_code}: {error_message}")

    async def _handle_subscribed(self, data: dict[str, Any]):
        """Handle subscription confirmation."""
        streams = data.get("streams", [])
        all_subscriptions = data.get("all_subscriptions", [])

        self.subscriptions = set(all_subscriptions)
        logger.info(f"Subscription confirmed for: {streams}")

    async def _handle_unsubscribed(self, data: dict[str, Any]):
        """Handle unsubscription confirmation."""
        streams = data.get("streams", [])
        all_subscriptions = data.get("all_subscriptions", [])

        self.subscriptions = set(all_subscriptions)
        logger.info(f"Unsubscription confirmed for: {streams}")

    async def _ping_loop(self):
        """Periodic ping loop."""
        logger.debug("Starting ping loop")

        while self.connected and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.ping_interval)
                if self.connected:
                    await self.ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if not self.reconnect_enabled:
            return

        self.state = ConnectionState.RECONNECTING
        attempt = 0

        while (
            not self.connected
            and not self._shutdown_event.is_set()
            and (
                self.max_reconnect_attempts == -1
                or attempt < self.max_reconnect_attempts
            )
        ):
            attempt += 1
            self.stats.reconnection_attempts += 1

            logger.info(
                f"Reconnection attempt {attempt} in {self.current_reconnect_delay:.1f}s..."
            )

            try:
                await asyncio.sleep(self.current_reconnect_delay)

                if await self.connect():
                    # Resubscribe to previous streams
                    if self.subscriptions:
                        await self.subscribe(list(self.subscriptions))
                    return

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {e}")

            # Exponential backoff
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * 2, self.max_reconnect_delay
            )

        # Failed to reconnect
        logger.error("All reconnection attempts failed")
        self.state = ConnectionState.ERROR
        await self._notify_error_handlers("Reconnection failed")

    async def _start_background_tasks(self):
        """Start background tasks."""
        # Message receiving task
        receive_task = asyncio.create_task(self._receive_loop())
        self._tasks.add(receive_task)

        # Ping task
        ping_task = asyncio.create_task(self._ping_loop())
        self._tasks.add(ping_task)

        logger.debug("Background tasks started")

    async def _stop_background_tasks(self):
        """Stop all background tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        logger.debug("Background tasks stopped")

    async def _notify_connection_handlers(self, connected: bool):
        """Notify all connection handlers."""
        for handler in self.connection_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(connected)
                else:
                    handler(connected)
            except Exception as e:
                logger.error(f"Error in connection handler: {e}")

    async def _notify_error_handlers(self, error_message: str):
        """Notify all error handlers."""
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_message)
                else:
                    handler(error_message)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
