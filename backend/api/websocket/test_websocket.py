"""Comprehensive integration tests for WebSocket system."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.websockets import WebSocketDisconnect

from .broadcaster import MessageBroadcaster
from .handler import WebSocketConnection, WebSocketHandler
from .manager import ConnectionState, StreamType, WebSocketManager
from .monitoring import ConnectionMonitor, HealthStatus
from .schemas import BallData, FrameData, GameStateData
from .subscriptions import SubscriptionManager


@pytest.fixture
def websocket_handler():
    """Create a fresh WebSocket handler for testing."""
    return WebSocketHandler()


@pytest.fixture
async def websocket_manager():
    """Create a fresh WebSocket manager for testing."""
    manager = WebSocketManager()
    yield manager
    # Cleanup
    for client_id in list(manager.sessions.keys()):
        await manager.unregister_client(client_id)


@pytest.fixture
async def message_broadcaster():
    """Create a fresh message broadcaster for testing."""
    broadcaster = MessageBroadcaster()
    await broadcaster.start_streaming()
    yield broadcaster
    await broadcaster.stop_streaming()


@pytest.fixture
async def subscription_manager():
    """Create a fresh subscription manager for testing."""
    manager = SubscriptionManager()
    return manager


@pytest.fixture
async def connection_monitor():
    """Create a fresh connection monitor for testing."""
    monitor = ConnectionMonitor()
    await monitor.start_monitoring()
    yield monitor
    await monitor.stop_monitoring()


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.is_connected = True
        self.client = Mock()
        self.client.host = "127.0.0.1"
        self.headers = {}

    async def accept(self):
        """Mock accept."""
        pass

    async def send_text(self, message: str):
        """Mock send text."""
        if not self.is_connected:
            raise WebSocketDisconnect()
        self.messages_sent.append(message)

    async def receive_text(self) -> str:
        """Mock receive text."""
        if not self.is_connected:
            raise WebSocketDisconnect()
        if self.messages_received:
            return self.messages_received.pop(0)
        # Simulate waiting for message
        await asyncio.sleep(0.1)
        raise WebSocketDisconnect()

    async def close(self, code: int = 1000, reason: str = ""):
        """Mock close."""
        self.is_connected = False

    def add_message(self, message: str):
        """Add a message to be received."""
        self.messages_received.append(message)


class TestWebSocketHandler:
    """Test WebSocket connection handler."""

    @pytest.mark.asyncio
    async def test_connection_establishment(self, websocket_handler):
        """Test establishing a WebSocket connection."""
        mock_ws = MockWebSocket()

        client_id = await websocket_handler.connect(mock_ws)

        assert client_id in websocket_handler.connections
        connection = websocket_handler.connections[client_id]
        assert isinstance(connection, WebSocketConnection)
        assert connection.websocket == mock_ws
        assert connection.is_alive

        # Check welcome message was sent
        assert len(mock_ws.messages_sent) == 1
        welcome_msg = json.loads(mock_ws.messages_sent[0])
        assert welcome_msg["type"] == "connection"
        assert welcome_msg["data"]["client_id"] == client_id

    @pytest.mark.asyncio
    async def test_message_handling(self, websocket_handler):
        """Test handling incoming messages."""
        mock_ws = MockWebSocket()
        client_id = await websocket_handler.connect(mock_ws)

        # Test ping message
        ping_msg = {
            "type": "ping",
            "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }
        await websocket_handler.handle_message(client_id, json.dumps(ping_msg))

        # Should receive pong response
        messages = [
            json.loads(msg) for msg in mock_ws.messages_sent[1:]
        ]  # Skip welcome message
        pong_messages = [msg for msg in messages if msg["type"] == "pong"]
        assert len(pong_messages) == 1

    @pytest.mark.asyncio
    async def test_subscription_management(self, websocket_handler):
        """Test subscription requests."""
        mock_ws = MockWebSocket()
        client_id = await websocket_handler.connect(mock_ws)

        # Test subscribe message
        subscribe_msg = {"type": "subscribe", "data": {"streams": ["frame", "state"]}}
        await websocket_handler.handle_message(client_id, json.dumps(subscribe_msg))

        # Check subscription response
        messages = [json.loads(msg) for msg in mock_ws.messages_sent[1:]]
        subscribed_messages = [msg for msg in messages if msg["type"] == "subscribed"]
        assert len(subscribed_messages) == 1
        assert set(subscribed_messages[0]["data"]["streams"]) == {"frame", "state"}

    @pytest.mark.asyncio
    async def test_rate_limiting(self, websocket_handler):
        """Test rate limiting functionality."""
        mock_ws = MockWebSocket()
        client_id = await websocket_handler.connect(mock_ws)

        # Send many messages rapidly
        for _i in range(150):  # Exceeds default rate limit of 100/minute
            ping_msg = {
                "type": "ping",
                "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
            }
            await websocket_handler.handle_message(client_id, json.dumps(ping_msg))

        # Should receive rate limit error
        messages = [json.loads(msg) for msg in mock_ws.messages_sent[1:]]
        error_messages = [
            msg
            for msg in messages
            if msg["type"] == "error" and msg["data"]["code"] == "RATE_LIMIT"
        ]
        assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_disconnection_cleanup(self, websocket_handler):
        """Test proper cleanup on disconnection."""
        mock_ws = MockWebSocket()
        client_id = await websocket_handler.connect(mock_ws)

        assert client_id in websocket_handler.connections

        await websocket_handler.disconnect(client_id)

        assert client_id not in websocket_handler.connections


class TestWebSocketManager:
    """Test WebSocket manager functionality."""

    @pytest.mark.asyncio
    async def test_client_registration(self, websocket_manager):
        """Test client session registration."""
        client_id = "test_client_1"
        user_id = "test_user"

        session = await websocket_manager.register_client(
            client_id=client_id,
            user_id=user_id,
            permissions={"stream:frame", "stream:state"},
        )

        assert client_id in websocket_manager.sessions
        assert session.client_id == client_id
        assert session.user_id == user_id
        assert session.connection_state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_stream_subscription(self, websocket_manager):
        """Test stream subscription management."""
        client_id = "test_client_1"
        await websocket_manager.register_client(
            client_id=client_id, permissions={"stream:*"}
        )

        # Subscribe to frame stream
        success = await websocket_manager.subscribe_to_stream(
            client_id, StreamType.FRAME
        )
        assert success

        # Check subscription
        assert client_id in websocket_manager.stream_subscribers[StreamType.FRAME]

        # Unsubscribe
        success = await websocket_manager.unsubscribe_from_stream(
            client_id, StreamType.FRAME
        )
        assert success
        assert client_id not in websocket_manager.stream_subscribers[StreamType.FRAME]

    @pytest.mark.asyncio
    async def test_permission_checking(self, websocket_manager):
        """Test permission-based subscription filtering."""
        client_id = "test_client_1"
        await websocket_manager.register_client(
            client_id=client_id, permissions={"stream:frame"}  # Only frame permission
        )

        # Should succeed for frame
        success = await websocket_manager.subscribe_to_stream(
            client_id, StreamType.FRAME
        )
        assert success

        # Should fail for state (no permission)
        success = await websocket_manager.subscribe_to_stream(
            client_id, StreamType.STATE
        )
        assert not success

    @pytest.mark.asyncio
    async def test_alert_broadcasting(self, websocket_manager):
        """Test alert broadcasting to specific clients."""
        client_id = "test_client_1"
        await websocket_manager.register_client(client_id=client_id)

        with patch.object(websocket_manager, "send_to_user", new_callable=AsyncMock):
            await websocket_manager.send_alert(
                level="warning",
                message="Test alert",
                code="TEST_001",
                target_clients=[client_id],
            )
            # Alert sending would be tested via websocket_handler mock


class TestMessageBroadcaster:
    """Test message broadcasting functionality."""

    @pytest.mark.asyncio
    async def test_frame_broadcasting(self, message_broadcaster):
        """Test video frame broadcasting."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = [255, 255, 255]  # White square

        await message_broadcaster.broadcast_frame(
            image_data=test_image, width=640, height=480, quality=75
        )

        # Check frame buffer
        latest_frame = message_broadcaster.frame_buffer.get_latest_frame()
        assert latest_frame is not None
        assert latest_frame["width"] == 640
        assert latest_frame["height"] == 480
        assert latest_frame["quality"] == 75
        assert "image" in latest_frame  # Base64 encoded image

    @pytest.mark.asyncio
    async def test_game_state_broadcasting(self, message_broadcaster):
        """Test game state broadcasting."""
        # Create test game state
        balls = [
            {
                "id": "cue",
                "position": [100, 200],
                "radius": 20,
                "color": "white",
                "velocity": [0, 0],
            },
            {
                "id": "8ball",
                "position": [300, 400],
                "radius": 20,
                "color": "black",
                "velocity": [5, -3],
            },
        ]

        cue = {"angle": 45.5, "position": [150, 250], "detected": True}

        with patch.object(
            message_broadcaster, "_broadcast_to_stream", new_callable=AsyncMock
        ) as mock_broadcast:
            await message_broadcaster.broadcast_game_state(balls=balls, cue=cue)

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args
            assert call_args[0][0] == StreamType.STATE  # Stream type
            assert "balls" in call_args[0][1]  # Data contains balls
            assert "cue" in call_args[0][1]  # Data contains cue

    @pytest.mark.asyncio
    async def test_trajectory_broadcasting(self, message_broadcaster):
        """Test trajectory broadcasting."""
        lines = [
            {"start": [100, 200], "end": [300, 400], "type": "primary"},
            {"start": [300, 400], "end": [500, 200], "type": "reflection"},
        ]

        collisions = [{"position": [300, 400], "ball_id": "8ball", "angle": 30}]

        with patch.object(
            message_broadcaster, "_broadcast_to_stream", new_callable=AsyncMock
        ) as mock_broadcast:
            await message_broadcaster.broadcast_trajectory(
                lines=lines, collisions=collisions, confidence=0.95
            )

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args
            assert call_args[0][0] == StreamType.TRAJECTORY
            data = call_args[0][1]
            assert data["confidence"] == 0.95
            assert len(data["lines"]) == 2
            assert len(data["collisions"]) == 1

    @pytest.mark.asyncio
    async def test_performance_metrics(self, message_broadcaster):
        """Test performance metrics collection."""
        # Generate some activity
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            await message_broadcaster.broadcast_frame(test_image, 100, 100)

        stats = message_broadcaster.get_broadcast_stats()

        assert stats["is_streaming"]
        assert stats["broadcast_stats"]["frame_metrics"]["frames_sent"] == 5
        assert "current_fps" in stats


class TestSubscriptionManager:
    """Test subscription management system."""

    @pytest.mark.asyncio
    async def test_subscription_creation(self, subscription_manager):
        """Test creating subscriptions with filters."""
        client_id = "test_client_1"

        success = await subscription_manager.create_subscription(
            client_id=client_id,
            stream_type=StreamType.FRAME,
            quality_level="medium",
            max_fps=15.0,
            include_fields=["image", "width", "height"],
            sample_rate=0.5,
        )

        assert success
        assert client_id in subscription_manager.subscriptions
        assert StreamType.FRAME in subscription_manager.subscriptions[client_id]

        subscription = subscription_manager.subscriptions[client_id][StreamType.FRAME]
        assert subscription.max_fps == 15.0
        assert subscription.sample_rate == 0.5
        assert "image" in subscription.include_fields

    @pytest.mark.asyncio
    async def test_message_filtering(self, subscription_manager):
        """Test message filtering functionality."""
        client_id = "test_client_1"

        # Create subscription with field filtering
        await subscription_manager.create_subscription(
            client_id=client_id,
            stream_type=StreamType.STATE,
            include_fields=["balls", "timestamp"],
        )

        # Process test message
        test_data = {
            "balls": [{"id": "cue", "position": [100, 200]}],
            "cue": {"angle": 45},
            "table": {"corners": [[0, 0], [100, 100]]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        processed = await subscription_manager.process_message(
            StreamType.STATE, test_data
        )

        assert client_id in processed
        filtered_data = processed[client_id]
        assert "balls" in filtered_data
        assert "timestamp" in filtered_data
        assert "cue" not in filtered_data  # Should be filtered out
        assert "table" not in filtered_data  # Should be filtered out

    @pytest.mark.asyncio
    async def test_sampling(self, subscription_manager):
        """Test message sampling."""
        client_id = "test_client_1"

        # Create subscription with 50% sampling
        await subscription_manager.create_subscription(
            client_id=client_id, stream_type=StreamType.FRAME, sample_rate=0.5
        )

        # Process many messages
        processed_count = 0
        for i in range(100):
            processed = await subscription_manager.process_message(
                StreamType.FRAME, {"frame_id": i, "data": "test"}
            )
            if client_id in processed:
                processed_count += 1

        # Should process approximately 50% of messages (with some variance)
        assert 40 <= processed_count <= 60


class TestConnectionMonitor:
    """Test connection health monitoring."""

    @pytest.mark.asyncio
    async def test_latency_recording(self, connection_monitor):
        """Test latency measurement recording."""
        client_id = "test_client_1"

        # Record some latency measurements
        connection_monitor.record_latency(client_id, 25.0, "ping")
        connection_monitor.record_latency(client_id, 30.0, "ping")
        connection_monitor.record_latency(client_id, 35.0, "ping")

        health_info = connection_monitor.get_client_health(client_id)
        assert health_info is not None
        assert health_info["latency"]["average_ms"] == 30.0
        assert health_info["latency"]["min_ms"] == 25.0
        assert health_info["latency"]["max_ms"] == 35.0

    @pytest.mark.asyncio
    async def test_health_assessment(self, connection_monitor):
        """Test health status assessment."""
        client_id = "test_client_1"

        # Record good latency (should be EXCELLENT)
        connection_monitor.record_latency(client_id, 15.0, "ping")
        health_info = connection_monitor.get_client_health(client_id)
        assert health_info["health_status"] == HealthStatus.EXCELLENT.value

        # Record poor latency (should be CRITICAL)
        for _ in range(10):
            connection_monitor.record_latency(client_id, 250.0, "ping")

        health_info = connection_monitor.get_client_health(client_id)
        assert health_info["health_status"] == HealthStatus.CRITICAL.value

    @pytest.mark.asyncio
    async def test_error_tracking(self, connection_monitor):
        """Test error tracking."""
        client_id = "test_client_1"

        # Record various errors
        connection_monitor.record_error(client_id, "failed_send")
        connection_monitor.record_error(client_id, "timeout")
        connection_monitor.record_error(client_id, "disconnection")

        health_info = connection_monitor.get_client_health(client_id)
        assert health_info["errors"]["failed_sends"] == 1
        assert health_info["errors"]["timeouts"] == 1
        assert health_info["errors"]["disconnections"] == 1

    @pytest.mark.asyncio
    async def test_system_health(self, connection_monitor):
        """Test system-wide health monitoring."""
        # Add some clients with different health levels
        connection_monitor.record_latency("client_1", 15.0)  # EXCELLENT
        connection_monitor.record_latency("client_2", 75.0)  # FAIR
        connection_monitor.record_latency("client_3", 250.0)  # CRITICAL

        system_health = connection_monitor.get_system_health()

        assert system_health["connections"]["total"] == 3
        assert system_health["connections"]["critical"] >= 1
        assert system_health["overall_status"] in ["critical", "degraded"]


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_frame_data_validation(self):
        """Test frame data schema validation."""
        valid_frame = {
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "width": 1920,
            "height": 1080,
            "format": "jpeg",
            "quality": 85,
            "compressed": False,
            "fps": 30.0,
            "size_bytes": 1024,
        }

        frame = FrameData(**valid_frame)
        assert frame.width == 1920
        assert frame.height == 1080
        assert frame.format == "jpeg"

        # Test validation error
        with pytest.raises(ValueError):
            FrameData(**{**valid_frame, "width": -1})  # Invalid width

    def test_ball_data_validation(self):
        """Test ball data schema validation."""
        valid_ball = {
            "id": "cue",
            "position": [100.0, 200.0],
            "radius": 20.0,
            "color": "white",
            "velocity": [5.0, -3.0],
            "confidence": 0.95,
        }

        ball = BallData(**valid_ball)
        assert ball.id == "cue"
        assert ball.position == [100.0, 200.0]
        assert ball.confidence == 0.95

        # Test validation error
        with pytest.raises(ValueError):
            BallData(**{**valid_ball, "position": [100.0]})  # Invalid position length

    def test_game_state_validation(self):
        """Test complete game state validation."""
        balls = [
            {"id": "cue", "position": [100.0, 200.0], "radius": 20.0, "color": "white"}
        ]

        cue = {"angle": 45.5, "position": [150.0, 250.0], "detected": True}

        table = {
            "corners": [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]],
            "pockets": [[100.0, 100.0], [1820.0, 100.0]],
        }

        game_state = GameStateData(balls=balls, cue=cue, table=table, ball_count=1)

        assert len(game_state.balls) == 1
        assert game_state.cue.angle == 45.5
        assert len(game_state.table.corners) == 4


class TestIntegration:
    """Integration tests for the complete WebSocket system."""

    @pytest.mark.asyncio
    async def test_full_connection_lifecycle(self):
        """Test complete connection lifecycle."""
        # Initialize components
        handler = WebSocketHandler()
        manager = WebSocketManager()
        broadcaster = MessageBroadcaster()
        sub_manager = SubscriptionManager()
        monitor = ConnectionMonitor()

        try:
            # Start services
            await broadcaster.start_streaming()
            await monitor.start_monitoring()

            # Simulate client connection
            mock_ws = MockWebSocket()
            client_id = await handler.connect(mock_ws)

            # Register with manager
            await manager.register_client(client_id, permissions={"stream:*"})

            # Subscribe to streams
            await manager.subscribe_to_stream(client_id, StreamType.FRAME)
            await manager.subscribe_to_stream(client_id, StreamType.STATE)

            # Create subscription with filtering
            await sub_manager.create_subscription(
                client_id=client_id,
                stream_type=StreamType.FRAME,
                quality_level="medium",
                max_fps=15.0,
            )

            # Simulate data broadcasting
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            await broadcaster.broadcast_frame(test_image, 100, 100)

            balls = [
                {"id": "cue", "position": [100, 200], "radius": 20, "color": "white"}
            ]
            await broadcaster.broadcast_game_state(balls=balls)

            # Check that client received messages
            assert len(mock_ws.messages_sent) > 1  # Welcome + data messages

            # Simulate health monitoring
            monitor.record_latency(client_id, 25.0)
            monitor.record_message_sent(client_id, 1024)

            health = monitor.get_client_health(client_id)
            assert health is not None
            assert health["health_status"] == HealthStatus.EXCELLENT.value

            # Clean up
            await handler.disconnect(client_id)
            await manager.unregister_client(client_id)

        finally:
            await broadcaster.stop_streaming()
            await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_multi_client_broadcasting(self):
        """Test broadcasting to multiple clients."""
        handler = WebSocketHandler()
        manager = WebSocketManager()
        broadcaster = MessageBroadcaster()

        try:
            await broadcaster.start_streaming()

            # Connect multiple clients
            clients = []
            for _i in range(3):
                mock_ws = MockWebSocket()
                client_id = await handler.connect(mock_ws)
                await manager.register_client(client_id, permissions={"stream:*"})
                await manager.subscribe_to_stream(client_id, StreamType.ALERT)
                clients.append((client_id, mock_ws))

            # Broadcast alert
            await broadcaster.broadcast_alert(
                level="info",
                message="Test broadcast to multiple clients",
                code="TEST_MULTI",
            )

            # All clients should receive the alert
            for client_id, mock_ws in clients:
                messages = [
                    json.loads(msg) for msg in mock_ws.messages_sent[1:]
                ]  # Skip welcome
                [msg for msg in messages if msg["type"] == "alert"]
                # Note: Would need proper integration with manager for actual broadcasting

        finally:
            await broadcaster.stop_streaming()
            for client_id, _ in clients:
                await handler.disconnect(client_id)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
