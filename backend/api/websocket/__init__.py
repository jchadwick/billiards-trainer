"""WebSocket module for real-time data streaming and communication."""

from .broadcaster import (
    BroadcastStats,
    FrameBuffer,
    MessageBroadcaster,
    message_broadcaster,
)
from .handler import WebSocketConnection, WebSocketHandler, websocket_handler
from .manager import (
    ClientSession,
    ConnectionState,
    StreamType,
    SubscriptionFilter,
    WebSocketManager,
    websocket_manager,
)
from .monitoring import ConnectionMonitor, HealthStatus, connection_monitor
from .schemas import (  # Base message types; Data structures; Client request schemas; Response schemas; Message factory functions; Validation functions; Constants
    VALID_ALERT_LEVELS,
    VALID_QUALITY_LEVELS,
    VALID_STREAM_TYPES,
    AlertData,
    AlertLevel,
    BallData,
    CollisionData,
    ConfigData,
    ConnectionData,
    CueData,
    ErrorData,
    FrameData,
    GameStateData,
    MessageType,
    MetricsData,
    PingRequest,
    PongResponse,
    QualityLevel,
    StatusData,
    StatusRequest,
    SubscribedResponse,
    SubscribeRequest,
    TableData,
    TrajectoryData,
    TrajectoryLine,
    UnsubscribedResponse,
    UnsubscribeRequest,
    WebSocketMessage,
    create_alert_message,
    create_config_message,
    create_connection_message,
    create_error_message,
    create_frame_message,
    create_metrics_message,
    create_state_message,
    create_status_message,
    create_trajectory_message,
    validate_client_message,
    validate_websocket_message,
)
from .subscriptions import (
    FilterCondition,
    FilterOperator,
    StreamSubscription,
    SubscriptionManager,
    subscription_manager,
)

# Global instances for easy access
__all__ = [
    # Handler classes and instances
    "WebSocketHandler",
    "WebSocketConnection",
    "websocket_handler",
    # Manager classes and instances
    "WebSocketManager",
    "ConnectionState",
    "StreamType",
    "SubscriptionFilter",
    "ClientSession",
    "websocket_manager",
    # Broadcaster classes and instances
    "MessageBroadcaster",
    "FrameBuffer",
    "BroadcastStats",
    "message_broadcaster",
    # Subscription classes and instances
    "SubscriptionManager",
    "StreamSubscription",
    "FilterCondition",
    "FilterOperator",
    "subscription_manager",
    # Monitoring classes and instances
    "ConnectionMonitor",
    "HealthStatus",
    "connection_monitor",
    # Schema classes
    "WebSocketMessage",
    "MessageType",
    "AlertLevel",
    "QualityLevel",
    "FrameData",
    "BallData",
    "CueData",
    "TableData",
    "GameStateData",
    "TrajectoryLine",
    "CollisionData",
    "TrajectoryData",
    "AlertData",
    "ConfigData",
    "ConnectionData",
    "StatusData",
    "ErrorData",
    "MetricsData",
    "SubscribeRequest",
    "UnsubscribeRequest",
    "PingRequest",
    "StatusRequest",
    "SubscribedResponse",
    "UnsubscribedResponse",
    "PongResponse",
    # Factory and validation functions
    "create_frame_message",
    "create_state_message",
    "create_trajectory_message",
    "create_alert_message",
    "create_config_message",
    "create_connection_message",
    "create_error_message",
    "create_status_message",
    "create_metrics_message",
    "validate_websocket_message",
    "validate_client_message",
    # Constants
    "VALID_STREAM_TYPES",
    "VALID_ALERT_LEVELS",
    "VALID_QUALITY_LEVELS",
]


async def initialize_websocket_system():
    """Initialize the complete WebSocket system."""
    # Start the message broadcaster
    await message_broadcaster.start_streaming()

    # Start the WebSocket handler monitoring
    await websocket_handler.start_monitoring()

    print("WebSocket system initialized successfully")


async def shutdown_websocket_system():
    """Gracefully shutdown the WebSocket system."""
    # Stop the message broadcaster
    await message_broadcaster.stop_streaming()

    # Stop the WebSocket handler monitoring
    await websocket_handler.stop_monitoring()

    # Disconnect all active connections
    active_connections = list(websocket_handler.connections.keys())
    for client_id in active_connections:
        await websocket_handler.disconnect(client_id)

    print("WebSocket system shutdown completed")
