"""Message broadcaster for efficient WebSocket data streaming with performance optimization."""

import asyncio
import base64
import contextlib
import json
import logging
import time
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np

from backend.config import config_manager

from .handler import websocket_handler
from .manager import StreamType, websocket_manager

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class FrameMetrics:
    """Frame streaming performance metrics."""

    frames_sent: int = 0
    bytes_sent: int = 0
    compression_ratio: float = 0.0
    average_latency: float = 0.0
    dropped_frames: int = 0
    target_fps: float = 30.0
    actual_fps: float = 0.0


@dataclass
class BroadcastStats:
    """Broadcasting performance statistics."""

    messages_sent: int = 0
    bytes_sent: int = 0
    failed_sends: int = 0
    average_latency: float = 0.0
    peak_latency: float = 0.0
    compression_enabled: bool = True
    frame_metrics: FrameMetrics = None
    validation_failures: int = 0

    def __post_init__(self):
        """Initialize frame metrics if not provided."""
        if self.frame_metrics is None:
            self.frame_metrics = FrameMetrics()


class FrameBuffer:
    """Circular buffer for frame data with automatic cleanup."""

    def __init__(
        self,
        max_size: int = None,
    ):
        """Initialize frame buffer with maximum size.

        Args:
            max_size: Maximum number of frames to store in buffer.
        """
        if max_size is None:
            max_size = config_manager.get(
                "api.websocket.broadcaster.frame_buffer.max_size", 100
            )
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.frame_times = deque(maxlen=max_size)

    def add_frame(self, frame_data: dict[str, Any]):
        """Add frame to buffer with timestamp."""
        current_time = time.time()
        self.frames.append(frame_data)
        self.frame_times.append(current_time)

    def get_latest_frame(self) -> Optional[dict[str, Any]]:
        """Get the most recent frame."""
        return self.frames[-1] if self.frames else None

    def get_frame_rate(self) -> float:
        """Calculate current frame rate."""
        if len(self.frame_times) < 2:
            return 0.0

        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0

        return (len(self.frame_times) - 1) / time_span

    def cleanup_old_frames(self, max_age_seconds: float = None):
        """Remove frames older than specified age."""
        if max_age_seconds is None:
            max_age_seconds = config_manager.get(
                "api.websocket.broadcaster.frame_buffer.max_age_seconds", 5.0
            )
        current_time = time.time()
        while (
            self.frame_times and (current_time - self.frame_times[0]) > max_age_seconds
        ):
            self.frames.popleft()
            self.frame_times.popleft()


class MessageBroadcaster:
    """High-performance WebSocket message broadcaster with streaming optimization."""

    def __init__(self):
        """Initialize message broadcaster with frame buffering and performance tracking."""
        self.frame_buffer = FrameBuffer()
        self.broadcast_stats = BroadcastStats()
        self.compression_threshold = config_manager.get(
            "api.websocket.broadcaster.compression.threshold_bytes", 1024
        )
        self.compression_level = config_manager.get(
            "api.websocket.broadcaster.compression.level", 6
        )
        self.compression_ratio_threshold = config_manager.get(
            "api.websocket.broadcaster.compression.ratio_threshold", 0.9
        )
        frame_queue_size = config_manager.get(
            "api.websocket.broadcaster.frame_queue.max_size", 10
        )
        self.frame_queue = asyncio.Queue(maxsize=frame_queue_size)
        self.broadcast_tasks: dict[str, asyncio.Task] = {}
        self.fps_limiter = defaultdict(lambda: 0.0)  # client_id -> last_frame_time
        self.sequence_counters = defaultdict(int)  # stream_type -> sequence_number
        self.is_streaming = False

    async def start_streaming(self):
        """Start the broadcasting service."""
        if self.is_streaming:
            return

        self.is_streaming = True

        # Start frame processing task
        self.broadcast_tasks["frame_processor"] = asyncio.create_task(
            self._process_frame_queue()
        )

        # Start cleanup task
        self.broadcast_tasks["cleanup"] = asyncio.create_task(self._cleanup_task())

        logger.info("Message broadcaster started")

    async def stop_streaming(self):
        """Stop the broadcasting service."""
        if not self.is_streaming:
            return

        self.is_streaming = False

        # Cancel all tasks
        for _task_name, task in self.broadcast_tasks.items():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.broadcast_tasks.clear()
        logger.info("Message broadcaster stopped")

    async def broadcast_frame(
        self,
        image_data: Union[np.ndarray, bytes],
        width: int,
        height: int,
        format: str = "JPEG",
        quality: int = 85,
        fps: Optional[float] = None,
    ):
        """Broadcast video frame to all subscribed clients."""
        try:
            # Convert image data to base64
            if isinstance(image_data, np.ndarray):
                # Assume it's an OpenCV image (BGR format)
                import cv2

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded_img = cv2.imencode(".jpg", image_data, encode_param)
                image_bytes = encoded_img.tobytes()
            else:
                image_bytes = image_data

            # Compress if enabled and data is large enough
            compressed = False
            if len(image_bytes) > self.compression_threshold:
                try:
                    compressed_data = zlib.compress(
                        image_bytes, level=self.compression_level
                    )
                    if (
                        len(compressed_data)
                        < len(image_bytes) * self.compression_ratio_threshold
                    ):
                        image_bytes = compressed_data
                        compressed = True
                        self.broadcast_stats.frame_metrics.compression_ratio = len(
                            compressed_data
                        ) / len(image_bytes)
                except Exception as e:
                    logger.warning(f"Frame compression failed: {e}")

            # Encode to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            frame_data = {
                "image": image_base64,
                "width": width,
                "height": height,
                "format": format.lower(),
                "quality": quality,
                "compressed": compressed,
                "fps": fps or self.frame_buffer.get_frame_rate(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": self._get_next_sequence("frame"),
                "size_bytes": len(image_bytes),
            }

            # Add to frame buffer
            self.frame_buffer.add_frame(frame_data)

            # DISABLED: Frame broadcasting via WebSocket to prevent browser crashes from large base64 images
            # Frames are still buffered for internal metrics and potential future use
            # If you need frame streaming, implement a separate optimized endpoint (e.g., MJPEG stream)
            # if self.is_streaming and not self.frame_queue.full():
            #     try:
            #         self.frame_queue.put_nowait(frame_data)
            #     except asyncio.QueueFull:
            #         self.broadcast_stats.frame_metrics.dropped_frames += 1
            #         logger.warning("Frame queue full, dropping frame")

            # Update metrics
            self.broadcast_stats.frame_metrics.frames_sent += 1
            self.broadcast_stats.frame_metrics.bytes_sent += len(image_bytes)
            self.broadcast_stats.frame_metrics.actual_fps = (
                self.frame_buffer.get_frame_rate()
            )

        except Exception as e:
            logger.error(f"Error broadcasting frame: {e}")

    async def broadcast_game_state(
        self,
        balls: list[dict[str, Any]],
        cue: Optional[dict[str, Any]] = None,
        table: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Broadcast current game state to subscribers with validation.

        Args:
            balls: List of ball dictionaries with position and other required fields
            cue: Optional cue stick information
            table: Optional table information
            timestamp: Optional timestamp for the state

        Returns:
            None. Logs warnings and returns early if validation fails.
        """
        # Validate balls parameter
        if not isinstance(balls, list):
            logger.warning(
                f"broadcast_game_state: 'balls' must be a list, got {type(balls).__name__}"
            )
            self.broadcast_stats.validation_failures += 1
            return

        if len(balls) == 0:
            logger.warning(
                "broadcast_game_state: 'balls' list is empty, skipping broadcast"
            )
            self.broadcast_stats.validation_failures += 1
            return

        # Validate each ball has required fields
        required_ball_fields = ["position"]
        for i, ball in enumerate(balls):
            if not isinstance(ball, dict):
                logger.warning(
                    f"broadcast_game_state: ball at index {i} is not a dict, got {type(ball).__name__}"
                )
                self.broadcast_stats.validation_failures += 1
                return

            missing_fields = [
                field for field in required_ball_fields if field not in ball
            ]
            if missing_fields:
                logger.warning(
                    f"broadcast_game_state: ball at index {i} missing required fields: {missing_fields}. "
                    f"Ball data: {ball}"
                )
                self.broadcast_stats.validation_failures += 1
                return

            # Validate position field
            position = ball["position"]
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                logger.warning(
                    f"broadcast_game_state: ball at index {i} has invalid position format. "
                    f"Expected list/tuple with at least 2 elements, got {type(position).__name__} with value {position}"
                )
                self.broadcast_stats.validation_failures += 1
                return

        state_data = {
            "balls": balls,
            "cue": cue,
            "table": table,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "sequence": self._get_next_sequence("state"),
            "ball_count": len(balls),
        }

        # Broadcast via WebSocket
        await self._broadcast_to_stream(StreamType.STATE, state_data)

    async def broadcast_trajectory(
        self,
        lines: list[dict[str, Any]],
        collisions: Optional[list[dict[str, Any]]] = None,
        confidence: float = 1.0,
        calculation_time_ms: float = 0.0,
    ):
        """Broadcast trajectory calculations to subscribers with validation.

        Args:
            lines: List of trajectory line dictionaries (must have at least 2 points)
            collisions: Optional list of collision dictionaries
            confidence: Confidence score for the trajectory calculation
            calculation_time_ms: Time taken to calculate the trajectory

        Returns:
            None. Logs warnings and returns early if validation fails.
        """
        # Validate lines parameter
        if not isinstance(lines, list):
            logger.warning(
                f"broadcast_trajectory: 'lines' must be a list, got {type(lines).__name__}"
            )
            self.broadcast_stats.validation_failures += 1
            return

        if len(lines) < 2:
            logger.warning(
                f"broadcast_trajectory: 'lines' must have at least 2 points for a valid trajectory, got {len(lines)}"
            )
            self.broadcast_stats.validation_failures += 1
            return

        # Validate each line is a dict
        for i, line in enumerate(lines):
            if not isinstance(line, dict):
                logger.warning(
                    f"broadcast_trajectory: line at index {i} is not a dict, got {type(line).__name__}"
                )
                self.broadcast_stats.validation_failures += 1
                return

        # Validate collisions parameter if provided
        if collisions is not None:
            if not isinstance(collisions, list):
                logger.warning(
                    f"broadcast_trajectory: 'collisions' must be a list, got {type(collisions).__name__}"
                )
                self.broadcast_stats.validation_failures += 1
                return

            # Validate each collision is a dict with expected structure
            for i, collision in enumerate(collisions):
                if not isinstance(collision, dict):
                    logger.warning(
                        f"broadcast_trajectory: collision at index {i} is not a dict, got {type(collision).__name__}"
                    )
                    self.broadcast_stats.validation_failures += 1
                    return

                # Common collision fields to check (if present)
                if "position" in collision:
                    position = collision["position"]
                    if not isinstance(position, (list, tuple)) or len(position) < 2:
                        logger.warning(
                            f"broadcast_trajectory: collision at index {i} has invalid position format. "
                            f"Expected list/tuple with at least 2 elements, got {type(position).__name__} with value {position}"
                        )
                        self.broadcast_stats.validation_failures += 1
                        return

        trajectory_data = {
            "lines": lines,
            "collisions": collisions or [],
            "confidence": confidence,
            "calculation_time_ms": calculation_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self._get_next_sequence("trajectory"),
            "line_count": len(lines),
            "collision_count": len(collisions or []),
        }

        # Broadcast via WebSocket
        await self._broadcast_to_stream(StreamType.TRAJECTORY, trajectory_data)

    async def broadcast_alert(
        self,
        level: str,
        message: str,
        code: str,
        details: Optional[dict[str, Any]] = None,
        target_clients: Optional[list[str]] = None,
        target_users: Optional[list[str]] = None,
    ):
        """Broadcast alert message with targeting options."""
        alert_data = {
            "level": level,
            "message": message,
            "code": code,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self._get_next_sequence("alert"),
        }

        if target_clients or target_users:
            # Use manager for targeted alerts
            await websocket_manager.send_alert(
                level, message, code, details, target_clients, target_users
            )
        else:
            # Broadcast to all alert subscribers
            await self._broadcast_to_stream(StreamType.ALERT, alert_data)

    async def broadcast_config_update(
        self,
        config_section: str,
        config_data: dict[str, Any],
        change_summary: Optional[str] = None,
    ):
        """Broadcast configuration updates to subscribers."""
        config_update_data = {
            "section": config_section,
            "config": config_data,
            "change_summary": change_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self._get_next_sequence("config"),
        }

        await self._broadcast_to_stream(StreamType.CONFIG, config_update_data)

    async def send_performance_metrics(
        self, target_clients: Optional[list[str]] = None
    ):
        """Send performance metrics to specified clients only.

        IMPORTANT: Metrics are NO LONGER automatically broadcast to all clients
        to prevent browser crashes from large metric payloads. Metrics should be
        requested via REST API endpoint or sent to specific clients only.
        """
        metrics_data = {
            "broadcast_stats": {
                "messages_sent": self.broadcast_stats.messages_sent,
                "bytes_sent": self.broadcast_stats.bytes_sent,
                "failed_sends": self.broadcast_stats.failed_sends,
                "validation_failures": self.broadcast_stats.validation_failures,
                "average_latency": self.broadcast_stats.average_latency,
                "peak_latency": self.broadcast_stats.peak_latency,
                "compression_enabled": self.broadcast_stats.compression_enabled,
            },
            "frame_metrics": {
                "frames_sent": self.broadcast_stats.frame_metrics.frames_sent,
                "bytes_sent": self.broadcast_stats.frame_metrics.bytes_sent,
                "compression_ratio": self.broadcast_stats.frame_metrics.compression_ratio,
                "average_latency": self.broadcast_stats.frame_metrics.average_latency,
                "dropped_frames": self.broadcast_stats.frame_metrics.dropped_frames,
                "target_fps": self.broadcast_stats.frame_metrics.target_fps,
                "actual_fps": self.broadcast_stats.frame_metrics.actual_fps,
            },
            "connection_stats": websocket_handler.get_connection_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send as a special metrics message
        message = {
            "type": "metrics",
            "data": metrics_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if target_clients:
            # Send to specific clients only
            tasks = [
                websocket_handler.send_to_client(client_id, message)
                for client_id in target_clients
                if client_id in websocket_handler.connections
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # DISABLED: Do not broadcast metrics to all clients to prevent browser overload
            # Use REST API endpoint /api/v1/websocket/metrics instead
            logger.warning(
                "send_performance_metrics called without target_clients. "
                "Metrics should be retrieved via REST API, not broadcast to all clients."
            )

    def get_broadcast_stats(self) -> dict[str, Any]:
        """Get current broadcasting statistics."""
        return {
            "is_streaming": self.is_streaming,
            "frame_buffer_size": len(self.frame_buffer.frames),
            "current_fps": self.frame_buffer.get_frame_rate(),
            "queue_size": self.frame_queue.qsize(),
            "broadcast_stats": self.broadcast_stats.__dict__,
            "active_tasks": list(self.broadcast_tasks.keys()),
        }

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics as data (for REST API use).

        This method returns metrics as a dictionary without broadcasting
        to clients, suitable for REST API endpoints.
        """
        return {
            "broadcast_stats": {
                "messages_sent": self.broadcast_stats.messages_sent,
                "bytes_sent": self.broadcast_stats.bytes_sent,
                "failed_sends": self.broadcast_stats.failed_sends,
                "validation_failures": self.broadcast_stats.validation_failures,
                "average_latency": self.broadcast_stats.average_latency,
                "peak_latency": self.broadcast_stats.peak_latency,
                "compression_enabled": self.broadcast_stats.compression_enabled,
            },
            "frame_metrics": {
                "frames_sent": self.broadcast_stats.frame_metrics.frames_sent,
                "bytes_sent": self.broadcast_stats.frame_metrics.bytes_sent,
                "compression_ratio": self.broadcast_stats.frame_metrics.compression_ratio,
                "average_latency": self.broadcast_stats.frame_metrics.average_latency,
                "dropped_frames": self.broadcast_stats.frame_metrics.dropped_frames,
                "target_fps": self.broadcast_stats.frame_metrics.target_fps,
                "actual_fps": self.broadcast_stats.frame_metrics.actual_fps,
            },
            "connection_stats": websocket_handler.get_connection_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _broadcast_to_stream(self, stream_type: StreamType, data: dict[str, Any]):
        """Internal method to broadcast to a stream with performance tracking."""
        start_time = time.time()

        try:
            await websocket_manager.broadcast_to_stream(
                stream_type, data, apply_filters=True
            )

            # Update statistics
            latency = (time.time() - start_time) * 1000  # ms
            self.broadcast_stats.messages_sent += 1
            self.broadcast_stats.average_latency = (
                self.broadcast_stats.average_latency
                * (self.broadcast_stats.messages_sent - 1)
                + latency
            ) / self.broadcast_stats.messages_sent
            self.broadcast_stats.peak_latency = max(
                self.broadcast_stats.peak_latency, latency
            )

            # Estimate bytes sent (rough approximation)
            message_size = len(json.dumps(data, cls=NumpyEncoder).encode())
            self.broadcast_stats.bytes_sent += message_size

        except Exception as e:
            self.broadcast_stats.failed_sends += 1
            logger.error(f"Failed to broadcast to {stream_type.value}: {e}")

    async def _process_frame_queue(self):
        """Process queued frames for streaming."""
        while self.is_streaming:
            try:
                # Wait for frame with timeout
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)

                # Apply FPS limiting per client
                current_time = time.time()
                subscribers = websocket_manager.stream_subscribers[
                    StreamType.FRAME
                ].copy()

                eligible_clients = []
                for client_id in subscribers:
                    if client_id not in websocket_manager.sessions:
                        continue

                    session = websocket_manager.sessions[client_id]

                    # Check FPS limiting
                    target_fps = config_manager.get(
                        "api.websocket.broadcaster.fps.default_target_fps", 30.0
                    )
                    if StreamType.FRAME in session.subscription_filters:
                        filter_config = session.subscription_filters[StreamType.FRAME]
                        if filter_config.max_fps:
                            target_fps = filter_config.max_fps

                    min_interval = 1.0 / target_fps if target_fps > 0 else 0
                    last_frame_time = self.fps_limiter[client_id]

                    if current_time - last_frame_time >= min_interval:
                        eligible_clients.append(client_id)
                        self.fps_limiter[client_id] = current_time

                # Send frame to eligible clients
                if eligible_clients:
                    message = {
                        "type": "frame",
                        "timestamp": frame_data["timestamp"],
                        "sequence": frame_data["sequence"],
                        "data": frame_data,
                    }

                    tasks = [
                        websocket_handler.send_to_client(client_id, message)
                        for client_id in eligible_clients
                    ]

                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.TimeoutError:
                # No frames to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")

    async def _cleanup_task(self):
        """Periodic cleanup of old data."""
        cleanup_interval = config_manager.get(
            "api.websocket.broadcaster.cleanup.interval_seconds", 30
        )
        fps_limiter_cleanup_age = config_manager.get(
            "api.websocket.broadcaster.fps.limiter_cleanup_age_seconds", 300
        )

        while self.is_streaming:
            try:
                await asyncio.sleep(cleanup_interval)

                # Clean up old frames
                self.frame_buffer.cleanup_old_frames()

                # Clean up old FPS limiter entries
                current_time = time.time()
                old_clients = [
                    client_id
                    for client_id, last_time in self.fps_limiter.items()
                    if current_time - last_time > fps_limiter_cleanup_age
                ]
                for client_id in old_clients:
                    del self.fps_limiter[client_id]

                logger.debug(
                    f"Cleanup completed: {len(old_clients)} old FPS entries removed"
                )

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    def _get_next_sequence(self, stream_type: str) -> int:
        """Get next sequence number for a stream type."""
        self.sequence_counters[stream_type] += 1
        return self.sequence_counters[stream_type]

    def _compress_data(self, data: bytes, level: int = None) -> tuple[bytes, bool]:
        """Compress data if beneficial."""
        if len(data) < self.compression_threshold:
            return data, False

        if level is None:
            level = self.compression_level

        try:
            compressed = zlib.compress(data, level=level)
            # Only use compression if it reduces size by configured threshold
            if len(compressed) < len(data) * self.compression_ratio_threshold:
                return compressed, True
        except Exception as e:
            logger.warning(f"Compression failed: {e}")

        return data, False


# Global broadcaster instance
message_broadcaster = MessageBroadcaster()
