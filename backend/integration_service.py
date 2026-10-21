"""Integration Service - Connects Vision → Core → Broadcast data flow.

This service is the critical missing piece that ties all modules together:
1. Polls Vision for detection results
2. Updates Core game state with detection data
3. Subscribes to Core events for state changes
4. Triggers WebSocket broadcasts on state updates
5. Calculates and broadcasts trajectories when cue detected
"""

import asyncio
import copy
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from backend.config import Config, config
from backend.core import CoreModule
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState, TableState
from backend.core.physics.trajectory import (
    MultiballTrajectoryResult,
    TrajectoryCalculator,
    TrajectoryQuality,
)
from backend.core.validation.physics import PhysicsValidator
from backend.core.validation.table_state import TableStateValidator
from backend.integration_service_conversion_helpers import (
    CoordinateConversionError,
    StateConversionHelpers,
    serialize_vision_ball,
    serialize_vision_cue,
    serialize_vision_table,
)
from backend.vision import VisionModule
from backend.vision.models import DetectionResult

logger = logging.getLogger(__name__)
_UNSET_OBSERVATION_STATUS = object()


@dataclass(frozen=True)
class _PredictionRequest:
    """Detached inputs and provenance for one background prediction."""

    cue_state: CueState
    ball_state: BallState
    table_state: TableState
    other_balls: list[BallState]
    quality: TrajectoryQuality
    max_collision_depth: int
    metadata: dict[str, Any]
    requested_at: float
    observation_age_seconds: Optional[float]
    maximum_observation_age_seconds: float
    lifecycle_generation: int
    timing: dict[str, float | None] = field(default_factory=dict, compare=False)


class BroadcastErrorType(Enum):
    """Types of broadcast errors for retry logic."""

    TRANSIENT = "transient"  # Network errors, temporary failures - retry
    VALIDATION = "validation"  # Data validation errors - don't retry
    UNKNOWN = "unknown"  # Unknown errors - retry with caution


@dataclass
class BroadcastMetrics:
    """Metrics for broadcast operations."""

    successful_broadcasts: int = 0
    failed_broadcasts: int = 0
    total_retries: int = 0

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.successful_broadcasts + self.failed_broadcasts
        return (self.successful_broadcasts / total * 100) if total > 0 else 0.0


class CircuitBreaker:
    """Circuit breaker pattern for broadcast failures.

    Implements the Circuit Breaker pattern to prevent cascade failures in WebSocket
    broadcasting. When multiple consecutive broadcast failures occur (e.g., all clients
    disconnect), the circuit "opens" and stops attempting broadcasts for a timeout period.

    REFACTORING NOTE: This is a new addition to improve system resilience. Previously,
    broadcast failures would continuously spam error logs and waste CPU cycles attempting
    to send to disconnected clients.

    States:
    - CLOSED: Normal operation, broadcasts are attempted
    - OPEN: Too many failures, broadcasts are blocked
    - HALF-OPEN: Timeout elapsed, trying one broadcast to test if system recovered

    Behavior:
    - After N consecutive failures (default 10), circuit opens
    - While open, all broadcast attempts are blocked (return immediately)
    - After timeout period (default 30s), circuit enters half-open state
    - One successful broadcast in half-open state closes the circuit
    - One failed broadcast in half-open state reopens the circuit

    This prevents:
    - CPU waste from attempting broadcasts when all clients are disconnected
    - Log spam from repeated broadcast failures
    - Resource exhaustion from queued broadcast tasks
    """

    def __init__(self, failure_threshold: int = 10, timeout_seconds: float = 30.0):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before circuit opens
                             (default: 10 failures)
            timeout_seconds: Seconds to wait before attempting to close circuit
                           (default: 30 seconds)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.consecutive_failures = 0
        self.circuit_open_time: Optional[float] = None
        self.is_open = False

    def record_success(self) -> None:
        """Record a successful operation.

        Resets the circuit breaker to CLOSED state and clears failure counters.
        This is called after any successful broadcast attempt.
        """
        if self.is_open:
            logger.info("Circuit breaker closing after successful operation")
        # Reset all state - back to normal operation
        self.consecutive_failures = 0
        self.circuit_open_time = None
        self.is_open = False

    def record_failure(self) -> None:
        """Record a failed operation.

        Increments the consecutive failure counter. If threshold is reached,
        opens the circuit and starts the timeout period.
        """
        self.consecutive_failures += 1

        # Check if we've hit the threshold to open the circuit
        if not self.is_open and self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            self.circuit_open_time = time.time()
            logger.error(
                f"Circuit breaker OPENED after {self.consecutive_failures} consecutive failures. "
                f"Will retry after {self.timeout_seconds} seconds."
            )

    def can_attempt(self) -> bool:
        """Check if operation can be attempted.

        This is called before each broadcast attempt to determine if the operation
        should proceed or be blocked.

        Returns:
            True if circuit is closed (normal operation) or timeout has elapsed
            (entering half-open state). False if circuit is open and timeout has
            not yet elapsed (blocking broadcasts).
        """
        # Circuit closed - normal operation
        if not self.is_open:
            return True

        # Circuit open - check if timeout has elapsed to enter half-open state
        if (
            self.circuit_open_time
            and (time.time() - self.circuit_open_time) >= self.timeout_seconds
        ):
            logger.info(
                f"Circuit breaker timeout elapsed ({self.timeout_seconds}s). "
                "Attempting to close circuit..."
            )
            # Half-open state: allow one attempt to test if system recovered
            # If this attempt succeeds, record_success() will close the circuit
            # If this attempt fails, record_failure() will reopen the circuit
            self.is_open = False
            return True

        # Circuit still open and timeout not elapsed - block the operation
        return False

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dictionary with status information
        """
        return {
            "is_open": self.is_open,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "seconds_until_retry": (
                max(0, self.timeout_seconds - (time.time() - self.circuit_open_time))
                if self.circuit_open_time
                else 0
            ),
        }


class IntegrationService:
    """Manages data flow between Vision, Core, and Broadcasting systems."""

    def __init__(
        self,
        vision_module: VisionModule,
        core_module: CoreModule,
        message_broadcaster: Any,  # MessageBroadcaster from api.websocket
        config_module: Optional[Config] = None,
    ):
        """Initialize integration service.

        Args:
            vision_module: Vision processing module
            core_module: Core game state and physics module
            message_broadcaster: WebSocket broadcaster
            config_module: Configuration module (optional, will use singleton if not provided)
        """
        self.vision = vision_module
        self.core = core_module
        self.broadcaster = message_broadcaster
        self.running = False
        self.integration_task: Optional[asyncio.Task[None]] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._audit_recorder = getattr(vision_module, "audit_recorder", None)
        self._audit_domains = getattr(vision_module, "audit_domains", {})
        self._vision_instance_id = str(uuid.uuid4())
        self._vision_process_id = os.getpid()
        self._last_processed_observation_identity: tuple[Any, ...] | None = None
        self.duplicate_observation_count = 0
        # Core emits state_updated while update_state is still in flight, but
        # its state model replaces the source frame/timestamp with Core-local
        # values. Serialize gateway-originated updates and keep a short-lived
        # FIFO handoff for the matching flattened state event. The callback can
        # run on an executor thread, so handoff access is protected by a lock.
        self._core_update_provenance_lock = asyncio.Lock()
        self._state_event_observation_lock = threading.Lock()
        self._pending_state_event_observations: list[
            tuple[object, Optional[dict[str, Any]]]
        ] = []

        # Initialize trajectory calculator for multiball predictions
        self.trajectory_calculator = TrajectoryCalculator()
        self._trajectory_calculation_lock = threading.Lock()
        self._prediction_task: Optional[asyncio.Task[None]] = None
        self._pending_prediction: Optional[_PredictionRequest] = None
        self._prediction_lifecycle_generation = 0
        self._latest_observation_received_at: Optional[float] = None

        # Initialize validators
        self.physics_validator = PhysicsValidator()
        self.table_state_validator = TableStateValidator()

        # Configuration (use provided or singleton)
        self.config = config_module if config_module is not None else config
        self._maximum_prediction_observation_age_seconds = float(
            self.config.get(
                "vision.processing.phase_scheduler.maximum_observation_age_seconds",
                0.5,
            )
        )

        # Initialize state conversion helpers
        self.state_converter = StateConversionHelpers(
            config=self.config, physics_validator=self.physics_validator
        )
        self.target_fps = self.config.get("integration.target_fps", 30)
        self.log_interval_frames = self.config.get(
            "integration.log_interval_frames", 300
        )
        self.error_retry_delay = self.config.get(
            "integration.error_retry_delay_sec", 0.1
        )
        self.shot_speed_estimate = self.config.get(
            "integration.shot_speed_estimate_m_per_s", 2.0
        )
        requested_quality = str(
            self.config.get("integration.trajectory_quality", "low")
        ).lower()
        try:
            self.trajectory_quality = TrajectoryQuality(requested_quality)
        except ValueError:
            logger.warning(
                "Invalid integration.trajectory_quality=%r; using configured fallback 'low'",
                requested_quality,
            )
            self.trajectory_quality = TrajectoryQuality.LOW
        try:
            self.max_collision_depth = int(
                self.config.get("integration.max_collision_depth", 2)
            )
        except (TypeError, ValueError):
            logger.warning(
                "Invalid integration.max_collision_depth; using fallback depth 2"
            )
            self.max_collision_depth = 2
        if self.max_collision_depth < 0:
            logger.warning(
                "integration.max_collision_depth must be non-negative; using fallback depth 2"
            )
            self.max_collision_depth = 2

        # Validation thresholds
        self.max_ball_velocity = self.config.get(
            "integration.max_ball_velocity_m_per_s", 10.0
        )
        self.max_position_x = self.config.get("integration.max_position_x", 3.0)
        self.max_position_y = self.config.get("integration.max_position_y", 2.0)

        # Broadcast retry configuration
        self.max_retries = self.config.get("integration.broadcast_max_retries", 3)
        self.retry_base_delay = self.config.get(
            "integration.broadcast_retry_base_delay_sec", 0.1
        )
        self.circuit_breaker_threshold = self.config.get(
            "integration.circuit_breaker_threshold", 10
        )
        self.circuit_breaker_timeout = self.config.get(
            "integration.circuit_breaker_timeout_sec", 30.0
        )

        # Performance tracking
        self.frame_count = 0
        self.error_count = 0
        self.last_detection_time = 0.0
        self.trajectory_frame_count = 0  # Track frames for periodic trajectory logging

        # The latest vision status is also used by request-triggered API
        # calculations. Keep this beside integration rather than the core game
        # state, whose model intentionally contains only billiards state.
        self.latest_observation_status: Optional[dict[str, Any]] = None
        self._last_vision_phase: Optional[str] = None
        self._last_calibration_revision: Any = None
        self.last_live_prediction: Optional[dict[str, Any]] = None
        self.frozen_comparison_prediction: Optional[dict[str, Any]] = None

        # Broadcast error recovery
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.circuit_breaker_threshold,
            timeout_seconds=self.circuit_breaker_timeout,
        )
        self.broadcast_metrics = BroadcastMetrics()

    async def start(self) -> None:
        """Start the integration service."""
        if self.running:
            logger.warning("Integration service already running")
            return

        logger.info("Starting integration service...")
        self._prediction_lifecycle_generation += 1
        self.running = True

        # Capture the event loop for use in event callbacks
        self._event_loop = asyncio.get_running_loop()

        # Start vision module camera capture
        logger.info("Starting vision module camera capture...")
        try:
            if self.vision.start_capture():
                logger.info("Vision camera capture started successfully")
            else:
                logger.error(
                    "Failed to start vision camera capture - integration will continue "
                    "but no detections will be available until camera is started"
                )
        except Exception as e:
            logger.error(
                f"Error starting vision camera capture: {e} - "
                "integration will continue but no detections will be available"
            )

        # Check vision calibration status
        try:
            # Check if vision module has calibration data
            if hasattr(self.vision, "calibrator") and self.vision.calibrator:
                if not self.vision.calibrator.is_calibrated():
                    logger.warning(
                        "Vision module is not calibrated - detection accuracy may be poor. "
                        "Use the supported chessboard workflow: start with "
                        "POST /api/v1/vision/calibration/camera/mode/start, capture "
                        "images with POST /api/v1/vision/calibration/camera/images/capture, "
                        "then process them with POST /api/v1/vision/calibration/camera/process."
                    )
            else:
                logger.warning(
                    "Vision module has no calibrator - using uncalibrated camera"
                )
        except Exception as e:
            logger.debug(f"Could not check calibration status: {e}")

        # Subscribe to Core events
        self._subscribe_to_core_events()

        # Start integration loop
        self.integration_task = asyncio.create_task(self._integration_loop())
        logger.info("Integration service started successfully")

    async def stop(self) -> None:
        """Stop the integration service."""
        if not self.running:
            await self._drain_prediction_task()
            return

        logger.info("Stopping integration service...")
        self.running = False
        self._prediction_lifecycle_generation += 1
        # Pending observations have no value after shutdown begins. The active
        # calculation is allowed to finish because cancelling asyncio.to_thread
        # would leave its worker thread running against the calculator.
        self._pending_prediction = None

        if self.integration_task:
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass

        await self._drain_prediction_task()

        # Stop vision module camera capture
        logger.info("Stopping vision module camera capture...")
        try:
            self.vision.stop_capture()
            logger.info("Vision camera capture stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping vision camera capture: {e}")

        logger.info("Integration service stopped")

    def _subscribe_to_core_events(self) -> None:
        """Subscribe to Core module events for broadcasting."""
        # Subscribe to state updates
        self.core.subscribe_to_events("state_updated", self._on_state_updated)

        # Subscribe to trajectory calculations
        self.core.subscribe_to_events(
            "trajectory_calculated", self._on_trajectory_calculated
        )

        logger.info("Subscribed to Core module events")

    async def _integration_loop(self) -> None:
        """Main integration loop - processes vision detections at configured FPS."""
        frame_interval = 1.0 / self.target_fps

        logger.info(f"Integration loop starting at {self.target_fps} FPS")

        while self.running:
            try:
                loop_start = asyncio.get_event_loop().time()

                # Get detection result from vision
                poll_started_ns = time.perf_counter_ns()
                detection_result = await self._get_vision_detection()
                poll_finished_ns = time.perf_counter_ns()
                self._record_audit_event(
                    "state",
                    {
                        "event_type": "integration_vision_poll",
                        "observation_available": detection_result is not None,
                        "poll_seconds": max(0, poll_finished_ns - poll_started_ns)
                        / 1e9,
                        "vision_queue_depth": (
                            self.vision._frame_queue.qsize()
                            if hasattr(self.vision, "_frame_queue")
                            else None
                        ),
                        "configured_target_fps": self.target_fps,
                    },
                )

                if detection_result:
                    # Convert to Core format and update state
                    observation_processed = await self._process_detection(
                        detection_result
                    )

                    if observation_processed:
                        self.frame_count += 1

                    # Log progress periodically
                    if (
                        observation_processed
                        and self.frame_count % self.log_interval_frames == 0
                    ):
                        logger.info(
                            f"Integration: {self.frame_count} frames processed, "
                            f"{self.error_count} errors"
                        )

                # Maintain target FPS
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in integration loop: {e}", exc_info=True)
                await asyncio.sleep(self.error_retry_delay)

        logger.info("Integration loop stopped")

    async def _get_vision_detection(self) -> Optional[DetectionResult]:
        """Get detection result from vision module.

        Returns:
            DetectionResult if available, None otherwise
        """
        try:
            # Process current frame
            detection = self.vision.process_frame()
            return detection
        except Exception as e:
            logger.error(f"Failed to get vision detection: {e}")
            return None

    async def _process_detection(self, detection: DetectionResult) -> bool:
        """Process one new source observation and update Core.

        Args:
            detection: Detection result from vision

        Returns:
            True when this result is admitted for processing. False when the
            result repeats the most recently admitted source observation or
            fails coordinate-contract validation. Results without usable
            identity are processed but cannot be deduplicated.
        """
        observation_identity = self._source_observation_identity(detection)
        if (
            observation_identity is not None
            and observation_identity == self._last_processed_observation_identity
        ):
            self.duplicate_observation_count += 1
            if self._audit_enabled("state"):
                self._record_audit_event(
                    "state",
                    {
                        "event_type": "integration_observation_duplicate",
                        "source_frame_number": getattr(
                            detection, "source_frame_number", detection.frame_number
                        ),
                        "source_process_id": getattr(
                            detection, "source_process_id", None
                        ),
                        "vision_instance_id": getattr(
                            self, "_vision_instance_id", None
                        ),
                        "duplicate_observation_count": self.duplicate_observation_count,
                    },
                )
            return False

        state_audit_enabled = self._audit_enabled("state")
        # A malformed coordinate contract must stop at this boundary instead
        # of injecting unscaled values into Core. Keep the integration loop
        # alive so a corrected frame can recover on the next observation.
        observation_started_ns = time.perf_counter_ns() if state_audit_enabled else None
        conversion_started_ns = observation_started_ns
        conversion_finished_ns: int | None = None
        core_started_ns: int | None = None
        core_finished_ns: int | None = None
        trajectory_started_ns: int | None = None
        trajectory_finished_ns: int | None = None
        lock_wait_seconds: float | None = None
        try:
            detection_data = self._convert_detection_to_core_format(detection)
            if state_audit_enabled:
                conversion_finished_ns = time.perf_counter_ns()
        except CoordinateConversionError as error:
            self.error_count += 1
            logger.error(
                "Skipping frame %s because its coordinate contract is invalid: %s",
                detection.frame_number,
                error,
            )
            if state_audit_enabled:
                conversion_finished_ns = time.perf_counter_ns()
                assert conversion_started_ns is not None
                self._record_audit_event(
                    "state",
                    {
                        "event_type": "integration_observation_rejected",
                        "source_frame_number": getattr(
                            detection, "source_frame_number", detection.frame_number
                        ),
                        "coordinate_conversion_seconds": max(
                            0, conversion_finished_ns - conversion_started_ns
                        )
                        / 1e9,
                        "reason": "coordinate_contract_invalid",
                    },
                )
            return False

        # Store the identity only after conversion succeeds. A failed coordinate
        # contract must not become an admitted observation; a corrected result
        # from the same source frame can still be retried.
        if observation_identity is not None:
            self._last_processed_observation_identity = observation_identity

        # Attach vision eligibility/provenance to the Core update. Public
        # freshness advances only after Core accepts this state update.
        status = self._observation_status_with_identity(detection)
        if status is not None:
            detection_data["observation_status"] = copy.deepcopy(status)

        state_event_token = object()
        update_lock = getattr(self, "_core_update_provenance_lock", None)
        if update_lock is None:
            # Lightweight unit-test doubles may bypass __init__.
            update_lock = asyncio.Lock()
            self._core_update_provenance_lock = update_lock

        # Keep the status handoff in registration order. Core's own state lock
        # serializes calls, and this integration lock ensures concurrent
        # gateway callers cannot reorder their handoffs while awaiting Core.
        lock_wait_started_ns = time.perf_counter_ns() if state_audit_enabled else None
        async with update_lock:
            if state_audit_enabled and lock_wait_started_ns is not None:
                lock_acquired_ns = time.perf_counter_ns()
                lock_wait_seconds = (
                    max(0, lock_acquired_ns - lock_wait_started_ns) / 1e9
                )
            self._register_state_event_observation(state_event_token, status)
            try:
                if state_audit_enabled:
                    core_started_ns = time.perf_counter_ns()
                await self.core.update_state(detection_data)
                if state_audit_enabled:
                    core_finished_ns = time.perf_counter_ns()
                core_update_status = "complete"
            except Exception as e:
                if state_audit_enabled:
                    core_finished_ns = time.perf_counter_ns()
                core_update_status = "failed"
                self.error_count += 1
                logger.error(f"Failed to update Core state: {e}", exc_info=True)
            finally:
                # If Core failed before emitting its event (or has no listener),
                # discard the pending handoff. If the callback already consumed
                # it, this is a no-op.
                self._discard_state_event_observation(state_event_token)

            # Freshness must describe the state actually held by Core. Updating
            # it before releasing this lock preserves ordering between
            # concurrent gateway callers.
            if core_update_status == "complete":
                self.latest_observation_status = status
                self._latest_observation_received_at = time.monotonic()

        # Freshness must describe the state actually held by Core. A failed
        # update leaves the prior status/receipt age untouched and must not
        # launch predictions from a mixture of old Core state and new vision.
        trajectory_admission_status = "skipped_core_update_failed"
        if core_update_status == "complete":
            if state_audit_enabled:
                trajectory_started_ns = time.perf_counter_ns()
            await self._check_trajectory_calculation(detection)
            if state_audit_enabled:
                trajectory_finished_ns = time.perf_counter_ns()
            trajectory_admission_status = "checked"
        elif state_audit_enabled:
            trajectory_finished_ns = time.perf_counter_ns()

        if state_audit_enabled:
            assert conversion_started_ns is not None
            assert conversion_finished_ns is not None
            assert observation_started_ns is not None
            assert trajectory_finished_ns is not None
            self._record_audit_event(
                "state",
                {
                    "event_type": "integration_observation_processed",
                    "source_frame_number": status.get("source_frame_number"),
                    "source_process_id": status.get("source_process_id"),
                    "vision_instance_id": status.get("vision_instance_id"),
                    "source_timestamp": status.get("source_timestamp"),
                    "source_timestamp_kind": status.get("source_timestamp_kind"),
                    "map_revision": status.get("map_revision"),
                    "calibration_revision": status.get("calibration_revision"),
                    "coordinate_conversion_seconds": max(
                        0, conversion_finished_ns - conversion_started_ns
                    )
                    / 1e9,
                    "integration_lock_wait_seconds": lock_wait_seconds,
                    "core_update_seconds": (
                        max(0, core_finished_ns - core_started_ns) / 1e9
                        if core_started_ns is not None and core_finished_ns is not None
                        else None
                    ),
                    "core_update_status": core_update_status,
                    "trajectory_admission_status": trajectory_admission_status,
                    "trajectory_admission_seconds": (
                        max(0, trajectory_finished_ns - trajectory_started_ns) / 1e9
                        if trajectory_started_ns is not None
                        and trajectory_finished_ns is not None
                        else None
                    ),
                    "ball_count": len(detection.balls),
                    "cue_present": detection.cue is not None,
                    "observation_status": status.get("component_observation_status"),
                    "service_elapsed_seconds": max(
                        0, trajectory_finished_ns - observation_started_ns
                    )
                    / 1e9,
                },
            )
        return True

    def _source_observation_identity(
        self, detection: DetectionResult
    ) -> tuple[Any, ...] | None:
        """Build a stable key for recognizing repeated cached source results.

        Producer PID and service instance identity scope frame counters across
        process/service restarts. A timestamp supplements the frame number so a
        reset or legacy counter can be disambiguated where provenance exists.
        These values identify the source result; map and calibration revisions
        intentionally do not turn an old camera frame into a new observation.
        """
        source_frame_number = getattr(detection, "source_frame_number", None)
        if source_frame_number is None:
            source_frame_number = getattr(detection, "frame_number", None)

        publication_monotonic_ns = getattr(
            detection, "source_publication_monotonic_ns", None
        )
        if (
            not isinstance(publication_monotonic_ns, int)
            or publication_monotonic_ns <= 0
        ):
            publication_monotonic_ns = None

        source_timestamp = getattr(detection, "source_timestamp", None)
        if source_timestamp is None:
            source_timestamp = getattr(detection, "timestamp", None)
        if not isinstance(source_timestamp, (int, float)) or not math.isfinite(
            source_timestamp
        ):
            source_timestamp = None

        if source_frame_number is None and source_timestamp is None:
            return None

        return (
            getattr(self, "_vision_instance_id", None),
            getattr(detection, "source_process_id", None),
            source_frame_number,
            getattr(detection, "source_timestamp_kind", None),
            publication_monotonic_ns,
            source_timestamp,
        )

    def _audit_enabled(self, domain: str) -> bool:
        """Check audit activation without constructing an event on the hot path."""
        recorder = getattr(self, "_audit_recorder", None)
        domains = getattr(self, "_audit_domains", {})
        return bool(
            recorder is not None
            and isinstance(domains, dict)
            and domains.get(domain, False)
        )

    def _record_audit_event(self, domain: str, event: dict[str, Any]) -> None:
        """Record an event only when its audit domain is explicitly enabled."""
        recorder = getattr(self, "_audit_recorder", None)
        if recorder is not None and self._audit_enabled(domain):
            recorder.record(event)

    def _register_state_event_observation(
        self,
        token: object,
        status: Optional[dict[str, Any]],
    ) -> None:
        """Publish one temporary event-provenance handoff to Core callbacks.

        The current Core event schema omits the source frame identity. Calls
        from this service are therefore serialized and paired in order with
        Core's synchronous, flattened state event. If Core gains another
        concurrent state-update producer, the event contract should carry an
        explicit request identity instead of relying on this handoff.
        """
        pending = getattr(self, "_pending_state_event_observations", None)
        lock = getattr(self, "_state_event_observation_lock", None)
        if pending is None or lock is None:
            # Support lightweight service doubles that bypass __init__ while
            # preserving correct production behavior.
            lock = threading.Lock()
            pending = []
            self._state_event_observation_lock = lock
            self._pending_state_event_observations = pending
        with lock:
            pending.append((token, copy.deepcopy(status)))

    def _take_state_event_observation(
        self, state: dict[str, Any]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Consume provenance only for a flattened Core state event.

        Other state_updated messages (including GameStateManager's smaller
        internal change notification) do not carry this full shape and must
        continue to use the last successfully applied observation status.
        """
        frame_number = state.get("frame_number")
        timestamp = state.get("timestamp")
        if (
            not all(
                field_name in state
                for field_name in (
                    "balls",
                    "table",
                    "timestamp",
                    "frame_number",
                    "game_type",
                    "cue",
                    "scores",
                    "coordinate_metadata",
                )
            )
            or not isinstance(state.get("balls"), list)
            or not isinstance(state.get("table"), dict)
            or not isinstance(state.get("game_type"), str)
            or not isinstance(frame_number, int)
            or isinstance(frame_number, bool)
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
        ):
            return False, None
        try:
            timestamp_seconds = float(timestamp)
        except (OverflowError, ValueError):
            return False, None
        if not math.isfinite(timestamp_seconds):
            return False, None
        pending = getattr(self, "_pending_state_event_observations", None)
        lock = getattr(self, "_state_event_observation_lock", None)
        if pending is None or lock is None:
            return False, None
        with lock:
            if not pending:
                return False, None
            _token, status = pending.pop(0)
        return True, copy.deepcopy(status)

    def _discard_state_event_observation(self, token: object) -> None:
        """Remove an unconsumed provenance handoff after Core returns."""
        pending = getattr(self, "_pending_state_event_observations", None)
        lock = getattr(self, "_state_event_observation_lock", None)
        if pending is None or lock is None:
            return
        with lock:
            self._pending_state_event_observations = [
                entry for entry in pending if entry[0] is not token
            ]

    def _observation_status_with_identity(
        self, detection: DetectionResult
    ) -> dict[str, Any]:
        """Add process-local restart identity to the source observation."""
        status = self._observation_status(detection) or {}
        status.setdefault(
            "source_frame_number",
            getattr(
                detection,
                "source_frame_number",
                getattr(detection, "frame_number", None),
            ),
        )
        status.setdefault(
            "source_timestamp", getattr(detection, "source_timestamp", None)
        )
        status.setdefault(
            "source_timestamp_kind",
            getattr(detection, "source_timestamp_kind", None),
        )
        status.setdefault(
            "source_process_id", getattr(detection, "source_process_id", None)
        )
        publication_monotonic_ns = getattr(
            detection, "source_publication_monotonic_ns", None
        )
        if isinstance(publication_monotonic_ns, int) and publication_monotonic_ns > 0:
            status["source_publication_monotonic_ns"] = publication_monotonic_ns
        status["vision_instance_id"] = getattr(self, "_vision_instance_id", None)
        status["vision_process_id"] = getattr(self, "_vision_process_id", None)
        return status

    async def _drain_prediction_task(self) -> None:
        """Wait for the active physics thread without cancelling its wrapper."""
        task = self._prediction_task
        if task is not None and not task.done():
            await asyncio.shield(task)

    def _convert_detection_to_core_format(
        self, detection: DetectionResult
    ) -> dict[str, Any]:
        """Convert vision DetectionResult to Core's expected format.

        Args:
            detection: Vision detection result

        Returns:
            Dictionary in Core's expected format
        """
        detection_data: dict[str, Any] = {
            "timestamp": detection.timestamp,
            "frame_number": detection.frame_number,
        }

        status = self._observation_status_with_identity(detection)
        if status is not None:
            detection_data["observation_status"] = status

        frame_resolution = detection.frame_size
        # Explicit null means the source did not provide valid dimensions;
        # Core must not substitute configured camera dimensions for this frame.
        detection_data["source_resolution"] = None
        # Keep the exact source dimensions with the observation. Geometry is
        # canonicalized below, but Core uses this value for camera provenance
        # and any legacy pixel payloads; substituting configured dimensions
        # would make that metadata inaccurate when the camera mode changes.
        if (
            isinstance(frame_resolution, (tuple, list))
            and len(frame_resolution) == 2
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value > 0
                and float(value).is_integer()
                for value in frame_resolution
            )
        ):
            detection_data["source_resolution"] = [
                int(frame_resolution[0]),
                int(frame_resolution[1]),
            ]

        # Convert balls. Each serializer resolves the object's own source
        # space and resolution, falling back only to this exact DetectionResult
        # frame size for pixel data without object-local resolution metadata.
        if detection.balls:
            detection_data["balls"] = [
                serialize_vision_ball(ball, frame_resolution)
                for ball in detection.balls
            ]

        # Convert cue
        if detection.cue:
            detection_data["cue"] = serialize_vision_cue(
                detection.cue, frame_resolution
            )

        # Convert table
        if detection.table:
            detection_data["table"] = serialize_vision_table(
                detection.table, frame_resolution
            )

        return detection_data

    @staticmethod
    def _phase_name(detection: DetectionResult) -> Optional[str]:
        """Return a normalized scheduler phase when the detector provides one."""
        value = getattr(detection, "phase", None)
        if value is None:
            return None
        value = getattr(value, "value", value)
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _observation_status(
        cls, detection: DetectionResult
    ) -> Optional[dict[str, Any]]:
        """Build the shared, serializable vision eligibility/status contract."""
        phase = cls._phase_name(detection)
        fields = (
            "phase_aware",
            "map_revision",
            "map_trusted",
            "aim_calculation_eligible",
            "component_observation_timestamps",
            "component_observation_status",
            "source_frame_number",
            "source_timestamp",
            "source_timestamp_kind",
            "source_publication_monotonic_ns",
            "frame_publication_age_seconds",
            "ball_positions_status",
            "ball_detection_status",
            "cue_detection_status",
            "motion_monitor_status",
            "ball_observation_source_timestamp",
            "ball_observation_source_frame_number",
            "ball_observation_source_timestamp_kind",
            "ball_observation_timestamp",
            "maximum_ball_observation_gap_seconds",
            "actual_path_history_max_points",
            "ball_detection_view",
            "calibration_revision",
            "moving_ball_count",
            "map_observation_fresh",
            "shot_detected",
            "balls_in_motion",
            "game_state",
        )
        has_contract = phase is not None or any(
            hasattr(detection, field_name) for field_name in fields
        )
        if not has_contract:
            return None

        explicit_phase_awareness = getattr(detection, "phase_aware", None)
        status: dict[str, Any] = {
            "phase_aware": (
                bool(explicit_phase_awareness)
                if explicit_phase_awareness is not None
                else phase is not None
            ),
            "phase": phase,
        }
        for field_name in fields:
            if hasattr(detection, field_name):
                value = getattr(detection, field_name)
                if value is not None:
                    status[field_name] = copy.deepcopy(value)
        return status

    @staticmethod
    def _phase_blocks_live_prediction(phase: Optional[str]) -> bool:
        """Motion and recovery phases may compare predictions, never calculate."""
        return phase in {
            "strike",
            "limited_motion",
            "many_balls_moving",
            "many_moving",
            "recovery",
            "recovering",
            "uncertain",
        }

    def _live_prediction_is_eligible(self, detection: DetectionResult) -> bool:
        """Apply the same explicit scheduler gate to automatic calculations."""
        status = self._observation_status(detection)
        return self._observation_status_is_eligible(status, detection)

    def _observation_status_is_eligible(
        self,
        status: Optional[dict[str, Any]],
        detection: Optional[DetectionResult] = None,
    ) -> bool:
        """Apply the scheduler gate to a detection or its saved status."""
        scheduler_enabled = bool(
            getattr(self, "config", None)
            and self.config.get("vision.processing.phase_scheduler.enabled", False)
        )
        if status and self._phase_blocks_live_prediction(status.get("phase")):
            return False
        if status is None or not status.get("phase_aware"):
            if scheduler_enabled:
                return False
            # Legacy mode remains available until the opt-in scheduler is
            # enabled. Existing callers still cannot calculate during a
            # directly reported shot/motion event. At completion time there
            # is no DetectionResult, so also consult the saved status snapshot.
            game_state = (
                status.get("game_state")
                if status is not None and "game_state" in status
                else getattr(detection, "game_state", "")
            )
            return not bool(
                (status or {}).get("shot_detected", False)
                or getattr(detection, "shot_detected", False)
                or (status or {}).get("balls_in_motion", 0)
                or getattr(detection, "balls_in_motion", 0)
                or str(game_state).lower() in {"moving", "shooting", "strike"}
            )

        phase = status.get("phase")
        if self._phase_blocks_live_prediction(phase):
            return False
        if status.get("phase_aware") and not (
            status.get("aim_calculation_eligible") is True
            and status.get("map_trusted") is True
        ):
            return False
        if status.get("aim_calculation_eligible") is False:
            return False
        if status.get("map_trusted") is False:
            return False
        if status.get("map_observation_fresh") is False:
            return False
        component_status = status.get("component_observation_status") or {}
        cue_status = component_status.get("cue")
        return not (
            isinstance(cue_status, str)
            and cue_status.lower() in {"stale", "missing", "failed", "uncertain"}
        )

    def _prediction_request_is_current(self, request: _PredictionRequest) -> bool:
        """Reject results whose phase, map, calibration, or age has changed."""
        if not getattr(self, "running", False):
            return False
        if request.lifecycle_generation != self._prediction_lifecycle_generation:
            return False

        current = getattr(self, "latest_observation_status", None)
        if not self._observation_status_is_eligible(current):
            return False
        if current is None:
            if any(
                request.metadata.get(field) is not None
                for field in ("phase", "map_revision", "calibration_revision")
            ):
                return False
        else:
            if current.get("phase") != request.metadata.get("phase"):
                return False
            if current.get("map_revision") != request.metadata.get("map_revision"):
                return False
            if current.get("calibration_revision") != request.metadata.get(
                "calibration_revision"
            ):
                return False
            current_age = current.get("frame_publication_age_seconds")
            received_at = self._latest_observation_received_at
            if current_age is not None and received_at is not None:
                if (
                    float(current_age) + time.monotonic() - received_at
                    > request.maximum_observation_age_seconds
                ):
                    return False

        if request.observation_age_seconds is not None:
            age = request.observation_age_seconds + (
                time.monotonic() - request.requested_at
            )
            if age > request.maximum_observation_age_seconds:
                return False
        return True

    def _queue_prediction(self, request: _PredictionRequest) -> None:
        """Keep only the newest request and supervise a single worker task."""
        previous = self._pending_prediction
        if previous is not None:
            self._record_audit_event(
                "trajectory",
                {
                    "event_type": "trajectory_request_coalesced",
                    "replaced_source_frame_number": previous.metadata.get(
                        "source_frame_number"
                    ),
                    "source_frame_number": request.metadata.get("source_frame_number"),
                },
            )
        self._pending_prediction = request
        task = self._prediction_task
        if task is None or task.done():
            self._prediction_task = asyncio.create_task(
                self._prediction_worker(), name="integration-trajectory-prediction"
            )
            self._prediction_task.add_done_callback(self._observe_prediction_task)

    def _create_prediction_request(
        self,
        *,
        cue_state: CueState,
        ball_state: BallState,
        table_state: TableState,
        other_balls: list[BallState],
        detection: DetectionResult,
        status: dict[str, Any],
    ) -> _PredictionRequest:
        """Capture owned physics inputs and immutable observation provenance."""
        return _PredictionRequest(
            cue_state=copy.deepcopy(cue_state),
            ball_state=copy.deepcopy(ball_state),
            table_state=copy.deepcopy(table_state),
            other_balls=copy.deepcopy(other_balls),
            quality=self.trajectory_quality,
            max_collision_depth=self.max_collision_depth,
            metadata=copy.deepcopy(self._prediction_metadata(detection, status)),
            requested_at=time.monotonic(),
            observation_age_seconds=status.get("frame_publication_age_seconds"),
            maximum_observation_age_seconds=(
                self._maximum_prediction_observation_age_seconds
            ),
            lifecycle_generation=self._prediction_lifecycle_generation,
        )

    @staticmethod
    def _observe_prediction_task(task: asyncio.Task[None]) -> None:
        """Retrieve unexpected task failures so they cannot go unnoticed."""
        if task.cancelled():
            logger.warning("Background trajectory prediction task was cancelled")
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Background trajectory prediction task failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _calculate_multiball_prediction(
        self, request: _PredictionRequest
    ) -> MultiballTrajectoryResult:
        """Run physics with detached values and serialized calculator access."""
        worker_started_ns = time.perf_counter_ns()
        executor_submitted_ns = request.timing.pop(
            "executor_submitted_monotonic_ns", None
        )
        request.timing["executor_wait_seconds"] = (
            max(0, worker_started_ns - executor_submitted_ns) / 1e9
            if isinstance(executor_submitted_ns, int)
            else None
        )
        lock_started_ns = worker_started_ns
        with self._trajectory_calculation_lock:
            compute_started_ns = time.perf_counter_ns()
            result = self.trajectory_calculator.predict_multiball_cue_shot(
                cue_state=request.cue_state,
                ball_state=request.ball_state,
                table_state=request.table_state,
                other_balls=request.other_balls,
                quality=request.quality,
                max_collision_depth=request.max_collision_depth,
            )
            compute_finished_ns = time.perf_counter_ns()
        request.timing["calculator_lock_wait_seconds"] = (
            max(0, compute_started_ns - lock_started_ns) / 1e9
        )
        request.timing["physics_compute_seconds"] = (
            max(0, compute_finished_ns - compute_started_ns) / 1e9
        )
        return result

    async def _prediction_worker(self) -> None:
        """Calculate queued predictions without blocking frame processing."""
        try:
            while self._pending_prediction is not None:
                request = self._pending_prediction
                self._pending_prediction = None
                worker_started = time.monotonic()
                request.timing["prediction_queue_wait_seconds"] = max(
                    0.0, worker_started - request.requested_at
                )
                request.timing["executor_submitted_monotonic_ns"] = (
                    time.perf_counter_ns()
                )
                try:
                    result = await asyncio.to_thread(
                        self._calculate_multiball_prediction, request
                    )
                except Exception:
                    logger.exception(
                        "Background trajectory calculation failed for source frame %s",
                        request.metadata.get("source_frame_number"),
                    )
                    self._record_audit_event(
                        "trajectory",
                        {
                            "event_type": "trajectory_prediction_failed",
                            "source_frame_number": request.metadata.get(
                                "source_frame_number"
                            ),
                            **request.timing,
                        },
                    )
                    continue

                if not self._prediction_request_is_current(request):
                    logger.debug(
                        "Discarding stale trajectory result for source frame %s",
                        request.metadata.get("source_frame_number"),
                    )
                    self._record_audit_event(
                        "trajectory",
                        {
                            "event_type": "trajectory_prediction_discarded_stale",
                            "source_frame_number": request.metadata.get(
                                "source_frame_number"
                            ),
                            **request.timing,
                        },
                    )
                    continue

                publish_started = time.perf_counter()
                try:
                    await self._emit_multiball_trajectory(
                        result, prediction_metadata=request.metadata
                    )
                except Exception:
                    logger.exception(
                        "Failed to publish trajectory for source frame %s",
                        request.metadata.get("source_frame_number"),
                    )
                    request.timing["trajectory_publication_seconds"] = max(
                        0.0, time.perf_counter() - publish_started
                    )
                    self._record_audit_event(
                        "trajectory",
                        {
                            "event_type": "trajectory_publication_failed",
                            "source_frame_number": request.metadata.get(
                                "source_frame_number"
                            ),
                            **request.timing,
                        },
                    )
                    continue

                request.timing["trajectory_publication_seconds"] = max(
                    0.0, time.perf_counter() - publish_started
                )
                self._record_audit_event(
                    "trajectory",
                    {
                        "event_type": "trajectory_prediction_published",
                        "source_frame_number": request.metadata.get(
                            "source_frame_number"
                        ),
                        "source_timestamp": request.metadata.get("source_timestamp"),
                        "prediction_status": "live",
                        **request.timing,
                    },
                )

                total_lines = sum(
                    max(len(trajectory.points) - 1, 0)
                    for trajectory in result.trajectories.values()
                )
                total_collisions = sum(
                    len(trajectory.collisions)
                    for trajectory in result.trajectories.values()
                )
                logger.info(
                    "Trajectory calculated and broadcast - %s ball(s) affected, "
                    "%s line segments, %s collisions",
                    len(result.trajectories),
                    total_lines,
                    total_collisions,
                )
        finally:
            # stop() never cancels this task; it clears pending work and waits
            # for this wrapper to drain the active physics thread.
            if self._prediction_task is asyncio.current_task():
                self._prediction_task = None

    async def _refresh_prediction_lifecycle(self, detection: DetectionResult) -> None:
        """Freeze a prior valid prediction on motion and invalidate stale geometry."""
        status = self._observation_status(detection)
        if status is None:
            return

        phase = status.get("phase")
        calibration_revision = status.get("calibration_revision")
        calibration_changed = (
            self._last_calibration_revision is not None
            and calibration_revision is not None
            and calibration_revision != self._last_calibration_revision
        )
        if calibration_revision is not None:
            self._last_calibration_revision = calibration_revision

        phase_entered_motion = self._phase_blocks_live_prediction(
            phase
        ) and not self._phase_blocks_live_prediction(self._last_vision_phase)
        if phase_entered_motion and self.frozen_comparison_prediction is None:
            snapshot = self._freeze_last_live_prediction(phase)
            if snapshot is not None and not calibration_changed:
                await self._broadcast_prediction_snapshot(snapshot)

        if calibration_changed:
            snapshot = self.frozen_comparison_prediction
            if snapshot is None:
                snapshot = self._freeze_last_live_prediction(phase)
            if snapshot is not None:
                invalidated = copy.deepcopy(snapshot)
                invalidated["spatial_valid"] = False
                invalidated["prediction_status"] = "invalidated"
                invalidated["phase"] = phase
                self.frozen_comparison_prediction = invalidated
                await self._broadcast_prediction_snapshot(
                    self.frozen_comparison_prediction
                )

        self._last_vision_phase = phase

    def _freeze_last_live_prediction(
        self, phase: Optional[str]
    ) -> Optional[dict[str, Any]]:
        """Copy the exact last valid live prediction as a retained reference."""
        if self.frozen_comparison_prediction is not None:
            return self.frozen_comparison_prediction
        if self.last_live_prediction is None:
            return None
        frozen = copy.deepcopy(self.last_live_prediction)
        frozen["prediction_status"] = "frozen_reference"
        frozen["spatial_valid"] = (
            frozen.get("calibration_revision") == self._last_calibration_revision
            or self._last_calibration_revision is None
        )
        frozen["phase"] = phase
        self.frozen_comparison_prediction = frozen
        return frozen

    async def _broadcast_prediction_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Send a saved prediction without recalculating or altering its geometry."""
        lines = copy.deepcopy(snapshot.get("lines", []))
        if len(lines) < 2:
            # There is no valid geometry to show. Never create an empty or
            # synthetic trajectory just to signal a phase transition.
            return
        collisions = copy.deepcopy(snapshot.get("collisions", []))
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_trajectory,
            "broadcast_prediction_reference",
            f"{len(lines)} retained lines, {len(collisions)} collisions",
            lines,
            collisions,
            confidence=snapshot.get("confidence", 1.0),
            calculation_time_ms=snapshot.get("calculation_time_ms", 0.0),
            prediction_status=snapshot.get("prediction_status", "frozen_reference"),
            spatial_valid=snapshot.get("spatial_valid", True),
            phase=snapshot.get("phase"),
            source_frame_number=snapshot.get("source_frame_number"),
            source_timestamp=snapshot.get("source_timestamp"),
            source_timestamp_kind=snapshot.get("source_timestamp_kind"),
            map_revision=snapshot.get("map_revision"),
            calibration_revision=snapshot.get("calibration_revision"),
            prediction_settings=copy.deepcopy(snapshot.get("prediction_settings", {})),
            source_observations=copy.deepcopy(snapshot.get("source_observations", {})),
        )

    def _find_ball_cue_is_pointing_at(self, cue: Any, balls: Any) -> Optional[Any]:
        """Find which ball the cue is currently pointing at.

        Args:
            cue: Detected cue stick with tip and tail position
            balls: List of detected balls

        Returns:
            The ball the cue is pointing at, or None
        """
        if not cue or not balls:
            return None

        # Calculate cue direction vector
        cue_dx = cue.tip_position[0] - cue.butt_position[0]
        cue_dy = cue.tip_position[1] - cue.butt_position[1]
        cue_length = np.sqrt(cue_dx**2 + cue_dy**2)

        if cue_length == 0:
            return None

        # Normalize direction
        cue_dx /= cue_length
        cue_dy /= cue_length

        # Find the closest ball along the cue direction
        closest_ball = None
        min_distance = float("inf")
        max_perpendicular_distance = 40  # pixels tolerance

        for ball in balls:
            # Vector from cue tip to ball center
            ball_dx = ball.position[0] - cue.tip_position[0]
            ball_dy = ball.position[1] - cue.tip_position[1]

            # Distance along cue direction (projection)
            distance_along_cue = ball_dx * cue_dx + ball_dy * cue_dy

            # Skip balls behind the cue tip
            if distance_along_cue < 0:
                continue

            # Calculate perpendicular distance from cue line to ball center
            perpendicular_distance = abs(ball_dx * cue_dy - ball_dy * cue_dx)

            # Check if ball is within tolerance and closer than current closest
            if (
                perpendicular_distance < max_perpendicular_distance
                and distance_along_cue < min_distance
            ):
                min_distance = distance_along_cue
                closest_ball = ball

        return closest_ball

    async def _check_trajectory_calculation(self, detection: DetectionResult) -> None:
        """Check if trajectory should be calculated and trigger if needed.

        Args:
            detection: Current detection result
        """
        # Increment trajectory frame counter for periodic logging
        self.trajectory_frame_count += 1
        should_log = (self.trajectory_frame_count % 30) == 0  # Log every 30 frames
        # Use this observation's provenance for any prediction request, but do
        # not mutate the latest-applied pointer here: another Core update may
        # already have committed while this optional trajectory work awaited.
        status = self._observation_status_with_identity(detection)

        # Observe phase transitions before considering this frame for a new
        # calculation. The retained reference is copied from the exact last
        # successful live prediction and is never recomputed during motion.
        await self._refresh_prediction_lifecycle(detection)

        if not self._live_prediction_is_eligible(detection):
            self._pending_prediction = None
            if should_log:
                logger.debug(
                    "Trajectory calculation skipped: phase/map/cue status is not eligible "
                    "for a live aiming prediction"
                )
            return

        # This observation supersedes any queued candidate even if it later
        # proves incomplete (for example, the cue or target ball disappeared).
        self._pending_prediction = None

        # Only calculate trajectory if:
        # 1. Cue is detected (removed aiming state requirement for more flexibility)
        # 2. All balls are stationary
        # 3. At least one ball is detected

        if not detection.cue:
            if should_log:
                logger.debug("Trajectory calculation skipped: No cue detected")
            return

        # Log cue detection with parameters
        if should_log:
            logger.debug(
                f"Cue detected - angle: {detection.cue.angle:.2f}deg, "
                f"tip: ({detection.cue.tip_position[0]:.1f}, {detection.cue.tip_position[1]:.1f}), "
                f"confidence: {detection.cue.confidence:.2f}"
            )

        # Check if all balls are stationary (no velocity data in detection)
        # This is a simplification - in real implementation, check Core state
        if not detection.balls:
            if should_log:
                logger.warning("Trajectory calculation skipped: No balls detected")
            return

        if should_log:
            logger.debug(
                f"Ball positions being used for trajectory: {len(detection.balls)} balls - "
                + ", ".join(
                    [
                        f"ball_{i}@({ball.position[0]:.1f},{ball.position[1]:.1f})"
                        for i, ball in enumerate(detection.balls)
                    ]
                )
            )

        # Find the ball the cue is pointing at (could be any ball, not just cue ball)
        target_ball = self._find_ball_cue_is_pointing_at(detection.cue, detection.balls)
        if not target_ball:
            if should_log:
                logger.warning(
                    f"Trajectory calculation skipped: No target ball found in cue direction. "
                    f"Cue pointing at angle {detection.cue.angle:.2f}deg from "
                    f"({detection.cue.tip_position[0]:.1f}, {detection.cue.tip_position[1]:.1f})"
                )
            return

        if should_log:
            logger.debug(
                f"Target ball found at ({target_ball.position[0]:.1f}, {target_ball.position[1]:.1f}), "
                f"type: {target_ball.ball_type.value if target_ball.ball_type else 'unknown'}"
            )

        # Calculate multiball trajectory using trajectory calculator directly
        try:
            if should_log:
                logger.debug("Starting trajectory calculation...")

            # Create CueState from detected cue using state converter (with validation)
            cue_state = self.state_converter.vision_cue_to_cue_state(
                detection.cue, timestamp=detection.timestamp
            )
            if cue_state is None:
                logger.error(
                    "Failed to convert cue state - aborting trajectory calculation"
                )
                return

            # Create BallState for target ball using state converter (with validation)
            ball_state = self.state_converter.vision_ball_to_ball_state(
                target_ball, is_target=True, timestamp=detection.timestamp
            )
            if ball_state is None:
                logger.error(
                    "Failed to convert target ball state - aborting trajectory calculation"
                )
                return

            # Create list of BallState for other balls using state converter (with validation)
            other_ball_states = []
            for ball in detection.balls:
                if ball == target_ball:
                    continue  # Skip the target ball
                ball_state_converted = self.state_converter.vision_ball_to_ball_state(
                    ball, is_target=False, timestamp=detection.timestamp
                )
                if ball_state_converted is not None:
                    other_ball_states.append(ball_state_converted)

            if should_log:
                logger.debug(
                    f"Trajectory calculation inputs - "
                    f"cue angle: {cue_state.angle:.2f}deg, "
                    f"cue force: {cue_state.estimated_force:.2f}, "
                    f"target ball: ({ball_state.position.x:.1f}, {ball_state.position.y:.1f}), "
                    f"other balls: {len(other_ball_states)}"
                )

            # Get table state from Core's current state
            if not self.core._current_state:
                logger.warning(
                    "Trajectory calculation skipped: No current game state available. "
                    "Core module may not be initialized properly."
                )
                return

            table_state = self.core._current_state.table

            # Validate table state before trajectory calculation
            is_valid, validation_errors = (
                self.table_state_validator.validate_for_trajectory(
                    table_state, require_playing_area=False
                )
            )
            if not is_valid:
                logger.error(
                    f"Trajectory calculation skipped: Invalid table state. "
                    f"Errors: {'; '.join(validation_errors)}"
                )
                return

            # Log validation summary periodically
            if should_log:
                summary = self.table_state_validator.get_validation_summary(table_state)
                logger.debug(f"Table state validation: {summary}")

            prediction_status = status or {}
            request = self._create_prediction_request(
                cue_state=cue_state,
                ball_state=ball_state,
                table_state=table_state,
                other_balls=other_ball_states,
                detection=detection,
                status=prediction_status,
            )
            self._queue_prediction(request)

        except Exception as e:
            logger.error(
                f"Trajectory preparation failed - "
                f"Error: {str(e)}, "
                f"Context: cue_angle={detection.cue.angle:.2f}deg, "
                f"cue_tip=({detection.cue.tip_position[0]:.1f},{detection.cue.tip_position[1]:.1f}), "
                f"balls_count={len(detection.balls)}, "
                f"target_ball={'found' if target_ball else 'not_found'}",
                exc_info=True,
            )

    def _prediction_metadata(
        self, detection: DetectionResult, status: dict[str, Any]
    ) -> dict[str, Any]:
        """Capture immutable provenance needed to compare prediction to motion."""
        return {
            "prediction_status": "live",
            "spatial_valid": True,
            "phase": status.get("phase"),
            "source_frame_number": status.get(
                "source_frame_number", getattr(detection, "frame_number", None)
            ),
            "source_timestamp": status.get(
                "source_timestamp", getattr(detection, "timestamp", None)
            ),
            "source_timestamp_kind": status.get("source_timestamp_kind"),
            "source_publication_monotonic_ns": status.get(
                "source_publication_monotonic_ns"
            ),
            "map_revision": status.get("map_revision"),
            "calibration_revision": status.get("calibration_revision"),
            "source_observations": {
                "component_timestamps": copy.deepcopy(
                    status.get("component_observation_timestamps", {})
                ),
                "component_status": copy.deepcopy(
                    status.get("component_observation_status", {})
                ),
                "vision_instance_id": status.get("vision_instance_id"),
                "vision_process_id": status.get("vision_process_id"),
                "source_process_id": status.get("source_process_id"),
                "source_publication_monotonic_ns": status.get(
                    "source_publication_monotonic_ns"
                ),
                "map_observation_fresh": status.get("map_observation_fresh"),
                "map_trusted": status.get("map_trusted"),
            },
            "prediction_settings": {
                "trajectory_quality": self.trajectory_quality.value,
                "max_collision_depth": self.max_collision_depth,
                "shot_speed_estimate_m_per_s": self.shot_speed_estimate,
            },
        }

    def _estimate_velocity_from_cue(self, cue: Any) -> dict[str, float]:
        """Estimate initial velocity from cue angle and estimated force.

        Args:
            cue: Detected cue stick

        Returns:
            Velocity dictionary with vx, vy components
        """
        # Estimate force based on cue position/state
        # This is a simplified estimation - real implementation would use
        # cue motion tracking or user input
        estimated_speed = self.shot_speed_estimate

        # Convert angle to velocity components
        angle_rad = np.deg2rad(cue.angle)
        vx = estimated_speed * np.cos(angle_rad)
        vy = estimated_speed * np.sin(angle_rad)

        return {"vx": vx, "vy": vy, "vz": 0.0}

    def _classify_broadcast_error(self, error: Exception) -> BroadcastErrorType:
        """Classify a broadcast error to determine retry strategy.

        Args:
            error: The exception that occurred

        Returns:
            BroadcastErrorType indicating how to handle the error
        """
        error_message = str(error).lower()

        # Validation errors - don't retry
        validation_keywords = [
            "validation",
            "invalid",
            "schema",
            "type error",
            "value error",
            "missing required",
        ]
        if any(keyword in error_message for keyword in validation_keywords):
            return BroadcastErrorType.VALIDATION

        # Transient errors - retry
        transient_keywords = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "unavailable",
            "refused",
        ]
        if any(keyword in error_message for keyword in transient_keywords):
            return BroadcastErrorType.TRANSIENT

        # Unknown - retry with caution
        return BroadcastErrorType.UNKNOWN

    async def _broadcast_with_retry(
        self,
        broadcast_func: Callable[..., Any],
        operation_name: str,
        data_summary: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Execute a broadcast with retry logic and circuit breaker.

        Args:
            broadcast_func: Async function to call for broadcasting
            operation_name: Name of the operation for logging
            data_summary: Summary of data being broadcast for logging
            *args: Arguments to pass to broadcast function
            **kwargs: Keyword arguments to pass to broadcast function

        Returns:
            True if broadcast succeeded, False otherwise
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            logger.warning(
                f"Circuit breaker OPEN - skipping {operation_name}. "
                f"Status: {self.circuit_breaker.get_status()}"
            )
            self.broadcast_metrics.failed_broadcasts += 1
            return False

        last_error = None
        error_type = BroadcastErrorType.UNKNOWN

        # Attempt broadcast with retries
        for attempt in range(self.max_retries + 1):
            try:
                await broadcast_func(*args, **kwargs)

                # Success!
                self.circuit_breaker.record_success()
                self.broadcast_metrics.successful_broadcasts += 1

                if attempt > 0:
                    logger.info(
                        f"{operation_name} succeeded after {attempt} retries. "
                        f"Data: {data_summary}"
                    )

                return True

            except Exception as e:
                last_error = e
                error_type = self._classify_broadcast_error(e)

                # Log error with context
                logger.error(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Error type: {error_type.value}. "
                    f"Data: {data_summary}. "
                    f"Error: {str(e)}"
                )

                # Don't retry validation errors
                if error_type == BroadcastErrorType.VALIDATION:
                    logger.error(
                        f"{operation_name} validation error - will not retry. "
                        f"Data: {data_summary}. "
                        f"Error: {str(e)}"
                    )
                    break

                # Calculate exponential backoff delay for retry
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt)
                    logger.debug(f"Retrying {operation_name} in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    self.broadcast_metrics.total_retries += 1

        # All attempts failed
        self.circuit_breaker.record_failure()
        self.broadcast_metrics.failed_broadcasts += 1

        logger.error(
            f"{operation_name} failed after {self.max_retries + 1} attempts. "
            f"Error type: {error_type.value}. "
            f"Data: {data_summary}. "
            f"Final error: {str(last_error)}. "
            f"Action: Giving up. "
            f"Circuit breaker status: {self.circuit_breaker.get_status()}"
        )

        return False

    def _on_state_updated(self, event_type: str, event_data: dict[str, Any]) -> None:
        """Handle Core state update events.

        This is a synchronous callback that schedules async work in the background.

        Args:
            event_type: Type of event (e.g., "state_updated")
            event_data: Event data containing updated state (flattened GameState dict)
        """
        # Schedule async processing using the stored event loop
        if self._event_loop is None:
            logger.error(
                "No event loop available - cannot process state update. "
                "Integration service may not be started properly."
            )
            return

        # Schedule the coroutine on the stored event loop
        try:
            has_matching_observation, matching_status = (
                self._take_state_event_observation(event_data)
                if isinstance(event_data, dict)
                else (False, None)
            )
            # Use request-local status for the event emitted inside
            # Core.update_state. Public freshness remains tied to successful
            # completion of that call. Unrelated Core events retain the last
            # successfully applied observation status.
            observation_status = copy.deepcopy(
                matching_status
                if has_matching_observation
                else self.latest_observation_status
            )
            self._event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._on_state_updated_async(
                        event_type, event_data, observation_status
                    ),
                    loop=self._event_loop,
                )
            )
        except Exception as e:
            logger.error(f"Failed to schedule state update processing: {e}")

    async def _on_state_updated_async(
        self,
        event_type: str,
        event_data: Any,
        observation_status: Any = _UNSET_OBSERVATION_STATUS,
    ) -> None:
        """Async handler for Core state update events.

        Args:
            event_type: Type of event (e.g., "state_updated")
            event_data: Event data containing updated state (flattened GameState dict)
        """
        # Validate event_data structure
        if not isinstance(event_data, dict):
            logger.error(
                f"Invalid event_data type: expected dict, got {type(event_data).__name__}. "
                f"Event will be skipped."
            )
            return

        # Event data is now flattened (directly from asdict(GameState))
        # Validate required fields exist
        required_fields = ["balls", "table", "timestamp"]
        missing_fields = [field for field in required_fields if field not in event_data]
        if missing_fields:
            logger.error(
                f"Missing required fields in event_data: {missing_fields}. "
                f"Available keys: {list(event_data.keys())}. "
                "Event will be skipped."
            )
            return

        # Extract data for broadcasting (directly from flattened event_data)
        balls = event_data.get("balls", [])
        cue = event_data.get("cue")
        table = event_data.get("table")
        game_type = event_data.get("game_type")

        # Validate balls is a list
        if not isinstance(balls, list):
            logger.error(
                f"Invalid balls type: expected list, got {type(balls).__name__}. "
                f"Balls value: {balls}. "
                "Event will be skipped."
            )
            return

        # Validate each ball is a dict
        for i, ball in enumerate(balls):
            if not isinstance(ball, dict):
                logger.error(
                    f"Invalid ball type at index {i}: expected dict, got {type(ball).__name__}. "
                    f"Ball value: {ball}. "
                    "Event will be skipped."
                )
                return

        # Validate cue is dict or None
        if cue is not None and not isinstance(cue, dict):
            logger.error(
                f"Invalid cue type: expected dict or None, got {type(cue).__name__}. "
                f"Cue value: {cue}. "
                "Event will be skipped."
            )
            return

        # Validate table is dict or None
        if table is not None and not isinstance(table, dict):
            logger.error(
                f"Invalid table type: expected dict or None, got {type(table).__name__}. "
                f"Table value: {table}. "
                "Event will be skipped."
            )
            return

        # Balls are already in the correct format (dict with x, y, scale)
        # The broadcaster validates that positions are dicts with x, y, and scale keys
        balls_converted = balls

        # Create summary for logging
        data_summary = f"{len(balls)} balls, cue={'present' if cue else 'absent'}, table={'present' if table else 'absent'}"

        # Log detailed structure on first broadcast to help with debugging
        if not hasattr(self, "_logged_state_structure"):
            logger.debug(
                f"State structure sample - "
                f"balls keys: {list(balls[0].keys()) if balls else 'no balls'}, "
                f"cue keys: {list(cue.keys()) if cue else 'no cue'}, "
                f"table keys: {list(table.keys()) if table else 'no table'}"
            )
            self._logged_state_structure = True

        # Broadcast with retry logic
        broadcast_started_ns = time.perf_counter_ns()
        broadcast_succeeded = await self._broadcast_with_retry(
            self.broadcaster.broadcast_game_state,
            "broadcast_game_state",
            data_summary,
            balls_converted,
            cue,
            table,
            game_type=game_type,
            observation_status=copy.deepcopy(
                self.latest_observation_status
                if observation_status is _UNSET_OBSERVATION_STATUS
                else observation_status
            ),
        )
        observation = (
            observation_status
            if isinstance(observation_status, dict)
            else self.latest_observation_status
        )
        self._record_audit_event(
            "state",
            {
                "event_type": "state_publication_enqueued",
                "source_frame_number": (
                    observation.get("source_frame_number")
                    if isinstance(observation, dict)
                    else None
                ),
                "source_timestamp": (
                    observation.get("source_timestamp")
                    if isinstance(observation, dict)
                    else None
                ),
                "source_timestamp_kind": (
                    observation.get("source_timestamp_kind")
                    if isinstance(observation, dict)
                    else None
                ),
                "map_revision": (
                    observation.get("map_revision")
                    if isinstance(observation, dict)
                    else None
                ),
                "calibration_revision": (
                    observation.get("calibration_revision")
                    if isinstance(observation, dict)
                    else None
                ),
                "serialization_and_enqueue_seconds": max(
                    0, time.perf_counter_ns() - broadcast_started_ns
                )
                / 1e9,
                "broadcast_succeeded": broadcast_succeeded,
                "payload_ball_count": len(balls_converted),
            },
        )

    def _on_trajectory_calculated(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Handle trajectory calculation events.

        This is a synchronous callback that schedules async work in the background.

        Args:
            event_type: Type of event (e.g., "trajectory_calculated")
            event_data: Event data containing trajectory
        """
        # Schedule async processing using the stored event loop
        if self._event_loop is None:
            logger.error(
                "No event loop available - cannot process trajectory calculation. "
                "Integration service may not be started properly."
            )
            return

        # Schedule the coroutine on the stored event loop
        try:
            self._event_loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._on_trajectory_calculated_async(event_type, event_data),
                    loop=self._event_loop,
                )
            )
        except Exception as e:
            logger.error(f"Failed to schedule trajectory calculation processing: {e}")

    async def _on_trajectory_calculated_async(
        self, event_type: str, event_data: dict[str, Any]
    ) -> None:
        """Async handler for trajectory calculation events.

        Args:
            event_type: Type of event (e.g., "trajectory_calculated")
            event_data: Event data containing trajectory
        """
        trajectory = event_data.get("trajectory")
        if not trajectory:
            return

        status = self.latest_observation_status
        phase = (status or {}).get("phase")
        scheduler_enabled = bool(
            self.config.get("vision.processing.phase_scheduler.enabled", False)
        )
        if (
            self._phase_blocks_live_prediction(phase)
            or (
                status
                and status.get("phase_aware")
                and not (
                    status.get("aim_calculation_eligible") is True
                    and status.get("map_trusted") is True
                )
            )
            or (scheduler_enabled and not (status and status.get("phase_aware")))
        ):
            logger.debug(
                "Ignoring trajectory event because current vision status is not "
                "eligible for a live aim prediction"
            )
            return

        # Extract trajectory lines and collision points
        lines = trajectory.get("lines", [])
        collisions = trajectory.get("collisions", [])

        # Create summary for logging
        data_summary = f"{len(lines)} lines, {len(collisions)} collisions"

        if len(lines) < 2:
            return

        metadata = {
            "phase": phase,
            "source_frame_number": (status or {}).get("source_frame_number"),
            "source_timestamp": (status or {}).get("source_timestamp"),
            "source_timestamp_kind": (status or {}).get("source_timestamp_kind"),
            "source_publication_monotonic_ns": (status or {}).get(
                "source_publication_monotonic_ns"
            ),
            "map_revision": (status or {}).get("map_revision"),
            "calibration_revision": (status or {}).get("calibration_revision"),
            "prediction_settings": {},
            "source_observations": {
                "component_timestamps": (status or {}).get(
                    "component_observation_timestamps", {}
                ),
                "component_status": (status or {}).get(
                    "component_observation_status", {}
                ),
                "source_publication_monotonic_ns": (status or {}).get(
                    "source_publication_monotonic_ns"
                ),
                "map_observation_fresh": (status or {}).get("map_observation_fresh"),
                "map_trusted": (status or {}).get("map_trusted"),
            },
        }
        live_prediction = {
            "lines": copy.deepcopy(lines),
            "collisions": copy.deepcopy(collisions),
            "confidence": 1.0,
            "calculation_time_ms": event_data.get("calculation_time_ms", 0.0),
            **metadata,
            "prediction_status": "live",
            "spatial_valid": True,
        }
        self.last_live_prediction = live_prediction
        self.frozen_comparison_prediction = None

        # Broadcast with retry logic
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_trajectory,
            "broadcast_trajectory",
            data_summary,
            lines,
            collisions,
            confidence=live_prediction["confidence"],
            calculation_time_ms=live_prediction["calculation_time_ms"],
            prediction_status="live",
            spatial_valid=True,
            phase=phase,
            source_frame_number=live_prediction.get("source_frame_number"),
            source_timestamp=live_prediction.get("source_timestamp"),
            source_timestamp_kind=live_prediction.get("source_timestamp_kind"),
            map_revision=live_prediction.get("map_revision"),
            calibration_revision=live_prediction.get("calibration_revision"),
            prediction_settings={},
            source_observations=live_prediction.get("source_observations", {}),
        )

    async def _emit_multiball_trajectory(
        self,
        multiball_result: MultiballTrajectoryResult,
        prediction_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Convert MultiballTrajectoryResult to event format and emit/broadcast.

        Args:
            multiball_result: Multiball trajectory calculation result
        """
        # Collect all lines and collisions from all ball trajectories
        all_lines = []
        all_collisions = []

        # Process each trajectory in the result
        for ball_id, trajectory in multiball_result.trajectories.items():
            # Determine line type: primary for cue ball, secondary for hit balls
            line_type = (
                "primary"
                if ball_id == multiball_result.primary_ball_id
                else "secondary"
            )

            # Convert trajectory points to line segments
            if trajectory.points and len(trajectory.points) >= 2:
                for i in range(len(trajectory.points) - 1):
                    start_point = trajectory.points[i]
                    end_point = trajectory.points[i + 1]
                    all_lines.append(
                        {
                            "start": self._trajectory_position_payload(
                                start_point.position
                            ),
                            "end": self._trajectory_position_payload(
                                end_point.position
                            ),
                            "type": line_type,
                            "ball_id": ball_id,
                        }
                    )

            # Convert collisions to event format
            for collision in trajectory.collisions:
                collision_position = self._trajectory_position_payload(
                    collision.position
                )
                all_collisions.append(
                    {
                        "time": collision.time,
                        "position": collision_position,
                        "type": collision.type.value,
                        "ball1_id": collision.ball1_id,  # Moving ball
                        "ball2_id": collision.ball2_id,  # Target ball (None for cushion/pocket)
                        "angle": collision.impact_angle,
                        "confidence": collision.confidence,
                    }
                )

        # Create summary for logging
        data_summary = (
            f"{len(all_lines)} lines, {len(all_collisions)} collisions "
            f"across {len(multiball_result.trajectories)} ball(s)"
        )

        if len(all_lines) < 2:
            logger.debug(
                "Trajectory result has fewer than two line segments; "
                "it is not retained as a valid prediction"
            )
            return

        metadata = copy.deepcopy(prediction_metadata or {})
        # Retain a detached copy before broadcasting. A later phase transition
        # must snapshot these exact segments and collisions, not recalculate.
        live_prediction = {
            "lines": copy.deepcopy(all_lines),
            "collisions": copy.deepcopy(all_collisions),
            "confidence": 1.0,
            "calculation_time_ms": multiball_result.total_calculation_time * 1000.0,
            **metadata,
        }
        self.last_live_prediction = live_prediction
        self.frozen_comparison_prediction = None

        # Broadcast with retry logic
        await self._broadcast_with_retry(
            self.broadcaster.broadcast_trajectory,
            "broadcast_multiball_trajectory",
            data_summary,
            all_lines,
            all_collisions,
            confidence=live_prediction["confidence"],
            calculation_time_ms=live_prediction["calculation_time_ms"],
            prediction_status="live",
            spatial_valid=True,
            phase=live_prediction.get("phase"),
            source_frame_number=live_prediction.get("source_frame_number"),
            source_timestamp=live_prediction.get("source_timestamp"),
            source_timestamp_kind=live_prediction.get("source_timestamp_kind"),
            map_revision=live_prediction.get("map_revision"),
            calibration_revision=live_prediction.get("calibration_revision"),
            prediction_settings=live_prediction.get("prediction_settings", {}),
            source_observations=live_prediction.get("source_observations", {}),
        )

    @staticmethod
    def _trajectory_position_payload(position: Vector2D) -> dict[str, Any]:
        """Serialize a Core vector as canonical x/y with explicit scale metadata."""
        canonical = position.to_4k_canonical()
        return {
            "x": canonical.x,
            "y": canonical.y,
            "scale": [canonical.scale[0], canonical.scale[1]],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get integration service statistics.

        Returns:
            Dictionary with service statistics
        """
        return {
            "running": self.running,
            "frames_processed": self.frame_count,
            "duplicate_observations": self.duplicate_observation_count,
            "errors": self.error_count,
            "error_rate": (
                self.error_count / self.frame_count if self.frame_count > 0 else 0
            ),
            "broadcast_metrics": {
                "successful_broadcasts": self.broadcast_metrics.successful_broadcasts,
                "failed_broadcasts": self.broadcast_metrics.failed_broadcasts,
                "total_retries": self.broadcast_metrics.total_retries,
                "success_rate_percent": self.broadcast_metrics.success_rate(),
            },
            "circuit_breaker": self.circuit_breaker.get_status(),
        }
