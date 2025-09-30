"""Integration interfaces for core module communication with other backend modules."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from .events.manager import EventManager, EventType
from .game_state import GameStateManager
from .models import BallState, GameState, ShotAnalysis, Trajectory

logger = logging.getLogger(__name__)


class ModuleInterface(Protocol):
    """Protocol for module interfaces."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the module with configuration."""
        ...

    def shutdown(self) -> None:
        """Shutdown the module."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Get module status."""
        ...


@dataclass
class ModuleCapabilities:
    """Module capabilities description."""

    name: str
    version: str
    features: list[str]
    requirements: list[str]
    interfaces: list[str]


class VisionInterface(ABC):
    """Interface for Vision module integration (FR-CORE-051)."""

    @abstractmethod
    def receive_detection_data(self, detection_data: dict[str, Any]) -> None:
        """Receive detection data from Vision module."""
        raise NotImplementedError(
            "Vision interface must implement receive_detection_data"
        )

    @abstractmethod
    def request_calibration(self, calibration_type: str) -> dict[str, Any]:
        """Request calibration data from Vision module."""
        raise NotImplementedError("Vision interface must implement request_calibration")

    @abstractmethod
    def set_detection_parameters(self, parameters: dict[str, Any]) -> bool:
        """Set detection parameters in Vision module."""
        raise NotImplementedError(
            "Vision interface must implement set_detection_parameters"
        )


class APIInterface(ABC):
    """Interface for API module integration (FR-CORE-052)."""

    @abstractmethod
    def send_state_update(self, state_data: dict[str, Any]) -> None:
        """Send state updates to API module."""
        raise NotImplementedError("API interface must implement send_state_update")

    @abstractmethod
    def send_event_notification(self, event_data: dict[str, Any]) -> None:
        """Send event notifications to API module."""
        raise NotImplementedError(
            "API interface must implement send_event_notification"
        )

    @abstractmethod
    def register_websocket_handler(self, handler: Callable) -> str:
        """Register WebSocket message handler."""
        raise NotImplementedError(
            "API interface must implement register_websocket_handler"
        )


class ProjectorInterface(ABC):
    """Interface for Projector module integration (FR-CORE-053)."""

    @abstractmethod
    def send_trajectory_data(self, trajectory_data: dict[str, Any]) -> None:
        """Send trajectory data to Projector module."""
        raise NotImplementedError(
            "Projector interface must implement send_trajectory_data"
        )

    @abstractmethod
    def send_overlay_data(self, overlay_data: dict[str, Any]) -> None:
        """Send overlay visualization data."""
        raise NotImplementedError(
            "Projector interface must implement send_overlay_data"
        )

    @abstractmethod
    def update_projection_settings(self, settings: dict[str, Any]) -> None:
        """Update projector settings."""
        raise NotImplementedError(
            "Projector interface must implement update_projection_settings"
        )


class ConfigInterface(ABC):
    """Interface for Config module integration (FR-CORE-054)."""

    @abstractmethod
    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        raise NotImplementedError("Config interface must implement get_module_config")

    @abstractmethod
    def update_module_config(self, module_name: str, config: dict[str, Any]) -> None:
        """Update configuration for a specific module."""
        raise NotImplementedError(
            "Config interface must implement update_module_config"
        )

    @abstractmethod
    def subscribe_config_changes(self, callback: Callable) -> str:
        """Subscribe to configuration changes."""
        raise NotImplementedError(
            "Config interface must implement subscribe_config_changes"
        )


class CoreModuleIntegrator:
    """Main integration coordinator for Core module (FR-CORE-051 to FR-CORE-055)."""

    def __init__(
        self, event_manager: EventManager, game_state_manager: GameStateManager
    ):
        """Initialize module integrator.

        Args:
            event_manager: EventManager instance
            game_state_manager: GameStateManager instance
        """
        self.event_manager = event_manager
        self.game_state_manager = game_state_manager
        self.logger = logging.getLogger(__name__)

        # Module interfaces
        self.vision_interface: Optional[VisionInterface] = None
        self.api_interface: Optional[APIInterface] = None
        self.projector_interface: Optional[ProjectorInterface] = None
        self.config_interface: Optional[ConfigInterface] = None

        # Integration state
        self.connected_modules: dict[str, ModuleCapabilities] = {}
        self.integration_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "last_activity": time.time(),
        }

        # Setup integration event handlers
        self._setup_integration_handlers()

    def _setup_integration_handlers(self):
        """Setup event handlers for module integration."""
        # Vision data reception (FR-CORE-051)
        self.event_manager.subscribe_to_events(
            EventType.VISION_DATA_RECEIVED.value, self._handle_vision_data
        )

        # State updates for API (FR-CORE-052)
        self.event_manager.subscribe_to_events(
            EventType.STATE_UPDATED.value, self._handle_state_update
        )

        # Trajectory data for Projector (FR-CORE-053)
        self.event_manager.subscribe_to_events(
            EventType.TRAJECTORY_CALCULATED.value, self._handle_trajectory_data
        )

        # Configuration changes (FR-CORE-054)
        self.event_manager.subscribe_to_events(
            EventType.CONFIG_CHANGED.value, self._handle_config_change
        )

        # Module lifecycle events (FR-CORE-055)
        self.event_manager.subscribe_to_events(
            EventType.MODULE_INITIALIZED.value, self._handle_module_initialization
        )

        self.event_manager.subscribe_to_events(
            EventType.MODULE_SHUTDOWN.value, self._handle_module_shutdown
        )

    def register_vision_interface(self, interface: VisionInterface) -> None:
        """Register Vision module interface."""
        self.vision_interface = interface
        self.logger.info("Vision interface registered")

        # Register module with coordinator
        self.event_manager.coordinator.register_module(
            "vision", interface, ["detection", "tracking", "calibration"]
        )

    def register_api_interface(self, interface: APIInterface) -> None:
        """Register API module interface."""
        self.api_interface = interface
        self.logger.info("API interface registered")

        self.event_manager.coordinator.register_module(
            "api", interface, ["websocket", "rest", "state_broadcast"]
        )

    def register_projector_interface(self, interface: ProjectorInterface) -> None:
        """Register Projector module interface."""
        self.projector_interface = interface
        self.logger.info("Projector interface registered")

        self.event_manager.coordinator.register_module(
            "projector", interface, ["visualization", "overlay", "calibration"]
        )

    def register_config_interface(self, interface: ConfigInterface) -> None:
        """Register Config module interface."""
        self.config_interface = interface
        self.logger.info("Config interface registered")

        self.event_manager.coordinator.register_module(
            "config", interface, ["configuration", "persistence", "validation"]
        )

    def _handle_vision_data(self, event_type: str, data: dict[str, Any]):
        """Handle vision detection data reception (FR-CORE-051)."""
        detection_data = data.get("detection_data", {})

        try:
            # Process through game state manager
            updated_state = self.game_state_manager.update_state(detection_data)

            # Forward processed data to other modules
            self._broadcast_state_update(updated_state)

            self.integration_stats["messages_received"] += 1
            self.integration_stats["last_activity"] = time.time()

        except Exception as e:
            self.integration_stats["errors"] += 1
            self.logger.error(f"Error processing vision data: {e}")

    def _handle_state_update(self, event_type: str, data: dict[str, Any]):
        """Handle state updates for API module (FR-CORE-052)."""
        if self.api_interface:
            try:
                self.api_interface.send_state_update(data)
                self.integration_stats["messages_sent"] += 1
            except Exception as e:
                self.integration_stats["errors"] += 1
                self.logger.error(f"Error sending state update to API: {e}")

    def _handle_trajectory_data(self, event_type: str, data: dict[str, Any]):
        """Handle trajectory data for Projector module (FR-CORE-053)."""
        if self.projector_interface:
            try:
                trajectory_data = data.get("trajectory_data", {})
                self.projector_interface.send_trajectory_data(trajectory_data)
                self.integration_stats["messages_sent"] += 1
            except Exception as e:
                self.integration_stats["errors"] += 1
                self.logger.error(f"Error sending trajectory data to Projector: {e}")

    def _handle_config_change(self, event_type: str, data: dict[str, Any]):
        """Handle configuration changes (FR-CORE-054)."""
        module_name = data.get("module_name")
        config = data.get("config")

        if module_name == "core" and config:
            # Apply core configuration changes
            self._apply_core_configuration(config)

        # Forward config changes to relevant modules
        if module_name and config:
            self._forward_config_change(module_name, config)

    def _handle_module_initialization(self, event_type: str, data: dict[str, Any]):
        """Handle module initialization (FR-CORE-055)."""
        module_name = data.get("module_name")
        capabilities = data.get("capabilities", [])

        if module_name:
            self.connected_modules[module_name] = ModuleCapabilities(
                name=module_name,
                version=data.get("version", "1.0.0"),
                features=capabilities,
                requirements=data.get("requirements", []),
                interfaces=data.get("interfaces", []),
            )

            self.logger.info(
                f"Module {module_name} initialized with capabilities: {capabilities}"
            )

            # Send welcome configuration
            self._send_welcome_configuration(module_name)

    def _handle_module_shutdown(self, event_type: str, data: dict[str, Any]):
        """Handle module shutdown (FR-CORE-055)."""
        module_name = data.get("module_name")

        if module_name in self.connected_modules:
            del self.connected_modules[module_name]
            self.logger.info(f"Module {module_name} disconnected")

            # Clean up module-specific resources
            self._cleanup_module_resources(module_name)

    def _broadcast_state_update(self, state: GameState):
        """Broadcast state update to all connected modules."""
        state_data = {
            "timestamp": state.timestamp,
            "frame_number": state.frame_number,
            "balls": [self._serialize_ball(ball) for ball in state.balls],
            "table": state.table.to_dict(),
            "cue": state.cue.to_dict() if state.cue else None,
            "game_type": state.game_type.value,
            "is_break": state.is_break,
            "scores": state.scores,
            "events": [event.to_dict() for event in state.events],
        }

        # Send to API module
        self.event_manager.send_state_update_to_api(state_data)

        # Send trajectory data to projector if balls are moving
        moving_balls = [ball for ball in state.balls if ball.is_moving()]
        if moving_balls:
            trajectory_data = {
                "moving_balls": [self._serialize_ball(ball) for ball in moving_balls],
                "timestamp": state.timestamp,
            }
            self.event_manager.provide_trajectory_to_projector(trajectory_data)

    def _serialize_ball(self, ball: BallState) -> dict[str, Any]:
        """Serialize ball state for module communication."""
        return {
            "id": ball.id,
            "position": {"x": ball.position.x, "y": ball.position.y},
            "velocity": {"x": ball.velocity.x, "y": ball.velocity.y},
            "radius": ball.radius,
            "mass": ball.mass,
            "spin": {"x": ball.spin.x, "y": ball.spin.y} if ball.spin else None,
            "is_cue_ball": ball.is_cue_ball,
            "is_pocketed": ball.is_pocketed,
            "number": ball.number,
            "confidence": ball.confidence,
        }

    def _apply_core_configuration(self, config: dict[str, Any]):
        """Apply configuration changes to core module."""
        if "validation_enabled" in config:
            self.game_state_manager.set_validation_config(
                enabled=config["validation_enabled"],
                auto_correct=config.get("auto_correct_enabled", True),
            )

        if "event_history_size" in config:
            # Update event manager history size
            self.event_manager.event_history.maxlen = config["event_history_size"]

        if "logging_level" in config:
            logging.getLogger("backend.core").setLevel(
                getattr(logging, config["logging_level"].upper())
            )

        self.logger.info("Core configuration updated")

    def _forward_config_change(self, module_name: str, config: dict[str, Any]):
        """Forward configuration changes to specific modules."""
        if module_name == "vision" and self.vision_interface:
            try:
                detection_params = config.get("detection_parameters", {})
                if detection_params:
                    self.vision_interface.set_detection_parameters(detection_params)
            except Exception as e:
                self.logger.error(f"Error forwarding config to Vision module: {e}")

        elif module_name == "projector" and self.projector_interface:
            try:
                projection_settings = config.get("projection_settings", {})
                if projection_settings:
                    self.projector_interface.update_projection_settings(
                        projection_settings
                    )
            except Exception as e:
                self.logger.error(f"Error forwarding config to Projector module: {e}")

    def _send_welcome_configuration(self, module_name: str):
        """Send initial configuration to newly connected module."""
        welcome_configs = {
            "vision": {
                "detection_frequency": 60,
                "confidence_threshold": 0.7,
                "tracking_enabled": True,
                "detection_parameters": {
                    "ball_radius_range": [0.025, 0.032],  # meters
                    "color_calibration": True,
                    "motion_detection": True,
                },
            },
            "api": {
                "update_frequency": 60,
                "enable_websocket": True,
                "cors_enabled": True,
                "state_compression": False,
            },
            "projector": {
                "projection_mode": "overlay",
                "brightness": 0.8,
                "show_trajectories": True,
                "show_assistance": True,
                "projection_settings": {
                    "resolution": [1920, 1080],
                    "refresh_rate": 60,
                    "calibration_required": True,
                },
            },
        }

        config = welcome_configs.get(module_name, {})
        if config:
            self.event_manager.exchange_config_with_module(module_name, config)

    def _cleanup_module_resources(self, module_name: str):
        """Clean up resources associated with disconnected module."""
        # Clear module-specific interface
        if module_name == "vision":
            self.vision_interface = None
        elif module_name == "api":
            self.api_interface = None
        elif module_name == "projector":
            self.projector_interface = None
        elif module_name == "config":
            self.config_interface = None

    # Public API methods for external module integration

    def send_trajectory_update(self, trajectories: list[Trajectory]):
        """Send trajectory update to Projector module."""
        trajectory_data = {
            "trajectories": [traj.to_dict() for traj in trajectories],
            "timestamp": time.time(),
        }

        self.event_manager.provide_trajectory_to_projector(trajectory_data)

    def send_shot_analysis(self, shot_analysis: ShotAnalysis):
        """Send shot analysis to API and Projector modules."""
        analysis_data = shot_analysis.to_dict()

        # Send to API for client access
        if self.api_interface:
            try:
                self.api_interface.send_event_notification(
                    {
                        "event_type": "shot_analysis",
                        "data": analysis_data,
                        "timestamp": time.time(),
                    }
                )
            except Exception as e:
                self.logger.error(f"Error sending shot analysis to API: {e}")

        # Send to Projector for visualization
        if self.projector_interface:
            try:
                overlay_data = {
                    "type": "shot_analysis",
                    "shot_type": analysis_data["shot_type"],
                    "recommended_angle": analysis_data["recommended_angle"],
                    "success_probability": analysis_data["success_probability"],
                    "difficulty": analysis_data["difficulty"],
                }
                self.projector_interface.send_overlay_data(overlay_data)
            except Exception as e:
                self.logger.error(f"Error sending shot analysis to Projector: {e}")

    def request_vision_calibration(
        self, calibration_type: str = "full"
    ) -> Optional[dict[str, Any]]:
        """Request calibration from Vision module."""
        if self.vision_interface:
            try:
                return self.vision_interface.request_calibration(calibration_type)
            except Exception as e:
                self.logger.error(f"Error requesting vision calibration: {e}")
        return None

    def get_module_status(self, module_name: str) -> Optional[dict[str, Any]]:
        """Get status of a specific module."""
        return self.event_manager.coordinator.get_module_status(module_name)

    def get_all_module_statuses(self) -> dict[str, dict[str, Any]]:
        """Get status of all connected modules."""
        return self.event_manager.coordinator.get_all_module_statuses()

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get integration statistics."""
        return {
            **self.integration_stats,
            "connected_modules": list(self.connected_modules.keys()),
            "module_count": len(self.connected_modules),
            "event_manager_stats": self.event_manager.get_statistics(),
        }

    def shutdown_integration(self):
        """Shutdown module integration."""
        # Notify all modules of shutdown
        for module_name in self.connected_modules:
            self.event_manager.coordinate_module_lifecycle("shutdown", module_name)

        # Clear interfaces
        self.vision_interface = None
        self.api_interface = None
        self.projector_interface = None
        self.config_interface = None

        self.logger.info("Module integration shutdown complete")


# Concrete implementation examples for reference


class VisionInterfaceImpl(VisionInterface):
    """Production implementation of Vision interface."""

    def __init__(self, event_manager: EventManager, vision_module: Any = None):
        """Initialize vision interface with event manager and optional vision module instance."""
        self.event_manager = event_manager
        self.vision_module = vision_module
        self._lock = threading.RLock()  # Thread safety
        self.calibration_data = {
            "table": {
                "corners": [],
                "pocket_positions": [],
                "calibration_matrix": None,
                "timestamp": 0.0,
            },
            "color": {
                "ball_colors": {},
                "table_color": None,
                "lighting_conditions": {},
                "timestamp": 0.0,
            },
            "detection": {
                "confidence_thresholds": {},
                "detection_regions": [],
                "tracking_parameters": {},
                "timestamp": 0.0,
            },
        }
        self.detection_parameters = {
            "ball_radius_range": [0.025, 0.032],
            "confidence_threshold": 0.7,
            "tracking_enabled": True,
            "color_calibration": True,
            "motion_detection": True,
            "detection_frequency": 60,
            "background_subtraction": True,
            "noise_reduction": True,
        }
        self.logger = logging.getLogger(__name__ + ".VisionInterface")

    def receive_detection_data(self, detection_data: dict[str, Any]) -> None:
        """Receive detection data from Vision module."""
        with self._lock:
            try:
                # Validate detection data structure
                required_fields = ["timestamp", "frame_number", "balls"]
                for field in required_fields:
                    if field not in detection_data:
                        self.logger.warning(
                            f"Missing required field in detection data: {field}"
                        )
                        return

                # Process ball detection data
                processed_balls = []
                for ball_data in detection_data.get("balls", []):
                    if self._validate_ball_detection(ball_data):
                        processed_balls.append(self._process_ball_detection(ball_data))

                # Add processed balls to detection data
                processed_data = {
                    **detection_data,
                    "processed_balls": processed_balls,
                    "detection_quality": self._assess_detection_quality(
                        processed_balls
                    ),
                }

                # Forward to event manager
                self.event_manager.receive_vision_data(processed_data)
                self.logger.debug(
                    f"Processed vision data with {len(processed_balls)} balls"
                )

            except Exception as e:
                self.logger.error(f"Error processing vision detection data: {e}")

    def request_calibration(self, calibration_type: str) -> dict[str, Any]:
        """Request calibration data from Vision module."""
        with self._lock:
            try:
                # If vision module is available, get live calibration data
                if self.vision_module and hasattr(
                    self.vision_module, "get_calibration_data"
                ):
                    try:
                        live_calibration = self.vision_module.get_calibration_data(
                            calibration_type
                        )
                        if live_calibration:
                            self.logger.info(
                                f"Retrieved live {calibration_type} calibration from vision module"
                            )
                            return live_calibration
                    except Exception as vision_error:
                        self.logger.warning(
                            f"Failed to get live calibration from vision module: {vision_error}"
                        )
                        # Fall through to cached data

                # Fall back to cached calibration data
                if calibration_type not in self.calibration_data:
                    self.logger.warning(f"Unknown calibration type: {calibration_type}")
                    return {}

                calibration = self.calibration_data[calibration_type].copy()

                # Check if calibration is stale (older than 1 hour)
                current_time = time.time()
                if current_time - calibration.get("timestamp", 0) > 3600:
                    self.logger.warning(
                        f"Calibration data for {calibration_type} is stale"
                    )
                    calibration["is_stale"] = True

                self.logger.info(
                    f"Providing cached {calibration_type} calibration data"
                )
                return calibration

            except Exception as e:
                self.logger.error(f"Error retrieving calibration data: {e}")
                return {}

    def set_detection_parameters(self, parameters: dict[str, Any]) -> bool:
        """Set detection parameters in Vision module."""
        with self._lock:
            try:
                # Validate parameters
                valid_params = {
                    "ball_radius_range",
                    "confidence_threshold",
                    "tracking_enabled",
                    "color_calibration",
                    "motion_detection",
                    "detection_frequency",
                    "background_subtraction",
                    "noise_reduction",
                }

                validated_params = {}
                for key, value in parameters.items():
                    if key in valid_params:
                        if self._validate_parameter(key, value):
                            validated_params[key] = value
                        else:
                            self.logger.warning(
                                f"Invalid value for parameter {key}: {value}"
                            )
                            return False
                    else:
                        self.logger.warning(f"Unknown parameter: {key}")

                # Update parameters
                self.detection_parameters.update(validated_params)

                # If vision module is available, apply parameters directly
                if self.vision_module and hasattr(
                    self.vision_module, "update_detection_parameters"
                ):
                    try:
                        self.vision_module.update_detection_parameters(validated_params)
                        self.logger.info(
                            "Applied detection parameters directly to vision module"
                        )
                    except Exception as vision_error:
                        self.logger.warning(
                            f"Failed to apply parameters to vision module: {vision_error}"
                        )

                # Also notify via event system for other listeners
                self.event_manager.send_config_update(
                    "vision", {"detection_parameters": validated_params}
                )

                self.logger.info(
                    f"Updated detection parameters: {list(validated_params.keys())}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error setting detection parameters: {e}")
                return False

    def _validate_ball_detection(self, ball_data: dict[str, Any]) -> bool:
        """Validate ball detection data."""
        required_fields = ["id", "position", "confidence"]
        return all(field in ball_data for field in required_fields)

    def _process_ball_detection(self, ball_data: dict[str, Any]) -> dict[str, Any]:
        """Process and enhance ball detection data."""
        processed = ball_data.copy()

        # Add velocity estimation if tracking is enabled
        if self.detection_parameters.get("tracking_enabled", True):
            processed["velocity"] = self._estimate_velocity(ball_data)

        # Apply confidence filtering
        confidence = ball_data.get("confidence", 0.0)
        threshold = self.detection_parameters.get("confidence_threshold", 0.7)
        processed["passes_confidence_filter"] = confidence >= threshold

        return processed

    def _estimate_velocity(self, ball_data: dict[str, Any]) -> dict[str, float]:
        """Estimate ball velocity from position tracking."""
        # Simplified velocity estimation - real implementation would use tracking history
        return {"x": 0.0, "y": 0.0}

    def _assess_detection_quality(self, balls: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess overall detection quality."""
        if not balls:
            return {"score": 0.0, "issues": ["No balls detected"]}

        avg_confidence = sum(ball.get("confidence", 0.0) for ball in balls) / len(balls)
        issues = []

        if avg_confidence < 0.8:
            issues.append("Low average confidence")

        if len(balls) < 2:
            issues.append("Few balls detected")

        return {
            "score": avg_confidence,
            "ball_count": len(balls),
            "average_confidence": avg_confidence,
            "issues": issues,
        }

    def _validate_parameter(self, key: str, value: Any) -> bool:
        """Validate parameter values."""
        validators = {
            "confidence_threshold": lambda v: 0.0 <= v <= 1.0,
            "detection_frequency": lambda v: 1 <= v <= 120,
            "ball_radius_range": lambda v: isinstance(v, list)
            and len(v) == 2
            and v[0] < v[1],
            "tracking_enabled": lambda v: isinstance(v, bool),
            "color_calibration": lambda v: isinstance(v, bool),
            "motion_detection": lambda v: isinstance(v, bool),
            "background_subtraction": lambda v: isinstance(v, bool),
            "noise_reduction": lambda v: isinstance(v, bool),
        }

        validator = validators.get(key)
        return validator(value) if validator else True


class APIInterfaceImpl(APIInterface):
    """Production implementation of API interface."""

    def __init__(
        self,
        event_manager: EventManager,
        websocket_manager: Any = None,
        message_broadcaster: Any = None,
    ):
        """Initialize API interface with event manager and WebSocket components."""
        self.event_manager = event_manager
        self.websocket_manager = websocket_manager
        self.message_broadcaster = message_broadcaster
        self._lock = threading.RLock()  # Thread safety
        self.websocket_handlers = {}
        self.message_queue = []
        self.connection_status = {
            "connected": False,
            "last_heartbeat": 0.0,
            "client_count": 0,
            "message_count": 0,
        }
        self.rate_limiter = {
            "state_updates": {"count": 0, "window_start": time.time(), "limit": 60},
            "events": {"count": 0, "window_start": time.time(), "limit": 100},
        }
        self.logger = logging.getLogger(__name__ + ".APIInterface")

    def send_state_update(self, state_data: dict[str, Any]) -> None:
        """Send state updates to API module."""
        with self._lock:
            try:
                # Apply rate limiting
                if not self._check_rate_limit("state_updates"):
                    self.logger.warning("State update rate limit exceeded")
                    return

                # Validate state data
                if not self._validate_state_data(state_data):
                    self.logger.error("Invalid state data format")
                    return

                # Prepare message for API
                message = {
                    "type": "state_update",
                    "timestamp": time.time(),
                    "data": self._sanitize_state_data(state_data),
                    "sequence_number": self.connection_status["message_count"],
                }

                # Add to message queue for delivery
                self._queue_message(message)

                # Update statistics
                self.connection_status["message_count"] += 1
                self.logger.debug(
                    f"Queued state update (sequence: {message['sequence_number']})"
                )

                # Trigger immediate delivery if connected
                if self.connection_status["connected"]:
                    self._flush_message_queue()

            except Exception as e:
                self.logger.error(f"Error sending state update: {e}")

    def send_event_notification(self, event_data: dict[str, Any]) -> None:
        """Send event notifications to API module."""
        with self._lock:
            try:
                # Apply rate limiting
                if not self._check_rate_limit("events"):
                    self.logger.warning("Event notification rate limit exceeded")
                    return

                # Validate event data
                if not self._validate_event_data(event_data):
                    self.logger.error("Invalid event data format")
                    return

                # Prepare message for API
                message = {
                    "type": "event_notification",
                    "timestamp": time.time(),
                    "data": event_data,
                    "sequence_number": self.connection_status["message_count"],
                    "priority": self._determine_event_priority(event_data),
                }

                # Add to message queue
                self._queue_message(message)

                # Update statistics
                self.connection_status["message_count"] += 1
                self.logger.info(
                    f"Queued event notification: {event_data.get('event_type', 'unknown')}"
                )

                # Trigger immediate delivery if connected
                if self.connection_status["connected"]:
                    self._flush_message_queue()

            except Exception as e:
                self.logger.error(f"Error sending event notification: {e}")

    def register_websocket_handler(self, handler: Callable) -> str:
        """Register WebSocket message handler."""
        with self._lock:
            try:
                handler_id = (
                    f"handler_{int(time.time() * 1000)}_{len(self.websocket_handlers)}"
                )

                # Wrap handler with error handling
                wrapped_handler = self._wrap_handler(handler, handler_id)
                self.websocket_handlers[handler_id] = {
                    "handler": wrapped_handler,
                    "registered_at": time.time(),
                    "message_count": 0,
                    "last_used": time.time(),
                }

                self.logger.info(f"Registered WebSocket handler: {handler_id}")
                return handler_id

            except Exception as e:
                self.logger.error(f"Error registering WebSocket handler: {e}")
                return ""

    def unregister_websocket_handler(self, handler_id: str) -> bool:
        """Unregister WebSocket message handler."""
        with self._lock:
            if handler_id in self.websocket_handlers:
                del self.websocket_handlers[handler_id]
                self.logger.info(f"Unregistered WebSocket handler: {handler_id}")
                return True
            return False

    def get_connection_status(self) -> dict[str, Any]:
        """Get current connection status."""
        return {
            **self.connection_status,
            "queue_size": len(self.message_queue),
            "handler_count": len(self.websocket_handlers),
            "uptime": time.time()
            - self.connection_status.get("start_time", time.time()),
        }

    def _validate_state_data(self, state_data: dict[str, Any]) -> bool:
        """Validate state data structure."""
        required_fields = ["timestamp", "frame_number", "balls"]
        return all(field in state_data for field in required_fields)

    def _validate_event_data(self, event_data: dict[str, Any]) -> bool:
        """Validate event data structure."""
        required_fields = ["event_type", "timestamp"]
        return all(field in event_data for field in required_fields)

    def _sanitize_state_data(self, state_data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize state data for API transmission."""
        # Remove internal fields that shouldn't be exposed
        sanitized = state_data.copy()

        # Remove validation errors from public API
        sanitized.pop("validation_errors", None)

        # Limit precision of floating point numbers
        if "balls" in sanitized:
            for ball in sanitized["balls"]:
                if "position" in ball:
                    ball["position"]["x"] = round(ball["position"]["x"], 4)
                    ball["position"]["y"] = round(ball["position"]["y"], 4)
                if "velocity" in ball:
                    ball["velocity"]["x"] = round(ball["velocity"]["x"], 4)
                    ball["velocity"]["y"] = round(ball["velocity"]["y"], 4)

        return sanitized

    def _determine_event_priority(self, event_data: dict[str, Any]) -> str:
        """Determine priority level for event."""
        event_type = event_data.get("event_type", "")

        high_priority_events = {"collision", "pocket", "shot_completed", "error"}
        medium_priority_events = {"shot_started", "ball_stopped", "state_changed"}

        if event_type in high_priority_events:
            return "high"
        elif event_type in medium_priority_events:
            return "medium"
        else:
            return "low"

    def _check_rate_limit(self, category: str) -> bool:
        """Check if rate limit allows this operation."""
        current_time = time.time()
        rate_info = self.rate_limiter[category]

        # Reset window if needed
        if current_time - rate_info["window_start"] > 60:  # 1 minute window
            rate_info["count"] = 0
            rate_info["window_start"] = current_time

        # Check limit
        if rate_info["count"] >= rate_info["limit"]:
            return False

        rate_info["count"] += 1
        return True

    def _queue_message(self, message: dict[str, Any]) -> None:
        """Add message to delivery queue."""
        # Prefer message_broadcaster for async WebSocket communication
        if self.message_broadcaster:
            try:
                import asyncio

                message_type = message.get("type", "")
                data = message.get("data", {})

                # Schedule async broadcast on event loop
                if message_type == "state_update":
                    balls = data.get("balls", [])
                    cue = data.get("cue")
                    table = data.get("table")
                    # Create async task to broadcast
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(
                            self.message_broadcaster.broadcast_game_state(
                                balls=balls, cue=cue, table=table
                            )
                        )
                        self.logger.debug(
                            "Scheduled game state broadcast via message_broadcaster"
                        )
                        return
                    except RuntimeError:
                        # No event loop, fall through to queueing
                        pass
                elif message_type == "event_notification":
                    # Broadcast as alert
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(
                            self.message_broadcaster.broadcast_alert(
                                level=data.get("level", "info"),
                                message=data.get("message", ""),
                                source=data.get("source", "system"),
                            )
                        )
                        self.logger.debug(
                            "Scheduled alert broadcast via message_broadcaster"
                        )
                        return
                    except RuntimeError:
                        pass
            except Exception as bc_error:
                self.logger.warning(
                    f"Failed to send via message_broadcaster: {bc_error}"
                )
                # Fall through to queueing

        # Fallback: try websocket_manager if available
        if self.websocket_manager:
            try:
                message_type = message.get("type", "")
                if message_type == "state_update" and hasattr(
                    self.websocket_manager, "broadcast_game_state"
                ):
                    self.websocket_manager.broadcast_game_state(message.get("data", {}))
                elif message_type == "event_notification" and hasattr(
                    self.websocket_manager, "broadcast_alert"
                ):
                    self.websocket_manager.broadcast_alert(message.get("data", {}))
                elif hasattr(self.websocket_manager, "broadcast"):
                    self.websocket_manager.broadcast(message)
                self.logger.debug(
                    f"Sent message directly to websocket_manager: {message_type}"
                )
                return
            except Exception as ws_error:
                self.logger.warning(f"Failed to send via websocket_manager: {ws_error}")
                # Fall through to queueing

        # Limit queue size to prevent memory issues
        max_queue_size = 1000
        if len(self.message_queue) >= max_queue_size:
            # Remove oldest messages
            self.message_queue = self.message_queue[-(max_queue_size // 2) :]
            self.logger.warning("Message queue full, discarded old messages")

        self.message_queue.append(message)

    def _flush_message_queue(self) -> None:
        """Deliver all queued messages to API."""
        if not self.message_queue:
            return

        try:
            # In a real implementation, this would send to actual API endpoints
            # For now, we'll just notify the event manager
            for message in self.message_queue:
                self.event_manager.send_api_message(message)

            delivered_count = len(self.message_queue)
            self.message_queue.clear()
            self.logger.debug(f"Delivered {delivered_count} queued messages")

        except Exception as e:
            self.logger.error(f"Error flushing message queue: {e}")

    def _wrap_handler(self, handler: Callable, handler_id: str) -> Callable:
        """Wrap handler with error handling and statistics."""

        def wrapped_handler(*args, **kwargs):
            try:
                # Update usage statistics
                self.websocket_handlers[handler_id]["message_count"] += 1
                self.websocket_handlers[handler_id]["last_used"] = time.time()

                # Call actual handler
                return handler(*args, **kwargs)

            except Exception as e:
                self.logger.error(f"Error in WebSocket handler {handler_id}: {e}")
                return None

        return wrapped_handler


class ProjectorInterfaceImpl(ProjectorInterface):
    """Production implementation of Projector interface."""

    def __init__(self, event_manager: EventManager, projector_module: Any = None):
        """Initialize projector interface with event manager and optional projector module instance."""
        self.event_manager = event_manager
        self.projector_module = projector_module
        self._lock = threading.RLock()  # Thread safety
        self.projection_settings = {
            "brightness": 0.8,
            "contrast": 1.0,
            "resolution": [1920, 1080],
            "refresh_rate": 60,
            "projection_mode": "overlay",
            "show_trajectories": True,
            "show_assistance": True,
            "show_aim_lines": True,
            "show_ball_ids": False,
            "calibration_points": [],
            "keystone_correction": {"x": 0, "y": 0},
        }
        self.active_overlays = {}
        self.trajectory_cache = {}
        self.calibration_status = {
            "is_calibrated": False,
            "last_calibration": 0.0,
            "calibration_error": 0.0,
        }
        self.performance_stats = {
            "frame_rate": 0.0,
            "latency_ms": 0.0,
            "dropped_frames": 0,
            "last_update": time.time(),
        }
        self.logger = logging.getLogger(__name__ + ".ProjectorInterface")

    def send_trajectory_data(self, trajectory_data: dict[str, Any]) -> None:
        """Send trajectory data to Projector module."""
        with self._lock:
            try:
                # Validate trajectory data
                if not self._validate_trajectory_data(trajectory_data):
                    self.logger.error("Invalid trajectory data format")
                    return

                # Process trajectory for visualization
                processed_trajectories = self._process_trajectories(trajectory_data)

                # Cache trajectories for performance
                self._cache_trajectories(processed_trajectories)

                # Create projection overlay
                overlay = self._create_trajectory_overlay(processed_trajectories)

                # Send to projector hardware/software
                projection_message = {
                    "type": "trajectory_update",
                    "timestamp": time.time(),
                    "overlay": overlay,
                    "settings": self.projection_settings,
                    "duration_ms": trajectory_data.get("duration_ms", 5000),
                }

                self._send_to_projector(projection_message)
                self.logger.debug(
                    f"Sent trajectory data for {len(processed_trajectories)} balls"
                )

            except Exception as e:
                self.logger.error(f"Error sending trajectory data: {e}")

    def send_overlay_data(self, overlay_data: dict[str, Any]) -> None:
        """Send overlay visualization data."""
        with self._lock:
            try:
                # Validate overlay data
                if not self._validate_overlay_data(overlay_data):
                    self.logger.error("Invalid overlay data format")
                    return

                overlay_type = overlay_data.get("type", "unknown")

                # Process different overlay types
                if overlay_type == "shot_analysis":
                    overlay = self._create_shot_analysis_overlay(overlay_data)
                elif overlay_type == "aim_assistance":
                    overlay = self._create_aim_assistance_overlay(overlay_data)
                elif overlay_type == "ball_tracking":
                    overlay = self._create_ball_tracking_overlay(overlay_data)
                elif overlay_type == "table_calibration":
                    overlay = self._create_calibration_overlay(overlay_data)
                else:
                    overlay = self._create_generic_overlay(overlay_data)

                # Store overlay for composite rendering
                self.active_overlays[overlay_type] = {
                    "overlay": overlay,
                    "timestamp": time.time(),
                    "priority": overlay_data.get("priority", 1),
                    "duration": overlay_data.get("duration_ms", 3000),
                }

                # Composite all active overlays
                composite_overlay = self._composite_overlays()

                # Send to projector
                projection_message = {
                    "type": "overlay_update",
                    "timestamp": time.time(),
                    "overlay": composite_overlay,
                    "settings": self.projection_settings,
                }

                self._send_to_projector(projection_message)
                self.logger.debug(f"Sent {overlay_type} overlay data")

            except Exception as e:
                self.logger.error(f"Error sending overlay data: {e}")

    def update_projection_settings(self, settings: dict[str, Any]) -> None:
        """Update projector settings."""
        with self._lock:
            try:
                # Validate settings
                valid_settings = self._validate_settings(settings)
                if not valid_settings:
                    self.logger.error("No valid settings provided")
                    return

                # Apply settings with validation
                old_settings = self.projection_settings.copy()
                self.projection_settings.update(valid_settings)

                # Check if calibration is needed
                if self._calibration_required(old_settings, valid_settings):
                    self.logger.info("Settings change requires recalibration")
                    self.calibration_status["is_calibrated"] = False

                # Send settings update to projector
                settings_message = {
                    "type": "settings_update",
                    "timestamp": time.time(),
                    "settings": self.projection_settings,
                    "requires_restart": self._requires_projector_restart(
                        valid_settings
                    ),
                }

                self._send_to_projector(settings_message)
                self.logger.info(
                    f"Updated projector settings: {list(valid_settings.keys())}"
                )

            except Exception as e:
                self.logger.error(f"Error updating projection settings: {e}")

    def get_projection_status(self) -> dict[str, Any]:
        """Get current projector status."""
        with self._lock:
            return {
                "settings": self.projection_settings.copy(),
                "calibration": self.calibration_status.copy(),
                "performance": self.performance_stats.copy(),
                "active_overlays": list(self.active_overlays.keys()),
                "cache_size": len(self.trajectory_cache),
            }

    def calibrate_projector(self, calibration_points: list[dict[str, Any]]) -> bool:
        """Perform projector calibration."""
        with self._lock:
            try:
                if len(calibration_points) < 4:
                    self.logger.error("Need at least 4 calibration points")
                    return False

                # Process calibration points
                processed_points = self._process_calibration_points(calibration_points)

                # Calculate transformation matrix
                transformation_matrix = self._calculate_transformation_matrix(
                    processed_points
                )

                if transformation_matrix is None:
                    self.logger.error("Failed to calculate transformation matrix")
                    return False

                # Update calibration
                self.projection_settings["calibration_points"] = processed_points
                self.projection_settings["transformation_matrix"] = (
                    transformation_matrix
                )

                self.calibration_status = {
                    "is_calibrated": True,
                    "last_calibration": time.time(),
                    "calibration_error": self._calculate_calibration_error(
                        processed_points
                    ),
                }

                self.logger.info("Projector calibration completed successfully")
                return True

            except Exception as e:
                self.logger.error(f"Error during projector calibration: {e}")
                return False

    def _validate_trajectory_data(self, data: dict[str, Any]) -> bool:
        """Validate trajectory data structure."""
        required_fields = ["trajectories", "timestamp"]
        return all(field in data for field in required_fields)

    def _validate_overlay_data(self, data: dict[str, Any]) -> bool:
        """Validate overlay data structure."""
        return "type" in data and "timestamp" in data

    def _validate_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Validate and filter projector settings."""
        valid_settings = {}

        validators = {
            "brightness": lambda v: 0.0 <= v <= 1.0,
            "contrast": lambda v: 0.1 <= v <= 2.0,
            "resolution": lambda v: isinstance(v, list) and len(v) == 2,
            "refresh_rate": lambda v: v in [30, 60, 120],
            "projection_mode": lambda v: v in ["overlay", "augmented", "full"],
            "show_trajectories": lambda v: isinstance(v, bool),
            "show_assistance": lambda v: isinstance(v, bool),
            "show_aim_lines": lambda v: isinstance(v, bool),
            "show_ball_ids": lambda v: isinstance(v, bool),
        }

        for key, value in settings.items():
            if key in validators:
                if validators[key](value):
                    valid_settings[key] = value
                else:
                    self.logger.warning(f"Invalid value for setting {key}: {value}")

        return valid_settings

    def _process_trajectories(
        self, trajectory_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process trajectory data for visualization."""
        trajectories = trajectory_data.get("trajectories", [])
        processed = []

        for trajectory in trajectories:
            # Convert to screen coordinates
            screen_points = self._convert_to_screen_coordinates(
                trajectory.get("points", [])
            )

            # Apply visual styling
            processed_trajectory = {
                "ball_id": trajectory.get("ball_id"),
                "points": screen_points,
                "color": self._get_trajectory_color(trajectory),
                "line_width": self._get_trajectory_width(trajectory),
                "style": self._get_trajectory_style(trajectory),
                "confidence": trajectory.get("confidence", 1.0),
            }
            processed.append(processed_trajectory)

        return processed

    def _create_trajectory_overlay(
        self, trajectories: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Create trajectory visualization overlay."""
        if not self.projection_settings.get("show_trajectories", True):
            return {"type": "empty"}

        overlay_elements = []

        for trajectory in trajectories:
            if trajectory["confidence"] < 0.5:
                continue  # Skip low-confidence trajectories

            # Create trajectory line
            line_element = {
                "type": "polyline",
                "points": trajectory["points"],
                "color": trajectory["color"],
                "width": trajectory["line_width"],
                "style": trajectory["style"],
            }
            overlay_elements.append(line_element)

            # Add collision markers
            if "collisions" in trajectory:
                for collision in trajectory["collisions"]:
                    marker = {
                        "type": "marker",
                        "position": collision["position"],
                        "symbol": "collision",
                        "color": "yellow",
                        "size": 8,
                    }
                    overlay_elements.append(marker)

        return {
            "type": "trajectory_overlay",
            "elements": overlay_elements,
            "timestamp": time.time(),
        }

    def _create_shot_analysis_overlay(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create shot analysis visualization overlay."""
        elements = []

        # Recommended shot line
        if "recommended_angle" in data:
            angle_line = {
                "type": "line",
                "angle": data["recommended_angle"],
                "color": "green",
                "width": 3,
                "style": "dashed",
            }
            elements.append(angle_line)

        # Success probability indicator
        if "success_probability" in data:
            probability_text = {
                "type": "text",
                "text": f"{data['success_probability']:.1%}",
                "position": data.get("text_position", [50, 50]),
                "color": "white",
                "size": 24,
            }
            elements.append(probability_text)

        return {
            "type": "shot_analysis_overlay",
            "elements": elements,
            "timestamp": time.time(),
        }

    def _create_aim_assistance_overlay(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create aim assistance visualization overlay."""
        if not self.projection_settings.get("show_assistance", True):
            return {"type": "empty"}

        elements = []

        # Aim line
        if "aim_line" in data and self.projection_settings.get("show_aim_lines", True):
            aim_line = {
                "type": "line",
                "start": data["aim_line"]["start"],
                "end": data["aim_line"]["end"],
                "color": "cyan",
                "width": 2,
                "style": "solid",
            }
            elements.append(aim_line)

        return {
            "type": "aim_assistance_overlay",
            "elements": elements,
            "timestamp": time.time(),
        }

    def _create_ball_tracking_overlay(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create ball tracking visualization overlay."""
        elements = []

        for ball in data.get("balls", []):
            # Ball ID overlay
            if self.projection_settings.get("show_ball_ids", False):
                id_text = {
                    "type": "text",
                    "text": str(ball.get("number", "?")),
                    "position": ball["position"],
                    "color": "white",
                    "size": 16,
                }
                elements.append(id_text)

            # Velocity indicator
            if (
                ball.get("velocity")
                and ball["velocity"]["x"] != 0
                or ball["velocity"]["y"] != 0
            ):
                velocity_arrow = {
                    "type": "arrow",
                    "start": ball["position"],
                    "velocity": ball["velocity"],
                    "color": "red",
                    "scale": 100,
                }
                elements.append(velocity_arrow)

        return {
            "type": "ball_tracking_overlay",
            "elements": elements,
            "timestamp": time.time(),
        }

    def _create_calibration_overlay(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create calibration visualization overlay."""
        elements = []

        for point in data.get("calibration_points", []):
            marker = {
                "type": "marker",
                "position": point["position"],
                "symbol": "crosshair",
                "color": "white",
                "size": 20,
            }
            elements.append(marker)

        return {
            "type": "calibration_overlay",
            "elements": elements,
            "timestamp": time.time(),
        }

    def _create_generic_overlay(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create generic overlay from data."""
        return {"type": "generic_overlay", "data": data, "timestamp": time.time()}

    def _composite_overlays(self) -> dict[str, Any]:
        """Composite all active overlays into single overlay."""
        # Note: _lock is already held when this is called
        current_time = time.time()
        composite_elements = []

        # Remove expired overlays
        expired_overlays = []
        for overlay_type, overlay_info in self.active_overlays.items():
            age_ms = (current_time - overlay_info["timestamp"]) * 1000
            if age_ms > overlay_info["duration"]:
                expired_overlays.append(overlay_type)

        for overlay_type in expired_overlays:
            del self.active_overlays[overlay_type]

        # Sort overlays by priority
        sorted_overlays = sorted(
            self.active_overlays.items(), key=lambda x: x[1]["priority"], reverse=True
        )

        # Composite elements
        for overlay_type, overlay_info in sorted_overlays:
            overlay = overlay_info["overlay"]
            if overlay.get("type") != "empty":
                composite_elements.extend(overlay.get("elements", []))

        return {
            "type": "composite_overlay",
            "elements": composite_elements,
            "timestamp": current_time,
        }

    def _cache_trajectories(self, trajectories: list[dict[str, Any]]) -> None:
        """Cache trajectories for performance optimization."""
        # Note: _lock is already held when this is called
        cache_key = str(hash(str(trajectories)))
        self.trajectory_cache[cache_key] = {
            "trajectories": trajectories,
            "timestamp": time.time(),
        }

        # Limit cache size
        if len(self.trajectory_cache) > 100:
            oldest_key = min(
                self.trajectory_cache.keys(),
                key=lambda k: self.trajectory_cache[k]["timestamp"],
            )
            del self.trajectory_cache[oldest_key]

    def _send_to_projector(self, message: dict[str, Any]) -> None:
        """Send message to actual projector hardware/software."""
        # If projector module is available, call it directly
        if self.projector_module:
            try:
                message_type = message.get("type", "")
                if message_type == "trajectory_update" and hasattr(
                    self.projector_module, "render_trajectory"
                ):
                    self.projector_module.render_trajectory(message.get("overlay", {}))
                elif message_type == "overlay_update" and hasattr(
                    self.projector_module, "render_overlay"
                ):
                    self.projector_module.render_overlay(message.get("overlay", {}))
                elif hasattr(self.projector_module, "send_message"):
                    self.projector_module.send_message(message)
                else:
                    self.logger.debug(
                        f"Projector module doesn't have handler for: {message_type}"
                    )
            except Exception as projector_error:
                self.logger.warning(
                    f"Failed to send to projector module: {projector_error}"
                )

        # Also notify via event system for other listeners
        self.logger.debug(f"Sending to projector: {message['type']}")
        self.event_manager.send_projector_command(message)

    def _convert_to_screen_coordinates(
        self, world_points: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """Convert world coordinates to screen coordinates."""
        # This would use the calibration transformation matrix
        # For now, return as-is (simplified)
        return world_points

    def _get_trajectory_color(self, trajectory: dict[str, Any]) -> str:
        """Get color for trajectory visualization."""
        confidence = trajectory.get("confidence", 1.0)
        if confidence > 0.8:
            return "lime"
        elif confidence > 0.5:
            return "yellow"
        else:
            return "orange"

    def _get_trajectory_width(self, trajectory: dict[str, Any]) -> int:
        """Get line width for trajectory."""
        confidence = trajectory.get("confidence", 1.0)
        return max(1, int(confidence * 4))

    def _get_trajectory_style(self, trajectory: dict[str, Any]) -> str:
        """Get line style for trajectory."""
        confidence = trajectory.get("confidence", 1.0)
        return "solid" if confidence > 0.7 else "dashed"

    def _calibration_required(self, old_settings: dict, new_settings: dict) -> bool:
        """Check if settings change requires recalibration."""
        calibration_sensitive = {"resolution", "keystone_correction", "projection_mode"}
        return any(key in calibration_sensitive for key in new_settings)

    def _requires_projector_restart(self, settings: dict[str, Any]) -> bool:
        """Check if settings require projector restart."""
        restart_required = {"resolution", "refresh_rate"}
        return any(key in restart_required for key in settings)

    def _process_calibration_points(
        self, points: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process calibration points."""
        return points  # Simplified for now

    def _calculate_transformation_matrix(
        self, points: list[dict[str, Any]]
    ) -> Optional[list[list[float]]]:
        """Calculate transformation matrix from calibration points."""
        # This would implement proper perspective transformation calculation
        # For now, return identity matrix
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def _calculate_calibration_error(self, points: list[dict[str, Any]]) -> float:
        """Calculate calibration error."""
        # This would calculate RMS error or similar metric
        return 0.0  # Simplified for now


class ConfigInterfaceImpl(ConfigInterface):
    """Production implementation of Config interface."""

    def __init__(self, event_manager: EventManager, config_module: Any = None):
        """Initialize config interface with event manager and optional config module instance."""
        self.event_manager = event_manager
        self.config_module = config_module
        self._lock = threading.RLock()  # Thread safety
        self.module_configs = {
            "core": {
                "validation_enabled": True,
                "auto_correct_enabled": True,
                "event_history_size": 1000,
                "logging_level": "INFO",
                "physics_enabled": True,
                "spin_enabled": True,
                "collision_detection": True,
            },
            "vision": {
                "detection_frequency": 60,
                "confidence_threshold": 0.7,
                "tracking_enabled": True,
                "detection_parameters": {
                    "ball_radius_range": [0.025, 0.032],
                    "color_calibration": True,
                    "motion_detection": True,
                    "background_subtraction": True,
                    "noise_reduction": True,
                },
                "camera_settings": {
                    "resolution": [1920, 1080],
                    "fps": 60,
                    "exposure": "auto",
                    "gain": "auto",
                },
            },
            "api": {
                "update_frequency": 60,
                "enable_websocket": True,
                "cors_enabled": True,
                "state_compression": False,
                "rate_limits": {"state_updates": 60, "events": 100},
                "auth_enabled": False,
                "port": 8000,
            },
            "projector": {
                "projection_mode": "overlay",
                "brightness": 0.8,
                "contrast": 1.0,
                "show_trajectories": True,
                "show_assistance": True,
                "projection_settings": {
                    "resolution": [1920, 1080],
                    "refresh_rate": 60,
                    "calibration_required": True,
                },
                "display_settings": {
                    "show_ball_ids": False,
                    "show_aim_lines": True,
                    "trajectory_duration": 5000,
                },
            },
        }
        self.config_subscribers = {}
        self.config_history = {}
        self.validation_schemas = self._initialize_validation_schemas()
        self.logger = logging.getLogger(__name__ + ".ConfigInterface")

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        with self._lock:
            try:
                # If config module is available, get live configuration
                if self.config_module and hasattr(
                    self.config_module, "get_module_config"
                ):
                    try:
                        live_config = self.config_module.get_module_config(module_name)
                        if live_config:
                            self.logger.debug(
                                f"Retrieved live configuration for module: {module_name}"
                            )
                            return live_config
                    except Exception as config_error:
                        self.logger.warning(
                            f"Failed to get live config from config module: {config_error}"
                        )
                        # Fall through to cached config

                # Fall back to cached configuration
                config = self.module_configs.get(module_name, {})
                if not config:
                    self.logger.warning(
                        f"No configuration found for module: {module_name}"
                    )
                    return {}

                # Add metadata
                enhanced_config = {
                    **config,
                    "_metadata": {
                        "module_name": module_name,
                        "last_updated": self.config_history.get(module_name, {}).get(
                            "last_updated", 0.0
                        ),
                        "version": self.config_history.get(module_name, {}).get(
                            "version", "1.0.0"
                        ),
                        "source": "cached_config",
                    },
                }

                self.logger.debug(
                    f"Retrieved cached configuration for module: {module_name}"
                )
                return enhanced_config

            except Exception as e:
                self.logger.error(f"Error retrieving config for {module_name}: {e}")
                return {}

    def update_module_config(self, module_name: str, config: dict[str, Any]) -> None:
        """Update configuration for a specific module."""
        with self._lock:
            try:
                # Validate configuration
                if not self._validate_config(module_name, config):
                    self.logger.error(
                        f"Configuration validation failed for {module_name}"
                    )
                    return

                # Backup current config
                old_config = self.module_configs.get(module_name, {}).copy()

                # Apply updates to cache
                if module_name not in self.module_configs:
                    self.module_configs[module_name] = {}

                self.module_configs[module_name].update(config)

                # If config module is available, persist configuration
                if self.config_module and hasattr(
                    self.config_module, "update_module_config"
                ):
                    try:
                        self.config_module.update_module_config(module_name, config)
                        self.logger.info(
                            f"Persisted configuration to config module for: {module_name}"
                        )
                    except Exception as config_error:
                        self.logger.warning(
                            f"Failed to persist config to config module: {config_error}"
                        )

                # Update history
                current_time = time.time()
                self.config_history[module_name] = {
                    "last_updated": current_time,
                    "version": self._increment_version(module_name),
                    "old_config": old_config,
                    "changes": self._calculate_changes(old_config, config),
                }

                # Notify subscribers
                self._notify_config_subscribers(module_name, config, old_config)

                # Notify event manager
                self.event_manager.exchange_config_with_module(
                    module_name, self.module_configs[module_name]
                )

                self.logger.info(f"Updated configuration for module: {module_name}")

            except Exception as e:
                self.logger.error(f"Error updating config for {module_name}: {e}")

    def subscribe_config_changes(self, callback: Callable) -> str:
        """Subscribe to configuration changes."""
        with self._lock:
            try:
                subscription_id = f"config_sub_{int(time.time() * 1000)}_{len(self.config_subscribers)}"

                # Wrap callback with error handling
                wrapped_callback = self._wrap_config_callback(callback, subscription_id)

                self.config_subscribers[subscription_id] = {
                    "callback": wrapped_callback,
                    "subscribed_at": time.time(),
                    "notification_count": 0,
                    "last_notified": 0.0,
                }

                self.logger.info(f"Registered config subscription: {subscription_id}")
                return subscription_id

            except Exception as e:
                self.logger.error(f"Error registering config subscription: {e}")
                return ""

    def unsubscribe_config_changes(self, subscription_id: str) -> bool:
        """Unsubscribe from configuration changes."""
        with self._lock:
            if subscription_id in self.config_subscribers:
                del self.config_subscribers[subscription_id]
                self.logger.info(f"Unregistered config subscription: {subscription_id}")
                return True
            return False

    def get_config_history(self, module_name: str) -> dict[str, Any]:
        """Get configuration change history for a module."""
        with self._lock:
            return self.config_history.get(module_name, {})

    def validate_config(
        self, module_name: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate configuration without applying changes."""
        with self._lock:
            try:
                validation_result = {
                    "valid": False,
                    "errors": [],
                    "warnings": [],
                    "suggestions": [],
                }

                # Check if module is known
                if module_name not in self.validation_schemas:
                    validation_result["errors"].append(f"Unknown module: {module_name}")
                    return validation_result

                # Validate against schema
                schema = self.validation_schemas[module_name]
                errors, warnings, suggestions = self._validate_against_schema(
                    config, schema
                )

                validation_result.update(
                    {
                        "valid": len(errors) == 0,
                        "errors": errors,
                        "warnings": warnings,
                        "suggestions": suggestions,
                    }
                )

                return validation_result

            except Exception as e:
                self.logger.error(f"Error validating config for {module_name}: {e}")
                return {
                    "valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "suggestions": [],
                }

    def reset_module_config(self, module_name: str) -> bool:
        """Reset module configuration to defaults."""
        with self._lock:
            try:
                default_configs = self._get_default_configs()
                if module_name not in default_configs:
                    self.logger.error(f"No default config available for {module_name}")
                    return False

                self.update_module_config(module_name, default_configs[module_name])
                self.logger.info(f"Reset configuration for module: {module_name}")
                return True

            except Exception as e:
                self.logger.error(f"Error resetting config for {module_name}: {e}")
                return False

    def _validate_config(self, module_name: str, config: dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        validation_result = self.validate_config(module_name, config)
        return validation_result["valid"]

    def _validate_against_schema(
        self, config: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[list[str], list[str], list[str]]:
        """Validate configuration against schema."""
        errors = []
        warnings = []
        suggestions = []

        for key, value in config.items():
            if key in schema:
                schema_entry = schema[key]
                expected_type = schema_entry.get("type")

                # Type validation
                if expected_type and not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )

                # Range validation
                if "range" in schema_entry and isinstance(value, (int, float)):
                    min_val, max_val = schema_entry["range"]
                    if not (min_val <= value <= max_val):
                        errors.append(
                            f"Value {key}={value} out of range [{min_val}, {max_val}]"
                        )

                # Choices validation
                if "choices" in schema_entry and value not in schema_entry["choices"]:
                    errors.append(
                        f"Invalid choice for {key}: {value} not in {schema_entry['choices']}"
                    )

                # Custom validation
                if "validator" in schema_entry:
                    try:
                        if not schema_entry["validator"](value):
                            errors.append(f"Custom validation failed for {key}")
                    except Exception as e:
                        errors.append(f"Validation error for {key}: {e}")

            else:
                warnings.append(f"Unknown configuration key: {key}")

        # Check for missing required fields
        for key, schema_entry in schema.items():
            if schema_entry.get("required", False) and key not in config:
                errors.append(f"Missing required configuration: {key}")

        return errors, warnings, suggestions

    def _notify_config_subscribers(
        self, module_name: str, new_config: dict[str, Any], old_config: dict[str, Any]
    ) -> None:
        """Notify all subscribers of configuration changes."""
        # Note: _lock is already held when this is called
        notification = {
            "module_name": module_name,
            "new_config": new_config,
            "old_config": old_config,
            "changes": self._calculate_changes(old_config, new_config),
            "timestamp": time.time(),
        }

        for subscription_id, subscriber_info in self.config_subscribers.items():
            try:
                subscriber_info["callback"](notification)
                subscriber_info["notification_count"] += 1
                subscriber_info["last_notified"] = time.time()
            except Exception as e:
                self.logger.error(
                    f"Error notifying config subscriber {subscription_id}: {e}"
                )

    def _wrap_config_callback(
        self, callback: Callable, subscription_id: str
    ) -> Callable:
        """Wrap config callback with error handling."""

        def wrapped_callback(notification):
            try:
                return callback(notification)
            except Exception as e:
                self.logger.error(f"Error in config callback {subscription_id}: {e}")
                return None

        return wrapped_callback

    def _calculate_changes(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate changes between configurations."""
        changes = {"added": [], "removed": [], "modified": []}

        old_keys = set(old_config.keys())
        new_keys = set(new_config.keys())

        # Added keys
        changes["added"] = list(new_keys - old_keys)

        # Removed keys
        changes["removed"] = list(old_keys - new_keys)

        # Modified keys
        for key in old_keys & new_keys:
            if old_config[key] != new_config[key]:
                changes["modified"].append(
                    {
                        "key": key,
                        "old_value": old_config[key],
                        "new_value": new_config[key],
                    }
                )

        return changes

    def _increment_version(self, module_name: str) -> str:
        """Increment version number for module config."""
        current_version = self.config_history.get(module_name, {}).get(
            "version", "1.0.0"
        )
        try:
            major, minor, patch = map(int, current_version.split("."))
            return f"{major}.{minor}.{patch + 1}"
        except:
            return "1.0.0"

    def _initialize_validation_schemas(self) -> dict[str, dict[str, Any]]:
        """Initialize validation schemas for each module."""
        return {
            "core": {
                "validation_enabled": {"type": bool, "required": True},
                "auto_correct_enabled": {"type": bool, "required": True},
                "event_history_size": {"type": int, "range": [100, 10000]},
                "logging_level": {
                    "type": str,
                    "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
                },
                "physics_enabled": {"type": bool},
                "spin_enabled": {"type": bool},
                "collision_detection": {"type": bool},
            },
            "vision": {
                "detection_frequency": {"type": int, "range": [1, 120]},
                "confidence_threshold": {"type": float, "range": [0.0, 1.0]},
                "tracking_enabled": {"type": bool},
            },
            "api": {
                "update_frequency": {"type": int, "range": [1, 120]},
                "enable_websocket": {"type": bool},
                "cors_enabled": {"type": bool},
                "port": {"type": int, "range": [1000, 65535]},
            },
            "projector": {
                "brightness": {"type": float, "range": [0.0, 1.0]},
                "contrast": {"type": float, "range": [0.1, 2.0]},
                "projection_mode": {
                    "type": str,
                    "choices": ["overlay", "augmented", "full"],
                },
                "show_trajectories": {"type": bool},
                "show_assistance": {"type": bool},
            },
        }

    def _get_default_configs(self) -> dict[str, dict[str, Any]]:
        """Get default configurations for all modules."""
        return {
            "core": {
                "validation_enabled": True,
                "auto_correct_enabled": True,
                "event_history_size": 1000,
                "logging_level": "INFO",
                "physics_enabled": True,
                "spin_enabled": True,
                "collision_detection": True,
            },
            "vision": {
                "detection_frequency": 60,
                "confidence_threshold": 0.7,
                "tracking_enabled": True,
            },
            "api": {
                "update_frequency": 60,
                "enable_websocket": True,
                "cors_enabled": True,
                "port": 8000,
            },
            "projector": {
                "brightness": 0.8,
                "contrast": 1.0,
                "projection_mode": "overlay",
                "show_trajectories": True,
                "show_assistance": True,
            },
        }
