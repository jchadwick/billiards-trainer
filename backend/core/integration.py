"""Integration interfaces for core module communication with other backend modules."""

import logging
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
        pass

    @abstractmethod
    def request_calibration(self, calibration_type: str) -> dict[str, Any]:
        """Request calibration data from Vision module."""
        pass

    @abstractmethod
    def set_detection_parameters(self, parameters: dict[str, Any]) -> bool:
        """Set detection parameters in Vision module."""
        pass


class APIInterface(ABC):
    """Interface for API module integration (FR-CORE-052)."""

    @abstractmethod
    def send_state_update(self, state_data: dict[str, Any]) -> None:
        """Send state updates to API module."""
        pass

    @abstractmethod
    def send_event_notification(self, event_data: dict[str, Any]) -> None:
        """Send event notifications to API module."""
        pass

    @abstractmethod
    def register_websocket_handler(self, handler: Callable) -> str:
        """Register WebSocket message handler."""
        pass


class ProjectorInterface(ABC):
    """Interface for Projector module integration (FR-CORE-053)."""

    @abstractmethod
    def send_trajectory_data(self, trajectory_data: dict[str, Any]) -> None:
        """Send trajectory data to Projector module."""
        pass

    @abstractmethod
    def send_overlay_data(self, overlay_data: dict[str, Any]) -> None:
        """Send overlay visualization data."""
        pass

    @abstractmethod
    def update_projection_settings(self, settings: dict[str, Any]) -> None:
        """Update projector settings."""
        pass


class ConfigInterface(ABC):
    """Interface for Config module integration (FR-CORE-054)."""

    @abstractmethod
    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        pass

    @abstractmethod
    def update_module_config(self, module_name: str, config: dict[str, Any]) -> None:
        """Update configuration for a specific module."""
        pass

    @abstractmethod
    def subscribe_config_changes(self, callback: Callable) -> str:
        """Subscribe to configuration changes."""
        pass


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
    """Example implementation of Vision interface."""

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.calibration_data = {}
        self.detection_parameters = {}

    def receive_detection_data(self, detection_data: dict[str, Any]) -> None:
        """Receive detection data from Vision module."""
        self.event_manager.receive_vision_data(detection_data)

    def request_calibration(self, calibration_type: str) -> dict[str, Any]:
        """Request calibration data from Vision module."""
        # This would interface with actual vision module
        return self.calibration_data.get(calibration_type, {})

    def set_detection_parameters(self, parameters: dict[str, Any]) -> bool:
        """Set detection parameters in Vision module."""
        self.detection_parameters.update(parameters)
        return True


class APIInterfaceImpl(APIInterface):
    """Example implementation of API interface."""

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.websocket_handlers = {}

    def send_state_update(self, state_data: dict[str, Any]) -> None:
        """Send state updates to API module."""
        # This would interface with actual API module
        pass

    def send_event_notification(self, event_data: dict[str, Any]) -> None:
        """Send event notifications to API module."""
        # This would interface with actual API module
        pass

    def register_websocket_handler(self, handler: Callable) -> str:
        """Register WebSocket message handler."""
        handler_id = f"handler_{len(self.websocket_handlers)}"
        self.websocket_handlers[handler_id] = handler
        return handler_id


class ProjectorInterfaceImpl(ProjectorInterface):
    """Example implementation of Projector interface."""

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.projection_settings = {}

    def send_trajectory_data(self, trajectory_data: dict[str, Any]) -> None:
        """Send trajectory data to Projector module."""
        # This would interface with actual projector module
        pass

    def send_overlay_data(self, overlay_data: dict[str, Any]) -> None:
        """Send overlay visualization data."""
        # This would interface with actual projector module
        pass

    def update_projection_settings(self, settings: dict[str, Any]) -> None:
        """Update projector settings."""
        self.projection_settings.update(settings)


class ConfigInterfaceImpl(ConfigInterface):
    """Example implementation of Config interface."""

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.module_configs = {}
        self.config_subscribers = {}

    def get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        return self.module_configs.get(module_name, {})

    def update_module_config(self, module_name: str, config: dict[str, Any]) -> None:
        """Update configuration for a specific module."""
        self.module_configs[module_name] = config
        self.event_manager.exchange_config_with_module(module_name, config)

    def subscribe_config_changes(self, callback: Callable) -> str:
        """Subscribe to configuration changes."""
        subscription_id = f"config_sub_{len(self.config_subscribers)}"
        self.config_subscribers[subscription_id] = callback
        return subscription_id
