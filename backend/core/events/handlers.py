"""Event handlers for system events and module coordination."""

import logging
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable

from .manager import EventType

if TYPE_CHECKING:
    from backend.core.models import BallState, CueState, GameState

logger = logging.getLogger(__name__)


class CoreEventHandlers:
    """Core system event handlers implementing FR-CORE-056 to FR-CORE-060."""

    def __init__(self, event_manager, game_state_manager=None):
        """Initialize core event handlers.

        Args:
            event_manager: EventManager instance
            game_state_manager: Optional GameStateManager instance
        """
        self.event_manager = event_manager
        self.game_state_manager = game_state_manager
        self.logger = logging.getLogger(__name__)

        # Handler registry
        self.handlers: dict[str, list[Callable]] = {
            EventType.STATE_UPDATED.value: [self.handle_state_change],
            EventType.VISION_DATA_RECEIVED.value: [self.handle_vision_data],
            EventType.BALL_MOVED.value: [self.handle_ball_movement],
            EventType.COLLISION_DETECTED.value: [self.handle_collision_event],
            EventType.SHOT_DETECTED.value: [self.handle_shot_event],
            EventType.BALL_POCKETED.value: [self.handle_ball_pocketed],
            EventType.GAME_RESET.value: [self.handle_game_reset],
            EventType.MODULE_INITIALIZED.value: [self.handle_module_initialization],
            EventType.MODULE_SHUTDOWN.value: [self.handle_module_shutdown],
            EventType.CONFIG_CHANGED.value: [self.handle_config_change],
            EventType.ERROR_OCCURRED.value: [self.handle_error],
            EventType.CALIBRATION_UPDATED.value: [self.handle_calibration_update],
        }

        # Statistics tracking
        self.handler_stats = {
            "events_handled": 0,
            "errors": 0,
            "last_activity": time.time(),
        }

        # Register handlers with event manager
        self._register_handlers()

    def _register_handlers(self):
        """Register all handlers with the event manager."""
        for event_type, handlers in self.handlers.items():
            for handler in handlers:
                self.event_manager.subscribe_to_events(
                    event_type, self._create_handler_wrapper(handler)
                )

    def _create_handler_wrapper(self, handler: Callable):
        """Create a wrapper for handlers to add error handling and statistics."""

        def wrapper(event_type: str, data: dict[str, Any]):
            try:
                self.handler_stats["events_handled"] += 1
                self.handler_stats["last_activity"] = time.time()
                handler(event_type, data)
            except Exception as e:
                self.handler_stats["errors"] += 1
                self.logger.error(f"Error in handler {handler.__name__}: {e}")
                # Emit error event
                self.event_manager.emit_event(
                    EventType.ERROR_OCCURRED.value,
                    {
                        "handler": handler.__name__,
                        "original_event_type": event_type,
                        "error": str(e),
                        "timestamp": time.time(),
                    },
                )

        return wrapper

    def handle_state_change(self, event_type: str, data: dict[str, Any]):
        """Handle game state change events."""
        frame_number = data.get("frame_number")
        timestamp = data.get("timestamp")
        balls_count = data.get("balls_count", 0)
        events = data.get("events", [])

        self.logger.debug(
            f"State updated: frame {frame_number}, {balls_count} balls, {len(events)} events"
        )

        # Process state-specific events
        for event in events:
            if event.get("event_type") == "ball_motion":
                self.event_manager.emit_event(EventType.BALL_MOVED.value, event)
            elif event.get("event_type") == "ball_pocketed":
                self.event_manager.emit_event(EventType.BALL_POCKETED.value, event)

        # Notify other modules of state update
        if self.game_state_manager:
            current_state = self.game_state_manager.get_current_state()
            if current_state:
                # Send to API module
                self.event_manager.send_state_update_to_api(
                    {
                        "state": self._serialize_state(current_state),
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                    }
                )

                # Send trajectory data to projector if balls are moving
                moving_balls = [
                    ball for ball in current_state.balls if ball.is_moving()
                ]
                if moving_balls:
                    self.event_manager.provide_trajectory_to_projector(
                        {
                            "moving_balls": [
                                self._serialize_ball(ball) for ball in moving_balls
                            ],
                            "timestamp": timestamp,
                        }
                    )

    def handle_vision_data(self, event_type: str, data: dict[str, Any]):
        """Handle vision detection data (FR-CORE-051)."""
        detection_data = data.get("detection_data", {})

        self.logger.debug(
            f"Received vision data with {len(detection_data.get('balls', []))} balls"
        )

        # Process vision data through game state manager
        if self.game_state_manager:
            try:
                updated_state = self.game_state_manager.update_state(detection_data)

                # Analyze for significant events
                self._analyze_state_for_events(updated_state)

            except Exception as e:
                self.logger.error(f"Error processing vision data: {e}")
                self.event_manager.emit_event(
                    EventType.ERROR_OCCURRED.value,
                    {
                        "source": "vision_data_processing",
                        "error": str(e),
                        "timestamp": time.time(),
                    },
                )

    def handle_ball_movement(self, event_type: str, data: dict[str, Any]):
        """Handle ball movement events."""
        ball_id = data.get("ball_id")
        distance = data.get("distance", 0)

        self.logger.debug(f"Ball {ball_id} moved {distance:.2f}mm")

        # Check for shot detection
        if distance > 50:  # Significant movement threshold
            self.event_manager.emit_event(
                EventType.SHOT_DETECTED.value,
                {
                    "initiating_ball": ball_id,
                    "initial_velocity": distance,  # Approximate
                    "timestamp": time.time(),
                },
            )

    def handle_collision_event(self, event_type: str, data: dict[str, Any]):
        """Handle collision detection events."""
        ball1_id = data.get("ball1_id")
        ball2_id = data.get("ball2_id")
        collision_type = data.get("type", "unknown")

        self.logger.debug(
            f"Collision detected: {ball1_id} with {ball2_id or 'cushion'} ({collision_type})"
        )

        # Log collision for analysis
        collision_data = {
            "ball1_id": ball1_id,
            "ball2_id": ball2_id,
            "type": collision_type,
            "timestamp": time.time(),
            "position": data.get("position"),
        }

        # Store collision for physics validation
        if hasattr(self, "collision_history"):
            self.collision_history.append(collision_data)

    def handle_shot_event(self, event_type: str, data: dict[str, Any]):
        """Handle shot detection events."""
        initiating_ball = data.get("initiating_ball")
        timestamp = data.get("timestamp")

        self.logger.info(f"Shot detected from ball {initiating_ball}")

        # Reset any previous trajectory predictions
        self.event_manager.emit_event(
            "trajectory_reset", {"reason": "new_shot_detected", "timestamp": timestamp}
        )

        # Request new trajectory calculation
        if self.game_state_manager:
            current_state = self.game_state_manager.get_current_state()
            if current_state and current_state.cue:
                self.event_manager.emit_event(
                    "trajectory_calculation_requested",
                    {
                        "cue_state": self._serialize_cue(current_state.cue),
                        "game_state": self._serialize_state(current_state),
                        "timestamp": timestamp,
                    },
                )

    def handle_ball_pocketed(self, event_type: str, data: dict[str, Any]):
        """Handle ball pocketed events."""
        ball_id = data.get("ball_id")
        ball_number = data.get("ball_number")
        is_cue_ball = data.get("is_cue_ball", False)

        if is_cue_ball:
            self.logger.warning("Cue ball pocketed - scratch detected")
            self.event_manager.emit_event(
                "scratch_detected", {"timestamp": time.time(), "ball_id": ball_id}
            )
        else:
            self.logger.info(f"Ball {ball_number or ball_id} pocketed")

    def handle_game_reset(self, event_type: str, data: dict[str, Any]):
        """Handle game reset events."""
        game_type = data.get("game_type")
        timestamp = data.get("timestamp")

        self.logger.info(f"Game reset to {game_type}")

        # Clear any cached data
        if hasattr(self, "collision_history"):
            self.collision_history.clear()

        # Notify all modules of reset
        self.event_manager.emit_event(
            "system_reset", {"game_type": game_type, "timestamp": timestamp}
        )

    def handle_module_initialization(self, event_type: str, data: dict[str, Any]):
        """Handle module initialization events (FR-CORE-055)."""
        module_name = data.get("module_name")
        capabilities = data.get("capabilities", [])

        self.logger.info(
            f"Module {module_name} initialized with capabilities: {capabilities}"
        )

        # Send welcome configuration if needed
        if module_name in ["api", "projector", "vision"]:
            self.event_manager.exchange_config_with_module(
                module_name, self._get_module_config(module_name)
            )

    def handle_module_shutdown(self, event_type: str, data: dict[str, Any]):
        """Handle module shutdown events (FR-CORE-055)."""
        module_name = data.get("module_name")

        self.logger.info(f"Module {module_name} shutting down")

        # Clean up any module-specific resources
        if hasattr(self.event_manager.coordinator, "module_states"):
            self.event_manager.coordinator.module_states.pop(module_name, None)

    def handle_config_change(self, event_type: str, data: dict[str, Any]):
        """Handle configuration change events (FR-CORE-054)."""
        module_name = data.get("module_name")
        config = data.get("config")

        self.logger.info(f"Configuration updated for {module_name}")

        # Apply configuration changes if they affect core module
        if module_name == "core" and config:
            self._apply_core_config(config)

    def handle_error(self, event_type: str, data: dict[str, Any]):
        """Handle error events."""
        source = data.get("source", "unknown")
        error = data.get("error")
        data.get("timestamp")

        self.logger.error(f"Error from {source}: {error}")

        # Could implement error recovery logic here
        # For now, just log and track

    def handle_calibration_update(self, event_type: str, data: dict[str, Any]):
        """Handle calibration update events."""
        calibration_type = data.get("type")
        parameters = data.get("parameters")

        self.logger.info(f"Calibration updated: {calibration_type}")

        # Notify relevant modules of calibration changes
        if calibration_type == "camera":
            self.event_manager.emit_targeted_event(
                "calibration_updated",
                {"type": calibration_type, "parameters": parameters},
                ["vision"],
            )
        elif calibration_type == "projector":
            self.event_manager.emit_targeted_event(
                "calibration_updated",
                {"type": calibration_type, "parameters": parameters},
                ["projector"],
            )

    def _analyze_state_for_events(self, state: "GameState"):
        """Analyze game state for significant events."""
        # Check for collisions based on ball proximity and velocities
        active_balls = [ball for ball in state.balls if not ball.is_pocketed]

        for i, ball1 in enumerate(active_balls):
            for ball2 in active_balls[i + 1 :]:
                if ball1.is_touching(ball2, tolerance=0.005):  # 5mm tolerance
                    self.event_manager.emit_event(
                        EventType.COLLISION_DETECTED.value,
                        {
                            "ball1_id": ball1.id,
                            "ball2_id": ball2.id,
                            "type": "ball",
                            "position": {
                                "x": (ball1.position.x + ball2.position.x) / 2,
                                "y": (ball1.position.y + ball2.position.y) / 2,
                            },
                            "timestamp": time.time(),
                        },
                    )

    def _serialize_state(self, state: "GameState") -> dict[str, Any]:
        """Serialize game state for transmission."""
        return {
            "timestamp": state.timestamp,
            "frame_number": state.frame_number,
            "balls": [self._serialize_ball(ball) for ball in state.balls],
            "table": (
                state.table.to_dict()
                if hasattr(state.table, "to_dict")
                else asdict(state.table)
            ),
            "cue": self._serialize_cue(state.cue) if state.cue else None,
            "game_type": state.game_type.value,
            "is_break": state.is_break,
            "scores": state.scores,
        }

    def _serialize_ball(self, ball: "BallState") -> dict[str, Any]:
        """Serialize ball state for transmission."""
        return {
            "id": ball.id,
            "position": {"x": ball.position.x, "y": ball.position.y},
            "velocity": {"x": ball.velocity.x, "y": ball.velocity.y},
            "radius": ball.radius,
            "is_cue_ball": ball.is_cue_ball,
            "is_pocketed": ball.is_pocketed,
            "number": ball.number,
            "confidence": ball.confidence,
        }

    def _serialize_cue(self, cue: "CueState") -> dict[str, Any]:
        """Serialize cue state for transmission."""
        return {
            "tip_position": {"x": cue.tip_position.x, "y": cue.tip_position.y},
            "angle": cue.angle,
            "elevation": cue.elevation,
            "estimated_force": cue.estimated_force,
            "impact_point": (
                {"x": cue.impact_point.x, "y": cue.impact_point.y}
                if cue.impact_point
                else None
            ),
            "confidence": cue.confidence,
        }

    def _get_module_config(self, module_name: str) -> dict[str, Any]:
        """Get configuration for a specific module."""
        configs = {
            "api": {
                "update_frequency": 60,  # Hz
                "enable_websocket": True,
                "cors_enabled": True,
            },
            "projector": {
                "projection_mode": "overlay",
                "brightness": 0.8,
                "show_trajectories": True,
                "show_assistance": True,
            },
            "vision": {
                "detection_frequency": 60,  # Hz
                "confidence_threshold": 0.7,
                "tracking_enabled": True,
            },
        }
        return configs.get(module_name, {})

    def _apply_core_config(self, config: dict[str, Any]):
        """Apply configuration changes to core module."""
        # Example configuration application
        if "validation_enabled" in config and self.game_state_manager:
            self.game_state_manager.set_validation_config(
                enabled=config["validation_enabled"],
                auto_correct=config.get("auto_correct_enabled", True),
            )

        if "logging_level" in config:
            logging.getLogger("backend.core").setLevel(
                getattr(logging, config["logging_level"].upper())
            )

    def get_handler_statistics(self) -> dict[str, Any]:
        """Get statistics about event handling."""
        return {
            **self.handler_stats,
            "registered_handlers": len(self.handlers),
            "total_event_types": len(self.handlers),
        }

    def add_custom_handler(self, event_type: str, handler: Callable):
        """Add a custom event handler (FR-CORE-060)."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)

        # Register with event manager
        self.event_manager.subscribe_to_events(
            event_type, self._create_handler_wrapper(handler)
        )

        self.logger.info(f"Custom handler added for event type: {event_type}")

    def remove_custom_handler(self, event_type: str, handler: Callable) -> bool:
        """Remove a custom event handler."""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            self.logger.info(f"Custom handler removed for event type: {event_type}")
            return True
        return False


class EventHandlers(CoreEventHandlers):
    """Legacy alias for backward compatibility."""

    pass
