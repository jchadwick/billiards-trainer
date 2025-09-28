"""Core Module - Main orchestration and integration layer.

This module provides the primary interface for all core functionality including:
- Game state management
- Physics calculations and trajectory prediction
- Shot analysis and assistance
- Event coordination
- Configuration management
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from .analysis.assistance import AssistanceEngine
from .analysis.prediction import OutcomePredictor
from .analysis.shot import ShotAnalysis, ShotAnalyzer
from .events.manager import EventManager

# Core component imports
from .game_state import BallState, GameState, GameStateManager, GameType
from .models import Collision, ShotType, Vector2D
from .physics.collision import CollisionDetector, CollisionResolver
from .physics.engine import PhysicsEngine
from .physics.trajectory import TrajectoryCalculator
from .utils.cache import CalculationCache
from .utils.geometry import GeometryUtils
from .utils.math import MathUtils

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class CoreModuleConfig:
    """Configuration for the Core Module."""

    physics_enabled: bool = True
    prediction_enabled: bool = True
    assistance_enabled: bool = True
    cache_size: int = 1000
    max_trajectory_time: float = 10.0
    collision_tolerance: float = 0.001
    state_history_limit: int = 1000
    performance_monitoring: bool = True
    async_processing: bool = True
    debug_mode: bool = False


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""

    total_updates: int = 0
    avg_update_time: float = 0.0
    avg_physics_time: float = 0.0
    avg_analysis_time: float = 0.0
    cache_hit_rate: float = 0.0
    last_update_time: float = 0.0
    errors_count: int = 0


class CoreModuleError(Exception):
    """Base exception for Core Module errors."""

    pass


class CoreModule:
    """Main Core Module - Orchestrates all core functionality.

    This class provides the primary interface for:
    - Game state management and updates
    - Physics calculations and trajectory prediction
    - Shot analysis and outcome prediction
    - Event management and coordination
    - Performance monitoring and optimization
    """

    def __init__(self, config: Optional[CoreModuleConfig] = None):
        """Initialize the Core Module.

        Args:
            config: Configuration for the core module
        """
        self.config = config or CoreModuleConfig()

        # Initialize performance metrics
        self.metrics = PerformanceMetrics()

        # Initialize core components
        self._initialize_components()

        # Initialize caches
        self._initialize_caches()

        # Initialize async locks
        self._state_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()

        # State tracking
        self._current_state: Optional[GameState] = None
        self._state_history: list[GameState] = []
        self._last_update_time = 0.0

        # Event callbacks
        self._event_callbacks: dict[str, list[Callable]] = {}

        logger.info("Core Module initialized successfully")

    def _initialize_components(self):
        """Initialize all core components."""
        try:
            # Core managers
            self.state_manager = GameStateManager()
            self.event_manager = EventManager()

            # Physics components
            physics_config = {
                "collision_tolerance": self.config.collision_tolerance,
                "max_simulation_time": self.config.max_trajectory_time,
            }
            self.physics_engine = PhysicsEngine(physics_config)
            self.trajectory_calculator = TrajectoryCalculator()
            self.collision_detector = CollisionDetector()
            self.collision_resolver = CollisionResolver()

            # Analysis components
            self.shot_analyzer = ShotAnalyzer()
            self.assistance_engine = AssistanceEngine()
            self.outcome_predictor = OutcomePredictor()

            # Utility components
            self.geometry_utils = GeometryUtils()
            self.math_utils = MathUtils()

            logger.info("All core components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize core components: {e}")
            raise CoreModuleError(f"Component initialization failed: {e}")

    def _initialize_caches(self):
        """Initialize caching systems for performance optimization."""
        self.trajectory_cache = CalculationCache(self.config.cache_size)
        self.analysis_cache = CalculationCache(self.config.cache_size)
        self.collision_cache = CalculationCache(self.config.cache_size)

        logger.debug("Caches initialized")

    async def update_state(self, detection_data: dict[str, Any]) -> GameState:
        """Update the game state from new detection data.

        Args:
            detection_data: Raw detection data from vision system

        Returns:
            Updated game state

        Raises:
            CoreModuleError: If state update fails
        """
        start_time = time.time()

        try:
            async with self._state_lock:
                # Update game state through state manager
                new_state = await self._async_update_state(detection_data)

                # Store in history
                self._add_to_history(new_state)

                # Update current state
                self._current_state = new_state
                self._last_update_time = time.time()

                # Emit state change event
                await self._emit_event(
                    "state_updated",
                    {"state": asdict(new_state), "timestamp": new_state.timestamp},
                )

                # Update performance metrics
                update_time = time.time() - start_time
                self._update_performance_metrics(update_time)

                logger.debug(f"State updated in {update_time:.4f}s")
                return new_state

        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"State update failed: {e}")
            raise CoreModuleError(f"State update failed: {e}")

    async def _async_update_state(self, detection_data: dict[str, Any]) -> GameState:
        """Async wrapper for state manager update."""
        if self.config.async_processing:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.state_manager.update_state, detection_data
            )
        else:
            return self.state_manager.update_state(detection_data)

    async def calculate_trajectory(
        self,
        ball_id: str,
        initial_velocity: Vector2D,
        time_limit: Optional[float] = None,
    ) -> list[Vector2D]:
        """Calculate ball trajectory using physics simulation.

        Args:
            ball_id: ID of the ball to calculate trajectory for
            initial_velocity: Initial velocity vector
            time_limit: Maximum simulation time (uses config default if None)

        Returns:
            List of positions along the trajectory

        Raises:
            CoreModuleError: If trajectory calculation fails
        """
        if not self._current_state:
            raise CoreModuleError("No current game state available")

        # Check cache first
        cache_key = f"{ball_id}_{initial_velocity.x}_{initial_velocity.y}"
        cached_result = self.trajectory_cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            start_time = time.time()

            # Get ball state
            ball = self._get_ball_by_id(ball_id)
            if not ball:
                raise CoreModuleError(f"Ball {ball_id} not found in current state")

            # Create a modified ball state with the new velocity
            ball_with_velocity = BallState(
                id=ball.id,
                position=ball.position,
                velocity=initial_velocity,
                radius=ball.radius,
                mass=ball.mass,
                spin=ball.spin,
                is_cue_ball=ball.is_cue_ball,
                is_pocketed=ball.is_pocketed,
                number=ball.number,
            )

            # Get other balls for collision detection
            other_balls = [
                b
                for b in self._current_state.balls
                if b.id != ball_id and not b.is_pocketed
            ]

            # Calculate trajectory using physics engine
            if self.config.async_processing:
                loop = asyncio.get_event_loop()
                trajectory_points = await loop.run_in_executor(
                    None,
                    self.physics_engine.calculate_trajectory,
                    ball_with_velocity,
                    self._current_state.table,
                    other_balls,
                    time_limit or self.config.max_trajectory_time,
                )
            else:
                trajectory_points = self.physics_engine.calculate_trajectory(
                    ball_with_velocity,
                    self._current_state.table,
                    other_balls,
                    time_limit or self.config.max_trajectory_time,
                )

            # Convert trajectory points to Vector2D positions
            trajectory = [point.position for point in trajectory_points]

            # Cache result
            self.trajectory_cache.set(cache_key, trajectory)

            # Update metrics
            calc_time = time.time() - start_time
            self.metrics.avg_physics_time = (
                self.metrics.avg_physics_time * self.metrics.total_updates + calc_time
            ) / (self.metrics.total_updates + 1)

            logger.debug(f"Trajectory calculated in {calc_time:.4f}s")
            return trajectory

        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"Trajectory calculation failed: {e}")
            raise CoreModuleError(f"Trajectory calculation failed: {e}")

    async def analyze_shot(
        self, target_ball: Optional[str] = None, include_alternatives: bool = True
    ) -> ShotAnalysis:
        """Analyze current shot setup and provide recommendations.

        Args:
            target_ball: ID of target ball (auto-detect if None)
            include_alternatives: Whether to include alternative shot options

        Returns:
            Shot analysis with recommendations

        Raises:
            CoreModuleError: If shot analysis fails
        """
        if not self._current_state:
            raise CoreModuleError("No current game state available")

        # Check cache
        cache_key = f"shot_{target_ball}_{self._current_state.timestamp}"
        cached_result = self.analysis_cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            start_time = time.time()

            # Perform shot analysis
            if self.config.async_processing:
                loop = asyncio.get_event_loop()
                analysis = await loop.run_in_executor(
                    None,
                    self.shot_analyzer.analyze_shot,
                    self._current_state,
                    target_ball,
                )
            else:
                analysis = self.shot_analyzer.analyze_shot(
                    self._current_state, target_ball
                )

            # Get alternatives if requested
            if include_alternatives and self.config.assistance_enabled:
                alternatives = await self._get_alternative_shots(target_ball)
                analysis.alternative_shots = alternatives

            # Cache result
            self.analysis_cache.set(cache_key, analysis)

            # Update metrics
            analysis_time = time.time() - start_time
            self.metrics.avg_analysis_time = (
                self.metrics.avg_analysis_time * self.metrics.total_updates
                + analysis_time
            ) / (self.metrics.total_updates + 1)

            logger.debug(f"Shot analyzed in {analysis_time:.4f}s")
            return analysis

        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"Shot analysis failed: {e}")
            raise CoreModuleError(f"Shot analysis failed: {e}")

    async def predict_outcomes(
        self, shot_velocity: Vector2D, num_predictions: int = 5
    ) -> list[dict[str, Any]]:
        """Predict possible outcomes for a given shot.

        Args:
            shot_velocity: Velocity to apply to cue ball
            num_predictions: Number of prediction scenarios

        Returns:
            List of predicted outcomes with probabilities

        Raises:
            CoreModuleError: If outcome prediction fails
        """
        if not self._current_state:
            raise CoreModuleError("No current game state available")

        try:
            start_time = time.time()

            # Get predictions
            if self.config.async_processing:
                loop = asyncio.get_event_loop()
                predictions = await loop.run_in_executor(
                    None,
                    self.outcome_predictor.predict_outcomes,
                    self._current_state,
                    shot_velocity,
                    num_predictions,
                )
            else:
                predictions = self.outcome_predictor.predict_outcomes(
                    self._current_state, shot_velocity, num_predictions
                )

            logger.debug(f"Outcomes predicted in {time.time() - start_time:.4f}s")
            return predictions

        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"Outcome prediction failed: {e}")
            raise CoreModuleError(f"Outcome prediction failed: {e}")

    async def suggest_shots(
        self,
        difficulty_filter: Optional[float] = None,
        shot_type_filter: Optional[ShotType] = None,
        max_suggestions: int = 3,
    ) -> list[ShotAnalysis]:
        """Suggest optimal shots based on current game state.

        Args:
            difficulty_filter: Maximum difficulty level (0.0-1.0)
            shot_type_filter: Filter by specific shot type
            max_suggestions: Maximum number of suggestions

        Returns:
            List of recommended shots

        Raises:
            CoreModuleError: If shot suggestion fails
        """
        if not self._current_state:
            raise CoreModuleError("No current game state available")

        try:
            start_time = time.time()

            # Get shot suggestions
            if self.config.async_processing:
                loop = asyncio.get_event_loop()
                suggestions = await loop.run_in_executor(
                    None,
                    self.assistance_engine.suggest_shots,
                    self._current_state,
                    difficulty_filter or 0.5,
                )
            else:
                suggestions = self.assistance_engine.suggest_shots(
                    self._current_state, difficulty_filter or 0.5
                )

            # Filter by shot type if specified
            if shot_type_filter:
                suggestions = [
                    s
                    for s in suggestions
                    if hasattr(s, "shot_type") and s.shot_type == shot_type_filter
                ]

            # Limit number of suggestions
            suggestions = suggestions[:max_suggestions]

            logger.debug(f"Shots suggested in {time.time() - start_time:.4f}s")
            return suggestions

        except Exception as e:
            self.metrics.errors_count += 1
            logger.error(f"Shot suggestion failed: {e}")
            raise CoreModuleError(f"Shot suggestion failed: {e}")

    async def validate_state(self) -> dict[str, Any]:
        """Validate current game state for consistency and correctness.

        Returns:
            Validation results with any issues found
        """
        if not self._current_state:
            return {"valid": False, "issues": ["No current game state available"]}

        try:
            # Basic validation checks
            issues = []

            # Check ball positions
            for ball in self._current_state.balls:
                if ball.position.x < 0 or ball.position.y < 0:
                    issues.append(f"Ball {ball.id} has invalid position")

                if ball.velocity.magnitude() > 50.0:  # Unrealistic velocity
                    issues.append(f"Ball {ball.id} has unrealistic velocity")

            # Check table bounds
            table = self._current_state.table
            if table.width <= 0 or table.height <= 0:
                issues.append("Invalid table dimensions")

            # Check for ball overlaps
            ball_overlaps = self._check_ball_overlaps()
            if ball_overlaps:
                issues.extend(ball_overlaps)

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "timestamp": self._current_state.timestamp,
            }

        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return {"valid": False, "issues": [f"Validation error: {e}"]}

    def get_current_state(self) -> Optional[GameState]:
        """Get the current game state."""
        return self._current_state

    def get_state_history(self, count: Optional[int] = None) -> list[GameState]:
        """Get historical game states.

        Args:
            count: Number of states to return (all if None)

        Returns:
            List of historical game states
        """
        if count is None:
            return self._state_history.copy()
        return self._state_history[-count:]

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Note: Cache hit rate calculation would need to be implemented
        # in the CalculationCache class to track hits/misses
        self.metrics.cache_hit_rate = 0.0  # Placeholder

        return self.metrics

    async def reset_game(self, game_type: GameType = GameType.PRACTICE) -> None:
        """Reset the game to initial state.

        Args:
            game_type: Type of game to initialize
        """
        try:
            async with self._state_lock:
                # Clear state history
                self._state_history.clear()
                self._current_state = None

                # Reset state manager
                self.state_manager.reset_game(game_type)

                # Clear caches
                self.trajectory_cache.clear()
                self.analysis_cache.clear()
                self.collision_cache.clear()

                # Reset performance metrics
                self.metrics = PerformanceMetrics()

                # Emit reset event
                await self._emit_event("game_reset", {"game_type": game_type.value})

                logger.info(f"Game reset to {game_type.value}")

        except Exception as e:
            logger.error(f"Game reset failed: {e}")
            raise CoreModuleError(f"Game reset failed: {e}")

    def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
        """Subscribe to core module events.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs

        Returns:
            Subscription ID for unsubscribing
        """
        return self.event_manager.subscribe_to_events(event_type, callback)

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe_to_events

        Returns:
            True if unsubscribed successfully
        """
        return self.event_manager.unsubscribe(subscription_id)

    # Private helper methods

    def _add_to_history(self, state: GameState) -> None:
        """Add state to history with size limit."""
        self._state_history.append(state)
        if len(self._state_history) > self.config.state_history_limit:
            self._state_history.pop(0)

    def _get_ball_by_id(self, ball_id: str) -> Optional[BallState]:
        """Find ball in current state by ID."""
        if not self._current_state:
            return None

        for ball in self._current_state.balls:
            if ball.id == ball_id:
                return ball
        return None

    def _check_ball_overlaps(self) -> list[str]:
        """Check for overlapping balls."""
        if not self._current_state:
            return []

        issues = []
        balls = self._current_state.balls

        for i, ball1 in enumerate(balls):
            for ball2 in balls[i + 1 :]:
                distance = self.geometry_utils.distance(ball1.position, ball2.position)
                min_distance = ball1.radius + ball2.radius

                if distance < min_distance:
                    issues.append(f"Balls {ball1.id} and {ball2.id} are overlapping")

        return issues

    async def _get_alternative_shots(
        self, target_ball: Optional[str]
    ) -> list[ShotAnalysis]:
        """Get alternative shot suggestions."""
        try:
            # Use the assistance engine to get alternative shots
            suggestions = await self.suggest_shots(max_suggestions=5)
            return suggestions
        except Exception as e:
            logger.warning(f"Failed to get alternative shots: {e}")
            return []

    def _update_performance_metrics(self, update_time: float) -> None:
        """Update performance tracking metrics."""
        self.metrics.total_updates += 1
        self.metrics.avg_update_time = (
            self.metrics.avg_update_time * (self.metrics.total_updates - 1)
            + update_time
        ) / self.metrics.total_updates
        self.metrics.last_update_time = update_time

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event through the event manager."""
        try:
            if self.config.async_processing:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self.event_manager.emit_event, event_type, data
                )
            else:
                self.event_manager.emit_event(event_type, data)
        except Exception as e:
            logger.warning(f"Failed to emit event {event_type}: {e}")

    def __str__(self) -> str:
        """String representation."""
        return f"CoreModule(state={'loaded' if self._current_state else 'empty'}, updates={self.metrics.total_updates})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"CoreModule("
            f"config={self.config}, "
            f"state_loaded={self._current_state is not None}, "
            f"history_size={len(self._state_history)}, "
            f"metrics={self.metrics}"
            f")"
        )


# Export main classes and types
__all__ = [
    "CoreModule",
    "CoreModuleConfig",
    "CoreModuleError",
    "PerformanceMetrics",
    "GameState",
    "GameType",
    "ShotType",
    "ShotAnalysis",
    "Vector2D",
    "Collision",
]
