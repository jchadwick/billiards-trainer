"""Model Conversion Utilities.

This module provides utilities for converting between API models and core backend models.
It handles the transformation of data between different layers of the application while
maintaining type safety and validation.

Key features:
- Convert between API and core data models
- Handle coordinate transformations
- Preserve validation and type safety
- Support batch conversions
- Error handling and logging
"""

from datetime import datetime
from typing import Any, Union

from backend.core.models import (
    BallState,
    Collision,
    CueState,
    GameEvent,
    GameState,
    GameType,
    ShotAnalysis,
    TableState,
    Trajectory,
    Vector2D,
)

from .common import Coordinate2D, ValidationResult
from .responses import (
    BallInfo,
    CueInfo,
    GameStateResponse,
    ShotAnalysisResponse,
    TableInfo,
    TrajectoryInfo,
)
from .responses import GameEvent as APIGameEvent
from .websocket import (
    BallStateData,
    CueStateData,
    GameStateData,
    TableStateData,
    TrajectoryData,
)
from .websocket import CollisionInfo as WSCollisionInfo

# =============================================================================
# Core to API Model Converters
# =============================================================================


def vector2d_to_coordinate2d(vector: Vector2D) -> Coordinate2D:
    """Convert core Vector2D to API Coordinate2D."""
    return Coordinate2D(x=vector.x, y=vector.y)


def vector2d_to_list(vector: Vector2D) -> list[float]:
    """Convert core Vector2D to list format [x, y]."""
    return [vector.x, vector.y]


def ball_state_to_ball_info(ball: BallState) -> BallInfo:
    """Convert core BallState to API BallInfo."""
    return BallInfo(
        id=ball.id,
        number=ball.number,
        position=vector2d_to_list(ball.position),
        velocity=vector2d_to_list(ball.velocity),
        is_cue_ball=ball.is_cue_ball,
        is_pocketed=ball.is_pocketed,
        confidence=ball.confidence,
        last_update=datetime.fromtimestamp(ball.last_update),
    )


def ball_state_to_websocket_data(ball: BallState) -> BallStateData:
    """Convert core BallState to WebSocket BallStateData."""
    return BallStateData(
        id=ball.id,
        number=ball.number,
        position=vector2d_to_list(ball.position),
        velocity=vector2d_to_list(ball.velocity),
        radius=ball.radius,
        is_cue_ball=ball.is_cue_ball,
        is_pocketed=ball.is_pocketed,
        confidence=ball.confidence,
    )


def cue_state_to_cue_info(cue: CueState) -> CueInfo:
    """Convert core CueState to API CueInfo."""
    return CueInfo(
        tip_position=vector2d_to_list(cue.tip_position),
        angle=cue.angle,
        elevation=cue.elevation,
        estimated_force=cue.estimated_force,
        is_visible=cue.is_visible,
        confidence=cue.confidence,
    )


def cue_state_to_websocket_data(cue: CueState) -> CueStateData:
    """Convert core CueState to WebSocket CueStateData."""
    return CueStateData(
        tip_position=vector2d_to_list(cue.tip_position),
        angle=cue.angle,
        elevation=cue.elevation,
        estimated_force=cue.estimated_force,
        is_visible=cue.is_visible,
        confidence=cue.confidence,
    )


def table_state_to_table_info(table: TableState) -> TableInfo:
    """Convert core TableState to API TableInfo."""
    return TableInfo(
        width=table.width,
        height=table.height,
        pocket_positions=[vector2d_to_list(pos) for pos in table.pocket_positions],
        pocket_radius=table.pocket_radius,
        surface_friction=table.surface_friction,
    )


def table_state_to_websocket_data(table: TableState) -> TableStateData:
    """Convert core TableState to WebSocket TableStateData."""
    return TableStateData(
        width=table.width,
        height=table.height,
        pocket_positions=[vector2d_to_list(pos) for pos in table.pocket_positions],
        pocket_radius=table.pocket_radius,
    )


def game_event_to_api_event(event: GameEvent) -> APIGameEvent:
    """Convert core GameEvent to API GameEvent."""
    return APIGameEvent(
        timestamp=datetime.fromtimestamp(event.timestamp),
        event_type=event.event_type,
        description=event.description,
        data=event.data,
    )


def game_state_to_response(state: GameState) -> GameStateResponse:
    """Convert core GameState to API GameStateResponse."""
    return GameStateResponse(
        timestamp=datetime.fromtimestamp(state.timestamp),
        frame_number=state.frame_number,
        balls=[ball_state_to_ball_info(ball) for ball in state.balls],
        cue=cue_state_to_cue_info(state.cue) if state.cue else None,
        table=table_state_to_table_info(state.table),
        game_type=state.game_type.value,
        is_valid=state.is_valid,
        confidence=state.state_confidence,
        events=[game_event_to_api_event(event) for event in state.events],
    )


def game_state_to_websocket_data(state: GameState) -> GameStateData:
    """Convert core GameState to WebSocket GameStateData."""
    return GameStateData(
        frame_number=state.frame_number,
        balls=[ball_state_to_websocket_data(ball) for ball in state.balls],
        cue=cue_state_to_websocket_data(state.cue) if state.cue else None,
        table=table_state_to_websocket_data(state.table),
        game_type=state.game_type.value,
        is_valid=state.is_valid,
        confidence=state.state_confidence,
    )


def collision_to_websocket_collision(collision: Collision) -> WSCollisionInfo:
    """Convert core Collision to WebSocket CollisionInfo."""
    return WSCollisionInfo(
        position=vector2d_to_list(collision.position),
        time=collision.time,
        type=collision.type,
        ball1_id=collision.ball1_id,
        ball2_id=collision.ball2_id,
        impact_angle=0.0,  # Calculate from collision data if needed
        confidence=collision.confidence,
    )


def trajectory_to_websocket_data(trajectory: Trajectory) -> TrajectoryData:
    """Convert core Trajectory to WebSocket TrajectoryData."""
    from .websocket import TrajectoryPoint

    # Convert trajectory points
    points = []
    time_per_point = (
        trajectory.time_to_rest / len(trajectory.points) if trajectory.points else 0
    )
    for i, point in enumerate(trajectory.points):
        points.append(
            TrajectoryPoint(
                position=vector2d_to_list(point),
                time=i * time_per_point,
                velocity=None,  # Would need to calculate from trajectory
            )
        )

    return TrajectoryData(
        ball_id=trajectory.ball_id,
        points=points,
        collisions=[collision_to_websocket_collision(c) for c in trajectory.collisions],
        will_be_pocketed=trajectory.will_be_pocketed,
        pocket_id=trajectory.pocket_id,
        time_to_rest=trajectory.time_to_rest,
        max_velocity=trajectory.max_velocity,
        confidence=trajectory.confidence,
    )


def trajectory_to_api_info(trajectory: Trajectory) -> TrajectoryInfo:
    """Convert core Trajectory to API TrajectoryInfo."""
    return TrajectoryInfo(
        ball_id=trajectory.ball_id,
        points=[vector2d_to_list(point) for point in trajectory.points],
        will_be_pocketed=trajectory.will_be_pocketed,
        pocket_id=trajectory.pocket_id,
        time_to_rest=trajectory.time_to_rest,
        max_velocity=trajectory.max_velocity,
        confidence=trajectory.confidence,
    )


def shot_analysis_to_response(analysis: ShotAnalysis) -> ShotAnalysisResponse:
    """Convert core ShotAnalysis to API ShotAnalysisResponse."""
    return ShotAnalysisResponse(
        shot_type=analysis.shot_type.value,
        difficulty=analysis.difficulty,
        success_probability=analysis.success_probability,
        recommended_force=analysis.recommended_force,
        recommended_angle=analysis.recommended_angle,
        target_ball_id=analysis.target_ball_id,
        target_pocket_id=analysis.target_pocket_id,
        potential_problems=analysis.potential_problems,
        risk_assessment=analysis.risk_assessment,
        trajectories=[],  # Would need trajectory data to populate
    )


# =============================================================================
# API to Core Model Converters
# =============================================================================


def coordinate2d_to_vector2d(coord: Coordinate2D) -> Vector2D:
    """Convert API Coordinate2D to core Vector2D."""
    return Vector2D(x=coord.x, y=coord.y)


def list_to_vector2d(coords: list[float]) -> Vector2D:
    """Convert list format [x, y] to core Vector2D."""
    if len(coords) != 2:
        raise ValueError("Coordinate list must have exactly 2 elements")
    return Vector2D(x=coords[0], y=coords[1])


def ball_info_to_ball_state(ball_info: BallInfo) -> BallState:
    """Convert API BallInfo to core BallState."""
    return BallState(
        id=ball_info.id,
        position=list_to_vector2d(ball_info.position),
        velocity=list_to_vector2d(ball_info.velocity),
        is_cue_ball=ball_info.is_cue_ball,
        is_pocketed=ball_info.is_pocketed,
        number=ball_info.number,
        confidence=ball_info.confidence,
        last_update=ball_info.last_update.timestamp(),
    )


def websocket_ball_data_to_ball_state(ball_data: BallStateData) -> BallState:
    """Convert WebSocket BallStateData to core BallState."""
    return BallState(
        id=ball_data.id,
        position=list_to_vector2d(ball_data.position),
        velocity=list_to_vector2d(ball_data.velocity),
        radius=ball_data.radius,
        is_cue_ball=ball_data.is_cue_ball,
        is_pocketed=ball_data.is_pocketed,
        number=ball_data.number,
        confidence=ball_data.confidence,
        last_update=datetime.now().timestamp(),
    )


def cue_info_to_cue_state(cue_info: CueInfo) -> CueState:
    """Convert API CueInfo to core CueState."""
    return CueState(
        tip_position=list_to_vector2d(cue_info.tip_position),
        angle=cue_info.angle,
        elevation=cue_info.elevation,
        estimated_force=cue_info.estimated_force,
        is_visible=cue_info.is_visible,
        confidence=cue_info.confidence,
        last_update=datetime.now().timestamp(),
    )


def websocket_cue_data_to_cue_state(cue_data: CueStateData) -> CueState:
    """Convert WebSocket CueStateData to core CueState."""
    return CueState(
        tip_position=list_to_vector2d(cue_data.tip_position),
        angle=cue_data.angle,
        elevation=cue_data.elevation,
        estimated_force=cue_data.estimated_force,
        is_visible=cue_data.is_visible,
        confidence=cue_data.confidence,
        last_update=datetime.now().timestamp(),
    )


def table_info_to_table_state(table_info: TableInfo) -> TableState:
    """Convert API TableInfo to core TableState."""
    return TableState(
        width=table_info.width,
        height=table_info.height,
        pocket_positions=[list_to_vector2d(pos) for pos in table_info.pocket_positions],
        pocket_radius=table_info.pocket_radius,
        surface_friction=table_info.surface_friction,
    )


def websocket_table_data_to_table_state(table_data: TableStateData) -> TableState:
    """Convert WebSocket TableStateData to core TableState."""
    return TableState(
        width=table_data.width,
        height=table_data.height,
        pocket_positions=[list_to_vector2d(pos) for pos in table_data.pocket_positions],
        pocket_radius=table_data.pocket_radius,
    )


def api_game_event_to_game_event(api_event: APIGameEvent) -> GameEvent:
    """Convert API GameEvent to core GameEvent."""
    return GameEvent(
        timestamp=api_event.timestamp.timestamp(),
        event_type=api_event.event_type,
        description=api_event.description,
        data=api_event.data,
        frame_number=0,  # Would need to be provided from context
    )


# =============================================================================
# Batch Conversion Functions
# =============================================================================


def convert_ball_states_to_api(balls: list[BallState]) -> list[BallInfo]:
    """Convert a list of core BallState objects to API BallInfo objects."""
    return [ball_state_to_ball_info(ball) for ball in balls]


def convert_ball_states_to_websocket(balls: list[BallState]) -> list[BallStateData]:
    """Convert a list of core BallState objects to WebSocket BallStateData objects."""
    return [ball_state_to_websocket_data(ball) for ball in balls]


def convert_api_balls_to_core(ball_infos: list[BallInfo]) -> list[BallState]:
    """Convert a list of API BallInfo objects to core BallState objects."""
    return [ball_info_to_ball_state(ball_info) for ball_info in ball_infos]


def convert_websocket_balls_to_core(
    ball_data_list: list[BallStateData],
) -> list[BallState]:
    """Convert a list of WebSocket BallStateData objects to core BallState objects."""
    return [
        websocket_ball_data_to_ball_state(ball_data) for ball_data in ball_data_list
    ]


def convert_trajectories_to_api(trajectories: list[Trajectory]) -> list[TrajectoryInfo]:
    """Convert a list of core Trajectory objects to API TrajectoryInfo objects."""
    return [trajectory_to_api_info(trajectory) for trajectory in trajectories]


def convert_trajectories_to_websocket(
    trajectories: list[Trajectory],
) -> list[TrajectoryData]:
    """Convert a list of core Trajectory objects to WebSocket TrajectoryData objects."""
    return [trajectory_to_websocket_data(trajectory) for trajectory in trajectories]


# =============================================================================
# Validation and Error Handling
# =============================================================================


def validate_coordinate_conversion(
    coords: list[float], field_name: str = "coordinates"
) -> None:
    """Validate coordinates before conversion."""
    if len(coords) != 2:
        raise ValueError(f"{field_name} must have exactly 2 values [x, y]")

    if not all(isinstance(c, (int, float)) for c in coords):
        raise ValueError(f"{field_name} must contain numeric values")

    # Check for reasonable bounds (adjust as needed)
    if not all(-1000 <= c <= 1000 for c in coords):
        raise ValueError(f"{field_name} values out of reasonable bounds")


def validate_ball_state_conversion(ball_data: dict[str, Any]) -> ValidationResult:
    """Validate ball state data before conversion."""
    result = ValidationResult(is_valid=True)

    # Check required fields
    required_fields = ["id", "position", "velocity"]
    for field in required_fields:
        if field not in ball_data:
            result.add_error(f"Missing required field: {field}")

    # Validate position and velocity
    for field in ["position", "velocity"]:
        if field in ball_data:
            try:
                validate_coordinate_conversion(ball_data[field], field)
            except ValueError as e:
                result.add_error(str(e))

    # Validate confidence
    if "confidence" in ball_data:
        confidence = ball_data["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            result.add_error("Confidence must be a number between 0 and 1")

    # Validate radius
    if "radius" in ball_data:
        radius = ball_data["radius"]
        if not isinstance(radius, (int, float)) or radius <= 0:
            result.add_error("Radius must be a positive number")

    return result


def safe_convert_game_state(
    state_data: dict[str, Any], validate: bool = True
) -> Union[GameState, ValidationResult]:
    """Safely convert game state data with validation."""
    if validate:
        # Perform validation first
        validation_result = ValidationResult(is_valid=True)

        # Validate required fields
        required_fields = ["timestamp", "frame_number", "balls", "table"]
        for field in required_fields:
            if field not in state_data:
                validation_result.add_error(f"Missing required field: {field}")

        # Validate balls
        if "balls" in state_data:
            for i, ball_data in enumerate(state_data["balls"]):
                ball_validation = validate_ball_state_conversion(ball_data)
                if not ball_validation.is_valid:
                    for error in ball_validation.errors:
                        validation_result.add_error(f"Ball {i}: {error}")

        if not validation_result.is_valid:
            return validation_result

    try:
        # Convert the data
        balls = [
            websocket_ball_data_to_ball_state(BallStateData.model_validate(ball_data))
            for ball_data in state_data["balls"]
        ]

        table = websocket_table_data_to_table_state(
            TableStateData.model_validate(state_data["table"])
        )

        cue = None
        if "cue" in state_data and state_data["cue"]:
            cue = websocket_cue_data_to_cue_state(
                CueStateData.model_validate(state_data["cue"])
            )

        game_type = GameType(state_data.get("game_type", "practice"))

        return GameState(
            timestamp=state_data["timestamp"],
            frame_number=state_data["frame_number"],
            balls=balls,
            table=table,
            cue=cue,
            game_type=game_type,
            is_valid=state_data.get("is_valid", True),
            state_confidence=state_data.get("confidence", 1.0),
        )

    except Exception as e:
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_error(f"Conversion error: {str(e)}")
        return validation_result


# =============================================================================
# Utility Functions
# =============================================================================


def create_conversion_summary(
    original_count: int, converted_count: int, errors: list[str]
) -> dict[str, Any]:
    """Create a summary of conversion results."""
    return {
        "original_count": original_count,
        "converted_count": converted_count,
        "success_rate": converted_count / original_count if original_count > 0 else 0,
        "error_count": len(errors),
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }


def estimate_conversion_time(item_count: int, base_time_ms: float = 0.1) -> float:
    """Estimate conversion time for a given number of items."""
    return item_count * base_time_ms


def get_conversion_capabilities() -> dict[str, list[str]]:
    """Get information about available conversion capabilities."""
    return {
        "core_to_api": [
            "BallState -> BallInfo",
            "CueState -> CueInfo",
            "TableState -> TableInfo",
            "GameState -> GameStateResponse",
            "Trajectory -> TrajectoryInfo",
            "ShotAnalysis -> ShotAnalysisResponse",
        ],
        "core_to_websocket": [
            "BallState -> BallStateData",
            "CueState -> CueStateData",
            "TableState -> TableStateData",
            "GameState -> GameStateData",
            "Trajectory -> TrajectoryData",
        ],
        "api_to_core": [
            "BallInfo -> BallState",
            "CueInfo -> CueState",
            "TableInfo -> TableState",
        ],
        "websocket_to_core": [
            "BallStateData -> BallState",
            "CueStateData -> CueState",
            "TableStateData -> TableState",
        ],
        "utility_functions": [
            "Vector2D <-> Coordinate2D",
            "Vector2D <-> List[float]",
            "Batch conversions",
            "Validation functions",
            "Safe conversion with error handling",
        ],
    }
