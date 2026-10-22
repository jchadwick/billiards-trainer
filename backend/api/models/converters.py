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

import math
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Sequence, Union

from ...core.coordinates import Vector2D
from ...core.models import (
    BallState,
    Collision,
    CueState,
    GameEvent,
    GameState,
    GameType,
    ShotAnalysis,
    TableState,
    Trajectory,
)
from ...perception.scene_state_adapter import SceneStateReadModel
from .common import (
    CANONICAL_PIXEL_UNIT,
    METERS_UNIT,
    Coordinate2D,
    PositionWithScale,
    RealWorldDimensions,
    ValidationResult,
)
from .responses import BallInfo, CueInfo
from .responses import GameEvent as APIGameEvent
from .responses import (
    GameSessionMetadataResponse,
    GameStateResponse,
    SceneBallInfo,
    SceneGameStateResponse,
    ShotAnalysisResponse,
    TableInfo,
    TrajectoryInfo,
)
from .websocket import BallStateData
from .websocket import CollisionInfo as WSCollisionInfo
from .websocket import CueStateData, GameStateData, TableStateData, TrajectoryData

# =============================================================================
# Core to API Model Converters
# =============================================================================


def _is_finite_number(value: Any) -> bool:
    """Check numeric input without accepting booleans or non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def vector2d_to_coordinate2d(vector: Vector2D) -> Coordinate2D:
    """Convert a core vector without discarding its coordinate scale."""
    if not _is_finite_number(vector.x) or not _is_finite_number(vector.y):
        raise ValueError("Vector coordinates must be finite numbers")
    return Coordinate2D(
        x=vector.x,
        y=vector.y,
        scale=list(_scale_pair(vector.scale)),
    )


def vector2d_to_dict(vector: Vector2D) -> dict[str, Any]:
    """Convert core Vector2D to dict with mandatory scale.

    Returns:
        Dictionary with x, y, and scale fields:
        {
            "x": float,
            "y": float,
            "scale": [scale_x, scale_y]
        }

    Raises:
        ValueError: If vector does not have scale metadata
    """
    if not _is_finite_number(vector.x) or not _is_finite_number(vector.y):
        raise ValueError("Vector coordinates must be finite numbers")
    if vector.scale is None:
        raise ValueError("Vector must have scale metadata")
    if not isinstance(vector.scale, (tuple, list)):
        raise ValueError(
            "Scale must be a 2-element tuple or list, "
            f"got {type(vector.scale).__name__}"
        )
    if len(vector.scale) != 2:
        raise ValueError("Scale must be a 2-element tuple or list")
    _scale_pair(vector.scale)

    return {
        "x": vector.x,
        "y": vector.y,
        "scale": list(vector.scale),  # Convert tuple to list for JSON serialization
    }


def _scale_pair(scale: Sequence[float]) -> tuple[float, float]:
    """Return validated fixed-size scale metadata for the core coordinate type."""
    if len(scale) != 2:
        raise ValueError("Scale must have exactly 2 values [scale_x, scale_y]")
    if not all(_is_finite_number(value) and value > 0 for value in scale):
        raise ValueError("Scale factors must be positive and finite")
    return scale[0], scale[1]


def _position_with_scale(vector: Vector2D) -> PositionWithScale:
    """Convert a core vector into the API's validated coordinate model."""
    return PositionWithScale(x=vector.x, y=vector.y, scale=list(vector.scale))


def _position_to_vector(position: PositionWithScale) -> Vector2D:
    """Validate a scale-aware transport point before constructing a core vector."""
    validate_coordinate_conversion(position)
    return Vector2D(
        x=position.x,
        y=position.y,
        scale=_scale_pair(position.scale),
    )


def _table_real_world_dimensions(table: TableState) -> Optional[RealWorldDimensions]:
    """Convert optional Core SI dimensions without conflating them with pixels."""
    if table.real_width_meters is None and table.real_height_meters is None:
        return None
    if table.real_width_meters is None or table.real_height_meters is None:
        raise ValueError("Core table must provide both physical dimensions together")
    return RealWorldDimensions(
        width=table.real_width_meters,
        height=table.real_height_meters,
        unit=METERS_UNIT,
    )


def _ball_transport_fields(ball: BallState) -> dict[str, Any]:
    """Return ball properties shared by the HTTP and WebSocket contracts.

    The API and WebSocket schemas intentionally remain separate because they
    carry different transport-specific fields (for example, ``last_update``
    versus ``radius``). Keeping the common mapping here prevents identity,
    confidence, pocket state, or coordinate metadata from drifting between them.
    """
    return {
        # Detection IDs are often numeric at runtime, while both transport
        # schemas define IDs as strings. Normalize once for HTTP and WebSocket.
        "id": str(ball.id),
        "number": ball.number,
        "position": vector2d_to_dict(ball.position),
        "velocity": vector2d_to_dict(ball.velocity),
        "is_cue_ball": ball.is_cue_ball,
        "is_pocketed": ball.is_pocketed,
        "is_observed": ball.is_observed,
        "confidence": ball.confidence,
    }


def ball_state_to_ball_info(ball: BallState) -> BallInfo:
    """Convert core BallState to API BallInfo.

    Note: This now returns position and velocity as dicts with mandatory scale metadata.
    """
    return BallInfo(
        **_ball_transport_fields(ball),
        last_update=datetime.fromtimestamp(ball.last_update),
    )


def ball_state_to_websocket_data(ball: BallState) -> BallStateData:
    """Convert core BallState to WebSocket BallStateData.

    Note: This now returns position and velocity as dicts with mandatory scale metadata.
    """
    return BallStateData(
        **_ball_transport_fields(ball),
        radius=ball.radius,
        radius_scale=list(_scale_pair(ball.radius_scale)),
        radius_unit=CANONICAL_PIXEL_UNIT,
    )


def cue_state_to_cue_info(cue: CueState) -> CueInfo:
    """Convert core CueState to API CueInfo.

    Note: This now returns tip_position as dict with mandatory scale metadata.
    """
    return CueInfo(
        tip_position=_position_with_scale(cue.tip_position),
        butt_position=(
            _position_with_scale(cue.butt_position)
            if cue.butt_position is not None
            else None
        ),
        angle=cue.angle,
        elevation=cue.elevation,
        estimated_force=cue.estimated_force,
        is_visible=cue.is_visible,
        confidence=cue.confidence,
    )


def cue_state_to_websocket_data(cue: CueState) -> CueStateData:
    """Convert core CueState to WebSocket CueStateData.

    Note: This now returns tip_position as dict with mandatory scale metadata.
    """
    return CueStateData(
        tip_position=_position_with_scale(cue.tip_position),
        butt_position=(
            _position_with_scale(cue.butt_position)
            if cue.butt_position is not None
            else None
        ),
        angle=cue.angle,
        elevation=cue.elevation,
        estimated_force=cue.estimated_force,
        is_visible=cue.is_visible,
        confidence=cue.confidence,
    )


def table_state_to_table_info(table: TableState) -> TableInfo:
    """Convert core TableState to API TableInfo.

    Note: This now returns pocket_positions as dicts with mandatory scale metadata.
    """
    return TableInfo(
        width=table.width,
        height=table.height,
        dimensions_scale=list(_scale_pair(table.dimensions_scale)),
        dimensions_unit=CANONICAL_PIXEL_UNIT,
        pocket_positions=[_position_with_scale(pos) for pos in table.pocket_positions],
        playing_area_corners=(
            [_position_with_scale(point) for point in table.playing_area_corners]
            if table.playing_area_corners is not None
            else None
        ),
        pocket_radius=table.pocket_radius,
        pocket_radius_scale=list(_scale_pair(table.pocket_radius_scale)),
        pocket_radius_unit=CANONICAL_PIXEL_UNIT,
        real_world_dimensions=_table_real_world_dimensions(table),
        surface_friction=table.surface_friction,
    )


def table_state_to_websocket_data(table: TableState) -> TableStateData:
    """Convert core TableState to WebSocket TableStateData.

    Note: This now returns pocket_positions as dicts with mandatory scale metadata.
    """
    return TableStateData(
        width=table.width,
        height=table.height,
        dimensions_scale=list(_scale_pair(table.dimensions_scale)),
        dimensions_unit=CANONICAL_PIXEL_UNIT,
        pocket_positions=[_position_with_scale(pos) for pos in table.pocket_positions],
        playing_area_corners=(
            [_position_with_scale(point) for point in table.playing_area_corners]
            if table.playing_area_corners is not None
            else None
        ),
        pocket_radius=table.pocket_radius,
        pocket_radius_scale=list(_scale_pair(table.pocket_radius_scale)),
        pocket_radius_unit=CANONICAL_PIXEL_UNIT,
        real_world_dimensions=_table_real_world_dimensions(table),
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


def scene_state_to_response(
    read_model: SceneStateReadModel,
    core_state: GameState | None,
    *,
    include_events: bool = True,
    session_metadata: dict[str, Any] | None = None,
) -> SceneGameStateResponse:
    """Project fresh scene locations independently of optional Core metadata.

    The scene read model owns location and provenance. Core may contribute game
    type and events, but its table is exposed only when both systems provide an
    equal calibration revision. Unknown alignment leaves table geometry absent.
    Core ball positions, cue state and prediction values never overwrite the
    localization read model.
    """
    payload = read_model.payload
    balls = [
        SceneBallInfo(
            id=ball["id"],
            number=ball.get("number"),
            position=ball["position"],
            velocity=None,
            is_cue_ball=None,
            is_pocketed=None,
            is_observed=bool(ball["is_observed"]),
            confidence=float(ball["confidence"]),
            last_update=read_model.ball_localization_received_at[ball["id"]],
            velocity_status="unavailable",
            cue_ball_status="unavailable",
            pocket_status="unavailable",
            confidence_kind=ball["confidence_kind"],
            last_observation_confidence=ball.get("last_observation_confidence"),
            identity_status=ball["identity_status"],
            identity_confidence=ball.get("identity_confidence"),
            identity_evidence=ball["identity_evidence"],
            ball_type=ball["type"],
            tracking_status=ball["tracking_status"],
            position_kind=ball["position_kind"],
            table_position=ball.get("table_position"),
            source_observation=ball["source_observation"],
            radius=ball["radius"],
            radius_unit=ball["radius_unit"],
            radius_source=ball["radius_source"],
        )
        for ball in payload.balls
    ]
    status = payload.observation_status
    transport_sequence = status.get("transport_sequence")
    if transport_sequence is None and status.get("phase") == "rust_scene_health":
        frame_number = None
    elif isinstance(transport_sequence, bool) or not isinstance(
        transport_sequence, int
    ):
        raise ValueError("scene transport_sequence must be an integer")
    else:
        frame_number = transport_sequence

    confidence_values = [
        ball.confidence
        for ball in balls
        if ball.confidence_kind == "detector_observation"
    ]
    scene_revision = _optional_revision(status.get("calibration_revision"))
    core_table = getattr(core_state, "table", None)
    table_revision = (
        _optional_revision(getattr(core_table, "calibration_revision", None))
        if core_table is not None
        else None
    )
    table_metadata_status: Literal["matched", "unverified", "mismatch", "unavailable"]
    if core_table is None:
        table_metadata_status = "unavailable"
        table = None
    elif scene_revision is None or table_revision is None:
        table_metadata_status = "unverified"
        table = None
    elif scene_revision != table_revision:
        table_metadata_status = "mismatch"
        table = None
    else:
        table_metadata_status = "matched"
        table = table_state_to_table_info(core_table)

    return SceneGameStateResponse(
        timestamp=read_model.updated_at,
        frame_number=frame_number,
        balls=balls,
        cue=None,
        table=table,
        game_type=(
            session_metadata["game_type"]
            if session_metadata is not None
            else core_state.game_type.value if core_state is not None else None
        ),
        session=(
            GameSessionMetadataResponse.model_validate(session_metadata)
            if session_metadata is not None
            else None
        ),
        is_valid=(
            status.get("phase") == "rust_scene_observation"
            and status.get("ball_detection_status") == "observed"
            and status.get("ball_positions_status") == "observed"
        ),
        confidence=(
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        ),
        confidence_status=(
            "mean_detector_confidence" if confidence_values else "unavailable"
        ),
        events=(
            [game_event_to_api_event(event) for event in core_state.events]
            if core_state is not None and include_events
            else []
        ),
        core_metadata_status=("available" if core_state is not None else "unavailable"),
        core_metadata_timestamp=(
            datetime.fromtimestamp(core_state.timestamp, tz=timezone.utc)
            if core_state is not None
            else None
        ),
        table_metadata_status=table_metadata_status,
        scene_calibration_revision=scene_revision,
        table_calibration_revision=table_revision,
        observation_status=status,
        perception_source="rust_scene",
    )


def _optional_revision(value: Any) -> int | None:
    """Accept only the integer calibration counters defined by scene v1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


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
    """Convert core Collision to WebSocket CollisionInfo.

    Note: This now returns position as dict with mandatory scale metadata.
    """
    return WSCollisionInfo(
        position=_position_with_scale(collision.position),
        time=collision.time,
        type=collision.type,
        ball1_id=collision.ball1_id,
        ball2_id=collision.ball2_id,
        impact_angle=0.0,  # Calculate from collision data if needed
        confidence=collision.confidence,
    )


def trajectory_to_websocket_data(trajectory: Trajectory) -> TrajectoryData:
    """Convert core Trajectory to WebSocket TrajectoryData.

    Note: This now returns trajectory points as dicts with mandatory scale metadata.
    """
    from .websocket import TrajectoryPoint

    # Convert trajectory points
    points = []
    time_per_point = (
        trajectory.time_to_rest / len(trajectory.points) if trajectory.points else 0
    )
    for i, point in enumerate(trajectory.points):
        points.append(
            TrajectoryPoint(
                position=_position_with_scale(point),
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
    """Convert core Trajectory to API TrajectoryInfo.

    Note: This now returns trajectory points as dicts with mandatory scale metadata.
    """
    return TrajectoryInfo(
        ball_id=trajectory.ball_id,
        points=[_position_with_scale(point) for point in trajectory.points],
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
    """Convert a scale-aware API coordinate into a core vector."""
    if not _is_finite_number(coord.x) or not _is_finite_number(coord.y):
        raise ValueError("Coordinate values must be finite numbers")
    return Vector2D(
        x=coord.x,
        y=coord.y,
        scale=_scale_pair(coord.scale),
    )


def ball_info_to_ball_state(ball_info: BallInfo) -> BallState:
    """Convert API BallInfo to core BallState.

    Note: BallInfo.position and BallInfo.velocity are now PositionWithScale objects.
    """
    return BallState(
        id=ball_info.id,
        position=_position_to_vector(ball_info.position),
        velocity=_position_to_vector(ball_info.velocity),
        is_cue_ball=ball_info.is_cue_ball,
        is_pocketed=ball_info.is_pocketed,
        is_observed=ball_info.is_observed,
        number=ball_info.number,
        confidence=ball_info.confidence,
        last_update=ball_info.last_update.timestamp(),
    )


def websocket_ball_data_to_ball_state(ball_data: BallStateData) -> BallState:
    """Convert WebSocket BallStateData to core BallState.

    Note: BallStateData.position and BallStateData.velocity are now PositionWithScale objects.
    """
    return BallState(
        id=ball_data.id,
        position=_position_to_vector(ball_data.position),
        velocity=_position_to_vector(ball_data.velocity),
        radius=ball_data.radius,
        radius_scale=_scale_pair(ball_data.radius_scale),
        is_cue_ball=ball_data.is_cue_ball,
        is_pocketed=ball_data.is_pocketed,
        is_observed=ball_data.is_observed,
        number=ball_data.number,
        confidence=ball_data.confidence,
        last_update=datetime.now().timestamp(),
    )


def cue_info_to_cue_state(cue_info: CueInfo) -> CueState:
    """Convert API CueInfo to core CueState.

    Note: CueInfo.tip_position is now a PositionWithScale object.
    """
    return CueState(
        tip_position=_position_to_vector(cue_info.tip_position),
        butt_position=(
            _position_to_vector(cue_info.butt_position)
            if cue_info.butt_position is not None
            else None
        ),
        angle=cue_info.angle,
        elevation=cue_info.elevation,
        estimated_force=cue_info.estimated_force,
        is_visible=cue_info.is_visible,
        confidence=cue_info.confidence,
        last_update=datetime.now().timestamp(),
    )


def websocket_cue_data_to_cue_state(cue_data: CueStateData) -> CueState:
    """Convert WebSocket CueStateData to core CueState.

    Note: CueStateData.tip_position is now a PositionWithScale object.
    """
    return CueState(
        tip_position=_position_to_vector(cue_data.tip_position),
        butt_position=(
            _position_to_vector(cue_data.butt_position)
            if cue_data.butt_position is not None
            else None
        ),
        angle=cue_data.angle,
        elevation=cue_data.elevation,
        estimated_force=cue_data.estimated_force,
        is_visible=cue_data.is_visible,
        confidence=cue_data.confidence,
        last_update=datetime.now().timestamp(),
    )


def table_info_to_table_state(table_info: TableInfo) -> TableState:
    """Convert API TableInfo to core TableState.

    Note: TableInfo.pocket_positions is now a list of PositionWithScale objects.
    """
    return TableState(
        width=table_info.width,
        height=table_info.height,
        dimensions_scale=_scale_pair(table_info.dimensions_scale),
        pocket_positions=[
            _position_to_vector(pos) for pos in table_info.pocket_positions
        ],
        playing_area_corners=(
            [_position_to_vector(point) for point in table_info.playing_area_corners]
            if table_info.playing_area_corners is not None
            else None
        ),
        pocket_radius=table_info.pocket_radius,
        pocket_radius_scale=_scale_pair(table_info.pocket_radius_scale),
        real_width_meters=(
            table_info.real_world_dimensions.width
            if table_info.real_world_dimensions is not None
            else None
        ),
        real_height_meters=(
            table_info.real_world_dimensions.height
            if table_info.real_world_dimensions is not None
            else None
        ),
        surface_friction=table_info.surface_friction,
    )


def websocket_table_data_to_table_state(table_data: TableStateData) -> TableState:
    """Convert WebSocket TableStateData to core TableState.

    Note: TableStateData.pocket_positions is now a list of PositionWithScale objects.
    """
    return TableState(
        width=table_data.width,
        height=table_data.height,
        dimensions_scale=_scale_pair(table_data.dimensions_scale),
        pocket_positions=[
            _position_to_vector(pos) for pos in table_data.pocket_positions
        ],
        playing_area_corners=(
            [_position_to_vector(point) for point in table_data.playing_area_corners]
            if table_data.playing_area_corners is not None
            else None
        ),
        pocket_radius=table_data.pocket_radius,
        pocket_radius_scale=_scale_pair(table_data.pocket_radius_scale),
        real_width_meters=(
            table_data.real_world_dimensions.width
            if table_data.real_world_dimensions is not None
            else None
        ),
        real_height_meters=(
            table_data.real_world_dimensions.height
            if table_data.real_world_dimensions is not None
            else None
        ),
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
    coords: Any, field_name: str = "coordinates"
) -> None:
    """Validate a scale-aware transport point without inferring its coordinate space.

    A bare ``[x, y]`` pair is ambiguous: it does not identify the source coordinate
    system or how its values map to canonical pixels. All API transport points must
    therefore include explicit positive scale metadata.
    """
    if isinstance(coords, PositionWithScale):
        point: dict[str, Any] = {
            "x": coords.x,
            "y": coords.y,
            "scale": coords.scale,
        }
    elif isinstance(coords, dict):
        point = coords
    else:
        raise ValueError(f"{field_name} must include x, y, and scale")

    if not all(key in point for key in ("x", "y", "scale")):
        raise ValueError(f"{field_name} must include x, y, and scale")
    values = (point["x"], point["y"])
    if not all(_is_finite_number(value) for value in values):
        raise ValueError(f"{field_name} must contain finite numeric x and y values")

    scale = point["scale"]
    if not isinstance(scale, (list, tuple)) or len(scale) != 2:
        raise ValueError(f"{field_name}.scale must have exactly 2 values")
    if not all(_is_finite_number(value) and value > 0 for value in scale):
        raise ValueError(f"{field_name}.scale values must be finite and positive")


def validate_ball_state_conversion(ball_data: dict[str, Any]) -> ValidationResult:
    """Validate ball state data before conversion."""
    result = ValidationResult(is_valid=True)

    # Check required fields
    required_fields = ["id", "position", "velocity"]
    for field in required_fields:
        if field not in ball_data:
            result.add_error(f"Missing required field: {field}", field)

    # Validate position and velocity
    for field in ["position", "velocity"]:
        if field in ball_data:
            try:
                validate_coordinate_conversion(ball_data[field], field)
            except ValueError as e:
                result.add_error(str(e), field)

    # Validate confidence
    if "confidence" in ball_data:
        confidence = ball_data["confidence"]
        if not _is_finite_number(confidence) or not 0 <= confidence <= 1:
            result.add_error(
                "Confidence must be a number between 0 and 1", "confidence"
            )

    # Validate radius
    if "radius" in ball_data:
        radius = ball_data["radius"]
        if not _is_finite_number(radius) or radius <= 0:
            result.add_error("Radius must be a positive number", "radius")

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
                validation_result.add_error(f"Missing required field: {field}", field)

        # Validate balls
        if "balls" in state_data:
            balls_data = state_data["balls"]
            if not isinstance(balls_data, list):
                validation_result.add_error("Balls must be a list", "balls")
            else:
                for i, ball_data in enumerate(balls_data):
                    if not isinstance(ball_data, dict):
                        validation_result.add_error(
                            f"Ball {i} must be an object", f"balls[{i}]"
                        )
                        continue

                    ball_validation = validate_ball_state_conversion(ball_data)
                    for field, errors in ball_validation.field_errors.items():
                        for error in errors:
                            validation_result.add_error(
                                f"Ball {i}: {error}", f"balls[{i}].{field}"
                            )

        if "table" in state_data:
            table_data = state_data["table"]
            if not isinstance(table_data, dict):
                validation_result.add_error("Table must be an object", "table")
            elif "pocket_positions" not in table_data:
                validation_result.add_error(
                    "Missing required field: pocket_positions",
                    "table.pocket_positions",
                )
            elif not isinstance(table_data["pocket_positions"], list):
                validation_result.add_error(
                    "Table pocket_positions must be a list", "table.pocket_positions"
                )
            else:
                for index, pocket_position in enumerate(table_data["pocket_positions"]):
                    field_path = f"table.pocket_positions[{index}]"
                    try:
                        validate_coordinate_conversion(pocket_position, field_path)
                    except ValueError as error:
                        validation_result.add_error(str(error), field_path)

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
