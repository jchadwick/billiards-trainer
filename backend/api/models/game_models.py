"""Game-related API models for data transformation."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class GameTypeEnum(str, Enum):
    """Game types."""

    PRACTICE = "practice"
    EIGHT_BALL = "8-ball"
    NINE_BALL = "9-ball"
    STRAIGHT_POOL = "straight_pool"


class ShotTypeEnum(str, Enum):
    """Shot types."""

    STRAIGHT = "straight"
    BANK = "bank"
    COMBINATION = "combination"
    CAROM = "carom"
    BREAK = "break"
    SAFETY = "safety"


class Vector2DModel(BaseModel):
    """2D vector model."""

    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return (self.x**2 + self.y**2) ** 0.5


class BallModel(BaseModel):
    """Ball model for API responses."""

    id: str = Field(..., description="Unique ball identifier")
    number: Optional[int] = Field(None, description="Ball number (1-15, 0 for cue)")
    position: Vector2DModel = Field(..., description="Ball position on table")
    velocity: Vector2DModel = Field(
        default=Vector2DModel(x=0, y=0), description="Ball velocity"
    )
    radius: float = Field(default=0.028575, description="Ball radius in meters")
    mass: float = Field(default=0.17, description="Ball mass in kg")
    is_cue_ball: bool = Field(default=False, description="Whether this is the cue ball")
    is_pocketed: bool = Field(default=False, description="Whether ball is pocketed")
    color: Optional[str] = Field(None, description="Ball color")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence"
    )


class PocketModel(BaseModel):
    """Pocket model."""

    id: str = Field(..., description="Pocket identifier")
    position: Vector2DModel = Field(..., description="Pocket center position")
    radius: float = Field(default=0.06, description="Pocket radius")
    type: str = Field(default="corner", description="Pocket type (corner/side)")


class TableModel(BaseModel):
    """Table model for API responses."""

    width: float = Field(..., description="Table width in meters")
    height: float = Field(..., description="Table height in meters")
    corners: list[Vector2DModel] = Field(..., description="Table corner coordinates")
    pockets: list[PocketModel] = Field(default=[], description="Table pockets")
    surface_color: Optional[list[int]] = Field(
        None, description="Table surface color (RGB)"
    )
    rails: Optional[list[Vector2DModel]] = Field(None, description="Rail boundaries")


class CueModel(BaseModel):
    """Cue stick model."""

    tip_position: Vector2DModel = Field(..., description="Cue tip position")
    angle: float = Field(..., description="Cue angle in radians")
    length: float = Field(default=1.47, description="Cue length in meters")
    is_aiming: bool = Field(default=False, description="Whether player is aiming")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence"
    )


class GameStateModel(BaseModel):
    """Complete game state model."""

    timestamp: float = Field(..., description="State timestamp")
    frame_number: Optional[int] = Field(None, description="Frame number")
    game_type: GameTypeEnum = Field(
        default=GameTypeEnum.PRACTICE, description="Type of game"
    )
    balls: list[BallModel] = Field(default=[], description="All balls on table")
    table: TableModel = Field(..., description="Table information")
    cue: Optional[CueModel] = Field(None, description="Cue stick information")
    current_player: Optional[str] = Field(None, description="Current player identifier")
    turn_number: int = Field(default=1, description="Current turn number")
    is_game_active: bool = Field(
        default=True, description="Whether game is in progress"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default={}, description="Additional metadata"
    )


class TrajectoryPointModel(BaseModel):
    """Single point in a trajectory."""

    position: Vector2DModel = Field(..., description="Position at this point")
    time: float = Field(..., description="Time offset from start")
    velocity: Vector2DModel = Field(..., description="Velocity at this point")


class TrajectoryModel(BaseModel):
    """Ball trajectory model."""

    ball_id: str = Field(..., description="Ball being tracked")
    points: list[TrajectoryPointModel] = Field(..., description="Trajectory points")
    total_time: float = Field(..., description="Total trajectory time")
    end_position: Vector2DModel = Field(..., description="Final resting position")
    collisions: list[dict[str, Any]] = Field(
        default=[], description="Predicted collisions"
    )


class ShotAnalysisModel(BaseModel):
    """Shot analysis results."""

    shot_type: ShotTypeEnum = Field(..., description="Type of shot")
    target_ball: Optional[str] = Field(None, description="Target ball ID")
    recommended_angle: float = Field(..., description="Recommended cue angle")
    recommended_force: float = Field(
        ..., ge=0.0, le=1.0, description="Recommended shot force"
    )
    success_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of success"
    )
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Shot difficulty")
    predicted_trajectory: Optional[TrajectoryModel] = Field(
        None, description="Predicted ball path"
    )
    alternative_shots: list["ShotAnalysisModel"] = Field(
        default=[], description="Alternative shot options"
    )
    warnings: list[str] = Field(default=[], description="Potential issues or warnings")
    tips: list[str] = Field(default=[], description="Helpful tips for the shot")


# Request models


class GameUpdateRequest(BaseModel):
    """Request to update game state."""

    detection_data: dict[str, Any] = Field(..., description="Raw detection data")
    frame_number: Optional[int] = Field(None, description="Frame number")
    timestamp: Optional[float] = Field(None, description="Custom timestamp")


class ShotAnalysisRequest(BaseModel):
    """Request for shot analysis."""

    target_ball: Optional[str] = Field(
        None, description="Target ball ID (auto-detect if not provided)"
    )
    include_alternatives: bool = Field(
        default=True, description="Include alternative shot suggestions"
    )
    max_alternatives: int = Field(
        default=3, ge=0, le=10, description="Maximum number of alternatives"
    )
    difficulty_filter: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Maximum difficulty level"
    )


class TrajectoryRequest(BaseModel):
    """Request for trajectory calculation."""

    ball_id: str = Field(..., description="Ball ID to calculate trajectory for")
    initial_velocity: Vector2DModel = Field(..., description="Initial velocity vector")
    time_limit: Optional[float] = Field(
        None, gt=0.0, description="Maximum simulation time"
    )
    include_collisions: bool = Field(
        default=True, description="Include collision predictions"
    )


class ShotSuggestionsRequest(BaseModel):
    """Request for shot suggestions."""

    difficulty_filter: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Maximum difficulty"
    )
    shot_type_filter: Optional[ShotTypeEnum] = Field(
        None, description="Filter by shot type"
    )
    max_suggestions: int = Field(
        default=3, ge=1, le=10, description="Maximum suggestions"
    )


class GameResetRequest(BaseModel):
    """Request to reset game."""

    game_type: GameTypeEnum = Field(
        default=GameTypeEnum.PRACTICE, description="Type of game to start"
    )
    preserve_settings: bool = Field(
        default=True, description="Preserve current settings"
    )


class OutcomePredictionRequest(BaseModel):
    """Request for outcome prediction."""

    shot_velocity: Vector2DModel = Field(
        ..., description="Velocity to apply to cue ball"
    )
    num_predictions: int = Field(
        default=5, ge=1, le=20, description="Number of prediction scenarios"
    )
    include_probabilities: bool = Field(
        default=True, description="Include probability estimates"
    )


# Response models


class GameStateResponse(BaseModel):
    """Game state response."""

    success: bool = Field(default=True, description="Operation success")
    game_state: GameStateModel = Field(..., description="Current game state")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ShotAnalysisResponse(BaseModel):
    """Shot analysis response."""

    success: bool = Field(default=True, description="Operation success")
    analysis: ShotAnalysisModel = Field(..., description="Shot analysis results")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class TrajectoryResponse(BaseModel):
    """Trajectory calculation response."""

    success: bool = Field(default=True, description="Operation success")
    trajectory: TrajectoryModel = Field(..., description="Calculated trajectory")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class ShotSuggestionsResponse(BaseModel):
    """Shot suggestions response."""

    success: bool = Field(default=True, description="Operation success")
    suggestions: list[ShotAnalysisModel] = Field(..., description="Shot suggestions")
    total_suggestions: int = Field(..., description="Total number of suggestions")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class OutcomePredictionResponse(BaseModel):
    """Outcome prediction response."""

    success: bool = Field(default=True, description="Operation success")
    predictions: list[dict[str, Any]] = Field(..., description="Predicted outcomes")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )


class GameValidationResponse(BaseModel):
    """Game state validation response."""

    success: bool = Field(default=True, description="Operation success")
    valid: bool = Field(..., description="Whether game state is valid")
    issues: list[str] = Field(default=[], description="Validation issues found")
    timestamp: float = Field(..., description="Validation timestamp")


# Validators


@validator("shot_type", pre=True)
def validate_shot_type(cls, v):
    """Validate shot type."""
    if isinstance(v, str):
        try:
            return ShotTypeEnum(v.lower())
        except ValueError:
            return ShotTypeEnum.STRAIGHT
    return v


@validator("game_type", pre=True)
def validate_game_type(cls, v):
    """Validate game type."""
    if isinstance(v, str):
        try:
            return GameTypeEnum(v.lower())
        except ValueError:
            return GameTypeEnum.PRACTICE
    return v


# Update forward references
ShotAnalysisModel.update_forward_refs()


__all__ = [
    # Enums
    "GameTypeEnum",
    "ShotTypeEnum",
    # Data models
    "Vector2DModel",
    "BallModel",
    "PocketModel",
    "TableModel",
    "CueModel",
    "GameStateModel",
    "TrajectoryPointModel",
    "TrajectoryModel",
    "ShotAnalysisModel",
    # Request models
    "GameUpdateRequest",
    "ShotAnalysisRequest",
    "TrajectoryRequest",
    "ShotSuggestionsRequest",
    "GameResetRequest",
    "OutcomePredictionRequest",
    # Response models
    "GameStateResponse",
    "ShotAnalysisResponse",
    "TrajectoryResponse",
    "ShotSuggestionsResponse",
    "OutcomePredictionResponse",
    "GameValidationResponse",
]
