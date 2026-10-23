"""Comprehensive Unit Tests for API Model Converters.

This test suite provides comprehensive coverage of all conversion functions in
api.models.converters, including:

1. Core to API conversions
   - vector2d_to_dict, vector2d_to_coordinate2d
   - ball_state_to_ball_info, ball_state_to_websocket_data
   - cue_state_to_cue_info, cue_state_to_websocket_data
   - table_state_to_table_info, table_state_to_websocket_data
   - game_state_to_response, game_state_to_websocket_data
   - trajectory_to_api_info, trajectory_to_websocket_data
   - shot_analysis_to_response, game_event_to_api_event

2. API to Core conversions
   - coordinate2d_to_vector2d
   - ball_info_to_ball_state, websocket_ball_data_to_ball_state
   - cue_info_to_cue_state, websocket_cue_data_to_cue_state
   - table_info_to_table_state, websocket_table_data_to_table_state
   - api_game_event_to_game_event

3. Validation and error handling
   - validate_coordinate_conversion
   - validate_ball_state_conversion
   - safe_convert_game_state

4. Batch conversions
   - convert_ball_states_to_api/websocket
   - convert_api/websocket_balls_to_core
   - convert_trajectories_to_api/websocket

Test Coverage:
- Scale metadata handling and validation
- Round-trip conversions (Core -> API -> Core)
- Error cases and validation failures
- Batch operations
- Edge cases and boundary conditions
- Type safety and data integrity

Total Tests: 60+
"""

import math
from datetime import datetime

import pytest
from pydantic import ValidationError

# Import API models
from backend.api.models.common import Coordinate2D, PositionWithScale, ValidationResult

# Import converter functions
from backend.api.models.converters import (  # Core to API; API to Core; Batch conversions; Validation
    api_game_event_to_game_event,
    ball_info_to_ball_state,
    ball_state_to_ball_info,
    ball_state_to_websocket_data,
    collision_to_websocket_collision,
    convert_api_balls_to_core,
    convert_ball_states_to_api,
    convert_ball_states_to_websocket,
    convert_trajectories_to_api,
    convert_trajectories_to_websocket,
    convert_websocket_balls_to_core,
    coordinate2d_to_vector2d,
    cue_info_to_cue_state,
    cue_state_to_cue_info,
    cue_state_to_websocket_data,
    game_event_to_api_event,
    game_state_to_response,
    game_state_to_websocket_data,
    safe_convert_game_state,
    shot_analysis_to_response,
    table_info_to_table_state,
    table_state_to_table_info,
    table_state_to_websocket_data,
    trajectory_to_api_info,
    trajectory_to_websocket_data,
    validate_ball_state_conversion,
    validate_coordinate_conversion,
    vector2d_to_coordinate2d,
    vector2d_to_dict,
    websocket_ball_data_to_ball_state,
    websocket_cue_data_to_cue_state,
    websocket_table_data_to_table_state,
)
from backend.api.models.responses import (
    BallInfo,
    CueInfo,
)
from backend.api.models.responses import GameEvent as APIGameEvent
from backend.api.models.responses import (
    GameStateResponse,
    ShotAnalysisResponse,
    TableInfo,
    TrajectoryInfo,
)
from backend.api.models.websocket import (
    BallStateData,
)
from backend.api.models.websocket import CollisionInfo as WSCollisionInfo
from backend.api.models.websocket import (
    CueStateData,
    GameStateData,
    TableStateData,
    TrajectoryData,
)

# Import core models
from backend.core.coordinates import Vector2D
from backend.core.models import (
    BallState,
    Collision,
    CueState,
    GameEvent,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
    Trajectory,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_vector_4k():
    """Create a sample Vector2D in 4K canonical coordinates."""
    return Vector2D.from_4k(1920.0, 1080.0)


@pytest.fixture
def sample_vector_1080p():
    """Create a sample Vector2D in 1080p coordinates."""
    return Vector2D.from_resolution(960.0, 540.0, (1920, 1080))


@pytest.fixture
def sample_ball_state():
    """Create a sample BallState with proper scale metadata."""
    return BallState(
        id="ball_1",
        number=1,
        position=Vector2D.from_4k(1000.0, 800.0),
        velocity=Vector2D.from_4k(100.0, 50.0),
        radius=36.0,
        mass=0.17,
        is_cue_ball=False,
        is_pocketed=False,
        confidence=0.95,
        last_update=datetime(2025, 1, 15, 12, 0, 0).timestamp(),
    )


@pytest.fixture
def sample_cue_ball():
    """Create a sample cue ball."""
    return BallState(
        id="cue",
        number=0,
        position=Vector2D.from_4k(1920.0, 1080.0),
        velocity=Vector2D.from_4k(0.0, 0.0),
        radius=36.0,
        mass=0.17,
        is_cue_ball=True,
        is_pocketed=False,
        confidence=0.99,
    )


@pytest.fixture
def sample_cue_state():
    """Create a sample CueState with proper scale metadata."""
    return CueState(
        tip_position=Vector2D.from_4k(1800.0, 1050.0),
        angle=45.0,
        elevation=15.0,
        estimated_force=0.7,
        is_visible=True,
        confidence=0.85,
    )


@pytest.fixture
def sample_table_state():
    """Create a sample TableState with pocket positions."""
    width = 3200.0
    height = 1600.0
    pockets = [
        Vector2D.from_4k(0, 0),
        Vector2D.from_4k(width / 2, 0),
        Vector2D.from_4k(width, 0),
        Vector2D.from_4k(0, height),
        Vector2D.from_4k(width / 2, height),
        Vector2D.from_4k(width, height),
    ]
    return TableState(
        width=width,
        height=height,
        pocket_positions=pockets,
        pocket_radius=72.0,
        cushion_elasticity=0.85,
        surface_friction=0.2,
        real_width_meters=2.54,
        real_height_meters=1.27,
    )


@pytest.fixture
def sample_game_state(
    sample_ball_state, sample_cue_ball, sample_cue_state, sample_table_state
):
    """Create a sample GameState with all components."""
    return GameState(
        timestamp=datetime(2025, 1, 15, 12, 0, 0).timestamp(),
        frame_number=100,
        balls=[sample_cue_ball, sample_ball_state],
        cue=sample_cue_state,
        table=sample_table_state,
        game_type=GameType.PRACTICE,
        is_valid=True,
        state_confidence=0.92,
        events=[],
    )


@pytest.fixture
def sample_trajectory(sample_ball_state):
    """Create a sample Trajectory."""
    points = [
        Vector2D.from_4k(1000.0, 800.0),
        Vector2D.from_4k(1100.0, 850.0),
        Vector2D.from_4k(1200.0, 900.0),
    ]
    collision = Collision(
        position=Vector2D.from_4k(1150.0, 875.0),
        time=0.5,
        type="ball",  # Fixed: must be "ball", "cushion", or "pocket"
        ball1_id="ball_1",
        ball2_id="ball_2",
        confidence=0.9,
    )
    return Trajectory(
        ball_id="ball_1",
        points=points,
        collisions=[collision],
        final_position=Vector2D.from_4k(1200.0, 900.0),  # Required field
        final_velocity=Vector2D.from_4k(0.0, 0.0),  # Required field
        will_be_pocketed=False,
        pocket_id=None,
        time_to_rest=2.5,
        max_velocity=150.0,
        confidence=0.88,
    )


# =============================================================================
# Vector2D Conversion Tests
# =============================================================================


class TestVector2DConversions:
    """Test Vector2D conversion functions."""

    def test_vector2d_to_dict_with_4k_scale(self, sample_vector_4k):
        """Test converting Vector2D to dict with 4K scale."""
        result = vector2d_to_dict(sample_vector_4k)

        assert result["x"] == 1920.0
        assert result["y"] == 1080.0
        assert result["scale"] == [1.0, 1.0]
        assert isinstance(result["scale"], list)

    def test_vector2d_to_dict_with_1080p_scale(self, sample_vector_1080p):
        """Test converting Vector2D to dict with 1080p scale."""
        result = vector2d_to_dict(sample_vector_1080p)

        assert result["x"] == 960.0
        assert result["y"] == 540.0
        assert result["scale"] == [2.0, 2.0]

    def test_vector2d_to_dict_without_scale_raises_error(self):
        """Test that vector without scale raises ValueError."""
        # Create vector without using factory methods (bypassing validation)
        vector = object.__new__(Vector2D)
        object.__setattr__(vector, "x", 100.0)
        object.__setattr__(vector, "y", 200.0)
        object.__setattr__(vector, "scale", None)

        with pytest.raises(ValueError, match="must have scale metadata"):
            vector2d_to_dict(vector)

    def test_vector2d_to_dict_with_invalid_scale_type(self):
        """Test that invalid scale type raises ValueError."""
        vector = object.__new__(Vector2D)
        object.__setattr__(vector, "x", 100.0)
        object.__setattr__(vector, "y", 200.0)
        object.__setattr__(vector, "scale", 1.0)  # Not a tuple/list

        with pytest.raises(ValueError, match="Scale must be a 2-element tuple or list"):
            vector2d_to_dict(vector)

    def test_vector2d_to_dict_with_wrong_scale_length(self):
        """Test that wrong scale length raises ValueError."""
        vector = object.__new__(Vector2D)
        object.__setattr__(vector, "x", 100.0)
        object.__setattr__(vector, "y", 200.0)
        object.__setattr__(vector, "scale", [1.0])  # Only 1 element

        with pytest.raises(ValueError, match="Scale must be a 2-element tuple or list"):
            vector2d_to_dict(vector)

    def test_vector2d_to_dict_with_negative_scale(self):
        """Test that negative scale raises ValueError."""
        vector = object.__new__(Vector2D)
        object.__setattr__(vector, "x", 100.0)
        object.__setattr__(vector, "y", 200.0)
        object.__setattr__(vector, "scale", (-1.0, 2.0))

        with pytest.raises(ValueError, match="Scale factors must be positive"):
            vector2d_to_dict(vector)

    def test_vector2d_to_dict_with_zero_scale(self):
        """Test that zero scale raises ValueError."""
        vector = object.__new__(Vector2D)
        object.__setattr__(vector, "x", 100.0)
        object.__setattr__(vector, "y", 200.0)
        object.__setattr__(vector, "scale", (0.0, 1.0))

        with pytest.raises(ValueError, match="Scale factors must be positive"):
            vector2d_to_dict(vector)

    def test_vector2d_to_coordinate2d(self, sample_vector_4k):
        """Converting to Coordinate2D retains explicit scale metadata."""
        result = vector2d_to_coordinate2d(sample_vector_4k)

        assert isinstance(result, Coordinate2D)
        assert result.x == 1920.0
        assert result.y == 1080.0
        assert result.scale == [1.0, 1.0]

    def test_coordinate2d_to_vector2d(self):
        """API coordinates retain their declared source scale."""
        coord = Coordinate2D(x=500.0, y=300.0, scale=[2.0, 3.0])

        result = coordinate2d_to_vector2d(coord)

        assert result.x == 500.0
        assert result.y == 300.0
        assert result.scale == (2.0, 3.0)

    def test_coordinate2d_round_trip_preserves_noncanonical_scale(self):
        vector = Vector2D(x=500.0, y=300.0, scale=(2.0, 3.0))

        result = coordinate2d_to_vector2d(vector2d_to_coordinate2d(vector))

        assert result == vector


# =============================================================================
# BallState Conversion Tests
# =============================================================================


class TestBallStateConversions:
    """Test BallState conversion functions."""

    def test_ball_state_to_ball_info(self, sample_ball_state):
        """Test converting BallState to BallInfo."""
        result = ball_state_to_ball_info(sample_ball_state)

        assert isinstance(result, BallInfo)
        assert result.id == "ball_1"
        assert result.number == 1
        assert result.is_cue_ball is False
        assert result.is_pocketed is False
        assert result.confidence == 0.95

        # Check position with scale
        assert result.position.x == 1000.0
        assert result.position.y == 800.0
        assert result.position.scale == [1.0, 1.0]

        # Check velocity with scale
        assert result.velocity.x == 100.0
        assert result.velocity.y == 50.0
        assert result.velocity.scale == [1.0, 1.0]

    def test_ball_state_to_ball_info_cue_ball(self, sample_cue_ball):
        """Test converting cue ball BallState to BallInfo."""
        result = ball_state_to_ball_info(sample_cue_ball)

        assert result.id == "cue"
        assert result.number == 0
        assert result.is_cue_ball is True
        assert result.confidence == 0.99

    def test_ball_state_to_websocket_data(self, sample_ball_state):
        """Test converting BallState to WebSocket BallStateData."""
        result = ball_state_to_websocket_data(sample_ball_state)

        assert isinstance(result, BallStateData)
        assert result.id == "ball_1"
        assert result.number == 1
        assert result.radius == 36.0
        assert result.radius_scale == [1.0, 1.0]
        assert result.radius_unit == "canonical_pixels"
        assert result.is_cue_ball is False
        assert result.is_pocketed is False
        assert result.confidence == 0.95

        # Check position with scale
        assert result.position.x == 1000.0
        assert result.position.y == 800.0
        assert result.position.scale == [1.0, 1.0]

    def test_api_and_websocket_ball_fields_stay_equivalent(
        self, sample_ball_state, sample_cue_ball
    ):
        """Transport models share core ball facts while retaining their own fields."""
        sample_cue_ball.is_pocketed = True
        for ball in (sample_ball_state, sample_cue_ball):
            api_data = ball_state_to_ball_info(ball).model_dump(mode="json")
            websocket_data = ball_state_to_websocket_data(ball).model_dump(mode="json")

            for field in (
                "id",
                "number",
                "position",
                "velocity",
                "is_cue_ball",
                "is_pocketed",
                "is_observed",
                "confidence",
            ):
                assert api_data[field] == websocket_data[field]

        assert set(BallInfo.model_fields) == {
            "id",
            "number",
            "position",
            "velocity",
            "is_cue_ball",
            "is_pocketed",
            "is_observed",
            "confidence",
            "last_update",
        }
        assert set(BallStateData.model_fields) == {
            "id",
            "number",
            "position",
            "velocity",
            "radius",
            "radius_scale",
            "radius_unit",
            "is_cue_ball",
            "is_pocketed",
            "is_observed",
            "confidence",
        }

    def test_ball_info_to_ball_state_roundtrip(self, sample_ball_state):
        """Test round-trip conversion BallState -> BallInfo -> BallState."""
        ball_info = ball_state_to_ball_info(sample_ball_state)
        result = ball_info_to_ball_state(ball_info)

        assert result.id == sample_ball_state.id
        assert result.number == sample_ball_state.number
        assert result.position.x == sample_ball_state.position.x
        assert result.position.y == sample_ball_state.position.y
        assert result.position.scale == sample_ball_state.position.scale
        assert result.velocity.x == sample_ball_state.velocity.x
        assert result.velocity.y == sample_ball_state.velocity.y
        assert result.is_cue_ball == sample_ball_state.is_cue_ball
        assert result.is_pocketed == sample_ball_state.is_pocketed
        assert result.confidence == sample_ball_state.confidence

    def test_websocket_ball_data_to_ball_state_roundtrip(self, sample_ball_state):
        """Test round-trip conversion BallState -> WebSocket -> BallState."""
        ws_data = ball_state_to_websocket_data(sample_ball_state)
        result = websocket_ball_data_to_ball_state(ws_data)

        assert result.id == sample_ball_state.id
        assert result.number == sample_ball_state.number
        assert result.radius == sample_ball_state.radius
        assert result.position.x == sample_ball_state.position.x
        assert result.position.y == sample_ball_state.position.y
        assert result.velocity.x == sample_ball_state.velocity.x
        assert result.velocity.y == sample_ball_state.velocity.y
        assert result.is_cue_ball == sample_ball_state.is_cue_ball

    def test_websocket_ball_pixel_metadata_matches_core_serialization(
        self, sample_ball_state
    ):
        core_payload = sample_ball_state.to_dict()
        api_payload = ball_state_to_websocket_data(sample_ball_state).model_dump(
            mode="json"
        )

        assert api_payload["radius"] == core_payload["radius"]
        assert api_payload["radius_scale"] == core_payload["radius_scale"]
        assert api_payload["radius_unit"] == core_payload["radius_unit"]


# =============================================================================
# CueState Conversion Tests
# =============================================================================


class TestCueStateConversions:
    """Test CueState conversion functions."""

    def test_cue_state_to_cue_info(self, sample_cue_state):
        """Test converting CueState to CueInfo."""
        result = cue_state_to_cue_info(sample_cue_state)

        assert isinstance(result, CueInfo)
        assert result.angle == 45.0
        assert result.elevation == 15.0
        assert result.estimated_force == 0.7
        assert result.is_visible is True
        assert result.confidence == 0.85

        # Check tip position with scale
        assert result.tip_position.x == 1800.0
        assert result.tip_position.y == 1050.0
        assert result.tip_position.scale == [1.0, 1.0]

    def test_cue_state_to_websocket_data(self, sample_cue_state):
        """Test converting CueState to WebSocket CueStateData."""
        result = cue_state_to_websocket_data(sample_cue_state)

        assert isinstance(result, CueStateData)
        assert result.angle == 45.0
        assert result.elevation == 15.0
        assert result.estimated_force == 0.7
        assert result.is_visible is True
        assert result.confidence == 0.85

        # Check tip position with scale
        assert result.tip_position.x == 1800.0
        assert result.tip_position.y == 1050.0

    def test_cue_info_to_cue_state_roundtrip(self, sample_cue_state):
        """Test round-trip conversion CueState -> CueInfo -> CueState."""
        cue_info = cue_state_to_cue_info(sample_cue_state)
        result = cue_info_to_cue_state(cue_info)

        assert result.tip_position.x == sample_cue_state.tip_position.x
        assert result.tip_position.y == sample_cue_state.tip_position.y
        assert result.tip_position.scale == sample_cue_state.tip_position.scale
        assert result.angle == sample_cue_state.angle
        assert result.elevation == sample_cue_state.elevation
        assert result.estimated_force == sample_cue_state.estimated_force
        assert result.is_visible == sample_cue_state.is_visible
        assert result.confidence == sample_cue_state.confidence

    def test_websocket_cue_data_to_cue_state_roundtrip(self, sample_cue_state):
        """Test round-trip conversion CueState -> WebSocket -> CueState."""
        ws_data = cue_state_to_websocket_data(sample_cue_state)
        result = websocket_cue_data_to_cue_state(ws_data)

        assert result.tip_position.x == sample_cue_state.tip_position.x
        assert result.tip_position.y == sample_cue_state.tip_position.y
        assert result.angle == sample_cue_state.angle
        assert result.elevation == sample_cue_state.elevation


# =============================================================================
# TableState Conversion Tests
# =============================================================================


class TestTableStateConversions:
    """Test TableState conversion functions."""

    def test_table_state_to_table_info(self, sample_table_state):
        """Test converting TableState to TableInfo."""
        result = table_state_to_table_info(sample_table_state)

        assert isinstance(result, TableInfo)
        assert result.width == 3200.0
        assert result.height == 1600.0
        assert result.pocket_radius == 72.0
        assert result.dimensions_scale == [1.0, 1.0]
        assert result.dimensions_unit == "canonical_pixels"
        assert result.pocket_radius_scale == [1.0, 1.0]
        assert result.pocket_radius_unit == "canonical_pixels"
        assert result.real_world_dimensions.model_dump() == {
            "width": 2.54,
            "height": 1.27,
            "unit": "meters",
        }
        assert result.surface_friction == 0.2
        assert len(result.pocket_positions) == 6

        # Check pocket positions have scale metadata
        for pocket in result.pocket_positions:
            assert hasattr(pocket, "scale")
            assert pocket.scale == [1.0, 1.0]

    def test_table_state_to_websocket_data(self, sample_table_state):
        """Test converting TableState to WebSocket TableStateData."""
        result = table_state_to_websocket_data(sample_table_state)

        assert isinstance(result, TableStateData)
        assert result.width == 3200.0
        assert result.height == 1600.0
        assert result.pocket_radius == 72.0
        assert result.dimensions_scale == [1.0, 1.0]
        assert result.dimensions_unit == "canonical_pixels"
        assert result.pocket_radius_scale == [1.0, 1.0]
        assert result.pocket_radius_unit == "canonical_pixels"
        assert result.real_world_dimensions.model_dump() == {
            "width": 2.54,
            "height": 1.27,
            "unit": "meters",
        }
        assert len(result.pocket_positions) == 6

    @pytest.mark.parametrize("source_scale", [1.0, 2.0])
    def test_playing_area_corners_roundtrip_as_canonical_scaled_points(
        self, sample_table_state, source_scale
    ):
        """Both API boundaries retain corners after Core canonicalizes pixels."""
        canonical_points = [
            (100.0, 80.0),
            (3100.0, 80.0),
            (3100.0, 1520.0),
            (100.0, 1520.0),
        ]
        table = TableState(
            width=sample_table_state.width,
            height=sample_table_state.height,
            pocket_positions=sample_table_state.pocket_positions,
            pocket_radius=sample_table_state.pocket_radius,
            real_width_meters=sample_table_state.real_width_meters,
            real_height_meters=sample_table_state.real_height_meters,
            playing_area_corners=[
                Vector2D(
                    x / source_scale,
                    y / source_scale,
                    scale=(source_scale, source_scale),
                )
                for x, y in canonical_points
            ],
        )

        # TableState consumes source scale once; converter outputs are canonical.
        assert table.playing_area_corners is not None
        for point, (expected_x, expected_y) in zip(
            table.playing_area_corners, canonical_points, strict=True
        ):
            assert (point.x, point.y, point.scale) == (
                expected_x,
                expected_y,
                (1.0, 1.0),
            )

        api = table_state_to_table_info(table)
        websocket = table_state_to_websocket_data(table)
        expected_payload = [
            {"x": x, "y": y, "scale": [1.0, 1.0]} for x, y in canonical_points
        ]
        assert [point.model_dump() for point in api.playing_area_corners] == (
            expected_payload
        )
        assert [point.model_dump() for point in websocket.playing_area_corners] == (
            expected_payload
        )

        for restored in (
            table_info_to_table_state(api),
            websocket_table_data_to_table_state(websocket),
        ):
            assert restored.playing_area_corners is not None
            assert [
                (point.x, point.y, point.scale)
                for point in restored.playing_area_corners
            ] == [(x, y, (1.0, 1.0)) for x, y in canonical_points]

    def test_playing_area_absence_is_preserved(self, sample_table_state):
        """Uncalibrated tables stay uncalibrated through both API formats."""
        assert sample_table_state.playing_area_corners is None

        api = table_state_to_table_info(sample_table_state)
        websocket = table_state_to_websocket_data(sample_table_state)
        assert api.playing_area_corners is None
        assert websocket.playing_area_corners is None
        assert api.model_dump()["playing_area_corners"] is None
        assert websocket.model_dump()["playing_area_corners"] is None
        assert table_info_to_table_state(api).playing_area_corners is None
        assert (
            websocket_table_data_to_table_state(websocket).playing_area_corners is None
        )

        invalid_corners = [{"x": 100.0, "y": 80.0, "scale": [1.0, 1.0]}] * 3
        with pytest.raises(ValidationError, match="exactly four corners"):
            TableInfo.model_validate(
                {**api.model_dump(), "playing_area_corners": invalid_corners}
            )
        with pytest.raises(ValidationError, match="exactly four corners"):
            TableStateData.model_validate(
                {
                    **websocket.model_dump(),
                    "playing_area_corners": invalid_corners,
                }
            )

    def test_table_info_to_table_state_roundtrip(self, sample_table_state):
        """Test round-trip conversion TableState -> TableInfo -> TableState."""
        table_info = table_state_to_table_info(sample_table_state)
        result = table_info_to_table_state(table_info)

        assert result.width == sample_table_state.width
        assert result.height == sample_table_state.height
        assert result.pocket_radius == sample_table_state.pocket_radius
        assert result.real_width_meters == sample_table_state.real_width_meters
        assert result.real_height_meters == sample_table_state.real_height_meters
        assert result.surface_friction == sample_table_state.surface_friction
        assert len(result.pocket_positions) == len(sample_table_state.pocket_positions)

        # Check pocket positions preserved
        for i, pocket in enumerate(result.pocket_positions):
            original = sample_table_state.pocket_positions[i]
            assert pocket.x == original.x
            assert pocket.y == original.y
            assert pocket.scale == original.scale

    def test_websocket_table_data_to_table_state_roundtrip(self, sample_table_state):
        """Test round-trip conversion TableState -> WebSocket -> TableState."""
        ws_data = table_state_to_websocket_data(sample_table_state)
        result = websocket_table_data_to_table_state(ws_data)

        assert result.width == sample_table_state.width
        assert result.height == sample_table_state.height
        assert result.pocket_radius == sample_table_state.pocket_radius
        assert result.real_width_meters == sample_table_state.real_width_meters
        assert result.real_height_meters == sample_table_state.real_height_meters
        assert len(result.pocket_positions) == len(sample_table_state.pocket_positions)

    def test_table_pixel_and_physical_metadata_match_core_serialization(
        self, sample_table_state
    ):
        core_payload = sample_table_state.to_dict()
        for api_payload in (
            table_state_to_table_info(sample_table_state).model_dump(mode="json"),
            table_state_to_websocket_data(sample_table_state).model_dump(mode="json"),
        ):
            for field in (
                "width",
                "height",
                "dimensions_scale",
                "dimensions_unit",
                "pocket_radius",
                "pocket_radius_scale",
                "pocket_radius_unit",
                "real_world_dimensions",
                "playing_area_corners",
            ):
                assert api_payload[field] == core_payload[field]


# =============================================================================
# GameState Conversion Tests
# =============================================================================


class TestGameStateConversions:
    """Test GameState conversion functions."""

    def test_game_state_to_response(self, sample_game_state):
        """Test converting GameState to GameStateResponse."""
        result = game_state_to_response(sample_game_state)

        assert isinstance(result, GameStateResponse)
        assert result.frame_number == 100
        assert result.game_type == "practice"
        assert result.is_valid is True
        assert result.confidence == 0.92
        assert len(result.balls) == 2
        assert result.cue is not None
        assert result.table is not None

    def test_game_state_to_response_no_cue(self, sample_game_state):
        """Test converting GameState without cue to GameStateResponse."""
        sample_game_state.cue = None
        result = game_state_to_response(sample_game_state)

        assert result.cue is None
        assert len(result.balls) == 2

    def test_game_state_to_websocket_data(self, sample_game_state):
        """Test converting GameState to WebSocket GameStateData."""
        result = game_state_to_websocket_data(sample_game_state)

        assert isinstance(result, GameStateData)
        assert result.frame_number == 100
        assert result.game_type == "practice"
        assert result.is_valid is True
        assert result.confidence == 0.92
        assert len(result.balls) == 2
        assert result.cue is not None

    def test_game_event_to_api_event(self):
        """Test converting GameEvent to API GameEvent."""
        event = GameEvent(
            timestamp=datetime(2025, 1, 15, 12, 0, 0).timestamp(),
            frame_number=100,
            event_type="ball_pocketed",
            description="Ball 8 was pocketed",
            data={"ball_id": "ball_8", "pocket_id": 1},
        )

        result = game_event_to_api_event(event)

        assert isinstance(result, APIGameEvent)
        assert result.event_type == "ball_pocketed"
        assert result.description == "Ball 8 was pocketed"
        assert result.data["ball_id"] == "ball_8"

    def test_api_game_event_to_game_event(self):
        """Test converting API GameEvent to core GameEvent."""
        api_event = APIGameEvent(
            timestamp=datetime(2025, 1, 15, 12, 0, 0),
            event_type="shot_detected",
            description="Shot detected",
            data={"force": 0.8},
        )

        result = api_game_event_to_game_event(api_event)

        assert isinstance(result, GameEvent)
        assert result.event_type == "shot_detected"
        assert result.description == "Shot detected"
        assert result.data["force"] == 0.8


# =============================================================================
# Trajectory and Analysis Conversion Tests
# =============================================================================


class TestTrajectoryConversions:
    """Test Trajectory conversion functions."""

    def test_trajectory_to_api_info(self, sample_trajectory):
        """Test converting Trajectory to TrajectoryInfo."""
        result = trajectory_to_api_info(sample_trajectory)

        assert isinstance(result, TrajectoryInfo)
        assert result.ball_id == "ball_1"
        assert result.will_be_pocketed is False
        assert result.pocket_id is None
        assert result.time_to_rest == 2.5
        assert result.max_velocity == 150.0
        assert result.confidence == 0.88
        assert len(result.points) == 3

        # Check points have scale metadata
        for point in result.points:
            assert hasattr(point, "scale")
            assert point.scale == [1.0, 1.0]

    def test_trajectory_to_websocket_data(self, sample_trajectory):
        """Test converting Trajectory to WebSocket TrajectoryData."""
        result = trajectory_to_websocket_data(sample_trajectory)

        assert isinstance(result, TrajectoryData)
        assert result.ball_id == "ball_1"
        assert result.will_be_pocketed is False
        assert result.time_to_rest == 2.5
        assert result.max_velocity == 150.0
        assert len(result.points) == 3
        assert len(result.collisions) == 1

    def test_collision_to_websocket_collision(self):
        """Test converting Collision to WebSocket CollisionInfo."""
        collision = Collision(
            position=Vector2D.from_4k(1500.0, 900.0),
            time=1.2,
            type="cushion",  # Fixed: must be "ball", "cushion", or "pocket"
            ball1_id="ball_5",
            ball2_id=None,
            confidence=0.87,
        )

        result = collision_to_websocket_collision(collision)

        assert isinstance(result, WSCollisionInfo)
        assert result.time == 1.2
        assert result.type == "cushion"
        assert result.ball1_id == "ball_5"
        assert result.confidence == 0.87
        assert result.position.x == 1500.0
        assert result.position.y == 900.0

    def test_shot_analysis_to_response(self):
        """Test converting ShotAnalysis to ShotAnalysisResponse."""
        analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,  # Fixed: use DIRECT instead of STRAIGHT
            difficulty=0.65,
            success_probability=0.75,
            recommended_force=0.6,
            recommended_angle=30.0,
            target_ball_id="ball_3",
            target_pocket_id=2,
            potential_problems=["scratch risk"],
            risk_assessment={"scratch": 0.15},
        )

        result = shot_analysis_to_response(analysis)

        assert isinstance(result, ShotAnalysisResponse)
        assert result.shot_type == "direct"
        assert result.difficulty == 0.65
        assert result.success_probability == 0.75
        assert result.recommended_force == 0.6
        assert result.recommended_angle == 30.0
        assert result.target_ball_id == "ball_3"
        assert result.target_pocket_id == 2
        assert "scratch risk" in result.potential_problems


# =============================================================================
# Batch Conversion Tests
# =============================================================================


class TestBatchConversions:
    """Test batch conversion functions."""

    def test_convert_ball_states_to_api(self, sample_ball_state, sample_cue_ball):
        """Test batch conversion of BallStates to BallInfo."""
        balls = [sample_cue_ball, sample_ball_state]
        result = convert_ball_states_to_api(balls)

        assert len(result) == 2
        assert all(isinstance(ball, BallInfo) for ball in result)
        assert result[0].is_cue_ball is True
        assert result[1].is_cue_ball is False

    def test_convert_ball_states_to_websocket(self, sample_ball_state, sample_cue_ball):
        """Test batch conversion of BallStates to WebSocket data."""
        balls = [sample_cue_ball, sample_ball_state]
        result = convert_ball_states_to_websocket(balls)

        assert len(result) == 2
        assert all(isinstance(ball, BallStateData) for ball in result)

    def test_convert_api_balls_to_core(self, sample_ball_state):
        """Test batch conversion of BallInfo to BallState."""
        ball_infos = [ball_state_to_ball_info(sample_ball_state)]
        result = convert_api_balls_to_core(ball_infos)

        assert len(result) == 1
        assert all(isinstance(ball, BallState) for ball in result)
        assert result[0].id == "ball_1"

    def test_convert_websocket_balls_to_core(self, sample_ball_state):
        """Test batch conversion of WebSocket data to BallState."""
        ws_data = [ball_state_to_websocket_data(sample_ball_state)]
        result = convert_websocket_balls_to_core(ws_data)

        assert len(result) == 1
        assert all(isinstance(ball, BallState) for ball in result)

    def test_convert_trajectories_to_api(self, sample_trajectory):
        """Test batch conversion of Trajectories to API info."""
        trajectories = [sample_trajectory]
        result = convert_trajectories_to_api(trajectories)

        assert len(result) == 1
        assert all(isinstance(t, TrajectoryInfo) for t in result)

    def test_convert_trajectories_to_websocket(self, sample_trajectory):
        """Test batch conversion of Trajectories to WebSocket data."""
        trajectories = [sample_trajectory]
        result = convert_trajectories_to_websocket(trajectories)

        assert len(result) == 1
        assert all(isinstance(t, TrajectoryData) for t in result)

    def test_batch_conversion_empty_list(self):
        """Test batch conversions with empty lists."""
        assert convert_ball_states_to_api([]) == []
        assert convert_ball_states_to_websocket([]) == []
        assert convert_api_balls_to_core([]) == []
        assert convert_websocket_balls_to_core([]) == []
        assert convert_trajectories_to_api([]) == []
        assert convert_trajectories_to_websocket([]) == []


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidationFunctions:
    """Test validation and error handling functions."""

    def test_validate_coordinate_conversion_valid(self):
        """Scale-aware coordinates validate without restricting canonical bounds."""
        coords = {"x": 100.0, "y": 200.0, "scale": [2.0, 3.0]}
        validate_coordinate_conversion(coords)

    @pytest.mark.parametrize("coords", [[100.0], [100.0, 200.0], [100.0, 200.0, 300.0]])
    def test_validate_coordinate_conversion_rejects_unscaled_lists(self, coords):
        """A coordinate pair without scale cannot identify its source pixel space."""
        with pytest.raises(ValueError, match="must include x, y, and scale"):
            validate_coordinate_conversion(coords)

    def test_validate_coordinate_conversion_non_numeric(self):
        """Scale metadata does not allow non-numeric coordinate components."""
        with pytest.raises(ValueError, match="finite numeric x and y values"):
            validate_coordinate_conversion(
                {"x": 100.0, "y": "invalid", "scale": [1.0, 1.0]}
            )

    @pytest.mark.parametrize(
        "coords",
        [
            {"x": 2000.0, "y": 200.0, "scale": [1.0, 1.0]},
            {"x": 100.0, "y": -2000.0, "scale": [1.0, 1.0]},
            {"x": 3840.0, "y": 2160.0, "scale": [1.0, 1.0]},
        ],
    )
    def test_validate_coordinate_conversion_canonical_coordinates(self, coords):
        """Finite scaled coordinates remain unrestricted by arbitrary world bounds."""
        assert validate_coordinate_conversion(coords) is None

    @pytest.mark.parametrize(
        "coords",
        [
            {"x": 3840.0, "y": 2160.0, "scale": [1.0, 1.0]},
            {"x": 960.0, "y": 540.0, "scale": (2.0, 2.0)},
            PositionWithScale(x=960.0, y=540.0, scale=[2.0, 2.0]),
        ],
    )
    def test_validate_coordinate_conversion_transport_coordinates(self, coords):
        """Both serialized transport points and parsed models retain metadata."""
        assert validate_coordinate_conversion(coords) is None

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_validate_coordinate_conversion_non_finite_component(self, value, axis):
        """Neither component may contain a non-finite number."""
        coords = {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]}
        coords["xy"[axis]] = value
        with pytest.raises(ValueError, match="finite numeric x and y values"):
            validate_coordinate_conversion(coords)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_validate_coordinate_conversion_non_finite_transport(self, value, axis):
        """Scale metadata cannot make a non-finite transport component valid."""
        coords = {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]}
        coords[axis] = value
        with pytest.raises(ValueError, match="finite numeric x and y values"):
            validate_coordinate_conversion(coords)

    @pytest.mark.parametrize(
        "scale",
        [
            [math.nan, 1.0],
            [1.0, math.inf],
            [-math.inf, 1.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [True, 1.0],
            [1.0, "2.0"],
        ],
    )
    def test_validate_coordinate_conversion_invalid_scale_values(self, scale):
        """Scale values must be finite positive numbers, excluding booleans."""
        coords = {"x": 100.0, "y": 200.0, "scale": scale}
        with pytest.raises(
            ValueError, match="scale values must be finite and positive"
        ):
            validate_coordinate_conversion(coords)

    @pytest.mark.parametrize("scale", [None, 1.0, [], [1.0], [1.0, 1.0, 1.0]])
    def test_validate_coordinate_conversion_invalid_scale_shape(self, scale):
        """Malformed scale metadata reports a structural validation error."""
        coords = {"x": 100.0, "y": 200.0, "scale": scale}
        with pytest.raises(ValueError, match="scale must have exactly 2 values"):
            validate_coordinate_conversion(coords)

    @pytest.mark.parametrize(
        "coords",
        [
            {"x": 100.0, "y": 200.0},
            {"x": 100.0, "scale": [1.0, 1.0]},
            {"y": 200.0, "scale": [1.0, 1.0]},
        ],
    )
    def test_validate_coordinate_conversion_missing_transport_metadata(self, coords):
        """Transport objects always require both coordinates and declared scale."""
        with pytest.raises(ValueError, match="position must include x, y, and scale"):
            validate_coordinate_conversion(coords, field_name="position")

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize("field", ["x", "y", "scale_x", "scale_y"])
    def test_position_with_scale_rejects_non_finite_numbers(self, value, field):
        """Direct model construction enforces the same numeric boundary."""
        values = {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]}
        if field.startswith("scale_"):
            values["scale"][0 if field == "scale_x" else 1] = value
            expected_location = ("scale",)
            expected_type = "value_error"
        else:
            values[field] = value
            expected_location = (field,)
            expected_type = "finite_number"

        with pytest.raises(ValidationError) as error:
            PositionWithScale.model_validate(values)

        failures = error.value.errors()
        assert len(failures) == 1
        assert failures[0]["loc"] == expected_location
        assert failures[0]["type"] == expected_type
        assert "finite" in failures[0]["msg"]

    def test_validate_coordinate_conversion_custom_field_name(self):
        """Error messages identify fields that lack their coordinate scale."""
        with pytest.raises(ValueError, match="velocity must include x, y, and scale"):
            validate_coordinate_conversion([100.0], field_name="velocity")

    def test_validate_ball_state_conversion_valid(self):
        """Test ball state validation with valid data."""
        ball_data = {
            "id": "ball_1",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            "velocity": {"x": 50.0, "y": 25.0, "scale": [1.0, 1.0]},
            "confidence": 0.95,
            "radius": 36.0,
        }
        result = validate_ball_state_conversion(ball_data)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_ball_state_conversion_rejects_unscaled_coordinates(self):
        """Ball payloads cannot omit scale metadata on positions or velocities."""
        result = validate_ball_state_conversion(
            {
                "id": "ball_1",
                "position": [100.0, 200.0],
                "velocity": [50.0, 25.0],
            }
        )

        assert result.is_valid is False
        assert set(result.field_errors) == {"position", "velocity"}
        assert all(
            "must include x, y, and scale" in errors[0]
            for errors in result.field_errors.values()
        )

    def test_validate_ball_state_conversion_missing_field(self):
        """Test ball state validation with missing required field."""
        ball_data = {
            "id": "ball_1",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            # Missing velocity
        }
        result = validate_ball_state_conversion(ball_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == ["Missing required field: velocity"]
        assert result.field_errors == {"velocity": ["Missing required field: velocity"]}

    def test_validate_ball_state_conversion_invalid_confidence(self):
        """Test ball state validation with invalid confidence."""
        ball_data = {
            "id": "ball_1",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            "velocity": {"x": 50.0, "y": 25.0, "scale": [1.0, 1.0]},
            "confidence": 1.5,  # Out of range
        }
        result = validate_ball_state_conversion(ball_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == ["Confidence must be a number between 0 and 1"]
        assert result.field_errors == {
            "confidence": ["Confidence must be a number between 0 and 1"]
        }

    def test_validate_ball_state_conversion_invalid_radius(self):
        """Test ball state validation with invalid radius."""
        ball_data = {
            "id": "ball_1",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            "velocity": {"x": 50.0, "y": 25.0, "scale": [1.0, 1.0]},
            "radius": -10.0,  # Negative
        }
        result = validate_ball_state_conversion(ball_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == ["Radius must be a positive number"]
        assert result.field_errors == {"radius": ["Radius must be a positive number"]}

    def test_safe_convert_game_state_valid(self, sample_game_state):
        """Test safe game state conversion with valid data."""
        # Create dict representation
        state_data = {
            "timestamp": sample_game_state.timestamp,
            "frame_number": sample_game_state.frame_number,
            "balls": [
                {
                    "id": ball.id,
                    "number": ball.number,
                    "position": {
                        "x": ball.position.x,
                        "y": ball.position.y,
                        "scale": list(ball.position.scale),
                    },
                    "velocity": {
                        "x": ball.velocity.x,
                        "y": ball.velocity.y,
                        "scale": list(ball.velocity.scale),
                    },
                    "radius": ball.radius,
                    "radius_scale": list(ball.radius_scale),
                    "radius_unit": "canonical_pixels",
                    "is_cue_ball": ball.is_cue_ball,
                    "is_pocketed": ball.is_pocketed,
                    "confidence": ball.confidence,
                }
                for ball in sample_game_state.balls
            ],
            "table": {
                "width": sample_game_state.table.width,
                "height": sample_game_state.table.height,
                "dimensions_scale": list(sample_game_state.table.dimensions_scale),
                "dimensions_unit": "canonical_pixels",
                "pocket_positions": [
                    {"x": pos.x, "y": pos.y, "scale": list(pos.scale)}
                    for pos in sample_game_state.table.pocket_positions
                ],
                "pocket_radius": sample_game_state.table.pocket_radius,
                "pocket_radius_scale": list(
                    sample_game_state.table.pocket_radius_scale
                ),
                "pocket_radius_unit": "canonical_pixels",
                "real_world_dimensions": {
                    "width": sample_game_state.table.real_width_meters,
                    "height": sample_game_state.table.real_height_meters,
                    "unit": "meters",
                },
            },
            "cue": {
                "tip_position": {
                    "x": sample_game_state.cue.tip_position.x,
                    "y": sample_game_state.cue.tip_position.y,
                    "scale": list(sample_game_state.cue.tip_position.scale),
                },
                "angle": sample_game_state.cue.angle,
                "elevation": sample_game_state.cue.elevation,
                "estimated_force": sample_game_state.cue.estimated_force,
                "is_visible": sample_game_state.cue.is_visible,
                "confidence": sample_game_state.cue.confidence,
            },
            "game_type": sample_game_state.game_type.value,
            "is_valid": sample_game_state.is_valid,
            "confidence": sample_game_state.state_confidence,
        }

        result = safe_convert_game_state(state_data, validate=True)

        assert isinstance(result, GameState), result
        assert result.timestamp == sample_game_state.timestamp
        assert result.frame_number == sample_game_state.frame_number
        assert result.game_type == sample_game_state.game_type
        assert result.state_confidence == sample_game_state.state_confidence
        assert result.is_valid is True
        assert [ball.id for ball in result.balls] == ["cue", "ball_1"]
        for actual, expected in zip(result.balls, sample_game_state.balls):
            assert actual.position == expected.position
            assert actual.velocity == expected.velocity
            assert actual.number == expected.number
            assert actual.radius == expected.radius
            assert actual.confidence == expected.confidence
            assert actual.is_cue_ball == expected.is_cue_ball
            assert actual.is_pocketed == expected.is_pocketed
        assert result.table.width == sample_game_state.table.width
        assert result.table.height == sample_game_state.table.height
        assert result.table.pocket_positions == sample_game_state.table.pocket_positions
        assert result.cue.tip_position == sample_game_state.cue.tip_position
        assert result.cue.angle == sample_game_state.cue.angle

    def test_safe_convert_game_state_missing_field(self):
        """Test safe game state conversion with missing required field."""
        state_data = {
            "timestamp": 123456.0,
            "frame_number": 100,
            # Missing balls and table
        }

        result = safe_convert_game_state(state_data, validate=True)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == [
            "Missing required field: balls",
            "Missing required field: table",
        ]

    @pytest.mark.parametrize(
        ("balls", "expected_message", "expected_field"),
        [
            (None, "Balls must be a list", "balls"),
            ([None], "Ball 0 must be an object", "balls[0]"),
        ],
    )
    def test_safe_convert_game_state_malformed_balls(
        self, sample_table_state, balls, expected_message, expected_field
    ):
        """Malformed ball collections return structured errors without raising."""
        state_data = {
            "timestamp": 123456.0,
            "frame_number": 100,
            "balls": balls,
            "table": table_state_to_websocket_data(sample_table_state).model_dump(),
        }

        result = safe_convert_game_state(state_data, validate=True)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == [expected_message]
        assert result.field_errors == {expected_field: [expected_message]}

    def test_safe_convert_game_state_rejects_non_finite_table_pocket(
        self, sample_table_state
    ):
        """A non-finite calibrated pocket cannot enter the physical table state."""
        table_data = table_state_to_websocket_data(sample_table_state).model_dump()
        table_data["pocket_positions"][0]["x"] = math.nan
        state_data = {
            "timestamp": 123456.0,
            "frame_number": 100,
            "balls": [],
            "table": table_data,
        }

        result = safe_convert_game_state(state_data, validate=True)

        message = "table.pocket_positions[0] must contain finite numeric x and y values"
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert result.errors == [message]
        assert result.field_errors == {"table.pocket_positions[0]": [message]}

    def test_safe_convert_game_state_no_validation(self, sample_game_state):
        """Test safe game state conversion without validation."""
        state_data = {
            "timestamp": sample_game_state.timestamp,
            "frame_number": sample_game_state.frame_number,
            "balls": [
                {
                    "id": ball.id,
                    "number": ball.number,
                    "position": {
                        "x": ball.position.x,
                        "y": ball.position.y,
                        "scale": list(ball.position.scale),
                    },
                    "velocity": {
                        "x": ball.velocity.x,
                        "y": ball.velocity.y,
                        "scale": list(ball.velocity.scale),
                    },
                    "radius": ball.radius,
                    "radius_scale": list(ball.radius_scale),
                    "radius_unit": "canonical_pixels",
                    "is_cue_ball": ball.is_cue_ball,
                    "is_pocketed": ball.is_pocketed,
                    "confidence": ball.confidence,
                }
                for ball in sample_game_state.balls
            ],
            "table": {
                "width": sample_game_state.table.width,
                "height": sample_game_state.table.height,
                "dimensions_scale": list(sample_game_state.table.dimensions_scale),
                "dimensions_unit": "canonical_pixels",
                "pocket_positions": [
                    {"x": pos.x, "y": pos.y, "scale": list(pos.scale)}
                    for pos in sample_game_state.table.pocket_positions
                ],
                "pocket_radius": sample_game_state.table.pocket_radius,
                "pocket_radius_scale": list(
                    sample_game_state.table.pocket_radius_scale
                ),
                "pocket_radius_unit": "canonical_pixels",
                "real_world_dimensions": {
                    "width": sample_game_state.table.real_width_meters,
                    "height": sample_game_state.table.real_height_meters,
                    "unit": "meters",
                },
            },
        }

        result = safe_convert_game_state(state_data, validate=False)

        assert isinstance(result, GameState), result
        assert result.timestamp == sample_game_state.timestamp
        assert result.frame_number == sample_game_state.frame_number
        assert result.balls[0].position == sample_game_state.balls[0].position
        assert result.balls[1].velocity == sample_game_state.balls[1].velocity
        assert result.table.pocket_positions == sample_game_state.table.pocket_positions
        assert result.cue is None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_ball_state_with_pocketed_ball(self):
        """Test converting pocketed ball state."""
        ball = BallState(
            id="ball_8",
            number=8,
            position=Vector2D.from_4k(0.0, 0.0),
            velocity=Vector2D.from_4k(0.0, 0.0),
            is_pocketed=True,
            confidence=1.0,
        )

        result = ball_state_to_ball_info(ball)
        assert result.is_pocketed is True

    def test_cue_state_not_visible(self):
        """Test converting invisible cue state."""
        cue = CueState(
            tip_position=Vector2D.from_4k(0.0, 0.0),
            angle=0.0,
            elevation=0.0,
            estimated_force=0.0,
            is_visible=False,
            confidence=0.0,
        )

        result = cue_state_to_cue_info(cue)
        assert result.is_visible is False

    def test_game_state_with_empty_events(self, sample_game_state):
        """Test converting game state with no events."""
        sample_game_state.events = []
        result = game_state_to_response(sample_game_state)

        assert len(result.events) == 0

    def test_trajectory_with_no_collisions(self):
        """Test converting trajectory with no collisions."""
        trajectory = Trajectory(
            ball_id="ball_1",
            points=[Vector2D.from_4k(100.0, 100.0)],
            collisions=[],
            final_position=Vector2D.from_4k(100.0, 100.0),  # Required field
            final_velocity=Vector2D.from_4k(0.0, 0.0),  # Required field
            will_be_pocketed=False,
            pocket_id=None,
            time_to_rest=1.0,
            max_velocity=50.0,
            confidence=0.9,
        )

        result = trajectory_to_api_info(trajectory)
        assert len(result.points) == 1

    def test_round_trip_preserves_scale_metadata(self, sample_ball_state):
        """Test that round-trip conversions preserve scale metadata."""
        # BallState -> BallInfo -> BallState
        ball_info = ball_state_to_ball_info(sample_ball_state)
        recovered = ball_info_to_ball_state(ball_info)

        # Verify scale is preserved
        assert recovered.position.scale == sample_ball_state.position.scale
        assert recovered.velocity.scale == sample_ball_state.velocity.scale

    def test_validation_result_add_error(self):
        """Errors update validity and preserve optional field-level messages."""
        result = ValidationResult(is_valid=True)

        result.add_error("Error 1", field="position")
        result.add_error("Error 2")

        assert result.is_valid is False
        assert result.errors == ["Error 1", "Error 2"]
        assert result.field_errors == {"position": ["Error 1"]}
        assert result.warnings == []
        independent = ValidationResult(is_valid=True)
        assert independent.errors == []
        assert independent.field_errors == {}
