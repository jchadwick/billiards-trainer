"""Comprehensive unit tests for core.models module.

This test suite provides complete coverage of BallState, TableState, and GameState
classes, including initialization, validation, methods, serialization, and factory methods.

Test Coverage:
- BallState: initialization, validation, kinetic_energy, is_moving, distance_to, copy, to/from_dict
- TableState: initialization, validation, is_point_in_pocket, is_point_on_table, is_point_in_playing_area, from_calibration
- GameState: get_ball_by_id, get_cue_ball, count_balls_by_player, validate_consistency, create_initial_state
- Additional model classes: CueState, GameEvent, Collision, Trajectory, ShotAnalysis, CoordinateMetadata
"""

import json
import math
from datetime import datetime

import pytest

from backend.core.coordinates import Vector2D
from backend.core.models import (
    BallState,
    Collision,
    CoordinateMetadata,
    CueState,
    GameEvent,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
    Trajectory,
    calculate_ball_separation,
    create_standard_ball_set,
    deserialize_from_json,
    find_closest_ball,
    serialize_to_json,
    validate_ball_physics,
)

# ============================================================================
# BallState Tests
# ============================================================================


class TestBallStateInitialization:
    """Tests for BallState initialization and validation."""

    def test_basic_initialization(self):
        """Test basic BallState creation with required parameters."""
        position = Vector2D.from_4k(1920.0, 1080.0)
        ball = BallState(id="test", position=position)

        assert ball.id == "test"
        assert ball.position.x == 1920.0
        assert ball.position.y == 1080.0
        assert ball.radius == 36.0  # Default 4K radius
        assert ball.mass == 0.17  # Default mass
        assert ball.is_cue_ball is False
        assert ball.is_pocketed is False
        assert ball.confidence == 1.0

    def test_initialization_with_all_parameters(self):
        """Test BallState with all parameters specified."""
        position = Vector2D.from_4k(100.0, 200.0)
        velocity = Vector2D.from_4k(50.0, 30.0)
        spin = Vector2D.from_4k(10.0, 5.0)

        ball = BallState(
            id="ball_1",
            position=position,
            velocity=velocity,
            radius=40.0,
            mass=0.18,
            spin=spin,
            is_cue_ball=True,
            is_pocketed=False,
            number=1,
            confidence=0.95,
        )

        assert ball.id == "ball_1"
        assert ball.velocity.x == 50.0
        assert ball.velocity.y == 30.0
        assert ball.radius == 40.0
        assert ball.mass == 0.18
        assert ball.spin.x == 10.0
        assert ball.is_cue_ball is True
        assert ball.number == 1
        assert ball.confidence == 0.95

    def test_negative_radius_raises_error(self):
        """Test that negative radius raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Ball radius must be positive"):
            BallState(id="test", position=position, radius=-1.0)

    def test_zero_radius_raises_error(self):
        """Test that zero radius raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Ball radius must be positive"):
            BallState(id="test", position=position, radius=0.0)

    def test_negative_mass_raises_error(self):
        """Test that negative mass raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Ball mass must be positive"):
            BallState(id="test", position=position, mass=-0.1)

    def test_zero_mass_raises_error(self):
        """Test that zero mass raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Ball mass must be positive"):
            BallState(id="test", position=position, mass=0.0)

    def test_confidence_below_zero_raises_error(self):
        """Test that confidence below 0.0 raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BallState(id="test", position=position, confidence=-0.1)

    def test_confidence_above_one_raises_error(self):
        """Test that confidence above 1.0 raises ValueError."""
        position = Vector2D.from_4k(100.0, 100.0)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            BallState(id="test", position=position, confidence=1.1)

    def test_confidence_at_boundaries(self):
        """Test that confidence at 0.0 and 1.0 is valid."""
        position = Vector2D.from_4k(100.0, 100.0)
        ball_min = BallState(id="test1", position=position, confidence=0.0)
        ball_max = BallState(id="test2", position=position, confidence=1.0)

        assert ball_min.confidence == 0.0
        assert ball_max.confidence == 1.0


class TestBallStateKineticEnergy:
    """Tests for BallState kinetic energy calculations."""

    def test_kinetic_energy_stationary_ball(self):
        """Test kinetic energy of stationary ball is zero."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(0.0, 0.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        assert ball.kinetic_energy() == 0.0

    def test_kinetic_energy_moving_ball_no_spin(self):
        """Test kinetic energy of moving ball without spin."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(10.0, 0.0)  # 10 units/s
        ball = BallState(id="test", position=position, velocity=velocity, mass=0.17)

        # KE = 0.5 * m * v^2 = 0.5 * 0.17 * 100 = 8.5
        expected_ke = 0.5 * 0.17 * 100
        assert ball.kinetic_energy() == pytest.approx(expected_ke)

    def test_kinetic_energy_with_spin(self):
        """Test kinetic energy includes rotational component."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(10.0, 0.0)
        spin = Vector2D.from_4k(5.0, 0.0)
        ball = BallState(
            id="test",
            position=position,
            velocity=velocity,
            spin=spin,
            radius=36.0,
            mass=0.17,
        )

        # Linear KE = 0.5 * m * v^2
        linear_ke = 0.5 * 0.17 * 100

        # Rotational KE = 0.5 * I * ω^2, where I = 0.4 * m * r^2
        moment_of_inertia = 0.4 * 0.17 * 36.0**2
        rotational_ke = 0.5 * moment_of_inertia * 25.0  # magnitude^2 of spin

        total_ke = linear_ke + rotational_ke
        assert ball.kinetic_energy() == pytest.approx(total_ke)


class TestBallStateMovement:
    """Tests for BallState movement detection."""

    def test_is_moving_stationary(self):
        """Test stationary ball is not moving."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(0.0, 0.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        assert ball.is_moving() is False

    def test_is_moving_fast(self):
        """Test fast-moving ball is detected as moving."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(10.0, 5.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        assert ball.is_moving() is True

    def test_is_moving_slow_above_threshold(self):
        """Test slow-moving ball above threshold is detected."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(0.002, 0.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        assert ball.is_moving(threshold=0.001) is True

    def test_is_moving_slow_below_threshold(self):
        """Test slow-moving ball below threshold is not detected."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(0.0005, 0.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        assert ball.is_moving(threshold=0.001) is False

    def test_is_spinning_no_spin(self):
        """Test ball with no spin."""
        position = Vector2D.from_4k(100.0, 100.0)
        spin = Vector2D.from_4k(0.0, 0.0)
        ball = BallState(id="test", position=position, spin=spin)

        assert ball.is_spinning() is False

    def test_is_spinning_with_spin(self):
        """Test ball with significant spin."""
        position = Vector2D.from_4k(100.0, 100.0)
        spin = Vector2D.from_4k(1.0, 1.0)
        ball = BallState(id="test", position=position, spin=spin)

        assert ball.is_spinning(threshold=0.1) is True


class TestBallStateDistanceAndTouching:
    """Tests for BallState distance and touching detection."""

    def test_distance_to_same_position(self):
        """Test distance to ball at same position is zero."""
        position = Vector2D.from_4k(100.0, 100.0)
        ball1 = BallState(id="ball1", position=position)
        ball2 = BallState(id="ball2", position=position)

        assert ball1.distance_to(ball2) == 0.0

    def test_distance_to_different_position(self):
        """Test distance calculation between balls."""
        pos1 = Vector2D.from_4k(0.0, 0.0)
        pos2 = Vector2D.from_4k(3.0, 4.0)
        ball1 = BallState(id="ball1", position=pos1)
        ball2 = BallState(id="ball2", position=pos2)

        assert ball1.distance_to(ball2) == pytest.approx(5.0)

    def test_is_touching_not_touching(self):
        """Test balls that are not touching."""
        pos1 = Vector2D.from_4k(0.0, 0.0)
        pos2 = Vector2D.from_4k(100.0, 0.0)
        ball1 = BallState(id="ball1", position=pos1, radius=36.0)
        ball2 = BallState(id="ball2", position=pos2, radius=36.0)

        assert ball1.is_touching(ball2) is False

    def test_is_touching_exactly_touching(self):
        """Test balls that are exactly touching."""
        pos1 = Vector2D.from_4k(0.0, 0.0)
        pos2 = Vector2D.from_4k(72.0, 0.0)  # 2 * radius
        ball1 = BallState(id="ball1", position=pos1, radius=36.0)
        ball2 = BallState(id="ball2", position=pos2, radius=36.0)

        assert ball1.is_touching(ball2) is True

    def test_is_touching_overlapping(self):
        """Test balls that are overlapping."""
        pos1 = Vector2D.from_4k(0.0, 0.0)
        pos2 = Vector2D.from_4k(60.0, 0.0)  # Less than 2 * radius
        ball1 = BallState(id="ball1", position=pos1, radius=36.0)
        ball2 = BallState(id="ball2", position=pos2, radius=36.0)

        assert ball1.is_touching(ball2) is True


class TestBallStateCopy:
    """Tests for BallState copy method."""

    def test_copy_preserves_all_fields(self):
        """Test that copy preserves all ball properties."""
        position = Vector2D.from_4k(100.0, 200.0)
        velocity = Vector2D.from_4k(10.0, 20.0)
        spin = Vector2D.from_4k(5.0, 5.0)

        ball = BallState(
            id="ball_1",
            position=position,
            velocity=velocity,
            radius=40.0,
            mass=0.18,
            spin=spin,
            is_cue_ball=True,
            is_pocketed=False,
            number=1,
            confidence=0.95,
        )

        ball_copy = ball.copy()

        assert ball_copy.id == ball.id
        assert ball_copy.position.x == ball.position.x
        assert ball_copy.position.y == ball.position.y
        assert ball_copy.velocity.x == ball.velocity.x
        assert ball_copy.velocity.y == ball.velocity.y
        assert ball_copy.radius == ball.radius
        assert ball_copy.mass == ball.mass
        assert ball_copy.spin.x == ball.spin.x
        assert ball_copy.is_cue_ball == ball.is_cue_ball
        assert ball_copy.is_pocketed == ball.is_pocketed
        assert ball_copy.number == ball.number
        assert ball_copy.confidence == ball.confidence

    def test_copy_preserves_scale_metadata(self):
        """Test that copy preserves Vector2D scale metadata."""
        position = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        ball = BallState(id="test", position=position)

        ball_copy = ball.copy()

        assert ball_copy.position.scale == ball.position.scale
        assert ball_copy.velocity.scale == ball.velocity.scale
        assert ball_copy.spin.scale == ball.spin.scale

    def test_copy_is_independent(self):
        """Test that copy is independent of original."""
        position = Vector2D.from_4k(100.0, 100.0)
        ball = BallState(id="test", position=position)
        ball_copy = ball.copy()

        # Modify copy
        ball_copy.position = Vector2D.from_4k(200.0, 200.0)

        # Original should be unchanged
        assert ball.position.x == 100.0
        assert ball.position.y == 100.0


class TestBallStateSerialization:
    """Tests for BallState serialization (to_dict/from_dict)."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        position = Vector2D.from_4k(100.0, 200.0)
        ball = BallState(id="test", position=position, number=1)

        data = ball.to_dict()

        assert data["id"] == "test"
        assert data["position"]["x"] == 100.0
        assert data["position"]["y"] == 200.0
        assert data["number"] == 1
        assert data["is_cue_ball"] is False

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        position = Vector2D.from_4k(100.0, 200.0)
        velocity = Vector2D.from_4k(10.0, 20.0)
        spin = Vector2D.from_4k(5.0, 5.0)

        ball = BallState(
            id="ball_1",
            position=position,
            velocity=velocity,
            radius=40.0,
            mass=0.18,
            spin=spin,
            is_cue_ball=True,
            is_pocketed=False,
            number=1,
            confidence=0.95,
        )

        data = ball.to_dict()

        assert "id" in data
        assert "position" in data
        assert "velocity" in data
        assert "radius" in data
        assert "mass" in data
        assert "spin" in data
        assert "is_cue_ball" in data
        assert "is_pocketed" in data
        assert "number" in data
        assert "confidence" in data
        assert "last_update" in data

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        data = {
            "id": "test",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            "velocity": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
            "spin": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
        }

        ball = BallState.from_dict(data)

        assert ball.id == "test"
        assert ball.position.x == 100.0
        assert ball.position.y == 200.0

    def test_from_dict_with_defaults(self):
        """Test from_dict uses proper defaults for missing fields."""
        data = {
            "id": "test",
            "position": {"x": 100.0, "y": 200.0, "scale": [1.0, 1.0]},
            "velocity": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
            "spin": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
        }

        ball = BallState.from_dict(data)

        assert ball.radius == 36.0  # Default 4K radius
        assert ball.mass == 0.17  # Default mass
        assert ball.confidence == 1.0  # Default confidence
        assert ball.is_cue_ball is False
        assert ball.is_pocketed is False

    def test_serialization_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        position = Vector2D.from_4k(100.0, 200.0)
        velocity = Vector2D.from_4k(10.0, 20.0)
        spin = Vector2D.from_4k(5.0, 5.0)

        ball = BallState(
            id="ball_1",
            position=position,
            velocity=velocity,
            radius=40.0,
            mass=0.18,
            spin=spin,
            is_cue_ball=True,
            is_pocketed=False,
            number=1,
            confidence=0.95,
        )

        data = ball.to_dict()
        ball_restored = BallState.from_dict(data)

        assert ball_restored.id == ball.id
        assert ball_restored.position.x == ball.position.x
        assert ball_restored.position.y == ball.position.y
        assert ball_restored.velocity.x == ball.velocity.x
        assert ball_restored.radius == ball.radius
        assert ball_restored.mass == ball.mass
        assert ball_restored.is_cue_ball == ball.is_cue_ball
        assert ball_restored.number == ball.number
        assert ball_restored.confidence == ball.confidence


class TestBallStateFactoryMethods:
    """Tests for BallState factory methods."""

    def test_from_4k_basic(self):
        """Test from_4k factory method."""
        ball = BallState.from_4k("test", 1920.0, 1080.0, number=1)

        assert ball.id == "test"
        assert ball.position.x == 1920.0
        assert ball.position.y == 1080.0
        assert ball.position.scale == (1.0, 1.0)
        assert ball.number == 1

    def test_from_4k_with_kwargs(self):
        """Test from_4k with additional kwargs."""
        ball = BallState.from_4k(
            "test", 100.0, 200.0, number=1, is_cue_ball=True, confidence=0.9
        )

        assert ball.is_cue_ball is True
        assert ball.confidence == 0.9

    def test_from_resolution_basic(self):
        """Test from_resolution factory method."""
        ball = BallState.from_resolution("test", 960.0, 540.0, (1920, 1080), number=1)

        assert ball.id == "test"
        assert ball.position.x == 960.0
        assert ball.position.y == 540.0
        assert ball.position.scale == (2.0, 2.0)
        assert ball.number == 1

    def test_from_resolution_converts_to_4k(self):
        """Test from_resolution can convert to 4K."""
        ball = BallState.from_resolution("test", 960.0, 540.0, (1920, 1080))

        pos_4k = ball.position.to_4k_canonical()
        assert pos_4k.x == pytest.approx(1920.0)
        assert pos_4k.y == pytest.approx(1080.0)


# ============================================================================
# TableState Tests
# ============================================================================


class TestTableStateInitialization:
    """Tests for TableState initialization and validation."""

    def test_basic_initialization(self):
        """Test basic TableState creation."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        assert table.width == 3200.0
        assert table.height == 1600.0
        assert len(table.pocket_positions) == 6

    def test_initialization_with_all_parameters(self):
        """Test TableState with all parameters."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        table = TableState(
            width=3200.0,
            height=1600.0,
            pocket_positions=pockets,
            pocket_radius=72.0,
            cushion_elasticity=0.9,
            surface_friction=0.15,
            surface_slope=0.0,
            cushion_height=0.064,
        )

        assert table.pocket_radius == 72.0
        assert table.cushion_elasticity == 0.9
        assert table.surface_friction == 0.15
        assert table.surface_slope == 0.0
        assert table.cushion_height == 0.064

    def test_negative_width_raises_error(self):
        """Test that negative width raises ValueError."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        with pytest.raises(ValueError, match="Table dimensions must be positive"):
            TableState(width=-100.0, height=1600.0, pocket_positions=pockets)

    def test_zero_height_raises_error(self):
        """Test that zero height raises ValueError."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        with pytest.raises(ValueError, match="Table dimensions must be positive"):
            TableState(width=3200.0, height=0.0, pocket_positions=pockets)

    def test_negative_pocket_radius_raises_error(self):
        """Test that negative pocket radius raises ValueError."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        with pytest.raises(ValueError, match="Pocket radius must be positive"):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                pocket_radius=-1.0,
            )

    def test_cushion_elasticity_above_one_raises_error(self):
        """Test that cushion elasticity above 1.0 raises ValueError."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        with pytest.raises(
            ValueError, match="Cushion elasticity must be between 0.0 and 1.0"
        ):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                cushion_elasticity=1.5,
            )

    def test_negative_surface_friction_raises_error(self):
        """Test that negative surface friction raises ValueError."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]

        with pytest.raises(ValueError, match="Surface friction must be non-negative"):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                surface_friction=-0.1,
            )

    def test_wrong_number_of_pockets_raises_error(self):
        """Test that wrong number of pockets raises ValueError."""
        pockets = [Vector2D.from_4k(0, 0), Vector2D.from_4k(100, 100)]  # Only 2 pockets

        with pytest.raises(ValueError, match="Table must have exactly 6 pockets"):
            TableState(width=3200.0, height=1600.0, pocket_positions=pockets)


class TestTableStatePocketDetection:
    """Tests for TableState pocket detection."""

    @pytest.fixture()
    def table(self):
        """Create a standard table for testing."""
        pockets = [
            Vector2D.from_4k(0.0, 0.0),
            Vector2D.from_4k(1600.0, 0.0),
            Vector2D.from_4k(3200.0, 0.0),
            Vector2D.from_4k(0.0, 1600.0),
            Vector2D.from_4k(1600.0, 1600.0),
            Vector2D.from_4k(3200.0, 1600.0),
        ]
        return TableState(
            width=3200.0, height=1600.0, pocket_positions=pockets, pocket_radius=72.0
        )

    def test_is_point_in_pocket_exact_center(self, table):
        """Test point at exact pocket center."""
        point = Vector2D.from_4k(0.0, 0.0)
        is_in_pocket, pocket_id = table.is_point_in_pocket(point)

        assert is_in_pocket is True
        assert pocket_id == 0

    def test_is_point_in_pocket_within_radius(self, table):
        """Test point within pocket radius."""
        point = Vector2D.from_4k(50.0, 0.0)  # 50 pixels from pocket center
        is_in_pocket, pocket_id = table.is_point_in_pocket(point)

        assert is_in_pocket is True
        assert pocket_id == 0

    def test_is_point_in_pocket_outside_radius(self, table):
        """Test point outside all pockets."""
        point = Vector2D.from_4k(1000.0, 800.0)  # Middle of table
        is_in_pocket, pocket_id = table.is_point_in_pocket(point)

        assert is_in_pocket is False
        assert pocket_id is None

    def test_is_point_in_pocket_with_tolerance(self, table):
        """Test pocket detection with tolerance."""
        point = Vector2D.from_4k(80.0, 0.0)  # Just beyond radius
        is_in_pocket_no_tolerance, _ = table.is_point_in_pocket(point)
        is_in_pocket_with_tolerance, pocket_id = table.is_point_in_pocket(
            point, tolerance=10.0
        )

        assert is_in_pocket_no_tolerance is False
        assert is_in_pocket_with_tolerance is True
        assert pocket_id == 0


class TestTableStatePointOnTable:
    """Tests for TableState point on table detection."""

    @pytest.fixture()
    def table(self):
        """Create a standard table for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        return TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

    def test_is_point_on_table_center(self, table):
        """Test point at table center is on table."""
        point = Vector2D.from_4k(1600.0, 800.0)
        assert table.is_point_on_table(point) is True

    def test_is_point_on_table_at_edge(self, table):
        """Test point at table edge is on table."""
        point = Vector2D.from_4k(0.0, 0.0)
        assert table.is_point_on_table(point) is True

    def test_is_point_on_table_outside(self, table):
        """Test point outside table bounds."""
        point = Vector2D.from_4k(-100.0, 800.0)
        assert table.is_point_on_table(point) is False

    def test_is_point_on_table_with_ball_radius(self, table):
        """Test point on table with ball radius margin."""
        point = Vector2D.from_4k(10.0, 800.0)
        assert table.is_point_on_table(point, ball_radius=36.0) is False
        assert table.is_point_on_table(point, ball_radius=5.0) is True


class TestTableStatePlayingArea:
    """Tests for TableState playing area detection."""

    @pytest.fixture()
    def table_with_playing_area(self):
        """Create a table with playing area corners."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        corners = [
            Vector2D.from_4k(100.0, 100.0),
            Vector2D.from_4k(3100.0, 100.0),
            Vector2D.from_4k(3100.0, 1500.0),
            Vector2D.from_4k(100.0, 1500.0),
        ]
        return TableState(
            width=3200.0,
            height=1600.0,
            pocket_positions=pockets,
            playing_area_corners=corners,
        )

    def test_is_point_in_playing_area_inside(self, table_with_playing_area):
        """Test point inside playing area."""
        point = Vector2D.from_4k(1600.0, 800.0)
        assert table_with_playing_area.is_point_in_playing_area(point) is True

    def test_is_point_in_playing_area_outside(self, table_with_playing_area):
        """Test point outside playing area."""
        point = Vector2D.from_4k(50.0, 50.0)
        assert table_with_playing_area.is_point_in_playing_area(point) is False

    def test_is_point_in_playing_area_with_margin(self, table_with_playing_area):
        """Test point in playing area with margin."""
        point = Vector2D.from_4k(110.0, 110.0)  # Just inside
        assert (
            table_with_playing_area.is_point_in_playing_area(point, margin=0.0) is True
        )
        assert (
            table_with_playing_area.is_point_in_playing_area(point, margin=20.0)
            is False
        )


class TestTableStateSerialization:
    """Tests for TableState serialization."""

    @pytest.fixture()
    def table(self):
        """Create a standard table for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        return TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

    def test_to_dict_basic(self, table):
        """Test basic to_dict serialization."""
        data = table.to_dict()

        assert data["width"] == 3200.0
        assert data["height"] == 1600.0
        assert len(data["pocket_positions"]) == 6

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        data = {
            "width": 3200.0,
            "height": 1600.0,
            "pocket_positions": [
                {"x": float(x), "y": float(y), "scale": [1.0, 1.0]}
                for x in [0, 1600, 3200]
                for y in [0, 1600]
            ],
        }

        table = TableState.from_dict(data)

        assert table.width == 3200.0
        assert table.height == 1600.0
        assert len(table.pocket_positions) == 6

    def test_serialization_round_trip(self, table):
        """Test that to_dict -> from_dict preserves data."""
        data = table.to_dict()
        table_restored = TableState.from_dict(data)

        assert table_restored.width == table.width
        assert table_restored.height == table.height
        assert table_restored.pocket_radius == table.pocket_radius
        assert table_restored.cushion_elasticity == table.cushion_elasticity


class TestTableStateFromCalibration:
    """Tests for TableState.from_calibration factory method."""

    def test_from_calibration_basic(self):
        """Test basic from_calibration creation."""
        # Camera corners in pixel coordinates
        corners = [(100.0, 100.0), (1820.0, 100.0), (1820.0, 980.0), (100.0, 980.0)]

        table = TableState.from_calibration(corners, camera_resolution=(1920, 1080))

        assert table.width > 0
        assert table.height > 0
        assert len(table.pocket_positions) == 6
        assert table.playing_area_corners is not None
        assert len(table.playing_area_corners) == 4

    def test_from_calibration_with_pocket_positions(self):
        """Test from_calibration with explicit pocket positions."""
        corners = [(100.0, 100.0), (1820.0, 100.0), (1820.0, 980.0), (100.0, 980.0)]
        pockets = [
            (100.0, 100.0),
            (960.0, 100.0),
            (1820.0, 100.0),
            (100.0, 980.0),
            (960.0, 980.0),
            (1820.0, 980.0),
        ]

        table = TableState.from_calibration(
            corners, camera_resolution=(1920, 1080), pocket_positions_pixel=pockets
        )

        assert len(table.pocket_positions) == 6


# ============================================================================
# GameState Tests
# ============================================================================


class TestGameStateInitialization:
    """Tests for GameState initialization and validation."""

    @pytest.fixture()
    def table(self):
        """Create a standard table for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        return TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

    @pytest.fixture()
    def balls(self):
        """Create a set of test balls."""
        cue_ball = BallState(
            id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(1600.0, 800.0), number=1
        )
        ball2 = BallState(
            id="ball_2", position=Vector2D.from_4k(1700.0, 800.0), number=2
        )
        return [cue_ball, ball1, ball2]

    def test_basic_initialization(self, table, balls):
        """Test basic GameState creation."""
        timestamp = datetime.now().timestamp()
        game_state = GameState(
            timestamp=timestamp, frame_number=1, balls=balls, table=table
        )

        assert game_state.timestamp == timestamp
        assert game_state.frame_number == 1
        assert len(game_state.balls) == 3
        assert game_state.table == table

    def test_initialization_with_all_parameters(self, table, balls):
        """Test GameState with all parameters."""
        timestamp = datetime.now().timestamp()
        game_state = GameState(
            timestamp=timestamp,
            frame_number=1,
            balls=balls,
            table=table,
            game_type=GameType.EIGHT_BALL,
            current_player=1,
            scores={1: 0, 2: 0},
            is_break=True,
            state_confidence=0.95,
        )

        assert game_state.game_type == GameType.EIGHT_BALL
        assert game_state.current_player == 1
        assert game_state.scores == {1: 0, 2: 0}
        assert game_state.is_break is True
        assert game_state.state_confidence == 0.95

    def test_negative_frame_number_raises_error(self, table, balls):
        """Test that negative frame number raises ValueError."""
        with pytest.raises(ValueError, match="Frame number must be non-negative"):
            GameState(
                timestamp=datetime.now().timestamp(),
                frame_number=-1,
                balls=balls,
                table=table,
            )

    def test_negative_timestamp_raises_error(self, table, balls):
        """Test that negative timestamp raises ValueError."""
        with pytest.raises(ValueError, match="Timestamp must be non-negative"):
            GameState(timestamp=-1.0, frame_number=1, balls=balls, table=table)

    def test_state_confidence_out_of_range_raises_error(self, table, balls):
        """Test that state confidence out of range raises ValueError."""
        with pytest.raises(
            ValueError, match="State confidence must be between 0.0 and 1.0"
        ):
            GameState(
                timestamp=datetime.now().timestamp(),
                frame_number=1,
                balls=balls,
                table=table,
                state_confidence=1.5,
            )


class TestGameStateBallRetrieval:
    """Tests for GameState ball retrieval methods."""

    @pytest.fixture()
    def game_state(self):
        """Create a game state for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        cue_ball = BallState(
            id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(1600.0, 800.0), number=1
        )
        ball2 = BallState(
            id="ball_2",
            position=Vector2D.from_4k(1700.0, 800.0),
            number=2,
            is_pocketed=True,
        )
        balls = [cue_ball, ball1, ball2]

        return GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

    def test_get_ball_by_id_exists(self, game_state):
        """Test getting ball by ID that exists."""
        ball = game_state.get_ball_by_id("ball_1")

        assert ball is not None
        assert ball.id == "ball_1"
        assert ball.number == 1

    def test_get_ball_by_id_not_exists(self, game_state):
        """Test getting ball by ID that doesn't exist."""
        ball = game_state.get_ball_by_id("ball_99")

        assert ball is None

    def test_get_cue_ball(self, game_state):
        """Test getting the cue ball."""
        cue_ball = game_state.get_cue_ball()

        assert cue_ball is not None
        assert cue_ball.id == "cue"
        assert cue_ball.is_cue_ball is True

    def test_get_numbered_balls(self, game_state):
        """Test getting all numbered balls."""
        numbered_balls = game_state.get_numbered_balls()

        assert len(numbered_balls) == 2
        assert all(not ball.is_cue_ball for ball in numbered_balls)

    def test_get_pocketed_balls(self, game_state):
        """Test getting pocketed balls."""
        pocketed_balls = game_state.get_pocketed_balls()

        assert len(pocketed_balls) == 1
        assert pocketed_balls[0].id == "ball_2"

    def test_get_active_balls(self, game_state):
        """Test getting active (non-pocketed) balls."""
        active_balls = game_state.get_active_balls()

        assert len(active_balls) == 2
        assert all(not ball.is_pocketed for ball in active_balls)


class TestGameStateMovingBalls:
    """Tests for GameState moving ball detection."""

    @pytest.fixture()
    def game_state_with_moving_balls(self):
        """Create a game state with some moving balls."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        cue_ball = BallState(
            id="cue",
            position=Vector2D.from_4k(800.0, 800.0),
            velocity=Vector2D.from_4k(10.0, 5.0),
            is_cue_ball=True,
        )
        ball1 = BallState(
            id="ball_1",
            position=Vector2D.from_4k(1600.0, 800.0),
            velocity=Vector2D.from_4k(0.0, 0.0),
            number=1,
        )
        ball2 = BallState(
            id="ball_2",
            position=Vector2D.from_4k(1700.0, 800.0),
            velocity=Vector2D.from_4k(2.0, 1.0),
            number=2,
        )
        balls = [cue_ball, ball1, ball2]

        return GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

    def test_get_moving_balls(self, game_state_with_moving_balls):
        """Test getting moving balls."""
        moving_balls = game_state_with_moving_balls.get_moving_balls()

        assert len(moving_balls) == 2
        assert all(ball.id in ["cue", "ball_2"] for ball in moving_balls)

    def test_is_table_clear_with_moving_balls(self, game_state_with_moving_balls):
        """Test table clear detection with moving balls."""
        assert game_state_with_moving_balls.is_table_clear() is False

    def test_is_table_clear_without_moving_balls(self):
        """Test table clear detection when all balls are stationary."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        cue_ball = BallState(
            id="cue",
            position=Vector2D.from_4k(800.0, 800.0),
            velocity=Vector2D.from_4k(0.0, 0.0),
            is_cue_ball=True,
        )
        ball1 = BallState(
            id="ball_1",
            position=Vector2D.from_4k(1600.0, 800.0),
            velocity=Vector2D.from_4k(0.0, 0.0),
            number=1,
        )
        balls = [cue_ball, ball1]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        assert game_state.is_table_clear() is True


class TestGameStateCountBallsByPlayer:
    """Tests for GameState count_balls_by_player method."""

    def test_count_eight_ball_groups(self):
        """Test counting solids and stripes for 8-ball."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        balls = [
            BallState(
                id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
            ),
            # Solids (1-7)
            BallState(id="ball_1", position=Vector2D.from_4k(1000.0, 800.0), number=1),
            BallState(id="ball_2", position=Vector2D.from_4k(1100.0, 800.0), number=2),
            BallState(
                id="ball_3",
                position=Vector2D.from_4k(1200.0, 800.0),
                number=3,
                is_pocketed=True,
            ),
            # Stripes (9-15)
            BallState(id="ball_9", position=Vector2D.from_4k(1300.0, 800.0), number=9),
            BallState(
                id="ball_10", position=Vector2D.from_4k(1400.0, 800.0), number=10
            ),
            BallState(
                id="ball_11",
                position=Vector2D.from_4k(1500.0, 800.0),
                number=11,
                is_pocketed=True,
            ),
            # 8-ball
            BallState(id="ball_8", position=Vector2D.from_4k(1600.0, 800.0), number=8),
        ]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        counts = game_state.count_balls_by_player(GameType.EIGHT_BALL)

        assert counts[1] == 2  # 2 solids remaining (1, 2)
        assert counts[2] == 2  # 2 stripes remaining (9, 10)

    def test_count_nine_ball_remaining(self):
        """Test counting remaining balls for 9-ball."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        balls = [
            BallState(
                id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
            ),
            BallState(id="ball_1", position=Vector2D.from_4k(1000.0, 800.0), number=1),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1100.0, 800.0),
                number=2,
                is_pocketed=True,
            ),
            BallState(id="ball_3", position=Vector2D.from_4k(1200.0, 800.0), number=3),
            BallState(id="ball_9", position=Vector2D.from_4k(1300.0, 800.0), number=9),
        ]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        counts = game_state.count_balls_by_player(GameType.NINE_BALL)

        # Both players have same target (remaining balls 1-9)
        assert counts[1] == 3  # 3 balls remaining (1, 3, 9)
        assert counts[2] == 3


class TestGameStateValidateConsistency:
    """Tests for GameState.validate_consistency method."""

    @pytest.fixture()
    def table(self):
        """Create a standard table for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        return TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

    def test_validate_consistency_valid_state(self, table):
        """Test validation of valid game state."""
        cue_ball = BallState(
            id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(1600.0, 800.0), number=1
        )
        balls = [cue_ball, ball1]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert len(errors) == 0

    def test_validate_consistency_duplicate_ball_ids(self, table):
        """Test validation catches duplicate ball IDs."""
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball2 = BallState(
            id="ball_1", position=Vector2D.from_4k(1600.0, 800.0), number=1
        )  # Same ID
        balls = [ball1, ball2]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert any("Duplicate ball IDs" in error for error in errors)

    def test_validate_consistency_no_cue_ball(self, table):
        """Test validation catches missing cue ball."""
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(800.0, 800.0), number=1
        )
        ball2 = BallState(
            id="ball_2", position=Vector2D.from_4k(1600.0, 800.0), number=2
        )
        balls = [ball1, ball2]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert any("Expected 1 cue ball" in error for error in errors)

    def test_validate_consistency_multiple_cue_balls(self, table):
        """Test validation catches multiple cue balls."""
        cue_ball1 = BallState(
            id="cue1", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        cue_ball2 = BallState(
            id="cue2", position=Vector2D.from_4k(1600.0, 800.0), is_cue_ball=True
        )
        balls = [cue_ball1, cue_ball2]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert any("Expected 1 cue ball, found 2" in error for error in errors)

    def test_validate_consistency_ball_outside_table(self, table):
        """Test validation catches ball outside table bounds."""
        cue_ball = BallState(
            id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball_outside = BallState(
            id="ball_1", position=Vector2D.from_4k(-100.0, 800.0), number=1
        )
        balls = [cue_ball, ball_outside]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert any("outside table bounds" in error for error in errors)

    def test_validate_consistency_overlapping_balls(self, table):
        """Test validation catches overlapping balls."""
        cue_ball = BallState(
            id="cue",
            position=Vector2D.from_4k(800.0, 800.0),
            is_cue_ball=True,
            radius=36.0,
        )
        ball_overlapping = BallState(
            id="ball_1", position=Vector2D.from_4k(810.0, 800.0), number=1, radius=36.0
        )  # Too close
        balls = [cue_ball, ball_overlapping]

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()

        assert any("overlapping" in error for error in errors)


class TestGameStateCreateInitialState:
    """Tests for GameState.create_initial_state factory method."""

    def test_create_initial_state_basic(self):
        """Test creating initial game state."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        game_state = GameState.create_initial_state(table)

        assert game_state.frame_number == 0
        assert len(game_state.balls) == 16  # 1 cue ball + 15 numbered balls
        assert game_state.is_break is True

    def test_create_initial_state_has_cue_ball(self):
        """Test initial state has exactly one cue ball."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        game_state = GameState.create_initial_state(table)

        cue_ball = game_state.get_cue_ball()
        assert cue_ball is not None
        assert cue_ball.is_cue_ball is True

    def test_create_initial_state_numbered_balls(self):
        """Test initial state has 15 numbered balls."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        game_state = GameState.create_initial_state(table)

        numbered_balls = game_state.get_numbered_balls()
        assert len(numbered_balls) == 15

        # Check all numbers 1-15 are present
        numbers = {ball.number for ball in numbered_balls}
        assert numbers == set(range(1, 16))

    def test_create_initial_state_with_game_type(self):
        """Test creating initial state with specific game type."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        game_state = GameState.create_initial_state(
            table, game_type=GameType.EIGHT_BALL
        )

        assert game_state.game_type == GameType.EIGHT_BALL


class TestGameStateSerialization:
    """Tests for GameState serialization."""

    @pytest.fixture()
    def game_state(self):
        """Create a game state for testing."""
        pockets = [Vector2D.from_4k(x, y) for x in [0, 1600, 3200] for y in [0, 1600]]
        table = TableState(width=3200.0, height=1600.0, pocket_positions=pockets)

        cue_ball = BallState(
            id="cue", position=Vector2D.from_4k(800.0, 800.0), is_cue_ball=True
        )
        ball1 = BallState(
            id="ball_1", position=Vector2D.from_4k(1600.0, 800.0), number=1
        )
        balls = [cue_ball, ball1]

        return GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
            game_type=GameType.EIGHT_BALL,
        )

    def test_to_dict_basic(self, game_state):
        """Test basic to_dict serialization."""
        data = game_state.to_dict()

        assert "timestamp" in data
        assert "frame_number" in data
        assert "balls" in data
        assert "table" in data
        assert "game_type" in data

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        data = {
            "timestamp": datetime.now().timestamp(),
            "frame_number": 1,
            "balls": [
                {
                    "id": "cue",
                    "position": {"x": 800.0, "y": 800.0, "scale": [1.0, 1.0]},
                    "velocity": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
                    "spin": {"x": 0.0, "y": 0.0, "scale": [1.0, 1.0]},
                    "is_cue_ball": True,
                }
            ],
            "table": {
                "width": 3200.0,
                "height": 1600.0,
                "pocket_positions": [
                    {"x": float(x), "y": float(y), "scale": [1.0, 1.0]}
                    for x in [0, 1600, 3200]
                    for y in [0, 1600]
                ],
            },
        }

        game_state = GameState.from_dict(data)

        assert game_state.frame_number == 1
        assert len(game_state.balls) == 1

    def test_serialization_round_trip(self, game_state):
        """Test that to_dict -> from_dict preserves data."""
        data = game_state.to_dict()
        game_state_restored = GameState.from_dict(data)

        assert game_state_restored.frame_number == game_state.frame_number
        assert len(game_state_restored.balls) == len(game_state.balls)
        assert game_state_restored.game_type == game_state.game_type


# ============================================================================
# Additional Model Tests
# ============================================================================


class TestCueState:
    """Tests for CueState model."""

    def test_cue_state_initialization(self):
        """Test CueState initialization."""
        tip_position = Vector2D.from_4k(800.0, 800.0)
        cue = CueState(tip_position=tip_position, angle=45.0)

        assert cue.tip_position.x == 800.0
        assert cue.angle == 45.0
        assert cue.elevation == 0.0

    def test_cue_state_validation_negative_length(self):
        """Test CueState validation with negative length."""
        tip_position = Vector2D.from_4k(800.0, 800.0)

        with pytest.raises(ValueError, match="Cue length must be positive"):
            CueState(tip_position=tip_position, angle=0.0, length=-1.0)

    def test_cue_state_get_direction_vector(self):
        """Test CueState direction vector calculation."""
        tip_position = Vector2D.from_4k(800.0, 800.0)
        cue = CueState(tip_position=tip_position, angle=0.0)  # Pointing right

        direction = cue.get_direction_vector()

        assert direction.x == pytest.approx(1.0)
        assert direction.y == pytest.approx(0.0)

    def test_cue_state_from_4k(self):
        """Test CueState.from_4k factory method."""
        cue = CueState.from_4k(800.0, 800.0, angle=45.0)

        assert cue.tip_position.x == 800.0
        assert cue.tip_position.y == 800.0
        assert cue.tip_position.scale == (1.0, 1.0)
        assert cue.angle == 45.0


class TestGameEvent:
    """Tests for GameEvent model."""

    def test_game_event_initialization(self):
        """Test GameEvent initialization."""
        timestamp = datetime.now().timestamp()
        event = GameEvent(
            timestamp=timestamp,
            event_type="shot",
            description="Shot taken",
            data={"force": 10.0},
            frame_number=1,
        )

        assert event.timestamp == timestamp
        assert event.event_type == "shot"
        assert event.description == "Shot taken"
        assert event.data == {"force": 10.0}
        assert event.frame_number == 1

    def test_game_event_serialization(self):
        """Test GameEvent serialization."""
        timestamp = datetime.now().timestamp()
        event = GameEvent(
            timestamp=timestamp,
            event_type="shot",
            description="Shot taken",
            frame_number=1,
        )

        data = event.to_dict()
        event_restored = GameEvent.from_dict(data)

        assert event_restored.timestamp == event.timestamp
        assert event_restored.event_type == event.event_type
        assert event_restored.description == event.description


class TestCollision:
    """Tests for Collision model."""

    def test_collision_initialization(self):
        """Test Collision initialization."""
        position = Vector2D.from_4k(800.0, 800.0)
        collision = Collision(
            time=1.0, position=position, ball1_id="cue", ball2_id="ball_1", type="ball"
        )

        assert collision.time == 1.0
        assert collision.ball1_id == "cue"
        assert collision.ball2_id == "ball_1"
        assert collision.type == "ball"

    def test_collision_validation_invalid_type(self):
        """Test Collision validation with invalid type."""
        position = Vector2D.from_4k(800.0, 800.0)

        with pytest.raises(ValueError, match="Collision type must be one of"):
            Collision(time=1.0, position=position, ball1_id="cue", type="invalid")

    def test_collision_validation_ball_requires_ball2_id(self):
        """Test Collision validation for ball collision."""
        position = Vector2D.from_4k(800.0, 800.0)

        with pytest.raises(ValueError, match="Ball collisions must have ball2_id"):
            Collision(time=1.0, position=position, ball1_id="cue", type="ball")

    def test_collision_is_ball_collision(self):
        """Test Collision.is_ball_collision method."""
        position = Vector2D.from_4k(800.0, 800.0)
        collision = Collision(
            time=1.0, position=position, ball1_id="cue", ball2_id="ball_1", type="ball"
        )

        assert collision.is_ball_collision() is True

    def test_collision_is_cushion_collision(self):
        """Test Collision.is_cushion_collision method."""
        position = Vector2D.from_4k(800.0, 800.0)
        collision = Collision(
            time=1.0, position=position, ball1_id="cue", type="cushion"
        )

        assert collision.is_cushion_collision() is True


class TestTrajectory:
    """Tests for Trajectory model."""

    def test_trajectory_initialization(self):
        """Test Trajectory initialization."""
        points = [Vector2D.from_4k(float(x), 800.0) for x in range(0, 1000, 100)]
        final_position = Vector2D.from_4k(1000.0, 800.0)
        final_velocity = Vector2D.from_4k(0.0, 0.0)

        trajectory = Trajectory(
            ball_id="cue",
            points=points,
            collisions=[],
            final_position=final_position,
            final_velocity=final_velocity,
            time_to_rest=5.0,
            will_be_pocketed=False,
        )

        assert trajectory.ball_id == "cue"
        assert len(trajectory.points) == 10
        assert trajectory.time_to_rest == 5.0
        assert trajectory.will_be_pocketed is False

    def test_trajectory_validation_negative_time(self):
        """Test Trajectory validation with negative time."""
        points = [Vector2D.from_4k(0.0, 0.0)]
        final_position = Vector2D.from_4k(0.0, 0.0)
        final_velocity = Vector2D.from_4k(0.0, 0.0)

        with pytest.raises(ValueError, match="Time to rest must be non-negative"):
            Trajectory(
                ball_id="cue",
                points=points,
                collisions=[],
                final_position=final_position,
                final_velocity=final_velocity,
                time_to_rest=-1.0,
                will_be_pocketed=False,
            )

    def test_trajectory_validation_pocketed_without_pocket_id(self):
        """Test Trajectory validation for pocketed ball without pocket_id."""
        points = [Vector2D.from_4k(0.0, 0.0)]
        final_position = Vector2D.from_4k(0.0, 0.0)
        final_velocity = Vector2D.from_4k(0.0, 0.0)

        with pytest.raises(ValueError, match="Pocketed balls must have a pocket_id"):
            Trajectory(
                ball_id="cue",
                points=points,
                collisions=[],
                final_position=final_position,
                final_velocity=final_velocity,
                time_to_rest=1.0,
                will_be_pocketed=True,
            )


class TestShotAnalysis:
    """Tests for ShotAnalysis model."""

    def test_shot_analysis_initialization(self):
        """Test ShotAnalysis initialization."""
        shot = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=10.0,
            recommended_angle=45.0,
        )

        assert shot.shot_type == ShotType.DIRECT
        assert shot.difficulty == 0.5
        assert shot.success_probability == 0.8
        assert shot.recommended_force == 10.0

    def test_shot_analysis_validation_difficulty_out_of_range(self):
        """Test ShotAnalysis validation with difficulty out of range."""
        with pytest.raises(ValueError, match="Difficulty must be between 0.0 and 1.0"):
            ShotAnalysis(
                shot_type=ShotType.DIRECT,
                difficulty=1.5,
                success_probability=0.8,
                recommended_force=10.0,
                recommended_angle=45.0,
            )

    def test_shot_analysis_is_safe_shot(self):
        """Test ShotAnalysis.is_safe_shot method."""
        shot = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=10.0,
            recommended_angle=45.0,
        )

        assert shot.is_safe_shot(safety_threshold=0.7) is True
        assert shot.is_safe_shot(safety_threshold=0.9) is False

    def test_shot_analysis_get_risk_score(self):
        """Test ShotAnalysis.get_risk_score method."""
        shot = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=10.0,
            recommended_angle=45.0,
        )

        risk_score = shot.get_risk_score()

        assert 0.0 <= risk_score <= 1.0


class TestCoordinateMetadata:
    """Tests for CoordinateMetadata model."""

    def test_coordinate_metadata_initialization(self):
        """Test CoordinateMetadata initialization."""
        metadata = CoordinateMetadata(
            camera_resolution=(1920, 1080),
            table_bounds=(100.0, 100.0, 1820.0, 980.0),
            coordinate_space="camera_pixels",
            pixels_per_meter=100.0,
        )

        assert metadata.camera_resolution == (1920, 1080)
        assert metadata.coordinate_space == "camera_pixels"
        assert metadata.pixels_per_meter == 100.0

    def test_coordinate_metadata_serialization(self):
        """Test CoordinateMetadata serialization."""
        metadata = CoordinateMetadata(camera_resolution=(1920, 1080))

        data = metadata.to_dict()
        metadata_restored = CoordinateMetadata.from_dict(data)

        assert metadata_restored.camera_resolution == metadata.camera_resolution


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions in models module."""

    def test_serialize_to_json_ball_state(self):
        """Test serialize_to_json with BallState."""
        position = Vector2D.from_4k(100.0, 200.0)
        ball = BallState(id="test", position=position, number=1)

        json_str = serialize_to_json(ball)

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["id"] == "test"

    def test_deserialize_from_json_ball_state(self):
        """Test deserialize_from_json with BallState."""
        position = Vector2D.from_4k(100.0, 200.0)
        ball = BallState(id="test", position=position, number=1)

        json_str = serialize_to_json(ball)
        ball_restored = deserialize_from_json(json_str, BallState)

        assert ball_restored.id == ball.id
        assert ball_restored.position.x == ball.position.x

    def test_calculate_ball_separation(self):
        """Test calculate_ball_separation function."""
        pos1 = Vector2D.from_4k(0.0, 0.0)
        pos2 = Vector2D.from_4k(100.0, 0.0)
        ball1 = BallState(id="ball1", position=pos1, radius=36.0)
        ball2 = BallState(id="ball2", position=pos2, radius=36.0)

        separation = calculate_ball_separation(ball1, ball2)

        assert separation == pytest.approx(28.0)  # 100 - 36 - 36

    def test_find_closest_ball(self):
        """Test find_closest_ball function."""
        target = BallState(id="target", position=Vector2D.from_4k(0.0, 0.0))
        ball1 = BallState(id="ball1", position=Vector2D.from_4k(100.0, 0.0))
        ball2 = BallState(id="ball2", position=Vector2D.from_4k(200.0, 0.0))
        ball3 = BallState(id="ball3", position=Vector2D.from_4k(50.0, 0.0))

        closest = find_closest_ball(target, [ball1, ball2, ball3])

        assert closest is not None
        assert closest.id == "ball3"

    def test_find_closest_ball_empty_list(self):
        """Test find_closest_ball with empty list."""
        target = BallState(id="target", position=Vector2D.from_4k(0.0, 0.0))

        closest = find_closest_ball(target, [])

        assert closest is None

    def test_validate_ball_physics_valid(self):
        """Test validate_ball_physics with valid ball."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(5.0, 5.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        errors = validate_ball_physics(ball, max_velocity=10.0)

        assert len(errors) == 0

    def test_validate_ball_physics_excessive_velocity(self):
        """Test validate_ball_physics with excessive velocity."""
        position = Vector2D.from_4k(100.0, 100.0)
        velocity = Vector2D.from_4k(15.0, 15.0)
        ball = BallState(id="test", position=position, velocity=velocity)

        errors = validate_ball_physics(ball, max_velocity=10.0)

        assert len(errors) > 0
        assert any("velocity" in error for error in errors)

    def test_create_standard_ball_set(self):
        """Test create_standard_ball_set function."""
        balls = create_standard_ball_set()

        assert len(balls) == 16  # 1 cue ball + 15 numbered balls

        # Check for cue ball
        cue_balls = [b for b in balls if b.is_cue_ball]
        assert len(cue_balls) == 1

        # Check for numbered balls
        numbered_balls = [b for b in balls if not b.is_cue_ball]
        assert len(numbered_balls) == 15
        numbers = {b.number for b in numbered_balls}
        assert numbers == set(range(1, 16))
