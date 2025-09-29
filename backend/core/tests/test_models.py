"""Comprehensive unit tests for core data models.

This module tests all the data models defined in the core module,
including validation, serialization, and utility methods.
"""

import math
import unittest
from datetime import datetime

import pytest

from ..models import (
    BallState,
    Collision,
    CueState,
    GameState,
    GameType,
    ShotAnalysis,
    ShotType,
    TableState,
    Trajectory,
    Vector2D,
    calculate_ball_separation,
    create_standard_ball_set,
    deserialize_from_json,
    find_closest_ball,
    serialize_to_json,
    validate_ball_physics,
)


class TestVector2D(unittest.TestCase):
    """Test Vector2D class functionality."""

    def test_creation(self):
        """Test vector creation."""
        v = Vector2D(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_magnitude(self):
        """Test magnitude calculation."""
        v = Vector2D(3.0, 4.0)
        self.assertAlmostEqual(v.magnitude(), 5.0)

        v_zero = Vector2D.zero()
        assert v_zero.magnitude() == 0.0

    def test_magnitude_squared(self):
        """Test squared magnitude calculation."""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude_squared() == 25.0

    def test_normalize(self):
        """Test vector normalization."""
        v = Vector2D(3.0, 4.0)
        normalized = v.normalize()
        self.assertAlmostEqual(normalized.magnitude(), 1.0)

        # Test zero vector normalization
        v_zero = Vector2D.zero()
        normalized_zero = v_zero.normalize()
        assert normalized_zero.x == 0.0
        assert normalized_zero.y == 0.0

    def test_dot_product(self):
        """Test dot product calculation."""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        assert v1.dot(v2) == 11.0

    def test_cross_product(self):
        """Test 2D cross product calculation."""
        v1 = Vector2D(1.0, 0.0)
        v2 = Vector2D(0.0, 1.0)
        assert v1.cross(v2) == 1.0

    def test_distance_to(self):
        """Test distance calculation."""
        v1 = Vector2D(0.0, 0.0)
        v2 = Vector2D(3.0, 4.0)
        self.assertAlmostEqual(v1.distance_to(v2), 5.0)

    def test_angle_to(self):
        """Test angle calculation."""
        v1 = Vector2D(0.0, 0.0)
        v2 = Vector2D(1.0, 0.0)
        self.assertAlmostEqual(v1.angle_to(v2), 0.0)

        v3 = Vector2D(0.0, 1.0)
        self.assertAlmostEqual(v1.angle_to(v3), math.pi / 2)

    def test_rotate(self):
        """Test vector rotation."""
        v = Vector2D(1.0, 0.0)
        rotated = v.rotate(math.pi / 2)
        self.assertAlmostEqual(rotated.x, 0.0, places=10)
        self.assertAlmostEqual(rotated.y, 1.0, places=10)

    def test_arithmetic_operations(self):
        """Test arithmetic operators."""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)

        # Addition
        v_add = v1 + v2
        assert v_add.x == 4.0
        assert v_add.y == 6.0

        # Subtraction
        v_sub = v2 - v1
        assert v_sub.x == 2.0
        assert v_sub.y == 2.0

        # Scalar multiplication
        v_mul = v1 * 2.0
        assert v_mul.x == 2.0
        assert v_mul.y == 4.0

        # Scalar division
        v_div = v1 / 2.0
        assert v_div.x == 0.5
        assert v_div.y == 1.0

        # Negation
        v_neg = -v1
        assert v_neg.x == -1.0
        assert v_neg.y == -2.0

    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        v = Vector2D(3.0, 4.0)
        data = v.to_dict()
        assert data == {"x": 3.0, "y": 4.0}

        v_restored = Vector2D.from_dict(data)
        assert v_restored.x == 3.0
        assert v_restored.y == 4.0

    def test_class_methods(self):
        """Test class factory methods."""
        v_zero = Vector2D.zero()
        assert v_zero.x == 0.0
        assert v_zero.y == 0.0

        v_unit_x = Vector2D.unit_x()
        assert v_unit_x.x == 1.0
        assert v_unit_x.y == 0.0

        v_unit_y = Vector2D.unit_y()
        assert v_unit_y.x == 0.0
        assert v_unit_y.y == 1.0


class TestBallState(unittest.TestCase):
    """Test BallState class functionality."""

    def test_creation(self):
        """Test ball state creation."""
        pos = Vector2D(1.0, 2.0)
        ball = BallState(id="test_ball", position=pos)
        assert ball.id == "test_ball"
        assert ball.position == pos
        assert not ball.is_cue_ball
        assert not ball.is_pocketed

    def test_validation(self):
        """Test ball state validation."""
        pos = Vector2D(1.0, 2.0)

        # Invalid radius
        with pytest.raises(ValueError):
            BallState(id="test", position=pos, radius=-1.0)

        # Invalid mass
        with pytest.raises(ValueError):
            BallState(id="test", position=pos, mass=-1.0)

        # Invalid confidence
        with pytest.raises(ValueError):
            BallState(id="test", position=pos, confidence=1.5)

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        pos = Vector2D(0.0, 0.0)
        vel = Vector2D(1.0, 0.0)
        spin = Vector2D(1.0, 0.0)

        ball = BallState(id="test", position=pos, velocity=vel, spin=spin)
        ke = ball.kinetic_energy()
        assert ke > 0.0

    def test_movement_detection(self):
        """Test movement and spin detection."""
        pos = Vector2D(0.0, 0.0)

        # Stationary ball
        ball = BallState(id="test", position=pos)
        assert not ball.is_moving()
        assert not ball.is_spinning()

        # Moving ball
        vel = Vector2D(1.0, 0.0)
        moving_ball = BallState(id="test", position=pos, velocity=vel)
        assert moving_ball.is_moving()

        # Spinning ball
        spin = Vector2D(1.0, 0.0)
        spinning_ball = BallState(id="test", position=pos, spin=spin)
        assert spinning_ball.is_spinning()

    def test_ball_interactions(self):
        """Test ball-to-ball interactions."""
        pos1 = Vector2D(0.0, 0.0)
        pos2 = Vector2D(0.1, 0.0)  # Close but not touching

        ball1 = BallState(id="ball1", position=pos1)
        ball2 = BallState(id="ball2", position=pos2)

        distance = ball1.distance_to(ball2)
        self.assertAlmostEqual(distance, 0.1)

        # Not touching (standard ball radius is ~0.0286m)
        assert not ball1.is_touching(ball2)

        # Make them touch
        pos3 = Vector2D(0.057, 0.0)  # Exactly touching
        ball3 = BallState(id="ball3", position=pos3)
        assert ball1.is_touching(ball3, tolerance=0.001)

    def test_copy(self):
        """Test ball state copying."""
        pos = Vector2D(1.0, 2.0)
        vel = Vector2D(0.5, 0.0)
        spin = Vector2D(0.1, 0.0)

        ball = BallState(
            id="test",
            position=pos,
            velocity=vel,
            spin=spin,
            number=8,
            is_cue_ball=False,
        )

        ball_copy = ball.copy()
        assert ball_copy.id == ball.id
        assert ball_copy.position.x == ball.position.x
        assert ball_copy.number == ball.number

        # Ensure it's a deep copy
        ball_copy.position.x = 5.0
        assert ball_copy.position.x != ball.position.x

    def test_serialization(self):
        """Test ball state serialization."""
        pos = Vector2D(1.0, 2.0)
        ball = BallState(id="test", position=pos, number=8)

        data = ball.to_dict()
        assert "id" in data
        assert "position" in data
        assert "number" in data

        ball_restored = BallState.from_dict(data)
        assert ball_restored.id == ball.id
        assert ball_restored.number == ball.number


class TestTableState(unittest.TestCase):
    """Test TableState class functionality."""

    def test_creation(self):
        """Test table state creation."""
        pockets = [
            Vector2D(0, 0),
            Vector2D(1, 0),
            Vector2D(2, 0),
            Vector2D(0, 1),
            Vector2D(1, 1),
            Vector2D(2, 1),
        ]
        table = TableState(width=2.0, height=1.0, pocket_positions=pockets)
        assert table.width == 2.0
        assert table.height == 1.0
        assert len(table.pocket_positions) == 6

    def test_validation(self):
        """Test table state validation."""
        pockets = [Vector2D(0, 0)] * 6  # 6 pockets required

        # Invalid dimensions
        with pytest.raises(ValueError):
            TableState(width=-1.0, height=1.0, pocket_positions=pockets)

        # Invalid pocket count
        with pytest.raises(ValueError):
            TableState(width=2.0, height=1.0, pocket_positions=[Vector2D(0, 0)])

    def test_legacy_conversion(self):
        """Test conversion from legacy mm measurements."""
        pockets = [Vector2D(0, 0)] * 6

        # Test conversion from mm to meters
        table = TableState(width=2540.0, height=1270.0, pocket_positions=pockets)
        self.assertAlmostEqual(table.width, 2.54)
        self.assertAlmostEqual(table.height, 1.27)

    def test_point_in_pocket(self):
        """Test pocket detection."""
        pockets = [
            Vector2D(0, 0),
            Vector2D(1, 0),
            Vector2D(2, 0),
            Vector2D(0, 1),
            Vector2D(1, 1),
            Vector2D(2, 1),
        ]
        table = TableState(width=2.0, height=1.0, pocket_positions=pockets)

        # Point in pocket
        in_pocket, pocket_id = table.is_point_in_pocket(Vector2D(0.01, 0.01))
        assert in_pocket
        assert pocket_id == 0

        # Point not in pocket
        not_in_pocket, _ = table.is_point_in_pocket(Vector2D(0.5, 0.5))
        assert not not_in_pocket

    def test_point_on_table(self):
        """Test table bounds checking."""
        pockets = [Vector2D(0, 0)] * 6
        table = TableState(width=2.0, height=1.0, pocket_positions=pockets)

        # Point on table
        assert table.is_point_on_table(Vector2D(1.0, 0.5))

        # Point off table
        assert not table.is_point_on_table(Vector2D(-0.1, 0.5))
        assert not table.is_point_on_table(Vector2D(2.1, 0.5))

    def test_closest_cushion(self):
        """Test closest cushion calculation."""
        pockets = [Vector2D(0, 0)] * 6
        table = TableState(width=2.0, height=1.0, pocket_positions=pockets)

        # Point closer to left cushion
        cushion, distance, normal = table.get_closest_cushion(Vector2D(0.1, 0.5))
        assert cushion == "left"
        self.assertAlmostEqual(distance, 0.1)
        assert normal.x == 1.0
        assert normal.y == 0.0

    def test_standard_table(self):
        """Test standard table creation."""
        table = TableState.standard_9ft_table()
        self.assertAlmostEqual(table.width, 2.54)
        self.assertAlmostEqual(table.height, 1.27)
        assert len(table.pocket_positions) == 6

    def test_serialization(self):
        """Test table state serialization."""
        table = TableState.standard_9ft_table()
        data = table.to_dict()

        table_restored = TableState.from_dict(data)
        self.assertAlmostEqual(table_restored.width, table.width)
        self.assertAlmostEqual(table_restored.height, table.height)


class TestCueState(unittest.TestCase):
    """Test CueState class functionality."""

    def test_creation(self):
        """Test cue state creation."""
        tip_pos = Vector2D(1.0, 2.0)
        cue = CueState(tip_position=tip_pos, angle=45.0)
        assert cue.tip_position == tip_pos
        assert cue.angle == 45.0

    def test_validation(self):
        """Test cue state validation."""
        tip_pos = Vector2D(1.0, 2.0)

        # Invalid length
        with pytest.raises(ValueError):
            CueState(tip_position=tip_pos, angle=0.0, length=-1.0)

        # Invalid confidence
        with pytest.raises(ValueError):
            CueState(tip_position=tip_pos, angle=0.0, confidence=1.5)

    def test_direction_vector(self):
        """Test direction vector calculation."""
        tip_pos = Vector2D(0.0, 0.0)

        # Pointing right (0 degrees)
        cue = CueState(tip_position=tip_pos, angle=0.0)
        direction = cue.get_direction_vector()
        self.assertAlmostEqual(direction.x, 1.0)
        self.assertAlmostEqual(direction.y, 0.0)

        # Pointing up (90 degrees)
        cue_up = CueState(tip_position=tip_pos, angle=90.0)
        direction_up = cue_up.get_direction_vector()
        self.assertAlmostEqual(direction_up.x, 0.0, places=10)
        self.assertAlmostEqual(direction_up.y, 1.0)

    def test_aim_line(self):
        """Test aim line calculation."""
        tip_pos = Vector2D(0.0, 0.0)
        cue = CueState(tip_position=tip_pos, angle=0.0)

        start, end = cue.get_aim_line(length=1.0)
        assert start == tip_pos
        self.assertAlmostEqual(end.x, 1.0)
        self.assertAlmostEqual(end.y, 0.0)

    def test_impact_velocity(self):
        """Test impact velocity calculation."""
        tip_pos = Vector2D(0.0, 0.0)
        cue = CueState(tip_position=tip_pos, angle=0.0, estimated_force=10.0)

        velocity = cue.calculate_impact_velocity()
        assert velocity.magnitude() > 0.0

    def test_serialization(self):
        """Test cue state serialization."""
        tip_pos = Vector2D(1.0, 2.0)
        cue = CueState(tip_position=tip_pos, angle=45.0)

        data = cue.to_dict()
        cue_restored = CueState.from_dict(data)
        assert cue_restored.angle == cue.angle


class TestCollision(unittest.TestCase):
    """Test Collision class functionality."""

    def test_creation(self):
        """Test collision creation."""
        pos = Vector2D(1.0, 2.0)
        collision = Collision(
            time=1.5, position=pos, ball1_id="ball1", ball2_id="ball2", type="ball"
        )
        assert collision.time == 1.5
        assert collision.ball1_id == "ball1"
        assert collision.ball2_id == "ball2"

    def test_validation(self):
        """Test collision validation."""
        pos = Vector2D(1.0, 2.0)

        # Invalid type
        with pytest.raises(ValueError):
            Collision(time=1.0, position=pos, ball1_id="ball1", type="invalid")

        # Ball collision without ball2_id
        with pytest.raises(ValueError):
            Collision(time=1.0, position=pos, ball1_id="ball1", type="ball")

    def test_collision_types(self):
        """Test collision type detection."""
        pos = Vector2D(1.0, 2.0)

        # Ball collision
        ball_collision = Collision(
            time=1.0, position=pos, ball1_id="ball1", ball2_id="ball2", type="ball"
        )
        assert ball_collision.is_ball_collision()
        assert not ball_collision.is_cushion_collision()
        assert not ball_collision.is_pocket_collision()

        # Cushion collision
        cushion_collision = Collision(
            time=1.0, position=pos, ball1_id="ball1", type="cushion"
        )
        assert not cushion_collision.is_ball_collision()
        assert cushion_collision.is_cushion_collision()

    def test_involved_balls(self):
        """Test involved balls detection."""
        pos = Vector2D(1.0, 2.0)
        collision = Collision(
            time=1.0, position=pos, ball1_id="ball1", ball2_id="ball2", type="ball"
        )

        balls = collision.get_involved_balls()
        assert "ball1" in balls
        assert "ball2" in balls
        assert len(balls) == 2

    def test_serialization(self):
        """Test collision serialization."""
        pos = Vector2D(1.0, 2.0)
        collision = Collision(
            time=1.0, position=pos, ball1_id="ball1", ball2_id="ball2", type="ball"
        )

        data = collision.to_dict()
        collision_restored = Collision.from_dict(data)
        assert collision_restored.time == collision.time
        assert collision_restored.ball1_id == collision.ball1_id


class TestTrajectory(unittest.TestCase):
    """Test Trajectory class functionality."""

    def test_creation(self):
        """Test trajectory creation."""
        points = [Vector2D(0, 0), Vector2D(1, 0), Vector2D(2, 0)]
        final_pos = Vector2D(2, 0)
        final_vel = Vector2D(0, 0)

        trajectory = Trajectory(
            ball_id="ball1",
            points=points,
            collisions=[],
            final_position=final_pos,
            final_velocity=final_vel,
            time_to_rest=2.0,
            will_be_pocketed=False,
        )

        assert trajectory.ball_id == "ball1"
        assert len(trajectory.points) == 3
        assert not trajectory.will_be_pocketed

    def test_validation(self):
        """Test trajectory validation."""
        points = [Vector2D(0, 0)]
        final_pos = Vector2D(1, 0)
        final_vel = Vector2D(0, 0)

        # Invalid time
        with pytest.raises(ValueError):
            Trajectory(
                ball_id="ball1",
                points=points,
                collisions=[],
                final_position=final_pos,
                final_velocity=final_vel,
                time_to_rest=-1.0,
                will_be_pocketed=False,
            )

        # Pocketed without pocket_id
        with pytest.raises(ValueError):
            Trajectory(
                ball_id="ball1",
                points=points,
                collisions=[],
                final_position=final_pos,
                final_velocity=final_vel,
                time_to_rest=1.0,
                will_be_pocketed=True,
            )

    def test_position_at_time(self):
        """Test position interpolation."""
        points = [Vector2D(0, 0), Vector2D(1, 0), Vector2D(2, 0)]
        final_pos = Vector2D(2, 0)
        final_vel = Vector2D(0, 0)

        trajectory = Trajectory(
            ball_id="ball1",
            points=points,
            collisions=[],
            final_position=final_pos,
            final_velocity=final_vel,
            time_to_rest=2.0,
            will_be_pocketed=False,
        )

        # Position at start
        pos_start = trajectory.get_position_at_time(0.0)
        assert pos_start is not None
        self.assertAlmostEqual(pos_start.x, 0.0)

        # Position at end
        pos_end = trajectory.get_position_at_time(2.0)
        assert pos_end is not None
        self.assertAlmostEqual(pos_end.x, 2.0)

        # Position beyond time
        pos_beyond = trajectory.get_position_at_time(3.0)
        assert pos_beyond is None

    def test_collision_filtering(self):
        """Test collision time filtering."""
        pos = Vector2D(1.0, 0.0)
        collision1 = Collision(time=0.5, position=pos, ball1_id="ball1", type="cushion")
        collision2 = Collision(time=1.5, position=pos, ball1_id="ball1", type="cushion")

        trajectory = Trajectory(
            ball_id="ball1",
            points=[Vector2D(0, 0), Vector2D(2, 0)],
            collisions=[collision1, collision2],
            final_position=Vector2D(2, 0),
            final_velocity=Vector2D(0, 0),
            time_to_rest=2.0,
            will_be_pocketed=False,
        )

        early_collisions = trajectory.get_collisions_before_time(1.0)
        assert len(early_collisions) == 1
        assert early_collisions[0].time == 0.5

    def test_serialization(self):
        """Test trajectory serialization."""
        points = [Vector2D(0, 0), Vector2D(1, 0)]
        trajectory = Trajectory(
            ball_id="ball1",
            points=points,
            collisions=[],
            final_position=Vector2D(1, 0),
            final_velocity=Vector2D(0, 0),
            time_to_rest=1.0,
            will_be_pocketed=False,
        )

        data = trajectory.to_dict()
        trajectory_restored = Trajectory.from_dict(data)
        assert trajectory_restored.ball_id == trajectory.ball_id
        assert len(trajectory_restored.points) == len(trajectory.points)


class TestShotAnalysis(unittest.TestCase):
    """Test ShotAnalysis class functionality."""

    def test_creation(self):
        """Test shot analysis creation."""
        analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        assert analysis.shot_type == ShotType.DIRECT
        assert analysis.difficulty == 0.5
        assert analysis.success_probability == 0.8

    def test_validation(self):
        """Test shot analysis validation."""
        # Invalid difficulty
        with pytest.raises(ValueError):
            ShotAnalysis(
                shot_type=ShotType.DIRECT,
                difficulty=1.5,
                success_probability=0.8,
                recommended_force=5.0,
                recommended_angle=45.0,
            )

        # Invalid success probability
        with pytest.raises(ValueError):
            ShotAnalysis(
                shot_type=ShotType.DIRECT,
                difficulty=0.5,
                success_probability=1.5,
                recommended_force=5.0,
                recommended_angle=45.0,
            )

    def test_safety_assessment(self):
        """Test safety and risk assessment."""
        safe_shot = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.3,
            success_probability=0.9,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        assert safe_shot.is_safe_shot()
        assert not safe_shot.is_high_risk()

        risky_shot = ShotAnalysis(
            shot_type=ShotType.MASSE,
            difficulty=0.9,
            success_probability=0.2,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        assert not risky_shot.is_safe_shot()
        assert risky_shot.is_high_risk()

    def test_risk_score(self):
        """Test risk score calculation."""
        analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        risk_score = analysis.get_risk_score()
        assert risk_score >= 0.0
        assert risk_score <= 1.0

    def test_problem_management(self):
        """Test problem and risk management."""
        analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        analysis.add_problem("Ball in the way")
        analysis.add_problem("Difficult angle")
        assert len(analysis.potential_problems) == 2

        # Adding duplicate problem
        analysis.add_problem("Ball in the way")
        assert len(analysis.potential_problems) == 2

        analysis.add_risk("scratch", 0.3)
        assert analysis.risk_assessment["scratch"] == 0.3

    def test_serialization(self):
        """Test shot analysis serialization."""
        analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.5,
            success_probability=0.8,
            recommended_force=5.0,
            recommended_angle=45.0,
        )

        data = analysis.to_dict()
        analysis_restored = ShotAnalysis.from_dict(data)
        assert analysis_restored.shot_type == analysis.shot_type
        assert analysis_restored.difficulty == analysis.difficulty


class TestGameState(unittest.TestCase):
    """Test GameState class functionality."""

    def test_creation(self):
        """Test game state creation."""
        timestamp = datetime.now().timestamp()
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        game_state = GameState(
            timestamp=timestamp, frame_number=1, balls=balls, table=table
        )

        assert game_state.frame_number == 1
        assert len(game_state.balls) == 16  # 15 numbered + cue

    def test_validation(self):
        """Test game state validation."""
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        # Invalid frame number
        with pytest.raises(ValueError):
            GameState(timestamp=1234567890.0, frame_number=-1, balls=balls, table=table)

    def test_ball_retrieval(self):
        """Test ball retrieval methods."""
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        # Get cue ball
        cue_ball = game_state.get_cue_ball()
        assert cue_ball is not None
        assert cue_ball.is_cue_ball

        # Get ball by ID
        ball1 = game_state.get_ball_by_id("ball_1")
        assert ball1 is not None
        assert ball1.number == 1

        # Get numbered balls
        numbered_balls = game_state.get_numbered_balls()
        assert len(numbered_balls) == 15

        # Get active balls (none pocketed initially)
        active_balls = game_state.get_active_balls()
        assert len(active_balls) == 16

    def test_movement_detection(self):
        """Test movement detection."""
        balls = create_standard_ball_set()

        # Add velocity to one ball
        balls[1].velocity = Vector2D(1.0, 0.0)

        table = TableState.standard_9ft_table()
        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        moving_balls = game_state.get_moving_balls()
        assert len(moving_balls) == 1

        assert not game_state.is_table_clear()

    def test_ball_counting(self):
        """Test ball counting by game type."""
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
            game_type=GameType.EIGHT_BALL,
        )

        counts = game_state.count_balls_by_player(GameType.EIGHT_BALL)
        assert counts[1] == 7  # Solids (1-7)
        assert counts[2] == 7  # Stripes (9-15)

    def test_event_management(self):
        """Test event management."""
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        game_state.add_event("Ball pocketed", "pocket", {"ball_id": "ball_8"})
        assert len(game_state.events) == 1
        assert game_state.events[0].event_type == "pocket"

    def test_consistency_validation(self):
        """Test game state consistency validation."""
        balls = create_standard_ball_set()
        table = TableState.standard_9ft_table()

        game_state = GameState(
            timestamp=datetime.now().timestamp(),
            frame_number=1,
            balls=balls,
            table=table,
        )

        errors = game_state.validate_consistency()
        assert len(errors) == 0  # Should be consistent initially

        # Add duplicate ball ID
        duplicate_ball = BallState(id="cue", position=Vector2D(0, 0))
        game_state.balls.append(duplicate_ball)

        errors = game_state.validate_consistency()
        assert len(errors) > 0

    def test_initial_state_creation(self):
        """Test initial state creation."""
        game_state = GameState.create_initial_state(GameType.EIGHT_BALL)

        assert game_state.game_type == GameType.EIGHT_BALL
        assert game_state.is_break
        assert len(game_state.balls) == 16

        cue_ball = game_state.get_cue_ball()
        assert cue_ball is not None

    def test_serialization(self):
        """Test game state serialization."""
        game_state = GameState.create_initial_state()

        data = game_state.to_dict()
        game_state_restored = GameState.from_dict(data)

        assert game_state_restored.frame_number == game_state.frame_number
        assert len(game_state_restored.balls) == len(game_state.balls)
        assert game_state_restored.game_type == game_state.game_type


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_json_serialization(self):
        """Test JSON serialization utilities."""
        v = Vector2D(3.0, 4.0)
        json_str = serialize_to_json(v)
        assert isinstance(json_str, str)

        v_restored = deserialize_from_json(json_str, Vector2D)
        assert v_restored.x == v.x
        assert v_restored.y == v.y

    def test_ball_separation(self):
        """Test ball separation calculation."""
        ball1 = BallState(id="ball1", position=Vector2D(0, 0))
        ball2 = BallState(id="ball2", position=Vector2D(0.1, 0))

        separation = calculate_ball_separation(ball1, ball2)
        expected = 0.1 - 2 * ball1.radius  # Distance minus radii
        self.assertAlmostEqual(separation, expected, places=5)

    def test_closest_ball_finding(self):
        """Test closest ball finding."""
        target = BallState(id="target", position=Vector2D(0, 0))
        ball1 = BallState(id="ball1", position=Vector2D(1, 0))
        ball2 = BallState(id="ball2", position=Vector2D(2, 0))
        ball3 = BallState(id="ball3", position=Vector2D(0.5, 0))

        others = [ball1, ball2, ball3]
        closest = find_closest_ball(target, others)

        assert closest.id == "ball3"

    def test_ball_physics_validation(self):
        """Test ball physics validation."""
        # Valid ball
        valid_ball = BallState(id="valid", position=Vector2D(0, 0))
        errors = validate_ball_physics(valid_ball)
        assert len(errors) == 0

        # Invalid ball (too fast)
        fast_ball = BallState(
            id="fast",
            position=Vector2D(0, 0),
            velocity=Vector2D(20, 0),  # Exceeds max_velocity of 10
        )
        errors = validate_ball_physics(fast_ball)
        assert len(errors) > 0

    def test_standard_ball_set_creation(self):
        """Test standard ball set creation."""
        balls = create_standard_ball_set()

        assert len(balls) == 16  # 15 numbered + cue

        cue_balls = [b for b in balls if b.is_cue_ball]
        assert len(cue_balls) == 1

        numbered_balls = [b for b in balls if b.number is not None]
        assert len(numbered_balls) == 15


if __name__ == "__main__":
    unittest.main()
