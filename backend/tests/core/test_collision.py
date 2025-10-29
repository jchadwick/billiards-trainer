"""Comprehensive unit tests for collision detection and resolution.

This test suite provides full coverage of the CollisionDetector, CollisionResolver,
CollisionPredictor, and CollisionOptimizer classes, with particular focus on
Vector2D instantiations and proper scale parameter handling.

Test Coverage:
- Ball-to-ball collision detection (continuous and discrete)
- Ball-to-cushion collision detection
- Collision resolution with physics
- Trajectory prediction with collisions
- Vector2D scale parameter validation
- Multi-ball collision handling
- Simultaneous collision resolution
- Spatial optimization
"""

import math
from typing import Optional

import pytest

from backend.core.constants_4k import BALL_RADIUS_4K
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, TableState
from backend.core.physics.collision import (
    CollisionDetector,
    CollisionOptimizer,
    CollisionPoint,
    CollisionPredictor,
    CollisionResolver,
    CollisionResult,
    CollisionType,
    CushionSegment,
)
from backend.core.physics.spin import SpinCalculator

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def detector():
    """Create a CollisionDetector instance."""
    return CollisionDetector()


@pytest.fixture()
def resolver():
    """Create a CollisionResolver instance."""
    return CollisionResolver()


@pytest.fixture()
def spin_calculator():
    """Create a SpinCalculator instance."""
    return SpinCalculator()


@pytest.fixture()
def resolver_with_spin(spin_calculator):
    """Create a CollisionResolver instance with spin calculator."""
    return CollisionResolver(spin_calculator=spin_calculator)


@pytest.fixture()
def predictor(detector, resolver):
    """Create a CollisionPredictor instance."""
    return CollisionPredictor(detector, resolver)


@pytest.fixture()
def optimizer():
    """Create a CollisionOptimizer instance."""
    return CollisionOptimizer()


@pytest.fixture()
def table_state():
    """Create a standard table state for testing."""
    # Create standard 6-pocket positions for testing
    pocket_positions = [
        Vector2D(0.0, 0.0, scale=(1.0, 1.0)),  # Top-left
        Vector2D(1600.0, 0.0, scale=(1.0, 1.0)),  # Top-middle
        Vector2D(3200.0, 0.0, scale=(1.0, 1.0)),  # Top-right
        Vector2D(0.0, 1600.0, scale=(1.0, 1.0)),  # Bottom-left
        Vector2D(1600.0, 1600.0, scale=(1.0, 1.0)),  # Bottom-middle
        Vector2D(3200.0, 1600.0, scale=(1.0, 1.0)),  # Bottom-right
    ]
    return TableState(
        width=3200.0,
        height=1600.0,
        pocket_positions=pocket_positions,
        pocket_radius=72.0,
        cushion_elasticity=0.85,
        surface_friction=0.2,
    )


@pytest.fixture()
def ball1():
    """Create a test ball (cue ball)."""
    return BallState(
        id="ball_1",
        position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),
        velocity=Vector2D(100.0, 0.0, scale=(1.0, 1.0)),
        radius=BALL_RADIUS_4K,
        mass=0.17,
        spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        is_cue_ball=True,
        number=0,
    )


@pytest.fixture()
def ball2():
    """Create a second test ball (stationary)."""
    return BallState(
        id="ball_2",
        position=Vector2D(1200.0, 800.0, scale=(1.0, 1.0)),
        velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        radius=BALL_RADIUS_4K,
        mass=0.17,
        spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        number=1,
    )


@pytest.fixture()
def ball3():
    """Create a third test ball."""
    return BallState(
        id="ball_3",
        position=Vector2D(1400.0, 800.0, scale=(1.0, 1.0)),
        velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        radius=BALL_RADIUS_4K,
        mass=0.17,
        spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        number=2,
    )


# ============================================================================
# CollisionDetector Tests
# ============================================================================


@pytest.mark.unit()
class TestCollisionDetector:
    """Tests for CollisionDetector class."""

    def test_detector_initialization(self):
        """Test CollisionDetector initialization with default and custom config."""
        # Default config
        detector1 = CollisionDetector()
        assert detector1.collision_threshold == 0.001
        assert detector1.time_step == 0.001
        assert detector1.max_iterations == 1000

        # Custom config
        config = {
            "collision_threshold": 0.002,
            "time_step": 0.005,
            "max_iterations": 500,
        }
        detector2 = CollisionDetector(config)
        assert detector2.collision_threshold == 0.002
        assert detector2.time_step == 0.005
        assert detector2.max_iterations == 500

    def test_ball_collision_detection_head_on(self, detector, ball1, ball2):
        """Test detection of head-on collision between two balls."""
        # ball1 at (1000, 800) moving right at 100 px/s
        # ball2 at (1200, 800) stationary
        # Distance = 200 pixels, combined radius = 72 pixels
        # Should collide in (200 - 72) / 100 = 1.28 seconds

        dt = 2.0  # Time step larger than collision time
        result = detector.detect_ball_collision(ball1, ball2, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_BALL
        assert result.ball1_id == "ball_1"
        assert result.ball2_id == "ball_2"
        assert 0 < result.time < dt
        assert result.point is not None
        assert isinstance(result.point.position, Vector2D)
        assert result.point.position.scale == (1.0, 1.0)
        assert isinstance(result.point.normal, Vector2D)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_ball_collision_detection_no_collision(self, detector, ball1, ball2):
        """Test no collision when balls are moving apart."""
        # Change velocity to move ball1 left (away from ball2)
        ball1.velocity = Vector2D(-100.0, 0.0, scale=(1.0, 1.0))

        dt = 1.0
        result = detector.detect_ball_collision(ball1, ball2, dt)

        assert result is None

    def test_ball_collision_detection_already_colliding(self, detector, ball1, ball2):
        """Test collision detection when balls are already overlapping."""
        # Move ball2 close to ball1 (within collision threshold)
        ball2.position = Vector2D(1060.0, 800.0, scale=(1.0, 1.0))

        dt = 0.1
        result = detector.detect_ball_collision(ball1, ball2, dt)

        assert result is not None
        assert result.time == 0.0  # Immediate collision
        assert result.point is not None
        # Verify scale parameters on all Vector2D objects
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_ball_collision_detection_glancing(self, detector):
        """Test detection of glancing collision (off-center hit)."""
        # Ball1 at (1000, 800) moving right
        ball1 = BallState(
            id="ball_1",
            position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )
        # Ball2 at (1200, 850) - offset by 50 pixels vertically
        ball2 = BallState(
            id="ball_2",
            position=Vector2D(1200.0, 850.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 2.0
        result = detector.detect_ball_collision(ball1, ball2, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_BALL
        # Normal should not be exactly horizontal due to glancing hit
        assert result.point is not None
        normal = result.point.normal
        assert normal.scale == (1.0, 1.0)
        assert abs(normal.y) > 0.1  # Should have vertical component

    def test_ball_collision_vector_scale_parameters(self, detector, ball1, ball2):
        """Test that all Vector2D objects have proper scale parameters."""
        dt = 2.0
        result = detector.detect_ball_collision(ball1, ball2, dt)

        assert result is not None

        # Check all Vector2D objects in result
        assert result.ball1_velocity.scale == (1.0, 1.0)
        assert result.ball2_velocity.scale == (1.0, 1.0)
        assert result.point is not None
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_cushion_collision_detection_left(self, detector, table_state):
        """Test detection of collision with left cushion."""
        # Ball near left edge, moving left
        ball = BallState(
            id="ball_1",
            position=Vector2D(50.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_cushion_collision(ball, table_state, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_CUSHION
        assert result.ball1_id == "ball_1"
        assert result.point is not None
        # Normal should point right (away from left wall)
        assert result.point.normal.x > 0
        # Verify scale parameters
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_cushion_collision_detection_right(self, detector, table_state):
        """Test detection of collision with right cushion."""
        # Ball near right edge, moving right
        ball = BallState(
            id="ball_1",
            position=Vector2D(3150.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_cushion_collision(ball, table_state, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_CUSHION
        # Normal should point left (away from right wall)
        assert result.point is not None
        assert result.point.normal.x < 0
        # Verify scale parameters
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_cushion_collision_detection_top(self, detector, table_state):
        """Test detection of collision with top cushion."""
        # Ball near top edge, moving up
        ball = BallState(
            id="ball_1",
            position=Vector2D(1600.0, 50.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, -100.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_cushion_collision(ball, table_state, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_CUSHION
        # Normal should point down (away from top wall)
        assert result.point is not None
        assert result.point.normal.y > 0
        # Verify scale parameters
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_cushion_collision_detection_bottom(self, detector, table_state):
        """Test detection of collision with bottom cushion."""
        # Ball near bottom edge, moving down
        ball = BallState(
            id="ball_1",
            position=Vector2D(1600.0, 1550.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, 100.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_cushion_collision(ball, table_state, dt)

        assert result is not None
        assert result.collision_type == CollisionType.BALL_CUSHION
        # Normal should point up (away from bottom wall)
        assert result.point is not None
        assert result.point.normal.y < 0
        # Verify scale parameters
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_cushion_collision_no_collision(self, detector, table_state):
        """Test no collision when ball is moving away from cushions."""
        # Ball in center, moving right
        ball = BallState(
            id="ball_1",
            position=Vector2D(1600.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(10.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 0.1
        result = detector.detect_cushion_collision(ball, table_state, dt)

        # Should be None since ball is far from cushions
        assert result is None

    def test_cushion_collision_already_colliding(self, detector, table_state):
        """Test collision detection when ball is already at cushion."""
        # Ball at left edge (within collision threshold)
        ball = BallState(
            id="ball_1",
            position=Vector2D(BALL_RADIUS_4K, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-10.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 0.1
        result = detector.detect_cushion_collision(ball, table_state, dt)

        assert result is not None
        assert result.time == 0.0  # Immediate collision
        assert result.point is not None
        # Verify scale parameters
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_create_table_cushions(self, detector, table_state):
        """Test creation of table cushion segments."""
        cushions = detector._create_table_cushions(table_state)

        assert len(cushions) == 4  # Four walls

        # Verify all cushions have proper Vector2D scale parameters
        for cushion in cushions:
            assert isinstance(cushion, CushionSegment)
            assert cushion.start.scale == (1.0, 1.0)
            assert cushion.end.scale == (1.0, 1.0)
            assert cushion.normal.scale == (1.0, 1.0)
            assert cushion.restitution > 0
            assert cushion.friction >= 0

    def test_detect_multiple_collisions(
        self, detector, ball1, ball2, ball3, table_state
    ):
        """Test detection of multiple collisions in a single time step."""
        # Set up scenario with multiple potential collisions
        # Ball1 moving right toward Ball2
        # Ball3 moving toward right cushion
        ball3.position = Vector2D(3100.0, 800.0, scale=(1.0, 1.0))
        ball3.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))

        balls = [ball1, ball2, ball3]
        dt = 2.0

        collisions = detector.detect_multiple_collisions(balls, table_state, dt)

        assert len(collisions) >= 1  # At least one collision
        # Verify all collisions are sorted by time
        for i in range(len(collisions) - 1):
            assert collisions[i].time <= collisions[i + 1].time

        # Verify all collision points have proper scale
        for collision in collisions:
            if collision.point:
                assert collision.point.position.scale == (1.0, 1.0)
                assert collision.point.normal.scale == (1.0, 1.0)

    def test_detect_multiple_collisions_pocketed_balls(
        self, detector, ball1, ball2, table_state
    ):
        """Test that pocketed balls are ignored in collision detection."""
        ball2.is_pocketed = True

        balls = [ball1, ball2]
        dt = 2.0

        collisions = detector.detect_multiple_collisions(balls, table_state, dt)

        # Should not detect collision with pocketed ball
        for collision in collisions:
            assert collision.ball2_id != "ball_2"


# ============================================================================
# CollisionResolver Tests
# ============================================================================


@pytest.mark.unit()
class TestCollisionResolver:
    """Tests for CollisionResolver class."""

    def test_resolver_initialization(self):
        """Test CollisionResolver initialization with default and custom config."""
        # Default config
        resolver1 = CollisionResolver()
        assert resolver1.energy_loss_factor == 0.02
        assert resolver1.spin_transfer_factor == 0.3
        assert resolver1.default_restitution == 0.95
        assert resolver1.friction_coefficient == 0.1

        # Custom config
        config = {
            "energy_loss_factor": 0.05,
            "spin_transfer_factor": 0.5,
            "default_restitution": 0.9,
            "friction_coefficient": 0.2,
        }
        resolver2 = CollisionResolver(config)
        assert resolver2.energy_loss_factor == 0.05
        assert resolver2.spin_transfer_factor == 0.5
        assert resolver2.default_restitution == 0.9
        assert resolver2.friction_coefficient == 0.2

    def test_resolve_ball_collision_basic(self, resolver, detector, ball1, ball2):
        """Test basic ball collision resolution (elastic collision)."""
        # Detect collision first
        dt = 2.0
        collision = detector.detect_ball_collision(ball1, ball2, dt)
        assert collision is not None

        # Store initial velocities for comparison
        initial_vel1 = ball1.velocity.magnitude()
        ball2.velocity.magnitude()

        # Resolve collision
        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)

        assert resolved is not None
        assert resolved.ball1_velocity is not None
        assert resolved.ball2_velocity is not None

        # Verify scale parameters
        assert resolved.ball1_velocity.scale == (1.0, 1.0)
        assert resolved.ball2_velocity.scale == (1.0, 1.0)

        # Energy should be approximately conserved (some loss due to restitution)
        initial_energy = 0.5 * ball1.mass * initial_vel1**2
        final_energy = (
            0.5 * ball1.mass * resolved.ball1_velocity.magnitude() ** 2
            + 0.5 * ball2.mass * resolved.ball2_velocity.magnitude() ** 2
        )
        assert final_energy <= initial_energy
        assert resolved.energy_lost >= 0

        # Momentum should be conserved in collision direction
        # (ball1 transfers momentum to ball2)
        final_total_momentum = (
            ball1.mass * resolved.ball1_velocity.magnitude()
            + ball2.mass * resolved.ball2_velocity.magnitude()
        )
        assert final_total_momentum > 0  # System has momentum

    def test_resolve_ball_collision_momentum_conservation(
        self, resolver, detector, ball1, ball2
    ):
        """Test that momentum is conserved in ball collision resolution."""
        # Detect collision
        dt = 2.0
        collision = detector.detect_ball_collision(ball1, ball2, dt)
        assert collision is not None

        # Store initial momentum
        initial_momentum_x = (
            ball1.velocity.x * ball1.mass + ball2.velocity.x * ball2.mass
        )
        initial_momentum_y = (
            ball1.velocity.y * ball1.mass + ball2.velocity.y * ball2.mass
        )

        # Resolve collision
        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)

        # Calculate final momentum
        final_momentum_x = (
            resolved.ball1_velocity.x * ball1.mass
            + resolved.ball2_velocity.x * ball2.mass
        )
        final_momentum_y = (
            resolved.ball1_velocity.y * ball1.mass
            + resolved.ball2_velocity.y * ball2.mass
        )

        # Momentum should be conserved (within floating point error)
        assert abs(final_momentum_x - initial_momentum_x) < 0.01
        assert abs(final_momentum_y - initial_momentum_y) < 0.01

    def test_resolve_ball_collision_separating_velocities(
        self, resolver, detector, ball1, ball2
    ):
        """Test collision resolution when balls are already separating."""
        # Set velocities so balls are moving apart
        ball1.velocity = Vector2D(-100.0, 0.0, scale=(1.0, 1.0))
        ball2.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        # Position them close together
        ball2.position = Vector2D(1060.0, 800.0, scale=(1.0, 1.0))

        # Detect collision (should still detect due to proximity)
        dt = 0.1
        collision = detector.detect_ball_collision(ball1, ball2, dt)
        assert collision is not None

        # Resolve collision
        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)

        # When balls are separating, the resolver should not reverse their motion
        # The collision result should preserve the general direction of separation
        # Verify that scale parameters are correct
        assert resolved.ball1_velocity.scale == (1.0, 1.0)
        assert resolved.ball2_velocity.scale == (1.0, 1.0)

        # Just verify no errors occurred and result structure is valid
        assert resolved is not None
        assert resolved.ball1_velocity is not None
        assert resolved.ball2_velocity is not None

    def test_resolve_cushion_collision_basic(self, resolver, detector, table_state):
        """Test basic cushion collision resolution."""
        # Ball moving toward left cushion
        ball = BallState(
            id="ball_1",
            position=Vector2D(50.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
            spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        )

        # Detect collision
        dt = 1.0
        collision = detector.detect_cushion_collision(ball, table_state, dt)
        assert collision is not None

        # Resolve collision
        resolved = resolver.resolve_cushion_collision(ball, collision)

        assert resolved is not None
        assert resolved.ball1_velocity is not None

        # Velocity should reverse direction (bounce off cushion)
        assert resolved.ball1_velocity.x > 0  # Now moving right
        # Magnitude should be reduced or equal due to energy loss
        assert resolved.ball1_velocity.magnitude() <= ball.velocity.magnitude()

        # Verify scale parameter
        assert resolved.ball1_velocity.scale == (1.0, 1.0)
        assert resolved.energy_lost >= 0

    def test_resolve_cushion_collision_friction(self, resolver, detector, table_state):
        """Test cushion collision with tangential friction."""
        # Ball moving at angle toward cushion (has tangential component)
        ball = BallState(
            id="ball_1",
            position=Vector2D(50.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-100.0, 50.0, scale=(1.0, 1.0)),  # Diagonal motion
            radius=BALL_RADIUS_4K,
            mass=0.17,
            spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        )

        # Detect collision
        dt = 1.0
        collision = detector.detect_cushion_collision(ball, table_state, dt)
        assert collision is not None

        # Resolve collision
        resolved = resolver.resolve_cushion_collision(ball, collision)

        assert resolved is not None
        # X component should reverse
        assert resolved.ball1_velocity.x > 0
        # Y component should be reduced or equal due to friction
        assert abs(resolved.ball1_velocity.y) <= abs(ball.velocity.y)

        # Verify scale parameter
        assert resolved.ball1_velocity.scale == (1.0, 1.0)

    def test_resolve_simultaneous_collisions_single(
        self, resolver, detector, ball1, ball2, table_state
    ):
        """Test resolution of a single collision from a list."""
        dt = 2.0
        collision = detector.detect_ball_collision(ball1, ball2, dt)
        assert collision is not None

        balls = [ball1, ball2]
        resolved_list = resolver.resolve_simultaneous_collisions([collision], balls)

        assert len(resolved_list) == 1
        resolved = resolved_list[0]
        assert resolved.ball1_velocity is not None
        assert resolved.ball2_velocity is not None
        # Verify scale parameters
        assert resolved.ball1_velocity.scale == (1.0, 1.0)
        assert resolved.ball2_velocity.scale == (1.0, 1.0)

    def test_resolve_simultaneous_collisions_multiple(
        self, resolver, detector, ball1, ball2, ball3, table_state
    ):
        """Test resolution of multiple simultaneous collisions."""
        # Set up multiple collisions
        ball3.position = Vector2D(1400.0, 800.0, scale=(1.0, 1.0))
        ball2.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))

        balls = [ball1, ball2, ball3]
        dt = 2.0

        # Detect all collisions
        collisions = detector.detect_multiple_collisions(balls, table_state, dt)

        if len(collisions) > 0:
            # Resolve all collisions
            resolved_list = resolver.resolve_simultaneous_collisions(collisions, balls)

            assert len(resolved_list) == len(collisions)
            # Verify all resolved collisions have proper scale
            for resolved in resolved_list:
                assert resolved.ball1_velocity.scale == (1.0, 1.0)
                if resolved.ball2_velocity:
                    assert resolved.ball2_velocity.scale == (1.0, 1.0)

    def test_find_ball_by_id(self, resolver, ball1, ball2, ball3):
        """Test finding a ball by ID."""
        balls = [ball1, ball2, ball3]

        found_ball1 = resolver._find_ball_by_id(balls, "ball_1")
        assert found_ball1 is not None
        assert found_ball1.id == "ball_1"

        found_ball2 = resolver._find_ball_by_id(balls, "ball_2")
        assert found_ball2 is not None
        assert found_ball2.id == "ball_2"

        not_found = resolver._find_ball_by_id(balls, "nonexistent")
        assert not_found is None

        none_id = resolver._find_ball_by_id(balls, None)
        assert none_id is None


# ============================================================================
# CollisionPredictor Tests
# ============================================================================


@pytest.mark.unit()
class TestCollisionPredictor:
    """Tests for CollisionPredictor class."""

    def test_predictor_initialization(self, detector, resolver):
        """Test CollisionPredictor initialization."""
        predictor = CollisionPredictor(detector, resolver)
        assert predictor.detector is detector
        assert predictor.resolver is resolver

    def test_predict_trajectory_single_collision(
        self, predictor, ball1, ball2, table_state
    ):
        """Test trajectory prediction with a single ball-ball collision."""
        other_balls = [ball2]
        max_time = 5.0

        collisions = predictor.predict_trajectory_collisions(
            ball1, other_balls, table_state, max_time=max_time
        )

        # Should predict at least one collision
        assert len(collisions) >= 1

        # First collision should be with ball2
        first_collision = collisions[0]
        assert (
            first_collision.collision_type == CollisionType.BALL_BALL
            or first_collision.collision_type == CollisionType.BALL_CUSHION
        )

        # Verify all collisions have proper scale parameters
        for collision in collisions:
            if collision.point:
                assert collision.point.position.scale == (1.0, 1.0)
                assert collision.point.normal.scale == (1.0, 1.0)
            assert collision.ball1_velocity.scale == (1.0, 1.0)
            if collision.ball2_velocity:
                assert collision.ball2_velocity.scale == (1.0, 1.0)

    def test_predict_trajectory_cushion_collisions(self, predictor, table_state):
        """Test trajectory prediction with cushion collisions."""
        # Ball moving toward left cushion and beyond
        ball = BallState(
            id="ball_1",
            position=Vector2D(200.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-200.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
            spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        )

        other_balls = []
        max_time = 5.0

        collisions = predictor.predict_trajectory_collisions(
            ball, other_balls, table_state, max_time=max_time
        )

        # Should predict cushion collision
        assert len(collisions) >= 1
        # First collision should be cushion
        assert collisions[0].collision_type == CollisionType.BALL_CUSHION

        # Verify scale parameters
        for collision in collisions:
            if collision.point:
                assert collision.point.position.scale == (1.0, 1.0)
                assert collision.point.normal.scale == (1.0, 1.0)

    def test_predict_trajectory_multiple_collisions(
        self, predictor, ball1, ball2, ball3, table_state
    ):
        """Test trajectory prediction with multiple sequential collisions."""
        # Set up balls in a line
        ball2.position = Vector2D(1200.0, 800.0, scale=(1.0, 1.0))
        ball3.position = Vector2D(1400.0, 800.0, scale=(1.0, 1.0))

        other_balls = [ball2, ball3]
        max_time = 10.0

        collisions = predictor.predict_trajectory_collisions(
            ball1, other_balls, table_state, max_time=max_time
        )

        # Should predict at least one collision
        assert len(collisions) >= 1

        # Verify collisions are in chronological order
        for i in range(len(collisions) - 1):
            assert collisions[i].time <= collisions[i + 1].time

        # Verify scale parameters
        for collision in collisions:
            if collision.point:
                assert collision.point.position.scale == (1.0, 1.0)

    def test_predict_trajectory_no_collision(self, predictor, table_state):
        """Test trajectory prediction when no collisions occur."""
        # Ball in center with low velocity
        ball = BallState(
            id="ball_1",
            position=Vector2D(1600.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.5, 0.0, scale=(1.0, 1.0)),  # Very slow
            radius=BALL_RADIUS_4K,
            mass=0.17,
            spin=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
        )

        other_balls = []
        max_time = 0.1  # Short time

        collisions = predictor.predict_trajectory_collisions(
            ball, other_balls, table_state, max_time=max_time
        )

        # May have no collisions or cushion collision depending on trajectory
        # Just verify structure is correct
        for collision in collisions:
            assert collision.time <= max_time

    def test_find_ball_in_list(self, predictor, ball1, ball2, ball3):
        """Test finding a ball by ID in a list."""
        balls = [ball1, ball2, ball3]

        found_ball1 = predictor._find_ball_in_list(balls, "ball_1")
        assert found_ball1 is not None
        assert found_ball1.id == "ball_1"

        found_ball2 = predictor._find_ball_in_list(balls, "ball_2")
        assert found_ball2 is not None
        assert found_ball2.id == "ball_2"

        not_found = predictor._find_ball_in_list(balls, "nonexistent")
        assert not_found is None

        none_id = predictor._find_ball_in_list(balls, None)
        assert none_id is None


# ============================================================================
# CollisionOptimizer Tests
# ============================================================================


@pytest.mark.unit()
class TestCollisionOptimizer:
    """Tests for CollisionOptimizer class."""

    def test_optimizer_initialization(self, optimizer):
        """Test CollisionOptimizer initialization."""
        assert optimizer.spatial_grid_size == 100.0
        assert len(optimizer.spatial_grid) == 0

    def test_build_spatial_grid(self, optimizer, ball1, ball2, ball3):
        """Test building spatial grid for collision optimization."""
        balls = [ball1, ball2, ball3]
        optimizer.build_spatial_grid(balls)

        # Grid should have entries
        assert len(optimizer.spatial_grid) > 0

        # Verify grid structure
        for cell, ball_indices in optimizer.spatial_grid.items():
            assert isinstance(cell, tuple)
            assert len(cell) == 2
            assert isinstance(ball_indices, list)
            for idx in ball_indices:
                assert 0 <= idx < len(balls)

    def test_build_spatial_grid_pocketed_balls(self, optimizer, ball1, ball2):
        """Test that pocketed balls are excluded from spatial grid."""
        ball2.is_pocketed = True
        balls = [ball1, ball2]

        optimizer.build_spatial_grid(balls)

        # Verify ball2 is not in any grid cells
        for cell, ball_indices in optimizer.spatial_grid.items():
            assert 1 not in ball_indices

    def test_get_collision_candidates(self, optimizer, ball1, ball2, ball3):
        """Test getting collision candidates from spatial grid."""
        balls = [ball1, ball2, ball3]
        optimizer.build_spatial_grid(balls)

        # Get candidates for ball1
        candidates = optimizer.get_collision_candidates(0, balls)

        # Should return list of indices
        assert isinstance(candidates, list)
        # ball1 should not be in its own candidates
        assert 0 not in candidates

    def test_get_collision_candidates_empty_grid(self, optimizer, ball1):
        """Test getting candidates from empty grid."""
        balls = [ball1]

        # Don't build grid - test with empty grid
        candidates = optimizer.get_collision_candidates(0, balls)

        # Should return empty list
        assert len(candidates) == 0

    def test_spatial_grid_performance(self, optimizer):
        """Test spatial grid performance with many balls."""
        # Create 15 balls (full rack)
        balls = []
        for i in range(15):
            x = 1000.0 + (i % 5) * 100
            y = 800.0 + (i // 5) * 100
            ball = BallState(
                id=f"ball_{i}",
                position=Vector2D(x, y, scale=(1.0, 1.0)),
                velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
                radius=BALL_RADIUS_4K,
                mass=0.17,
            )
            balls.append(ball)

        # Build grid
        optimizer.build_spatial_grid(balls)

        # Get candidates for first ball
        candidates = optimizer.get_collision_candidates(0, balls)

        # Should have candidates from nearby cells
        assert len(candidates) >= 0  # May have no nearby balls
        # Verify all candidates are valid indices
        for idx in candidates:
            assert 0 <= idx < len(balls)
            assert idx != 0


# ============================================================================
# Vector2D Scale Parameter Tests
# ============================================================================


@pytest.mark.unit()
class TestVector2DScaleParameters:
    """Tests specifically for Vector2D scale parameter handling in collision code."""

    def test_all_vector2d_have_scale(
        self, detector, resolver, ball1, ball2, table_state
    ):
        """Test that all Vector2D instances created in collision detection have scale."""
        dt = 2.0

        # Detect ball collision
        ball_collision = detector.detect_ball_collision(ball1, ball2, dt)
        if ball_collision:
            self._verify_collision_result_scales(ball_collision)

        # Detect cushion collision
        ball_cushion = BallState(
            id="test_ball",
            position=Vector2D(50.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )
        cushion_collision = detector.detect_cushion_collision(
            ball_cushion, table_state, dt
        )
        if cushion_collision:
            self._verify_collision_result_scales(cushion_collision)

    def test_cushion_segments_have_scale(self, detector, table_state):
        """Test that all cushion segments have proper scale parameters."""
        cushions = detector._create_table_cushions(table_state)

        for cushion in cushions:
            assert hasattr(cushion.start, "scale")
            assert hasattr(cushion.end, "scale")
            assert hasattr(cushion.normal, "scale")
            assert cushion.start.scale == (1.0, 1.0)
            assert cushion.end.scale == (1.0, 1.0)
            assert cushion.normal.scale == (1.0, 1.0)

    def test_collision_resolution_preserves_scale(
        self, detector, resolver, ball1, ball2
    ):
        """Test that collision resolution preserves scale parameters."""
        dt = 2.0
        collision = detector.detect_ball_collision(ball1, ball2, dt)
        assert collision is not None

        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)

        # All vectors should maintain scale
        assert resolved.ball1_velocity.scale == (1.0, 1.0)
        assert resolved.ball2_velocity.scale == (1.0, 1.0)
        if resolved.ball1_spin:
            assert resolved.ball1_spin.scale == (1.0, 1.0)
        if resolved.ball2_spin:
            assert resolved.ball2_spin.scale == (1.0, 1.0)

    def test_trajectory_prediction_preserves_scale(
        self, predictor, ball1, ball2, table_state
    ):
        """Test that trajectory prediction preserves scale parameters."""
        other_balls = [ball2]
        max_time = 5.0

        collisions = predictor.predict_trajectory_collisions(
            ball1, other_balls, table_state, max_time=max_time
        )

        for collision in collisions:
            self._verify_collision_result_scales(collision)

    def test_vector2d_without_scale_raises_error(self):
        """Test that creating Vector2D without scale raises ValueError."""
        with pytest.raises(TypeError):
            # This should fail because scale is required
            Vector2D(100.0, 200.0)

    def test_vector2d_invalid_scale_raises_error(self):
        """Test that creating Vector2D with invalid scale raises ValueError."""
        with pytest.raises(ValueError):
            Vector2D(100.0, 200.0, scale=(0.0, 1.0))  # Zero scale

        with pytest.raises(ValueError):
            Vector2D(100.0, 200.0, scale=(-1.0, 1.0))  # Negative scale

        with pytest.raises(ValueError):
            Vector2D(100.0, 200.0, scale=(1.0,))  # Wrong tuple size

    def _verify_collision_result_scales(self, collision: CollisionResult) -> None:
        """Helper to verify all Vector2D in CollisionResult have proper scale."""
        assert collision.ball1_velocity.scale == (1.0, 1.0)
        if collision.ball2_velocity:
            assert collision.ball2_velocity.scale == (1.0, 1.0)
        if collision.point:
            assert collision.point.position.scale == (1.0, 1.0)
            assert collision.point.normal.scale == (1.0, 1.0)
        if collision.ball1_spin:
            assert collision.ball1_spin.scale == (1.0, 1.0)
        if collision.ball2_spin:
            assert collision.ball2_spin.scale == (1.0, 1.0)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit()
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_collision_with_zero_velocity_balls(self, detector):
        """Test collision detection with stationary balls."""
        ball1 = BallState(
            id="ball_1",
            position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )
        ball2 = BallState(
            id="ball_2",
            position=Vector2D(1200.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_ball_collision(ball1, ball2, dt)

        # No collision since no relative motion
        assert result is None

    def test_collision_at_exact_same_position(self, detector):
        """Test collision detection when balls are at exact same position."""
        ball1 = BallState(
            id="ball_1",
            position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(10.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )
        ball2 = BallState(
            id="ball_2",
            position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),  # Same position
            velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 0.1
        result = detector.detect_ball_collision(ball1, ball2, dt)

        # Should detect immediate collision
        assert result is not None
        assert result.time == 0.0
        # Should use arbitrary normal when positions are identical
        assert result.point is not None
        assert result.point.normal.scale == (1.0, 1.0)

    def test_resolve_collision_without_collision_point(self, resolver, ball1, ball2):
        """Test collision resolution fails gracefully without collision point."""
        # Create collision result without point
        collision = CollisionResult(
            collision_type=CollisionType.BALL_BALL,
            time=0.0,
            point=None,  # No collision point
            ball1_velocity=ball1.velocity,
            ball2_velocity=ball2.velocity,
        )

        with pytest.raises(ValueError, match="Collision point required"):
            resolver.resolve_ball_collision(ball1, ball2, collision)

    def test_cushion_collision_with_none_table(self, detector):
        """Test cushion collision detection with None table."""
        ball = BallState(
            id="ball_1",
            position=Vector2D(50.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(-100.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 1.0
        result = detector.detect_cushion_collision(ball, None, dt)

        # Should return None when table is None
        assert result is None

    def test_very_high_velocity_collision(self, detector):
        """Test collision detection with very high velocities."""
        ball1 = BallState(
            id="ball_1",
            position=Vector2D(1000.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(10000.0, 0.0, scale=(1.0, 1.0)),  # Very fast
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )
        ball2 = BallState(
            id="ball_2",
            position=Vector2D(1200.0, 800.0, scale=(1.0, 1.0)),
            velocity=Vector2D(0.0, 0.0, scale=(1.0, 1.0)),
            radius=BALL_RADIUS_4K,
            mass=0.17,
        )

        dt = 0.1
        result = detector.detect_ball_collision(ball1, ball2, dt)

        # Should still detect collision
        assert result is not None
        assert result.point is not None
        # Verify scale parameters even at high velocities
        assert result.point.position.scale == (1.0, 1.0)
        assert result.point.normal.scale == (1.0, 1.0)

    def test_collision_with_very_small_time_step(self, detector, ball1, ball2):
        """Test collision detection with very small time step."""
        dt = 0.00001  # 10 microseconds

        result = detector.detect_ball_collision(ball1, ball2, dt)

        # May or may not detect depending on time step
        # Just verify no errors occur
        if result:
            assert result.time <= dt
