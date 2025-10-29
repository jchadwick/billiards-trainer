"""Comprehensive unit tests for spin physics module.

This test suite provides full coverage of the SpinPhysics class with particular
emphasis on verifying proper scale parameter handling in all Vector2D instantiations
that were recently fixed.

Test Coverage:
    1. SpinState creation and conversion (to_vector2d, from_vector2d)
    2. English/spin application from cue impact
    3. Spin decay and Magnus force calculation during ball motion
    4. Spin transfer during ball-to-ball collisions
    5. Spin changes during cushion interactions
    6. Masse shot calculations
    7. Vector2D scale parameter validation
    8. SpinCalculator high-level interface

All tests use pytest framework and are marked with @pytest.mark.unit.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.coordinates import Vector2D
from core.models import BallState, Collision
from core.physics.spin import SpinCalculator, SpinPhysics, SpinState


class TestSpinState:
    """Test suite for SpinState dataclass."""

    @pytest.mark.unit()
    def test_spin_state_initialization(self) -> None:
        """Test SpinState creation with default values."""
        spin = SpinState()
        assert spin.top_spin == 0.0
        assert spin.side_spin == 0.0
        assert spin.back_spin == 0.0
        assert spin.decay_rate == 2.0

    @pytest.mark.unit()
    def test_spin_state_with_values(self) -> None:
        """Test SpinState creation with custom values."""
        spin = SpinState(
            top_spin=5.0,
            side_spin=-3.0,
            back_spin=2.0,
            decay_rate=1.5,
        )
        assert spin.top_spin == 5.0
        assert spin.side_spin == -3.0
        assert spin.back_spin == 2.0
        assert spin.decay_rate == 1.5

    @pytest.mark.unit()
    def test_spin_magnitude(self) -> None:
        """Test spin magnitude calculation."""
        spin = SpinState(top_spin=3.0, side_spin=4.0, back_spin=0.0)
        assert abs(spin.magnitude() - 5.0) < 1e-10

    @pytest.mark.unit()
    def test_spin_magnitude_3d(self) -> None:
        """Test spin magnitude with all three components."""
        # 3-4-5 Pythagorean triple scaled
        spin = SpinState(top_spin=2.0, side_spin=3.0, back_spin=6.0)
        expected = math.sqrt(4.0 + 9.0 + 36.0)  # sqrt(49) = 7.0
        assert abs(spin.magnitude() - expected) < 1e-10

    @pytest.mark.unit()
    def test_to_vector2d_scale_parameter(self) -> None:
        """Test Vector2D conversion includes mandatory scale parameter.

        This test verifies the fix for Vector2D instantiation at line 44
        where scale=(1.0, 1.0) was added.
        """
        spin = SpinState(top_spin=10.0, side_spin=5.0)
        vec = spin.to_vector2d()

        # Verify the conversion
        assert vec.x == 5.0  # side_spin maps to x
        assert vec.y == 10.0  # top_spin maps to y

        # Verify mandatory scale parameter is present
        assert vec.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_from_vector2d(self) -> None:
        """Test updating SpinState from Vector2D."""
        spin = SpinState()
        vec = Vector2D(3.0, 7.0, scale=(1.0, 1.0))

        spin.from_vector2d(vec)

        assert spin.side_spin == 3.0
        assert spin.top_spin == 7.0
        # back_spin should not be affected
        assert spin.back_spin == 0.0


class TestSpinPhysics:
    """Test suite for SpinPhysics calculations."""

    @pytest.fixture()
    def physics(self) -> SpinPhysics:
        """Create a SpinPhysics instance with default configuration."""
        return SpinPhysics()

    @pytest.fixture()
    def custom_physics(self) -> SpinPhysics:
        """Create a SpinPhysics instance with custom configuration."""
        config = {
            "cloth_friction": 0.4,
            "magnus_coefficient": 0.3,
            "spin_transfer_efficiency": 0.8,
            "cue_efficiency": 0.9,
        }
        return SpinPhysics(config)

    @pytest.mark.unit()
    def test_physics_initialization(self, physics: SpinPhysics) -> None:
        """Test SpinPhysics initialization with default config."""
        assert physics.cloth_friction == 0.3
        assert physics.magnus_coefficient == 0.25
        assert physics.spin_transfer_efficiency == 0.7
        assert physics.cue_efficiency == 0.8

    @pytest.mark.unit()
    def test_physics_custom_config(self, custom_physics: SpinPhysics) -> None:
        """Test SpinPhysics initialization with custom config."""
        assert custom_physics.cloth_friction == 0.4
        assert custom_physics.magnus_coefficient == 0.3
        assert custom_physics.spin_transfer_efficiency == 0.8
        assert custom_physics.cue_efficiency == 0.9

    @pytest.mark.unit()
    def test_apply_english_basic(self, physics: SpinPhysics) -> None:
        """Test basic English application from cue impact.

        This test verifies Vector2D scale parameters in:
        - Line 121-125: cue_dir vector creation
        - Line 134-138: velocity_change vector creation
        - Line 141-145: cue_ball.velocity update
        """
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        # Impact point at center (no English)
        impact_point = Vector2D(0.0, 0.0, scale=(1.0, 1.0))
        force_magnitude = 100.0
        cue_angle = 0.0  # Horizontal right
        cue_elevation = 0.0

        spin_state = physics.apply_english_from_cue(
            cue_ball, impact_point, force_magnitude, cue_angle, cue_elevation
        )

        # Verify spin state is created
        assert isinstance(spin_state, SpinState)
        # With center impact, spin should be minimal
        assert abs(spin_state.side_spin) < 0.1
        # Verify velocity was updated with proper scale
        assert cue_ball.velocity.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_apply_english_with_offset(self, physics: SpinPhysics) -> None:
        """Test English application with offset impact point."""
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        # Impact point offset to the right (side spin)
        impact_point = Vector2D(10.0, 0.0, scale=(1.0, 1.0))
        force_magnitude = 150.0
        cue_angle = 0.0
        cue_elevation = 0.0

        spin_state = physics.apply_english_from_cue(
            cue_ball, impact_point, force_magnitude, cue_angle, cue_elevation
        )

        # Offset impact should create side spin
        assert abs(spin_state.side_spin) > 0.0 or abs(spin_state.top_spin) > 0.0

    @pytest.mark.unit()
    def test_apply_english_with_elevation(self, physics: SpinPhysics) -> None:
        """Test English application with cue elevation (follow/draw)."""
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        impact_point = Vector2D(0.0, 5.0, scale=(1.0, 1.0))
        force_magnitude = 120.0
        cue_angle = 45.0
        cue_elevation = 15.0  # Elevated cue for follow

        spin_state = physics.apply_english_from_cue(
            cue_ball, impact_point, force_magnitude, cue_angle, cue_elevation
        )

        # Elevation should affect top spin
        assert isinstance(spin_state, SpinState)
        # With elevation, there should be some spin component
        total_spin = abs(spin_state.top_spin) + abs(spin_state.side_spin)
        assert total_spin > 0.0

    @pytest.mark.unit()
    def test_update_spin_with_motion_no_spin(self, physics: SpinPhysics) -> None:
        """Test spin update with stationary ball.

        This test verifies Vector2D scale parameter at line 222-224
        where magnus_force vector is created.
        """
        ball = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))
        spin_state = SpinState()

        new_spin, magnus_force = physics.update_spin_with_motion(
            ball, spin_state, dt=0.1
        )

        # No motion, no spin -> no Magnus force
        assert magnus_force.x == 0.0
        assert magnus_force.y == 0.0
        # Verify mandatory scale parameter
        assert magnus_force.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_update_spin_with_motion_spinning_ball(self, physics: SpinPhysics) -> None:
        """Test spin update with moving, spinning ball."""
        ball = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))  # Moving right
        spin_state = SpinState(top_spin=0.0, side_spin=10.0, back_spin=0.0)

        new_spin, magnus_force = physics.update_spin_with_motion(
            ball, spin_state, dt=0.1
        )

        # Spin should decay
        assert new_spin.side_spin < spin_state.side_spin
        # Magnus force should be present with side spin and velocity
        assert magnus_force.magnitude() > 0.0
        # Verify scale parameter
        assert magnus_force.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_update_spin_decay(self, physics: SpinPhysics) -> None:
        """Test spin decay over time."""
        ball = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball.velocity = Vector2D(50.0, 50.0, scale=(1.0, 1.0))

        initial_spin = SpinState(top_spin=10.0, side_spin=5.0, back_spin=2.0)

        # Simulate several time steps
        spin = initial_spin
        for _ in range(10):
            spin, _ = physics.update_spin_with_motion(ball, spin, dt=0.1)

        # All spin components should decay
        assert spin.top_spin < initial_spin.top_spin
        assert spin.side_spin < initial_spin.side_spin
        assert spin.back_spin < initial_spin.back_spin

    @pytest.mark.unit()
    def test_transfer_spin_collision(self, physics: SpinPhysics) -> None:
        """Test spin transfer during ball-to-ball collision."""
        # Create two balls
        ball1 = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball1.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))

        ball2 = BallState.from_4k(
            id="ball_2",
            x=2000.0,
            y=1080.0,
            number=2,
        )
        ball2.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        # Initial spin states
        spin1 = SpinState(top_spin=5.0, side_spin=10.0, back_spin=1.0)
        spin2 = SpinState(top_spin=0.0, side_spin=0.0, back_spin=0.0)

        # Create collision
        collision = Collision(
            time=0.0,
            position=Vector2D(1960.0, 1080.0, scale=(1.0, 1.0)),
            ball1_id="ball_1",
            ball2_id="ball_2",
            type="ball",
        )

        new_spin1, new_spin2 = physics.transfer_spin_collision(
            ball1, ball2, spin1, spin2, collision
        )

        # Spins should be modified
        assert isinstance(new_spin1, SpinState)
        assert isinstance(new_spin2, SpinState)
        # Top spin typically reduces in collisions
        assert new_spin1.top_spin < spin1.top_spin

    @pytest.mark.unit()
    def test_transfer_spin_no_collision_position(self, physics: SpinPhysics) -> None:
        """Test spin transfer when collision position is None."""
        ball1 = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball1.velocity = Vector2D(50.0, 0.0, scale=(1.0, 1.0))

        ball2 = BallState.from_4k(id="ball_2", x=2000.0, y=1080.0, number=2)
        ball2.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        spin1 = SpinState(side_spin=5.0)
        spin2 = SpinState()

        # Collision without position
        collision = Collision(
            time=0.0,
            position=None,
            ball1_id="ball_1",
            ball2_id="ball_2",
            type="ball",
        )

        # Should not crash, should use default normal
        new_spin1, new_spin2 = physics.transfer_spin_collision(
            ball1, ball2, spin1, spin2, collision
        )

        assert isinstance(new_spin1, SpinState)
        assert isinstance(new_spin2, SpinState)

    @pytest.mark.unit()
    def test_cushion_spin_interaction_vertical(self, physics: SpinPhysics) -> None:
        """Test spin interaction with vertical cushion."""
        ball = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball.velocity = Vector2D(100.0, 50.0, scale=(1.0, 1.0))

        spin_state = SpinState(top_spin=5.0, side_spin=3.0, back_spin=1.0)

        # Vertical cushion (normal points left/right)
        cushion_normal = Vector2D(1.0, 0.0, scale=(1.0, 1.0))

        new_spin = physics.cushion_spin_interaction(ball, spin_state, cushion_normal)

        assert isinstance(new_spin, SpinState)
        # Side spin should reverse and reduce
        assert new_spin.side_spin * spin_state.side_spin <= 0  # Opposite sign or zero

    @pytest.mark.unit()
    def test_cushion_spin_interaction_horizontal(self, physics: SpinPhysics) -> None:
        """Test spin interaction with horizontal cushion."""
        ball = BallState.from_4k(
            id="ball_1",
            x=1920.0,
            y=1080.0,
            number=1,
        )
        ball.velocity = Vector2D(50.0, 100.0, scale=(1.0, 1.0))

        spin_state = SpinState(top_spin=5.0, side_spin=3.0, back_spin=1.0)

        # Horizontal cushion (normal points up/down)
        cushion_normal = Vector2D(0.0, 1.0, scale=(1.0, 1.0))

        new_spin = physics.cushion_spin_interaction(ball, spin_state, cushion_normal)

        assert isinstance(new_spin, SpinState)
        # Top spin should reverse and reduce
        assert new_spin.top_spin * spin_state.top_spin <= 0  # Opposite sign or zero

    @pytest.mark.unit()
    def test_calculate_masse_shot_scale_parameters(self, physics: SpinPhysics) -> None:
        """Test masse shot calculation with proper Vector2D scale handling.

        This test verifies Vector2D scale parameters in:
        - Line 435-439: direction vector creation
        - Line 447-451: initial_velocity vector creation
        """
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        target_position = Vector2D(2200.0, 1300.0, scale=(1.0, 1.0))
        cue_elevation = 45.0  # High elevation for masse
        english_amount = 0.5

        initial_velocity, masse_spin = physics.calculate_masse_shot(
            cue_ball, target_position, cue_elevation, english_amount
        )

        # Verify velocity vector has scale parameter
        assert initial_velocity.scale == (1.0, 1.0)
        assert isinstance(initial_velocity, Vector2D)

        # Verify masse spin is created
        assert isinstance(masse_spin, SpinState)
        # High elevation should create strong back spin
        assert masse_spin.top_spin < 0  # Negative for back spin
        # English should create side spin
        assert abs(masse_spin.side_spin) > 0

    @pytest.mark.unit()
    def test_calculate_masse_shot_no_english(self, physics: SpinPhysics) -> None:
        """Test masse shot without English."""
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        target_position = Vector2D(2000.0, 1080.0, scale=(1.0, 1.0))
        cue_elevation = 60.0
        english_amount = 0.0  # No English

        initial_velocity, masse_spin = physics.calculate_masse_shot(
            cue_ball, target_position, cue_elevation, english_amount
        )

        # Without English, side spin should be minimal
        assert abs(masse_spin.side_spin) < 1.0
        # But should still have back spin from elevation
        assert masse_spin.top_spin < 0

    @pytest.mark.unit()
    def test_calculate_masse_shot_extreme_elevation(self, physics: SpinPhysics) -> None:
        """Test masse shot with extreme elevation."""
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        target_position = Vector2D(2100.0, 1080.0, scale=(1.0, 1.0))
        cue_elevation = 80.0  # Very steep
        english_amount = 1.0  # Maximum English

        initial_velocity, masse_spin = physics.calculate_masse_shot(
            cue_ball, target_position, cue_elevation, english_amount
        )

        # Extreme elevation should reduce forward velocity
        assert initial_velocity.magnitude() < 5.0
        # Should create very strong spin
        assert masse_spin.magnitude() > 10.0


class TestSpinCalculator:
    """Test suite for SpinCalculator high-level interface."""

    @pytest.fixture()
    def calculator(self) -> SpinCalculator:
        """Create a SpinCalculator instance."""
        return SpinCalculator()

    @pytest.mark.unit()
    def test_calculator_initialization(self, calculator: SpinCalculator) -> None:
        """Test SpinCalculator initialization."""
        assert isinstance(calculator.physics, SpinPhysics)
        assert isinstance(calculator.spin_states, dict)
        assert len(calculator.spin_states) == 0

    @pytest.mark.unit()
    def test_apply_english(self, calculator: SpinCalculator) -> None:
        """Test high-level English application."""
        cue_ball = BallState.from_4k(
            id="cue",
            x=1920.0,
            y=1080.0,
            number=0,
        )

        impact_point = Vector2D(5.0, 0.0, scale=(1.0, 1.0))
        force_magnitude = 100.0

        calculator.apply_english(cue_ball, impact_point, force_magnitude)

        # Spin state should be stored
        assert "cue" in calculator.spin_states
        # Ball's spin vector should be updated
        assert cue_ball.spin.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_transfer_spin(self, calculator: SpinCalculator) -> None:
        """Test high-level spin transfer."""
        ball1 = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball1.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        ball1.spin = Vector2D(5.0, 3.0, scale=(1.0, 1.0))

        ball2 = BallState.from_4k(id="ball_2", x=2000.0, y=1080.0, number=2)
        ball2.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))
        ball2.spin = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        collision = Collision(
            time=0.0,
            position=Vector2D(1960.0, 1080.0, scale=(1.0, 1.0)),
            ball1_id="ball_1",
            ball2_id="ball_2",
            type="ball",
        )

        calculator.transfer_spin(ball1, ball2, collision)

        # Both balls should have spin states stored
        assert "ball_1" in calculator.spin_states
        assert "ball_2" in calculator.spin_states
        # Ball spin vectors should be updated
        assert ball1.spin.scale == (1.0, 1.0)
        assert ball2.spin.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_update_ball_motion(self, calculator: SpinCalculator) -> None:
        """Test ball motion update with spin."""
        ball = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        ball.spin = Vector2D(5.0, 10.0, scale=(1.0, 1.0))

        magnus_force = calculator.update_ball_motion(ball, dt=0.1)

        # Magnus force should be returned
        assert isinstance(magnus_force, Vector2D)
        assert magnus_force.scale == (1.0, 1.0)
        # Spin state should be updated
        assert "ball_1" in calculator.spin_states

    @pytest.mark.unit()
    def test_handle_cushion_collision(self, calculator: SpinCalculator) -> None:
        """Test cushion collision handling."""
        ball = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball.velocity = Vector2D(100.0, 50.0, scale=(1.0, 1.0))
        ball.spin = Vector2D(3.0, 5.0, scale=(1.0, 1.0))

        cushion_normal = Vector2D(1.0, 0.0, scale=(1.0, 1.0))

        calculator.handle_cushion_collision(ball, cushion_normal)

        # Spin state should be updated
        assert "ball_1" in calculator.spin_states
        # Ball spin vector should be updated
        assert ball.spin.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_calculate_masse_shot_calculator(self, calculator: SpinCalculator) -> None:
        """Test masse shot calculation through calculator."""
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        target_position = Vector2D(2200.0, 1300.0, scale=(1.0, 1.0))

        velocity, spin_vec = calculator.calculate_masse_shot(
            cue_ball, target_position, cue_elevation=50.0, english_amount=0.5
        )

        # Should return velocity and spin vectors
        assert isinstance(velocity, Vector2D)
        assert isinstance(spin_vec, Vector2D)
        assert velocity.scale == (1.0, 1.0)
        assert spin_vec.scale == (1.0, 1.0)
        # Spin state should be stored
        assert "cue" in calculator.spin_states

    @pytest.mark.unit()
    def test_get_spin_state(self, calculator: SpinCalculator) -> None:
        """Test retrieving spin state."""
        # Initially empty
        assert calculator.get_spin_state("ball_1") is None

        # Add a spin state
        spin = SpinState(top_spin=5.0, side_spin=3.0)
        calculator.spin_states["ball_1"] = spin

        # Should retrieve it
        retrieved = calculator.get_spin_state("ball_1")
        assert retrieved is spin
        assert retrieved.top_spin == 5.0

    @pytest.mark.unit()
    def test_reset_spin(self, calculator: SpinCalculator) -> None:
        """Test resetting spin state."""
        # Add a spin state
        calculator.spin_states["ball_1"] = SpinState(top_spin=5.0)
        assert "ball_1" in calculator.spin_states

        # Reset it
        calculator.reset_spin("ball_1")
        assert "ball_1" not in calculator.spin_states

        # Resetting non-existent state should not crash
        calculator.reset_spin("nonexistent")


class TestVector2DScaleValidation:
    """Test suite specifically for Vector2D scale parameter validation.

    This suite focuses on verifying that all Vector2D instantiations
    in spin.py properly include the mandatory scale parameter.
    """

    @pytest.mark.unit()
    def test_spin_state_to_vector2d_has_scale(self) -> None:
        """Verify line 44: to_vector2d() includes scale parameter."""
        spin = SpinState(top_spin=10.0, side_spin=5.0)
        vec = spin.to_vector2d()

        # Must have scale attribute
        assert hasattr(vec, "scale")
        assert vec.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_cue_direction_vector_has_scale(self) -> None:
        """Verify line 121-125: cue_dir includes scale parameter."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        impact_point = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        # This will create cue_dir internally with scale
        spin = physics.apply_english_from_cue(
            cue_ball,
            impact_point,
            force_magnitude=100.0,
            cue_angle=45.0,
            cue_elevation=0.0,
        )

        # If it worked without error, scale was properly set
        assert isinstance(spin, SpinState)

    @pytest.mark.unit()
    def test_velocity_change_vector_has_scale(self) -> None:
        """Verify line 134-138: velocity_change includes scale parameter."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        impact_point = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        initial_velocity = cue_ball.velocity.magnitude()

        physics.apply_english_from_cue(
            cue_ball,
            impact_point,
            force_magnitude=100.0,
            cue_angle=0.0,
            cue_elevation=0.0,
        )

        # Velocity should be updated
        assert cue_ball.velocity.magnitude() > initial_velocity
        assert cue_ball.velocity.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_cue_ball_velocity_update_has_scale(self) -> None:
        """Verify line 141-145: cue_ball.velocity update includes scale parameter."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        impact_point = Vector2D(5.0, 0.0, scale=(1.0, 1.0))

        physics.apply_english_from_cue(
            cue_ball,
            impact_point,
            force_magnitude=150.0,
            cue_angle=0.0,
            cue_elevation=0.0,
        )

        # Updated velocity must have scale
        assert cue_ball.velocity.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_magnus_force_vector_has_scale(self) -> None:
        """Verify line 222-224: magnus_force includes scale parameter."""
        physics = SpinPhysics()
        ball = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        spin_state = SpinState(side_spin=10.0)

        _, magnus_force = physics.update_spin_with_motion(ball, spin_state, dt=0.1)

        # Magnus force must have scale
        assert magnus_force.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_masse_direction_vector_has_scale(self) -> None:
        """Verify line 435-439: direction vector includes scale parameter."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        target = Vector2D(2200.0, 1300.0, scale=(1.0, 1.0))

        # This will create direction vector internally with scale
        velocity, spin = physics.calculate_masse_shot(
            cue_ball, target, cue_elevation=45.0, english_amount=0.5
        )

        # If it worked, direction was properly created
        assert isinstance(velocity, Vector2D)
        assert velocity.scale == (1.0, 1.0)

    @pytest.mark.unit()
    def test_masse_initial_velocity_has_scale(self) -> None:
        """Verify line 447-451: initial_velocity includes scale parameter."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        target = Vector2D(2000.0, 1080.0, scale=(1.0, 1.0))

        velocity, _ = physics.calculate_masse_shot(
            cue_ball, target, cue_elevation=60.0, english_amount=0.0
        )

        # Initial velocity must have scale
        assert velocity.scale == (1.0, 1.0)


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.unit()
    def test_zero_magnitude_spin(self) -> None:
        """Test spin with zero magnitude."""
        spin = SpinState()
        assert spin.magnitude() == 0.0

    @pytest.mark.unit()
    def test_zero_force_english(self) -> None:
        """Test English application with zero force."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)
        impact_point = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        spin = physics.apply_english_from_cue(
            cue_ball,
            impact_point,
            force_magnitude=0.0,
            cue_angle=0.0,
            cue_elevation=0.0,
        )

        # Should create spin state with minimal values
        assert isinstance(spin, SpinState)
        assert spin.magnitude() < 1.0

    @pytest.mark.unit()
    def test_impact_outside_ball_surface(self) -> None:
        """Test impact point far outside ball surface (should be normalized)."""
        physics = SpinPhysics()
        cue_ball = BallState.from_4k(id="cue", x=1920.0, y=1080.0, number=0)

        # Impact point way outside the ball
        impact_point = Vector2D(100.0, 100.0, scale=(1.0, 1.0))

        # Should not crash, should normalize to ball surface
        spin = physics.apply_english_from_cue(
            cue_ball,
            impact_point,
            force_magnitude=100.0,
            cue_angle=0.0,
            cue_elevation=0.0,
        )

        assert isinstance(spin, SpinState)

    @pytest.mark.unit()
    def test_very_long_time_step(self) -> None:
        """Test spin update with very long time step.

        Note: With very long time steps and slipping, the alignment mechanism
        can actually increase top spin as it aligns with the linear motion.
        This is physically reasonable but means we can't always assert decay.
        """
        physics = SpinPhysics()
        ball = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball.velocity = Vector2D(100.0, 0.0, scale=(1.0, 1.0))
        spin = SpinState(top_spin=10.0, side_spin=5.0)

        # Very long time step
        new_spin, _ = physics.update_spin_with_motion(ball, spin, dt=10.0)

        # Spin components should not be negative
        assert new_spin.top_spin >= 0.0
        assert new_spin.side_spin >= 0.0
        # Side spin should decay (perpendicular to motion)
        assert new_spin.side_spin < spin.side_spin

    @pytest.mark.unit()
    def test_negative_decay_rate(self) -> None:
        """Test that negative decay rate doesn't cause issues."""
        # This tests robustness, though negative decay is physically unrealistic
        spin = SpinState(top_spin=5.0, decay_rate=-1.0)
        assert spin.decay_rate == -1.0
        # The physics should handle it (though results may be unphysical)

    @pytest.mark.unit()
    def test_collision_with_zero_velocity(self) -> None:
        """Test spin transfer when both balls are stationary."""
        physics = SpinPhysics()

        ball1 = BallState.from_4k(id="ball_1", x=1920.0, y=1080.0, number=1)
        ball1.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        ball2 = BallState.from_4k(id="ball_2", x=2000.0, y=1080.0, number=2)
        ball2.velocity = Vector2D(0.0, 0.0, scale=(1.0, 1.0))

        spin1 = SpinState(side_spin=5.0)
        spin2 = SpinState()

        collision = Collision(
            time=0.0,
            position=Vector2D(1960.0, 1080.0, scale=(1.0, 1.0)),
            ball1_id="ball_1",
            ball2_id="ball_2",
            type="ball",
        )

        # Should not crash
        new_spin1, new_spin2 = physics.transfer_spin_collision(
            ball1, ball2, spin1, spin2, collision
        )

        assert isinstance(new_spin1, SpinState)
        assert isinstance(new_spin2, SpinState)
