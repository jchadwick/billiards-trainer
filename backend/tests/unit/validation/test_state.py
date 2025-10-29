"""Comprehensive unit tests for StateValidator class.

This test suite provides complete coverage of the StateValidator class,
testing all validation methods including:
- Game state validation
- Ball position validation
- Table state validation
- Cue state validation
- Physics validation
- Overlap detection
- Error handling and correction

Test Coverage:
- All StateValidator public methods
- All validation rules and constraints
- Error detection and reporting
- Auto-correction functionality
- Edge cases and boundary conditions
- Minimum 60+ test cases
"""

import math
import sys
from pathlib import Path

import pytest

# Add backend to path for direct imports
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

from core.constants_4k import BALL_RADIUS_4K, CANONICAL_HEIGHT, CANONICAL_WIDTH
from core.coordinates import Vector2D
from core.models import BallState, CueState, GameState, TableState
from core.validation.state import StateValidator, ValidationResult

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture()
def validator():
    """Create a standard validator with default settings."""
    return StateValidator()


@pytest.fixture()
def strict_validator():
    """Create a validator with strict settings (no auto-correction)."""
    return StateValidator(
        max_velocity=5000.0,
        overlap_tolerance=0.5,
        enable_auto_correction=False,
    )


@pytest.fixture()
def lenient_validator():
    """Create a validator with lenient settings."""
    return StateValidator(
        max_velocity=20000.0,
        overlap_tolerance=5.0,
        enable_auto_correction=True,
    )


@pytest.fixture()
def simple_table():
    """Create a simple test table with proper 4K dimensions."""
    width = 3200.0  # pixels
    height = 1600.0  # pixels

    pockets = [
        Vector2D.from_4k(0, 0),  # Top-left
        Vector2D.from_4k(width / 2, 0),  # Top-middle
        Vector2D.from_4k(width, 0),  # Top-right
        Vector2D.from_4k(0, height),  # Bottom-left
        Vector2D.from_4k(width / 2, height),  # Bottom-middle
        Vector2D.from_4k(width, height),  # Bottom-right
    ]

    return TableState(
        width=width,
        height=height,
        pocket_positions=pockets,
        pocket_radius=72.0,
        cushion_elasticity=0.85,
        surface_friction=0.2,
    )


@pytest.fixture()
def cue_ball():
    """Create a cue ball at table center."""
    return BallState(
        id="cue",
        position=Vector2D.from_4k(1600.0, 800.0),
        velocity=Vector2D.from_4k(0.0, 0.0),
        radius=BALL_RADIUS_4K,
        mass=0.17,
        is_cue_ball=True,
        confidence=1.0,
    )


@pytest.fixture()
def object_ball():
    """Create a standard object ball."""
    return BallState(
        id="ball_1",
        position=Vector2D.from_4k(2000.0, 800.0),
        velocity=Vector2D.from_4k(0.0, 0.0),
        radius=BALL_RADIUS_4K,
        mass=0.17,
        number=1,
        confidence=1.0,
    )


@pytest.fixture()
def simple_cue():
    """Create a simple cue state."""
    return CueState(
        tip_position=Vector2D.from_4k(1500.0, 800.0),
        angle=0.0,
        length=1.47,
        tip_radius=0.006,
        confidence=1.0,
    )


@pytest.fixture()
def simple_game_state(simple_table, cue_ball, object_ball):
    """Create a simple valid game state."""
    return GameState(
        timestamp=0.0,
        frame_number=0,
        balls=[cue_ball, object_ball],
        table=simple_table,
        state_confidence=1.0,
    )


# ============================================================================
# VALIDATION RESULT TESTS
# ============================================================================


class TestValidationResult:
    """Test ValidationResult class."""

    @pytest.mark.unit()
    def test_default_validation_result(self):
        """Test default ValidationResult is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        assert len(result.corrected_values) == 0
        assert result.confidence == 1.0

    @pytest.mark.unit()
    def test_add_error(self):
        """Test adding error invalidates result."""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert "Test error" in result.errors
        assert len(result.errors) == 1

    @pytest.mark.unit()
    def test_add_warning(self):
        """Test adding warning keeps result valid."""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert "Test warning" in result.warnings
        assert len(result.warnings) == 1

    @pytest.mark.unit()
    def test_add_correction(self):
        """Test adding correction."""
        result = ValidationResult()
        result.add_correction("test_field", 42)

        assert result.is_valid is True
        assert result.corrected_values["test_field"] == 42

    @pytest.mark.unit()
    def test_merge_results(self):
        """Test merging two validation results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")
        result1.confidence = 0.8

        result2 = ValidationResult()
        result2.add_error("Error 2")
        result2.add_warning("Warning 2")
        result2.confidence = 0.9

        merged = result1.merge(result2)

        assert merged.is_valid is False
        assert len(merged.errors) == 2
        assert len(merged.warnings) == 2
        assert merged.confidence == 0.8  # Minimum confidence


# ============================================================================
# VALIDATOR INITIALIZATION TESTS
# ============================================================================


class TestStateValidatorInitialization:
    """Test StateValidator initialization."""

    @pytest.mark.unit()
    def test_default_initialization(self):
        """Test validator initializes with default settings."""
        validator = StateValidator()

        assert validator.max_velocity > 0
        assert validator.max_acceleration > 0
        assert validator.overlap_tolerance > 0
        assert validator.position_tolerance > 0
        assert validator.enable_auto_correction is True

    @pytest.mark.unit()
    def test_custom_initialization(self):
        """Test validator with custom parameters."""
        validator = StateValidator(
            max_velocity=5000.0,
            max_acceleration=10000.0,
            overlap_tolerance=2.0,
            position_tolerance=0.5,
            enable_auto_correction=False,
        )

        assert validator.max_velocity == 5000.0
        assert validator.max_acceleration == 10000.0
        assert validator.overlap_tolerance == 2.0
        assert validator.position_tolerance == 0.5
        assert validator.enable_auto_correction is False

    @pytest.mark.unit()
    def test_get_validation_statistics(self, validator):
        """Test getting validation statistics."""
        stats = validator.get_validation_statistics()

        assert "max_velocity" in stats
        assert "max_acceleration" in stats
        assert "overlap_tolerance" in stats
        assert "position_tolerance" in stats
        assert "auto_correction_enabled" in stats


# ============================================================================
# BALL VALIDATION TESTS
# ============================================================================


class TestBallValidation:
    """Test individual ball validation."""

    @pytest.mark.unit()
    def test_valid_ball(self, validator, cue_ball):
        """Test validation of a valid ball."""
        result = validator._validate_individual_ball(cue_ball)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_invalid_ball_radius(self, validator):
        """Test ball with invalid radius."""
        # Ball creation should fail in __post_init__
        with pytest.raises(ValueError):
            BallState(
                id="test",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=-10.0,  # Invalid negative radius
                mass=0.17,
            )

    @pytest.mark.unit()
    def test_invalid_ball_mass(self, validator):
        """Test ball with invalid mass."""
        # Ball creation should fail in __post_init__
        with pytest.raises(ValueError):
            BallState(
                id="test",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=-0.1,  # Invalid negative mass
            )

    @pytest.mark.unit()
    def test_invalid_ball_confidence(self, validator):
        """Test ball with invalid confidence."""
        # Ball creation should fail in __post_init__
        with pytest.raises(ValueError):
            BallState(
                id="test",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                confidence=1.5,  # Invalid (> 1.0)
            )

    @pytest.mark.unit()
    def test_ball_excessive_velocity_no_correction(self, strict_validator):
        """Test ball with excessive velocity without auto-correction."""
        ball = BallState(
            id="test",
            position=Vector2D.from_4k(1000.0, 800.0),
            velocity=Vector2D.from_4k(10000.0, 10000.0),  # Very high velocity
            radius=36.0,
            mass=0.17,
        )

        result = strict_validator._validate_individual_ball(ball)
        assert result.is_valid is False
        assert any("velocity" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_ball_excessive_velocity_with_correction(self, lenient_validator):
        """Test ball with excessive velocity with auto-correction."""
        ball = BallState(
            id="test",
            position=Vector2D.from_4k(1000.0, 800.0),
            velocity=Vector2D.from_4k(50000.0, 50000.0),  # Very high velocity
            radius=36.0,
            mass=0.17,
        )

        result = lenient_validator._validate_individual_ball(ball)
        # Should have correction
        assert any("velocity" in key for key in result.corrected_values)

    @pytest.mark.unit()
    def test_ball_none_velocity(self, validator):
        """Test ball with None velocity."""
        ball = BallState(
            id="test",
            position=Vector2D.from_4k(1000.0, 800.0),
            velocity=None,  # Invalid None velocity
            radius=36.0,
            mass=0.17,
        )

        result = validator._validate_individual_ball(ball)
        assert result.is_valid is False
        assert any("velocity" in e.lower() for e in result.errors)


# ============================================================================
# BALL OVERLAP TESTS
# ============================================================================


class TestBallOverlaps:
    """Test ball overlap detection."""

    @pytest.mark.unit()
    def test_no_overlaps(self, validator):
        """Test balls with no overlaps."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1200.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._check_ball_overlaps(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_overlapping_balls_no_correction(self, strict_validator):
        """Test overlapping balls without auto-correction."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1050.0, 800.0),  # Overlapping
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = strict_validator._check_ball_overlaps(balls)
        assert result.is_valid is False
        assert any("overlap" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_overlapping_balls_with_correction(self, validator):
        """Test overlapping balls with auto-correction."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1070.0, 800.0),  # Slightly overlapping
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._check_ball_overlaps(balls)
        # Should have position corrections
        assert len(result.corrected_values) > 0

    @pytest.mark.unit()
    def test_touching_balls_within_tolerance(self, validator):
        """Test balls touching but within tolerance."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1072.5, 800.0),  # Just touching
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._check_ball_overlaps(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_multiple_overlapping_balls(self, validator):
        """Test detection of multiple overlapping balls."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1050.0, 800.0),  # Overlaps with ball_1
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_3",
                position=Vector2D.from_4k(1100.0, 800.0),  # Overlaps with ball_2
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._check_ball_overlaps(balls)
        # Should detect multiple overlaps or have corrections
        assert not result.is_valid or len(result.corrected_values) > 0


# ============================================================================
# BALL POSITIONS VALIDATION TESTS
# ============================================================================


class TestBallPositionsValidation:
    """Test validate_ball_positions method."""

    @pytest.mark.unit()
    def test_empty_ball_list(self, validator):
        """Test validation of empty ball list."""
        result = validator.validate_ball_positions([])
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_valid_ball_positions(self, validator, cue_ball, object_ball):
        """Test validation of valid ball positions."""
        result = validator.validate_ball_positions([cue_ball, object_ball])
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_duplicate_ball_ids(self, validator):
        """Test detection of duplicate ball IDs."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_1",  # Duplicate ID
                position=Vector2D.from_4k(1200.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is False
        assert any("duplicate" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_no_cue_ball(self, validator, object_ball):
        """Test detection of missing cue ball."""
        balls = [object_ball]  # No cue ball

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is False
        assert any("cue ball" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_multiple_cue_balls(self, validator):
        """Test detection of multiple cue balls."""
        balls = [
            BallState(
                id="cue_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
            BallState(
                id="cue_2",
                position=Vector2D.from_4k(1200.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
        ]

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is False
        assert any(
            "multiple" in e.lower() and "cue ball" in e.lower() for e in result.errors
        )

    @pytest.mark.unit()
    def test_pocketed_balls_ignored_for_cue_ball_check(self, validator):
        """Test that pocketed cue balls don't count."""
        balls = [
            BallState(
                id="cue_pocketed",
                position=Vector2D.from_4k(0.0, 0.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
                is_pocketed=True,
            ),
            BallState(
                id="cue_active",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
                is_pocketed=False,
            ),
        ]

        result = validator.validate_ball_positions(balls)
        # Should be valid - only one active cue ball
        assert result.is_valid is True


# ============================================================================
# TABLE VALIDATION TESTS
# ============================================================================


class TestTableValidation:
    """Test table state validation."""

    @pytest.mark.unit()
    def test_valid_table(self, validator, simple_table):
        """Test validation of valid table."""
        result = validator.validate_table_state(simple_table)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_invalid_table_width(self, validator):
        """Test table with invalid width."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=-100.0,  # Invalid negative width
                height=1600.0,
                pocket_positions=pockets,
            )

    @pytest.mark.unit()
    def test_invalid_table_height(self, validator):
        """Test table with invalid height."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=3200.0,
                height=0.0,  # Invalid zero height
                pocket_positions=pockets,
            )

    @pytest.mark.unit()
    def test_table_exceeds_4k_bounds(self, validator):
        """Test table dimensions exceeding 4K bounds."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=5000.0,  # Exceeds CANONICAL_WIDTH
            height=3000.0,  # Exceeds CANONICAL_HEIGHT
            pocket_positions=pockets,
        )

        result = validator.validate_table_state(table)
        assert result.is_valid is False

    @pytest.mark.unit()
    def test_unusual_aspect_ratio(self, validator):
        """Test table with unusual aspect ratio."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=3200.0,
            height=3200.0,  # Square table (unusual for pool)
            pocket_positions=pockets,
        )

        result = validator.validate_table_state(table)
        # Should have warning about aspect ratio
        assert len(result.warnings) > 0

    @pytest.mark.unit()
    def test_wrong_number_of_pockets(self, validator):
        """Test table with wrong number of pockets."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(4)]  # Only 4 pockets
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
            )

    @pytest.mark.unit()
    def test_invalid_cushion_elasticity(self, validator):
        """Test table with invalid cushion elasticity."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                cushion_elasticity=1.5,  # Invalid (> 1.0)
            )

    @pytest.mark.unit()
    def test_negative_surface_friction(self, validator):
        """Test table with negative surface friction."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                surface_friction=-0.1,  # Invalid negative
            )

    @pytest.mark.unit()
    def test_invalid_pocket_radius(self, validator):
        """Test table with invalid pocket radius."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        # Table creation should fail in __post_init__
        with pytest.raises(ValueError):
            TableState(
                width=3200.0,
                height=1600.0,
                pocket_positions=pockets,
                pocket_radius=-10.0,  # Invalid negative
            )


# ============================================================================
# CUE VALIDATION TESTS
# ============================================================================


class TestCueValidation:
    """Test cue state validation."""

    @pytest.mark.unit()
    def test_valid_cue(self, validator, simple_cue, simple_table):
        """Test validation of valid cue."""
        result = validator.validate_cue_state(simple_cue, simple_table)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_invalid_cue_length(self, validator, simple_table):
        """Test cue with invalid length."""
        # Cue creation should fail in __post_init__
        with pytest.raises(ValueError):
            CueState(
                tip_position=Vector2D.from_4k(1500.0, 800.0),
                angle=0.0,
                length=-1.0,  # Invalid negative length
                tip_radius=0.006,
            )

    @pytest.mark.unit()
    def test_invalid_cue_tip_radius(self, validator, simple_table):
        """Test cue with invalid tip radius."""
        # Cue creation should fail in __post_init__
        with pytest.raises(ValueError):
            CueState(
                tip_position=Vector2D.from_4k(1500.0, 800.0),
                angle=0.0,
                length=1.47,
                tip_radius=0.0,  # Invalid zero radius
            )

    @pytest.mark.unit()
    def test_invalid_cue_confidence(self, validator, simple_table):
        """Test cue with invalid confidence."""
        # Cue creation should fail in __post_init__
        with pytest.raises(ValueError):
            CueState(
                tip_position=Vector2D.from_4k(1500.0, 800.0),
                angle=0.0,
                length=1.47,
                tip_radius=0.006,
                confidence=1.5,  # Invalid (> 1.0)
            )

    @pytest.mark.unit()
    def test_negative_cue_force(self, validator, simple_table):
        """Test cue with negative force."""
        cue = CueState(
            tip_position=Vector2D.from_4k(1500.0, 800.0),
            angle=0.0,
            length=1.47,
            tip_radius=0.006,
            estimated_force=-10.0,  # Invalid negative
        )

        result = validator.validate_cue_state(cue, simple_table)
        assert result.is_valid is False
        assert any("force" in e.lower() for e in result.errors)


# ============================================================================
# GAME STATE VALIDATION TESTS
# ============================================================================


class TestGameStateValidation:
    """Test complete game state validation."""

    @pytest.mark.unit()
    def test_valid_game_state(self, validator, simple_game_state):
        """Test validation of valid game state."""
        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_invalid_frame_number(self, validator, simple_game_state):
        """Test game state with invalid frame number."""
        simple_game_state.frame_number = -1

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any("frame" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_invalid_timestamp(self, validator, simple_game_state):
        """Test game state with invalid timestamp."""
        simple_game_state.timestamp = -1.0

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any("timestamp" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_invalid_state_confidence(self, validator, simple_game_state):
        """Test game state with invalid confidence."""
        simple_game_state.state_confidence = 1.5

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any("confidence" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_game_state_with_cue(self, validator, simple_game_state, simple_cue):
        """Test game state validation with cue."""
        simple_game_state.cue = simple_cue

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_game_state_ball_outside_table(self, validator, simple_game_state):
        """Test game state with ball outside table bounds."""
        # Move ball outside table
        simple_game_state.balls[1].position = Vector2D.from_4k(5000.0, 5000.0)

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any(
            "outside" in e.lower() or "bounds" in e.lower() for e in result.errors
        )

    @pytest.mark.unit()
    def test_game_state_with_negative_score(self, validator, simple_game_state):
        """Test game state with negative score."""
        simple_game_state.scores = {1: -5}

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any("score" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_game_state_with_invalid_player(self, validator, simple_game_state):
        """Test game state with invalid current player."""
        simple_game_state.current_player = 5  # Invalid player number

        result = validator.validate_game_state(simple_game_state)
        assert result.is_valid is False
        assert any("player" in e.lower() for e in result.errors)


# ============================================================================
# PHYSICS VALIDATION TESTS
# ============================================================================


class TestPhysicsValidation:
    """Test physics consistency validation."""

    @pytest.mark.unit()
    def test_physics_validation_no_previous_state(self, validator, simple_game_state):
        """Test physics validation with no previous state."""
        result = validator.validate_physics(simple_game_state, None)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_physics_validation_invalid_dt(self, validator, simple_game_state):
        """Test physics validation with invalid time delta."""
        previous_state = simple_game_state.copy()

        result = validator.validate_physics(simple_game_state, previous_state, dt=-0.1)
        assert result.is_valid is False
        assert any("time" in e.lower() or "delta" in e.lower() for e in result.errors)

    @pytest.mark.unit()
    def test_physics_validation_with_previous_state(self, validator, simple_game_state):
        """Test physics validation with previous state."""
        previous_state = simple_game_state.copy()
        current_state = simple_game_state.copy()

        # Move ball slightly
        current_state.balls[0].position = Vector2D.from_4k(1605.0, 800.0)

        result = validator.validate_physics(current_state, previous_state, dt=0.033)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_physics_validation_new_ball(self, validator, simple_game_state):
        """Test physics validation with new ball added."""
        previous_state = simple_game_state.copy()
        current_state = simple_game_state.copy()

        # Add new ball
        new_ball = BallState(
            id="ball_new",
            position=Vector2D.from_4k(1800.0, 800.0),
            velocity=Vector2D.from_4k(0.0, 0.0),
            radius=36.0,
            mass=0.17,
            number=2,
        )
        current_state.balls.append(new_ball)

        result = validator.validate_physics(current_state, previous_state, dt=0.033)
        # Should still be valid - new balls are allowed
        assert result.is_valid is True


# ============================================================================
# OBJECT CONSISTENCY TESTS
# ============================================================================


class TestObjectConsistency:
    """Test consistency between objects."""

    @pytest.mark.unit()
    def test_ball_in_pocket_auto_correction(self, validator, simple_table):
        """Test auto-correction for ball in pocket."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(5.0, 5.0),  # Very close to pocket at (0, 0)
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_pocketed=False,  # Not marked as pocketed
            ),
        ]

        result = validator._validate_object_consistency(balls, simple_table, None)
        # Should have correction to mark as pocketed
        if validator.enable_auto_correction:
            assert len(result.corrected_values) > 0

    @pytest.mark.unit()
    def test_cue_ball_interaction(self, validator, simple_table, simple_cue):
        """Test cue-ball interaction validation."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
        ]

        result = validator._validate_object_consistency(balls, simple_table, simple_cue)
        assert result.is_valid is True


# ============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit()
    def test_single_ball(self, validator):
        """Test validation with single ball."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
        ]

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_many_balls(self, validator):
        """Test validation with many balls."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            )
        ]

        # Add 15 numbered balls
        for i in range(1, 16):
            ball = BallState(
                id=f"ball_{i}",
                position=Vector2D.from_4k(2000.0 + i * 100, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                number=i,
            )
            balls.append(ball)

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_ball_at_table_edge(self, validator, simple_table):
        """Test ball at table edge."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(36.0, 800.0),  # At left edge
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._validate_object_consistency(balls, simple_table, None)
        # Ball at edge should be valid
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_zero_velocity_balls(self, validator):
        """Test balls with zero velocity."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
            ),
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(2000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_all_balls_pocketed_except_cue(self, validator):
        """Test with all balls pocketed except cue ball."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D.from_4k(1600.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                is_cue_ball=True,
                is_pocketed=False,
            ),
        ]

        # Add pocketed balls
        for i in range(1, 16):
            ball = BallState(
                id=f"ball_{i}",
                position=Vector2D.from_4k(0.0, 0.0),  # Position doesn't matter
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
                number=i,
                is_pocketed=True,
            )
            balls.append(ball)

        result = validator.validate_ball_positions(balls)
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_minimum_table_size(self, validator):
        """Test with minimum valid table size."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=100.0,  # Very small but positive
            height=50.0,
            pocket_positions=pockets,
        )

        result = validator.validate_table_state(table)
        # Should be valid even if small
        assert result.is_valid is True

    @pytest.mark.unit()
    def test_maximum_table_size(self, validator):
        """Test with maximum valid table size (4K bounds)."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=CANONICAL_WIDTH,
            height=CANONICAL_HEIGHT,
            pocket_positions=pockets,
        )

        result = validator.validate_table_state(table)
        assert result.is_valid is True


# ============================================================================
# AUTO-CORRECTION TESTS
# ============================================================================


class TestAutoCorrection:
    """Test auto-correction functionality."""

    @pytest.mark.unit()
    def test_velocity_clamping(self, validator):
        """Test velocity clamping with auto-correction."""
        ball = BallState(
            id="test",
            position=Vector2D.from_4k(1000.0, 800.0),
            velocity=Vector2D.from_4k(50000.0, 50000.0),
            radius=36.0,
            mass=0.17,
        )

        result = validator._validate_individual_ball(ball)
        # Should have velocity correction
        assert any("velocity" in key for key in result.corrected_values)

    @pytest.mark.unit()
    def test_none_velocity_correction(self, validator):
        """Test None velocity correction."""
        ball = BallState(
            id="test",
            position=Vector2D.from_4k(1000.0, 800.0),
            velocity=None,
            radius=36.0,
            mass=0.17,
        )

        result = validator._validate_individual_ball(ball)
        if validator.enable_auto_correction:
            assert any("velocity" in key for key in result.corrected_values)

    @pytest.mark.unit()
    def test_overlap_separation(self, validator):
        """Test overlap separation with auto-correction."""
        balls = [
            BallState(
                id="ball_1",
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
            BallState(
                id="ball_2",
                position=Vector2D.from_4k(1070.0, 800.0),  # Slightly overlapping
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        result = validator._check_ball_overlaps(balls)
        if validator.enable_auto_correction:
            # Should have position corrections for separation
            assert len(result.corrected_values) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test integration of multiple validation components."""

    @pytest.mark.unit()
    def test_full_game_state_validation_flow(self, validator, simple_game_state):
        """Test complete validation flow through game state."""
        result = validator.validate_game_state(simple_game_state)

        # Should validate all components
        assert result.is_valid is True
        # No errors
        assert len(result.errors) == 0

    @pytest.mark.unit()
    def test_multiple_validation_errors(self, validator):
        """Test game state with multiple validation errors."""
        # Create invalid game state - some validation happens in __post_init__
        # so we create valid objects but with invalid game logic
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=3200.0,
            height=1600.0,
            pocket_positions=pockets,
            cushion_elasticity=0.85,
        )

        balls = [
            BallState(
                id="ball_1",  # No cue ball - this is a validation error
                position=Vector2D.from_4k(1000.0, 800.0),
                velocity=Vector2D.from_4k(0.0, 0.0),
                radius=36.0,
                mass=0.17,
            ),
        ]

        # Create game state with validation errors caught by validator (not __post_init__)
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            state_confidence=1.0,
            scores={1: -5},  # Negative score
            current_player=5,  # Invalid player
        )

        result = validator.validate_game_state(game_state)

        # Should have multiple errors (no cue ball, negative score, invalid player)
        assert result.is_valid is False
        assert len(result.errors) >= 2

    @pytest.mark.unit()
    def test_validation_with_warnings_only(self, validator):
        """Test validation that passes but has warnings."""
        pockets = [Vector2D.from_4k(0, 0) for _ in range(6)]
        table = TableState(
            width=2000.0,
            height=2000.0,  # Square (unusual aspect ratio) but within 4K bounds
            pocket_positions=pockets,
        )

        result = validator.validate_table_state(table)

        # Should be valid but have warnings about aspect ratio
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("aspect ratio" in w.lower() for w in result.warnings)
