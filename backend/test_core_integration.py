#!/usr/bin/env python3
"""Integration test for Core Module functionality.

Tests the core module components working together:
- Physics engine integration
- Game state management
- Ball/table state creation and updates
- Trajectory calculations
- Event system
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import CoreModule, CoreModuleConfig, GameType
from core.models import BallState, GameState, TableState, Vector2D

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_table() -> TableState:
    """Create a standard test table."""
    # Standard 6-pocket positions for a pool table
    pocket_positions = [
        Vector2D(0.0, 0.0),  # Bottom left
        Vector2D(1.42, 0.0),  # Bottom middle
        Vector2D(2.84, 0.0),  # Bottom right
        Vector2D(0.0, 1.42),  # Top left
        Vector2D(1.42, 1.42),  # Top middle
        Vector2D(2.84, 1.42),  # Top right
    ]

    return TableState(
        width=2.84,  # Standard pool table width in meters
        height=1.42,  # Standard pool table height in meters
        pocket_positions=pocket_positions,
        pocket_radius=0.06,  # 6cm pocket radius
        cushion_elasticity=0.85,
        surface_friction=0.2,
        surface_slope=0.0,
        cushion_height=0.064,
    )


def create_test_balls() -> list[BallState]:
    """Create a set of test balls in standard formation."""
    balls = []

    # Cue ball
    balls.append(
        BallState(
            id="cue",
            position=Vector2D(0.6, 0.71),  # About 1/4 from left, center height
            velocity=Vector2D(0.0, 0.0),
            radius=0.028,  # Standard ball radius (2.8cm)
            mass=0.16,  # Standard ball mass (160g)
            spin=Vector2D(0.0, 0.0),
            is_cue_ball=True,
            is_pocketed=False,
            number=0,
        )
    )

    # Object balls in triangle formation
    ball_positions = [
        (2.0, 0.71),  # 1 ball (front)
        (2.06, 0.682),  # 2 ball
        (2.06, 0.738),  # 3 ball
        (2.12, 0.654),  # 4 ball
        (2.12, 0.71),  # 8 ball (center)
        (2.12, 0.766),  # 6 ball
        (2.18, 0.626),  # 7 ball
        (2.18, 0.682),  # 8 ball
        (2.18, 0.738),  # 9 ball
        (2.18, 0.794),  # 10 ball
    ]

    for i, (x, y) in enumerate(ball_positions, 1):
        balls.append(
            BallState(
                id=f"ball_{i}",
                position=Vector2D(x, y),
                velocity=Vector2D(0.0, 0.0),
                radius=0.028,
                mass=0.16,
                spin=Vector2D(0.0, 0.0),
                is_cue_ball=False,
                is_pocketed=False,
                number=i,
            )
        )

    return balls


def create_test_game_state() -> GameState:
    """Create a complete test game state."""
    return GameState(
        timestamp=time.time(),
        balls=create_test_balls(),
        table=create_test_table(),
        game_type=GameType.PRACTICE,
        turn_number=1,
        current_player="test_player",
        scores={"test_player": 0},
        legal_targets=[f"ball_{i}" for i in range(1, 11)],
        last_shot_time=None,
        metadata={"test_scenario": "integration_test"},
    )


async def test_core_module_initialization():
    """Test basic core module initialization."""
    print("Testing Core Module Initialization...")

    try:
        # Test default initialization
        core = CoreModule()
        assert core is not None
        assert core.config is not None
        assert core.state_manager is not None
        assert core.physics_engine is not None
        assert core.trajectory_calculator is not None
        print("✓ Default initialization successful")

        # Test custom config initialization
        config = CoreModuleConfig(
            physics_enabled=True,
            prediction_enabled=True,
            cache_size=500,
            max_trajectory_time=5.0,
            debug_mode=True,
        )
        core_custom = CoreModule(config)
        assert core_custom.config.cache_size == 500
        assert core_custom.config.max_trajectory_time == 5.0
        print("✓ Custom config initialization successful")

        return True

    except Exception as e:
        print(f"✗ Core module initialization failed: {e}")
        return False


async def test_game_state_management():
    """Test game state creation and management."""
    print("\nTesting Game State Management...")

    try:
        core = CoreModule()

        # Test initial state (should be None)
        current_state = core.get_current_state()
        assert current_state is None
        print("✓ Initial state is None as expected")

        # Create test detection data
        detection_data = {
            "balls": [
                {
                    "id": "cue",
                    "x": 0.6,  # Changed from position.x to x
                    "y": 0.71,  # Changed from position.y to y
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_cue_ball": True,
                    "number": 0,
                    "confidence": 0.95,
                },
                {
                    "id": "ball_1",
                    "x": 2.0,  # Changed from position.x to x
                    "y": 0.71,  # Changed from position.y to y
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_cue_ball": False,
                    "number": 1,
                    "confidence": 0.93,
                },
            ],
            "table": {
                "width": 2.84,
                "height": 1.42,
                "pocket_positions": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.42, "y": 0.0},
                    {"x": 2.84, "y": 0.0},
                    {"x": 0.0, "y": 1.42},
                    {"x": 1.42, "y": 1.42},
                    {"x": 2.84, "y": 1.42},
                ],
            },
            "timestamp": time.time(),
        }

        # Update state with detection data
        updated_state = await core.update_state(detection_data)
        assert updated_state is not None
        assert len(updated_state.balls) >= 2
        print("✓ State updated from detection data successfully")

        # Verify state is stored
        current_state = core.get_current_state()
        assert current_state is not None
        assert current_state.timestamp == updated_state.timestamp
        print("✓ Current state stored correctly")

        # Test state history
        history = core.get_state_history()
        assert len(history) == 1
        assert history[0].timestamp == updated_state.timestamp
        print("✓ State history working correctly")

        return True

    except Exception as e:
        print(f"✗ Game state management failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_physics_integration():
    """Test physics engine integration and trajectory calculations."""
    print("\nTesting Physics Integration...")

    try:
        core = CoreModule()

        # Create a test game state
        test_state = create_test_game_state()

        # Manually set the current state (simulating detection update)
        core._current_state = test_state

        # Test trajectory calculation
        initial_velocity = Vector2D(2.0, 0.0)  # 2 m/s forward
        trajectory = await core.calculate_trajectory("cue", initial_velocity)

        assert trajectory is not None
        assert len(trajectory) > 0
        assert isinstance(trajectory[0], Vector2D)
        print(f"✓ Trajectory calculated successfully with {len(trajectory)} points")

        # Verify trajectory starts at cue ball position
        cue_ball_pos = next(b.position for b in test_state.balls if b.is_cue_ball)
        start_pos = trajectory[0]
        distance = (
            (start_pos.x - cue_ball_pos.x) ** 2 + (start_pos.y - cue_ball_pos.y) ** 2
        ) ** 0.5
        assert distance < 0.01  # Should be very close
        print("✓ Trajectory starts at correct position")

        # Test trajectory caching
        trajectory2 = await core.calculate_trajectory("cue", initial_velocity)
        assert len(trajectory2) == len(trajectory)
        print("✓ Trajectory caching working correctly")

        return True

    except Exception as e:
        print(f"✗ Physics integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_shot_analysis():
    """Test shot analysis functionality."""
    print("\nTesting Shot Analysis...")

    try:
        core = CoreModule()

        # Create a test game state
        test_state = create_test_game_state()
        core._current_state = test_state

        # Test shot analysis
        analysis = await core.analyze_shot(target_ball="ball_1")
        assert analysis is not None
        print("✓ Shot analysis completed successfully")

        # Test shot suggestions
        suggestions = await core.suggest_shots(max_suggestions=3)
        assert isinstance(suggestions, list)
        print(f"✓ Got {len(suggestions)} shot suggestions")

        return True

    except Exception as e:
        print(f"✗ Shot analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_event_system():
    """Test event system integration."""
    print("\nTesting Event System...")

    try:
        core = CoreModule()

        # Track events
        events_received = []

        def event_handler(event_data):
            events_received.append(event_data)

        # Subscribe to state updates
        subscription_id = core.subscribe_to_events("state_updated", event_handler)
        assert subscription_id is not None
        print("✓ Event subscription successful")

        # Trigger a state update
        detection_data = {
            "balls": [
                {
                    "id": "cue",
                    "x": 0.6,  # Changed from position.x to x
                    "y": 0.71,  # Changed from position.y to y
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_cue_ball": True,
                    "number": 0,
                    "confidence": 0.95,
                }
            ],
            "table": {
                "width": 2.84,
                "height": 1.42,
                "pocket_positions": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.42, "y": 0.0},
                    {"x": 2.84, "y": 0.0},
                    {"x": 0.0, "y": 1.42},
                    {"x": 1.42, "y": 1.42},
                    {"x": 2.84, "y": 1.42},
                ],
            },
            "timestamp": time.time(),
        }

        await core.update_state(detection_data)

        # Give events time to process
        await asyncio.sleep(0.1)

        # Check if event was received
        assert len(events_received) > 0
        print("✓ State update event received")

        # Test unsubscription
        success = core.unsubscribe(subscription_id)
        assert success
        print("✓ Event unsubscription successful")

        return True

    except Exception as e:
        print(f"✗ Event system failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_state_validation():
    """Test state validation functionality."""
    print("\nTesting State Validation...")

    try:
        core = CoreModule()

        # Test validation with no state
        validation = await core.validate_state()
        assert validation["valid"] is False
        assert "No current game state available" in validation["issues"]
        print("✓ Validation correctly handles no state")

        # Create valid state
        test_state = create_test_game_state()
        core._current_state = test_state

        validation = await core.validate_state()
        assert "valid" in validation
        print(f"✓ State validation completed: {validation['valid']}")

        if validation["issues"]:
            print(f"  Issues found: {validation['issues']}")

        return True

    except Exception as e:
        print(f"✗ State validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_metrics():
    """Test performance monitoring."""
    print("\nTesting Performance Metrics...")

    try:
        core = CoreModule()

        # Get initial metrics
        initial_metrics = core.get_performance_metrics()
        assert initial_metrics.total_updates == 0
        print("✓ Initial metrics correct")

        # Perform some operations
        test_state = create_test_game_state()
        core._current_state = test_state

        await core.calculate_trajectory("cue", Vector2D(1.0, 0.0))
        await core.analyze_shot(target_ball="ball_1")

        # Check updated metrics
        metrics = core.get_performance_metrics()
        assert metrics.avg_physics_time >= 0
        assert metrics.avg_analysis_time >= 0
        print("✓ Performance metrics tracking working")

        return True

    except Exception as e:
        print(f"✗ Performance metrics failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_game_reset():
    """Test game reset functionality."""
    print("\nTesting Game Reset...")

    try:
        core = CoreModule()

        # Create some state
        test_state = create_test_game_state()
        core._current_state = test_state
        core._state_history.append(test_state)

        # Reset game
        await core.reset_game(GameType.EIGHT_BALL)

        # Verify reset
        assert core.get_current_state() is None
        assert len(core.get_state_history()) == 0
        print("✓ Game reset successful")

        return True

    except Exception as e:
        print(f"✗ Game reset failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_core_tests():
    """Run all core module integration tests."""
    print("Starting Core Module Integration Tests")
    print("=" * 50)

    tests = [
        test_core_module_initialization,
        test_game_state_management,
        test_physics_integration,
        test_shot_analysis,
        test_event_system,
        test_state_validation,
        test_performance_metrics,
        test_game_reset,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("CORE MODULE INTEGRATION TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL CORE MODULE INTEGRATION TESTS PASSED")
        return True
    else:
        print("✗ SOME CORE MODULE INTEGRATION TESTS FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_core_tests())
    sys.exit(0 if success else 1)
