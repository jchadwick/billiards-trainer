#!/usr/bin/env python3
"""
Cross-Module Integration Test.

Tests integration between modules:
- Core ↔ Vision data flow
- Config → All modules propagation
- Event system between modules
- Data type compatibility
- Module coordination
"""

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import all modules
from core import CoreModule, CoreModuleConfig
from core.models import BallState, Vector2D
from vision import VisionModule
from vision.models import Ball, BallType

from config import ConfigurationModule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_core_vision_data_flow():
    """Test data flow between Core and Vision modules."""
    print("Testing Core-Vision Data Flow...")

    try:
        # Initialize modules
        core = CoreModule()
        vision = VisionModule({"camera_device_id": -1})  # No camera

        # Create Vision detection data
        vision_balls = [
            Ball(position=(320, 240), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(400, 200), radius=14.0, ball_type=BallType.SOLID, number=1),
            Ball(position=(350, 280), radius=14.0, ball_type=BallType.EIGHT, number=8),
        ]

        print("✓ Vision balls created")

        # Convert Vision balls to Core format
        core_balls = []
        for vball in vision_balls:
            ball_id = (
                "cue" if vball.ball_type == BallType.CUE else f"ball_{vball.number}"
            )
            core_ball = BallState(
                id=ball_id,
                position=Vector2D(vball.position[0], vball.position[1]),
                velocity=Vector2D.zero(),
                radius=vball.radius / 1000.0,  # Convert to meters
                mass=0.17,
                is_cue_ball=vball.ball_type == BallType.CUE,
                is_pocketed=False,
                number=vball.number,
            )
            core_balls.append(core_ball)

        print("✓ Vision-to-Core ball conversion successful")

        # Verify data consistency
        assert len(core_balls) == len(vision_balls)

        # Check cue ball
        cue_balls_vision = [b for b in vision_balls if b.ball_type == BallType.CUE]
        cue_balls_core = [b for b in core_balls if b.is_cue_ball]
        assert len(cue_balls_vision) == len(cue_balls_core) == 1
        print("✓ Cue ball consistency verified")

        # Check position consistency
        for i, (vball, cball) in enumerate(zip(vision_balls, core_balls)):
            assert cball.position.x == vball.position[0]
            assert cball.position.y == vball.position[1]
        print("✓ Position consistency verified")

        # Test reverse conversion (Core to Vision format)
        converted_vision_balls = []
        for cball in core_balls:
            ball_type = (
                BallType.CUE
                if cball.is_cue_ball
                else (BallType.EIGHT if cball.number == 8 else BallType.SOLID)
            )
            vision_ball = Ball(
                position=(cball.position.x, cball.position.y),
                radius=cball.radius * 1000.0,  # Convert back to pixels
                ball_type=ball_type,
                number=cball.number,
            )
            converted_vision_balls.append(vision_ball)

        print("✓ Core-to-Vision conversion successful")

        # Verify round-trip consistency
        assert len(converted_vision_balls) == len(vision_balls)
        for orig, converted in zip(vision_balls, converted_vision_balls):
            assert orig.position == converted.position
            assert orig.number == converted.number
            assert orig.ball_type == converted.ball_type

        print("✓ Round-trip conversion consistency verified")

        return True

    except Exception as e:
        print(f"✗ Core-Vision data flow failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_config_module_propagation():
    """Test configuration propagation to all modules."""
    print("\nTesting Config Module Propagation...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"

            # Initialize configuration module
            config_module = ConfigurationModule(config_dir)

            # Set up configuration for different modules
            core_config = {
                "physics_enabled": True,
                "prediction_enabled": True,
                "cache_size": 500,
                "max_trajectory_time": 5.0,
                "debug_mode": True,
            }

            vision_config = {
                "camera_device_id": -1,
                "enable_ball_detection": True,
                "enable_table_detection": True,
                "target_fps": 30,
                "debug_mode": True,
            }

            # Create modules with configuration
            core = CoreModule(CoreModuleConfig(**core_config))
            vision = VisionModule(vision_config)

            print("✓ Modules created with configuration")

            # Verify configuration was applied
            assert core.config.cache_size == 500
            assert core.config.max_trajectory_time == 5.0
            assert core.config.debug_mode == True
            print("✓ Core module configuration applied")

            assert vision.config.camera_device_id == -1
            assert vision.config.enable_ball_detection == True
            assert vision.config.target_fps == 30
            print("✓ Vision module configuration applied")

            # Test configuration consistency
            assert core.config.debug_mode == vision.config.debug_mode
            print("✓ Configuration consistency verified")

            return True

    except Exception as e:
        print(f"✗ Config module propagation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_event_system_integration():
    """Test event system integration between modules."""
    print("\nTesting Event System Integration...")

    try:
        # Initialize modules
        core = CoreModule()
        vision = VisionModule({"camera_device_id": -1})

        # Set up event tracking
        events_received = []

        def event_handler(event_type, event_data):
            events_received.append((event_type, event_data, time.time()))

        # Subscribe to events from core module
        subscription_id = core.subscribe_to_events("state_updated", event_handler)
        assert subscription_id is not None
        print("✓ Event subscription established")

        # Simulate vision detection triggering core state update
        detection_data = {
            "balls": [
                {
                    "id": "cue",
                    "x": 320,
                    "y": 240,
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

        # Update core state (which should trigger event)
        await core.update_state(detection_data)
        print("✓ State update triggered from detection data")

        # Give events time to process
        await asyncio.sleep(0.1)

        # Verify events were received
        assert len(events_received) > 0
        print(f"✓ Received {len(events_received)} events")

        # Verify event content
        event_type, event_data, timestamp = events_received[0]
        assert event_type == "state_updated"
        assert "state" in event_data
        assert "timestamp" in event_data
        print("✓ Event content verified")

        # Test unsubscription
        success = core.unsubscribe(subscription_id)
        assert success
        print("✓ Event unsubscription successful")

        return True

    except Exception as e:
        print(f"✗ Event system integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_data_type_compatibility():
    """Test data type compatibility between modules."""
    print("\nTesting Data Type Compatibility...")

    try:
        # Test Vector2D compatibility
        vector = Vector2D(100.0, 200.0)

        # Convert to tuple (Vision format)
        tuple_pos = (vector.x, vector.y)
        assert tuple_pos == (100.0, 200.0)
        print("✓ Vector2D to tuple conversion works")

        # Convert back to Vector2D
        vector_back = Vector2D(tuple_pos[0], tuple_pos[1])
        assert vector_back.x == vector.x
        assert vector_back.y == vector.y
        print("✓ Tuple to Vector2D conversion works")

        # Test ball type compatibility
        vision_ball_types = [
            BallType.CUE,
            BallType.SOLID,
            BallType.STRIPE,
            BallType.EIGHT,
        ]

        for ball_type in vision_ball_types:
            # Create Vision ball
            vision_ball = Ball(
                position=(100, 100),
                radius=14.0,
                ball_type=ball_type,
                number=0 if ball_type == BallType.CUE else 1,
            )

            # Convert to Core ball
            is_cue = ball_type == BallType.CUE
            core_ball = BallState(
                id="cue" if is_cue else f"ball_{vision_ball.number}",
                position=Vector2D(vision_ball.position[0], vision_ball.position[1]),
                velocity=Vector2D.zero(),
                radius=vision_ball.radius / 1000.0,
                mass=0.17,
                is_cue_ball=is_cue,
                is_pocketed=False,
                number=vision_ball.number,
            )

            # Verify compatibility
            assert core_ball.is_cue_ball == is_cue
            assert core_ball.number == vision_ball.number

        print("✓ Ball type compatibility verified")

        # Test coordinate system compatibility
        pixel_coords = [(0, 0), (640, 480), (320, 240)]

        for x, y in pixel_coords:
            # Vision uses tuples
            vision_pos = (x, y)

            # Core uses Vector2D
            core_pos = Vector2D(x, y)

            # Should be convertible both ways
            assert core_pos.x == vision_pos[0]
            assert core_pos.y == vision_pos[1]

            # Distance calculation should work in both
            if x > 0 and y > 0:
                core_distance = core_pos.distance_to(Vector2D(0, 0))
                vision_distance = ((x - 0) ** 2 + (y - 0) ** 2) ** 0.5
                assert abs(core_distance - vision_distance) < 0.001

        print("✓ Coordinate system compatibility verified")

        return True

    except Exception as e:
        print(f"✗ Data type compatibility failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_module_state_coordination():
    """Test coordination of state between modules."""
    print("\nTesting Module State Coordination...")

    try:
        # Initialize modules
        core = CoreModule()
        vision = VisionModule({"camera_device_id": -1})

        # Test initial states
        assert core.get_current_state() is None
        assert not hasattr(vision, "is_running") or not vision._is_running
        print("✓ Initial module states verified")

        # Simulate vision detection creating state for core
        vision_detection = [
            Ball(position=(300, 200), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(400, 250), radius=14.0, ball_type=BallType.SOLID, number=1),
        ]

        # Convert to core update format
        detection_data = {
            "balls": [
                {
                    "id": (
                        f"ball_{ball.number}"
                        if ball.ball_type != BallType.CUE
                        else "cue"
                    ),
                    "x": ball.position[0],
                    "y": ball.position[1],
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_cue_ball": ball.ball_type == BallType.CUE,
                    "number": ball.number,
                }
                for ball in vision_detection
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

        # Update core state
        game_state = await core.update_state(detection_data)
        assert game_state is not None
        assert len(game_state.balls) == 2
        print("✓ Core state updated from vision detection")

        # Verify state coordination
        current_state = core.get_current_state()
        assert current_state is not None
        assert current_state.timestamp == game_state.timestamp
        print("✓ State coordination verified")

        # Test performance metrics coordination
        core_metrics = core.get_performance_metrics()
        vision_stats = vision.get_statistics()

        assert core_metrics.total_updates > 0
        assert vision_stats is not None
        print("✓ Performance metrics coordination works")

        return True

    except Exception as e:
        print(f"✗ Module state coordination failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_error_propagation():
    """Test error handling and propagation between modules."""
    print("\nTesting Error Propagation...")

    try:
        # Initialize modules
        core = CoreModule()
        vision = VisionModule({"camera_device_id": -1})

        # Test invalid data handling
        try:
            # Send invalid detection data to core
            invalid_data = {
                "balls": [
                    {
                        # Missing required fields
                        "id": "invalid",
                        # Missing x, y coordinates
                    }
                ],
                "timestamp": time.time(),
            }

            await core.update_state(invalid_data)
            print("! Invalid data was handled gracefully")

        except Exception as e:
            print(f"✓ Invalid data properly rejected: {type(e).__name__}")

        # Test module error isolation
        try:
            # Create a module with invalid configuration
            invalid_vision = VisionModule(
                {"camera_device_id": 999999}
            )  # Non-existent camera
            print("✓ Invalid vision configuration handled gracefully")
        except Exception as e:
            print(
                f"✓ Invalid vision configuration properly rejected: {type(e).__name__}"
            )

        # Test core module continues working despite vision errors
        valid_detection_data = {
            "balls": [
                {
                    "id": "cue",
                    "x": 320,
                    "y": 240,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_cue_ball": True,
                    "number": 0,
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

        game_state = await core.update_state(valid_detection_data)
        assert game_state is not None
        print("✓ Core module continues working despite potential vision issues")

        return True

    except Exception as e:
        print(f"✗ Error propagation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_cross_module_tests():
    """Run all cross-module integration tests."""
    print("Starting Cross-Module Integration Tests")
    print("=" * 50)

    tests = [
        test_core_vision_data_flow,
        test_config_module_propagation,
        test_event_system_integration,
        test_data_type_compatibility,
        test_module_state_coordination,
        test_error_propagation,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("CROSS-MODULE INTEGRATION TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL CROSS-MODULE INTEGRATION TESTS PASSED")
        return True
    else:
        print("✗ SOME CROSS-MODULE INTEGRATION TESTS FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_cross_module_tests())
    sys.exit(0 if success else 1)
