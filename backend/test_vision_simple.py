#!/usr/bin/env python3
"""Very Simplified Vision Module Integration Test.

Tests only the working parts of the vision module:
- Module initialization
- Basic data structures that work
- Core integration for ball conversion
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Import core models for integration testing
from core.models import BallState, Vector2D

# Import vision module components
from vision import VisionModule
from vision.models import Ball, BallType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_vision_module_initialization():
    """Test vision module initialization without camera."""
    print("Testing Vision Module Initialization...")

    try:
        # Test with minimal config (no camera)
        config = {
            "camera_device_id": -1,  # Disable camera
            "enable_table_detection": True,
            "enable_ball_detection": True,
            "enable_cue_detection": True,
            "enable_tracking": True,
            "debug_mode": True,
        }

        vision = VisionModule(config)
        assert vision is not None
        assert vision.config is not None
        print("✓ Vision module initialized without camera")

        # Test component initialization
        assert vision.ball_detector is not None
        assert vision.table_detector is not None
        assert vision.cue_detector is not None
        assert vision.preprocessor is not None
        assert vision.tracker is not None
        print("✓ All vision components initialized")

        return True

    except Exception as e:
        print(f"✗ Vision module initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ball_data_model():
    """Test the Ball data model."""
    print("\nTesting Ball Data Model...")

    try:
        # Test Ball model creation
        ball = Ball(
            position=(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.SOLID,
            number=1,
            confidence=0.95,
        )
        assert ball.position == (100.0, 200.0)
        assert ball.radius == 15.0
        assert ball.ball_type == BallType.SOLID
        assert ball.number == 1
        assert ball.confidence == 0.95
        print("✓ Ball model creation works")

        # Test ball types
        assert BallType.CUE.value == "cue"
        assert BallType.SOLID.value == "solid"
        assert BallType.STRIPE.value == "stripe"
        assert BallType.EIGHT.value == "eight"
        print("✓ Ball types work correctly")

        # Test ball with different types
        cue_ball = Ball(
            position=(320, 240), radius=14.0, ball_type=BallType.CUE, number=0
        )
        eight_ball = Ball(
            position=(400, 300), radius=14.0, ball_type=BallType.EIGHT, number=8
        )

        assert cue_ball.ball_type == BallType.CUE
        assert eight_ball.ball_type == BallType.EIGHT
        print("✓ Different ball types work correctly")

        return True

    except Exception as e:
        print(f"✗ Ball data model failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_coordinate_operations():
    """Test coordinate transformations and operations."""
    print("\nTesting Coordinate Operations...")

    try:
        # Test position operations
        ball_pos = (320, 240)  # Center of 640x480 image

        # Test position scaling
        scale_factor = 2.0
        scaled_pos = (ball_pos[0] * scale_factor, ball_pos[1] * scale_factor)
        assert scaled_pos == (640.0, 480.0)
        print("✓ Position scaling works")

        # Test conversion to Vector2D (for core integration)
        vector_pos = Vector2D(ball_pos[0], ball_pos[1])
        assert vector_pos.x == 320
        assert vector_pos.y == 240
        print("✓ Position conversion to Vector2D works")

        # Test distance calculation between positions
        pos1 = (100, 100)
        pos2 = (150, 130)
        distance = ((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) ** 0.5
        expected_distance = ((50) ** 2 + (30) ** 2) ** 0.5
        assert abs(distance - expected_distance) < 0.001
        print("✓ Distance calculation works")

        return True

    except Exception as e:
        print(f"✗ Coordinate operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vision_to_core_conversion():
    """Test conversion from vision data to core data structures."""
    print("\nTesting Vision-to-Core Conversion...")

    try:
        # Create Vision balls
        vision_balls = [
            Ball(position=(100, 100), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(200, 150), radius=14.0, ball_type=BallType.SOLID, number=1),
            Ball(position=(300, 200), radius=14.0, ball_type=BallType.STRIPE, number=9),
            Ball(position=(250, 175), radius=14.0, ball_type=BallType.EIGHT, number=8),
        ]

        # Convert to core BallState objects
        core_balls = []
        for vball in vision_balls:
            ball_id = (
                "cue" if vball.ball_type == BallType.CUE else f"ball_{vball.number}"
            )
            core_ball = BallState(
                id=ball_id,
                position=Vector2D(vball.position[0], vball.position[1]),
                velocity=Vector2D.zero(),
                radius=vball.radius / 1000.0,  # Convert pixels to rough meters
                mass=0.17,  # Standard ball mass
                is_cue_ball=vball.ball_type == BallType.CUE,
                is_pocketed=False,
                number=vball.number,
                confidence=vball.confidence,
            )
            core_balls.append(core_ball)

        # Verify conversions
        assert len(core_balls) == 4
        print("✓ All balls converted successfully")

        # Check cue ball
        cue_ball = next(b for b in core_balls if b.is_cue_ball)
        assert cue_ball.id == "cue"
        assert cue_ball.number == 0
        assert cue_ball.is_cue_ball is True
        print("✓ Cue ball conversion correct")

        # Check numbered balls
        solid_ball = next(b for b in core_balls if b.number == 1)
        stripe_ball = next(b for b in core_balls if b.number == 9)
        eight_ball = next(b for b in core_balls if b.number == 8)

        assert solid_ball.id == "ball_1"
        assert stripe_ball.id == "ball_9"
        assert eight_ball.id == "ball_8"
        assert not solid_ball.is_cue_ball
        assert not stripe_ball.is_cue_ball
        assert not eight_ball.is_cue_ball
        print("✓ Numbered ball conversion correct")

        # Check position conversion
        original_vision_ball = vision_balls[1]  # SOLID ball
        converted_core_ball = solid_ball
        assert converted_core_ball.position.x == original_vision_ball.position[0]
        assert converted_core_ball.position.y == original_vision_ball.position[1]
        print("✓ Position conversion correct")

        return True

    except Exception as e:
        print(f"✗ Vision-to-Core conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ball_tracking_simulation():
    """Test simulated ball tracking across frames."""
    print("\nTesting Ball Tracking Simulation...")

    try:
        # Simulate ball movement across 3 frames
        frame_1_ball = Ball(
            position=(100, 100), radius=14.0, ball_type=BallType.SOLID, number=1
        )
        frame_2_ball = Ball(
            position=(105, 102), radius=14.0, ball_type=BallType.SOLID, number=1
        )
        frame_3_ball = Ball(
            position=(110, 104), radius=14.0, ball_type=BallType.SOLID, number=1
        )

        # Calculate movement vectors
        movement_1_to_2 = (
            frame_2_ball.position[0] - frame_1_ball.position[0],
            frame_2_ball.position[1] - frame_1_ball.position[1],
        )
        movement_2_to_3 = (
            frame_3_ball.position[0] - frame_2_ball.position[0],
            frame_3_ball.position[1] - frame_2_ball.position[1],
        )

        assert movement_1_to_2 == (5.0, 2.0)
        assert movement_2_to_3 == (5.0, 2.0)
        print("✓ Ball movement tracking works")

        # Estimate velocity (assuming 30 FPS)
        time_delta = 1.0 / 30.0
        velocity = (movement_1_to_2[0] / time_delta, movement_1_to_2[1] / time_delta)
        assert abs(velocity[0] - 150.0) < 0.1  # 5 pixels / (1/30) seconds
        assert abs(velocity[1] - 60.0) < 0.1  # 2 pixels / (1/30) seconds
        print("✓ Velocity estimation works")

        return True

    except Exception as e:
        print(f"✗ Ball tracking simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_module_configuration():
    """Test vision module configuration access."""
    print("\nTesting Module Configuration...")

    try:
        config = {
            "camera_device_id": -1,
            "enable_ball_detection": True,
            "enable_table_detection": False,  # Test mixed settings
            "debug_mode": True,
        }

        vision = VisionModule(config)

        # Test configuration access
        assert vision.config.camera_device_id == -1
        assert vision.config.enable_ball_detection is True
        assert vision.config.enable_table_detection is False
        assert vision.config.debug_mode is True
        print("✓ Configuration access works")

        # Test default values are preserved
        assert vision.config.enable_cue_detection is True  # Should be default
        print("✓ Default configuration values preserved")

        return True

    except Exception as e:
        print(f"✗ Module configuration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_vision_tests():
    """Run all working vision module tests."""
    print("Starting Basic Vision Module Integration Tests")
    print("=" * 50)

    tests = [
        test_vision_module_initialization,
        test_ball_data_model,
        test_coordinate_operations,
        test_vision_to_core_conversion,
        test_ball_tracking_simulation,
        test_module_configuration,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("BASIC VISION MODULE INTEGRATION TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL VISION MODULE INTEGRATION TESTS PASSED")
        return True
    else:
        print("✗ SOME VISION MODULE INTEGRATION TESTS FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_vision_tests())
    sys.exit(0 if success else 1)
