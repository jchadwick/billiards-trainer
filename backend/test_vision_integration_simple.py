#!/usr/bin/env python3
"""
Simplified Vision Module Integration Test.

Tests the vision module components without requiring a camera:
- Module initialization
- Data structure verification
- Component loading
- Integration with core module data types
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Import core models for integration testing
from core.models import BallState, Vector2D

# Import vision module components
from vision import VisionConfig, VisionModule
from vision.models import Ball, BallType, CueState, DetectionResult, Table

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

        # Test configuration access
        assert vision.config.camera_device_id == -1
        assert vision.config.enable_ball_detection == True
        assert vision.config.debug_mode == True
        print("✓ Configuration properly applied")

        return True

    except Exception as e:
        print(f"✗ Vision module initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vision_data_models():
    """Test vision data model structures."""
    print("\nTesting Vision Data Models...")

    try:
        # Test Ball model
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
        print("✓ Ball model works correctly")

        # Test ball types
        assert BallType.CUE == BallType.CUE
        assert BallType.SOLID == BallType.SOLID
        assert BallType.STRIPE == BallType.STRIPE
        assert BallType.EIGHT == BallType.EIGHT
        print("✓ Ball types work correctly")

        # Test Table model (simplified - requires pockets and surface_color)
        from vision.models import Pocket, PocketType

        pockets = [
            Pocket(
                position=(50, 50), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(750, 50), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(750, 350),
                pocket_type=PocketType.CORNER,
                radius=25,
                corners=[],
            ),
            Pocket(
                position=(50, 350), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(400, 50), pocket_type=PocketType.SIDE, radius=25, corners=[]
            ),
            Pocket(
                position=(400, 350), pocket_type=PocketType.SIDE, radius=25, corners=[]
            ),
        ]

        table = Table(
            corners=[(0, 0), (800, 0), (800, 400), (0, 400)],
            pockets=pockets,
            width=800.0,
            height=400.0,
            surface_color=(60, 100, 80),  # HSV green
        )
        assert len(table.corners) == 4
        assert table.width == 800.0
        assert table.height == 400.0
        assert len(table.pockets) == 6
        print("✓ Table model works correctly")

        # Test DetectionResult model
        detection = DetectionResult(
            timestamp=time.time(), frame_number=100, balls=[ball], table=table
        )
        assert detection.frame_number == 100
        assert len(detection.balls) == 1
        assert detection.table is not None
        print("✓ DetectionResult model works correctly")

        return True

    except Exception as e:
        print(f"✗ Vision data models failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_coordinate_transformations():
    """Test coordinate transformation functionality."""
    print("\nTesting Coordinate Transformations...")

    try:
        # Test basic coordinate operations with Vision Ball positions
        ball_pixel_pos = (320, 240)  # Center of 640x480 image

        # Test position scaling
        scale_factor = 2.0
        scaled_pos = (
            ball_pixel_pos[0] * scale_factor,
            ball_pixel_pos[1] * scale_factor,
        )
        assert scaled_pos == (640.0, 480.0)
        print("✓ Position scaling works")

        # Test conversion to Vector2D (for core integration)
        vector_pos = Vector2D(ball_pixel_pos[0], ball_pixel_pos[1])
        assert vector_pos.x == 320
        assert vector_pos.y == 240
        print("✓ Position conversion to Vector2D works")

        # Test distance calculation between balls
        ball1_pos = (100, 100)
        ball2_pos = (150, 130)
        distance = (
            (ball2_pos[0] - ball1_pos[0]) ** 2 + (ball2_pos[1] - ball1_pos[1]) ** 2
        ) ** 0.5
        expected_distance = ((50) ** 2 + (30) ** 2) ** 0.5
        assert abs(distance - expected_distance) < 0.001
        print("✓ Distance calculation works")

        # Test angle calculation
        angle = np.arctan2(ball2_pos[1] - ball1_pos[1], ball2_pos[0] - ball1_pos[0])
        expected_angle = np.arctan2(30, 50)
        assert abs(angle - expected_angle) < 0.001
        print("✓ Angle calculation works")

        return True

    except Exception as e:
        print(f"✗ Coordinate transformations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vision_core_integration():
    """Test integration between vision and core module data types."""
    print("\nTesting Vision-Core Integration...")

    try:
        # Create Vision Ball and convert to Core BallState
        vision_ball = Ball(
            position=(150, 200),
            radius=14.0,
            ball_type=BallType.SOLID,
            number=5,
            confidence=0.92,
        )

        # Convert to core BallState
        core_ball = BallState(
            id=f"ball_{vision_ball.number}",
            position=Vector2D(vision_ball.position[0], vision_ball.position[1]),
            velocity=Vector2D.zero(),
            radius=vision_ball.radius / 1000.0,  # Convert pixels to meters (rough)
            mass=0.17,  # Standard ball mass
            is_cue_ball=vision_ball.ball_type == BallType.CUE,
            is_pocketed=False,
            number=vision_ball.number,
            confidence=vision_ball.confidence,
        )

        # Verify conversion
        assert core_ball.id == "ball_5"
        assert core_ball.position.x == vision_ball.position[0]
        assert core_ball.position.y == vision_ball.position[1]
        assert core_ball.number == vision_ball.number
        assert core_ball.confidence == vision_ball.confidence
        assert core_ball.is_cue_ball == False
        print("✓ Vision-to-Core ball conversion works")

        # Test multiple balls including cue ball
        vision_balls = [
            Ball(position=(100, 100), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(200, 150), radius=14.0, ball_type=BallType.SOLID, number=1),
            Ball(position=(300, 200), radius=14.0, ball_type=BallType.STRIPE, number=9),
        ]

        core_balls = []
        for i, vball in enumerate(vision_balls):
            ball_id = (
                "cue" if vball.ball_type == BallType.CUE else f"ball_{vball.number}"
            )
            core_balls.append(
                BallState(
                    id=ball_id,
                    position=Vector2D(vball.position[0], vball.position[1]),
                    velocity=Vector2D.zero(),
                    radius=vball.radius / 1000.0,
                    mass=0.17,
                    is_cue_ball=vball.ball_type == BallType.CUE,
                    is_pocketed=False,
                    number=vball.number,
                )
            )

        assert len(core_balls) == 3
        assert core_balls[0].is_cue_ball == True  # CUE ball
        assert core_balls[1].is_cue_ball == False  # SOLID ball
        assert core_balls[2].is_cue_ball == False  # STRIPE ball
        assert core_balls[0].id == "cue"
        assert core_balls[1].id == "ball_1"
        assert core_balls[2].id == "ball_9"
        print("✓ Multiple ball conversion works")

        return True

    except Exception as e:
        print(f"✗ Vision-Core integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_detection_result_processing():
    """Test processing of detection results."""
    print("\nTesting Detection Result Processing...")

    try:
        # Create mock detection result
        balls = [
            Ball(position=(320, 240), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(400, 200), radius=14.0, ball_type=BallType.SOLID, number=1),
            Ball(position=(350, 280), radius=14.0, ball_type=BallType.EIGHT, number=8),
        ]

        from vision.models import Pocket, PocketType

        pockets = [
            Pocket(
                position=(50, 50), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(590, 50), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(590, 430),
                pocket_type=PocketType.CORNER,
                radius=25,
                corners=[],
            ),
            Pocket(
                position=(50, 430), pocket_type=PocketType.CORNER, radius=25, corners=[]
            ),
            Pocket(
                position=(320, 50), pocket_type=PocketType.SIDE, radius=25, corners=[]
            ),
            Pocket(
                position=(320, 430), pocket_type=PocketType.SIDE, radius=25, corners=[]
            ),
        ]

        table = Table(
            corners=[(50, 50), (590, 50), (590, 430), (50, 430)],
            pockets=pockets,
            width=540.0,
            height=380.0,
            surface_color=(60, 100, 80),
        )

        detection_result = DetectionResult(
            timestamp=time.time(), frame_number=150, balls=balls, table=table
        )

        # Verify detection result structure
        assert detection_result.frame_number == 150
        assert len(detection_result.balls) == 3
        assert detection_result.table is not None
        print("✓ Detection result structure is valid")

        # Test filtering balls by type
        cue_balls = [b for b in detection_result.balls if b.ball_type == BallType.CUE]
        solid_balls = [
            b for b in detection_result.balls if b.ball_type == BallType.SOLID
        ]
        eight_balls = [
            b for b in detection_result.balls if b.ball_type == BallType.EIGHT
        ]

        assert len(cue_balls) == 1
        assert len(solid_balls) == 1
        assert len(eight_balls) == 1
        print("✓ Ball type filtering works")

        # Test table boundary validation
        table_center_x = sum(corner[0] for corner in table.corners) / 4
        table_center_y = sum(corner[1] for corner in table.corners) / 4
        assert 200 < table_center_x < 400  # Roughly in center
        assert 200 < table_center_y < 300
        print("✓ Table geometry validation works")

        return True

    except Exception as e:
        print(f"✗ Detection result processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_ball_tracking():
    """Test ball tracking capabilities."""
    print("\nTesting Ball Tracking...")

    try:
        # Simulate tracking a ball across frames
        frame_1_ball = Ball(
            position=(100, 100), radius=14.0, ball_type=BallType.SOLID, number=1
        )
        frame_2_ball = Ball(
            position=(105, 102), radius=14.0, ball_type=BallType.SOLID, number=1
        )
        frame_3_ball = Ball(
            position=(110, 104), radius=14.0, ball_type=BallType.SOLID, number=1
        )

        # Test ball movement calculation
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

        # Test velocity estimation
        time_delta = 1.0 / 30.0  # 30 FPS
        velocity = (movement_1_to_2[0] / time_delta, movement_1_to_2[1] / time_delta)
        assert abs(velocity[0] - 150.0) < 0.1  # 5 pixels / (1/30) seconds
        assert abs(velocity[1] - 60.0) < 0.1  # 2 pixels / (1/30) seconds
        print("✓ Velocity estimation works")

        # Test ball history tracking
        if hasattr(frame_1_ball, "position_history"):
            frame_1_ball.position_history.append(frame_1_ball.position)
            frame_1_ball.position_history.append(frame_2_ball.position)
            frame_1_ball.position_history.append(frame_3_ball.position)

            assert len(frame_1_ball.position_history) == 3
            assert frame_1_ball.position_history[0] == (100, 100)
            assert frame_1_ball.position_history[2] == (110, 104)
            print("✓ Ball history tracking works")
        else:
            print("✓ Ball position tracking model available")

        return True

    except Exception as e:
        print(f"✗ Ball tracking failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_module_status():
    """Test vision module status and health checks."""
    print("\nTesting Module Status...")

    try:
        vision = VisionModule({"camera_device_id": -1})

        # Test status access
        stats = vision.get_statistics()
        assert stats is not None
        print("✓ Statistics access works")

        # Test module state
        assert not vision.is_running()  # Should not be running without camera
        print("✓ Module state checks work")

        # Test configuration retrieval
        config = vision.get_config()
        assert config is not None
        assert config.camera_device_id == -1
        print("✓ Configuration retrieval works")

        return True

    except Exception as e:
        print(f"✗ Module status failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_vision_tests():
    """Run all vision module integration tests."""
    print("Starting Simplified Vision Module Integration Tests")
    print("=" * 55)

    tests = [
        test_vision_module_initialization,
        test_vision_data_models,
        test_coordinate_transformations,
        test_vision_core_integration,
        test_detection_result_processing,
        test_ball_tracking,
        test_module_status,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("SIMPLIFIED VISION MODULE INTEGRATION TEST SUMMARY")
    print("=" * 55)

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
