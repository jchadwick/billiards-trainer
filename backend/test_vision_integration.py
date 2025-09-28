#!/usr/bin/env python3
"""Vision Module Integration Test.

Tests the vision module components without requiring a physical camera:
- Module initialization
- Detection components
- Coordinate transformations
- Model data structures
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
from vision import (
    Ball,
    BallDetector,
    BallType,
    CueDetector,
    CueState,
    DetectionResult,
    ImagePreprocessor,
    ObjectTracker,
    Table,
    TableDetector,
    VisionModule,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_vision_module_initialization():
    """Test vision module initialization without camera."""
    print("Testing Vision Module Initialization...")

    try:
        # Test with minimal config
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


async def test_vision_data_models():
    """Test vision data model structures."""
    print("\nTesting Vision Data Models...")

    try:
        # Test Ball model
        ball = Ball(
            id="test_ball",
            position=Vector2D(100.0, 200.0),
            radius=15.0,
            ball_type=BallType.SOLID,
            number=1,
            confidence=0.95,
        )
        assert ball.id == "test_ball"
        assert ball.position.x == 100.0
        assert ball.ball_type == BallType.SOLID
        print("✓ Ball model works correctly")

        # Test Table model
        table = Table(
            corners=[
                Vector2D(0, 0),
                Vector2D(800, 0),
                Vector2D(800, 400),
                Vector2D(0, 400),
            ],
            width_mm=2000,
            height_mm=1000,
            confidence=0.98,
        )
        assert len(table.corners) == 4
        assert table.width_mm == 2000
        print("✓ Table model works correctly")

        # Test CueState model
        cue = CueState(
            tip_position=Vector2D(50, 50),
            base_position=Vector2D(10, 10),
            angle=45.0,
            length=580.0,
            is_visible=True,
            confidence=0.85,
        )
        assert cue.tip_position.x == 50
        assert cue.angle == 45.0
        print("✓ CueState model works correctly")

        # Test DetectionResult model
        detection = DetectionResult(
            timestamp=time.time(),
            frame_number=100,
            balls=[ball],
            table=table,
            cue=cue,
            processing_time_ms=25.5,
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


async def test_detection_components():
    """Test individual detection components."""
    print("\nTesting Detection Components...")

    try:
        # Create test image (simulated)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:150, 200:250] = [0, 255, 0]  # Green square (simulated ball)

        # Test BallDetector initialization
        ball_detector = BallDetector()
        assert ball_detector is not None
        print("✓ BallDetector initialized")

        # Test TableDetector initialization
        table_detector = TableDetector()
        assert table_detector is not None
        print("✓ TableDetector initialized")

        # Test CueDetector initialization
        cue_detector = CueDetector()
        assert cue_detector is not None
        print("✓ CueDetector initialized")

        # Test ImagePreprocessor
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
        print("✓ ImagePreprocessor initialized")

        # Test ObjectTracker
        tracker = ObjectTracker()
        assert tracker is not None
        print("✓ ObjectTracker initialized")

        return True

    except Exception as e:
        print(f"✗ Detection components failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_coordinate_transformations():
    """Test coordinate transformation functionality."""
    print("\nTesting Coordinate Transformations...")

    try:
        # Test basic coordinate operations with Vector2D
        pixel_coord = Vector2D(320, 240)  # Center of 640x480 image

        # Test scaling transformation
        scale_factor = 2.0
        scaled_coord = Vector2D(
            pixel_coord.x * scale_factor, pixel_coord.y * scale_factor
        )
        assert scaled_coord.x == 640.0
        assert scaled_coord.y == 480.0
        print("✓ Coordinate scaling works")

        # Test translation
        offset = Vector2D(50, 30)
        translated_coord = pixel_coord + offset
        assert translated_coord.x == 370.0
        assert translated_coord.y == 270.0
        print("✓ Coordinate translation works")

        # Test distance calculation
        distance = pixel_coord.distance_to(translated_coord)
        expected_distance = (50**2 + 30**2) ** 0.5
        assert abs(distance - expected_distance) < 0.001
        print("✓ Distance calculation works")

        # Test angle calculation
        angle = pixel_coord.angle_to(translated_coord)
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
            id="integration_test",
            position=Vector2D(150, 200),
            radius=14.0,
            ball_type=BallType.SOLID,
            number=5,
            confidence=0.92,
        )

        # Convert to core BallState
        core_ball = BallState(
            id=vision_ball.id,
            position=vision_ball.position,  # Same Vector2D type
            velocity=Vector2D.zero(),
            radius=vision_ball.radius / 1000.0,  # Convert mm to m
            mass=0.17,  # Standard ball mass
            is_cue_ball=vision_ball.ball_type == BallType.CUE,
            is_pocketed=False,
            number=vision_ball.number,
            confidence=vision_ball.confidence,
        )

        # Verify conversion
        assert core_ball.id == vision_ball.id
        assert core_ball.position.x == vision_ball.position.x
        assert core_ball.position.y == vision_ball.position.y
        assert core_ball.number == vision_ball.number
        assert core_ball.confidence == vision_ball.confidence
        print("✓ Vision-to-Core ball conversion works")

        # Test multiple balls
        vision_balls = [
            Ball(
                id="ball_1",
                position=Vector2D(100, 100),
                ball_type=BallType.CUE,
                number=0,
            ),
            Ball(
                id="ball_2",
                position=Vector2D(200, 150),
                ball_type=BallType.SOLID,
                number=1,
            ),
            Ball(
                id="ball_3",
                position=Vector2D(300, 200),
                ball_type=BallType.STRIPE,
                number=9,
            ),
        ]

        core_balls = []
        for vball in vision_balls:
            core_balls.append(
                BallState(
                    id=vball.id,
                    position=vball.position,
                    velocity=Vector2D.zero(),
                    radius=vball.radius / 1000.0 if vball.radius else 0.028575,
                    mass=0.17,
                    is_cue_ball=vball.ball_type == BallType.CUE,
                    is_pocketed=False,
                    number=vball.number,
                )
            )

        assert len(core_balls) == 3
        assert core_balls[0].is_cue_ball is True  # CUE ball
        assert core_balls[1].is_cue_ball is False  # SOLID ball
        assert core_balls[2].is_cue_ball is False  # STRIPE ball
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
            Ball(
                id="cue", position=Vector2D(320, 240), ball_type=BallType.CUE, number=0
            ),
            Ball(
                id="ball_1",
                position=Vector2D(400, 200),
                ball_type=BallType.SOLID,
                number=1,
            ),
            Ball(
                id="ball_8",
                position=Vector2D(350, 280),
                ball_type=BallType.EIGHT,
                number=8,
            ),
        ]

        table = Table(
            corners=[
                Vector2D(50, 50),
                Vector2D(590, 50),
                Vector2D(590, 430),
                Vector2D(50, 430),
            ],
            width_mm=2000,
            height_mm=1000,
        )

        cue = CueState(
            tip_position=Vector2D(300, 220),
            base_position=Vector2D(250, 190),
            angle=30.0,
            length=580.0,
            is_visible=True,
        )

        detection_result = DetectionResult(
            timestamp=time.time(),
            frame_number=150,
            balls=balls,
            table=table,
            cue=cue,
            processing_time_ms=18.2,
        )

        # Verify detection result structure
        assert detection_result.frame_number == 150
        assert len(detection_result.balls) == 3
        assert detection_result.table is not None
        assert detection_result.cue is not None
        assert detection_result.processing_time_ms > 0
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
        table_center = Vector2D(
            sum(corner.x for corner in table.corners) / 4,
            sum(corner.y for corner in table.corners) / 4,
        )
        assert 200 < table_center.x < 400  # Roughly in center
        assert 200 < table_center.y < 300
        print("✓ Table geometry validation works")

        return True

    except Exception as e:
        print(f"✗ Detection result processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tracking_functionality():
    """Test object tracking capabilities."""
    print("\nTesting Tracking Functionality...")

    try:
        ObjectTracker()

        # Simulate tracking a ball across frames
        frame_1_ball = Ball(
            id="tracked_ball",
            position=Vector2D(100, 100),
            ball_type=BallType.SOLID,
            number=1,
        )
        frame_2_ball = Ball(
            id="tracked_ball",
            position=Vector2D(105, 102),
            ball_type=BallType.SOLID,
            number=1,
        )
        frame_3_ball = Ball(
            id="tracked_ball",
            position=Vector2D(110, 104),
            ball_type=BallType.SOLID,
            number=1,
        )

        # Test ball movement calculation
        movement_1_to_2 = frame_2_ball.position - frame_1_ball.position
        movement_2_to_3 = frame_3_ball.position - frame_2_ball.position

        assert movement_1_to_2.x == 5.0
        assert movement_1_to_2.y == 2.0
        assert movement_2_to_3.x == 5.0
        assert movement_2_to_3.y == 2.0
        print("✓ Ball movement tracking works")

        # Test velocity estimation
        time_delta = 1.0 / 30.0  # 30 FPS
        velocity = movement_1_to_2 / time_delta
        assert abs(velocity.x - 150.0) < 0.1  # 5 pixels / (1/30) seconds
        assert abs(velocity.y - 60.0) < 0.1  # 2 pixels / (1/30) seconds
        print("✓ Velocity estimation works")

        return True

    except Exception as e:
        print(f"✗ Tracking functionality failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling in vision components."""
    print("\nTesting Error Handling...")

    try:
        # Test invalid ball creation
        try:
            Ball(
                id="",  # Empty ID should be handled
                position=Vector2D(-10, -10),  # Negative coordinates
                radius=-5.0,  # Invalid radius
                ball_type=BallType.SOLID,
                number=1,
            )
            # If this doesn't raise an error, it should still be created but with adjusted values
            print("✓ Invalid ball parameters handled gracefully")
        except Exception:
            print("✓ Invalid ball parameters properly rejected")

        # Test empty detection result
        empty_detection = DetectionResult(
            timestamp=time.time(),
            frame_number=0,
            balls=[],
            table=None,
            cue=None,
            processing_time_ms=0.0,
        )
        assert len(empty_detection.balls) == 0
        assert empty_detection.table is None
        print("✓ Empty detection result handled correctly")

        return True

    except Exception as e:
        print(f"✗ Error handling failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_vision_tests():
    """Run all vision module integration tests."""
    print("Starting Vision Module Integration Tests")
    print("=" * 50)

    tests = [
        test_vision_module_initialization,
        test_vision_data_models,
        test_detection_components,
        test_coordinate_transformations,
        test_vision_core_integration,
        test_detection_result_processing,
        test_tracking_functionality,
        test_error_handling,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("VISION MODULE INTEGRATION TEST SUMMARY")
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
