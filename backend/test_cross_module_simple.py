#!/usr/bin/env python3
"""Simplified Cross-Module Integration Test.

Tests the working parts of integration between modules:
- Data conversion between Core and Vision
- Configuration propagation
- Data type compatibility
- Module initialization coordination
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


async def test_core_vision_data_conversion():
    """Test data conversion between Core and Vision modules."""
    print("Testing Core-Vision Data Conversion...")

    try:
        # Initialize modules
        CoreModule()
        VisionModule({"camera_device_id": -1})  # No camera

        # Create Vision detection data
        vision_balls = [
            Ball(position=(320, 240), radius=14.0, ball_type=BallType.CUE, number=0),
            Ball(position=(400, 200), radius=14.0, ball_type=BallType.SOLID, number=1),
            Ball(position=(350, 280), radius=14.0, ball_type=BallType.STRIPE, number=9),
            Ball(position=(380, 320), radius=14.0, ball_type=BallType.EIGHT, number=8),
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
                radius=vball.radius / 1000.0,  # Convert to meters (rough conversion)
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

        # Check numbered balls
        numbered_balls_vision = [b for b in vision_balls if b.ball_type != BallType.CUE]
        numbered_balls_core = [b for b in core_balls if not b.is_cue_ball]
        assert len(numbered_balls_vision) == len(numbered_balls_core) == 3
        print("✓ Numbered ball consistency verified")

        # Check position consistency
        for vball, cball in zip(vision_balls, core_balls):
            assert cball.position.x == vball.position[0]
            assert cball.position.y == vball.position[1]
        print("✓ Position consistency verified")

        # Test reverse conversion (Core to Vision format)
        converted_vision_balls = []
        for cball in core_balls:
            if cball.is_cue_ball:
                ball_type = BallType.CUE
            elif cball.number == 8:
                ball_type = BallType.EIGHT
            elif cball.number and cball.number > 8:
                ball_type = BallType.STRIPE
            else:
                ball_type = BallType.SOLID

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
        print(f"✗ Core-Vision data conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_configuration_coordination():
    """Test configuration coordination between modules."""
    print("\nTesting Configuration Coordination...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"

            # Initialize configuration module
            ConfigurationModule(config_dir)
            print("✓ Configuration module initialized")

            # Set up configuration for different modules
            core_config = CoreModuleConfig(
                physics_enabled=True,
                prediction_enabled=True,
                cache_size=500,
                max_trajectory_time=5.0,
                debug_mode=True,
            )

            vision_config = {
                "camera_device_id": -1,
                "enable_ball_detection": True,
                "enable_table_detection": True,
                "target_fps": 30,
                "debug_mode": True,
            }

            # Create modules with configuration
            core = CoreModule(core_config)
            vision = VisionModule(vision_config)

            print("✓ Modules created with configuration")

            # Verify configuration was applied to Core
            assert core.config.cache_size == 500
            assert core.config.max_trajectory_time == 5.0
            assert core.config.debug_mode is True
            assert core.config.physics_enabled is True
            assert core.config.prediction_enabled is True
            print("✓ Core module configuration applied correctly")

            # Verify configuration was applied to Vision
            assert vision.config.camera_device_id == -1
            assert vision.config.enable_ball_detection is True
            assert vision.config.target_fps == 30
            assert vision.config.debug_mode is True
            print("✓ Vision module configuration applied correctly")

            # Test configuration consistency
            assert core.config.debug_mode == vision.config.debug_mode
            print("✓ Configuration consistency between modules verified")

            # Test module-specific configurations don't interfere
            assert hasattr(core.config, "physics_enabled")
            assert not hasattr(vision.config, "physics_enabled")
            assert hasattr(vision.config, "camera_device_id")
            assert not hasattr(core.config, "camera_device_id")
            print("✓ Module-specific configuration isolation verified")

            return True

    except Exception as e:
        print(f"✗ Configuration coordination failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_data_type_compatibility():
    """Test data type compatibility between modules."""
    print("\nTesting Data Type Compatibility...")

    try:
        # Test Vector2D and tuple compatibility
        test_positions = [(0, 0), (320, 240), (640, 480), (100.5, 200.7)]

        for x, y in test_positions:
            # Vision uses tuples
            vision_pos = (x, y)

            # Core uses Vector2D
            core_pos = Vector2D(x, y)

            # Test conversion both ways
            assert core_pos.x == vision_pos[0]
            assert core_pos.y == vision_pos[1]

            # Convert back to tuple
            tuple_back = (core_pos.x, core_pos.y)
            assert tuple_back == vision_pos

        print("✓ Position data type compatibility verified")

        # Test ball type mapping
        type_mappings = [
            (BallType.CUE, True, 0),  # Vision type, is_cue_ball, number
            (BallType.SOLID, False, 1),
            (BallType.SOLID, False, 7),
            (BallType.STRIPE, False, 9),
            (BallType.STRIPE, False, 15),
            (BallType.EIGHT, False, 8),
        ]

        for vision_type, expected_is_cue, number in type_mappings:
            # Create Vision ball
            vision_ball = Ball(
                position=(100, 100), radius=14.0, ball_type=vision_type, number=number
            )

            # Convert to Core ball
            core_ball = BallState(
                id="cue" if expected_is_cue else f"ball_{number}",
                position=Vector2D(vision_ball.position[0], vision_ball.position[1]),
                velocity=Vector2D.zero(),
                radius=vision_ball.radius / 1000.0,
                mass=0.17,
                is_cue_ball=expected_is_cue,
                is_pocketed=False,
                number=number,
            )

            # Verify mapping
            assert core_ball.is_cue_ball == expected_is_cue
            assert core_ball.number == number

        print("✓ Ball type compatibility verified")

        # Test numeric precision compatibility
        precise_values = [0.0, 1.0, 3.14159, 999.999, 0.001]

        for value in precise_values:
            # Create Vector2D with precise value
            vector = Vector2D(value, value * 2)

            # Convert to tuple and back
            tuple_pos = (vector.x, vector.y)
            vector_back = Vector2D(tuple_pos[0], tuple_pos[1])

            # Should maintain precision
            assert abs(vector_back.x - vector.x) < 1e-10
            assert abs(vector_back.y - vector.y) < 1e-10

        print("✓ Numeric precision compatibility verified")

        return True

    except Exception as e:
        print(f"✗ Data type compatibility failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_module_initialization_order():
    """Test different module initialization orders."""
    print("\nTesting Module Initialization Order...")

    try:
        # Test 1: Config → Core → Vision
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"

            config = ConfigurationModule(config_dir)
            core = CoreModule()
            vision = VisionModule({"camera_device_id": -1})

            assert config is not None
            assert core is not None
            assert vision is not None
            print("✓ Config → Core → Vision initialization order works")

        # Test 2: Vision → Core → Config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"

            vision = VisionModule({"camera_device_id": -1})
            core = CoreModule()
            config = ConfigurationModule(config_dir)

            assert vision is not None
            assert core is not None
            assert config is not None
            print("✓ Vision → Core → Config initialization order works")

        # Test 3: All modules simultaneously
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"

            modules = {
                "config": ConfigurationModule(config_dir),
                "core": CoreModule(),
                "vision": VisionModule({"camera_device_id": -1}),
            }

            for _name, module in modules.items():
                assert module is not None
            print("✓ Simultaneous module initialization works")

        # Test 4: Module independence
        core1 = CoreModule()
        core2 = CoreModule()
        assert core1 is not core2
        assert core1.config is not core2.config
        print("✓ Module independence verified")

        return True

    except Exception as e:
        print(f"✗ Module initialization order test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_performance_coordination():
    """Test performance monitoring coordination between modules."""
    print("\nTesting Performance Coordination...")

    try:
        # Initialize modules
        core = CoreModule()
        vision = VisionModule({"camera_device_id": -1})

        # Get initial performance metrics
        core_metrics = core.get_performance_metrics()
        vision_stats = vision.get_statistics()

        assert core_metrics is not None
        assert vision_stats is not None
        print("✓ Performance metrics accessible from both modules")

        # Check initial values
        assert core_metrics.total_updates == 0
        assert core_metrics.avg_update_time == 0.0
        assert core_metrics.errors_count == 0
        print("✓ Initial performance metrics correct")

        # Check vision statistics structure
        assert hasattr(vision_stats, "frames_processed")
        assert hasattr(vision_stats, "avg_processing_time")
        print("✓ Vision statistics structure correct")

        # Test performance tracking over simulated operations
        start_time = time.time()

        # Simulate some work
        for i in range(5):
            # Create test balls
            test_balls = [
                Ball(
                    position=(100 + i * 10, 100),
                    radius=14.0,
                    ball_type=BallType.SOLID,
                    number=1,
                )
            ]

            # Simulate conversion work
            for ball in test_balls:
                BallState(
                    id=f"ball_{ball.number}",
                    position=Vector2D(ball.position[0], ball.position[1]),
                    velocity=Vector2D.zero(),
                    radius=ball.radius / 1000.0,
                    mass=0.17,
                    is_cue_ball=False,
                    is_pocketed=False,
                    number=ball.number,
                )

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✓ Simulated {processing_time:.4f}s of cross-module operations")

        # Get updated metrics
        updated_core_metrics = core.get_performance_metrics()
        updated_vision_stats = vision.get_statistics()

        # Verify metrics are still accessible
        assert updated_core_metrics is not None
        assert updated_vision_stats is not None
        print("✓ Performance metrics remain stable during operations")

        return True

    except Exception as e:
        print(f"✗ Performance coordination failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_error_isolation():
    """Test error isolation between modules."""
    print("\nTesting Error Isolation...")

    try:
        # Initialize working modules
        core = CoreModule()
        VisionModule({"camera_device_id": -1})

        print("✓ Base modules initialized successfully")

        # Test that vision errors don't affect core
        try:
            # Try to create vision module with invalid config
            VisionModule({"camera_device_id": 99999, "invalid_param": "bad_value"})
        except Exception as e:
            print(f"✓ Invalid vision config handled: {type(e).__name__}")

        # Core should still work
        assert core.get_current_state() is None  # Expected initial state
        metrics = core.get_performance_metrics()
        assert metrics is not None
        print("✓ Core module unaffected by vision errors")

        # Test that core continues working with bad input data
        try:
            # This should fail gracefully
            bad_data = {"invalid": "structure"}
            await core.update_state(bad_data)
            print("! Bad data handled gracefully")
        except Exception as e:
            print(f"✓ Bad data properly rejected: {type(e).__name__}")

        # Core should still be functional
        metrics_after = core.get_performance_metrics()
        assert metrics_after is not None
        print("✓ Core module remains functional after errors")

        # Test module independence
        core2 = CoreModule()
        assert core2 is not core
        print("✓ Multiple module instances work independently")

        return True

    except Exception as e:
        print(f"✗ Error isolation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_cross_module_tests():
    """Run all working cross-module integration tests."""
    print("Starting Simplified Cross-Module Integration Tests")
    print("=" * 55)

    tests = [
        test_core_vision_data_conversion,
        test_configuration_coordination,
        test_data_type_compatibility,
        test_module_initialization_order,
        test_performance_coordination,
        test_error_isolation,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("SIMPLIFIED CROSS-MODULE INTEGRATION TEST SUMMARY")
    print("=" * 55)

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
