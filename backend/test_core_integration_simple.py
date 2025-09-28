#!/usr/bin/env python3
"""
Simplified Core Module Integration Test.

Tests the core module working independently:
- Module initialization
- Component interaction
- Basic functionality without complex state creation
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core import CoreModule, CoreModuleConfig, GameType
from core.models import Vector2D

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_core_module_initialization():
    """Test core module initialization and component setup."""
    print("Testing Core Module Initialization...")

    try:
        # Test default initialization
        core = CoreModule()
        assert core is not None
        assert core.config is not None
        assert core.state_manager is not None
        assert core.physics_engine is not None
        assert core.trajectory_calculator is not None
        assert core.collision_detector is not None
        assert core.shot_analyzer is not None
        print("✓ Core module components initialized successfully")

        # Test configuration
        config = CoreModuleConfig(
            physics_enabled=True,
            prediction_enabled=True,
            cache_size=500,
            max_trajectory_time=5.0,
            debug_mode=True,
        )
        core_custom = CoreModule(config)
        assert core_custom.config.cache_size == 500
        print("✓ Custom configuration works correctly")

        # Test performance metrics
        metrics = core.get_performance_metrics()
        assert metrics is not None
        assert metrics.total_updates == 0
        print("✓ Performance metrics initialized")

        return True

    except Exception as e:
        print(f"✗ Core module initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_vector_math_operations():
    """Test Vector2D mathematical operations."""
    print("\nTesting Vector2D Math Operations...")

    try:
        # Basic vector operations
        v1 = Vector2D(3.0, 4.0)
        v2 = Vector2D(1.0, 2.0)

        # Test magnitude
        assert abs(v1.magnitude() - 5.0) < 0.001
        print("✓ Vector magnitude calculation works")

        # Test normalization
        normalized = v1.normalize()
        assert abs(normalized.magnitude() - 1.0) < 0.001
        print("✓ Vector normalization works")

        # Test dot product
        dot = v1.dot(v2)
        assert abs(dot - 11.0) < 0.001  # 3*1 + 4*2 = 11
        print("✓ Vector dot product works")

        # Test addition/subtraction
        sum_v = v1 + v2
        assert sum_v.x == 4.0 and sum_v.y == 6.0
        diff_v = v1 - v2
        assert diff_v.x == 2.0 and diff_v.y == 2.0
        print("✓ Vector arithmetic operations work")

        # Test distance
        distance = v1.distance_to(v2)
        expected = ((3 - 1) ** 2 + (4 - 2) ** 2) ** 0.5  # sqrt(4 + 4) = 2.828...
        assert abs(distance - expected) < 0.001
        print("✓ Vector distance calculation works")

        return True

    except Exception as e:
        print(f"✗ Vector math operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_event_system():
    """Test event subscription and emission."""
    print("\nTesting Event System...")

    try:
        core = CoreModule()
        events_received = []

        def event_handler(event_type, event_data):
            events_received.append((event_type, event_data))

        # Test subscription
        subscription_id = core.subscribe_to_events("test_event", event_handler)
        assert subscription_id is not None
        print("✓ Event subscription successful")

        # Test manual event emission (through event manager)
        core.event_manager.emit_event("test_event", {"test": "data"})

        # Give time for event processing
        await asyncio.sleep(0.1)

        # Check events were received
        assert len(events_received) > 0
        print("✓ Event emission and reception works")

        # Test unsubscription
        success = core.unsubscribe(subscription_id)
        assert success
        print("✓ Event unsubscription works")

        return True

    except Exception as e:
        print(f"✗ Event system failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_caching_system():
    """Test the caching functionality."""
    print("\nTesting Caching System...")

    try:
        core = CoreModule()

        # Test trajectory cache
        cache_key = "test_trajectory"
        test_data = [Vector2D(1.0, 1.0), Vector2D(2.0, 2.0)]

        # Set cache
        core.trajectory_cache.set(cache_key, test_data)
        print("✓ Cache set operation works")

        # Get from cache
        cached_data = core.trajectory_cache.get(cache_key)
        assert cached_data is not None
        assert len(cached_data) == 2
        print("✓ Cache get operation works")

        # Test cache miss
        missing_data = core.trajectory_cache.get("non_existent_key")
        assert missing_data is None
        print("✓ Cache miss handling works")

        # Test cache clear
        core.trajectory_cache.clear()
        cleared_data = core.trajectory_cache.get(cache_key)
        assert cleared_data is None
        print("✓ Cache clear operation works")

        return True

    except Exception as e:
        print(f"✗ Caching system failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_physics_components():
    """Test physics components independently."""
    print("\nTesting Physics Components...")

    try:
        core = CoreModule()

        # Test physics engine initialization
        assert core.physics_engine is not None
        print("✓ Physics engine initialized")

        # Test trajectory calculator
        assert core.trajectory_calculator is not None
        print("✓ Trajectory calculator initialized")

        # Test collision detector
        assert core.collision_detector is not None
        print("✓ Collision detector initialized")

        # Test collision resolver
        assert core.collision_resolver is not None
        print("✓ Collision resolver initialized")

        return True

    except Exception as e:
        print(f"✗ Physics components failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_analysis_components():
    """Test analysis components independently."""
    print("\nTesting Analysis Components...")

    try:
        core = CoreModule()

        # Test shot analyzer
        assert core.shot_analyzer is not None
        print("✓ Shot analyzer initialized")

        # Test assistance engine
        assert core.assistance_engine is not None
        print("✓ Assistance engine initialized")

        # Test outcome predictor
        assert core.outcome_predictor is not None
        print("✓ Outcome predictor initialized")

        return True

    except Exception as e:
        print(f"✗ Analysis components failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_utility_components():
    """Test utility components."""
    print("\nTesting Utility Components...")

    try:
        core = CoreModule()

        # Test geometry utils
        assert core.geometry_utils is not None
        print("✓ Geometry utils initialized")

        # Test math utils
        assert core.math_utils is not None
        print("✓ Math utils initialized")

        # Test state manager
        assert core.state_manager is not None
        print("✓ State manager initialized")

        return True

    except Exception as e:
        print(f"✗ Utility components failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_state_management_basic():
    """Test basic state management without complex objects."""
    print("\nTesting Basic State Management...")

    try:
        core = CoreModule()

        # Test initial state
        current_state = core.get_current_state()
        assert current_state is None
        print("✓ Initial state is None as expected")

        # Test state history
        history = core.get_state_history()
        assert isinstance(history, list)
        assert len(history) == 0
        print("✓ Initial history is empty")

        return True

    except Exception as e:
        print(f"✗ Basic state management failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_game_reset():
    """Test game reset functionality."""
    print("\nTesting Game Reset...")

    try:
        core = CoreModule()

        # Reset game to practice mode
        await core.reset_game(GameType.PRACTICE)
        print("✓ Game reset to practice mode successful")

        # Reset to different game type
        await core.reset_game(GameType.EIGHT_BALL)
        print("✓ Game reset to 8-ball mode successful")

        return True

    except Exception as e:
        print(f"✗ Game reset failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_core_tests():
    """Run all simplified core module tests."""
    print("Starting Simplified Core Module Integration Tests")
    print("=" * 55)

    tests = [
        test_core_module_initialization,
        test_vector_math_operations,
        test_event_system,
        test_caching_system,
        test_physics_components,
        test_analysis_components,
        test_utility_components,
        test_state_management_basic,
        test_game_reset,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    # Summary
    print("\n" + "=" * 55)
    print("SIMPLIFIED CORE MODULE INTEGRATION TEST SUMMARY")
    print("=" * 55)

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
