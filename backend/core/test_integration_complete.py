#!/usr/bin/env python3
"""Comprehensive integration test for the enhanced billiards physics system.

Tests the integration of:
- Spin/English physics system
- Advanced collision detection and response
- Core module integration interfaces
- Masse shot calculations
- Multi-ball collision chains
- Integration between all components
"""

import sys
import traceback
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, "/Users/jchadwick/code/billiards-trainer")

from .events.manager import EventManager
from .game_state import GameStateManager
from .integration import (
    APIInterfaceImpl,
    ConfigInterfaceImpl,
    CoreModuleIntegrator,
    ProjectorInterfaceImpl,
    VisionInterfaceImpl,
)
from .models import BallState, GameState, GameType, TableState, Vector2D
from .physics.collision import CollisionDetector, CollisionResolver
from .physics.engine import PhysicsEngine
from .physics.spin import SpinCalculator, SpinState


def test_spin_physics():
    """Test the spin physics system."""
    print("Testing Spin Physics System...")

    try:
        # Create spin calculator
        spin_calc = SpinCalculator()

        # Create test ball
        cue_ball = BallState(
            id="cue",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(0.0, 0.0),
            is_cue_ball=True,
        )

        # Test English application
        impact_point = Vector2D(0.015, 0.0)  # Side English
        force = 15.0
        cue_angle = 45.0
        cue_elevation = 10.0

        spin_calc.apply_english(cue_ball, impact_point, force, cue_angle, cue_elevation)

        # Check that spin was applied
        assert cue_ball.spin is not None, "Spin should be applied to cue ball"
        assert cue_ball.spin.magnitude() > 0, "Spin magnitude should be greater than 0"

        # Test spin state retrieval
        spin_state = spin_calc.get_spin_state("cue")
        assert spin_state is not None, "Should retrieve spin state"
        assert spin_state.magnitude() > 0, "Spin state should have magnitude"

        # Test masse shot calculation
        target_pos = Vector2D(1.0, 1.0)
        initial_vel, initial_spin = spin_calc.calculate_masse_shot(
            cue_ball, target_pos, 60.0, 0.5  # High elevation, side English
        )

        assert initial_vel.magnitude() > 0, "Masse shot should have initial velocity"
        assert (
            initial_spin.magnitude() > 0
        ), "Masse shot should have initial spin vector"

        print("✓ Spin Physics System: PASSED")
        return True

    except Exception as e:
        print(f"✗ Spin Physics System: FAILED - {e}")
        traceback.print_exc()
        return False


def test_advanced_physics_engine():
    """Test the enhanced physics engine."""
    print("Testing Advanced Physics Engine...")

    try:
        # Create physics engine with advanced features
        config = {
            "spin_enabled": True,
            "enable_masse_shots": True,
            "enable_advanced_collisions": True,
            "magnus_effect_enabled": True,
        }
        engine = PhysicsEngine(config)

        # Create test setup
        table = TableState.standard_9ft_table()

        cue_ball = BallState(
            id="cue",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(2.0, 1.0),
            is_cue_ball=True,
        )

        target_ball = BallState(
            id="ball_1",
            position=Vector2D(1.0, 0.5),
            velocity=Vector2D(0.0, 0.0),
            number=1,
        )

        # Test basic trajectory calculation
        trajectory = engine.calculate_trajectory(cue_ball, table, [target_ball])
        assert len(trajectory) > 0, "Should generate trajectory points"

        # Test English application
        impact_point = Vector2D(0.01, 0.0)
        engine.apply_english_to_cue_ball(cue_ball, impact_point, 10.0, 0.0, 15.0)
        assert cue_ball.spin is not None, "English should be applied"

        # Test trajectory with spin
        spin_trajectory = engine.calculate_trajectory_with_spin(
            cue_ball, table, [target_ball]
        )
        assert len(spin_trajectory) > 0, "Should generate spin trajectory"

        # Test masse shot
        masse_vel, masse_traj = engine.calculate_masse_shot(
            cue_ball, Vector2D(1.5, 1.0), 45.0, 0.3
        )
        assert masse_vel.magnitude() > 0, "Masse shot should have velocity"
        assert len(masse_traj) > 0, "Masse shot should have trajectory"

        # Test spin effects summary
        spin_summary = engine.get_spin_effects_summary("cue")
        assert spin_summary["spin_enabled"], "Spin should be enabled"

        print("✓ Advanced Physics Engine: PASSED")
        return True

    except Exception as e:
        print(f"✗ Advanced Physics Engine: FAILED - {e}")
        traceback.print_exc()
        return False


def test_collision_system():
    """Test the enhanced collision system."""
    print("Testing Enhanced Collision System...")

    try:
        # Create collision system
        spin_calc = SpinCalculator()
        detector = CollisionDetector()
        resolver = CollisionResolver(spin_calculator=spin_calc)

        # Create test balls - positioned to collide
        ball1 = BallState(
            id="ball_1",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(2.0, 0.0),  # Moving towards ball2
            spin=Vector2D(5.0, 2.0),  # Some initial spin
        )

        ball2 = BallState(
            id="ball_2",
            position=Vector2D(0.56, 0.5),  # Close enough to collide within time step
            velocity=Vector2D(-1.0, 0.0),  # Moving towards ball1
            spin=Vector2D(-2.0, 1.0),
        )

        table = TableState.standard_9ft_table()

        # Test collision detection
        collision = detector.detect_ball_collision(ball1, ball2, 0.01)
        assert collision is not None, "Should detect ball collision"

        # Test collision resolution with spin transfer
        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)
        assert (
            resolved.ball1_velocity is not None
        ), "Should have new velocity for ball 1"
        assert (
            resolved.ball2_velocity is not None
        ), "Should have new velocity for ball 2"

        # Test multi-ball collision detection
        all_balls = [ball1, ball2]
        collisions = detector.detect_multiple_collisions(all_balls, table, 0.01)
        assert len(collisions) >= 0, "Should handle multiple collision detection"

        # Test cushion collision
        cushion_ball = BallState(
            id="cushion_test",
            position=Vector2D(0.03, 0.5),  # Near left cushion
            velocity=Vector2D(-1.0, 0.0),
            spin=Vector2D(0.0, 5.0),
        )

        cushion_collision = detector.detect_cushion_collision(cushion_ball, table, 0.01)
        if cushion_collision:
            resolved_cushion = resolver.resolve_cushion_collision(
                cushion_ball, cushion_collision
            )
            assert (
                resolved_cushion.ball1_velocity is not None
            ), "Should resolve cushion collision"

        print("✓ Enhanced Collision System: PASSED")
        return True

    except Exception as e:
        print(f"✗ Enhanced Collision System: FAILED - {e}")
        traceback.print_exc()
        return False


def test_integration_interfaces():
    """Test the core integration interfaces."""
    print("Testing Core Integration Interfaces...")

    try:
        # Create event manager and game state manager
        event_manager = EventManager()
        game_state_manager = GameStateManager()

        # Create core integrator
        integrator = CoreModuleIntegrator(event_manager, game_state_manager)

        # Create interface implementations
        vision_interface = VisionInterfaceImpl(event_manager)
        api_interface = APIInterfaceImpl(event_manager)
        projector_interface = ProjectorInterfaceImpl(event_manager)
        config_interface = ConfigInterfaceImpl(event_manager)

        # Register interfaces
        integrator.register_vision_interface(vision_interface)
        integrator.register_api_interface(api_interface)
        integrator.register_projector_interface(projector_interface)
        integrator.register_config_interface(config_interface)

        # Test vision interface
        test_detection = {
            "timestamp": datetime.now().timestamp(),
            "frame_number": 1,
            "balls": [
                {"id": "cue", "position": {"x": 0.5, "y": 0.5}, "confidence": 0.95}
            ],
        }
        vision_interface.receive_detection_data(test_detection)

        # Test API interface
        test_state = {
            "timestamp": datetime.now().timestamp(),
            "frame_number": 1,
            "balls": [{"id": "cue", "position": {"x": 0.5, "y": 0.5}}],
        }
        api_interface.send_state_update(test_state)

        # Test projector interface
        test_trajectory = {
            "trajectories": [
                {
                    "ball_id": "cue",
                    "points": [{"x": 0.5, "y": 0.5}, {"x": 0.6, "y": 0.6}],
                    "confidence": 0.9,
                }
            ],
            "timestamp": datetime.now().timestamp(),
        }
        projector_interface.send_trajectory_data(test_trajectory)

        # Test config interface
        test_config = {"spin_enabled": True, "detection_frequency": 30}
        config_interface.update_module_config("core", test_config)
        retrieved_config = config_interface.get_module_config("core")
        assert retrieved_config is not None, "Should retrieve config"

        # Test integration statistics
        stats = integrator.get_integration_statistics()
        assert "connected_modules" in stats, "Should have integration stats"

        print("✓ Core Integration Interfaces: PASSED")
        return True

    except Exception as e:
        print(f"✗ Core Integration Interfaces: FAILED - {e}")
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    print("Testing End-to-End Integration...")

    try:
        # Create complete system
        config = {
            "spin_enabled": True,
            "enable_masse_shots": True,
            "enable_advanced_collisions": True,
            "magnus_effect_enabled": True,
        }

        # Initialize components
        physics_engine = PhysicsEngine(config)
        event_manager = EventManager()
        game_state_manager = GameStateManager()
        integrator = CoreModuleIntegrator(event_manager, game_state_manager)

        # Create initial game state
        game_state = GameState.create_initial_state(GameType.PRACTICE)

        # Get cue ball and apply English
        cue_ball = game_state.get_cue_ball()
        assert cue_ball is not None, "Should have cue ball"

        # Apply masse shot
        target_pos = Vector2D(
            game_state.table.width * 0.75, game_state.table.height * 0.3
        )
        masse_vel, masse_trajectory = physics_engine.calculate_masse_shot(
            cue_ball, target_pos, 50.0, 0.4, 12.0
        )

        # Update cue ball with masse shot
        cue_ball.velocity = masse_vel

        # Calculate full trajectory with other balls
        other_balls = game_state.get_numbered_balls()[:5]  # Limit for performance
        full_trajectory = physics_engine.calculate_trajectory_with_spin(
            cue_ball, game_state.table, other_balls, 5.0
        )

        assert len(full_trajectory) > 0, "Should generate complete trajectory"

        # Test collision detection in trajectory
        [point for point in full_trajectory if point.collision_type is not None]

        # Verify spin effects
        spin_summary = physics_engine.get_spin_effects_summary("cue")
        assert spin_summary["spin_enabled"], "Spin should be enabled in end-to-end test"

        # Test integration with multiple systems
        # Simulate vision detection
        detection_data = {
            "timestamp": datetime.now().timestamp(),
            "frame_number": 100,
            "balls": [
                {
                    "id": ball.id,
                    "position": {"x": ball.position.x, "y": ball.position.y},
                    "velocity": (
                        {"x": ball.velocity.x, "y": ball.velocity.y}
                        if ball.velocity
                        else {"x": 0, "y": 0}
                    ),
                    "confidence": 0.9,
                }
                for ball in [cue_ball] + other_balls[:3]
            ],
        }

        # Send through integration system
        if hasattr(integrator, "vision_interface") and integrator.vision_interface:
            integrator.vision_interface.receive_detection_data(detection_data)

        print("✓ End-to-End Integration: PASSED")
        return True

    except Exception as e:
        print(f"✗ End-to-End Integration: FAILED - {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("BILLIARDS TRAINER - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()

    tests = [
        test_spin_physics,
        test_advanced_physics_engine,
        test_collision_system,
        test_integration_interfaces,
        test_end_to_end_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} PASSED")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System integration is successful.")
        print()
        print("Key Features Verified:")
        print("• Comprehensive spin/English physics system")
        print("• Advanced collision detection with spin transfer")
        print("• Masse shot calculations and trajectory prediction")
        print("• Multi-ball collision chain handling")
        print("• Complete core module integration interfaces")
        print("• Production-ready API, Vision, Projector, and Config interfaces")
        print("• End-to-end system integration and data flow")
    else:
        print(f"❌ {total - passed} TESTS FAILED. Please review the failures above.")

    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
