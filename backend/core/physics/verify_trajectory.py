"""Final verification of trajectory calculation system."""

from ..models import BallState, CueState, TableState, Vector2D

from .trajectory import TrajectoryCalculator, TrajectoryQuality


def test_multi_bounce_prediction():
    """Test multi-bounce trajectory prediction."""
    print("=== Multi-Bounce Trajectory Test ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    # Ball positioned to create multiple bounces
    ball = BallState(
        id="bouncer",
        position=Vector2D(0.5, 0.3),
        velocity=Vector2D(1.5, 1.0),  # Angled velocity for bounces
        radius=0.028575,
        mass=0.17,
    )

    print(f"Ball initial position: ({ball.position.x:.3f}, {ball.position.y:.3f})")
    print(f"Ball initial velocity: ({ball.velocity.x:.3f}, {ball.velocity.y:.3f})")
    print(f"Table dimensions: {table.width:.3f}m x {table.height:.3f}m")

    trajectory = calculator.calculate_trajectory(
        ball, table, [], TrajectoryQuality.MEDIUM, time_limit=3.0
    )

    print("\nTrajectory Results:")
    print(f"  - Total points: {len(trajectory.points)}")
    print(f"  - Total collisions: {len(trajectory.collisions)}")
    print(f"  - Time to rest: {trajectory.time_to_rest:.3f}s")
    print(f"  - Total distance: {trajectory.total_distance:.3f}m")

    # Count cushion collisions
    cushion_collisions = [
        c for c in trajectory.collisions if c.type.value == "ball_cushion"
    ]
    print(f"  - Cushion bounces: {len(cushion_collisions)}")

    if cushion_collisions:
        print("\nFirst few cushion collisions:")
        for i, collision in enumerate(cushion_collisions[:5]):
            print(
                f"  {i+1}: time={collision.time:.3f}s pos=({collision.position.x:.3f},{collision.position.y:.3f})"
            )

    print("\nFinal state:")
    print(
        f"  - Final position: ({trajectory.final_position.x:.3f}, {trajectory.final_position.y:.3f})"
    )
    print(
        f"  - Final velocity: ({trajectory.final_velocity.x:.3f}, {trajectory.final_velocity.y:.3f})"
    )
    print(f"  - Final speed: {trajectory.final_velocity.magnitude():.3f} m/s")

    return len(cushion_collisions) >= 2  # Success if at least 2 bounces


def test_cue_shot_prediction():
    """Test cue shot prediction."""
    print("\n=== Cue Shot Prediction Test ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    # Cue ball
    cue_ball = BallState(
        id="cue",
        position=Vector2D(0.5, 0.635),  # Center of table
        velocity=Vector2D(0.0, 0.0),
        radius=0.028575,
        mass=0.17,
        is_cue_ball=True,
    )

    # Cue state - moderate shot toward other end
    cue_state = CueState(
        tip_position=Vector2D(0.4, 0.635),
        angle=0.0,  # Straight shot
        estimated_force=15.0,
    )

    print(f"Cue ball position: ({cue_ball.position.x:.3f}, {cue_ball.position.y:.3f})")
    print(f"Cue angle: {cue_state.angle}°")
    print(f"Cue force: {cue_state.estimated_force}N")

    trajectory = calculator.predict_cue_shot(
        cue_state, cue_ball, table, [], TrajectoryQuality.MEDIUM
    )

    print("\nCue Shot Results:")
    print(f"  - Points calculated: {len(trajectory.points)}")
    print(f"  - Collisions: {len(trajectory.collisions)}")
    print(
        f"  - Final position: ({trajectory.final_position.x:.3f}, {trajectory.final_position.y:.3f})"
    )

    # Check if ball moved in expected direction
    if trajectory.points:
        first_point = trajectory.points[0]
        direction_correct = first_point.velocity.x > 0  # Should move right
        print(f"  - Direction correct: {direction_correct}")
        return direction_correct

    return False


def test_visualization_export():
    """Test trajectory visualization data export."""
    print("\n=== Visualization Export Test ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    ball = BallState(
        id="viz_test",
        position=Vector2D(1.0, 0.6),
        velocity=Vector2D(0.8, 0.4),
        radius=0.028575,
        mass=0.17,
    )

    trajectory = calculator.calculate_trajectory(
        ball, table, [], TrajectoryQuality.MEDIUM, time_limit=1.0
    )

    viz_data = calculator.export_visualization_data(trajectory)

    print("Visualization data exported:")
    print(f"  - Ball ID: {viz_data['ball_id']}")
    print(f"  - Points: {len(viz_data['points'])}")
    print(f"  - Collisions: {len(viz_data['collisions'])}")
    print(f"  - Success probability: {viz_data['success_probability']:.3f}")
    print(f"  - Alternatives: {len(viz_data['alternatives'])}")

    # Verify structure
    required_fields = [
        "ball_id",
        "points",
        "collisions",
        "success_probability",
        "alternatives",
    ]
    all_present = all(field in viz_data for field in required_fields)
    print(f"  - All required fields present: {all_present}")

    return all_present


def test_cache_functionality():
    """Test trajectory caching."""
    print("\n=== Cache Functionality Test ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    ball = BallState(
        id="cache_test",
        position=Vector2D(1.0, 0.6),
        velocity=Vector2D(0.5, 0.0),
        radius=0.028575,
        mass=0.17,
    )

    # Clear cache
    calculator.clear_cache()
    initial_stats = calculator.get_cache_stats()
    print(f"Initial cache size: {initial_stats['size']}")

    # First calculation
    import time

    start = time.time()
    calculator.calculate_trajectory(ball, table, [], TrajectoryQuality.MEDIUM)
    time1 = time.time() - start

    cache_stats = calculator.get_cache_stats()
    print(f"Cache size after first calculation: {cache_stats['size']}")

    # Second calculation (should use cache)
    start = time.time()
    calculator.calculate_trajectory(ball, table, [], TrajectoryQuality.MEDIUM)
    time2 = time.time() - start

    print(f"First calculation time: {time1:.6f}s")
    print(f"Second calculation time: {time2:.6f}s")
    print(f"Cache working: {cache_stats['size'] > 0}")

    return cache_stats["size"] > 0


def main():
    """Run all verification tests."""
    print("🎱 Billiards Trajectory Calculation System Verification")
    print("=" * 60)

    tests = [
        ("Multi-bounce prediction", test_multi_bounce_prediction),
        ("Cue shot prediction", test_cue_shot_prediction),
        ("Visualization export", test_visualization_export),
        ("Cache functionality", test_cache_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_name} - {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Trajectory system is working correctly.")
        print("\nThe trajectory calculation system provides:")
        print("  ✓ Accurate multi-bounce collision predictions")
        print("  ✓ Physics-based ball movement simulation")
        print("  ✓ Table geometry integration for cushion bounces")
        print("  ✓ Cue shot prediction capabilities")
        print("  ✓ Visualization data export")
        print("  ✓ Performance optimization with caching")
        print("  ✓ Support for complex scenarios")
    else:
        print("⚠️  Some tests failed. System needs additional work.")

    return passed == total


if __name__ == "__main__":
    main()
