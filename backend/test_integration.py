#!/usr/bin/env python3
"""Integration tests for module interactions."""

import time

import psutil

from .core.models import BallState, GameState, TableState, Vector2D
from .core.physics.engine import PhysicsEngine
from .vision.models import Ball, BallType, DetectionResult, FrameStatistics


def test_core_physics_integration():
    """Test core module and physics engine integration."""
    print("Testing Core-Physics Integration...")

    # Create game state
    table = TableState.standard_9ft_table()
    balls = [
        BallState(
            id="cue",
            position=Vector2D(1.0, 1.0),
            velocity=Vector2D(2.0, 0.0),
            is_cue_ball=True,
        ),
        BallState(id="1", position=Vector2D(2.5, 1.0), number=1),  # Target ball
        BallState(id="2", position=Vector2D(2.0, 0.5), number=2),  # Another ball
    ]

    game_state = GameState(
        timestamp=time.time(), frame_number=0, balls=balls, table=table
    )

    # Test physics simulation
    physics_engine = PhysicsEngine()
    cue_ball = game_state.get_cue_ball()
    other_balls = game_state.get_numbered_balls()

    start_time = time.perf_counter()
    trajectory = physics_engine.calculate_trajectory(
        cue_ball, table, other_balls, time_limit=1.0
    )
    physics_time = time.perf_counter() - start_time

    print(f"  ✓ Physics simulation completed in {physics_time*1000:.2f}ms")
    print(f"  ✓ Trajectory has {len(trajectory)} points")

    if trajectory:
        final_pos = trajectory[-1].position
        print(f"  ✓ Cue ball final position: ({final_pos.x:.3f}, {final_pos.y:.3f})")

    return physics_time


def test_vision_core_integration():
    """Test vision module to core module data conversion."""
    print("Testing Vision-Core Integration...")

    # Create vision detection result
    vision_balls = [
        Ball(position=(960, 540), radius=20, ball_type=BallType.CUE, confidence=0.95),
        Ball(
            position=(800, 400),
            radius=20,
            ball_type=BallType.SOLID,
            number=1,
            confidence=0.90,
        ),
        Ball(
            position=(1100, 600),
            radius=20,
            ball_type=BallType.EIGHT,
            number=8,
            confidence=0.85,
        ),
    ]

    stats = FrameStatistics(
        frame_number=1,
        timestamp=time.time(),
        processing_time=15.0,
        balls_detected=len(vision_balls),
    )

    detection = DetectionResult(
        frame_number=1,
        timestamp=time.time(),
        balls=vision_balls,
        cue=None,
        table=None,
        statistics=stats,
    )

    # Convert to core models (simulate conversion logic)
    start_time = time.perf_counter()

    table = TableState.standard_9ft_table()
    core_balls = []

    for vision_ball in vision_balls:
        # Convert pixel coordinates to table coordinates (simplified)
        table_x = (vision_ball.position[0] / 1920.0) * table.width
        table_y = (vision_ball.position[1] / 1080.0) * table.height

        core_ball = BallState(
            id=(
                "cue"
                if vision_ball.ball_type == BallType.CUE
                else f"ball_{vision_ball.number or 'unknown'}"
            ),
            position=Vector2D(table_x, table_y),
            radius=vision_ball.radius / 1000.0,  # Convert to meters
            is_cue_ball=(vision_ball.ball_type == BallType.CUE),
            number=vision_ball.number,
            confidence=vision_ball.confidence,
        )
        core_balls.append(core_ball)

    game_state = GameState(
        timestamp=detection.timestamp,
        frame_number=detection.frame_number,
        balls=core_balls,
        table=table,
    )

    conversion_time = time.perf_counter() - start_time

    print(f"  ✓ Vision-to-Core conversion completed in {conversion_time*1000:.2f}ms")
    print(f"  ✓ Converted {len(core_balls)} balls")
    print(f"  ✓ Cue ball found: {game_state.get_cue_ball() is not None}")

    # Validate conversion
    errors = game_state.validate_consistency()
    if errors:
        print(f"  ⚠ Validation errors: {errors}")
    else:
        print("  ✓ Game state validation passed")

    return conversion_time


def test_memory_usage():
    """Test memory usage during operations."""
    print("Testing Memory Usage...")

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create large game state
    table = TableState.standard_9ft_table()
    balls = []

    # Create many balls to test memory usage
    for i in range(100):
        ball = BallState(
            id=f"ball_{i}",
            position=Vector2D(i * 0.1, i * 0.05),
            velocity=Vector2D(i * 0.01, i * 0.01),
            number=i % 15 + 1 if i > 0 else None,
            is_cue_ball=(i == 0),
        )
        balls.append(ball)

    GameState(timestamp=time.time(), frame_number=0, balls=balls, table=table)

    # Run physics simulation
    physics_engine = PhysicsEngine()
    for i in range(10):  # Multiple iterations
        cue_ball = balls[0]  # First ball is cue
        other_balls = balls[1:11]  # Use subset to avoid too long simulation
        physics_engine.calculate_trajectory(
            cue_ball, table, other_balls, time_limit=0.1
        )

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print(f"  ✓ Initial memory: {initial_memory:.2f} MB")
    print(f"  ✓ Final memory: {final_memory:.2f} MB")
    print(f"  ✓ Memory increase: {memory_increase:.2f} MB")

    if memory_increase > 100:  # More than 100MB increase might indicate a leak
        print(f"  ⚠ High memory usage detected: {memory_increase:.2f} MB")
    else:
        print("  ✓ Memory usage within acceptable limits")

    return memory_increase


def test_fps_performance():
    """Test frame processing performance to validate FPS capabilities."""
    print("Testing FPS Performance...")

    table = TableState.standard_9ft_table()

    # Simulate frame processing loop
    frame_count = 100
    start_time = time.perf_counter()

    for frame_num in range(frame_count):
        # Simulate vision detection (create detection data)
        balls = [
            BallState(
                id="cue",
                position=Vector2D(1.0 + frame_num * 0.01, 1.0),
                velocity=Vector2D(1.0, 0.0),
                is_cue_ball=True,
            ),
            BallState(id="1", position=Vector2D(2.0, 1.0), number=1),
        ]

        # Create game state
        game_state = GameState(
            timestamp=time.time(), frame_number=frame_num, balls=balls, table=table
        )

        # Validate game state
        game_state.validate_consistency()

        # Simple physics calculation
        physics_engine = PhysicsEngine()
        if (
            frame_num % 10 == 0
        ):  # Only do physics every 10th frame to simulate real usage
            cue_ball = game_state.get_cue_ball()
            if cue_ball:
                physics_engine.calculate_trajectory(
                    cue_ball, table, game_state.get_numbered_balls(), time_limit=0.1
                )

    total_time = time.perf_counter() - start_time
    fps = frame_count / total_time

    print(f"  ✓ Processed {frame_count} frames in {total_time:.3f}s")
    print(f"  ✓ Average FPS: {fps:.2f}")
    print(f"  ✓ Frame time: {1000/fps:.2f}ms")

    target_fps = 30
    if fps >= target_fps:
        print(f"  ✓ Performance meets target of {target_fps} FPS")
    else:
        print(f"  ⚠ Performance below target: {fps:.2f} < {target_fps} FPS")

    return fps


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("BILLIARDS TRAINER BACKEND - INTEGRATION TEST SUITE")
    print("=" * 60)

    results = {}

    try:
        results["physics_time"] = test_core_physics_integration()
        print()

        results["conversion_time"] = test_vision_core_integration()
        print()

        results["memory_increase"] = test_memory_usage()
        print()

        results["fps"] = test_fps_performance()
        print()

        print("=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Physics simulation time: {results['physics_time']*1000:.2f}ms")
        print(f"Vision-Core conversion time: {results['conversion_time']*1000:.2f}ms")
        print(f"Memory increase: {results['memory_increase']:.2f} MB")
        print(f"Frame processing FPS: {results['fps']:.2f}")

        # Overall assessment
        all_good = (
            results["physics_time"] < 0.1
            and results["conversion_time"] < 0.01  # Physics under 100ms
            and results["memory_increase"] < 50  # Conversion under 10ms
            and results["fps"] >= 30  # Memory under 50MB  # At least 30 FPS
        )

        if all_good:
            print("\n🎉 All integration tests PASSED!")
        else:
            print("\n⚠ Some performance targets not met, but system functional")

        return all_good

    except Exception as e:
        print(f"\n❌ Integration tests FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
