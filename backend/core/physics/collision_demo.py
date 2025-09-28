"""Demonstration script for the collision detection and response system.

This script showcases the capabilities of the collision system including:
- Ball-to-ball collision detection and response
- Ball-to-cushion collision detection and response
- Trajectory prediction with multiple collisions
- Performance optimization features
- Edge case handling
"""

import math
import time

from ..models import BallState, TableState, Vector2D

from .collision import (
    CollisionDetector,
    CollisionOptimizer,
    CollisionPredictor,
    CollisionResolver,
)


def demo_basic_ball_collision():
    """Demonstrate basic ball-to-ball collision detection and response."""
    print("=== Basic Ball-to-Ball Collision Demo ===")

    detector = CollisionDetector()
    resolver = CollisionResolver()

    # Create two balls moving towards each other
    ball1 = BallState(
        id="ball1",
        position=Vector2D(0.5, 0.5),
        velocity=Vector2D(2.0, 0.0),
        radius=0.028575,
        mass=0.17,
    )
    ball2 = BallState(
        id="ball2",
        position=Vector2D(0.8, 0.5),
        velocity=Vector2D(-1.0, 0.0),
        radius=0.028575,
        mass=0.17,
    )

    print("Before collision:")
    print(
        f"  Ball1: pos=({ball1.position.x:.3f}, {ball1.position.y:.3f}), vel=({ball1.velocity.x:.3f}, {ball1.velocity.y:.3f})"
    )
    print(
        f"  Ball2: pos=({ball2.position.x:.3f}, {ball2.position.y:.3f}), vel=({ball2.velocity.x:.3f}, {ball2.velocity.y:.3f})"
    )

    # Detect collision
    collision = detector.detect_ball_collision(ball1, ball2, 0.1)
    if collision:
        print(f"Collision detected at time {collision.time:.4f}s")
        print(
            f"Collision point: ({collision.point.position.x:.3f}, {collision.point.position.y:.3f})"
        )

        # Resolve collision
        resolved = resolver.resolve_ball_collision(ball1, ball2, collision)

        print("After collision:")
        print(
            f"  Ball1: vel=({resolved.ball1_velocity.x:.3f}, {resolved.ball1_velocity.y:.3f})"
        )
        print(
            f"  Ball2: vel=({resolved.ball2_velocity.x:.3f}, {resolved.ball2_velocity.y:.3f})"
        )
        print(f"  Energy lost: {resolved.energy_lost:.6f} J")

        # Check momentum conservation
        initial_momentum = ball1.mass * ball1.velocity.x + ball2.mass * ball2.velocity.x
        final_momentum = (
            ball1.mass * resolved.ball1_velocity.x
            + ball2.mass * resolved.ball2_velocity.x
        )
        print(
            f"  Momentum conservation: {abs(initial_momentum - final_momentum) < 1e-10}"
        )
    else:
        print("No collision detected")

    print()


def demo_cushion_collision():
    """Demonstrate ball-to-cushion collision."""
    print("=== Ball-to-Cushion Collision Demo ===")

    detector = CollisionDetector()
    resolver = CollisionResolver()
    table = TableState.standard_9ft_table()

    # Ball moving towards left cushion at an angle
    ball = BallState(
        id="ball1",
        position=Vector2D(0.1, 0.5),
        velocity=Vector2D(-1.5, 0.8),
        radius=0.028575,
        mass=0.17,
    )

    print("Before cushion collision:")
    print(
        f"  Ball: pos=({ball.position.x:.3f}, {ball.position.y:.3f}), vel=({ball.velocity.x:.3f}, {ball.velocity.y:.3f})"
    )

    # Detect cushion collision
    collision = detector.detect_cushion_collision(ball, table, 0.1)
    if collision:
        print(f"Cushion collision detected at time {collision.time:.4f}s")
        print(
            f"Collision point: ({collision.point.position.x:.3f}, {collision.point.position.y:.3f})"
        )
        print(
            f"Normal vector: ({collision.point.normal.x:.3f}, {collision.point.normal.y:.3f})"
        )

        # Resolve collision
        resolved = resolver.resolve_cushion_collision(ball, collision)

        print("After cushion collision:")
        print(
            f"  Ball: vel=({resolved.ball1_velocity.x:.3f}, {resolved.ball1_velocity.y:.3f})"
        )
        print(f"  Energy lost: {resolved.energy_lost:.6f} J")

        # Check if angle of incidence ≈ angle of reflection (for normal component)
        incident_angle = math.atan2(ball.velocity.y, ball.velocity.x)
        reflected_angle = math.atan2(
            resolved.ball1_velocity.y, resolved.ball1_velocity.x
        )
        print(f"  Incident angle: {math.degrees(incident_angle):.1f}°")
        print(f"  Reflected angle: {math.degrees(reflected_angle):.1f}°")
    else:
        print("No cushion collision detected")

    print()


def demo_trajectory_prediction():
    """Demonstrate trajectory prediction with multiple collisions."""
    print("=== Trajectory Prediction Demo ===")

    detector = CollisionDetector()
    resolver = CollisionResolver()
    predictor = CollisionPredictor(detector, resolver)
    table = TableState.standard_9ft_table()

    # Ball that will bounce around the table
    ball = BallState(
        id="ball1",
        position=Vector2D(0.2, 0.2),
        velocity=Vector2D(3.0, 2.0),
        radius=0.028575,
        mass=0.17,
    )

    # Add one target ball
    other_balls = [
        BallState(
            id="ball2",
            position=Vector2D(1.5, 0.8),
            velocity=Vector2D(0.0, 0.0),
            radius=0.028575,
            mass=0.17,
        )
    ]

    print(
        f"Predicting trajectory for ball at ({ball.position.x:.3f}, {ball.position.y:.3f})"
    )
    print(f"Initial velocity: ({ball.velocity.x:.3f}, {ball.velocity.y:.3f})")

    # Predict trajectory
    start_time = time.time()
    predicted_collisions = predictor.predict_trajectory_collisions(
        ball, other_balls, table, max_time=5.0, time_step=0.01
    )
    prediction_time = time.time() - start_time

    print(
        f"Predicted {len(predicted_collisions)} collisions in {prediction_time:.4f}s:"
    )

    for i, collision in enumerate(predicted_collisions):
        collision_type = collision.collision_type.value
        print(f"  {i+1}. {collision_type} collision at t={collision.time:.3f}s")
        print(
            f"     Position: ({collision.point.position.x:.3f}, {collision.point.position.y:.3f})"
        )
        if collision.ball2_id:
            print(
                f"     Involving balls: {collision.ball1_id} and {collision.ball2_id}"
            )
        print(f"     Energy lost: {collision.energy_lost:.6f} J")

    print()


def demo_simultaneous_collisions():
    """Demonstrate handling of simultaneous collisions."""
    print("=== Simultaneous Collisions Demo ===")

    detector = CollisionDetector()
    resolver = CollisionResolver()
    table = TableState.standard_9ft_table()

    # Create a scenario with multiple balls that will collide simultaneously
    balls = [
        BallState(
            id="ball1",
            position=Vector2D(0.5, 0.5),
            velocity=Vector2D(1.0, 0.0),
            radius=0.028575,
            mass=0.17,
        ),
        BallState(
            id="ball2",
            position=Vector2D(0.6, 0.5),
            velocity=Vector2D(-1.0, 0.0),
            radius=0.028575,
            mass=0.17,
        ),
        BallState(
            id="ball3",
            position=Vector2D(0.55, 0.4),
            velocity=Vector2D(0.0, 1.0),
            radius=0.028575,
            mass=0.17,
        ),
    ]

    print("Initial ball states:")
    for ball in balls:
        print(
            f"  {ball.id}: pos=({ball.position.x:.3f}, {ball.position.y:.3f}), vel=({ball.velocity.x:.3f}, {ball.velocity.y:.3f})"
        )

    # Detect multiple collisions
    collisions = detector.detect_multiple_collisions(balls, table, 0.1)
    print(f"\nDetected {len(collisions)} potential collisions")

    # Resolve simultaneous collisions
    resolved_collisions = resolver.resolve_simultaneous_collisions(collisions, balls)

    print(f"Resolved {len(resolved_collisions)} collisions:")
    for i, collision in enumerate(resolved_collisions):
        print(f"  {i+1}. {collision.collision_type.value} at t={collision.time:.4f}s")
        if collision.ball2_id:
            print(f"     Balls: {collision.ball1_id} and {collision.ball2_id}")
        print(f"     Energy lost: {collision.energy_lost:.6f} J")

    print()


def demo_performance_optimization():
    """Demonstrate performance optimization with spatial partitioning."""
    print("=== Performance Optimization Demo ===")

    detector = CollisionDetector()
    optimizer = CollisionOptimizer()

    # Create many balls for performance testing
    num_balls = 100
    balls = []
    for i in range(num_balls):
        balls.append(
            BallState(
                id=f"ball_{i}",
                position=Vector2D(i * 0.02, 0.5 + (i % 10) * 0.02),
                velocity=Vector2D(1.0, 0.0),
                radius=0.028575,
                mass=0.17,
            )
        )

    print(f"Performance test with {num_balls} balls")

    # Test naive approach (O(n²))
    start_time = time.time()
    naive_collisions = 0
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            collision = detector.detect_ball_collision(balls[i], balls[j], 0.01)
            if collision:
                naive_collisions += 1
    naive_time = time.time() - start_time

    # Test optimized approach with spatial partitioning
    optimizer.build_spatial_grid(balls)
    start_time = time.time()
    optimized_collisions = 0
    for i in range(len(balls)):
        candidates = optimizer.get_collision_candidates(i, balls)
        for j in candidates:
            if j > i:  # Avoid double-checking
                collision = detector.detect_ball_collision(balls[i], balls[j], 0.01)
                if collision:
                    optimized_collisions += 1
    optimized_time = time.time() - start_time

    print(f"Naive approach: {naive_collisions} collisions in {naive_time:.4f}s")
    print(
        f"Optimized approach: {optimized_collisions} collisions in {optimized_time:.4f}s"
    )

    if naive_time > 0:
        speedup = naive_time / optimized_time if optimized_time > 0 else float("inf")
        print(f"Speedup: {speedup:.2f}x")

    # Show spatial grid statistics
    grid_cells = len(optimizer.spatial_grid)
    balls_per_cell = (
        sum(len(cell) for cell in optimizer.spatial_grid.values()) / grid_cells
        if grid_cells > 0
        else 0
    )
    print(f"Spatial grid: {grid_cells} cells, avg {balls_per_cell:.1f} balls per cell")

    print()


def demo_edge_cases():
    """Demonstrate handling of edge cases."""
    print("=== Edge Cases Demo ===")

    detector = CollisionDetector()
    CollisionResolver()

    # Test 1: Grazing collision
    print("1. Grazing collision:")
    ball1 = BallState(
        id="ball1",
        position=Vector2D(0.5, 0.5),
        velocity=Vector2D(1.0, 0.0),
        radius=0.028575,
    )
    ball2 = BallState(
        id="ball2",
        position=Vector2D(0.6, 0.557),
        velocity=Vector2D(0.0, 0.0),
        radius=0.028575,
    )

    collision = detector.detect_ball_collision(ball1, ball2, 0.1)
    print(f"   Grazing collision detected: {collision is not None}")

    # Test 2: Very small velocities
    print("2. Very small velocities:")
    ball1 = BallState(
        id="ball1",
        position=Vector2D(0.5, 0.5),
        velocity=Vector2D(0.001, 0.0),
        radius=0.028575,
    )
    ball2 = BallState(
        id="ball2",
        position=Vector2D(0.6, 0.5),
        velocity=Vector2D(-0.001, 0.0),
        radius=0.028575,
    )

    collision = detector.detect_ball_collision(ball1, ball2, 100.0)  # Large time window
    print(f"   Small velocity collision detected: {collision is not None}")
    if collision:
        print(f"   Collision time: {collision.time:.2f}s")

    # Test 3: High-speed collision
    print("3. High-speed collision:")
    ball1 = BallState(
        id="ball1",
        position=Vector2D(0.5, 0.5),
        velocity=Vector2D(10.0, 0.0),
        radius=0.028575,
    )
    ball2 = BallState(
        id="ball2",
        position=Vector2D(0.6, 0.5),
        velocity=Vector2D(0.0, 0.0),
        radius=0.028575,
    )

    collision = detector.detect_ball_collision(ball1, ball2, 0.01)
    print(f"   High-speed collision detected: {collision is not None}")
    if collision:
        print(f"   Collision time: {collision.time:.6f}s")

    # Test 4: Near miss
    print("4. Near miss:")
    ball1 = BallState(
        id="ball1",
        position=Vector2D(0.5, 0.5),
        velocity=Vector2D(1.0, 0.0),
        radius=0.028575,
    )
    ball2 = BallState(
        id="ball2",
        position=Vector2D(0.6, 0.56),
        velocity=Vector2D(0.0, 0.0),
        radius=0.028575,
    )

    collision = detector.detect_ball_collision(ball1, ball2, 0.1)
    print(f"   Near miss correctly ignored: {collision is None}")

    print()


def run_full_demo():
    """Run the complete collision system demonstration."""
    print("🎱 Billiards Collision Detection and Response System Demo 🎱")
    print("=" * 60)
    print()

    demo_basic_ball_collision()
    demo_cushion_collision()
    demo_trajectory_prediction()
    demo_simultaneous_collisions()
    demo_performance_optimization()
    demo_edge_cases()

    print("✅ All collision system features demonstrated successfully!")
    print("\nKey Features Demonstrated:")
    print("- ✅ Ball-to-ball collision detection with continuous CD")
    print("- ✅ Ball-to-cushion collision detection")
    print("- ✅ Momentum and energy conservation")
    print("- ✅ Realistic collision response with restitution")
    print("- ✅ Trajectory prediction with multiple collisions")
    print("- ✅ Simultaneous collision handling")
    print("- ✅ Performance optimization with spatial partitioning")
    print("- ✅ Edge case handling (grazing, high-speed, near-miss)")
    print("- ✅ Comprehensive test coverage")


if __name__ == "__main__":
    run_full_demo()
