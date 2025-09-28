"""Debug trajectory calculation system."""

import math

from ..models import BallState, TableState, Vector2D
from ..utils.cache import CacheManager

from .trajectory import TrajectoryCalculator, TrajectoryQuality


def debug_simple_trajectory():
    """Debug simple trajectory calculation."""
    print("=== Debug Simple Trajectory ===")

    # Create calculator
    cache_manager = CacheManager()
    calculator = TrajectoryCalculator(cache_manager)

    # Standard table
    table = TableState.standard_9ft_table()
    print(f"Table dimensions: {table.width}m x {table.height}m")

    # Simple ball
    ball = BallState(
        id="test",
        position=Vector2D(1.0, 1.0),  # Center of table approximately
        velocity=Vector2D(0.5, 0.0),  # Moving right slowly
        radius=0.028575,
        mass=0.17,
    )

    print(f"Initial ball position: ({ball.position.x:.3f}, {ball.position.y:.3f})")
    print(f"Initial ball velocity: ({ball.velocity.x:.3f}, {ball.velocity.y:.3f})")
    print(f"Initial speed: {ball.velocity.magnitude():.3f} m/s")

    # Calculate trajectory
    trajectory = calculator.calculate_trajectory(
        ball, table, [], TrajectoryQuality.MEDIUM, time_limit=2.0
    )

    print("\nTrajectory calculated:")
    print(f"  - Points: {len(trajectory.points)}")
    print(f"  - Collisions: {len(trajectory.collisions)}")
    print(f"  - Time to rest: {trajectory.time_to_rest:.3f}s")
    print(
        f"  - Final position: ({trajectory.final_position.x:.3f}, {trajectory.final_position.y:.3f})"
    )
    print(
        f"  - Final velocity: ({trajectory.final_velocity.x:.3f}, {trajectory.final_velocity.y:.3f})"
    )
    print(f"  - Final speed: {trajectory.final_velocity.magnitude():.3f} m/s")
    print(f"  - Total distance: {trajectory.total_distance:.3f}m")
    print(f"  - Calculation time: {trajectory.calculation_time:.3f}s")

    # Show first few trajectory points
    print("\nFirst 5 trajectory points:")
    for i, point in enumerate(trajectory.points[:5]):
        print(
            f"  {i}: t={point.time:.3f}s pos=({point.position.x:.3f},{point.position.y:.3f}) "
            f"vel=({point.velocity.x:.3f},{point.velocity.y:.3f}) speed={point.velocity.magnitude():.3f}"
        )

    if len(trajectory.points) > 10:
        print("  ...")
        # Show last few points
        for i, point in enumerate(trajectory.points[-3:], len(trajectory.points) - 3):
            print(
                f"  {i}: t={point.time:.3f}s pos=({point.position.x:.3f},{point.position.y:.3f}) "
                f"vel=({point.velocity.x:.3f},{point.velocity.y:.3f}) speed={point.velocity.magnitude():.3f}"
            )

    return trajectory


def debug_ball_collision():
    """Debug ball-to-ball collision."""
    print("\n=== Debug Ball Collision ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    # Moving ball
    ball1 = BallState(
        id="moving",
        position=Vector2D(0.5, 1.0),
        velocity=Vector2D(1.0, 0.0),  # Moving right
        radius=0.028575,
        mass=0.17,
    )

    # Stationary ball in path
    ball2 = BallState(
        id="stationary",
        position=Vector2D(1.0, 1.0),  # Directly in path
        velocity=Vector2D(0.0, 0.0),
        radius=0.028575,
        mass=0.17,
    )

    print(
        f"Ball 1: pos=({ball1.position.x:.3f},{ball1.position.y:.3f}) vel=({ball1.velocity.x:.3f},{ball1.velocity.y:.3f})"
    )
    print(
        f"Ball 2: pos=({ball2.position.x:.3f},{ball2.position.y:.3f}) vel=({ball2.velocity.x:.3f},{ball2.velocity.y:.3f})"
    )

    distance = math.sqrt(
        (ball2.position.x - ball1.position.x) ** 2
        + (ball2.position.y - ball1.position.y) ** 2
    )
    combined_radius = ball1.radius + ball2.radius
    print(f"Distance between centers: {distance:.6f}m")
    print(f"Combined radius: {combined_radius:.6f}m")
    print(f"Gap: {distance - combined_radius:.6f}m")

    trajectory = calculator.calculate_trajectory(
        ball1, table, [ball2], TrajectoryQuality.HIGH, time_limit=1.0
    )

    print(f"\nCollisions detected: {len(trajectory.collisions)}")
    for i, collision in enumerate(trajectory.collisions):
        print(
            f"  Collision {i}: type={collision.type.value} time={collision.time:.3f}s "
            f"pos=({collision.position.x:.3f},{collision.position.y:.3f})"
        )

    return trajectory


def debug_cushion_collision():
    """Debug cushion collision."""
    print("\n=== Debug Cushion Collision ===")

    calculator = TrajectoryCalculator()
    table = TableState.standard_9ft_table()

    # Ball heading toward right cushion
    ball = BallState(
        id="cushion_test",
        position=Vector2D(2.0, 1.0),  # Near right edge
        velocity=Vector2D(1.0, 0.0),  # Moving right
        radius=0.028575,
        mass=0.17,
    )

    print(
        f"Ball: pos=({ball.position.x:.3f},{ball.position.y:.3f}) vel=({ball.velocity.x:.3f},{ball.velocity.y:.3f})"
    )
    print(f"Table width: {table.width:.3f}m")
    print(f"Distance to right cushion: {table.width - ball.position.x:.3f}m")
    print(f"Ball radius: {ball.radius:.6f}m")
    print(
        f"Time to collision (approx): {(table.width - ball.radius - ball.position.x) / ball.velocity.x:.3f}s"
    )

    trajectory = calculator.calculate_trajectory(
        ball, table, [], TrajectoryQuality.HIGH, time_limit=2.0
    )

    print(f"\nCollisions detected: {len(trajectory.collisions)}")
    for i, collision in enumerate(trajectory.collisions):
        print(
            f"  Collision {i}: type={collision.type.value} time={collision.time:.3f}s "
            f"pos=({collision.position.x:.3f},{collision.position.y:.3f})"
        )

    return trajectory


if __name__ == "__main__":
    try:
        debug_simple_trajectory()
        debug_ball_collision()
        debug_cushion_collision()
        print("\n=== All debug tests completed ===")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
