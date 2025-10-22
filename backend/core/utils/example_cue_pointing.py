#!/usr/bin/env python3
"""Example usage of find_ball_cue_is_pointing_at function.

This example demonstrates how to use the find_ball_cue_is_pointing_at function
with different data types from both the backend core models and vision models.

All examples use 4K coordinate system (3840×2160 pixels).
"""

from backend.core.constants_4k import BALL_RADIUS_4K, CANONICAL_HEIGHT, CANONICAL_WIDTH
from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState
from backend.core.utils.geometry import find_ball_cue_is_pointing_at


def example_with_backend_models():
    """Example using backend CueState and BallState models (4K coordinates)."""
    print("=" * 60)
    print("Example 1: Using backend CueState and BallState models (4K)")
    print("=" * 60)

    # Create a cue state pointing at 45 degrees (4K coordinates)
    cue = CueState(
        tip_position=Vector2D(200, 400),  # 4K: doubled from 1080p (100, 200)
        angle=45.0,  # degrees
        elevation=0.0,
        estimated_force=5.0,
        confidence=0.95,
    )

    # Create some balls (4K coordinates)
    balls = [
        BallState(
            id="ball_1",
            position=Vector2D(300, 500),  # Close to cue line (4K: doubled)
            is_cue_ball=False,
        ),
        BallState(
            id="ball_2",
            position=Vector2D(400, 600),  # Further along cue line (4K: doubled)
            is_cue_ball=False,
        ),
        BallState(
            id="ball_3",
            position=Vector2D(600, 200),  # Off to the side (4K: doubled)
            is_cue_ball=False,
        ),
    ]

    # Extract positions for the function
    ball_positions = [ball.position for ball in balls]

    # Find which ball the cue is pointing at (4K tolerance: 40 * 2 = 80 pixels)
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue.tip_position,
        cue_angle=cue.angle,
        balls=ball_positions,
        max_perpendicular_distance=80.0,  # 4K: doubled from 1080p (40.0)
    )

    if target_idx is not None:
        target_ball = balls[target_idx]
        print(f"Cue is pointing at: {target_ball.id}")
        print(f"  Position: ({target_ball.position.x}, {target_ball.position.y})")
    else:
        print("Cue is not pointing at any ball")


def example_with_cue_direction_vector():
    """Example using cue direction vector instead of angle (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 2: Using cue direction vector (4K)")
    print("=" * 60)

    # Create a cue with direction vector (4K coordinates)
    cue_tip = Vector2D(200, 400)  # 4K: doubled
    cue_direction = Vector2D(1, 1).normalize()  # Pointing up-right

    # Create some balls (as tuples for variety, 4K coordinates)
    balls = [(300, 500), (400, 600), (600, 200)]  # 4K: doubled

    # Find which ball the cue is pointing at
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_direction=cue_direction,
        balls=balls,
    )

    if target_idx is not None:
        print(f"Cue is pointing at ball #{target_idx + 1}")
        print(f"  Position: {balls[target_idx]}")
    else:
        print("Cue is not pointing at any ball")


def example_with_custom_tolerance():
    """Example demonstrating custom tolerance values (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 3: Custom perpendicular distance tolerance (4K)")
    print("=" * 60)

    cue_tip = (200, 400)  # 4K: doubled
    cue_angle = 0.0  # Pointing right

    # Ball is 100 pixels above the cue line (4K: 50 * 2)
    balls = [(400, 500)]  # 4K: doubled

    # Try with default tolerance (80 pixels, 4K: 40 * 2)
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=80.0,  # 4K: doubled
    )
    print(f"With 80px tolerance: {target_idx} (ball is 100px off-axis)")

    # Try with larger tolerance (120 pixels, 4K: 60 * 2)
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=120.0,  # 4K: doubled
    )
    print(f"With 120px tolerance: {target_idx} (ball is 100px off-axis)")


def example_with_vision_models():
    """Example showing how to use with vision detection models (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 4: Using with vision detection results (4K)")
    print("=" * 60)

    # Simulate vision detection results (would come from VisionModule)
    # In real usage, these would be from backend.vision.models
    # All coordinates in 4K
    detected_cue = {
        "tip_position": (200, 400),  # 4K: doubled
        "angle": 45.0,
        "butt_position": (100, 300),  # 4K: doubled
    }

    detected_balls = [
        {"position": (300, 500), "track_id": 1},  # 4K: doubled
        {"position": (400, 600), "track_id": 2},  # 4K: doubled
        {"position": (600, 200), "track_id": 3},  # 4K: doubled
    ]

    # Extract just the positions
    ball_positions = [ball["position"] for ball in detected_balls]

    # Find target
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=detected_cue["tip_position"],
        cue_angle=detected_cue["angle"],
        balls=ball_positions,
    )

    if target_idx is not None:
        target_ball = detected_balls[target_idx]
        print(f"Cue is pointing at ball with track_id: {target_ball['track_id']}")
        print(f"  Position: {target_ball['position']}")
    else:
        print("Cue is not pointing at any ball")


def example_multiple_balls_in_line():
    """Example showing behavior when multiple balls are in line (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 5: Multiple balls in line (picks closest) - 4K")
    print("=" * 60)

    cue_tip = Vector2D(200, 400)  # 4K: doubled
    cue_angle = 0.0  # Pointing right

    # Three balls in a line, all within tolerance (4K coordinates)
    balls = [
        Vector2D(400, 400),  # Closest (4K: doubled)
        Vector2D(600, 400),  # Middle (4K: doubled)
        Vector2D(800, 400),  # Farthest (4K: doubled)
    ]

    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip, cue_angle=cue_angle, balls=balls
    )

    if target_idx is not None:
        print(f"Selected ball #{target_idx + 1} (closest along cue direction)")
        print(f"  Position: ({balls[target_idx].x}, {balls[target_idx].y})")


if __name__ == "__main__":
    # Run all examples
    example_with_backend_models()
    example_with_cue_direction_vector()
    example_with_custom_tolerance()
    example_with_vision_models()
    example_multiple_balls_in_line()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
