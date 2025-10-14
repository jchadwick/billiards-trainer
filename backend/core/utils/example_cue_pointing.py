#!/usr/bin/env python3
"""Example usage of find_ball_cue_is_pointing_at function.

This example demonstrates how to use the find_ball_cue_is_pointing_at function
with different data types from both the backend core models and vision models.
"""

from backend.core.models import BallState, CueState, Vector2D
from backend.core.utils.geometry import find_ball_cue_is_pointing_at


def example_with_backend_models():
    """Example using backend CueState and BallState models."""
    print("=" * 60)
    print("Example 1: Using backend CueState and BallState models")
    print("=" * 60)

    # Create a cue state pointing at 45 degrees
    cue = CueState(
        tip_position=Vector2D(100, 200),
        angle=45.0,  # degrees
        elevation=0.0,
        estimated_force=5.0,
        confidence=0.95,
    )

    # Create some balls
    balls = [
        BallState(
            id="ball_1",
            position=Vector2D(150, 250),  # Close to cue line
            is_cue_ball=False,
        ),
        BallState(
            id="ball_2",
            position=Vector2D(200, 300),  # Further along cue line
            is_cue_ball=False,
        ),
        BallState(
            id="ball_3",
            position=Vector2D(300, 100),  # Off to the side
            is_cue_ball=False,
        ),
    ]

    # Extract positions for the function
    ball_positions = [ball.position for ball in balls]

    # Find which ball the cue is pointing at
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue.tip_position,
        cue_angle=cue.angle,
        balls=ball_positions,
        max_perpendicular_distance=40.0,
    )

    if target_idx is not None:
        target_ball = balls[target_idx]
        print(f"Cue is pointing at: {target_ball.id}")
        print(f"  Position: ({target_ball.position.x}, {target_ball.position.y})")
    else:
        print("Cue is not pointing at any ball")


def example_with_cue_direction_vector():
    """Example using cue direction vector instead of angle."""
    print("\n" + "=" * 60)
    print("Example 2: Using cue direction vector")
    print("=" * 60)

    # Create a cue with direction vector
    cue_tip = Vector2D(100, 200)
    cue_direction = Vector2D(1, 1).normalize()  # Pointing up-right

    # Create some balls (as tuples for variety)
    balls = [(150, 250), (200, 300), (300, 100)]

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
    """Example demonstrating custom tolerance values."""
    print("\n" + "=" * 60)
    print("Example 3: Custom perpendicular distance tolerance")
    print("=" * 60)

    cue_tip = (100, 200)
    cue_angle = 0.0  # Pointing right

    # Ball is 50 pixels above the cue line
    balls = [(200, 250)]

    # Try with default tolerance (40 pixels)
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=40.0,
    )
    print(f"With 40px tolerance: {target_idx} (ball is 50px off-axis)")

    # Try with larger tolerance (60 pixels)
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=60.0,
    )
    print(f"With 60px tolerance: {target_idx} (ball is 50px off-axis)")


def example_with_vision_models():
    """Example showing how to use with vision detection models."""
    print("\n" + "=" * 60)
    print("Example 4: Using with vision detection results")
    print("=" * 60)

    # Simulate vision detection results (would come from VisionModule)
    # In real usage, these would be from backend.vision.models
    detected_cue = {
        "tip_position": (100, 200),
        "angle": 45.0,
        "butt_position": (50, 150),
    }

    detected_balls = [
        {"position": (150, 250), "track_id": 1},
        {"position": (200, 300), "track_id": 2},
        {"position": (300, 100), "track_id": 3},
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
    """Example showing behavior when multiple balls are in line."""
    print("\n" + "=" * 60)
    print("Example 5: Multiple balls in line (picks closest)")
    print("=" * 60)

    cue_tip = Vector2D(100, 200)
    cue_angle = 0.0  # Pointing right

    # Three balls in a line, all within tolerance
    balls = [
        Vector2D(200, 200),  # Closest
        Vector2D(300, 200),  # Middle
        Vector2D(400, 200),  # Farthest
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
