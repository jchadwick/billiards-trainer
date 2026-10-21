#!/usr/bin/env python3
"""Example usage of find_ball_cue_is_pointing_at function.

This example demonstrates how to use the find_ball_cue_is_pointing_at function
with different data types from both the backend core models and vision models.

The geometry helper consumes canonical 4K coordinates (3840×2160 pixels).
Examples that begin with 1080p camera values convert them before calling it.
"""

from backend.core.coordinates import Vector2D
from backend.core.models import BallState, CueState
from backend.core.resolution_converter import ResolutionConverter
from backend.core.utils.geometry import find_ball_cue_is_pointing_at

CAMERA_RESOLUTION = (1920, 1080)


def example_with_backend_models() -> None:
    """Example using backend CueState and BallState models (4K coordinates)."""
    print("=" * 60)
    print("Example 1: Using backend CueState and BallState models (4K)")
    print("=" * 60)

    # The detector points below are in a 1080p camera frame. The model factories
    # attach that source scale and normalize positions to canonical 4K pixels.
    cue = CueState(
        tip_position=Vector2D.from_resolution(100, 200, CAMERA_RESOLUTION),
        angle=45.0,  # degrees
        elevation=0.0,
        estimated_force=5.0,
        confidence=0.95,
    )

    # BallState normalizes these source-frame positions and pixel radii.
    balls = [
        BallState(
            id="ball_1",
            position=Vector2D.from_resolution(150, 250, CAMERA_RESOLUTION),
            radius=18.0,
            radius_scale=ResolutionConverter.calculate_scale_to_4k(CAMERA_RESOLUTION),
            is_cue_ball=False,
        ),
        BallState(
            id="ball_2",
            position=Vector2D.from_resolution(200, 300, CAMERA_RESOLUTION),
            radius=18.0,
            radius_scale=ResolutionConverter.calculate_scale_to_4k(CAMERA_RESOLUTION),
            is_cue_ball=False,
        ),
        BallState(
            id="ball_3",
            position=Vector2D.from_resolution(300, 100, CAMERA_RESOLUTION),
            radius=18.0,
            radius_scale=ResolutionConverter.calculate_scale_to_4k(CAMERA_RESOLUTION),
            is_cue_ball=False,
        ),
    ]

    # Extract positions for the function
    ball_positions: list[Vector2D | tuple[float, float]] = [
        ball.position for ball in balls
    ]

    # The helper expects a canonical-pixel tolerance, so scale the 1080p value.
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue.tip_position,
        cue_angle=cue.angle,
        balls=ball_positions,
        max_perpendicular_distance=ResolutionConverter.scale_distance_to_4k(
            40.0, CAMERA_RESOLUTION
        ),
    )

    if target_idx is not None:
        target_ball = balls[target_idx]
        print(f"Cue is pointing at: {target_ball.id}")
        print(f"  Position: ({target_ball.position.x}, {target_ball.position.y})")
    else:
        print("Cue is not pointing at any ball")


def example_with_cue_direction_vector() -> None:
    """Example using cue direction vector instead of angle (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 2: Using cue direction vector (4K)")
    print("=" * 60)

    # Create a cue with direction vector (4K coordinates)
    cue_tip = Vector2D.from_resolution(100, 200, CAMERA_RESOLUTION).to_4k_canonical()
    cue_direction = Vector2D.from_4k(1, 1).normalize()  # Dimensionless direction

    # Model positions carry their source resolution and are normalized by BallState.
    balls: list[Vector2D | tuple[float, float]] = [
        BallState(
            id=f"ball_{index}",
            position=Vector2D.from_resolution(x, y, CAMERA_RESOLUTION),
        ).position
        for index, (x, y) in enumerate(((150, 250), (200, 300), (300, 100)), 1)
    ]

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


def example_with_custom_tolerance() -> None:
    """Example demonstrating custom tolerance values (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 3: Custom perpendicular distance tolerance (4K)")
    print("=" * 60)

    # This low-level API accepts bare tuples only in canonical 4K pixels. Convert
    # the 1080p sample point and its distance threshold before passing them in.
    cue_tip = ResolutionConverter.scale_to_4k(100, 200, CAMERA_RESOLUTION)
    cue_angle = 0.0  # Pointing right

    # This ball is 50 pixels above the line in 1080p, or 100 canonical pixels.
    balls: list[Vector2D | tuple[float, float]] = [
        ResolutionConverter.scale_to_4k(200, 250, CAMERA_RESOLUTION)
    ]
    base_tolerance_4k = ResolutionConverter.scale_distance_to_4k(
        40.0, CAMERA_RESOLUTION
    )

    # Try with a 40px-at-1080p tolerance, represented canonically.
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=base_tolerance_4k,
    )
    print(f"With 80px tolerance: {target_idx} (ball is 100px off-axis)")

    # Try with a 60px-at-1080p tolerance, represented canonically.
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip,
        cue_angle=cue_angle,
        balls=balls,
        max_perpendicular_distance=ResolutionConverter.scale_distance_to_4k(
            60.0, CAMERA_RESOLUTION
        ),
    )
    print(f"With 120px tolerance: {target_idx} (ball is 100px off-axis)")


def example_with_vision_models() -> None:
    """Example showing how to use with vision detection models (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 4: Using with vision detection results (4K)")
    print("=" * 60)

    # VisionModule exposes canonical image-space positions. Keep the scale on
    # each point, even though the low-level helper can also accept canonical tuples.
    detected_cue_tip = Vector2D.from_4k(200, 400)
    detected_cue_angle = 45.0
    detected_balls: list[tuple[int, Vector2D]] = [
        (1, Vector2D.from_4k(300, 500)),
        (2, Vector2D.from_4k(400, 600)),
        (3, Vector2D.from_4k(600, 200)),
    ]

    # Keep the model vectors (and their scale metadata) through geometry calls.
    ball_positions: list[Vector2D | tuple[float, float]] = [
        position for _, position in detected_balls
    ]

    # Find target
    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=detected_cue_tip,
        cue_angle=detected_cue_angle,
        balls=ball_positions,
    )

    if target_idx is not None:
        track_id, position = detected_balls[target_idx]
        print(f"Cue is pointing at ball with track_id: {track_id}")
        print(f"  Position: {position}")
    else:
        print("Cue is not pointing at any ball")


def example_multiple_balls_in_line() -> None:
    """Example showing behavior when multiple balls are in line (4K coordinates)."""
    print("\n" + "=" * 60)
    print("Example 5: Multiple balls in line (picks closest) - 4K")
    print("=" * 60)

    cue_tip = Vector2D.from_resolution(100, 200, CAMERA_RESOLUTION).to_4k_canonical()
    cue_angle = 0.0  # Pointing right

    # Three balls in a line, all within tolerance (4K coordinates)
    balls: list[Vector2D | tuple[float, float]] = [
        Vector2D.from_resolution(200, 200, CAMERA_RESOLUTION).to_4k_canonical(),
        Vector2D.from_resolution(300, 200, CAMERA_RESOLUTION).to_4k_canonical(),
        Vector2D.from_resolution(400, 200, CAMERA_RESOLUTION).to_4k_canonical(),
    ]

    target_idx = find_ball_cue_is_pointing_at(
        cue_tip=cue_tip, cue_angle=cue_angle, balls=balls
    )

    if target_idx is not None:
        print(f"Selected ball #{target_idx + 1} (closest along cue direction)")
        target_ball = balls[target_idx]
        if isinstance(target_ball, Vector2D):
            print(f"  Position: ({target_ball.x}, {target_ball.y})")
        else:
            print(f"  Position: ({target_ball[0]}, {target_ball[1]})")


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
