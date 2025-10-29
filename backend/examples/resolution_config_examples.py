"""Usage examples for resolution_config module.

This file demonstrates common usage patterns for the resolution configuration utilities.
"""

from backend.core.coordinates import Vector2D
from backend.core.resolution_config import (
    CoordinateSpace,
    ResolutionConfig,
    StandardResolution,
    TableSize,
    get_standard_resolution,
    get_table_dimensions,
    validate_point_in_space,
)


def example_1_get_standard_resolutions():
    """Example 1: Getting standard resolutions."""
    print("Example 1: Getting Standard Resolutions")
    print("-" * 50)

    # Using enum directly
    hd_res = StandardResolution.HD_1080
    print(f"HD 1080: {hd_res.width}x{hd_res.height}")
    print(f"Aspect ratio: {hd_res.aspect_ratio:.2f}")
    print(f"Total pixels: {hd_res.total_pixels:,}")

    # Using helper function with aliases
    res_4k = get_standard_resolution("4K")
    print(f"\n4K resolution: {res_4k}")

    res_1080 = get_standard_resolution("1080p")
    print(f"1080p resolution: {res_1080}")

    print()


def example_2_get_table_dimensions():
    """Example 2: Getting table dimensions."""
    print("Example 2: Getting Table Dimensions")
    print("-" * 50)

    # Using enum directly
    table = TableSize.NINE_FOOT
    print(f"9-foot table: {table.width}m x {table.height}m")
    print(f"Playing area: {table.area:.2f} m²")

    # Using helper function with aliases
    dims_9ft = get_table_dimensions("9ft")
    print(f"\n9ft (via helper): {dims_9ft}")

    dims_8 = get_table_dimensions("8")
    print(f"8-foot table: {dims_8}")

    print()


def example_3_create_coordinate_spaces():
    """Example 3: Creating coordinate spaces."""
    print("Example 3: Creating Coordinate Spaces")
    print("-" * 50)

    # Create pixel coordinate space for HD camera
    pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
    print(f"Pixel space: {pixel_space.width}x{pixel_space.height} {pixel_space.unit}")
    print(f"Center: ({pixel_space.center.x}, {pixel_space.center.y})")

    # Create table coordinate space (origin at corner)
    table_space = ResolutionConfig.create_table_space((2.54, 1.27))
    print(f"\nTable space: {table_space.width}x{table_space.height} {table_space.unit}")
    print(f"Center: ({table_space.center.x:.2f}, {table_space.center.y:.2f})")

    # Create centered table coordinate space (origin at center)
    centered_space = ResolutionConfig.create_table_space((2.54, 1.27), centered=True)
    print(
        f"\nCentered table space origin: ({centered_space.origin_x:.2f}, {centered_space.origin_y:.2f})"
    )
    print(f"Center: ({centered_space.center.x:.2f}, {centered_space.center.y:.2f})")

    print()


def example_4_validate_coordinates():
    """Example 4: Validating coordinates."""
    print("Example 4: Validating Coordinates")
    print("-" * 50)

    table_space = ResolutionConfig.create_table_space((2.54, 1.27))

    # Test various ball positions
    test_positions = [
        Vector2D(1.27, 0.64),  # Center of table - valid
        Vector2D(0.1, 0.1),  # Near corner - valid
        Vector2D(-0.5, 0.5),  # Outside table - invalid
        Vector2D(3.0, 1.0),  # Outside table - invalid
    ]

    for pos in test_positions:
        is_valid, error = ResolutionConfig.validate_vector(pos, table_space)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{status}: ({pos.x:.2f}, {pos.y:.2f})")
        if error:
            print(f"  Error: {error}")

    print()


def example_5_scale_coordinates():
    """Example 5: Scaling coordinates between spaces."""
    print("Example 5: Scaling Coordinates Between Spaces")
    print("-" * 50)

    # Setup: HD camera viewing a 9-foot table
    pixel_space = ResolutionConfig.create_pixel_space((1920, 1080))
    table_space = ResolutionConfig.create_table_space((2.54, 1.27))

    # Simulate ball detection in pixel coordinates
    detected_positions_pixels = [
        Vector2D(960, 540),  # Center of image
        Vector2D(100, 100),  # Near top-left corner
        Vector2D(1800, 1000),  # Near bottom-right corner
    ]

    print("Converting pixel coordinates to table coordinates:")
    for pixel_pos in detected_positions_pixels:
        table_pos = ResolutionConfig.scale_vector(pixel_pos, pixel_space, table_space)
        print(
            f"  Pixel ({pixel_pos.x:.0f}, {pixel_pos.y:.0f}) -> "
            f"Table ({table_pos.x:.2f}m, {table_pos.y:.2f}m)"
        )

    print()


def example_6_clamp_coordinates():
    """Example 6: Clamping coordinates to bounds."""
    print("Example 6: Clamping Coordinates to Bounds")
    print("-" * 50)

    table_space = ResolutionConfig.create_table_space((2.54, 1.27))

    # Test positions that might be out of bounds
    test_positions = [
        Vector2D(-0.5, 0.5),  # Outside left edge
        Vector2D(3.0, 1.0),  # Outside right edge
        Vector2D(1.0, 1.5),  # Outside top edge
        Vector2D(1.0, 0.5),  # Inside (should stay the same)
    ]

    print("Clamping positions to table bounds:")
    for pos in test_positions:
        clamped = table_space.clamp_vector(pos)
        changed = pos.x != clamped.x or pos.y != clamped.y
        marker = "→" if changed else " "
        print(
            f"  {marker} ({pos.x:.2f}, {pos.y:.2f}) -> ({clamped.x:.2f}, {clamped.y:.2f})"
        )

    print()


def example_7_normalize_coordinates():
    """Example 7: Normalizing coordinates to [0, 1] range."""
    print("Example 7: Normalizing Coordinates")
    print("-" * 50)

    space = CoordinateSpace(width=1920, height=1080, unit="pixels")

    # Test positions
    test_positions = [
        (0, 0),  # Top-left corner
        (960, 540),  # Center
        (1920, 1080),  # Bottom-right corner
        (480, 270),  # Quarter point
    ]

    print("Normalizing pixel coordinates to [0, 1] range:")
    for x, y in test_positions:
        norm_x, norm_y = space.normalize_point(x, y)
        print(f"  ({x:4}, {y:4}) -> ({norm_x:.3f}, {norm_y:.3f})")

    print()


def example_8_complete_workflow():
    """Example 8: Complete workflow - Ball detection to validation."""
    print("Example 8: Complete Workflow - Ball Detection Pipeline")
    print("-" * 50)

    # Step 1: Setup coordinate spaces
    print("Step 1: Setup coordinate spaces")
    camera_resolution = get_standard_resolution("1080p")
    table_dimensions = get_table_dimensions("9")

    pixel_space = ResolutionConfig.create_pixel_space(camera_resolution)
    table_space = ResolutionConfig.create_table_space(table_dimensions)

    print(f"  Camera: {camera_resolution[0]}x{camera_resolution[1]}")
    print(f"  Table: {table_dimensions[0]}m x {table_dimensions[1]}m")

    # Step 2: Simulate ball detection (in pixels)
    print("\nStep 2: Detect balls in camera image (pixel coordinates)")
    detected_balls = [
        Vector2D(960, 540),  # Cue ball at center
        Vector2D(1440, 540),  # Object ball to the right
        Vector2D(480, 540),  # Object ball to the left
    ]
    for i, ball in enumerate(detected_balls):
        print(f"  Ball {i+1}: ({ball.x:.0f}, {ball.y:.0f}) pixels")

    # Step 3: Convert to table coordinates
    print("\nStep 3: Convert to table coordinates (meters)")
    table_positions = []
    for i, pixel_pos in enumerate(detected_balls):
        table_pos = ResolutionConfig.scale_vector(pixel_pos, pixel_space, table_space)
        table_positions.append(table_pos)
        print(f"  Ball {i+1}: ({table_pos.x:.2f}, {table_pos.y:.2f}) meters")

    # Step 4: Validate positions
    print("\nStep 4: Validate ball positions")
    valid_count = 0
    # Add margin for ball radius (standard pool ball is ~0.057m diameter)
    ball_radius = 0.028575  # meters
    for i, pos in enumerate(table_positions):
        is_valid, error = ResolutionConfig.validate_vector(
            pos, table_space, margin=ball_radius
        )
        if is_valid:
            valid_count += 1
            print(f"  Ball {i+1}: ✓ Valid position")
        else:
            print(f"  Ball {i+1}: ✗ Invalid - {error}")

    print(f"\nResult: {valid_count}/{len(table_positions)} balls have valid positions")

    print()


def run_all_examples():
    """Run all examples."""
    print("=" * 60)
    print("Resolution Config Module - Usage Examples")
    print("=" * 60)
    print()

    example_1_get_standard_resolutions()
    example_2_get_table_dimensions()
    example_3_create_coordinate_spaces()
    example_4_validate_coordinates()
    example_5_scale_coordinates()
    example_6_clamp_coordinates()
    example_7_normalize_coordinates()
    example_8_complete_workflow()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
