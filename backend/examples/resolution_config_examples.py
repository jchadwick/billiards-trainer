"""Executable examples for the supported resolution and coordinate helpers.

Image points and bounds enter as frame pixels with a declared source resolution;
``Vector2D`` and ``ResolutionConverter`` express them in canonical 4K pixels.
Physical table dimensions remain explicitly measured meters and are never stored
in a pixel ``Vector2D``.
"""

from backend.core.coordinates import Vector2D
from backend.core.models import TableState
from backend.core.resolution_config import (
    StandardResolution,
    get_standard_resolution,
    validate_point_in_bounds,
)
from backend.core.resolution_converter import ResolutionConverter


def example_1_get_standard_resolutions() -> None:
    """Read named source dimensions supported by the resolution helper."""
    print("Example 1: Standard resolutions")
    hd_resolution = StandardResolution.HD_1080
    print(f"HD 1080: {hd_resolution.width}x{hd_resolution.height}")
    print(f"Aspect ratio: {hd_resolution.aspect_ratio:.2f}")
    print(f"Total pixels: {hd_resolution.total_pixels:,}")

    canonical_resolution = get_standard_resolution("4K")
    if canonical_resolution is None:
        raise RuntimeError("The configured 4K alias is not recognized")
    print(f"4K resolution: {canonical_resolution[0]}x{canonical_resolution[1]}")


def example_2_scale_points_and_bounds() -> tuple[Vector2D, Vector2D, float]:
    """Scale a frame point, image boundary, extent, and radius together."""
    print("\nExample 2: Scaling points and boundaries")
    frame_resolution = (1920, 1080)
    frame_point = Vector2D.from_resolution(960, 540, frame_resolution)
    canonical_point = frame_point.to_4k_canonical()
    print(
        f"Frame center {frame_resolution}: ({frame_point.x:g}, {frame_point.y:g}) "
        f"-> canonical ({canonical_point.x:g}, {canonical_point.y:g})"
    )

    # Bounds and point are both expressed in the same declared frame resolution.
    point_is_inside = validate_point_in_bounds(
        frame_point.x, frame_point.y, *frame_resolution, margin=18
    )
    print(f"Frame-space point is inside its 18px margin: {point_is_inside}")

    # Convert both endpoints of the inclusive image boundary and its extent.
    top_left = Vector2D.from_resolution(0, 0, frame_resolution).to_4k_canonical()
    bottom_right = Vector2D.from_resolution(*frame_resolution, frame_resolution)
    canonical_bottom_right = bottom_right.to_4k_canonical()
    canonical_extent = ResolutionConverter.scale_to_4k(
        frame_resolution[0], frame_resolution[1], frame_resolution
    )
    radius_4k = ResolutionConverter.scale_distance_to_4k(18, frame_resolution)
    print(
        "Canonical image bounds: "
        f"({top_left.x:g}, {top_left.y:g}).."
        f"({canonical_bottom_right.x:g}, {canonical_bottom_right.y:g})"
    )
    print(f"Canonical extent: {canonical_extent[0]:g}x{canonical_extent[1]:g}")
    print(f"18px frame radius: {radius_4k:g} canonical pixels")
    return canonical_point, canonical_bottom_right, radius_4k


def example_3_convert_between_resolutions() -> None:
    """Convert a point between two explicitly named image resolutions."""
    print("\nExample 3: Converting between resolutions")
    source_resolution = (1280, 720)
    target_resolution = (1920, 1080)
    point_source = (640, 360)
    point_target = ResolutionConverter.scale_between_resolutions(
        *point_source, source_resolution, target_resolution
    )
    print(
        f"{point_source} at {source_resolution} -> {point_target} "
        f"at {target_resolution}"
    )


def example_4_calibrated_table_units() -> TableState:
    """Keep calibrated image geometry and real-world table dimensions distinct."""
    print("\nExample 4: Calibrated table with explicit physical measurements")
    camera_resolution = (1920, 1080)
    # These four points are camera-frame pixels in TL, TR, BR, BL order.
    corners_camera_pixels: list[tuple[float, float]] = [
        (100.0, 100.0),
        (1820.0, 100.0),
        (1820.0, 980.0),
        (100.0, 980.0),
    ]
    # Physical dimensions are measured SI values, not pixel vectors.
    table_width_meters = 2.54
    table_height_meters = 1.27
    table = TableState.from_calibration(
        table_corners_pixel=corners_camera_pixels,
        camera_resolution=camera_resolution,
        table_dimensions_real=(table_width_meters, table_height_meters),
    )
    canonical_corners = [
        corner.to_4k_canonical() for corner in table.playing_area_corners or []
    ]
    print(
        f"Image geometry normalized to canonical pixels: "
        f"{[(point.x, point.y) for point in canonical_corners]}"
    )
    print(
        f"Measured table dimensions: {table.real_width_meters:g}m x "
        f"{table.real_height_meters:g}m"
    )
    pixels_per_meter = table.calculate_pixels_per_meter()
    print(f"Derived canonical pixels per meter: {pixels_per_meter}")
    return table


def run_all_examples() -> None:
    """Run every hardware-free resolution example."""
    print("Resolution and coordinate examples")
    example_1_get_standard_resolutions()
    example_2_scale_points_and_bounds()
    example_3_convert_between_resolutions()
    example_4_calibrated_table_units()
    print("All resolution examples completed successfully.")


if __name__ == "__main__":
    run_all_examples()
