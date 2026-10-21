"""Coordinate utilities shared by vision consumers.

The former metrics, transformation, and visualization exports pointed at
modules that no longer exist. Coordinate conversion now lives in the vision
package's explicit scale-aware module and is re-exported here for callers that
want the utility namespace.
"""

from backend.vision.coordinates import (
    canonical_to_frame,
    playing_area_to_canonical,
    playing_area_to_frame,
    point_is_within_scaled_radius,
    point_to_canonical,
    point_to_frame,
    points_to_frame,
    radius_to_frame_ellipse,
)

__all__ = [
    "canonical_to_frame",
    "playing_area_to_canonical",
    "playing_area_to_frame",
    "point_is_within_scaled_radius",
    "point_to_canonical",
    "point_to_frame",
    "points_to_frame",
    "radius_to_frame_ellipse",
]
