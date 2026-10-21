"""Conversions for configured image geometry used by vision consumers.

Configured table points carry explicit scale metadata describing how their
stored pixel coordinates map into the canonical 4K image plane. Consumers
must use that metadata rather than infer a source resolution: calibration
resolution settings describe a camera mode and can diverge from a point's
actual reference image.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard

from backend.core.constants_4k import CANONICAL_RESOLUTION


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def point_to_canonical(point: Mapping[str, Any]) -> tuple[float, float]:
    """Convert one configured ``{x, y, scale}`` point to canonical pixels.

    ``scale`` contains independent canonical-pixels-per-source-pixel factors
    for x and y. Missing, invalid, or non-finite metadata is rejected rather
    than silently reconstructed from calibration dimensions.
    """
    if not isinstance(point, Mapping):
        raise ValueError("Configured point must be an object")

    x, y = point.get("x"), point.get("y")
    scale = point.get("scale")
    if not _finite_number(x) or not _finite_number(y):
        raise ValueError("Configured point coordinates must be finite numbers")
    if (
        not isinstance(scale, Sequence)
        or isinstance(scale, (str, bytes))
        or len(scale) != 2
        or not all(_finite_number(component) and component > 0 for component in scale)
    ):
        raise ValueError("Configured point requires a positive two-axis scale")

    scaled = float(x) * float(scale[0]), float(y) * float(scale[1])
    if not all(math.isfinite(component) for component in scaled):
        raise ValueError("Scaled point coordinates must be finite")
    return scaled


def point_to_frame(
    point: Mapping[str, Any], frame_size: tuple[int, int]
) -> tuple[float, float]:
    """Convert configured point to the current frame using canonical pixels."""
    width, height = _valid_frame_size(frame_size)
    return canonical_to_frame(point_to_canonical(point), (width, height))


def canonical_to_frame(
    point: tuple[float, float], frame_size: tuple[int, int]
) -> tuple[float, float]:
    """Convert a canonical image point to current-frame pixel coordinates."""
    width, height = _valid_frame_size(frame_size)
    canonical_x, canonical_y = point
    scaled = (
        canonical_x * width / CANONICAL_RESOLUTION[0],
        canonical_y * height / CANONICAL_RESOLUTION[1],
    )
    if not all(math.isfinite(component) for component in scaled):
        raise ValueError("Frame point coordinates must be finite")
    return scaled


def points_to_frame(
    points: Sequence[Mapping[str, Any]], frame_size: tuple[int, int]
) -> list[tuple[float, float]]:
    """Convert all configured points, rejecting the complete set on bad data."""
    return [point_to_frame(point, frame_size) for point in points]


def playing_area_to_canonical(
    corners: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    """Convert and validate the configured four-corner playing-area polygon."""
    if len(corners) != 4:
        raise ValueError("Playing area requires exactly four corners")
    points = [point_to_canonical(corner) for corner in corners]
    crosses: list[float] = []
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        third = points[(index + 2) % 4]
        cross = (second[0] - first[0]) * (third[1] - second[1]) - (
            second[1] - first[1]
        ) * (third[0] - second[0])
        if not math.isfinite(cross) or abs(cross) <= 1e-9:
            raise ValueError("Playing area corners must form a non-degenerate polygon")
        crosses.append(cross)
    if not (all(value > 0 for value in crosses) or all(value < 0 for value in crosses)):
        raise ValueError("Playing area corners must form a convex polygon in order")
    return points


def playing_area_to_frame(
    corners: Sequence[Mapping[str, Any]], frame_size: tuple[int, int]
) -> list[tuple[float, float]]:
    """Convert a valid calibrated playing-area polygon to current-frame pixels."""
    canonical_points = playing_area_to_canonical(corners)
    return [canonical_to_frame(point, frame_size) for point in canonical_points]


def radius_to_frame_ellipse(
    radius_px: float,
    point_scale: Sequence[float],
    frame_size: tuple[int, int],
) -> tuple[float, float]:
    """Scale a source-space circular radius to an anisotropic frame ellipse.

    The configured radius is expressed in the same source-pixel space as its
    marker point. Its per-axis canonical scale follows that marker's metadata,
    then the canonical-to-frame transform produces x/y radii. This retains
    circular masks for uniform scaling and correctly handles nonuniform source
    or target aspect ratios.
    """
    width, height = _valid_frame_size(frame_size)
    if not _finite_number(radius_px) or radius_px < 0:
        raise ValueError("Mask radius must be a finite non-negative number")
    if (
        not isinstance(point_scale, Sequence)
        or isinstance(point_scale, (str, bytes))
        or len(point_scale) != 2
        or not all(
            _finite_number(component) and component > 0 for component in point_scale
        )
    ):
        raise ValueError("Configured point requires a positive two-axis scale")

    radius_x = (
        float(radius_px) * float(point_scale[0]) * width / CANONICAL_RESOLUTION[0]
    )
    radius_y = (
        float(radius_px) * float(point_scale[1]) * height / CANONICAL_RESOLUTION[1]
    )
    if not math.isfinite(radius_x) or not math.isfinite(radius_y):
        raise ValueError("Scaled mask radius must be finite")
    return radius_x, radius_y


def point_is_within_scaled_radius(
    x: float,
    y: float,
    center: tuple[float, float],
    radius_px: float,
    point_scale: Sequence[float],
    frame_size: tuple[int, int],
) -> bool:
    """Check a point against the scale-aware elliptical marker exclusion area."""
    radius_x, radius_y = radius_to_frame_ellipse(radius_px, point_scale, frame_size)
    if radius_x == 0 or radius_y == 0:
        return x == center[0] and y == center[1]
    normalized_distance_squared = ((x - center[0]) / radius_x) ** 2 + (
        (y - center[1]) / radius_y
    ) ** 2
    return normalized_distance_squared < 1.0


def _valid_frame_size(frame_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(frame_size, (tuple, list))
        or len(frame_size) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, float))
            or not math.isfinite(dimension)
            or dimension <= 0
            for dimension in frame_size
        )
    ):
        raise ValueError("Frame size must contain positive finite dimensions")
    return int(frame_size[0]), int(frame_size[1])
