"""Explicit coordinate metadata and pixel scaling helpers for developer tools.

Tool inputs use 4K canonical pixels as the shared interchange space. Image
coordinates and extents carry a source resolution and an explicit scale to that
canonical space so tools never infer a resolution from coordinate magnitudes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

CANONICAL_RESOLUTION = (3840, 2160)


def _resolution(value: Any, field_name: str) -> tuple[int, int]:
    """Validate a width/height pair represented as a list or mapping."""
    if isinstance(value, Mapping):
        pair = (value.get("width"), value.get("height"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pair = (value[0], value[1])
    else:
        raise ValueError(f"{field_name} must declare [width, height]")

    result: list[int] = []
    for dimension in pair:
        if isinstance(dimension, bool):
            raise ValueError(f"{field_name} dimensions must be positive integers")
        try:
            numeric = float(dimension)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{field_name} dimensions must be positive integers"
            ) from error
        if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise ValueError(f"{field_name} dimensions must be positive integers")
        result.append(int(numeric))
    return result[0], result[1]


def _scale_pair(value: Any, field_name: str) -> tuple[float, float]:
    """Validate a positive, finite X/Y scale pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must declare [scale_x, scale_y]")
    try:
        scale_x, scale_y = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} values must be positive and finite") from error
    if not all(math.isfinite(scale) and scale > 0 for scale in (scale_x, scale_y)):
        raise ValueError(f"{field_name} values must be positive and finite")
    return scale_x, scale_y


@dataclass(frozen=True)
class CoordinateMetadata:
    """Coordinate space and scale declared by a tool input."""

    coordinate_space: str
    source_resolution: tuple[int, int]
    coordinate_scale: tuple[float, float]

    @classmethod
    def for_values(
        cls, coordinate_space: str, source_resolution: Any = None
    ) -> CoordinateMetadata:
        """Create metadata for a coordinate-bearing model object.

        Canonical values are already in 4K pixels; a pixel-space object must
        carry its exact source resolution. A canonical object's optional source
        resolution is provenance and does not change the current value space.
        """
        if coordinate_space == "4k_canonical":
            return cls("4k_canonical", CANONICAL_RESOLUTION, (1.0, 1.0))
        if coordinate_space != "pixel":
            raise ValueError(
                f"Unsupported coordinate space for image geometry: {coordinate_space!r}"
            )
        resolution = _resolution(source_resolution, "source_resolution")
        return cls(
            "pixel",
            resolution,
            (
                CANONICAL_RESOLUTION[0] / resolution[0],
                CANONICAL_RESOLUTION[1] / resolution[1],
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        resolution_key: str = "source_resolution",
        scale_key: str = "coordinate_scale",
        space_key: str = "coordinate_space",
    ) -> CoordinateMetadata:
        """Parse and cross-check coordinate metadata without guessing defaults."""
        coordinate_space = value.get(space_key)
        if coordinate_space not in {"pixel", "4k_canonical"}:
            raise ValueError(f"{space_key} must be 'pixel' or '4k_canonical'")
        resolution = _resolution(value.get(resolution_key), resolution_key)
        scale = _scale_pair(value.get(scale_key), scale_key)
        expected = (
            CANONICAL_RESOLUTION[0] / resolution[0],
            CANONICAL_RESOLUTION[1] / resolution[1],
        )
        if not all(
            math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-9)
            for actual, target in zip(scale, expected)
        ):
            raise ValueError(
                f"{scale_key} {scale} does not match {resolution_key} {resolution} "
                f"for 4K canonical pixels (expected {expected})"
            )
        if coordinate_space == "4k_canonical" and resolution != CANONICAL_RESOLUTION:
            raise ValueError(
                "4k_canonical coordinates must declare source_resolution [3840, 2160]"
            )
        return cls(coordinate_space, resolution, scale)

    def to_canonical(self, x: float, y: float) -> tuple[float, float]:
        """Convert one source point into canonical pixels."""
        x_value, y_value = _finite_pair(x, y, "point")
        return (
            x_value * self.coordinate_scale[0],
            y_value * self.coordinate_scale[1],
        )

    def point_to_resolution(
        self, x: float, y: float, target_resolution: Any
    ) -> tuple[float, float]:
        """Convert one source point to another image resolution."""
        target_width, target_height = _resolution(
            target_resolution, "target_resolution"
        )
        canonical_x, canonical_y = self.to_canonical(x, y)
        return (
            canonical_x * target_width / CANONICAL_RESOLUTION[0],
            canonical_y * target_height / CANONICAL_RESOLUTION[1],
        )

    def extent_to_resolution(
        self, width: float, height: float, target_resolution: Any
    ) -> tuple[float, float]:
        """Scale an X/Y extent, such as a box width and height."""
        width_value, height_value = _finite_pair(width, height, "extent")
        target_width, target_height = _resolution(
            target_resolution, "target_resolution"
        )
        return (
            width_value
            * self.coordinate_scale[0]
            * target_width
            / CANONICAL_RESOLUTION[0],
            height_value
            * self.coordinate_scale[1]
            * target_height
            / CANONICAL_RESOLUTION[1],
        )

    def radius_to_resolution(self, radius: float, target_resolution: Any) -> float:
        """Scale a circular radius using the canonical X-axis radius convention.

        The runtime ball model stores one scalar radius and converts it through
        the horizontal pixel scale. Centers and rectangular extents still use
        independent X/Y factors, so this matches the shared model contract.
        """
        radius_value = _finite_number(radius, "radius")
        if radius_value <= 0:
            raise ValueError("radius must be positive")
        target_width, _ = _resolution(target_resolution, "target_resolution")
        return (
            radius_value
            * self.coordinate_scale[0]
            * target_width
            / CANONICAL_RESOLUTION[0]
        )

    def bounds_to_resolution(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        target_resolution: Any,
    ) -> tuple[float, float, float, float]:
        """Scale inclusive/exclusive rectangle edges as points, preserving extents."""
        left_value, top_value = self.point_to_resolution(left, top, target_resolution)
        right_value, bottom_value = self.point_to_resolution(
            right, bottom, target_resolution
        )
        if right_value <= left_value or bottom_value <= top_value:
            raise ValueError("bounds must have right > left and bottom > top")
        return left_value, top_value, right_value, bottom_value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _finite_pair(x: Any, y: Any, field_name: str) -> tuple[float, float]:
    return _finite_number(x, f"{field_name}.x"), _finite_number(y, f"{field_name}.y")


def canonical_point_to_resolution(
    x: float, y: float, target_resolution: Any
) -> tuple[float, float]:
    """Convert canonical 4K point coordinates to a target frame."""
    target_width, target_height = _resolution(target_resolution, "target_resolution")
    x_value, y_value = _finite_pair(x, y, "canonical_point")
    return (
        x_value * target_width / CANONICAL_RESOLUTION[0],
        y_value * target_height / CANONICAL_RESOLUTION[1],
    )


def canonical_radius_to_resolution(radius: float, target_resolution: Any) -> float:
    """Convert a canonical scalar ball radius using the runtime X-axis convention."""
    radius_value = _finite_number(radius, "canonical_radius")
    if radius_value <= 0:
        raise ValueError("canonical_radius must be positive")
    target_width, _ = _resolution(target_resolution, "target_resolution")
    return radius_value * target_width / CANONICAL_RESOLUTION[0]


def point_for_frame(
    x: float,
    y: float,
    coordinate_space: str,
    source_resolution: Any,
    target_resolution: Any,
) -> tuple[float, float]:
    """Convert a model point from its declared space to actual frame pixels."""
    metadata = CoordinateMetadata.for_values(coordinate_space, source_resolution)
    canonical_x, canonical_y = metadata.to_canonical(x, y)
    return canonical_point_to_resolution(canonical_x, canonical_y, target_resolution)


def scalar_extent_for_frame(
    extent: float,
    coordinate_space: str,
    source_resolution: Any,
    target_resolution: Any,
) -> float:
    """Convert a scalar pixel extent from model coordinates to frame pixels."""
    metadata = CoordinateMetadata.for_values(coordinate_space, source_resolution)
    canonical_extent = _finite_number(extent, "extent") * metadata.coordinate_scale[0]
    return canonical_radius_to_resolution(canonical_extent, target_resolution)
