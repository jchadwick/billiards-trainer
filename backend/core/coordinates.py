"""4K Pixel-Based Coordinate System for the billiards trainer system.

This module provides a simplified coordinate system based on 4K pixels (3840×2160)
as the canonical storage format. All coordinates carry mandatory scale metadata
that enables conversion to/from other resolutions.

The module introduces:
- Vector2D: 2D vector with mandatory scale metadata for resolution tracking

Design Goals:
1. Single Canonical Format - All coordinates stored in 4K pixels
2. Mandatory Scale Metadata - No ambiguity about coordinate resolution
3. Simple Conversions - Scale-based transformations between resolutions
4. Type Safety - Prevent coordinate space mixing errors

See: thoughts/4k_standardization_plan.md for full design specification
"""

import math
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class Vector2D:
    """2D vector with mandatory scale metadata for 4K canonical coordinate system.

    All vectors track their scale relative to 4K canonical resolution (3840×2160).
    The scale metadata enables conversion to/from other resolutions while maintaining
    4K as the canonical storage format.

    Attributes:
        x: X coordinate value in pixels
        y: Y coordinate value in pixels
        scale: MANDATORY tuple of (scale_x, scale_y) to reach 4K canonical
               - For 4K coordinates: scale = (1.0, 1.0)
               - For 1080p coordinates: scale = (2.0, 2.0)
               - For lower resolutions: scale > 1.0
               - For higher resolutions: scale < 1.0

    Mathematical Operations:
        All operations work in 4K canonical space and preserve scale metadata.
        Operations convert to 4K, perform calculation, then convert back.

    Example:
        >>> # Create from 4K canonical (scale = 1.0)
        >>> pos_4k = Vector2D.from_4k(1920.0, 1080.0)
        >>> pos_4k.scale
        (1.0, 1.0)

        >>> # Create from 1080p (auto-calculates scale = 2.0)
        >>> pos_1080p = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
        >>> pos_1080p.scale
        (2.0, 2.0)

        >>> # Convert to 4K canonical
        >>> pos_4k = pos_1080p.to_4k_canonical()
        >>> pos_4k.x, pos_4k.y
        (1920.0, 1080.0)
    """

    x: float
    y: float
    scale: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate vector after initialization."""
        if self.scale is None:
            raise ValueError("Scale metadata is MANDATORY for all Vector2D instances")
        if len(self.scale) != 2:
            raise ValueError(
                f"Scale must be a tuple of (scale_x, scale_y), got {self.scale}"
            )
        if self.scale[0] <= 0 or self.scale[1] <= 0:
            raise ValueError(f"Scale factors must be positive, got {self.scale}")

    # =========================================================================
    # Geometric Operations
    # =========================================================================

    def magnitude(self) -> float:
        """Calculate the length/magnitude of the vector.

        Returns:
            Length of the vector (always non-negative)

        Example:
            >>> v = Vector2D(3.0, 4.0)
            >>> v.magnitude()
            5.0
        """
        return math.sqrt(self.x**2 + self.y**2)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (avoids sqrt for performance).

        Useful for distance comparisons where actual distance isn't needed.

        Returns:
            Squared magnitude of the vector

        Example:
            >>> v = Vector2D(3.0, 4.0)
            >>> v.magnitude_squared()
            25.0
        """
        return self.x**2 + self.y**2

    def normalize(self) -> "Vector2D":
        """Return a normalized (unit) vector in the same direction.

        Preserves scale metadata from original vector.

        Returns:
            Unit vector (magnitude = 1) in same direction, or zero vector if
            magnitude is zero

        Example:
            >>> v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
            >>> u = v.normalize()
            >>> u.magnitude()
            1.0
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(
                0.0,
                0.0,
                scale=self.scale,
            )
        return Vector2D(
            self.x / mag,
            self.y / mag,
            scale=self.scale,
        )

    def dot(self, other: "Vector2D") -> float:
        """Calculate the dot product with another vector.

        Args:
            other: Vector to compute dot product with

        Returns:
            Scalar dot product value

        Note:
            Does not validate coordinate spaces. Use with caution when
            comparing vectors from different spaces.

        Example:
            >>> v1 = Vector2D(1.0, 0.0)
            >>> v2 = Vector2D(0.0, 1.0)
            >>> v1.dot(v2)
            0.0
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2D") -> float:
        """Calculate the 2D cross product (returns scalar z-component).

        Args:
            other: Vector to compute cross product with

        Returns:
            Z-component of 3D cross product (scalar)

        Example:
            >>> v1 = Vector2D(1.0, 0.0)
            >>> v2 = Vector2D(0.0, 1.0)
            >>> v1.cross(v2)
            1.0
        """
        return self.x * other.y - self.y * other.x

    def distance_to(self, other: "Vector2D") -> float:
        """Calculate the Euclidean distance to another point.

        Args:
            other: Point to measure distance to

        Returns:
            Distance between points

        Note:
            Does not validate coordinate spaces. Ensure both vectors are
            in the same coordinate space before calling.

        Example:
            >>> p1 = Vector2D(0.0, 0.0)
            >>> p2 = Vector2D(3.0, 4.0)
            >>> p1.distance_to(p2)
            5.0
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def angle_to(self, other: "Vector2D") -> float:
        """Calculate the angle to another vector in radians.

        Args:
            other: Target vector/point

        Returns:
            Angle in radians from this vector to other vector
            Range: [-π, π]

        Example:
            >>> origin = Vector2D(0.0, 0.0)
            >>> right = Vector2D(1.0, 0.0)
            >>> origin.angle_to(right)
            0.0
        """
        return math.atan2(other.y - self.y, other.x - self.x)

    def rotate(self, angle: float) -> "Vector2D":
        """Rotate the vector by the given angle in radians.

        Preserves scale metadata.

        Args:
            angle: Rotation angle in radians (positive = counter-clockwise)

        Returns:
            Rotated vector

        Example:
            >>> v = Vector2D(1.0, 0.0, scale=(1.0, 1.0))
            >>> v_rotated = v.rotate(math.pi / 2)  # 90 degrees
            >>> abs(v_rotated.x) < 1e-10  # Close to zero
            True
            >>> abs(v_rotated.y - 1.0) < 1e-10  # Close to 1.0
            True
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
            scale=self.scale,
        )

    def scale_by(self, factor: float) -> "Vector2D":
        """Scale the vector by a scalar factor.

        Preserves scale metadata. Note: This scales the vector coordinates,
        not the scale metadata itself.

        Args:
            factor: Scaling factor

        Returns:
            Scaled vector

        Example:
            >>> v = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
            >>> v2 = v.scale_by(2.0)
            >>> (v2.x, v2.y)
            (6.0, 8.0)
        """
        return Vector2D(
            self.x * factor,
            self.y * factor,
            scale=self.scale,
        )

    # =========================================================================
    # Operator Overloads
    # =========================================================================

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Add two vectors.

        Operations are performed in 4K canonical space, then converted back to
        the scale of the left operand (self).

        Args:
            other: Vector to add

        Returns:
            Sum vector in self's scale

        Example:
            >>> v1 = Vector2D(1.0, 2.0, scale=(1.0, 1.0))
            >>> v2 = Vector2D(3.0, 4.0, scale=(1.0, 1.0))
            >>> v3 = v1 + v2
            >>> (v3.x, v3.y)
            (4.0, 6.0)
        """
        # Convert both to 4K canonical
        self_4k = self.to_4k_canonical()
        other_4k = other.to_4k_canonical()

        # Perform addition in 4K space
        result_4k = Vector2D(
            self_4k.x + other_4k.x,
            self_4k.y + other_4k.y,
            scale=(1.0, 1.0),
        )

        # Convert back to self's scale if needed
        if self.scale == (1.0, 1.0):
            return result_4k
        else:
            return result_4k.to_scale(self.scale)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Subtract two vectors.

        Operations are performed in 4K canonical space, then converted back to
        the scale of the left operand (self).

        Args:
            other: Vector to subtract

        Returns:
            Difference vector in self's scale
        """
        # Convert both to 4K canonical
        self_4k = self.to_4k_canonical()
        other_4k = other.to_4k_canonical()

        # Perform subtraction in 4K space
        result_4k = Vector2D(
            self_4k.x - other_4k.x,
            self_4k.y - other_4k.y,
            scale=(1.0, 1.0),
        )

        # Convert back to self's scale if needed
        if self.scale == (1.0, 1.0):
            return result_4k
        else:
            return result_4k.to_scale(self.scale)

    def __mul__(self, scalar: float) -> "Vector2D":
        """Multiply vector by a scalar.

        Preserves scale metadata.

        Args:
            scalar: Scalar multiplier

        Returns:
            Scaled vector
        """
        return Vector2D(
            self.x * scalar,
            self.y * scalar,
            scale=self.scale,
        )

    def __truediv__(self, scalar: float) -> "Vector2D":
        """Divide vector by a scalar.

        Preserves scale metadata.

        Args:
            scalar: Scalar divisor (must be non-zero)

        Returns:
            Scaled vector

        Raises:
            ZeroDivisionError: If scalar is zero
        """
        return Vector2D(
            self.x / scalar,
            self.y / scalar,
            scale=self.scale,
        )

    def __neg__(self) -> "Vector2D":
        """Negate the vector.

        Preserves scale metadata.

        Returns:
            Negated vector
        """
        return Vector2D(
            -self.x,
            -self.y,
            scale=self.scale,
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another vector.

        Vectors are equal if their x, y coordinates match exactly.
        Coordinate space metadata is NOT considered for equality.

        Args:
            other: Object to compare with

        Returns:
            True if coordinates match, False otherwise
        """
        if not isinstance(other, Vector2D):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """Calculate hash for use in sets and dicts.

        Only based on x, y coordinates (not metadata).

        Returns:
            Hash value
        """
        return hash((self.x, self.y))

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def to_4k_canonical(self) -> "Vector2D":
        """Convert to 4K canonical coordinates (scale = 1.0).

        Returns:
            New Vector2D in 4K canonical coordinates

        Example:
            >>> # 1080p coordinates
            >>> v = Vector2D(960.0, 540.0, scale=(2.0, 2.0))
            >>> v_4k = v.to_4k_canonical()
            >>> v_4k.x, v_4k.y, v_4k.scale
            (1920.0, 1080.0, (1.0, 1.0))
        """
        return Vector2D(
            x=self.x * self.scale[0],
            y=self.y * self.scale[1],
            scale=(1.0, 1.0),
        )

    def to_resolution(self, target_resolution: tuple[int, int]) -> "Vector2D":
        """Convert to target resolution coordinates.

        Args:
            target_resolution: Target (width, height) in pixels

        Returns:
            New Vector2D in target resolution

        Example:
            >>> # Convert 4K to 1080p
            >>> v_4k = Vector2D(1920.0, 1080.0, scale=(1.0, 1.0))
            >>> v_1080p = v_4k.to_resolution((1920, 1080))
            >>> v_1080p.x, v_1080p.y, v_1080p.scale
            (960.0, 540.0, (2.0, 2.0))
        """
        # First convert to 4K canonical
        canonical = self.to_4k_canonical()

        # Import here to avoid circular dependency
        from .resolution_converter import ResolutionConverter

        # Calculate scale from 4K to target
        target_scale = ResolutionConverter.calculate_scale_from_4k(target_resolution)

        return Vector2D(
            x=canonical.x * target_scale[0],
            y=canonical.y * target_scale[1],
            scale=ResolutionConverter.calculate_scale_to_4k(target_resolution),
        )

    def to_scale(self, target_scale: tuple[float, float]) -> "Vector2D":
        """Convert to a specific scale.

        This is an internal helper method for operations.

        Args:
            target_scale: Target scale tuple

        Returns:
            New Vector2D with target scale
        """
        if self.scale == target_scale:
            return self

        # Convert to 4K canonical first
        canonical = self.to_4k_canonical()

        # Calculate inverse scale to get to target
        inverse_scale = (1.0 / target_scale[0], 1.0 / target_scale[1])

        return Vector2D(
            x=canonical.x * inverse_scale[0],
            y=canonical.y * inverse_scale[1],
            scale=target_scale,
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Includes mandatory scale metadata.

        Returns:
            Dictionary representation with scale

        Example:
            >>> v = Vector2D.from_4k(1920.0, 1080.0)
            >>> d = v.to_dict()
            >>> d['scale']
            [1.0, 1.0]
        """
        return {
            "x": self.x,
            "y": self.y,
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vector2D":
        """Create from dictionary.

        Args:
            data: Dictionary with 'x', 'y', and 'scale'

        Returns:
            Vector2D instance

        Example:
            >>> data = {'x': 960, 'y': 540, 'scale': [2.0, 2.0]}
            >>> v = Vector2D.from_dict(data)
            >>> v.scale
            (2.0, 2.0)
        """
        scale = data.get("scale")
        if scale is None:
            raise ValueError("Scale is required in dictionary data")

        return cls(
            x=data["x"],
            y=data["y"],
            scale=tuple(scale),
        )

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_4k(cls, x: float, y: float) -> "Vector2D":
        """Create from 4K canonical coordinates (scale = 1.0).

        Args:
            x: X coordinate in 4K pixels
            y: Y coordinate in 4K pixels

        Returns:
            Vector2D with scale = (1.0, 1.0)

        Example:
            >>> v = Vector2D.from_4k(1920.0, 1080.0)
            >>> v.scale
            (1.0, 1.0)
        """
        return cls(x=x, y=y, scale=(1.0, 1.0))

    @classmethod
    def from_resolution(
        cls, x: float, y: float, resolution: tuple[int, int]
    ) -> "Vector2D":
        """Create from any resolution with auto-calculated scale to 4K.

        Args:
            x: X coordinate in source resolution
            y: Y coordinate in source resolution
            resolution: Source (width, height) in pixels

        Returns:
            Vector2D with auto-calculated scale to reach 4K

        Example:
            >>> # Create from 1080p coordinates
            >>> v = Vector2D.from_resolution(960.0, 540.0, (1920, 1080))
            >>> v.scale
            (2.0, 2.0)
            >>> # Verify converts to 4K correctly
            >>> v_4k = v.to_4k_canonical()
            >>> v_4k.x, v_4k.y
            (1920.0, 1080.0)
        """
        from .resolution_converter import ResolutionConverter

        scale = ResolutionConverter.calculate_scale_to_4k(resolution)
        return cls(x=x, y=y, scale=scale)

    # =========================================================================
    # Utility Factory Methods
    # =========================================================================

    @classmethod
    def zero(cls, scale: tuple[float, float] = (1.0, 1.0)) -> "Vector2D":
        """Create a zero vector.

        Args:
            scale: Scale metadata (defaults to 4K canonical)

        Returns:
            Zero vector (0, 0)

        Example:
            >>> v = Vector2D.zero()
            >>> (v.x, v.y)
            (0.0, 0.0)
        """
        return cls(0.0, 0.0, scale=scale)

    @classmethod
    def unit_x(cls, scale: tuple[float, float] = (1.0, 1.0)) -> "Vector2D":
        """Create a unit vector in the x direction.

        Args:
            scale: Scale metadata (defaults to 4K canonical)

        Returns:
            Unit vector (1, 0)
        """
        return cls(1.0, 0.0, scale=scale)

    @classmethod
    def unit_y(cls, scale: tuple[float, float] = (1.0, 1.0)) -> "Vector2D":
        """Create a unit vector in the y direction.

        Args:
            scale: Scale metadata (defaults to 4K canonical)

        Returns:
            Unit vector (0, 1)
        """
        return cls(0.0, 1.0, scale=scale)
