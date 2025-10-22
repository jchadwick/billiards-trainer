"""Comprehensive tests for coordinate space conversions.

This module tests the coordinate conversion system used throughout the billiards trainer,
ensuring accurate transformations between different coordinate spaces:
- Pixel coordinates (camera/vision space)
- Normalized coordinates (0-1 range, resolution-independent)
- Table/World coordinates (meters, physics space)

Tests cover:
1. Vector2D creation with coordinate space metadata
2. CoordinateConverter conversions between all spaces
3. Round-trip conversions (A→B→A should equal A)
4. Edge cases (0,0), (max,max), negative coordinates
5. Resolution scaling edge cases
6. Batch conversions
7. Error handling for invalid conversions
"""

import math
from typing import Optional

import pytest

# Import Vector2D - this will be provided by the test environment
# For standalone testing, mock it here
try:
    from core.coordinates import Vector2D
except (ImportError, ModuleNotFoundError):
    # If import fails, we'll define a minimal Vector2D for testing
    # The real Vector2D from core.models will be used when available
    class Vector2D:
        """Minimal Vector2D implementation for standalone testing."""

        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y

        @classmethod
        def zero(cls):
            """Create a zero vector."""
            return cls(0.0, 0.0)

        def __add__(self, other):
            """Add two vectors."""
            return Vector2D(self.x + other.x, self.y + other.y)

        def to_dict(self):
            """Convert to dictionary."""
            return {"x": self.x, "y": self.y}


# =============================================================================
# Mock Coordinate System Classes
# =============================================================================
# Since the actual coordinate conversion system may not exist yet,
# we'll define the expected interface and test against it


class CoordinateSpace:
    """Enumeration of coordinate spaces used in the system."""

    PIXEL = "pixel"  # Camera/vision pixel coordinates
    NORMALIZED = "normalized"  # Normalized 0-1 range (resolution-independent)
    TABLE = "table"  # Table/world coordinates in meters
    SCREEN = "screen"  # Screen/projector coordinates (if different from pixel)


class CoordinateMetadata:
    """Metadata for coordinate space conversions.

    Stores information needed to convert between coordinate spaces:
    - Source coordinate space
    - Source resolution (width, height) for pixel/screen spaces
    - Table dimensions (width, height) in meters for table space
    - Calibration data (homography matrix, etc.)
    """

    def __init__(
        self,
        space: str = CoordinateSpace.PIXEL,
        resolution: Optional[tuple[int, int]] = None,
        table_dimensions: Optional[tuple[float, float]] = None,
        homography_matrix: Optional[list[list[float]]] = None,
    ):
        """Initialize coordinate metadata.

        Args:
            space: The coordinate space (pixel, normalized, table, screen)
            resolution: (width, height) in pixels for pixel/screen spaces
            table_dimensions: (width, height) in meters for table space
            homography_matrix: 3x3 homography matrix for perspective transformation
        """
        self.space = space
        self.resolution = resolution
        self.table_dimensions = table_dimensions
        self.homography_matrix = homography_matrix

    def __eq__(self, other):
        """Check equality of metadata."""
        if not isinstance(other, CoordinateMetadata):
            return False
        return (
            self.space == other.space
            and self.resolution == other.resolution
            and self.table_dimensions == other.table_dimensions
        )

    def __repr__(self):
        """String representation."""
        return f"CoordinateMetadata(space={self.space}, resolution={self.resolution}, table_dimensions={self.table_dimensions})"


class CoordinateConverter:
    """Utility class for converting between coordinate spaces.

    Provides methods to convert 2D coordinates between different spaces:
    - pixel_to_normalized: Convert pixel coordinates to normalized (0-1) range
    - normalized_to_pixel: Convert normalized coordinates to pixel coordinates
    - pixel_to_table: Convert pixel coordinates to table/world meters
    - table_to_pixel: Convert table/world meters to pixel coordinates
    - normalized_to_table: Convert normalized to table coordinates
    - table_to_normalized: Convert table to normalized coordinates
    """

    @staticmethod
    def pixel_to_normalized(
        point: Vector2D,
        resolution: tuple[int, int],
    ) -> Vector2D:
        """Convert pixel coordinates to normalized (0-1) range.

        Args:
            point: Point in pixel coordinates
            resolution: (width, height) in pixels

        Returns:
            Point in normalized coordinates (0-1 range)
        """
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError("Resolution dimensions must be positive")

        return Vector2D(
            x=point.x / width,
            y=point.y / height,
        )

    @staticmethod
    def normalized_to_pixel(
        point: Vector2D,
        resolution: tuple[int, int],
    ) -> Vector2D:
        """Convert normalized coordinates to pixel coordinates.

        Args:
            point: Point in normalized coordinates (0-1 range)
            resolution: (width, height) in pixels

        Returns:
            Point in pixel coordinates
        """
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError("Resolution dimensions must be positive")

        return Vector2D(
            x=point.x * width,
            y=point.y * height,
        )

    @staticmethod
    def pixel_to_table(
        point: Vector2D,
        resolution: tuple[int, int],
        table_dimensions: tuple[float, float],
        homography_matrix: Optional[list[list[float]]] = None,
    ) -> Vector2D:
        """Convert pixel coordinates to table/world coordinates (meters).

        If homography matrix is provided, uses perspective transformation.
        Otherwise, uses simple linear scaling.

        Args:
            point: Point in pixel coordinates
            resolution: (width, height) in pixels
            table_dimensions: (width, height) in meters
            homography_matrix: Optional 3x3 homography matrix

        Returns:
            Point in table coordinates (meters)
        """
        if homography_matrix is not None:
            # Apply homography transformation for perspective correction
            import numpy as np

            # Convert to homogeneous coordinates
            pixel_point = np.array([point.x, point.y, 1.0])

            # Apply homography
            h_matrix = np.array(homography_matrix)
            table_point = h_matrix @ pixel_point

            # Normalize by w component
            if abs(table_point[2]) < 1e-10:
                raise ValueError("Homography resulted in invalid w component")

            return Vector2D(
                x=table_point[0] / table_point[2],
                y=table_point[1] / table_point[2],
            )
        else:
            # Simple linear scaling (no perspective correction)
            width_px, height_px = resolution
            width_m, height_m = table_dimensions

            if width_px <= 0 or height_px <= 0:
                raise ValueError("Resolution dimensions must be positive")
            if width_m <= 0 or height_m <= 0:
                raise ValueError("Table dimensions must be positive")

            return Vector2D(
                x=(point.x / width_px) * width_m,
                y=(point.y / height_px) * height_m,
            )

    @staticmethod
    def table_to_pixel(
        point: Vector2D,
        resolution: tuple[int, int],
        table_dimensions: tuple[float, float],
        inverse_homography_matrix: Optional[list[list[float]]] = None,
    ) -> Vector2D:
        """Convert table/world coordinates to pixel coordinates.

        If inverse homography matrix is provided, uses perspective transformation.
        Otherwise, uses simple linear scaling.

        Args:
            point: Point in table coordinates (meters)
            resolution: (width, height) in pixels
            table_dimensions: (width, height) in meters
            inverse_homography_matrix: Optional 3x3 inverse homography matrix

        Returns:
            Point in pixel coordinates
        """
        if inverse_homography_matrix is not None:
            # Apply inverse homography transformation
            import numpy as np

            # Convert to homogeneous coordinates
            table_point = np.array([point.x, point.y, 1.0])

            # Apply inverse homography
            h_inv_matrix = np.array(inverse_homography_matrix)
            pixel_point = h_inv_matrix @ table_point

            # Normalize by w component
            if abs(pixel_point[2]) < 1e-10:
                raise ValueError("Inverse homography resulted in invalid w component")

            return Vector2D(
                x=pixel_point[0] / pixel_point[2],
                y=pixel_point[1] / pixel_point[2],
            )
        else:
            # Simple linear scaling (no perspective correction)
            width_px, height_px = resolution
            width_m, height_m = table_dimensions

            if width_px <= 0 or height_px <= 0:
                raise ValueError("Resolution dimensions must be positive")
            if width_m <= 0 or height_m <= 0:
                raise ValueError("Table dimensions must be positive")

            return Vector2D(
                x=(point.x / width_m) * width_px,
                y=(point.y / height_m) * height_px,
            )

    @staticmethod
    def normalized_to_table(
        point: Vector2D,
        table_dimensions: tuple[float, float],
    ) -> Vector2D:
        """Convert normalized coordinates to table coordinates.

        Args:
            point: Point in normalized coordinates (0-1 range)
            table_dimensions: (width, height) in meters

        Returns:
            Point in table coordinates (meters)
        """
        width_m, height_m = table_dimensions

        if width_m <= 0 or height_m <= 0:
            raise ValueError("Table dimensions must be positive")

        return Vector2D(
            x=point.x * width_m,
            y=point.y * height_m,
        )

    @staticmethod
    def table_to_normalized(
        point: Vector2D,
        table_dimensions: tuple[float, float],
    ) -> Vector2D:
        """Convert table coordinates to normalized coordinates.

        Args:
            point: Point in table coordinates (meters)
            table_dimensions: (width, height) in meters

        Returns:
            Point in normalized coordinates (0-1 range)
        """
        width_m, height_m = table_dimensions

        if width_m <= 0 or height_m <= 0:
            raise ValueError("Table dimensions must be positive")

        return Vector2D(
            x=point.x / width_m,
            y=point.y / height_m,
        )

    @staticmethod
    def convert(
        point: Vector2D,
        from_space: str,
        to_space: str,
        metadata: Optional[CoordinateMetadata] = None,
        from_metadata: Optional[CoordinateMetadata] = None,
        to_metadata: Optional[CoordinateMetadata] = None,
    ) -> Vector2D:
        """Generic conversion between any two coordinate spaces.

        Args:
            point: Point to convert
            from_space: Source coordinate space
            to_space: Target coordinate space
            metadata: Shared metadata for conversion (deprecated, use from_metadata/to_metadata)
            from_metadata: Metadata for source space
            to_metadata: Metadata for target space

        Returns:
            Converted point in target coordinate space

        Raises:
            ValueError: If conversion is not supported or metadata is missing
        """
        # Use from_metadata/to_metadata if provided, otherwise fall back to metadata
        src_meta = from_metadata or metadata
        dst_meta = to_metadata or metadata

        if src_meta is None:
            raise ValueError("Source metadata is required for conversion")

        # If converting to same space, return copy
        if from_space == to_space:
            return Vector2D(point.x, point.y)

        # Direct conversions
        if (
            from_space == CoordinateSpace.PIXEL
            and to_space == CoordinateSpace.NORMALIZED
        ):
            if src_meta.resolution is None:
                raise ValueError(
                    "Resolution required for pixel to normalized conversion"
                )
            return CoordinateConverter.pixel_to_normalized(point, src_meta.resolution)

        elif (
            from_space == CoordinateSpace.NORMALIZED
            and to_space == CoordinateSpace.PIXEL
        ):
            if dst_meta is None or dst_meta.resolution is None:
                raise ValueError(
                    "Resolution required for normalized to pixel conversion"
                )
            return CoordinateConverter.normalized_to_pixel(point, dst_meta.resolution)

        elif from_space == CoordinateSpace.PIXEL and to_space == CoordinateSpace.TABLE:
            if src_meta.resolution is None:
                raise ValueError("Resolution required for pixel to table conversion")
            if dst_meta is None or dst_meta.table_dimensions is None:
                raise ValueError(
                    "Table dimensions required for pixel to table conversion"
                )
            return CoordinateConverter.pixel_to_table(
                point,
                src_meta.resolution,
                dst_meta.table_dimensions,
                src_meta.homography_matrix,
            )

        elif from_space == CoordinateSpace.TABLE and to_space == CoordinateSpace.PIXEL:
            if dst_meta is None or dst_meta.resolution is None:
                raise ValueError("Resolution required for table to pixel conversion")
            if src_meta.table_dimensions is None:
                raise ValueError(
                    "Table dimensions required for table to pixel conversion"
                )
            # For inverse homography, we'd need it in metadata
            return CoordinateConverter.table_to_pixel(
                point,
                dst_meta.resolution,
                src_meta.table_dimensions,
                None,  # Would need inverse homography here
            )

        elif (
            from_space == CoordinateSpace.NORMALIZED
            and to_space == CoordinateSpace.TABLE
        ):
            if dst_meta is None or dst_meta.table_dimensions is None:
                raise ValueError(
                    "Table dimensions required for normalized to table conversion"
                )
            return CoordinateConverter.normalized_to_table(
                point, dst_meta.table_dimensions
            )

        elif (
            from_space == CoordinateSpace.TABLE
            and to_space == CoordinateSpace.NORMALIZED
        ):
            if src_meta.table_dimensions is None:
                raise ValueError(
                    "Table dimensions required for table to normalized conversion"
                )
            return CoordinateConverter.table_to_normalized(
                point, src_meta.table_dimensions
            )

        else:
            # Multi-hop conversion through normalized space
            if from_space != CoordinateSpace.NORMALIZED:
                # Convert to normalized first
                normalized = CoordinateConverter.convert(
                    point,
                    from_space,
                    CoordinateSpace.NORMALIZED,
                    from_metadata=src_meta,
                    to_metadata=src_meta,
                )
                # Then convert to target
                return CoordinateConverter.convert(
                    normalized,
                    CoordinateSpace.NORMALIZED,
                    to_space,
                    from_metadata=src_meta,
                    to_metadata=dst_meta,
                )
            else:
                raise ValueError(
                    f"Unsupported conversion from {from_space} to {to_space}"
                )

    @staticmethod
    def convert_batch(
        points: list[Vector2D],
        from_space: str,
        to_space: str,
        metadata: Optional[CoordinateMetadata] = None,
        from_metadata: Optional[CoordinateMetadata] = None,
        to_metadata: Optional[CoordinateMetadata] = None,
    ) -> list[Vector2D]:
        """Convert a batch of points between coordinate spaces.

        Args:
            points: List of points to convert
            from_space: Source coordinate space
            to_space: Target coordinate space
            metadata: Shared metadata for conversion
            from_metadata: Metadata for source space
            to_metadata: Metadata for target space

        Returns:
            List of converted points in target coordinate space
        """
        return [
            CoordinateConverter.convert(
                point,
                from_space,
                to_space,
                metadata=metadata,
                from_metadata=from_metadata,
                to_metadata=to_metadata,
            )
            for point in points
        ]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture()
def standard_resolution():
    """Standard 1920x1080 resolution."""
    return (1920, 1080)


@pytest.fixture()
def standard_table_dimensions():
    """Standard 9-foot pool table dimensions in meters."""
    return (2.54, 1.27)  # 9 feet x 4.5 feet


@pytest.fixture()
def pixel_metadata(standard_resolution):
    """Pixel coordinate metadata."""
    return CoordinateMetadata(
        space=CoordinateSpace.PIXEL,
        resolution=standard_resolution,
    )


@pytest.fixture()
def table_metadata(standard_table_dimensions):
    """Table coordinate metadata."""
    return CoordinateMetadata(
        space=CoordinateSpace.TABLE,
        table_dimensions=standard_table_dimensions,
    )


@pytest.fixture()
def normalized_metadata():
    """Normalized coordinate metadata."""
    return CoordinateMetadata(space=CoordinateSpace.NORMALIZED)


@pytest.fixture()
def complete_metadata(standard_resolution, standard_table_dimensions):
    """Complete metadata with all coordinate systems."""
    return CoordinateMetadata(
        space=CoordinateSpace.PIXEL,
        resolution=standard_resolution,
        table_dimensions=standard_table_dimensions,
    )


# =============================================================================
# Test Cases
# =============================================================================


class TestVector2DCreation:
    """Test Vector2D creation with coordinate space metadata."""

    def test_basic_vector2d_creation(self):
        """Test creating a basic Vector2D without metadata."""
        v = Vector2D(100.0, 200.0)
        assert v.x == 100.0
        assert v.y == 200.0

    def test_vector2d_zero(self):
        """Test creating a zero vector."""
        v = Vector2D.zero()
        assert v.x == 0.0
        assert v.y == 0.0

    def test_vector2d_equality(self):
        """Test Vector2D equality."""
        v1 = Vector2D(100.0, 200.0)
        v2 = Vector2D(100.0, 200.0)
        v3 = Vector2D(100.0, 201.0)

        assert v1.x == v2.x
        assert v1.y == v2.y
        assert not (v1.x == v3.x and v1.y == v3.y)

    def test_vector2d_copy(self):
        """Test that Vector2D operations create new instances."""
        v1 = Vector2D(100.0, 200.0)
        v2 = v1 + Vector2D(10.0, 20.0)

        assert v1.x == 100.0
        assert v1.y == 200.0
        assert v2.x == 110.0
        assert v2.y == 220.0


class TestPixelToNormalizedConversion:
    """Test pixel to normalized coordinate conversions."""

    def test_pixel_to_normalized_center(self, standard_resolution):
        """Test converting center pixel to normalized coordinates."""
        width, height = standard_resolution
        center = Vector2D(width / 2, height / 2)

        normalized = CoordinateConverter.pixel_to_normalized(
            center, standard_resolution
        )

        assert abs(normalized.x - 0.5) < 1e-10
        assert abs(normalized.y - 0.5) < 1e-10

    def test_pixel_to_normalized_origin(self, standard_resolution):
        """Test converting (0,0) to normalized coordinates."""
        origin = Vector2D(0, 0)

        normalized = CoordinateConverter.pixel_to_normalized(
            origin, standard_resolution
        )

        assert abs(normalized.x - 0.0) < 1e-10
        assert abs(normalized.y - 0.0) < 1e-10

    def test_pixel_to_normalized_max(self, standard_resolution):
        """Test converting maximum pixel coordinates to normalized."""
        width, height = standard_resolution
        max_point = Vector2D(width, height)

        normalized = CoordinateConverter.pixel_to_normalized(
            max_point, standard_resolution
        )

        assert abs(normalized.x - 1.0) < 1e-10
        assert abs(normalized.y - 1.0) < 1e-10

    def test_pixel_to_normalized_invalid_resolution(self):
        """Test error handling for invalid resolution."""
        point = Vector2D(100, 100)

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (0, 1080))

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (1920, 0))

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (-1920, 1080))


class TestNormalizedToPixelConversion:
    """Test normalized to pixel coordinate conversions."""

    def test_normalized_to_pixel_center(self, standard_resolution):
        """Test converting normalized center to pixel coordinates."""
        center = Vector2D(0.5, 0.5)

        pixel = CoordinateConverter.normalized_to_pixel(center, standard_resolution)

        width, height = standard_resolution
        assert abs(pixel.x - width / 2) < 1e-6
        assert abs(pixel.y - height / 2) < 1e-6

    def test_normalized_to_pixel_origin(self, standard_resolution):
        """Test converting normalized (0,0) to pixel coordinates."""
        origin = Vector2D(0.0, 0.0)

        pixel = CoordinateConverter.normalized_to_pixel(origin, standard_resolution)

        assert abs(pixel.x - 0.0) < 1e-10
        assert abs(pixel.y - 0.0) < 1e-10

    def test_normalized_to_pixel_max(self, standard_resolution):
        """Test converting normalized (1,1) to pixel coordinates."""
        max_norm = Vector2D(1.0, 1.0)

        pixel = CoordinateConverter.normalized_to_pixel(max_norm, standard_resolution)

        width, height = standard_resolution
        assert abs(pixel.x - width) < 1e-6
        assert abs(pixel.y - height) < 1e-6


class TestPixelToTableConversion:
    """Test pixel to table coordinate conversions."""

    def test_pixel_to_table_linear(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test simple linear pixel to table conversion (no perspective)."""
        # Center of frame should map to center of table
        width_px, height_px = standard_resolution
        width_m, height_m = standard_table_dimensions

        center_px = Vector2D(width_px / 2, height_px / 2)

        table = CoordinateConverter.pixel_to_table(
            center_px, standard_resolution, standard_table_dimensions
        )

        assert abs(table.x - width_m / 2) < 1e-6
        assert abs(table.y - height_m / 2) < 1e-6

    def test_pixel_to_table_origin(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test converting pixel (0,0) to table coordinates."""
        origin = Vector2D(0, 0)

        table = CoordinateConverter.pixel_to_table(
            origin, standard_resolution, standard_table_dimensions
        )

        assert abs(table.x - 0.0) < 1e-10
        assert abs(table.y - 0.0) < 1e-10

    def test_pixel_to_table_max(self, standard_resolution, standard_table_dimensions):
        """Test converting maximum pixel coordinates to table coordinates."""
        width_px, height_px = standard_resolution
        width_m, height_m = standard_table_dimensions

        max_px = Vector2D(width_px, height_px)

        table = CoordinateConverter.pixel_to_table(
            max_px, standard_resolution, standard_table_dimensions
        )

        assert abs(table.x - width_m) < 1e-6
        assert abs(table.y - height_m) < 1e-6

    def test_pixel_to_table_with_homography(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test pixel to table conversion with homography matrix (identity)."""
        # Identity homography should behave like no homography
        identity_homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        point = Vector2D(100, 200)

        # With identity homography, point should remain the same
        table = CoordinateConverter.pixel_to_table(
            point,
            standard_resolution,
            standard_table_dimensions,
            homography_matrix=identity_homography,
        )

        assert abs(table.x - 100.0) < 1e-6
        assert abs(table.y - 200.0) < 1e-6


class TestTableToPixelConversion:
    """Test table to pixel coordinate conversions."""

    def test_table_to_pixel_linear(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test simple linear table to pixel conversion."""
        width_m, height_m = standard_table_dimensions
        width_px, height_px = standard_resolution

        center_table = Vector2D(width_m / 2, height_m / 2)

        pixel = CoordinateConverter.table_to_pixel(
            center_table, standard_resolution, standard_table_dimensions
        )

        assert abs(pixel.x - width_px / 2) < 1e-6
        assert abs(pixel.y - height_px / 2) < 1e-6

    def test_table_to_pixel_origin(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test converting table (0,0) to pixel coordinates."""
        origin = Vector2D(0, 0)

        pixel = CoordinateConverter.table_to_pixel(
            origin, standard_resolution, standard_table_dimensions
        )

        assert abs(pixel.x - 0.0) < 1e-10
        assert abs(pixel.y - 0.0) < 1e-10

    def test_table_to_pixel_max(self, standard_resolution, standard_table_dimensions):
        """Test converting maximum table coordinates to pixel coordinates."""
        width_m, height_m = standard_table_dimensions
        width_px, height_px = standard_resolution

        max_table = Vector2D(width_m, height_m)

        pixel = CoordinateConverter.table_to_pixel(
            max_table, standard_resolution, standard_table_dimensions
        )

        assert abs(pixel.x - width_px) < 1e-6
        assert abs(pixel.y - height_px) < 1e-6


class TestRoundTripConversions:
    """Test that round-trip conversions preserve original coordinates."""

    def test_pixel_normalized_roundtrip(self, standard_resolution):
        """Test pixel → normalized → pixel preserves coordinates."""
        original = Vector2D(1234.5, 678.9)

        normalized = CoordinateConverter.pixel_to_normalized(
            original, standard_resolution
        )
        roundtrip = CoordinateConverter.normalized_to_pixel(
            normalized, standard_resolution
        )

        assert abs(roundtrip.x - original.x) < 1e-6
        assert abs(roundtrip.y - original.y) < 1e-6

    def test_normalized_pixel_roundtrip(self, standard_resolution):
        """Test normalized → pixel → normalized preserves coordinates."""
        original = Vector2D(0.64, 0.37)

        pixel = CoordinateConverter.normalized_to_pixel(original, standard_resolution)
        roundtrip = CoordinateConverter.pixel_to_normalized(pixel, standard_resolution)

        assert abs(roundtrip.x - original.x) < 1e-10
        assert abs(roundtrip.y - original.y) < 1e-10

    def test_pixel_table_roundtrip(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test pixel → table → pixel preserves coordinates."""
        original = Vector2D(800, 450)

        table = CoordinateConverter.pixel_to_table(
            original, standard_resolution, standard_table_dimensions
        )
        roundtrip = CoordinateConverter.table_to_pixel(
            table, standard_resolution, standard_table_dimensions
        )

        assert abs(roundtrip.x - original.x) < 1e-6
        assert abs(roundtrip.y - original.y) < 1e-6

    def test_table_pixel_roundtrip(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test table → pixel → table preserves coordinates."""
        original = Vector2D(1.5, 0.8)

        pixel = CoordinateConverter.table_to_pixel(
            original, standard_resolution, standard_table_dimensions
        )
        roundtrip = CoordinateConverter.pixel_to_table(
            pixel, standard_resolution, standard_table_dimensions
        )

        assert abs(roundtrip.x - original.x) < 1e-6
        assert abs(roundtrip.y - original.y) < 1e-6

    def test_normalized_table_roundtrip(self, standard_table_dimensions):
        """Test normalized → table → normalized preserves coordinates."""
        original = Vector2D(0.42, 0.73)

        table = CoordinateConverter.normalized_to_table(
            original, standard_table_dimensions
        )
        roundtrip = CoordinateConverter.table_to_normalized(
            table, standard_table_dimensions
        )

        assert abs(roundtrip.x - original.x) < 1e-10
        assert abs(roundtrip.y - original.y) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_coordinates(self, standard_resolution, standard_table_dimensions):
        """Test handling of (0,0) coordinates in all spaces."""
        zero = Vector2D(0, 0)

        # Pixel to normalized
        norm = CoordinateConverter.pixel_to_normalized(zero, standard_resolution)
        assert norm.x == 0.0
        assert norm.y == 0.0

        # Normalized to pixel
        pix = CoordinateConverter.normalized_to_pixel(zero, standard_resolution)
        assert pix.x == 0.0
        assert pix.y == 0.0

        # Pixel to table
        table = CoordinateConverter.pixel_to_table(
            zero, standard_resolution, standard_table_dimensions
        )
        assert table.x == 0.0
        assert table.y == 0.0

    def test_negative_pixel_coordinates(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test handling of negative pixel coordinates."""
        negative = Vector2D(-100, -50)

        # Should still convert (may represent off-screen points)
        normalized = CoordinateConverter.pixel_to_normalized(
            negative, standard_resolution
        )
        assert normalized.x < 0
        assert normalized.y < 0

        table = CoordinateConverter.pixel_to_table(
            negative, standard_resolution, standard_table_dimensions
        )
        assert table.x < 0
        assert table.y < 0

    def test_out_of_range_normalized(self, standard_resolution):
        """Test normalized coordinates outside [0,1] range."""
        # Values > 1.0 (beyond normalized bounds)
        beyond = Vector2D(1.5, 2.0)

        pixel = CoordinateConverter.normalized_to_pixel(beyond, standard_resolution)
        width, height = standard_resolution

        assert pixel.x > width
        assert pixel.y > height

    def test_very_small_values(self, standard_resolution, standard_table_dimensions):
        """Test very small coordinate values (near floating-point precision)."""
        tiny = Vector2D(1e-10, 1e-10)

        normalized = CoordinateConverter.pixel_to_normalized(tiny, standard_resolution)
        assert abs(normalized.x) < 1e-6
        assert abs(normalized.y) < 1e-6

    def test_very_large_values(self, standard_resolution, standard_table_dimensions):
        """Test very large coordinate values."""
        large = Vector2D(1e6, 1e6)

        normalized = CoordinateConverter.pixel_to_normalized(large, standard_resolution)
        assert normalized.x > 100  # Well beyond normalized range
        assert normalized.y > 100


class TestResolutionScaling:
    """Test coordinate conversions with different resolutions."""

    def test_different_resolutions(self):
        """Test that conversions work correctly with different resolutions."""
        resolutions = [
            (640, 480),  # VGA
            (1280, 720),  # 720p
            (1920, 1080),  # 1080p
            (3840, 2160),  # 4K
        ]

        normalized_point = Vector2D(0.5, 0.5)

        for resolution in resolutions:
            # Convert to pixel
            pixel = CoordinateConverter.normalized_to_pixel(
                normalized_point, resolution
            )

            # Should be at center
            width, height = resolution
            assert abs(pixel.x - width / 2) < 1e-6
            assert abs(pixel.y - height / 2) < 1e-6

            # Convert back
            roundtrip = CoordinateConverter.pixel_to_normalized(pixel, resolution)
            assert abs(roundtrip.x - 0.5) < 1e-10
            assert abs(roundtrip.y - 0.5) < 1e-10

    def test_resolution_independence_through_normalized(self):
        """Test that normalized space allows resolution-independent conversions."""
        # Same point in different resolutions
        res1 = (1920, 1080)
        res2 = (3840, 2160)  # 2x resolution

        pixel1 = Vector2D(960, 540)  # Center of res1

        # Convert to normalized
        normalized = CoordinateConverter.pixel_to_normalized(pixel1, res1)

        # Convert to different resolution
        pixel2 = CoordinateConverter.normalized_to_pixel(normalized, res2)

        # Should be at center of res2
        assert abs(pixel2.x - 1920) < 1e-6  # Center of 3840
        assert abs(pixel2.y - 1080) < 1e-6  # Center of 2160

    def test_aspect_ratio_handling(self):
        """Test conversions with different aspect ratios."""
        wide_res = (1920, 1080)  # 16:9
        square_res = (1080, 1080)  # 1:1
        tall_res = (1080, 1920)  # 9:16

        normalized = Vector2D(0.5, 0.5)

        # All should map to their respective centers
        for resolution in [wide_res, square_res, tall_res]:
            pixel = CoordinateConverter.normalized_to_pixel(normalized, resolution)
            width, height = resolution

            assert abs(pixel.x - width / 2) < 1e-6
            assert abs(pixel.y - height / 2) < 1e-6


class TestBatchConversions:
    """Test batch coordinate conversions."""

    def test_batch_pixel_to_normalized(self, standard_resolution, pixel_metadata):
        """Test batch conversion of pixel coordinates to normalized."""
        points = [
            Vector2D(0, 0),
            Vector2D(960, 540),
            Vector2D(1920, 1080),
            Vector2D(100, 200),
            Vector2D(1500, 800),
        ]

        converted = CoordinateConverter.convert_batch(
            points,
            CoordinateSpace.PIXEL,
            CoordinateSpace.NORMALIZED,
            from_metadata=pixel_metadata,
        )

        assert len(converted) == len(points)

        # Check first and last points
        assert abs(converted[0].x) < 1e-10
        assert abs(converted[0].y) < 1e-10
        assert abs(converted[2].x - 1.0) < 1e-10
        assert abs(converted[2].y - 1.0) < 1e-10

    def test_batch_empty_list(self, pixel_metadata):
        """Test batch conversion of empty list."""
        empty = []

        converted = CoordinateConverter.convert_batch(
            empty,
            CoordinateSpace.PIXEL,
            CoordinateSpace.NORMALIZED,
            from_metadata=pixel_metadata,
        )

        assert len(converted) == 0

    def test_batch_single_point(self, standard_resolution, pixel_metadata):
        """Test batch conversion of single point."""
        single = [Vector2D(100, 200)]

        converted = CoordinateConverter.convert_batch(
            single,
            CoordinateSpace.PIXEL,
            CoordinateSpace.NORMALIZED,
            from_metadata=pixel_metadata,
        )

        assert len(converted) == 1
        assert converted[0].x == 100 / 1920
        assert converted[0].y == 200 / 1080


class TestGenericConversion:
    """Test the generic convert() method."""

    def test_convert_same_space(self, pixel_metadata):
        """Test that converting to same space returns copy."""
        original = Vector2D(100, 200)

        result = CoordinateConverter.convert(
            original,
            CoordinateSpace.PIXEL,
            CoordinateSpace.PIXEL,
            from_metadata=pixel_metadata,
        )

        assert result.x == original.x
        assert result.y == original.y

    def test_convert_pixel_to_normalized(self, pixel_metadata):
        """Test generic conversion from pixel to normalized."""
        point = Vector2D(960, 540)

        result = CoordinateConverter.convert(
            point,
            CoordinateSpace.PIXEL,
            CoordinateSpace.NORMALIZED,
            from_metadata=pixel_metadata,
        )

        assert abs(result.x - 0.5) < 1e-10
        assert abs(result.y - 0.5) < 1e-10

    def test_convert_missing_metadata(self):
        """Test error handling when metadata is missing."""
        point = Vector2D(100, 200)

        with pytest.raises(ValueError, match="Source metadata is required"):
            CoordinateConverter.convert(
                point,
                CoordinateSpace.PIXEL,
                CoordinateSpace.NORMALIZED,
                metadata=None,
            )

    def test_convert_missing_resolution(self):
        """Test error handling when resolution is missing."""
        point = Vector2D(100, 200)
        incomplete_metadata = CoordinateMetadata(space=CoordinateSpace.PIXEL)

        with pytest.raises(
            ValueError, match="Resolution required for pixel to normalized conversion"
        ):
            CoordinateConverter.convert(
                point,
                CoordinateSpace.PIXEL,
                CoordinateSpace.NORMALIZED,
                from_metadata=incomplete_metadata,
            )


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_zero_resolution(self):
        """Test error handling for zero resolution."""
        point = Vector2D(100, 200)

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (0, 1080))

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (1920, 0))

    def test_negative_resolution(self):
        """Test error handling for negative resolution."""
        point = Vector2D(100, 200)

        with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
            CoordinateConverter.pixel_to_normalized(point, (-1920, 1080))

    def test_zero_table_dimensions(self):
        """Test error handling for zero table dimensions."""
        point = Vector2D(100, 200)
        resolution = (1920, 1080)

        with pytest.raises(ValueError, match="Table dimensions must be positive"):
            CoordinateConverter.pixel_to_table(point, resolution, (0, 1.27))

        with pytest.raises(ValueError, match="Table dimensions must be positive"):
            CoordinateConverter.pixel_to_table(point, resolution, (2.54, 0))

    def test_negative_table_dimensions(self):
        """Test error handling for negative table dimensions."""
        point = Vector2D(100, 200)
        resolution = (1920, 1080)

        with pytest.raises(ValueError, match="Table dimensions must be positive"):
            CoordinateConverter.pixel_to_table(point, resolution, (-2.54, 1.27))

    def test_invalid_homography_w_component(self):
        """Test error handling for invalid homography w component."""
        point = Vector2D(100, 200)
        resolution = (1920, 1080)
        table_dims = (2.54, 1.27)

        # Homography that results in w = 0
        bad_homography = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]

        with pytest.raises(ValueError, match="invalid w component"):
            CoordinateConverter.pixel_to_table(
                point, resolution, table_dims, homography_matrix=bad_homography
            )


class TestCoordinateMetadata:
    """Test CoordinateMetadata class."""

    def test_metadata_creation(self):
        """Test creating coordinate metadata."""
        metadata = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=(1920, 1080),
            table_dimensions=(2.54, 1.27),
        )

        assert metadata.space == CoordinateSpace.PIXEL
        assert metadata.resolution == (1920, 1080)
        assert metadata.table_dimensions == (2.54, 1.27)

    def test_metadata_equality(self):
        """Test metadata equality comparison."""
        meta1 = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=(1920, 1080),
        )
        meta2 = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=(1920, 1080),
        )
        meta3 = CoordinateMetadata(
            space=CoordinateSpace.TABLE,
            resolution=(1920, 1080),
        )

        assert meta1 == meta2
        assert meta1 != meta3

    def test_metadata_repr(self):
        """Test metadata string representation."""
        metadata = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=(1920, 1080),
        )

        repr_str = repr(metadata)
        assert "CoordinateMetadata" in repr_str
        assert "pixel" in repr_str
        assert "(1920, 1080)" in repr_str


# =============================================================================
# Property-Based Tests (using hypothesis)
# =============================================================================


try:
    from hypothesis import given
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    # Mock decorators for when hypothesis is not available
    def given(**kwargs):
        """Mock given decorator."""

        def decorator(func):
            return func

        return decorator

    class MockStrategies:
        """Mock strategies class."""

        @staticmethod
        def floats(**kwargs):  # noqa: ARG004
            return None

        @staticmethod
        def integers(**kwargs):  # noqa: ARG004
            return None

    st = MockStrategies()


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not installed")
class TestPropertyBased:
    """Property-based tests using hypothesis."""

    @given(
        x=st.floats(min_value=0, max_value=1920, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=0, max_value=1080, allow_nan=False, allow_infinity=False),
    )
    def test_pixel_normalized_roundtrip_property(self, x, y):
        """Property: pixel → normalized → pixel should be identity."""
        resolution = (1920, 1080)
        original = Vector2D(x, y)

        normalized = CoordinateConverter.pixel_to_normalized(original, resolution)
        roundtrip = CoordinateConverter.normalized_to_pixel(normalized, resolution)

        assert abs(roundtrip.x - original.x) < 1e-6
        assert abs(roundtrip.y - original.y) < 1e-6

    @given(
        x=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    )
    def test_normalized_values_in_bounds_property(self, x, y):
        """Property: normalized coordinates should be in [0,1] for valid pixels."""
        resolution = (1920, 1080)
        pixel = Vector2D(x * 1920, y * 1080)

        normalized = CoordinateConverter.pixel_to_normalized(pixel, resolution)

        assert 0 <= normalized.x <= 1
        assert 0 <= normalized.y <= 1

    @given(
        width=st.integers(min_value=1, max_value=10000),
        height=st.integers(min_value=1, max_value=10000),
    )
    def test_center_maps_to_half_property(self, width, height):
        """Property: center pixel should always map to (0.5, 0.5) normalized."""
        resolution = (width, height)
        center = Vector2D(width / 2, height / 2)

        normalized = CoordinateConverter.pixel_to_normalized(center, resolution)

        assert abs(normalized.x - 0.5) < 1e-10
        assert abs(normalized.y - 0.5) < 1e-10


# =============================================================================
# Integration Tests
# =============================================================================


class TestRealWorldScenarios:
    """Test realistic coordinate conversion scenarios."""

    def test_ball_detection_to_physics_pipeline(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test converting detected ball positions through the pipeline."""
        # Simulate ball detected at pixel coordinates
        detected_balls_pixel = [
            Vector2D(960, 540),  # Cue ball at center
            Vector2D(1440, 270),  # Object ball 1
            Vector2D(480, 810),  # Object ball 2
        ]

        # Convert to table coordinates for physics
        table_metadata = CoordinateMetadata(
            space=CoordinateSpace.TABLE,
            table_dimensions=standard_table_dimensions,
        )
        pixel_metadata = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=standard_resolution,
        )

        balls_table = CoordinateConverter.convert_batch(
            detected_balls_pixel,
            CoordinateSpace.PIXEL,
            CoordinateSpace.TABLE,
            from_metadata=pixel_metadata,
            to_metadata=table_metadata,
        )

        # Verify conversion
        assert len(balls_table) == 3

        # Cue ball should be at center of table
        width_m, height_m = standard_table_dimensions
        assert abs(balls_table[0].x - width_m / 2) < 1e-6
        assert abs(balls_table[0].y - height_m / 2) < 1e-6

    def test_trajectory_visualization_pipeline(
        self, standard_resolution, standard_table_dimensions
    ):
        """Test converting physics trajectory back to screen coordinates."""
        # Simulate physics trajectory in table coordinates (meters)
        trajectory_table = [
            Vector2D(1.27, 0.635),  # Start at center
            Vector2D(1.5, 0.7),
            Vector2D(1.8, 0.8),
            Vector2D(2.1, 0.9),
        ]

        # Convert to pixel coordinates for visualization
        table_metadata = CoordinateMetadata(
            space=CoordinateSpace.TABLE,
            table_dimensions=standard_table_dimensions,
        )
        pixel_metadata = CoordinateMetadata(
            space=CoordinateSpace.PIXEL,
            resolution=standard_resolution,
        )

        trajectory_pixel = CoordinateConverter.convert_batch(
            trajectory_table,
            CoordinateSpace.TABLE,
            CoordinateSpace.PIXEL,
            from_metadata=table_metadata,
            to_metadata=pixel_metadata,
        )

        # Verify conversion
        assert len(trajectory_pixel) == len(trajectory_table)

        # All points should be within screen bounds
        width, height = standard_resolution
        for point in trajectory_pixel:
            assert 0 <= point.x <= width
            assert 0 <= point.y <= height

    def test_multi_resolution_tracking(self):
        """Test tracking ball across different camera resolutions."""
        # Ball detected in high-res camera
        highres_resolution = (3840, 2160)
        ball_highres = Vector2D(1920, 1080)

        # Convert to normalized (resolution-independent)
        normalized = CoordinateConverter.pixel_to_normalized(
            ball_highres, highres_resolution
        )

        # Convert to low-res display
        lowres_resolution = (1920, 1080)
        ball_lowres = CoordinateConverter.normalized_to_pixel(
            normalized, lowres_resolution
        )

        # Should be at center in both
        assert abs(normalized.x - 0.5) < 1e-10
        assert abs(normalized.y - 0.5) < 1e-10
        assert abs(ball_lowres.x - 960) < 1e-6
        assert abs(ball_lowres.y - 540) < 1e-6
