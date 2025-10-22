#!/usr/bin/env python
"""Standalone test runner for coordinate conversion tests.

This script runs the coordinate conversion tests without relying on pytest,
useful for quick validation and debugging.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

# Import Vector2D directly from models file to avoid circular imports
import importlib.util

spec = importlib.util.spec_from_file_location(
    "models", backend_path / "core" / "models.py"
)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)
Vector2D = models_module.Vector2D

from tests.unit.test_coordinate_conversion import (
    CoordinateConverter,
    CoordinateMetadata,
    CoordinateSpace,
)


def test_basic_operations():
    """Test basic Vector2D operations."""
    print("Testing basic Vector2D operations...")

    # Test creation
    v = Vector2D(100.0, 200.0)
    assert v.x == 100.0
    assert v.y == 200.0
    print("  ✓ Vector2D creation")

    # Test zero
    v_zero = Vector2D.zero()
    assert v_zero.x == 0.0
    assert v_zero.y == 0.0
    print("  ✓ Vector2D zero")

    print("✓ All basic operations passed\n")


def test_pixel_to_normalized():
    """Test pixel to normalized conversions."""
    print("Testing pixel to normalized conversions...")

    resolution = (1920, 1080)

    # Test center
    center = Vector2D(960, 540)
    normalized = CoordinateConverter.pixel_to_normalized(center, resolution)
    assert abs(normalized.x - 0.5) < 1e-10
    assert abs(normalized.y - 0.5) < 1e-10
    print("  ✓ Center conversion")

    # Test origin
    origin = Vector2D(0, 0)
    normalized = CoordinateConverter.pixel_to_normalized(origin, resolution)
    assert abs(normalized.x - 0.0) < 1e-10
    assert abs(normalized.y - 0.0) < 1e-10
    print("  ✓ Origin conversion")

    # Test max
    max_point = Vector2D(1920, 1080)
    normalized = CoordinateConverter.pixel_to_normalized(max_point, resolution)
    assert abs(normalized.x - 1.0) < 1e-10
    assert abs(normalized.y - 1.0) < 1e-10
    print("  ✓ Max point conversion")

    print("✓ All pixel to normalized conversions passed\n")


def test_normalized_to_pixel():
    """Test normalized to pixel conversions."""
    print("Testing normalized to pixel conversions...")

    resolution = (1920, 1080)

    # Test center
    center = Vector2D(0.5, 0.5)
    pixel = CoordinateConverter.normalized_to_pixel(center, resolution)
    assert abs(pixel.x - 960) < 1e-6
    assert abs(pixel.y - 540) < 1e-6
    print("  ✓ Center conversion")

    # Test origin
    origin = Vector2D(0.0, 0.0)
    pixel = CoordinateConverter.normalized_to_pixel(origin, resolution)
    assert abs(pixel.x - 0.0) < 1e-10
    assert abs(pixel.y - 0.0) < 1e-10
    print("  ✓ Origin conversion")

    # Test max
    max_norm = Vector2D(1.0, 1.0)
    pixel = CoordinateConverter.normalized_to_pixel(max_norm, resolution)
    assert abs(pixel.x - 1920) < 1e-6
    assert abs(pixel.y - 1080) < 1e-6
    print("  ✓ Max point conversion")

    print("✓ All normalized to pixel conversions passed\n")


def test_roundtrip_conversions():
    """Test round-trip conversions preserve values."""
    print("Testing round-trip conversions...")

    resolution = (1920, 1080)

    # Pixel -> Normalized -> Pixel
    original = Vector2D(1234.5, 678.9)
    normalized = CoordinateConverter.pixel_to_normalized(original, resolution)
    roundtrip = CoordinateConverter.normalized_to_pixel(normalized, resolution)
    assert abs(roundtrip.x - original.x) < 1e-6
    assert abs(roundtrip.y - original.y) < 1e-6
    print("  ✓ Pixel → Normalized → Pixel")

    # Normalized -> Pixel -> Normalized
    original = Vector2D(0.64, 0.37)
    pixel = CoordinateConverter.normalized_to_pixel(original, resolution)
    roundtrip = CoordinateConverter.pixel_to_normalized(pixel, resolution)
    assert abs(roundtrip.x - original.x) < 1e-10
    assert abs(roundtrip.y - original.y) < 1e-10
    print("  ✓ Normalized → Pixel → Normalized")

    print("✓ All round-trip conversions passed\n")


def test_table_conversions():
    """Test table coordinate conversions."""
    print("Testing table coordinate conversions...")

    resolution = (1920, 1080)
    table_dimensions = (2.54, 1.27)

    # Pixel to table - center
    center_px = Vector2D(960, 540)
    table = CoordinateConverter.pixel_to_table(center_px, resolution, table_dimensions)
    assert abs(table.x - 1.27) < 1e-6
    assert abs(table.y - 0.635) < 1e-6
    print("  ✓ Pixel to table (center)")

    # Table to pixel - center
    center_table = Vector2D(1.27, 0.635)
    pixel = CoordinateConverter.table_to_pixel(
        center_table, resolution, table_dimensions
    )
    assert abs(pixel.x - 960) < 1e-6
    assert abs(pixel.y - 540) < 1e-6
    print("  ✓ Table to pixel (center)")

    # Round-trip
    original = Vector2D(800, 450)
    table = CoordinateConverter.pixel_to_table(original, resolution, table_dimensions)
    roundtrip = CoordinateConverter.table_to_pixel(table, resolution, table_dimensions)
    assert abs(roundtrip.x - original.x) < 1e-6
    assert abs(roundtrip.y - original.y) < 1e-6
    print("  ✓ Pixel → Table → Pixel roundtrip")

    print("✓ All table conversions passed\n")


def test_batch_conversions():
    """Test batch coordinate conversions."""
    print("Testing batch conversions...")

    resolution = (1920, 1080)
    pixel_metadata = CoordinateMetadata(
        space=CoordinateSpace.PIXEL,
        resolution=resolution,
    )

    points = [
        Vector2D(0, 0),
        Vector2D(960, 540),
        Vector2D(1920, 1080),
    ]

    converted = CoordinateConverter.convert_batch(
        points,
        CoordinateSpace.PIXEL,
        CoordinateSpace.NORMALIZED,
        from_metadata=pixel_metadata,
    )

    assert len(converted) == 3
    assert abs(converted[0].x) < 1e-10
    assert abs(converted[0].y) < 1e-10
    assert abs(converted[1].x - 0.5) < 1e-10
    assert abs(converted[1].y - 0.5) < 1e-10
    assert abs(converted[2].x - 1.0) < 1e-10
    assert abs(converted[2].y - 1.0) < 1e-10

    print("  ✓ Batch pixel to normalized")
    print("✓ All batch conversions passed\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...")

    point = Vector2D(100, 200)

    # Test zero resolution
    try:
        CoordinateConverter.pixel_to_normalized(point, (0, 1080))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Resolution dimensions must be positive" in str(e)
        print("  ✓ Zero resolution error")

    # Test negative resolution
    try:
        CoordinateConverter.pixel_to_normalized(point, (-1920, 1080))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Resolution dimensions must be positive" in str(e)
        print("  ✓ Negative resolution error")

    # Test zero table dimensions
    try:
        CoordinateConverter.pixel_to_table(point, (1920, 1080), (0, 1.27))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Table dimensions must be positive" in str(e)
        print("  ✓ Zero table dimensions error")

    print("✓ All error handling tests passed\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Running Coordinate Conversion Tests")
    print("=" * 70)
    print()

    try:
        test_basic_operations()
        test_pixel_to_normalized()
        test_normalized_to_pixel()
        test_roundtrip_conversions()
        test_table_conversions()
        test_batch_conversions()
        test_error_handling()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
