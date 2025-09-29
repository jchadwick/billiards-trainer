#!/usr/bin/env python3
"""Test script for coordinate transformation utilities.

This script tests the coordinate transformation system including camera calibration,
perspective correction, and coordinate system conversions.
"""

import math
import sys
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.vision.utils.transforms import (
    CoordinateTransformer,
    Point2D,
    Point3D,
    TransformationMatrix,
    CameraCalibration,
    CoordinateSystem,
    create_perspective_matrix,
    apply_perspective_correction,
    normalize_table_coordinates,
    denormalize_table_coordinates,
)


def test_point_classes():
    """Test Point2D and Point3D classes."""
    print("🧪 Testing Point Classes")
    print("=" * 50)

    # Test Point2D
    p1 = Point2D(10.0, 20.0)
    p2 = Point2D(5.0, 15.0)

    # Test basic operations
    p_sum = p1 + p2
    p_diff = p1 - p2
    p_scaled = p1 * 2.0

    print(f"Point2D operations:")
    print(f"  p1: {p1.to_tuple()}")
    print(f"  p2: {p2.to_tuple()}")
    print(f"  p1 + p2: {p_sum.to_tuple()}")
    print(f"  p1 - p2: {p_diff.to_tuple()}")
    print(f"  p1 * 2: {p_scaled.to_tuple()}")
    print(f"  Distance p1 to p2: {p1.distance_to(p2):.2f}")

    # Test Point3D
    p3d1 = Point3D(1.0, 2.0, 3.0)
    p2d_from_3d = p3d1.to_2d()

    print(f"\nPoint3D operations:")
    print(f"  p3d1: {p3d1.to_tuple()}")
    print(f"  p3d1 to 2D: {p2d_from_3d.to_tuple()}")

    # Verify calculations
    expected_sum = (15.0, 35.0)
    expected_diff = (5.0, 5.0)
    expected_scaled = (20.0, 40.0)
    expected_distance = math.sqrt(5**2 + 5**2)

    success = (
        abs(p_sum.x - expected_sum[0]) < 0.01 and
        abs(p_sum.y - expected_sum[1]) < 0.01 and
        abs(p_diff.x - expected_diff[0]) < 0.01 and
        abs(p_diff.y - expected_diff[1]) < 0.01 and
        abs(p_scaled.x - expected_scaled[0]) < 0.01 and
        abs(p_scaled.y - expected_scaled[1]) < 0.01 and
        abs(p1.distance_to(p2) - expected_distance) < 0.01
    )

    if success:
        print("✅ Point classes working correctly")
    else:
        print("❌ Point class calculations failed")

    return success


def test_camera_calibration():
    """Test camera calibration data structure."""
    print("\n📷 Testing Camera Calibration")
    print("=" * 50)

    # Create test camera calibration
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    distortion_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])
    image_size = (640, 480)

    calibration = CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        image_size=image_size,
        reprojection_error=0.5
    )

    # Test properties
    fx, fy = calibration.focal_length
    cx, cy = calibration.principal_point

    print(f"Camera calibration:")
    print(f"  Image size: {calibration.image_size}")
    print(f"  Focal length: fx={fx}, fy={fy}")
    print(f"  Principal point: cx={cx}, cy={cy}")
    print(f"  Reprojection error: {calibration.reprojection_error}")

    # Verify values
    success = (
        fx == 800.0 and fy == 800.0 and
        cx == 320.0 and cy == 240.0 and
        calibration.image_size == (640, 480)
    )

    if success:
        print("✅ Camera calibration working correctly")
    else:
        print("❌ Camera calibration failed")

    return success


def test_transformation_matrix():
    """Test transformation matrix class."""
    print("\n🔄 Testing Transformation Matrix")
    print("=" * 50)

    # Create test transformation matrix (2D rotation by 45 degrees)
    angle = math.pi / 4  # 45 degrees
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    transform = TransformationMatrix(
        matrix=rotation_matrix,
        source_system=CoordinateSystem.CAMERA,
        target_system=CoordinateSystem.TABLE,
        is_homogeneous=True
    )

    print(f"Transformation matrix:")
    print(f"  Source: {transform.source_system.value}")
    print(f"  Target: {transform.target_system.value}")
    print(f"  Matrix shape: {transform.matrix.shape}")

    # Test inverse
    inverse_transform = transform.inverse

    print(f"  Inverse target: {inverse_transform.target_system.value}")
    print(f"  Inverse source: {inverse_transform.source_system.value}")

    # Test that forward and inverse cancel out
    identity = transform.matrix @ inverse_transform.matrix
    is_identity = np.allclose(identity, np.eye(3), atol=1e-10)

    if is_identity:
        print("✅ Transformation matrix and inverse working correctly")
    else:
        print("❌ Transformation matrix inverse failed")

    return is_identity


def test_coordinate_transformer_basic():
    """Test basic coordinate transformer functionality."""
    print("\n🗺️  Testing Coordinate Transformer - Basic")
    print("=" * 50)

    transformer = CoordinateTransformer()

    # Create a simple scaling transformation (camera to table)
    scale_matrix = np.array([
        [2.0, 0.0, 0.0],  # Scale x by 2
        [0.0, 2.0, 0.0],  # Scale y by 2
        [0.0, 0.0, 1.0]   # Homogeneous
    ])

    transform = TransformationMatrix(
        matrix=scale_matrix,
        source_system=CoordinateSystem.CAMERA,
        target_system=CoordinateSystem.TABLE,
        is_homogeneous=True
    )

    transformer.add_transformation(transform)

    # Test transformation
    test_point = Point2D(10.0, 20.0)
    transformed = transformer.transform_point(
        test_point,
        CoordinateSystem.CAMERA,
        CoordinateSystem.TABLE
    )

    print(f"Original point: {test_point.to_tuple()}")
    print(f"Transformed point: {transformed.to_tuple()}")

    # Should be scaled by 2
    expected = (20.0, 40.0)
    success = (
        abs(transformed.x - expected[0]) < 0.01 and
        abs(transformed.y - expected[1]) < 0.01
    )

    if success:
        print("✅ Basic coordinate transformation working correctly")
    else:
        print("❌ Basic coordinate transformation failed")

    return success


def test_perspective_transformation():
    """Test perspective transformation functionality."""
    print("\n📐 Testing Perspective Transformation")
    print("=" * 50)

    transformer = CoordinateTransformer()

    # Define a simple perspective transformation
    # Source: unit square corners
    source_points = [
        Point2D(0, 0),    # Top-left
        Point2D(100, 0),  # Top-right
        Point2D(100, 100), # Bottom-right
        Point2D(0, 100)   # Bottom-left
    ]

    # Target: slightly skewed quadrilateral
    target_points = [
        Point2D(10, 10),
        Point2D(110, 5),
        Point2D(105, 105),
        Point2D(5, 110)
    ]

    # Compute perspective transformation
    homography = transformer.compute_perspective_transform(source_points, target_points)

    print(f"Perspective transformation computed:")
    print(f"  Matrix shape: {homography.shape}")
    print(f"  Matrix determinant: {np.linalg.det(homography):.6f}")

    # Test transformation
    transformed_points = transformer.apply_perspective_transform(source_points, homography)

    print(f"Transformation results:")
    for i, (src, dst, result) in enumerate(zip(source_points, target_points, transformed_points)):
        error = Point2D(result.x - dst.x, result.y - dst.y)
        error_magnitude = math.sqrt(error.x**2 + error.y**2)
        print(f"  Point {i}: {src.to_tuple()} -> {result.to_tuple()} (expected {dst.to_tuple()}, error: {error_magnitude:.3f})")

    # Check accuracy - transformed points should be close to target points
    max_error = 0.0
    for result, target in zip(transformed_points, target_points):
        error = math.sqrt((result.x - target.x)**2 + (result.y - target.y)**2)
        max_error = max(max_error, error)

    success = max_error < 1.0  # Allow up to 1 pixel error

    if success:
        print("✅ Perspective transformation working correctly")
    else:
        print(f"❌ Perspective transformation failed (max error: {max_error:.3f})")

    return success


def test_coordinate_normalization():
    """Test coordinate normalization functions."""
    print("\n📏 Testing Coordinate Normalization")
    print("=" * 50)

    transformer = CoordinateTransformer()

    # Test pixel coordinate normalization
    test_point = Point2D(320, 240)  # Center of 640x480 image
    image_size = (640, 480)

    normalized = transformer.normalize_coordinates(test_point, image_size)
    denormalized = transformer.denormalize_coordinates(normalized, image_size)

    print(f"Pixel coordinate normalization:")
    print(f"  Original: {test_point.to_tuple()}")
    print(f"  Normalized: {normalized.to_tuple()}")
    print(f"  Denormalized: {denormalized.to_tuple()}")

    # Test table coordinate normalization
    table_points = [(500, 250), (1000, 750)]  # Points on 2000x1000mm table
    table_size = (2000, 1000)

    normalized_table = normalize_table_coordinates(table_points, table_size)
    denormalized_table = denormalize_table_coordinates(normalized_table, table_size)

    print(f"Table coordinate normalization:")
    print(f"  Original: {table_points}")
    print(f"  Normalized: {normalized_table}")
    print(f"  Denormalized: {denormalized_table}")

    # Check round-trip accuracy
    pixel_error = abs(denormalized.x - test_point.x) + abs(denormalized.y - test_point.y)
    table_error = sum(abs(d[0] - o[0]) + abs(d[1] - o[1]) for d, o in zip(denormalized_table, table_points))

    success = pixel_error < 0.01 and table_error < 0.01

    if success:
        print("✅ Coordinate normalization working correctly")
    else:
        print(f"❌ Coordinate normalization failed (pixel error: {pixel_error}, table error: {table_error})")

    return success


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🛠️  Testing Convenience Functions")
    print("=" * 50)

    # Test perspective matrix creation
    source_corners = [(0, 0), (100, 0), (100, 100), (0, 100)]
    target_corners = [(10, 10), (110, 5), (105, 105), (5, 110)]

    matrix = create_perspective_matrix(source_corners, target_corners)
    print(f"Perspective matrix created: {matrix.shape}")

    # Test perspective correction
    test_points = [(25, 25), (75, 25), (75, 75), (25, 75)]
    corrected_points = apply_perspective_correction(test_points, source_corners, target_corners)

    print(f"Perspective correction:")
    for orig, corrected in zip(test_points, corrected_points):
        print(f"  {orig} -> {corrected}")

    # Basic sanity check - points should be transformed
    all_different = all(
        abs(orig[0] - corr[0]) > 0.1 or abs(orig[1] - corr[1]) > 0.1
        for orig, corr in zip(test_points, corrected_points)
    )

    if all_different and len(corrected_points) == len(test_points):
        print("✅ Convenience functions working correctly")
        return True
    else:
        print("❌ Convenience functions failed")
        return False


def test_table_calibration():
    """Test table calibration functionality."""
    print("\n🎱 Testing Table Calibration")
    print("=" * 50)

    transformer = CoordinateTransformer()

    # Simulate table corner detection in camera image
    camera_corners = [
        Point2D(100, 80),   # Top-left
        Point2D(540, 85),   # Top-right
        Point2D(530, 400),  # Bottom-right
        Point2D(110, 395)   # Bottom-left
    ]

    # Real table corners in world coordinates (mm)
    world_corners = [
        Point2D(0, 0),        # Top-left
        Point2D(2540, 0),     # Top-right
        Point2D(2540, 1270),  # Bottom-right
        Point2D(0, 1270)      # Bottom-left
    ]

    table_size = (2540, 1270)  # Standard 9-foot pool table

    # Set table calibration
    transformer.set_table_calibration(camera_corners, world_corners, table_size)

    print(f"Table calibration set:")
    print(f"  Table size: {table_size}mm")
    print(f"  Camera corners: {[p.to_tuple() for p in camera_corners]}")
    print(f"  World corners: {[p.to_tuple() for p in world_corners]}")

    # Test transformation
    test_camera_point = Point2D(320, 240)  # Center of image
    try:
        world_point = transformer.transform_point(
            test_camera_point,
            CoordinateSystem.CAMERA,
            CoordinateSystem.TABLE
        )

        print(f"Camera to table transformation:")
        print(f"  Camera point: {test_camera_point.to_tuple()}")
        print(f"  Table point: {world_point.to_tuple()}")

        # Should be somewhere in the middle of the table
        success = (
            0 <= world_point.x <= table_size[0] and
            0 <= world_point.y <= table_size[1]
        )

        if success:
            print("✅ Table calibration working correctly")
        else:
            print("❌ Table calibration transformation out of bounds")

    except Exception as e:
        print(f"❌ Table calibration failed: {e}")
        success = False

    return success


def test_transformation_availability():
    """Test transformation availability checking."""
    print("\n🔍 Testing Transformation Availability")
    print("=" * 50)

    transformer = CoordinateTransformer()

    # Initially no transformations available
    available_before = transformer.is_transformation_available(
        CoordinateSystem.CAMERA,
        CoordinateSystem.TABLE
    )

    # Add a transformation
    identity_matrix = np.eye(3)
    transform = TransformationMatrix(
        matrix=identity_matrix,
        source_system=CoordinateSystem.CAMERA,
        target_system=CoordinateSystem.TABLE,
        is_homogeneous=True
    )
    transformer.add_transformation(transform)

    # Now transformation should be available
    available_after = transformer.is_transformation_available(
        CoordinateSystem.CAMERA,
        CoordinateSystem.TABLE
    )

    # Inverse should also be available
    available_inverse = transformer.is_transformation_available(
        CoordinateSystem.TABLE,
        CoordinateSystem.CAMERA
    )

    print(f"Transformation availability:")
    print(f"  Before adding transform: {available_before}")
    print(f"  After adding transform: {available_after}")
    print(f"  Inverse available: {available_inverse}")

    # Get list of available transformations
    available_transforms = transformer.get_available_transformations()
    print(f"  Available transformations: {len(available_transforms)}")

    success = not available_before and available_after and available_inverse

    if success:
        print("✅ Transformation availability checking working correctly")
    else:
        print("❌ Transformation availability checking failed")

    return success


def main():
    """Run all coordinate transformation tests."""
    print("🚀 Starting Coordinate Transformation Tests")
    print("=" * 60)

    tests = [
        ("Point Classes", test_point_classes),
        ("Camera Calibration", test_camera_calibration),
        ("Transformation Matrix", test_transformation_matrix),
        ("Coordinate Transformer Basic", test_coordinate_transformer_basic),
        ("Perspective Transformation", test_perspective_transformation),
        ("Coordinate Normalization", test_coordinate_normalization),
        ("Convenience Functions", test_convenience_functions),
        ("Table Calibration", test_table_calibration),
        ("Transformation Availability", test_transformation_availability),
    ]

    results = {}

    try:
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            results[test_name] = test_func()

        # Summary
        print(f"\n🏁 Test Summary")
        print("=" * 70)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All coordinate transformation tests passed! System is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {total - passed} tests failed. Check the implementation.")
            return 1

    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
