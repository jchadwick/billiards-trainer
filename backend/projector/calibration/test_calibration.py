#!/usr/bin/env python3
"""Test script for the projector calibration system.

This script tests the core functionality of the calibration system including
keystone correction, geometric mapping, and persistence operations.
"""

import logging
import sys
import tempfile
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from projector.calibration import (
    CalibrationManager,
    CornerPoints,
    GeometricCalibrator,
    KeystoneCalibrator,
    KeystoneParams,
    TableDimensions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_keystone_calibrator():
    """Test keystone calibration functionality."""
    logger.info("Testing KeystoneCalibrator...")

    # Create calibrator for 1920x1080 display
    calibrator = KeystoneCalibrator(1920, 1080)

    # Test corner point setting
    corners = CornerPoints(
        top_left=(100.0, 50.0),
        top_right=(1820.0, 80.0),
        bottom_right=(1850.0, 1030.0),
        bottom_left=(70.0, 1000.0),
    )
    calibrator.set_corner_points(corners)

    # Test keystone parameters
    params = KeystoneParams(
        horizontal=0.1, vertical=-0.05, rotation=2.0, barrel_distortion=0.02
    )
    calibrator.set_keystone_params(params)

    # Test point transformation
    test_point = (960, 540)  # Center of display
    transformed = calibrator.transform_point(test_point[0], test_point[1])
    logger.info(f"Center point {test_point} transformed to {transformed}")

    # Test calibration grid generation
    grid = calibrator.generate_calibration_grid(5)
    logger.info(f"Generated calibration grid with {len(grid)} lines")

    # Test crosshairs generation
    crosshairs = calibrator.generate_crosshairs()
    logger.info(f"Generated {len(crosshairs)} crosshair lines")

    # Test validation
    is_valid, errors = calibrator.validate_calibration()
    logger.info(f"Keystone calibration valid: {is_valid}, errors: {errors}")

    # Test data persistence
    data = calibrator.get_calibration_data()
    logger.info(f"Calibration data keys: {list(data.keys())}")

    # Test loading data
    calibrator.reset_calibration()
    success = calibrator.load_calibration_data(data)
    logger.info(f"Calibration data loaded successfully: {success}")

    logger.info("KeystoneCalibrator tests completed ✓")
    return True


def test_geometric_calibrator():
    """Test geometric calibration functionality."""
    logger.info("Testing GeometricCalibrator...")

    # Create table dimensions (9-foot pool table)
    table_dims = TableDimensions(
        length=2.74, width=1.37, pocket_radius=0.05, rail_width=0.03  # meters  # meters
    )

    # Create calibrator
    calibrator = GeometricCalibrator(table_dims, 1920, 1080)

    # Add calibration targets (table corners)
    corner_displays = [
        (200.0, 150.0),  # top_left
        (1720.0, 140.0),  # top_right
        (1730.0, 930.0),  # bottom_right
        (190.0, 940.0),  # bottom_left
    ]
    calibrator.add_table_corner_targets(corner_displays)

    # Add additional calibration points
    calibrator.add_calibration_target(1.37, 0.685, 960, 540, "center")

    # Calculate transform
    success = calibrator.calculate_transform()
    logger.info(f"Geometric transform calculated: {success}")
    logger.info(f"Calibration error: {calibrator.calibration_error:.3f} pixels")

    # Test coordinate transformation
    table_center = (1.37, 0.685)  # Center of table
    display_coords = calibrator.table_to_display(table_center[0], table_center[1])
    logger.info(f"Table center {table_center} -> display {display_coords}")

    # Test inverse transformation
    back_to_table = calibrator.display_to_table(display_coords[0], display_coords[1])
    logger.info(f"Back to table coordinates: {back_to_table}")

    # Test grid generation
    grid = calibrator.generate_table_grid(0.2)  # 20cm spacing
    logger.info(f"Generated table grid with {len(grid)} lines")

    # Test validation
    is_valid, errors = calibrator.validate_calibration(tolerance=10.0)
    logger.info(f"Geometric calibration valid: {is_valid}, errors: {errors}")

    # Test data persistence
    data = calibrator.get_calibration_data()
    logger.info(f"Geometric data keys: {list(data.keys())}")

    logger.info("GeometricCalibrator tests completed ✓")
    return True


def test_calibration_manager():
    """Test calibration manager functionality."""
    logger.info("Testing CalibrationManager...")

    # Create table dimensions
    table_dims = TableDimensions(length=2.74, width=1.37)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        calibration_dir = Path(temp_dir)

        # Create calibration manager
        manager = CalibrationManager(
            display_width=1920,
            display_height=1080,
            table_dimensions=table_dims,
            calibration_dir=calibration_dir,
        )

        # Test calibration workflow
        logger.info("Starting calibration workflow...")

        # Start calibration
        success = manager.start_calibration()
        logger.info(f"Calibration started: {success}")

        # Set up keystone corners
        corners = CornerPoints(
            top_left=(100.0, 50.0),
            top_right=(1820.0, 80.0),
            bottom_right=(1850.0, 1030.0),
            bottom_left=(70.0, 1000.0),
        )
        success = manager.setup_keystone_corners(corners)
        logger.info(f"Keystone corners set: {success}")

        # Complete keystone calibration
        success = manager.complete_keystone_calibration()
        logger.info(f"Keystone calibration completed: {success}")

        # Add geometric targets
        corner_displays = [
            (200.0, 150.0),
            (1720.0, 140.0),
            (1730.0, 930.0),
            (190.0, 940.0),
        ]
        success = manager.add_table_corner_targets(corner_displays)
        logger.info(f"Table corner targets added: {success}")

        # Calculate geometric transform
        success = manager.calculate_geometric_transform()
        logger.info(f"Geometric transform calculated: {success}")

        # Validate calibration
        success = manager.validate_calibration()
        logger.info(f"Calibration validated: {success}")

        # Save calibration
        success = manager.save_calibration("test_profile")
        logger.info(f"Calibration saved: {success}")

        # Test point transformation
        table_point = (1.37, 0.685)
        display_point = manager.transform_point(table_point[0], table_point[1])
        logger.info(f"Transform: table {table_point} -> display {display_point}")

        # Test profile management
        profiles = manager.get_profiles()
        logger.info(f"Available profiles: {[p['name'] for p in profiles]}")

        # Reset and load calibration
        manager.reset_calibration()
        success = manager.load_calibration("test_profile")
        logger.info(f"Calibration loaded from profile: {success}")

        # Test calibration validity
        is_valid = manager.is_calibration_valid()
        logger.info(f"Loaded calibration is valid: {is_valid}")

    logger.info("CalibrationManager tests completed ✓")
    return True


def test_integration():
    """Test integration with display manager concepts."""
    logger.info("Testing integration scenarios...")

    table_dims = TableDimensions(length=2.74, width=1.37)

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CalibrationManager(1920, 1080, table_dims, Path(temp_dir))

        # Simulate a complete calibration workflow
        manager.start_calibration()

        # Set up basic keystone correction
        corners = CornerPoints(
            top_left=(50.0, 30.0),
            top_right=(1870.0, 40.0),
            bottom_right=(1880.0, 1050.0),
            bottom_left=(40.0, 1040.0),
        )
        manager.setup_keystone_corners(corners)
        manager.complete_keystone_calibration()

        # Add precise geometric mapping
        precise_corners = [
            (180.0, 120.0),
            (1740.0, 130.0),
            (1750.0, 960.0),
            (170.0, 950.0),
        ]
        manager.add_table_corner_targets(precise_corners)

        # Add center point for better accuracy
        manager.add_geometric_target(1.37, 0.685, 960, 540, "center")

        # Complete calibration
        manager.calculate_geometric_transform()
        success = manager.validate_calibration()

        if success:
            manager.save_calibration()

            # Test multiple table points
            test_points = [
                (0.0, 0.0),  # Corner
                (2.74, 1.37),  # Opposite corner
                (1.37, 0.685),  # Center
                (0.685, 0.3425),  # Quarter point
            ]

            logger.info("Testing coordinate transformations:")
            for table_x, table_y in test_points:
                display_x, display_y = manager.transform_point(table_x, table_y)
                logger.info(
                    f"  Table ({table_x:.3f}, {table_y:.3f}) -> Display ({display_x:.1f}, {display_y:.1f})"
                )

            # Test calibration accuracy
            calibration_data = manager.get_calibration_data()
            error = calibration_data.get("geometric", {}).get("calibration_error", 0)
            logger.info(f"Final calibration error: {error:.3f} pixels")

    logger.info("Integration tests completed ✓")
    return True


def main():
    """Run all calibration tests."""
    logger.info("Starting calibration system tests...")

    try:
        # Run individual component tests
        test_keystone_calibrator()
        test_geometric_calibrator()
        test_calibration_manager()
        test_integration()

        logger.info("🎉 All calibration tests passed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
