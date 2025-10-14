#!/usr/bin/env python3
"""Test configuration validation module.

This script tests the configuration validation to ensure it works correctly
with both valid and invalid configurations.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config.manager import ConfigurationModule
from backend.config.validation import (
    ConfigValidationError,
    ConfigValidator,
    validate_configuration,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_valid_configuration():
    """Test validation with valid configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Valid Configuration")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()
    validator = ConfigValidator(config_manager)

    success = validator.validate_all()

    if success:
        logger.info("✅ Valid configuration test PASSED")
    else:
        logger.error("❌ Valid configuration test FAILED")
        for error in validator.errors:
            logger.error(f"  {error}")

    return success


def test_missing_defaults():
    """Test that defaults are applied for missing parameters."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Missing Defaults")
    logger.info("=" * 60)

    # Create a fresh config manager
    config_manager = ConfigurationModule()

    # Remove some vision parameters
    config_manager.set("vision.tracking.min_hits", None)
    config_manager.set("vision.detection.yolo_confidence", None)

    validator = ConfigValidator(config_manager)
    validator.apply_defaults()

    # Check that defaults were applied
    min_hits = config_manager.get("vision.tracking.min_hits")
    yolo_conf = config_manager.get("vision.detection.yolo_confidence")

    if min_hits == 10 and yolo_conf == 0.15:
        logger.info("✅ Defaults applied correctly")
        logger.info(f"  vision.tracking.min_hits = {min_hits}")
        logger.info(f"  vision.detection.yolo_confidence = {yolo_conf}")
        return True
    else:
        logger.error("❌ Defaults not applied correctly")
        logger.error(f"  vision.tracking.min_hits = {min_hits} (expected 10)")
        logger.error(
            f"  vision.detection.yolo_confidence = {yolo_conf} (expected 0.15)"
        )
        return False


def test_invalid_range():
    """Test validation catches invalid ranges."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Invalid Range Detection")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()

    # Set invalid values
    config_manager.set("vision.tracking.min_hits", -5)  # Must be > 0
    config_manager.set("vision.detection.yolo_confidence", 1.5)  # Must be 0-1
    config_manager.set("vision.camera.fps", 200)  # Must be <= 120

    validator = ConfigValidator(config_manager)
    success = validator.validate_all()

    if not success and len(validator.errors) >= 3:
        logger.info("✅ Invalid range detection test PASSED")
        logger.info("  Caught errors:")
        for error in validator.errors:
            logger.info(f"    - {error}")
        return True
    else:
        logger.error("❌ Invalid range detection test FAILED")
        logger.error(f"  Expected at least 3 errors, got {len(validator.errors)}")
        return False


def test_suboptimal_warnings():
    """Test that suboptimal values generate warnings."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Suboptimal Value Warnings")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()

    # Set suboptimal values
    config_manager.set("vision.tracking.min_hits", 2)  # Too low
    config_manager.set("vision.detection.yolo_confidence", 0.8)  # Too high
    config_manager.set("vision.tracking.max_distance", 300.0)  # Too high

    validator = ConfigValidator(config_manager)
    validator.validate_all()

    if len(validator.warnings) >= 3:
        logger.info("✅ Suboptimal warnings test PASSED")
        logger.info("  Generated warnings:")
        for warning in validator.warnings:
            logger.info(f"    ⚠️  {warning}")
        return True
    else:
        logger.error("❌ Suboptimal warnings test FAILED")
        logger.error(f"  Expected at least 3 warnings, got {len(validator.warnings)}")
        return False


def test_cross_field_validation():
    """Test cross-field validations."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Cross-Field Validation")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()

    # Set conflicting values
    config_manager.set("vision.detection.min_ball_radius", 50)
    config_manager.set("vision.detection.max_ball_radius", 30)  # Less than min!

    validator = ConfigValidator(config_manager)
    success = validator.validate_all()

    if not success and any(
        "min_ball_radius" in err and "max_ball_radius" in err
        for err in validator.errors
    ):
        logger.info("✅ Cross-field validation test PASSED")
        logger.info("  Caught error:")
        for error in validator.errors:
            if "min_ball_radius" in error and "max_ball_radius" in error:
                logger.info(f"    - {error}")
        return True
    else:
        logger.error("❌ Cross-field validation test FAILED")
        logger.error("  Did not catch min/max ball radius conflict")
        return False


def test_kernel_size_validation():
    """Test that kernel sizes must be odd."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Kernel Size Validation")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()

    # Set even kernel sizes (invalid)
    config_manager.set("vision.processing.blur_kernel_size", 4)  # Must be odd
    config_manager.set("vision.processing.morphology_kernel_size", 6)  # Must be odd

    validator = ConfigValidator(config_manager)
    success = validator.validate_all()

    if not success and any("odd" in err.lower() for err in validator.errors):
        logger.info("✅ Kernel size validation test PASSED")
        logger.info("  Caught errors:")
        for error in validator.errors:
            if "odd" in error.lower():
                logger.info(f"    - {error}")
        return True
    else:
        logger.error("❌ Kernel size validation test FAILED")
        logger.error("  Did not catch even kernel sizes")
        return False


def test_fail_fast():
    """Test that validate_configuration raises exception on critical errors."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Fail Fast Behavior")
    logger.info("=" * 60)

    config_manager = ConfigurationModule()

    # Set multiple invalid values
    config_manager.set("vision.tracking.min_hits", -10)
    config_manager.set("vision.detection.yolo_confidence", 2.0)

    try:
        validate_configuration(config_manager)
        logger.error("❌ Fail fast test FAILED - no exception raised")
        return False
    except ConfigValidationError as e:
        logger.info("✅ Fail fast test PASSED")
        logger.info("  Exception raised as expected:")
        logger.info(f"    {str(e)[:100]}...")
        return True


def main():
    """Run all validation tests."""
    logger.info("\n" + "=" * 80)
    logger.info("CONFIGURATION VALIDATION TESTS")
    logger.info("=" * 80)

    tests = [
        ("Valid Configuration", test_valid_configuration),
        ("Missing Defaults", test_missing_defaults),
        ("Invalid Range Detection", test_invalid_range),
        ("Suboptimal Value Warnings", test_suboptimal_warnings),
        ("Cross-Field Validation", test_cross_field_validation),
        ("Kernel Size Validation", test_kernel_size_validation),
        ("Fail Fast Behavior", test_fail_fast),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            import traceback

            logger.error(traceback.format_exc())
            results.append((test_name, False))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
