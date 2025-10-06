#!/usr/bin/env python3
"""Test script for TPU detection and fallback logic.

This script tests:
1. TPU detection when pycoral is available
2. Graceful fallback to CPU when TPU unavailable
3. Configuration validation
4. Model loading with different formats

Usage:
    python test_tpu_detector.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision.detection.yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_tpu_detection():
    """Test TPU device detection."""
    logger.info("=" * 60)
    logger.info("Test 1: TPU Detection")
    logger.info("=" * 60)

    # Create detector with TPU device (no model)
    detector = YOLODetector(model_path=None, device="tpu", auto_fallback=True)

    # Check TPU availability
    logger.info(f"TPU available: {detector.tpu_available}")
    logger.info(f"Device: {detector.device}")
    logger.info(f"Using TPU: {detector.stats.get('using_tpu', False)}")

    if detector.tpu_available:
        logger.info("✓ TPU detected successfully")
    else:
        logger.info("✗ TPU not available (expected if no TPU hardware)")

    return detector.tpu_available


def test_fallback_to_cpu():
    """Test fallback to CPU when TPU unavailable."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Fallback to CPU")
    logger.info("=" * 60)

    # Try to create detector with TPU device but non-existent model
    detector = YOLODetector(
        model_path="nonexistent_model.tflite", device="tpu", auto_fallback=True
    )

    # Should have fallen back gracefully
    logger.info(f"Model loaded: {detector.model_loaded}")
    logger.info(f"Fallback mode: {detector.stats['fallback_mode']}")
    logger.info(f"Device: {detector.device}")

    if detector.stats["fallback_mode"]:
        logger.info("✓ Graceful fallback to CPU mode")
        return True
    else:
        logger.error("✗ Fallback failed")
        return False


def test_cpu_detector():
    """Test standard CPU detector."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: CPU Detector (no model)")
    logger.info("=" * 60)

    detector = YOLODetector(model_path=None, device="cpu", auto_fallback=True)

    logger.info(f"Model loaded: {detector.model_loaded}")
    logger.info(f"Device: {detector.device}")
    logger.info(f"Fallback mode: {detector.stats['fallback_mode']}")

    # Test inference with no model (should return empty list)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_balls(test_frame)

    logger.info(f"Detections (expected 0): {len(detections)}")

    if len(detections) == 0:
        logger.info("✓ CPU detector works correctly")
        return True
    else:
        logger.error("✗ Unexpected detections")
        return False


def test_model_info():
    """Test model info reporting."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Model Info")
    logger.info("=" * 60)

    # Test with TPU device
    detector = YOLODetector(
        model_path=None,
        device="tpu",
        tpu_device_path="usb",
        confidence=0.5,
        nms_threshold=0.4,
        auto_fallback=True,
    )

    info = detector.get_model_info()

    logger.info("Model Info:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    required_keys = [
        "model_path",
        "model_format",
        "model_loaded",
        "device",
        "confidence_threshold",
        "nms_threshold",
        "fallback_mode",
        "using_tpu",
    ]

    all_present = all(key in info for key in required_keys)

    if all_present:
        logger.info("✓ All required info fields present")
        return True
    else:
        logger.error("✗ Missing info fields")
        return False


def test_create_detector_helper():
    """Test create_detector helper function."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: create_detector Helper")
    logger.info("=" * 60)

    from vision.detection.yolo_detector import create_detector

    # Test with config dictionary
    config = {
        "device": "tpu",
        "confidence": 0.45,
        "nms_threshold": 0.5,
        "tpu_device_path": None,
        "auto_fallback": True,
    }

    detector = create_detector(model_path=None, config=config)

    logger.info(f"Device: {detector.device}")
    logger.info(f"Confidence: {detector.confidence}")
    logger.info(f"NMS threshold: {detector.nms_threshold}")
    logger.info(f"TPU device path: {detector.tpu_device_path}")

    if (
        detector.confidence == 0.45
        and detector.nms_threshold == 0.5
        and detector.tpu_device_path is None
    ):
        logger.info("✓ Helper function works correctly")
        return True
    else:
        logger.error("✗ Configuration not applied correctly")
        return False


def test_statistics():
    """Test statistics tracking."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Statistics Tracking")
    logger.info("=" * 60)

    detector = YOLODetector(model_path=None, device="cpu", auto_fallback=True)

    stats = detector.get_statistics()

    logger.info("Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    required_stats = [
        "total_inferences",
        "total_detections",
        "avg_inference_time",
        "fallback_mode",
        "using_tpu",
    ]

    all_present = all(key in stats for key in required_stats)

    if all_present:
        logger.info("✓ All required statistics present")
        return True
    else:
        logger.error("✗ Missing statistics")
        return False


def main():
    """Run all tests."""
    logger.info("Starting TPU Detector Tests")
    logger.info("=" * 60)

    results = []

    # Run tests
    try:
        results.append(("TPU Detection", test_tpu_detection()))
    except Exception as e:
        logger.error(f"TPU Detection test failed: {e}")
        results.append(("TPU Detection", False))

    try:
        results.append(("Fallback to CPU", test_fallback_to_cpu()))
    except Exception as e:
        logger.error(f"Fallback test failed: {e}")
        results.append(("Fallback to CPU", False))

    try:
        results.append(("CPU Detector", test_cpu_detector()))
    except Exception as e:
        logger.error(f"CPU Detector test failed: {e}")
        results.append(("CPU Detector", False))

    try:
        results.append(("Model Info", test_model_info()))
    except Exception as e:
        logger.error(f"Model Info test failed: {e}")
        results.append(("Model Info", False))

    try:
        results.append(("create_detector Helper", test_create_detector_helper()))
    except Exception as e:
        logger.error(f"create_detector test failed: {e}")
        results.append(("create_detector Helper", False))

    try:
        results.append(("Statistics", test_statistics()))
    except Exception as e:
        logger.error(f"Statistics test failed: {e}")
        results.append(("Statistics", False))

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        logger.info(f"{symbol} {test_name}: {status}")

        if result:
            passed += 1
        else:
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Total: {len(results)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failed == 0:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
