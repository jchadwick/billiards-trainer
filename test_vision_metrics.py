#!/usr/bin/env python3
"""Test script for vision metrics collection and reporting functionality.

This script tests the metrics collection system for the vision pipeline
including performance tracking, FPS monitoring, and detection accuracy.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.vision.utils.metrics import (
    VisionMetricsCollector,
    MetricType,
    AggregationType,
    get_metrics_collector,
    time_function,
    record_fps,
    record_detection,
    get_current_fps,
    get_system_stats
)


def simulate_ball_detection(detector_name: str, accuracy: float = 0.9) -> dict:
    """Simulate a ball detection operation."""
    import random

    start_time = time.perf_counter()

    # Simulate processing time
    processing_time = random.uniform(0.01, 0.05)  # 10-50ms
    time.sleep(processing_time)

    # Simulate detection result
    is_correct = random.random() < accuracy
    confidence = random.uniform(0.7, 0.95) if is_correct else random.uniform(0.3, 0.7)

    detection_time = time.perf_counter() - start_time

    return {
        "detected": True,
        "confidence": confidence,
        "is_correct": is_correct,
        "detection_time": detection_time,
        "detector": detector_name
    }


def simulate_table_detection() -> dict:
    """Simulate table detection operation."""
    import random

    start_time = time.perf_counter()

    # Simulate longer processing time for table detection
    processing_time = random.uniform(0.02, 0.1)  # 20-100ms
    time.sleep(processing_time)

    detection_time = time.perf_counter() - start_time
    success = random.random() > 0.1  # 90% success rate

    return {
        "success": success,
        "corners_found": 4 if success else random.randint(0, 3),
        "detection_time": detection_time,
        "confidence": random.uniform(0.8, 0.98) if success else random.uniform(0.2, 0.6)
    }


@time_function
def simulate_frame_processing():
    """Simulate processing a vision frame."""
    import random

    # Simulate frame processing
    time.sleep(random.uniform(0.02, 0.08))  # 20-80ms processing time

    # Record frame processed for FPS calculation
    record_fps()

    return True


def test_basic_metrics_collection():
    """Test basic metrics collection functionality."""
    print("🧪 Testing Basic Metrics Collection")
    print("=" * 50)

    # Get metrics collector
    collector = get_metrics_collector()

    # Test recording different metric types
    test_metrics = [
        (MetricType.LATENCY, "ball_detection", 25.5),
        (MetricType.ACCURACY, "detection_accuracy", 0.92),
        (MetricType.FPS, "pipeline_fps", 30.0),
        (MetricType.MEMORY, "memory_usage", 128.5),
        (MetricType.CPU, "cpu_usage", 65.2)
    ]

    print("Recording test metrics...")
    for metric_type, name, value in test_metrics:
        collector.record_metric(metric_type, name, value)
        print(f"✓ Recorded {metric_type.value}_{name}: {value}")

    # Test metric retrieval
    print("\nRetrieving metrics...")
    for metric_type, name, expected_value in test_metrics:
        retrieved_value = collector.get_metric_stats(metric_type, name, AggregationType.MEAN)
        if retrieved_value is not None and abs(retrieved_value - expected_value) < 0.01:
            print(f"✅ {metric_type.value}_{name}: {retrieved_value}")
        else:
            print(f"❌ {metric_type.value}_{name}: Expected {expected_value}, got {retrieved_value}")

    return True


def test_fps_tracking():
    """Test FPS tracking functionality."""
    print("\n🎬 Testing FPS Tracking")
    print("=" * 50)

    collector = get_metrics_collector()

    # Simulate processing multiple frames
    print("Simulating frame processing...")
    for i in range(20):
        # Record frame processing
        record_fps()
        time.sleep(0.033)  # ~30 FPS

    # Check FPS calculation
    current_fps = get_current_fps()
    print(f"Current FPS: {current_fps:.2f}")

    # Should be close to 30 FPS
    if 25 <= current_fps <= 35:
        print("✅ FPS tracking working correctly")
        return True
    else:
        print(f"❌ FPS tracking failed: expected ~30, got {current_fps}")
        return False


def test_detection_accuracy_tracking():
    """Test detection accuracy metrics."""
    print("\n🎯 Testing Detection Accuracy Tracking")
    print("=" * 50)

    collector = get_metrics_collector()

    # Simulate ball detections
    print("Simulating ball detections...")
    for i in range(50):
        result = simulate_ball_detection("ball_detector", accuracy=0.85)

        record_detection(
            detector_name=result["detector"],
            is_correct=result["is_correct"],
            confidence=result["confidence"],
            detection_time=result["detection_time"]
        )

    # Get detection metrics
    detection_metrics = collector.get_detection_metrics("ball_detector")

    if detection_metrics:
        print(f"Detection Results:")
        print(f"  Precision: {detection_metrics.precision:.3f}")
        print(f"  Recall: {detection_metrics.recall:.3f}")
        print(f"  F1 Score: {detection_metrics.f1_score:.3f}")
        print(f"  Accuracy: {detection_metrics.accuracy:.3f}")
        print(f"  Average Confidence: {detection_metrics.average_confidence:.3f}")
        print(f"  Average Detection Time: {detection_metrics.average_detection_time*1000:.2f}ms")

        # Check if metrics are reasonable
        if 0.7 <= detection_metrics.accuracy <= 1.0:
            print("✅ Detection accuracy tracking working correctly")
            return True
        else:
            print(f"❌ Detection accuracy seems off: {detection_metrics.accuracy}")
            return False
    else:
        print("❌ No detection metrics found")
        return False


def test_performance_profiling():
    """Test performance profiling functionality."""
    print("\n⚡ Testing Performance Profiling")
    print("=" * 50)

    collector = get_metrics_collector()

    # Test performance timer context manager
    print("Testing performance timer...")

    with collector.time_component("test_operation"):
        # Simulate some work
        time.sleep(0.1)

    # Test the time_function decorator
    print("Testing decorated function timing...")
    for _ in range(5):
        simulate_frame_processing()

    # Get performance profiles
    test_profile = collector.get_performance_profile("test_operation")
    frame_profile = collector.get_performance_profile("simulate_frame_processing")

    success = True

    if test_profile:
        print(f"Test Operation Profile:")
        print(f"  Average time: {test_profile.average_time*1000:.2f}ms")
        print(f"  Call count: {test_profile.call_count}")
        print(f"  Min time: {test_profile.min_time*1000:.2f}ms")
        print(f"  Max time: {test_profile.max_time*1000:.2f}ms")

        if 90 <= test_profile.average_time*1000 <= 110:  # Should be ~100ms
            print("✅ Test operation profiling working correctly")
        else:
            print(f"❌ Test operation timing off: expected ~100ms, got {test_profile.average_time*1000:.2f}ms")
            success = False
    else:
        print("❌ Test operation profile not found")
        success = False

    if frame_profile:
        print(f"Frame Processing Profile:")
        print(f"  Average time: {frame_profile.average_time*1000:.2f}ms")
        print(f"  Call count: {frame_profile.call_count}")

        if frame_profile.call_count == 5:
            print("✅ Frame processing profiling working correctly")
        else:
            print(f"❌ Frame processing call count wrong: expected 5, got {frame_profile.call_count}")
            success = False
    else:
        print("❌ Frame processing profile not found")
        success = False

    return success


def test_system_monitoring():
    """Test system resource monitoring."""
    print("\n🖥️  Testing System Monitoring")
    print("=" * 50)

    # Get system stats
    stats = get_system_stats()

    print("System Metrics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Check if we got reasonable metrics
    expected_keys = ["cpu_percent", "memory_mb", "memory_percent"]
    if all(key in stats for key in expected_keys):
        print("✅ System monitoring working correctly")
        return True
    else:
        print("❌ Missing system metrics")
        return False


def test_comprehensive_reporting():
    """Test comprehensive metrics reporting."""
    print("\n📊 Testing Comprehensive Reporting")
    print("=" * 50)

    collector = get_metrics_collector()

    # Generate some more test data
    for i in range(10):
        collector.record_metric(MetricType.LATENCY, "ball_detection", 20 + i)
        collector.record_metric(MetricType.LATENCY, "table_detection", 50 + i*2)
        record_fps()
        time.sleep(0.01)

    # Get comprehensive report
    report = collector.get_comprehensive_report()

    print("Comprehensive Report Generated:")
    print(f"  Report timestamp: {report['timestamp']}")
    print(f"  FPS data: {report['fps']}")
    print(f"  System metrics: {list(report['system'].keys())}")
    print(f"  Performance profiles: {len(report['performance_profiles'])}")
    print(f"  Detection metrics: {len(report['detection_metrics'])}")
    print(f"  Latency stats: {len(report['latency_stats'])}")

    # Check report structure
    required_sections = ["fps", "system", "performance_profiles", "latency_stats"]
    if all(section in report for section in required_sections):
        print("✅ Comprehensive reporting working correctly")
        return True
    else:
        print("❌ Missing report sections")
        return False


def test_metrics_reset():
    """Test metrics reset functionality."""
    print("\n🔄 Testing Metrics Reset")
    print("=" * 50)

    collector = get_metrics_collector()

    # Add some metrics
    collector.record_metric(MetricType.LATENCY, "test_reset", 100.0)

    # Verify metric exists
    value_before = collector.get_metric_stats(MetricType.LATENCY, "test_reset")
    print(f"Value before reset: {value_before}")

    # Reset specific metric
    collector.reset_metrics(MetricType.LATENCY, "test_reset")

    # Verify metric is reset
    value_after = collector.get_metric_stats(MetricType.LATENCY, "test_reset")
    print(f"Value after reset: {value_after}")

    if value_before is not None and value_after is None:
        print("✅ Metrics reset working correctly")
        return True
    else:
        print("❌ Metrics reset failed")
        return False


async def main():
    """Run all vision metrics tests."""
    print("🚀 Starting Vision Metrics System Tests")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    tests = [
        ("Basic Metrics Collection", test_basic_metrics_collection),
        ("FPS Tracking", test_fps_tracking),
        ("Detection Accuracy Tracking", test_detection_accuracy_tracking),
        ("Performance Profiling", test_performance_profiling),
        ("System Monitoring", test_system_monitoring),
        ("Comprehensive Reporting", test_comprehensive_reporting),
        ("Metrics Reset", test_metrics_reset),
    ]

    results = {}

    try:
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            results[test_name] = test_func()

        # Summary
        print(f"\n🏁 Test Summary")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All vision metrics tests passed! System is working correctly.")
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
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
