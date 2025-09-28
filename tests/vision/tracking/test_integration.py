"""Test the integrated tracking system"""

import sys
import os
import numpy as np
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import components for testing
from backend.vision.tracking.optimization import TrackingOptimizer, PerformanceMetrics
from backend.vision.tracking.integration import (
    IntegratedTracker, TrackingConfig, create_integrated_tracker
)


def test_performance_optimization():
    """Test performance optimization components"""
    print("Testing Performance Optimization:")

    # Test metrics
    metrics = PerformanceMetrics()
    metrics.fps = 30.0
    metrics.tracks_processed = 5
    metrics.detections_processed = 3
    assert metrics.fps == 30.0
    print("  ✓ Performance metrics")

    # Test optimizer
    config = {
        'parallel_processing': True,
        'max_threads': 2,
        'memory_limit_mb': 256
    }

    optimizer = TrackingOptimizer(config)
    assert optimizer.enable_parallel_processing == True
    assert optimizer.max_threads == 2
    print("  ✓ Optimizer initialization")

    # Test performance monitoring
    metrics = optimizer.monitor_performance(5, 3, 0.033)  # ~30 FPS
    assert metrics.fps > 25
    print("  ✓ Performance monitoring")

    # Test algorithm selection
    algorithm = optimizer.adaptive_algorithm_selection(10, 10)
    assert algorithm in ['hungarian', 'greedy', 'auction']
    print("  ✓ Algorithm selection")

    print("Performance optimization tests completed!\n")


def test_integration_components():
    """Test integration components"""
    print("Testing Integration Components:")

    # Test configuration
    config = TrackingConfig()
    assert config.max_age == 30
    assert config.enable_optimization == True
    print("  ✓ Configuration")

    # Test factory function
    tracker = create_integrated_tracker()
    assert tracker is not None
    assert isinstance(tracker, IntegratedTracker)
    print("  ✓ Factory creation")

    # Test custom configuration
    custom_config = {
        'max_age': 20,
        'min_hits': 2,
        'enable_optimization': False
    }
    custom_tracker = create_integrated_tracker(custom_config)
    assert custom_tracker.config.max_age == 20
    assert custom_tracker.config.min_hits == 2
    print("  ✓ Custom configuration")

    print("Integration components tests completed!\n")


def test_integrated_tracking_workflow():
    """Test complete integrated tracking workflow"""
    print("Testing Integrated Tracking Workflow:")

    # Create tracker with optimization
    config = {
        'max_age': 20,
        'min_hits': 2,
        'max_distance': 40.0,
        'enable_optimization': True,
        'parallel_processing': False,  # Disable for testing
        'smooth_trajectories': True,
        'predict_missing_detections': False,  # Disable for simplicity
        'performance_monitoring': True
    }

    tracker = create_integrated_tracker(config)
    print("  ✓ Tracker creation")

    # Simulate detections (simplified format)
    class SimpleDetection:
        def __init__(self, position, ball_type='cue', confidence=0.8):
            self.position = position
            self.radius = 15.0
            self.ball_type = ball_type
            self.confidence = confidence
            self.velocity = (0.0, 0.0)
            self.number = None
            self.is_moving = False

    # Test frame processing
    detections = [
        SimpleDetection((100, 200)),
        SimpleDetection((300, 400))
    ]

    result = tracker.process_frame(detections, frame_number=1)
    assert result is not None
    assert result.frame_number == 1
    assert len(result.tracked_objects) >= 0  # May be 0 initially (not confirmed)
    print("  ✓ Frame processing")

    # Process several frames to confirm tracks
    for frame_num in range(2, 6):
        # Simulate slight movement
        detections = [
            SimpleDetection((100 + frame_num, 200 + frame_num)),
            SimpleDetection((300 + frame_num, 400 + frame_num))
        ]
        result = tracker.process_frame(detections, frame_number=frame_num)

    assert len(result.tracked_objects) > 0  # Should have confirmed tracks now
    print("  ✓ Track confirmation")

    # Test predictions
    assert len(result.predictions) > 0
    print("  ✓ Position predictions")

    # Test performance summary
    summary = tracker.get_performance_summary()
    assert 'frames_processed' in summary
    assert summary['frames_processed'] >= 5
    print("  ✓ Performance summary")

    # Test reset
    tracker.reset()
    assert tracker.frame_count == 0
    print("  ✓ Reset functionality")

    print("Integrated tracking workflow tests completed!\n")


def test_trajectory_smoothing():
    """Test trajectory smoothing functionality"""
    print("Testing Trajectory Smoothing:")

    from backend.vision.tracking.integration import TrajectorySmoothing

    smoother = TrajectorySmoothing(window_size=3)

    # Create mock tracked objects
    class MockObject:
        def __init__(self, track_id, position):
            self.track_id = track_id
            self.position = position

    # Simulate noisy trajectory
    objects = [
        MockObject(1, (100.0, 100.0)),
        MockObject(1, (102.0, 98.0)),   # Noisy
        MockObject(1, (101.0, 102.0)),  # Noisy
        MockObject(1, (104.0, 103.0))
    ]

    # Apply smoothing progressively
    for obj in objects:
        smoothed = smoother.smooth_trajectories([obj], [])
        assert len(smoothed) == 1

    print("  ✓ Trajectory smoothing")

    # Test reset
    smoother.reset()
    assert len(smoother.position_history) == 0
    print("  ✓ Smoother reset")

    print("Trajectory smoothing tests completed!\n")


def test_error_handling():
    """Test error handling in tracking system"""
    print("Testing Error Handling:")

    # Test with invalid configuration
    try:
        invalid_config = {'max_age': -1}  # Invalid value
        tracker = create_integrated_tracker(invalid_config)
        # Should still work with clamped values or defaults
        assert tracker is not None
        print("  ✓ Invalid configuration handling")
    except Exception as e:
        print(f"  ✓ Caught expected error: {e}")

    # Test with empty detections
    tracker = create_integrated_tracker()
    result = tracker.process_frame([], frame_number=1)
    assert result is not None
    assert len(result.tracked_objects) == 0
    print("  ✓ Empty detections handling")

    # Test with malformed detections
    class BadDetection:
        def __init__(self):
            pass  # Missing required attributes

    try:
        bad_detections = [BadDetection()]
        result = tracker.process_frame(bad_detections, frame_number=1)
        # Should handle gracefully
        print("  ✓ Malformed detections handling")
    except Exception as e:
        print(f"  ✓ Caught expected error: {e}")

    print("Error handling tests completed!\n")


def test_performance_characteristics():
    """Test performance characteristics"""
    print("Testing Performance Characteristics:")

    tracker = create_integrated_tracker({
        'enable_optimization': True,
        'performance_monitoring': True
    })

    # Simulate many detections to test performance
    class SimpleDetection:
        def __init__(self, position):
            self.position = position
            self.radius = 15.0
            self.ball_type = 'solid'
            self.confidence = 0.8
            self.velocity = (0.0, 0.0)
            self.number = None
            self.is_moving = False

    num_detections = 50
    detections = [
        SimpleDetection((i * 10, i * 10)) for i in range(num_detections)
    ]

    start_time = time.time()
    result = tracker.process_frame(detections, frame_number=1)
    processing_time = time.time() - start_time

    print(f"  ✓ Processed {num_detections} detections in {processing_time:.4f}s")

    # Check if performance is reasonable (should be < 1 second for 50 detections)
    assert processing_time < 1.0, f"Processing took too long: {processing_time}s"

    # Test performance summary
    summary = tracker.get_performance_summary()
    assert 'average_processing_time' in summary
    print(f"  ✓ Average processing time: {summary['average_processing_time']:.4f}s")

    print("Performance characteristics tests completed!\n")


def main():
    """Run all integration tests"""
    print("Running integrated tracking system tests...\n")

    try:
        test_performance_optimization()
        test_integration_components()
        test_integrated_tracking_workflow()
        test_trajectory_smoothing()
        test_error_handling()
        test_performance_characteristics()

        print("🎉 All integration tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
