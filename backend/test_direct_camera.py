#!/usr/bin/env python3
"""
Test script for DirectCameraModule.

This script validates that the DirectCameraModule works correctly by:
1. Creating an instance with simple configuration
2. Starting capture
3. Retrieving frames using different methods
4. Collecting and printing statistics
5. Properly stopping capture

This script can be run locally (without actual camera) to verify the code structure,
and on the target environment to verify actual camera functionality.
"""

import sys
import time
import logging
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from vision.direct_camera import DirectCameraModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_direct_camera():
    """Test DirectCameraModule functionality."""

    print_section("DirectCameraModule Test Script")

    # Test configuration
    config = {
        'device_id': 0,
        'resolution': (640, 480),
        'fps': 30,
        'buffer_size': 1
    }

    print(f"\nConfiguration:")
    print(f"  Device ID: {config['device_id']}")
    print(f"  Resolution: {config['resolution']}")
    print(f"  FPS: {config['fps']}")
    print(f"  Buffer Size: {config['buffer_size']}")

    # Statistics tracking
    stats = {
        'current_frame_count': 0,
        'current_frame_success': 0,
        'current_frame_none': 0,
        'streaming_frame_count': 0,
        'streaming_frame_success': 0,
        'streaming_frame_none': 0,
        'frame_shapes': set(),
        'streaming_shapes': set(),
    }

    camera = None

    try:
        # Step 1: Create instance
        print_section("Step 1: Creating DirectCameraModule Instance")
        camera = DirectCameraModule(config=config)
        print("  ✓ DirectCameraModule instance created successfully")

        # Step 2: Start capture
        print_section("Step 2: Starting Camera Capture")
        success = camera.start_capture()

        if not success:
            print("  ✗ Failed to start camera capture")
            print("  Note: This is expected if no camera is available (e.g., on local machine)")
            return False

        print("  ✓ Camera capture started successfully")

        # Wait for first frame
        print("\n  Waiting for first frame (max 3 seconds)...")
        start_wait = time.time()
        first_frame = None

        while first_frame is None and (time.time() - start_wait) < 3.0:
            first_frame = camera.get_current_frame()
            if first_frame is None:
                time.sleep(0.1)

        if first_frame is None:
            print("  ✗ No frame received within 3 seconds")
            return False

        print(f"  ✓ First frame received (shape: {first_frame.shape})")

        # Step 3: Get 10 frames using get_current_frame()
        print_section("Step 3: Testing get_current_frame() - 10 iterations")

        for i in range(10):
            frame = camera.get_current_frame()
            stats['current_frame_count'] += 1

            if frame is not None:
                stats['current_frame_success'] += 1
                stats['frame_shapes'].add(frame.shape)
                print(f"  Frame {i+1}/10: ✓ (shape: {frame.shape})")
            else:
                stats['current_frame_none'] += 1
                print(f"  Frame {i+1}/10: None (no frame available)")

            time.sleep(0.05)  # Small delay between requests

        # Step 4: Get 10 frames using get_frame_for_streaming()
        print_section("Step 4: Testing get_frame_for_streaming(scale=0.5) - 10 iterations")

        for i in range(10):
            frame = camera.get_frame_for_streaming(scale=0.5)
            stats['streaming_frame_count'] += 1

            if frame is not None:
                stats['streaming_frame_success'] += 1
                stats['streaming_shapes'].add(frame.shape)
                print(f"  Frame {i+1}/10: ✓ (shape: {frame.shape})")
            else:
                stats['streaming_frame_none'] += 1
                print(f"  Frame {i+1}/10: None (rate limited or unavailable)")

            # Streaming has 15 FPS rate limit (1/15 = ~0.067s), so wait a bit longer
            time.sleep(0.1)

        # Step 5: Get camera statistics
        print_section("Step 5: Camera Statistics")

        cam_stats = camera.get_statistics()

        print(f"\n  Capture Status:")
        print(f"    Is Capturing: {cam_stats['is_capturing']}")
        print(f"    Device ID: {cam_stats['device_id']}")
        print(f"    Resolution: {cam_stats['resolution']}")
        print(f"    Target FPS: {cam_stats['target_fps']}")

        print(f"\n  Frame Statistics:")
        print(f"    Total Frames Captured: {cam_stats['frame_count']}")
        print(f"    Uptime: {cam_stats['uptime_seconds']:.2f} seconds")
        print(f"    Actual FPS: {cam_stats['actual_fps']:.2f}")

        print(f"\n  Rate Limiting:")
        print(f"    Processing Rate Limit: {cam_stats['rate_limits']['processing']:.4f}s "
              f"({1.0/cam_stats['rate_limits']['processing']:.1f} FPS)")
        print(f"    Streaming Rate Limit: {cam_stats['rate_limits']['streaming']:.4f}s "
              f"({1.0/cam_stats['rate_limits']['streaming']:.1f} FPS)")

        print(f"\n  Dropped Frames:")
        print(f"    Processing: {cam_stats['dropped_frames']['processing']}")
        print(f"    Streaming: {cam_stats['dropped_frames']['streaming']}")

        # Step 6: Print test statistics
        print_section("Step 6: Test Statistics")

        print(f"\n  get_current_frame() Results:")
        print(f"    Total Attempts: {stats['current_frame_count']}")
        print(f"    Successful: {stats['current_frame_success']} "
              f"({stats['current_frame_success']/stats['current_frame_count']*100:.1f}%)")
        print(f"    None Returned: {stats['current_frame_none']} "
              f"({stats['current_frame_none']/stats['current_frame_count']*100:.1f}%)")
        if stats['frame_shapes']:
            print(f"    Frame Shapes: {sorted(stats['frame_shapes'])}")

        print(f"\n  get_frame_for_streaming() Results:")
        print(f"    Total Attempts: {stats['streaming_frame_count']}")
        print(f"    Successful: {stats['streaming_frame_success']} "
              f"({stats['streaming_frame_success']/stats['streaming_frame_count']*100:.1f}%)")
        print(f"    None Returned: {stats['streaming_frame_none']} "
              f"({stats['streaming_frame_none']/stats['streaming_frame_count']*100:.1f}%)")
        if stats['streaming_shapes']:
            print(f"    Frame Shapes: {sorted(stats['streaming_shapes'])}")

        # Verify expected behavior
        print_section("Step 7: Validation")

        validation_passed = True

        # Check that we got at least some frames
        if stats['current_frame_success'] == 0:
            print("  ✗ FAIL: No frames retrieved via get_current_frame()")
            validation_passed = False
        else:
            print(f"  ✓ PASS: Retrieved {stats['current_frame_success']} frames via get_current_frame()")

        # Check streaming frames (may be rate limited, so some None is OK)
        if stats['streaming_frame_success'] == 0:
            print("  ✗ FAIL: No frames retrieved via get_frame_for_streaming()")
            validation_passed = False
        else:
            print(f"  ✓ PASS: Retrieved {stats['streaming_frame_success']} frames via get_frame_for_streaming()")

        # Check that camera is still capturing
        if not cam_stats['is_capturing']:
            print("  ✗ FAIL: Camera stopped capturing unexpectedly")
            validation_passed = False
        else:
            print("  ✓ PASS: Camera still capturing")

        # Check actual FPS is reasonable
        if cam_stats['actual_fps'] < 5.0:
            print(f"  ⚠ WARNING: Actual FPS ({cam_stats['actual_fps']:.2f}) is very low")
        elif cam_stats['actual_fps'] > config['fps'] * 1.5:
            print(f"  ⚠ WARNING: Actual FPS ({cam_stats['actual_fps']:.2f}) exceeds target significantly")
        else:
            print(f"  ✓ PASS: Actual FPS ({cam_stats['actual_fps']:.2f}) is reasonable")

        # Step 7: Stop capture
        print_section("Step 8: Stopping Camera Capture")
        camera.stop_capture()
        print("  ✓ Camera capture stopped")

        # Verify camera stopped
        if camera.is_capturing():
            print("  ✗ WARNING: Camera still reports as capturing after stop")
        else:
            print("  ✓ Camera confirmed stopped")

        # Final result
        print_section("Test Results")

        if validation_passed:
            print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓")
            print("\n  DirectCameraModule is functioning correctly!")
            return True
        else:
            print("\n  ✗✗✗ SOME TESTS FAILED ✗✗✗")
            print("\n  Please review the failures above.")
            return False

    except KeyboardInterrupt:
        print("\n\n  Test interrupted by user (Ctrl+C)")
        return False

    except Exception as e:
        print_section("ERROR")
        print(f"\n  ✗ Unexpected error occurred: {e}")
        logger.error("Test failed with exception", exc_info=True)
        return False

    finally:
        # Ensure cleanup
        if camera is not None:
            try:
                camera.stop_capture()
                print("\n  Cleanup: Camera capture stopped")
            except Exception as e:
                print(f"\n  Warning: Error during cleanup: {e}")


def main():
    """Main entry point."""
    print("\n" + "█" * 60)
    print("  DirectCameraModule Test Script")
    print("  Python version:", sys.version.split()[0])
    print("█" * 60)

    try:
        success = test_direct_camera()

        print("\n" + "█" * 60)
        if success:
            print("  Test completed successfully!")
            sys.exit(0)
        else:
            print("  Test completed with failures")
            sys.exit(1)

    except Exception as e:
        print("\n" + "█" * 60)
        print(f"  Fatal error: {e}")
        logger.error("Fatal error in main", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
