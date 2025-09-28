#!/usr/bin/env python3
"""Camera Capture Demo Script.

Demonstrates the camera capture interface functionality including:
- Camera initialization with various configurations
- Real-time frame capture and display
- Health monitoring and statistics
- Error handling and recovery
- Multi-backend support testing

Usage:
    python demo_capture.py [--device-id ID] [--backend BACKEND] [--resolution WIDTHxHEIGHT] [--fps FPS]

Examples:
    python demo_capture.py --device-id 0 --backend auto --resolution 640x480 --fps 30
    python demo_capture.py --backend v4l2 --resolution 1920x1080 --fps 60
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Add the backend path to import vision modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vision.capture import CameraCapture, CameraHealth, CameraStatus
    from vision.models import FrameInfo
except ImportError as e:
    print(f"Error importing vision modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CameraDemonstration:
    """Demonstration class for camera capture functionality."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the demonstration with camera configuration."""
        self.config = config
        self.capture: Optional[CameraCapture] = None
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0

        # Statistics
        self.total_frames = 0
        self.total_errors = 0
        self.fps_history = []

        # Display window name
        self.window_name = "Camera Capture Demo"

    def status_callback(self, status: CameraStatus) -> None:
        """Handle camera status changes."""
        status_messages = {
            CameraStatus.DISCONNECTED: "Camera disconnected",
            CameraStatus.CONNECTING: "Connecting to camera...",
            CameraStatus.CONNECTED: "Camera connected successfully",
            CameraStatus.ERROR: "Camera error occurred",
            CameraStatus.RECONNECTING: "Attempting to reconnect...",
        }

        message = status_messages.get(status, f"Unknown status: {status}")
        logger.info(f"Camera status: {message}")

        # Change window title to reflect status
        try:
            cv2.setWindowTitle(self.window_name, f"Camera Demo - {message}")
        except cv2.error:
            pass  # Window might not exist yet

    def create_info_overlay(
        self, frame: np.ndarray, health: CameraHealth, frame_info: FrameInfo
    ) -> np.ndarray:
        """Create an informational overlay on the frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)

        # Text color
        color = (0, 255, 0) if health.status == CameraStatus.CONNECTED else (0, 0, 255)

        # Display information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 20
        y_start = 35

        info_lines = [
            f"Frame: {frame_info.frame_number}",
            f"Resolution: {frame_info.size[0]}x{frame_info.size[1]}",
            f"FPS: {health.fps:.1f}",
            f"Status: {health.status.value}",
            f"Captured: {health.frames_captured}",
            f"Dropped: {health.frames_dropped}",
            f"Errors: {health.error_count}",
            f"Uptime: {health.uptime:.1f}s",
        ]

        for i, line in enumerate(info_lines):
            y = y_start + i * line_height
            cv2.putText(overlay, line, (20, y), font, font_scale, color, thickness)

        # Add last error if present
        if health.last_error:
            error_y = y_start + len(info_lines) * line_height
            cv2.putText(
                overlay,
                f"Last Error: {health.last_error[:40]}...",
                (20, error_y),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
            )

        return overlay

    def draw_crosshair(self, frame: np.ndarray) -> np.ndarray:
        """Draw a crosshair in the center of the frame."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Draw crosshair
        cv2.line(
            frame,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (0, 255, 255),
            2,
        )
        cv2.line(
            frame,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (0, 255, 255),
            2,
        )
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 255), 2)

        return frame

    def draw_frame_border(self, frame: np.ndarray, health: CameraHealth) -> np.ndarray:
        """Draw a colored border based on camera health."""
        h, w = frame.shape[:2]

        # Border color based on status
        border_colors = {
            CameraStatus.CONNECTED: (0, 255, 0),  # Green
            CameraStatus.CONNECTING: (0, 255, 255),  # Yellow
            CameraStatus.RECONNECTING: (0, 165, 255),  # Orange
            CameraStatus.ERROR: (0, 0, 255),  # Red
            CameraStatus.DISCONNECTED: (128, 128, 128),  # Gray
        }

        color = border_colors.get(health.status, (255, 255, 255))
        thickness = 3

        # Draw border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

        return frame

    def test_camera_backends(self) -> None:
        """Test different camera backends."""
        backends = ["auto", "v4l2", "dshow", "gstreamer", "opencv"]

        print("\n=== Testing Camera Backends ===")

        for backend in backends:
            print(f"\nTesting backend: {backend}")

            test_config = self.config.copy()
            test_config["backend"] = backend
            test_config["auto_reconnect"] = False  # Don't auto-reconnect for tests

            test_capture = CameraCapture(test_config)
            test_capture.set_status_callback(lambda s: print(f"  Status: {s.value}"))

            if test_capture.start_capture():
                time.sleep(1.0)  # Let it capture a few frames
                health = test_capture.get_health()
                info = test_capture.get_camera_info()

                print(f"  ✓ Backend {backend} working")
                print(f"    Frames captured: {health.frames_captured}")
                print(f"    FPS: {health.fps:.1f}")
                print(f"    Resolution: {info.get('resolution', 'Unknown')}")

                test_capture.stop_capture()
            else:
                print(f"  ✗ Backend {backend} failed")

            time.sleep(0.5)  # Brief pause between tests

    def run_live_demo(self) -> None:
        """Run the live camera demonstration."""
        print("\n=== Starting Live Camera Demo ===")
        print(f"Configuration: {self.config}")
        print("Press 'q' to quit, 's' for statistics, 'h' for health info")

        # Initialize camera
        self.capture = CameraCapture(self.config)
        self.capture.set_status_callback(self.status_callback)

        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_RESIZABLE)

        if not self.capture.start_capture():
            logger.error("Failed to start camera capture")
            return

        self.running = True
        self.start_time = time.time()

        try:
            while self.running:
                # Get latest frame
                frame_data = self.capture.get_latest_frame()

                if frame_data is None:
                    # No frame available, show waiting message
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        waiting_frame,
                        "Waiting for camera...",
                        (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(self.window_name, waiting_frame)
                else:
                    frame, frame_info = frame_data
                    self.total_frames += 1

                    # Get health information
                    health = self.capture.get_health()

                    # Process frame for display
                    display_frame = frame.copy()
                    display_frame = self.draw_crosshair(display_frame)
                    display_frame = self.draw_frame_border(display_frame, health)
                    display_frame = self.create_info_overlay(
                        display_frame, health, frame_info
                    )

                    # Update FPS calculation
                    if len(self.fps_history) >= 30:
                        self.fps_history.pop(0)
                    self.fps_history.append(time.time())

                    # Display frame
                    cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Quit requested by user")
                    self.running = False
                elif key == ord("s"):
                    self.print_statistics()
                elif key == ord("h"):
                    self.print_health_info()
                elif key == ord("r"):
                    self.test_reconnection()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def test_reconnection(self) -> None:
        """Test camera reconnection by stopping and starting."""
        print("\n=== Testing Reconnection ===")

        if self.capture:
            print("Stopping camera...")
            self.capture.stop_capture()

            time.sleep(2.0)

            print("Restarting camera...")
            if self.capture.start_capture():
                print("Reconnection successful")
            else:
                print("Reconnection failed")

    def print_statistics(self) -> None:
        """Print detailed statistics."""
        if not self.capture:
            return

        health = self.capture.get_health()
        info = self.capture.get_camera_info()
        elapsed = time.time() - self.start_time

        print("\n=== Camera Statistics ===")
        print(f"Runtime: {elapsed:.1f}s")
        print(f"Total frames: {self.total_frames}")
        print(f"Average FPS: {self.total_frames / elapsed:.1f}")
        print(f"Capture FPS: {health.fps:.1f}")
        print(f"Frames captured: {health.frames_captured}")
        print(f"Frames dropped: {health.frames_dropped}")
        print(
            f"Drop rate: {health.frames_dropped / max(health.frames_captured, 1) * 100:.1f}%"
        )
        print(f"Error count: {health.error_count}")
        print(f"Connection attempts: {health.connection_attempts}")
        print(f"Camera info: {info}")

    def print_health_info(self) -> None:
        """Print detailed health information."""
        if not self.capture:
            return

        health = self.capture.get_health()

        print("\n=== Camera Health ===")
        print(f"Status: {health.status.value}")
        print(f"FPS: {health.fps:.1f}")
        print(f"Uptime: {health.uptime:.1f}s")
        print(f"Last frame: {time.time() - health.last_frame_time:.1f}s ago")
        print(f"Error count: {health.error_count}")
        if health.last_error:
            print(f"Last error: {health.last_error}")

    def cleanup(self) -> None:
        """Clean up resources."""
        print("\nCleaning up...")

        if self.capture:
            self.capture.stop_capture()

        cv2.destroyAllWindows()

        # Print final statistics
        self.print_statistics()


def parse_resolution(resolution_str: str) -> tuple:
    """Parse resolution string like '640x480' into tuple."""
    try:
        width, height = resolution_str.split("x")
        return (int(width), int(height))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"Invalid resolution format: {resolution_str}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Camera Capture Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --device-id 0 --backend auto --resolution 640x480 --fps 30
  %(prog)s --backend v4l2 --resolution 1920x1080 --fps 60
  %(prog)s --test-backends
        """,
    )

    parser.add_argument(
        "--device-id", type=int, default=0, help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "v4l2", "dshow", "gstreamer", "opencv"],
        default="auto",
        help="Camera backend (default: auto)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(640, 480),
        help="Camera resolution WIDTHxHEIGHT (default: 640x480)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frame rate (default: 30)"
    )
    parser.add_argument(
        "--exposure-mode",
        choices=["auto", "manual"],
        default="auto",
        help="Exposure mode (default: auto)",
    )
    parser.add_argument(
        "--exposure-value", type=float, help="Manual exposure value (0.0-1.0)"
    )
    parser.add_argument(
        "--gain", type=float, default=1.0, help="Camera gain (default: 1.0)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1, help="Frame buffer size (default: 1)"
    )
    parser.add_argument(
        "--no-reconnect", action="store_true", help="Disable automatic reconnection"
    )
    parser.add_argument(
        "--test-backends",
        action="store_true",
        help="Test all available backends and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build configuration
    config = {
        "device_id": args.device_id,
        "backend": args.backend,
        "resolution": args.resolution,
        "fps": args.fps,
        "exposure_mode": args.exposure_mode,
        "gain": args.gain,
        "buffer_size": args.buffer_size,
        "auto_reconnect": not args.no_reconnect,
        "reconnect_delay": 1.0,
        "max_reconnect_attempts": 5,
    }

    if args.exposure_value is not None:
        config["exposure_value"] = args.exposure_value

    # Create demonstration instance
    demo = CameraDemonstration(config)

    try:
        if args.test_backends:
            demo.test_camera_backends()
        else:
            demo.run_live_demo()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
