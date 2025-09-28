#!/usr/bin/env python3
"""Vision Module Demo Script.

Demonstrates the complete vision pipeline including:
- Camera capture with error handling
- Real-time image processing
- Table, ball, and cue detection
- Calibration procedures
- Performance monitoring
- Visual feedback and debugging

Usage:
    python demo.py [options]

Options:
    --camera-id ID          Camera device ID (default: 0)
    --resolution WxH        Camera resolution (default: 1920x1080)
    --fps FPS              Target frame rate (default: 30)
    --debug                Enable debug mode with visualizations
    --calibrate            Run calibration procedure first
    --save-video PATH      Save processed video to file
    --config PATH          Load configuration from file
    --no-gui               Run without GUI (headless mode)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Add the backend to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from vision import VisionConfig, VisionModule, VisionStatistics
    from vision.calibration.camera import CameraCalibrator
    from vision.models import Ball, CueStick, DetectionResult, Table
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VisionDemo:
    """Complete vision module demonstration with real-time processing.

    Features:
    - Live camera feed processing
    - Real-time detection visualization
    - Performance monitoring
    - Calibration workflows
    - Statistics display
    """

    def __init__(self, args):
        """Initialize the demo with command line arguments."""
        self.args = args
        self.running = False

        # Create vision configuration
        self.vision_config = self._create_vision_config()

        # Initialize vision module
        self.vision = VisionModule(self.vision_config)

        # Video writer for saving output
        self.video_writer = None
        if args.save_video:
            self._setup_video_writer()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []

        logger.info("Vision demo initialized successfully")

    def _create_vision_config(self) -> dict[str, Any]:
        """Create vision configuration from command line arguments."""
        width, height = map(int, self.args.resolution.split("x"))

        config = {
            "camera_device_id": self.args.camera_id,
            "camera_resolution": (width, height),
            "camera_fps": self.args.fps,
            "target_fps": self.args.fps,
            "enable_threading": True,
            "enable_table_detection": True,
            "enable_ball_detection": True,
            "enable_cue_detection": True,
            "enable_tracking": True,
            "debug_mode": self.args.debug,
            "preprocessing_enabled": True,
            "enable_gpu": False,  # Can be enabled if GPU support is available
        }

        # Load additional config from file if specified
        if self.args.config:
            try:
                import json

                with open(self.args.config) as f:
                    file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {self.args.config}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        return config

    def _setup_video_writer(self):
        """Setup video writer for saving output."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width, height = self.vision_config["camera_resolution"]

            self.video_writer = cv2.VideoWriter(
                self.args.save_video, fourcc, self.args.fps, (width, height)
            )
            logger.info(f"Video will be saved to {self.args.save_video}")
        except Exception as e:
            logger.error(f"Failed to setup video writer: {e}")
            self.video_writer = None

    def run_calibration(self) -> bool:
        """Run the calibration procedure."""
        logger.info("Starting calibration procedure...")

        try:
            # Start camera for calibration
            if not self.vision.start_capture():
                logger.error("Failed to start camera for calibration")
                return False

            # Wait for camera to stabilize
            time.sleep(2)

            # Perform automatic calibration
            success = self.vision.calibrate_camera()

            if success:
                logger.info("Camera calibration completed successfully")

                # Perform color calibration with current frame
                frame = self.vision.get_current_frame()
                if frame is not None:
                    color_results = self.vision.calibrate_colors(frame)
                    if color_results:
                        logger.info("Color calibration completed successfully")
                    else:
                        logger.warning("Color calibration failed")

                return True
            else:
                logger.error("Camera calibration failed")
                return False

        except Exception as e:
            logger.error(f"Calibration procedure failed: {e}")
            return False
        finally:
            # Stop camera after calibration
            self.vision.stop_capture()

    def run_detection_demo(self):
        """Run the main detection demonstration."""
        logger.info("Starting vision detection demo...")

        try:
            # Start vision processing
            if not self.vision.start_capture():
                logger.error("Failed to start vision processing")
                return

            logger.info(
                "Vision processing started. Press 'q' to quit, 's' for statistics"
            )

            self.running = True
            self.start_time = time.time()

            while self.running:
                # Process frame
                start_frame_time = time.time()

                detection_result = self.vision.process_frame()

                if detection_result is not None:
                    self._process_detection_result(detection_result)

                # Update performance tracking
                frame_time = time.time() - start_frame_time
                self._update_performance_tracking(frame_time)

                # Handle GUI and user input
                if not self.args.no_gui and not self._handle_gui():
                    break

                # Small sleep to prevent overwhelming the system
                time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
        finally:
            self._cleanup()

    def _process_detection_result(self, result: DetectionResult):
        """Process and visualize detection results."""
        self.frame_count += 1

        # Get current frame for visualization
        frame = self.vision.get_current_frame()
        if frame is None:
            return

        # Create visualization
        if not self.args.no_gui or self.video_writer:
            vis_frame = self._create_visualization(frame, result)

            # Display frame if GUI enabled
            if not self.args.no_gui:
                cv2.imshow("Billiards Vision Demo", vis_frame)

            # Save to video if enabled
            if self.video_writer:
                self.video_writer.write(vis_frame)

        # Log interesting events
        self._log_detection_events(result)

    def _create_visualization(
        self, frame: np.ndarray, result: DetectionResult
    ) -> np.ndarray:
        """Create visualization overlay on frame."""
        vis_frame = frame.copy()

        # Draw table detection
        if result.table:
            self._draw_table(vis_frame, result.table)

        # Draw ball detections
        for ball in result.balls:
            self._draw_ball(vis_frame, ball)

        # Draw cue detection
        if result.cue:
            self._draw_cue(vis_frame, result.cue)

        # Draw performance info
        self._draw_performance_info(vis_frame, result)

        return vis_frame

    def _draw_table(self, frame: np.ndarray, table: Table):
        """Draw table detection overlay."""
        # Draw table corners
        corners = np.array(table.corners, dtype=np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

        # Draw pockets
        for pocket in table.pockets:
            center = (int(pocket[0]), int(pocket[1]))
            cv2.circle(frame, center, 15, (255, 0, 0), 2)

        # Draw table info
        cv2.putText(
            frame,
            f"Table: {table.width:.0f}x{table.height:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def _draw_ball(self, frame: np.ndarray, ball: Ball):
        """Draw ball detection overlay."""
        center = (int(ball.position[0]), int(ball.position[1]))
        radius = int(ball.radius)

        # Color based on ball type
        color_map = {
            "cue": (255, 255, 255),  # White
            "solid": (0, 255, 0),  # Green
            "stripe": (0, 255, 255),  # Yellow
            "eight": (0, 0, 0),  # Black
            "unknown": (128, 128, 128),  # Gray
        }
        color = color_map.get(ball.ball_type.value, (128, 128, 128))

        # Draw ball circle
        cv2.circle(frame, center, radius, color, 2)

        # Draw center point
        cv2.circle(frame, center, 3, color, -1)

        # Draw ball number if available
        if ball.number is not None:
            cv2.putText(
                frame,
                str(ball.number),
                (center[0] - 10, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Draw velocity vector if moving
        if ball.is_moving:
            end_x = int(center[0] + ball.velocity[0] * 0.1)
            end_y = int(center[1] + ball.velocity[1] * 0.1)
            cv2.arrowedLine(frame, center, (end_x, end_y), (255, 0, 255), 2)

    def _draw_cue(self, frame: np.ndarray, cue: CueStick):
        """Draw cue stick detection overlay."""
        tip = (int(cue.tip_position[0]), int(cue.tip_position[1]))

        # Calculate butt position from tip, angle, and length
        butt_x = tip[0] - cue.length * np.cos(np.radians(cue.angle))
        butt_y = tip[1] - cue.length * np.sin(np.radians(cue.angle))
        butt = (int(butt_x), int(butt_y))

        # Color based on state
        if cue.state.value == "striking":
            color = (0, 0, 255)  # Red
        elif cue.state.value == "aiming":
            color = (0, 255, 255)  # Yellow
        else:
            color = (128, 128, 128)  # Gray

        # Draw cue line
        cv2.line(frame, tip, butt, color, 3)

        # Draw tip marker
        cv2.circle(frame, tip, 8, color, -1)

        # Draw cue info
        info_text = f"Cue: {cue.angle:.1f}° {cue.state.value} ({cue.confidence:.2f})"
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw velocity vector if available
        if hasattr(cue, "tip_velocity") and cue.tip_velocity != (0.0, 0.0):
            vel_end_x = int(tip[0] + cue.tip_velocity[0] * 5)
            vel_end_y = int(tip[1] + cue.tip_velocity[1] * 5)
            cv2.arrowedLine(frame, tip, (vel_end_x, vel_end_y), (255, 255, 0), 2)

    def _draw_performance_info(self, frame: np.ndarray, result: DetectionResult):
        """Draw performance information overlay."""
        # FPS calculation
        current_fps = len(self.fps_history) / max(1, time.time() - self.start_time + 1)

        # Performance text
        perf_text = [
            f"FPS: {current_fps:.1f}",
            f"Frame: {self.frame_count}",
            f"Processing: {result.statistics.processing_time:.1f}ms",
            f"Balls: {len(result.balls)}",
            f"Cue: {'Yes' if result.cue else 'No'}",
            f"Table: {'Yes' if result.table else 'No'}",
        ]

        # Draw performance info
        y_offset = frame.shape[0] - 120
        for i, text in enumerate(perf_text):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def _log_detection_events(self, result: DetectionResult):
        """Log interesting detection events."""
        # Log shots
        if result.shot_detected:
            logger.info(f"Shot detected at frame {result.frame_number}")

        # Log significant ball movement
        moving_balls = [b for b in result.balls if b.is_moving]
        if moving_balls:
            logger.debug(f"Moving balls: {len(moving_balls)}")

        # Log cue state changes
        if result.cue and hasattr(result.cue, "state"):
            # Would track state changes here with previous state
            pass

    def _update_performance_tracking(self, frame_time: float):
        """Update performance tracking metrics."""
        self.fps_history.append(time.time())

        # Keep only recent history
        cutoff_time = time.time() - 5.0  # Last 5 seconds
        self.fps_history = [t for t in self.fps_history if t > cutoff_time]

    def _handle_gui(self) -> bool:
        """Handle GUI events and user input."""
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Quit requested by user")
            return False
        elif key == ord("s"):
            self._print_statistics()
        elif key == ord("c"):
            self._save_current_frame()
        elif key == ord("r"):
            self._reset_statistics()
        elif key == ord("h"):
            self._print_help()

        return True

    def _print_statistics(self):
        """Print current statistics."""
        stats = self.vision.get_statistics()

        print("\n" + "=" * 50)
        print("VISION MODULE STATISTICS")
        print("=" * 50)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Frames dropped: {stats['frames_dropped']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"Detection accuracy: {stats['detection_accuracy']}")
        print(f"Uptime: {stats['uptime_seconds']:.1f} seconds")
        print(f"Camera connected: {stats['camera_connected']}")

        if stats["last_error"]:
            print(f"Last error: {stats['last_error']}")

        print("=" * 50 + "\n")

    def _save_current_frame(self):
        """Save current frame to file."""
        frame = self.vision.get_current_frame()
        if frame is not None:
            timestamp = int(time.time())
            filename = f"vision_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Frame saved to {filename}")

    def _reset_statistics(self):
        """Reset performance statistics."""
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        logger.info("Statistics reset")

    def _print_help(self):
        """Print help information."""
        help_text = """
        VISION DEMO CONTROLS:

        q - Quit the demo
        s - Show statistics
        c - Capture current frame
        r - Reset statistics
        h - Show this help

        ESC - Quit the demo
        """
        print(help_text)

    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        self.running = False

        # Stop vision processing
        self.vision.stop_capture()

        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            logger.info("Video saved successfully")

        # Close GUI windows
        if not self.args.no_gui:
            cv2.destroyAllWindows()

        # Print final statistics
        self._print_statistics()

        logger.info("Demo completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Billiards Vision Module Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--camera-id", type=int, default=0, help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="Camera resolution WxH (default: 1920x1080)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frame rate (default: 30)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with extra visualizations",
    )
    parser.add_argument(
        "--calibrate", action="store_true", help="Run calibration procedure first"
    )
    parser.add_argument(
        "--save-video", metavar="PATH", help="Save processed video to file"
    )
    parser.add_argument(
        "--config", metavar="PATH", help="Load configuration from JSON file"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Run without GUI (headless mode)"
    )

    args = parser.parse_args()

    # Validate arguments
    try:
        width, height = map(int, args.resolution.split("x"))
        if width <= 0 or height <= 0:
            raise ValueError("Invalid resolution")
    except ValueError:
        print("Error: Invalid resolution format. Use WxH (e.g., 1920x1080)")
        return 1

    if args.fps <= 0:
        print("Error: FPS must be positive")
        return 1

    # Create and run demo
    try:
        demo = VisionDemo(args)

        # Run calibration if requested
        if args.calibrate and not demo.run_calibration():
            print("Calibration failed. Continuing with demo...")

        # Run main demo
        demo.run_detection_demo()

        return 0

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
