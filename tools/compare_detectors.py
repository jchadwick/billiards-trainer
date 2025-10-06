#!/usr/bin/env python3
"""Visual detector comparison tool for billiards vision system.

This tool provides side-by-side comparison of YOLO and OpenCV ball detectors.
It processes test videos and displays results in real-time or saves comparison videos.

Features:
- Load and process test videos
- Run YOLO and OpenCV detectors simultaneously
- Display results side-by-side in GUI or saved video
- Highlight differences (different detections, position variances)
- Save comparison video with annotations
- Performance metrics and statistics

Usage:
    # Compare both detectors on video
    python compare_detectors.py --video test_video.mp4 --detector both --output comparison.mp4

    # Show only YOLO results
    python compare_detectors.py --video test_video.mp4 --detector yolo --display

    # Compare with custom config
    python compare_detectors.py --video test_video.mp4 --detector both --config config.json

Requirements:
    - OpenCV (cv2)
    - NumPy
    - Vision detection modules (BallDetector, YOLODetector)
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vision.detection.balls import BallDetector
from vision.detection.yolo_detector import YOLODetector
from vision.models import Ball

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionComparison:
    """Comparison data for a single frame."""

    frame_number: int
    timestamp: float

    # OpenCV results
    opencv_balls: list[Ball]
    opencv_time: float  # milliseconds

    # YOLO results
    yolo_balls: list[Ball]
    yolo_time: float  # milliseconds

    # Comparison metrics
    opencv_count: int = 0
    yolo_count: int = 0
    matched_balls: int = 0  # Balls detected by both
    opencv_only: int = 0  # Balls only in OpenCV
    yolo_only: int = 0  # Balls only in YOLO
    position_differences: list[float] = field(
        default_factory=list
    )  # Pixel distances for matched balls
    avg_position_diff: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        self.opencv_count = len(self.opencv_balls)
        self.yolo_count = len(self.yolo_balls)


@dataclass
class ComparisonStatistics:
    """Overall comparison statistics."""

    total_frames: int = 0
    opencv_avg_time: float = 0.0
    yolo_avg_time: float = 0.0
    opencv_avg_balls: float = 0.0
    yolo_avg_balls: float = 0.0
    avg_matched_balls: float = 0.0
    avg_position_diff: float = 0.0
    opencv_fps: float = 0.0
    yolo_fps: float = 0.0


class DetectorComparator:
    """Compares YOLO and OpenCV ball detectors side-by-side.

    Processes video frames through both detectors and provides visual comparison
    with performance metrics and detection accuracy analysis.
    """

    def __init__(
        self,
        opencv_config: Optional[dict[str, Any]] = None,
        yolo_model_path: Optional[str] = None,
        yolo_config: Optional[dict[str, Any]] = None,
        match_threshold: float = 30.0,
    ):
        """Initialize detector comparator.

        Args:
            opencv_config: Configuration for OpenCV detector
            yolo_model_path: Path to YOLO model file
            yolo_config: Configuration for YOLO detector
            match_threshold: Maximum pixel distance to consider balls as matched
        """
        self.match_threshold = match_threshold

        # Initialize OpenCV detector
        logger.info("Initializing OpenCV detector...")
        opencv_cfg = opencv_config or {}
        self.opencv_detector = BallDetector(opencv_cfg)

        # Initialize YOLO detector
        logger.info("Initializing YOLO detector...")
        yolo_cfg = yolo_config or {}
        self.yolo_detector = YOLODetector(
            model_path=yolo_model_path,
            device=yolo_cfg.get("device", "cpu"),
            confidence=yolo_cfg.get("confidence", 0.4),
            nms_threshold=yolo_cfg.get("nms_threshold", 0.45),
            auto_fallback=True,
        )

        # Statistics
        self.comparisons: list[DetectionComparison] = []
        self.stats = ComparisonStatistics()

        logger.info(f"YOLO model available: {self.yolo_detector.is_available()}")

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set background reference frame for both detectors.

        Args:
            frame: Background frame (empty table)
        """
        logger.info("Setting background frame for both detectors")
        self.opencv_detector.set_background_frame(frame)
        # YOLO doesn't use background subtraction

    def compare_frame(
        self, frame: NDArray[np.uint8], frame_number: int, timestamp: float
    ) -> DetectionComparison:
        """Compare detectors on a single frame.

        Args:
            frame: Input frame
            frame_number: Frame index
            timestamp: Frame timestamp

        Returns:
            Comparison results
        """
        # Run OpenCV detector
        start_time = time.time()
        opencv_balls = self.opencv_detector.detect_balls(frame)
        opencv_time = (time.time() - start_time) * 1000

        # Run YOLO detector
        yolo_balls = []
        if self.yolo_detector.is_available():
            start_time = time.time()
            yolo_detections = self.yolo_detector.detect_balls(frame)
            yolo_time = (time.time() - start_time) * 1000

            # Convert YOLO detections to Ball objects
            yolo_balls = self._convert_yolo_detections(yolo_detections)
        else:
            yolo_time = 0.0

        # Create comparison
        comparison = DetectionComparison(
            frame_number=frame_number,
            timestamp=timestamp,
            opencv_balls=opencv_balls,
            opencv_time=opencv_time,
            yolo_balls=yolo_balls,
            yolo_time=yolo_time,
        )

        # Analyze matches and differences
        self._analyze_comparison(comparison)

        # Store comparison
        self.comparisons.append(comparison)

        return comparison

    def _convert_yolo_detections(self, detections: list[Any]) -> list[Ball]:
        """Convert YOLO Detection objects to Ball objects.

        Args:
            detections: List of YOLO Detection objects

        Returns:
            List of Ball objects
        """
        from vision.detection.yolo_detector import ball_class_to_type
        from vision.models import BallType

        balls = []
        for det in detections:
            # Get ball type from class ID
            ball_type_str, number = ball_class_to_type(det.class_id)

            # Convert string to BallType enum
            ball_type_map = {
                "cue": BallType.CUE,
                "solid": BallType.SOLID,
                "stripe": BallType.STRIPE,
                "eight": BallType.EIGHT,
                "unknown": BallType.UNKNOWN,
            }
            ball_type = ball_type_map.get(ball_type_str, BallType.UNKNOWN)

            # Estimate radius from bounding box
            radius = min(det.width, det.height) / 2.0

            ball = Ball(
                position=det.center,
                radius=radius,
                ball_type=ball_type,
                number=number,
                confidence=det.confidence,
            )
            balls.append(ball)

        return balls

    def _analyze_comparison(self, comparison: DetectionComparison) -> None:
        """Analyze comparison to find matches and differences.

        Args:
            comparison: Comparison data to analyze
        """
        opencv_balls = comparison.opencv_balls
        yolo_balls = comparison.yolo_balls

        # Match balls between detectors using Hungarian algorithm (greedy approximation)
        matched = set()
        yolo_matched = set()
        position_diffs = []

        for i, opencv_ball in enumerate(opencv_balls):
            best_match = None
            best_distance = float("inf")

            for j, yolo_ball in enumerate(yolo_balls):
                if j in yolo_matched:
                    continue

                # Calculate distance
                distance = np.sqrt(
                    (opencv_ball.position[0] - yolo_ball.position[0]) ** 2
                    + (opencv_ball.position[1] - yolo_ball.position[1]) ** 2
                )

                if distance < best_distance and distance < self.match_threshold:
                    best_distance = distance
                    best_match = j

            if best_match is not None:
                matched.add(i)
                yolo_matched.add(best_match)
                position_diffs.append(best_distance)

        # Calculate metrics
        comparison.matched_balls = len(matched)
        comparison.opencv_only = comparison.opencv_count - len(matched)
        comparison.yolo_only = comparison.yolo_count - len(yolo_matched)
        comparison.position_differences = position_diffs
        comparison.avg_position_diff = (
            np.mean(position_diffs) if position_diffs else 0.0
        )

    def visualize_comparison(
        self,
        frame: NDArray[np.uint8],
        comparison: DetectionComparison,
        show_metrics: bool = True,
    ) -> NDArray[np.uint8]:
        """Create side-by-side visualization of detector results.

        Args:
            frame: Original frame
            comparison: Comparison data
            show_metrics: Whether to show performance metrics

        Returns:
            Composite image with both detectors' results
        """
        # Create two copies of the frame
        opencv_frame = frame.copy()
        yolo_frame = frame.copy()

        # Draw OpenCV detections (green)
        for ball in comparison.opencv_balls:
            x, y = int(ball.position[0]), int(ball.position[1])
            r = int(ball.radius)
            cv2.circle(opencv_frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(opencv_frame, (x, y), 3, (0, 255, 0), -1)

            # Label with type and confidence
            label = f"{ball.ball_type.value}"
            if ball.number:
                label += f":{ball.number}"
            label += f" {ball.confidence:.2f}"

            cv2.putText(
                opencv_frame,
                label,
                (x - r, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        # Draw YOLO detections (blue)
        for ball in comparison.yolo_balls:
            x, y = int(ball.position[0]), int(ball.position[1])
            r = int(ball.radius)
            cv2.circle(yolo_frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(yolo_frame, (x, y), 3, (255, 0, 0), -1)

            # Label with type and confidence
            label = f"{ball.ball_type.value}"
            if ball.number:
                label += f":{ball.number}"
            label += f" {ball.confidence:.2f}"

            cv2.putText(
                yolo_frame,
                label,
                (x - r, y - r - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )

        # Add titles
        cv2.putText(
            opencv_frame,
            "OpenCV Detector",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            yolo_frame,
            "YOLO Detector",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

        # Add metrics if requested
        if show_metrics:
            metrics_y = 60
            metrics = [
                f"Balls: {comparison.opencv_count}",
                f"Time: {comparison.opencv_time:.1f}ms",
            ]
            for i, metric in enumerate(metrics):
                cv2.putText(
                    opencv_frame,
                    metric,
                    (10, metrics_y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            metrics = [
                f"Balls: {comparison.yolo_count}",
                f"Time: {comparison.yolo_time:.1f}ms",
            ]
            for i, metric in enumerate(metrics):
                cv2.putText(
                    yolo_frame,
                    metric,
                    (10, metrics_y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        # Stack frames horizontally
        combined = np.hstack([opencv_frame, yolo_frame])

        # Add comparison metrics at bottom
        if show_metrics:
            metrics_text = (
                f"Frame {comparison.frame_number} | "
                f"Matched: {comparison.matched_balls} | "
                f"OpenCV only: {comparison.opencv_only} | "
                f"YOLO only: {comparison.yolo_only}"
            )
            if comparison.avg_position_diff > 0:
                metrics_text += f" | Avg diff: {comparison.avg_position_diff:.1f}px"

            cv2.putText(
                combined,
                metrics_text,
                (10, combined.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1,
            )

        return combined

    def calculate_statistics(self) -> ComparisonStatistics:
        """Calculate overall comparison statistics.

        Returns:
            Statistics object
        """
        if not self.comparisons:
            return self.stats

        stats = ComparisonStatistics()
        stats.total_frames = len(self.comparisons)

        # Calculate averages
        opencv_times = [c.opencv_time for c in self.comparisons]
        yolo_times = [c.yolo_time for c in self.comparisons]
        opencv_counts = [c.opencv_count for c in self.comparisons]
        yolo_counts = [c.yolo_count for c in self.comparisons]
        matched_counts = [c.matched_balls for c in self.comparisons]
        position_diffs = [
            c.avg_position_diff for c in self.comparisons if c.avg_position_diff > 0
        ]

        stats.opencv_avg_time = np.mean(opencv_times)
        stats.yolo_avg_time = np.mean(yolo_times)
        stats.opencv_avg_balls = np.mean(opencv_counts)
        stats.yolo_avg_balls = np.mean(yolo_counts)
        stats.avg_matched_balls = np.mean(matched_counts)
        stats.avg_position_diff = np.mean(position_diffs) if position_diffs else 0.0

        # Calculate FPS (frames per second each detector could handle)
        stats.opencv_fps = 1000.0 / stats.opencv_avg_time if stats.opencv_avg_time > 0 else 0
        stats.yolo_fps = 1000.0 / stats.yolo_avg_time if stats.yolo_avg_time > 0 else 0

        self.stats = stats
        return stats

    def print_statistics(self) -> None:
        """Print comparison statistics to console."""
        stats = self.calculate_statistics()

        print("\n" + "=" * 70)
        print("DETECTOR COMPARISON STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {stats.total_frames}")
        print()
        print("OpenCV Detector:")
        print(f"  Average processing time: {stats.opencv_avg_time:.2f}ms")
        print(f"  Theoretical max FPS: {stats.opencv_fps:.1f}")
        print(f"  Average balls detected: {stats.opencv_avg_balls:.1f}")
        print()
        print("YOLO Detector:")
        print(f"  Average processing time: {stats.yolo_avg_time:.2f}ms")
        print(f"  Theoretical max FPS: {stats.yolo_fps:.1f}")
        print(f"  Average balls detected: {stats.yolo_avg_balls:.1f}")
        print()
        print("Comparison:")
        print(f"  Average matched balls: {stats.avg_matched_balls:.1f}")
        print(f"  Average position difference: {stats.avg_position_diff:.2f} pixels")
        print()

        # Performance comparison
        if stats.opencv_avg_time > 0 and stats.yolo_avg_time > 0:
            speedup = stats.opencv_avg_time / stats.yolo_avg_time
            faster = "YOLO" if speedup > 1 else "OpenCV"
            ratio = max(speedup, 1.0 / speedup)
            print(f"  {faster} is {ratio:.2f}x faster")
        print("=" * 70 + "\n")

    def save_statistics_json(self, output_path: str) -> None:
        """Save statistics to JSON file.

        Args:
            output_path: Output JSON file path
        """
        stats = self.calculate_statistics()

        data = {
            "total_frames": stats.total_frames,
            "opencv": {
                "avg_time_ms": stats.opencv_avg_time,
                "max_fps": stats.opencv_fps,
                "avg_balls_detected": stats.opencv_avg_balls,
            },
            "yolo": {
                "avg_time_ms": stats.yolo_avg_time,
                "max_fps": stats.yolo_fps,
                "avg_balls_detected": stats.yolo_avg_balls,
            },
            "comparison": {
                "avg_matched_balls": stats.avg_matched_balls,
                "avg_position_diff_pixels": stats.avg_position_diff,
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Statistics saved to {output_path}")


def process_video(
    video_path: str,
    detector_mode: str,
    output_path: Optional[str] = None,
    display: bool = False,
    opencv_config: Optional[dict] = None,
    yolo_model_path: Optional[str] = None,
    yolo_config: Optional[dict] = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    background_frame_path: Optional[str] = None,
) -> Optional[DetectorComparator]:
    """Process video with detector comparison.

    Args:
        video_path: Path to input video
        detector_mode: 'opencv', 'yolo', or 'both'
        output_path: Path to save output video (optional)
        display: Whether to display results in real-time
        opencv_config: OpenCV detector configuration
        yolo_model_path: Path to YOLO model
        yolo_config: YOLO detector configuration
        start_frame: Frame to start processing from
        max_frames: Maximum number of frames to process
        background_frame_path: Path to background frame image (optional)

    Returns:
        DetectorComparator if mode is 'both', else None
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path}")
    logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

    # Initialize comparator for 'both' mode
    comparator = None
    if detector_mode == "both":
        comparator = DetectorComparator(
            opencv_config=opencv_config,
            yolo_model_path=yolo_model_path,
            yolo_config=yolo_config,
        )

        # Set background frame if provided
        if background_frame_path:
            bg_frame = cv2.imread(background_frame_path)
            if bg_frame is not None:
                comparator.set_background_frame(bg_frame)
            else:
                logger.warning(f"Failed to load background frame: {background_frame_path}")

    # Initialize single detector for 'opencv' or 'yolo' mode
    elif detector_mode == "opencv":
        detector = BallDetector(opencv_config or {})
        if background_frame_path:
            bg_frame = cv2.imread(background_frame_path)
            if bg_frame is not None:
                detector.set_background_frame(bg_frame)
    elif detector_mode == "yolo":
        yolo_cfg = yolo_config or {}
        detector = YOLODetector(
            model_path=yolo_model_path,
            device=yolo_cfg.get("device", "cpu"),
            confidence=yolo_cfg.get("confidence", 0.4),
            nms_threshold=yolo_cfg.get("nms_threshold", 0.45),
        )

    # Setup video writer if output requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_width = width * 2 if detector_mode == "both" else width
        writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
        logger.info(f"Output will be saved to: {output_path}")

    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        logger.info(f"Starting from frame {start_frame}")

    # Process frames
    frame_count = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and processed_count >= max_frames:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Process frame
            if detector_mode == "both":
                comparison = comparator.compare_frame(frame, frame_count, timestamp)
                vis_frame = comparator.visualize_comparison(frame, comparison)
            else:
                # Single detector mode
                vis_frame = frame.copy()
                start_time = time.time()

                if detector_mode == "opencv":
                    balls = detector.detect_balls(frame)
                    color = (0, 255, 0)  # Green
                    title = "OpenCV Detector"
                else:  # yolo
                    detections = detector.detect_balls(frame)
                    balls = comparator._convert_yolo_detections(detections) if comparator else []
                    color = (255, 0, 0)  # Blue
                    title = "YOLO Detector"

                proc_time = (time.time() - start_time) * 1000

                # Draw detections
                for ball in balls:
                    x, y = int(ball.position[0]), int(ball.position[1])
                    r = int(ball.radius)
                    cv2.circle(vis_frame, (x, y), r, color, 2)
                    cv2.circle(vis_frame, (x, y), 3, color, -1)

                # Add title and metrics
                cv2.putText(
                    vis_frame,
                    title,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )
                cv2.putText(
                    vis_frame,
                    f"Balls: {len(balls)} | Time: {proc_time:.1f}ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Write frame to output
            if writer:
                writer.write(vis_frame)

            # Display frame
            if display:
                cv2.imshow("Detector Comparison", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User requested quit")
                    break
                elif key == ord(" "):
                    # Pause on space
                    cv2.waitKey(0)

            processed_count += 1
            if processed_count % 30 == 0:
                logger.info(f"Processed {processed_count} frames...")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        logger.info(f"Processed {processed_count} frames total")

    return comparator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare YOLO and OpenCV ball detectors visually",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both detectors on video
  python compare_detectors.py --video test.mp4 --detector both --output comparison.mp4

  # Show only OpenCV results with display
  python compare_detectors.py --video test.mp4 --detector opencv --display

  # Compare with YOLO model and background frame
  python compare_detectors.py --video test.mp4 --detector both \\
      --yolo-model models/billiards.pt --background empty_table.jpg --output out.mp4

  # Process subset of frames
  python compare_detectors.py --video test.mp4 --detector both \\
      --start-frame 100 --max-frames 300 --display
        """,
    )

    parser.add_argument(
        "--video", required=True, help="Path to input video file"
    )

    parser.add_argument(
        "--detector",
        choices=["opencv", "yolo", "both"],
        default="both",
        help="Which detector(s) to use (default: both)",
    )

    parser.add_argument(
        "--output", help="Path to save output video (optional)"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display results in real-time window",
    )

    parser.add_argument(
        "--config", help="Path to JSON config file for OpenCV detector"
    )

    parser.add_argument(
        "--yolo-model", help="Path to YOLO model file (.pt or .onnx)"
    )

    parser.add_argument(
        "--yolo-config", help="Path to JSON config file for YOLO detector"
    )

    parser.add_argument(
        "--background", help="Path to background frame image (empty table)"
    )

    parser.add_argument(
        "--start-frame", type=int, default=0, help="Frame to start processing from"
    )

    parser.add_argument(
        "--max-frames", type=int, help="Maximum number of frames to process"
    )

    parser.add_argument(
        "--stats-json", help="Path to save statistics JSON file"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configs
    opencv_config = None
    if args.config:
        with open(args.config) as f:
            opencv_config = json.load(f)

    yolo_config = None
    if args.yolo_config:
        with open(args.yolo_config) as f:
            yolo_config = json.load(f)

    # Validate inputs
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        return 1

    if args.yolo_model and not Path(args.yolo_model).exists():
        logger.error(f"YOLO model file not found: {args.yolo_model}")
        return 1

    if args.background and not Path(args.background).exists():
        logger.error(f"Background frame not found: {args.background}")
        return 1

    # Process video
    comparator = process_video(
        video_path=args.video,
        detector_mode=args.detector,
        output_path=args.output,
        display=args.display,
        opencv_config=opencv_config,
        yolo_model_path=args.yolo_model,
        yolo_config=yolo_config,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        background_frame_path=args.background,
    )

    # Print and save statistics if in comparison mode
    if comparator:
        comparator.print_statistics()

        if args.stats_json:
            comparator.save_statistics_json(args.stats_json)

    logger.info("Processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
