#!/usr/bin/env python3
"""Performance benchmark tool comparing YOLO vs OpenCV ball detection.

This tool provides comprehensive benchmarking capabilities to compare the performance,
accuracy, and quality of YOLO-based deep learning detection versus traditional OpenCV
computer vision detection for billiards ball detection.

Features:
- Load test video or image sequence for benchmarking
- Run both YOLO and OpenCV detectors on identical frames
- Measure FPS, latency, and throughput
- Calculate detection accuracy metrics
- Count false positives and false negatives
- Generate visual comparison outputs
- Export detailed results in JSON and text formats
- Optional ground truth comparison
- Optional visualization with side-by-side comparison

Requirements:
- Test video file or image sequence
- Optionally: YOLO model file (.pt or .onnx)
- Optionally: Ground truth annotations for accuracy metrics

Usage:
    # Basic benchmark with video
    python benchmark_detectors.py --video test_video.mp4

    # With YOLO model
    python benchmark_detectors.py --video test_video.mp4 --yolo-model models/billiards_yolo.pt

    # Limited frames with visualization
    python benchmark_detectors.py --video test_video.mp4 --frames 100 --visualize

    # With ground truth annotations
    python benchmark_detectors.py --video test_video.mp4 --ground-truth annotations.json

    # Save visualizations to directory
    python benchmark_detectors.py --video test_video.mp4 --visualize --output-dir results/

Output:
- Console: Real-time progress and summary table
- JSON: models/benchmark_results.json (detailed metrics)
- Text: models/benchmark_report.txt (human-readable report)
- Optional: Visualization frames saved to output directory
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vision.detection.balls import BallDetector, BallDetectionConfig
from vision.detection.yolo_detector import YOLODetector
from vision.models import Ball

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Detection performance metrics for a single detector."""

    detector_name: str
    total_frames: int = 0
    total_detections: int = 0
    total_processing_time: float = 0.0  # seconds
    frame_times: list[float] = None  # individual frame processing times
    detection_counts: list[int] = None  # detections per frame

    # Accuracy metrics (require ground truth)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Performance metrics
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    median_fps: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Quality metrics
    avg_detections_per_frame: float = 0.0
    detection_variance: float = 0.0  # Consistency metric
    avg_confidence: float = 0.0

    def __post_init__(self):
        """Initialize lists."""
        if self.frame_times is None:
            self.frame_times = []
        if self.detection_counts is None:
            self.detection_counts = []

    def calculate_statistics(self):
        """Calculate derived statistics from raw measurements."""
        if not self.frame_times:
            return

        # Convert times to FPS
        fps_values = [1.0 / t if t > 0 else 0 for t in self.frame_times]

        # FPS statistics
        self.avg_fps = np.mean(fps_values) if fps_values else 0.0
        self.min_fps = np.min(fps_values) if fps_values else 0.0
        self.max_fps = np.max(fps_values) if fps_values else 0.0
        self.median_fps = np.median(fps_values) if fps_values else 0.0

        # Latency statistics (milliseconds)
        latencies_ms = [t * 1000 for t in self.frame_times]
        self.avg_latency_ms = np.mean(latencies_ms) if latencies_ms else 0.0
        self.min_latency_ms = np.min(latencies_ms) if latencies_ms else 0.0
        self.max_latency_ms = np.max(latencies_ms) if latencies_ms else 0.0

        # Detection statistics
        if self.detection_counts:
            self.avg_detections_per_frame = np.mean(self.detection_counts)
            self.detection_variance = np.var(self.detection_counts)

        # Accuracy metrics (if available)
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (
                self.true_positives + self.false_positives
            )
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (
                self.true_positives + self.false_negatives
            )
        if self.precision + self.recall > 0:
            self.f1_score = (
                2 * (self.precision * self.recall) / (self.precision + self.recall)
            )


@dataclass
class BenchmarkResults:
    """Complete benchmark results comparing detectors."""

    opencv_metrics: DetectionMetrics
    yolo_metrics: Optional[DetectionMetrics] = None
    test_video_path: str = ""
    test_video_fps: float = 0.0
    test_video_frames: int = 0
    benchmark_date: str = ""
    ground_truth_available: bool = False

    # Comparative analysis
    yolo_speedup_factor: float = 0.0  # YOLO FPS / OpenCV FPS
    yolo_accuracy_gain: float = 0.0  # YOLO F1 - OpenCV F1
    winner: str = ""  # "yolo", "opencv", or "tie"

    def calculate_comparison(self):
        """Calculate comparative metrics."""
        if self.yolo_metrics is None:
            self.winner = "opencv"
            return

        # Speed comparison
        if self.opencv_metrics.avg_fps > 0:
            self.yolo_speedup_factor = (
                self.yolo_metrics.avg_fps / self.opencv_metrics.avg_fps
            )

        # Accuracy comparison (if ground truth available)
        if self.ground_truth_available:
            self.yolo_accuracy_gain = (
                self.yolo_metrics.f1_score - self.opencv_metrics.f1_score
            )

            # Determine winner based on F1 score
            if abs(self.yolo_accuracy_gain) < 0.05:  # Within 5% is a tie
                self.winner = "tie"
            elif self.yolo_accuracy_gain > 0:
                self.winner = "yolo"
            else:
                self.winner = "opencv"
        else:
            # Without ground truth, use detection consistency and speed
            yolo_score = self.yolo_metrics.avg_fps - self.yolo_metrics.detection_variance
            opencv_score = (
                self.opencv_metrics.avg_fps - self.opencv_metrics.detection_variance
            )

            if abs(yolo_score - opencv_score) < 5:
                self.winner = "tie"
            elif yolo_score > opencv_score:
                self.winner = "yolo"
            else:
                self.winner = "opencv"


class DetectorBenchmark:
    """Benchmark framework for comparing ball detectors."""

    def __init__(
        self,
        video_path: str,
        yolo_model_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        ground_truth_path: Optional[str] = None,
        visualize: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Initialize benchmark.

        Args:
            video_path: Path to test video file
            yolo_model_path: Optional path to YOLO model
            max_frames: Optional limit on frames to process
            ground_truth_path: Optional path to ground truth annotations
            visualize: Whether to generate visualization frames
            output_dir: Directory to save visualization frames
        """
        self.video_path = video_path
        self.yolo_model_path = yolo_model_path
        self.max_frames = max_frames
        self.ground_truth_path = ground_truth_path
        self.visualize = visualize
        self.output_dir = Path(output_dir) if output_dir else None

        # Create output directory if needed
        if self.output_dir and self.visualize:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ground truth if provided
        self.ground_truth = self._load_ground_truth() if ground_truth_path else None

        # Initialize detectors
        self._init_detectors()

        # Video properties
        self.video_fps = 0.0
        self.video_frame_count = 0

    def _load_ground_truth(self) -> dict[int, list[dict[str, Any]]]:
        """Load ground truth annotations.

        Expected format:
        {
            "frame_number": [
                {"x": float, "y": float, "radius": float, "type": str},
                ...
            ],
            ...
        }

        Returns:
            Dictionary mapping frame numbers to ball annotations
        """
        try:
            with open(self.ground_truth_path, "r") as f:
                data = json.load(f)
            logger.info(
                f"Loaded ground truth for {len(data)} frames from {self.ground_truth_path}"
            )
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return {}

    def _init_detectors(self):
        """Initialize OpenCV and YOLO detectors."""
        # OpenCV detector with standard configuration
        opencv_config = {
            "detection_method": "combined",
            "hough_dp": 1.0,
            "hough_min_dist_ratio": 0.8,
            "hough_param1": 50,
            "hough_param2": 30,
            "min_radius": 15,
            "max_radius": 26,
            "expected_radius": 20,
            "radius_tolerance": 0.30,
            "min_circularity": 0.75,
            "min_confidence": 0.4,
            "max_overlap_ratio": 0.30,
            "debug_mode": False,
        }
        self.opencv_detector = BallDetector(opencv_config)
        logger.info("OpenCV detector initialized")

        # YOLO detector (if model provided)
        if self.yolo_model_path:
            try:
                self.yolo_detector = YOLODetector(
                    model_path=self.yolo_model_path,
                    device="cpu",  # Use CPU for fair comparison
                    confidence=0.4,
                    nms_threshold=0.45,
                    auto_fallback=False,
                )
                logger.info(f"YOLO detector initialized with model: {self.yolo_model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO detector: {e}")
                self.yolo_detector = None
        else:
            self.yolo_detector = None
            logger.info("No YOLO model provided, benchmarking OpenCV only")

    def _calculate_iou(
        self, ball1: tuple[float, float, float], ball2: tuple[float, float, float]
    ) -> float:
        """Calculate Intersection over Union for two balls.

        Args:
            ball1: (x, y, radius) tuple
            ball2: (x, y, radius) tuple

        Returns:
            IoU value between 0.0 and 1.0
        """
        x1, y1, r1 = ball1
        x2, y2, r2 = ball2

        # Calculate distance between centers
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # If circles don't overlap
        if distance >= r1 + r2:
            return 0.0

        # If one circle is inside the other
        if distance <= abs(r1 - r2):
            smaller_area = np.pi * min(r1, r2) ** 2
            larger_area = np.pi * max(r1, r2) ** 2
            return smaller_area / larger_area

        # Partial overlap
        # Calculate intersection area using geometric formula
        r1_sq = r1 * r1
        r2_sq = r2 * r2
        d_sq = distance * distance

        alpha = np.arccos((d_sq + r1_sq - r2_sq) / (2 * distance * r1))
        beta = np.arccos((d_sq + r2_sq - r1_sq) / (2 * distance * r2))

        intersection_area = (
            r1_sq * alpha + r2_sq * beta - 0.5 * (r1_sq * np.sin(2 * alpha) + r2_sq * np.sin(2 * beta))
        )

        # Union area
        union_area = np.pi * r1_sq + np.pi * r2_sq - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _match_detections_to_ground_truth(
        self, detections: list[Ball], ground_truth: list[dict[str, Any]], iou_threshold: float = 0.5
    ) -> tuple[int, int, int]:
        """Match detections to ground truth and count TP/FP/FN.

        Args:
            detections: List of detected balls
            ground_truth: List of ground truth ball annotations
            iou_threshold: Minimum IoU to consider a match

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        if not ground_truth:
            # No ground truth, count all detections as false positives
            return 0, len(detections), 0

        # Convert detections to (x, y, r) tuples
        det_circles = [(b.position[0], b.position[1], b.radius) for b in detections]

        # Convert ground truth to (x, y, r) tuples
        gt_circles = [(g["x"], g["y"], g["radius"]) for g in ground_truth]

        # Track matched ground truth
        matched_gt = set()
        matched_det = set()

        # For each detection, find best matching ground truth
        for i, det in enumerate(det_circles):
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(gt_circles):
                if j in matched_gt:
                    continue

                iou = self._calculate_iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # If good match found
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_det.add(i)

        # Calculate metrics
        true_positives = len(matched_det)
        false_positives = len(detections) - len(matched_det)
        false_negatives = len(ground_truth) - len(matched_gt)

        return true_positives, false_positives, false_negatives

    def _visualize_comparison(
        self,
        frame: NDArray[np.uint8],
        opencv_balls: list[Ball],
        yolo_balls: list[Ball],
        frame_number: int,
    ) -> NDArray[np.uint8]:
        """Create side-by-side visualization of detections.

        Args:
            frame: Original frame
            opencv_balls: OpenCV detections
            yolo_balls: YOLO detections (can be empty list)
            frame_number: Frame number for labeling

        Returns:
            Visualization frame with side-by-side comparison
        """
        # Create two copies of the frame
        opencv_frame = frame.copy()
        yolo_frame = frame.copy()

        # Draw OpenCV detections
        for ball in opencv_balls:
            x, y = int(ball.position[0]), int(ball.position[1])
            r = int(ball.radius)
            cv2.circle(opencv_frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(opencv_frame, (x, y), 2, (0, 0, 255), -1)

        # Draw YOLO detections
        for ball in yolo_balls:
            x, y = int(ball.position[0]), int(ball.position[1])
            r = int(ball.radius)
            cv2.circle(yolo_frame, (x, y), r, (255, 0, 0), 2)
            cv2.circle(yolo_frame, (x, y), 2, (0, 0, 255), -1)

        # Add labels
        cv2.putText(
            opencv_frame,
            f"OpenCV ({len(opencv_balls)} balls)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            yolo_frame,
            f"YOLO ({len(yolo_balls)} balls)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

        # Concatenate side by side
        vis_frame = np.hstack([opencv_frame, yolo_frame])

        # Add frame number
        cv2.putText(
            vis_frame,
            f"Frame {frame_number}",
            (10, vis_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return vis_frame

    def run_benchmark(self) -> BenchmarkResults:
        """Run complete benchmark and return results.

        Returns:
            BenchmarkResults with complete metrics
        """
        logger.info(f"Starting benchmark on video: {self.video_path}")

        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Get video properties
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Video: {self.video_frame_count} frames @ {self.video_fps:.2f} FPS"
        )

        # Initialize metrics
        opencv_metrics = DetectionMetrics(detector_name="OpenCV")
        yolo_metrics = (
            DetectionMetrics(detector_name="YOLO") if self.yolo_detector else None
        )

        # Process frames
        frame_number = 0
        frames_to_process = (
            self.max_frames if self.max_frames else self.video_frame_count
        )

        logger.info(f"Processing {frames_to_process} frames...")

        while frame_number < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Benchmark OpenCV detector
            start_time = time.time()
            opencv_balls = self.opencv_detector.detect_balls(frame)
            opencv_time = time.time() - start_time

            opencv_metrics.frame_times.append(opencv_time)
            opencv_metrics.detection_counts.append(len(opencv_balls))
            opencv_metrics.total_detections += len(opencv_balls)

            # Calculate average confidence for OpenCV
            if opencv_balls:
                avg_conf = sum(b.confidence for b in opencv_balls) / len(opencv_balls)
                opencv_metrics.avg_confidence += avg_conf

            # Benchmark YOLO detector
            yolo_balls = []
            if self.yolo_detector:
                start_time = time.time()
                yolo_detections = self.yolo_detector.detect_balls(frame)
                yolo_time = time.time() - start_time

                # Convert YOLO detections to Ball objects
                from vision.models import BallType

                for det in yolo_detections:
                    # Extract ball type from class_id
                    if det.class_id == 0:
                        ball_type = BallType.CUE
                    elif det.class_id == 8:
                        ball_type = BallType.EIGHT
                    elif 1 <= det.class_id <= 7:
                        ball_type = BallType.SOLID
                    elif 9 <= det.class_id <= 15:
                        ball_type = BallType.STRIPE
                    else:
                        ball_type = BallType.UNKNOWN

                    # Calculate radius from bbox
                    radius = (det.width + det.height) / 4  # Average of width/height / 2

                    ball = Ball(
                        position=det.center,
                        radius=radius,
                        ball_type=ball_type,
                        number=det.class_id if det.class_id > 0 else None,
                        confidence=det.confidence,
                        velocity=(0.0, 0.0),
                        is_moving=False,
                    )
                    yolo_balls.append(ball)

                yolo_metrics.frame_times.append(yolo_time)
                yolo_metrics.detection_counts.append(len(yolo_balls))
                yolo_metrics.total_detections += len(yolo_balls)

                # Calculate average confidence for YOLO
                if yolo_balls:
                    avg_conf = sum(b.confidence for b in yolo_balls) / len(yolo_balls)
                    yolo_metrics.avg_confidence += avg_conf

            # Match against ground truth if available
            if self.ground_truth and frame_number in self.ground_truth:
                gt = self.ground_truth[frame_number]

                # OpenCV accuracy
                tp, fp, fn = self._match_detections_to_ground_truth(opencv_balls, gt)
                opencv_metrics.true_positives += tp
                opencv_metrics.false_positives += fp
                opencv_metrics.false_negatives += fn

                # YOLO accuracy
                if self.yolo_detector:
                    tp, fp, fn = self._match_detections_to_ground_truth(yolo_balls, gt)
                    yolo_metrics.true_positives += tp
                    yolo_metrics.false_positives += fp
                    yolo_metrics.false_negatives += fn

            # Visualize if requested
            if self.visualize and self.output_dir:
                vis_frame = self._visualize_comparison(
                    frame, opencv_balls, yolo_balls, frame_number
                )
                output_path = self.output_dir / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(output_path), vis_frame)

            frame_number += 1

            # Progress reporting
            if frame_number % 10 == 0:
                progress = (frame_number / frames_to_process) * 100
                logger.info(
                    f"Progress: {frame_number}/{frames_to_process} ({progress:.1f}%)"
                )

        cap.release()

        # Finalize metrics
        opencv_metrics.total_frames = frame_number
        opencv_metrics.total_processing_time = sum(opencv_metrics.frame_times)
        if opencv_metrics.total_frames > 0:
            opencv_metrics.avg_confidence /= opencv_metrics.total_frames
        opencv_metrics.calculate_statistics()

        if yolo_metrics:
            yolo_metrics.total_frames = frame_number
            yolo_metrics.total_processing_time = sum(yolo_metrics.frame_times)
            if yolo_metrics.total_frames > 0:
                yolo_metrics.avg_confidence /= yolo_metrics.total_frames
            yolo_metrics.calculate_statistics()

        # Create results
        from datetime import datetime

        results = BenchmarkResults(
            opencv_metrics=opencv_metrics,
            yolo_metrics=yolo_metrics,
            test_video_path=self.video_path,
            test_video_fps=self.video_fps,
            test_video_frames=self.video_frame_count,
            benchmark_date=datetime.now().isoformat(),
            ground_truth_available=self.ground_truth is not None,
        )
        results.calculate_comparison()

        logger.info("Benchmark complete!")
        return results


def print_results_table(results: BenchmarkResults):
    """Print results in a formatted table.

    Args:
        results: Benchmark results to display
    """
    print("\n" + "=" * 80)
    print(" DETECTOR BENCHMARK RESULTS ".center(80, "="))
    print("=" * 80)

    print(f"\nTest Video: {results.test_video_path}")
    print(f"Video Properties: {results.test_video_frames} frames @ {results.test_video_fps:.2f} FPS")
    print(f"Benchmark Date: {results.benchmark_date}")
    print(f"Ground Truth: {'Available' if results.ground_truth_available else 'Not Available'}")

    # Performance metrics
    print("\n" + "-" * 80)
    print(" PERFORMANCE METRICS ".center(80, "-"))
    print("-" * 80)

    opencv = results.opencv_metrics
    yolo = results.yolo_metrics

    print(f"\n{'Metric':<30} {'OpenCV':>20} {'YOLO':>20}")
    print("-" * 72)
    print(f"{'Total Frames Processed':<30} {opencv.total_frames:>20} {yolo.total_frames if yolo else 'N/A':>20}")
    print(f"{'Total Detections':<30} {opencv.total_detections:>20} {yolo.total_detections if yolo else 'N/A':>20}")
    print(f"{'Avg Detections/Frame':<30} {opencv.avg_detections_per_frame:>20.2f} {yolo.avg_detections_per_frame if yolo else 'N/A':>20}")
    print(f"{'Detection Variance':<30} {opencv.detection_variance:>20.2f} {yolo.detection_variance if yolo else 'N/A':>20}")
    print()
    print(f"{'Average FPS':<30} {opencv.avg_fps:>20.2f} {yolo.avg_fps if yolo else 'N/A':>20}")
    print(f"{'Min FPS':<30} {opencv.min_fps:>20.2f} {yolo.min_fps if yolo else 'N/A':>20}")
    print(f"{'Max FPS':<30} {opencv.max_fps:>20.2f} {yolo.max_fps if yolo else 'N/A':>20}")
    print(f"{'Median FPS':<30} {opencv.median_fps:>20.2f} {yolo.median_fps if yolo else 'N/A':>20}")
    print()
    print(f"{'Average Latency (ms)':<30} {opencv.avg_latency_ms:>20.2f} {yolo.avg_latency_ms if yolo else 'N/A':>20}")
    print(f"{'Min Latency (ms)':<30} {opencv.min_latency_ms:>20.2f} {yolo.min_latency_ms if yolo else 'N/A':>20}")
    print(f"{'Max Latency (ms)':<30} {opencv.max_latency_ms:>20.2f} {yolo.max_latency_ms if yolo else 'N/A':>20}")
    print()
    print(f"{'Average Confidence':<30} {opencv.avg_confidence:>20.3f} {yolo.avg_confidence if yolo else 'N/A':>20}")

    # Accuracy metrics (if ground truth available)
    if results.ground_truth_available:
        print("\n" + "-" * 80)
        print(" ACCURACY METRICS (vs Ground Truth) ".center(80, "-"))
        print("-" * 80)

        print(f"\n{'Metric':<30} {'OpenCV':>20} {'YOLO':>20}")
        print("-" * 72)
        print(f"{'True Positives':<30} {opencv.true_positives:>20} {yolo.true_positives if yolo else 'N/A':>20}")
        print(f"{'False Positives':<30} {opencv.false_positives:>20} {yolo.false_positives if yolo else 'N/A':>20}")
        print(f"{'False Negatives':<30} {opencv.false_negatives:>20} {yolo.false_negatives if yolo else 'N/A':>20}")
        print()
        print(f"{'Precision':<30} {opencv.precision:>20.3f} {yolo.precision if yolo else 'N/A':>20}")
        print(f"{'Recall':<30} {opencv.recall:>20.3f} {yolo.recall if yolo else 'N/A':>20}")
        print(f"{'F1 Score':<30} {opencv.f1_score:>20.3f} {yolo.f1_score if yolo else 'N/A':>20}")

    # Comparison
    if yolo:
        print("\n" + "-" * 80)
        print(" COMPARATIVE ANALYSIS ".center(80, "-"))
        print("-" * 80)

        print(f"\nYOLO Speedup Factor: {results.yolo_speedup_factor:.2f}x")
        if results.yolo_speedup_factor > 1.0:
            print(f"  → YOLO is {results.yolo_speedup_factor:.2f}x FASTER than OpenCV")
        else:
            print(f"  → OpenCV is {1.0/results.yolo_speedup_factor:.2f}x FASTER than YOLO")

        if results.ground_truth_available:
            print(f"\nYOLO Accuracy Gain: {results.yolo_accuracy_gain:+.3f} (F1 score difference)")
            if results.yolo_accuracy_gain > 0.05:
                print(f"  → YOLO is MORE ACCURATE than OpenCV")
            elif results.yolo_accuracy_gain < -0.05:
                print(f"  → OpenCV is MORE ACCURATE than YOLO")
            else:
                print(f"  → Both detectors have SIMILAR ACCURACY")

        print(f"\nWinner: {results.winner.upper()}")

    print("\n" + "=" * 80 + "\n")


def save_results_json(results: BenchmarkResults, output_path: str):
    """Save results to JSON file.

    Args:
        results: Benchmark results
        output_path: Path to output JSON file
    """
    # Convert to dictionary
    data = {
        "opencv_metrics": asdict(results.opencv_metrics),
        "yolo_metrics": asdict(results.yolo_metrics) if results.yolo_metrics else None,
        "test_video_path": results.test_video_path,
        "test_video_fps": results.test_video_fps,
        "test_video_frames": results.test_video_frames,
        "benchmark_date": results.benchmark_date,
        "ground_truth_available": results.ground_truth_available,
        "yolo_speedup_factor": results.yolo_speedup_factor,
        "yolo_accuracy_gain": results.yolo_accuracy_gain,
        "winner": results.winner,
    }

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def save_results_text(results: BenchmarkResults, output_path: str):
    """Save results to text report file.

    Args:
        results: Benchmark results
        output_path: Path to output text file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Redirect print to file
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    print_results_table(results)

    report_text = sys.stdout.getvalue()
    sys.stdout = old_stdout

    # Save to file
    with open(output_file, "w") as f:
        f.write(report_text)

    logger.info(f"Text report saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO vs OpenCV ball detection performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--video", required=True, help="Path to test video file or image sequence"
    )
    parser.add_argument(
        "--yolo-model", help="Path to YOLO model file (.pt or .onnx)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Number of frames to test (default: all frames)",
    )
    parser.add_argument(
        "--ground-truth",
        help="Path to ground truth annotations JSON file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate side-by-side comparison visualizations",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_output",
        help="Directory to save visualization frames (default: benchmark_output)",
    )
    parser.add_argument(
        "--results-json",
        default="models/benchmark_results.json",
        help="Path to save JSON results (default: models/benchmark_results.json)",
    )
    parser.add_argument(
        "--results-text",
        default="models/benchmark_report.txt",
        help="Path to save text report (default: models/benchmark_report.txt)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate video file exists
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Validate YOLO model if provided
    if args.yolo_model and not Path(args.yolo_model).exists():
        logger.error(f"YOLO model file not found: {args.yolo_model}")
        sys.exit(1)

    # Validate ground truth if provided
    if args.ground_truth and not Path(args.ground_truth).exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    # Create benchmark
    benchmark = DetectorBenchmark(
        video_path=args.video,
        yolo_model_path=args.yolo_model,
        max_frames=args.frames,
        ground_truth_path=args.ground_truth,
        visualize=args.visualize,
        output_dir=args.output_dir if args.visualize else None,
    )

    # Run benchmark
    try:
        results = benchmark.run_benchmark()

        # Display results
        print_results_table(results)

        # Save results
        save_results_json(results, args.results_json)
        save_results_text(results, args.results_text)

        logger.info("Benchmark complete!")

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
