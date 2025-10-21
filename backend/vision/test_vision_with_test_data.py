"""Comprehensive vision system tests using test_data ground truth images.

This test suite validates the vision system against curated test images and
their ground truth annotations. These tests ensure the system meets quality
requirements across various real-world scenarios.

Test Coverage:
- Empty table (false positive testing)
- Multiple balls detection and accuracy
- Clustered balls separation
- Cue stick detection and tracking
- Motion blur handling
- Calibration accuracy
- Edge cases and challenging conditions

Ground Truth Format:
Each test image has a corresponding .json file with:
- Balls: position, radius, bounding box
- Cues: position, bounding box
- Other annotations as needed

Quality Requirements (from SPECS.md):
- Ball detection accuracy > 98% (NFR-VIS-006)
- Position accuracy within 2mm / 2 pixels (NFR-VIS-008, FR-VIS-023)
- False positive rate < 1% (NFR-VIS-007)
- Color classification accuracy > 95% (NFR-VIS-010)
"""

import json
import logging
import unittest
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from backend.vision.detection.balls import BallDetectionConfig, BallDetector
from backend.vision.detection.cue import CueDetector
from backend.vision.detection.table import TableDetector
from backend.vision.detection.yolo_detector import YOLODetector
from backend.vision.models import Ball, BallType, CueStick

logger = logging.getLogger(__name__)

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class YOLOBallDetectorAdapter:
    """Adapter to make YOLODetector compatible with BallDetector interface for tests."""

    def __init__(self, yolo_detector: YOLODetector):
        """Initialize adapter with YOLODetector instance.

        Args:
            yolo_detector: YOLODetector instance to wrap
        """
        self.yolo_detector = yolo_detector

    def detect_balls(
        self, frame: np.ndarray, table_mask: Optional[np.ndarray] = None
    ) -> list[Ball]:
        """Detect balls using YOLO and convert to Ball objects.

        Args:
            frame: Input image
            table_mask: Optional table mask (not used by YOLO)

        Returns:
            List of Ball objects
        """
        from backend.vision.detection.detector_adapter import yolo_detections_to_balls

        # Get YOLO detections
        yolo_detections = self.yolo_detector.detect_balls(frame)

        if not yolo_detections:
            return []

        # Convert Detection objects to Ball objects
        balls = []
        for det in yolo_detections:
            # Create detection dict for adapter
            detection_dict = {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }

            # Convert using adapter (handles single detection at a time)
            converted_balls = yolo_detections_to_balls(
                [detection_dict],
                (frame.shape[0], frame.shape[1]),
                min_confidence=0.15,
                bbox_format="xyxy",
            )

            balls.extend(converted_balls)

        return balls


# Tolerance thresholds (based on SPECS.md requirements)
POSITION_TOLERANCE_PIXELS = 2.0  # FR-VIS-023: ±2 pixel accuracy (for average error)
MATCHING_TOLERANCE_PIXELS = (
    20.0  # Matching tolerance for greedy assignment (more lenient)
)
RADIUS_TOLERANCE_RATIO = 0.15  # 15% tolerance for radius
DETECTION_ACCURACY_THRESHOLD = 0.98  # NFR-VIS-006: >98% detection accuracy
FALSE_POSITIVE_THRESHOLD = 0.01  # NFR-VIS-007: <1% false positive rate


def load_ground_truth(json_path: Path) -> dict[str, Any]:
    """Load ground truth annotations from JSON file.

    Args:
        json_path: Path to JSON annotation file

    Returns:
        Dictionary with 'balls' and 'cues' annotations
    """
    with open(json_path) as f:
        return json.load(f)


def calculate_detection_metrics(
    detected: list[Ball], ground_truth: list[dict]
) -> dict[str, float]:
    """Calculate detection performance metrics.

    Args:
        detected: List of detected Ball objects
        ground_truth: List of ground truth ball annotations

    Returns:
        Dictionary with metrics:
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: Harmonic mean of precision and recall
        - avg_position_error: Average position error in pixels
        - avg_radius_error: Average radius error ratio
    """
    if not ground_truth:
        # Empty table case
        if not detected:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "avg_position_error": 0.0,
                "avg_radius_error": 0.0,
            }
        else:
            # False positives on empty table
            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1_score": 0.0,
                "avg_position_error": 0.0,
                "avg_radius_error": 0.0,
            }

    # Match detected balls to ground truth using Hungarian algorithm (greedy for now)
    matched = []
    unmatched_detected = list(detected)
    unmatched_gt = list(ground_truth)

    position_errors = []
    radius_errors = []

    # Greedy matching: find closest detected ball for each ground truth
    for gt_ball in ground_truth:
        gt_pos = (gt_ball["center"]["x"], gt_ball["center"]["y"])
        gt_radius = gt_ball["radius"]

        best_match = None
        best_distance = float("inf")

        for det_ball in unmatched_detected:
            det_pos = det_ball.position
            distance = np.sqrt(
                (det_pos[0] - gt_pos[0]) ** 2 + (det_pos[1] - gt_pos[1]) ** 2
            )

            if distance < best_distance and distance <= MATCHING_TOLERANCE_PIXELS:
                best_distance = distance
                best_match = det_ball

        if best_match:
            matched.append((best_match, gt_ball))
            unmatched_detected.remove(best_match)
            unmatched_gt.remove(
                gt_ball
            )  # Fix: Remove matched GT ball from unmatched list
            position_errors.append(best_distance)

            # Calculate radius error
            radius_error = abs(best_match.radius - gt_radius) / gt_radius
            radius_errors.append(radius_error)

    # Calculate metrics
    true_positives = len(matched)
    false_positives = len(unmatched_detected)
    false_negatives = len(unmatched_gt)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    avg_position_error = np.mean(position_errors) if position_errors else 0.0
    avg_radius_error = np.mean(radius_errors) if radius_errors else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_position_error": avg_position_error,
        "avg_radius_error": avg_radius_error,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


class TestVisionWithGroundTruth(unittest.TestCase):
    """Test vision system against ground truth test data.

    These tests validate the vision system meets the quality requirements
    specified in SPECS.md using real-world test images.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Verify test data directory exists
        if not TEST_DATA_DIR.exists():
            raise FileNotFoundError(
                f"Test data directory not found: {TEST_DATA_DIR}\n"
                f"Please ensure test_data/ contains the required test images and annotations."
            )

        # Initialize YOLO detector (using v117 model which works well)
        yolo_detector = YOLODetector(
            model_path="backend/vision/models/training_runs/yolov8n_pool_v117/weights/best.pt",
            device="cpu",  # Use CPU for consistency
            confidence=0.15,  # Default from yolo_detector.py
            nms_threshold=0.45,
            enable_opencv_classification=False,  # Model already classifies balls
            min_ball_size=20,
        )

        # Wrap YOLO detector to match BallDetector interface
        cls.ball_detector = YOLOBallDetectorAdapter(yolo_detector)

        # Initialize cue detector with YOLO support
        cls.yolo_detector = yolo_detector  # Keep reference for cue detector
        cls.cue_detector = CueDetector(
            cls._get_default_cue_config(), yolo_detector=yolo_detector
        )

        # Track overall performance metrics across all tests
        cls.all_metrics = []

    @classmethod
    def _get_default_ball_config(cls) -> dict[str, Any]:
        """Get default ball detection configuration."""
        return {
            "detection_method": "combined",
            "hough_circles": {
                "dp": 1.0,
                "min_dist_ratio": 0.8,
                "param1": 50,
                "param2": 30,
                "gaussian_blur_kernel": 9,
                "gaussian_blur_sigma": 2,
            },
            "size_constraints": {
                "min_radius": 15,
                "max_radius": 70,
                "expected_radius": 50,  # Balls in test images are ~50-65px radius
                "radius_tolerance": 0.40,  # Balanced tolerance
            },
            "quality": {
                "min_circularity": 0.75,
                "min_confidence": 0.25,  # Lowered to allow detection with partial color info
                "max_overlap_ratio": 0.30,
            },
            "debug": {
                "debug_mode": False,
                "save_debug_images": False,
            },
        }

    @classmethod
    def _get_default_cue_config(cls) -> dict[str, Any]:
        """Get default cue detection configuration."""
        return {
            "geometry": {
                "min_cue_length": 150,
                "max_cue_length": 800,
                "ball_radius": 15,
            },
            "hough": {
                "threshold": 100,
                "min_line_length": 100,
                "max_line_gap": 20,
            },
            "filtering": {
                "angle_stability_threshold": 5.0,
                "position_stability_threshold": 10.0,
            },
        }

    def _load_test_image(self, image_name: str) -> Optional[np.ndarray]:
        """Load test image from test_data directory.

        Args:
            image_name: Name of image file (e.g., 'empty_table.png')

        Returns:
            Loaded image as numpy array, or None if not found
        """
        image_path = TEST_DATA_DIR / image_name
        if not image_path.exists():
            logger.warning(f"Test image not found: {image_path}")
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
        return image

    def _load_ground_truth_for_image(self, image_name: str) -> Optional[dict]:
        """Load ground truth annotations for an image.

        Args:
            image_name: Name of image file (e.g., 'empty_table.png')

        Returns:
            Ground truth dictionary or None if not found
        """
        json_name = image_name.replace(".png", ".json").replace(".jpg", ".json")
        json_path = TEST_DATA_DIR / json_name

        if not json_path.exists():
            logger.warning(f"Ground truth not found: {json_path}")
            return None

        return load_ground_truth(json_path)

    # =========================================================================
    # Empty Table Tests - False Positive Detection
    # =========================================================================

    def test_empty_table_no_false_positives(self):
        """Test that empty table produces no false positive ball detections.

        Requirement: NFR-VIS-007 - False positive rate < 1%
        """
        image = self._load_test_image("empty_table.png")
        self.assertIsNotNone(image, "empty_table.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("empty_table.png")
        self.assertIsNotNone(ground_truth)
        self.assertEqual(
            len(ground_truth["balls"]), 0, "Empty table should have 0 balls"
        )

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])

        # Empty table should have perfect precision (no false positives)
        self.assertEqual(
            metrics["precision"],
            1.0,
            f"Empty table produced {metrics.get('false_positives', 0)} false positives. "
            f"Expected 0 detections but got {len(detected_balls)}",
        )

    # =========================================================================
    # Multiple Balls Tests - Detection Accuracy
    # =========================================================================

    def test_multiple_balls_detection_accuracy(self):
        """Test detection accuracy with multiple balls on table.

        Requirements:
        - NFR-VIS-006: Ball detection accuracy > 98%
        - FR-VIS-023: Track ball positions with ±2 pixel accuracy
        """
        image = self._load_test_image("multiple_balls.png")
        self.assertIsNotNone(image, "multiple_balls.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("multiple_balls.png")
        self.assertIsNotNone(ground_truth)

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])
        self.all_metrics.append(("multiple_balls", metrics))

        # Assert detection accuracy
        self.assertGreaterEqual(
            metrics["recall"],
            DETECTION_ACCURACY_THRESHOLD,
            f"Detection recall {metrics['recall']:.2%} is below required "
            f"{DETECTION_ACCURACY_THRESHOLD:.2%}. "
            f"Detected {metrics['true_positives']}/{len(ground_truth['balls'])} balls. "
            f"Missed {metrics['false_negatives']} balls.",
        )

        # Assert position accuracy
        self.assertLessEqual(
            metrics["avg_position_error"],
            POSITION_TOLERANCE_PIXELS,
            f"Average position error {metrics['avg_position_error']:.2f}px exceeds "
            f"tolerance {POSITION_TOLERANCE_PIXELS}px",
        )

        # Assert radius accuracy
        self.assertLessEqual(
            metrics["avg_radius_error"],
            RADIUS_TOLERANCE_RATIO,
            f"Average radius error {metrics['avg_radius_error']:.2%} exceeds "
            f"tolerance {RADIUS_TOLERANCE_RATIO:.2%}",
        )

    # =========================================================================
    # Clustered Balls Tests - Separation Accuracy
    # =========================================================================

    def test_clustered_balls_separation(self):
        """Test ball separation with tightly clustered balls.

        This is a challenging scenario that tests the detector's ability to
        separate balls that are very close together.

        Requirement: FR-VIS-020 - Detect all balls on the table surface
        """
        image = self._load_test_image("clustered_balls.png")
        self.assertIsNotNone(image, "clustered_balls.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("clustered_balls.png")
        self.assertIsNotNone(ground_truth)

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])
        self.all_metrics.append(("clustered_balls", metrics))

        # For clustered balls, we allow slightly lower accuracy (95% instead of 98%)
        # since this is a challenging scenario
        CLUSTERED_THRESHOLD = 0.95

        self.assertGreaterEqual(
            metrics["recall"],
            CLUSTERED_THRESHOLD,
            f"Detection recall {metrics['recall']:.2%} for clustered balls is below "
            f"threshold {CLUSTERED_THRESHOLD:.2%}. "
            f"Detected {metrics['true_positives']}/{len(ground_truth['balls'])} balls. "
            f"Missed {metrics['false_negatives']} balls.",
        )

    # =========================================================================
    # Full Table Tests - Stress Testing
    # =========================================================================

    def test_full_table_all_balls(self):
        """Test detection with all 15 balls on table (stress test).

        This tests the system's ability to handle maximum ball count.

        Requirement: FR-VIS-020 - Detect all balls on the table surface
        """
        image = self._load_test_image("full_table.png")
        if image is None:
            self.skipTest("full_table.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("full_table.png")
        if ground_truth is None:
            self.skipTest("full_table.json not found in test_data")

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])
        self.all_metrics.append(("full_table", metrics))

        # With 15 balls, expect high recall
        self.assertGreaterEqual(
            metrics["recall"],
            DETECTION_ACCURACY_THRESHOLD,
            f"Detection recall {metrics['recall']:.2%} is below required "
            f"{DETECTION_ACCURACY_THRESHOLD:.2%}. "
            f"Detected {metrics['true_positives']}/{len(ground_truth['balls'])} balls.",
        )

    # =========================================================================
    # Cue Detection Tests
    # =========================================================================

    def test_cue_detection_frame_with_cue(self):
        """Test cue stick detection in frame with visible cue.

        Requirements:
        - FR-VIS-030: Detect cue stick using line detection
        - FR-VIS-031: Determine cue angle relative to cue ball
        """
        image = self._load_test_image("frame_with_cue.png")
        self.assertIsNotNone(image, "frame_with_cue.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("frame_with_cue.png")
        self.assertIsNotNone(ground_truth)

        # Detect cue
        detected_cue = self.cue_detector.detect_cue(image, None, None)

        # Verify cue was detected if ground truth shows cue present
        if ground_truth["cues"]:
            self.assertIsNotNone(
                detected_cue,
                "Cue stick should be detected when present in frame",
            )
            if detected_cue:
                # Verify cue has reasonable properties
                self.assertGreater(
                    detected_cue.length, 0, "Cue length should be positive"
                )
                self.assertGreater(
                    detected_cue.confidence, 0, "Cue confidence should be positive"
                )
        else:
            # If no cue in ground truth, either None or low confidence is acceptable
            if detected_cue:
                self.assertLess(
                    detected_cue.confidence,
                    0.5,
                    "High confidence cue detection when no cue present",
                )

    def test_cue_detection_aiming(self):
        """Test cue detection during aiming scenario.

        Requirement: FR-VIS-033 - Detect cue movement patterns (aiming vs striking)
        """
        image = self._load_test_image("cue_aiming.png")
        self.assertIsNotNone(image, "cue_aiming.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("cue_aiming.png")
        self.assertIsNotNone(ground_truth)

        # Detect cue
        detected_cue = self.cue_detector.detect_cue(image, None, None)

        # Verify cue was detected
        if ground_truth["cues"]:
            self.assertIsNotNone(
                detected_cue,
                "Cue should be detected in aiming scenario",
            )

    # =========================================================================
    # Motion Blur Tests - Challenging Conditions
    # =========================================================================

    def test_motion_blur_handling(self):
        """Test detection with motion blur (fast-moving balls).

        This tests the system's robustness to motion blur.

        Requirement: NFR-VIS-014 - Tolerate various image quality issues
        """
        image = self._load_test_image("motion_blur.png")
        if image is None:
            self.skipTest("motion_blur.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("motion_blur.png")
        if ground_truth is None:
            self.skipTest("motion_blur.json not found in test_data")

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])
        self.all_metrics.append(("motion_blur", metrics))

        # For motion blur, we allow lower accuracy (85% instead of 98%)
        MOTION_BLUR_THRESHOLD = 0.85

        self.assertGreaterEqual(
            metrics["recall"],
            MOTION_BLUR_THRESHOLD,
            f"Detection with motion blur {metrics['recall']:.2%} is below "
            f"threshold {MOTION_BLUR_THRESHOLD:.2%}. System should handle motion blur.",
        )

    # =========================================================================
    # Calibration Tests
    # =========================================================================

    def test_calibration_straight_on_view(self):
        """Test detection with straight-on calibration view.

        This represents optimal camera positioning.

        Requirement: FR-VIS-039 - Perform automatic camera calibration
        """
        image = self._load_test_image("calibration_straight_on.png")
        if image is None:
            self.skipTest("calibration_straight_on.png not found in test_data")

        ground_truth = self._load_ground_truth_for_image("calibration_straight_on.png")
        if ground_truth is None:
            self.skipTest("calibration_straight_on.json not found in test_data")

        # Detect balls
        detected_balls = self.ball_detector.detect_balls(image, table_mask=None)

        # Calculate metrics
        metrics = calculate_detection_metrics(detected_balls, ground_truth["balls"])
        self.all_metrics.append(("calibration_straight_on", metrics))

        # Straight-on view should have excellent accuracy
        if ground_truth["balls"]:
            self.assertGreaterEqual(
                metrics["recall"],
                DETECTION_ACCURACY_THRESHOLD,
                f"Straight-on calibration view should achieve high accuracy. "
                f"Got {metrics['recall']:.2%}",
            )

    # =========================================================================
    # Summary Report
    # =========================================================================

    @classmethod
    def tearDownClass(cls):
        """Generate summary report of all test results."""
        if cls.all_metrics:
            print("\n" + "=" * 80)
            print("VISION SYSTEM TEST SUMMARY - Ground Truth Validation")
            print("=" * 80)

            for test_name, metrics in cls.all_metrics:
                print(f"\n{test_name}:")
                print(f"  Precision:          {metrics['precision']:.2%}")
                print(f"  Recall:             {metrics['recall']:.2%}")
                print(f"  F1 Score:           {metrics['f1_score']:.2%}")
                print(f"  Position Error:     {metrics['avg_position_error']:.2f} px")
                print(f"  Radius Error:       {metrics['avg_radius_error']:.2%}")
                print(f"  True Positives:     {metrics.get('true_positives', 'N/A')}")
                print(f"  False Positives:    {metrics.get('false_positives', 'N/A')}")
                print(f"  False Negatives:    {metrics.get('false_negatives', 'N/A')}")

            # Calculate overall metrics
            avg_precision = np.mean([m[1]["precision"] for m in cls.all_metrics])
            avg_recall = np.mean([m[1]["recall"] for m in cls.all_metrics])
            avg_f1 = np.mean([m[1]["f1_score"] for m in cls.all_metrics])

            print("\n" + "-" * 80)
            print("OVERALL METRICS:")
            print(f"  Average Precision:  {avg_precision:.2%}")
            print(f"  Average Recall:     {avg_recall:.2%}")
            print(f"  Average F1 Score:   {avg_f1:.2%}")
            print("=" * 80 + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run tests
    unittest.main(verbosity=2)
