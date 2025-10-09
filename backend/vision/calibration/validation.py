"""Calibration validation and accuracy testing."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.config.manager import config_manager

from .camera import CameraCalibrator
from .color import ColorCalibrator
from .geometry import GeometricCalibrator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Calibration validation result."""

    test_name: str
    timestamp: float
    passed: bool
    accuracy_score: float  # 0.0-1.0
    error_metrics: dict[str, float]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CalibrationReport:
    """Comprehensive calibration validation report."""

    session_id: str
    test_date: str
    camera_validation: Optional[ValidationResult]
    color_validation: Optional[ValidationResult]
    geometry_validation: Optional[ValidationResult]
    integration_validation: Optional[ValidationResult]
    overall_score: float
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "session_id": self.session_id,
            "test_date": self.test_date,
            "camera_validation": (
                self.camera_validation.to_dict() if self.camera_validation else None
            ),
            "color_validation": (
                self.color_validation.to_dict() if self.color_validation else None
            ),
            "geometry_validation": (
                self.geometry_validation.to_dict() if self.geometry_validation else None
            ),
            "integration_validation": (
                self.integration_validation.to_dict()
                if self.integration_validation
                else None
            ),
            "overall_score": self.overall_score,
            "recommendations": self.recommendations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationReport":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            test_date=data["test_date"],
            camera_validation=(
                ValidationResult.from_dict(data["camera_validation"])
                if data["camera_validation"]
                else None
            ),
            color_validation=(
                ValidationResult.from_dict(data["color_validation"])
                if data["color_validation"]
                else None
            ),
            geometry_validation=(
                ValidationResult.from_dict(data["geometry_validation"])
                if data["geometry_validation"]
                else None
            ),
            integration_validation=(
                ValidationResult.from_dict(data["integration_validation"])
                if data["integration_validation"]
                else None
            ),
            overall_score=data["overall_score"],
            recommendations=data["recommendations"],
        )


class CalibrationValidator:
    """Comprehensive calibration validation and testing system.

    Provides validation for all calibration components:
    - Camera intrinsic/extrinsic parameters
    - Color threshold accuracy
    - Geometric transformation accuracy
    - End-to-end system integration
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize calibration validator.

        Args:
            cache_dir: Directory to save validation reports
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.cwd() / "validation_cache"
        )
        self.cache_dir.mkdir(exist_ok=True)

        # Accuracy thresholds for different validation tests - load from config
        thresholds_config = config_manager.get(
            "vision.calibration.validation.accuracy_thresholds", {}
        )
        self.accuracy_thresholds = {
            "camera_reprojection_error": thresholds_config.get(
                "camera_reprojection_error_pixels", 2.0
            ),
            "camera_distortion_quality": thresholds_config.get(
                "camera_distortion_quality", 0.8
            ),
            "color_detection_accuracy": thresholds_config.get(
                "color_detection_accuracy", 0.85
            ),
            "color_false_positive_rate": thresholds_config.get(
                "color_false_positive_rate", 0.1
            ),
            "geometry_pixel_error": thresholds_config.get("geometry_pixel_error", 5.0),
            "geometry_world_error": thresholds_config.get(
                "geometry_world_error_meters", 0.02
            ),
            "integration_ball_accuracy": thresholds_config.get(
                "integration_ball_accuracy", 0.9
            ),
            "integration_table_accuracy": thresholds_config.get(
                "integration_table_accuracy", 0.95
            ),
        }

    def validate_camera_calibration(
        self, camera_calibrator: CameraCalibrator, test_images: list[np.ndarray]
    ) -> ValidationResult:
        """Validate camera calibration accuracy.

        Args:
            camera_calibrator: Camera calibrator instance
            test_images: Test images with known patterns

        Returns:
            Camera validation result
        """
        if camera_calibrator.camera_params is None:
            return ValidationResult(
                test_name="camera_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"error": "No camera parameters available"},
                details={"status": "failed", "reason": "missing_parameters"},
            )

        try:
            # Test reprojection accuracy using chessboard patterns
            reprojection_errors = []
            detection_rate = 0
            total_images = len(test_images)

            pattern_points = camera_calibrator.generate_chessboard_points()

            for _i, image in enumerate(test_images):
                gray = (
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if len(image.shape) == 3
                    else image
                )

                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(
                    gray, camera_calibrator.chessboard_size, None
                )

                if ret:
                    detection_rate += 1

                    # Refine corners
                    refinement_config = config_manager.get(
                        "vision.calibration.validation.camera_validation.corner_refinement",
                        {},
                    )
                    max_iter = refinement_config.get("max_iterations", 30)
                    epsilon = refinement_config.get("epsilon", 0.001)
                    window_size = refinement_config.get("window_size", [11, 11])
                    zero_zone = refinement_config.get("zero_zone", [-1, -1])

                    criteria = (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        max_iter,
                        epsilon,
                    )
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, tuple(window_size), tuple(zero_zone), criteria
                    )

                    # Project 3D points to image
                    rvec = np.zeros((3, 1))
                    tvec = np.zeros((3, 1))

                    # Solve PnP to get pose
                    ret_pnp, rvec, tvec = cv2.solvePnP(
                        pattern_points,
                        corners_refined,
                        camera_calibrator.camera_params.camera_matrix,
                        camera_calibrator.camera_params.distortion_coefficients,
                    )

                    if ret_pnp:
                        # Reproject points
                        projected_points, _ = cv2.projectPoints(
                            pattern_points,
                            rvec,
                            tvec,
                            camera_calibrator.camera_params.camera_matrix,
                            camera_calibrator.camera_params.distortion_coefficients,
                        )

                        # Calculate reprojection error
                        error = cv2.norm(
                            corners_refined, projected_points, cv2.NORM_L2
                        ) / len(projected_points)
                        reprojection_errors.append(error)

            detection_rate /= total_images if total_images > 0 else 1

            # Calculate metrics
            mean_reprojection_error = (
                np.mean(reprojection_errors) if reprojection_errors else float("inf")
            )
            max_reprojection_error = (
                np.max(reprojection_errors) if reprojection_errors else float("inf")
            )

            # Test distortion correction quality
            distortion_quality = self._test_distortion_correction(
                camera_calibrator, test_images[0] if test_images else None
            )

            # Calculate accuracy score
            reprojection_score = max(
                0.0,
                1.0
                - (
                    mean_reprojection_error
                    / self.accuracy_thresholds["camera_reprojection_error"]
                ),
            )
            detection_score = detection_rate
            distortion_score = distortion_quality

            accuracy_score = (
                reprojection_score + detection_score + distortion_score
            ) / 3.0

            # Determine if validation passed
            min_detection_rate = config_manager.get(
                "vision.calibration.validation.camera_validation.min_detection_rate",
                0.7,
            )
            passed = (
                mean_reprojection_error
                < self.accuracy_thresholds["camera_reprojection_error"]
                and detection_rate > min_detection_rate
                and distortion_quality
                > self.accuracy_thresholds["camera_distortion_quality"]
            )

            return ValidationResult(
                test_name="camera_calibration",
                timestamp=time.time(),
                passed=passed,
                accuracy_score=accuracy_score,
                error_metrics={
                    "mean_reprojection_error": mean_reprojection_error,
                    "max_reprojection_error": max_reprojection_error,
                    "detection_rate": detection_rate,
                    "distortion_quality": distortion_quality,
                },
                details={
                    "num_test_images": total_images,
                    "successful_detections": len(reprojection_errors),
                    "calibration_error": camera_calibrator.camera_params.calibration_error,
                },
            )

        except Exception as e:
            logger.error(f"Camera validation failed: {e}")
            return ValidationResult(
                test_name="camera_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"exception": str(e)},
                details={"status": "exception", "error": str(e)},
            )

    def _test_distortion_correction(
        self,
        camera_calibrator: CameraCalibrator,
        test_image: Optional[NDArray[np.float64]],
    ) -> float:
        """Test quality of distortion correction."""
        if test_image is None or camera_calibrator.camera_params is None:
            return 0.0

        try:
            # Apply undistortion
            undistorted = camera_calibrator.undistort_image(test_image)

            # Measure straightness of lines (simple heuristic)
            gray = (
                cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                if len(undistorted.shape) == 3
                else undistorted
            )
            edge_config = config_manager.get(
                "vision.calibration.validation.camera_validation.edge_detection", {}
            )
            canny_low = edge_config.get("canny_low_threshold", 50)
            canny_high = edge_config.get("canny_high_threshold", 150)
            edges = cv2.Canny(gray, canny_low, canny_high)

            # Detect lines
            hough_threshold = config_manager.get(
                "vision.calibration.validation.camera_validation.hough_lines_threshold",
                100,
            )
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=hough_threshold)

            if lines is not None and len(lines) > 5:
                # Measure how close lines are to being perfectly straight
                angle_deviations = []
                for line in lines:
                    rho, theta = line[0]
                    # Check deviation from horizontal/vertical
                    angle_deviation = min(
                        abs(theta),
                        abs(theta - np.pi / 2),
                        abs(theta - np.pi),
                        abs(theta - 3 * np.pi / 2),
                    )
                    angle_deviations.append(angle_deviation)

                mean_deviation = np.mean(angle_deviations)
                quality = max(
                    0.0, 1.0 - (mean_deviation / (np.pi / 4))
                )  # Normalize to 0-1
                return quality

            return 0.5  # Neutral score if no lines detected

        except Exception as e:
            logger.warning(f"Distortion test failed: {e}")
            return 0.0

    def validate_color_calibration(
        self,
        color_calibrator: ColorCalibrator,
        test_frame: NDArray[np.float64],
        ground_truth_labels: dict[str, list[tuple[int, int, int, int]]],
    ) -> ValidationResult:
        """Validate color calibration accuracy.

        Args:
            color_calibrator: Color calibrator instance
            test_frame: Test frame with known objects
            ground_truth_labels: Ground truth object regions {object_type: [(x, y, w, h), ...]}

        Returns:
            Color validation result
        """
        if color_calibrator.current_profile is None:
            return ValidationResult(
                test_name="color_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"error": "No color profile available"},
                details={"status": "failed", "reason": "missing_profile"},
            )

        try:
            hsv = cv2.cvtColor(test_frame, cv2.COLOR_BGR2HSV)

            detection_results = {}
            overall_accuracy = 0.0
            total_objects = 0

            # Test each object type
            for obj_type, regions in ground_truth_labels.items():
                if (
                    obj_type not in color_calibrator.current_profile.ball_colors
                    and obj_type != "table"
                ):
                    continue

                if obj_type == "table":
                    thresholds = color_calibrator.current_profile.table_color
                else:
                    thresholds = color_calibrator.current_profile.ball_colors[obj_type]

                # Apply color mask
                mask = thresholds.apply_mask(hsv)

                # Calculate detection accuracy for this object type
                true_positives = 0
                false_positives = 0
                total_pixels = 0

                # Create ground truth mask
                gt_mask = np.zeros(mask.shape, dtype=np.uint8)
                for x, y, w, h in regions:
                    gt_mask[y : y + h, x : x + w] = 255
                    total_pixels += w * h

                # Calculate overlap
                intersection = cv2.bitwise_and(mask, gt_mask)
                cv2.bitwise_or(mask, gt_mask)

                true_positives = np.sum(intersection > 0)
                false_positives = np.sum(mask > gt_mask)

                # Calculate metrics
                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0.0
                )
                recall = true_positives / total_pixels if total_pixels > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                detection_results[obj_type] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "total_pixels": total_pixels,
                }

                overall_accuracy += f1_score
                total_objects += 1

            overall_accuracy /= total_objects if total_objects > 0 else 1

            # Calculate false positive rate across entire image
            all_masks = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for obj_type in color_calibrator.current_profile.ball_colors:
                if obj_type in ground_truth_labels:
                    mask = color_calibrator.current_profile.ball_colors[
                        obj_type
                    ].apply_mask(hsv)
                    all_masks = cv2.bitwise_or(all_masks, mask)

            false_positive_rate = np.sum(all_masks > 0) / all_masks.size

            # Determine if validation passed
            passed = (
                overall_accuracy > self.accuracy_thresholds["color_detection_accuracy"]
                and false_positive_rate
                < self.accuracy_thresholds["color_false_positive_rate"]
            )

            return ValidationResult(
                test_name="color_calibration",
                timestamp=time.time(),
                passed=passed,
                accuracy_score=overall_accuracy,
                error_metrics={
                    "overall_accuracy": overall_accuracy,
                    "false_positive_rate": false_positive_rate,
                    "detection_results": detection_results,
                },
                details={
                    "num_object_types": total_objects,
                    "lighting_level": color_calibrator.current_profile.ambient_light_level,
                    "profile_name": color_calibrator.current_profile.name,
                },
            )

        except Exception as e:
            logger.error(f"Color validation failed: {e}")
            return ValidationResult(
                test_name="color_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"exception": str(e)},
                details={"status": "exception", "error": str(e)},
            )

    def validate_geometric_calibration(
        self,
        geometry_calibrator: GeometricCalibrator,
        test_points: list[tuple[float, float]],
        expected_world_points: list[tuple[float, float]],
    ) -> ValidationResult:
        """Validate geometric calibration accuracy.

        Args:
            geometry_calibrator: Geometric calibrator instance
            test_points: Test points in pixel coordinates
            expected_world_points: Expected world coordinates

        Returns:
            Geometry validation result
        """
        if geometry_calibrator.current_calibration is None:
            return ValidationResult(
                test_name="geometric_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"error": "No geometric calibration available"},
                details={"status": "failed", "reason": "missing_calibration"},
            )

        try:
            # Test coordinate transformation accuracy
            validation_metrics = geometry_calibrator.validate_geometry(
                test_points, expected_world_points
            )

            # Test reverse transformation
            reverse_errors = []
            for expected_world, test_pixel in zip(expected_world_points, test_points):
                # Convert world to pixel
                calculated_pixel = geometry_calibrator.world_to_pixel_coordinates(
                    expected_world
                )
                if calculated_pixel:
                    error = np.linalg.norm(
                        np.array(calculated_pixel) - np.array(test_pixel)
                    )
                    reverse_errors.append(error)

            mean_reverse_error = (
                np.mean(reverse_errors) if reverse_errors else float("inf")
            )

            # Test perspective correction quality
            perspective_quality = 0.0
            if geometry_calibrator.current_calibration.perspective_correction:
                perspective_quality = (
                    geometry_calibrator.current_calibration.perspective_correction.correction_quality
                )

            # Calculate accuracy score
            pixel_accuracy = max(
                0.0,
                1.0
                - (
                    validation_metrics["mean_error"]
                    / self.accuracy_thresholds["geometry_world_error"]
                ),
            )
            reverse_accuracy = max(
                0.0,
                1.0
                - (
                    mean_reverse_error
                    / self.accuracy_thresholds["geometry_pixel_error"]
                ),
            )

            accuracy_score = (
                pixel_accuracy + reverse_accuracy + perspective_quality
            ) / 3.0

            # Determine if validation passed
            min_perspective_quality = config_manager.get(
                "vision.calibration.validation.camera_validation.min_perspective_quality",
                0.7,
            )
            passed = (
                validation_metrics["mean_error"]
                < self.accuracy_thresholds["geometry_world_error"]
                and mean_reverse_error
                < self.accuracy_thresholds["geometry_pixel_error"]
                and perspective_quality > min_perspective_quality
            )

            return ValidationResult(
                test_name="geometric_calibration",
                timestamp=time.time(),
                passed=passed,
                accuracy_score=accuracy_score,
                error_metrics={
                    "forward_mean_error": validation_metrics["mean_error"],
                    "forward_max_error": validation_metrics["max_error"],
                    "reverse_mean_error": mean_reverse_error,
                    "perspective_quality": perspective_quality,
                },
                details={
                    "num_test_points": len(test_points),
                    "calibration_error": geometry_calibrator.current_calibration.calibration_error,
                    "table_dimensions": geometry_calibrator.current_calibration.table_dimensions_real,
                },
            )

        except Exception as e:
            logger.error(f"Geometry validation failed: {e}")
            return ValidationResult(
                test_name="geometric_calibration",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"exception": str(e)},
                details={"status": "exception", "error": str(e)},
            )

    def validate_integration(
        self,
        camera_calibrator: CameraCalibrator,
        color_calibrator: ColorCalibrator,
        geometry_calibrator: GeometricCalibrator,
        test_frame: NDArray[np.float64],
        ground_truth: dict[str, Any],
    ) -> ValidationResult:
        """Validate end-to-end calibration integration.

        Args:
            camera_calibrator: Camera calibrator instance
            color_calibrator: Color calibrator instance
            geometry_calibrator: Geometric calibrator instance
            test_frame: Test frame
            ground_truth: Ground truth data with ball positions, table corners, etc.

        Returns:
            Integration validation result
        """
        try:
            # Apply all calibrations in sequence
            corrected_frame = test_frame.copy()

            # 1. Apply camera distortion correction
            if camera_calibrator.camera_params:
                corrected_frame = camera_calibrator.undistort_image(corrected_frame)

            # 2. Apply perspective correction
            if (
                geometry_calibrator.current_calibration
                and geometry_calibrator.current_calibration.perspective_correction
            ):
                corrected_frame = geometry_calibrator.correct_keystone_distortion(
                    corrected_frame,
                    geometry_calibrator.current_calibration.perspective_correction,
                )

            # 3. Test ball detection with color calibration
            ball_detection_accuracy = 0.0
            if color_calibrator.current_profile and "balls" in ground_truth:
                detected_balls = self._detect_balls_with_color_calibration(
                    corrected_frame, color_calibrator
                )
                expected_balls = ground_truth["balls"]
                ball_detection_accuracy = self._calculate_ball_detection_accuracy(
                    detected_balls, expected_balls
                )

            # 4. Test table detection
            table_detection_accuracy = 0.0
            if "table_corners" in ground_truth:
                if geometry_calibrator.current_calibration:
                    detected_corners = (
                        geometry_calibrator.current_calibration.table_corners_pixel
                    )
                    expected_corners = ground_truth["table_corners"]
                    table_detection_accuracy = self._calculate_table_detection_accuracy(
                        detected_corners, expected_corners
                    )

            # Calculate overall integration score
            accuracy_score = (ball_detection_accuracy + table_detection_accuracy) / 2.0

            # Determine if validation passed
            passed = (
                ball_detection_accuracy
                > self.accuracy_thresholds["integration_ball_accuracy"]
                and table_detection_accuracy
                > self.accuracy_thresholds["integration_table_accuracy"]
            )

            return ValidationResult(
                test_name="integration_validation",
                timestamp=time.time(),
                passed=passed,
                accuracy_score=accuracy_score,
                error_metrics={
                    "ball_detection_accuracy": ball_detection_accuracy,
                    "table_detection_accuracy": table_detection_accuracy,
                },
                details={
                    "camera_calibrated": camera_calibrator.camera_params is not None,
                    "color_calibrated": color_calibrator.current_profile is not None,
                    "geometry_calibrated": geometry_calibrator.current_calibration
                    is not None,
                },
            )

        except Exception as e:
            logger.error(f"Integration validation failed: {e}")
            return ValidationResult(
                test_name="integration_validation",
                timestamp=time.time(),
                passed=False,
                accuracy_score=0.0,
                error_metrics={"exception": str(e)},
                details={"status": "exception", "error": str(e)},
            )

    def _detect_balls_with_color_calibration(
        self, frame: NDArray[np.uint8], color_calibrator: ColorCalibrator
    ) -> list[tuple[float, float]]:
        """Detect balls using color calibration."""
        if not color_calibrator.current_profile:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_balls = []

        for (
            _ball_type,
            thresholds,
        ) in color_calibrator.current_profile.ball_colors.items():
            mask = thresholds.apply_mask(hsv)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_area = config_manager.get(
                "vision.calibration.validation.color_validation.min_area_threshold", 100
            )
            min_circularity = config_manager.get(
                "vision.calibration.validation.color_validation.min_circularity", 0.5
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:  # Minimum area threshold
                    continue

                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < min_circularity:  # Minimum circularity for balls
                    continue

                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    detected_balls.append((cx, cy))

        return detected_balls

    def _calculate_ball_detection_accuracy(
        self, detected: list[tuple[float, float]], expected: list[tuple[float, float]]
    ) -> float:
        """Calculate ball detection accuracy."""
        if not expected:
            return 1.0 if not detected else 0.0

        if not detected:
            return 0.0

        # Match detected balls to expected balls (nearest neighbor)
        matched = 0
        threshold = config_manager.get(
            "vision.calibration.validation.ball_detection.matching_threshold_pixels",
            50.0,
        )

        for exp_ball in expected:
            min_distance = float("inf")
            for det_ball in detected:
                distance = np.linalg.norm(np.array(exp_ball) - np.array(det_ball))
                if distance < min_distance:
                    min_distance = distance

            if min_distance < threshold:
                matched += 1

        return matched / len(expected)

    def _calculate_table_detection_accuracy(
        self, detected: list[tuple[float, float]], expected: list[tuple[float, float]]
    ) -> float:
        """Calculate table detection accuracy."""
        if len(detected) != 4 or len(expected) != 4:
            return 0.0

        # Calculate average distance between corresponding corners
        errors = []
        for det_corner, exp_corner in zip(detected, expected):
            error = np.linalg.norm(np.array(det_corner) - np.array(exp_corner))
            errors.append(error)

        mean_error = np.mean(errors)
        normalization_factor = config_manager.get(
            "vision.calibration.validation.ball_detection.normalization_factor_pixels",
            100.0,
        )
        accuracy = max(0.0, 1.0 - (mean_error / normalization_factor))  # Normalize

        return accuracy

    def generate_comprehensive_report(
        self,
        camera_calibrator: CameraCalibrator,
        color_calibrator: ColorCalibrator,
        geometry_calibrator: GeometricCalibrator,
        test_data: dict[str, Any],
    ) -> CalibrationReport:
        """Generate comprehensive calibration validation report.

        Args:
            camera_calibrator: Camera calibrator instance
            color_calibrator: Color calibrator instance
            geometry_calibrator: Geometric calibrator instance
            test_data: Test data including images, ground truth, etc.

        Returns:
            Complete calibration report
        """
        import uuid
        from datetime import datetime

        session_id = str(uuid.uuid4())
        test_date = datetime.now().isoformat()

        # Run all validation tests
        camera_validation = None
        if "camera_test_images" in test_data:
            camera_validation = self.validate_camera_calibration(
                camera_calibrator, test_data["camera_test_images"]
            )

        color_validation = None
        if "color_test_frame" in test_data and "color_ground_truth" in test_data:
            color_validation = self.validate_color_calibration(
                color_calibrator,
                test_data["color_test_frame"],
                test_data["color_ground_truth"],
            )

        geometry_validation = None
        if (
            "geometry_test_points" in test_data
            and "geometry_expected_points" in test_data
        ):
            geometry_validation = self.validate_geometric_calibration(
                geometry_calibrator,
                test_data["geometry_test_points"],
                test_data["geometry_expected_points"],
            )

        integration_validation = None
        if (
            "integration_test_frame" in test_data
            and "integration_ground_truth" in test_data
        ):
            integration_validation = self.validate_integration(
                camera_calibrator,
                color_calibrator,
                geometry_calibrator,
                test_data["integration_test_frame"],
                test_data["integration_ground_truth"],
            )

        # Calculate overall score
        scores = []
        if camera_validation:
            scores.append(camera_validation.accuracy_score)
        if color_validation:
            scores.append(color_validation.accuracy_score)
        if geometry_validation:
            scores.append(geometry_validation.accuracy_score)
        if integration_validation:
            scores.append(integration_validation.accuracy_score)

        overall_score = np.mean(scores) if scores else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            camera_validation,
            color_validation,
            geometry_validation,
            integration_validation,
        )

        report = CalibrationReport(
            session_id=session_id,
            test_date=test_date,
            camera_validation=camera_validation,
            color_validation=color_validation,
            geometry_validation=geometry_validation,
            integration_validation=integration_validation,
            overall_score=overall_score,
            recommendations=recommendations,
        )

        # Save report
        self._save_report(report)

        return report

    def _generate_recommendations(
        self,
        camera_val: Optional[ValidationResult],
        color_val: Optional[ValidationResult],
        geometry_val: Optional[ValidationResult],
        integration_val: Optional[ValidationResult],
    ) -> list[str]:
        """Generate calibration improvement recommendations."""
        recommendations = []

        integration_config = config_manager.get(
            "vision.calibration.validation.integration_validation", {}
        )

        if camera_val and not camera_val.passed:
            reprojection_threshold = integration_config.get(
                "reprojection_error_threshold_pixels", 3.0
            )
            detection_rate_threshold = integration_config.get("min_detection_rate", 0.8)

            if (
                camera_val.error_metrics.get("mean_reprojection_error", 0)
                > reprojection_threshold
            ):
                recommendations.append(
                    "Camera calibration needs improvement. Take more calibration images with better chessboard visibility."
                )
            if (
                camera_val.error_metrics.get("detection_rate", 0)
                < detection_rate_threshold
            ):
                recommendations.append(
                    "Improve chessboard detection rate by ensuring good lighting and focus."
                )

        if color_val and not color_val.passed:
            color_accuracy_threshold = integration_config.get("min_color_accuracy", 0.8)
            false_positive_threshold = integration_config.get(
                "max_false_positive_rate", 0.15
            )

            if (
                color_val.error_metrics.get("overall_accuracy", 0)
                < color_accuracy_threshold
            ):
                recommendations.append(
                    "Color calibration accuracy is low. Recalibrate color thresholds with better samples."
                )
            if (
                color_val.error_metrics.get("false_positive_rate", 1)
                > false_positive_threshold
            ):
                recommendations.append(
                    "Too many false color detections. Tighten color thresholds."
                )

        if geometry_val and not geometry_val.passed:
            geometry_error_threshold = integration_config.get(
                "max_geometry_error_meters", 0.03
            )
            perspective_quality_threshold = integration_config.get(
                "min_perspective_quality", 0.7
            )

            if (
                geometry_val.error_metrics.get("forward_mean_error", 0)
                > geometry_error_threshold
            ):
                recommendations.append(
                    "Geometric calibration has high coordinate transformation error. Redetect table corners."
                )
            if (
                geometry_val.error_metrics.get("perspective_quality", 0)
                < perspective_quality_threshold
            ):
                recommendations.append(
                    "Perspective correction quality is low. Ensure table corners are accurately detected."
                )

        if integration_val and not integration_val.passed:
            recommendations.append(
                "End-to-end system integration needs improvement. Check all calibration components."
            )

        if not recommendations:
            recommendations.append(
                "All calibrations are performing well. System is ready for operation."
            )

        return recommendations

    def _save_report(self, report: CalibrationReport) -> None:
        """Save validation report to cache."""
        try:
            report_file = (
                self.cache_dir / f"validation_report_{report.session_id[:8]}.json"
            )
            with open(report_file, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Validation report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    def load_report(self, session_id: str) -> Optional[CalibrationReport]:
        """Load validation report by session ID."""
        try:
            report_file = self.cache_dir / f"validation_report_{session_id[:8]}.json"
            with open(report_file) as f:
                data = json.load(f)
            return CalibrationReport.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load validation report: {e}")
            return None

    def stability_test(
        self,
        calibrators: dict[str, Any],
        test_frames: list[np.ndarray],
        duration_minutes: int = 10,
    ) -> dict[str, Any]:
        """Test calibration stability over time.

        Args:
            calibrators: Dictionary of calibrator instances
            test_frames: Sequence of test frames
            duration_minutes: Test duration in minutes

        Returns:
            Stability test results
        """
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        stability_metrics = {
            "camera": [],
            "color": [],
            "geometry": [],
            "timestamps": [],
        }

        frame_idx = 0
        while time.time() < end_time:
            current_time = time.time()
            test_frame = test_frames[frame_idx % len(test_frames)]

            # Test camera stability (if available)
            if "camera" in calibrators and calibrators["camera"].camera_params:
                # Measure processing time consistency
                start_proc = time.time()
                calibrators["camera"].undistort_image(test_frame)
                proc_time = time.time() - start_proc
                stability_metrics["camera"].append(proc_time)

            # Test color stability
            if "color" in calibrators and calibrators["color"].current_profile:
                # Measure color detection consistency
                light_level = calibrators["color"]._estimate_ambient_light(test_frame)
                stability_metrics["color"].append(light_level)

            # Test geometry stability
            if (
                "geometry" in calibrators
                and calibrators["geometry"].current_calibration
            ):
                # Test coordinate transformation consistency
                test_point = (test_frame.shape[1] // 2, test_frame.shape[0] // 2)
                world_point = calibrators["geometry"].pixel_to_world_coordinates(
                    test_point
                )
                if world_point:
                    stability_metrics["geometry"].append(world_point)

            stability_metrics["timestamps"].append(current_time - start_time)

            frame_idx += 1
            sleep_interval = config_manager.get(
                "vision.calibration.validation.stability_test.sleep_interval_seconds",
                1.0,
            )
            time.sleep(sleep_interval)  # Process frames at configured interval

        # Calculate stability statistics
        results = {}

        for metric_type, values in stability_metrics.items():
            if metric_type == "timestamps":
                continue

            if values:
                if metric_type in ["camera", "color"]:
                    # Scalar values
                    values_array = np.array(values)
                    results[metric_type] = {
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array)),
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "stability_score": max(
                            0.0, 1.0 - (np.std(values_array) / np.mean(values_array))
                        ),
                    }
                else:
                    # Vector values (geometry)
                    if values and len(values[0]) == 2:
                        x_values = [v[0] for v in values]
                        y_values = [v[1] for v in values]
                        results[metric_type] = {
                            "x_std": float(np.std(x_values)),
                            "y_std": float(np.std(y_values)),
                            "position_stability": max(
                                0.0,
                                1.0
                                - np.sqrt(
                                    np.std(x_values) ** 2 + np.std(y_values) ** 2
                                ),
                            ),
                        }

        results["test_duration"] = duration_minutes
        results["total_frames_tested"] = frame_idx

        return results
