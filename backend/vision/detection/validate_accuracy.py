"""Ball Detection Accuracy Validation Script.

This script validates that the ball detection system meets the >95% accuracy
requirement as specified in FR-VIS-020 to FR-VIS-029.
"""

import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .ball_tracker import BallTrackingSystem
from .balls import BallDetector, DetectionMethod


class AccuracyValidator:
    """Validates ball detection accuracy against known ground truth."""

    def __init__(self):
        self.detector = BallDetector(
            {
                "detection_method": DetectionMethod.COMBINED,
                "min_radius": 15,
                "max_radius": 25,
                "expected_radius": 20,
                "min_confidence": 0.3,
                "debug_mode": False,
            }
        )

        self.tracking_system = BallTrackingSystem(
            {
                "detection_method": DetectionMethod.COMBINED,
                "enable_tracking": True,
                "position_accuracy_threshold": 2.0,
                "debug_mode": False,
            }
        )

    def create_synthetic_frame(
        self, ball_configs: list, frame_size=(600, 800), noise_level=10
    ):
        """Create synthetic frame with known ball positions.

        Args:
            ball_configs: List of (x, y, radius, color) tuples
            frame_size: (height, width) of frame
            noise_level: Amount of noise to add for realism

        Returns:
            frame: Synthetic frame
            ground_truth: List of ground truth ball positions
        """
        height, width = frame_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create realistic table background
        frame[:, :] = (35, 75, 35)  # Green felt

        # Add texture/noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ground_truth = []

        for x, y, radius, color in ball_configs:
            # Ensure ball is within frame
            if radius < x < width - radius and radius < y < height - radius:
                # Draw realistic ball with shading
                self._draw_realistic_ball(frame, (x, y), radius, color)
                ground_truth.append(
                    {"position": (x, y), "radius": radius, "color": color}
                )

        return frame, ground_truth

    def _draw_realistic_ball(self, frame, center, radius, color) -> None:
        """Draw a realistic-looking ball with shading."""
        x, y = center

        # Main ball body
        cv2.circle(frame, center, radius, color, -1)

        # Add highlight for 3D effect
        highlight_center = (x - radius // 3, y - radius // 3)
        highlight_radius = max(2, radius // 4)
        highlight_color = tuple(min(255, c + 60) for c in color)
        cv2.circle(frame, highlight_center, highlight_radius, highlight_color, -1)

        # Add shadow/darker edge
        shadow_color = tuple(max(0, c - 30) for c in color)
        cv2.circle(frame, center, radius, shadow_color, 2)

        # Add slight gradient
        for i in range(3):
            gradient_radius = radius - i * 2
            if gradient_radius > 0:
                gradient_color = tuple(max(0, c - i * 5) for c in color)
                cv2.circle(frame, center, gradient_radius, gradient_color, 1)

    def generate_test_scenarios(self, num_scenarios=50) -> None:
        """Generate diverse test scenarios."""
        scenarios = []

        for _ in range(num_scenarios):
            # Random number of balls (3-12)
            num_balls = random.randint(3, 12)

            ball_configs = []
            attempts = 0
            max_attempts = 100

            while len(ball_configs) < num_balls and attempts < max_attempts:
                attempts += 1

                # Random position
                x = random.randint(50, 750)
                y = random.randint(50, 550)
                radius = random.randint(18, 22)

                # Random color
                colors = [
                    (255, 255, 255),  # White
                    (0, 0, 0),  # Black
                    (255, 255, 0),  # Yellow
                    (0, 0, 255),  # Red
                    (0, 255, 0),  # Green
                    (255, 0, 255),  # Purple
                    (255, 128, 0),  # Orange
                    (0, 255, 255),  # Cyan
                ]
                color = random.choice(colors)

                # Check for overlap with existing balls
                valid = True
                for existing_x, existing_y, existing_r, _ in ball_configs:
                    distance = np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2)
                    if distance < (radius + existing_r + 10):  # Minimum separation
                        valid = False
                        break

                if valid:
                    ball_configs.append((x, y, radius, color))

            scenarios.append(ball_configs)

        return scenarios

    def test_detection_accuracy(self, scenarios) -> None:
        """Test detection accuracy across multiple scenarios."""
        results = {
            "total_scenarios": len(scenarios),
            "total_ground_truth_balls": 0,
            "total_detected_balls": 0,
            "correct_detections": 0,
            "false_positives": 0,
            "missed_balls": 0,
            "position_errors": [],
            "scenario_accuracies": [],
        }

        print(f"Testing detection accuracy across {len(scenarios)} scenarios...")

        for i, ball_configs in enumerate(scenarios):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(scenarios)} scenarios")

            # Create frame and ground truth
            frame, ground_truth = self.create_synthetic_frame(ball_configs)

            # Detect balls
            detected_balls = self.detector.detect_balls(frame)

            # Analyze results
            scenario_correct = 0
            scenario_total = len(ground_truth)

            for gt_ball in ground_truth:
                gt_pos = gt_ball["position"]
                matched = False

                for det_ball in detected_balls:
                    det_pos = det_ball.position
                    distance = np.sqrt(
                        (gt_pos[0] - det_pos[0]) ** 2 + (gt_pos[1] - det_pos[1]) ** 2
                    )

                    if distance <= 5.0:  # 5 pixel tolerance for matching
                        scenario_correct += 1
                        results["correct_detections"] += 1
                        results["position_errors"].append(distance)
                        matched = True
                        break

                if not matched:
                    results["missed_balls"] += 1

            # Count false positives (detections not matching any ground truth)
            for det_ball in detected_balls:
                det_pos = det_ball.position
                matched = False

                for gt_ball in ground_truth:
                    gt_pos = gt_ball["position"]
                    distance = np.sqrt(
                        (gt_pos[0] - det_pos[0]) ** 2 + (gt_pos[1] - det_pos[1]) ** 2
                    )

                    if distance <= 5.0:
                        matched = True
                        break

                if not matched:
                    results["false_positives"] += 1

            # Calculate scenario accuracy
            scenario_accuracy = (
                scenario_correct / scenario_total if scenario_total > 0 else 0.0
            )
            results["scenario_accuracies"].append(scenario_accuracy)

            results["total_ground_truth_balls"] += scenario_total
            results["total_detected_balls"] += len(detected_balls)

        return results

    def test_tracking_accuracy(self, num_frames=20) -> None:
        """Test tracking accuracy over multiple frames."""
        print(f"Testing tracking accuracy over {num_frames} frames...")

        # Create a ball that moves in a predictable pattern
        ball_trajectory = []
        for frame_num in range(num_frames):
            x = 200 + frame_num * 10  # Moving right
            y = 300 + int(20 * np.sin(frame_num * 0.5))  # Slight vertical oscillation
            ball_trajectory.append((x, y, 20, (255, 255, 255)))

        tracking_results = {
            "total_frames": num_frames,
            "frames_with_detection": 0,
            "frames_with_tracking": 0,
            "position_errors": [],
            "velocity_estimates": [],
        }

        for frame_num in range(num_frames):
            ball_config = [ball_trajectory[frame_num]]
            frame, ground_truth = self.create_synthetic_frame(ball_config)

            # Process with tracking system
            result = self.tracking_system.process_frame(frame, frame_num)

            if result.balls:
                tracking_results["frames_with_detection"] += 1

                # Check tracking accuracy
                detected_ball = result.balls[0]
                gt_pos = ground_truth[0]["position"]
                det_pos = detected_ball.position

                position_error = np.sqrt(
                    (gt_pos[0] - det_pos[0]) ** 2 + (gt_pos[1] - det_pos[1]) ** 2
                )
                tracking_results["position_errors"].append(position_error)

                if (
                    hasattr(detected_ball, "track_id")
                    and detected_ball.track_id is not None
                ):
                    tracking_results["frames_with_tracking"] += 1

                # Record velocity if available
                if hasattr(detected_ball, "velocity"):
                    velocity_magnitude = np.sqrt(
                        detected_ball.velocity[0] ** 2 + detected_ball.velocity[1] ** 2
                    )
                    tracking_results["velocity_estimates"].append(velocity_magnitude)

        return tracking_results

    def test_performance_requirements(self) -> None:
        """Test performance requirements (processing speed)."""
        print("Testing performance requirements...")

        # Test frame
        ball_configs = [(200, 200, 20, (255, 255, 255)), (400, 300, 20, (255, 255, 0))]
        frame, _ = self.create_synthetic_frame(ball_configs)

        # Measure processing times
        processing_times = []
        num_iterations = 100

        for _ in range(num_iterations):
            start_time = time.perf_counter()
            self.detector.detect_balls(frame)
            end_time = time.perf_counter()

            processing_times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)

        # Calculate FPS
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0

        return {
            "average_processing_time_ms": avg_time,
            "max_processing_time_ms": max_time,
            "min_processing_time_ms": min_time,
            "average_fps": avg_fps,
            "meets_30fps_requirement": avg_fps >= 30,
        }

    def print_results(
        self, detection_results, tracking_results, performance_results
    ) -> None:
        """Print comprehensive results."""
        print("\n" + "=" * 80)
        print("BALL DETECTION ACCURACY VALIDATION RESULTS")
        print("=" * 80)

        # Detection accuracy
        print("\n1. DETECTION ACCURACY")
        print("-" * 40)

        total_gt = detection_results["total_ground_truth_balls"]
        correct = detection_results["correct_detections"]
        false_pos = detection_results["false_positives"]
        missed = detection_results["missed_balls"]

        detection_rate = (correct / total_gt * 100) if total_gt > 0 else 0
        precision = (
            (correct / (correct + false_pos) * 100) if (correct + false_pos) > 0 else 0
        )
        recall = (correct / (correct + missed) * 100) if (correct + missed) > 0 else 0

        print(f"Total scenarios tested: {detection_results['total_scenarios']}")
        print(f"Total ground truth balls: {total_gt}")
        print(f"Correct detections: {correct}")
        print(f"False positives: {false_pos}")
        print(f"Missed balls: {missed}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"Precision: {precision:.1f}%")
        print(f"Recall: {recall:.1f}%")

        if detection_results["position_errors"]:
            avg_pos_error = np.mean(detection_results["position_errors"])
            max_pos_error = np.max(detection_results["position_errors"])
            print(f"Average position error: {avg_pos_error:.2f} pixels")
            print(f"Maximum position error: {max_pos_error:.2f} pixels")

            # Check ±2 pixel requirement
            within_2px = sum(
                1 for err in detection_results["position_errors"] if err <= 2.0
            )
            accuracy_2px = within_2px / len(detection_results["position_errors"]) * 100
            print(f"Positions within ±2 pixels: {accuracy_2px:.1f}%")

        # Scenario accuracy distribution
        if detection_results["scenario_accuracies"]:
            scenarios_95_plus = sum(
                1 for acc in detection_results["scenario_accuracies"] if acc >= 0.95
            )
            pct_95_plus = (
                scenarios_95_plus / len(detection_results["scenario_accuracies"]) * 100
            )
            print(f"Scenarios with ≥95% accuracy: {pct_95_plus:.1f}%")

        # Tracking accuracy
        print("\n2. TRACKING ACCURACY")
        print("-" * 40)

        detection_consistency = (
            tracking_results["frames_with_detection"]
            / tracking_results["total_frames"]
            * 100
        )
        tracking_consistency = (
            tracking_results["frames_with_tracking"]
            / tracking_results["total_frames"]
            * 100
        )

        print(f"Frame detection consistency: {detection_consistency:.1f}%")
        print(f"Frame tracking consistency: {tracking_consistency:.1f}%")

        if tracking_results["position_errors"]:
            avg_track_error = np.mean(tracking_results["position_errors"])
            max_track_error = np.max(tracking_results["position_errors"])
            print(f"Average tracking error: {avg_track_error:.2f} pixels")
            print(f"Maximum tracking error: {max_track_error:.2f} pixels")

        # Performance
        print("\n3. PERFORMANCE REQUIREMENTS")
        print("-" * 40)

        print(
            f"Average processing time: {performance_results['average_processing_time_ms']:.2f} ms"
        )
        print(f"Average FPS: {performance_results['average_fps']:.1f}")
        print(
            f"Meets 30 FPS requirement: {'✓' if performance_results['meets_30fps_requirement'] else '✗'}"
        )

        # Overall assessment
        print("\n4. REQUIREMENT COMPLIANCE")
        print("-" * 40)

        requirements_met = []

        # FR-VIS-020: Detect all balls on table surface
        requirements_met.append(("FR-VIS-020 (Ball detection)", detection_rate >= 95))

        # FR-VIS-023: Track positions with ±2 pixel accuracy
        accuracy_2px = (
            (
                sum(1 for err in detection_results["position_errors"] if err <= 2.0)
                / len(detection_results["position_errors"])
                * 100
            )
            if detection_results["position_errors"]
            else 0
        )
        requirements_met.append(("FR-VIS-023 (±2 pixel accuracy)", accuracy_2px >= 95))

        # Performance requirement
        requirements_met.append(
            ("Performance (30+ FPS)", performance_results["meets_30fps_requirement"])
        )

        for req_name, met in requirements_met:
            status = "✓ PASS" if met else "✗ FAIL"
            print(f"{req_name}: {status}")

        # Overall score
        total_met = sum(1 for _, met in requirements_met if met)
        overall_score = total_met / len(requirements_met) * 100

        print(
            f"\nOVERALL COMPLIANCE: {overall_score:.1f}% ({total_met}/{len(requirements_met)} requirements met)"
        )

        if overall_score >= 95:
            print("✓ SYSTEM MEETS >95% ACCURACY REQUIREMENT")
        else:
            print("✗ SYSTEM DOES NOT MEET 95% ACCURACY REQUIREMENT")

        return overall_score >= 95


def main():
    """Run comprehensive accuracy validation."""
    print("Ball Detection System - Accuracy Validation")
    print("Requirements: FR-VIS-020 to FR-VIS-029")
    print("Target: >95% accuracy with robust tracking\n")

    validator = AccuracyValidator()

    try:
        # Generate test scenarios
        scenarios = validator.generate_test_scenarios(
            num_scenarios=25
        )  # Reduced for faster testing

        # Test detection accuracy
        detection_results = validator.test_detection_accuracy(scenarios)

        # Test tracking accuracy
        tracking_results = validator.test_tracking_accuracy(num_frames=10)

        # Test performance
        performance_results = validator.test_performance_requirements()

        # Print comprehensive results
        success = validator.print_results(
            detection_results, tracking_results, performance_results
        )

        return success

    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
