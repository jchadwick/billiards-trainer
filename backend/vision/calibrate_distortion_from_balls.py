#!/usr/bin/env python3
"""Camera distortion calibration using ball grid pattern.

This script detects billiard balls arranged in a regular grid pattern and uses
their positions to calibrate camera lens distortion. The balls should form a
regular rectangular grid, but barrel/pincushion distortion will make them appear
curved. By comparing detected positions to an ideal grid, we can solve for the
camera's distortion coefficients.

Usage:
    python calibrate_distortion_from_balls.py [--image PATH] [--output PATH]

The script will:
1. Detect all balls in the image using blob detection and Hough circles
2. Analyze ball positions to infer the grid structure (rows/columns)
3. Sort balls into their grid positions
4. Calculate ideal grid positions (no distortion, regular spacing)
5. Use OpenCV's calibrateCamera to solve for distortion coefficients
6. Save calibration to YAML file
7. Validate by applying undistortion and measuring grid improvement
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BallGridCalibrator:
    """Calibrate camera distortion using a grid of billiard balls."""

    def __init__(self, min_ball_radius: int = 20, max_ball_radius: int = 80):
        """Initialize calibrator.

        Args:
            min_ball_radius: Minimum ball radius in pixels
            max_ball_radius: Maximum ball radius in pixels
        """
        self.min_ball_radius = min_ball_radius
        self.max_ball_radius = max_ball_radius
        self.debug_images = {}

    def detect_balls(
        self, image: NDArray[np.uint8]
    ) -> list[tuple[float, float, float]]:
        """Detect all balls in the image.

        Uses Hough circles and filters out detections near image borders
        (likely pockets or false positives).

        Args:
            image: Input BGR image

        Returns:
            List of (x, y, radius) tuples for each detected ball
        """
        logger.info("Detecting balls in image...")

        h, w = image.shape[:2]

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_ball_radius * 2,  # Balls should be at least 2*radius apart
            param1=50,  # Canny edge threshold
            param2=30,  # Accumulator threshold (lower = more circles)
            minRadius=self.min_ball_radius,
            maxRadius=self.max_ball_radius,
        )

        balls = []
        filtered_balls = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            # Filter out circles near borders and with unusual colors
            border_margin = max(w, h) * 0.05  # 5% margin

            for x, y, r in circles:
                # Skip if too close to border (likely pockets)
                if (
                    x < border_margin
                    or x > w - border_margin
                    or y < border_margin
                    or y > h - border_margin
                ):
                    balls.append((float(x), float(y), float(r)))
                    continue

                # Check color - billiard balls should be colorful, not dark/black
                roi_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(roi_mask, (x, y), r, 255, -1)
                mean_val = cv2.mean(image, mask=roi_mask)

                # Skip if too dark (likely pockets/shadows)
                brightness = (mean_val[0] + mean_val[1] + mean_val[2]) / 3
                if brightness < 50:
                    balls.append((float(x), float(y), float(r)))
                    continue

                filtered_balls.append((float(x), float(y), float(r)))

        logger.info(
            f"Detected {len(circles[0]) if circles is not None else 0} circles total, "
            f"{len(filtered_balls)} passed filters"
        )

        # Create debug visualization showing all detections
        debug_img = image.copy()

        # Draw filtered out balls in red
        for x, y, r in balls:
            cv2.circle(debug_img, (int(x), int(y)), int(r), (0, 0, 255), 2)

        # Draw valid balls in green
        for x, y, r in filtered_balls:
            cv2.circle(debug_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(x), int(y)), 2, (0, 0, 255), 3)

        self.debug_images["detected_balls"] = debug_img

        return filtered_balls

    def infer_grid_structure(
        self, balls: list[tuple[float, float, float]]
    ) -> tuple[int, int, NDArray[np.float32]]:
        """Infer grid structure from ball positions.

        Analyzes ball positions to determine rows and columns, then sorts
        balls into their grid positions. Uses clustering to group balls
        by row and column positions.

        Args:
            balls: List of (x, y, radius) tuples

        Returns:
            Tuple of (rows, cols, sorted_centers) where sorted_centers is
            a (rows, cols, 2) array of ball centers
        """
        logger.info("Inferring grid structure from ball positions...")

        # Extract just the centers
        centers = np.array([(x, y) for x, y, _ in balls], dtype=np.float32)

        if len(centers) < 4:
            raise ValueError(f"Need at least 4 balls to infer grid, got {len(centers)}")

        # Use clustering on y-coordinates to find rows
        from sklearn.cluster import DBSCAN

        y_coords = centers[:, 1].reshape(-1, 1)

        # Estimate epsilon from y-coordinate spread
        y_range = np.ptp(y_coords)
        # Assume at least 3 rows, so max spacing is y_range/2
        eps_y = y_range / 6  # Conservative estimate

        clustering_y = DBSCAN(eps=eps_y, min_samples=2).fit(y_coords)
        row_labels = clustering_y.labels_

        # Filter out noise points (label = -1)
        valid_mask = row_labels >= 0
        centers_filtered = centers[valid_mask]
        row_labels_filtered = row_labels[valid_mask]

        if len(centers_filtered) < 4:
            logger.warning(
                f"Too many noise points. Using {len(centers_filtered)} balls."
            )
            if len(centers_filtered) == 0:
                raise ValueError(
                    "All balls classified as noise. Try adjusting detection parameters."
                )

        # Group into rows
        unique_rows = np.unique(row_labels_filtered)
        num_rows = len(unique_rows)

        # Create row mapping (sorted by average y-coordinate)
        row_avg_y = []
        for row_label in unique_rows:
            mask = row_labels_filtered == row_label
            avg_y = np.mean(centers_filtered[mask, 1])
            row_avg_y.append((row_label, avg_y))

        row_avg_y.sort(key=lambda x: x[1])
        row_mapping = {old: new for new, (old, _) in enumerate(row_avg_y)}

        # Build rows list
        rows = [[] for _ in range(num_rows)]
        for center, row_label in zip(centers_filtered, row_labels_filtered):
            row_idx = row_mapping[row_label]
            rows[row_idx].append(center)

        # Convert to arrays and sort by x-coordinate
        for i in range(num_rows):
            rows[i] = np.array(rows[i])
            rows[i] = rows[i][np.argsort(rows[i][:, 0])]

        # Determine number of columns (most common row length)
        row_lengths = [len(row) for row in rows]
        from collections import Counter

        num_cols = Counter(row_lengths).most_common(1)[0][0]

        logger.info(f"Detected {num_rows} rows × {num_cols} columns grid")
        logger.info(f"Balls per row: {row_lengths}")

        # Create grid array
        grid_centers = np.full((num_rows, num_cols, 2), np.nan, dtype=np.float32)

        for row_idx, row in enumerate(rows):
            if len(row) == num_cols:
                # Perfect match - fill directly
                grid_centers[row_idx, :, :] = row
            elif len(row) < num_cols:
                # Fewer balls - try to align them properly
                # Use x-coordinates to determine column positions
                if len(row) > 0:
                    # Find expected column positions based on other rows
                    ref_x_positions = []
                    for other_row in rows:
                        if len(other_row) == num_cols:
                            ref_x_positions = other_row[:, 0]
                            break

                    if len(ref_x_positions) > 0:
                        # Match each ball to closest reference position
                        for ball in row:
                            distances = np.abs(ref_x_positions - ball[0])
                            col_idx = np.argmin(distances)
                            grid_centers[row_idx, col_idx] = ball
                    else:
                        # No reference - distribute evenly
                        start_col = (num_cols - len(row)) // 2
                        for col_idx, center in enumerate(row):
                            grid_centers[row_idx, start_col + col_idx] = center
            else:
                # More balls than expected - take first num_cols
                logger.warning(
                    f"Row {row_idx} has {len(row)} balls, expected {num_cols}"
                )
                grid_centers[row_idx, :, :] = row[:num_cols]

        return num_rows, num_cols, grid_centers

    def calculate_ideal_grid(
        self, grid_centers: NDArray[np.float32], rows: int, cols: int
    ) -> NDArray[np.float32]:
        """Calculate ideal grid positions (no distortion).

        Uses the average spacing to determine what the grid should look like
        with perfect regularity.

        Args:
            grid_centers: (rows, cols, 2) array of detected ball centers
            rows: Number of rows
            cols: Number of columns

        Returns:
            (rows, cols, 2) array of ideal ball positions
        """
        logger.info("Calculating ideal grid positions...")

        # Filter out NaN values for spacing calculation
        valid_centers = []
        for r in range(rows):
            for c in range(cols):
                if not np.isnan(grid_centers[r, c, 0]):
                    valid_centers.append(grid_centers[r, c])

        valid_centers = np.array(valid_centers)

        # Calculate average spacing
        # Horizontal spacing
        h_spacings = []
        for r in range(rows):
            row_centers = grid_centers[r, :, :]
            valid_in_row = row_centers[~np.isnan(row_centers[:, 0])]
            if len(valid_in_row) > 1:
                for i in range(len(valid_in_row) - 1):
                    h_spacings.append(
                        np.linalg.norm(valid_in_row[i + 1] - valid_in_row[i])
                    )

        # Vertical spacing
        v_spacings = []
        for c in range(cols):
            col_centers = grid_centers[:, c, :]
            valid_in_col = col_centers[~np.isnan(col_centers[:, 0])]
            if len(valid_in_col) > 1:
                for i in range(len(valid_in_col) - 1):
                    v_spacings.append(
                        np.linalg.norm(valid_in_col[i + 1] - valid_in_col[i])
                    )

        avg_h_spacing = np.mean(h_spacings) if h_spacings else 100.0
        avg_v_spacing = np.mean(v_spacings) if v_spacings else 100.0

        logger.info(
            f"Average spacing: horizontal={avg_h_spacing:.1f}px, vertical={avg_v_spacing:.1f}px"
        )

        # Calculate center of detected grid
        center_x = np.nanmean(grid_centers[:, :, 0])
        center_y = np.nanmean(grid_centers[:, :, 1])

        logger.info(f"Grid center: ({center_x:.1f}, {center_y:.1f})")

        # Create ideal grid centered at the same point
        ideal_grid = np.zeros((rows, cols, 2), dtype=np.float32)

        # Start from center and work outward
        for r in range(rows):
            for c in range(cols):
                # Calculate position relative to center
                x_offset = (c - (cols - 1) / 2) * avg_h_spacing
                y_offset = (r - (rows - 1) / 2) * avg_v_spacing

                ideal_grid[r, c, 0] = center_x + x_offset
                ideal_grid[r, c, 1] = center_y + y_offset

        return ideal_grid

    def calibrate_distortion(
        self,
        image: NDArray[np.uint8],
        detected_grid: NDArray[np.float32],
        ideal_grid: NDArray[np.float32],
        rows: int,
        cols: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Calibrate camera distortion from grid positions.

        Args:
            image: Input image
            detected_grid: (rows, cols, 2) array of detected ball positions
            ideal_grid: (rows, cols, 2) array of ideal ball positions
            rows: Number of rows
            cols: Number of columns

        Returns:
            Tuple of (camera_matrix, distortion_coefficients, rms_error)
        """
        logger.info("Calibrating camera distortion...")

        h, w = image.shape[:2]

        # Prepare 3D object points (z=0, using ideal grid positions)
        object_points = []
        image_points = []

        # Real-world ball spacing (approximate - we'll use pixels as units)
        # This creates a planar calibration pattern
        for r in range(rows):
            for c in range(cols):
                if not np.isnan(detected_grid[r, c, 0]):
                    # 3D point: use ideal grid position with z=0
                    obj_pt = np.array(
                        [ideal_grid[r, c, 0], ideal_grid[r, c, 1], 0.0],
                        dtype=np.float32,
                    )
                    object_points.append(obj_pt)

                    # 2D image point: use detected position
                    img_pt = detected_grid[r, c]
                    image_points.append(img_pt)

        if len(object_points) < 4:
            raise ValueError(
                f"Need at least 4 valid points for calibration, got {len(object_points)}"
            )

        logger.info(f"Using {len(object_points)} ball positions for calibration")

        # Convert to proper format for calibrateCamera
        # It expects lists of arrays (for multiple images)
        obj_points_array = [np.array(object_points, dtype=np.float32)]
        img_points_array = [np.array(image_points, dtype=np.float32)]

        # Initial camera matrix estimate
        focal_length = w * 0.8  # Typical for webcams/security cameras
        cx, cy = w / 2, h / 2

        camera_matrix = np.array(
            [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64
        )

        # Initial distortion coefficients (all zeros)
        distortion_coeffs = np.zeros((5, 1), dtype=np.float64)

        # Calibration flags:
        # - Use intrinsic guess
        # - Fix principal point (assume it's at image center)
        # - Fix aspect ratio (assume square pixels)
        # - Fix focal length (only optimize distortion)
        # - Zero tangential distortion (assume only radial)
        calibration_flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_FOCAL_LENGTH
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3  # Only optimize k1, k2
        )

        try:
            rms_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
                obj_points_array,
                img_points_array,
                (w, h),
                camera_matrix,
                distortion_coeffs,
                flags=calibration_flags,
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100,
                    1e-6,
                ),
            )

            logger.info(f"Calibration completed with RMS error: {rms_error:.4f} pixels")
            logger.info(f"Camera matrix:\n{K}")
            logger.info(f"Distortion coefficients (k1,k2,p1,p2,k3): {D.ravel()}")

            # Clamp distortion coefficients if they're extreme
            k1, k2 = float(D[0]), float(D[1])
            if abs(k1) > 0.5 or abs(k2) > 0.5:
                logger.warning(
                    f"Extreme distortion values detected: k1={k1:.3f}, k2={k2:.3f}"
                )
                logger.warning("Clamping to reasonable range [-0.3, 0.3]")
                k1 = np.clip(k1, -0.3, 0.3)
                k2 = np.clip(k2, -0.3, 0.3)
                D[0], D[1] = k1, k2

            # Convert to fisheye-compatible format (4 coefficients)
            fisheye_dist = np.zeros((4, 1), dtype=np.float64)
            fisheye_dist[0] = D[0]  # k1
            fisheye_dist[1] = D[1]  # k2

            return K, fisheye_dist, rms_error

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise

    def validate_calibration(
        self,
        image: NDArray[np.uint8],
        camera_matrix: NDArray[np.float64],
        dist_coeffs: NDArray[np.float64],
        detected_grid: NDArray[np.float32],
        rows: int,
        cols: int,
    ) -> dict:
        """Validate calibration by measuring grid straightness improvement.

        Args:
            image: Original image
            camera_matrix: Calibrated camera matrix
            dist_coeffs: Calibrated distortion coefficients
            detected_grid: Original detected grid positions
            rows: Number of rows
            cols: Number of columns

        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating calibration...")

        # Undistort the image
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(
            image, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        self.debug_images["undistorted"] = undistorted

        # Re-detect balls in undistorted image
        balls_undistorted = self.detect_balls(undistorted)

        if len(balls_undistorted) < 4:
            logger.warning("Too few balls detected in undistorted image")
            return {"error": "insufficient_detections"}

        # Measure grid straightness before and after
        def measure_grid_straightness(grid_centers, rows, cols):
            """Measure how straight the grid lines are."""
            row_straightness = []
            col_straightness = []

            # Check row straightness (should be horizontal)
            for r in range(rows):
                row_pts = []
                for c in range(cols):
                    if not np.isnan(grid_centers[r, c, 0]):
                        row_pts.append(grid_centers[r, c])

                if len(row_pts) >= 3:
                    row_pts = np.array(row_pts)
                    # Fit a line and measure deviation
                    vx, vy, x0, y0 = cv2.fitLine(
                        row_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
                    )
                    # Calculate perpendicular distance from each point to the line
                    for pt in row_pts:
                        # Distance from point to line
                        d = abs((pt[1] - y0) * vx - (pt[0] - x0) * vy) / np.sqrt(
                            vx**2 + vy**2
                        )
                        row_straightness.append(float(d))

            # Check column straightness (should be vertical)
            for c in range(cols):
                col_pts = []
                for r in range(rows):
                    if not np.isnan(grid_centers[r, c, 0]):
                        col_pts.append(grid_centers[r, c])

                if len(col_pts) >= 3:
                    col_pts = np.array(col_pts)
                    vx, vy, x0, y0 = cv2.fitLine(
                        col_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
                    )
                    for pt in col_pts:
                        d = abs((pt[1] - y0) * vx - (pt[0] - x0) * vy) / np.sqrt(
                            vx**2 + vy**2
                        )
                        col_straightness.append(float(d))

            return {
                "mean_row_deviation": (
                    np.mean(row_straightness) if row_straightness else 0
                ),
                "mean_col_deviation": (
                    np.mean(col_straightness) if col_straightness else 0
                ),
                "max_row_deviation": (
                    np.max(row_straightness) if row_straightness else 0
                ),
                "max_col_deviation": (
                    np.max(col_straightness) if col_straightness else 0
                ),
            }

        original_straightness = measure_grid_straightness(detected_grid, rows, cols)

        # Try to infer grid from undistorted detections
        try:
            _, _, undistorted_grid = self.infer_grid_structure(balls_undistorted)
            undistorted_straightness = measure_grid_straightness(
                undistorted_grid, rows, cols
            )
        except Exception as e:
            logger.warning(f"Could not analyze undistorted grid: {e}")
            undistorted_straightness = {
                "mean_row_deviation": float("nan"),
                "mean_col_deviation": float("nan"),
                "max_row_deviation": float("nan"),
                "max_col_deviation": float("nan"),
            }

        improvement = {
            "original_mean_deviation": (
                original_straightness["mean_row_deviation"]
                + original_straightness["mean_col_deviation"]
            )
            / 2,
            "undistorted_mean_deviation": (
                undistorted_straightness["mean_row_deviation"]
                + undistorted_straightness["mean_col_deviation"]
            )
            / 2,
            "original_max_deviation": max(
                original_straightness["max_row_deviation"],
                original_straightness["max_col_deviation"],
            ),
            "undistorted_max_deviation": max(
                undistorted_straightness["max_row_deviation"],
                undistorted_straightness["max_col_deviation"],
            ),
        }

        logger.info("Grid straightness metrics:")
        logger.info(
            f"  Original mean deviation: {improvement['original_mean_deviation']:.2f}px"
        )
        logger.info(
            f"  Undistorted mean deviation: {improvement['undistorted_mean_deviation']:.2f}px"
        )
        logger.info(
            f"  Original max deviation: {improvement['original_max_deviation']:.2f}px"
        )
        logger.info(
            f"  Undistorted max deviation: {improvement['undistorted_max_deviation']:.2f}px"
        )

        return improvement

    def save_calibration(
        self,
        filepath: str,
        camera_matrix: NDArray[np.float64],
        dist_coeffs: NDArray[np.float64],
        resolution: tuple[int, int],
        calibration_error: float,
    ) -> bool:
        """Save calibration to OpenCV YAML format.

        Args:
            filepath: Path to save YAML file
            camera_matrix: Camera matrix
            dist_coeffs: Distortion coefficients
            resolution: Image resolution (width, height)
            calibration_error: RMS calibration error

        Returns:
            True if saved successfully
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", camera_matrix)
            fs.write("dist_coeffs", dist_coeffs)
            fs.write("image_width", resolution[0])
            fs.write("image_height", resolution[1])
            fs.write("calibration_error", float(calibration_error))
            fs.write("calibration_date", datetime.now().isoformat())
            fs.write("calibration_method", "ball_grid")
            fs.write(
                "notes",
                "Camera calibration computed from billiard ball grid pattern",
            )
            fs.release()

            logger.info(f"Calibration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def save_debug_images(self, output_dir: str) -> None:
        """Save debug visualization images.

        Args:
            output_dir: Directory to save debug images
        """
        debug_dir = Path(output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        for name, img in self.debug_images.items():
            output_path = debug_dir / f"{name}.jpg"
            cv2.imwrite(str(output_path), img)
            logger.info(f"Saved debug image: {output_path}")

    def visualize_grid(
        self,
        image: NDArray[np.uint8],
        detected_grid: NDArray[np.float32],
        ideal_grid: NDArray[np.float32],
        rows: int,
        cols: int,
    ) -> NDArray[np.uint8]:
        """Create visualization showing detected vs ideal grid.

        Args:
            image: Input image
            detected_grid: Detected ball positions
            ideal_grid: Ideal ball positions
            rows: Number of rows
            cols: Number of columns

        Returns:
            Visualization image
        """
        vis = image.copy()

        # Draw detected grid in green
        for r in range(rows):
            for c in range(cols):
                if not np.isnan(detected_grid[r, c, 0]):
                    pt = (int(detected_grid[r, c, 0]), int(detected_grid[r, c, 1]))
                    cv2.circle(vis, pt, 5, (0, 255, 0), -1)

                    # Draw lines to neighbors
                    if c < cols - 1 and not np.isnan(detected_grid[r, c + 1, 0]):
                        next_pt = (
                            int(detected_grid[r, c + 1, 0]),
                            int(detected_grid[r, c + 1, 1]),
                        )
                        cv2.line(vis, pt, next_pt, (0, 255, 0), 1)

                    if r < rows - 1 and not np.isnan(detected_grid[r + 1, c, 0]):
                        next_pt = (
                            int(detected_grid[r + 1, c, 0]),
                            int(detected_grid[r + 1, c, 1]),
                        )
                        cv2.line(vis, pt, next_pt, (0, 255, 0), 1)

        # Draw ideal grid in blue
        for r in range(rows):
            for c in range(cols):
                pt = (int(ideal_grid[r, c, 0]), int(ideal_grid[r, c, 1]))
                cv2.circle(vis, pt, 3, (255, 0, 0), -1)

                # Draw lines to neighbors
                if c < cols - 1:
                    next_pt = (
                        int(ideal_grid[r, c + 1, 0]),
                        int(ideal_grid[r, c + 1, 1]),
                    )
                    cv2.line(vis, pt, next_pt, (255, 0, 0), 1)

                if r < rows - 1:
                    next_pt = (
                        int(ideal_grid[r + 1, c, 0]),
                        int(ideal_grid[r + 1, c, 1]),
                    )
                    cv2.line(vis, pt, next_pt, (255, 0, 0), 1)

        return vis


def main():
    """Main calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate camera distortion using ball grid pattern"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="backend/vision/test_data/grid.jpg",
        help="Path to grid image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/calibration_data/camera/camera_params.yaml",
        help="Path to save calibration YAML",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="backend/calibration_data/camera/debug",
        help="Directory to save debug images",
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=20,
        help="Minimum ball radius in pixels",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=80,
        help="Maximum ball radius in pixels",
    )

    args = parser.parse_args()

    # Load image
    logger.info(f"Loading image: {args.image}")
    image = cv2.imread(args.image)

    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return 1

    h, w = image.shape[:2]
    logger.info(f"Image resolution: {w}x{h}")

    # Create calibrator
    calibrator = BallGridCalibrator(
        min_ball_radius=args.min_radius, max_ball_radius=args.max_radius
    )

    # Step 1: Detect balls
    balls = calibrator.detect_balls(image)

    if len(balls) < 4:
        logger.error(f"Too few balls detected ({len(balls)}). Need at least 4.")
        return 1

    # Step 2: Infer grid structure
    rows, cols, detected_grid = calibrator.infer_grid_structure(balls)

    # Step 3: Calculate ideal grid
    ideal_grid = calibrator.calculate_ideal_grid(detected_grid, rows, cols)

    # Create grid visualization
    grid_vis = calibrator.visualize_grid(image, detected_grid, ideal_grid, rows, cols)
    calibrator.debug_images["grid_analysis"] = grid_vis

    # Step 4: Calibrate distortion
    try:
        camera_matrix, dist_coeffs, rms_error = calibrator.calibrate_distortion(
            image, detected_grid, ideal_grid, rows, cols
        )
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        calibrator.save_debug_images(args.debug_dir)
        return 1

    # Step 5: Validate calibration
    validation = calibrator.validate_calibration(
        image, camera_matrix, dist_coeffs, detected_grid, rows, cols
    )

    # Step 6: Save calibration
    calibrator.save_calibration(
        args.output, camera_matrix, dist_coeffs, (w, h), rms_error
    )

    # Save debug images
    calibrator.save_debug_images(args.debug_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Grid structure: {rows} rows × {cols} columns")
    logger.info(f"Calibration points: {rows * cols}")
    logger.info(f"RMS error: {rms_error:.4f} pixels")
    logger.info(f"\nCamera matrix:\n{camera_matrix}")
    logger.info(f"\nDistortion coefficients: {dist_coeffs.ravel()}")
    logger.info(f"  k1 (radial): {dist_coeffs[0, 0]:.6f}")
    logger.info(f"  k2 (radial): {dist_coeffs[1, 0]:.6f}")

    if "error" not in validation:
        improvement_pct = (
            (
                validation["original_mean_deviation"]
                - validation["undistorted_mean_deviation"]
            )
            / validation["original_mean_deviation"]
            * 100
        )
        logger.info(f"\nGrid straightness improvement: {improvement_pct:.1f}%")
        logger.info(
            f"  Before: {validation['original_mean_deviation']:.2f}px mean deviation"
        )
        logger.info(
            f"  After:  {validation['undistorted_mean_deviation']:.2f}px mean deviation"
        )

    logger.info(f"\nCalibration saved to: {args.output}")
    logger.info(f"Debug images saved to: {args.debug_dir}/")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
