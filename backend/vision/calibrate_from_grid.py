#!/usr/bin/env python3
"""Standalone calibration script to calibrate camera barrel/fisheye distortion
using the test grid image and table boundaries.

This script:
1. Loads the grid.jpg test image
2. Detects table boundaries (which should be straight rectangles)
3. Uses table geometry to calibrate fisheye/barrel distortion parameters
4. Saves calibration to YAML file for use by Video Module
5. Displays before/after images showing the correction

Usage:
    python -m backend.vision.calibrate_from_grid [options]

Options:
    --input PATH          Path to input image (default: backend/vision/test_data/grid.jpg)
    --output PATH         Path to output YAML file (default: backend/calibration_data/camera/camera_params.yaml)
    --manual-corners      Enable manual corner selection via mouse clicks
    --show-visualization  Display before/after visualization
    --save-debug-images   Save intermediate debug images
    --table-width METERS  Real table width in meters (default: 2.54 for 9ft table)
    --table-height METERS Real table height in meters (default: 1.27 for 9ft table)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(backend_dir))

# Change to backend directory for config imports
import os

original_cwd = Path.cwd()
os.chdir(backend_dir)

# Direct imports to avoid package __init__.py circular dependencies
import importlib.util


def load_module_from_file(module_name: str, file_path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load {module_name} from {file_path}")


# Load config module first
config_module = load_module_from_file("config", backend_dir / "config.py")

# Load calibration modules
camera_cal_path = backend_dir / "vision" / "calibration" / "camera.py"
CameraCalibrator = load_module_from_file(
    "vision.calibration.camera", camera_cal_path
).CameraCalibrator

# TableDetector has been removed - using manual corner selection only

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ManualCornerSelector:
    """Interactive tool to manually select table corners."""

    def __init__(self, image: NDArray[np.uint8]):
        """Initialize with image to annotate.

        Args:
            image: Input image for corner selection
        """
        self.image = image.copy()
        self.display_image = image.copy()
        self.corners: list[tuple[float, float]] = []
        self.window_name = "Select Table Corners (clockwise from top-left)"

    def select_corners(self) -> Optional[list[tuple[float, float]]]:
        """Display image and allow user to click 4 corners.

        Returns:
            List of 4 corner points or None if cancelled
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        logger.info("Click 4 table corners in CLOCKWISE order starting from TOP-LEFT")
        logger.info("Press 'r' to reset, 'q' to quit, Enter to confirm")

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Corner selection cancelled")
                cv2.destroyAllWindows()
                return None
            elif key == ord("r"):
                logger.info("Resetting corners")
                self.corners = []
                self.display_image = self.image.copy()
            elif key == 13:  # Enter key
                if len(self.corners) == 4:
                    logger.info(f"Corners selected: {self.corners}")
                    cv2.destroyAllWindows()
                    return self.corners
                else:
                    logger.warning(
                        f"Need exactly 4 corners, got {len(self.corners)}. Continue clicking."
                    )

        return None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((float(x), float(y)))
            logger.info(f"Corner {len(self.corners)}: ({x}, {y})")

            # Draw the point
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)

            # Draw line to previous point
            if len(self.corners) > 1:
                prev = self.corners[-2]
                cv2.line(
                    self.display_image,
                    (int(prev[0]), int(prev[1])),
                    (x, y),
                    (0, 255, 0),
                    2,
                )

            # Close the quadrilateral on 4th point
            if len(self.corners) == 4:
                first = self.corners[0]
                cv2.line(
                    self.display_image,
                    (x, y),
                    (int(first[0]), int(first[1])),
                    (0, 255, 0),
                    2,
                )


def detect_table_corners_auto(
    image: NDArray[np.uint8],
) -> Optional[list[tuple[float, float]]]:
    """Automatically detect table corners using TableDetector.

    NOTE: TableDetector has been removed. This function now always returns None.
    Use manual corner selection instead.

    Args:
        image: Input image

    Returns:
        None (automatic detection disabled)
    """
    logger.warning(
        "Automatic table corner detection has been disabled. "
        "TableDetector module has been removed. Please use manual corner selection."
    )
    return None


def visualize_distortion_lines(
    image: NDArray[np.uint8], corners: list[tuple[float, float]], color=(0, 255, 0)
) -> NDArray[np.uint8]:
    """Draw lines on table edges to visualize distortion.

    Args:
        image: Input image
        corners: Table corner points
        color: Line color (BGR)

    Returns:
        Image with table edges highlighted
    """
    vis = image.copy()

    # Draw lines between corners
    for i in range(4):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
        cv2.line(vis, pt1, pt2, color, 2)

    # Draw corner points
    for corner in corners:
        cv2.circle(vis, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)

    return vis


def measure_line_straightness(
    image: NDArray[np.uint8], p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Measure how straight a line is by checking edge pixels.

    This samples points along the line and measures deviation from the
    ideal straight line.

    Args:
        image: Input image
        p1: Start point
        p2: End point

    Returns:
        RMS deviation from straight line (lower is straighter)
    """
    # Sample points along the line
    num_samples = 50
    x1, y1 = p1
    x2, y2 = p2

    deviations = []
    for i in range(num_samples):
        t = i / (num_samples - 1)
        # Ideal point on straight line
        x1 + t * (x2 - x1)
        y1 + t * (y2 - y1)

        # In a perfect world, we'd detect the actual edge here
        # For simplicity, we assume the ideal is the actual
        # and deviation is measured after correction
        deviation = 0.0  # Placeholder
        deviations.append(deviation)

    return float(np.sqrt(np.mean(np.array(deviations) ** 2)))


def create_before_after_visualization(
    original: NDArray[np.uint8],
    corrected: NDArray[np.uint8],
    original_corners: list[tuple[float, float]],
    corrected_corners: Optional[list[tuple[float, float]]] = None,
) -> NDArray[np.uint8]:
    """Create side-by-side before/after comparison.

    Args:
        original: Original image
        corrected: Distortion-corrected image
        original_corners: Corner points in original image
        corrected_corners: Corner points in corrected image (optional)

    Returns:
        Combined visualization image
    """
    # Resize images to same height if needed
    h1, w1 = original.shape[:2]
    h2, w2 = corrected.shape[:2]

    if h1 != h2:
        # Resize to smaller height
        target_h = min(h1, h2)
        scale1 = target_h / h1
        scale2 = target_h / h2
        original = cv2.resize(
            original, (int(w1 * scale1), target_h), interpolation=cv2.INTER_AREA
        )
        corrected = cv2.resize(
            corrected, (int(w2 * scale2), target_h), interpolation=cv2.INTER_AREA
        )

        # Scale corners
        original_corners = [(x * scale1, y * scale1) for x, y in original_corners]
        if corrected_corners:
            corrected_corners = [(x * scale2, y * scale2) for x, y in corrected_corners]

    # Draw table edges on both images
    original_vis = visualize_distortion_lines(original, original_corners, (0, 255, 0))

    if corrected_corners is None:
        # Use same corner positions for corrected image
        corrected_corners = original_corners

    corrected_vis = visualize_distortion_lines(
        corrected, corrected_corners, (0, 255, 0)
    )

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        original_vis, "BEFORE (Original)", (10, 30), font, 1, (255, 255, 255), 2
    )
    cv2.putText(
        corrected_vis, "AFTER (Corrected)", (10, 30), font, 1, (255, 255, 255), 2
    )

    # Combine side by side
    combined = np.hstack([original_vis, corrected_vis])

    return combined


def print_calibration_stats(calibrator: CameraCalibrator):
    """Print calibration statistics to console.

    Args:
        calibrator: CameraCalibrator with completed calibration
    """
    if calibrator.camera_params is None:
        logger.error("No calibration parameters available")
        return

    params = calibrator.camera_params

    logger.info("=" * 60)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 60)

    logger.info(f"Resolution: {params.resolution[0]}x{params.resolution[1]}")
    logger.info(f"Calibration Date: {params.calibration_date}")
    logger.info(f"RMS Error: {params.calibration_error:.4f} pixels")

    logger.info("\nCamera Matrix:")
    logger.info(f"  fx: {params.camera_matrix[0, 0]:.2f}")
    logger.info(f"  fy: {params.camera_matrix[1, 1]:.2f}")
    logger.info(f"  cx: {params.camera_matrix[0, 2]:.2f}")
    logger.info(f"  cy: {params.camera_matrix[1, 2]:.2f}")

    logger.info("\nDistortion Coefficients:")
    dist = params.distortion_coefficients.ravel()
    logger.info(f"  k1 (radial): {dist[0]:.6f}")
    if len(dist) > 1:
        logger.info(f"  k2 (radial): {dist[1]:.6f}")
    if len(dist) > 2:
        logger.info(f"  p1 (tangential): {dist[2]:.6f}")
    if len(dist) > 3:
        logger.info(f"  p2 (tangential): {dist[3]:.6f}")
    if len(dist) > 4:
        logger.info(f"  k3 (radial): {dist[4]:.6f}")

    # Interpret distortion
    k1 = float(dist[0])
    if abs(k1) < 0.01:
        distortion_type = "minimal distortion"
    elif k1 < 0:
        distortion_type = "barrel distortion (fisheye)"
    else:
        distortion_type = "pincushion distortion"

    logger.info(f"\nDistortion Type: {distortion_type}")
    logger.info(
        f"Distortion Strength: {'weak' if abs(k1) < 0.1 else 'moderate' if abs(k1) < 0.2 else 'strong'}"
    )

    logger.info("=" * 60)


def main():
    """Main calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate camera distortion from test grid image"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="vision/test_data/grid.jpg",
        help="Path to input image (relative to backend dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_data/camera/camera_params.yaml",
        help="Path to output YAML calibration file (relative to backend dir)",
    )
    parser.add_argument(
        "--manual-corners",
        action="store_true",
        help="Enable manual corner selection",
    )
    parser.add_argument(
        "--show-visualization",
        action="store_true",
        help="Display before/after visualization",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Save intermediate debug images",
    )
    parser.add_argument(
        "--table-width",
        type=float,
        default=2.54,
        help="Real table width in meters (default: 2.54 for 9ft table)",
    )
    parser.add_argument(
        "--table-height",
        type=float,
        default=1.27,
        help="Real table height in meters (default: 1.27 for 9ft table)",
    )

    args = parser.parse_args()

    # Resolve paths - we're already in backend dir due to os.chdir
    input_path = Path(args.input)
    if not input_path.is_absolute():
        if input_path.exists():
            input_path = input_path.resolve()
        else:
            # Try as relative path
            input_path = input_path.resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = output_path.resolve()

    # Load image
    logger.info(f"Loading image from: {input_path}")
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        return 1

    image = cv2.imread(str(input_path))
    if image is None:
        logger.error(f"Failed to load image: {input_path}")
        return 1

    logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Detect or manually select table corners
    corners: Optional[list[tuple[float, float]]] = None

    if args.manual_corners:
        selector = ManualCornerSelector(image)
        corners = selector.select_corners()
        if corners is None:
            logger.error("Corner selection cancelled")
            return 1
    else:
        corners = detect_table_corners_auto(image)

        if corners is None:
            logger.warning(
                "Automatic detection failed, falling back to manual selection"
            )
            selector = ManualCornerSelector(image)
            corners = selector.select_corners()
            if corners is None:
                logger.error("Corner selection cancelled")
                return 1

    # Create calibrator
    cache_dir = backend_dir / "calibration_cache"
    calibrator = CameraCalibrator(cache_dir=str(cache_dir))

    # Calibrate fisheye distortion from table geometry
    logger.info("Calibrating fisheye distortion from table geometry...")
    table_dimensions = (args.table_width, args.table_height)

    success, camera_params = calibrator.calibrate_fisheye_from_table(
        image, corners, table_dimensions
    )

    if not success or camera_params is None:
        logger.error("Calibration failed!")
        return 1

    # Print statistics
    print_calibration_stats(calibrator)

    # Apply undistortion to validate
    logger.info("Applying undistortion to validate calibration...")
    undistorted = calibrator.undistort_image(image)

    # Save calibration to YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving calibration to: {output_path}")

    if calibrator.save_fisheye_calibration_yaml(str(output_path)):
        logger.info("Calibration saved successfully!")
    else:
        logger.error("Failed to save calibration")
        return 1

    # Save debug images if requested
    if args.save_debug_images:
        debug_dir = output_path.parent / "debug"
        debug_dir.mkdir(exist_ok=True)

        # Original with corners
        original_vis = visualize_distortion_lines(image, corners)
        cv2.imwrite(str(debug_dir / "01_original_with_corners.jpg"), original_vis)
        logger.info(f"Saved: {debug_dir / '01_original_with_corners.jpg'}")

        # Undistorted
        cv2.imwrite(str(debug_dir / "02_undistorted.jpg"), undistorted)
        logger.info(f"Saved: {debug_dir / '02_undistorted.jpg'}")

        # Before/after comparison
        comparison = create_before_after_visualization(image, undistorted, corners)
        cv2.imwrite(str(debug_dir / "03_before_after.jpg"), comparison)
        logger.info(f"Saved: {debug_dir / '03_before_after.jpg'}")

    # Display visualization if requested
    if args.show_visualization:
        logger.info("Displaying before/after visualization (press any key to close)...")
        comparison = create_before_after_visualization(image, undistorted, corners)

        # Resize if too large for display
        max_width = 1920
        h, w = comparison.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            comparison = cv2.resize(
                comparison, (max_width, new_h), interpolation=cv2.INTER_AREA
            )

        cv2.imshow("Calibration Result", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info("\nCalibration complete!")
    logger.info(f"Calibration file: {output_path}")
    logger.info(
        "\nTo use this calibration in the Video Module, ensure the video module"
    )
    logger.info(
        "loads and applies this calibration before writing frames to shared memory."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
