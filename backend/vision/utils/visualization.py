"""Debugging visualization utilities for table detection.

Provides comprehensive visualization functions for debugging and analyzing
table detection algorithms including corner detection, pocket identification,
and perspective correction.
"""

from typing import Optional

import cv2
import numpy as np

from ..detection.table import Pocket, PocketType, TableCorners, TableDetectionResult


class TableVisualization:
    """Visualization tools for table detection debugging."""

    @staticmethod
    def draw_table_detection(
        image: np.ndarray,
        detection: TableDetectionResult,
        show_corners: bool = True,
        show_pockets: bool = True,
        show_perspective: bool = True,
        show_info: bool = True,
    ) -> np.ndarray:
        """Draw complete table detection visualization on image.

        Args:
            image: Input image to draw on
            detection: Table detection result
            show_corners: Whether to draw corner markers
            show_pockets: Whether to draw pocket markers
            show_perspective: Whether to draw perspective grid
            show_info: Whether to draw information text

        Returns:
            Image with visualization overlays
        """
        if image is None or detection is None:
            return image

        vis_image = image.copy()

        # Draw table boundaries
        TableVisualization.draw_table_boundaries(vis_image, detection.corners)

        # Draw corners
        if show_corners:
            TableVisualization.draw_corners(vis_image, detection.corners)

        # Draw pockets
        if show_pockets:
            TableVisualization.draw_pockets(vis_image, detection.pockets)

        # Draw perspective grid
        if show_perspective and detection.perspective_transform is not None:
            TableVisualization.draw_perspective_grid(vis_image, detection)

        # Draw information text
        if show_info:
            TableVisualization.draw_detection_info(vis_image, detection)

        return vis_image

    @staticmethod
    def draw_table_boundaries(
        image: np.ndarray,
        corners: TableCorners,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw table boundary lines."""
        if image is None or corners is None:
            return image

        corner_points = [(int(x), int(y)) for x, y in corners.to_list()]

        # Draw boundary lines
        for i in range(4):
            pt1 = corner_points[i]
            pt2 = corner_points[(i + 1) % 4]
            cv2.line(image, pt1, pt2, color, thickness)

        return image

    @staticmethod
    def draw_corners(
        image: np.ndarray,
        corners: TableCorners,
        color: tuple[int, int, int] = (255, 0, 0),
        radius: int = 8,
        thickness: int = -1,
    ) -> np.ndarray:
        """Draw corner markers with labels."""
        if image is None or corners is None:
            return image

        corner_points = corners.to_list()
        labels = [
            "TL",
            "TR",
            "BL",
            "BR",
        ]  # Top-Left, Top-Right, Bottom-Left, Bottom-Right

        for _i, ((x, y), label) in enumerate(zip(corner_points, labels)):
            center = (int(x), int(y))

            # Draw corner circle
            cv2.circle(image, center, radius, color, thickness)

            # Draw label
            label_pos = (center[0] + 15, center[1] - 15)
            cv2.putText(
                image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Draw precise coordinates
            coord_text = f"({x:.1f},{y:.1f})"
            coord_pos = (center[0] + 15, center[1] + 5)
            cv2.putText(
                image, coord_text, coord_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        return image

    @staticmethod
    def draw_pockets(
        image: np.ndarray,
        pockets: list[Pocket],
        corner_color: tuple[int, int, int] = (255, 255, 0),
        side_color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw pocket markers with type indicators."""
        if image is None or not pockets:
            return image

        for i, pocket in enumerate(pockets):
            center = (int(pocket.position[0]), int(pocket.position[1]))
            radius = int(pocket.size)

            # Choose color based on pocket type
            color = (
                corner_color if pocket.pocket_type == PocketType.CORNER else side_color
            )

            # Draw pocket circle
            cv2.circle(image, center, radius, color, thickness)

            # Draw pocket type label
            pocket_label = f"P{i+1}-{pocket.pocket_type.value[0].upper()}"
            label_pos = (center[0] - 20, center[1] - radius - 10)
            cv2.putText(
                image, pocket_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

            # Draw confidence
            conf_text = f"{pocket.confidence:.2f}"
            conf_pos = (center[0] - 15, center[1] + radius + 15)
            cv2.putText(
                image, conf_text, conf_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        return image

    @staticmethod
    def draw_perspective_grid(
        image: np.ndarray,
        detection: TableDetectionResult,
        grid_size: int = 50,
        color: tuple[int, int, int] = (128, 128, 128),
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw perspective correction grid."""
        if (
            image is None
            or detection is None
            or detection.perspective_transform is None
        ):
            return image

        transform = detection.perspective_transform
        width, height = int(detection.width), int(detection.height)

        # Create grid points in rectified space
        grid_points = []
        for y in range(0, height, grid_size):
            for x in range(0, width, grid_size):
                grid_points.append([x, y, 1])

        # Transform grid points back to image space
        grid_points = np.array(grid_points).T
        inv_transform = cv2.invert(transform)[1]
        transformed_points = inv_transform @ grid_points

        # Normalize homogeneous coordinates
        transformed_points = transformed_points[:2] / transformed_points[2]

        # Draw grid lines
        grid_coords = transformed_points.T.reshape(-1, 2)

        # Draw horizontal lines
        for y in range(0, height, grid_size):
            line_points = []
            for x in range(0, width, grid_size):
                idx = (y // grid_size) * (width // grid_size) + (x // grid_size)
                if idx < len(grid_coords):
                    line_points.append(grid_coords[idx])

            for i in range(len(line_points) - 1):
                pt1 = (int(line_points[i][0]), int(line_points[i][1]))
                pt2 = (int(line_points[i + 1][0]), int(line_points[i + 1][1]))
                cv2.line(image, pt1, pt2, color, thickness)

        # Draw vertical lines
        for x in range(0, width, grid_size):
            line_points = []
            for y in range(0, height, grid_size):
                idx = (y // grid_size) * (width // grid_size) + (x // grid_size)
                if idx < len(grid_coords):
                    line_points.append(grid_coords[idx])

            for i in range(len(line_points) - 1):
                pt1 = (int(line_points[i][0]), int(line_points[i][1]))
                pt2 = (int(line_points[i + 1][0]), int(line_points[i + 1][1]))
                cv2.line(image, pt1, pt2, color, thickness)

        return image

    @staticmethod
    def draw_detection_info(
        image: np.ndarray,
        detection: TableDetectionResult,
        position: tuple[int, int] = (10, 30),
        font_scale: float = 0.6,
        color: tuple[int, int, int] = (255, 255, 255),
        background_color: Optional[tuple[int, int, int]] = (0, 0, 0),
    ) -> np.ndarray:
        """Draw detection information text overlay."""
        if image is None or detection is None:
            return image

        lines = [
            f"Table Detection Confidence: {detection.confidence:.3f}",
            f"Dimensions: {detection.width:.1f} x {detection.height:.1f}",
            f"Pockets Detected: {len(detection.pockets)}",
            f"Surface Color (HSV): {detection.surface_color}",
            f"Aspect Ratio: {detection.width/detection.height:.2f}",
        ]

        # Add corner coordinates
        corner_list = detection.corners.to_list()
        for i, (x, y) in enumerate(corner_list):
            lines.append(f"Corner {i+1}: ({x:.1f}, {y:.1f})")

        # Draw background rectangle if specified
        if background_color is not None:
            text_height = len(lines) * 25
            cv2.rectangle(
                image,
                (position[0] - 5, position[1] - 20),
                (position[0] + 350, position[1] + text_height),
                background_color,
                -1,
            )

        # Draw text lines
        y_offset = position[1]
        for line in lines:
            cv2.putText(
                image,
                line,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
            )
            y_offset += 25

        return image

    @staticmethod
    def create_detection_comparison(
        original: np.ndarray,
        detection: TableDetectionResult,
        debug_images: list[tuple[str, np.ndarray]],
    ) -> np.ndarray:
        """Create side-by-side comparison of detection stages."""
        if original is None:
            return np.array([])

        # Create visualization of final detection
        final_vis = TableVisualization.draw_table_detection(original, detection)

        # Prepare debug images
        debug_vis = []
        for name, debug_img in debug_images[:3]:  # Limit to 3 debug images
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

            # Add title to debug image
            titled_img = debug_img.copy()
            cv2.putText(
                titled_img,
                name,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            debug_vis.append(titled_img)

        # Resize all images to same height
        target_height = 300
        images_to_concat = [final_vis]

        # Resize original and final visualization
        aspect_ratio = final_vis.shape[1] / final_vis.shape[0]
        target_width = int(target_height * aspect_ratio)
        final_vis_resized = cv2.resize(final_vis, (target_width, target_height))
        images_to_concat = [final_vis_resized]

        # Resize debug images
        for debug_img in debug_vis:
            aspect_ratio = debug_img.shape[1] / debug_img.shape[0]
            target_width = int(target_height * aspect_ratio)
            debug_resized = cv2.resize(debug_img, (target_width, target_height))
            images_to_concat.append(debug_resized)

        # Concatenate horizontally
        if len(images_to_concat) > 1:
            comparison = np.hstack(images_to_concat)
        else:
            comparison = images_to_concat[0]

        return comparison

    @staticmethod
    def visualize_color_segmentation(
        image: np.ndarray, mask: np.ndarray, title: str = "Color Segmentation"
    ) -> np.ndarray:
        """Visualize color segmentation results."""
        if image is None or mask is None:
            return np.array([])

        # Create colored overlay
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        # Blend with original image
        alpha = 0.6
        blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        # Add title
        cv2.putText(
            blended, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        return blended

    @staticmethod
    def draw_contours_analysis(
        image: np.ndarray, contours: list[np.ndarray], title: str = "Contour Analysis"
    ) -> np.ndarray:
        """Visualize contour detection analysis."""
        if image is None or not contours:
            return image

        vis_image = image.copy()

        # Draw all contours with different colors
        for _i, contour in enumerate(contours):
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            cv2.drawContours(vis_image, [contour], -1, color, 2)

            # Draw contour info
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 1)

            # Draw area text
            text_pos = (x, y - 10)
            cv2.putText(
                vis_image,
                f"Area: {area:.0f}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Add title
        cv2.putText(
            vis_image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return vis_image

    @staticmethod
    def create_debug_dashboard(
        original: np.ndarray,
        detection: TableDetectionResult,
        debug_images: list[tuple[str, np.ndarray]],
    ) -> np.ndarray:
        """Create comprehensive debug dashboard."""
        if original is None:
            return np.array([])

        # Create main visualization
        main_vis = TableVisualization.draw_table_detection(original, detection)

        # Prepare grid layout (2x3)
        dashboard_images = []

        # Main detection (top-left, larger)
        main_resized = cv2.resize(main_vis, (400, 300))
        dashboard_images.append(main_resized)

        # Debug images (smaller panels)
        for name, debug_img in debug_images[:5]:  # Limit to 5 debug images
            if len(debug_img.shape) == 2:
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

            debug_resized = cv2.resize(debug_img, (200, 150))

            # Add title
            cv2.putText(
                debug_resized,
                name,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            dashboard_images.append(debug_resized)

        # Fill remaining slots with black images if needed
        while len(dashboard_images) < 6:
            black_img = np.zeros((150, 200, 3), dtype=np.uint8)
            dashboard_images.append(black_img)

        # Arrange in 2x3 grid
        top_row = np.hstack(
            [dashboard_images[0], np.vstack([dashboard_images[1], dashboard_images[2]])]
        )
        bottom_row = np.hstack(
            [dashboard_images[3], dashboard_images[4], dashboard_images[5]]
        )

        # Ensure same width
        if top_row.shape[1] != bottom_row.shape[1]:
            width = min(top_row.shape[1], bottom_row.shape[1])
            top_row = cv2.resize(top_row, (width, top_row.shape[0]))
            bottom_row = cv2.resize(bottom_row, (width, bottom_row.shape[0]))

        dashboard = np.vstack([top_row, bottom_row])

        # Add overall title
        title_height = 40
        title_panel = np.zeros((title_height, dashboard.shape[1], 3), dtype=np.uint8)
        title_text = (
            f"Table Detection Debug Dashboard - Confidence: {detection.confidence:.3f}"
        )
        cv2.putText(
            title_panel,
            title_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        final_dashboard = np.vstack([title_panel, dashboard])

        return final_dashboard

    @staticmethod
    def save_debug_images(
        debug_images: list[tuple[str, np.ndarray]],
        output_dir: str,
        frame_number: int = 0,
    ) -> list[str]:
        """Save debug images to disk for analysis."""
        import os

        saved_paths = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for name, image in debug_images:
            filename = f"frame_{frame_number:06d}_{name}.png"
            filepath = os.path.join(output_dir, filename)

            try:
                cv2.imwrite(filepath, image)
                saved_paths.append(filepath)
            except Exception as e:
                print(f"Failed to save debug image {filename}: {e}")

        return saved_paths
