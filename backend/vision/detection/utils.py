"""Common detection utilities."""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Line:
    """Represents a detected line."""

    start_point: tuple[float, float]
    end_point: tuple[float, float]
    angle: float  # In degrees
    length: float


@dataclass
class Circle:
    """Represents a detected circle."""

    center: tuple[float, float]
    radius: float
    confidence: float


class DetectionUtils:
    """Common utilities for vision detection."""

    @staticmethod
    def apply_color_threshold(
        frame: np.ndarray, lower: tuple, upper: tuple, color_space: str = "HSV"
    ) -> np.ndarray:
        """Apply color thresholding to frame."""
        if frame is None or frame.size == 0:
            return np.array([])

        # Convert color space if needed
        if color_space.upper() == "HSV":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color_space.upper() == "LAB":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif color_space.upper() == "RGB":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            converted = frame

        # Apply thresholding
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(converted, lower_np, upper_np)
        return mask

    @staticmethod
    def find_circles(
        frame: np.ndarray,
        min_radius: int,
        max_radius: int,
        param1: float = 50,
        param2: float = 30,
        min_dist: Optional[int] = None,
    ) -> list[Circle]:
        """Find circles using Hough circle detection."""
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Set minimum distance between circle centers
        if min_dist is None:
            min_dist = min_radius * 2

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                # Calculate confidence based on circle quality
                confidence = DetectionUtils._calculate_circle_confidence(
                    gray, (x, y), r
                )
                detected_circles.append(
                    Circle(
                        center=(float(x), float(y)),
                        radius=float(r),
                        confidence=confidence,
                    )
                )

        # Sort by confidence
        detected_circles.sort(key=lambda c: c.confidence, reverse=True)
        return detected_circles

    @staticmethod
    def find_lines(
        frame: np.ndarray,
        threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
        rho: float = 1,
        theta: float = np.pi / 180,
    ) -> list[Line]:
        """Find lines using Hough line detection."""
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using HoughLinesP (probabilistic)
        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line properties
                length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                detected_lines.append(
                    Line(
                        start_point=(float(x1), float(y1)),
                        end_point=(float(x2), float(y2)),
                        angle=angle,
                        length=length,
                    )
                )

        # Sort by length (longer lines first)
        detected_lines.sort(key=lambda l: l.length, reverse=True)
        return detected_lines

    @staticmethod
    def calculate_contour_center(contour: np.ndarray) -> tuple[float, float]:
        """Calculate center point of contour using moments."""
        if contour is None or len(contour) == 0:
            return (0.0, 0.0)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            return (float(x + w // 2), float(y + h // 2))

        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        return (float(center_x), float(center_y))

    @staticmethod
    def filter_contours_by_area(
        contours: list[np.ndarray], min_area: float = 0, max_area: float = float("inf")
    ) -> list[np.ndarray]:
        """Filter contours by area."""
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered.append(contour)
        return filtered

    @staticmethod
    def filter_contours_by_aspect_ratio(
        contours: list[np.ndarray], target_ratio: float, tolerance: float = 0.3
    ) -> list[np.ndarray]:
        """Filter contours by aspect ratio."""
        filtered = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect_ratio = w / h
            if abs(aspect_ratio - target_ratio) <= tolerance:
                filtered.append(contour)
        return filtered

    @staticmethod
    def filter_contours_by_circularity(
        contours: list[np.ndarray], min_circularity: float = 0.7
    ) -> list[np.ndarray]:
        """Filter contours by circularity (how circle-like they are)."""
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity >= min_circularity:
                filtered.append(contour)
        return filtered

    @staticmethod
    def apply_morphological_operations(
        mask: np.ndarray,
        operation: str = "close",
        kernel_size: int = 5,
        iterations: int = 1,
    ) -> np.ndarray:
        """Apply morphological operations to clean up binary masks."""
        if mask is None or mask.size == 0:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        if operation.lower() == "close":
            result = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
        elif operation.lower() == "open":
            result = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=iterations
            )
        elif operation.lower() == "erode":
            result = cv2.erode(mask, kernel, iterations=iterations)
        elif operation.lower() == "dilate":
            result = cv2.dilate(mask, kernel, iterations=iterations)
        elif operation.lower() == "gradient":
            result = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        else:
            result = mask

        return result

    @staticmethod
    def calculate_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def calculate_angle_between_points(
        p1: tuple[float, float], p2: tuple[float, float]
    ) -> float:
        """Calculate angle in degrees between two points."""
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    @staticmethod
    def point_in_polygon(
        point: tuple[float, float], polygon: list[tuple[float, float]]
    ) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def find_quadrilateral_corners(
        contour: np.ndarray, epsilon_factor: float = 0.02
    ) -> Optional[np.ndarray]:
        """Approximate contour to quadrilateral and return corners."""
        if contour is None or len(contour) < 4:
            return None

        # Try different epsilon values to get exactly 4 points
        for factor in [epsilon_factor, 0.01, 0.03, 0.04, 0.05, 0.1]:
            epsilon = factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                return approx.reshape(-1, 2).astype(np.float32)

        return None

    @staticmethod
    def enhance_contrast(
        image: np.ndarray, alpha: float = 1.5, beta: int = 0
    ) -> np.ndarray:
        """Enhance contrast using linear transformation."""
        if image is None or image.size == 0:
            return image
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def apply_gaussian_blur(
        image: np.ndarray, kernel_size: int = 5, sigma: float = 0
    ) -> np.ndarray:
        """Apply Gaussian blur for noise reduction."""
        if image is None or image.size == 0:
            return image
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def create_roi_mask(
        image_shape: tuple[int, int], roi_points: list[tuple[float, float]]
    ) -> np.ndarray:
        """Create a region of interest mask."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if len(roi_points) >= 3:
            roi_array = np.array(roi_points, dtype=np.int32)
            cv2.fillPoly(mask, [roi_array], 255)
        return mask

    @staticmethod
    def _calculate_circle_confidence(
        gray: np.ndarray, center: tuple[int, int], radius: int
    ) -> float:
        """Calculate confidence score for detected circle based on edge strength."""
        try:
            # Create circular mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, 2)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Calculate ratio of edge pixels on the circle
            circle_pixels = np.sum(mask > 0)
            edge_pixels = np.sum(cv2.bitwise_and(edges, mask) > 0)

            if circle_pixels == 0:
                return 0.0

            confidence = edge_pixels / circle_pixels
            return min(1.0, confidence * 2.0)  # Scale to 0-1 range

        except Exception:
            return 0.5  # Default confidence if calculation fails
