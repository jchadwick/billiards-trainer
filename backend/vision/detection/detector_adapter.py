"""YOLO detection adapter for converting YOLO outputs to Vision module dataclasses.

This adapter converts YOLOv8 detection results (bounding boxes, class IDs, confidence scores)
into the Vision module's Ball and CueStick dataclasses. It handles:
- Coordinate conversion from YOLO format (normalized or pixel bbox) to center+radius
- Class ID mapping to BallType enum and ball numbers
- Confidence score extraction and filtering
- CueStick parameter extraction from bounding box orientation

YOLO Format:
- Bounding boxes: [x, y, w, h] where (x,y) is top-left corner
- Can be normalized [0-1] or pixel coordinates
- Class IDs: Map to ball types (e.g., "cue", "solid_1", "stripe_9", "eight")
- Confidence scores: [0-1] detection confidence

Implements FR-VIS-020 through FR-VIS-024, FR-VIS-030, FR-VIS-031
"""

import logging
import math
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..models import Ball, BallType, CueState, CueStick

logger = logging.getLogger(__name__)


# =============================================================================
# Class ID to Ball Type Mapping
# =============================================================================

# Standard YOLO class naming conventions for pool balls
# Expected class names: "cue", "eight", "solid_1", "solid_2", ..., "solid_7",
#                       "stripe_9", "stripe_10", ..., "stripe_15"
# Or simplified: "ball_0" (cue), "ball_8" (eight), "ball_1" through "ball_15"


def parse_ball_class_name(class_name: str) -> tuple[BallType, Optional[int]]:
    """Parse YOLO class name to extract ball type and number.

    Supports multiple naming conventions (simplified - no solid/stripe distinction):
    - "ball" -> (BallType.UNKNOWN, None) - generic ball, needs further classification
    - "cue" or "ball_0" -> (BallType.CUE, None)
    - "eight" or "ball_8" -> (BallType.EIGHT, 8)
    - "solid_1", "stripe_9", "ball_1", "ball_9" -> (BallType.OTHER, None) - all numbered balls

    Args:
        class_name: YOLO class name string

    Returns:
        Tuple of (BallType, ball_number) - ball_number is always None for OTHER type
    """
    class_name_lower = class_name.lower().strip()

    # Handle generic "ball" - needs further classification
    if class_name_lower == "ball":
        return BallType.UNKNOWN, None

    # Handle "cue" or "cue_ball"
    if "cue" in class_name_lower:
        return BallType.CUE, None

    # Handle "eight" or "8ball" or "ball_8"
    if "eight" in class_name_lower or class_name_lower in ["8ball", "ball_8"]:
        return BallType.EIGHT, 8

    # Handle "solid_N" format - now maps to OTHER
    if class_name_lower.startswith("solid_"):
        try:
            number = int(class_name_lower.split("_")[1])
            if 1 <= number <= 7:
                return BallType.OTHER, None  # No longer track specific numbers
        except (ValueError, IndexError):
            pass

    # Handle "stripe_N" format - now maps to OTHER
    if class_name_lower.startswith("stripe_"):
        try:
            number = int(class_name_lower.split("_")[1])
            if 9 <= number <= 15:
                return BallType.OTHER, None  # No longer track specific numbers
        except (ValueError, IndexError):
            pass

    # Handle "ball_N" format (unified naming)
    if class_name_lower.startswith("ball_"):
        try:
            number = int(class_name_lower.split("_")[1])
            if number == 0:
                return BallType.CUE, None
            elif number == 8:
                return BallType.EIGHT, 8
            elif 1 <= number <= 15:  # All other numbered balls -> OTHER
                return BallType.OTHER, None
        except (ValueError, IndexError):
            pass

    # Fallback: try to extract any number from the class name
    import re

    numbers = re.findall(r"\d+", class_name_lower)
    if numbers:
        try:
            number = int(numbers[0])
            if number == 0:
                return BallType.CUE, None
            elif number == 8:
                return BallType.EIGHT, 8
            elif 1 <= number <= 15:  # All other numbered balls -> OTHER
                return BallType.OTHER, None
        except ValueError:
            pass

    logger.warning(f"Unknown ball class name: {class_name}, defaulting to UNKNOWN")
    return BallType.UNKNOWN, None


def map_class_id_to_ball_type(
    class_id: int, class_names: Optional[list[str]] = None
) -> tuple[BallType, Optional[int]]:
    """Map YOLO class ID to ball type and number.

    Simplified mapping - no solid/stripe distinction:
    - 0 = cue
    - 8 = eight
    - 1-7, 9-15 = other (all numbered balls)

    Args:
        class_id: YOLO class ID (integer)
        class_names: Optional list of class names (index = class_id)

    Returns:
        Tuple of (BallType, ball_number) - ball_number is always None for OTHER type
    """
    # If class names provided, use them
    if class_names and 0 <= class_id < len(class_names):
        return parse_ball_class_name(class_names[class_id])

    # Fallback: simplified mapping (no solid/stripe distinction)
    # 0 = cue, 8 = eight, all others = other
    if class_id == 0:
        return BallType.CUE, None
    elif class_id == 8:
        return BallType.EIGHT, 8
    elif 1 <= class_id <= 15:
        return BallType.OTHER, None  # All other numbered balls
    else:
        logger.warning(f"Unknown class ID: {class_id}, defaulting to UNKNOWN")
        return BallType.UNKNOWN, None


# =============================================================================
# Coordinate Conversion Functions
# =============================================================================


def bbox_to_center_radius(
    bbox: Union[list[float], NDArray[np.float64]],
    image_shape: tuple[int, int],
    normalized: bool = False,
) -> tuple[tuple[float, float], float]:
    """Convert YOLO bounding box to center coordinates and radius.

    YOLO bounding boxes are in format [x, y, w, h] where:
    - (x, y) is the top-left corner
    - (w, h) is width and height
    - Can be normalized [0-1] or pixel coordinates

    Args:
        bbox: Bounding box [x, y, w, h]
        image_shape: Image dimensions (height, width)
        normalized: Whether bbox coordinates are normalized [0-1]

    Returns:
        Tuple of ((center_x, center_y), radius)
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]

    # Convert normalized coordinates to pixels
    if normalized:
        x *= img_w
        y *= img_h
        w *= img_w
        h *= img_h

    # Calculate center from top-left corner
    center_x = x + w / 2.0
    center_y = y + h / 2.0

    # Estimate radius from bounding box dimensions
    # For circles, use average of width and height divided by 2
    radius = (w + h) / 4.0

    return (center_x, center_y), radius


def bbox_center_to_center_radius(
    bbox: Union[list[float], NDArray[np.float64]],
    image_shape: tuple[int, int],
    normalized: bool = False,
) -> tuple[tuple[float, float], float]:
    """Convert YOLO center-format bounding box to center coordinates and radius.

    Some YOLO formats use center coordinates: [center_x, center_y, w, h]

    Args:
        bbox: Bounding box [center_x, center_y, w, h]
        image_shape: Image dimensions (height, width)
        normalized: Whether bbox coordinates are normalized [0-1]

    Returns:
        Tuple of ((center_x, center_y), radius)
    """
    center_x, center_y, w, h = bbox
    img_h, img_w = image_shape[:2]

    # Convert normalized coordinates to pixels
    if normalized:
        center_x *= img_w
        center_y *= img_h
        w *= img_w
        h *= img_h

    # Estimate radius from bounding box dimensions
    radius = (w + h) / 4.0

    return (center_x, center_y), radius


def estimate_ball_radius_from_bbox(
    bbox: Union[list[float], NDArray[np.float64]],
    image_shape: tuple[int, int],
    normalized: bool = False,
) -> float:
    """Estimate ball radius from bounding box dimensions.

    Uses the average of width and height, divided by 2.
    Can apply corrections based on perspective distortion if needed.

    Args:
        bbox: Bounding box [x, y, w, h] or [center_x, center_y, w, h]
        image_shape: Image dimensions (height, width)
        normalized: Whether bbox coordinates are normalized [0-1]

    Returns:
        Estimated radius in pixels
    """
    # Extract width and height (last two elements)
    w, h = bbox[2], bbox[3]
    img_h, img_w = image_shape[:2]

    # Convert normalized dimensions to pixels
    if normalized:
        w *= img_w
        h *= img_h

    # Average diameter divided by 2 = radius
    radius = (w + h) / 4.0

    return radius


# =============================================================================
# YOLO to Ball Conversion
# =============================================================================


def yolo_to_ball(
    detection: dict[str, Any],
    image_shape: tuple[int, int],
    class_names: Optional[list[str]] = None,
    min_confidence: float = 0.25,
    bbox_format: str = "xyxy",
) -> Optional[Ball]:
    """Convert YOLO detection to Ball dataclass.

    Args:
        detection: YOLO detection dictionary with keys:
            - "bbox": Bounding box coordinates
            - "confidence" or "conf": Detection confidence
            - "class_id" or "class": Class ID
            - "class_name" (optional): Class name string
        image_shape: Image dimensions (height, width)
        class_names: Optional list of class names for mapping
        min_confidence: Minimum confidence threshold to accept detection
        bbox_format: Format of bbox coordinates:
            - "xyxy": [x1, y1, x2, y2] (top-left, bottom-right)
            - "xywh": [x, y, w, h] (top-left, width, height)
            - "cxcywh": [cx, cy, w, h] (center, width, height)
            - "normalized": Any format but with [0-1] coordinates

    Returns:
        Ball object or None if detection is invalid or below confidence threshold
    """
    try:
        # Extract confidence score
        confidence = detection.get("confidence", detection.get("conf", 0.0))

        # Filter by minimum confidence
        if confidence < min_confidence:
            return None

        # Extract bounding box
        bbox = detection.get("bbox", detection.get("box"))
        if bbox is None:
            logger.warning("Detection missing 'bbox' field")
            return None

        # Convert bbox to numpy array for easier manipulation
        bbox = np.array(bbox, dtype=np.float32)

        # Convert bbox to xywh format if needed
        normalized = "normalized" in bbox_format
        if "xyxy" in bbox_format:
            # Convert [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = bbox
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])

        # Extract center and radius based on format
        if "cxcywh" in bbox_format or "center" in bbox_format:
            position, radius = bbox_center_to_center_radius(
                bbox, image_shape, normalized
            )
        else:
            position, radius = bbox_to_center_radius(bbox, image_shape, normalized)

        # Extract class information
        class_id = detection.get("class_id", detection.get("class"))
        class_name = detection.get("class_name")

        # Map to ball type and number
        if class_name:
            ball_type, ball_number = parse_ball_class_name(class_name)
        elif class_id is not None:
            ball_type, ball_number = map_class_id_to_ball_type(
                int(class_id), class_names
            )
        else:
            logger.warning("Detection missing both 'class_id' and 'class_name'")
            return None

        # Create Ball object
        ball = Ball(
            position=position,
            radius=radius,
            ball_type=ball_type,
            number=ball_number,
            confidence=float(confidence),
            velocity=(0.0, 0.0),  # Will be calculated by tracking module
            acceleration=(0.0, 0.0),
            is_moving=False,  # Will be determined by tracking module
        )

        return ball

    except Exception as e:
        logger.error(f"Failed to convert YOLO detection to Ball: {e}", exc_info=True)
        return None


def yolo_detections_to_balls(
    detections: list[dict[str, Any]],
    image_shape: tuple[int, int],
    class_names: Optional[list[str]] = None,
    min_confidence: float = 0.25,
    bbox_format: str = "xyxy",
) -> list[Ball]:
    """Convert multiple YOLO detections to Ball objects.

    Args:
        detections: List of YOLO detection dictionaries
        image_shape: Image dimensions (height, width)
        class_names: Optional list of class names for mapping
        min_confidence: Minimum confidence threshold
        bbox_format: Format of bbox coordinates

    Returns:
        List of Ball objects
    """
    balls = []

    for detection in detections:
        ball = yolo_to_ball(
            detection, image_shape, class_names, min_confidence, bbox_format
        )

        if ball is not None:
            balls.append(ball)

    return balls


# =============================================================================
# YOLO to CueStick Conversion
# =============================================================================


def estimate_cue_angle_from_bbox(
    bbox: Union[list[float], NDArray[np.float64]],
    image_shape: tuple[int, int],
    normalized: bool = False,
) -> float:
    """Estimate cue stick angle from bounding box orientation.

    For elongated objects like cue sticks, the bounding box aspect ratio
    and orientation provide information about the angle.

    Args:
        bbox: Bounding box [x, y, w, h]
        image_shape: Image dimensions (height, width)
        normalized: Whether bbox coordinates are normalized [0-1]

    Returns:
        Angle in degrees from horizontal (0-360)
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]

    # Convert normalized coordinates to pixels
    if normalized:
        x *= img_w
        y *= img_h
        w *= img_w
        h *= img_h

    # Calculate center
    x + w / 2.0
    y + h / 2.0

    # For an axis-aligned bounding box, we need to determine orientation
    # If width > height, cue is more horizontal
    # If height > width, cue is more vertical

    if w > h:
        # Horizontal orientation
        angle = 0.0  # Pointing right
    else:
        # Vertical orientation
        angle = 90.0  # Pointing down

    # Note: This is a simplified estimation. For better accuracy,
    # YOLO models should include rotation prediction or use oriented bounding boxes (OBB)
    # For now, we return a basic estimate

    return angle


def yolo_to_cue(
    detection: dict[str, Any],
    image_shape: tuple[int, int],
    min_confidence: float = 0.3,
    bbox_format: str = "xyxy",
) -> Optional[CueStick]:
    """Convert YOLO cue detection to CueStick dataclass.

    Args:
        detection: YOLO detection dictionary with keys:
            - "bbox": Bounding box coordinates
            - "confidence" or "conf": Detection confidence
            - "angle" (optional): Rotation angle if available from oriented bbox
            - "keypoints" (optional): Keypoints if available (tip, butt positions)
        image_shape: Image dimensions (height, width)
        min_confidence: Minimum confidence threshold
        bbox_format: Format of bbox coordinates

    Returns:
        CueStick object or None if detection is invalid
    """
    try:
        # Extract confidence score
        confidence = detection.get("confidence", detection.get("conf", 0.0))

        # Filter by minimum confidence
        if confidence < min_confidence:
            return None

        # Extract bounding box
        bbox = detection.get("bbox", detection.get("box"))
        if bbox is None:
            logger.warning("Cue detection missing 'bbox' field")
            return None

        # Convert bbox to numpy array
        bbox = np.array(bbox, dtype=np.float32)

        # Convert bbox to xywh format if needed
        normalized = "normalized" in bbox_format
        if "xyxy" in bbox_format:
            # Convert [x1, y1, x2, y2] to [x, y, w, h]
            x1, y1, x2, y2 = bbox
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])

        # Extract tip position
        # Check if keypoints are available (more accurate)
        keypoints = detection.get("keypoints")
        if keypoints and len(keypoints) >= 1:
            # Assume first keypoint is tip
            tip_position = (float(keypoints[0][0]), float(keypoints[0][1]))
        else:
            # Estimate from bbox center
            if "cxcywh" in bbox_format or "center" in bbox_format:
                position, _ = bbox_center_to_center_radius(
                    bbox, image_shape, normalized
                )
            else:
                position, _ = bbox_to_center_radius(bbox, image_shape, normalized)
            tip_position = position

        # Extract or estimate angle
        angle = detection.get("angle")
        if angle is None:
            # Estimate from bbox
            angle = estimate_cue_angle_from_bbox(bbox, image_shape, normalized)
        else:
            angle = float(angle)

        # Calculate length from bbox
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        if normalized:
            w *= img_w
            h *= img_h

        # Length is the longer dimension
        length = max(w, h)

        # Create CueStick object
        cue = CueStick(
            tip_position=tip_position,
            angle=angle,
            length=length,
            confidence=float(confidence),
            state=CueState.AIMING,  # Default state, will be determined by motion analysis
            is_aiming=True,
            tip_velocity=(0.0, 0.0),  # Will be calculated by tracking
            angular_velocity=0.0,
        )

        return cue

    except Exception as e:
        logger.error(
            f"Failed to convert YOLO detection to CueStick: {e}", exc_info=True
        )
        return None


def yolo_cue_to_cue_stick(
    detection: Any,
    image_shape: tuple[int, int],
    min_confidence: float = 0.3,
) -> Optional[CueStick]:
    """Convert YOLO Detection object (from YOLODetector) to CueStick dataclass.

    This function handles Detection objects directly from the YOLODetector class.

    Args:
        detection: Detection object from YOLODetector with attributes:
            - bbox: (x1, y1, x2, y2) bounding box
            - confidence: Detection confidence
            - angle: Rotation angle (calculated from Hough lines)
            - center: (cx, cy) center position
            - line_center: (cx, cy) line center if available
            - line_end1, line_end2: Line endpoints if available
        image_shape: Image dimensions (height, width)
        min_confidence: Minimum confidence threshold

    Returns:
        CueStick object or None if detection is invalid
    """
    try:
        # Check confidence
        if detection.confidence < min_confidence:
            return None

        # Extract tip position
        # Use line_center if available (more accurate from Hough line detection)
        if hasattr(detection, "line_center") and detection.line_center:
            tip_position = detection.line_center
        else:
            tip_position = detection.center

        # Extract angle (already calculated by YOLODetector._estimate_cue_angle)
        angle = detection.angle if hasattr(detection, "angle") else 0.0

        # Calculate length from bbox
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        length = max(width, height)

        # Create CueStick object
        cue = CueStick(
            tip_position=tip_position,
            angle=angle,
            length=length,
            confidence=float(detection.confidence),
            state=CueState.AIMING,  # Default state, will be determined by motion analysis
            is_aiming=True,
            tip_velocity=(0.0, 0.0),  # Will be calculated by tracking
            angular_velocity=0.0,
        )

        return cue

    except Exception as e:
        logger.error(
            f"Failed to convert YOLO Detection to CueStick: {e}", exc_info=True
        )
        return None


# =============================================================================
# Helper Functions
# =============================================================================


def filter_detections_by_confidence(
    detections: list[dict[str, Any]], min_confidence: float = 0.25
) -> list[dict[str, Any]]:
    """Filter YOLO detections by confidence threshold.

    Args:
        detections: List of YOLO detection dictionaries
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of detections
    """
    filtered = []
    for detection in detections:
        confidence = detection.get("confidence", detection.get("conf", 0.0))
        if confidence >= min_confidence:
            filtered.append(detection)

    return filtered


def filter_ball_detections(
    detections: list[dict[str, Any]], class_names: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Filter YOLO detections to keep only ball detections.

    Args:
        detections: List of YOLO detection dictionaries
        class_names: Optional list of class names for filtering

    Returns:
        Filtered list containing only ball detections
    """
    ball_detections = []

    for detection in detections:
        class_id = detection.get("class_id", detection.get("class"))
        class_name = detection.get("class_name", "")

        # Check if this is a ball detection
        is_ball = False

        if class_name:
            # Check class name contains ball-related keywords
            class_name_lower = class_name.lower()
            # Exclude "cue_stick", "stick", etc - only include actual balls
            if "stick" in class_name_lower:
                is_ball = False
            elif (
                any(
                    keyword in class_name_lower
                    for keyword in ["ball", "solid", "stripe", "eight"]
                )
                or class_name_lower == "cue"
            ):  # Only "cue" alone, not "cue_stick"
                is_ball = True

        elif class_id is not None and class_names:
            # Check class name from ID
            if 0 <= class_id < len(class_names):
                name_lower = class_names[class_id].lower()
                if "stick" not in name_lower:
                    if (
                        any(
                            keyword in name_lower
                            for keyword in ["ball", "solid", "stripe", "eight"]
                        )
                        or name_lower == "cue"
                    ):
                        is_ball = True

        elif class_id is not None:
            # Assume IDs 0-15 are balls
            if 0 <= class_id <= 15:
                is_ball = True

        if is_ball:
            ball_detections.append(detection)

    return ball_detections


def filter_cue_detections(
    detections: list[dict[str, Any]], class_names: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Filter YOLO detections to keep only cue stick detections.

    Args:
        detections: List of YOLO detection dictionaries
        class_names: Optional list of class names for filtering

    Returns:
        Filtered list containing only cue detections
    """
    cue_detections = []

    for detection in detections:
        class_id = detection.get("class_id", detection.get("class"))
        class_name = detection.get("class_name", "")

        # Check if this is a cue detection
        is_cue = False

        if class_name:
            # Check class name - must contain "stick" or "cue_stick" or similar
            # But NOT just "cue" alone (that's the cue ball)
            class_name_lower = class_name.lower()
            if "stick" in class_name_lower:
                is_cue = True
            elif "cue_" in class_name_lower and "ball" not in class_name_lower:
                # Matches "cue_stick" but not "cue_ball" or just "cue"
                is_cue = True

        elif class_id is not None and class_names:
            # Check class name from ID
            if 0 <= class_id < len(class_names):
                name_lower = class_names[class_id].lower()
                if "stick" in name_lower or (
                    "cue_" in name_lower and "ball" not in name_lower
                ):
                    is_cue = True

        if is_cue:
            cue_detections.append(detection)

    return cue_detections


# =============================================================================
# Batch Conversion Functions
# =============================================================================


def process_yolo_detections(
    detections: list[dict[str, Any]],
    image_shape: tuple[int, int],
    class_names: Optional[list[str]] = None,
    min_ball_confidence: float = 0.25,
    min_cue_confidence: float = 0.3,
    bbox_format: str = "xyxy",
) -> dict[str, Any]:
    """Process YOLO detections and convert to Vision module objects.

    This is a convenience function that:
    1. Filters detections by type (balls vs cue)
    2. Converts to appropriate dataclasses
    3. Returns organized results

    Args:
        detections: List of YOLO detection dictionaries
        image_shape: Image dimensions (height, width)
        class_names: Optional list of class names
        min_ball_confidence: Minimum confidence for ball detections
        min_cue_confidence: Minimum confidence for cue detections
        bbox_format: Format of bbox coordinates

    Returns:
        Dictionary with keys:
            - "balls": List of Ball objects
            - "cues": List of CueStick objects
            - "ball_count": Number of balls detected
            - "cue_count": Number of cues detected
    """
    # Separate ball and cue detections
    ball_detections = filter_ball_detections(detections, class_names)
    cue_detections = filter_cue_detections(detections, class_names)

    # Convert balls
    balls = yolo_detections_to_balls(
        ball_detections, image_shape, class_names, min_ball_confidence, bbox_format
    )

    # Convert cues
    cues = []
    for detection in cue_detections:
        cue = yolo_to_cue(detection, image_shape, min_cue_confidence, bbox_format)
        if cue is not None:
            cues.append(cue)

    return {
        "balls": balls,
        "cues": cues,
        "ball_count": len(balls),
        "cue_count": len(cues),
    }
