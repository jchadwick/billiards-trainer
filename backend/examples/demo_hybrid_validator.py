"""Hardware-free example of refining an explicitly scaled ball detection.

The old script referred to a ``HybridValidator`` API that is no longer present.
This example uses the current ``BallPositionRefiner`` and demonstrates how a
frame-pixel detection is normalized before canonical geometry is consumed.
"""

import cv2
import numpy as np

from backend.common.models import Ball
from backend.core.resolution_converter import ResolutionConverter
from backend.vision.detection.hybrid_validator import BallPositionRefiner
from backend.vision.models import BallType


def make_synthetic_frame(
    resolution: tuple[int, int], center: tuple[int, int], radius: int
) -> np.ndarray:
    """Create a deterministic frame whose geometry is local to its resolution."""
    width, height = resolution
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(frame, center, radius, (255, 255, 255), -1)
    cv2.circle(
        frame,
        (center[0] - max(1, radius // 3), center[1] - max(1, radius // 3)),
        max(1, radius // 5),
        (235, 235, 235),
        -1,
    )
    return frame


def run_example() -> tuple[Ball, Ball]:
    """Refine one ball and return its source and refined canonical models."""
    frame_resolution = (320, 180)
    frame_center = (160, 90)
    frame_radius = 24
    frame = make_synthetic_frame(frame_resolution, frame_center, frame_radius)

    # The source ball center and radius are frame pixels. The resolution metadata
    # lets conversion scale both before the model crosses into canonical space.
    detection_in_frame = Ball(
        position=frame_center,
        radius=frame_radius,
        ball_type=BallType.CUE,
        confidence=0.95,
        coordinate_space="pixel",
        source_resolution=frame_resolution,
    )
    detection_canonical = detection_in_frame.to_4k_canonical()

    refiner = BallPositionRefiner(max_refinement_distance=5.0)
    refined = refiner.refine_ball_position(detection_canonical, frame)

    print(f"Frame resolution: {frame_resolution[0]}x{frame_resolution[1]} pixels")
    print(
        "Source detection: "
        f"{detection_in_frame.position}, radius {detection_in_frame.radius}px"
    )
    print(
        "Canonical detection: "
        f"{detection_canonical.position}, radius {detection_canonical.radius}px, "
        f"space={detection_canonical.coordinate_space}"
    )
    print(
        "Refined canonical detection: "
        f"{refined.position}, radius {refined.radius}px, "
        f"space={refined.coordinate_space}"
    )
    print(
        f"Canonical position scale: {ResolutionConverter.calculate_scale_to_4k(frame_resolution)}"
    )
    print(f"Refiner statistics: {refiner.stats}")
    return detection_canonical, refined


if __name__ == "__main__":
    run_example()
