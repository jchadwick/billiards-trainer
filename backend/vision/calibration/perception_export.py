"""Export retained Python table calibration into the Rust v1 bundle.

The existing camera calibrator stores its homography in meters and does not
record the source image mode or camera profile. This adapter requires that
missing provenance as explicit arguments and converts units without changing
the historical calibration cache format.
"""

from __future__ import annotations

import math
import unicodedata
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.vision.calibration.camera import TableTransform

ImageSpaceName = Literal["raw", "rectified"]
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
MAX_IDENTIFIER_CHARACTERS = 128
MAX_SOURCE_DIMENSION_PX = 4_294_967_295
CORNER_REPROJECTION_ABS_TOLERANCE_METERS = 1e-6
CORNER_REPROJECTION_REL_TOLERANCE = 1e-6


def build_table_calibration_v1(
    transform: TableTransform,
    *,
    calibration_revision: int,
    camera_profile_id: str,
    rectification_revision: int,
    image_space: ImageSpaceName,
    source_width_px: int,
    source_height_px: int,
) -> dict[str, object]:
    """Build a fail-closed, revision-bound pixel-to-table calibration bundle.

    The caller must supply provenance the legacy ``TableTransform`` does not
    retain. Its matrix maps pixels to meters with center-origin axes; the v1
    bundle scales only the output-coordinate rows to millimeters. A numerical
    corner check catches a transform whose stored matrix and declared corner
    convention no longer agree. This check is not a physical-accuracy claim.
    """
    _validate_revision(calibration_revision, "calibration_revision")
    _validate_revision(rectification_revision, "rectification_revision")
    _validate_identifier(camera_profile_id)
    _validate_source_dimensions(source_width_px, source_height_px)
    if image_space not in ("raw", "rectified"):
        raise ValueError("image_space must be 'raw' or 'rectified'")

    table_width_m, table_height_m = transform.table_dimensions
    if (
        not math.isfinite(table_width_m)
        or not math.isfinite(table_height_m)
        or table_width_m <= 0
        or table_height_m <= 0
    ):
        raise ValueError("table dimensions must be finite and positive")

    expected_world_corners = [
        (-table_width_m / 2, -table_height_m / 2),
        (table_width_m / 2, -table_height_m / 2),
        (table_width_m / 2, table_height_m / 2),
        (-table_width_m / 2, table_height_m / 2),
    ]
    if len(transform.table_corners_world) != len(expected_world_corners):
        raise ValueError("table calibration must contain four ordered world corners")
    for actual, expected in zip(
        transform.table_corners_world, expected_world_corners, strict=True
    ):
        if len(actual) != 2 or any(
            not math.isclose(
                float(value),
                target,
                rel_tol=CORNER_REPROJECTION_REL_TOLERANCE,
                abs_tol=CORNER_REPROJECTION_ABS_TOLERANCE_METERS,
            )
            for value, target in zip(actual, expected, strict=True)
        ):
            raise ValueError(
                "table world corners do not use the v1 center-origin frame"
            )

    source_corners = np.asarray(transform.table_corners_pixel, dtype=np.float64)
    if source_corners.shape != (4, 2) or not np.isfinite(source_corners).all():
        raise ValueError("table calibration must contain four finite image corners")
    if (
        (source_corners[:, 0] < 0).any()
        or (source_corners[:, 0] > source_width_px).any()
        or (source_corners[:, 1] < 0).any()
        or (source_corners[:, 1] > source_height_px).any()
    ):
        raise ValueError("table calibration corners lie outside the source image")

    matrix_m = np.asarray(transform.homography_matrix, dtype=np.float64)
    if matrix_m.shape != (3, 3) or not np.isfinite(matrix_m).all():
        raise ValueError("table homography must be a finite 3-by-3 matrix")
    largest = float(np.max(np.abs(matrix_m)))
    if largest == 0 or not math.isfinite(largest):
        raise ValueError("table homography must be invertible")
    normalized = matrix_m / largest
    determinant = float(np.linalg.det(normalized))
    frobenius_norm = float(np.linalg.norm(normalized))
    relative_guard = np.finfo(float).eps * 64 * frobenius_norm**3
    if not math.isfinite(determinant) or abs(determinant) <= relative_guard:
        raise ValueError("table homography must be invertible")

    projected_corners = _project_corners(matrix_m, source_corners)
    for projected, expected in zip(
        projected_corners, expected_world_corners, strict=True
    ):
        if not all(
            math.isclose(
                float(actual),
                target,
                rel_tol=CORNER_REPROJECTION_REL_TOLERANCE,
                abs_tol=CORNER_REPROJECTION_ABS_TOLERANCE_METERS,
            )
            for actual, target in zip(projected, expected, strict=True)
        ):
            raise ValueError(
                "table homography does not map its ordered corners to the declared frame"
            )

    matrix_mm = matrix_m.copy()
    matrix_mm[:2, :] *= 1000.0
    return {
        "schema_version": 1,
        "calibration_revision": calibration_revision,
        "camera_profile_id": camera_profile_id,
        "rectification_revision": rectification_revision,
        "image_space": image_space,
        "source_width_px": source_width_px,
        "source_height_px": source_height_px,
        "coordinate_frame": "center_origin_x_corner0_to1_y_corner0_to3",
        "table_width_mm": table_width_m * 1000.0,
        "table_height_mm": table_height_m * 1000.0,
        "pixel_to_table_mm": matrix_mm.tolist(),
    }


def _validate_revision(revision: int, field_name: str) -> None:
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or not 0 <= revision <= MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(f"{field_name} must be a nonnegative safe JSON integer")


def _validate_identifier(identifier: str) -> None:
    if (
        not isinstance(identifier, str)
        or not 1 <= len(identifier) <= MAX_IDENTIFIER_CHARACTERS
        or any(unicodedata.category(character) == "Cc" for character in identifier)
    ):
        raise ValueError("camera_profile_id must be a bounded non-control identifier")


def _validate_source_dimensions(width: int, height: int) -> None:
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width > MAX_SOURCE_DIMENSION_PX
        or height > MAX_SOURCE_DIMENSION_PX
    ):
        raise ValueError(
            "source image dimensions must be positive u32 integers for the Rust contract"
        )


def _project_corners(
    matrix: NDArray[np.float64], corners: NDArray[np.float64]
) -> list[tuple[float, float]]:
    scale = float(np.max(np.abs(matrix)))
    normalized = matrix / scale
    projected: list[tuple[float, float]] = []
    for x_px, y_px in corners:
        homogeneous = normalized @ np.array([x_px, y_px, 1.0])
        x_m, y_m, denominator = homogeneous
        denominator_scale = float(np.sum(np.abs(normalized[2] * [x_px, y_px, 1.0])))
        if (
            not math.isfinite(float(denominator))
            or abs(float(denominator)) <= np.finfo(float).eps * 64 * denominator_scale
        ):
            raise ValueError("table homography has a singular calibration corner")
        x_m, y_m = float(x_m / denominator), float(y_m / denominator)
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError(
                "table homography produces a non-finite calibration corner"
            )
        projected.append((x_m, y_m))
    return projected
