#!/usr/bin/env python3
"""Audit a WPA-style table-mark grid and rail-to-cloth plane transfer.

This is a reproducible, diagnostics-only experiment. It uses saved image mark
annotations, an existing checkerboard intrinsics fit, and standards-derived
mark locations. It never fits the table map from ball positions and does not
change production behavior.

Run with the repository's Python runtime containing NumPy, for example:
  python3 thoughts/calibration/offline-mark-grid-calibration-audit-20261027.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "thoughts" / "calibration"
REGISTRATION = CAL / "table-feature-registration-audit-20261027" / "summary.json"
INTRINSICS = CAL / "checkerboard-eight-view-baseline.json"
CORPUS = CAL / "native-localization-corpus-v1.json"
MEASUREMENTS = CAL / "table-grid-20261026-114241-measurements.json"
OUT = CAL / "mark-grid-calibration-audit-20261027"

TABLE_LENGTH_MM = 2336.8
TABLE_WIDTH_MM = 1155.7
BALL_DIAMETER_MM = 57.15
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2.0
# WPA Recommended Equipment Specifications, §6 (8-ft table): 3 11/16 in from
# cushion nose, with the published text showing +3.175 mm. For diagnostics we
# report a symmetric ±3.175 mm sensitivity allowance; this is not a claim that
# the source specifies a symmetric tolerance, nor that a home table conforms.
RAIL_SIGHT_OFFSET_MM = 3.6875 * 25.4
RAIL_SIGHT_OFFSET_SENSITIVITY_MM = 0.125 * 25.4
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
MARK_IMAGE_IDS = ("measured_720", "framing_before_720")

# Long-rail sights observed in the saved images. The center sight at 4/8 is
# absent at the side pocket; the remaining positions are the official 1/8
# divisions. Four lower-rail sightings are precisely seeded by the prior
# annotation audit at 2/8, 3/8, 6/8, and 7/8. A fifth sight at 5/8 is visible
# under glare and retained separately as an approximate held-out check.
LONG_RAIL_K = {
    "top_rail_surface_1": 1,
    "top_rail_surface_2": 2,
    "top_rail_surface_3": 3,
    "top_rail_surface_4": 5,
    "top_rail_surface_5": 6,
    "top_rail_surface_6": 7,
    "bottom_rail_surface_1": 2,
    "bottom_rail_surface_2": 3,
    "bottom_rail_surface_3": 6,
    "bottom_rail_surface_4": 7,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Fit a normalized 2D homography using a deterministic SVD DLT."""
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and destination must both be N×2")
    if len(source) < 4:
        raise ValueError("at least four point pairs are required")

    def normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(points, axis=0)
        average_distance = float(np.mean(np.linalg.norm(points - center, axis=1)))
        if average_distance <= 1e-12:
            raise ValueError("degenerate point distribution")
        scale = math.sqrt(2.0) / average_distance
        transform = np.array(
            [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]]
        )
        homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed = (transform @ homogeneous.T).T
        return transformed[:, :2], transform

    src, ts = normalize(source)
    dst, td = normalize(destination)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=True)
    hn = vh[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    if abs(h[2, 2]) > 1e-12:
        h /= h[2, 2]
    else:
        h /= np.linalg.norm(h)
    return h


def project_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (h @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("homography mapped a point to infinity")
    return projected[:, :2] / projected[:, 2:3]


def robust_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "count": int(len(values)),
        "rms": float(np.sqrt(np.mean(values**2))) if len(values) else math.nan,
        "median": float(np.median(values)) if len(values) else math.nan,
        "p95": float(np.quantile(values, 0.95)) if len(values) else math.nan,
        "max": float(np.max(values)) if len(values) else math.nan,
    }


def get_rail_data(registration: dict[str, Any], image_id: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    annotations = registration["rail_dot_annotations_by_image"][image_id]
    ids: list[str] = []
    world: list[tuple[float, float]] = []
    image: list[tuple[float, float]] = []
    for feature_id, k in LONG_RAIL_K.items():
        item = annotations[feature_id]
        if item["status"] != "localized":
            continue
        x = TABLE_LENGTH_MM * k / 8.0
        y = -RAIL_SIGHT_OFFSET_MM if feature_id.startswith("top_") else TABLE_WIDTH_MM + RAIL_SIGHT_OFFSET_MM
        ids.append(feature_id)
        world.append((x, y))
        image.append(tuple(item["center_px"]))
    if len(ids) < 8:
        raise ValueError(f"expected at least 8 long-rail sight observations, got {len(ids)}")
    return ids, np.asarray(world), np.asarray(image)


def fit_and_score_rail_plane(ids: list[str], world: np.ndarray, image: np.ndarray) -> dict[str, Any]:
    h = normalized_dlt(world, image)
    prediction = project_homography(h, world)
    errors = np.linalg.norm(prediction - image, axis=1)
    loo = []
    for index, feature_id in enumerate(ids):
        keep = np.arange(len(ids)) != index
        h_loo = normalized_dlt(world[keep], image[keep])
        pred = project_homography(h_loo, world[index : index + 1])[0]
        loo.append({"feature_id": feature_id, "error_px": float(np.linalg.norm(pred - image[index]))})
    return {
        "homography_world_mm_to_image_px": h.tolist(),
        "fit_error_px": robust_stats(errors),
        "leave_one_sight_out_error_px": robust_stats(np.asarray([x["error_px"] for x in loo])),
        "per_sight_fit_error_px": [
            {"feature_id": feature_id, "error_px": float(error)} for feature_id, error in zip(ids, errors, strict=True)
        ],
        "per_sight_leave_one_out_px": loo,
    }


def decompose_pose(h: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a metric plane homography into a camera pose (rail plane z=0)."""
    b = np.linalg.inv(k) @ h
    c1, c2, c3 = b[:, 0], b[:, 1], b[:, 2]
    scale = 2.0 / (np.linalg.norm(c1) + np.linalg.norm(c2))
    if c3[2] * scale < 0:
        scale = -scale
    raw_r1 = c1 * scale
    raw_r2 = c2 * scale
    t = c3 * scale
    raw_r3 = np.cross(raw_r1, raw_r2)
    raw_rotation = np.column_stack((raw_r1, raw_r2, raw_r3))
    u, _, vh = np.linalg.svd(raw_rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    # A single sign flip of both in-plane columns and translation is equivalent
    # in the homography. Keep all observed rail points in front of the camera.
    sample = np.column_stack((np.linspace(0.0, TABLE_LENGTH_MM, 8), np.zeros(8), np.zeros(8)))
    depths = (rotation @ sample.T + t[:, None])[2]
    if np.median(depths) <= 0:
        rotation[:, :2] *= -1
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
        t *= -1
    return rotation, t


def project_camera(k: np.ndarray, rotation: np.ndarray, translation: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    camera = (rotation @ xyz.T + translation[:, None]).T
    if np.any(camera[:, 2] <= 0):
        raise ValueError("point projected behind the camera")
    pixels_h = (k @ camera.T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def solve_plane_separation(
    k: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    world_xy: np.ndarray,
    image_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit one common vertical separation h for cloth spots at z=+h."""
    lo, hi = 0.0, 350.0
    # A deterministic golden-section search; the objective is low-dimensional.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - phi * (hi - lo)
    d = lo + phi * (hi - lo)

    def objective(height: float) -> float:
        xyz = np.column_stack((world_xy, np.full(len(world_xy), height)))
        residual = project_camera(k, rotation, translation, xyz) - image_points
        return float(np.sum(residual**2))

    fc, fd = objective(c), objective(d)
    for _ in range(100):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = objective(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = objective(d)
    height = (lo + hi) / 2.0
    predictions = project_camera(k, rotation, translation, np.column_stack((world_xy, np.full(len(world_xy), height))))
    errors = np.linalg.norm(predictions - image_points, axis=1)
    return height, predictions, errors


def unproject_to_fixed_plane(
    k: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    image_point: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    """Intersect one calibrated image ray with a fixed physical height plane."""
    camera_ray = np.linalg.inv(k) @ np.array([image_point[0], image_point[1], 1.0])
    world_ray = rotation.T @ camera_ray
    camera_center = -rotation.T @ translation
    if abs(world_ray[2]) < 1e-10:
        raise ValueError("image ray is parallel to requested plane")
    lam = (plane_z - camera_center[2]) / world_ray[2]
    point = camera_center + lam * world_ray
    return point[:2]


def assign_side_sights(
    h: np.ndarray,
    annotations: dict[str, Any],
    rail: str,
) -> dict[str, Any]:
    """Match ambiguous saved side-rail detections against official quarter marks."""
    x = -RAIL_SIGHT_OFFSET_MM if rail == "left" else TABLE_LENGTH_MM + RAIL_SIGHT_OFFSET_MM
    expected_world = np.asarray(
        [[x, TABLE_WIDTH_MM * fraction] for fraction in (0.25, 0.5, 0.75)], dtype=float
    )
    expected_pixels = project_homography(h, expected_world)
    prefix = f"{rail}_rail_vertical_face_"
    observed = []
    for name, item in annotations.items():
        if name.startswith(prefix) and item.get("status") == "localized":
            observed.append((name, np.asarray(item["center_px"], dtype=float)))
    if len(observed) < 2:
        return {"rail": rail, "status": "insufficient_observations", "observed_count": len(observed)}
    best: tuple[float, tuple[tuple[int, int], ...], int] | None = None
    # Match the smaller side of the two sets. This tolerates missing side-rail
    # annotations and leaves surplus annotations explicitly unassigned.
    match_count = min(len(observed), len(expected_pixels))
    for observed_subset in itertools.combinations(range(len(observed)), match_count):
        for expected_subset in itertools.combinations(range(len(expected_pixels)), match_count):
            for observed_permutation in itertools.permutations(observed_subset):
                pairs = tuple(zip(expected_subset, observed_permutation, strict=True))
                cost = sum(
                    float(np.sum((observed[observed_idx][1] - expected_pixels[expected_idx]) ** 2))
                    for expected_idx, observed_idx in pairs
                )
                assigned = set(observed_permutation)
                unassigned = next((idx for idx in range(len(observed)) if idx not in assigned), -1)
                candidate = (cost, pairs, unassigned)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    assert best is not None
    assignments = []
    for expected_index, observed_index in best[1]:
        name, point = observed[observed_index]
        assignments.append(
            {
                "expected_fraction": (0.25, 0.5, 0.75)[expected_index],
                "feature_id": name,
                "predicted_px_from_long_rail_grid": expected_pixels[expected_index].tolist(),
                "observed_px": point.tolist(),
                "transfer_residual_px": float(np.linalg.norm(point - expected_pixels[expected_index])),
            }
        )
    surplus = None if best[2] < 0 else {"feature_id": observed[best[2]][0], "observed_px": observed[best[2]][1].tolist()}
    return {
        "rail": rail,
        "assignment_basis": "nearest three-permutation to expected quarter marks predicted from long-rail-only fit",
        "assignments": assignments,
        "surplus_unassigned_detection": surplus,
        "note": "These side-rail marks are held out; they do not train the homography or pose.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    registration = json.loads(REGISTRATION.read_text())
    intrinsics_data = json.loads(INTRINSICS.read_text())
    corpus = json.loads(CORPUS.read_text())
    measurements = json.loads(MEASUREMENTS.read_text())
    baseline_fit = intrinsics_data["models"]["zero"]["full_fit"]
    k = np.asarray(baseline_fit["camera_matrix"], dtype=float)

    frame_results: dict[str, Any] = {}
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    homographies: dict[str, np.ndarray] = {}
    for image_id in MARK_IMAGE_IDS:
        ids, world, image = get_rail_data(registration, image_id)
        scored = fit_and_score_rail_plane(ids, world, image)
        h = np.asarray(scored["homography_world_mm_to_image_px"], dtype=float)
        rotation, translation = decompose_pose(h, k)
        poses[image_id] = (rotation, translation)
        homographies[image_id] = h
        annotations = registration["rail_dot_annotations_by_image"][image_id]
        side = {rail: assign_side_sights(h, annotations, rail) for rail in ("left", "right")}
        frame_results[image_id] = {
            "fit_points": len(ids),
            "fit_ids": ids,
            "rail_sight_fit": scored,
            "held_out_side_sight_checks": side,
            "pose_from_checkerboard_intrinsics": {
                "rotation_world_to_camera": rotation.tolist(),
                "translation_mm": translation.tolist(),
                "intrinsics_id": "checkerboard-eight-view-baseline.models.zero.full_fit",
            },
        }

    # A fifth lower-rail dot is visible near a strong specular reflection. It
    # was not in the previous seeded annotation set, so keep it held out and
    # record only an approximate visual center rather than training on it.
    fifth_bottom_expected_xy = np.asarray(
        [[TABLE_LENGTH_MM * 5.0 / 8.0, TABLE_WIDTH_MM + RAIL_SIGHT_OFFSET_MM]], dtype=float
    )
    fifth_bottom_predicted_px = project_homography(
        homographies["framing_before_720"], fifth_bottom_expected_xy
    )[0]
    fifth_bottom_visual_center_px = np.asarray([784.0, 690.0], dtype=float)
    fifth_bottom_visual_residual_px = float(
        np.linalg.norm(fifth_bottom_predicted_px - fifth_bottom_visual_center_px)
    )
    frame_results["framing_before_720"]["additional_unseeded_bottom_sight_check"] = {
        "feature_id": "bottom_rail_surface_unseeded_5",
        "expected_fraction": "5/8 of table length",
        "world_xy_mm": fifth_bottom_expected_xy[0].tolist(),
        "predicted_pixel_from_top_and_other_bottom_marks": fifth_bottom_predicted_px.tolist(),
        "approximate_visual_center_px": fifth_bottom_visual_center_px.tolist(),
        "approximate_visual_residual_px": fifth_bottom_visual_residual_px,
        "quality": "held-out qualitative check only; the sight is under glare and the pixel center was visually estimated",
    }

    # The two cloth circles occur only in this framing scene. Official rules
    # define Head/Foot Strings at the quarter lines; this identity/location is
    # treated as a hypothesis pending table-specific confirmation.
    cloth_annotations = registration["cloth_marker_annotations_by_image"]["framing_before_720"]
    cloth_points = np.asarray([x["center_px"] for x in cloth_annotations], dtype=float)
    cloth_xy = np.asarray(
        [[TABLE_LENGTH_MM / 4.0, TABLE_WIDTH_MM / 2.0], [3.0 * TABLE_LENGTH_MM / 4.0, TABLE_WIDTH_MM / 2.0]],
        dtype=float,
    )
    framing_rotation, framing_translation = poses["framing_before_720"]
    fitted_height, cloth_predictions, cloth_errors = solve_plane_separation(
        k, framing_rotation, framing_translation, cloth_xy, cloth_points
    )
    rail_plane_spot_predictions = project_homography(homographies["framing_before_720"], cloth_xy)
    rail_plane_spot_errors = np.linalg.norm(rail_plane_spot_predictions - cloth_points, axis=1)
    rail_inverse = np.linalg.inv(homographies["framing_before_720"])
    cloth_as_rail_xy_h = (rail_inverse @ np.column_stack((cloth_points, np.ones(2))).T).T
    cloth_as_rail_xy = cloth_as_rail_xy_h[:, :2] / cloth_as_rail_xy_h[:, 2:3]
    cloth_as_rail_xy_error = np.linalg.norm(cloth_as_rail_xy - cloth_xy, axis=1)
    height_from_each = []
    for index in range(2):
        other = 1 - index
        h_i, _, _ = solve_plane_separation(
            k,
            framing_rotation,
            framing_translation,
            cloth_xy[index : index + 1],
            cloth_points[index : index + 1],
        )
        pred_other = project_camera(
            k,
            framing_rotation,
            framing_translation,
            np.column_stack((cloth_xy[other : other + 1], [h_i])),
        )[0]
        height_from_each.append(
            {
                "fit_marker_index": index,
                "estimated_rail_to_cloth_plane_separation_mm": h_i,
                "heldout_marker_index": other,
                "heldout_prediction_px": pred_other.tolist(),
                "heldout_residual_px": float(np.linalg.norm(pred_other - cloth_points[other])),
            }
        )

    # Checkerboard full fit plus its eight leave-one-view-out intrinsics
    # estimate how much this geometry depends on the weakly constrained K.
    # This is a sensitivity envelope, not a statistical confidence interval.
    k_sensitivity = []
    for fold in intrinsics_data["models"]["zero"]["folds"]:
        k_fold = np.asarray(fold["training_fit"]["camera_matrix"], dtype=float)
        h = homographies["framing_before_720"]
        r, t = decompose_pose(h, k_fold)
        h_fit, _, e = solve_plane_separation(k_fold, r, t, cloth_xy, cloth_points)
        cross = []
        for index in range(2):
            other = 1 - index
            hi, _, _ = solve_plane_separation(k_fold, r, t, cloth_xy[index : index + 1], cloth_points[index : index + 1])
            pred = project_camera(k_fold, r, t, np.column_stack((cloth_xy[other : other + 1], [hi])))[0]
            cross.append(float(np.linalg.norm(pred - cloth_points[other])))
        k_sensitivity.append(
            {
                "held_out_checkerboard_view": fold["held_out_view"],
                "h_mm_joint": h_fit,
                "cloth_marker_joint_fit_residuals_px": e.tolist(),
                "one_marker_to_other_marker_residuals_px": cross,
            }
        )

    # Held-out ball references are evaluated only after the table geometry is
    # established; they are neither used to fit the rail map nor the plane gap.
    measured_rotation, measured_translation = poses["measured_720"]
    image_record = next(x for x in corpus["images"] if x["id"] == "measured-grid-720")
    references = {
        x["id"]: x for x in corpus["physical_reference_sets"]["measured_grid_nine"]["observations"]
    }
    proposal_by_id = {x["ball_id"]: x for x in image_record["ball_proposals"]["proposals"]}
    ball_diagnostics = []
    for ball_id, reference in references.items():
        proposal = proposal_by_id[ball_id]
        image_center = np.asarray(proposal["bbox_center_source_px"], dtype=float)
        inferred_xy = unproject_to_fixed_plane(
            k, measured_rotation, measured_translation, image_center, fitted_height - BALL_RADIUS_MM
        )
        target_xy = np.asarray(reference["center_from_upper_left_mm"], dtype=float)
        error = inferred_xy - target_xy
        ball_diagnostics.append(
            {
                "ball_id": ball_id,
                "coarse_bbox_midpoint_px": image_center.tolist(),
                "reference_xy_mm": target_xy.tolist(),
                "inferred_xy_mm_using_fixed_ball_radius_plane": inferred_xy.tolist(),
                "assumed_ball_center_plane_z_mm_relative_to_rail_cap": fitted_height - BALL_RADIUS_MM,
                "xy_error_mm": error.tolist(),
                "radial_error_mm": float(np.linalg.norm(error)),
                "quality_limit": "diagnostic only: manual bbox midpoint is not a validated ball-center observation and physical reference uncertainty is unknown",
            }
        )

    # Compare a naïve rail-plane inverse and the 3D plane-aware ray result for
    # the same bbox midpoints. The latter is the physically meaningful model.
    rail_h_inv = np.linalg.inv(homographies["measured_720"])
    for row in ball_diagnostics:
        pixel = np.asarray(row["coarse_bbox_midpoint_px"], dtype=float)
        homogeneous = rail_h_inv @ np.array([pixel[0], pixel[1], 1.0])
        row["naive_same_rail_plane_xy_mm"] = (homogeneous[:2] / homogeneous[2]).tolist()
        row["naive_same_plane_radial_error_mm"] = float(
            np.linalg.norm(np.asarray(row["naive_same_rail_plane_xy_mm"]) - np.asarray(row["reference_xy_mm"]))
        )

    # Intrinsic sensitivity for ball XY is summarized across the 8 LOO models.
    ball_k_sensitivity = []
    for fold in intrinsics_data["models"]["zero"]["folds"]:
        k_fold = np.asarray(fold["training_fit"]["camera_matrix"], dtype=float)
        r, t = decompose_pose(homographies["measured_720"], k_fold)
        # Plane offset is re-estimated from the two cloth spots for each K.
        rf, tf = decompose_pose(homographies["framing_before_720"], k_fold)
        hf, _, _ = solve_plane_separation(k_fold, rf, tf, cloth_xy, cloth_points)
        rows = []
        for row in ball_diagnostics:
            xy = unproject_to_fixed_plane(
                k_fold,
                r,
                t,
                np.asarray(row["coarse_bbox_midpoint_px"]),
                hf - BALL_RADIUS_MM,
            )
            rows.append(xy.tolist())
        ball_k_sensitivity.append(
            {"held_out_checkerboard_view": fold["held_out_view"], "fitted_h_mm": hf, "ball_xy_mm": rows}
        )
    per_ball_sensitivity = []
    for index, row in enumerate(ball_diagnostics):
        coords = np.asarray([entry["ball_xy_mm"][index] for entry in ball_k_sensitivity])
        per_ball_sensitivity.append(
            {
                "ball_id": row["ball_id"],
                "fold_min_xy_mm": np.min(coords, axis=0).tolist(),
                "fold_max_xy_mm": np.max(coords, axis=0).tolist(),
                "fold_peak_to_peak_mm": np.ptp(coords, axis=0).tolist(),
                "fold_max_radial_deviation_from_full_fit_mm": float(
                    np.max(
                        np.linalg.norm(
                            coords - np.asarray(row["inferred_xy_mm_using_fixed_ball_radius_plane"]), axis=1
                        )
                    )
                ),
            }
        )

    report = {
        "schema_version": 1,
        "audit_date": "2026-10-27",
        "purpose": "diagnostics-only test of standards-based rail sight coordinates and rail-cap-to-cloth-plane transfer",
        "production_behavior_changed": False,
        "no_camera_capture_performed": True,
        "no_ball_coordinates_used_for_fitting": True,
        "user_coordinate_contract": {
            "origin": "upper-left cushion-nose intersection in the saved image",
            "positive_x": "image-right along the 92-inch long axis",
            "positive_y": "image-down along the 45.5-inch short axis",
            "units": "millimeters",
            "table_length_mm": TABLE_LENGTH_MM,
            "table_width_mm": TABLE_WIDTH_MM,
            "physical_ball_diameter_mm": BALL_DIAMETER_MM,
            "outputs": "X/Y only; ball radius is a fixed known geometry term, not an estimated Z result",
        },
        "official_layout_basis": {
            "current_wpa_rules_url": "https://wpapool.com/wp-content/uploads/2025/10/2025.09.15-WPA-Rules-NP.pdf",
            "recommended_equipment_url": "https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf",
            "layout_summary": "WPA rules describe sights at 1/8 length and 1/4 width; equipment specs state 18 sights, 11.5 in spacing on an 8-ft table and center of sights 3 11/16 in (+3.175 mm in the source notation) from cushion nose.",
            "home_table_caveat": "WPA equipment specs say these competition recommendations do not necessarily apply to commercial home-use tables. The user's 45.5 in measured width differs from the 46 in 8-ft specification by 0.5 in. The grid is a testable hypothesis, not confirmed table-specific truth.",
            "assumed_sight_coordinates": {
                "long_rails": "x = k*92 in/8 for k={1,2,3,5,6,7}; center sight is absent at the side pocket; y = -offset or width+offset",
                "short_rails_held_out": "x = -offset or length+offset; y = width*{1/4,1/2,3/4}",
                "offset_mm": RAIL_SIGHT_OFFSET_MM,
                "sensitivity_allowance_mm": RAIL_SIGHT_OFFSET_SENSITIVITY_MM,
            },
            "felt_mark_hypothesis": "The two saved cloth circles are treated as Head Spot/Foot Spot at x=L/4 and 3L/4, y=W/2 based on appearance and WPA string definitions; their identity is not yet user-confirmed.",
        },
        "intrinsics": {
            "source": "checkerboard-eight-view-baseline.json models.zero.full_fit",
            "camera_matrix": k.tolist(),
            "radial_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "checkerboard_validation": "8-view training-only full fit; leave-one-view-out focal range and principal point variation are reported in the referenced baseline; this is not a production-approved calibration.",
        },
        "saved_input_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (REGISTRATION, INTRINSICS, CORPUS, MEASUREMENTS)
        },
        "rail_plane_and_pose_by_frame": frame_results,
        "rail_cap_to_cloth_transfer": {
            "training_frame": "framing_before_720",
            "cloth_marker_observations_px": cloth_points.tolist(),
            "assumed_marker_xy_mm": cloth_xy.tolist(),
            "joint_estimated_plane_separation_mm": fitted_height,
            "joint_fit_predictions_px": cloth_predictions.tolist(),
            "joint_fit_residuals_px": cloth_errors.tolist(),
            "joint_fit_residual_stats_px": robust_stats(cloth_errors),
            "naive_single_rail_plane_prediction_px": rail_plane_spot_predictions.tolist(),
            "naive_single_rail_plane_residuals_px": rail_plane_spot_errors.tolist(),
            "naive_single_rail_plane_residual_stats_px": robust_stats(rail_plane_spot_errors),
            "naive_single_rail_plane_inverse_xy_mm": cloth_as_rail_xy.tolist(),
            "naive_single_rail_plane_xy_error_mm": cloth_as_rail_xy_error.tolist(),
            "single_marker_cross_validation": height_from_each,
            "intrinsics_leave_one_checkerboard_view_sensitivity": k_sensitivity,
            "interpretation": "The 3D plane bridge is evaluated explicitly; do not apply the rail-plane homography directly to cloth/ball pixels.",
        },
        "held_out_nine_ball_reference_diagnostics": {
            "used_for_fit": False,
            "source_uncertainty": "Physical reference uncertainty is unknown; coarse bbox midpoint is not a ground-truth center.",
            "fixed_ball_center_height_mm": BALL_RADIUS_MM,
            "results": ball_diagnostics,
            "radial_error_summary_mm": robust_stats(np.asarray([x["radial_error_mm"] for x in ball_diagnostics])),
            "intrinsics_loo_sensitivity": per_ball_sensitivity,
        },
        "limitations": [
            "The WPA source is a recommended tournament standard and explicitly warns it may not apply to home tables.",
            "The exact 45.5 in table width conflicts with the nominal 46 in 8-ft table; exact long-rail spacing can still match 11.5 in.",
            "Rail sight centers use manual seeded bright-component annotations with image-level uncertainty; lower-rail dots are only partially visible.",
            "The side rail dots are held-out and manually ambiguous; their agreement is a layout check, not an independent survey.",
            "Two cloth spots determine only one shared plane-separation scalar after calibrated camera pose; if their identity or quarter-line location is wrong, the estimate is invalid.",
            "Checkerboard intrinsics are exploratory and weakly constrained by the small 10 x 7 in target and available views; leave-one-view-out K sensitivity is not a confidence interval.",
            "No table-specific measurement of rail-cap height or sight-center offsets was made; the published nominal offset and +3.175 mm notation do not establish this home table's exact geometry or millimeter-level qualification.",
            "Ball midpoint comparisons are diagnostics only and do not qualify localization accuracy or correct sphere silhouette centers.",
            "No live cadence, tracking, availability, or shot-coaching result can be inferred from still images.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }

    # Executable self-checks: transform recovery and fixed-plane ray round trip.
    synthetic_h = np.array([[1.07, 0.02, 125.0], [-0.01, 0.93, 84.0], [0.0002, -0.0001, 1.0]])
    synthetic_xy = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 30.0], [50.0, 30.0], [25.0, 15.0]])
    synthetic_px = project_homography(synthetic_h, synthetic_xy)
    recovered_h = normalized_dlt(synthetic_xy, synthetic_px)
    synthetic_h_error = float(np.max(np.abs(recovered_h - synthetic_h)))
    if synthetic_h_error > 1e-8:
        raise AssertionError(f"synthetic homography recovery failed: {synthetic_h_error}")
    # Noiseless pose-based point projection/inversion must recover X/Y at a
    # fixed, known plane. Use a simple oblique pose and fixed K.
    theta = math.radians(12.0)
    synthetic_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(theta), -math.sin(theta)], [0.0, math.sin(theta), math.cos(theta)]]
    )
    synthetic_translation = np.array([-400.0, -250.0, 2100.0])
    synthetic_k = np.array([[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    test_xyz = np.array([[200.0, 100.0, 28.575], [1800.0, 900.0, 28.575]])
    test_px = project_camera(synthetic_k, synthetic_rotation, synthetic_translation, test_xyz)
    test_xy = np.asarray(
        [unproject_to_fixed_plane(synthetic_k, synthetic_rotation, synthetic_translation, point, 28.575) for point in test_px]
    )
    ray_roundtrip_error = float(np.max(np.abs(test_xy - test_xyz[:, :2])))
    if ray_roundtrip_error > 1e-7:
        raise AssertionError(f"fixed-plane ray round trip failed: {ray_roundtrip_error}")
    report["synthetic_self_test"] = {
        "homography_recovery_max_abs_matrix_error": synthetic_h_error,
        "fixed_plane_ray_roundtrip_max_xy_error_mm": ray_roundtrip_error,
        "status": "passed",
    }

    output_json = args.output / "summary.json"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Mark-grid calibration audit",
        "",
        "This is an offline, diagnostics-only experiment. No production logic or camera settings changed.",
        "",
        "## Decision",
        "",
        "The saved rail dots strongly resemble the WPA 8-foot sight lattice, and the saved cloth marks are visually consistent with the head/foot spots. The standard supplies a useful candidate coordinate grid. However, WPA explicitly cautions that the competition specification may not apply to home tables, the user's measured width is 0.5 in smaller than the nominal 8-foot width, and the saved checkerboard intrinsics are not yet qualified. The measured results below therefore test feasibility but do not yet establish a production metric map.",
        "",
        "The key check is not a same-plane rail homography: it decomposes that homography with saved checkerboard intrinsics, estimates one common rail-cap-to-cloth plane separation from the two visible cloth spots, and checks each spot against the other.",
        "",
        "## Main measurements",
        "",
        f"- Rail sight offset used: {RAIL_SIGHT_OFFSET_MM:.4f} mm; symmetric sensitivity allowance ±{RAIL_SIGHT_OFFSET_SENSITIVITY_MM:.4f} mm (published specification text shows +3.175 mm, not a symmetric tolerance).",
        f"- Checkerboard-derived rail-to-cloth separation: {fitted_height:.3f} mm.",
        f"- Two-spot joint reprojection RMS: {report['rail_cap_to_cloth_transfer']['joint_fit_residual_stats_px']['rms']:.3f} px; max {report['rail_cap_to_cloth_transfer']['joint_fit_residual_stats_px']['max']:.3f} px.",
        f"- Applying the rail-plane homography directly at those cloth XY positions misses the spots by {', '.join(f'{x:.2f}' for x in rail_plane_spot_errors)} px (RMS {report['rail_cap_to_cloth_transfer']['naive_single_rail_plane_residual_stats_px']['rms']:.3f} px).",
        "- Single-spot-to-other-spot checks:",
    ]
    for entry in height_from_each:
        report_lines.append(
            f"  - Marker {entry['fit_marker_index'] + 1} estimates h={entry['estimated_rail_to_cloth_plane_separation_mm']:.3f} mm; predicts marker {entry['heldout_marker_index'] + 1} with {entry['heldout_residual_px']:.3f} px residual."
        )
    report_lines += ["", "## Rail-grid fit and independent checks", ""]
    for image_id in MARK_IMAGE_IDS:
        scored = frame_results[image_id]["rail_sight_fit"]
        report_lines.append(
            f"- `{image_id}`: {frame_results[image_id]['fit_points']} long-rail marks; fit RMS {scored['fit_error_px']['rms']:.3f} px; leave-one-sight-out RMS {scored['leave_one_sight_out_error_px']['rms']:.3f} px (max {scored['leave_one_sight_out_error_px']['max']:.3f} px)."
        )
        for rail, check in frame_results[image_id]["held_out_side_sight_checks"].items():
            residuals = [x["transfer_residual_px"] for x in check.get("assignments", [])]
            report_lines.append(
                f"  - Held-out {rail} rail marks: matched {len(residuals)} expected quarter positions; assigned residuals {', '.join(f'{x:.2f}' for x in residuals)} px; unassigned detection {check.get('surplus_unassigned_detection')}."
            )
    extra_bottom = frame_results["framing_before_720"]["additional_unseeded_bottom_sight_check"]
    report_lines.append(
        f"- An additional bottom-rail dot visible under glare was not in the seeded annotation set: the long-rail-only map predicts the 5/8 sight at ({extra_bottom['predicted_pixel_from_top_and_other_bottom_marks'][0]:.2f}, {extra_bottom['predicted_pixel_from_top_and_other_bottom_marks'][1]:.2f}) px; visual center estimate ({extra_bottom['approximate_visual_center_px'][0]:.1f}, {extra_bottom['approximate_visual_center_px'][1]:.1f}) px differs by {extra_bottom['approximate_visual_residual_px']:.2f} px. This is a qualitative confirmation, not a scored precise holdout."
    )
    report_lines += ["", "## Ball-reference diagnostic only", ""]
    report_lines.append(
        f"The nine saved measured-ball references were not used to fit any parameter. Mapping coarse manually annotated box midpoints to the fixed ball-center-height plane gives radial error RMS {report['held_out_nine_ball_reference_diagnostics']['radial_error_summary_mm']['rms']:.3f} mm, p95 {report['held_out_nine_ball_reference_diagnostics']['radial_error_summary_mm']['p95']:.3f} mm, max {report['held_out_nine_ball_reference_diagnostics']['radial_error_summary_mm']['max']:.3f} mm. This is not an accuracy result: the physical reference uncertainties are unknown and the box midpoint is not a validated sphere center. See the per-ball rows and intrinsics sensitivity in `summary.json`."
    )
    report_lines += [
        "",
        "## Interpretation and next step",
        "",
        "The user can avoid measuring round balls: rail sights and marked head/foot spots can establish a candidate grid if this table follows the standard. The minimum next calibration capture is a larger rigid-backed checkerboard at the unchanged camera zoom/focus/resolution, moved through sharp views spanning the image center and corners with tilts about both axes. The saved rail/cloth-mark framing already supplies the plane bridge; use it as the held-out table-specific check after recalibrating intrinsics. Do not ship the map until the calibration model is stable across held-out views and the observed sight lattice/cloth-spot identities agree.",
        "",
        "A full-table print is not required. A target roughly twice the present linear size (for example, a 20 × 14 in pattern with 2 in squares) would improve corner precision while remaining portable; several deliberate poses across center/corners are more useful than another small target view at the image center. No ball jig or round-ball measurements are needed.",
        "",
        "## Source references",
        "",
        "- WPA Rules of Play (effective 2025-09-15), §2.1 defines sights at 1/8 of length and 1/4 of width and identifies head/foot spots at string intersections: https://wpapool.com/wp-content/uploads/2025/10/2025.09.15-WPA-Rules-NP.pdf",
        "- WPA Recommended Equipment Specifications, §6 gives the 8-foot 11.5-inch sight spacing and a 3 11/16-inch sight-to-cushion-nose offset with +3.175 mm in the published text; the specification says it may not apply to home-use tables: https://www.wpapool.com/wp-content/uploads/2024/01/RECOMMENDED-EQUIPMENT-SPECIFICATIONS.pdf",
        "",
        "## Reproduction",
        "",
        "Run `offline-mark-grid-calibration-audit-20261027.py`. It writes `summary.json` and this report. Synthetic homography recovery and fixed-plane ray round-trip checks must pass.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(report_lines))
    print(json.dumps({"summary": str(output_json), "report": str(args.output / "README.md"), "h_mm": fitted_height, "cloth_errors_px": cloth_errors.tolist(), "ball_rms_mm": report["held_out_nine_ball_reference_diagnostics"]["radial_error_summary_mm"]["rms"]}, indent=2))


if __name__ == "__main__":
    main()
