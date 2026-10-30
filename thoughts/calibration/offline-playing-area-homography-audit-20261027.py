#!/usr/bin/env python3
"""Report-only metric mapping diagnostic from saved cushion boundary traces.

The source edge transitions have not been surveyed against the physical
cushion noses. Every smoothing/band variant is therefore retained as a
candidate. Ball references are used only after the candidate table mapping is
fixed, for held-out diagnostics; they never choose an edge or fit a homography.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "thoughts" / "calibration"
OUT = CAL / "playing-area-homography-audit-20261027"
EDGE_FILE = CAL / "cushion-straightness.json"
EDGE_EXTRACTOR_FILE = CAL / "analyze_cushion_straightness.py"
MEASUREMENTS_FILE = CAL / "table-grid-20261026-114241-measurements.json"
SEGMENTATION_FILE = CAL / "table-grid-feasibility-results.json"
IMAGE_FILE = CAL / "table-grid-20261026-114241.jpg"
INTRINSICS_FILE = CAL / "checkerboard-eight-view-baseline.json"
WIDTH_MM = 2336.8
HEIGHT_MM = 1155.7
BALL_RADIUS_MM = 28.575
CORNER_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
WORLD_CORNERS = np.array(
    [[0.0, 0.0], [WIDTH_MM, 0.0], [WIDTH_MM, HEIGHT_MM], [0.0, HEIGHT_MM]],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalize_line(coefficients: np.ndarray) -> np.ndarray:
    line = np.asarray(coefficients, dtype=np.float64)
    norm = float(np.linalg.norm(line[:2]))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("degenerate image line")
    line /= norm
    if line[0] < -1e-12 or (abs(line[0]) <= 1e-12 and line[1] < 0):
        line = -line
    return line


def robust_line(points: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit one deterministic Huber-weighted total-least-squares line."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("line fitting needs at least four 2D samples")
    weights = np.ones(len(points), dtype=np.float64)
    line: np.ndarray | None = None
    for _ in range(30):
        center = np.average(points, axis=0, weights=weights)
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / weights.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.array([-direction[1], direction[0]])
        line = normalize_line(np.array([normal[0], normal[1], -normal @ center]))
        residuals = np.abs(points @ line[:2] + line[2])
        median = float(np.median(residuals))
        scale = max(1.4826 * float(np.median(np.abs(residuals - median))), 0.03)
        cutoff = 1.5 * scale
        new_weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-12))
        if np.max(np.abs(new_weights - weights)) < 1e-7:
            weights = new_weights
            break
        weights = new_weights
    assert line is not None
    residuals = points @ line[:2] + line[2]
    t = points @ np.array([-line[1], line[0]])
    return line, {
        "sample_count": int(len(points)),
        "rms_px": float(np.sqrt(np.mean(residuals**2))),
        "weighted_rms_px": float(np.sqrt(np.average(residuals**2, weights=weights))),
        "p95_abs_px": float(np.percentile(np.abs(residuals), 95)),
        "max_abs_px": float(np.max(np.abs(residuals))),
        "span_px": float(np.ptp(t)),
    }


def intersect_lines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.vstack([first[:2], second[:2]])
    det = float(np.linalg.det(a))
    if abs(det) < 1e-7:
        raise ValueError("adjacent boundary lines are nearly parallel")
    return np.linalg.solve(a, -np.array([first[2], second[2]]))


def quadrilateral_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def image_to_surface_homography(corners_px: np.ndarray) -> np.ndarray:
    h = cv2.getPerspectiveTransform(
        np.asarray(corners_px, dtype=np.float32), WORLD_CORNERS.astype(np.float32)
    ).astype(np.float64)
    if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-12:
        raise ValueError("invalid image-to-table homography")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (h @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-10):
        raise ValueError("point maps to infinity")
    result = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("homography produced nonfinite table coordinates")
    return result


def camera_center_plane_homography(
    image_to_surface: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shift the plane by one common known radius toward the camera.

    This is an internal fixed-plane geometric correction, not a per-ball
    height/depth estimate. Only X/Y coordinates are returned to consumers.
    """
    surface_to_image = np.linalg.inv(image_to_surface)
    normalized = np.linalg.inv(k) @ surface_to_image
    h1, h2, h3 = normalized[:, 0], normalized[:, 1], normalized[:, 2]
    n1, n2 = float(np.linalg.norm(h1)), float(np.linalg.norm(h2))
    if min(n1, n2) <= 1e-12:
        raise ValueError("homography decomposition has a zero basis vector")
    scale = 2.0 / (n1 + n2)
    r1, r2 = scale * h1, scale * h2
    r3 = np.cross(r1, r2)
    rotation_raw = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(rotation_raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = scale * h3
    camera_center_world = -rotation.T @ translation
    if abs(float(camera_center_world[2])) <= 1e-6:
        raise ValueError("decomposed camera lies too close to the table plane")
    toward_camera_sign = 1.0 if camera_center_world[2] > 0 else -1.0
    center_offset_world_z = toward_camera_sign * BALL_RADIUS_MM
    center_plane_to_image = k @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation + center_offset_world_z * rotation[:, 2]]
    )
    if abs(center_plane_to_image[2, 2]) < 1e-12:
        raise ValueError("invalid center-plane homography")
    center_plane_to_image /= center_plane_to_image[2, 2]
    image_to_center_plane = np.linalg.inv(center_plane_to_image)
    image_to_center_plane /= image_to_center_plane[2, 2]
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    reconstructed_surface = k @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )
    reconstructed_surface /= reconstructed_surface[2, 2]
    surface_reconstruction_error = float(
        np.linalg.norm(reconstructed_surface - surface_to_image, ord="fro")
    )
    expected_corners = apply_homography(surface_to_image, WORLD_CORNERS)
    reconstructed_corners = apply_homography(reconstructed_surface, WORLD_CORNERS)
    corner_errors = np.linalg.norm(reconstructed_corners - expected_corners, axis=1)
    diagnostics = {
        "surface_homography_reconstruction_frobenius_error": surface_reconstruction_error,
        "surface_homography_reconstruction_corner_rms_px": float(np.sqrt(np.mean(corner_errors**2))),
        "surface_homography_reconstruction_corner_max_px": float(np.max(corner_errors)),
        "rotation_orthogonality_frobenius_error": orthogonality_error,
        "center_plane_table_output_dimensions": ["X", "Y"],
        "per_ball_height_or_depth_estimated": False,
        "fixed_radius_offset_magnitude_mm_applied_uniformly": BALL_RADIUS_MM,
    }
    return image_to_center_plane, diagnostics


def homography_round_trip_test() -> dict[str, Any]:
    k = np.array([[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    angle_y = math.radians(7.0)
    angle_x = math.radians(-4.0)
    ry = np.array(
        [[math.cos(angle_y), 0.0, math.sin(angle_y)], [0.0, 1.0, 0.0], [-math.sin(angle_y), 0.0, math.cos(angle_y)]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(angle_x), -math.sin(angle_x)], [0.0, math.sin(angle_x), math.cos(angle_x)]]
    )
    rotation = ry @ rx
    translation = np.array([-1150.0, -570.0, 2100.0])
    surface_to_image = k @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )
    surface_to_image /= surface_to_image[2, 2]
    image_to_surface = np.linalg.inv(surface_to_image)
    surface_corners_image = apply_homography(surface_to_image, WORLD_CORNERS)
    surface_mapping = image_to_surface_homography(surface_corners_image)
    probe = np.array([[50.0, 40.0], [WIDTH_MM * 0.5, HEIGHT_MM * 0.5], [WIDTH_MM - 35.0, HEIGHT_MM - 45.0]])
    recovered_surface = apply_homography(surface_mapping, apply_homography(surface_to_image, probe))
    surface_error = float(np.max(np.linalg.norm(recovered_surface - probe, axis=1)))
    center_mapping, _ = camera_center_plane_homography(surface_mapping, k)
    center_world = probe.copy()
    center_world_z = -BALL_RADIUS_MM
    center_to_image = k @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation + center_world_z * rotation[:, 2]]
    )
    center_pixels = apply_homography(center_to_image / center_to_image[2, 2], center_world)
    recovered_center_xy = apply_homography(center_mapping, center_pixels)
    center_error = float(np.max(np.linalg.norm(recovered_center_xy - probe, axis=1)))
    if surface_error > 1e-4 or center_error > 1e-3:
        raise AssertionError(f"synthetic plane round trip failed: surface={surface_error}, center={center_error}")
    return {
        "status": "passed",
        "known_rectangle_round_trip_max_error_mm": surface_error,
        "fixed_center_plane_round_trip_max_error_mm": center_error,
        "table_output_dimensions": ["X", "Y"],
    }


def feature_index(edge_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene = next(
        item for item in edge_data["images"]
        if Path(item["source"]).name == IMAGE_FILE.name
    )
    return {feature["id"]: feature for feature in scene["features"]}


def variant_map(feature: dict[str, Any]) -> dict[tuple[float, int], np.ndarray]:
    output: dict[tuple[float, int], np.ndarray] = {}
    for item in feature["variants"]:
        if "points_raw_px" not in item:
            continue
        key = (float(item["sigma"]), int(item["band_shift"]))
        output[key] = np.asarray(item["points_raw_px"], dtype=np.float64)
    return output


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": float(min(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(max(values)),
    }


def summarize_xy_spread(stack: np.ndarray, reference_xy: np.ndarray) -> dict[str, Any]:
    # Shape: mapping candidate x ball x XY.
    output: dict[str, Any] = {"mapping_count": int(stack.shape[0]), "points": {}}
    per_point_max_radius = []
    all_reference_errors = []
    for index, ref in enumerate(reference_xy):
        sample = stack[:, index, :]
        median = np.median(sample, axis=0)
        deviations = np.linalg.norm(sample - median, axis=1)
        radii = np.linalg.norm(sample - ref, axis=1)
        per_point_max_radius.append(float(np.max(deviations)))
        all_reference_errors.extend(radii.tolist())
        output["points"][str(index)] = {
            "median_xy_mm": median.tolist(),
            "x_range_mm": [float(np.min(sample[:, 0])), float(np.max(sample[:, 0]))],
            "y_range_mm": [float(np.min(sample[:, 1])), float(np.max(sample[:, 1]))],
            "maximum_distance_from_median_xy_mm": float(np.max(deviations)),
            "reference_error_quantiles_mm": {
                "median": float(np.median(radii)),
                "p95": float(np.percentile(radii, 95)),
                "maximum": float(np.max(radii)),
            },
        }
    output["maximum_per_point_coordinate_spread_mm"] = float(max(per_point_max_radius))
    output["aggregate_reference_error_mm"] = summarize(all_reference_errors)
    return output


def build_report() -> dict[str, Any]:
    edge_data = load_json(EDGE_FILE)
    measurements = load_json(MEASUREMENTS_FILE)
    segmentation = load_json(SEGMENTATION_FILE)
    baseline = load_json(INTRINSICS_FILE)
    image = cv2.imread(str(IMAGE_FILE), cv2.IMREAD_COLOR)
    if image is None or image.shape[1::-1] != (1280, 720):
        raise ValueError("the measured saved scene must be 1280 x 720")
    k = np.asarray(baseline["models"]["zero"]["full_fit"]["camera_matrix"], dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError("frozen camera intrinsic matrix must be 3 x 3")
    features = feature_index(edge_data)
    group_specs = {
        "top": ("top-left", "top-right"),
        "bottom_nose": ("bottom-left-nose", "bottom-right-nose"),
        "bottom_seam_alternative": ("bottom-left-seam", "bottom-right-seam"),
        "left": ("left-nose",),
        "right": ("right-nose",),
    }
    variants = {name: {fid: variant_map(features[fid]) for fid in ids} for name, ids in group_specs.items()}
    variant_keys = sorted(variants["top"]["top-left"].keys())
    if len(variant_keys) != 9:
        raise ValueError(f"expected nine extracted line variants; found {len(variant_keys)}")

    line_records: dict[str, list[dict[str, Any]]] = {}
    line_coefficients: dict[str, dict[tuple[float, int], np.ndarray]] = {}
    line_stats: dict[str, dict[tuple[float, int], dict[str, float | int]]] = {}
    for group, feature_ids in group_specs.items():
        line_records[group] = []
        line_coefficients[group] = {}
        line_stats[group] = {}
        for key in variant_keys:
            points = np.vstack([variants[group][fid][key] for fid in feature_ids])
            line, metrics = robust_line(points)
            line_coefficients[group][key] = line
            line_stats[group][key] = metrics
            line_records[group].append(
                {
                    "variant_id": f"sigma_{key[0]:.1f}_band_{key[1]:+d}",
                    "sigma": key[0],
                    "band_shift_px": key[1],
                    "feature_ids": list(feature_ids),
                    "line_ax_by_c_eq_0": line.tolist(),
                    **metrics,
                }
            )

    segmentation_centers = np.asarray(
        [item["center_px"] for item in segmentation["observations"]], dtype=np.float64
    )
    reference_xy = np.asarray(
        [[point["center_from_upper_left_mm"]["x"], point["center_from_upper_left_mm"]["y"]] for point in measurements["points"]],
        dtype=np.float64,
    )
    if len(segmentation_centers) != 9 or len(reference_xy) != 9:
        raise ValueError("expected exactly nine segmentation centers and physical references")
    center_ids = [point["id"] for point in measurements["points"]]
    baseline_loo = segmentation["results"]["homography"]["leave_one_position_out_error_mm"]
    for observation, point in zip(segmentation["observations"], measurements["points"]):
        if observation["id"] != point["id"]:
            raise ValueError("segmentation-center and physical-reference point order does not match")

    def map_one(keys: dict[str, tuple[float, int]], bottom_group: str) -> dict[str, Any]:
        top_line = line_coefficients["top"][keys["top"]]
        bottom_line = line_coefficients[bottom_group][keys["bottom"]]
        left_line = line_coefficients["left"][keys["left"]]
        right_line = line_coefficients["right"][keys["right"]]
        corners = np.asarray(
            [
                intersect_lines(top_line, left_line),
                intersect_lines(top_line, right_line),
                intersect_lines(bottom_line, right_line),
                intersect_lines(bottom_line, left_line),
            ],
            dtype=np.float64,
        )
        area = quadrilateral_area(corners)
        if area < 10000 or not np.isfinite(corners).all():
            raise ValueError("candidate rectangle is degenerate")
        surface_h = image_to_surface_homography(corners)
        center_h, camera_diagnostics = camera_center_plane_homography(surface_h, k)
        mapped = apply_homography(center_h, segmentation_centers)
        errors = mapped - reference_xy
        return {
            "corners_px": corners,
            "area_px2": area,
            "surface_h": surface_h,
            "center_h": center_h,
            "camera_diagnostics": camera_diagnostics,
            "mapped_xy_mm": mapped,
            "error_vectors_xy_mm": errors,
            "errors_xy_mm": np.linalg.norm(errors, axis=1),
        }

    nominal_key = (1.0, 0)
    nominal_keys = {"top": nominal_key, "bottom": nominal_key, "left": nominal_key, "right": nominal_key}
    nominal = map_one(nominal_keys, "bottom_nose")
    nominal_lines = {
        name: line_records[name][variant_keys.index(nominal_key)]
        for name in ["top", "bottom_nose", "left", "right"]
    }
    nominal_points = []
    for index, ball_id in enumerate(center_ids):
        nominal_points.append(
            {
                "point_id": ball_id,
                "segmentation_center_px": segmentation_centers[index].tolist(),
                "mapped_xy_mm_user_convention": nominal["mapped_xy_mm"][index].tolist(),
                "reference_xy_mm": reference_xy[index].tolist(),
                "error_vector_xy_mm": nominal["error_vectors_xy_mm"][index].tolist(),
                "error_norm_xy_mm": float(nominal["errors_xy_mm"][index]),
            }
        )

    candidate_corner_mappings = []
    sensitivity_by_bottom: dict[str, Any] = {}
    total_candidates = 0
    for bottom_group in ["bottom_nose", "bottom_seam_alternative"]:
        mapped_results: list[np.ndarray] = []
        rms_results: list[float] = []
        valid_corner_rows: list[np.ndarray] = []
        failures = 0
        for top_key, bottom_key, left_key, right_key in itertools.product(
            variant_keys, variant_keys, variant_keys, variant_keys
        ):
            total_candidates += 1
            keys = {"top": top_key, "bottom": bottom_key, "left": left_key, "right": right_key}
            corner_record: dict[str, Any] = {
                "candidate_id": total_candidates - 1,
                "bottom_edge_family": bottom_group,
                "variant_ids": {
                    name: f"sigma_{key[0]:.1f}_band_{key[1]:+d}"
                    for name, key in keys.items()
                },
            }
            try:
                result = map_one(keys, bottom_group)
                corners = result["corners_px"]
                pred = result["mapped_xy_mm"]
                errors = result["errors_xy_mm"]
                rms = float(np.sqrt(np.mean(errors**2)))
                corner_record.update(
                    {
                        "status": "valid_report_only_mapping",
                        "image_corners_px": corners.tolist(),
                        "corner_polygon_area_px2": float(result["area_px2"]),
                        "reference_error_xy_mm_by_point": errors.tolist(),
                        "reference_rms_xy_mm_diagnostic_only": rms,
                    }
                )
                valid_corner_rows.append(corners)
                mapped_results.append(pred)
                rms_results.append(rms)
            except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                failures += 1
                corner_record.update({"status": "rejected_degenerate_mapping", "reason": str(error)})
            candidate_corner_mappings.append(corner_record)
        if not mapped_results:
            raise ValueError(f"no valid candidates for lower-edge family {bottom_group}")
        mapped_stack = np.stack(mapped_results)
        corner_stack = np.stack(valid_corner_rows)
        corner_median = np.median(corner_stack, axis=0)
        corner_spreads = np.linalg.norm(corner_stack - corner_median, axis=2)
        sensitivity_by_bottom[bottom_group] = {
            "attempted_mappings": len(variant_keys) ** 4,
            "valid_mappings": len(mapped_results),
            "rejected_degenerate_mappings": failures,
            "corner_median_px": corner_median.tolist(),
            "corner_max_distance_from_median_px": np.max(corner_spreads, axis=0).tolist(),
            "ball_xy_mapping_sensitivity": summarize_xy_spread(mapped_stack, reference_xy),
            "reference_rms_xy_mm_diagnostic_distribution": summarize(rms_results),
            "line_residual_ranges_px": {
                name: {
                    metric: [
                        float(min(record[metric] for record in line_records[name])),
                        float(max(record[metric] for record in line_records[name])),
                    ]
                    for metric in ["rms_px", "p95_abs_px", "max_abs_px", "span_px"]
                }
                for name in ["top", bottom_group, "left", "right"]
            },
        }

    source_paths = [
        EDGE_FILE,
        EDGE_EXTRACTOR_FILE,
        MEASUREMENTS_FILE,
        SEGMENTATION_FILE,
        IMAGE_FILE,
        INTRINSICS_FILE,
        Path(__file__),
    ]
    nominal_rms = float(np.sqrt(np.mean(nominal["errors_xy_mm"] ** 2)))
    all_nominal_errors = nominal["errors_xy_mm"].tolist()
    report = {
        "schema_version": 1,
        "purpose": "report-only X/Y mapping from saved playing-area boundary lines; no production activation",
        "production_behavior_changed": False,
        "metric_table_asset_created": False,
        "coordinate_contract": {
            "output_dimensions": ["X", "Y"],
            "origin": "upper-left cushion-nose intersection",
            "positive_x": "image-right",
            "positive_y": "image-down",
            "playing_area_width_mm": WIDTH_MM,
            "playing_area_height_mm": HEIGHT_MM,
            "ball_diameter_mm": BALL_RADIUS_MM * 2,
            "fixed_ball_center_plane_offset_used_for_projection_mm": BALL_RADIUS_MM,
            "per_ball_height_or_depth_estimated": False,
            "z_or_depth_output": False,
        },
        "reference_scope": {
            "nine_ball_xy_used_to_fit_or_select_edges": False,
            "nine_ball_xy_used_only_after_mapping_was_fixed": True,
            "measurement_uncertainty_mm": None,
            "ball_center_image_points": "existing segmentation ellipse centers; not independent pixel ground truth",
            "no_dot_spacing_assumed": True,
        },
        "edge_identity_and_physical_assignment": {
            "top": "nose-associated brightness transition; not physically surveyed",
            "bottom_nose": "weak inner cushion transition; adjacent outer seam retained as a separate alternative",
            "bottom_seam_alternative": "outer dark seam, explicitly not assumed to be the nose",
            "left": "inner shadow-to-cloth transition; may not equal the physical nose",
            "right": "cloth-to-inner-shadow transition; known ambiguous/jumpy extraction",
            "physical_nose_identified_independently": False,
            "mapping_is_physical_calibration": False,
        },
        "camera_model": {
            "intrinsics_model": "frozen eight-view checkerboard zero-radial full-fit matrix",
            "intrinsics_source": str(INTRINSICS_FILE.relative_to(ROOT)),
            "camera_matrix": k.tolist(),
            "distortion_applied": False,
            "intrinsics_weak_identification_limit": "See source checkerboard report; no uncertainty propagated here.",
        },
        "line_candidate_sets": line_records,
        "nominal_candidate": {
            "variant_selection_rule": "pre-existing nominal sigma=1.0, band_shift=0; not selected using ball-reference errors",
            "bottom_edge_family": "bottom_nose",
            "line_fits": nominal_lines,
            "image_corners_px": nominal["corners_px"].tolist(),
            "image_to_cloth_surface_homography_px_to_mm": nominal["surface_h"].tolist(),
            "image_to_fixed_ball_center_plane_homography_px_to_mm": nominal["center_h"].tolist(),
            "fixed_radius_projection_diagnostics": nominal["camera_diagnostics"],
            "segmentation_center_ball_diagnostics": nominal_points,
            "reference_rms_xy_mm_diagnostic_only": nominal_rms,
            "reference_error_distribution_xy_mm": summarize(all_nominal_errors),
        },
        "existing_ball_center_homography_comparison": {
            "source": str(SEGMENTATION_FILE.relative_to(ROOT)),
            "method": "segmentation-derived centers fitted jointly to the same nine user ball references",
            "training_rms_mm": segmentation["results"]["homography"]["training_rms_mm"],
            "leave_one_out_rms_mm": segmentation["results"]["homography"]["leave_one_position_out_rms_mm"],
            "leave_one_out_errors_mm": baseline_loo,
            "is_comparable_as_independent_accuracy": False,
        },
        "edge_choice_and_corner_extrapolation_sensitivity": sensitivity_by_bottom,
        "all_candidate_corner_mappings": candidate_corner_mappings,
        "candidate_mapping_count": total_candidates,
        "limitations": [
            "The extracted transitions were described in the source as nose-associated or weak inner boundaries, not independently surveyed physical cushion noses.",
            "Right-nose points contain large local jumps and edge-band sensitivity; all nine source extraction variants are retained rather than choosing one by ball residual.",
            "The bottom outer seam is a nearby but distinct transition; it is included only as a sensitivity alternative, not promoted as the physical boundary.",
            "Corner positions are extrapolated beyond the sampled rail segments and are especially sensitive to which transition is treated as the edge.",
            "The center-plane correction uses one fixed 28.575 mm offset for all balls and frozen intrinsics; it does not estimate per-ball height/depth and is still vulnerable to incorrect boundary identification and intrinsic uncertainty.",
            "Segmentation centers are not independently labeled physical sphere centers; sphere silhouette-center bias is not corrected by this rectangle mapping.",
            "The nine physical references have unknown uncertainty and are used only for evaluation after the homography candidates are fixed.",
            "Low held-out residuals against these nine user references would not establish independent physical accuracy, live availability, or corner/pocket accuracy.",
        ],
        "input_manifest": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in source_paths
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "synthetic_self_test": homography_round_trip_test(),
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "summary.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    nominal = report["nominal_candidate"]
    sensitivity = report["edge_choice_and_corner_extrapolation_sensitivity"]
    lines = [
        "# Playing-area rectangle homography audit — 2026-10-27",
        "",
        "Offline saved-image diagnostic only. No new physical measurement, jig, or camera capture was used. No production mapping was changed.",
        "",
        "## Method and edge identity",
        "",
        "The four image boundaries are Huber-fit to all existing cushion-straightness samples from the measured 1280×720 image. Top, bottom, left, and right each have nine smoothing/search-band variants; both the weak inner bottom transition and the nearby outer seam are retained. The physical nose is not independently identified by these brightness transitions. The nine ball references never fit or select a boundary.",
        "",
        "A cloth-surface mapping is formed from the line intersections and the verified 2336.8 × 1155.7 mm rectangle. The saved segmentation centers correspond to balls with one known common radius. For the ball-center diagnostic, the same fixed radius offset is applied uniformly using the frozen checkerboard K; no individual height or depth is estimated, and only X/Y is reported.",
        "",
        "## Nominal diagnostic",
        "",
        f"The pre-existing nominal edge extraction (σ=1.0, band shift 0) gives X/Y RMS **{nominal['reference_rms_xy_mm_diagnostic_only']:.3f} mm** against the nine physical references, with median **{nominal['reference_error_distribution_xy_mm']['median']:.3f} mm** and maximum **{nominal['reference_error_distribution_xy_mm']['maximum']:.3f} mm**. These references have unknown uncertainty. This is a report-only comparison; it does not establish the physical nose or accuracy.",
        "",
        "| Boundary line | Samples | Line RMS (px) | 95th percentile (px) | Maximum (px) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, fit in nominal["line_fits"].items():
        lines.append(
            f"| {name} | {fit['sample_count']} | {fit['rms_px']:.3f} | {fit['p95_abs_px']:.3f} | {fit['max_abs_px']:.3f} |"
        )
    diagnostics = nominal["fixed_radius_projection_diagnostics"]
    lines += [
        "",
        f"The nominal four-corner homography and frozen checkerboard intrinsics do not reproduce one another exactly: the fixed-radius pose decomposition misses the four extracted corners by **{diagnostics['surface_homography_reconstruction_corner_rms_px']:.2f} px RMS** (max **{diagnostics['surface_homography_reconstruction_corner_max_px']:.2f} px**). This limits confidence in projecting from the cloth boundary to the common ball-center plane.",
        "",
        "| Ball | Mapped X (mm) | Mapped Y (mm) | Diagnostic residual (mm) |",
        "|---|---:|---:|---:|",
    ]
    for point in nominal["segmentation_center_ball_diagnostics"]:
        xy = point["mapped_xy_mm_user_convention"]
        lines.append(
            f"| {point['point_id']} | {xy[0]:.2f} | {xy[1]:.2f} | {point['error_norm_xy_mm']:.2f} |"
        )
    lines += [
        "",
        "The existing segmentation-center homography fitted to those same nine references has training RMS **3.893 mm** and leave-one-out RMS **8.706 mm**. The nominal rectangle map is independent of the ball references, but the physical edge is uncertain; the two numbers have different fitting protocols and neither is independent accuracy evidence.",
        "",
        "## Edge and corner sensitivity",
        "",
        "The audit evaluates all 6,561 combinations of the nine top, bottom, left, and right extraction variants for each lower-edge family (13,122 mappings total). Each candidate mapping, its four extrapolated image corners, and all nine reference-error magnitudes are retained in `summary.json`.",
        "",
        "| Lower edge choice | Valid mappings | Maximum corner variation from median (px), TL/TR/BR/BL | Maximum per-ball X/Y spread from median (mm) | Diagnostic RMS range (mm) |",
        "|---|---:|---|---:|---|",
    ]
    for edge, record in sensitivity.items():
        corner_values = record["corner_max_distance_from_median_px"]
        corner_string = ", ".join(f"{value:.2f}" for value in corner_values)
        ref = record["reference_rms_xy_mm_diagnostic_distribution"]
        lines.append(
            f"| {edge} | {record['valid_mappings']} / {record['attempted_mappings']} | {corner_string} | {record['ball_xy_mapping_sensitivity']['maximum_per_point_coordinate_spread_mm']:.2f} | {ref['minimum']:.2f}–{ref['maximum']:.2f} |"
        )
    lines += [
        "",
        "Do not choose the lowest-error edge family. The seam is an explicit alternative transition, not the intended cushion nose. The wide variation is evidence that the current images do not identify the physical boundary tightly enough for a reliable metric map.",
        "",
        "## Interpretation and limits",
        "",
        "This is a geometric sensitivity experiment, not a qualified table calibration. The sampled lines are brightness/shadow transitions with no surveyed labels; the right side is notably jumpy, the lower inner line is weak, and corner points must be extrapolated around pockets. The frozen-intrinsics decomposition's nominal surface-corner reconstruction error is reported in the JSON and is another check on the fixed-radius projection correction. The saved segmentation ellipse centers are not independently verified projections of physical sphere centers.",
        "",
        "The nine user-provided positions were applied only after all line candidates and mappings were computed. Their measurement uncertainty is unknown, so residuals cannot diagnose whether the measurements or the map are wrong. No metric asset, precise coordinates, or guidance behavior is activated.",
        "",
        "## Reproduction",
        "",
        "```sh",
        "python thoughts/calibration/offline-playing-area-homography-audit-20261027.py --self-test",
        "python thoughts/calibration/offline-playing-area-homography-audit-20261027.py",
        "```",
        "",
        "The runner requires NumPy and OpenCV Python; the saved calibration environment is `/tmp/billiards-checkerboard-env/bin/python`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(homography_round_trip_test(), indent=2, sort_keys=True))
        return
    report = build_report()
    write_report(report)
    nominal = report["nominal_candidate"]
    print(
        json.dumps(
            {
                "report": str((OUT / "README.md").relative_to(ROOT)),
                "candidate_mappings": report["candidate_mapping_count"],
                "nominal_reference_rms_xy_mm": nominal["reference_rms_xy_mm_diagnostic_only"],
                "sensitivity": {
                    name: {
                        "valid_mappings": item["valid_mappings"],
                        "max_per_ball_coordinate_spread_mm": item["ball_xy_mapping_sensitivity"]["maximum_per_point_coordinate_spread_mm"],
                        "reference_rms_range_mm": [item["reference_rms_xy_mm_diagnostic_distribution"].get("minimum"), item["reference_rms_xy_mm_diagnostic_distribution"].get("maximum")],
                    }
                    for name, item in report["edge_choice_and_corner_extrapolation_sensitivity"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
