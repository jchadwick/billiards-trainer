#!/usr/bin/env python3
"""Saved-data fixed-sphere geometry diagnostic; no production behavior changes."""

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
OUT = CAL / "sphere-projection-audit-20261027"
MEASUREMENTS = CAL / "table-grid-20261026-114241-measurements.json"
INTRINSICS_SOURCE = CAL / "checkerboard-frozen-validation-view-009.json"
IMAGE = CAL / "table-grid-20261026-114241.jpg"
IMAGE_CENTER_BASELINE = CAL / "table-grid-feasibility-results.json"
CASE_DIR = CAL / "contour-color-audit-20261027" / "cases"
RADIUS_MM = 28.575
MAX_SEEDS = 10000
MAX_ASSIGNMENT_STEPS = 20
NEAR_OPTIMAL_DIAGNOSTIC_MM = 1.0
ANCHOR_QUADRILATERAL_COUNT = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def conic_from_ellipse(
    center: np.ndarray, axes: np.ndarray, angle_degrees: float
) -> np.ndarray:
    """Return a homogeneous image conic for an ellipse fitted in source pixels."""
    a, b = axes / 2.0
    if not np.isfinite([*center, a, b, angle_degrees]).all() or min(a, b) <= 0:
        raise ValueError("ellipse dimensions or parameters are invalid")
    angle = math.radians(float(angle_degrees))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    q = rotation @ np.diag([1.0 / (a * a), 1.0 / (b * b)]) @ rotation.T
    linear = -q @ center
    c = float(center @ q @ center - 1.0)
    return np.block([[q, linear[:, None]], [linear[None, :], np.array([[c]])]])


def invert_sphere_conic(
    points_px: np.ndarray, k: np.ndarray, radius_mm: float = RADIUS_MM
) -> dict[str, Any]:
    """Fit an image ellipse and recover the tangent sphere center in camera space.

    For normalized camera rays, a sphere silhouette has conic
    C C^T - (||C||^2 - R^2) I, up to sign/scale. The isolated eigenvector gives
    the center direction; the eigenvalue ratio gives range. No per-ball height
    or depth is fitted: the known physical radius is constant.
    """
    points = np.asarray(points_px, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError("at least five 2D outline points are required")
    (cx, cy), (width, height), angle = cv2.fitEllipse(points.reshape(-1, 1, 2))
    center = np.array([cx, cy], dtype=float)
    axes = np.array([width, height], dtype=float)
    image_conic = conic_from_ellipse(center, axes, float(angle))
    calibrated_conic = k.T @ image_conic @ k
    eigenvalues, eigenvectors = np.linalg.eigh(calibrated_conic)

    possible: list[tuple[float, int, list[int]]] = []
    for isolated in range(3):
        repeated = [index for index in range(3) if index != isolated]
        if eigenvalues[isolated] * eigenvalues[repeated[0]] >= 0:
            continue
        if eigenvalues[isolated] * eigenvalues[repeated[1]] >= 0:
            continue
        repeated_magnitudes = np.abs(eigenvalues[repeated])
        mean_magnitude = float(np.mean(repeated_magnitudes))
        if mean_magnitude <= 0:
            continue
        mismatch = float(
            abs(repeated_magnitudes[0] - repeated_magnitudes[1]) / mean_magnitude
        )
        possible.append((mismatch, isolated, repeated))
    if not possible:
        raise ValueError("calibrated conic does not have a sphere tangent-cone sign pattern")
    mismatch, isolated, repeated = min(possible, key=lambda value: (value[0], value[1]))
    isolated_magnitude = abs(float(eigenvalues[isolated]))
    repeated_mean = float(np.mean(np.abs(eigenvalues[repeated])))
    if isolated_magnitude <= np.finfo(float).eps * repeated_mean:
        raise ValueError("sphere range is numerically unbounded")

    distance = radius_mm * math.sqrt(1.0 + repeated_mean / isolated_magnitude)
    direction = eigenvectors[:, isolated]
    if direction[2] < 0:
        direction = -direction
    if direction[2] <= 0:
        raise ValueError("sphere center ray points behind the camera")
    sphere_center_camera_mm = distance * direction

    # Evaluate how closely the sampled outline follows the fitted ellipse.
    angle_rad = math.radians(float(angle))
    rotation = np.array(
        [[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]
    )
    local = (points.astype(float) - center) @ rotation
    radial = np.sqrt((local[:, 0] / (axes[0] / 2.0)) ** 2 + (local[:, 1] / (axes[1] / 2.0)) ** 2)
    ellipse_residual_px = np.abs(radial - 1.0) * float(np.mean(axes) / 2.0)
    return {
        "ellipse_center_px": center.tolist(),
        "ellipse_axes_px": axes.tolist(),
        "ellipse_angle_degrees": float(angle),
        "ellipse_radial_residual_rms_px": float(np.sqrt(np.mean(ellipse_residual_px**2))),
        "calibrated_conic_eigenvalues": eigenvalues.tolist(),
        "isolated_eigenvalue_index": int(isolated),
        "repeated_eigenvalue_indices": repeated,
        "repeated_eigenvalue_relative_mismatch": mismatch,
        "sphere_center_camera_mm": sphere_center_camera_mm.tolist(),
        "sphere_center_range_mm": float(distance),
        "direction_unit": direction.tolist(),
    }


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Find proper rigid transform target ~= rotation @ source + translation."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("rigid alignment requires matching N by 3 point arrays")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ sign @ u.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation, singular_values.tolist()


def pose_errors(
    rotation: np.ndarray, translation: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    return np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)


def convex_hull_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def fit_homography_normalized(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match the saved 9-point diagnostic's normalized linear H fit."""
    x, y = source.T
    u, v = target.T
    zero = np.zeros(len(source))
    one = np.ones(len(source))
    design = np.vstack(
        (
            np.column_stack((x, y, one, zero, zero, zero, -u * x, -u * y)),
            np.column_stack((zero, zero, zero, x, y, one, -v * x, -v * y)),
        )
    )
    coefficients = np.linalg.lstsq(design, np.r_[u, v], rcond=None)[0]
    return np.r_[coefficients, 1.0].reshape(3, 3)


def project_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def leave_one_out_image_center_homography(
    points: list[dict[str, Any]], target_xyz: np.ndarray
) -> dict[str, Any]:
    """Reproduce the existing segmentation-ellipse 2D homography diagnostic."""
    baseline = load_json(IMAGE_CENTER_BASELINE)
    observations = {entry["id"]: entry for entry in baseline["observations"]}
    image_xy = np.asarray([observations[point["id"]]["center_px"] for point in points])
    user_xy = np.asarray(
        [[point["center_from_upper_left_mm"][axis] for axis in ("x", "y")] for point in points]
    )
    source_normalized = (image_xy - np.array([640.0, 360.0])) / 640.0
    target_normalized = (user_xy - np.array([1168.4, 577.85])) / 1168.4
    errors = []
    for heldout in range(len(points)):
        train = [index for index in range(len(points)) if index != heldout]
        homography = fit_homography_normalized(
            source_normalized[train], target_normalized[train]
        )
        prediction = (
            project_homography(source_normalized[[heldout]], homography)[0] * 1168.4
            + np.array([1168.4, 577.85])
        )
        errors.append(float(np.linalg.norm(prediction - user_xy[heldout])))
    return {
        "method": "existing saved segmentation-ellipse centers with the original 8-parameter normalized homography",
        "source": str(IMAGE_CENTER_BASELINE.relative_to(ROOT)),
        "fit_is_independent_physical_ground_truth": False,
        "training_indices_per_fold": 8,
        "leave_one_out_errors_mm": {
            point["id"]: errors[index] for index, point in enumerate(points)
        },
        "leave_one_out_mean_mm": float(np.mean(errors)),
        "leave_one_out_rms_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "leave_one_out_median_mm": float(np.median(errors)),
        "leave_one_out_max_mm": float(np.max(errors)),
        "reference_uncertainty_mm": None,
        "limitations": "This is the prior segmentation-derived center mapping diagnostic; image localization, measurement/contact uncertainty, shared image/measurement setup, and homography error are mixed. It is not independent XY accuracy.",
    }


def best_anchor_indices(indices: list[int], target_xyz: np.ndarray) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    best_area = -1.0
    for combination in itertools.combinations(indices, 4):
        area = convex_hull_area(target_xyz[list(combination), :2])
        if area > best_area:
            best_area = area
            best = combination
    if best is None or best_area < 1.0:
        raise ValueError("training reference points do not span a table region")
    return best


def top_anchor_quadrilaterals(
    indices: list[int], target_xyz: np.ndarray, count: int = ANCHOR_QUADRILATERAL_COUNT
) -> list[tuple[int, ...]]:
    scored = [
        (convex_hull_area(target_xyz[list(combo), :2]), combo)
        for combo in itertools.combinations(indices, 4)
    ]
    return [combo for area, combo in sorted(scored, key=lambda item: (-item[0], item[1]))[:count] if area >= 1.0]


def describe_solution(
    solution: dict[str, Any],
    candidates: list[list[dict[str, Any]]],
    target_xyz: np.ndarray,
    training_indices: list[int],
) -> dict[str, Any]:
    chosen = [candidates[i][solution["assignment"][i]] for i in training_indices]
    observed = np.array([x["sphere_center_camera_mm"] for x in chosen])
    predicted = target_xyz[training_indices]
    rotation, translation, singular_values = kabsch(observed, predicted)
    table_residuals = (rotation @ observed.T).T + translation - predicted
    xy_residuals = table_residuals[:, :2]
    xy_norms = np.linalg.norm(xy_residuals, axis=1)
    return {
        "assignment": solution["assignment"],
        "training_indices": training_indices,
        "fit_ids": [x["fit_id"] for x in chosen],
        "rotation_camera_to_table": rotation.tolist(),
        "translation_camera_to_table_mm": translation.tolist(),
        "alignment_singular_values": singular_values,
        "training_residuals_xy_mm": xy_residuals.tolist(),
        "training_residual_norms_xy_mm": xy_norms.tolist(),
        "training_predicted_xy_mm_internal": ((rotation @ observed.T).T + translation)[:, :2].tolist(),
        "training_rms_xy_mm": float(np.sqrt(np.mean(xy_norms**2))),
        "training_max_xy_mm": float(np.max(xy_norms)),
    }


def public_pose_fit(fit: dict[str, Any], point_ids: list[str]) -> dict[str, Any]:
    """Persist only 2D mapping outputs; keep camera depth internal to the solver."""
    def public_solution(solution: dict[str, Any]) -> dict[str, Any]:
        associations = [
            {"point_id": point_ids[index], "fit_id": fit_id}
            for index, fit_id in zip(fit["training_indices"], solution["fit_ids"])
        ]
        return {
            "associations": associations,
            "training_residuals_xy_mm": solution["training_residuals_xy_mm"],
            "training_residual_norms_xy_mm": solution["training_residual_norms_xy_mm"],
            "training_predicted_xy_mm_user_convention": [
                [float(prediction[0]), float(-prediction[1])]
                for prediction in solution["training_predicted_xy_mm_internal"]
            ],
            "training_rms_xy_mm": solution["training_rms_xy_mm"],
            "training_max_xy_mm": solution["training_max_xy_mm"],
        }

    all_solutions = [public_solution(solution) for solution in fit["all_distinct_solutions"]]
    best_public = public_solution(fit["best"])
    near = [
        solution
        for solution in all_solutions
        if solution["training_rms_xy_mm"] <= best_public["training_rms_xy_mm"] + NEAR_OPTIMAL_DIAGNOSTIC_MM
    ]
    return {
        "training_point_ids": [point_ids[index] for index in fit["training_indices"]],
        "anchor_quadrilaterals_point_ids": [
            [point_ids[index] for index in anchors] for anchors in fit["anchor_quadrilaterals"]
        ],
        "anchor_candidate_seed_count": fit["candidate_seed_count"],
        "seeds_examined": fit["seeds_examined"],
        "assignment_search_is_exhaustive_over_all_ball_combinations": False,
        "distinct_converged_assignments": len(all_solutions),
        "best_candidate_associations_and_xy_residuals": best_public,
        "all_distinct_converged_assignment_results": all_solutions,
        "assignment_count_within_1mm_xy_rms_of_best": len(near),
    }


def fit_candidate_pose(
    candidates: list[list[dict[str, Any]]],
    target_xyz: np.ndarray,
    training_indices: list[int],
    max_seeds: int = MAX_SEEDS,
) -> dict[str, Any]:
    """Bounded deterministic candidate assignment/rigid-pose alternating search."""
    anchor_sets = top_anchor_quadrilaterals(training_indices, target_xyz)
    if not anchor_sets:
        raise ValueError("training reference points do not span a table region")
    seed_count = sum(math.prod(len(candidates[index]) for index in anchors) for anchors in anchor_sets)
    if seed_count > max_seeds:
        raise ValueError(f"anchor candidate product {seed_count} exceeds bound {max_seeds}")

    solutions: dict[tuple[int, ...], dict[str, Any]] = {}
    seeds_examined = 0
    for anchors in anchor_sets:
        products = [range(len(candidates[index])) for index in anchors]
        for anchor_choice in itertools.product(*products):
            seeds_examined += 1
            assignment = [0] * len(candidates)
            for index, choice in zip(anchors, anchor_choice):
                assignment[index] = choice
            anchor_source = np.array(
                [candidates[index][assignment[index]]["sphere_center_camera_mm"] for index in anchors]
            )
            rotation, translation, _ = kabsch(anchor_source, target_xyz[list(anchors)])
            seen: set[tuple[int, ...]] = set()
            for _ in range(MAX_ASSIGNMENT_STEPS):
                key = tuple(assignment[i] for i in range(len(candidates)))
                if key in seen:
                    break
                seen.add(key)
                for index in training_indices:
                    points = np.array([x["sphere_center_camera_mm"] for x in candidates[index]])
                    residual = (rotation @ points.T).T + translation - target_xyz[index]
                    assignment[index] = int(np.argmin(np.linalg.norm(residual, axis=1)))

                chosen = [candidates[i][assignment[i]]["sphere_center_camera_mm"] for i in training_indices]
                observed = np.asarray(chosen)
                rotation, translation, _ = kabsch(observed, target_xyz[training_indices])
                errors = pose_errors(rotation, translation, observed, target_xyz[training_indices])
                key = tuple(assignment[i] for i in range(len(candidates)))
                score = float(np.sqrt(np.mean(errors**2)))
                previous = solutions.get(key)
                if previous is None or score < previous["score"]:
                    solutions[key] = {
                        "assignment": assignment.copy(),
                        "score": score,
                    }

    ranked = sorted(solutions.values(), key=lambda value: (value["score"], tuple(value["assignment"])))
    if not ranked:
        raise ValueError("bounded candidate search produced no pose")
    detailed = [describe_solution(x, candidates, target_xyz, training_indices) for x in ranked]
    best = detailed[0]
    near = [
        x
        for x in detailed
        if x["training_rms_xy_mm"] <= best["training_rms_xy_mm"] + NEAR_OPTIMAL_DIAGNOSTIC_MM
    ]
    return {
        "training_indices": training_indices,
        "anchors": list(anchor_sets[0]),
        "anchor_quadrilaterals": [list(x) for x in anchor_sets],
        "candidate_seed_count": seed_count,
        "seeds_examined": seeds_examined,
        "distinct_assignments_found": len(ranked),
        "best": best,
        "all_distinct_solutions": detailed,
        "near_best_within_1mm_xy_count": len(near),
    }


def projected_sphere_outline(
    center_camera_mm: np.ndarray,
    radius_mm: float,
    k: np.ndarray,
    count: int = 128,
    angle_start: float = 0.0,
    angle_span: float = 2.0 * math.pi,
) -> np.ndarray:
    """Generate exact tangent points on a sphere then project them to pixels."""
    center = np.asarray(center_camera_mm, dtype=float)
    distance = float(np.linalg.norm(center))
    direction = center / distance
    seed = np.eye(3)[int(np.argmin(np.abs(direction)))]
    u = np.cross(direction, seed)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    ring_center = (1.0 - radius_mm**2 / distance**2) * center
    ring_radius = radius_mm * math.sqrt(1.0 - radius_mm**2 / distance**2)
    angles = angle_start + np.linspace(0.0, angle_span, count, endpoint=False)
    tangent_points = np.array(
        [ring_center + ring_radius * (u * math.cos(a) + v * math.sin(a)) for a in angles]
    )
    projected = (k @ tangent_points.T).T
    return projected[:, :2] / projected[:, 2:3]


def self_test() -> dict[str, Any]:
    k = np.array([[696.173158, 0.0, 644.175562], [0.0, 695.361177, 349.671524], [0.0, 0.0, 1.0]])
    radius = RADIUS_MM
    rotation = np.diag([1.0, -1.0, -1.0])
    # Internal table frame is right-handed: +X right, +Y toward the top rail,
    # +Z above cloth. This puts the camera above and looking down at the plane.
    translation = np.array([1170.0, -575.0, 1728.575])
    targets = np.array(
        [[150.0, -150.0, radius], [2200.0, -200.0, radius], [180.0, -1000.0, radius], [2180.0, -1050.0, radius], [1170.0, -575.0, radius], [560.0, -40.0, radius]],
        dtype=float,
    )
    centers = (rotation.T @ (targets - translation).T).T
    exact_xy_errors: list[float] = []
    noisy_xy_errors: list[float] = []
    partial_xy_errors: list[float] = []
    for index, center in enumerate(centers):
        points = projected_sphere_outline(center, radius, k, count=128)
        recovered = invert_sphere_conic(points, k, radius)
        exact_center_error = float(np.linalg.norm(np.array(recovered["sphere_center_camera_mm"]) - center))
        exact_table = rotation @ np.asarray(recovered["sphere_center_camera_mm"]) + translation
        exact_xy_errors.append(float(np.linalg.norm(exact_table[:2] - targets[index, :2])))
        rng = np.random.default_rng(1000 + index)
        noisy = points + rng.normal(0.0, 0.15, size=points.shape)
        noisy_fit = invert_sphere_conic(noisy, k, radius)
        noisy_center_error = float(np.linalg.norm(np.array(noisy_fit["sphere_center_camera_mm"]) - center))
        noisy_table = rotation @ np.asarray(noisy_fit["sphere_center_camera_mm"]) + translation
        noisy_xy_errors.append(float(np.linalg.norm(noisy_table[:2] - targets[index, :2])))
        partial = projected_sphere_outline(
            center, radius, k, count=80, angle_span=math.radians(200.0)
        )
        partial_fit = invert_sphere_conic(partial, k, radius)
        partial_center_error = float(np.linalg.norm(np.array(partial_fit["sphere_center_camera_mm"]) - center))
        partial_table = rotation @ np.asarray(partial_fit["sphere_center_camera_mm"]) + translation
        partial_xy_errors.append(float(np.linalg.norm(partial_table[:2] - targets[index, :2])))
        if exact_center_error > 0.002 or noisy_center_error > 8.0 or partial_center_error > 5.0:
            raise AssertionError("internal tangent-cone geometry self-check exceeded its numeric bound")

    source_true: list[list[dict[str, Any]]] = []
    for index, (center, target) in enumerate(zip(centers, targets)):
        points = projected_sphere_outline(center, radius, k, count=128)
        true_candidate = {
            "fit_id": index * 10,
            **invert_sphere_conic(points, k, radius),
        }
        options = [true_candidate]
        if index == 1:
            displaced = points + np.array([1.0, -0.4])
            options.append({"fit_id": index * 10 + 1, **invert_sphere_conic(displaced, k, radius)})
        source_true.append(options)
    fit = fit_candidate_pose(source_true, targets, list(range(len(targets))))
    chosen = fit["best"]["fit_ids"]
    expected = [x[0]["fit_id"] for x in source_true]
    if max(exact_xy_errors) > 0.002:
        raise AssertionError(f"noiseless table X/Y inversion failed: {max(exact_xy_errors)} mm")
    if max(noisy_xy_errors) > 8.0:
        raise AssertionError(f"noisy table X/Y inversion too unstable: {max(noisy_xy_errors)} mm")
    if max(partial_xy_errors) > 5.0:
        raise AssertionError(f"partial-arc table X/Y inversion too unstable: {max(partial_xy_errors)} mm")
    if chosen != expected:
        raise AssertionError(f"competing-contour assignment failed: {chosen} != {expected}")
    if fit["best"]["training_rms_xy_mm"] > 0.002:
        raise AssertionError("rigid table X/Y transform recovery failed")
    return {
        "status": "passed",
        "physical_radius_mm": radius,
        "exact_edge_positions": len(targets),
        "max_noiseless_table_xy_error_mm": max(exact_xy_errors),
        "max_0_15px_noise_table_xy_error_mm": max(noisy_xy_errors),
        "max_200_degree_partial_arc_table_xy_error_mm": max(partial_xy_errors),
        "competing_outline_assignment": "true outline selected over the displaced 1.0 x -0.4 px alternative",
        "rigid_table_xy_transform_recovery": "passed",
        "outputs_are_xy_only": True,
    }


def load_scene() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    measurements = load_json(MEASUREMENTS)
    frozen = load_json(INTRINSICS_SOURCE)
    model = frozen["models"]["zero"]
    k = np.asarray(model["camera_matrix"], dtype=float)
    source_w = measurements.get("source_width_px", 1280)
    source_h = measurements.get("source_height_px", 720)
    if k.shape != (3, 3):
        raise ValueError("frozen camera matrix is not 3x3")
    target_xyz = []
    points = []
    for point in measurements["points"]:
        xy = point["center_from_upper_left_mm"]
        # User coordinates increase Y toward the bottom rail. The internal
        # 3D frame is right-handed with Y toward the top rail and Z above cloth.
        target_xyz.append([xy["x"], -xy["y"], RADIUS_MM])
        points.append(point)
    target_xyz_array = np.asarray(target_xyz, dtype=float)
    if len(points) != 9 or not np.allclose(target_xyz_array[:, 2], RADIUS_MM):
        raise ValueError("expected nine points and one fixed known center plane")

    cases: list[dict[str, Any]] = []
    candidates: list[list[dict[str, Any]]] = []
    for point in points:
        path = CASE_DIR / f"measured-grid-720--{point['id']}.json"
        case = load_json(path)
        if case["source_dimensions_px"] != [source_w, source_h]:
            source_w, source_h = case["source_dimensions_px"]
        if case["source_image_sha256"] != sha256(IMAGE):
            raise ValueError(f"case {path.name} refers to a different saved image")
        case_candidates = []
        for item in case["candidate_evaluations"]:
            entry: dict[str, Any] = {
                "fit_id": item["fit_id"],
                "iteration": item["iteration"],
                "is_qualified": item["is_qualified"],
                "rejection": item["rejection"],
                "support": item["support"],
                "source_residual_rms_px": item["source_residual_rms_px"],
                "quality": item["quality"],
                "outline_point_count": len(item.get("outline_image_px", [])),
                "sphere_geometry_status": "not_attempted_unqualified",
            }
            if item["is_qualified"] and len(item.get("outline_image_px", [])) >= 5:
                try:
                    geometry = invert_sphere_conic(
                        np.asarray(item["outline_image_px"], dtype=float), k
                    )
                    entry.update(geometry)
                    entry["sphere_geometry_status"] = "usable_candidate_diagnostic"
                    case_candidates.append(entry)
                except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                    entry["sphere_geometry_status"] = "conic_rejected"
                    entry["sphere_geometry_error"] = str(error)
            cases.append(
                {
                    "point_id": point["id"],
                    "fit_id": item["fit_id"],
                    "candidate": entry,
                }
            )
        candidates.append(case_candidates)
    return points, target_xyz_array, k, np.array([source_w, source_h]), [
        {"all_candidates": cases, "usable_candidates_by_point": candidates}
    ]


def build_report() -> dict[str, Any]:
    points, target_xyz, k, source_size, loaded = load_scene()
    case_records = loaded[0]["all_candidates"]
    candidates = loaded[0]["usable_candidates_by_point"]
    if any(not group for group in candidates):
        raise ValueError("at least one measured position has no usable qualified outline")

    full_indices = list(range(len(points)))
    full = fit_candidate_pose(candidates, target_xyz, full_indices)
    image_center_baseline = leave_one_out_image_center_homography(points, target_xyz)
    segmentation_centers = {
        entry["id"]: np.asarray(entry["center_px"], dtype=float)
        for entry in load_json(IMAGE_CENTER_BASELINE)["observations"]
    }
    selected_full_fit_diagnostics = []
    for index, point in enumerate(points):
        chosen_index = full["best"]["assignment"][index]
        selected = candidates[index][chosen_index]
        baseline_center = segmentation_centers[point["id"]]
        candidate_distances = [
            float(np.linalg.norm(np.asarray(candidate["ellipse_center_px"]) - baseline_center))
            for candidate in candidates[index]
        ]
        selected_full_fit_diagnostics.append(
            {
                "point_id": point["id"],
                "fit_id_selected_by_all_nine_reference_pose": selected["fit_id"],
                "selected_outline_center_px": selected["ellipse_center_px"],
                "fixed_segmentation_ellipse_center_prior_px": baseline_center.tolist(),
                "selected_outline_to_segmentation_center_distance_px": candidate_distances[chosen_index],
                "candidate_nearest_segmentation_center_fit_id": candidates[index][
                    int(np.argmin(candidate_distances))
                ]["fit_id"],
                "all_nine_pose_selected_candidate_is_nearest_to_saved_segmentation_center": (
                    candidate_distances[chosen_index] <= min(candidate_distances) + 1e-9
                ),
                "segmentation_center_is_ground_truth": False,
            }
        )
    leave_one_out = []
    point_ids = [point["id"] for point in points]
    for heldout in full_indices:
        training = [index for index in full_indices if index != heldout]
        fit = fit_candidate_pose(candidates, target_xyz, training)
        best = fit["best"]
        rotation = np.asarray(best["rotation_camera_to_table"], dtype=float)
        translation = np.asarray(best["translation_camera_to_table_mm"], dtype=float)
        predictions = []
        for candidate in candidates[heldout]:
            camera_center = np.asarray(candidate["sphere_center_camera_mm"], dtype=float)
            predicted = rotation @ camera_center + translation
            delta = predicted - target_xyz[heldout]
            predictions.append(
                {
                    "fit_id": candidate["fit_id"],
                    "predicted_xy_mm_user_convention": [float(predicted[0]), float(-predicted[1])],
                    "xy_error_mm": float(np.linalg.norm(delta[:2])),
                    "outline_center_px": candidate["ellipse_center_px"],
                    "distance_to_saved_segmentation_center_px": float(
                        np.linalg.norm(
                            np.asarray(candidate["ellipse_center_px"])
                            - segmentation_centers[points[heldout]["id"]]
                        )
                    ),
                }
            )
        ranked_predictions = sorted(predictions, key=lambda item: (item["xy_error_mm"], item["fit_id"]))
        leave_one_out.append(
            {
                "held_out_point_id": points[heldout]["id"],
                "training_point_ids": [points[index]["id"] for index in training],
                "pose_only_candidate_predictions_and_residuals": predictions,
                "oracle_best_candidate_for_error_lower_bound": ranked_predictions[0],
                "oracle_choice_uses_held_out_known_position_for_scoring_only": True,
                "training_pose_fit_summary_xy_only": public_pose_fit(fit, point_ids),
            }
        )

    all_xy = [
        [record["oracle_best_candidate_for_error_lower_bound"]["xy_error_mm"], record["held_out_point_id"]]
        for record in leave_one_out
    ]
    loo_errors = [item[0] for item in all_xy]
    source_manifest = [MEASUREMENTS, INTRINSICS_SOURCE, IMAGE, IMAGE_CENTER_BASELINE]
    source_manifest.extend(
        CASE_DIR / f"measured-grid-720--{point['id']}.json" for point in points
    )
    source_manifest.append(Path(__file__))
    public_candidate_fields = {
        "fit_id",
        "iteration",
        "is_qualified",
        "rejection",
        "support",
        "source_residual_rms_px",
        "quality",
        "outline_point_count",
        "sphere_geometry_status",
        "sphere_geometry_error",
        "ellipse_center_px",
        "ellipse_axes_px",
        "ellipse_angle_degrees",
        "ellipse_radial_residual_rms_px",
        "repeated_eigenvalue_relative_mismatch",
    }
    public_candidate_records = [
        {
            "point_id": record["point_id"],
            "fit_id": record["fit_id"],
            "candidate": {
                key: value
                for key, value in record["candidate"].items()
                if key in public_candidate_fields
            },
        }
        for record in case_records
    ]
    full_public = public_pose_fit(full, point_ids)
    return {
        "schema_version": 1,
        "purpose": "offline fixed-diameter sphere projection and 2D table consistency diagnostic",
        "production_behavior_changed": False,
        "physical_model": {
            "ball_diameter_mm": RADIUS_MM * 2.0,
            "radius_mm": RADIUS_MM,
            "center_plane_offset_from_cloth_mm": RADIUS_MM,
            "one_fixed_center_plane_for_all_balls": True,
            "table_output_dimensions": ["X", "Y"],
            "user_coordinate_convention": "X right, Y down from the upper-left table origin",
        },
        "camera": {
            "identity": "MOKOSE UC40M, 1280x720 saved frame; profile identity per prior calibration work",
            "source_dimensions_px": source_size.tolist(),
            "intrinsics_model": "frozen zero-radial checkerboard validation model",
            "camera_matrix": k.tolist(),
            "intrinsics_source": str(INTRINSICS_SOURCE.relative_to(ROOT)),
            "distortion": "zero-radial Brown model; no fisheye correction",
        },
        "reference_limits": {
            "point_count": len(points),
            "source": "user physical measurements already recorded in the saved measurements JSON",
            "measurement_uncertainty_mm": None,
            "touching_cushion_positions_derive_center_from_known_radius": True,
            "consistency_is_independent_accuracy": False,
            "no_new_measurements_or_fixture": True,
        },
        "solver": {
            "algorithm": "deterministic bounded 4-anchor initialization; alternate closest candidate assignment and proper Kabsch rigid alignment",
            "maximum_seed_combinations": MAX_SEEDS,
            "maximum_assignment_steps_per_seed": MAX_ASSIGNMENT_STEPS,
            "near_best_spread_window_mm": NEAR_OPTIMAL_DIAGNOSTIC_MM,
            "candidate_selection_uses_color_or_rank": False,
            "candidate_selection_uses_reference_coordinates_for_training": True,
            "leave_one_out_candidate_choice_is_oracle_scoring_only": True,
        },
        "candidate_inventory": {
            "qualified_usable_outline_count_by_point": {
                point["id"]: len(candidates[index]) for index, point in enumerate(points)
            },
            "all_refiner_candidates_with_qualification_and_rejection": public_candidate_records,
        },
        "all_points_training_fit": full_public,
        "synthetic_self_test": self_test(),
        "all_nine_fit_candidate_vs_existing_segmentation_center": selected_full_fit_diagnostics,
        "leave_one_point_out": leave_one_out,
        "existing_image_center_homography_baseline": image_center_baseline,
        "leave_one_point_out_xy_error_summary_mm": {
            "mean": float(statistics.mean(loo_errors)),
            "rms": float(np.sqrt(np.mean(np.square(loo_errors)))),
            "median": float(statistics.median(loo_errors)),
            "p95": float(np.percentile(loo_errors, 95)),
            "maximum": float(max(loo_errors)),
            "point_errors": {name: err for err, name in all_xy},
        },
        "interpretation": {
            "fixed_sphere_conic_geometry_used_internally": True,
            "two_dimensional_output_only": True,
            "training_fit_is_accuracy_claim": False,
            "candidate_families_all_preserved": True,
            "activation_recommended": False,
            "reason": "The source outlines are refiner-generated qualified fit contours rather than independently labeled physical silhouettes; the fixed-sphere fit is only a cross-point consistency diagnostic using measurements with unknown uncertainty and a frozen, imperfectly identified intrinsic calibration.",
        },
        "limitations": [
            "All nine user-reported positions have unknown uncertainty; several rail-contact positions derive X or Y from the shared 28.575 mm radius.",
            "The candidate outlines are fitted source-space projections from an estimator that rejects each measured observation as ambiguous; they are not ground-truth silhouette labels.",
            "The conic inversion assumes a pinhole camera and a true sphere tangent contour. Residual lens distortion and calibration uncertainty are not propagated.",
            "The checkerboard intrinsics are weakly identified by the existing small target and limited tilt variety; this analysis does not repair that calibration.",
            "Candidate assignments and rigid pose use the same nine references for training; leave-one-out scores are consistency diagnostics, not independent physical accuracy.",
            "A stable low residual would not qualify live availability, FPS, robustness, rail-contact performance, or shot coaching.",
        ],
        "input_manifest": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in source_manifest
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def write_report(report: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    full = report["all_points_training_fit"]["best_candidate_associations_and_xy_residuals"]
    loo = report["leave_one_point_out_xy_error_summary_mm"]
    baseline = report["existing_image_center_homography_baseline"]
    selected_nearest_count = sum(
        item["all_nine_pose_selected_candidate_is_nearest_to_saved_segmentation_center"]
        for item in report["all_nine_fit_candidate_vs_existing_segmentation_center"]
    )
    lines = [
        "# Fixed-sphere projection audit — 2026-10-27",
        "",
        "Offline saved-image diagnostic only. No camera was accessed, no new measurement or fixture was used, and no production behavior changed.",
        "",
        "## Method",
        "",
        "Each individually qualified source-space outline was fit as an ellipse. With the frozen 1280×720 zero-radial camera matrix, calibrated conic geometry was applied under the fixed 28.575 mm ball radius. The same fixed center plane is used for every ball; the report contains table X/Y only. One common rigid camera-to-table transform was fit from candidate assignments and the nine existing user-confirmed X/Y references, with user-facing Y-down restored at output.",
        "",
        f"The solver is deterministic and bounded. On the full nine-point fit it examined **{report['all_points_training_fit']['seeds_examined']}** initial candidate combinations across the four highest-area anchor quadrilaterals, then alternated nearest candidate assignment and proper rigid alignment. This is a bounded heuristic, not an exhaustive search across every nine-ball candidate combination; the JSON records every distinct converged assignment found, plus every individual candidate including unqualified/rejected fits.",
        "",
        "## Results",
        "",
        f"Qualified outline candidates per reference: `{json.dumps(report['candidate_inventory']['qualified_usable_outline_count_by_point'], sort_keys=True)}`.",
        "",
        f"Best all-nine training X/Y RMS: **{full['training_rms_xy_mm']:.3f} mm**; maximum per-point X/Y residual: **{full['training_max_xy_mm']:.3f} mm**.",
        "",
        f"The all-nine pose selected a candidate whose ellipse center was nearest the saved segmentation-ellipse center in **{selected_nearest_count}/9** cases. That fixed center file is an alternate same-image algorithm, not physical truth.",
        "",
        f"Existing image-center homography baseline LOO RMS: **{baseline['leave_one_out_rms_mm']:.3f} mm** (recorded report: 8.71 mm). It uses nine segmentation-derived centers and has the same unknown reference uncertainty.",
        "",
        f"For comparison, the sphere-conic pose produces an **oracle-best-candidate lower bound** with LOO X/Y RMS **{loo['rms']:.3f} mm**, mean **{loo['mean']:.3f} mm**, median **{loo['median']:.3f} mm**, p95 **{loo['p95']:.3f} mm**, max **{loo['maximum']:.3f} mm**. This is not an automatic localization result: for each held-out position, it scores all candidate projections against that position and selects the lowest-error candidate after seeing the held-out reference.",
        "",
        f"The existing segmentation-center homography has LOO X/Y RMS **{baseline['leave_one_out_rms_mm']:.3f} mm** on the same positions. Even the oracle sphere result is worse in this diagnostic. No global infeasibility claim is made because the assignment search is heuristic.",
        "",
        "| Held-out position | Oracle fit ID for lower bound | Oracle X/Y error (mm) | Homography X/Y error (mm) |",
        "|---|---:|---:|---:|",
    ]
    for entry in report["leave_one_point_out"]:
        best = entry["oracle_best_candidate_for_error_lower_bound"]
        homography_error = baseline["leave_one_out_errors_mm"][entry["held_out_point_id"]]
        lines.append(
            f"| {entry['held_out_point_id']} | {best['fit_id']} | {best['xy_error_mm']:.3f} | {homography_error:.3f} |"
        )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "This experiment tests whether one camera pose and the fixed-size sphere constraint can make competing measured-ball outlines globally consistent with the nine known table positions. It does not establish the physical silhouette or center, because the refiner rejects the observations as ambiguous and the fitted candidates are generated from the same image evidence. The physical measurements have no stated uncertainty, and the frozen checkerboard intrinsics remain weakly identified. Consequently, a low training or leave-one-out residual cannot be reported as millimeter accuracy or used to activate a production measurement.",
        "",
        "The sphere LOO values are an optimistic oracle lower bound: each held-out point's known physical position is used only after pose fitting to score all that point's outlines and retain the least residual. The method has not shown how to choose that boundary without the physical answer. The prior homography instead uses one fixed segmentation-derived pixel center per ball. Both compare against the same user-reported points and share uncertain references, so neither establishes accuracy.",
        "",
        "Synthetic tests exercise exact tangent-cone recovery, 200-degree arcs, a 0.15-pixel contour noise level, table edges, and a competing contour. A small contour perturbation produces several millimeters of X/Y error in this geometry, showing how outline errors can affect the result. These ideal synthetic cases do not characterize the saved RGB edge error.",
        "",
        "The complete JSON includes every refiner fit and rejection, ellipse/conic quality metrics, selected associations, every distinct converged assignment found by the bounded heuristic, each candidate's projected table X/Y under each leave-one-out pose, all X/Y residuals, input hashes, and runtime versions. It contains no height, depth, or out-of-plane residual values.",
        "",
        "## Reproduction",
        "",
        "```sh",
        "python thoughts/calibration/offline-sphere-projection-audit-20261027.py --self-test",
        "python thoughts/calibration/offline-sphere-projection-audit-20261027.py",
        "```",
        "",
        "The audit requires NumPy and OpenCV Python. The current saved environment is `/tmp/billiards-checkerboard-env/bin/python`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    report = build_report()
    write_report(report)
    print(
        json.dumps(
            {
                "report": str((OUT / "README.md").relative_to(ROOT)),
                "all_points_training_xy_rms_mm": report["all_points_training_fit"]["best_candidate_associations_and_xy_residuals"]["training_rms_xy_mm"],
                "loo_xy_error_mm": report["leave_one_point_out_xy_error_summary_mm"],
                "homography_loo_rms_mm": report["existing_image_center_homography_baseline"]["leave_one_out_rms_mm"],
                "candidate_counts": report["candidate_inventory"]["qualified_usable_outline_count_by_point"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
