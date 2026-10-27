#!/usr/bin/env python3
"""Relative registration audit for saved billiards-table images.

This audit uses visible fixed table marks only. It does not infer physical dot
coordinates, transfer ball positions between arrangements, or create a metric
table calibration. The 4K frame is downsampled by exactly 3 to compare it in
the 1280x720 image coordinate system used by the other saved scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "thoughts" / "calibration"
OUT = CAL / "table-feature-registration-audit-20261027"

IMAGE_PATHS = {
    "measured_720": CAL / "table-grid-20261026-114241.jpg",
    "alternate_720": CAL / "table-grid-2026-10-26.jpg",
    "framing_before_720": CAL / "stream-4k-verification" / "before.jpg",
    "framing_after_4k": CAL / "stream-4k-verification" / "after.jpg",
}

# Pixel seeds were placed on visible white rail sights in the measured 720p
# frame, then reused as local search centers after checking each named scene.
# Coordinates are image annotations only; none encode physical X or Y.
RAIL_SEEDS = {
    "top_rail_surface": [
        (210.0, 35.0),
        (353.0, 35.0),
        (496.0, 35.0),
        (784.0, 33.0),
        (930.0, 31.0),
        (1077.0, 30.0),
    ],
    "bottom_rail_surface": [
        (359.0, 687.0),
        (500.0, 687.0),
        (927.0, 692.0),
        (1072.0, 693.0),
    ],
    "left_rail_vertical_face": [(32.0, 221.0), (36.0, 504.0)],
    "right_rail_vertical_face": [
        (1265.0, 217.0),
        (1263.0, 363.0),
        (1267.0, 476.0),
        (1261.0, 508.0),
    ],
}

METRIC_REFERENCE_RECORDS = [
    CAL / "table-grid-20261026-114241-measurements.json",
    CAL / "table-grid-feasibility-results.json",
    CAL / "native-localization-corpus-v1.json",
    CAL / "stream-4k-framing.json",
]

# The cloth markers are clearly visible only in the framing pair. Their precise
# physical table coordinates are absent from the repository, so they are used
# solely as held-out image landmarks for a cross-plane transfer check.
CLOTH_MARKER_SEEDS = [(366.5, 361.5), (918.5, 361.5)]

PAIR_SPECS = [
    {
        "pair_id": "measured_to_alternate_720",
        "source": "measured_720",
        "destination": "alternate_720",
        "heldout_planes": ["left_rail_vertical_face", "right_rail_vertical_face"],
    },
    {
        "pair_id": "framing_before_to_after_720_equivalent",
        "source": "framing_before_720",
        "destination": "framing_after_4k",
        "heldout_planes": [
            "left_rail_vertical_face",
            "right_rail_vertical_face",
            "cloth_plane_marker",
        ],
    },
]

# Hold out one sight from each end of the top rail and two bottom sights. The
# remaining top/bottom points span both image axes for a planar homography.
TRAIN_IDS = {
    "top_rail_surface": {"top_rail_surface_2", "top_rail_surface_3", "top_rail_surface_4", "top_rail_surface_5"},
    "bottom_rail_surface": {"bottom_rail_surface_2", "bottom_rail_surface_3"},
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_image(image_id: str) -> tuple[np.ndarray, list[int], str]:
    path = IMAGE_PATHS[image_id]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    source_size = [int(image.shape[1]), int(image.shape[0])]
    if image_id == "framing_after_4k":
        if source_size != [3840, 2160]:
            raise ValueError(f"Expected native 3840x2160 framing image, got {source_size}")
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    elif source_size != [1280, 720]:
        raise ValueError(f"Expected 1280x720 image for {image_id}, got {source_size}")
    return image, source_size, file_sha256(path)


def detect_seeded_rail_sight(
    image: np.ndarray, seed_xy: tuple[float, float], *, radius: int = 8, threshold: int = 175
) -> dict[str, Any] | None:
    """Refine a manually seeded white rail dot using a bounded bright component."""
    cx, cy = (int(round(seed_xy[0])), int(round(seed_xy[1])))
    patch = image[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        return None
    binary = (patch >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    possibilities = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if not (8 <= area <= 60 and 2 <= width <= 9 and 2 <= height <= 9):
            continue
        aspect = width / height
        if not 0.55 <= aspect <= 1.8:
            continue
        detected = np.array([cx - radius + centroids[label][0], cy - radius + centroids[label][1]])
        distance = float(np.linalg.norm(detected - np.asarray(seed_xy)))
        possibilities.append((distance, detected, area, width, height))
    if not possibilities:
        return None
    distance, detected, area, width, height = min(possibilities, key=lambda item: (item[0], item[2]))
    return {
        "seed_px": [float(seed_xy[0]), float(seed_xy[1])],
        "center_px": [float(detected[0]), float(detected[1])],
        "component_area_px": int(area),
        "component_size_px": [int(width), int(height)],
        "centroid_method": "binary_component_centroid",
        "threshold_gray": threshold,
        "seed_distance_px": distance,
    }


def detect_rail_annotations() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    annotations: dict[str, dict[str, Any]] = {}
    image_metadata: dict[str, dict[str, Any]] = {}
    for image_id in IMAGE_PATHS:
        gray, original_size, digest = load_image(image_id)
        image_metadata[image_id] = {
            "path": str(IMAGE_PATHS[image_id].relative_to(ROOT)),
            "sha256": digest,
            "source_dimensions_px": original_size,
            "analysis_dimensions_px": [1280, 720],
            "analysis_transform": (
                "area_downsample_exactly_3x"
                if image_id == "framing_after_4k"
                else "identity"
            ),
        }
        image_annotations: dict[str, Any] = {}
        for plane, seeds in RAIL_SEEDS.items():
            for position, seed in enumerate(seeds, start=1):
                feature_id = f"{plane}_{position}"
                annotation = detect_seeded_rail_sight(gray, seed)
                image_annotations[feature_id] = {
                    "plane_class": plane,
                    "status": "localized" if annotation else "not_detected",
                    **(annotation or {}),
                }
        annotations[image_id] = image_annotations
    return annotations, image_metadata


def detect_cloth_markers() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for image_id in ("framing_before_720", "framing_after_4k"):
        image, _, _ = load_image(image_id)
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=100.0,
            param1=100.0,
            param2=12.0,
            minRadius=4,
            maxRadius=10,
        )
        detections = [] if circles is None else circles[0].astype(float).tolist()
        selected = []
        used: set[int] = set()
        for marker_index, seed in enumerate(CLOTH_MARKER_SEEDS, start=1):
            eligible = [
                (float(np.linalg.norm(np.asarray([x, y]) - np.asarray(seed))), index, x, y, radius)
                for index, (x, y, radius) in enumerate(detections)
                if 250.0 < x < 1050.0 and 280.0 < y < 500.0 and index not in used
            ]
            if not eligible:
                selected.append(
                    {
                        "feature_id": f"cloth_marker_{marker_index}",
                        "plane_class": "cloth_plane_marker",
                        "status": "not_visible_or_occluded",
                        "seed_px": list(seed),
                    }
                )
                continue
            distance, index, x, y, radius = min(eligible, key=lambda item: item[0])
            if distance > 15.0:
                selected.append(
                    {
                        "feature_id": f"cloth_marker_{marker_index}",
                        "plane_class": "cloth_plane_marker",
                        "status": "not_visible_or_occluded",
                        "seed_px": list(seed),
                    }
                )
                continue
            used.add(index)
            selected.append(
                {
                    "feature_id": f"cloth_marker_{marker_index}",
                    "plane_class": "cloth_plane_marker",
                    "status": "localized",
                    "center_px": [float(x), float(y)],
                    "radius_px": float(radius),
                    "method": "Hough circle center on dark/light target boundary",
                    "seed_px": list(seed),
                    "seed_distance_px": distance,
                    "physical_xy_recorded": False,
                }
            )
        result[image_id] = selected
    # The markers are covered by balls in both table-grid arrangements. They
    # are not used as failed or extrapolated points for those images.
    result["measured_720"] = [
        {"feature_id": f"cloth_marker_{index}", "plane_class": "cloth_plane_marker", "status": "occluded_by_ball"}
        for index in range(1, 3)
    ]
    result["alternate_720"] = [
        {"feature_id": f"cloth_marker_{index}", "plane_class": "cloth_plane_marker", "status": "occluded_by_ball"}
        for index in range(1, 3)
    ]
    return result


def fit_homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Corresponding point arrays must both be N×2")
    if len(source) < 4:
        raise ValueError("At least four correspondences are required")
    matrix, _ = cv2.findHomography(source.astype(np.float64), destination.astype(np.float64), method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Homography fit failed")
    return matrix / matrix[2, 2]


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    warped = (homography @ homogeneous.T).T
    if np.any(np.abs(warped[:, 2]) < 1e-12):
        raise ValueError("Homography produced an invalid projective denominator")
    return warped[:, :2] / warped[:, 2:3]


def error_summary(errors: list[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=float)
    return {
        "count": int(len(values)),
        "rms_px": float(np.sqrt(np.mean(values**2))) if len(values) else 0.0,
        "median_px": float(np.median(values)) if len(values) else 0.0,
        "p95_px": float(np.percentile(values, 95)) if len(values) else 0.0,
        "max_px": float(np.max(values)) if len(values) else 0.0,
    }


def matched_rail_points(
    annotations: dict[str, dict[str, Any]], source_id: str, destination_id: str, planes: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    feature_ids: list[str] = []
    source_points: list[list[float]] = []
    destination_points: list[list[float]] = []
    for plane in planes:
        features = sorted(
            (feature_id for feature_id in annotations[source_id] if feature_id.startswith(plane + "_")),
            key=lambda feature_id: int(feature_id.rsplit("_", 1)[1]),
        )
        for feature_id in features:
            first = annotations[source_id][feature_id]
            second = annotations[destination_id][feature_id]
            if first.get("status") != "localized" or second.get("status") != "localized":
                continue
            feature_ids.append(feature_id)
            source_points.append(first["center_px"])
            destination_points.append(second["center_px"])
    return feature_ids, np.asarray(source_points), np.asarray(destination_points)


def evaluate_pair(
    pair_spec: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    cloth_markers: dict[str, Any],
) -> dict[str, Any]:
    source_id = pair_spec["source"]
    destination_id = pair_spec["destination"]
    ids, source_points, destination_points = matched_rail_points(
        annotations,
        source_id,
        destination_id,
        ["top_rail_surface", "bottom_rail_surface"],
    )
    # Hold out fixed outer sights on both long-rail rows; fit only the remaining
    # rail-top/bottom points. This split is identical for both scene pairs.
    holdout_ids = {
        "top_rail_surface_1",
        "top_rail_surface_6",
        "bottom_rail_surface_1",
        "bottom_rail_surface_4",
    }
    train_indices = [index for index, feature_id in enumerate(ids) if feature_id not in holdout_ids]
    held_indices = [index for index, feature_id in enumerate(ids) if feature_id in holdout_ids]
    if len(train_indices) < 4 or len(held_indices) < 2:
        raise ValueError(f"Not enough localized rail sights for {pair_spec['pair_id']}")
    homography = fit_homography(source_points[train_indices], destination_points[train_indices])
    train_prediction = project_points(source_points[train_indices], homography)
    held_prediction = project_points(source_points[held_indices], homography)
    train_errors = np.linalg.norm(train_prediction - destination_points[train_indices], axis=1)
    held_errors = np.linalg.norm(held_prediction - destination_points[held_indices], axis=1)
    heldout = [
        {
            "feature_id": ids[index],
            "plane_class": "top_or_bottom_rail_surface",
            "source_px": source_points[index].tolist(),
            "destination_px": destination_points[index].tolist(),
            "predicted_destination_px": held_prediction[offset].tolist(),
            "transfer_error_px": float(held_errors[offset]),
        }
        for offset, index in enumerate(held_indices)
    ]

    # Side-rail dots are a different plane class; evaluate them without fitting
    # the homography so parallax or model mismatch remains visible.
    side_ids, side_source, side_destination = matched_rail_points(
        annotations,
        source_id,
        destination_id,
        ["left_rail_vertical_face", "right_rail_vertical_face"],
    )
    if len(side_ids):
        side_prediction = project_points(side_source, homography)
        side_errors = np.linalg.norm(side_prediction - side_destination, axis=1)
        side_holdout = [
            {
                "feature_id": feature_id,
                "plane_class": (
                    "left_rail_vertical_face"
                    if feature_id.startswith("left_rail")
                    else "right_rail_vertical_face"
                ),
                "source_px": source.tolist(),
                "destination_px": destination.tolist(),
                "predicted_destination_px": predicted.tolist(),
                "transfer_error_px": float(error),
            }
            for feature_id, source, destination, predicted, error in zip(
                side_ids, side_source, side_destination, side_prediction, side_errors
            )
        ]
    else:
        side_errors = np.asarray([], dtype=float)
        side_holdout = []

    marker_records = []
    source_markers = cloth_markers.get(source_id, [])
    destination_markers = cloth_markers.get(destination_id, [])
    destination_by_id = {item["feature_id"]: item for item in destination_markers}
    for source_marker in source_markers:
        destination_marker = destination_by_id.get(source_marker["feature_id"], {})
        if source_marker.get("status") != "localized" or destination_marker.get("status") != "localized":
            marker_records.append(
                {
                    "feature_id": source_marker["feature_id"],
                    "plane_class": "cloth_plane_marker",
                    "status": "not_comparable",
                    "source_status": source_marker.get("status"),
                    "destination_status": destination_marker.get("status"),
                }
            )
            continue
        source_marker_xy = np.asarray(source_marker["center_px"], dtype=float)[None, :]
        destination_marker_xy = np.asarray(destination_marker["center_px"], dtype=float)
        prediction = project_points(source_marker_xy, homography)[0]
        marker_records.append(
            {
                "feature_id": source_marker["feature_id"],
                "plane_class": "cloth_plane_marker",
                "status": "held_out_cross_plane_transfer",
                "source_px": source_marker_xy[0].tolist(),
                "destination_px": destination_marker_xy.tolist(),
                "predicted_destination_px": prediction.tolist(),
                "transfer_error_px": float(np.linalg.norm(prediction - destination_marker_xy)),
                "physical_xy_recorded": False,
            }
        )
    return {
        "pair_id": pair_spec["pair_id"],
        "source_image_id": source_id,
        "destination_image_id": destination_id,
        "fit_feature_ids": [ids[index] for index in train_indices],
        "heldout_rail_feature_ids": [ids[index] for index in held_indices],
        "homography_source_to_destination_720_equivalent_px": homography.tolist(),
        "fit_point_count": int(len(train_indices)),
        "heldout_rail_point_count": int(len(held_indices)),
        "training_error": error_summary(train_errors.tolist()),
        "heldout_long_rail_error": error_summary(held_errors.tolist()),
        "heldout_long_rail_landmarks": heldout,
        "heldout_other_plane_error": error_summary(side_errors.tolist()),
        "heldout_other_plane_landmarks": side_holdout,
        "heldout_cloth_plane_markers": marker_records,
        "registration_is_metric": False,
        "balls_used_as_registration_points": False,
    }


def self_test() -> dict[str, Any]:
    source = np.array(
        [[10.0, 10.0], [200.0, 15.0], [15.0, 150.0], [220.0, 180.0], [100.0, 75.0], [300.0, 100.0]]
    )
    truth = np.array(
        [[1.02, 0.015, -12.0], [-0.007, 0.98, 8.0], [0.00004, -0.00002, 1.0]],
        dtype=float,
    )
    destination = project_points(source, truth)
    fit = fit_homography(source[:4], destination[:4])
    holdout = np.linalg.norm(project_points(source[4:], fit) - destination[4:], axis=1)
    matrix_error = float(np.linalg.norm(fit / fit[2, 2] - truth / truth[2, 2]))
    max_transfer = float(np.max(holdout))
    if matrix_error > 1e-5 or max_transfer > 1e-4:
        raise AssertionError(
            f"synthetic registration failed: matrix {matrix_error}, holdout {max_transfer}"
        )
    return {
        "status": "passed",
        "known_transform_matrix_error": matrix_error,
        "heldout_transfer_max_error_px": max_transfer,
        "balls_used": False,
        "metric_table_coordinates_created": False,
    }


def build_report() -> dict[str, Any]:
    annotations, image_metadata = detect_rail_annotations()
    cloth_markers = detect_cloth_markers()
    pairs = [evaluate_pair(spec, annotations, cloth_markers) for spec in PAIR_SPECS]
    measurements = json.loads(
        (CAL / "table-grid-20261026-114241-measurements.json").read_text()
    )
    table_dots_in_records = []
    for record_path in METRIC_REFERENCE_RECORDS:
        record = json.loads(record_path.read_text())
        encoded = json.dumps(record).lower()
        if "table_dot_xy_mm" in encoded or "rail_dot_xy_mm" in encoded:
            table_dots_in_records.append(str(record_path.relative_to(ROOT)))
    return {
        "schema_version": 1,
        "purpose": "relative image registration from static table features only",
        "production_behavior_changed": False,
        "metric_table_mapping_created": False,
        "metric_reference_inventory": {
            "table_dimensions_mm": measurements["playing_area_inside_cushion_noses_mm"],
            "physical_ball_diameter_mm": measurements["ball_diameter_mm"],
            "known_dot_coordinate_records_found": table_dots_in_records,
            "exact_physical_dot_xy_available_in_repository": False,
            "inspected_records": [str(path.relative_to(ROOT)) for path in METRIC_REFERENCE_RECORDS],
            "cloth_marker_physical_xy_available": False,
            "no_dot_spacing_assumed": True,
            "interpretation": "The user previously said table dots are precisely measured, but this repository contains no exact dot X/Y values. Visible marker positions are used only for image registration.",
        },
        "coordinate_contract": {
            "reported_coordinates": "source/destination image pixels only, normalized to 1280x720",
            "no_table_xy_estimated": True,
            "no_height_depth_or_z_output": True,
            "moving_balls_used_as_registration_points": False,
        },
        "image_sources": image_metadata,
        "annotation_method": {
            "rail_sights": "manually seeded static rail-dot neighborhoods, refined to local bright connected-component centroids",
            "rail_threshold_gray": 175,
            "rail_patch_half_size_px": 8,
            "clothing_markers": "Hough circle center on two visible high-contrast circular cloth markers in framing pair only",
            "cloth_marker_status_in_grid_scenes": "occluded by balls; not inferred",
            "annotation_plane_classes": list(RAIL_SEEDS.keys()) + ["cloth_plane_marker"],
        },
        "rail_dot_annotations_by_image": annotations,
        "cloth_marker_annotations_by_image": cloth_markers,
        "pairwise_registration": pairs,
        "limitations": [
            "Exact physical X/Y values of the visible rail dots or cloth markers are not present in the saved project records.",
            "No absolute metric table-plane homography or ball-center accuracy can be estimated without verified metric point correspondences on the cloth plane.",
            "The measured/alternate 720p images have balls over the two cloth markers; ball coordinates are not used to register those images.",
            "Raised rail features and cloth-plane markers are separate physical planes. Transfer between them checks image-coordinate consistency only and cannot calibrate the cloth plane.",
            "The 720p/4K framing pair uses the same camera and scene and tests registration, not optical or physical accuracy. The earlier perimeter SIFT audit remains a separate reference.",
            "Connected-component and Hough centers have pixel uncertainty from JPEG, threshold, blur, and resampling; this audit does not propagate physical uncertainty.",
            "Saved stills cannot establish tracking availability, temporal stability, live FPS, or shot-coaching accuracy.",
        ],
        "synthetic_self_test": self_test(),
        "input_manifest": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in [
                *IMAGE_PATHS.values(),
                *METRIC_REFERENCE_RECORDS,
                Path(__file__),
            ]
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def save_report(report: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_readme(report)


def write_readme(report: dict[str, Any]) -> None:
    lines = [
        "# Static table-feature registration audit — 2026-10-27",
        "",
        "Offline saved-image registration only. No camera or new physical measurement was used. No production behavior changed.",
        "",
        "## Coordinate and reference scope",
        "",
        "The report contains pixel coordinates only, normalized to the 1280×720 frame. The 3840×2160 framing image is reduced exactly 3× before comparison. The inspected measurement, feasibility, corpus, and framing records contain the 2336.8 × 1155.7 mm playing-area dimensions and nine ball positions, but no exact physical X/Y coordinates for the visible rail dots or cloth markers. No spacing is assumed from table conventions. No metric table mapping is emitted.",
        "",
        "Two high-contrast circular markers are visible on the cloth plane in the framing-before/after pair. They are covered by balls in the measured and alternate grid scenes, so they are recorded as occluded rather than inferred. White sights on the upper/lower rails are used for registration; side-rail sights and cloth markers are held out from that fit because they are on other physical planes.",
        "",
        "## Results",
        "",
    ]
    for pair in report["pairwise_registration"]:
        lines.extend(
            [
                f"### {pair['pair_id']}",
                "",
                f"Fit {pair['fit_point_count']} upper/lower rail sights. Their training RMS is **{pair['training_error']['rms_px']:.3f} px**. Held-out upper/lower rail transfer RMS is **{pair['heldout_long_rail_error']['rms_px']:.3f} px** (max {pair['heldout_long_rail_error']['max_px']:.3f} px).",
                f"Held-out side-face sight transfer RMS is **{pair['heldout_other_plane_error']['rms_px']:.3f} px** (max {pair['heldout_other_plane_error']['max_px']:.3f} px).",
                "",
            ]
        )
        for marker in pair["heldout_cloth_plane_markers"]:
            if marker["status"] == "held_out_cross_plane_transfer":
                lines.append(
                    f"Held-out cloth marker `{marker['feature_id']}` transfers with **{marker['transfer_error_px']:.3f} px** error; physical table coordinates are not recorded."
                )
            else:
                lines.append(
                    f"Held-out cloth marker `{marker['feature_id']}` status: `{marker['status']}`."
                )
        lines.extend([""])
    lines.extend(
        [
            "## Annotations",
            "",
            "Every detected rail sight and cloth marker is listed with its image ID, plane class, refined pixel center, detector settings, and source hashes in `summary.json`. Balls are excluded from registration by construction: only seeded rail-sight neighborhoods and the two static cloth-marker neighborhoods are annotated.",
            "",
            "## Interpretation and limits",
            "",
            "A low pixel transfer error shows that static landmarks can register these saved frames consistently. It does not establish absolute table X/Y, physical ball-center accuracy, or that all points share one physical plane. The exact metric coordinates of the table dots are not present in the inspected project records, despite the user saying they had been measured precisely. Raised-rail sight points cannot substitute for cloth-plane calibration without their physical plane and coordinates.",
            "",
            "The measured-to-alternate registration uses only upper/lower rail sights for fitting. It never transfers the measured arrangement's ball positions. The 720p/4K pair adds held-out vertical-side rail sights and visible cloth markers to test cross-plane pixel transfer; the prior 4K perimeter SIFT report is not duplicated as a metric calibration.",
            "",
            "## Reproduction",
            "",
            "```sh",
            "python thoughts/calibration/offline-table-feature-registration-audit-20261027.py --self-test",
            "python thoughts/calibration/offline-table-feature-registration-audit-20261027.py",
            "```",
            "",
            "The audit requires NumPy and OpenCV Python. The saved environment is `/tmp/billiards-checkerboard-env/bin/python`.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    report = build_report()
    save_report(report)
    print(
        json.dumps(
            {
                "report": str((OUT / "README.md").relative_to(ROOT)),
                "pairs": [
                    {
                        "pair_id": pair["pair_id"],
                        "rail_holdout_rms_px": pair["heldout_long_rail_error"]["rms_px"],
                        "side_holdout_rms_px": pair["heldout_other_plane_error"]["rms_px"],
                        "cloth_markers": [
                            (item["feature_id"], item.get("transfer_error_px"), item["status"])
                            for item in pair["heldout_cloth_plane_markers"]
                        ],
                    }
                    for pair in report["pairwise_registration"]
                ],
                "exact_physical_dot_xy_available": report["metric_reference_inventory"][
                    "exact_physical_dot_xy_available_in_repository"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
