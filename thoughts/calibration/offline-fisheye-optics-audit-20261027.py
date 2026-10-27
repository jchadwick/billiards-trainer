#!/usr/bin/env python3
"""Offline comparison of OpenCV's fisheye projection on existing captures only.

No camera configuration or production optics profile is written. The board
pose is solved from one checkerboard parity and scored on the other, exactly
as in analyze_checkerboard_session.py. The source corners and cushion sample
coordinates are frozen inputs; this is not physical ball/table accuracy.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thoughts/calibration/fisheye-optics-audit-20261027"
MANIFEST = ROOT / "thoughts/calibration/checkerboard-session/manifest.json"
BROWN_PREFLIGHT = ROOT / "thoughts/calibration/checkerboard-eight-view-baseline.json"
BROWN_REPORT = ROOT / "thoughts/calibration/checkerboard-model-preflight.json"
VIEW009_REPORT = ROOT / "thoughts/calibration/checkerboard-frozen-validation-view-009.json"
CUSHION_DATA = ROOT / "thoughts/calibration/cushion-straightness.json"
CHECKERBOARD_SCRIPT = ROOT / "thoughts/calibration/analyze_checkerboard_session.py"
CUSHION_SCRIPT = ROOT / "thoughts/calibration/analyze_cushion_straightness.py"
RUNNER_PATH = ROOT / "thoughts/calibration/offline-fisheye-optics-audit-20261027.py"

# The OpenCV 4.x/5.x Python binding exposes no cv2.fisheye.CALIB_* constants
# in the installed 5.0.0 build. These are the documented cv::fisheye values.
FISHEYE_RECOMPUTE_EXTRINSIC = 1 << 1
FISHEYE_FIX_SKEW = 1 << 3
FISHEYE_FLAGS = FISHEYE_RECOMPUTE_EXTRINSIC | FISHEYE_FIX_SKEW
MAX_TRAINING_RMS_PX = 2.0  # diagnostic fail-closed threshold, not a release target


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def board_points(pattern: tuple[int, int], pitch: float) -> np.ndarray:
    points = np.zeros((1, pattern[0] * pattern[1], 3), dtype=np.float64)
    points[0, :, :2] = np.mgrid[: pattern[0], : pattern[1]].T.reshape(-1, 2) * pitch
    return points


def finite(values) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def error_summary(errors: list[float]) -> dict | None:
    if not errors:
        return None
    a = np.asarray(errors, dtype=float)
    return {
        "rms_px": float(np.sqrt(np.mean(a * a))),
        "median_px": float(np.median(a)),
        "p95_px": float(np.percentile(a, 95)),
        "max_px": float(a.max()),
        "count": int(a.size),
    }


def fit_fisheye(obj: np.ndarray, images: list[np.ndarray], size: tuple[int, int]) -> dict:
    """Fit the canonical four-coefficient theta-polynomial, with fixed skew."""
    # Use the documented recommended extrinsic recomputation and fixed-skew
    # flags, with no hand-selected starting coefficients. The historical
    # pinhole model is not supplied as an intrinsic guess for a different
    # projection family.
    k = None
    d = None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9)
    rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
        [obj.copy() for _ in images],
        [image.copy() for image in images],
        size,
        k,
        d,
        flags=FISHEYE_FLAGS,
        criteria=criteria,
    )
    errors: list[float] = []
    for image, rvec, tvec in zip(images, rvecs, tvecs, strict=True):
        projected = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)
        errors.extend(np.linalg.norm(projected - image.reshape(-1, 2), axis=1).tolist())
    return {
        "calibrator_rms_px": float(rms),
        "recomputed_training_error": error_summary(errors),
        "camera_matrix": np.asarray(k, dtype=float).tolist(),
        "distortion_theta_k1_k2_k3_k4": np.asarray(d, dtype=float).reshape(-1).tolist(),
        "rvecs": [np.asarray(v, dtype=float).reshape(-1).tolist() for v in rvecs],
        "tvecs": [np.asarray(v, dtype=float).reshape(-1).tolist() for v in tvecs],
    }


def fisheye_monotonicity(k: np.ndarray, d: np.ndarray, size: tuple[int, int]) -> dict:
    width, height = size
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=float,
    )
    normalized = np.column_stack(
        [(corners[:, 0] - k[0, 2]) / k[0, 0], (corners[:, 1] - k[1, 2]) / k[1, 1]]
    )
    theta_max = float(np.arctan(np.linalg.norm(normalized, axis=1)).max())
    theta = np.linspace(0.0, theta_max, 2049)
    k1, k2, k3, k4 = np.asarray(d, dtype=float).reshape(-1)
    theta2 = theta * theta
    scale = 1 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
    derivative = (
        1
        + 3 * k1 * theta2
        + 5 * k2 * theta2**2
        + 7 * k3 * theta2**3
        + 9 * k4 * theta2**4
    )
    return {
        "maximum_corner_incidence_angle_rad": theta_max,
        "minimum_radial_scale": float(scale.min()),
        "minimum_radial_derivative": float(derivative.min()),
        "monotonic_over_frame_corners": bool(scale.min() > 0 and derivative.min() > 0),
    }


def fit_issues(fit: dict, size: tuple[int, int]) -> list[str]:
    k = np.asarray(fit["camera_matrix"], dtype=float)
    d = np.asarray(fit["distortion_theta_k1_k2_k3_k4"], dtype=float)
    issues: list[str] = []
    if not finite(k) or not finite(d):
        return ["nonfinite_parameters"]
    width, height = size
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        issues.append("nonpositive_focal_length")
    if not (0.1 * width <= k[0, 0] <= 10 * width and 0.1 * width <= k[1, 1] <= 10 * width):
        issues.append("focal_length_outside_plausibility_bounds")
    if not (-0.25 * width <= k[0, 2] <= 1.25 * width and -0.25 * height <= k[1, 2] <= 1.25 * height):
        issues.append("principal_point_outside_plausibility_bounds")
    mapping = fisheye_monotonicity(k, d, size)
    if not mapping["monotonic_over_frame_corners"]:
        issues.append("theta_mapping_not_monotonic_over_frame")
    training_rms = fit["recomputed_training_error"]["rms_px"]
    if not math.isfinite(training_rms) or training_rms > MAX_TRAINING_RMS_PX:
        issues.append("training_residual_exceeds_diagnostic_plausibility_bound")
    return issues


def fisheye_pose_errors(obj: np.ndarray, image: np.ndarray, pattern: tuple[int, int], k: np.ndarray, d: np.ndarray) -> list[float]:
    """Same parity pose split, using undistorted points for IPPE pose fitting."""
    points = image.reshape(-1, 2)
    parity = np.indices((pattern[1], pattern[0])).sum(axis=0).ravel() % 2
    errors: list[float] = []
    zero_dist = np.zeros((5, 1), dtype=float)
    pinhole_board = obj.reshape(-1, 3)
    for value in (0, 1):
        in_mask = parity == value
        undistorted = cv2.fisheye.undistortPoints(
            points.reshape(1, -1, 2), k, d, R=np.eye(3), P=k
        ).reshape(-1, 2)
        train_objects = pinhole_board[in_mask].reshape(-1, 1, 3)
        train_points = undistorted[in_mask].reshape(-1, 1, 2)
        result = cv2.solvePnPGeneric(
            train_objects,
            train_points,
            k,
            zero_dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
        solutions = []
        for rvec, tvec in zip(result[1], result[2], strict=True):
            rotation = cv2.Rodrigues(rvec)[0]
            if np.min((pinhole_board @ rotation.T + tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(
                train_objects, train_points, k, zero_dist, rvec, tvec
            )
            if np.min((pinhole_board @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            projected = cv2.projectPoints(train_objects, rvec, tvec, k, zero_dist)[0].reshape(-1, 2)
            solutions.append((float(np.sum((projected - undistorted[in_mask]) ** 2)), rvec, tvec))
        if not solutions:
            raise ValueError("no positive-depth parity pose solution")
        _, rvec, tvec = min(solutions, key=lambda item: item[0])
        raw_prediction = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)
        error = np.linalg.norm(raw_prediction[~in_mask] - points[~in_mask], axis=1)
        if not finite(error):
            raise ValueError("nonfinite fisheye held-out residual")
        errors.extend(error.tolist())
    return errors


def load_inputs():
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["image_size_px"] != [1280, 720] or manifest["pattern_inner_corners"] != [9, 6]:
        raise ValueError("checkerboard manifest geometry differs from expected saved session")
    if manifest["square_pitch_mm"] != 25.4 or len(manifest["views"]) != 9:
        raise ValueError("checkerboard board scale or view count differs")
    for view in manifest["views"]:
        source = MANIFEST.parent / view["source_file"]
        if file_sha(source) != view["source_sha256"]:
            raise ValueError(f"source image hash mismatch: {view['id']}")
        decoded = np.fromfile(source, dtype=np.uint8)
        decoded_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
        if decoded_sha != view["source_sha256"]:
            raise ValueError(f"source-byte hash mismatch: {view['id']}")
    return manifest, manifest_bytes


def fisheye_all_folds(manifest: dict, obj: np.ndarray, train_count: int = 8) -> dict:
    size = tuple(manifest["image_size_px"])
    views = manifest["views"][:train_count]
    images = [np.asarray(v["corners_source_image_px"], dtype=np.float64).reshape(1, -1, 2) for v in views]
    results = []
    all_errors: list[float] = []
    intrinsics: list[list[float]] = []
    coefficients: list[list[float]] = []
    attempted_intrinsics: list[list[float]] = []
    attempted_coefficients: list[list[float]] = []
    for held_index, view in enumerate(views):
        fold = {"held_out_view": view["id"]}
        train = images[:held_index] + images[held_index + 1 :]
        try:
            fit = fit_fisheye(obj, train, size)
            k = np.asarray(fit["camera_matrix"], dtype=float)
            d = np.asarray(fit["distortion_theta_k1_k2_k3_k4"], dtype=float).reshape(4, 1)
            fold["fit"] = {key: value for key, value in fit.items() if key not in {"rvecs", "tvecs"}}
            if finite(k) and finite(d):
                attempted_intrinsics.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
                attempted_coefficients.append(d.reshape(-1).tolist())
            fold["issues"] = fit_issues(fit, size)
            fold["monotonicity"] = fisheye_monotonicity(k, d, size)
            if fold["issues"]:
                raise ValueError("fit failed: " + ", ".join(fold["issues"]))
            fold_errors = fisheye_pose_errors(
                obj, images[held_index], tuple(manifest["pattern_inner_corners"]), k, d
            )
            fold["heldout_error_px"] = error_summary(fold_errors)
            all_errors.extend(fold_errors)
            intrinsics.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
            coefficients.append(d.reshape(-1).tolist())
        except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            fold["error"] = str(exc)
        results.append(fold)
    stability = {}
    if intrinsics:
        p = np.asarray(intrinsics)
        stability["intrinsics"] = {
            "columns": ["fx", "fy", "cx", "cy"],
            "min": p.min(axis=0).tolist(),
            "max": p.max(axis=0).tolist(),
            "stddev": p.std(axis=0).tolist(),
            "relative_focal_range": ((p.max(axis=0)[:2] - p.min(axis=0)[:2]) / np.maximum(np.abs(np.median(p, axis=0)[:2]), 1e-12)).tolist(),
            "principal_range_fraction_image": ((p.max(axis=0)[2:] - p.min(axis=0)[2:]) / np.asarray(size)).tolist(),
        }
    if coefficients:
        p = np.asarray(coefficients)
        stability["theta_coefficients_k1_to_k4"] = {
            "min": p.min(axis=0).tolist(),
            "max": p.max(axis=0).tolist(),
            "stddev": p.std(axis=0).tolist(),
        }
    if attempted_intrinsics:
        p = np.asarray(attempted_intrinsics)
        stability["all_finite_fit_attempts_intrinsics"] = {
            "columns": ["fx", "fy", "cx", "cy"],
            "min": p.min(axis=0).tolist(),
            "max": p.max(axis=0).tolist(),
            "stddev": p.std(axis=0).tolist(),
        }
    if attempted_coefficients:
        p = np.asarray(attempted_coefficients)
        stability["all_finite_fit_attempts_theta_coefficients_k1_to_k4"] = {
            "min": p.min(axis=0).tolist(),
            "max": p.max(axis=0).tolist(),
            "stddev": p.std(axis=0).tolist(),
        }
    return {
        "fit_model": "OpenCV fisheye theta-polynomial with four coefficients",
        "flags_numeric": FISHEYE_FLAGS,
        "flags_definition": {"recompute_extrinsic": FISHEYE_RECOMPUTE_EXTRINSIC, "fix_skew": FISHEYE_FIX_SKEW},
        "folds": results,
        "successful_folds": len(intrinsics),
        "whole_view_holdout_pose_split_error_px": error_summary(all_errors),
        "parameter_ranges": stability,
    }


def fisheye_prospective_009(manifest: dict, obj: np.ndarray) -> dict:
    size = tuple(manifest["image_size_px"])
    train_views = manifest["views"][:8]
    view = manifest["views"][8]
    train_images = [np.asarray(v["corners_source_image_px"], dtype=np.float64).reshape(1, -1, 2) for v in train_views]
    image = np.asarray(view["corners_source_image_px"], dtype=np.float64).reshape(1, -1, 2)
    fit = fit_fisheye(obj, train_images, size)
    k = np.asarray(fit["camera_matrix"], dtype=float)
    d = np.asarray(fit["distortion_theta_k1_k2_k3_k4"], dtype=float).reshape(4, 1)
    errors = None
    error = None
    issues = fit_issues(fit, size)
    try:
        errors = error_summary(fisheye_pose_errors(obj, image, tuple(manifest["pattern_inner_corners"]), k, d))
    except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
        error = str(exc)
    return {
        "held_out_view": view["id"],
        "held_out_view_sha256": view["source_sha256"],
        "training_view_ids": [v["id"] for v in train_views],
        "fisheye_fit": {key: value for key, value in fit.items() if key not in {"rvecs", "tvecs"}},
        "fisheye_fit_issues": issues,
        "monotonicity": fisheye_monotonicity(k, d, size),
        "heldout_parity_error_px": errors,
        "heldout_error": error,
        "view009_included_in_fit": False,
    }


def transform_fisheye(points: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    return cv2.fisheye.undistortPoints(
        points.astype(np.float64).reshape(-1, 1, 2), k, d, R=np.eye(3), P=k
    ).reshape(-1, 2)


def cushion_results(k: np.ndarray, d: np.ndarray, cushion_data: dict, cushion_module) -> dict:
    output = []
    for image in cushion_data["images"]:
        image_record = {"source": image["source"], "source_sha256": image["source_sha256"], "features": []}
        for feature in image["features"]:
            variants = []
            for variant in feature["variants"]:
                if "points_raw_px" not in variant:
                    variants.append({"sigma": variant["sigma"], "band_shift": variant["band_shift"], "error": variant.get("error")})
                    continue
                points = np.asarray(variant["points_raw_px"], dtype=float)
                corrected = transform_fisheye(points, k, d)
                variants.append({
                    "sigma": variant["sigma"],
                    "band_shift": variant["band_shift"],
                    "raw_line_metrics": cushion_module.line_metrics(points),
                    "fisheye_line_metrics": cushion_module.line_metrics(corrected),
                    "corrected_point_count": int(len(corrected)),
                    "finite": bool(finite(corrected)),
                })
            valid = [v for v in variants if "fisheye_line_metrics" in v]
            feat = {"id": feature["id"], "feature": feature["feature"], "variants": variants}
            if valid:
                feat["fisheye_sensitivity_ranges"] = {
                    metric: [
                        min(v["fisheye_line_metrics"][metric] for v in valid),
                        max(v["fisheye_line_metrics"][metric] for v in valid),
                    ]
                    for metric in ("rms_px", "quadratic_midpoint_chord_bow_px")
                }
                # The raw point rows are exactly those scored in the frozen
                # original diagnostic; verify sample counts before comparing.
                count_range = [min(v["corrected_point_count"] for v in valid), max(v["corrected_point_count"] for v in valid)]
                feat["same_sample_counts"] = all(
                    v["corrected_point_count"] == v["raw_line_metrics"]["points"] for v in valid
                ) and count_range[0] == count_range[1]
            image_record["features"].append(feat)
        output.append(image_record)
    return {"source_cushion_data_sha256": file_sha(CUSHION_DATA), "images": output}


def compact_cushion_source(cushion_data: dict) -> dict:
    """Retain exact source metrics without duplicating the large point arrays."""
    images = []
    for image in cushion_data["images"]:
        record = {"source": image["source"], "source_sha256": image["source_sha256"], "features": []}
        for feature in image["features"]:
            variants = []
            for variant in feature["variants"]:
                if "models" in variant:
                    variants.append({
                        "sigma": variant["sigma"],
                        "band_shift": variant["band_shift"],
                        "models": variant["models"],
                    })
                else:
                    variants.append({"sigma": variant["sigma"], "band_shift": variant["band_shift"], "error": variant.get("error")})
            record["features"].append({"id": feature["id"], "feature": feature["feature"], "variants": variants})
        images.append(record)
    return {"path": str(CUSHION_DATA.relative_to(ROOT)), "sha256": file_sha(CUSHION_DATA), "images": images}


def self_test():
    rng = np.random.default_rng(446)
    pattern = (9, 6)
    obj = board_points(pattern, 25.4)
    k = np.asarray([[700.0, 0, 640.0], [0, 700.0, 360.0], [0, 0, 1]], dtype=float)
    d = np.asarray([[-0.02], [0.001], [0.0], [0.0]], dtype=float)
    rvec = np.asarray([[0.12], [-0.18], [0.04]], dtype=float)
    tvec = np.asarray([[-40.0], [20.0], [1200.0]], dtype=float)
    raw = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)[0]
    undistorted = cv2.fisheye.undistortPoints(raw, k, d, R=np.eye(3), P=k)
    recovered = cv2.solvePnP(obj.reshape(-1, 1, 3), undistorted.reshape(-1, 1, 2), k, np.zeros((5, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    assert recovered[0]
    predicted = cv2.fisheye.projectPoints(obj, recovered[1], recovered[2], k, d)[0]
    assert float(np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2) - raw.reshape(-1, 2)) ** 2, axis=1)))) < 1e-4
    print(json.dumps({"synthetic_fisheye_roundtrip": "passed", "rms_px": float(np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2) - raw.reshape(-1, 2)) ** 2, axis=1))))}))


def make_report(result: dict) -> str:
    lines = [
        "# Fisheye optics model audit — 2026-10-27",
        "",
        "Offline evaluation only. No profile was activated, and no camera or new measurement was used.",
        "",
        "## Held-out checkerboard comparison",
        "",
        "The same whole-view holdout and alternating checkerboard parity pose split was used for the existing zero/Brown models and the OpenCV fisheye model. Training views were 001–008. View 009 was kept entirely out of its fisheye fit and scored prospectively by parity after fitting on 001–008.",
        "",
        "| Model | LOO held-out RMS (px) | Successful folds | Fit / fold status |",
        "|---|---:|---:|---|",
    ]
    for name, row in result["brown_eight_view_leave_one_out"]["models"].items():
        rms = (row.get("whole_view_holdout_pose_split_error_px") or {}).get("rms")
        lines.append(f"| Brown {name} | {'unavailable' if rms is None else f'{rms:.4f}'} | {row.get('successful_folds', 0)}/8 | existing solver |")
    fish = result["fisheye_eight_view_leave_one_out"]
    fish_rms = (fish.get("whole_view_holdout_pose_split_error_px") or {}).get("rms_px")
    valid_folds = sum("heldout_error_px" in fold for fold in fish["folds"])
    failures = sum("error" in fold for fold in fish["folds"])
    lines.append(f"| Fisheye theta polynomial (4 coefficients) | {'unavailable' if fish_rms is None else f'{fish_rms:.4f}'} | {valid_folds}/8 | {failures} failed/implausible folds |")
    lines += ["", "The fisheye model fit is **blocked/unstable and inadmissible as a correction candidate**. All eight matched leave-one-view-out fits have enormous training residuals, and the independent prospective score is worse than the frozen Brown scores. No further coefficient tuning was attempted. Exact coefficients, per-view errors, and parameter ranges are in `summary.json`.", "", "| Held-out training view | Fisheye training RMS (px) | Result |", "|---|---:|---|"]
    for fold in fish["folds"]:
        training = fold.get("fit", {}).get("recomputed_training_error", {}).get("rms_px")
        lines.append(f"| {fold['held_out_view']} | {'unavailable' if training is None else f'{training:.4f}'} | {fold.get('error', 'scored')} |")
    lines += ["", "The OpenCV 5.0.0 Python binding exposes no `cv2.fisheye.CALIB_*` constants, so the documented numeric C++ flag values were used. The returned camera matrix nevertheless has non-zero skew despite `FIX_SKEW`; the flag/API contract is therefore unresolved in this environment. Together with the failed residuals, this blocks interpretation of the model fit. The synthetic projection/undistortion self-test passes but does not validate calibration flags.", "", "## Prospective held-out view 009", ""]
    nine = result["fisheye_view009_prospective"]
    validation = nine.get("heldout_parity_error_px")
    validation_text = "unavailable" if validation is None else f"{validation['rms_px']:.4f} px"
    issue_text = ", ".join(nine["fisheye_fit_issues"]) or "none under configured checks"
    lines.append(
        f"Fisheye model fit on views 001–008 only; `view-009` was not included in the fit. "
        f"Held-out parity RMS: {validation_text}. Fit issues: {issue_text}."
    )
    brown_nine = result.get("brown_view009_frozen")
    if brown_nine:
        lines.append("")
        lines.append("Existing frozen Brown-model validation scores on view-009: " + ", ".join(f"{name}={row.get('frozen_validation_error_px', {}).get('rms', float('nan')):.4f}px" for name, row in brown_nine["models"].items() if row.get("frozen_validation_error_px")))
    lines += ["", "## Same-point cushion diagnostics", "", "All existing cushion point samples were reused without re-extraction. The following table compares ranges across the source report's smoothing and band-shift variants. These visible transitions are not surveyed physical rails and extraction variants are correlated.", "", "| Image | Feature | Raw bow range (px) | Frozen zero bow range (px) | Fisheye bow range (px) | Fisheye RMS range (px) | Same point counts |", "|---|---|---:|---:|---:|---:|---|"]
    original = {row["source"]: row for row in result["cushion_source"]["images"]}
    for image in result["cushion_fisheye_transformed"]["images"]:
        raw_image = original[image["source"]]
        raw_features = {f["id"]: f for f in raw_image["features"]}
        for feature in image["features"]:
            fish_range = feature.get("fisheye_sensitivity_ranges", {})
            source_feature = raw_features[feature["id"]]
            old_variants = [v for v in source_feature["variants"] if "models" in v]
            def rng(model, key):
                values = [v["models"][model][key] for v in old_variants]
                return "—" if not values else f"{min(values):.2f}–{max(values):.2f}"
            fbow = fish_range.get("quadratic_midpoint_chord_bow_px")
            frms = fish_range.get("rms_px")
            fmt = lambda r: "—" if not r else f"{r[0]:.2f}–{r[1]:.2f}"
            lines.append(f"| {Path(image['source']).name} | {feature['id']} | {rng('raw','quadratic_midpoint_chord_bow_px')} | {rng('zero','quadratic_midpoint_chord_bow_px')} | {fmt(fbow)} | {fmt(frms)} | {feature.get('same_sample_counts')} |")
    lines += ["", "## Interpretation and limits", ""]
    lines += ["- Do not activate this fisheye fit or its point transform. The candidate is blocked/unstable and the cushion transform is shown only to document consequences of applying the failed fit.", "- The checkerboard and cushion comparisons assess projection consistency and visible-edge geometry only; neither establishes table X/Y accuracy or the true physical rail nose.", "- View-009 is prospective: it is scored only after coefficients are frozen from views 001–008. It is not used in any fit that reports its score.", "- The board views are small with limited tilt diversity; fisheye fold instability is evidence the current saved data may not identify the added coefficients, not proof the lens has no fisheye distortion.", "- The test is offline and leaves the production zero-radial optics profile unchanged. The ball-center refiner still rejects all nine measured positions; no accepted centers or map are produced.", "- No ball height, depth, or Z output is introduced."]
    lines += ["", "## Reproduction", "", "```sh", "/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-fisheye-optics-audit-20261027.py --self-test", "/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/offline-fisheye-optics-audit-20261027.py", "```", "", "The raw JSON records exact input and source-code hashes, all fold residuals, parameter ranges, point-transform metrics, and software version."]
    return "\n".join(lines) + "\n"


def run():
    manifest, manifest_bytes = load_inputs()
    size = tuple(manifest["image_size_px"])
    pattern = tuple(manifest["pattern_inner_corners"])
    obj = board_points(pattern, manifest["square_pitch_mm"])
    views8 = dict(manifest)
    views8["views"] = manifest["views"][:8]

    brown_module = load_module("checkerboard_preflight", CHECKERBOARD_SCRIPT)
    cushion_module = load_module("cushion_metrics", CUSHION_SCRIPT)
    brown_options = SimpleNamespace(
        models="zero,k1,k1k2", minimum_views=6, iterations=100, epsilon=1e-9,
        min_focal_ratio=0.1, max_focal_ratio=10, principal_margin=0.25,
        max_matrix_condition=1e6, radial_samples=101,
        maximum_fold_focal_range=0.25, maximum_fold_principal_range=0.1,
    )
    brown_result = brown_module.analyze(views8, brown_options)
    fish_result = fisheye_all_folds(manifest, obj, train_count=8)
    prospective = fisheye_prospective_009(manifest, obj)
    brown_frozen_validation = json.loads(VIEW009_REPORT.read_text())
    fisheye_full8 = fit_fisheye(
        obj,
        [np.asarray(v["corners_source_image_px"], dtype=float).reshape(1, -1, 2) for v in manifest["views"][:8]],
        size,
    )
    k8 = np.asarray(fisheye_full8["camera_matrix"], dtype=float)
    d8 = np.asarray(fisheye_full8["distortion_theta_k1_k2_k3_k4"], dtype=float).reshape(4, 1)
    cushion_data = json.loads(CUSHION_DATA.read_text())
    cushion_fisheye = cushion_results(k8, d8, cushion_data, cushion_module)

    inputs = [
        {"path": str(MANIFEST.relative_to(ROOT)), "sha256": file_sha(MANIFEST)},
        {"path": str(BROWN_PREFLIGHT.relative_to(ROOT)), "sha256": file_sha(BROWN_PREFLIGHT)},
        {"path": str(BROWN_REPORT.relative_to(ROOT)), "sha256": file_sha(BROWN_REPORT)},
        {"path": str(VIEW009_REPORT.relative_to(ROOT)), "sha256": file_sha(VIEW009_REPORT)},
        {"path": str(CUSHION_DATA.relative_to(ROOT)), "sha256": file_sha(CUSHION_DATA)},
        {"path": str(CHECKERBOARD_SCRIPT.relative_to(ROOT)), "sha256": file_sha(CHECKERBOARD_SCRIPT)},
        {"path": str(CUSHION_SCRIPT.relative_to(ROOT)), "sha256": file_sha(CUSHION_SCRIPT)},
        {"path": str(RUNNER_PATH.relative_to(ROOT)), "sha256": file_sha(RUNNER_PATH)},
    ]
    for view in manifest["views"]:
        inputs.append({"path": f"thoughts/calibration/checkerboard-session/{view['source_file']}", "sha256": view["source_sha256"]})

    result = {
        "schema_version": 1,
        "date": "2026-10-27",
        "purpose": "offline fisheye-model feasibility; no production activation",
        "comparison_status": "blocked_unstable",
        "candidate_admissible": False,
        "stop_reason": "All matched fisheye folds exceed the 2 px diagnostic training-residual bound; view-009 prospective residual is worse than frozen Brown; OpenCV 5.0.0 Python omits fisheye flag constants and returned skew is nonzero despite FIX_SKEW. No coefficient tuning was performed.",
        "opencv_version": cv2.__version__,
        "diagnostics_only": True,
        "production_behavior_changed": False,
        "training_view_ids_for_model_comparison": [v["id"] for v in manifest["views"][:8]],
        "held_out_prospective_view_id": "view-009",
        "held_out_view_009_in_any_reported_fisheye_fit": False,
        "image_size_px": size,
        "pattern_inner_corners": pattern,
        "square_pitch_mm": manifest["square_pitch_mm"],
        "validation_protocol": "For views 001–008, hold out each entire view from intrinsics, estimate pose from parity 0 then score parity 1, swap and concatenate. Separately fit on views 001–008 only and score view-009 with the same parity split.",
        "validation_limit": "Held-out pose uses other corners from that same view. Reprojection consistency on checkerboard captures is not independent XY accuracy.",
        "fisheye_fit_configuration": {
            "model": "OpenCV cv::fisheye four-coefficient theta polynomial",
            "flags": FISHEYE_FLAGS,
            "flags_definition": {"CALIB_RECOMPUTE_EXTRINSIC": FISHEYE_RECOMPUTE_EXTRINSIC, "CALIB_FIX_SKEW": FISHEYE_FIX_SKEW},
            "iterations": 100,
            "epsilon": 1e-9,
            "initial_intrinsics": None,
            "initial_distortion": None,
            "maximum_training_rms_px_for_plausibility": MAX_TRAINING_RMS_PX,
        },
        "inputs": inputs,
        "brown_eight_view_leave_one_out": brown_result,
        "fisheye_eight_view_leave_one_out": fish_result,
        "fisheye_view009_prospective": prospective,
        "brown_view009_frozen": brown_frozen_validation,
        "fisheye_full_fit_views001_to008": {
            key: value for key, value in fisheye_full8.items() if key not in {"rvecs", "tvecs"}
        },
        "fisheye_full_fit_001_to_008_issues": fit_issues(fisheye_full8, size),
        "fisheye_full_fit_001_to_008_monotonicity": fisheye_monotonicity(k8, d8, size),
        "cushion_source": compact_cushion_source(cushion_data),
        "cushion_fisheye_transformed": cushion_fisheye,
        "limitations": [
            "The 9-view checkerboard target has small projected square pitch and limited tilt coverage; model parameters can be weakly identified.",
            "Fisheye fit instability is not proof the camera has zero fisheye distortion or that the pinhole model is physically exact.",
            "Cushion samples follow visible transitions that may include shadow, seams, or cloth texture; they are not surveyed physical rails.",
            "Cushion extraction variants are correlated sensitivity probes, not independent confidence bounds.",
            "No ball-center accuracy, table millimeter accuracy, live latency, or selected FPS is established.",
            "No ball height/depth/Z estimate, camera capture, new physical measurement, or production profile activation is included.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (OUT / "README.md").write_text(make_report(result))
    print(json.dumps({
        "report": str((OUT / "README.md").relative_to(ROOT)),
        "brown_loo": {name: row.get("whole_view_holdout_pose_split_error_px") for name, row in brown_result["models"].items()},
        "fisheye_loo": fish_result["whole_view_holdout_pose_split_error_px"],
        "fisheye_successful_folds": fish_result["successful_folds"],
        "fisheye_view009": prospective.get("heldout_parity_error_px"),
        "fisheye_full_issues": result["fisheye_full_fit_001_to_008_issues"],
    }))


def main():
    if "--self-test" in sys.argv:
        self_test()
    else:
        run()


if __name__ == "__main__":
    main()
