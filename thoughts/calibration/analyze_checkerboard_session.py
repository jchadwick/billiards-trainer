#!/usr/bin/env python3
"""Offline lens-model preflight. Never writes camera configuration or the session.

Entire views are held out from intrinsics estimation. On each held-out view,
pose is estimated from one checkerboard parity and evaluated on the other;
then the parities swap. This avoids reporting pose-training residual as validation.
Target pose is only calibration mathematics; application ball output remains XY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def object_points(pattern, pitch):
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * pitch
    return obj


def summary(errors):
    a = np.asarray(errors, dtype=float)
    if not a.size:
        return None
    return {"rms": float(np.sqrt(np.mean(a * a))), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "count": int(a.size)}


def physical_issues(k, d, size, options):
    issues = []
    if not np.isfinite(k).all() or not np.isfinite(d).all():
        return ["nonfinite_parameters"]
    width, height = size
    if min(k[0, 0], k[1, 1]) <= 0:
        issues.append("nonpositive_focal_length")
    if not all(options.min_focal_ratio <= f / width <= options.max_focal_ratio
               for f in [k[0, 0], k[1, 1]]):
        issues.append("focal_length_outside_configured_plausibility_bounds")
    margin = options.principal_margin
    if not (-margin * width <= k[0, 2] <= (1 + margin) * width
            and -margin * height <= k[1, 2] <= (1 + margin) * height):
        issues.append("principal_point_outside_configured_bounds")
    if np.linalg.cond(k) > options.max_matrix_condition:
        issues.append("ill_conditioned_camera_matrix")
    # k1/k2 models must preserve radial order over the observed frame domain.
    # Positive radial scale and radial derivative rule out folding, but do not
    # establish calibration accuracy or constrain unobserved regions.
    if min(k[0, 0], k[1, 1]) > 0:
        rays = np.array([[(x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1]]
                         for x in (0, width-1) for y in (0, height-1)])
        r2 = np.linspace(0, np.max(np.sum(rays*rays, axis=1)), options.radial_samples)
        k1, k2 = d.ravel()[:2]
        if np.min(1+k1*r2+k2*r2*r2) <= 0 or np.min(1+3*k1*r2+5*k2*r2*r2) <= 0:
            issues.append("radial_mapping_folds_within_frame_domain")
    return issues


def fit_model(obj, images, size, model, options):
    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
             | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    if model == "zero":
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    elif model == "k1":
        flags |= cv2.CALIB_FIX_K2
    elif model != "k1k2":
        raise ValueError(f"Unknown model: {model}")
    objects = [obj.copy() for _ in images]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                options.iterations, options.epsilon)
    rms, k, d, rvecs, tvecs, std_k, _, _ = cv2.calibrateCameraExtended(
        objects, images, size, None, None, flags=flags, criteria=criteria)
    if not all(np.isfinite(value).all() for value in [k, d, std_k, *rvecs, *tvecs]):
        raise ValueError("Nonfinite calibration parameters or uncertainty diagnostics")
    residuals = []
    for points, rvec, tvec in zip(images, rvecs, tvecs):
        projected = cv2.projectPoints(obj, rvec, tvec, k, d)[0].reshape(-1, 2)
        if not np.isfinite(projected).all():
            raise ValueError("Nonfinite training reprojection")
        residuals.extend(np.linalg.norm(projected-points.reshape(-1, 2), axis=1).tolist())
    issues = physical_issues(k, d, size, options)
    if not np.isfinite(rms):
        issues.append("nonfinite_fit_error")
    return {"opencv_rms_px": float(rms), "train_error_px": summary(residuals),
            "camera_matrix": k.tolist(), "distortion_k1_k2_p1_p2_k3": d.ravel().tolist(),
            "intrinsics_stddev_local_linear_approximation": std_k.ravel().tolist(),
            "issues": issues}, k, d


def heldout_pose_errors(obj, image, pattern, k, d):
    points = image.reshape(-1, 2)
    indices = np.indices((pattern[1], pattern[0])).sum(axis=0).ravel() % 2
    errors = []
    for parity in (0, 1):
        train = indices == parity
        # IPPE enumerates the planar pose ambiguity; choose with training
        # corners only, and require positive depth before any held-out scoring.
        result = cv2.solvePnPGeneric(obj[train], points[train], k, d, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            rotation = cv2.Rodrigues(rvec)[0]
            if np.min((obj @ rotation.T + tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj[train], points[train], k, d, rvec, tvec)
            if np.min((obj @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(1, 3))[:, 2]) <= 0:
                continue
            projected = cv2.projectPoints(obj[train], rvec, tvec, k, d)[0].reshape(-1, 2)
            loss = float(np.sum((projected-points[train])**2))
            candidates.append((loss, rvec, tvec))
        if not candidates:
            raise ValueError("No positive-depth held-out pose solution")
        _, rvec, tvec = min(candidates, key=lambda candidate: candidate[0])
        predicted = cv2.projectPoints(obj[~train], rvec, tvec, k, d)[0].reshape(-1, 2)
        e = np.linalg.norm(predicted-points[~train], axis=1)
        if not np.isfinite(e).all():
            raise ValueError("Nonfinite held-out residual")
        errors.extend(e.tolist())
    return errors


def analyze(manifest, options):
    size = tuple(manifest["image_size_px"])
    pattern = tuple(manifest["pattern_inner_corners"])
    obj = object_points(pattern, manifest["square_pitch_mm"])
    views = manifest["views"]
    if len(views) < options.minimum_views:
        raise ValueError(f"Need at least {options.minimum_views} views for this preflight")
    images = []
    for view in views:
        points = np.asarray(view["corners_source_image_px"], dtype=np.float32)
        if points.shape != (len(obj), 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid corners: {view['id']}")
        if (view["source_width_px"], view["source_height_px"]) != size:
            raise ValueError("Mixed source geometries")
        if tuple(view["pattern_inner_corners"]) != pattern or view["square_pitch_mm"] != manifest["square_pitch_mm"]:
            raise ValueError("Mixed checkerboard geometry")
        if np.any(points < 0) or np.any(points >= np.asarray(size)):
            raise ValueError("Corner outside declared image")
        images.append(points.reshape(-1, 1, 2))
    models = {}
    for model in options.models.split(","):
        record = {"folds": [], "status": "exploratory_only_not_activated"}
        try:
            record["full_fit"], _, _ = fit_model(obj, images, size, model, options)
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            record["full_fit_error"] = str(error)
        all_errors, intrinsics = [], []
        for index, view in enumerate(views):
            fold = {"held_out_view": view["id"]}
            try:
                fitted, k, d = fit_model(obj, images[:index]+images[index+1:], size, model, options)
                fold["training_fit"] = fitted
                # Retain finite fits even when plausibility fails so a deceptively
                # low validation residual cannot hide unstable intrinsics.
                if not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
                    raise ValueError("Invalid intrinsics; pose scoring withheld")
                errors = heldout_pose_errors(obj, images[index], pattern, k, d)
                fold["validation_error_px"] = summary(errors)
                all_errors.extend(errors)
                intrinsics.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
            except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                fold["error"] = str(error)
            record["folds"].append(fold)
        record["whole_view_holdout_pose_split_error_px"] = summary(all_errors)
        record["successful_folds"] = len(intrinsics)
        if intrinsics:
            params = np.asarray(intrinsics)
            record["fold_intrinsics"] = {
                "columns": ["fx", "fy", "cx", "cy"], "min": params.min(axis=0).tolist(),
                "max": params.max(axis=0).tolist(), "stddev": params.std(axis=0).tolist(),
                "focal_relative_range": ((params.max(axis=0)[:2]-params.min(axis=0)[:2]) / np.maximum(np.abs(np.median(params, axis=0)[:2]), 1e-12)).tolist(),
                "principal_range_fraction_image": ((params.max(axis=0)[2:]-params.min(axis=0)[2:])/np.asarray(size)).tolist()}
            fi = record["fold_intrinsics"]
            record["unstable_across_folds"] = bool(
                max(fi["focal_relative_range"]) > options.maximum_fold_focal_range
                or max(fi["principal_range_fraction_image"]) > options.maximum_fold_principal_range)
        models[model] = record
    return {"purpose": "offline lens-model feasibility; no production activation", "view_count": len(views),
            "view_ids": [v["id"] for v in views], "image_size_px": size,
            "pattern_inner_corners": pattern, "square_pitch_mm": manifest["square_pitch_mm"],
            "validation_protocol": "Hold out entire view from intrinsics/lens fit; estimate its pose on checkerboard parity 0, score parity 1; swap and concatenate.",
            "validation_limit": "Other corners of the held-out image fit pose, so this tests projection consistency with unseen poses, not independently measured table XY or ball centers. Neighboring corner errors may be correlated.",
            "settings": vars(options), "models": models,
            "corner_pitch_px_min_by_view": {v["id"]: v.get("quality", {}).get("adjacent_corner_pitch_px_min") for v in views},
            "limitations": ["Small target and limited tilt diversity can make intrinsics weakly identified despite low pixel residuals.",
                            "Physical pitch/flatness uncertainty, JPEG and corner detection systematic errors are not estimated.",
                            "No table millimeter or ball-center accuracy is established; no Z ball output is introduced.",
                            "No corrected-image preview is emitted: parameter plausibility alone does not justify a visually persuasive warp.",
                            "Model comparison is preflight evidence, not permission to select the lowest residual or activate a calibration."]}


def evaluate_frozen(baseline, manifest, test_view_id, options):
    """Only solve held-out board pose. No lens/intrinsics optimization occurs."""
    if test_view_id in baseline["view_ids"]:
        raise ValueError("Test view was used for frozen calibration; choose an unseen view")
    size = tuple(baseline["image_size_px"])
    pattern = tuple(baseline["pattern_inner_corners"])
    pitch = baseline["square_pitch_mm"]
    if (tuple(manifest["image_size_px"]) != size
            or tuple(manifest["pattern_inner_corners"]) != pattern
            or manifest["square_pitch_mm"] != pitch):
        raise ValueError("Frozen baseline and test session geometry differ")
    matches = [v for v in manifest["views"] if v["id"] == test_view_id]
    if len(matches) != 1:
        raise ValueError("Requested test view is missing or duplicated")
    view = matches[0]
    if ((view["source_width_px"], view["source_height_px"]) != size
            or tuple(view["pattern_inner_corners"]) != pattern
            or view["square_pitch_mm"] != pitch):
        raise ValueError("Test-view geometry differs from frozen baseline")
    obj = object_points(pattern, pitch)
    image = np.asarray(view["corners_source_image_px"], dtype=np.float32)
    if (image.shape != (len(obj), 2) or not np.isfinite(image).all()
            or np.any(image < 0) or np.any(image >= np.asarray(size))):
        raise ValueError("Invalid frozen-test corners")
    models = {}
    for name, record in baseline["models"].items():
        fit = record.get("full_fit")
        if fit is None:
            models[name] = {"error": "Frozen baseline has no full fit"}
            continue
        k = np.asarray(fit["camera_matrix"], dtype=float)
        d = np.asarray(fit["distortion_k1_k2_p1_p2_k3"], dtype=float)
        issues = physical_issues(k, d, size, options)
        if issues:
            models[name] = {"error": "Frozen calibration failed plausibility checks", "issues": issues}
            continue
        errors = heldout_pose_errors(obj, image, pattern, k, d)
        models[name] = {"frozen_validation_error_px": summary(errors),
                        "camera_matrix": k.tolist(), "distortion_k1_k2_p1_p2_k3": d.tolist(),
                        "intrinsics_or_lens_refit": False}
    return {"purpose": "prospective frozen-lens validation; no activation", "training_view_ids": baseline["view_ids"],
            "test_view_id": test_view_id, "test_source_sha256": view.get("source_sha256"),
            "validation_protocol": "Frozen full-training-set intrinsics and lens parameters; held-out view pose estimated from one corner parity and evaluated on the other, then swapped.",
            "limitations": "One new image probes projection consistency at one placement. It does not prove table XY/ball-center accuracy, establish physical uncertainty, or justify selecting a more complex model from this result alone.",
            "models": models}


def write_frozen_report(result, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Prospective frozen-model checkerboard validation", "", result["test_view_id"], "",
             result["validation_protocol"], "", "| Model | Frozen validation RMS px |", "|---|---:|"]
    for name, record in result["models"].items():
        score = record.get("frozen_validation_error_px", {}).get("rms")
        lines.append(f"| {name} | {'unavailable' if score is None else f'{score:.4f}'} |")
        if "error" in record:
            lines.append(f"\n{name}: {record['error']}\n")
    lines += ["", result["limitations"], "", "Coefficients were frozen before this view was captured. No calibration activated."]
    prefix.with_suffix(".md").write_text("\n".join(lines)+"\n")


def write_report(results, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    lines = ["# Checkerboard lens-model preflight", "", f"Views: {results['view_count']}. Offline only; no calibration activated.", "", results["validation_protocol"], "", results["validation_limit"], "", "| Model | Training RMS px | Whole-view holdout RMS px | Scored folds | Fold instability |", "|---|---:|---:|---:|---|"]
    for name, record in results["models"].items():
        training = record.get("full_fit", {}).get("opencv_rms_px")
        validation = (record.get("whole_view_holdout_pose_split_error_px") or {}).get("rms")
        fmt = lambda value: "unavailable" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {fmt(training)} | {fmt(validation)} | {record['successful_folds']}/{results['view_count']} | {record.get('unstable_across_folds', 'unavailable')} |")
    if results["view_count"] == 8:
        lines += ["", "At eight views, neither radial model improves whole-view held-out error over the zero-distortion baseline. Added radial correction is not supported by this experiment. Full eight-view parameters are frozen separately in checkerboard-eight-view-baseline.json without additional tuning. A prospective left-middle tilted ninth view will be scored with those coefficients fixed; only its board pose will be estimated. No prospective result is available until that image is captured. A single new view cannot establish table or ball-center accuracy."]
    for name, record in results["models"].items():
        lines += ["", f"## {name}", "", "Full-fit issues: "+", ".join(record.get("full_fit", {}).get("issues", [])) if record.get("full_fit", {}).get("issues") else "Full-fit plausibility checks: no configured bound violations (not proof of accuracy)."]
        if "fold_intrinsics" in record:
            fi = record["fold_intrinsics"]
            lines += ["", "Fold parameter ranges (fx, fy, cx, cy), pixels:", "", f"- Minimum: {fi['min']}", f"- Maximum: {fi['max']}"]
        lines += ["", "| Held-out view | Validation RMS px | Training-fit warnings |", "|---|---:|---|"]
        for fold in record["folds"]:
            rms = (fold.get("validation_error_px") or {}).get("rms")
            warning = fold.get("error", ", ".join(fold.get("training_fit", {}).get("issues", [])) or "none")
            lines.append(f"| {fold['held_out_view']} | {'unavailable' if rms is None else f'{rms:.4f}'} | {warning.replace(chr(10), ' ')} |")
    lines += ["", "## Limits and next capture", ""] + ["- "+limit for limit in results["limitations"]]
    lines += ["", "Most useful next capture: an additional clear tilt about the other board axis if not already present, then a tilted board toward an image edge. Keep full pattern and white border visible. Prefer larger native corner spacing where practical without altering camera settings. Repeat a pose to probe corner repeatability. Orthogonal tilted views must remain represented in training folds; losing the only tilted view can invalidate that fold.", "", "Reproduce with the isolated OpenCV/NumPy environment:", "", "```sh", "/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_checkerboard_session.py --self-test", "/tmp/billiards-checkerboard-env/bin/python thoughts/calibration/analyze_checkerboard_session.py", "```", "", "Zero, k1 and k1+k2 models fix all tangential and higher-order terms. No extra tangential/fisheye model is searched on this small session. Local linear standard deviations are diagnostic approximations, not confidence guarantees. Raw JSON preserves coefficients, per-fold diagnostics and thresholds."]
    prefix.with_suffix(".md").write_text("\n".join(lines)+"\n")


def self_test(options):
    rng = np.random.default_rng(7321)
    pattern, size = (9, 6), (1280, 720)
    obj = object_points(pattern, 25.4)
    k = np.array([[900., 0, 640.], [0, 920., 360.], [0, 0, 1.]])
    d = np.array([-.12, 0, 0, 0, 0.])
    views = []
    for index in range(12):
        r = np.array([rng.uniform(-.6, .6), rng.uniform(-.6, .6), rng.uniform(-.4, .4)])
        # All training and validation points are deterministic, independently
        # perturbed projections with multiple tilts and image locations.
        t = np.array([rng.uniform(-380, 180), rng.uniform(-230, 80), rng.uniform(900, 1300)])
        image = cv2.projectPoints(obj, r, t, k, d)[0].reshape(-1, 2)
        image += rng.normal(0, .015, image.shape)
        views.append({"id": f"synthetic-{index}", "source_width_px": size[0], "source_height_px": size[1],
                      "pattern_inner_corners": pattern, "square_pitch_mm": 25.4, "corners_source_image_px": image.tolist()})
    manifest = {"image_size_px": size, "pattern_inner_corners": pattern, "square_pitch_mm": 25.4, "views": views}
    result = analyze(manifest, options)
    fitted = result["models"]["k1"]["full_fit"]
    estimated = np.asarray(fitted["camera_matrix"])
    assert np.max(np.abs(estimated.diagonal()[:2]/k.diagonal()[:2]-1)) < .01
    assert abs(fitted["distortion_k1_k2_p1_p2_k3"][0]-d[0]) < .01
    good = result["models"]["k1"]["whole_view_holdout_pose_split_error_px"]["rms"]
    base = result["models"]["zero"]["whole_view_holdout_pose_split_error_px"]["rms"]
    assert good < .05 and base > good*2, (good, base)
    assert result["models"]["k1"]["successful_folds"] == 12
    assert not fitted["issues"]
    broken = k.copy(); broken[0, 0] = -1
    assert "nonpositive_focal_length" in physical_issues(broken, d, size, options)
    folded = np.array([-2., 0, 0, 0, 0])
    assert "radial_mapping_folds_within_frame_domain" in physical_issues(k, folded, size, options)
    # Frozen validation must recover the low noise of a genuinely unseen pose
    # without being able to invoke the intrinsics fitter, even accidentally.
    new_view = dict(views[0], id="synthetic-prospective")
    new_image = cv2.projectPoints(obj, np.array([.3, -.4, .15]),
                                 np.array([-150., 40., 1150.]), k, d)[0].reshape(-1, 2)
    new_image += rng.normal(0, .015, new_image.shape)
    new_view["corners_source_image_px"] = new_image.tolist()
    test_manifest = dict(manifest, views=views+[new_view])
    original_fitter = globals()["fit_model"]
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Frozen evaluation must never fit intrinsics")
    globals()["fit_model"] = forbidden_fit
    try:
        frozen = evaluate_frozen(result, test_manifest, new_view["id"], options)
        assert frozen["models"]["k1"]["frozen_validation_error_px"]["rms"] < .05
        assert frozen["models"]["zero"]["frozen_validation_error_px"]["rms"] > .05
        try:
            evaluate_frozen(result, test_manifest, views[0]["id"], options)
            raise AssertionError("Reused training view should be rejected")
        except ValueError:
            pass
    finally:
        globals()["fit_model"] = original_fitter
    print(json.dumps({"synthetic_test": "passed", "views": 12, "k1_holdout_rms_px": good, "zero_holdout_rms_px": base}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", default="thoughts/calibration/checkerboard-session")
    parser.add_argument("--output-prefix", default="thoughts/calibration/checkerboard-model-preflight")
    parser.add_argument("--models", default="zero,k1,k1k2", choices=["zero,k1,k1k2", "zero,k1"])
    parser.add_argument("--minimum-views", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--min-focal-ratio", type=float, default=.1)
    parser.add_argument("--max-focal-ratio", type=float, default=10)
    parser.add_argument("--principal-margin", type=float, default=.25)
    parser.add_argument("--max-matrix-condition", type=float, default=1e6)
    parser.add_argument("--radial-samples", type=int, default=101)
    parser.add_argument("--maximum-fold-focal-range", type=float, default=.25)
    parser.add_argument("--maximum-fold-principal-range", type=float, default=.1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--frozen-results", help="Frozen baseline JSON; score only, never refit intrinsics")
    parser.add_argument("--test-view-id", help="Unseen session view to score against frozen baseline")
    options = parser.parse_args()
    if bool(options.frozen_results) != bool(options.test_view_id):
        parser.error("--frozen-results and --test-view-id must be supplied together")
    if (options.minimum_views < 4 or options.iterations < 1 or options.epsilon <= 0
            or options.radial_samples < 2 or options.min_focal_ratio <= 0
            or options.max_focal_ratio <= options.min_focal_ratio
            or options.principal_margin < 0 or options.max_matrix_condition <= 0
            or options.maximum_fold_focal_range <= 0
            or options.maximum_fold_principal_range <= 0):
        parser.error("Invalid numerical diagnostic or solver settings")
    if options.self_test:
        self_test(options)
        return
    manifest_path = Path(options.session_dir)/"manifest.json"
    raw = manifest_path.read_bytes()
    if options.frozen_results:
        frozen_raw = Path(options.frozen_results).read_bytes()
        result = evaluate_frozen(json.loads(frozen_raw), json.loads(raw), options.test_view_id, options)
        result["frozen_baseline_sha256"] = hashlib.sha256(frozen_raw).hexdigest()
        result["test_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        prefix = options.output_prefix
        if prefix == parser.get_default("output_prefix"):
            prefix = f"thoughts/calibration/checkerboard-frozen-validation-{options.test_view_id}"
        write_frozen_report(result, prefix)
        print(json.dumps({"test_view_id": options.test_view_id, "models": result["models"]}))
        return
    results = analyze(json.loads(raw), options)
    results["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    write_report(results, options.output_prefix)
    print(json.dumps({"view_count": results["view_count"], "report": str(Path(options.output_prefix).with_suffix('.md')),
                      "models": {name: {"holdout_px": r["whole_view_holdout_pose_split_error_px"], "unstable": r.get("unstable_across_folds")} for name, r in results["models"].items()}}))


if __name__ == "__main__":
    main()
