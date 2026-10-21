#!/usr/bin/env python3
"""Compare a retained YOLO reference with a model or recorded Rust output.

This utility compares detection outputs, with optional CVAT image-box scoring.
It does not measure real-time
performance, physical localization accuracy, tracking, cue geometry, or
trajectory equivalence. The reference model and one candidate source must be
selected explicitly so a failed model load cannot silently become an
OpenCV-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

DEFAULT_OUTPUT_DIRECTORY = "thoughts/performance-audits/equivalence-tests"
DEFAULT_FRAME_LIMIT = 100
DEFAULT_IMAGE_SIZE = 1280
DEFAULT_CONFIDENCE = 0.4
DEFAULT_IOU = 0.45
DEFAULT_MATCH_TOLERANCE = 2.0
DEFAULT_GROUND_TRUTH_RADIUS_PX = 20.0
DEFAULT_MAX_TRACE_FRAMES = 100
DEFAULT_MAX_DETECTIONS = 300
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
RUST_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
RUST_RESULT_SCHEMA_VERSION = 1
RUST_RESULT_WORKLOAD = "recorded_static_images_unpaced"
RUST_PACED_RESULT_WORKLOAD = "recorded_images_paced_latest_frame"
BATCH_COORDINATE_ROUNDING_TOLERANCE_PX = 1e-3
BATCH_CONFIDENCE_ROUNDING_TOLERANCE = 1e-6
DETECTION_BATCH_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "perception/contracts/detection-batch/v1/detection-batch.schema.json"
)

logger = logging.getLogger("backend_equivalence")


def _sha256_path(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a model file or package using stable relative component names."""
    digest = hashlib.sha256()
    if path.is_file():
        file_digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                file_digest.update(chunk)
        value = file_digest.hexdigest()
        return value, [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": value}
        ]

    components = []
    if not path.is_dir():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    for component in sorted(item for item in path.rglob("*") if item.is_file()):
        component_hash, _ = _sha256_path(component)
        relative = component.relative_to(path).as_posix()
        components.append(
            {
                "path": relative,
                "size_bytes": component.stat().st_size,
                "sha256": component_hash,
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(component_hash))
        digest.update(b"\0")
    if not components:
        raise ValueError(f"Model package is empty: {path}")
    return digest.hexdigest(), components


def _manifest_evidence(model_path: Path) -> dict[str, Any]:
    """Include an adjacent exporter manifest when one accompanies the model."""
    manifest_path = Path(f"{model_path}.manifest.json")
    if not manifest_path.is_file():
        return {"status": "not_present"}
    manifest_hash, _ = _sha256_path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read model manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Model manifest must contain a JSON object: {manifest_path}")
    return {
        "status": "present",
        "path": str(manifest_path),
        "sha256": manifest_hash,
        "contents": manifest,
    }


def _model_class_names(model: Any) -> dict[int, str]:
    """Normalize Ultralytics' label map to stable integer IDs."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            return {int(class_id): str(name) for class_id, name in names.items()}
        except (TypeError, ValueError) as error:
            raise ValueError(f"Model has an invalid class map: {names!r}") from error
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    raise ValueError("Model does not expose a usable class-name mapping")


def _extract_detections(
    result: Any, *, include_bounds: bool = False
) -> list[dict[str, Any]]:
    """Convert Ultralytics boxes to a small, JSON-compatible comparison form."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    centers = boxes.xywh[:, :2].cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    detections: list[dict[str, Any]] = [
        {
            "center": [float(center[0]), float(center[1])],
            "class_id": int(class_id),
            "confidence": float(confidence),
        }
        for center, class_id, confidence in zip(centers, classes, confidences)
    ]
    if include_bounds:
        for detection, bounds in zip(detections, boxes.xyxy.cpu().numpy()):
            detection["bbox_xyxy_px"] = [float(value) for value in bounds]
    detections.sort(
        key=lambda detection: (
            detection["class_id"],
            detection["center"][0],
            detection["center"][1],
            -detection["confidence"],
        )
    )
    return detections


def compare_detections(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    class_names: dict[int, str],
    match_tolerance_px: float,
) -> dict[str, Any]:
    """Match same-class centers and report unmatched detections explicitly.

    The tolerance is a comparison association radius, not a production accuracy
    guarantee. A match farther away is represented as two unmatched detections.
    """
    if not np.isfinite(match_tolerance_px) or match_tolerance_px <= 0:
        raise ValueError("match_tolerance_px must be finite and positive")

    matches = []
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    class_ids = sorted(
        {item["class_id"] for item in reference}
        | {item["class_id"] for item in candidate}
    )
    for class_id in class_ids:
        reference_indices = [
            index
            for index, item in enumerate(reference)
            if item["class_id"] == class_id
        ]
        candidate_indices = [
            index
            for index, item in enumerate(candidate)
            if item["class_id"] == class_id
        ]
        if not reference_indices or not candidate_indices:
            continue

        distances = np.empty((len(reference_indices), len(candidate_indices)))
        for row, reference_index in enumerate(reference_indices):
            reference_center = np.asarray(reference[reference_index]["center"])
            for column, candidate_index in enumerate(candidate_indices):
                candidate_center = np.asarray(candidate[candidate_index]["center"])
                distances[row, column] = float(
                    np.linalg.norm(reference_center - candidate_center)
                )

        # Penalize out-of-radius edges above every possible in-radius total so
        # the assignment first maximizes valid matches, then minimizes error.
        penalty = (match_tolerance_px + 1.0) * (
            min(len(reference_indices), len(candidate_indices)) + 1
        )
        assignment_costs = np.where(distances <= match_tolerance_px, distances, penalty)
        rows, columns = linear_sum_assignment(assignment_costs)
        for row, column in zip(rows, columns):
            reference_index = reference_indices[int(row)]
            candidate_index = candidate_indices[int(column)]
            center_error = float(distances[row, column])
            if center_error > match_tolerance_px:
                continue
            reference_detection = reference[reference_index]
            candidate_detection = candidate[candidate_index]
            matches.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"unknown_{class_id}"),
                    "reference_index": reference_index,
                    "candidate_index": candidate_index,
                    "reference_center": reference_detection["center"],
                    "candidate_center": candidate_detection["center"],
                    "center_error_px": center_error,
                    "confidence_abs_error": abs(
                        reference_detection["confidence"]
                        - candidate_detection["confidence"]
                    ),
                }
            )
            used_reference.add(reference_index)
            used_candidate.add(candidate_index)

    unmatched_reference = [
        {"index": index, **detection}
        for index, detection in enumerate(reference)
        if index not in used_reference
    ]
    unmatched_candidate = [
        {"index": index, **detection}
        for index, detection in enumerate(candidate)
        if index not in used_candidate
    ]
    matches.sort(key=lambda item: (item["class_id"], item["reference_index"]))
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched_count": len(matches),
        "unmatched_reference": unmatched_reference,
        "unmatched_candidate": unmatched_candidate,
        "matches": matches,
    }


def _normalized_annotation_name(name: str) -> str:
    """Normalize platform-dependent CVAT separators without discarding folders."""
    return "/".join(
        part for part in name.replace("\\", "/").split("/") if part and part != "."
    )


def _load_cvat_ground_truth(
    path: Path, class_names: dict[int, str]
) -> list[dict[str, Any]]:
    """Load CVAT image boxes in original image-pixel coordinates.

    Labels must match model class names exactly. Unsupported geometry is
    rejected rather than silently turning annotated objects into false positives.
    Explicitly empty image annotations are valid negative examples. Video tracks
    need a separate source-time mapping and are not guessed here.
    """
    try:
        root = ET.fromstring(path.read_bytes())
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"Unable to read CVAT XML {path}: {error}") from error
    if root.tag != "annotations" or root.findall("track"):
        raise ValueError(
            "Ground truth requires CVAT image annotations, not video tracks"
        )
    images = root.findall("image")
    if not images:
        raise ValueError("CVAT XML contains no image annotations")
    label_ids = {name: class_id for class_id, name in class_names.items()}
    if len(label_ids) != len(class_names):
        raise ValueError("Ground-truth scoring requires unique model class names")
    seen_names: set[str] = set()
    annotations = []
    for image in images:
        name = _normalized_annotation_name(image.get("name", ""))
        if not name or name in seen_names:
            raise ValueError(f"CVAT image name is empty or duplicated: {name!r}")
        seen_names.add(name)
        try:
            width = int(image.attrib["width"])
            height = int(image.attrib["height"])
            if width <= 0 or height <= 0:
                raise ValueError("dimensions must be positive")
        except (KeyError, ValueError) as error:
            raise ValueError(f"CVAT image {name!r} has invalid dimensions") from error
        detections = []
        for shape in image:
            if shape.tag != "box":
                raise ValueError(f"Unsupported CVAT geometry {shape.tag!r} in {name!r}")
            label = shape.get("label", "")
            if label not in label_ids:
                raise ValueError(
                    f"CVAT label {label!r} is absent from the model class map"
                )
            try:
                bounds = [
                    float(shape.attrib[key]) for key in ("xtl", "ytl", "xbr", "ybr")
                ]
                left, top, right, bottom = bounds
                if not all(math.isfinite(value) for value in bounds):
                    raise ValueError("coordinates must be finite")
                if left >= right or top >= bottom:
                    raise ValueError("box extents must be positive")
                center = [(left + right) / 2, (top + bottom) / 2]
                if not 0 <= center[0] < width or not 0 <= center[1] < height:
                    raise ValueError("center lies outside image dimensions")
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid CVAT box in {name!r}: {error}") from error
            detections.append(
                {
                    "center": center,
                    "class_id": label_ids[label],
                    "confidence": 1.0,
                    "bbox_xyxy_px": bounds,
                }
            )
        annotations.append(
            {"name": name, "resolution": [width, height], "detections": detections}
        )
    return annotations


def _map_cvat_ground_truth(
    annotations: list[dict[str, Any]], frame_sources: list[str]
) -> dict[str, dict[str, Any]]:
    """Join exact paths first, then unique path suffixes across moved corpora.

    Basename matching is permitted only when unambiguous. Annotation IDs and
    sorted positions are never image identities: export IDs may differ from
    frame order, and paced Rust replay may skip source frames.
    """
    mapped: dict[str, dict[str, Any]] = {}
    used_names: set[str] = set()
    for source in frame_sources:
        name = _normalized_annotation_name(source)
        matches = [item for item in annotations if item["name"] == name]
        if not matches:
            matches = [
                item
                for item in annotations
                if item["name"].endswith("/" + name)
                or name.endswith("/" + item["name"])
            ]
        if not matches:
            matches = [
                item
                for item in annotations
                if item["name"].split("/")[-1] == name.split("/")[-1]
            ]
        if len(matches) != 1:
            raise ValueError(
                f"CVAT frame {source!r} has {len(matches)} annotation matches; require an unambiguous filename mapping"
            )
        annotation = matches[0]
        if annotation["name"] in used_names:
            raise ValueError(
                f"Multiple input frames map to CVAT image {annotation['name']!r}"
            )
        used_names.add(annotation["name"])
        mapped[source] = annotation
    return mapped


def _score_ground_truth(
    annotations: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    *,
    class_names: dict[int, str],
    radius_px: float,
) -> dict[str, Any]:
    """Score class-aware, maximum-cardinality one-to-one image-center matches."""
    comparison = compare_detections(
        annotations, detections, class_names=class_names, match_tolerance_px=radius_px
    )
    for match in comparison["matches"]:
        match.pop("confidence_abs_error")
    return {
        "true_positives": comparison["matched_count"],
        "false_positives": len(comparison["unmatched_candidate"]),
        "false_negatives": len(comparison["unmatched_reference"]),
        "matches": comparison["matches"],
        "unmatched_annotations": comparison["unmatched_reference"],
        "unmatched_detections": comparison["unmatched_candidate"],
    }


def _ground_truth_metrics(
    tp: int, fp: int, fn: int, errors: list[float]
) -> dict[str, Any]:
    """Compute micro-averaged metrics; undefined ratios and errors remain null."""
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "center_errors_px": errors,
        "mean_center_error_px": statistics.fmean(errors) if errors else None,
        "p50_center_error_px": float(np.percentile(errors, 50)) if errors else None,
        "p95_center_error_px": float(np.percentile(errors, 95)) if errors else None,
        "max_center_error_px": max(errors) if errors else None,
    }


def _summarize_ground_truth(
    scores: list[dict[str, Any]], class_names: dict[int, str]
) -> dict[str, Any]:
    """Aggregate all scored frames, independent of the bounded frame trace."""
    matches = [match for score in scores for match in score["matches"]]
    false_positives = [
        item for score in scores for item in score["unmatched_detections"]
    ]
    false_negatives = [
        item for score in scores for item in score["unmatched_annotations"]
    ]
    return {
        "frames_scored": len(scores),
        **_ground_truth_metrics(
            len(matches),
            len(false_positives),
            len(false_negatives),
            [match["center_error_px"] for match in matches],
        ),
        "per_class": {
            name: _ground_truth_metrics(
                sum(match["class_id"] == class_id for match in matches),
                sum(item["class_id"] == class_id for item in false_positives),
                sum(item["class_id"] == class_id for item in false_negatives),
                [
                    match["center_error_px"]
                    for match in matches
                    if match["class_id"] == class_id
                ],
            )
            for class_id, name in class_names.items()
        },
    }


def _iter_frames(
    source: Path, max_frames: int
) -> Iterable[tuple[int, str, np.ndarray]]:
    """Yield decoded frames from one image, a directory, or a video file."""
    if source.is_dir():
        paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No supported images were found in: {source}")
        for frame_number, path in enumerate(paths[:max_frames]):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Unable to decode input image: {path}")
            yield frame_number, path.relative_to(source).as_posix(), frame
        return

    if not source.is_file():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to decode input image: {source}")
        yield 0, source.name, frame
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open input video: {source}")
    try:
        for frame_number in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_number, f"{source.name}#frame={frame_number}", frame
    finally:
        capture.release()


def _runtime_evidence(model: Any) -> dict[str, Any]:
    """Return only device/provider data observable on the initialized predictor."""
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    device = getattr(backend, "device", None)
    if device is not None:
        device = str(device)
    providers = None
    session = getattr(backend, "session", None)
    if session is not None and hasattr(session, "get_providers"):
        providers = list(session.get_providers())
    return {
        "backend_type": type(backend).__name__ if backend is not None else None,
        "observed_device": device,
        "observed_providers": providers,
    }


def _model_prediction(
    model: Any,
    frame: np.ndarray,
    *,
    image_size: int,
    confidence: float,
    iou: float,
    device: str,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    include_bounds: bool = False,
) -> list[dict[str, Any]]:
    """Run the same decoded BGR frame with identical Ultralytics arguments."""
    results = model.predict(
        frame,
        imgsz=image_size,
        rect=False,
        conf=confidence,
        iou=iou,
        max_det=max_detections,
        device=device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("Model inference returned no result object")
    return _extract_detections(results[0], include_bounds=include_bounds)


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase or uppercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _finite_number(value: Any, field: str) -> float:
    """Validate JSON numeric evidence without accepting booleans or infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Rust result {field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Rust result {field} must be a finite number")
    return number


def _load_rust_result(path: Path) -> dict[str, Any]:
    """Read and validate the stable recorded-image result schema from Rust."""
    try:
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read Rust candidate result {path}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise ValueError("Rust candidate result must contain a JSON object")
    schema_version = report.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RUST_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported Rust candidate-result schema version: " f"{schema_version!r}"
        )
    workload = report.get("workload")
    if workload not in {RUST_RESULT_WORKLOAD, RUST_PACED_RESULT_WORKLOAD}:
        raise ValueError(
            "Rust candidate result must use a supported recorded-image workload; "
            f"found {workload!r}"
        )
    report["_candidate_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()

    model = report.get("model")
    runtime = report.get("runtime")
    settings = report.get("settings")
    results = report.get("results")
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise ValueError("Rust candidate result is missing model or runtime provenance")
    for field in ("onnx_path", "source_checkpoint_path", "manifest_path"):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(f"Rust candidate result has invalid model.{field}")
    if not isinstance(settings, dict) or not isinstance(results, list) or not results:
        raise ValueError("Rust candidate result is missing settings or image results")
    for field in ("onnx_sha256", "source_checkpoint_sha256"):
        if not _is_sha256(model.get(field)):
            raise ValueError(f"Rust candidate result has an invalid model.{field}")
    class_names = model.get("class_names")
    if (
        not isinstance(class_names, list)
        or not class_names
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError("Rust candidate result has an invalid model.class_names map")
    requested_provider = runtime.get("requested_execution_provider")
    observed_providers = runtime.get("observed_execution_providers")
    if not isinstance(requested_provider, str) or not requested_provider.strip():
        raise ValueError(
            "Rust candidate result has invalid requested provider evidence"
        )
    if not isinstance(observed_providers, list) or any(
        not isinstance(provider, str) or not provider.strip()
        for provider in observed_providers
    ):
        raise ValueError("Rust candidate result has invalid observed provider evidence")

    image_size = settings.get("image_size")
    max_detections = settings.get("max_detections")
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size < 1
    ):
        raise ValueError("Rust candidate result has invalid settings.image_size")
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or max_detections < 1
    ):
        raise ValueError("Rust candidate result has invalid settings.max_detections")
    confidence = _finite_number(settings.get("confidence"), "settings.confidence")
    iou = _finite_number(settings.get("iou"), "settings.iou")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("Rust candidate confidence and IoU settings must be in [0, 1]")

    paced_replay = None
    if workload == RUST_PACED_RESULT_WORKLOAD:
        replay_rate = _finite_number(
            settings.get("replay_rate_fps"), "settings.replay_rate_fps"
        )
        if replay_rate <= 0:
            raise ValueError("Rust paced replay rate must be positive")
        timings = report.get("timings")
        paced_replay = (
            timings.get("paced_replay") if isinstance(timings, dict) else None
        )
        if not isinstance(paced_replay, dict):
            raise ValueError("Rust paced result is missing paced replay measurements")
        if not math.isclose(
            _finite_number(
                paced_replay.get("requested_source_fps"),
                "timings.paced_replay.requested_source_fps",
            ),
            replay_rate,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Paced replay rate differs from effective settings")
        scheduled_count = paced_replay.get("scheduled_frame_count")
        decoded_count = paced_replay.get("decoded_frame_count")
        processed_count = paced_replay.get("processed_frame_count")
        replaced_count = paced_replay.get("replaced_pending_frame_count")
        if (
            isinstance(scheduled_count, bool)
            or not isinstance(scheduled_count, int)
            or isinstance(decoded_count, bool)
            or not isinstance(decoded_count, int)
            or isinstance(processed_count, bool)
            or not isinstance(processed_count, int)
            or isinstance(replaced_count, bool)
            or not isinstance(replaced_count, int)
        ):
            raise ValueError("Paced replay frame counts must be integers")
        if (
            scheduled_count < 1
            or decoded_count > scheduled_count
            or processed_count != len(results)
            or processed_count > decoded_count
            or replaced_count != decoded_count - processed_count
        ):
            raise ValueError(
                "Paced replay counts are inconsistent with processed results"
            )

    seen_report_paths: set[str] = set()
    previous_source_index = -1
    for index, image_result in enumerate(results):
        if not isinstance(image_result, dict):
            raise ValueError(f"Rust result image entry {index} must be an object")
        source_path = image_result.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(f"Rust result image entry {index} has no source path")
        normalized_path = Path(source_path).as_posix()
        if normalized_path in seen_report_paths:
            raise ValueError(
                f"Rust result repeats source frame mapping {source_path!r}"
            )
        seen_report_paths.add(normalized_path)
        source_index = image_result.get("source_frame_index")
        if source_index is not None:
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index <= previous_source_index
            ):
                raise ValueError(
                    f"Rust result image entry {index} has an invalid or unordered source frame index"
                )
            previous_source_index = source_index
            if (
                paced_replay is not None
                and source_index >= paced_replay["scheduled_frame_count"]
            ):
                raise ValueError(
                    f"Rust paced result image entry {index} source index exceeds its schedule"
                )
        elif workload == RUST_PACED_RESULT_WORKLOAD:
            raise ValueError(
                f"Rust paced result image entry {index} is missing source_frame_index"
            )
        for timing_field in (
            "source_schedule_lateness_ms",
            "source_to_inference_start_ms",
            "source_to_inference_complete_ms",
        ):
            timing_value = image_result.get(timing_field)
            if (
                timing_value is not None
                and _finite_number(timing_value, f"results[{index}].{timing_field}") < 0
            ):
                raise ValueError(
                    f"Rust result image entry {index} has negative {timing_field}"
                )
        if not _is_sha256(image_result.get("encoded_image_sha256")):
            raise ValueError(
                f"Rust result image entry {index} has an invalid encoded-image SHA-256"
            )
        width = image_result.get("source_width")
        height = image_result.get("source_height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
        ):
            raise ValueError(
                f"Rust result image entry {index} has invalid source dimensions"
            )
        detections = image_result.get("detections")
        if not isinstance(detections, list):
            raise ValueError(f"Rust result image entry {index} has invalid detections")
        for detection_index, detection in enumerate(detections):
            prefix = f"results[{index}].detections[{detection_index}]"
            if not isinstance(detection, dict):
                raise ValueError(f"Rust {prefix} must be an object")
            class_id = detection.get("class_id")
            if (
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or not 0 <= class_id < len(class_names)
            ):
                raise ValueError(f"Rust {prefix}.class_id is outside the class map")
            if detection.get("class_name") != class_names[class_id]:
                raise ValueError(
                    f"Rust {prefix}.class_name conflicts with model.class_names"
                )
            center = detection.get("center_xy_px")
            if not isinstance(center, list) or len(center) != 2:
                raise ValueError(f"Rust {prefix}.center_xy_px must contain two numbers")
            confidence_value = _finite_number(
                detection.get("confidence"), f"{prefix}.confidence"
            )
            if not 0 <= confidence_value <= 1:
                raise ValueError(f"Rust {prefix}.confidence must be in [0, 1]")
            center_x = _finite_number(center[0], f"{prefix}.center_xy_px[0]")
            center_y = _finite_number(center[1], f"{prefix}.center_xy_px[1]")
            if not 0 <= center_x < width or not 0 <= center_y < height:
                raise ValueError(
                    f"Rust {prefix}.center_xy_px lies outside the reported source image"
                )
            detection["_equivalence_detection"] = {
                "center": [center_x, center_y],
                "class_id": class_id,
                "confidence": confidence_value,
            }
    return report


def _attach_detection_batches(report: dict[str, Any], path: Path) -> dict[str, Any]:
    """Validate Rust JSONL batches and join them to aggregate source metadata.

    The aggregate report carries source paths and encoded-image hashes; JSONL
    carries the public detection-batch contract. Joining both prevents a valid
    batch file from being scored against the wrong input or model run.
    """
    try:
        batch_bytes = path.read_bytes()
        schema = json.loads(DETECTION_BATCH_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read detection batch input: {error}") from error
    if not batch_bytes:
        raise ValueError("Detection batch JSONL file is empty")

    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    try:
        lines = batch_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("Detection batch JSONL must be UTF-8") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Detection batch JSONL contains an empty record")

    metadata = report.get("detection_batch_output")
    results = report.get("results")
    if not isinstance(metadata, dict) or not isinstance(results, list):
        raise ValueError("Aggregate Rust report has no detection-batch output metadata")
    supported_exports = {
        (
            "newline_delimited_json",
            "synchronous_export_during_unpaced_static_replay",
        ),
        (
            "length_framed_json",
            "synchronous_loopback_delivery_during_unpaced_static_replay",
        ),
        (
            "newline_delimited_json",
            "synchronous_export_during_paced_latest_frame_replay",
        ),
        (
            "length_framed_json",
            "synchronous_loopback_delivery_during_paced_latest_frame_replay",
        ),
    }
    is_paced = report.get("workload") == RUST_PACED_RESULT_WORKLOAD
    expected_source_time = (
        "synthetic_monotonic_frame_arrival"
        if is_paced
        else "unavailable_for_still_image_replay"
    )
    if (
        metadata.get("format"),
        metadata.get("mode"),
    ) not in supported_exports or metadata.get("source_time") != expected_source_time:
        raise ValueError("Rust batch output is not a supported recorded-image export")
    batch_count = metadata.get("batch_count")
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count != len(lines)
        or len(results) != len(lines)
    ):
        raise ValueError(
            "Detection batch count does not match the aggregate report's image count"
        )

    model = report["model"]
    settings = report["settings"]
    expected_classes = [
        {"class_id": class_id, "class_name": class_name}
        for class_id, class_name in enumerate(model["class_names"])
    ]
    expected_engine_id = metadata.get("engine_instance_id")
    expected_session_id = metadata.get("source_session_id")
    seen_frames: set[tuple[str, str, str]] = set()
    adapted_results = []

    for index, (line, source_result) in enumerate(zip(lines, results)):
        try:
            batch = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Detection batch JSONL record {index + 1} is invalid JSON: {error}"
            ) from error
        try:
            validator.validate(batch)
        except ValidationError as error:
            raise ValueError(
                f"Detection batch JSONL record {index + 1} violates v1 schema: "
                f"{error.message}"
            ) from error

        prefix = f"detection batch {index + 1}"
        if batch["batch_sequence"] != index + 1:
            raise ValueError(
                f"{prefix} sequence is missing, duplicated, or out of order"
            )
        if batch["engine_instance_id"] != expected_engine_id:
            raise ValueError(f"{prefix} engine instance differs from aggregate report")
        if batch["source"]["session_id"] != expected_session_id:
            raise ValueError(f"{prefix} source session differs from aggregate report")
        frame_key = (
            batch["engine_instance_id"],
            batch["source"]["session_id"],
            batch["source"]["frame_id"],
        )
        if frame_key in seen_frames:
            raise ValueError(f"{prefix} repeats a source frame identity")
        seen_frames.add(frame_key)
        if batch["outcome"] != "success":
            raise ValueError(f"{prefix} is not a successful detection observation")
        source_timestamp = batch["source"]["timestamp"]
        if is_paced:
            source_index = source_result.get("source_frame_index")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or batch["source"]["frame_id"] != f"frame-{source_index + 1:06}"
            ):
                raise ValueError(
                    f"{prefix} source frame identity differs from its result"
                )
            if (
                not isinstance(source_timestamp, dict)
                or source_timestamp["stage"] != "frame_arrival"
                or source_timestamp["clock"] != "monotonic"
                or source_timestamp["domain_id"] != expected_engine_id
                or int(source_timestamp["ticks"])
                > int(batch["inference_started_at"]["ticks"])
            ):
                raise ValueError(f"{prefix} has invalid synthetic frame-arrival time")
        elif source_timestamp is not None:
            raise ValueError(
                f"{prefix} is not an unpaced still-image observation with unknown source time"
            )
        if batch["image_space"] != "raw":
            raise ValueError(f"{prefix} must use the original source image coordinates")
        if batch["model"]["artifact_sha256"] != model["onnx_sha256"]:
            raise ValueError(
                f"{prefix} model artifact hash differs from aggregate report"
            )
        if (
            batch["model"]["source_checkpoint_sha256"]
            != model["source_checkpoint_sha256"]
        ):
            raise ValueError(f"{prefix} checkpoint hash differs from aggregate report")
        if batch["model"]["classes"] != expected_classes:
            raise ValueError(f"{prefix} class map differs from aggregate report")
        batch_settings = batch["settings"]
        if (
            batch_settings["input_width_px"] != settings["image_size"]
            or batch_settings["input_height_px"] != settings["image_size"]
            or batch_settings["max_detections"] != settings["max_detections"]
            or not math.isclose(
                batch_settings["confidence_threshold"],
                settings["confidence"],
                rel_tol=0,
                abs_tol=1e-7,
            )
            or not math.isclose(
                batch_settings["iou_threshold"],
                settings["iou"],
                rel_tol=0,
                abs_tol=1e-7,
            )
        ):
            raise ValueError(
                f"{prefix} inference settings differ from aggregate report"
            )
        if (
            batch["inference_started_at"]["domain_id"]
            != batch["inference_completed_at"]["domain_id"]
            or batch["inference_started_at"]["domain_id"]
            != batch["published_at"]["domain_id"]
            or int(batch["inference_completed_at"]["ticks"])
            < int(batch["inference_started_at"]["ticks"])
            or int(batch["published_at"]["ticks"])
            < int(batch["inference_completed_at"]["ticks"])
        ):
            raise ValueError(f"{prefix} has inconsistent monotonic timestamp ordering")

        source = batch["source"]
        if (source["width_px"], source["height_px"]) != (
            source_result["source_width"],
            source_result["source_height"],
        ):
            raise ValueError(f"{prefix} source dimensions differ from aggregate report")
        detections = []
        if len(batch["detections"]) > batch_settings["max_detections"]:
            raise ValueError(f"{prefix} exceeds the configured detection cap")
        aggregate_detections = source_result["detections"]
        if len(batch["detections"]) != len(aggregate_detections):
            raise ValueError(f"{prefix} detection count differs from aggregate report")
        for detection_index, (detection, aggregate_detection) in enumerate(
            zip(batch["detections"], aggregate_detections)
        ):
            detection_prefix = f"{prefix} detection {detection_index}"
            if detection["detection_index"] != detection_index:
                raise ValueError(f"{detection_prefix} index does not match array order")
            class_id = detection["class_id"]
            if (
                class_id >= len(model["class_names"])
                or detection["class_name"] != model["class_names"][class_id]
            ):
                raise ValueError(f"{detection_prefix} class conflicts with model map")
            box = detection["bbox_image_px"]
            center = detection["center_image_px"]
            width = source["width_px"]
            height = source["height_px"]
            if (
                box["left"] >= box["right"]
                or box["top"] >= box["bottom"]
                or box["right"] > width
                or box["bottom"] > height
                or center["x"] >= width
                or center["y"] >= height
                or not math.isclose(
                    center["x"], (box["left"] + box["right"]) / 2, abs_tol=1e-4
                )
                or not math.isclose(
                    center["y"], (box["top"] + box["bottom"]) / 2, abs_tol=1e-4
                )
            ):
                raise ValueError(
                    f"{detection_prefix} has invalid source-image geometry"
                )
            aggregate_box = aggregate_detection.get("bbox_xyxy_px")
            aggregate_center = aggregate_detection.get("center_xy_px")
            if (
                not isinstance(aggregate_box, list)
                or len(aggregate_box) != 4
                or not isinstance(aggregate_center, list)
                or len(aggregate_center) != 2
            ):
                raise ValueError(
                    f"{detection_prefix} has no matching aggregate geometry"
                )
            try:
                aggregate_geometry = [
                    _finite_number(value, f"{detection_prefix}.aggregate_geometry")
                    for value in [*aggregate_box, *aggregate_center]
                ]
            except ValueError as error:
                raise ValueError(
                    f"{detection_prefix} has invalid aggregate geometry"
                ) from error
            batch_geometry = [
                box["left"],
                box["top"],
                box["right"],
                box["bottom"],
                center["x"],
                center["y"],
            ]
            if detection["class_id"] != aggregate_detection["class_id"] or not (
                math.isclose(
                    detection["confidence"],
                    aggregate_detection["confidence"],
                    rel_tol=0,
                    abs_tol=BATCH_CONFIDENCE_ROUNDING_TOLERANCE,
                )
                and all(
                    math.isclose(
                        batch_value,
                        aggregate_value,
                        rel_tol=0,
                        abs_tol=BATCH_COORDINATE_ROUNDING_TOLERANCE_PX,
                    )
                    for batch_value, aggregate_value in zip(
                        batch_geometry, aggregate_geometry
                    )
                )
            ):
                raise ValueError(
                    f"{detection_prefix} differs from aggregate detection output"
                )
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "center_xy_px": [center["x"], center["y"]],
                    "_equivalence_detection": {
                        "center": [center["x"], center["y"]],
                        "class_id": class_id,
                        "confidence": detection["confidence"],
                    },
                }
            )

        adapted_result = dict(source_result)
        adapted_result["detections"] = detections
        adapted_result["_detection_batch_frame_id"] = source["frame_id"]
        adapted_results.append(adapted_result)

    report["results"] = adapted_results
    report["_candidate_batch_jsonl_sha256"] = hashlib.sha256(batch_bytes).hexdigest()
    report["_candidate_batch_jsonl_path"] = str(path)
    return report


def _input_image_files(source: Path) -> list[tuple[str, Path]]:
    """List supported source images with the same stable order as frame replay."""
    if source.is_dir():
        unsupported_paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.suffix.lower() not in RUST_IMAGE_SUFFIXES
        )
        if unsupported_paths:
            examples = ", ".join(str(path) for path in unsupported_paths[:3])
            raise ValueError(
                "Rust saved-result comparison supports only PNG/JPEG source files; "
                f"unsupported image suffixes found: {examples}"
            )
        paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in RUST_IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No supported images were found in: {source}")
        return [(path.relative_to(source).as_posix(), path.resolve()) for path in paths]
    if source.is_file():
        if source.suffix.lower() in RUST_IMAGE_SUFFIXES:
            return [(source.name, source.resolve())]
        if source.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError(
                "Rust saved-result comparison supports only PNG/JPEG source files; "
                f"unsupported image suffix: {source.suffix}"
            )
    raise ValueError(
        "Rust saved-result comparison supports only an image or image directory; "
        "video frame mapping is unsupported"
    )


def _map_rust_results_to_inputs(
    report: dict[str, Any], source: Path, report_path: Path
) -> list[dict[str, Any]]:
    """Map processed Rust frames to unique input files and verify their hashes."""
    inputs = _input_image_files(source)
    by_absolute_path = {path.resolve(): (name, path) for name, path in inputs}
    by_relative_name = {name: (name, path) for name, path in inputs}
    input_indices = {path: index for index, (_, path) in enumerate(inputs)}
    mapped: list[dict[str, Any]] = []
    used_inputs: set[Path] = set()

    for entry in report["results"]:
        raw_path = Path(entry["path"])
        candidates: set[Path] = set()
        if raw_path.is_absolute():
            resolved = raw_path.resolve()
            if resolved in by_absolute_path:
                candidates.add(resolved)
        else:
            for base in (Path.cwd(), source, report_path.parent):
                resolved = (base / raw_path).resolve()
                if resolved in by_absolute_path:
                    candidates.add(resolved)

            normalized = raw_path.as_posix().removeprefix("./")
            if normalized in by_relative_name:
                candidates.add(by_relative_name[normalized][1])
            # Rust reports commonly retain a repository-relative prefix while
            # callers pass the narrower image directory. Suffix matching is
            # accepted only when the path plus content hash is unique.
            path_parts = raw_path.parts
            suffix_matches = {
                path
                for relative_name, path in inputs
                if len(path_parts) <= len(Path(relative_name).parts)
                and Path(relative_name).parts[-len(path_parts) :] == path_parts
            }
            candidates.update(suffix_matches)

        expected_hash = entry["encoded_image_sha256"].lower()
        reported_source_index = entry.get("source_frame_index")
        indexed_path = None
        if reported_source_index is not None:
            if isinstance(reported_source_index, bool) or not isinstance(
                reported_source_index, int
            ):
                raise ValueError(
                    f"Rust source frame index for {entry['path']!r} must be an integer"
                )
            if not 0 <= reported_source_index < len(inputs):
                raise ValueError(
                    f"Rust source frame index {reported_source_index} is outside the "
                    f"{len(inputs)}-image input sequence"
                )
            indexed_path = inputs[reported_source_index][1]
            if candidates and indexed_path not in candidates:
                raise ValueError(
                    f"Rust source frame index {reported_source_index} conflicts with "
                    f"reported path {entry['path']!r}"
                )
            if _sha256_path(indexed_path)[0] != expected_hash:
                raise ValueError(
                    f"Rust source frame index {reported_source_index} does not match "
                    "its encoded-image SHA-256"
                )

        hash_matches = {
            path for path in candidates if _sha256_path(path)[0] == expected_hash
        }
        if not hash_matches and not candidates:
            # A stale absolute root may differ between the Rust and comparison
            # runs. Content hashes can find the input again when path roots differ;
            # the optional source index below resolves byte-identical inputs.
            hash_matches = {
                path for _, path in inputs if _sha256_path(path)[0] == expected_hash
            }
        if indexed_path is not None and indexed_path in hash_matches:
            hash_matches = {indexed_path}
        if len(hash_matches) != 1:
            if not hash_matches:
                raise ValueError(
                    f"No input image matches Rust source frame {entry['path']!r} "
                    "and its encoded-image SHA-256"
                )
            raise ValueError(
                f"Rust source frame {entry['path']!r} maps to multiple input images; "
                "provide an unambiguous image directory"
            )
        input_path = next(iter(hash_matches))
        expected_source_index = input_indices[input_path]
        if (
            reported_source_index is not None
            and reported_source_index != expected_source_index
        ):
            raise ValueError(
                f"Rust source frame index {reported_source_index!r} does not match "
                f"sorted input index {expected_source_index} for {input_path}"
            )
        if input_path in used_inputs:
            raise ValueError(
                f"Multiple Rust results map to input frame {input_path}; "
                "source mapping must be one-to-one"
            )
        used_inputs.add(input_path)
        source_name = next(name for name, path in inputs if path == input_path)
        mapped.append(
            {
                "entry": entry,
                "input_path": input_path,
                "source_name": source_name,
                "source_index": input_indices[input_path],
            }
        )
    missing_inputs = sorted(path for _, path in inputs if path not in used_inputs)
    if report.get("workload") == RUST_PACED_RESULT_WORKLOAD:
        paced_replay = report["timings"]["paced_replay"]
        if (
            paced_replay["scheduled_frame_count"] != len(inputs)
            or paced_replay["decoded_frame_count"] != len(inputs)
            or paced_replay["processed_frame_count"] != len(mapped)
            or paced_replay["replaced_pending_frame_count"] != len(missing_inputs)
        ):
            raise ValueError(
                "Paced replay schedule/drop counts do not reconcile with the input corpus"
            )
    elif missing_inputs:
        examples = ", ".join(str(path) for path in missing_inputs[:3])
        raise ValueError(
            "Rust result does not cover every input image; "
            f"{len(missing_inputs)} image(s) are missing from the result "
            f"(for example: {examples})"
        )
    return mapped


def _rust_result_candidate_evidence(
    result_path: Path, report: dict[str, Any]
) -> dict[str, Any]:
    """Translate Rust report provenance into the common model comparison schema."""
    model = report["model"]
    runtime = report["runtime"]
    return {
        "path": model["onnx_path"],
        "sha256": model["onnx_sha256"],
        "components": [],
        "export_manifest": {
            "status": "path_recorded",
            "path": model["manifest_path"],
        },
        "class_names": dict(enumerate(model["class_names"])),
        "runtime": {
            "backend_type": runtime.get("binding"),
            "observed_device": None,
            "observed_providers": runtime["observed_execution_providers"],
        },
        "provenance": {
            "kind": (
                "saved_rust_detection_batch_jsonl"
                if "_candidate_batch_jsonl_sha256" in report
                else "saved_rust_inference_result"
            ),
            "rust_schema_version": report["schema_version"],
            "workload": report["workload"],
            "candidate_report_path": str(result_path),
            "candidate_report_sha256": report["_candidate_report_sha256"],
            "onnx_sha256": model["onnx_sha256"],
            "source_checkpoint_sha256": model["source_checkpoint_sha256"],
            "requested_execution_provider": runtime["requested_execution_provider"],
            "observed_execution_providers": runtime["observed_execution_providers"],
            "effective_inference_settings": {
                "image_size": report["settings"]["image_size"],
                "confidence": report["settings"]["confidence"],
                "iou": report["settings"]["iou"],
                "max_detections": report["settings"]["max_detections"],
            },
            **(
                {
                    "candidate_batch_jsonl_path": report["_candidate_batch_jsonl_path"],
                    "candidate_batch_jsonl_sha256": report[
                        "_candidate_batch_jsonl_sha256"
                    ],
                    "candidate_batch_count": report["detection_batch_output"][
                        "batch_count"
                    ],
                }
                if "_candidate_batch_jsonl_sha256" in report
                else {}
            ),
        },
    }


def _summarize(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate bounded per-frame comparisons without conflating accuracy."""
    errors = [
        match["center_error_px"] for frame in frames for match in frame["matches"]
    ]
    confidence_errors = [
        match["confidence_abs_error"] for frame in frames for match in frame["matches"]
    ]
    frames_with_differences = sum(
        bool(frame["unmatched_reference"] or frame["unmatched_candidate"])
        for frame in frames
    )
    return {
        "frames_processed": len(frames),
        "frames_with_differences": frames_with_differences,
        "matched_detections": len(errors),
        "unmatched_reference_detections": sum(
            len(frame["unmatched_reference"]) for frame in frames
        ),
        "unmatched_candidate_detections": sum(
            len(frame["unmatched_candidate"]) for frame in frames
        ),
        "mean_center_error_px": statistics.fmean(errors) if errors else None,
        "p95_center_error_px": (float(np.percentile(errors, 95)) if errors else None),
        "max_center_error_px": max(errors) if errors else None,
        "max_confidence_abs_error": (
            max(confidence_errors) if confidence_errors else None
        ),
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    """Write a compact report with scope limitations and comparison evidence."""
    summary = result["summary"]
    lines = [
        "# Detector backend equivalence comparison",
        "",
        f"Run: `{result['run_id']}`",
        "",
        "This report compares model outputs on identical decoded frames. It is not a physical accuracy, tracking, or performance qualification.",
        "",
        f"- Input: `{result['input']['path']}`",
        f"- Reference model SHA-256: `{result['models']['reference']['sha256']}`",
        f"- Candidate model SHA-256: `{result['models']['candidate']['sha256']}`",
        f"- Frames processed: {summary['frames_processed']}",
        f"- Frames with unmatched detections: {summary['frames_with_differences']}",
        f"- Matched detections: {summary['matched_detections']}",
        f"- Unmatched reference/candidate detections: {summary['unmatched_reference_detections']}/{summary['unmatched_candidate_detections']}",
        f"- Center error mean/p95/max (px): {summary['mean_center_error_px']}/{summary['p95_center_error_px']}/{summary['max_center_error_px']}",
        f"- Match association tolerance: {result['comparison']['match_tolerance_px']} px",
        f"- Inference settings: imgsz={result['comparison']['imgsz']}, confidence={result['comparison']['confidence']}, IoU={result['comparison']['iou']}, max detections={result['comparison']['max_detections']}",
        "",
        "Detailed per-frame data is in the sibling JSON report.",
    ]
    provenance = result["models"]["candidate"].get("provenance")
    if provenance is not None:
        lines.extend(
            [
                "",
                "## Saved Rust output provenance",
                "",
                f"- Candidate report SHA-256: `{provenance['candidate_report_sha256']}`",
                f"- Source checkpoint SHA-256: `{provenance['source_checkpoint_sha256']}`",
                f"- Requested execution provider: `{provenance['requested_execution_provider']}`",
                f"- Observed execution providers: `{', '.join(provenance['observed_execution_providers']) or 'unavailable'}`",
                f"- Candidate result frames omitted by `--frames`: {result['comparison']['candidate_result_frames_omitted']}",
            ]
        )
        skipped_by_pacing = result["comparison"].get(
            "candidate_source_frames_skipped_by_pacing"
        )
        if skipped_by_pacing is not None:
            lines.append(
                f"- Candidate source frames skipped by latest-frame replay: {skipped_by_pacing}"
            )
        if "candidate_batch_jsonl_sha256" in provenance:
            lines.extend(
                [
                    f"- Detection batch JSONL SHA-256: `{provenance['candidate_batch_jsonl_sha256']}`",
                    f"- Detection batches joined: {provenance['candidate_batch_count']}",
                ]
            )
    ground_truth = result.get("ground_truth")
    if ground_truth is not None:
        lines.extend(
            [
                "",
                "## CVAT image-center accuracy",
                "",
                f"- Annotation XML SHA-256: `{ground_truth['sha256']}`",
                f"- Annotation images scored/available: {ground_truth['annotation_images_scored']}/{ground_truth['annotation_images_available']}",
                f"- Audit association radius: {ground_truth['association_radius_px']} px (configurable matching threshold, not a production acceptance criterion)",
                "- Errors describe matched box centers in original image pixels; unmatched objects remain FP/FN. Undefined ratios and errors are reported as None.",
                "",
                "| Source | Class | TP | FP | FN | Precision | Recall | F1 | Center error mean/p50/p95/max (px) |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for role in ("reference", "candidate"):
            summary_rows = [
                ("overall", ground_truth[role]),
                *ground_truth[role]["per_class"].items(),
            ]
            for label, metrics in summary_rows:
                lines.append(
                    f"| {role} | {label} | {metrics['true_positives']} | {metrics['false_positives']} | {metrics['false_negatives']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1']} | {metrics['mean_center_error_px']}/{metrics['p50_center_error_px']}/{metrics['p95_center_error_px']}/{metrics['max_center_error_px']} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_equivalence_test(
    arguments: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    """Run the black-box comparison and save versioned machine/human reports."""
    reference_path = Path(arguments.reference_model).expanduser().resolve()
    source_path = Path(arguments.input).expanduser().resolve()
    candidate_path = (
        Path(arguments.candidate_model).expanduser().resolve()
        if arguments.candidate_model is not None
        else None
    )
    candidate_result_path = (
        Path(arguments.candidate_result).expanduser().resolve()
        if arguments.candidate_result is not None
        else None
    )
    candidate_batches_path = (
        Path(arguments.candidate_batches_jsonl).expanduser().resolve()
        if arguments.candidate_batches_jsonl is not None
        else None
    )
    for model_path in (reference_path, candidate_path):
        if model_path is not None and not model_path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Input does not exist: {source_path}")
    if candidate_result_path is not None and not candidate_result_path.is_file():
        raise FileNotFoundError(
            f"Rust candidate result does not exist: {candidate_result_path}"
        )
    if candidate_batches_path is not None:
        if candidate_result_path is None:
            raise ValueError("--candidate-batches-jsonl requires --candidate-result")
        if not candidate_batches_path.is_file():
            raise FileNotFoundError(
                f"Rust detection batch JSONL does not exist: {candidate_batches_path}"
            )
        if candidate_batches_path in (source_path, candidate_result_path):
            raise ValueError(
                "Detection batch JSONL must differ from the input and aggregate report"
            )

    reference_hash, reference_components = _sha256_path(reference_path)
    reference_model = YOLO(str(reference_path))
    reference_names = _model_class_names(reference_model)
    rust_report = None
    rust_mapped_results = None
    rust_candidate_evidence = None
    candidate_model = None
    candidate_source_frames_skipped = None
    if candidate_result_path is not None:
        if candidate_result_path == source_path:
            raise ValueError("Rust candidate result cannot also be the input source")
        rust_report = _load_rust_result(candidate_result_path)
        if candidate_batches_path is not None:
            rust_report = _attach_detection_batches(rust_report, candidate_batches_path)
        candidate_names = dict(enumerate(rust_report["model"]["class_names"]))
        if reference_hash != rust_report["model"]["source_checkpoint_sha256"]:
            raise ValueError(
                "Reference checkpoint SHA-256 does not match the Rust result's "
                "source checkpoint provenance"
            )
        if reference_names != candidate_names:
            raise ValueError(
                "Reference and Rust candidate class maps differ: "
                f"{reference_names!r} != {candidate_names!r}"
            )
        settings = rust_report["settings"]
        if settings["image_size"] != arguments.imgsz:
            raise ValueError(
                "Rust result image_size does not match --imgsz: "
                f"{settings['image_size']} != {arguments.imgsz}"
            )
        if not math.isclose(
            settings["confidence"], arguments.confidence, rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError(
                "Rust result confidence threshold does not match --confidence"
            )
        if not math.isclose(settings["iou"], arguments.iou, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Rust result NMS IoU does not match --iou")
        rust_mapped_results = _map_rust_results_to_inputs(
            rust_report, source_path, candidate_result_path
        )
        if rust_report["workload"] == RUST_PACED_RESULT_WORKLOAD:
            paced_replay = rust_report["timings"]["paced_replay"]
            candidate_source_frames_skipped = (
                paced_replay["scheduled_frame_count"]
                - paced_replay["processed_frame_count"]
            )
        rust_candidate_evidence = _rust_result_candidate_evidence(
            candidate_result_path, rust_report
        )
    else:
        if candidate_path is None:
            raise ValueError("Select either --candidate-model or --candidate-result")
        candidate_hash, candidate_components = _sha256_path(candidate_path)
        candidate_model = YOLO(str(candidate_path))
        candidate_names = _model_class_names(candidate_model)
        if reference_names != candidate_names:
            raise ValueError(
                "Reference and candidate class maps differ: "
                f"{reference_names!r} != {candidate_names!r}"
            )

    ground_truth_path = (
        Path(arguments.ground_truth_cvat).expanduser().resolve()
        if getattr(arguments, "ground_truth_cvat", None) is not None
        else None
    )
    ground_truth_annotations = (
        _load_cvat_ground_truth(ground_truth_path, reference_names)
        if ground_truth_path is not None
        else None
    )
    ground_truth_radius = getattr(
        arguments, "ground_truth_radius_px", DEFAULT_GROUND_TRUTH_RADIUS_PX
    )
    if ground_truth_annotations is not None and (
        not math.isfinite(ground_truth_radius) or ground_truth_radius <= 0
    ):
        raise ValueError("--ground-truth-radius-px must be finite and positive")

    frame_results: list[dict[str, Any]] = []
    reported_frames: list[dict[str, Any]] = []
    if rust_mapped_results is not None:
        frame_inputs = []
        source_ordered_results = sorted(
            rust_mapped_results, key=lambda mapped: mapped["source_index"]
        )
        for frame_number, mapped in enumerate(
            source_ordered_results[: arguments.frames]
        ):
            frame = cv2.imread(str(mapped["input_path"]), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(
                    f"Unable to decode input image: {mapped['input_path']}"
                )
            entry = mapped["entry"]
            if [int(frame.shape[1]), int(frame.shape[0])] != [
                entry["source_width"],
                entry["source_height"],
            ]:
                raise ValueError(
                    f"Source dimensions changed for Rust frame {entry['path']!r}"
                )
            frame_inputs.append((frame_number, mapped["source_name"], frame, entry))
    else:
        frame_inputs = [
            (frame_number, frame_source, frame, None)
            for frame_number, frame_source, frame in _iter_frames(
                source_path, arguments.frames
            )
        ]

    ground_truth_mapping = (
        _map_cvat_ground_truth(
            ground_truth_annotations, [source for _, source, _, _ in frame_inputs]
        )
        if ground_truth_annotations is not None
        else {}
    )
    for _, frame_source, frame, _ in frame_inputs:
        if frame_source in ground_truth_mapping and ground_truth_mapping[frame_source][
            "resolution"
        ] != [int(frame.shape[1]), int(frame.shape[0])]:
            raise ValueError(
                f"CVAT dimensions differ from decoded input frame {frame_source!r}"
            )

    for frame_number, frame_source, frame, rust_entry in frame_inputs:
        reference_detections = _model_prediction(
            reference_model,
            frame,
            image_size=arguments.imgsz,
            confidence=arguments.confidence,
            iou=arguments.iou,
            device=arguments.device,
            include_bounds=ground_truth_annotations is not None,
            max_detections=(
                rust_report["settings"]["max_detections"]
                if rust_report is not None
                else DEFAULT_MAX_DETECTIONS
            ),
        )
        if rust_entry is not None:
            candidate_detections = [
                dict(detection["_equivalence_detection"])
                for detection in rust_entry["detections"]
            ]
            if ground_truth_annotations is not None:
                for normalized, detection in zip(
                    candidate_detections, rust_entry["detections"]
                ):
                    if "bbox_xyxy_px" in detection:
                        normalized["bbox_xyxy_px"] = detection["bbox_xyxy_px"]
        else:
            candidate_detections = _model_prediction(
                candidate_model,
                frame,
                image_size=arguments.imgsz,
                confidence=arguments.confidence,
                iou=arguments.iou,
                device=arguments.device,
                include_bounds=ground_truth_annotations is not None,
            )
        comparison = compare_detections(
            reference_detections,
            candidate_detections,
            class_names=reference_names,
            match_tolerance_px=arguments.tolerance,
        )
        frame_result = {
            "frame_number": frame_number,
            "frame_source": frame_source,
            "resolution": [int(frame.shape[1]), int(frame.shape[0])],
            **comparison,
        }
        if ground_truth_annotations is not None:
            annotation = ground_truth_mapping[frame_source]
            frame_result["ground_truth"] = {
                "annotation_name": annotation["name"],
                "annotations": annotation["detections"],
                "reference_detections": reference_detections,
                "candidate_detections": candidate_detections,
                **{
                    role: _score_ground_truth(
                        annotation["detections"],
                        detections,
                        class_names=reference_names,
                        radius_px=ground_truth_radius,
                    )
                    for role, detections in (
                        ("reference", reference_detections),
                        ("candidate", candidate_detections),
                    )
                },
            }
        frame_results.append(frame_result)
        if len(reported_frames) < arguments.max_trace_frames:
            reported_frames.append(frame_result)

    if not frame_results:
        raise ValueError(f"No frames were decoded from: {source_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"equivalence-{run_id}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    if rust_candidate_evidence is not None:
        candidate_model_evidence = rust_candidate_evidence
    else:
        assert candidate_path is not None
        candidate_model_evidence = {
            "path": str(candidate_path),
            "sha256": candidate_hash,
            "components": candidate_components,
            "export_manifest": _manifest_evidence(candidate_path),
            "class_names": candidate_names,
            "runtime": _runtime_evidence(candidate_model),
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "model detection output equivalence; not accuracy or performance",
        "input": {
            "path": str(source_path),
            "sha256": _sha256_path(source_path)[0],
            "kind": (
                "image_directory"
                if source_path.is_dir()
                else (
                    "image" if source_path.suffix.lower() in IMAGE_SUFFIXES else "video"
                )
            ),
        },
        "models": {
            "reference": {
                "path": str(reference_path),
                "sha256": reference_hash,
                "components": reference_components,
                "class_names": reference_names,
                "runtime": _runtime_evidence(reference_model),
            },
            "candidate": candidate_model_evidence,
        },
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": importlib_version("ultralytics"),
            "torch": importlib_version("torch"),
            "onnxruntime": importlib_version("onnxruntime"),
            "opencv": cv2.__version__,
        },
        "comparison": {
            "imgsz": arguments.imgsz,
            "rect": False,
            "confidence": arguments.confidence,
            "iou": arguments.iou,
            "max_detections": (
                rust_report["settings"]["max_detections"]
                if rust_report is not None
                else DEFAULT_MAX_DETECTIONS
            ),
            "requested_device": arguments.device,
            "match_tolerance_px": arguments.tolerance,
            "matching": "class-constrained minimum-cost one-to-one center distance",
            "max_frames": arguments.frames,
            "max_trace_frames": arguments.max_trace_frames,
            "candidate_kind": (
                "saved_rust_detection_batch_jsonl"
                if candidate_batches_path is not None
                else ("saved_rust_result" if rust_report is not None else "model")
            ),
            "candidate_result_frames_available": (
                len(rust_mapped_results) if rust_mapped_results is not None else None
            ),
            "candidate_result_frames_compared": (
                len(frame_inputs) if rust_mapped_results is not None else None
            ),
            "candidate_result_frames_omitted": (
                len(rust_mapped_results) - len(frame_inputs)
                if rust_mapped_results is not None
                else None
            ),
            "candidate_source_frames_skipped_by_pacing": candidate_source_frames_skipped,
        },
        "summary": _summarize(frame_results),
        "frames": reported_frames,
        "frames_omitted_from_details": len(frame_results) - len(reported_frames),
        "limitations": [
            "No ground-truth accuracy is computed; model agreement is not evidence of correctness.",
            "No temporal tracking, cue geometry, physics, or camera-to-client path is exercised.",
            "The match tolerance associates corresponding outputs and is not a production acceptance threshold.",
            "Inference timing is intentionally not reported by this equivalence tool.",
            *(
                [
                    "Encoded-file SHA-256 verifies source-file identity but does not prove decoded pixel equality across image decoders.",
                    "Rust timing fields are preserved as provenance only and are not scored or interpreted as a real-time performance claim.",
                    *(
                        [
                            "Paced replay may omit scheduled frames through latest-frame replacement; comparisons include only processed source frames.",
                        ]
                        if candidate_source_frames_skipped is not None
                        else []
                    ),
                ]
                if rust_report is not None
                else []
            ),
        ],
    }
    if ground_truth_annotations is not None:
        assert ground_truth_path is not None
        result["scope"] = (
            "model detection output equivalence and annotated image-center accuracy; not physical accuracy or performance"
        )
        result["limitations"][
            0
        ] = "CVAT accuracy scores image-coordinate box centers on annotated processed frames only; they do not establish physical localization accuracy."
        result["limitations"].append(
            "Ground-truth radius is a configurable audit association threshold, not a production acceptance criterion. Center errors describe matched detections only; misses are counted separately."
        )
        result["ground_truth"] = {
            "path": str(ground_truth_path),
            "sha256": _sha256_path(ground_truth_path)[0],
            "format": "CVAT image XML boxes",
            "coordinate_space": "original image pixels",
            "association_radius_px": ground_truth_radius,
            "matching": "class-constrained maximum-cardinality minimum-cost one-to-one box-center distance",
            "annotation_images_available": len(ground_truth_annotations),
            "annotation_images_scored": len(ground_truth_mapping),
            "annotation_images_not_scored": len(ground_truth_annotations)
            - len(ground_truth_mapping),
            **{
                role: _summarize_ground_truth(
                    [frame["ground_truth"][role] for frame in frame_results],
                    reference_names,
                )
                for role in ("reference", "candidate")
            },
        }
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(markdown_path, result)
    return json_path, markdown_path, result


def importlib_version(distribution: str) -> str | None:
    """Return a package version if installed, otherwise explicit unavailable."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_argument_parser() -> argparse.ArgumentParser:
    """Create a CLI with explicit reference and candidate provenance."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare retained Python YOLO outputs against another model or a "
            "saved Rust recorded-image result."
        )
    )
    parser.add_argument(
        "input", type=Path, help="Video, image, or directory of images."
    )
    parser.add_argument("--reference-model", required=True, type=Path)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-model", type=Path)
    candidates.add_argument(
        "--candidate-result",
        type=Path,
        help=(
            "Saved Rust billiards-perception-infer JSON result. Supports only "
            "recorded static images, not video frames."
        ),
    )
    parser.add_argument(
        "--candidate-batches-jsonl",
        type=Path,
        help=(
            "Optional detection-batch v1 JSONL paired with --candidate-result; "
            "every line is validated and joined by recorded frame order."
        ),
    )
    parser.add_argument(
        "--ground-truth-cvat",
        type=Path,
        help="Optional CVAT image XML with boxes labelled using the model class names; all processed images must map unambiguously by filename.",
    )
    parser.add_argument(
        "--ground-truth-radius-px",
        type=float,
        default=DEFAULT_GROUND_TRUTH_RADIUS_PX,
        help="Class-aware box-center audit association radius in original image pixels (default: %(default)s); not a production acceptance criterion.",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_LIMIT)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_MATCH_TOLERANCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIRECTORY)
    )
    parser.add_argument(
        "--max-trace-frames", type=int, default=DEFAULT_MAX_TRACE_FRAMES
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison, returning a nonzero code only for invalid/failed runs."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.frames < 1:
        parser.error("--frames must be positive")
    if arguments.imgsz < 1:
        parser.error("--imgsz must be positive")
    if arguments.max_trace_frames < 0:
        parser.error("--max-trace-frames cannot be negative")
    if not 0 <= arguments.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if not 0 <= arguments.iou <= 1:
        parser.error("--iou must be between 0 and 1")
    if not np.isfinite(arguments.tolerance) or arguments.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")

    if (
        not math.isfinite(arguments.ground_truth_radius_px)
        or arguments.ground_truth_radius_px <= 0
    ):
        parser.error("--ground-truth-radius-px must be finite and positive")

    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        json_path, markdown_path, result = run_equivalence_test(arguments)
    except Exception as error:
        logger.error("Equivalence comparison failed: %s", error)
        return 1

    summary = result["summary"]
    print(f"Frames: {summary['frames_processed']}")
    print(f"Frames with unmatched detections: {summary['frames_with_differences']}")
    print(
        "Matched detections: "
        f"{summary['matched_detections']} (center max: {summary['max_center_error_px']} px)"
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
