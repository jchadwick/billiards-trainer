#!/usr/bin/env python3
"""Replay recorded billiards video against candidate phase-aware profiles.

This benchmark reports measured runtime, inference cadence, stationary-position
jitter, cue-angle stability, track observation gaps, and recovery time. It does
not select or write production defaults. Ground-truth accuracy and aim-assist
latency are reported as unavailable unless separately annotated evaluation
data is added.

Example:
    python backend/tools/benchmark_vision_profiles.py \
        --video test_data/demo2_short.mp4 --frames 300 --output /tmp/vision.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import config
from backend.tools.temporal_evaluation import (
    MAX_SAFE_JSON_INTEGER,
    evaluate_temporal_localization,
    load_temporal_annotations,
    unavailable_temporal_metrics,
)
from backend.vision import VisionModule
from backend.vision.detection.detector_adapter import canonical_point_to_source_pixels
from backend.vision.motion_monitor import MotionMonitor
from backend.vision.phase_scheduler import VisionPhase, VisionPhaseScheduler


class ReplayClock:
    """Clock whose time advances with source video presentation timestamps."""

    def __init__(self) -> None:
        self.value = time.monotonic()

    def __call__(self) -> float:
        return self.value

    def set_elapsed(self, elapsed_seconds: float) -> None:
        self.value = self.start + elapsed_seconds

    def start_replay(self) -> None:
        self.start = time.monotonic()
        self.value = self.start


def build_candidate_settings(
    base_settings: dict[str, Any],
    *,
    maximum_dimension: int | None,
    cadence_multiplier: float,
    crop_to_table: bool = False,
) -> dict[str, Any]:
    """Build an isolated profile candidate without changing stored config."""
    settings = copy.deepcopy(base_settings)
    settings["enabled"] = True
    settings["processing_max_dimension"] = {
        phase.value: maximum_dimension for phase in VisionPhase
    }
    settings["ball_detection_crop"] = {
        phase.value: crop_to_table for phase in VisionPhase
    }
    intervals = settings.get("intervals_seconds", {})
    for phase_values in intervals.values():
        for job_name, value in phase_values.items():
            # A lower multiplier schedules work more often; zero stays every
            # processed frame, as defined by the scheduler contract.
            phase_values[job_name] = float(value) * cadence_multiplier
    return settings


def parse_dimensions(value: str) -> list[tuple[str, int | None]]:
    candidates: list[tuple[str, int | None]] = []
    for token in value.split(","):
        normalized = token.strip().lower()
        if normalized == "native":
            candidates.append(("native", None))
        elif normalized in {"720", "720p"}:
            # The scheduler's limit is the longest image side. 1280 therefore
            # yields conventional 1280x720 analysis on a 16:9 source.
            candidates.append(("720p", 1280))
        else:
            raise argparse.ArgumentTypeError(
                f"unsupported processing dimension {token!r}; use native or 720p"
            )
    if not candidates:
        raise argparse.ArgumentTypeError("at least one dimension candidate is required")
    return candidates


def parse_factors(value: str) -> list[float]:
    factors = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not factors or any(not math.isfinite(item) or item <= 0 for item in factors):
        raise argparse.ArgumentTypeError("cadence factors must be finite and positive")
    return factors


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _phase_name(result: Any) -> str:
    phase = getattr(result, "phase", None) or "unknown"
    return str(getattr(phase, "value", phase)).lower()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporal_ball_center(
    ball: Any, source_resolution: tuple[int, int]
) -> tuple[float, float] | None:
    """Convert a replay ball position into the annotation source-pixel space."""
    coordinate_space = getattr(ball, "coordinate_space", "4k_canonical")
    position = getattr(ball, "position", None)
    if position is None:
        return None
    if coordinate_space == "4k_canonical":
        center = canonical_point_to_source_pixels(position, source_resolution)
    elif coordinate_space == "pixel":
        center = (float(position[0]), float(position[1]))
    else:
        return None
    if not all(math.isfinite(float(value)) for value in center):
        return None
    return float(center[0]), float(center[1])


def _temporal_cue_endpoints(
    cue: Any, source_resolution: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Convert a replay cue's tip and butt into source-pixel coordinates."""
    coordinate_space = getattr(cue, "coordinate_space", "4k_canonical")
    tip = getattr(cue, "tip_position", None)
    butt = getattr(cue, "butt_position", None)
    if tip is None or butt is None:
        return None
    if coordinate_space == "4k_canonical":
        tip = canonical_point_to_source_pixels(tip, source_resolution)
        butt = canonical_point_to_source_pixels(butt, source_resolution)
    elif coordinate_space != "pixel":
        return None
    normalized_tip = (float(tip[0]), float(tip[1]))
    normalized_butt = (float(butt[0]), float(butt[1]))
    if not all(math.isfinite(value) for value in (*normalized_tip, *normalized_butt)):
        return None
    # CueStick defaults butt_position to (0, 0) for detector paths that only
    # return a tip. Do not present that default as a measured shaft endpoint.
    if math.dist(normalized_tip, normalized_butt) <= 1.0:
        return None
    return normalized_tip, normalized_butt


def _normalize_temporal_result(
    result: Any,
    *,
    frame_index: int,
    presentation_time_seconds: float,
    source_epoch: float,
    presentation_time_offset_seconds: float,
    source_resolution: tuple[int, int],
) -> dict[str, Any]:
    """Adapt one Python result into the engine-neutral evaluator contract."""
    ball_status = getattr(result, "ball_positions_status", None) or "unavailable"
    source_observation_time: float | None = None
    source_timestamp = _number(
        getattr(result, "ball_observation_source_timestamp", None)
    )
    timestamp_kind = getattr(result, "ball_observation_source_timestamp_kind", None)
    if timestamp_kind == "replay_video_presentation" and source_timestamp is not None:
        source_observation_time = (
            source_timestamp - source_epoch + presentation_time_offset_seconds
        )

    balls = []
    for ball in getattr(result, "balls", []):
        # Some trackers retain or predict positions alongside a successful
        # detector pass. Such items must not inflate fresh localization scores.
        if not getattr(ball, "is_observed", True):
            continue
        center = _temporal_ball_center(ball, source_resolution)
        if center is None:
            continue
        track_id = getattr(ball, "track_id", None)
        balls.append(
            {
                "track_id": track_id,
                "center": {"x": center[0], "y": center[1]},
            }
        )

    cue_status = getattr(result, "cue_detection_status", None) or "unavailable"
    if cue_status == "stale":
        # A stale cached cue is no current observation. Keep it distinct from
        # a fresh negative result so it cannot count as a true negative.
        cue_status = "unavailable"
    cue = getattr(result, "cue", None)
    cue_endpoints = _temporal_cue_endpoints(cue, source_resolution) if cue else None
    cue_present: bool | None = None
    if cue_status == "observed":
        cue_present = cue is not None
    if cue_present and cue_endpoints is None:
        # A world-space cue cannot be compared with pixel-space labels. Preserve
        # the rest of the frame but report cue quality as unavailable.
        cue_status = "unavailable"
        cue_present = None
    record: dict[str, Any] = {
        "source_frame_index": frame_index,
        "presentation_time_seconds": presentation_time_seconds,
        "ball_positions_status": ball_status,
        "ball_observation_source_frame_index": getattr(
            result, "ball_observation_source_frame_number", None
        ),
        "ball_observation_presentation_time_seconds": source_observation_time,
        "balls": balls,
        "cue_observation_status": cue_status,
        "cue_observation_source_frame_index": (
            frame_index if cue_status == "observed" else None
        ),
        "cue_present": cue_present,
        "calibration_revision": (
            result.calibration_revision
            if type(getattr(result, "calibration_revision", None)) is int
            and 0 <= result.calibration_revision <= MAX_SAFE_JSON_INTEGER
            else None
        ),
        # The replay has no original camera process identity. Keep this null;
        # session fidelity must not be inferred from the annotation labels.
        "source_session_epoch": (
            str(result.source_process_id)
            if getattr(result, "source_process_id", None) is not None
            else None
        ),
    }
    if cue_present and cue_endpoints is not None:
        tip, butt = cue_endpoints
        record.update(
            {
                "cue_tip": {"x": tip[0], "y": tip[1]},
                "cue_butt": {"x": butt[0], "y": butt[1]},
                "cue_angle_degrees": _number(getattr(cue, "angle", None)),
            }
        )
    return record


def _hash_path(path: Path) -> str:
    """Hash a model file or directory without treating an unset path as cwd."""
    if path.is_file():
        return _hash_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_hash_file(child).encode("ascii"))
    return digest.hexdigest()


def _run_metadata(video_path: Path, output_path: Path | None) -> dict[str, Any]:
    """Capture a reproducible identity for each recorded-input benchmark."""

    def git_bytes(*arguments: str) -> bytes | None:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=ROOT, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError):
            return None

    excluded_output: str | None = None
    if output_path is not None:
        try:
            excluded_output = output_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            pass

    tracked = git_bytes("diff", "--name-only", "-z", "HEAD", "--") or b""
    untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z") or b""
    paths = sorted(
        {
            path.decode("utf-8", errors="replace")
            for path in (*tracked.split(b"\0"), *untracked.split(b"\0"))
            if path
            and not (
                excluded_output
                and (
                    path.decode("utf-8", errors="replace") == excluded_output
                    or path.decode("utf-8", errors="replace").startswith(
                        excluded_output + "/"
                    )
                )
            )
        }
    )
    dirty_digest = hashlib.sha256()
    for relative_path in paths:
        path = ROOT / relative_path
        dirty_digest.update(relative_path.encode("utf-8"))
        dirty_digest.update(b"\0")
        if path.is_file():
            dirty_digest.update(_hash_file(path).encode("ascii"))
        elif path.is_symlink():
            dirty_digest.update(str(path.readlink()).encode("utf-8"))
        else:
            dirty_digest.update(b"<deleted-or-non-file>")

    model_value = config.get("vision.detection.yolo_model_path")
    model_path = (
        Path(str(model_value)).expanduser()
        if isinstance(model_value, str) and model_value.strip()
        else None
    )
    if model_path is not None and not model_path.is_absolute():
        model_path = ROOT / model_path
    model_hash = (
        _hash_path(model_path)
        if model_path is not None and (model_path.is_file() or model_path.is_dir())
        else None
    )
    try:
        import psutil

        hardware = {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "system_memory_bytes": psutil.virtual_memory().total,
        }
    except Exception:
        hardware = {
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": None,
            "system_memory_bytes": None,
        }

    try:
        revision_bytes = git_bytes("rev-parse", "HEAD")
        revision = revision_bytes.decode("utf-8").strip() if revision_bytes else None
    except UnicodeDecodeError:
        revision = None
    return {
        "revision": revision,
        "dirty_worktree_sha256": dirty_digest.hexdigest(),
        "dirty_worktree_files": paths,
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "hardware": hardware,
        "python": platform.python_version(),
        "model": {
            "configured_path": str(model_value) if model_value else None,
            "exists": bool(model_path and model_path.exists()),
            "sha256": model_hash,
        },
        "recorded_input": {
            "path": str(video_path.resolve()),
            "sha256": _hash_path(video_path),
        },
        "effective_configuration": {
            "camera": config.get("vision.camera", {}),
            "detection": config.get("vision.detection", {}),
            "base_phase_scheduler": config.get("vision.processing.phase_scheduler", {}),
            "calibration": config.get("table", {}),
        },
    }


def benchmark_static_ground_truth(
    module: VisionModule,
    dataset_directory: Path,
    *,
    dimension_name: str,
    maximum_dimension: int | None,
    edge_margin_fraction: float,
    crop_to_table: bool,
) -> dict[str, Any]:
    """Compare ball centers and radii with repository-annotated still frames."""
    detector = module.detector
    if detector is None:
        raise RuntimeError("static benchmark requires an enabled ball detector")

    annotation_files = sorted(dataset_directory.glob("*.json"))
    distance_errors: list[float] = []
    radius_errors: list[float] = []
    edge_distance_errors: list[float] = []
    ground_truth_count = prediction_count = matched_count = 0
    edge_ground_truth_count = edge_matched_count = image_count = 0

    for annotation_file in annotation_files:
        try:
            annotation = json.loads(annotation_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        image_path = dataset_directory / str(annotation.get("image", ""))
        loaded_image = cv2.imread(str(image_path))
        if loaded_image is None:
            continue
        # OpenCV's type stubs expose multiple image representations. Normalize
        # the decoded image to the uint8 frame contract used by VisionModule.
        image = np.asarray(loaded_image, dtype=np.uint8)
        image_count += 1
        source_height, source_width = image.shape[:2]
        source_resolution = (source_width, source_height)
        processing_frame = module._resize_for_phase(image, maximum_dimension)
        processing_resolution = (processing_frame.shape[1], processing_frame.shape[0])
        crop_rect = None
        if crop_to_table:
            polygon, _ = module._phase_table_geometry(image.shape)
            processing_frame, crop_rect = module._crop_to_calibrated_table(
                processing_frame,
                polygon,
                source_resolution,
                module.phase_scheduler.settings.get("ball_crop_margin_fraction", 0.08),
            )
        predictions = detector.detect_balls_with_classification(
            processing_frame,
            frame_key=(
                "static-benchmark",
                annotation_file.name,
                dimension_name,
                crop_to_table,
                processing_resolution,
            ),
            classification_frame=processing_frame,
        )
        if crop_rect is None:
            predictions = module._normalise_balls_to_source(
                list(predictions), source_resolution, processing_resolution
            )
        else:
            predictions = module._normalise_cropped_balls_to_source(
                list(predictions),
                source_resolution,
                processing_resolution,
                crop_rect,
            )
        predicted_centers: list[tuple[float, float]] = []
        predicted_radii: list[float] = []
        for ball in predictions:
            if getattr(ball, "coordinate_space", "4k_canonical") == "4k_canonical":
                center = canonical_point_to_source_pixels(
                    ball.position, source_resolution
                )
                from backend.core.resolution_converter import ResolutionConverter

                radius = ResolutionConverter.scale_distance_from_4k(
                    ball.radius, source_resolution
                )
            elif getattr(ball, "coordinate_space", None) == "pixel":
                center = (float(ball.position[0]), float(ball.position[1]))
                radius = float(ball.radius)
            else:
                continue
            predicted_centers.append((float(center[0]), float(center[1])))
            predicted_radii.append(float(radius))

        ground_truth = annotation.get("balls", [])
        ground_truth_count += len(ground_truth)
        prediction_count += len(predicted_centers)
        pairs = []
        for truth_index, truth in enumerate(ground_truth):
            center = truth.get("center", {})
            truth_center = (float(center.get("x", 0)), float(center.get("y", 0)))
            for prediction_index, prediction_center in enumerate(predicted_centers):
                pairs.append(
                    (
                        math.dist(truth_center, prediction_center),
                        truth_index,
                        prediction_index,
                    )
                )

        assigned_truth: set[int] = set()
        assigned_predictions: set[int] = set()
        for distance, truth_index, prediction_index in sorted(pairs):
            truth = ground_truth[truth_index]
            truth_radius = float(truth.get("radius", 0))
            if distance > max(1.0, truth_radius * 0.5):
                continue
            if (
                truth_index in assigned_truth
                or prediction_index in assigned_predictions
            ):
                continue
            assigned_truth.add(truth_index)
            assigned_predictions.add(prediction_index)
            matched_count += 1
            distance_errors.append(distance)
            radius_errors.append(abs(truth_radius - predicted_radii[prediction_index]))
            truth_center = truth["center"]
            edge = (
                float(truth_center["x"]) <= source_width * edge_margin_fraction
                or float(truth_center["x"]) >= source_width * (1 - edge_margin_fraction)
                or float(truth_center["y"]) <= source_height * edge_margin_fraction
                or float(truth_center["y"])
                >= source_height * (1 - edge_margin_fraction)
            )
            if edge:
                edge_matched_count += 1
                edge_distance_errors.append(distance)

        for truth in ground_truth:
            center = truth.get("center", {})
            if (
                float(center.get("x", 0)) <= source_width * edge_margin_fraction
                or float(center.get("x", 0))
                >= source_width * (1 - edge_margin_fraction)
                or float(center.get("y", 0)) <= source_height * edge_margin_fraction
                or float(center.get("y", 0))
                >= source_height * (1 - edge_margin_fraction)
            ):
                edge_ground_truth_count += 1

    precision = matched_count / prediction_count if prediction_count else None
    recall = matched_count / ground_truth_count if ground_truth_count else None
    return {
        "dataset_directory": str(dataset_directory),
        "annotated_images": image_count,
        "ground_truth_balls": ground_truth_count,
        "predicted_balls": prediction_count,
        "matched_balls": matched_count,
        "center_match_radius": "half the annotated ball radius",
        "ball_precision": precision,
        "ball_recall": recall,
        "ball_f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        ),
        "center_error_px_mean": _mean(distance_errors),
        "center_error_px_p95": _percentile(distance_errors, 95),
        "radius_error_px_mean": _mean(radius_errors),
        "edge_margin_fraction": edge_margin_fraction,
        "edge_ground_truth_balls": edge_ground_truth_count,
        "edge_matched_balls": edge_matched_count,
        "edge_center_error_px_mean": _mean(edge_distance_errors),
        "calibrated_table_crop": crop_to_table,
    }


def benchmark_candidate(
    module: VisionModule,
    clock: ReplayClock,
    video_path: Path,
    *,
    name: str,
    settings: dict[str, Any],
    frame_limit: int | None,
    start_seconds: float,
    pace: str = "capacity",
    temporal_annotations: dict[str, Any] | None = None,
    temporal_freshness_budget_seconds: float = 0.1,
    temporal_matching_radius_fraction: float = 0.5,
) -> dict[str, Any]:
    """Replay one candidate and collect phase-specific measurements."""
    module.phase_scheduler = VisionPhaseScheduler(settings)
    module.motion_monitor = MotionMonitor(
        config.get("vision.processing.motion_monitor", {})
    )
    tracker = module.tracker
    if tracker is None:
        raise RuntimeError("profile replay requires an enabled object tracker")
    tracker.reset()
    tracker.clock = clock
    module._phase_geometry_signature = None
    module._phase_last_ball_map = []
    module._phase_last_ball_timestamp = None
    module._phase_last_ball_source_timestamp = None
    module._phase_last_ball_source_frame_number = None
    module._phase_last_ball_source_timestamp_kind = None
    module._phase_last_cue = None
    module._phase_last_cue_timestamp = None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    source_fps = _number(capture.get(cv2.CAP_PROP_FPS)) or float(
        config.get("vision.camera.fps", 30.0)
    )
    frame_period = 1.0 / max(source_fps, 1.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    source_frame_index_base = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    clock.start_replay()
    source_epoch = time.time()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    phase_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "frames": 0,
            "ball_inference_attempts": 0,
            "ball_observations": 0,
            "cue_inference_attempts": 0,
            "cue_observations": 0,
            "motion_monitor_observations": 0,
            "processing_ms": [],
            "stationary_positions": defaultdict(list),
            "cue_angles_radians": [],
            "trusted_map_frames": 0,
        }
    )
    track_observation_times: dict[str, list[float]] = defaultdict(list)
    latest_motion_elapsed: float | None = None
    recovery_times: list[float] = []
    temporal_frames: list[dict[str, Any]] = []
    frame_index = 0
    previous_presentation_time = -frame_period

    try:
        while frame_limit is None or frame_index < frame_limit:
            success, frame = capture.read()
            if not success:
                break
            frame_uint8: NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
            presentation_time = _number(capture.get(cv2.CAP_PROP_POS_MSEC))
            elapsed = (
                presentation_time / 1000.0 - start_seconds
                if presentation_time is not None
                else frame_index * frame_period
            )
            if elapsed <= previous_presentation_time:
                elapsed = frame_index * frame_period
            previous_presentation_time = elapsed
            if pace == "realtime":
                due_at = started_wall + elapsed
                delay = due_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            clock.set_elapsed(elapsed)
            publication_time = source_epoch + elapsed
            result = module._process_single_frame(
                frame_uint8,
                frame_index,
                publication_time,
                source_frame_number=source_frame_index_base + frame_index,
                source_timestamp=publication_time,
                source_timestamp_kind="replay_video_presentation",
            )
            frame_index += 1
            if result is None:
                continue

            temporal_frames.append(
                _normalize_temporal_result(
                    result,
                    frame_index=source_frame_index_base + frame_index - 1,
                    presentation_time_seconds=elapsed + start_seconds,
                    source_epoch=source_epoch,
                    presentation_time_offset_seconds=start_seconds,
                    source_resolution=(source_width, source_height),
                )
            )

            phase = _phase_name(result)
            metrics = phase_metrics[phase]
            metrics["frames"] += 1
            processing_ms = _number(result.statistics.processing_time)
            if processing_ms is not None:
                metrics["processing_ms"].append(processing_ms)
            if result.ball_detection_status in {"observed", "failed"}:
                metrics["ball_inference_attempts"] += 1
            if result.ball_detection_status == "observed":
                metrics["ball_observations"] += 1
            if result.cue_detection_status in {"observed", "failed"}:
                metrics["cue_inference_attempts"] += 1
            if result.cue_detection_status == "observed":
                metrics["cue_observations"] += 1
            if result.motion_monitor_status == "observed":
                metrics["motion_monitor_observations"] += 1
            if result.map_trusted:
                metrics["trusted_map_frames"] += 1

            if phase in {"limited_motion", "many_moving"}:
                latest_motion_elapsed = elapsed
            elif (
                latest_motion_elapsed is not None
                and phase in {"aiming", "idle"}
                and result.map_trusted
            ):
                recovery_times.append(max(0.0, elapsed - latest_motion_elapsed))
                latest_motion_elapsed = None

            if result.ball_positions_status == "observed":
                for ball in result.balls:
                    track_id = getattr(ball, "track_id", None)
                    if track_id is not None:
                        track_observation_times[str(track_id)].append(elapsed)
                    if (
                        phase in {"aiming", "idle"}
                        and not getattr(ball, "is_moving", False)
                        and track_id is not None
                    ):
                        position = getattr(ball, "position", None)
                        if position is not None:
                            metrics["stationary_positions"][str(track_id)].append(
                                (float(position[0]), float(position[1]))
                            )

            cue = getattr(result, "cue", None)
            if phase == "aiming" and cue is not None:
                angle = _number(getattr(cue, "angle", None))
                if angle is not None:
                    metrics["cue_angles_radians"].append(angle)
    finally:
        capture.release()

    formatted_phases: dict[str, Any] = {}
    for phase, metrics in phase_metrics.items():
        positions = metrics.pop("stationary_positions")
        cue_angles = metrics.pop("cue_angles_radians")
        jitter_by_track = []
        for samples in positions.values():
            if len(samples) > 1:
                coordinates = np.asarray(samples, dtype=float)
                jitter_by_track.append(
                    float(np.sqrt(np.mean(np.var(coordinates, axis=0))))
                )
        unwrapped_cue_angles = np.unwrap(np.asarray(cue_angles, dtype=float))
        formatted_phases[phase] = {
            **metrics,
            "processing_ms_p50": _percentile(metrics["processing_ms"], 50),
            "processing_ms_p95": _percentile(metrics["processing_ms"], 95),
            "processing_ms": None,
            "stationary_ball_position_jitter_px": _mean(jitter_by_track),
            "cue_angle_stability_std_degrees": (
                float(np.degrees(np.std(unwrapped_cue_angles)))
                if unwrapped_cue_angles.size > 1
                else None
            ),
        }

    all_processing = [
        value
        for metrics in phase_metrics.values()
        for value in metrics.get("processing_ms", [])
    ]
    gaps = [
        current - previous
        for observations in track_observation_times.values()
        for previous, current in zip(observations, observations[1:])
    ]
    temporal_segment_annotations: dict[str, Any] | None = None
    if temporal_annotations is not None and frame_index > 0:
        source_frame_end_exclusive = source_frame_index_base + frame_index
        selected_annotation_frames = [
            frame
            for frame in temporal_annotations["frames"]
            if source_frame_index_base
            <= frame["source_frame_index"]
            < source_frame_end_exclusive
        ]
        if selected_annotation_frames:
            temporal_segment_annotations = {
                **temporal_annotations,
                "frames": selected_annotation_frames,
            }
    if temporal_segment_annotations is not None:
        temporal_metrics = evaluate_temporal_localization(
            temporal_segment_annotations,
            temporal_frames,
            matching_radius_fraction=temporal_matching_radius_fraction,
            freshness_budget_seconds=temporal_freshness_budget_seconds,
        )
    elif temporal_annotations is not None:
        temporal_metrics = unavailable_temporal_metrics(
            "no annotation frames overlap the selected replay segment"
        )
    else:
        temporal_metrics = unavailable_temporal_metrics()
    return {
        "candidate": name,
        "pace": pace,
        "video": str(video_path),
        "start_seconds": start_seconds,
        "source_resolution": [source_width, source_height],
        "source_fps": source_fps,
        "frames_processed": frame_index,
        "temporal_evaluation_segment": {
            "first_source_frame_index": (
                source_frame_index_base if frame_index else None
            ),
            "last_source_frame_index": (
                source_frame_index_base + frame_index - 1 if frame_index else None
            ),
            "annotated_frames": (
                len(temporal_segment_annotations["frames"])
                if temporal_segment_annotations is not None
                else 0
            ),
        },
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "processing_ms_p50": _percentile(all_processing, 50),
        "processing_ms_p95": _percentile(all_processing, 95),
        "moving_ball_observation_gap_seconds_p95": _percentile(gaps, 95),
        "moving_ball_observation_gap_seconds_max": max(gaps) if gaps else None,
        "motion_to_trusted_map_seconds_mean": _mean(recovery_times),
        "phases": formatted_phases,
        "temporal_localization": temporal_metrics,
        "unmeasured_metrics": [
            "stationary map jitter per live phase and cue-angle error against labels",
            "aim-assist calculation latency",
            "missed collision opportunities",
            "camera publication-age freshness (file timestamps are replay timestamps)",
            "manual move, occlusion, and edge-ball recovery unless labeled per frame",
        ],
        "limitations": [
            "processing frequency follows recorded presentation timestamps through an injected clock",
            "uses configured table.playing_area_corners; verify they match the recording",
            "candidate measurements do not validate the phase profile for production defaults",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--pace",
        choices=("realtime", "capacity"),
        default="capacity",
        help="pace replay by source timestamps or process as fast as possible",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--dimensions", default="native,720p")
    parser.add_argument("--cadence-factors", default="0.5,1,2")
    parser.add_argument("--crop-modes", default="full-frame,table")
    parser.add_argument(
        "--static-ground-truth-dir",
        type=Path,
        default=ROOT / "backend" / "vision" / "test_data",
    )
    parser.add_argument("--edge-margin-fraction", type=float, default=0.15)
    parser.add_argument("--no-static-ground-truth", action="store_true")
    parser.add_argument(
        "--temporal-annotations",
        type=Path,
        help="version-1 temporal localization JSON labels for this exact video",
    )
    parser.add_argument("--temporal-matching-radius-fraction", type=float, default=0.5)
    parser.add_argument("--temporal-freshness-budget-seconds", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video.is_file():
        raise SystemExit(f"video does not exist: {args.video}")
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        raise SystemExit("--start-seconds must be finite and non-negative")

    dimensions = parse_dimensions(args.dimensions)
    cadence_factors = parse_factors(args.cadence_factors)
    crop_modes = [token.strip().lower() for token in args.crop_modes.split(",")]
    if not crop_modes or set(crop_modes) - {"full-frame", "table"}:
        raise SystemExit("--crop-modes must contain full-frame and/or table")
    if not math.isfinite(args.temporal_matching_radius_fraction) or not (
        0 < args.temporal_matching_radius_fraction <= 2
    ):
        raise SystemExit("--temporal-matching-radius-fraction must be in (0, 2]")
    if (
        not math.isfinite(args.temporal_freshness_budget_seconds)
        or args.temporal_freshness_budget_seconds < 0
    ):
        raise SystemExit(
            "--temporal-freshness-budget-seconds must be finite and non-negative"
        )

    identity_capture = cv2.VideoCapture(str(args.video))
    if not identity_capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    video_geometry = (
        int(identity_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(identity_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    identity_capture.release()
    temporal_annotations = None
    if args.temporal_annotations is not None:
        try:
            temporal_annotations = load_temporal_annotations(
                args.temporal_annotations,
                expected_video_sha256=_hash_file(args.video),
                expected_geometry=video_geometry,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid temporal annotations: {error}") from error

    base_settings = config.get("vision.processing.phase_scheduler", {})
    clock = ReplayClock()
    module = VisionModule(config={"phase_scheduler": base_settings}, clock=clock)
    if module.detector is None or not module.detector.model_loaded:
        module.close()
        raise SystemExit(
            "recorded-profile benchmarks require a configured, loaded detector model; "
            "set vision.detection.yolo_model_path to the installed model"
        )
    static_by_dimension: dict[tuple[str, bool], Any] = {}
    if not args.no_static_ground_truth:
        if not 0.0 <= args.edge_margin_fraction < 0.5:
            raise SystemExit("--edge-margin-fraction must be between 0 and 0.5")
        for label, maximum_dimension in dimensions:
            for crop_mode in crop_modes:
                crop_to_table = crop_mode == "table"
                print(
                    f"Benchmarking annotated stills at {label} resolution "
                    f"with {crop_mode} ball view..."
                )
                static_by_dimension[(label, crop_to_table)] = (
                    benchmark_static_ground_truth(
                        module,
                        args.static_ground_truth_dir,
                        dimension_name=label,
                        maximum_dimension=maximum_dimension,
                        edge_margin_fraction=args.edge_margin_fraction,
                        crop_to_table=crop_to_table,
                    )
                )
    candidates = []
    for label, maximum_dimension in dimensions:
        for cadence in cadence_factors:
            for crop_mode in crop_modes:
                crop_to_table = crop_mode == "table"
                settings = build_candidate_settings(
                    base_settings,
                    maximum_dimension=maximum_dimension,
                    cadence_multiplier=cadence,
                    crop_to_table=crop_to_table,
                )
                name = f"{label}-{crop_mode}-cadence-x{cadence:g}"
                print(f"Benchmarking {name} on {args.video}...")
                candidate_report = benchmark_candidate(
                    module,
                    clock,
                    args.video,
                    name=name,
                    settings=settings,
                    frame_limit=args.frames,
                    start_seconds=args.start_seconds,
                    pace=args.pace,
                    temporal_annotations=temporal_annotations,
                    temporal_freshness_budget_seconds=args.temporal_freshness_budget_seconds,
                    temporal_matching_radius_fraction=args.temporal_matching_radius_fraction,
                )
                candidate_report["static_ball_accuracy"] = static_by_dimension.get(
                    (label, crop_to_table)
                )
                candidates.append(candidate_report)

    report = {
        "benchmark": "phase-aware-vision-profiles",
        "default_selection": "unchanged; reduced profiles require labeled repeatable validation",
        "metadata": _run_metadata(args.video.resolve(), args.output),
        "detector": {
            "implementation": type(module.detector).__name__,
            "model_loaded": bool(module.detector and module.detector.model_loaded),
            "device": getattr(module.detector, "device", None),
            "model_format": getattr(
                getattr(module.detector, "model_format", None), "value", None
            ),
        },
        "pace_semantics": {
            "realtime": "source presentation timestamps paced against wall time; measures freshness behavior",
            "capacity": "frames processed as quickly as possible; measures processing capacity, not live freshness",
        },
        "accuracy_scope": {
            "available": [
                "annotated still-frame ball precision and recall",
                "ball center and radius error",
                *(
                    [
                        "temporal localization, tracking, cue, freshness, and boundary metrics"
                    ]
                    if temporal_annotations is not None
                    else []
                ),
            ],
            "missing_temporal_annotations": (
                []
                if temporal_annotations is not None
                else [
                    "track identity continuity across motion and occlusion",
                    "cue presence, position, and angle quality over time",
                    "component observation freshness against labeled observations",
                    "calibration changes, camera interruption, and restart recovery",
                ]
            ),
            "temporal_annotation_status": (
                "available" if temporal_annotations is not None else "unavailable"
            ),
        },
        "temporal_evaluation": {
            "status": (
                "available" if temporal_annotations is not None else "unavailable"
            ),
            "annotation_path": (
                str(args.temporal_annotations.resolve())
                if args.temporal_annotations is not None
                else None
            ),
            "annotation_sha256": (
                _hash_file(args.temporal_annotations)
                if args.temporal_annotations is not None
                else None
            ),
            "schema_version": 1 if temporal_annotations is not None else None,
            "matching_radius_fraction": args.temporal_matching_radius_fraction,
            "freshness_budget_seconds": args.temporal_freshness_budget_seconds,
            "timing_basis": "recorded-video presentation time; not camera exposure time",
        },
        "candidates": candidates,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
