"""Strict temporal localization annotations and engine-neutral scoring.

Temporal scores are tied to source-frame identity. A cached position is useful
state, but it is not a new detection and therefore cannot improve detection
precision, recall, or freshness. All timestamps in this module refer to
recorded-video presentation time; they are not camera exposure timestamps.

The JSON Schema defines the annotation wire format. The semantic validator
below adds constraints that JSON Schema cannot conveniently express, notably
strict ordering, per-frame stable-ID uniqueness, finite floating-point values,
and compatibility with the actual input video.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import jsonschema  # type: ignore[import-untyped]
import numpy as np
from jsonschema import Draft202012Validator
from scipy.optimize import linear_sum_assignment

SCHEMA_PATH = (
    Path(__file__).parent
    / "schemas"
    / "temporal_localization_annotations_v1.schema.json"
)
SCENE_SNAPSHOT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "perception"
    / "contracts"
    / "v1"
    / "scene-snapshot.schema.json"
)
_STATUSES = {"observed", "cached", "skipped", "failed", "unavailable"}
MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991
_RUST_REPLAY_FRAME_ID = re.compile(r"^frame-(\d+)$")


class TemporalAnnotationError(ValueError):
    """Raised when annotations or normalized engine records are invalid."""


def _reject_json_constant(token: str) -> None:
    """Reject non-standard NaN and infinity tokens accepted by Python's JSON parser."""
    raise TemporalAnnotationError(f"non-finite JSON number is not allowed: {token}")


def _assert_finite_tree(value: Any, location: str = "$") -> None:
    """Reject non-finite values, including values represented by huge exponents."""
    if isinstance(value, float) and not math.isfinite(value):
        raise TemporalAnnotationError(f"non-finite number at {location}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{location}[{index}]")


def _load_schema() -> dict[str, Any]:
    try:
        schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not load temporal schema at {SCHEMA_PATH}"
        ) from error
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def validate_annotation_document(
    document: Mapping[str, Any],
    *,
    expected_video_sha256: str | None = None,
    expected_geometry: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Validate schema, sequence identity, ordered frame times, and unique IDs.

    The input hash and geometry are checked before evaluation so annotations
    from a visually similar but different recording cannot silently be scored.
    The returned value is a shallow plain-dict copy suitable for serialization.
    """
    if not isinstance(document, Mapping):
        raise TemporalAnnotationError("annotation root must be an object")
    normalized = dict(document)
    _assert_finite_tree(normalized)
    try:
        jsonschema.validate(normalized, _load_schema())
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        raise TemporalAnnotationError(
            f"annotation schema violation at {location}: {error.message}"
        ) from error

    sequence = normalized["sequence"]
    for dimension_name in ("width", "height"):
        dimension = sequence[dimension_name]
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TemporalAnnotationError(
                f"sequence {dimension_name} must be an integer"
            )
    schema_version = normalized["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TemporalAnnotationError("schema_version must be the integer 1")
    actual_hash = str(sequence["video_sha256"]).lower()
    if (
        expected_video_sha256 is not None
        and actual_hash != expected_video_sha256.lower()
    ):
        raise TemporalAnnotationError("annotation video SHA-256 does not match input")
    actual_geometry = (int(sequence["width"]), int(sequence["height"]))
    if expected_geometry is not None and actual_geometry != expected_geometry:
        raise TemporalAnnotationError(
            "annotation geometry does not match input video "
            f"({actual_geometry[0]}x{actual_geometry[1]} != "
            f"{expected_geometry[0]}x{expected_geometry[1]})"
        )

    previous_index = -1
    previous_timestamp = -math.inf
    for frame_number, frame in enumerate(normalized["frames"]):
        source_index = frame["source_frame_index"]
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TemporalAnnotationError(
                f"annotation source_frame_index in frame {frame_number} must be an integer"
            )
        presentation_time = float(frame["presentation_time_seconds"])
        if source_index <= previous_index:
            raise TemporalAnnotationError(
                "annotation source_frame_index values must be strictly increasing "
                f"(frame {frame_number})"
            )
        if presentation_time <= previous_timestamp:
            raise TemporalAnnotationError(
                "annotation presentation timestamps must be strictly increasing "
                f"(frame {frame_number})"
            )
        previous_index = source_index
        previous_timestamp = presentation_time

        calibration_revision = frame["expected_calibration_revision"]
        if calibration_revision is not None and (
            isinstance(calibration_revision, bool)
            or not isinstance(calibration_revision, int)
            or not 0 <= calibration_revision <= MAX_SAFE_JSON_INTEGER
        ):
            raise TemporalAnnotationError(
                "expected_calibration_revision must be a non-negative safe integer or null"
            )

        ball_ids = [ball["ball_id"] for ball in frame["balls"]]
        if len(ball_ids) != len(set(ball_ids)):
            raise TemporalAnnotationError(
                f"duplicate ground-truth ball_id in frame {source_index}"
            )
    return normalized


def load_temporal_annotations(
    path: Path,
    *,
    expected_video_sha256: str,
    expected_geometry: tuple[int, int],
) -> dict[str, Any]:
    """Read and validate temporal annotations against a recorded video."""
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalAnnotationError(
            f"could not read annotation file: {path}"
        ) from error
    return validate_annotation_document(
        document,
        expected_video_sha256=expected_video_sha256,
        expected_geometry=expected_geometry,
    )


def load_rust_scene_snapshots(
    path: Path,
    annotations: Mapping[str, Any],
    *,
    frame_index_offset: int = 0,
) -> list[dict[str, Any]]:
    """Normalize Rust scene-snapshot JSONL into this evaluator's frame contract.

    Rust replay frame IDs are one-based (``frame-000001``); annotation indices
    are zero-based. ``frame_index_offset`` explicitly maps segmented or
    trimmed replays into the annotation sequence. Only observation snapshots
    are scored. Annotation/health snapshots are publications of retained state,
    not new detector results. Rust monotonic ticks are deliberately ignored:
    presentation time comes only from the matching annotation frame.
    """
    if isinstance(frame_index_offset, bool) or not isinstance(frame_index_offset, int):
        raise TemporalAnnotationError("frame_index_offset must be an integer")
    normalized_annotations = validate_annotation_document(annotations)
    annotation_by_index = {
        frame["source_frame_index"]: frame for frame in normalized_annotations["frames"]
    }
    try:
        scene_schema = json.loads(
            SCENE_SNAPSHOT_SCHEMA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not load Rust scene schema at {SCENE_SNAPSHOT_SCHEMA_PATH}"
        ) from error
    Draft202012Validator.check_schema(scene_schema)
    validator = Draft202012Validator(scene_schema)

    normalized_frames: list[dict[str, Any]] = []
    last_transport_sequence: dict[tuple[str, str], int] = {}
    seen_observation_indices: set[int] = set()
    previous_observation_index = -1
    try:
        source = path.open("r", encoding="utf-8")
    except OSError as error:
        raise TemporalAnnotationError(
            f"could not read Rust scene JSONL: {path}"
        ) from error

    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise TemporalAnnotationError(
                    f"blank Rust scene JSONL line at {line_number}"
                )
            try:
                snapshot = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, TemporalAnnotationError) as error:
                raise TemporalAnnotationError(
                    f"invalid Rust scene JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(snapshot, Mapping):
                raise TemporalAnnotationError(
                    f"Rust scene at line {line_number} must be an object"
                )
            _assert_finite_tree(snapshot, f"scene[{line_number}]")
            errors = sorted(
                validator.iter_errors(snapshot),
                key=lambda error: list(error.absolute_path),
            )
            if errors:
                schema_error = errors[0]
                location = (
                    ".".join(str(part) for part in schema_error.absolute_path) or "$"
                )
                raise TemporalAnnotationError(
                    f"Rust scene schema violation at line {line_number} "
                    f"({location}): {schema_error.message}"
                )

            source_record = snapshot["source"]
            source_session = source_record["session_id"]
            identity_key = (snapshot["engine_instance_id"], source_session)
            sequence = snapshot["transport_sequence"]
            previous_sequence = last_transport_sequence.get(identity_key)
            if previous_sequence is not None and sequence <= previous_sequence:
                raise TemporalAnnotationError(
                    f"Rust transport sequence is not increasing at line {line_number}"
                )
            last_transport_sequence[identity_key] = sequence

            match = _RUST_REPLAY_FRAME_ID.fullmatch(source_record["frame_id"])
            if match is None or int(match.group(1)) < 1:
                raise TemporalAnnotationError(
                    "Rust replay frame_id must use one-based frame-N identifiers "
                    f"(line {line_number})"
                )
            source_index = int(match.group(1)) - 1 + frame_index_offset
            if source_index < 0:
                raise TemporalAnnotationError(
                    f"Rust frame index maps below zero at line {line_number}"
                )
            if snapshot["update_kind"] != "observation":
                continue
            if source_index in seen_observation_indices:
                raise TemporalAnnotationError(
                    f"duplicate Rust observation source frame {source_index}"
                )
            if source_index <= previous_observation_index:
                raise TemporalAnnotationError(
                    "Rust observation frames must be strictly increasing"
                )
            seen_observation_indices.add(source_index)
            previous_observation_index = source_index

            annotation_frame = annotation_by_index.get(source_index)
            if annotation_frame is None:
                # Sparse temporal labels are valid; keep only corresponding
                # engine frames so neither replay nor wall-clock time is used
                # to invent timestamps for unlabelled frames.
                continue
            geometry = (
                source_record["width_px"],
                source_record["height_px"],
            )
            expected_geometry = (
                normalized_annotations["sequence"]["width"],
                normalized_annotations["sequence"]["height"],
            )
            if geometry != expected_geometry:
                raise TemporalAnnotationError(
                    "Rust scene geometry does not match temporal labels "
                    f"at source frame {source_index}"
                )

            outcome = snapshot["outcome"]
            if outcome == "failed":
                ball_status = "failed"
            elif outcome == "skipped":
                ball_status = "skipped"
            elif outcome in {"success", "degraded"}:
                ball_status = "observed"
            else:  # schema validation makes this unreachable
                ball_status = "unavailable"

            balls: list[dict[str, Any]] = []
            for ball in snapshot["balls"]:
                if ball["status"] != "observed":
                    continue
                measurement = ball["measurement"]
                if measurement is None:
                    raise TemporalAnnotationError(
                        f"observed Rust ball has no measurement at source frame {source_index}"
                    )
                if measurement["image_space"] != "raw":
                    raise TemporalAnnotationError(
                        "Rust temporal scoring currently requires raw source-pixel "
                        f"coordinates (source frame {source_index})"
                    )
                if (
                    measurement["image_width_px"],
                    measurement["image_height_px"],
                ) != expected_geometry:
                    raise TemporalAnnotationError(
                        "Rust ball measurement geometry differs from the labeled "
                        f"source image at frame {source_index}"
                    )
                last_observation = ball["last_observation"]
                if (
                    last_observation["session_id"] != source_session
                    or last_observation["frame_id"] != source_record["frame_id"]
                ):
                    raise TemporalAnnotationError(
                        "observed Rust ball provenance does not match its scene "
                        f"source frame {source_index}"
                    )
                balls.append(
                    {
                        "track_id": ball["track_id"],
                        "center": measurement["image_center_px"],
                    }
                )

            cue_status = "unavailable"
            cue_present: bool | None = None
            cue_center: Mapping[str, Any] | None = None
            cue = snapshot["cue"]
            if cue is not None and cue["status"] in {"observed", "not_observed"}:
                cue_observation = cue["last_observation"]
                if (
                    cue_observation is not None
                    and cue_observation["session_id"] == source_session
                    and cue_observation["frame_id"] == source_record["frame_id"]
                ):
                    cue_status = "observed"
                    cue_present = cue["status"] == "observed"
                    cue_center = cue["image_center_px"]
                    if cue_center is not None:
                        cue_geometry = (
                            cue["image_width_px"],
                            cue["image_height_px"],
                        )
                        if (
                            cue["image_space"] != "raw"
                            or cue_geometry != expected_geometry
                        ):
                            cue_status = "unavailable"
                            cue_present = None
                            cue_center = None

            revisions = snapshot["revisions"]
            calibration_revision = (
                revisions["calibration"]
                if revisions["calibration_state"] == "valid"
                else None
            )
            frame_record: dict[str, Any] = {
                "source_frame_index": source_index,
                "presentation_time_seconds": annotation_frame[
                    "presentation_time_seconds"
                ],
                "ball_positions_status": ball_status,
                "ball_observation_source_frame_index": (
                    source_index if ball_status == "observed" else None
                ),
                "ball_observation_presentation_time_seconds": (
                    annotation_frame["presentation_time_seconds"]
                    if ball_status == "observed"
                    else None
                ),
                "balls": balls if ball_status == "observed" else [],
                "cue_observation_status": cue_status,
                "cue_observation_source_frame_index": (
                    source_index if cue_status == "observed" else None
                ),
                "cue_present": cue_present,
                "calibration_revision": calibration_revision,
                "source_session_epoch": source_session,
            }
            if cue_present and cue_center is not None:
                frame_record["cue_center"] = dict(cue_center)
            normalized_frames.append(frame_record)

    if not normalized_frames:
        raise TemporalAnnotationError(
            "no Rust observation frames overlap the labeled temporal sequence"
        )
    return normalized_frames


def _finite_number(value: Any, label: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalAnnotationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise TemporalAnnotationError(f"{label} must be a finite number")
    return number


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Mapping) or "x" not in value or "y" not in value:
        raise TemporalAnnotationError(f"{label} must contain x and y")
    x = _finite_number(value["x"], f"{label}.x")
    y = _finite_number(value["y"], f"{label}.y")
    if x is None or y is None:
        raise TemporalAnnotationError(f"{label} coordinates cannot be null")
    return x, y


def _validate_engine_frames(
    engine_frames: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Validate and index normalized engine records, rejecting ambiguous order."""
    indexed: dict[int, dict[str, Any]] = {}
    previous_index = -1
    previous_time = -math.inf
    for offset, raw_frame in enumerate(engine_frames):
        if not isinstance(raw_frame, Mapping):
            raise TemporalAnnotationError(f"engine frame {offset} must be an object")
        frame = dict(raw_frame)
        index = frame.get("source_frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise TemporalAnnotationError(
                f"engine frame {offset} source_frame_index must be a non-negative integer"
            )
        if index in indexed:
            raise TemporalAnnotationError(f"duplicate engine source frame {index}")
        if index <= previous_index:
            raise TemporalAnnotationError(
                "engine source frames must be strictly ordered"
            )
        presentation_time = _finite_number(
            frame.get("presentation_time_seconds"),
            f"engine frame {index} presentation_time_seconds",
        )
        assert presentation_time is not None
        if presentation_time < 0 or presentation_time <= previous_time:
            raise TemporalAnnotationError(
                "engine presentation timestamps must be non-negative and strictly increasing"
            )
        previous_index = index
        previous_time = presentation_time

        status = frame.get("ball_positions_status", "unavailable")
        if status not in _STATUSES:
            raise TemporalAnnotationError(f"invalid ball_positions_status: {status!r}")
        balls = frame.get("balls", [])
        if not isinstance(balls, list):
            raise TemporalAnnotationError(
                f"engine frame {index} balls must be an array"
            )
        normalized_balls: list[dict[str, Any]] = []
        for ball_number, raw_ball in enumerate(balls):
            if not isinstance(raw_ball, Mapping):
                raise TemporalAnnotationError(
                    f"engine ball {ball_number} in frame {index} must be an object"
                )
            center = _point(raw_ball.get("center"), f"engine ball {ball_number}.center")
            track_id = raw_ball.get("track_id")
            if track_id is not None and (
                isinstance(track_id, bool) or not isinstance(track_id, (str, int))
            ):
                raise TemporalAnnotationError(
                    "engine track_id must be a string, integer, or null"
                )
            normalized_balls.append(
                {
                    "center": center,
                    "track_id": None if track_id is None else str(track_id),
                }
            )
        frame["balls"] = normalized_balls

        source_index = frame.get("ball_observation_source_frame_index")
        if source_index is not None and (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise TemporalAnnotationError(
                "ball_observation_source_frame_index must be a non-negative integer or null"
            )
        source_time = _finite_number(
            frame.get("ball_observation_presentation_time_seconds"),
            "ball_observation_presentation_time_seconds",
            optional=True,
        )
        if source_time is not None and source_time < 0:
            raise TemporalAnnotationError(
                "ball observation presentation time cannot be negative"
            )
        frame["ball_observation_source_frame_index"] = source_index
        frame["ball_observation_presentation_time_seconds"] = source_time

        cue_status = frame.get("cue_observation_status", "unavailable")
        if cue_status not in _STATUSES:
            raise TemporalAnnotationError(
                f"invalid cue_observation_status: {cue_status!r}"
            )
        cue_present = frame.get("cue_present")
        if cue_status == "observed" and not isinstance(cue_present, bool):
            raise TemporalAnnotationError(
                "cue_present must be a boolean when cue_observation_status is observed"
            )
        if cue_present is True:
            if frame.get("cue_center") is not None:
                frame["cue_center"] = _point(frame["cue_center"], "cue_center")
            has_endpoint_geometry = (
                frame.get("cue_tip") is not None or frame.get("cue_butt") is not None
            )
            if has_endpoint_geometry:
                frame["cue_tip"] = _point(frame.get("cue_tip"), "cue_tip")
                frame["cue_butt"] = _point(frame.get("cue_butt"), "cue_butt")
                angle = _finite_number(
                    frame.get("cue_angle_degrees"),
                    "cue_angle_degrees",
                    optional=True,
                )
                if angle is None:
                    tip_x, tip_y = frame["cue_tip"]
                    butt_x, butt_y = frame["cue_butt"]
                    angle = math.degrees(math.atan2(tip_y - butt_y, tip_x - butt_x))
                frame["cue_angle_degrees"] = angle
        cue_source_index = frame.get("cue_observation_source_frame_index")
        if cue_source_index is not None and (
            isinstance(cue_source_index, bool)
            or not isinstance(cue_source_index, int)
            or cue_source_index < 0
        ):
            raise TemporalAnnotationError(
                "cue_observation_source_frame_index must be a non-negative integer or null"
            )
        frame["cue_observation_source_frame_index"] = cue_source_index
        indexed[index] = frame
        calibration_revision = frame.get("calibration_revision")
        if calibration_revision is not None and (
            isinstance(calibration_revision, bool)
            or not isinstance(calibration_revision, int)
            or not 0 <= calibration_revision <= MAX_SAFE_JSON_INTEGER
        ):
            raise TemporalAnnotationError(
                "engine calibration_revision must be a non-negative safe integer or null"
            )
    return indexed


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return (
        float(np.percentile(np.asarray(values, dtype=float), percentile))
        if values
        else None
    )


def _summary(values: Sequence[float], prefix: str) -> dict[str, float | int | None]:
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": fmean(values) if values else None,
        f"{prefix}_p50": _percentile(values, 50),
        f"{prefix}_p95": _percentile(values, 95),
    }


def _global_spatial_assignment(
    truths: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    matching_radius_fraction: float,
) -> list[tuple[int, int, float]]:
    """Maximize valid one-to-one matches, then minimize total center distance.

    Dummy columns make a match preferable to any unmatched assignment. The
    global Hungarian assignment avoids order-dependent greedy collisions when
    nearby balls compete for the same detector output.
    """
    if not truths or not predictions:
        return []
    truth_count = len(truths)
    prediction_count = len(predictions)
    maximum_radius = max(float(item["radius_px"]) for item in truths)
    gate_upper_bound = maximum_radius * matching_radius_fraction
    unmatched_cost = (gate_upper_bound + 1.0) * (min(truth_count, prediction_count) + 1)
    invalid_cost = unmatched_cost * (truth_count + prediction_count + 2)
    costs = np.full((truth_count, prediction_count + truth_count), unmatched_cost)
    distances: dict[tuple[int, int], float] = {}
    for truth_index, truth in enumerate(truths):
        truth_center = _point(truth["center"], "ground-truth center")
        threshold = float(truth["radius_px"]) * matching_radius_fraction
        for prediction_index, prediction in enumerate(predictions):
            distance = math.dist(truth_center, prediction["center"])
            distances[(truth_index, prediction_index)] = distance
            costs[truth_index, prediction_index] = (
                distance if distance <= threshold else invalid_cost
            )
    rows, columns = linear_sum_assignment(costs)
    matches: list[tuple[int, int, float]] = []
    for row, column in zip(rows, columns):
        if column < prediction_count:
            distance = distances[(int(row), int(column))]
            threshold = float(truths[int(row)]["radius_px"]) * matching_radius_fraction
            if distance <= threshold:
                matches.append((int(row), int(column), distance))
    return matches


def _boundary_metrics(
    annotation_frames: Sequence[Mapping[str, Any]],
    engine_by_index: Mapping[int, Mapping[str, Any]],
    expected_key: str,
    actual_key: str,
    label: str,
    *,
    compare_frame_values: bool = True,
) -> dict[str, Any]:
    """Compare revision values or opaque-epoch transitions where both are known."""
    expected_values = [frame.get(expected_key) for frame in annotation_frames]
    actual_values = [
        engine_by_index.get(frame["source_frame_index"], {}).get(actual_key)
        for frame in annotation_frames
    ]
    expected_count = sum(value is not None for value in expected_values)
    covered_count = sum(
        expected is not None and actual is not None
        for expected, actual in zip(expected_values, actual_values)
    )
    frame_mismatches = (
        sum(
            expected is not None and actual is not None and expected != actual
            for expected, actual in zip(expected_values, actual_values)
        )
        if compare_frame_values
        else None
    )
    expected_boundaries = actual_boundaries = covered_boundaries = (
        boundary_mismatches
    ) = 0
    for offset in range(1, len(annotation_frames)):
        expected_previous, expected_current = expected_values[offset - 1 : offset + 1]
        actual_previous, actual_current = actual_values[offset - 1 : offset + 1]
        if expected_previous is not None and expected_current is not None:
            expected_changed = expected_previous != expected_current
            expected_boundaries += int(expected_changed)
            if actual_previous is not None and actual_current is not None:
                actual_changed = actual_previous != actual_current
                covered_boundaries += 1
                actual_boundaries += int(actual_changed)
                boundary_mismatches += int(expected_changed != actual_changed)
    return {
        "status": (
            "unavailable"
            if expected_count == 0 or covered_count == 0
            else ("complete" if covered_count == expected_count else "partial")
        ),
        "expected_labeled_frames": expected_count,
        "covered_frames": covered_count,
        "frame_mismatch_count": (
            frame_mismatches if covered_count and compare_frame_values else None
        ),
        "expected_boundary_count": expected_boundaries,
        "observed_boundary_count": actual_boundaries if covered_boundaries else None,
        "covered_boundary_transitions": covered_boundaries,
        "boundary_mismatch_count": boundary_mismatches if covered_boundaries else None,
        "label": label,
    }


def evaluate_temporal_localization(
    annotations: Mapping[str, Any],
    engine_frames: Sequence[Mapping[str, Any]],
    *,
    matching_radius_fraction: float = 0.5,
    freshness_budget_seconds: float = 0.1,
) -> dict[str, Any]:
    """Score normalized engine observations against version-1 temporal labels.

    Matching maximizes the number of ball matches under a radius-relative gate,
    then minimizes aggregate center distance. Detection metrics only consume
    observations explicitly marked fresh for the same source frame. Absent
    engine frames remain unavailable and contribute ground-truth misses.
    """
    if (
        not math.isfinite(matching_radius_fraction)
        or matching_radius_fraction <= 0
        or matching_radius_fraction > 2
    ):
        raise TemporalAnnotationError("matching_radius_fraction must be in (0, 2]")
    if not math.isfinite(freshness_budget_seconds) or freshness_budget_seconds < 0:
        raise TemporalAnnotationError(
            "freshness_budget_seconds must be finite and non-negative"
        )
    document = validate_annotation_document(annotations)
    annotation_frames: list[dict[str, Any]] = document["frames"]
    engine_by_index = _validate_engine_frames(engine_frames)
    presentation_by_index = {
        frame["source_frame_index"]: float(frame["presentation_time_seconds"])
        for frame in annotation_frames
    }

    visible_count = prediction_count = matched_count = 0
    center_errors: list[float] = []
    observation_ages: list[float] = []
    freshness_misses = stale_observation_count = 0
    missing_age_samples = 0
    freshness_expected_samples = len(annotation_frames)
    missing_engine_frames = 0
    track_predictions: dict[str, list[tuple[float, str | None]]] = defaultdict(list)
    visible_starts: dict[str, float] = {}
    first_detection: dict[str, float] = {}
    occlusion_previous: dict[str, str] = {}
    pending_reacquisitions: dict[str, float] = {}
    reacquisition_delays: list[float] = []
    reacquisition_attempts = reacquisition_successes = 0

    cue_tp = cue_fp = cue_fn = cue_tn = 0
    cue_labeled_count = cue_unavailable_engine_count = 0
    cue_tip_errors: list[float] = []
    cue_butt_errors: list[float] = []
    cue_angle_errors: list[float] = []
    cue_center_errors: list[float] = []
    id_switches = identity_unavailable_matches = 0
    previous_track_id: dict[str, str] = {}
    previous_labeled_session: str | None = None
    source_session_boundary_id_resets = 0

    for truth_frame in annotation_frames:
        source_index = int(truth_frame["source_frame_index"])
        current_time = float(truth_frame["presentation_time_seconds"])
        labeled_session = truth_frame.get("expected_source_session_epoch")
        if (
            labeled_session is not None
            and previous_labeled_session is not None
            and labeled_session != previous_labeled_session
        ):
            # Tracker IDs may legitimately restart with a new source process.
            # Do not call that an identity switch across a labeled session edge.
            previous_track_id.clear()
            occlusion_previous.clear()
            pending_reacquisitions.clear()
            source_session_boundary_id_resets += 1
        if labeled_session is not None:
            previous_labeled_session = str(labeled_session)

        result = engine_by_index.get(source_index)
        age: float | None = None
        if result is None:
            missing_engine_frames += 1
        else:
            result_status = result.get("ball_positions_status", "unavailable")
            observation_source_index = result.get("ball_observation_source_frame_index")
            source_time = result.get("ball_observation_presentation_time_seconds")
            if (
                source_time is None
                and observation_source_index in presentation_by_index
            ):
                source_time = presentation_by_index[observation_source_index]
            if source_time is not None and result_status in {"observed", "cached"}:
                age = current_time - float(source_time)
                if age < -1e-9:
                    raise TemporalAnnotationError(
                        f"engine ball observation is from the future at source frame {source_index}"
                    )
                age = max(0.0, age)
            fresh = (
                result_status == "observed" and observation_source_index == source_index
            )
            if fresh and age is None:
                # Source-frame identity proves this observation belongs to this
                # presentation frame even when an explicit PTS was omitted.
                age = 0.0
            if result_status == "observed" and not fresh:
                stale_observation_count += 1
            fresh_predictions = result["balls"] if fresh else []
        if result is None:
            fresh = False
            fresh_predictions = []

        if age is None:
            missing_age_samples += 1
            freshness_misses += 1
        else:
            observation_ages.append(age)
            freshness_misses += int(age > freshness_budget_seconds)

        # Establish visibility episodes before scoring this frame so a first
        # detection on the very first frame after occlusion is a zero-delay
        # reacquisition rather than being credited one frame late.
        for truth_ball in truth_frame["balls"]:
            track_key = str(truth_ball["ball_id"])
            state = str(truth_ball["status"])
            previous_state = occlusion_previous.get(track_key)
            if state == "occluded":
                occlusion_previous[track_key] = "occluded"
            elif state == "visible":
                if previous_state == "occluded":
                    pending_reacquisitions[track_key] = current_time
                    reacquisition_attempts += 1
                occlusion_previous[track_key] = "visible"
            else:
                if track_key in pending_reacquisitions:
                    pending_reacquisitions.pop(track_key)
                occlusion_previous[track_key] = "removed"

        visible_truths = [
            ball for ball in truth_frame["balls"] if ball["status"] == "visible"
        ]
        visible_count += len(visible_truths)
        prediction_count += len(fresh_predictions)
        matches = _global_spatial_assignment(
            visible_truths, fresh_predictions, matching_radius_fraction
        )
        matched_truth_indices: set[int] = set()
        for truth_index, prediction_index, distance in matches:
            truth_ball = visible_truths[truth_index]
            track_key = str(truth_ball["ball_id"])
            prediction_track_id = fresh_predictions[prediction_index]["track_id"]
            matched_truth_indices.add(truth_index)
            matched_count += 1
            center_errors.append(distance)
            track_predictions[track_key].append((current_time, prediction_track_id))
            if track_key not in visible_starts:
                visible_starts[track_key] = current_time
            if track_key not in first_detection:
                first_detection[track_key] = current_time
            if prediction_track_id is None:
                identity_unavailable_matches += 1
            else:
                old_track_id = previous_track_id.get(track_key)
                if old_track_id is not None and old_track_id != prediction_track_id:
                    id_switches += 1
                previous_track_id[track_key] = prediction_track_id
            if track_key in pending_reacquisitions:
                reacquisition_delays.append(
                    current_time - pending_reacquisitions.pop(track_key)
                )
                reacquisition_successes += 1

        for truth_index, truth_ball in enumerate(visible_truths):
            track_key = str(truth_ball["ball_id"])
            visible_starts.setdefault(track_key, current_time)
            if (
                track_key in pending_reacquisitions
                and truth_index not in matched_truth_indices
            ):
                # The episode remains pending across missed visible samples.
                pass

        cue_truth = truth_frame["cue"]
        cue_state = cue_truth["status"]
        cue_result_observed = bool(
            result is not None
            and result.get("cue_observation_status", "unavailable") == "observed"
            and result.get("cue_observation_source_frame_index") == source_index
        )
        if cue_state in {"visible", "absent"}:
            cue_labeled_count += 1
            if not cue_result_observed:
                cue_unavailable_engine_count += 1
                continue
            predicted_present = bool(result is not None and result.get("cue_present"))
            expected_present = cue_state == "visible"
            if expected_present and predicted_present:
                cue_tp += 1
                assert result is not None
                truth_tip = _point(cue_truth["tip"], "cue tip")
                truth_butt = _point(cue_truth["butt"], "cue butt")
                predicted_center = result.get("cue_center")
                if predicted_center is not None:
                    cue_center_errors.append(
                        math.dist(
                            (
                                (truth_tip[0] + truth_butt[0]) / 2,
                                (truth_tip[1] + truth_butt[1]) / 2,
                            ),
                            predicted_center,
                        )
                    )
                if "cue_tip" in result and "cue_butt" in result:
                    tip_error = math.dist(truth_tip, result["cue_tip"])
                    butt_error = math.dist(truth_butt, result["cue_butt"])
                    cue_tip_errors.append(tip_error)
                    cue_butt_errors.append(butt_error)
                    expected_angle = math.degrees(
                        math.atan2(
                            truth_tip[1] - truth_butt[1],
                            truth_tip[0] - truth_butt[0],
                        )
                    )
                    angle_difference = (
                        float(result["cue_angle_degrees"]) - expected_angle + 180.0
                    ) % 360.0 - 180.0
                    cue_angle_errors.append(abs(angle_difference))
            elif expected_present:
                cue_fn += 1
            elif predicted_present:
                cue_fp += 1
            else:
                cue_tn += 1

    visible_tracks = set(visible_starts)
    detected_tracks = set(first_detection)
    delays = [first_detection[key] - visible_starts[key] for key in detected_tracks]
    identity_observations = sum(
        1
        for samples in track_predictions.values()
        for _, track_id in samples
        if track_id is not None
    )
    ball_precision = matched_count / prediction_count if prediction_count else None
    ball_recall = matched_count / visible_count if visible_count else None
    cue_precision = cue_tp / (cue_tp + cue_fp) if cue_tp + cue_fp else None
    cue_recall = cue_tp / (cue_tp + cue_fn) if cue_tp + cue_fn else None

    return {
        "status": "available",
        "schema_version": 1,
        "timing_basis": "recorded-video presentation time; not camera exposure time",
        "matching": {
            "algorithm": "global minimum-cost assignment with maximum valid cardinality",
            "radius_fraction": matching_radius_fraction,
        },
        "availability": {
            "annotated_frames": len(annotation_frames),
            "engine_frames": len(engine_by_index),
            "missing_engine_frames": missing_engine_frames,
            "missing_frames_are_unavailable": True,
            "stale_observed_ball_results": stale_observation_count,
        },
        "ball_localization": {
            "visible_ground_truth_balls": visible_count,
            "fresh_predictions": prediction_count,
            "matched_balls": matched_count,
            "false_positives": prediction_count - matched_count,
            "false_negatives": visible_count - matched_count,
            "precision": ball_precision,
            "recall": ball_recall,
            "f1": (
                2 * ball_precision * ball_recall / (ball_precision + ball_recall)
                if ball_precision is not None
                and ball_recall is not None
                and ball_precision + ball_recall > 0
                else None
            ),
            **_summary(center_errors, "center_error_px"),
        },
        "tracking": {
            "visible_tracks": len(visible_tracks),
            "tracks_ever_detected": len(detected_tracks),
            "tracks_never_detected": len(visible_tracks - detected_tracks),
            "identity_switches": id_switches,
            "source_session_boundary_id_resets": source_session_boundary_id_resets,
            "matches_without_track_identity": identity_unavailable_matches,
            "identified_matches": identity_observations,
            "first_detection_delay_seconds": _summary(delays, "delay_seconds"),
            "occlusion_reacquisition_attempts": reacquisition_attempts,
            "occlusion_reacquisition_successes": reacquisition_successes,
            "occlusion_reacquisition_unresolved": reacquisition_attempts
            - reacquisition_successes,
            "occlusion_reacquisition_delay_seconds": _summary(
                reacquisition_delays, "delay_seconds"
            ),
        },
        "observation_freshness": {
            "available_samples": len(observation_ages),
            "expected_samples": freshness_expected_samples,
            "missing_age_samples": missing_age_samples,
            "freshness_budget_seconds": freshness_budget_seconds,
            "freshness_budget_misses": freshness_misses,
            "freshness_budget_miss_rate": (
                freshness_misses / freshness_expected_samples
                if freshness_expected_samples
                else None
            ),
            **_summary(observation_ages, "age_seconds"),
        },
        "cue": {
            "labeled_presence_frames": cue_labeled_count,
            "scored_presence_frames": cue_tp + cue_fp + cue_fn + cue_tn,
            "unavailable_engine_frames": cue_unavailable_engine_count,
            "true_positive": cue_tp,
            "false_positive": cue_fp,
            "false_negative": cue_fn,
            "true_negative": cue_tn,
            "presence_precision": cue_precision,
            "presence_recall": cue_recall,
            "center_error_px": _summary(cue_center_errors, "error_px"),
            "tip_error_px": _summary(cue_tip_errors, "error_px"),
            "butt_error_px": _summary(cue_butt_errors, "error_px"),
            "wrapped_angle_error_degrees": _summary(cue_angle_errors, "error_degrees"),
            "occluded_frames_excluded": sum(
                frame["cue"]["status"] == "occluded" for frame in annotation_frames
            ),
        },
        "calibration": _boundary_metrics(
            annotation_frames,
            engine_by_index,
            "expected_calibration_revision",
            "calibration_revision",
            "calibration revision",
        ),
        "source_session": _boundary_metrics(
            annotation_frames,
            engine_by_index,
            "expected_source_session_epoch",
            "source_session_epoch",
            "source-session epoch",
            compare_frame_values=False,
        ),
    }


def unavailable_temporal_metrics(
    reason: str = "temporal annotations not supplied",
) -> dict[str, Any]:
    """Return the explicit non-claim used when a benchmark has no labels."""
    return {
        "status": "unavailable",
        "reason": reason,
        "timing_basis": "recorded-video presentation time; not camera exposure time",
        "metrics": None,
    }
