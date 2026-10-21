"""Import explicitly reviewed CVAT video tracks into temporal evaluation labels.

Only CVAT keyframes at frames listed in ``--reviewed-frames`` become ground
truth. Track interpolation is useful for annotation navigation, but is never
promoted to an independently reviewed label by this importer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from bisect import bisect_right
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.tools.temporal_evaluation import (
    MAX_SAFE_JSON_INTEGER,
    TemporalAnnotationError,
    validate_annotation_document,
)

_BOX_COORDINATES = ("xtl", "ytl", "xbr", "ybr")
_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: str | None, label: str) -> float:
    if value is None:
        raise TemporalAnnotationError(f"{label} is required")
    try:
        number = float(value)
    except ValueError as error:
        raise TemporalAnnotationError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise TemporalAnnotationError(f"{label} must be finite")
    return number


def _non_negative_int(value: str | None, label: str) -> int:
    if value is None or not value.isdecimal():
        raise TemporalAnnotationError(f"{label} must be a non-negative integer")
    return int(value)


def _execute_ffprobe(
    command: Sequence[str], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run ffprobe in a small seam that deterministic tests can replace."""
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _flag(element: ET.Element, name: str, *, default: str | None = None) -> bool:
    value = element.get(name, default)
    if value not in {"0", "1"}:
        raise TemporalAnnotationError(f"CVAT {element.tag} {name} must be 0 or 1")
    return value == "1"


def _parse_point_pair(points: str | None) -> tuple[dict[str, float], dict[str, float]]:
    if points is None:
        raise TemporalAnnotationError("CVAT cue polyline is missing points")
    coordinates = points.split(";")
    if len(coordinates) != 2:
        raise TemporalAnnotationError(
            "CVAT cue polyline must have exactly two points ordered tip then butt"
        )
    parsed: list[dict[str, float]] = []
    for point_index, coordinate in enumerate(coordinates):
        components = coordinate.split(",")
        if len(components) != 2:
            raise TemporalAnnotationError(
                f"CVAT cue point {point_index} must contain x,y"
            )
        parsed.append(
            {
                "x": _finite_float(components[0], "CVAT cue x"),
                "y": _finite_float(components[1], "CVAT cue y"),
            }
        )
    return parsed[0], parsed[1]


def _read_video_timeline(
    video_path: Path,
    *,
    video_stream_index: int,
    ffprobe: str,
    timeout_seconds: float,
) -> tuple[int, int, list[float], float]:
    """Read frame presentation timestamps and source geometry from ffprobe."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        f"v:{video_stream_index}",
        "-show_frames",
        "-show_streams",
        "-show_entries",
        "stream=codec_type,width,height:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = _execute_ffprobe(command, timeout_seconds)
    except FileNotFoundError as error:
        raise TemporalAnnotationError(
            f"ffprobe executable was not found: {ffprobe}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise TemporalAnnotationError(
            f"ffprobe exceeded the {timeout_seconds:g}-second timeout"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "ffprobe returned a non-zero exit status"
        raise TemporalAnnotationError(f"could not inspect video: {detail}") from error

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise TemporalAnnotationError("ffprobe returned invalid JSON") from error
    if not isinstance(probe, Mapping):
        raise TemporalAnnotationError("ffprobe output root must be an object")
    streams = probe.get("streams")
    frames = probe.get("frames")
    if not isinstance(streams, list) or len(streams) != 1:
        raise TemporalAnnotationError(
            "video must contain exactly one selected video stream"
        )
    if not isinstance(frames, list) or not frames:
        raise TemporalAnnotationError("video contains no decodable video frames")
    stream = streams[0]
    if not isinstance(stream, Mapping) or stream.get("codec_type") != "video":
        raise TemporalAnnotationError("ffprobe did not identify a video stream")
    width = stream.get("width")
    height = stream.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise TemporalAnnotationError("video stream has invalid width or height")

    presentation_times: list[float] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise TemporalAnnotationError(
                f"ffprobe frame {frame_index} must be an object"
            )
        raw_time = frame.get("best_effort_timestamp_time")
        if raw_time is None:
            raise TemporalAnnotationError(
                f"ffprobe frame {frame_index} has no presentation timestamp"
            )
        try:
            timestamp = float(raw_time)
        except (TypeError, ValueError) as error:
            raise TemporalAnnotationError(
                f"ffprobe frame {frame_index} has an invalid presentation timestamp"
            ) from error
        if not math.isfinite(timestamp):
            raise TemporalAnnotationError(
                f"ffprobe frame {frame_index} presentation timestamp is not finite"
            )
        if presentation_times and timestamp <= presentation_times[-1]:
            raise TemporalAnnotationError(
                "video presentation timestamps must be strictly increasing"
            )
        presentation_times.append(timestamp)

    origin = presentation_times[0]
    return (
        width,
        height,
        [timestamp - origin for timestamp in presentation_times],
        origin,
    )


def _parse_reviewed_frames(path: Path, *, frame_count: int) -> tuple[list[int], bytes]:
    try:
        content = path.read_bytes()
        raw = json.loads(content)
    except OSError as error:
        raise TemporalAnnotationError(
            f"could not read reviewed-frame list: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise TemporalAnnotationError(
            "reviewed-frame list must be a JSON array"
        ) from error
    if not isinstance(raw, list) or not raw:
        raise TemporalAnnotationError(
            "reviewed-frame list must be a non-empty JSON array"
        )
    reviewed: list[int] = []
    previous = -1
    for position, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TemporalAnnotationError(
                f"reviewed frame at position {position} must be an integer"
            )
        if value <= previous:
            raise TemporalAnnotationError(
                "reviewed frame indices must be unique and strictly increasing"
            )
        if value < 0 or value >= frame_count:
            raise TemporalAnnotationError(
                f"reviewed frame index {value} is outside the video"
            )
        reviewed.append(value)
        previous = value
    return reviewed, content


def _parse_epoch_events(path: Path | None) -> tuple[list[dict[str, Any]], bytes | None]:
    if path is None:
        return [], None
    try:
        content = path.read_bytes()
        raw = json.loads(content)
    except OSError as error:
        raise TemporalAnnotationError(
            f"could not read epoch-event file: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise TemporalAnnotationError(
            "epoch-event file must be a JSON array"
        ) from error
    if not isinstance(raw, list):
        raise TemporalAnnotationError("epoch-event file must be a JSON array")

    events: list[dict[str, Any]] = []
    previous = -1
    for position, event in enumerate(raw):
        if not isinstance(event, Mapping):
            raise TemporalAnnotationError(f"epoch event {position} must be an object")
        expected_keys = {
            "source_frame_index",
            "expected_calibration_revision",
            "expected_source_session_epoch",
        }
        if set(event) != expected_keys:
            raise TemporalAnnotationError(
                f"epoch event {position} must contain exactly {sorted(expected_keys)}"
            )
        frame_index = event["source_frame_index"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TemporalAnnotationError(
                f"epoch event {position} source_frame_index must be an integer"
            )
        if frame_index < 0:
            raise TemporalAnnotationError(
                f"epoch event {position} source_frame_index must be non-negative"
            )
        if frame_index <= previous:
            raise TemporalAnnotationError(
                "epoch event frame indices must be unique and strictly increasing"
            )
        revision = event["expected_calibration_revision"]
        if revision is not None and (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or revision > MAX_SAFE_JSON_INTEGER
        ):
            raise TemporalAnnotationError(
                f"epoch event {position} calibration revision must be a safe integer or null"
            )
        session_epoch = event["expected_source_session_epoch"]
        if session_epoch is not None and (
            not isinstance(session_epoch, str) or not session_epoch
        ):
            raise TemporalAnnotationError(
                f"epoch event {position} source session epoch must be a non-empty string or null"
            )
        events.append(
            {
                "source_frame_index": frame_index,
                "expected_calibration_revision": revision,
                "expected_source_session_epoch": session_epoch,
            }
        )
        previous = frame_index
    return events, content


def _task_dimensions(root: ET.Element) -> tuple[int, int]:
    task = root.find("./meta/task")
    original_size = task.find("original_size") if task is not None else None
    if original_size is None:
        raise TemporalAnnotationError("CVAT video task is missing original_size")
    width = _non_negative_int(original_size.findtext("width"), "CVAT original width")
    height = _non_negative_int(original_size.findtext("height"), "CVAT original height")
    if width == 0 or height == 0:
        raise TemporalAnnotationError("CVAT original_size dimensions must be positive")
    return width, height


def _parse_cvat_tracks(
    annotation_path: Path,
    *,
    video_geometry: tuple[int, int],
    frame_count: int,
    ball_label: str,
    cue_label: str,
    allow_empty_ball_sequence: bool,
) -> dict[str, list[dict[str, Any]]]:
    try:
        root = ET.parse(annotation_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise TemporalAnnotationError(
            f"could not parse CVAT XML: {annotation_path}"
        ) from error
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise TemporalAnnotationError("CVAT XML must use the 1.1 annotations format")
    task = root.find("./meta/task")
    if task is None or task.findtext("mode") != "interpolation":
        raise TemporalAnnotationError(
            "CVAT XML must be exported from a video interpolation task"
        )
    if _task_dimensions(root) != video_geometry:
        raise TemporalAnnotationError(
            "CVAT task original_size does not match source video"
        )
    task_size = _non_negative_int(task.findtext("size"), "CVAT task size")
    if task_size != frame_count:
        raise TemporalAnnotationError(
            "CVAT task size does not match the selected source video stream "
            f"({task_size} != {frame_count}); create annotations for the full stream"
        )
    labels_element = task.find("labels")
    declared_labels = (
        {
            label.findtext("name")
            for label in labels_element.findall("label")
            if label.findtext("name")
        }
        if labels_element is not None
        else set()
    )
    missing_labels = {ball_label, cue_label} - declared_labels
    if missing_labels:
        raise TemporalAnnotationError(
            "CVAT task does not declare the configured labels: "
            + ", ".join(sorted(missing_labels))
        )

    tracks: dict[str, list[dict[str, Any]]] = {"ball": [], "cue": []}
    seen_track_ids: set[str] = set()
    for track in root.findall("track"):
        track_id = track.get("id")
        label = track.get("label")
        if track_id is None or not track_id or label not in {ball_label, cue_label}:
            if label in {ball_label, cue_label}:
                raise TemporalAnnotationError("CVAT track is missing a stable track id")
            continue
        if track_id in seen_track_ids:
            raise TemporalAnnotationError(f"duplicate CVAT track id: {track_id}")
        seen_track_ids.add(track_id)
        kind = "ball" if label == ball_label else "cue"
        parsed_shapes: list[dict[str, Any]] = []
        previous_frame = -1
        expected_shape = "box" if kind == "ball" else "polyline"
        for shape in list(track):
            if shape.tag not in {"box", "polyline"}:
                raise TemporalAnnotationError(
                    f"unsupported CVAT {kind} track shape: {shape.tag}"
                )
            if shape.tag != expected_shape:
                raise TemporalAnnotationError(
                    f"CVAT {kind} tracks must use {expected_shape} shapes"
                )
            frame = _non_negative_int(shape.get("frame"), "CVAT shape frame")
            if frame >= frame_count:
                raise TemporalAnnotationError(
                    f"CVAT shape frame {frame} is outside the source video"
                )
            if frame <= previous_frame:
                raise TemporalAnnotationError(
                    f"CVAT shapes in track {track_id} must be strictly ordered by frame"
                )
            previous_frame = frame
            outside = _flag(shape, "outside")
            occluded = _flag(shape, "occluded")
            keyframe = _flag(shape, "keyframe")
            if outside and occluded:
                raise TemporalAnnotationError(
                    f"CVAT shape in track {track_id} cannot be both outside and occluded"
                )
            record: dict[str, Any] = {
                "frame": frame,
                "outside": outside,
                "occluded": occluded,
                "keyframe": keyframe,
            }
            if shape.tag == "box":
                coordinates = {
                    name: _finite_float(shape.get(name), f"CVAT box {name}")
                    for name in _BOX_COORDINATES
                }
                if not outside and (
                    coordinates["xbr"] <= coordinates["xtl"]
                    or coordinates["ybr"] <= coordinates["ytl"]
                ):
                    raise TemporalAnnotationError(
                        f"visible CVAT ball box in track {track_id} has non-positive dimensions"
                    )
                record["coordinates"] = coordinates
            elif not outside and not occluded:
                record["tip"], record["butt"] = _parse_point_pair(shape.get("points"))
            parsed_shapes.append(record)
        if not parsed_shapes:
            raise TemporalAnnotationError(f"CVAT track {track_id} has no shapes")
        tracks[kind].append({"track_id": track_id, "shapes": parsed_shapes})
    if not tracks["ball"] and not allow_empty_ball_sequence:
        raise TemporalAnnotationError(
            "CVAT contains no ball tracks; pass --allow-empty-ball-sequence only "
            "for a fully reviewed empty-table sequence"
        )
    return tracks


def _state_at_reviewed_frame(
    shapes: Sequence[Mapping[str, Any]],
    frame_index: int,
    *,
    track_id: str,
) -> Mapping[str, Any] | None:
    shape_frames = [int(shape["frame"]) for shape in shapes]
    position = bisect_right(shape_frames, frame_index) - 1
    if position < 0:
        return None
    shape = shapes[position]
    if int(shape["frame"]) == frame_index:
        if not shape["keyframe"]:
            raise TemporalAnnotationError(
                f"reviewed frame {frame_index} has a non-keyframe shape for track {track_id}"
            )
        return shape
    if shape["outside"]:
        # An explicit outside keyframe marks the track removed until a later
        # reviewed keyframe reintroduces it.
        return {"outside": True, "occluded": False}
    raise TemporalAnnotationError(
        f"reviewed frame {frame_index} relies on interpolated state for active track {track_id}; "
        "add an explicit CVAT keyframe after reviewing this frame"
    )


def build_temporal_annotations(
    *,
    video_path: Path,
    cvat_xml_path: Path,
    reviewed_frames_path: Path,
    output_path: Path,
    expected_video_sha256: str,
    frame_index_offset: int = 0,
    video_stream_index: int = 0,
    epoch_events_path: Path | None = None,
    ffprobe: str = "ffprobe",
    ffprobe_timeout_seconds: float = 120.0,
    ball_label: str = "ball",
    cue_label: str = "cue",
    allow_empty_ball_sequence: bool = False,
) -> dict[str, Any]:
    """Build validated labels from exact video PTS and explicitly reviewed keys."""
    for name, path in (
        ("video", video_path),
        ("CVAT XML", cvat_xml_path),
        ("reviewed-frame list", reviewed_frames_path),
    ):
        if not path.is_file():
            raise TemporalAnnotationError(f"{name} does not exist: {path}")
    if epoch_events_path is not None and not epoch_events_path.is_file():
        raise TemporalAnnotationError(
            f"epoch-event file does not exist: {epoch_events_path}"
        )
    if (
        isinstance(frame_index_offset, bool)
        or not isinstance(frame_index_offset, int)
        or frame_index_offset < 0
    ):
        raise TemporalAnnotationError("frame_index_offset must be non-negative")
    if (
        isinstance(video_stream_index, bool)
        or not isinstance(video_stream_index, int)
        or video_stream_index < 0
    ):
        raise TemporalAnnotationError("video_stream_index must be non-negative")
    if (
        isinstance(ffprobe_timeout_seconds, bool)
        or not isinstance(ffprobe_timeout_seconds, (int, float))
        or not math.isfinite(ffprobe_timeout_seconds)
        or ffprobe_timeout_seconds <= 0
    ):
        raise TemporalAnnotationError(
            "ffprobe timeout must be a positive finite number"
        )
    if (
        not isinstance(ball_label, str)
        or not ball_label
        or not isinstance(cue_label, str)
        or not cue_label
        or ball_label == cue_label
    ):
        raise TemporalAnnotationError(
            "ball and cue labels must be distinct non-empty names"
        )
    if not isinstance(allow_empty_ball_sequence, bool):
        raise TemporalAnnotationError("allow_empty_ball_sequence must be a boolean")
    if not isinstance(expected_video_sha256, str) or not _SHA256_PATTERN.fullmatch(
        expected_video_sha256
    ):
        raise TemporalAnnotationError(
            "expected_video_sha256 must be a 64-character SHA-256 hex digest "
            "recorded for the CVAT source video"
        )
    actual_video_sha256 = _sha256_file(video_path)
    if actual_video_sha256.lower() != expected_video_sha256.lower():
        raise TemporalAnnotationError(
            "source video SHA-256 does not match the identity recorded for CVAT"
        )
    protected_paths = {
        video_path.resolve(),
        cvat_xml_path.resolve(),
        reviewed_frames_path.resolve(),
    }
    if epoch_events_path is not None:
        protected_paths.add(epoch_events_path.resolve())
    if output_path.resolve() in protected_paths:
        raise TemporalAnnotationError("output path must not overwrite an input file")

    width, height, presentation_times, presentation_time_origin = _read_video_timeline(
        video_path,
        video_stream_index=video_stream_index,
        ffprobe=ffprobe,
        timeout_seconds=ffprobe_timeout_seconds,
    )
    reviewed_frames, reviewed_content = _parse_reviewed_frames(
        reviewed_frames_path, frame_count=len(presentation_times)
    )
    events, epoch_content = _parse_epoch_events(epoch_events_path)
    tracks = _parse_cvat_tracks(
        cvat_xml_path,
        video_geometry=(width, height),
        frame_count=len(presentation_times),
        ball_label=ball_label,
        cue_label=cue_label,
        allow_empty_ball_sequence=allow_empty_ball_sequence,
    )
    event_frames = [int(event["source_frame_index"]) for event in events]

    frames: list[dict[str, Any]] = []
    for cvat_frame in reviewed_frames:
        source_frame = cvat_frame + frame_index_offset
        epoch_position = bisect_right(event_frames, source_frame) - 1
        epoch = events[epoch_position] if epoch_position >= 0 else None
        balls: list[dict[str, Any]] = []
        for track in tracks["ball"]:
            state = _state_at_reviewed_frame(
                track["shapes"], cvat_frame, track_id=track["track_id"]
            )
            if state is None:
                continue
            if state["outside"]:
                balls.append({"ball_id": track["track_id"], "status": "removed"})
            elif state["occluded"]:
                balls.append({"ball_id": track["track_id"], "status": "occluded"})
            else:
                coordinates = state["coordinates"]
                x1 = float(coordinates["xtl"])
                y1 = float(coordinates["ytl"])
                x2 = float(coordinates["xbr"])
                y2 = float(coordinates["ybr"])
                balls.append(
                    {
                        "ball_id": track["track_id"],
                        "status": "visible",
                        "center": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
                        "radius_px": ((x2 - x1) + (y2 - y1)) / 4,
                    }
                )

        cue_state: Mapping[str, Any] | None = None
        if tracks["cue"]:
            cue_state = _state_at_reviewed_frame(
                tracks["cue"][0]["shapes"],
                cvat_frame,
                track_id=tracks["cue"][0]["track_id"],
            )
            if len(tracks["cue"]) > 1:
                raise TemporalAnnotationError(
                    "CVAT video must contain at most one cue track"
                )
        if cue_state is None or cue_state["outside"]:
            cue: dict[str, Any] = {"status": "absent"}
        elif cue_state["occluded"]:
            cue = {"status": "occluded"}
        else:
            cue = {
                "status": "visible",
                "tip": dict(cue_state["tip"]),
                "butt": dict(cue_state["butt"]),
            }
        frames.append(
            {
                "source_frame_index": source_frame,
                "presentation_time_seconds": presentation_times[cvat_frame],
                "expected_calibration_revision": (
                    epoch["expected_calibration_revision"] if epoch else None
                ),
                "expected_source_session_epoch": (
                    epoch["expected_source_session_epoch"] if epoch else None
                ),
                "balls": balls,
                "cue": cue,
            }
        )

    document: dict[str, Any] = {
        "schema_version": 1,
        "sequence": {
            "video_sha256": actual_video_sha256,
            "width": width,
            "height": height,
            "coordinate_space": "source_pixels",
        },
        "frames": frames,
        "annotation_metadata": {
            "source_annotation_format": "cvat_video_1.1",
            "source_annotation_sha256": _sha256_file(cvat_xml_path),
            "reviewed_frames_sha256": _sha256_bytes(reviewed_content),
            "epoch_events_sha256": (
                _sha256_bytes(epoch_content) if epoch_content is not None else None
            ),
            "frame_index_offset": frame_index_offset,
            "source_video_stream_index": video_stream_index,
            "presentation_time_source": "ffprobe_best_effort_timestamp_time",
            "presentation_time_origin_seconds": presentation_time_origin,
            "ball_radius_method": "mean_box_half_extent",
            "cue_polyline_point_order": "first_tip_second_butt",
            "tool_version": 1,
        },
    }
    return validate_annotation_document(
        document,
        expected_video_sha256=document["sequence"]["video_sha256"],
        expected_geometry=(width, height),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert explicitly reviewed CVAT video keyframes to versioned temporal labels. "
            "Interpolated shapes are never treated as ground truth."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--expected-video-sha256",
        required=True,
        help="SHA-256 recorded for the video uploaded to the CVAT task",
    )
    parser.add_argument("--cvat-xml", type=Path, required=True)
    parser.add_argument(
        "--reviewed-frames",
        type=Path,
        required=True,
        help="JSON array of fully reviewed, zero-based CVAT frame ordinals",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=0,
        help="add this offset to CVAT frame ordinals for full source-frame identity",
    )
    parser.add_argument(
        "--video-stream-index",
        type=int,
        default=0,
        help="zero-based index of the source video stream selected by CVAT",
    )
    parser.add_argument(
        "--epoch-events",
        type=Path,
        help="optional JSON calibration/session boundary event list",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffprobe-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--ball-label", default="ball")
    parser.add_argument("--cue-label", default="cue")
    parser.add_argument(
        "--allow-empty-ball-sequence",
        action="store_true",
        help="explicitly permit a fully reviewed sequence with no ball tracks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = build_temporal_annotations(
            video_path=args.video,
            cvat_xml_path=args.cvat_xml,
            reviewed_frames_path=args.reviewed_frames,
            output_path=args.output,
            expected_video_sha256=args.expected_video_sha256,
            frame_index_offset=args.frame_index_offset,
            video_stream_index=args.video_stream_index,
            epoch_events_path=args.epoch_events,
            ffprobe=args.ffprobe,
            ffprobe_timeout_seconds=args.ffprobe_timeout_seconds,
            ball_label=args.ball_label,
            cue_label=args.cue_label,
            allow_empty_ball_sequence=args.allow_empty_ball_sequence,
        )
        serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=args.output.parent,
                prefix=f".{args.output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.replace(args.output)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    except (OSError, TemporalAnnotationError, ValueError) as error:
        raise SystemExit(f"CVAT temporal annotation import failed: {error}") from error

    print(
        json.dumps(
            {
                "output": str(args.output),
                "video_sha256": document["sequence"]["video_sha256"],
                "video_geometry": {
                    "width": document["sequence"]["width"],
                    "height": document["sequence"]["height"],
                },
                "source_video_stream_index": document["annotation_metadata"][
                    "source_video_stream_index"
                ],
                "reviewed_frames": len(document["frames"]),
                "labeled_ball_instances": sum(
                    len(frame["balls"]) for frame in document["frames"]
                ),
                "timestamp_basis": "ffprobe best_effort_timestamp_time",
                "interpolated_states_used_as_ground_truth": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
