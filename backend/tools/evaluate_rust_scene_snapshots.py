"""Score Rust scene-snapshot replay output against versioned temporal labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2

from backend.tools.temporal_evaluation import (
    TemporalAnnotationError,
    evaluate_temporal_localization,
    load_rust_scene_snapshots,
    load_temporal_annotations,
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video_geometry(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"video has invalid frame geometry: {path}")
        return width, height
    finally:
        capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Rust scene-snapshot JSONL against versioned temporal labels. "
            "Replay timestamps are taken from annotations, never Rust monotonic ticks."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--scene-snapshots", type=Path, required=True)
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=0,
        help="add this offset after converting Rust's one-based frame ID to zero-based",
    )
    parser.add_argument("--matching-radius-fraction", type=float, default=0.5)
    parser.add_argument("--freshness-budget-seconds", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    return parser


def evaluate_files(
    *,
    video_path: Path,
    annotation_path: Path,
    scene_snapshot_path: Path,
    frame_index_offset: int = 0,
    matching_radius_fraction: float = 0.5,
    freshness_budget_seconds: float = 0.1,
) -> dict[str, Any]:
    """Load exact input identity, normalize Rust scenes, and build a report."""
    if not video_path.is_file():
        raise ValueError(f"video does not exist: {video_path}")
    if not annotation_path.is_file():
        raise ValueError(f"annotation file does not exist: {annotation_path}")
    if not scene_snapshot_path.is_file():
        raise ValueError(f"Rust scene file does not exist: {scene_snapshot_path}")
    video_hash = _hash_file(video_path)
    video_geometry = _read_video_geometry(video_path)
    annotations = load_temporal_annotations(
        annotation_path,
        expected_video_sha256=video_hash,
        expected_geometry=video_geometry,
    )
    rust_frames = load_rust_scene_snapshots(
        scene_snapshot_path,
        annotations,
        frame_index_offset=frame_index_offset,
    )
    metrics = evaluate_temporal_localization(
        annotations,
        rust_frames,
        matching_radius_fraction=matching_radius_fraction,
        freshness_budget_seconds=freshness_budget_seconds,
    )
    return {
        "schema_version": 1,
        "benchmark": "rust-scene-temporal-localization",
        "metadata": {
            "video_path": str(video_path.resolve()),
            "video_sha256": video_hash,
            "video_geometry": {
                "width": video_geometry[0],
                "height": video_geometry[1],
            },
            "annotations_path": str(annotation_path.resolve()),
            "annotations_sha256": _hash_file(annotation_path),
            "scene_snapshots_path": str(scene_snapshot_path.resolve()),
            "scene_snapshots_sha256": _hash_file(scene_snapshot_path),
            "frame_index_offset": frame_index_offset,
            "scored_engine_frames": len(rust_frames),
            "timestamp_basis": "annotation video presentation time",
            "rust_monotonic_timestamps_used_for_latency": False,
            "wall_clock_output_latency_measured": False,
        },
        "temporal_localization": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_files(
            video_path=args.video,
            annotation_path=args.annotations,
            scene_snapshot_path=args.scene_snapshots,
            frame_index_offset=args.frame_index_offset,
            matching_radius_fraction=args.matching_radius_fraction,
            freshness_budget_seconds=args.freshness_budget_seconds,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Rust temporal evaluation failed: {error}") from error
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
