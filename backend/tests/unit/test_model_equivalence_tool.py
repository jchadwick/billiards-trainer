"""Tests for class-aware matching in the retained model-equivalence tool."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import tools.test_backend_equivalence as equivalence_tool
from tools.test_backend_equivalence import (
    _attach_detection_batches,
    _iter_frames,
    _load_rust_result,
    _manifest_evidence,
    _map_rust_results_to_inputs,
    _sha256_path,
    build_argument_parser,
    compare_detections,
)


def test_detection_matching_is_one_to_one_and_class_constrained():
    reference = [
        {"center": [100.0, 100.0], "class_id": 1, "confidence": 0.9},
        {"center": [300.0, 300.0], "class_id": 1, "confidence": 0.8},
    ]
    candidate = [
        {"center": [301.0, 300.0], "class_id": 1, "confidence": 0.81},
        {"center": [100.5, 100.0], "class_id": 1, "confidence": 0.89},
        {"center": [100.1, 100.0], "class_id": 0, "confidence": 0.7},
    ]

    result = compare_detections(
        reference,
        candidate,
        class_names={0: "cue", 1: "ball"},
        match_tolerance_px=2.0,
    )

    assert result["matched_count"] == 2
    assert [match["reference_center"] for match in result["matches"]] == [
        [100.0, 100.0],
        [300.0, 300.0],
    ]
    assert result["unmatched_reference"] == []
    assert [item["class_id"] for item in result["unmatched_candidate"]] == [0]


def test_detection_outside_match_radius_remains_unmatched_on_both_sides():
    result = compare_detections(
        [{"center": [100.0, 100.0], "class_id": 1, "confidence": 0.9}],
        [{"center": [103.0, 100.0], "class_id": 1, "confidence": 0.9}],
        class_names={1: "ball"},
        match_tolerance_px=2.0,
    )

    assert result["matched_count"] == 0
    assert result["unmatched_reference"][0]["index"] == 0
    assert result["unmatched_candidate"][0]["index"] == 0


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_detection_matching_rejects_invalid_tolerance(tolerance: float):
    with pytest.raises(ValueError, match="finite and positive"):
        compare_detections([], [], class_names={}, match_tolerance_px=tolerance)


def test_model_hashes_cover_directory_component_names_and_contents(tmp_path: Path):
    package = tmp_path / "model.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Data" / "weights.bin").write_bytes(b"weights")
    (package / "Manifest.json").write_text("{}", encoding="utf-8")

    digest, components = _sha256_path(package)

    assert len(digest) == 64
    assert [component["path"] for component in components] == [
        "Data/weights.bin",
        "Manifest.json",
    ]


def test_export_manifest_is_included_with_content_hash(tmp_path: Path):
    model_path = tmp_path / "candidate.onnx"
    model_path.write_bytes(b"model")
    manifest_path = Path(f"{model_path}.manifest.json")
    manifest_path.write_text('{"schema_version": 1}', encoding="utf-8")

    evidence = _manifest_evidence(model_path)

    assert evidence["status"] == "present"
    assert evidence["path"] == str(manifest_path)
    assert len(evidence["sha256"]) == 64
    assert evidence["contents"] == {"schema_version": 1}


def test_invalid_export_manifest_is_rejected(tmp_path: Path):
    model_path = tmp_path / "candidate.onnx"
    model_path.write_bytes(b"model")
    Path(f"{model_path}.manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        _manifest_evidence(model_path)


def test_image_directory_frames_record_stable_relative_source_names(
    tmp_path: Path,
):
    (tmp_path / "nested").mkdir()
    for name in ("z-last.png", "nested/a-first.png"):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / name), image)

    frames = list(_iter_frames(tmp_path, max_frames=10))

    assert [source_name for _, source_name, _ in frames] == [
        "nested/a-first.png",
        "z-last.png",
    ]
    assert [number for number, _, _ in frames] == [0, 1]
    assert all(frame.shape == (4, 5, 3) for _, _, frame in frames)


def test_cli_requires_explicit_reference_and_candidate_models():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["input.mp4"])

    arguments = parser.parse_args(
        [
            "input.mp4",
            "--reference-model",
            "source.pt",
            "--candidate-model",
            "candidate.onnx",
        ]
    )
    assert arguments.reference_model == Path("source.pt")
    assert arguments.candidate_model == Path("candidate.onnx")


def _rust_result_fixture(image_path: Path, checkpoint_path: Path) -> dict:
    """Build a small valid Rust result using real encoded-image/checkpoint hashes."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    return {
        "schema_version": 1,
        "workload": "recorded_static_images_unpaced",
        "parity_scope": "test fixture",
        "model": {
            "onnx_path": "model.onnx",
            "onnx_sha256": hashlib.sha256(b"onnx model").hexdigest(),
            "source_checkpoint_path": str(checkpoint_path),
            "source_checkpoint_sha256": _sha256_path(checkpoint_path)[0],
            "manifest_path": "model.onnx.manifest.json",
            "class_names": ["cue", "ball"],
        },
        "runtime": {
            "requested_execution_provider": "CPUExecutionProvider",
            "observed_execution_providers": ["CPUExecutionProvider"],
            "binding": "ort",
        },
        "settings": {
            "image_size": 1280,
            "confidence": 0.4,
            "iou": 0.45,
            "max_detections": 300,
        },
        "results": [
            {
                "path": str(image_path),
                "encoded_image_sha256": _sha256_path(image_path)[0],
                "source_width": int(image.shape[1]),
                "source_height": int(image.shape[0]),
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "ball",
                        "confidence": 0.9,
                        "bbox_xyxy_px": [1.0, 2.0, 3.0, 4.0],
                        "center_xy_px": [2.0, 3.0],
                    }
                ],
            }
        ],
    }


def _configure_paced_result(
    report: dict, input_paths: list[Path], processed_indices: list[int]
) -> dict:
    """Describe a paced result whose source-index gaps are real dropped inputs."""
    import copy

    template = report["results"][0]
    results = []
    for source_index in processed_indices:
        path = input_paths[source_index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        entry = copy.deepcopy(template)
        entry.update(
            {
                "path": str(path),
                "source_frame_index": source_index,
                "encoded_image_sha256": _sha256_path(path)[0],
                "source_width": int(image.shape[1]),
                "source_height": int(image.shape[0]),
                "source_schedule_lateness_ms": 1.25,
                "source_to_inference_start_ms": 2.5,
                "source_to_inference_complete_ms": 5.0,
            }
        )
        results.append(entry)

    report["workload"] = equivalence_tool.RUST_PACED_RESULT_WORKLOAD
    report["settings"]["replay_rate_fps"] = 30.0
    report["results"] = results
    report["timings"] = {
        "paced_replay": {
            "requested_source_fps": 30.0,
            "scheduled_frame_count": len(input_paths),
            "decoded_frame_count": len(input_paths),
            "processed_frame_count": len(processed_indices),
            "replaced_pending_frame_count": len(input_paths) - len(processed_indices),
        }
    }
    return report


def _detection_batch_fixture(
    report: dict,
    *,
    batch_sequence: int = 1,
    source_index: int | None = None,
) -> dict:
    """Build one valid frame-scoped record paired with a Rust audit report."""
    image_result = report["results"][0]
    engine_id = "engine-test-01"
    session_id = "session-test-01"
    source_index = (
        image_result.get("source_frame_index", batch_sequence - 1)
        if source_index is None
        else source_index
    )
    timestamp_base = batch_sequence * 10
    is_paced = report.get("workload") == equivalence_tool.RUST_PACED_RESULT_WORKLOAD
    return {
        "schema_version": 1,
        "engine_instance_id": engine_id,
        "batch_sequence": batch_sequence,
        "source": {
            "session_id": session_id,
            "frame_id": f"frame-{source_index + 1:06}",
            "timestamp": (
                {
                    "ticks": str(timestamp_base - 1),
                    "unit": "nanoseconds",
                    "clock": "monotonic",
                    "domain_id": engine_id,
                    "stage": "frame_arrival",
                }
                if is_paced
                else None
            ),
            "width_px": image_result["source_width"],
            "height_px": image_result["source_height"],
        },
        "image_space": "raw",
        "model": {
            "artifact_sha256": report["model"]["onnx_sha256"],
            "source_checkpoint_sha256": report["model"]["source_checkpoint_sha256"],
            "classes": [
                {"class_id": 0, "class_name": "cue"},
                {"class_id": 1, "class_name": "ball"},
            ],
        },
        "settings": {
            "input_width_px": 1280,
            "input_height_px": 1280,
            "confidence_threshold": 0.4,
            "iou_threshold": 0.45,
            "max_detections": 300,
        },
        "inference_started_at": {
            "ticks": str(timestamp_base),
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": engine_id,
            "stage": "inference_started",
        },
        "inference_completed_at": {
            "ticks": str(timestamp_base + 1),
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": engine_id,
            "stage": "inference_complete",
        },
        "published_at": {
            "ticks": str(timestamp_base + 2),
            "unit": "nanoseconds",
            "clock": "monotonic",
            "domain_id": engine_id,
            "stage": "batch_publication",
        },
        "outcome": "success",
        "reason": None,
        "detections": [
            {
                "detection_index": 0,
                "class_id": 1,
                "class_name": "ball",
                "confidence": 0.9,
                "bbox_image_px": {
                    "left": 1.0,
                    "top": 2.0,
                    "right": 3.0,
                    "bottom": 4.0,
                },
                "center_image_px": {"x": 2.0, "y": 3.0},
            }
        ],
    }


def _add_batch_output_metadata(
    report: dict, transport: str = "jsonl", *, paced: bool = False
) -> None:
    if transport == "tcp":
        destination = "tcp://127.0.0.1:9000"
        output_format = "length_framed_json"
        mode = (
            "synchronous_loopback_delivery_during_paced_latest_frame_replay"
            if paced
            else "synchronous_loopback_delivery_during_unpaced_static_replay"
        )
    else:
        destination = "detection-batches.jsonl"
        output_format = "newline_delimited_json"
        mode = (
            "synchronous_export_during_paced_latest_frame_replay"
            if paced
            else "synchronous_export_during_unpaced_static_replay"
        )
    report["detection_batch_output"] = {
        "destination": destination,
        "format": output_format,
        "mode": mode,
        "engine_instance_id": "engine-test-01",
        "source_session_id": "session-test-01",
        "batch_count": len(report["results"]),
        "source_time": (
            "synthetic_monotonic_frame_arrival"
            if paced
            else "unavailable_for_still_image_replay"
        ),
    }


def test_saved_rust_result_compares_against_python_reference_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    rust_result_path = tmp_path / "rust-result.json"
    rust_result = _rust_result_fixture(image_path, checkpoint_path)
    rust_result_path.write_text(json.dumps(rust_result), encoding="utf-8")

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    monkeypatch.setattr(
        equivalence_tool,
        "_model_prediction",
        lambda *_args, **_kwargs: [
            {"center": [2.0, 3.0], "class_id": 1, "confidence": 0.9}
        ],
    )
    parser = build_argument_parser()
    arguments = parser.parse_args(
        [
            str(image_path),
            "--reference-model",
            str(checkpoint_path),
            "--candidate-result",
            str(rust_result_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    json_path, _, comparison = equivalence_tool.run_equivalence_test(arguments)

    candidate = comparison["models"]["candidate"]
    assert comparison["summary"]["matched_detections"] == 1
    assert comparison["comparison"]["candidate_result_frames_omitted"] == 0
    assert candidate["provenance"]["kind"] == "saved_rust_inference_result"
    assert candidate["provenance"]["onnx_sha256"] == rust_result["model"]["onnx_sha256"]
    assert (
        candidate["provenance"]["source_checkpoint_sha256"]
        == _sha256_path(checkpoint_path)[0]
    )
    assert (
        candidate["provenance"]["requested_execution_provider"]
        == "CPUExecutionProvider"
    )
    assert candidate["provenance"]["observed_execution_providers"] == [
        "CPUExecutionProvider"
    ]
    assert candidate["provenance"]["effective_inference_settings"] == {
        "image_size": 1280,
        "confidence": 0.4,
        "iou": 0.45,
        "max_detections": 300,
    }
    assert comparison["comparison"]["max_detections"] == 300
    assert (
        candidate["provenance"]["candidate_report_sha256"]
        == _sha256_path(rust_result_path)[0]
    )
    assert "does not prove decoded pixel equality" in " ".join(
        comparison["limitations"]
    )
    assert json_path.is_file()


def test_detection_batch_jsonl_is_schema_checked_and_joined_to_report_metadata(
    tmp_path: Path,
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    _add_batch_output_metadata(report)
    batch_path = tmp_path / "detection-batches.jsonl"
    batch_path.write_text(
        json.dumps(_detection_batch_fixture(report)) + "\n", encoding="utf-8"
    )

    joined = _attach_detection_batches(report, batch_path)

    assert joined["results"][0]["_detection_batch_frame_id"] == "frame-000001"
    assert joined["results"][0]["detections"][0]["center_xy_px"] == [2.0, 3.0]
    assert joined["_candidate_batch_jsonl_path"] == str(batch_path)
    assert len(joined["_candidate_batch_jsonl_sha256"]) == 64


def test_paced_result_maps_only_processed_source_frames_and_joins_batches(
    tmp_path: Path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    input_paths = []
    for index in range(3):
        path = image_dir / f"frame-{index}.png"
        assert cv2.imwrite(str(path), np.full((8, 10, 3), index * 50, dtype=np.uint8))
        input_paths.append(path)
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _configure_paced_result(
        _rust_result_fixture(input_paths[0], checkpoint_path), input_paths, [0, 2]
    )
    result_path = tmp_path / "paced-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = _load_rust_result(result_path)
    mapped = _map_rust_results_to_inputs(loaded, image_dir, result_path)

    assert [item["source_index"] for item in mapped] == [0, 2]
    _add_batch_output_metadata(report, paced=True)
    batches = [
        _detection_batch_fixture(report, batch_sequence=1, source_index=0),
        _detection_batch_fixture(report, batch_sequence=2, source_index=2),
    ]
    batch_path = tmp_path / "paced-batches.jsonl"
    batch_path.write_text(
        "".join(json.dumps(batch) + "\n" for batch in batches), encoding="utf-8"
    )

    joined = _attach_detection_batches(report, batch_path)

    assert [item["_detection_batch_frame_id"] for item in joined["results"]] == [
        "frame-000001",
        "frame-000003",
    ]


def test_paced_result_rejects_unaccounted_source_frame_gaps(tmp_path: Path):
    image_path = _write_test_image(tmp_path / "frame.png")
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _configure_paced_result(
        _rust_result_fixture(image_path, checkpoint_path), [image_path], [0]
    )
    report["timings"]["paced_replay"]["replaced_pending_frame_count"] = 1
    result_path = tmp_path / "paced-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="counts are inconsistent"):
        _load_rust_result(result_path)


def test_received_tcp_batches_use_the_existing_equivalence_scorer(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    _add_batch_output_metadata(report, transport="tcp")
    batch_path = tmp_path / "received-detection-batches.jsonl"
    batch_path.write_text(
        json.dumps(_detection_batch_fixture(report)) + "\n", encoding="utf-8"
    )

    joined = _attach_detection_batches(report, batch_path)

    assert joined["results"][0]["_detection_batch_frame_id"] == "frame-000001"
    assert joined["detection_batch_output"]["format"] == "length_framed_json"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: batch.update({"unexpected": True}),
            "violates v1 schema",
        ),
        (
            lambda batch: batch.update({"batch_sequence": 2}),
            "sequence is missing, duplicated, or out of order",
        ),
        (
            lambda batch: batch["detections"][0].update({"confidence": 0.8}),
            "differs from aggregate detection output",
        ),
    ],
)
def test_detection_batch_jsonl_rejects_invalid_or_incomplete_records(
    tmp_path: Path, mutate, message: str
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    _add_batch_output_metadata(report)
    batch = _detection_batch_fixture(report)
    mutate(batch)
    batch_path = tmp_path / "detection-batches.jsonl"
    batch_path.write_text(json.dumps(batch) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _attach_detection_batches(report, batch_path)


def test_equivalence_runner_scores_rust_detection_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    _add_batch_output_metadata(report)
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")
    batch_path = tmp_path / "detection-batches.jsonl"
    batch_path.write_text(
        json.dumps(_detection_batch_fixture(report)) + "\n", encoding="utf-8"
    )

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    monkeypatch.setattr(
        equivalence_tool,
        "_model_prediction",
        lambda *_args, **_kwargs: [
            {"center": [2.0, 3.0], "class_id": 1, "confidence": 0.9}
        ],
    )
    arguments = build_argument_parser().parse_args(
        [
            str(image_path),
            "--reference-model",
            str(checkpoint_path),
            "--candidate-result",
            str(result_path),
            "--candidate-batches-jsonl",
            str(batch_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    json_path, markdown_path, comparison = equivalence_tool.run_equivalence_test(
        arguments
    )

    candidate = comparison["models"]["candidate"]
    assert (
        comparison["comparison"]["candidate_kind"] == "saved_rust_detection_batch_jsonl"
    )
    assert comparison["summary"]["matched_detections"] == 1
    assert candidate["provenance"]["kind"] == "saved_rust_detection_batch_jsonl"
    assert candidate["provenance"]["candidate_batch_count"] == 1
    assert len(candidate["provenance"]["candidate_batch_jsonl_sha256"]) == 64
    assert json_path.is_file()
    assert "Detection batches joined: 1" in markdown_path.read_text()


def test_equivalence_runner_reports_latest_frame_drops_separately_from_comparison_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    input_paths = []
    for index in range(3):
        path = image_dir / f"frame-{index}.png"
        assert cv2.imwrite(str(path), np.full((8, 10, 3), index * 50, dtype=np.uint8))
        input_paths.append(path)
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _configure_paced_result(
        _rust_result_fixture(input_paths[0], checkpoint_path), input_paths, [0, 2]
    )
    _add_batch_output_metadata(report, paced=True)
    result_path = tmp_path / "paced-rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")
    batch_path = tmp_path / "paced-detection-batches.jsonl"
    batch_path.write_text(
        "".join(
            json.dumps(batch) + "\n"
            for batch in (
                _detection_batch_fixture(report, batch_sequence=1, source_index=0),
                _detection_batch_fixture(report, batch_sequence=2, source_index=2),
            )
        ),
        encoding="utf-8",
    )

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    monkeypatch.setattr(
        equivalence_tool,
        "_model_prediction",
        lambda *_args, **_kwargs: [
            {"center": [2.0, 3.0], "class_id": 1, "confidence": 0.9}
        ],
    )
    arguments = build_argument_parser().parse_args(
        [
            str(image_dir),
            "--reference-model",
            str(checkpoint_path),
            "--candidate-result",
            str(result_path),
            "--candidate-batches-jsonl",
            str(batch_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    _, markdown_path, comparison = equivalence_tool.run_equivalence_test(arguments)

    assert comparison["summary"]["frames_processed"] == 2
    assert comparison["summary"]["matched_detections"] == 2
    assert comparison["comparison"]["candidate_result_frames_omitted"] == 0
    assert comparison["comparison"]["candidate_source_frames_skipped_by_pacing"] == 1
    assert (
        "Candidate source frames skipped by latest-frame replay: 1"
        in markdown_path.read_text()
    )


def test_saved_rust_result_reports_frames_omitted_by_comparison_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_path = _write_test_image(image_dir / "first.png")
    second_path = image_dir / "second.png"
    assert cv2.imwrite(str(second_path), np.full((8, 10, 3), 127, dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _rust_result_fixture(first_path, checkpoint_path)
    second_entry = dict(report["results"][0])
    second_entry["path"] = str(second_path)
    second_entry["encoded_image_sha256"] = _sha256_path(second_path)[0]
    report["results"].append(second_entry)
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    monkeypatch.setattr(equivalence_tool, "_model_prediction", lambda *_a, **_k: [])
    arguments = build_argument_parser().parse_args(
        [
            str(image_dir),
            "--reference-model",
            str(checkpoint_path),
            "--candidate-result",
            str(result_path),
            "--frames",
            "1",
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    _, markdown_path, comparison = equivalence_tool.run_equivalence_test(arguments)

    assert comparison["summary"]["frames_processed"] == 1
    assert comparison["comparison"]["candidate_result_frames_available"] == 2
    assert comparison["comparison"]["candidate_result_frames_compared"] == 1
    assert comparison["comparison"]["candidate_result_frames_omitted"] == 1
    assert (
        "Candidate result frames omitted by `--frames`: 1" in markdown_path.read_text()
    )


def test_saved_rust_result_rejects_source_encoded_hash_mismatch(tmp_path: Path):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    result_path = tmp_path / "rust-result.json"
    report = _rust_result_fixture(image_path, checkpoint_path)
    report["results"][0]["encoded_image_sha256"] = hashlib.sha256(
        b"different encoded input"
    ).hexdigest()
    result_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = _load_rust_result(result_path)
    with pytest.raises(ValueError, match="No input image matches Rust source frame"):
        _map_rust_results_to_inputs(loaded, image_path, result_path)


def test_saved_rust_result_rejects_reference_checkpoint_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_path = _write_test_image(tmp_path / "frame.png")
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained Python checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    report["model"]["source_checkpoint_sha256"] = hashlib.sha256(
        b"different checkpoint"
    ).hexdigest()
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    arguments = build_argument_parser().parse_args(
        [
            str(image_path),
            "--reference-model",
            str(checkpoint_path),
            "--candidate-result",
            str(result_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )

    with pytest.raises(ValueError, match="source checkpoint provenance"):
        equivalence_tool.run_equivalence_test(arguments)


def test_saved_rust_result_rejects_ambiguous_source_mapping(tmp_path: Path):
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    pixels = np.zeros((8, 10, 3), dtype=np.uint8)
    left_path = tmp_path / "left" / "frame.png"
    right_path = tmp_path / "right" / "frame.png"
    assert cv2.imwrite(str(left_path), pixels)
    assert cv2.imwrite(str(right_path), pixels)
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(left_path, checkpoint_path)
    report["results"][0]["path"] = "old-root/frame.png"
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="maps to multiple input images"):
        _map_rust_results_to_inputs(report, tmp_path, result_path)


def test_paced_source_index_disambiguates_identical_images_at_stale_absolute_paths(
    tmp_path: Path,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    pixels = np.full((8, 10, 3), 64, dtype=np.uint8)
    first_path = image_dir / "first.png"
    second_path = image_dir / "second.png"
    assert cv2.imwrite(str(first_path), pixels)
    assert cv2.imwrite(str(second_path), pixels)
    assert _sha256_path(first_path)[0] == _sha256_path(second_path)[0]
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _configure_paced_result(
        _rust_result_fixture(first_path, checkpoint_path),
        [first_path, second_path],
        [0, 1],
    )
    for index, entry in enumerate(report["results"]):
        entry["path"] = f"/container/repo/images/{index + 1}.png"
    result_path = tmp_path / "container-rust-result.json"

    mapped = _map_rust_results_to_inputs(report, image_dir, result_path)

    assert [item["source_index"] for item in mapped] == [0, 1]
    assert [item["input_path"] for item in mapped] == [first_path, second_path]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 99, "schema version"),
        ("schema_version", 1.0, "schema version"),
        ("workload", "video_replay", "supported recorded-image workload"),
    ],
)
def test_saved_rust_result_rejects_unsupported_schema_or_workload(
    tmp_path: Path, field: str, value: object, message: str
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    report[field] = value
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_rust_result(result_path)


def test_saved_rust_result_rejects_partial_input_coverage(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_path = _write_test_image(image_dir / "first.png")
    _write_test_image(image_dir / "second.png")
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(first_path, checkpoint_path)
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover every input image"):
        _map_rust_results_to_inputs(report, image_dir, result_path)


def test_saved_rust_result_rejects_unsupported_input_image_suffix(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = _write_test_image(image_dir / "frame.png")
    assert cv2.imwrite(
        str(image_dir / "unsupported.tiff"), np.zeros((8, 10, 3), dtype=np.uint8)
    )
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)

    with pytest.raises(ValueError, match="supports only PNG/JPEG source files"):
        _map_rust_results_to_inputs(report, image_dir, tmp_path / "rust-result.json")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_class_name", "invalid model.class_names"),
        ("unknown_class_id", "class_id is outside the class map"),
        ("nonfinite_center", "must be a finite number"),
        ("out_of_bounds_center", "lies outside the reported source image"),
    ],
)
def test_saved_rust_result_rejects_malformed_schema_and_class_map(
    tmp_path: Path, mutation: str, message: str
):
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(image_path, checkpoint_path)
    if mutation == "duplicate_class_name":
        report["model"]["class_names"] = ["ball", "ball"]
    elif mutation == "unknown_class_id":
        report["results"][0]["detections"][0]["class_id"] = 7
    elif mutation == "nonfinite_center":
        report["results"][0]["detections"][0]["center_xy_px"][0] = float("nan")
    else:
        report["results"][0]["detections"][0]["center_xy_px"][0] = 10.0
    result_path = tmp_path / "rust-result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_rust_result(result_path)


def test_rust_result_cli_selection_is_mutually_exclusive_and_required():
    parser = build_argument_parser()
    base = ["frame.png", "--reference-model", "reference.pt"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)

    model_args = parser.parse_args([*base, "--candidate-model", "candidate.onnx"])
    result_args = parser.parse_args([*base, "--candidate-result", "rust-result.json"])
    assert model_args.candidate_model == Path("candidate.onnx")
    assert model_args.candidate_result is None
    assert result_args.candidate_model is None
    assert result_args.candidate_result == Path("rust-result.json")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *base,
                "--candidate-model",
                "candidate.onnx",
                "--candidate-result",
                "rust-result.json",
            ]
        )


def test_video_input_fails_for_saved_rust_image_results(tmp_path: Path):
    video_path = tmp_path / "recording.mp4"
    video_path.write_bytes(b"video placeholder")
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report = _rust_result_fixture(
        _write_test_image(tmp_path / "frame.png"), checkpoint_path
    )
    with pytest.raises(ValueError, match="video frame mapping is unsupported"):
        _map_rust_results_to_inputs(report, video_path, tmp_path / "rust-result.json")


def _write_test_image(path: Path) -> Path:
    """Write one small deterministic fixture image and return its path."""
    assert cv2.imwrite(str(path), np.zeros((8, 10, 3), dtype=np.uint8))
    return path


def _write_cvat_annotations(tmp_path: Path, images: str) -> Path:
    """Write a minimal real XML fixture with caller-controlled image records."""
    path = tmp_path / "annotations.xml"
    path.write_text(f"<annotations>{images}</annotations>", encoding="utf-8")
    return path


def test_cvat_loads_box_centers_and_explicit_empty_images(tmp_path: Path):
    path = _write_cvat_annotations(
        tmp_path,
        '<image id="99" name="folder\\frame.png" width="10" height="8">'
        '<box label="ball" xtl="1" ytl="2" xbr="3" ybr="4"/>'
        '</image><image id="1" name="empty.png" width="10" height="8"/>',
    )

    annotations = equivalence_tool._load_cvat_ground_truth(path, {1: "ball"})

    assert annotations[0]["name"] == "folder/frame.png"
    assert annotations[0]["resolution"] == [10, 8]
    assert annotations[0]["detections"] == [
        {
            "center": [2.0, 3.0],
            "class_id": 1,
            "confidence": 1.0,
            "bbox_xyxy_px": [1.0, 2.0, 3.0, 4.0],
        }
    ]
    assert annotations[1]["detections"] == []


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ("<track/>", "image annotations"),
        ("", "no image annotations"),
        ('<image name="a.png" width="0" height="8"/>', "invalid dimensions"),
        (
            '<image name="a.png" width="10" height="8"/><image name="./a.png" width="10" height="8"/>',
            "duplicated",
        ),
        (
            '<image name="a.png" width="10" height="8"><polygon label="ball"/></image>',
            "Unsupported CVAT geometry",
        ),
        (
            '<image name="a.png" width="10" height="8"><box label="unknown"/></image>',
            "absent from the model class map",
        ),
        (
            '<image name="a.png" width="10" height="8"><box label="ball" xtl="1" ytl="2" xbr="nan" ybr="4"/></image>',
            "finite",
        ),
        (
            '<image name="a.png" width="10" height="8"><box label="ball" xtl="3" ytl="2" xbr="1" ybr="4"/></image>',
            "positive",
        ),
        (
            '<image name="a.png" width="10" height="8"><box label="ball" xtl="20" ytl="2" xbr="22" ybr="4"/></image>',
            "outside",
        ),
    ],
)
def test_cvat_rejects_unscoreable_or_invalid_annotations(
    tmp_path: Path, images: str, message: str
):
    with pytest.raises(ValueError, match=message):
        equivalence_tool._load_cvat_ground_truth(
            _write_cvat_annotations(tmp_path, images), {1: "ball"}
        )


def test_cvat_rejects_malformed_xml(tmp_path: Path):
    path = tmp_path / "invalid.xml"
    path.write_text("<annotations>", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read CVAT XML"):
        equivalence_tool._load_cvat_ground_truth(path, {1: "ball"})


def test_cvat_mapping_uses_names_and_unique_suffixes_not_annotation_order():
    annotations = [
        {"name": "export/second/frame.png"},
        {"name": "export/first/frame.png"},
        {"name": "old-prefix/unique.png"},
    ]
    mapped = equivalence_tool._map_cvat_ground_truth(
        annotations, ["first/frame.png", "second/frame.png", "new-prefix/unique.png"]
    )
    assert mapped["first/frame.png"] is annotations[1]
    assert mapped["second/frame.png"] is annotations[0]
    assert mapped["new-prefix/unique.png"] is annotations[2]


@pytest.mark.parametrize(
    ("annotations", "sources", "message"),
    [
        (
            [{"name": "left/frame.png"}, {"name": "right/frame.png"}],
            ["frame.png"],
            "2 annotation matches",
        ),
        ([{"name": "other.png"}], ["missing.png"], "0 annotation matches"),
        (
            [{"name": "frame.png"}],
            ["left/frame.png", "right/frame.png"],
            "Multiple input frames",
        ),
    ],
)
def test_cvat_rejects_ambiguous_missing_and_reused_mappings(
    annotations: list, sources: list, message: str
):
    with pytest.raises(ValueError, match=message):
        equivalence_tool._map_cvat_ground_truth(annotations, sources)


def test_cvat_scoring_counts_misses_duplicates_and_class_errors():
    annotations = [
        {"center": [0, 0], "class_id": 1, "confidence": 1},
        {"center": [10, 0], "class_id": 1, "confidence": 1},
        {"center": [20, 0], "class_id": 0, "confidence": 1},
    ]
    detections = [
        {"center": [3, 4], "class_id": 1, "confidence": 0.9},
        {"center": [3, 4], "class_id": 1, "confidence": 0.8},
        {"center": [20, 0], "class_id": 1, "confidence": 0.9},
    ]
    names = {0: "cue", 1: "ball"}
    score = equivalence_tool._score_ground_truth(
        annotations, detections, class_names=names, radius_px=5
    )
    summary = equivalence_tool._summarize_ground_truth([score], names)

    assert summary["true_positives"] == 1
    assert summary["false_positives"] == 2
    assert summary["false_negatives"] == 2
    assert summary["precision"] == pytest.approx(1 / 3)
    assert summary["recall"] == pytest.approx(1 / 3)
    assert summary["f1"] == pytest.approx(1 / 3)
    assert summary["center_errors_px"] == [5.0]
    assert summary["p95_center_error_px"] == 5.0
    assert summary["per_class"]["ball"]["false_negatives"] == 1
    assert summary["per_class"]["cue"]["false_negatives"] == 1
    assert "confidence_abs_error" not in score["matches"][0]


def test_cvat_assignment_maximizes_valid_match_count_before_center_error():
    annotations = [
        {"center": [0, 0], "class_id": 1, "confidence": 1},
        {"center": [4, 0], "class_id": 1, "confidence": 1},
    ]
    detections = [
        {"center": [1, 0], "class_id": 1, "confidence": 0.9},
        {"center": [-2, 0], "class_id": 1, "confidence": 0.9},
    ]
    score = equivalence_tool._score_ground_truth(
        annotations, detections, class_names={1: "ball"}, radius_px=3
    )
    assert score["true_positives"] == 2
    assert score["false_positives"] == 0
    assert score["false_negatives"] == 0


@pytest.mark.parametrize(
    "detections", [[], [{"center": [1, 1], "class_id": 1, "confidence": 0.9}]]
)
def test_cvat_empty_images_have_explicit_false_positives_and_undefined_recall(
    detections: list,
):
    score = equivalence_tool._score_ground_truth(
        [], detections, class_names={1: "ball"}, radius_px=5
    )
    summary = equivalence_tool._summarize_ground_truth([score], {1: "ball"})
    assert summary["false_positives"] == len(detections)
    assert summary["recall"] is None
    assert summary["mean_center_error_px"] is None
    assert summary["center_errors_px"] == []


@pytest.mark.parametrize("with_ground_truth", [False, True])
def test_cvat_rust_report_scoring_is_optional_and_survives_trace_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_ground_truth: bool
):
    image_path = _write_test_image(tmp_path / "frame.png")
    checkpoint_path = tmp_path / "reference.pt"
    checkpoint_path.write_bytes(b"retained checkpoint")
    rust_result_path = tmp_path / "rust.json"
    rust_result_path.write_text(
        json.dumps(_rust_result_fixture(image_path, checkpoint_path)), encoding="utf-8"
    )
    annotations_path = _write_cvat_annotations(
        tmp_path,
        '<image id="500" name="old/root/frame.png" width="10" height="8">'
        '<box label="ball" xtl="1" ytl="2" xbr="3" ybr="4"/></image>',
    )

    class FakeReferenceModel:
        names = {0: "cue", 1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeReferenceModel())
    monkeypatch.setattr(
        equivalence_tool, "_model_prediction", lambda *_args, **_kwargs: []
    )
    options = [
        str(image_path),
        "--reference-model",
        str(checkpoint_path),
        "--candidate-result",
        str(rust_result_path),
        "--output-dir",
        str(tmp_path / "reports"),
        "--max-trace-frames",
        "0",
    ]
    if with_ground_truth:
        options.extend(
            [
                "--ground-truth-cvat",
                str(annotations_path),
                "--ground-truth-radius-px",
                "2",
            ]
        )
    arguments = build_argument_parser().parse_args(options)

    json_path, markdown_path, result = equivalence_tool.run_equivalence_test(arguments)
    assert result["frames"] == []
    assert result["frames_omitted_from_details"] == 1
    assert ("ground_truth" in result) is with_ground_truth
    assert json.loads(json_path.read_text()) == json.loads(json.dumps(result))
    if with_ground_truth:
        assert result["ground_truth"]["reference"]["false_negatives"] == 1
        assert result["ground_truth"]["candidate"]["true_positives"] == 1
        assert result["ground_truth"]["candidate"]["per_class"]["ball"]["f1"] == 1
        assert result["ground_truth"]["association_radius_px"] == 2
        assert "No ground-truth accuracy" not in str(result["limitations"])
        assert "| candidate | ball | 1 | 0 | 0 |" in markdown_path.read_text()
        assert "not a production acceptance criterion" in markdown_path.read_text()
    else:
        assert result["limitations"][0].startswith("No ground-truth accuracy")
        assert "CVAT" not in markdown_path.read_text()


@pytest.mark.parametrize("radius", ["0", "-1", "nan", "inf"])
def test_cvat_cli_rejects_invalid_association_radius(radius: str):
    with pytest.raises(SystemExit):
        equivalence_tool.main(
            [
                "input.png",
                "--reference-model",
                "reference.pt",
                "--candidate-model",
                "candidate.onnx",
                "--ground-truth-radius-px",
                radius,
            ]
        )


def test_ground_truth_bounds_are_preserved_only_when_requested():
    from types import SimpleNamespace

    import torch

    class FakeBoxes:
        xywh = torch.tensor([[2.0, 3.0, 2.0, 2.0]])
        xyxy = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        cls = torch.tensor([1.0])
        conf = torch.tensor([0.9])

        def __len__(self):
            return 1

    result = SimpleNamespace(boxes=FakeBoxes())
    original = equivalence_tool._extract_detections(result)
    with_bounds = equivalence_tool._extract_detections(result, include_bounds=True)

    assert "bbox_xyxy_px" not in original[0]
    assert with_bounds[0]["bbox_xyxy_px"] == [1.0, 2.0, 3.0, 4.0]
    assert with_bounds[0]["center"] == original[0]["center"]


def test_cvat_dimensions_are_checked_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image_path = _write_test_image(tmp_path / "frame.png")
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.onnx"
    reference.write_bytes(b"reference")
    candidate.write_bytes(b"candidate")
    annotations = _write_cvat_annotations(
        tmp_path, '<image name="frame.png" width="999" height="8"/>'
    )

    class FakeModel:
        names = {1: "ball"}

    monkeypatch.setattr(equivalence_tool, "YOLO", lambda _path: FakeModel())

    def unexpected_inference(*_args, **_kwargs):
        pytest.fail("Invalid annotation dimensions must fail before inference")

    monkeypatch.setattr(equivalence_tool, "_model_prediction", unexpected_inference)
    arguments = build_argument_parser().parse_args(
        [
            str(image_path),
            "--reference-model",
            str(reference),
            "--candidate-model",
            str(candidate),
            "--ground-truth-cvat",
            str(annotations),
        ]
    )
    with pytest.raises(ValueError, match="CVAT dimensions differ"):
        equivalence_tool.run_equivalence_test(arguments)
