"""Deterministic tests for candidate-profile generation and parsing."""

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.tools.benchmark_vision_profiles as benchmark_module
from backend.tools.benchmark_vision_profiles import (
    ReplayClock,
    _normalize_temporal_result,
    benchmark_candidate,
    build_candidate_settings,
    parse_dimensions,
    parse_factors,
)
from backend.tools.temporal_evaluation import (
    TemporalAnnotationError,
    evaluate_temporal_localization,
    load_rust_scene_snapshots,
    load_temporal_annotations,
    unavailable_temporal_metrics,
    validate_annotation_document,
)
from backend.vision.phase_scheduler import VisionPhase


def test_candidate_settings_are_isolated_and_scale_each_phase_job():
    base = {
        "enabled": False,
        "processing_max_dimension": {phase.value: None for phase in VisionPhase},
        "intervals_seconds": {
            phase.value: {
                "motion_monitor": 0.1,
                "ball_map_verification": 0.5,
                "cue_detection": 0.25,
                "detailed_tracking": 0.0,
            }
            for phase in VisionPhase
        },
    }

    candidate = build_candidate_settings(
        base, maximum_dimension=720, cadence_multiplier=2.0
    )

    assert candidate["enabled"] is True
    assert set(candidate["processing_max_dimension"].values()) == {720}
    assert candidate["intervals_seconds"]["AIMING"]["motion_monitor"] == 0.2
    assert candidate["intervals_seconds"]["AIMING"]["detailed_tracking"] == 0.0
    assert base["enabled"] is False
    assert base["processing_max_dimension"]["AIMING"] is None
    assert base["intervals_seconds"]["AIMING"]["motion_monitor"] == 0.1

    cropped = build_candidate_settings(
        base,
        maximum_dimension=None,
        cadence_multiplier=1.0,
        crop_to_table=True,
    )
    assert set(cropped["ball_detection_crop"].values()) == {True}


def test_benchmark_candidate_parsers_validate_and_normalize_inputs():
    assert parse_dimensions("native,720") == [("native", None), ("720p", 1280)]
    assert parse_factors("0.5,1,2") == [0.5, 1.0, 2.0]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_dimensions("1080p")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_factors("0,-1")


VIDEO_HASH = "a" * 64


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": x, "y": y}


def _ball(ball_id: str, x: float, y: float, *, radius: float = 10.0) -> dict:
    return {
        "ball_id": ball_id,
        "status": "visible",
        "center": _point(x, y),
        "radius_px": radius,
    }


def _frame(
    index: int,
    *,
    balls: list[dict] | None = None,
    cue: dict | None = None,
    calibration: int | None = 0,
    session: str | None = "session-a",
) -> dict:
    return {
        "source_frame_index": index,
        "presentation_time_seconds": float(index),
        "expected_calibration_revision": calibration,
        "expected_source_session_epoch": session,
        "balls": [] if balls is None else balls,
        "cue": {"status": "absent"} if cue is None else cue,
    }


def _annotations(frames: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "sequence": {
            "video_sha256": VIDEO_HASH,
            "width": 100,
            "height": 80,
            "coordinate_space": "source_pixels",
        },
        "frames": frames,
    }


def _engine_frame(
    index: int,
    *,
    balls: list[dict] | None = None,
    status: str = "observed",
    observation_index: int | None = None,
    observation_time: float | None = None,
    cue_status: str = "observed",
    cue_present: bool = False,
    cue_tip: tuple[float, float] = (0, 0),
    cue_butt: tuple[float, float] = (0, 0),
    cue_angle: float = 0.0,
    calibration: int | None = 0,
    session: str | None = "session-a",
) -> dict:
    return {
        "source_frame_index": index,
        "presentation_time_seconds": float(index),
        "ball_positions_status": status,
        "ball_observation_source_frame_index": (
            index
            if observation_index is None and status == "observed"
            else observation_index
        ),
        "ball_observation_presentation_time_seconds": (
            float(index)
            if observation_time is None and status == "observed"
            else observation_time
        ),
        "balls": [] if balls is None else balls,
        "cue_observation_status": cue_status,
        "cue_observation_source_frame_index": (
            index if cue_status == "observed" else None
        ),
        "cue_present": cue_present if cue_status == "observed" else None,
        "cue_tip": {"x": cue_tip[0], "y": cue_tip[1]},
        "cue_butt": {"x": cue_butt[0], "y": cue_butt[1]},
        "cue_angle_degrees": cue_angle,
        "calibration_revision": calibration,
        "source_session_epoch": session,
    }


def _observed_ball(track_id: str | int | None, x: float, y: float) -> dict:
    return {"track_id": track_id, "center": _point(x, y)}


def test_temporal_annotation_schema_and_sequence_identity_are_strict(tmp_path: Path):
    document = _annotations([_frame(0, balls=[_ball("cue", 12, 15)])])
    validated = validate_annotation_document(
        document,
        expected_video_sha256=VIDEO_HASH,
        expected_geometry=(100, 80),
    )
    assert validated["schema_version"] == 1

    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert (
        load_temporal_annotations(
            path,
            expected_video_sha256=VIDEO_HASH,
            expected_geometry=(100, 80),
        )["frames"][0]["source_frame_index"]
        == 0
    )
    with pytest.raises(TemporalAnnotationError, match="SHA-256"):
        load_temporal_annotations(
            path,
            expected_video_sha256="b" * 64,
            expected_geometry=(100, 80),
        )
    with pytest.raises(TemporalAnnotationError, match="geometry"):
        validate_annotation_document(document, expected_geometry=(80, 100))


def test_shared_temporal_evaluation_fixture_matches_the_python_reference():
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "perception/contracts/v1/temporal-evaluation-golden.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    annotations = validate_annotation_document(fixture["annotations"])

    assert (
        evaluate_temporal_localization(annotations, fixture["engine_frames"])
        == fixture["expected"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: d.update(schema_version=2), "schema violation"),
        (
            lambda d: d.update(schema_version=1.0),
            "schema_version must be the integer 1",
        ),
        (
            lambda d: d["frames"][0].update(expected_calibration_revision="cal-a"),
            "schema violation",
        ),
        (
            lambda d: d["frames"][0].update(
                expected_calibration_revision=9_007_199_254_740_992
            ),
            "schema violation",
        ),
        (
            lambda d: d["sequence"].update(width=100.0),
            "sequence width must be an integer",
        ),
        (
            lambda d: d["frames"][0].update(source_frame_index=0.0),
            "source_frame_index in frame 0 must be an integer",
        ),
        (
            lambda d: d["frames"].append(copy.deepcopy(d["frames"][0])),
            "strictly increasing",
        ),
        (
            lambda d: d["frames"].append(
                {**copy.deepcopy(d["frames"][0]), "source_frame_index": 1}
            ),
            "timestamps must be strictly increasing",
        ),
        (
            lambda d: d.update(
                frames=[
                    {**copy.deepcopy(d["frames"][0]), "source_frame_index": 2},
                    {
                        **copy.deepcopy(d["frames"][0]),
                        "source_frame_index": 1,
                        "presentation_time_seconds": 1,
                    },
                ]
            ),
            "source_frame_index values must be strictly increasing",
        ),
        (
            lambda d: d["frames"][0]["balls"].append(
                copy.deepcopy(d["frames"][0]["balls"][0])
            ),
            "duplicate ground-truth ball_id",
        ),
        (
            lambda d: d["frames"][0]["balls"][0].update(
                status="visible", center={"x": math.inf, "y": 0}
            ),
            "non-finite",
        ),
        (
            lambda d: d["frames"][0]["balls"][0].update(status="occluded"),
            "schema violation",
        ),
    ],
)
def test_temporal_annotation_semantic_and_schema_failures(mutation, message):
    document = _annotations([_frame(0, balls=[_ball("cue", 12, 15)])])
    mutation(document)
    with pytest.raises(TemporalAnnotationError, match=message):
        validate_annotation_document(document)


def _rust_scene_fixture() -> dict:
    path = (
        Path(__file__).resolve().parents[3]
        / "perception/contracts/v1/scene-rust-tracked-success.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_rust_scene_adapter_uses_label_pts_and_scores_only_source_observations(
    tmp_path: Path,
):
    scene = _rust_scene_fixture()
    annotation = _annotations(
        [
            _frame(
                0,
                balls=[_ball("ball-1", 1882.4, 1024.7)],
                cue={
                    "status": "visible",
                    "tip": _point(600.0, 400.0),
                    "butt": _point(400.0, 400.0),
                },
                calibration=None,
                session="replay-session-01",
            )
        ]
    )
    annotation["sequence"].update(
        {"width": 3840, "height": 2160, "video_sha256": VIDEO_HASH}
    )
    annotation["frames"][0]["presentation_time_seconds"] = 0.25
    source = scene["source"]
    source["timestamp"]["ticks"] = "41607041"
    scene["published_at"]["ticks"] = "94901583"
    scene_path = tmp_path / "scene-snapshots.jsonl"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")

    frames = load_rust_scene_snapshots(scene_path, annotation)
    report = evaluate_temporal_localization(annotation, frames)

    assert frames[0]["presentation_time_seconds"] == 0.25
    assert report["ball_localization"]["precision"] == 1.0
    assert report["ball_localization"]["recall"] == 1.0
    assert report["observation_freshness"]["age_seconds_p50"] == 0.0
    assert report["cue"]["presence_recall"] == 1.0
    assert report["cue"]["center_error_px"]["error_px_p50"] == 0.0
    assert report["cue"]["tip_error_px"]["error_px_count"] == 0


def test_rust_scene_adapter_excludes_cached_ball_positions(tmp_path: Path):
    scene = _rust_scene_fixture()
    scene["balls"][0]["status"] = "tracked"
    scene["balls"][0]["estimate"] = {
        "kind": "filtered",
        "as_of": scene["source"]["timestamp"],
        "image_center_px": {"x": 1882.4, "y": 1024.7},
        "image_space": "raw",
        "image_width_px": 3840,
        "image_height_px": 2160,
        "table_center_mm": None,
        "table_position_uncertainty": {
            "status": "unavailable",
            "sigma_x_mm": None,
            "sigma_y_mm": None,
        },
    }
    annotation = _annotations(
        [
            _frame(
                0, balls=[_ball("ball-1", 1882.4, 1024.7)], session="replay-session-01"
            )
        ]
    )
    annotation["sequence"].update({"width": 3840, "height": 2160})
    scene_path = tmp_path / "scene-snapshots.jsonl"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")

    frames = load_rust_scene_snapshots(scene_path, annotation)
    report = evaluate_temporal_localization(annotation, frames)

    assert frames[0]["balls"] == []
    assert report["ball_localization"]["fresh_predictions"] == 0
    assert report["ball_localization"]["recall"] == 0.0


def test_rust_scene_adapter_rejects_bad_geometry_and_frame_identity(tmp_path: Path):
    scene = _rust_scene_fixture()
    annotation = _annotations([_frame(0, session="replay-session-01")])
    annotation["sequence"].update({"width": 100, "height": 80})
    scene_path = tmp_path / "scene-snapshots.jsonl"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    with pytest.raises(TemporalAnnotationError, match="geometry"):
        load_rust_scene_snapshots(scene_path, annotation)

    scene["source"]["frame_id"] = "frame-zero"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    annotation["sequence"].update({"width": 3840, "height": 2160})
    with pytest.raises(TemporalAnnotationError, match="one-based frame-N"):
        load_rust_scene_snapshots(scene_path, annotation)


def test_rust_scene_adapter_rejects_measurement_geometry_and_nonfinite_numbers(
    tmp_path: Path,
):
    scene = _rust_scene_fixture()
    annotation = _annotations(
        [_frame(0, session="replay-session-01", balls=[_ball("ball-1", 10, 10)])]
    )
    annotation["sequence"].update({"width": 3840, "height": 2160})
    scene_path = tmp_path / "scene-snapshots.jsonl"
    scene["balls"][0]["measurement"]["image_width_px"] = 1920
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    with pytest.raises(TemporalAnnotationError, match="measurement geometry"):
        load_rust_scene_snapshots(scene_path, annotation)

    scene_path.write_text(
        json.dumps(_rust_scene_fixture()).replace("1882.4", "1e999") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TemporalAnnotationError, match="non-finite number"):
        load_rust_scene_snapshots(scene_path, annotation)


def test_rust_scene_adapter_requires_raw_coordinates_and_explicit_offset(
    tmp_path: Path,
):
    scene = _rust_scene_fixture()
    scene["balls"][0]["measurement"]["image_space"] = "rectified"
    annotation = _annotations(
        [
            _frame(
                4, balls=[_ball("ball-1", 1882.4, 1024.7)], session="replay-session-01"
            )
        ]
    )
    annotation["sequence"].update({"width": 3840, "height": 2160})
    scene_path = tmp_path / "scene-snapshots.jsonl"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    with pytest.raises(TemporalAnnotationError, match="requires raw source-pixel"):
        load_rust_scene_snapshots(scene_path, annotation, frame_index_offset=4)

    scene["balls"][0]["measurement"]["image_space"] = "raw"
    scene_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
    frames = load_rust_scene_snapshots(scene_path, annotation, frame_index_offset=4)
    assert frames[0]["source_frame_index"] == 4


def test_temporal_json_loader_rejects_nonstandard_and_overflow_numbers(tmp_path: Path):
    path = tmp_path / "annotations.json"
    path.write_text('{"number": NaN}', encoding="utf-8")
    with pytest.raises(TemporalAnnotationError, match="non-finite"):
        load_temporal_annotations(
            path, expected_video_sha256=VIDEO_HASH, expected_geometry=(100, 80)
        )

    document = _annotations([_frame(0)])
    document["frames"][0]["presentation_time_seconds"] = 1e999
    with pytest.raises(TemporalAnnotationError, match="non-finite"):
        validate_annotation_document(document)

    document["frames"][0]["presentation_time_seconds"] = 0.0
    path.write_text(json.dumps(document).replace("0.0", "1e999"), encoding="utf-8")
    with pytest.raises(TemporalAnnotationError, match="non-finite"):
        load_temporal_annotations(
            path, expected_video_sha256=VIDEO_HASH, expected_geometry=(100, 80)
        )


def test_global_assignment_maximizes_matches_under_radius_gate():
    annotations = _annotations(
        [
            _frame(
                0, balls=[_ball("left", 0, 0, radius=8), _ball("right", 5, 0, radius=8)]
            )
        ]
    )
    # Greedy nearest-pair matching would take left->1 and strand right. The
    # global assignment instead chooses left->-4 and right->1 for two matches.
    engine = [
        _engine_frame(
            0,
            balls=[_observed_ball("track-1", 1, 0), _observed_ball("track-2", -4, 0)],
        )
    ]
    report = evaluate_temporal_localization(
        annotations, engine, matching_radius_fraction=0.5
    )
    assert report["ball_localization"]["matched_balls"] == 2
    assert report["ball_localization"]["precision"] == 1.0
    assert report["ball_localization"]["recall"] == 1.0


def test_missing_and_cached_frames_are_misses_and_freshness_uses_observation_time():
    annotations = _annotations(
        [
            _frame(0, balls=[_ball("cue", 10, 10)]),
            _frame(1, balls=[_ball("cue", 10, 10)]),
        ]
    )
    engine = [
        _engine_frame(0, balls=[_observed_ball("track", 10, 10)]),
        _engine_frame(
            1,
            balls=[_observed_ball("track", 10, 10)],
            status="cached",
            observation_index=0,
            observation_time=0,
        ),
    ]
    report = evaluate_temporal_localization(
        annotations, engine, freshness_budget_seconds=0.25
    )
    assert report["ball_localization"]["fresh_predictions"] == 1
    assert report["ball_localization"]["false_negatives"] == 1
    assert report["observation_freshness"]["age_seconds_mean"] == 0.5
    assert report["observation_freshness"]["freshness_budget_misses"] == 1
    assert report["observation_freshness"]["expected_samples"] == 2
    assert report["observation_freshness"]["missing_age_samples"] == 0
    assert report["observation_freshness"]["freshness_budget_miss_rate"] == 0.5

    missing = evaluate_temporal_localization(annotations, engine[:1])
    assert missing["availability"]["missing_engine_frames"] == 1
    assert missing["ball_localization"]["false_negatives"] == 1
    assert missing["observation_freshness"]["missing_age_samples"] == 1
    assert missing["observation_freshness"]["freshness_budget_misses"] == 1


def test_missing_observation_age_is_counted_as_a_freshness_miss():
    annotations = _annotations([_frame(0), _frame(1)])
    engine = [
        _engine_frame(
            0,
            balls=[],
            status="observed",
            observation_time=None,
        ),
        _engine_frame(
            1,
            balls=[],
            status="failed",
            observation_index=None,
            observation_time=None,
        ),
    ]
    # An observed current-frame empty map is fresh; a failed frame with no
    # timestamp is an unknown-age freshness miss.
    engine[0]["ball_observation_presentation_time_seconds"] = None
    report = evaluate_temporal_localization(annotations, engine)
    freshness = report["observation_freshness"]
    assert freshness["expected_samples"] == 2
    assert freshness["available_samples"] == 1
    assert freshness["missing_age_samples"] == 1
    assert freshness["freshness_budget_misses"] == 1
    assert freshness["freshness_budget_miss_rate"] == 0.5


def test_wrong_source_frame_observation_is_rejected_as_stale_for_detection():
    annotations = _annotations([_frame(0, balls=[_ball("cue", 10, 10)])])
    engine = [
        _engine_frame(
            0,
            balls=[_observed_ball("track", 10, 10)],
            status="observed",
            observation_index=4,
            observation_time=0,
        )
    ]
    report = evaluate_temporal_localization(annotations, engine)
    assert report["availability"]["stale_observed_ball_results"] == 1
    assert report["ball_localization"]["fresh_predictions"] == 0
    assert report["ball_localization"]["false_negatives"] == 1


def test_track_switch_first_detection_delay_and_occlusion_reacquisition():
    annotations = _annotations(
        [
            _frame(0, balls=[_ball("stable", 10, 10)]),
            _frame(
                1,
                balls=[{"ball_id": "stable", "status": "occluded"}],
            ),
            _frame(2, balls=[_ball("stable", 10, 10)]),
        ]
    )
    engine = [
        _engine_frame(0, balls=[_observed_ball("engine-a", 10, 10)]),
        _engine_frame(
            1,
            balls=[_observed_ball("engine-a", 10, 10)],
            status="cached",
            observation_index=0,
            observation_time=0,
        ),
        _engine_frame(2, balls=[_observed_ball("engine-b", 10, 10)]),
    ]
    report = evaluate_temporal_localization(annotations, engine)
    tracking = report["tracking"]
    assert tracking["identity_switches"] == 1
    assert tracking["first_detection_delay_seconds"]["delay_seconds_mean"] == 0
    assert tracking["occlusion_reacquisition_attempts"] == 1
    assert tracking["occlusion_reacquisition_successes"] == 1
    assert tracking["occlusion_reacquisition_delay_seconds"]["delay_seconds_mean"] == 0


def test_cue_presence_endpoints_and_wrapped_angle_are_scored_separately():
    cue_visible = {
        "status": "visible",
        "tip": _point(-1, 0.1763269807),
        "butt": _point(0, 0),
    }
    annotations = _annotations(
        [
            _frame(0, cue=cue_visible),
            _frame(1, cue={"status": "absent"}),
            _frame(2, cue={"status": "occluded"}),
        ]
    )
    engine = [
        _engine_frame(
            0,
            cue_present=True,
            cue_tip=(0, 0.1763269807),
            cue_butt=(0, 0),
            cue_angle=-170,
        ),
        _engine_frame(1, cue_present=True),
        _engine_frame(2, cue_status="skipped"),
    ]
    report = evaluate_temporal_localization(annotations, engine)
    cue = report["cue"]
    assert cue["presence_precision"] == 0.5
    assert cue["presence_recall"] == 1.0
    assert cue["tip_error_px"]["error_px_mean"] == 1.0
    assert cue["butt_error_px"]["error_px_mean"] == 0.0
    assert cue["wrapped_angle_error_degrees"]["error_degrees_mean"] == pytest.approx(
        20.0
    )
    assert cue["occluded_frames_excluded"] == 1


def test_stale_or_skipped_cue_inference_does_not_count_as_a_true_negative():
    annotations = _annotations([_frame(0, cue={"status": "absent"})])
    stale_result = SimpleNamespace(
        ball_positions_status="unavailable",
        ball_observation_source_frame_number=None,
        ball_observation_source_timestamp=None,
        ball_observation_source_timestamp_kind=None,
        balls=[],
        cue_detection_status="stale",
        cue=None,
        calibration_revision=0,
        source_process_id=None,
    )
    normalized_stale = _normalize_temporal_result(
        stale_result,
        frame_index=0,
        presentation_time_seconds=0.0,
        source_epoch=100.0,
        presentation_time_offset_seconds=0.0,
        source_resolution=(100, 80),
    )
    assert normalized_stale["cue_observation_status"] == "unavailable"
    assert normalized_stale["cue_present"] is None

    for engine_frame in (normalized_stale, _engine_frame(0, cue_status="skipped")):
        cue_metrics = evaluate_temporal_localization(annotations, [engine_frame])["cue"]
        assert cue_metrics["labeled_presence_frames"] == 1
        assert cue_metrics["scored_presence_frames"] == 0
        assert cue_metrics["unavailable_engine_frames"] == 1
        assert cue_metrics["true_negative"] == 0


def test_calibration_and_session_boundary_coverage_and_mismatches_are_independent():
    annotations = _annotations(
        [
            _frame(0, calibration=0, session="session-a"),
            _frame(1, calibration=0, session="session-a"),
            _frame(2, calibration=1, session="session-b"),
        ]
    )
    engine = [
        _engine_frame(0, calibration=0, session="session-a"),
        _engine_frame(1, calibration=1, session="session-a"),
        _engine_frame(2, calibration=1, session="session-a"),
    ]
    report = evaluate_temporal_localization(annotations, engine)
    assert report["calibration"]["frame_mismatch_count"] == 1
    assert report["calibration"]["expected_boundary_count"] == 1
    assert report["calibration"]["boundary_mismatch_count"] == 2
    assert report["source_session"]["frame_mismatch_count"] is None
    assert report["source_session"]["expected_boundary_count"] == 1
    assert report["source_session"]["boundary_mismatch_count"] == 1

    unknown_session = [dict(frame, source_session_epoch=None) for frame in engine]
    without_session = evaluate_temporal_localization(annotations, unknown_session)
    assert without_session["source_session"]["status"] == "unavailable"
    assert without_session["source_session"]["frame_mismatch_count"] is None
    assert without_session["source_session"]["expected_boundary_count"] == 1


def test_session_boundaries_compare_transitions_not_run_specific_ids():
    annotations = _annotations(
        [
            _frame(0, session="labeled-epoch-a"),
            _frame(1, session="labeled-epoch-a"),
            _frame(2, session="labeled-epoch-b"),
        ]
    )
    engine = [
        _engine_frame(0, session="rust-process-session-17"),
        _engine_frame(1, session="rust-process-session-17"),
        _engine_frame(2, session="rust-process-session-18"),
    ]

    metrics = evaluate_temporal_localization(annotations, engine)["source_session"]

    assert metrics["frame_mismatch_count"] is None
    assert metrics["expected_boundary_count"] == 1
    assert metrics["observed_boundary_count"] == 1
    assert metrics["boundary_mismatch_count"] == 0


@pytest.mark.parametrize("calibration", ["0", -1, 9_007_199_254_740_992, True])
def test_engine_calibration_revision_requires_nonnegative_safe_integer(calibration):
    annotations = _annotations([_frame(0, calibration=0)])
    engine = [_engine_frame(0, calibration=calibration)]
    with pytest.raises(TemporalAnnotationError, match="engine calibration_revision"):
        evaluate_temporal_localization(annotations, engine)


def test_labeled_session_boundary_resets_identity_switch_continuity():
    annotations = _annotations(
        [
            _frame(0, balls=[_ball("stable", 10, 10)], session="session-a"),
            _frame(1, balls=[_ball("stable", 10, 10)], session="session-b"),
        ]
    )
    engine = [
        _engine_frame(0, balls=[_observed_ball("before-restart", 10, 10)]),
        _engine_frame(1, balls=[_observed_ball("after-restart", 10, 10)]),
    ]
    report = evaluate_temporal_localization(annotations, engine)
    assert report["tracking"]["source_session_boundary_id_resets"] == 1
    assert report["tracking"]["identity_switches"] == 0


@pytest.mark.parametrize(
    "engine",
    [
        [
            _engine_frame(0),
            _engine_frame(0),
        ],
        [
            _engine_frame(1),
            _engine_frame(0),
        ],
        [
            _engine_frame(0, balls=[{"track_id": "x", "center": _point(math.inf, 1)}]),
        ],
    ],
)
def test_engine_frame_duplicates_order_and_nonfinite_values_fail_closed(engine):
    annotations = _annotations([_frame(0)])
    with pytest.raises(TemporalAnnotationError):
        evaluate_temporal_localization(annotations, engine)


def test_temporal_metrics_without_annotations_are_explicitly_unavailable():
    report = unavailable_temporal_metrics()
    assert report["status"] == "unavailable"
    assert report["metrics"] is None
    assert "not camera exposure" in report["timing_basis"]


def test_python_replay_adapter_uses_current_source_frame_and_presentation_time():
    result = SimpleNamespace(
        ball_positions_status="observed",
        ball_observation_source_frame_number=7,
        ball_observation_source_timestamp=120.25,
        ball_observation_source_timestamp_kind="replay_video_presentation",
        balls=[
            SimpleNamespace(
                position=(100, 200),
                coordinate_space="pixel",
                track_id=3,
                is_observed=True,
            ),
            SimpleNamespace(
                position=(50, 50),
                coordinate_space="pixel",
                track_id=4,
                is_observed=False,
            ),
        ],
        cue_detection_status="observed",
        cue=SimpleNamespace(
            coordinate_space="pixel",
            tip_position=(12, 13),
            butt_position=(1, 2),
            angle=90,
        ),
        calibration_revision=0,
        source_process_id=None,
    )
    frame = _normalize_temporal_result(
        result,
        frame_index=7,
        presentation_time_seconds=0.25,
        source_epoch=120.0,
        presentation_time_offset_seconds=0.0,
        source_resolution=(100, 80),
    )
    assert frame["ball_positions_status"] == "observed"
    assert frame["ball_observation_source_frame_index"] == 7
    assert frame["ball_observation_presentation_time_seconds"] == 0.25
    assert len(frame["balls"]) == 1
    assert frame["cue_tip"] == _point(12, 13)
    assert frame["cue_angle_degrees"] == 90
    assert frame["source_session_epoch"] is None
    assert frame["calibration_revision"] == 0


def test_python_replay_candidate_includes_compact_temporal_metrics(monkeypatch):
    class FakeCapture:
        def __init__(self, _video_path):
            self.reads = 0
            self.source_base = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == benchmark_module.cv2.CAP_PROP_FPS:
                return 30.0
            if prop == benchmark_module.cv2.CAP_PROP_FRAME_WIDTH:
                return 100
            if prop == benchmark_module.cv2.CAP_PROP_FRAME_HEIGHT:
                return 80
            if prop == benchmark_module.cv2.CAP_PROP_POS_MSEC:
                return (self.source_base + max(0, self.reads - 1)) * (1000.0 / 30.0)
            if prop == benchmark_module.cv2.CAP_PROP_POS_FRAMES:
                return self.source_base + self.reads
            return 0.0

        def set(self, prop, value):
            if prop == benchmark_module.cv2.CAP_PROP_POS_MSEC:
                self.source_base = round(value * 30.0 / 1000.0)
                return True
            return False

        def read(self):
            if self.reads:
                return False, None
            self.reads += 1
            import numpy as np

            return True, np.zeros((80, 100, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(benchmark_module.cv2, "VideoCapture", FakeCapture)
    frame_result = SimpleNamespace(
        phase="aiming",
        statistics=SimpleNamespace(processing_time=2.0),
        ball_detection_status="observed",
        cue_detection_status="observed",
        motion_monitor_status="observed",
        ball_positions_status="observed",
        ball_observation_source_frame_number=0,
        ball_observation_source_timestamp_kind="replay_video_presentation",
        balls=[
            SimpleNamespace(
                position=(20.0, 30.0),
                coordinate_space="pixel",
                track_id=11,
                is_observed=True,
                is_moving=False,
            )
        ],
        cue=None,
        map_trusted=True,
        calibration_revision=0,
    )

    class FakeTracker:
        def reset(self):
            pass

    class FakeModule:
        tracker = FakeTracker()

        def _process_single_frame(self, _image, _frame_number, timestamp, **_kwargs):
            frame_result.ball_observation_source_timestamp = timestamp
            frame_result.ball_observation_source_frame_number = _kwargs[
                "source_frame_number"
            ]
            return frame_result

        def _phase_frame_geometry_for_test(self):
            pass

    module = FakeModule()
    module._phase_geometry_signature = None
    module._phase_last_ball_map = []
    module._phase_last_ball_timestamp = None
    module._phase_last_ball_source_timestamp = None
    module._phase_last_ball_source_frame_number = None
    module._phase_last_ball_source_timestamp_kind = None
    module._phase_last_cue = None
    module._phase_last_cue_timestamp = None
    annotation_first = _frame(30, balls=[_ball("gt", 20, 30)])
    annotation_first["presentation_time_seconds"] = 1.0
    annotation_second = _frame(31, balls=[_ball("gt", 20, 30)])
    annotation_second["presentation_time_seconds"] = 31 / 30
    annotation = _annotations([annotation_first, annotation_second])
    report = benchmark_candidate(
        module,
        ReplayClock(),
        Path("fake.mp4"),
        name="fixture",
        settings={"enabled": True},
        frame_limit=1,
        start_seconds=1.0,
        temporal_annotations=annotation,
    )
    assert report["temporal_localization"]["ball_localization"]["recall"] == 1.0
    assert report["temporal_localization"]["availability"]["missing_engine_frames"] == 0
    assert (
        report["temporal_localization"]["ball_localization"][
            "visible_ground_truth_balls"
        ]
        == 1
    )
    assert report["temporal_evaluation_segment"]["annotated_frames"] == 1
    assert report["temporal_evaluation_segment"]["first_source_frame_index"] == 30
    assert "frames" not in report["temporal_localization"]
