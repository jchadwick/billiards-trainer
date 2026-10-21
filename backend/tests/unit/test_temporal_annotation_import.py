"""Strict tests for importing reviewed CVAT video tracks as temporal labels."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import backend.tools.import_cvat_video_temporal_annotations as importer
from backend.tools.temporal_evaluation import TemporalAnnotationError


def _probe_output(
    *,
    times: list[str] | None = None,
    width: int = 100,
    height: int = 80,
) -> dict[str, Any]:
    timestamps = ["12.000", "12.040", "12.110"] if times is None else times
    return {
        "streams": [{"codec_type": "video", "width": width, "height": height}],
        "frames": [
            {"best_effort_timestamp_time": timestamp} for timestamp in timestamps
        ],
    }


def _cvat_xml(
    *,
    ball_two_frame: str = '<box frame="2" outside="0" occluded="0" keyframe="1" xtl="40" ytl="30" xbr="60" ybr="50"/>',
    geometry: tuple[int, int] = (100, 80),
    cue_shapes: str = (
        '<polyline frame="0" outside="0" occluded="0" keyframe="1" '
        'points="4,5;80,50"/><polyline frame="2" outside="0" '
        'occluded="1" keyframe="1" points="5,6;81,51"/>'
    ),
    include_ball_tracks: bool = True,
) -> str:
    width, height = geometry
    ball_tracks = (
        """<track id="ball-a" label="ball">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
    <box frame="2" outside="1" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
  </track>
  <track id="ball-b" label="ball">"""
        + ball_two_frame
        + "</track>"
        if include_ball_tracks
        else ""
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><size>3</size><mode>interpolation</mode><original_size>
    <width>{width}</width><height>{height}</height>
  </original_size><labels><label><name>ball</name><type>rectangle</type></label>
    <label><name>cue</name><type>polyline</type></label></labels></task></meta>
  {ball_tracks}
  <track id="cue-a" label="cue">{cue_shapes}</track>
</annotations>
"""


def _write_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    xml: str | None = None,
    reviewed_frames: Any = None,
    epoch_events: Any = None,
    probe: dict[str, Any] | None = None,
    stream_index: int = 0,
) -> tuple[Path, Path, Path, Path, Path | None]:
    if reviewed_frames is None:
        reviewed_frames = [0, 2]
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video bytes for importer test")
    annotations = tmp_path / "cvat.xml"
    annotations.write_text(xml or _cvat_xml(), encoding="utf-8")
    reviewed = tmp_path / "reviewed.json"
    reviewed.write_text(json.dumps(reviewed_frames), encoding="utf-8")
    output = tmp_path / "labels.json"
    event_path: Path | None = None
    if epoch_events is not None:
        event_path = tmp_path / "epochs.json"
        event_path.write_text(json.dumps(epoch_events), encoding="utf-8")
    probe_document = _probe_output() if probe is None else probe

    def fake_run(
        command: list[str], timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]:
        assert "-show_frames" in command
        assert command[command.index("-select_streams") + 1] == f"v:{stream_index}"
        assert timeout_seconds > 0
        return subprocess.CompletedProcess(command, 0, json.dumps(probe_document), "")

    monkeypatch.setattr(importer, "_execute_ffprobe", fake_run)
    return video, annotations, reviewed, output, event_path


def _build_temporal_annotations(*, video_path: Path, **kwargs: Any) -> dict[str, Any]:
    """Use the video hash as the pre-CVAT source identity attestation."""
    expected_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    return importer.build_temporal_annotations(
        video_path=video_path,
        expected_video_sha256=expected_hash,
        **kwargs,
    )


def test_import_uses_reviewed_keyframes_pts_offsets_and_epoch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {
            "source_frame_index": 0,
            "expected_calibration_revision": 3,
            "expected_source_session_epoch": "camera-session-before-clip",
        },
        {
            "source_frame_index": 100,
            "expected_calibration_revision": 4,
            "expected_source_session_epoch": "camera-session-a",
        },
        {
            "source_frame_index": 102,
            "expected_calibration_revision": 5,
            "expected_source_session_epoch": "camera-session-b",
        },
    ]
    video, annotations, reviewed, output, event_path = _write_inputs(
        tmp_path, monkeypatch, epoch_events=events
    )
    document = _build_temporal_annotations(
        video_path=video,
        cvat_xml_path=annotations,
        reviewed_frames_path=reviewed,
        output_path=output,
        frame_index_offset=100,
        epoch_events_path=event_path,
    )

    first, second = document["frames"]
    assert [frame["source_frame_index"] for frame in document["frames"]] == [100, 102]
    assert [frame["presentation_time_seconds"] for frame in document["frames"]] == [
        0.0,
        pytest.approx(0.11),
    ]
    assert first["expected_calibration_revision"] == 4
    assert second["expected_source_session_epoch"] == "camera-session-b"
    assert first["balls"] == [
        {
            "ball_id": "ball-a",
            "status": "visible",
            "center": {"x": 20.0, "y": 30.0},
            "radius_px": 10.0,
        }
    ]
    assert first["cue"] == {
        "status": "visible",
        "tip": {"x": 4.0, "y": 5.0},
        "butt": {"x": 80.0, "y": 50.0},
    }
    assert second["balls"] == [
        {"ball_id": "ball-a", "status": "removed"},
        {
            "ball_id": "ball-b",
            "status": "visible",
            "center": {"x": 50.0, "y": 40.0},
            "radius_px": 10.0,
        },
    ]
    assert second["cue"] == {"status": "occluded"}
    metadata = document["annotation_metadata"]
    assert metadata["presentation_time_origin_seconds"] == 12.0
    assert (
        metadata["source_annotation_sha256"]
        == hashlib.sha256(annotations.read_bytes()).hexdigest()
    )
    assert event_path is not None
    assert (
        metadata["epoch_events_sha256"]
        == hashlib.sha256(event_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("reviewed", "xml", "message"),
    [
        ([True], None, "must be an integer"),
        ([2, 1], None, "strictly increasing"),
        ([3], None, "outside the video"),
        (
            [2],
            _cvat_xml(
                ball_two_frame='<box frame="2" outside="0" occluded="0" keyframe="0" xtl="40" ytl="30" xbr="60" ybr="50"/>'
            ),
            "non-keyframe",
        ),
        (
            [1],
            _cvat_xml(),
            "interpolated state",
        ),
    ],
)
def test_import_rejects_unreviewed_or_invalid_frame_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reviewed: list[Any],
    xml: str | None,
    message: str,
) -> None:
    video, annotations, reviewed_path, output, _ = _write_inputs(
        tmp_path, monkeypatch, xml=xml, reviewed_frames=reviewed
    )
    with pytest.raises(TemporalAnnotationError, match=message):
        _build_temporal_annotations(
            video_path=video,
            cvat_xml_path=annotations,
            reviewed_frames_path=reviewed_path,
            output_path=output,
        )


@pytest.mark.parametrize(
    ("probe", "xml", "message"),
    [
        (_probe_output(times=["1.0", "1.0", "2.0"]), None, "strictly increasing"),
        (_probe_output(times=["1.0", "NaN", "2.0"]), None, "not finite"),
        (_probe_output(width=101), None, "does not match source video"),
        (_probe_output(times=["1", "2", "3", "4"]), None, "task size"),
        (
            _probe_output(),
            _cvat_xml(
                ball_two_frame='<box frame="3" outside="1" occluded="0" keyframe="1" xtl="40" ytl="30" xbr="60" ybr="50"/>'
            ),
            "outside the source video",
        ),
        (
            _probe_output(),
            _cvat_xml().replace("<name>ball</name>", "<name>eight</name>"),
            "does not declare the configured labels",
        ),
        (
            _probe_output(),
            _cvat_xml(
                cue_shapes='<polyline frame="0" outside="0" occluded="0" keyframe="1" points="1,2;3,4;5,6"/>'
            ),
            "exactly two points",
        ),
        (_probe_output(), _cvat_xml(geometry=(200, 80)), "does not match source video"),
    ],
)
def test_import_rejects_invalid_timestamps_geometry_and_cue_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: dict[str, Any],
    xml: str | None,
    message: str,
) -> None:
    video, annotations, reviewed, output, _ = _write_inputs(
        tmp_path, monkeypatch, xml=xml, probe=probe
    )
    with pytest.raises(TemporalAnnotationError, match=message):
        _build_temporal_annotations(
            video_path=video,
            cvat_xml_path=annotations,
            reviewed_frames_path=reviewed,
            output_path=output,
        )


def test_epoch_events_are_strictly_ordered_and_revision_is_safe_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_events = [
        [
            {
                "source_frame_index": 1,
                "expected_calibration_revision": 2,
                "expected_source_session_epoch": "a",
            },
            {
                "source_frame_index": 1,
                "expected_calibration_revision": 3,
                "expected_source_session_epoch": "b",
            },
        ],
        [
            {
                "source_frame_index": 0,
                "expected_calibration_revision": 9_007_199_254_740_992,
                "expected_source_session_epoch": "a",
            }
        ],
    ]
    for events in invalid_events:
        video, annotations, reviewed, output, event_path = _write_inputs(
            tmp_path,
            monkeypatch,
            epoch_events=events,
        )
        with pytest.raises(TemporalAnnotationError, match="epoch event"):
            _build_temporal_annotations(
                video_path=video,
                cvat_xml_path=annotations,
                reviewed_frames_path=reviewed,
                output_path=output,
                epoch_events_path=event_path,
            )


def test_import_prevents_overwriting_input_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, annotations, reviewed, _, _ = _write_inputs(tmp_path, monkeypatch)
    with pytest.raises(TemporalAnnotationError, match="must not overwrite"):
        _build_temporal_annotations(
            video_path=video,
            cvat_xml_path=annotations,
            reviewed_frames_path=reviewed,
            output_path=video,
        )


def test_import_requires_expected_video_identity_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, annotations, reviewed, output, _ = _write_inputs(tmp_path, monkeypatch)
    with pytest.raises(TemporalAnnotationError, match="SHA-256 does not match"):
        importer.build_temporal_annotations(
            video_path=video,
            cvat_xml_path=annotations,
            reviewed_frames_path=reviewed,
            output_path=output,
            expected_video_sha256="f" * 64,
        )


def test_empty_ball_sequence_requires_an_explicit_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = _cvat_xml(include_ball_tracks=False)
    video, annotations, reviewed, output, _ = _write_inputs(
        tmp_path, monkeypatch, xml=xml
    )
    with pytest.raises(TemporalAnnotationError, match="no ball tracks"):
        _build_temporal_annotations(
            video_path=video,
            cvat_xml_path=annotations,
            reviewed_frames_path=reviewed,
            output_path=output,
        )
    document = _build_temporal_annotations(
        video_path=video,
        cvat_xml_path=annotations,
        reviewed_frames_path=reviewed,
        output_path=output,
        allow_empty_ball_sequence=True,
    )
    assert all(frame["balls"] == [] for frame in document["frames"])


def test_parser_exposes_review_provenance_and_input_options() -> None:
    args = importer.build_parser().parse_args(
        [
            "--video",
            "capture.mp4",
            "--expected-video-sha256",
            "a" * 64,
            "--cvat-xml",
            "annotations.xml",
            "--reviewed-frames",
            "reviewed.json",
            "--output",
            "labels.json",
            "--epoch-events",
            "epochs.json",
            "--frame-index-offset",
            "100",
            "--video-stream-index",
            "2",
        ]
    )
    assert args.frame_index_offset == 100
    assert args.video_stream_index == 2
    assert args.epoch_events == Path("epochs.json")


def test_cli_writes_versioned_annotations_and_prints_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video, annotations, reviewed, _, _ = _write_inputs(tmp_path, monkeypatch)
    output = tmp_path / "nested" / "labels.json"
    assert (
        importer.main(
            [
                "--video",
                str(video),
                "--expected-video-sha256",
                hashlib.sha256(video.read_bytes()).hexdigest(),
                "--cvat-xml",
                str(annotations),
                "--reviewed-frames",
                str(reviewed),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    serialized = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert serialized["schema_version"] == 1
    assert len(serialized["frames"]) == 2
    assert summary["reviewed_frames"] == 2
    assert summary["interpolated_states_used_as_ground_truth"] is False


def test_import_selects_and_records_configured_video_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, annotations, reviewed, output, _ = _write_inputs(
        tmp_path, monkeypatch, stream_index=2
    )
    document = _build_temporal_annotations(
        video_path=video,
        cvat_xml_path=annotations,
        reviewed_frames_path=reviewed,
        output_path=output,
        video_stream_index=2,
    )
    assert document["annotation_metadata"]["source_video_stream_index"] == 2
