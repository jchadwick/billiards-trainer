//! End-to-end process coverage for the bounded recorded-video replay CLI.
//!
//! Fake ffprobe/ffmpeg programs keep the test independent of system multimedia
//! packages and physical video files. Inference itself uses the checked-in
//! tiny ONNX detector, so this verifies the real command and runtime rather
//! than an instrument-only simulation.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::tempdir;

const TEST_ONNX: &[u8] = include_bytes!("fixtures/tiny_two_class_detector.onnx");
const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;
const FRAME_COUNT: usize = 2;

struct ReplayFixture {
    video: std::path::PathBuf,
    model: std::path::PathBuf,
    checkpoint: std::path::PathBuf,
    ffprobe: std::path::PathBuf,
    ffmpeg: std::path::PathBuf,
    output: std::path::PathBuf,
    frame_count: usize,
}

#[test]
fn compiled_video_replay_cli_runs_real_inference_and_writes_bounded_audit_artifacts() {
    let temporary = tempdir().unwrap();
    let fixture = create_fixture(temporary.path(), &[3.0, 3.04]);
    let result = run_cli(&fixture, "capacity", None);
    assert_success(&result);

    let report_path = fixture.output.join("video-replay-report.json");
    let markdown_path = fixture.output.join("video-replay-report.md");
    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let markdown = fs::read_to_string(&markdown_path).unwrap();

    assert_eq!(report["mode"], "capacity");
    assert_eq!(report["source"]["source_frame_count"], FRAME_COUNT);
    assert_eq!(report["source"]["width_px"], WIDTH);
    assert_eq!(report["source"]["height_px"], HEIGHT);
    assert_eq!(report["source"]["sha256"], sha256_file(&fixture.video));
    assert_eq!(report["model"]["onnx_sha256"], sha256_file(&fixture.model));
    assert_eq!(
        report["model"]["source_checkpoint_sha256"],
        sha256_file(&fixture.checkpoint)
    );
    assert_eq!(
        report["measurement"]["scheduled_source_frames"],
        FRAME_COUNT
    );
    assert_eq!(
        report["measurement"]["processed_source_frames"],
        FRAME_COUNT
    );
    assert_eq!(report["artifacts"]["detailed_trace_written"], 0);
    assert_eq!(report["temporal_accuracy"]["status"], "unavailable");
    assert_eq!(
        report["temporal_accuracy"]["reason"],
        "no reviewed temporal annotation file was supplied"
    );
    assert!(markdown.contains("# Recorded-video performance audit"));
    assert!(markdown.contains("unpaced processing capacity"));
    assert!(markdown.contains("Status: **unavailable**"));
    assert!(markdown.contains("no reviewed temporal annotation file was supplied"));
}

#[test]
fn paced_video_replay_cli_reports_replaced_frames_and_freshness_artifacts() {
    let temporary = tempdir().unwrap();
    // Nanosecond-spaced PTS values put the source schedule far below scheduler
    // granularity. Multiple frames are due when real replay processing begins,
    // exercising actual frame replacement instead of a simulated scheduler.
    let presentation_times = (0..20)
        .map(|index| index as f64 / 1_000_000_000.0)
        .collect::<Vec<_>>();
    let fixture = create_fixture(temporary.path(), &presentation_times);
    let details_path = fixture.output.join("frame-details.jsonl");
    let result = run_cli(&fixture, "realtime-latest-frame", Some(&details_path));
    assert_success(&result);

    let report_path = fixture.output.join("video-replay-report.json");
    let markdown_path = fixture.output.join("video-replay-report.md");
    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let markdown = fs::read_to_string(markdown_path).unwrap();
    let measurement = &report["measurement"];

    assert_eq!(report["mode"], "realtime_latest_frame");
    assert_eq!(
        report["source"]["source_frame_count"],
        presentation_times.len()
    );
    assert_eq!(
        measurement["scheduled_source_frames"],
        presentation_times.len()
    );
    assert_eq!(
        measurement["decoded_source_frames"],
        presentation_times.len()
    );
    let processed = measurement["processed_source_frames"].as_u64().unwrap() as usize;
    let replacements = measurement["latest_frame_replacements"].as_u64().unwrap() as usize;
    assert!((1..presentation_times.len()).contains(&processed));
    assert_eq!(processed + replacements, presentation_times.len());
    assert!(measurement["processed_source_deadline_misses"].is_number());

    // Freshness percentiles cover every processed source observation and are
    // independently present for inference start and inference completion.
    for metric_name in [
        "source_to_inference_start_ms",
        "source_to_inference_complete_ms",
    ] {
        let metric = &measurement[metric_name];
        assert_eq!(metric["sample_count"], processed);
        for percentile in ["p50_ms", "p95_ms", "p99_ms"] {
            let latency = metric[percentile].as_f64().unwrap();
            assert!(
                latency.is_finite() && latency >= 0.0,
                "{metric_name}.{percentile}"
            );
        }
    }

    assert_eq!(report["artifacts"]["detailed_trace_written"], processed);
    assert_eq!(report["artifacts"]["detailed_trace_dropped_by_limit"], 0);
    let details = fs::read_to_string(details_path).unwrap();
    let trace_records = details
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).expect("trace line must contain valid JSON")
        })
        .collect::<Vec<_>>();
    assert_eq!(trace_records.len(), processed);
    let source_indexes = trace_records
        .iter()
        .map(|record| record["source_frame_index"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert!(source_indexes.windows(2).all(|pair| pair[0] < pair[1]));
    let trace_replacements = source_indexes.first().copied().unwrap_or(0)
        + source_indexes
            .windows(2)
            .map(|pair| pair[1] - pair[0] - 1)
            .sum::<usize>();
    assert_eq!(trace_replacements, replacements);
    for record in &trace_records {
        assert!(record["source_to_inference_start_ms"].as_f64().is_some());
        assert!(record["source_to_inference_complete_ms"].as_f64().is_some());
    }
    assert!(markdown.contains("PTS-paced latest-frame replay"));
    assert!(markdown.contains(&format!("latest-frame replacements: {replacements}")));
}

fn create_fixture(root: &Path, presentation_times: &[f64]) -> ReplayFixture {
    let inputs = root.join("inputs");
    let output = root.join("artifacts");
    fs::create_dir_all(&inputs).unwrap();

    let video = inputs.join("synthetic-video.fixture");
    fs::write(&video, b"deterministic synthetic video identity\n").unwrap();
    let model = inputs.join("tiny-two-class.onnx");
    fs::write(&model, TEST_ONNX).unwrap();
    let checkpoint = inputs.join("source-checkpoint.fixture");
    fs::write(&checkpoint, b"deterministic source checkpoint identity\n").unwrap();
    write_model_manifest(&model, &checkpoint);

    let ffprobe = root.join("bin/ffprobe");
    let ffmpeg = root.join("bin/ffmpeg");
    let frames = presentation_times
        .iter()
        .map(|presentation_time| {
            serde_json::json!({
                "width": WIDTH,
                "height": HEIGHT,
                "best_effort_timestamp_time": format!("{presentation_time:.9}")
            })
        })
        .collect::<Vec<_>>();
    let probe_document = serde_json::json!({
        "streams": [{"index": 0, "codec_type": "video", "width": WIDTH, "height": HEIGHT}],
        "frames": frames
    });
    let ffprobe_script = format!(
        "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then\n  printf '%s\\n' 'ffprobe version deterministic-fixture'\n  exit 0\nfi\nfor arg in \"$@\"; do\n  if [ \"$arg\" = 'stream=index' ]; then\n    printf '%s\\n' '0'\n    exit 0\n  fi\ndone\ncat <<'JSON'\n{}\nJSON\n",
        serde_json::to_string(&probe_document).unwrap()
    );
    write_executable(&ffprobe, &ffprobe_script);
    let decoded_bytes = u64::from(WIDTH) * u64::from(HEIGHT) * 3 * presentation_times.len() as u64;
    let ffmpeg_script = format!(
        "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then\n  printf '%s\\n' 'ffmpeg version deterministic-fixture'\n  exit 0\nfi\nhead -c {decoded_bytes} /dev/zero\n"
    );
    write_executable(&ffmpeg, &ffmpeg_script);

    ReplayFixture {
        video,
        model,
        checkpoint,
        ffprobe,
        ffmpeg,
        output,
        frame_count: presentation_times.len(),
    }
}

fn run_cli(fixture: &ReplayFixture, mode: &str, details_jsonl: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_billiards-perception-video-replay"));
    command
        .arg("--video")
        .arg(&fixture.video)
        .arg("--onnx")
        .arg(&fixture.model)
        .arg("--source-checkpoint")
        .arg(&fixture.checkpoint)
        .args(["--provider", "cpu", "--image-size", "32"])
        .args(["--confidence", "0.25", "--iou", "0.45"])
        .args(["--mode", mode, "--warmup-runs", "0"])
        .arg("--ffprobe")
        .arg(&fixture.ffprobe)
        .arg("--ffmpeg")
        .arg(&fixture.ffmpeg)
        .arg("--maximum-frame-count")
        .arg(fixture.frame_count.to_string())
        .arg("--maximum-pixel-count")
        .arg((WIDTH * HEIGHT).to_string())
        .arg("--maximum-decoded-video-bytes")
        .arg((u64::from(WIDTH) * u64::from(HEIGHT) * 3 * fixture.frame_count as u64).to_string())
        .arg("--maximum-video-bytes")
        .arg("1024")
        .arg("--maximum-probe-json-bytes")
        .arg("4096")
        .arg("--subprocess-timeout-seconds")
        .arg("5")
        .arg("--output-dir")
        .arg(&fixture.output)
        .arg("--run-id")
        .arg("cli-smoke");
    if let Some(details_jsonl) = details_jsonl {
        command
            .arg("--details-jsonl")
            .arg(details_jsonl)
            .arg("--maximum-detail-records")
            .arg(fixture.frame_count.to_string());
    }
    command
        .output()
        .expect("could not launch compiled video replay CLI")
}

fn assert_success(result: &Output) {
    assert!(
        result.status.success(),
        "video replay CLI failed with {}\nstdout:\n{}\nstderr:\n{}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

fn write_model_manifest(model: &Path, checkpoint: &Path) {
    let manifest_path = model.with_file_name(format!(
        "{}.manifest.json",
        model.file_name().unwrap().to_string_lossy()
    ));
    let manifest = serde_json::json!({
        "manifest_schema_version": 1,
        "source_model": { "sha256": sha256_file(checkpoint) },
        "exported_model": {
            "path": model.file_name().unwrap().to_string_lossy(),
            "sha256": sha256_file(model)
        },
        "model": {
            "class_count": 2,
            "class_names": [
                { "id": 0, "name": "cue" },
                { "id": 1, "name": "ball" }
            ]
        }
    });
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

fn sha256_file(path: &Path) -> String {
    format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
}

fn write_executable(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}
