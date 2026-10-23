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

#[test]
fn compiled_video_replay_cli_runs_real_inference_and_writes_bounded_audit_artifacts() {
    let temporary = tempdir().unwrap();
    let inputs = temporary.path().join("inputs");
    let output = temporary.path().join("artifacts");
    fs::create_dir_all(&inputs).unwrap();

    let video = inputs.join("synthetic-video.fixture");
    fs::write(&video, b"deterministic synthetic video identity\n").unwrap();
    let model = inputs.join("tiny-two-class.onnx");
    fs::write(&model, TEST_ONNX).unwrap();
    let checkpoint = inputs.join("source-checkpoint.fixture");
    fs::write(&checkpoint, b"deterministic source checkpoint identity\n").unwrap();
    write_model_manifest(&model, &checkpoint);

    let ffprobe = temporary.path().join("bin/ffprobe");
    let ffmpeg = temporary.path().join("bin/ffmpeg");
    write_executable(
        &ffprobe,
        r##"#!/bin/sh
if [ "$1" = "-version" ]; then
  printf '%s\n' 'ffprobe version deterministic-fixture'
  exit 0
fi
for arg in "$@"; do
  if [ "$arg" = 'stream=index' ]; then
    printf '%s\n' '0'
    exit 0
  fi
done
cat <<'JSON'
{"streams":[{"index":0,"codec_type":"video","width":32,"height":32}],"frames":[{"width":32,"height":32,"best_effort_timestamp_time":"3.000000"},{"width":32,"height":32,"best_effort_timestamp_time":"3.040000"}]}
JSON
"##,
    );
    write_executable(
        &ffmpeg,
        r##"#!/bin/sh
if [ "$1" = "-version" ]; then
  printf '%s\n' 'ffmpeg version deterministic-fixture'
  exit 0
fi
head -c 6144 /dev/zero
"##,
    );

    let result = run_cli(&video, &model, &checkpoint, &ffprobe, &ffmpeg, &output);
    assert_success(&result);

    let report_path = output.join("video-replay-report.json");
    let markdown_path = output.join("video-replay-report.md");
    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let markdown = fs::read_to_string(&markdown_path).unwrap();

    assert_eq!(report["mode"], "capacity");
    assert_eq!(report["source"]["source_frame_count"], FRAME_COUNT);
    assert_eq!(report["source"]["width_px"], WIDTH);
    assert_eq!(report["source"]["height_px"], HEIGHT);
    assert_eq!(report["source"]["sha256"], sha256_file(&video));
    assert_eq!(report["model"]["onnx_sha256"], sha256_file(&model));
    assert_eq!(
        report["model"]["source_checkpoint_sha256"],
        sha256_file(&checkpoint)
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

fn run_cli(
    video: &Path,
    model: &Path,
    checkpoint: &Path,
    ffprobe: &Path,
    ffmpeg: &Path,
    output: &Path,
) -> Output {
    Command::new(env!("CARGO_BIN_EXE_billiards-perception-video-replay"))
        .arg("--video")
        .arg(video)
        .arg("--onnx")
        .arg(model)
        .arg("--source-checkpoint")
        .arg(checkpoint)
        .args(["--provider", "cpu", "--image-size", "32"])
        .args(["--confidence", "0.25", "--iou", "0.45"])
        .args(["--mode", "capacity", "--warmup-runs", "0"])
        .arg("--ffprobe")
        .arg(ffprobe)
        .arg("--ffmpeg")
        .arg(ffmpeg)
        .arg("--maximum-frame-count")
        .arg(FRAME_COUNT.to_string())
        .arg("--maximum-pixel-count")
        .arg((WIDTH * HEIGHT).to_string())
        .arg("--maximum-decoded-video-bytes")
        .arg((u64::from(WIDTH) * u64::from(HEIGHT) * 3 * FRAME_COUNT as u64).to_string())
        .arg("--maximum-video-bytes")
        .arg("1024")
        .arg("--maximum-probe-json-bytes")
        .arg("4096")
        .arg("--subprocess-timeout-seconds")
        .arg("5")
        .arg("--output-dir")
        .arg(output)
        .arg("--run-id")
        .arg("cli-smoke")
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
