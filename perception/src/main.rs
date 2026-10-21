use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use billiards_perception::detection_batch::ImageSpace;
use billiards_perception::inference::{
    run_audit, AuditOptions, BatchOutputDestination, BatchOutputOptions, CalibrationAuditOptions,
    ProviderSelection, SceneSnapshotOutputDestination, SceneSnapshotOutputOptions,
};
use billiards_perception::tracking::TrackingConfigV1;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProviderArgument {
    Cpu,
    Auto,
    Openvino,
}

impl From<ProviderArgument> for ProviderSelection {
    fn from(value: ProviderArgument) -> Self {
        match value {
            ProviderArgument::Cpu => Self::Cpu,
            ProviderArgument::Auto => Self::Auto,
            ProviderArgument::Openvino => Self::Openvino,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SourceImageSpaceArgument {
    Raw,
    Rectified,
}

impl From<SourceImageSpaceArgument> for ImageSpace {
    fn from(value: SourceImageSpaceArgument) -> Self {
        match value {
            SourceImageSpaceArgument::Raw => Self::Raw,
            SourceImageSpaceArgument::Rectified => Self::Rectified,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "billiards-perception-infer",
    about = "Run recorded PNG/JPEG images through the current Rust ONNX inference prototype",
    long_about = "Recorded-image evaluation only. Model paths, provider policy, confidence, IoU, and fixed model image size are required so this command never silently selects a model, provider policy, or production thresholds."
)]
struct Arguments {
    /// Explicit ONNX export. A matching exporter manifest is required beside it.
    #[arg(long)]
    onnx: PathBuf,

    /// Explicit source .pt checkpoint whose hash must match the export manifest.
    #[arg(long)]
    source_checkpoint: PathBuf,

    /// One PNG/JPEG image or a directory searched recursively.
    #[arg(long)]
    input: PathBuf,

    /// Select ONNX Runtime CPU, the native OpenVINO CPU backend, or ONNX Runtime's max-efficiency policy.
    #[arg(long, value_enum)]
    provider: ProviderArgument,

    /// Fixed square ONNX input width and height; must match the export.
    #[arg(long)]
    image_size: u32,

    /// Minimum class score, explicitly selected for this audit run.
    #[arg(long)]
    confidence: f32,

    /// Class-aware NMS IoU threshold, explicitly selected for this audit run.
    #[arg(long)]
    iou: f32,

    /// Maximum detections retained after NMS (matches Ultralytics' common cap when set to 300).
    #[arg(long, default_value_t = 300)]
    max_detections: usize,

    /// Number of unreported inferences used to warm the measurement session.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Optional synthetic input presentation rate. A latest-frame mailbox
    /// replaces stale pending images when inference falls behind this rate.
    #[arg(long)]
    replay_rate_fps: Option<f64>,

    /// Coordinate space of the source image bytes. Required when auditing a calibration.
    #[arg(long, value_enum)]
    source_image_space: Option<SourceImageSpaceArgument>,

    /// Optional immutable table-calibration v1 bundle for calibrated replay analysis.
    #[arg(long)]
    calibration_bundle: Option<PathBuf>,

    /// Camera profile the replay runtime says it applied; required with --calibration-bundle.
    #[arg(long)]
    applied_camera_profile_id: Option<String>,

    /// Calibration revision the replay runtime says it applied.
    #[arg(long)]
    applied_calibration_revision: Option<u64>,

    /// Rectification revision the replay runtime says it applied.
    #[arg(long)]
    applied_rectification_revision: Option<u64>,

    /// Optional inference thread count applied to the selected backend when supported.
    /// Omit to use the backend defaults.
    #[arg(long)]
    intra_op_threads: Option<usize>,

    /// Keep the ONNX Runtime provider probe profile at this path for inspection.
    #[arg(long)]
    retain_provider_profile: Option<PathBuf>,

    /// Export raw little-endian float32 NCHW tensors for direct Python/OpenCV comparison.
    #[arg(long)]
    dump_preprocessed_directory: Option<PathBuf>,

    /// Write one detection-batch v1 JSON object per line into a new file during replay.
    #[arg(long, conflicts_with = "detection_batches_tcp")]
    detection_batches_jsonl: Option<PathBuf>,

    /// Send length-framed detection-batch v1 JSON to a loopback receiver during replay.
    #[arg(long, conflicts_with = "detection_batches_jsonl")]
    detection_batches_tcp: Option<SocketAddr>,

    /// Maximum JSON payload bytes per TCP frame; required with --detection-batches-tcp.
    #[arg(long)]
    detection_batches_max_message_bytes: Option<usize>,

    /// TCP connection deadline in milliseconds; required with --detection-batches-tcp.
    #[arg(long)]
    detection_batches_connect_timeout_ms: Option<u64>,

    /// TCP write deadline in milliseconds; required with --detection-batches-tcp.
    #[arg(long)]
    detection_batches_write_timeout_ms: Option<u64>,

    /// Unique identity for this inference engine run; required with --detection-batches-jsonl.
    #[arg(long)]
    engine_instance_id: Option<String>,

    /// Identity for the recorded input session; required with --detection-batches-jsonl.
    #[arg(long)]
    source_session_id: Option<String>,

    /// Write tracked scene-snapshot v1 JSONL during paced replay.
    #[arg(long, conflicts_with = "scene_snapshots_tcp")]
    scene_snapshots_jsonl: Option<PathBuf>,

    /// Send length-framed scene-snapshot v1 JSON to a loopback receiver during replay.
    #[arg(long, conflicts_with = "scene_snapshots_jsonl")]
    scene_snapshots_tcp: Option<SocketAddr>,

    /// Maximum JSON payload bytes per TCP frame; required with --scene-snapshots-tcp.
    #[arg(long)]
    scene_snapshots_max_message_bytes: Option<usize>,

    /// TCP connection and handshake deadline in milliseconds; required with --scene-snapshots-tcp.
    #[arg(long)]
    scene_snapshots_connect_timeout_ms: Option<u64>,

    /// TCP write deadline in milliseconds; required with --scene-snapshots-tcp.
    #[arg(long)]
    scene_snapshots_write_timeout_ms: Option<u64>,

    /// Initial asynchronous scene reconnect delay, in milliseconds.
    #[arg(long, default_value_t = 100)]
    scene_snapshots_reconnect_initial_delay_ms: u64,

    /// Maximum asynchronous scene reconnect delay, in milliseconds.
    #[arg(long, default_value_t = 2000)]
    scene_snapshots_reconnect_maximum_delay_ms: u64,

    /// Model class selected as pool balls; required with scene snapshot output.
    #[arg(long)]
    tracking_ball_class_id: Option<u32>,

    /// Optional model class selected as the cue ball.
    #[arg(long)]
    tracking_cue_class_id: Option<u32>,

    /// Camera profile associated with tracked coordinates.
    #[arg(long)]
    tracking_camera_profile_id: Option<String>,

    /// Maximum center displacement used for association, in source pixels.
    #[arg(long)]
    tracking_association_gate_px: Option<f64>,

    /// Maximum forward prediction horizon used only for association, in ms.
    #[arg(long)]
    tracking_association_prediction_horizon_ms: Option<u64>,

    /// Age after which an unmatched ball is marked lost, in ms.
    #[arg(long)]
    tracking_lost_after_ms: Option<u64>,

    /// Maximum observation age before a track is removed, in ms.
    #[arg(long)]
    tracking_maximum_track_age_ms: Option<u64>,

    /// Bound on simultaneously retained ball tracks.
    #[arg(long)]
    tracking_maximum_active_tracks: Option<usize>,

    /// Number of recent source frame IDs retained for duplicate detection.
    #[arg(long)]
    tracking_recent_observation_capacity: Option<usize>,

    /// Effective tracking configuration revision.
    #[arg(long)]
    tracking_configuration_revision: Option<u64>,

    /// Effective source image rectification revision.
    #[arg(long)]
    tracking_rectification_revision: Option<u64>,

    /// Also write the machine-readable report to this path; JSON is always printed to stdout.
    #[arg(long)]
    output_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();

    let detection_batch_output = batch_output_options(&arguments)?;
    let calibration = calibration_audit_options(&arguments, detection_batch_output.is_some())?;
    let scene_snapshot_output = scene_snapshot_output_options(
        &arguments,
        detection_batch_output.is_some(),
        calibration.as_ref(),
    )?;

    let report = run_audit(&AuditOptions {
        onnx_path: arguments.onnx,
        source_checkpoint_path: arguments.source_checkpoint,
        input_path: arguments.input,
        provider: arguments.provider.into(),
        image_size: arguments.image_size,
        confidence: arguments.confidence,
        iou: arguments.iou,
        max_detections: arguments.max_detections,
        intra_op_threads: arguments.intra_op_threads,
        warmup_runs: arguments.warmup_runs,
        replay_rate_fps: arguments.replay_rate_fps,
        source_image_space: arguments
            .source_image_space
            .map(ImageSpace::from)
            .unwrap_or(ImageSpace::Raw),
        retained_provider_profile: arguments.retain_provider_profile,
        dump_preprocessed_directory: arguments.dump_preprocessed_directory,
        detection_batch_output,
        calibration,
        scene_snapshot_output,
    })?;
    let json = serde_json::to_string_pretty(&report).context("failed to serialize audit report")?;
    if let Some(path) = arguments.output_json {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create output directory {}", parent.display())
                })?;
            }
        }
        fs::write(&path, &json)
            .with_context(|| format!("failed to write report {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn calibration_audit_options(
    arguments: &Arguments,
    has_detection_batch_output: bool,
) -> Result<Option<CalibrationAuditOptions>> {
    let has_calibration_setting = arguments.calibration_bundle.is_some()
        || arguments.applied_camera_profile_id.is_some()
        || arguments.applied_calibration_revision.is_some()
        || arguments.applied_rectification_revision.is_some();
    if !has_calibration_setting {
        return Ok(None);
    }

    let bundle_path = arguments
        .calibration_bundle
        .clone()
        .context("--calibration-bundle is required for calibrated replay analysis")?;
    ensure!(
        has_detection_batch_output,
        "calibrated replay analysis requires detection-batch output to preserve raw results"
    );
    ensure!(
        arguments.source_image_space.is_some(),
        "--source-image-space must be explicit when --calibration-bundle is used"
    );
    let applied_camera_profile_id = arguments
        .applied_camera_profile_id
        .clone()
        .context("--applied-camera-profile-id is required with --calibration-bundle")?;
    let applied_calibration_revision = arguments
        .applied_calibration_revision
        .context("--applied-calibration-revision is required with --calibration-bundle")?;
    let applied_rectification_revision = arguments
        .applied_rectification_revision
        .context("--applied-rectification-revision is required with --calibration-bundle")?;

    Ok(Some(CalibrationAuditOptions::load(
        bundle_path,
        applied_camera_profile_id,
        applied_calibration_revision,
        applied_rectification_revision,
    )?))
}

fn batch_output_options(arguments: &Arguments) -> Result<Option<BatchOutputOptions>> {
    let has_batch_setting = arguments.detection_batches_jsonl.is_some()
        || arguments.detection_batches_tcp.is_some()
        || arguments.detection_batches_max_message_bytes.is_some()
        || arguments.detection_batches_connect_timeout_ms.is_some()
        || arguments.detection_batches_write_timeout_ms.is_some()
        || arguments.engine_instance_id.is_some()
        || arguments.source_session_id.is_some();
    if !has_batch_setting {
        return Ok(None);
    }

    let engine_instance_id = arguments
        .engine_instance_id
        .clone()
        .context("--engine-instance-id is required with batch output")?;
    let source_session_id = arguments
        .source_session_id
        .clone()
        .context("--source-session-id is required with batch output")?;

    let destination = match (
        arguments.detection_batches_jsonl.clone(),
        arguments.detection_batches_tcp,
        arguments.detection_batches_max_message_bytes,
        arguments.detection_batches_connect_timeout_ms,
        arguments.detection_batches_write_timeout_ms,
    ) {
        (Some(path), None, None, None, None) => {
            ensure!(
                arguments.output_json.as_ref() != Some(&path),
                "aggregate report and detection batch output must use different paths"
            );
            BatchOutputDestination::JsonlFile { path }
        }
        (None, Some(address), Some(maximum_message_bytes), Some(connect_ms), Some(write_ms)) => {
            ensure!(maximum_message_bytes > 0, "maximum framed message size must be positive");
            ensure!(connect_ms > 0, "TCP connect timeout must be positive");
            ensure!(write_ms > 0, "TCP write timeout must be positive");
            BatchOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes,
                connect_timeout: Duration::from_millis(connect_ms),
                write_timeout: Duration::from_millis(write_ms),
            }
        }
        _ => bail!(
            "select --detection-batches-jsonl, or provide --detection-batches-tcp with its maximum-message and timeout settings"
        ),
    };

    Ok(Some(BatchOutputOptions {
        destination,
        engine_instance_id,
        source_session_id,
    }))
}

fn scene_snapshot_output_options(
    arguments: &Arguments,
    has_detection_batch_output: bool,
    calibration: Option<&CalibrationAuditOptions>,
) -> Result<Option<SceneSnapshotOutputOptions>> {
    let has_tracking_setting = arguments.tracking_ball_class_id.is_some()
        || arguments.tracking_cue_class_id.is_some()
        || arguments.tracking_camera_profile_id.is_some()
        || arguments.tracking_association_gate_px.is_some()
        || arguments
            .tracking_association_prediction_horizon_ms
            .is_some()
        || arguments.tracking_lost_after_ms.is_some()
        || arguments.tracking_maximum_track_age_ms.is_some()
        || arguments.tracking_maximum_active_tracks.is_some()
        || arguments.tracking_recent_observation_capacity.is_some()
        || arguments.tracking_configuration_revision.is_some()
        || arguments.tracking_rectification_revision.is_some();
    let has_tcp_setting = arguments.scene_snapshots_max_message_bytes.is_some()
        || arguments.scene_snapshots_connect_timeout_ms.is_some()
        || arguments.scene_snapshots_write_timeout_ms.is_some();
    let destination = match (
        arguments.scene_snapshots_jsonl.clone(),
        arguments.scene_snapshots_tcp,
        arguments.scene_snapshots_max_message_bytes,
        arguments.scene_snapshots_connect_timeout_ms,
        arguments.scene_snapshots_write_timeout_ms,
    ) {
        (None, None, None, None, None) => {
            ensure!(
                !has_tracking_setting,
                "tracking settings require scene snapshot output"
            );
            ensure!(!has_tcp_setting, "scene TCP settings require a TCP destination");
            return Ok(None);
        }
        (Some(path), None, None, None, None) => {
            ensure!(
                arguments.output_json.as_ref() != Some(&path),
                "aggregate report and scene snapshot output must use different paths"
            );
            if let Some(batch_path) = &arguments.detection_batches_jsonl {
                ensure!(
                    batch_path != &path,
                    "detection batch and scene snapshot output must use different paths"
                );
            }
            SceneSnapshotOutputDestination::JsonlFile { path }
        }
        (None, Some(address), Some(maximum_message_bytes), Some(connect_ms), Some(write_ms)) => {
            ensure!(
                address.ip().is_loopback(),
                "scene snapshot TCP address must be loopback"
            );
            ensure!(
                maximum_message_bytes > 0 && maximum_message_bytes <= u32::MAX as usize,
                "scene maximum framed message size must fit a non-zero u32"
            );
            ensure!(connect_ms > 0, "scene TCP connect timeout must be positive");
            ensure!(write_ms > 0, "scene TCP write timeout must be positive");
            ensure!(
                arguments.scene_snapshots_reconnect_initial_delay_ms > 0,
                "scene reconnect initial delay must be positive"
            );
            ensure!(
                arguments.scene_snapshots_reconnect_maximum_delay_ms
                    >= arguments.scene_snapshots_reconnect_initial_delay_ms,
                "scene reconnect maximum delay cannot be smaller than its initial delay"
            );
            SceneSnapshotOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes,
                connect_timeout: Duration::from_millis(connect_ms),
                write_timeout: Duration::from_millis(write_ms),
                reconnect_initial_delay: Duration::from_millis(
                    arguments.scene_snapshots_reconnect_initial_delay_ms,
                ),
                reconnect_maximum_delay: Duration::from_millis(
                    arguments.scene_snapshots_reconnect_maximum_delay_ms,
                ),
            }
        }
        _ => bail!(
            "select --scene-snapshots-jsonl, or provide --scene-snapshots-tcp with its maximum-message and timeout settings"
        ),
    };
    ensure!(
        has_detection_batch_output,
        "scene snapshot output requires detection-batch output"
    );
    ensure!(
        arguments.replay_rate_fps.is_some(),
        "scene snapshot output requires --replay-rate-fps"
    );
    if let Some(calibration) = calibration {
        if let Some(camera_profile_id) = &arguments.tracking_camera_profile_id {
            ensure!(
                camera_profile_id == calibration.applied_camera_profile_id(),
                "tracking and applied calibration camera profiles must match"
            );
        }
        if let Some(rectification_revision) = arguments.tracking_rectification_revision {
            ensure!(
                rectification_revision == calibration.applied_rectification_revision(),
                "tracking and applied calibration rectification revisions must match"
            );
        }
    }

    let tracking = TrackingConfigV1::new(
        arguments
            .tracking_ball_class_id
            .context("--tracking-ball-class-id is required with scene snapshot output")?,
        arguments.tracking_cue_class_id,
        arguments
            .tracking_camera_profile_id
            .clone()
            .context("--tracking-camera-profile-id is required with scene snapshot output")?,
        arguments
            .tracking_association_gate_px
            .context("--tracking-association-gate-px is required with scene snapshot output")?,
        Duration::from_millis(arguments.tracking_association_prediction_horizon_ms.context(
            "--tracking-association-prediction-horizon-ms is required with scene snapshot output",
        )?),
        Duration::from_millis(arguments.tracking_lost_after_ms.context(
            "--tracking-lost-after-ms is required with scene snapshot output",
        )?),
        Duration::from_millis(arguments.tracking_maximum_track_age_ms.context(
            "--tracking-maximum-track-age-ms is required with scene snapshot output",
        )?),
        arguments
            .tracking_maximum_active_tracks
            .context("--tracking-maximum-active-tracks is required with scene snapshot output")?,
        arguments
            .tracking_recent_observation_capacity
            .context("--tracking-recent-observation-capacity is required with scene snapshot output")?,
        arguments
            .tracking_configuration_revision
            .context("--tracking-configuration-revision is required with scene snapshot output")?,
        arguments
            .tracking_rectification_revision
            .context("--tracking-rectification-revision is required with scene snapshot output")?,
    )
    .context("invalid scene tracking configuration")?;
    Ok(Some(SceneSnapshotOutputOptions {
        destination,
        tracking,
    }))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        calibration_audit_options, scene_snapshot_output_options, Arguments,
        SourceImageSpaceArgument,
    };
    use billiards_perception::detection_batch::ImageSpace;
    use billiards_perception::inference::SceneSnapshotOutputDestination;
    use clap::Parser;

    #[test]
    fn replay_rate_is_an_explicit_optional_cli_setting() {
        let arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--replay-rate-fps",
            "30",
        ])
        .unwrap();

        assert_eq!(arguments.replay_rate_fps, Some(30.0));
    }

    #[test]
    fn unpaced_mode_remains_the_default_cli_behavior() {
        let arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
        ])
        .unwrap();

        assert_eq!(arguments.replay_rate_fps, None);
    }

    #[test]
    fn source_image_space_is_explicitly_selectable_for_replay_batches() {
        let arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--source-image-space",
            "rectified",
        ])
        .unwrap();

        assert!(matches!(
            arguments.source_image_space,
            Some(SourceImageSpaceArgument::Rectified)
        ));
        assert_eq!(
            arguments.source_image_space.map(ImageSpace::from),
            Some(ImageSpace::Rectified)
        );
    }

    #[test]
    fn calibration_audit_requires_batch_output_and_explicit_applied_context() {
        let arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--calibration-bundle",
            "calibration.json",
        ])
        .unwrap();

        let error = calibration_audit_options(&arguments, false)
            .expect_err("calibrated measurements need retained raw batches");
        assert!(error
            .to_string()
            .contains("requires detection-batch output"));
    }

    #[test]
    fn calibration_audit_loads_the_versioned_bundle_and_hashes_its_bytes() {
        use sha2::{Digest, Sha256};

        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("contracts/calibration/v1/table-calibration-python-export.json");
        let arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--source-image-space",
            "rectified",
            "--detection-batches-jsonl",
            "batches.jsonl",
            "--engine-instance-id",
            "engine-test",
            "--source-session-id",
            "session-test",
            "--calibration-bundle",
            fixture_path.to_str().unwrap(),
            "--applied-camera-profile-id",
            "test-camera-profile",
            "--applied-calibration-revision",
            "3",
            "--applied-rectification-revision",
            "2",
        ])
        .unwrap();

        let calibration = calibration_audit_options(&arguments, true)
            .unwrap()
            .expect("the explicit bundle settings enable geometry audit");
        assert_eq!(calibration.calibration_revision(), 3);
        let expected_sha256 = format!("{:x}", Sha256::digest(std::fs::read(fixture_path).unwrap()));
        assert_eq!(calibration.bundle_sha256(), expected_sha256);
        assert_eq!(
            arguments.source_image_space.map(ImageSpace::from),
            Some(ImageSpace::Rectified)
        );
    }

    #[test]
    fn scene_output_requires_complete_tracking_policy_and_paced_replay() {
        let incomplete = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--scene-snapshots-jsonl",
            "scene.jsonl",
        ])
        .unwrap();
        let error = scene_snapshot_output_options(&incomplete, false, None)
            .expect_err("scene records need retained detector output");
        assert!(error
            .to_string()
            .contains("requires detection-batch output"));
    }

    #[test]
    fn scene_output_accepts_explicit_bounded_tracking_configuration() {
        let mut arguments = Arguments::try_parse_from([
            "billiards-perception-infer",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--input",
            "frames",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--replay-rate-fps",
            "30",
            "--detection-batches-jsonl",
            "batches.jsonl",
            "--engine-instance-id",
            "engine-test",
            "--source-session-id",
            "session-test",
            "--scene-snapshots-jsonl",
            "scene.jsonl",
            "--tracking-ball-class-id",
            "1",
            "--tracking-cue-class-id",
            "0",
            "--tracking-camera-profile-id",
            "camera-a",
            "--tracking-association-gate-px",
            "18.5",
            "--tracking-association-prediction-horizon-ms",
            "50",
            "--tracking-lost-after-ms",
            "200",
            "--tracking-maximum-track-age-ms",
            "1000",
            "--tracking-maximum-active-tracks",
            "24",
            "--tracking-recent-observation-capacity",
            "32",
            "--tracking-configuration-revision",
            "4",
            "--tracking-rectification-revision",
            "7",
        ])
        .unwrap();
        arguments.replay_rate_fps = None;
        let error = scene_snapshot_output_options(&arguments, true, None)
            .expect_err("tracking requires source time from paced replay");
        assert!(error.to_string().contains("requires --replay-rate-fps"));

        arguments.replay_rate_fps = Some(30.0);
        let output = scene_snapshot_output_options(&arguments, true, None)
            .unwrap()
            .expect("scene settings enable scene output");
        assert!(matches!(
            output.destination,
            SceneSnapshotOutputDestination::JsonlFile { path }
                if path.as_path() == std::path::Path::new("scene.jsonl")
        ));
        assert_eq!(output.tracking.ball_class_id(), 1);
        assert_eq!(output.tracking.cue_class_id(), Some(0));
        assert_eq!(output.tracking.camera_profile_id(), "camera-a");
        assert_eq!(output.tracking.maximum_active_tracks(), 24);

        arguments.scene_snapshots_jsonl = None;
        arguments.scene_snapshots_tcp = Some("127.0.0.1:9000".parse().unwrap());
        arguments.scene_snapshots_max_message_bytes = Some(64 * 1024);
        arguments.scene_snapshots_connect_timeout_ms = Some(1000);
        arguments.scene_snapshots_write_timeout_ms = Some(1000);
        let tcp_output = scene_snapshot_output_options(&arguments, true, None)
            .unwrap()
            .expect("replay TCP settings enable scene output");
        assert!(matches!(
            tcp_output.destination,
            SceneSnapshotOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes,
                reconnect_initial_delay,
                reconnect_maximum_delay,
                ..
            } if maximum_message_bytes == 64 * 1024
                && address == "127.0.0.1:9000".parse().unwrap()
                && reconnect_initial_delay == std::time::Duration::from_millis(100)
                && reconnect_maximum_delay == std::time::Duration::from_millis(2000)
        ));

        arguments.scene_snapshots_reconnect_initial_delay_ms = 0;
        let error = scene_snapshot_output_options(&arguments, true, None)
            .expect_err("scene reconnect delay must be positive");
        assert!(error.to_string().contains("initial delay must be positive"));
        arguments.scene_snapshots_reconnect_initial_delay_ms = 100;
        arguments.scene_snapshots_reconnect_maximum_delay_ms = 99;
        let error = scene_snapshot_output_options(&arguments, true, None)
            .expect_err("maximum reconnect delay must cover the initial delay");
        assert!(error.to_string().contains("cannot be smaller"));
        arguments.scene_snapshots_reconnect_maximum_delay_ms = 2000;

        arguments.scene_snapshots_tcp = Some("192.0.2.1:9000".parse().unwrap());
        let error = scene_snapshot_output_options(&arguments, true, None)
            .expect_err("scene stream must remain local to the gateway");
        assert!(error.to_string().contains("must be loopback"));
    }
}
