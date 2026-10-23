use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{ensure, Context, Result};
use billiards_perception::detection_batch::ImageSpace;
use billiards_perception::inference::{
    run_video_audit, AuditOptions, BatchOutputDestination, BatchOutputOptions, ProviderSelection,
    SceneSnapshotOutputDestination, SceneSnapshotOutputOptions, VideoReplayMode,
    VideoReplayOptions, DEFAULT_VIDEO_MAXIMUM_BYTES, DEFAULT_VIDEO_MAXIMUM_DECODED_BYTES,
    DEFAULT_VIDEO_MAXIMUM_DETAIL_RECORDS, DEFAULT_VIDEO_MAXIMUM_FRAME_COUNT,
    DEFAULT_VIDEO_MAXIMUM_PIXEL_COUNT, DEFAULT_VIDEO_MAXIMUM_PROBE_JSON_BYTES,
    DEFAULT_VIDEO_SUBPROCESS_TIMEOUT_SECONDS,
};
use billiards_perception::temporal_evaluation::{
    evaluate_files, unavailable_temporal_metrics, EvaluationConfig, FileEvaluationConfig,
    DEFAULT_FRESHNESS_BUDGET_SECONDS, DEFAULT_MATCHING_RADIUS_FRACTION,
};
use billiards_perception::tracking::TrackingConfigV1;
use clap::{Parser, ValueEnum};
use serde_json::Value;

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
enum ModeArgument {
    RealtimeLatestFrame,
    Capacity,
}

impl From<ModeArgument> for VideoReplayMode {
    fn from(value: ModeArgument) -> Self {
        match value {
            ModeArgument::RealtimeLatestFrame => Self::RealtimeLatestFrame,
            ModeArgument::Capacity => Self::Capacity,
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
    name = "billiards-perception-video-replay",
    about = "Measure the Rust perception pipeline on a bounded recorded video",
    long_about = "Video replay has two distinct meanings: realtime-latest-frame preserves the video's presentation timing and drops stale queued frames; capacity decodes and processes every frame as quickly as possible. Choose a mode explicitly. Model, provider, image dimensions, confidence, and IoU are also required explicitly."
)]
struct Arguments {
    /// Input video file. The selected stream is decoded without resizing.
    #[arg(long)]
    video: PathBuf,

    /// Explicit ONNX export. A matching exporter manifest is required beside it.
    #[arg(long)]
    onnx: PathBuf,

    /// Explicit source .pt checkpoint whose hash must match the export manifest.
    #[arg(long)]
    source_checkpoint: PathBuf,

    /// Select ONNX Runtime CPU, native OpenVINO CPU, or ONNX Runtime's max-efficiency policy.
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

    /// Maximum detections retained after NMS.
    #[arg(long, default_value_t = 300)]
    max_detections: usize,

    /// Number of unreported inferences used to warm the measurement session.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Optional inference thread count; omit to use backend defaults.
    #[arg(long)]
    intra_op_threads: Option<usize>,

    /// Source video stream ordinal (zero based among video streams).
    #[arg(long, default_value_t = 0)]
    video_stream_ordinal: u32,

    /// Select PTS-paced latest-frame replay or unpaced sequential capacity measurement.
    #[arg(long, value_enum)]
    mode: ModeArgument,

    /// ffprobe executable used to inspect the selected video stream.
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,

    /// ffmpeg executable used to decode the selected stream.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,

    /// Maximum encoded source-video bytes.
    #[arg(long, default_value_t = DEFAULT_VIDEO_MAXIMUM_BYTES)]
    maximum_video_bytes: u64,

    /// Maximum ffprobe JSON bytes retained for stream and frame timing metadata.
    #[arg(long, default_value_t = DEFAULT_VIDEO_MAXIMUM_PROBE_JSON_BYTES)]
    maximum_probe_json_bytes: usize,

    /// Maximum source frames admitted by one audit run.
    #[arg(long, default_value_t = DEFAULT_VIDEO_MAXIMUM_FRAME_COUNT)]
    maximum_frame_count: usize,

    /// Maximum width*height accepted from stream metadata.
    #[arg(long, default_value_t = DEFAULT_VIDEO_MAXIMUM_PIXEL_COUNT)]
    maximum_pixel_count: u64,

    /// Maximum aggregate decoded RGB bytes accepted during the run.
    #[arg(long, default_value_t = DEFAULT_VIDEO_MAXIMUM_DECODED_BYTES)]
    maximum_decoded_video_bytes: u64,

    /// Wall-clock bound for each ffprobe/ffmpeg subprocess operation.
    #[arg(long, default_value_t = DEFAULT_VIDEO_SUBPROCESS_TIMEOUT_SECONDS)]
    subprocess_timeout_seconds: u64,

    /// Optional bounded detailed trace JSONL. Disabled by default to avoid measuring trace I/O.
    #[arg(long)]
    details_jsonl: Option<PathBuf>,

    /// Maximum detail records retained when --details-jsonl is enabled.
    #[arg(long)]
    maximum_detail_records: Option<usize>,

    /// Directory for this run's artifacts. Defaults to thoughts/performance-audits/<run-id>.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Run identity used for the default artifact directory; generated as a UUID when omitted.
    #[arg(long)]
    run_id: Option<String>,

    /// Aggregate report JSON path. Relative paths are resolved under --output-dir.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Markdown summary path. Relative paths are resolved under --output-dir.
    #[arg(long)]
    output_markdown: Option<PathBuf>,

    /// Reviewed Rust temporal annotation JSON for optional accuracy scoring.
    /// Requires scene-snapshot output and the exact annotated source video.
    #[arg(long)]
    annotations: Option<PathBuf>,

    /// Frame index offset between one-based Rust frame IDs and zero-based labels.
    #[arg(long, default_value_t = 0)]
    accuracy_frame_index_offset: i64,

    /// Matching gate as a fraction of each annotated ball radius.
    #[arg(long, default_value_t = DEFAULT_MATCHING_RADIUS_FRACTION)]
    accuracy_matching_radius_fraction: f64,

    /// Maximum acceptable observation age in source-video seconds.
    #[arg(long, default_value_t = DEFAULT_FRESHNESS_BUDGET_SECONDS)]
    accuracy_freshness_budget_seconds: f64,

    /// Input coordinate space. Defaults to raw camera pixels.
    #[arg(long, value_enum, default_value = "raw")]
    source_image_space: SourceImageSpaceArgument,

    /// Optional raw detection-batch JSONL accuracy artifact.
    #[arg(long)]
    detection_batches_jsonl: Option<PathBuf>,

    /// Unique inference-engine identity; required with --detection-batches-jsonl.
    #[arg(long)]
    engine_instance_id: Option<String>,

    /// Recorded source-session identity; required with --detection-batches-jsonl.
    #[arg(long)]
    source_session_id: Option<String>,

    /// Optional tracked scene-snapshot JSONL accuracy artifact.
    #[arg(long)]
    scene_snapshots_jsonl: Option<PathBuf>,

    /// Model class selected as pool balls; required with scene snapshots.
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

    /// Maximum forward prediction horizon used only for association, in milliseconds.
    #[arg(long)]
    tracking_association_prediction_horizon_ms: Option<u64>,

    /// Age after which an unmatched ball is marked lost, in milliseconds.
    #[arg(long)]
    tracking_lost_after_ms: Option<u64>,

    /// Maximum observation age before a track is removed, in milliseconds.
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
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let run_id = run_identity(arguments.run_id.as_deref())?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("perception crate is missing its repository root")?
        .to_path_buf();
    let output_dir = arguments
        .output_dir
        .as_deref()
        .map(resolve_from_current_directory)
        .unwrap_or_else(|| repo_root.join("thoughts/performance-audits").join(&run_id));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    validate_inference_arguments(&arguments)?;
    let output_json = resolve_output_path(
        arguments.output_json.as_deref(),
        &output_dir,
        "video-replay-report.json",
    );
    let output_markdown = resolve_output_path(
        arguments.output_markdown.as_deref(),
        &output_dir,
        "video-replay-report.md",
    );
    let details_jsonl = arguments
        .details_jsonl
        .as_deref()
        .map(|path| resolve_output_path(Some(path), &output_dir, ""));
    ensure!(
        details_jsonl.is_some() || arguments.maximum_detail_records.is_none(),
        "--maximum-detail-records requires --details-jsonl"
    );
    if let Some(limit) = arguments.maximum_detail_records {
        ensure!(limit > 0, "maximum detail records must be positive");
    }
    let maximum_detail_records = if details_jsonl.is_some() {
        arguments
            .maximum_detail_records
            .unwrap_or(DEFAULT_VIDEO_MAXIMUM_DETAIL_RECORDS)
    } else {
        0
    };
    let detection_batches_jsonl = arguments
        .detection_batches_jsonl
        .as_deref()
        .map(|path| resolve_output_path(Some(path), &output_dir, ""));
    let scene_snapshots_jsonl = arguments
        .scene_snapshots_jsonl
        .as_deref()
        .map(|path| resolve_output_path(Some(path), &output_dir, ""));
    ensure!(
        arguments.annotations.is_none() || scene_snapshots_jsonl.is_some(),
        "--annotations requires --scene-snapshots-jsonl so localization can be scored"
    );
    ensure!(
        arguments.accuracy_matching_radius_fraction.is_finite()
            && arguments.accuracy_matching_radius_fraction > 0.0,
        "accuracy matching radius fraction must be finite and positive"
    );
    ensure!(
        arguments.accuracy_freshness_budget_seconds.is_finite()
            && arguments.accuracy_freshness_budget_seconds >= 0.0,
        "accuracy freshness budget must be finite and non-negative"
    );
    let report_paths = [
        Some(output_json.as_path()),
        Some(output_markdown.as_path()),
        details_jsonl.as_deref(),
        detection_batches_jsonl.as_deref(),
        scene_snapshots_jsonl.as_deref(),
    ];
    validate_output_paths(&report_paths)?;
    for path in report_paths.iter().flatten() {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create output directory {}", parent.display())
                })?;
            }
        }
    }

    let batch_output = batch_output_options(
        detection_batches_jsonl,
        arguments.engine_instance_id.clone(),
        arguments.source_session_id.clone(),
    )?;
    let scene_output =
        scene_snapshot_output_options(scene_snapshots_jsonl, batch_output.is_some(), &arguments)?;
    let scoring_scene_path = scene_output
        .as_ref()
        .map(|output| match &output.destination {
            SceneSnapshotOutputDestination::JsonlFile { path } => path.clone(),
            SceneSnapshotOutputDestination::LoopbackTcp { .. } => unreachable!(),
        });
    let scored_video_path = arguments.video.clone();
    let ffprobe_for_scoring = arguments.ffprobe.clone();

    let mut report = run_video_audit(&VideoReplayOptions {
        audit: AuditOptions {
            onnx_path: arguments.onnx,
            source_checkpoint_path: arguments.source_checkpoint,
            input_path: arguments.video,
            provider: arguments.provider.into(),
            image_size: arguments.image_size,
            confidence: arguments.confidence,
            iou: arguments.iou,
            max_detections: arguments.max_detections,
            intra_op_threads: arguments.intra_op_threads,
            warmup_runs: arguments.warmup_runs,
            retained_provider_profile: None,
            dump_preprocessed_directory: None,
            source_image_space: arguments.source_image_space.into(),
            detection_batch_output: batch_output,
            calibration: None,
            scene_snapshot_output: scene_output,
            replay_rate_fps: None,
            replay_repeat_count: 1,
            maximum_replay_frames: None,
        },
        video_stream_ordinal: arguments.video_stream_ordinal,
        ffprobe_executable: arguments.ffprobe,
        ffmpeg_executable: arguments.ffmpeg,
        mode: arguments.mode.into(),
        maximum_video_bytes: arguments.maximum_video_bytes,
        maximum_probe_json_bytes: arguments.maximum_probe_json_bytes,
        maximum_frame_count: arguments.maximum_frame_count,
        maximum_pixel_count: arguments.maximum_pixel_count,
        maximum_decoded_video_bytes: arguments.maximum_decoded_video_bytes,
        subprocess_timeout: Duration::from_secs(arguments.subprocess_timeout_seconds),
        details_jsonl_path: details_jsonl,
        maximum_detail_records,
    })?;
    report.run_id = Some(run_id.clone());
    report.temporal_accuracy = if let Some(annotation_path) = arguments.annotations.as_deref() {
        let annotation_path = resolve_from_current_directory(annotation_path);
        let scene_path = scoring_scene_path
            .as_deref()
            .context("scene snapshot output path disappeared before accuracy scoring")?;
        Some(evaluate_files(
            &scored_video_path,
            &annotation_path,
            scene_path,
            &ffprobe_for_scoring,
            FileEvaluationConfig {
                frame_index_offset: arguments.accuracy_frame_index_offset,
                evaluation: EvaluationConfig {
                    matching_radius_fraction: arguments.accuracy_matching_radius_fraction,
                    freshness_budget_seconds: arguments.accuracy_freshness_budget_seconds,
                },
                ffprobe_timeout: Duration::from_secs(arguments.subprocess_timeout_seconds),
                maximum_video_bytes: arguments.maximum_video_bytes,
                ..FileEvaluationConfig::default()
            },
        )?)
    } else {
        Some(unavailable_temporal_metrics(
            "no reviewed temporal annotation file was supplied",
        ))
    };

    let json =
        serde_json::to_string_pretty(&report).context("failed to serialize video audit report")?;
    if let Some(parent) = output_json.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create report directory {}", parent.display()))?;
    }
    fs::write(&output_json, &json)
        .with_context(|| format!("failed to write report {}", output_json.display()))?;
    let markdown = render_markdown(&report, &output_json, &output_markdown);
    fs::write(&output_markdown, markdown)
        .with_context(|| format!("failed to write report {}", output_markdown.display()))?;
    print_terminal_summary(&report, &output_json, &output_markdown);
    println!("{json}");
    Ok(())
}

fn render_markdown(
    report: &billiards_perception::inference::VideoReplayReport,
    json_path: &Path,
    markdown_path: &Path,
) -> String {
    let measurement = &report.measurement;
    let mut output = format!(
        "# Recorded-video performance audit\n\nRun `{}` measured **{}** using `{}`.\n\n",
        report.run_id.as_deref().unwrap_or("unassigned"),
        match report.mode {
            VideoReplayMode::RealtimeLatestFrame => "PTS-paced latest-frame replay",
            VideoReplayMode::Capacity => "unpaced processing capacity",
        },
        report.source.path,
    );
    output.push_str("## Input and runtime\n\n");
    output.push_str(&format!(
        "- Video SHA-256: `{}`\n- Selected video stream: ordinal {}, global index {} ({}×{}, {} source frames)\n- Source presentation duration: {} seconds; ffprobe: `{}`; ffmpeg: `{}`\n- Model SHA-256: `{}`; checkpoint SHA-256: `{}`; manifest SHA-256: `{}`\n- Provider requested: `{}`; observed providers: `{}`\n- Settings: image size {}, confidence {}, IoU {}, maximum detections {}, inference threads {}\n- Resource caps: video {} bytes, probe JSON {} bytes, {} frames, {} pixels/frame, decoded RGB {} bytes, subprocess {} ms, trace records {}\n- Revision: `{}`; dirty worktree: {}; dirty status SHA-256: `{}`\n- Host: `{} {}`; processor `{}` at {} MHz; logical CPUs: {}; total memory: {} bytes\n\n",
        report.source.sha256,
        report.source.video_stream_ordinal,
        report.source.ffprobe_global_stream_index,
        report.source.width_px,
        report.source.height_px,
        report.source.source_frame_count,
        optional_float(report.source.source_duration_seconds),
        report.source.ffprobe_version,
        report.source.ffmpeg_version,
        report.model.onnx_sha256,
        report.model.source_checkpoint_sha256,
        report.model.manifest_sha256,
        report.runtime.requested_execution_provider,
        if report.runtime.observed_execution_providers.is_empty() {
            "unavailable".to_owned()
        } else {
            report.runtime.observed_execution_providers.join(", ")
        },
        report.settings.image_size,
        report.settings.confidence,
        report.settings.iou,
        report.settings.max_detections,
        report.settings.intra_op_threads.map_or_else(|| "backend default".to_owned(), |threads| threads.to_string()),
        report.settings.maximum_video_bytes,
        report.settings.maximum_probe_json_bytes,
        report.settings.maximum_frame_count,
        report.settings.maximum_pixel_count,
        report.settings.maximum_decoded_video_bytes,
        report.settings.subprocess_timeout_ms,
        report.settings.maximum_detail_records,
        report.environment.repository_revision.as_deref().unwrap_or("unavailable"),
        report.environment.working_tree_dirty.map_or_else(|| "unavailable".to_owned(), |dirty| dirty.to_string()),
        report.environment.dirty_tree_status_sha256.as_deref().unwrap_or("unavailable"),
        report.environment.operating_system,
        report.environment.architecture,
        report.environment.processor_brand.as_deref().unwrap_or("unavailable"),
        report.environment.processor_frequency_mhz.map_or_else(|| "unavailable".to_owned(), |frequency| frequency.to_string()),
        report.environment.logical_cpu_count.map_or_else(|| "unavailable".to_owned(), |count| count.to_string()),
        report.environment.total_memory_bytes.map_or_else(|| "unavailable".to_owned(), |bytes| bytes.to_string()),
    ));
    output.push_str("## Results\n\n");
    output.push_str(&format!(
        "- Processed {}/{} source observations; latest-frame replacements: {}\n- Processed rate: {} observations/s; source rate: {} observations/s\n- Processed-frame source deadlines missed: {}\n- Source-to-inference-start latency ({} samples): p50 {}, p95 {}, p99 {} ms\n- Source-to-inference-complete latency ({} samples): p50 {}, p95 {}, p99 {} ms\n- Mean preprocessing: {:.2} ms; inference: {:.2} ms; decode/NMS: {:.2} ms\n- Measured wall time: {:.2} ms; offline decode {:.2} ms was excluded\n- Detailed trace records: {} written, {} omitted by the configured limit\n\n",
        measurement.processed_source_frames,
        measurement.scheduled_source_frames,
        measurement.latest_frame_replacements,
        measurement.processed_source_deadline_misses.map_or_else(|| "unavailable".to_owned(), |count| count.to_string()),
        optional_float(measurement.processed_observations_per_second),
        optional_float(measurement.source_observations_per_second),
        measurement.source_to_inference_start_ms.sample_count,
        optional_float(measurement.source_to_inference_start_ms.p50_ms),
        optional_float(measurement.source_to_inference_start_ms.p95_ms),
        optional_float(measurement.source_to_inference_start_ms.p99_ms),
        measurement.source_to_inference_complete_ms.sample_count,
        optional_float(measurement.source_to_inference_complete_ms.p50_ms),
        optional_float(measurement.source_to_inference_complete_ms.p95_ms),
        optional_float(measurement.source_to_inference_complete_ms.p99_ms),
        measurement.mean_preprocess_ms,
        measurement.mean_inference_ms,
        measurement.mean_decode_and_nms_ms,
        measurement.measured_wall_ms,
        report.source.offline_decode_ms,
        report.artifacts.detailed_trace_written,
        report.artifacts.detailed_trace_dropped_by_limit,
    ));
    if let Some(accuracy) = &report.temporal_accuracy {
        let status = accuracy
            .pointer("/temporal_localization/status")
            .or_else(|| accuracy.get("status"))
            .and_then(Value::as_str)
            .unwrap_or("unavailable");
        output.push_str(&format!(
            "## Temporal accuracy\n\nStatus: **{status}**.\n\n"
        ));
        if let Some(metrics) = accuracy.get("temporal_localization") {
            for key in [
                "ball_localization",
                "tracking",
                "cue",
                "observation_freshness",
            ] {
                if let Some(value) = metrics.get(key) {
                    output.push_str(&format!(
                        "### {}\n\n```json\n{}\n```\n\n",
                        key.replace('_', " "),
                        serde_json::to_string_pretty(value)
                            .unwrap_or_else(|_| "unavailable".to_owned())
                    ));
                }
            }
        } else if let Some(reason) = accuracy.get("reason").and_then(Value::as_str) {
            output.push_str(&format!("{reason}\n\n"));
        }
    }
    output.push_str("## Limits\n\n");
    output.push_str("Video is decoded into a bounded temporary RGB spool before measurement, so codec startup and offline decoding are excluded. PTS pacing measures detector-side replay freshness; it does not qualify physical camera exposure timing, live decoder contention, viewer/render interference, process CPU/memory trends, or display latency. Use a reviewed annotation set for accuracy claims.\n\n");
    output.push_str(&format!(
        "- JSON report: `{}`\n- Markdown report: `{}`\n",
        json_path.display(),
        markdown_path.display()
    ));
    output
}

fn print_terminal_summary(
    report: &billiards_perception::inference::VideoReplayReport,
    json_path: &Path,
    markdown_path: &Path,
) {
    let measurement = &report.measurement;
    eprintln!(
        "Video audit {}: mode={:?}, processed={}/{}, replaced={}, deadline_misses={}, source_rate={}/s, processed_rate={}/s, completion_p95={} ms, report={}",
        report.run_id.as_deref().unwrap_or("unassigned"),
        report.mode,
        measurement.processed_source_frames,
        measurement.scheduled_source_frames,
        measurement.latest_frame_replacements,
        measurement.processed_source_deadline_misses.map_or_else(|| "unavailable".to_owned(), |count| count.to_string()),
        optional_float(measurement.source_observations_per_second),
        optional_float(measurement.processed_observations_per_second),
        optional_float(measurement.source_to_inference_complete_ms.p95_ms),
        json_path.display(),
    );
    eprintln!("Markdown report: {}", markdown_path.display());
}

fn optional_float(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"))
}

fn run_identity(requested: Option<&str>) -> Result<String> {
    let value = requested
        .map(str::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value != "."
            && value != ".."
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "run ID must be 1-128 ASCII letters, digits, hyphens, or underscores"
    );
    Ok(value)
}

fn resolve_from_current_directory(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn resolve_output_path(path: Option<&Path>, output_dir: &Path, default_name: &str) -> PathBuf {
    match path {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => output_dir.join(path),
        None => output_dir.join(default_name),
    }
}

fn validate_inference_arguments(arguments: &Arguments) -> Result<()> {
    ensure!(arguments.image_size > 0, "image size must be positive");
    ensure!(
        arguments.confidence.is_finite() && (0.0..=1.0).contains(&arguments.confidence),
        "confidence must be finite and within [0, 1]"
    );
    ensure!(
        arguments.iou.is_finite() && (0.0..=1.0).contains(&arguments.iou),
        "IoU must be finite and within [0, 1]"
    );
    ensure!(
        arguments.max_detections > 0,
        "maximum detections must be positive"
    );
    ensure!(
        arguments.intra_op_threads != Some(0),
        "inference thread count must be positive"
    );
    ensure!(
        arguments.maximum_video_bytes > 0,
        "maximum video bytes must be positive"
    );
    ensure!(
        arguments.maximum_probe_json_bytes > 0,
        "maximum probe JSON bytes must be positive"
    );
    ensure!(
        arguments.maximum_frame_count > 0,
        "maximum frame count must be positive"
    );
    ensure!(
        arguments.maximum_pixel_count > 0,
        "maximum pixel count must be positive"
    );
    ensure!(
        arguments.maximum_decoded_video_bytes > 0,
        "maximum decoded video bytes must be positive"
    );
    ensure!(
        arguments.subprocess_timeout_seconds > 0,
        "subprocess timeout must be positive"
    );
    Ok(())
}

fn validate_output_paths(paths: &[Option<&Path>]) -> Result<()> {
    let mut seen = HashSet::new();
    for path in paths.iter().flatten() {
        let normalized = normalize_path(path);
        ensure!(
            seen.insert(normalized.clone()),
            "output paths must be distinct: {}",
            path.display()
        );
        ensure!(
            !path.exists(),
            "refusing to overwrite existing output {}",
            path.display()
        );
    }
    Ok(())
}

fn normalize_path(path: &Path) -> PathBuf {
    let absolute = resolve_from_current_directory(path);
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

fn batch_output_options(
    path: Option<PathBuf>,
    engine_instance_id: Option<String>,
    source_session_id: Option<String>,
) -> Result<Option<BatchOutputOptions>> {
    if path.is_none() {
        ensure!(
            engine_instance_id.is_none() && source_session_id.is_none(),
            "batch identities require --detection-batches-jsonl"
        );
        return Ok(None);
    }

    let engine_instance_id =
        engine_instance_id.context("--engine-instance-id is required with batch output")?;
    let source_session_id =
        source_session_id.context("--source-session-id is required with batch output")?;
    Ok(Some(BatchOutputOptions {
        destination: BatchOutputDestination::JsonlFile {
            path: path.context("batch output path disappeared")?,
        },
        engine_instance_id,
        source_session_id,
    }))
}

fn scene_snapshot_output_options(
    path: Option<PathBuf>,
    has_detection_batch_output: bool,
    arguments: &Arguments,
) -> Result<Option<SceneSnapshotOutputOptions>> {
    let tracking_settings_present = arguments.tracking_ball_class_id.is_some()
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
    let Some(path) = path else {
        ensure!(
            !tracking_settings_present,
            "tracking settings require --scene-snapshots-jsonl"
        );
        return Ok(None);
    };
    ensure!(
        has_detection_batch_output,
        "scene snapshot output requires detection-batch output"
    );

    let tracking = TrackingConfigV1::new(
        arguments
            .tracking_ball_class_id
            .context("--tracking-ball-class-id is required with scene snapshots")?,
        arguments.tracking_cue_class_id,
        arguments
            .tracking_camera_profile_id
            .clone()
            .context("--tracking-camera-profile-id is required with scene snapshots")?,
        arguments
            .tracking_association_gate_px
            .context("--tracking-association-gate-px is required with scene snapshots")?,
        Duration::from_millis(
            arguments
                .tracking_association_prediction_horizon_ms
                .context(
                    "--tracking-association-prediction-horizon-ms is required with scene snapshots",
                )?,
        ),
        Duration::from_millis(
            arguments
                .tracking_lost_after_ms
                .context("--tracking-lost-after-ms is required with scene snapshots")?,
        ),
        Duration::from_millis(
            arguments
                .tracking_maximum_track_age_ms
                .context("--tracking-maximum-track-age-ms is required with scene snapshots")?,
        ),
        arguments
            .tracking_maximum_active_tracks
            .context("--tracking-maximum-active-tracks is required with scene snapshots")?,
        arguments
            .tracking_recent_observation_capacity
            .context("--tracking-recent-observation-capacity is required with scene snapshots")?,
        arguments
            .tracking_configuration_revision
            .context("--tracking-configuration-revision is required with scene snapshots")?,
        arguments
            .tracking_rectification_revision
            .context("--tracking-rectification-revision is required with scene snapshots")?,
    )
    .context("invalid scene tracking configuration")?;

    Ok(Some(SceneSnapshotOutputOptions {
        destination: SceneSnapshotOutputDestination::JsonlFile { path },
        tracking,
        delivery_schema_version: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::{normalize_path, run_identity, Arguments, ModeArgument};
    use clap::Parser;
    use std::path::Path;

    fn required_args() -> Vec<&'static str> {
        vec![
            "billiards-perception-video-replay",
            "--video",
            "input.mp4",
            "--onnx",
            "model.onnx",
            "--source-checkpoint",
            "model.pt",
            "--provider",
            "cpu",
            "--image-size",
            "640",
            "--confidence",
            "0.4",
            "--iou",
            "0.45",
            "--mode",
            "realtime-latest-frame",
        ]
    }

    #[test]
    fn mode_is_required_and_values_are_distinct() {
        let mut args = required_args();
        args.truncate(args.len() - 2);
        assert!(Arguments::try_parse_from(args).is_err());

        let args = Arguments::try_parse_from(required_args()).unwrap();
        assert!(matches!(args.mode, ModeArgument::RealtimeLatestFrame));
        let mut args = required_args();
        let mode_index = args
            .iter()
            .position(|arg| *arg == "realtime-latest-frame")
            .unwrap();
        args[mode_index] = "capacity";
        let args = Arguments::try_parse_from(args).unwrap();
        assert!(matches!(args.mode, ModeArgument::Capacity));
    }

    #[test]
    fn default_limits_are_bounded_and_report_is_saved_by_default() {
        let args = Arguments::try_parse_from(required_args()).unwrap();
        assert!(args.maximum_video_bytes > 0);
        assert!(args.maximum_probe_json_bytes > 0);
        assert!(args.maximum_frame_count > 0);
        assert!(args.maximum_pixel_count > 0);
        assert!(args.maximum_decoded_video_bytes > 0);
        assert!(args.subprocess_timeout_seconds > 0);
        assert!(args.details_jsonl.is_none());
        assert!(args.detection_batches_jsonl.is_none());
        assert!(args.scene_snapshots_jsonl.is_none());
        assert!(args.output_json.is_none());
    }

    #[test]
    fn run_id_rejects_paths_and_unsafe_characters() {
        assert!(run_identity(Some("audit-01")).is_ok());
        assert!(run_identity(Some("../audit")).is_err());
        assert!(run_identity(Some("bad id")).is_err());
    }

    #[test]
    fn normalized_output_paths_collapse_parent_components() {
        let path = normalize_path(Path::new("one/../two/report.json"));
        assert!(path.ends_with("two/report.json"));
    }
}
