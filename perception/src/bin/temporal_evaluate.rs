use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use billiards_perception::temporal_evaluation::{
    evaluate_files, EvaluationConfig, FileEvaluationConfig, DEFAULT_FFPROBE_TIMEOUT_SECONDS,
    DEFAULT_FRESHNESS_BUDGET_SECONDS, DEFAULT_MATCHING_RADIUS_FRACTION,
    DEFAULT_MAXIMUM_ANNOTATION_BYTES, DEFAULT_MAXIMUM_CUE_GEOMETRY_JSONL_BYTES,
    DEFAULT_MAXIMUM_CUE_GEOMETRY_LINE_BYTES, DEFAULT_MAXIMUM_SCENE_JSONL_BYTES,
    DEFAULT_MAXIMUM_SCENE_LINE_BYTES, DEFAULT_MAXIMUM_VIDEO_BYTES,
};

#[derive(Debug, Parser)]
#[command(
    name = "billiards-temporal-evaluate",
    about = "Score Rust scene snapshots against exact recorded-video temporal labels"
)]
struct Arguments {
    /// Original video; SHA-256 and the annotation-selected stream geometry are verified.
    #[arg(long)]
    video: PathBuf,
    /// Versioned temporal localization annotation JSON.
    #[arg(long)]
    annotations: PathBuf,
    /// Rust SceneSnapshotV1 JSON Lines output.
    #[arg(long)]
    scene_snapshots: PathBuf,
    /// Optional scheduled cue-geometry JSONL trace to score against cue labels.
    #[arg(long)]
    cue_geometry_jsonl: Option<PathBuf>,
    /// Executable used to inspect video stream geometry.
    #[arg(long, default_value = "ffprobe")]
    ffprobe_executable: PathBuf,
    /// Maximum wall time allowed for the ffprobe geometry query.
    #[arg(long, default_value_t = DEFAULT_FFPROBE_TIMEOUT_SECONDS)]
    ffprobe_timeout_seconds: u64,
    /// Map one-based Rust frame IDs to zero-based annotation frame indices.
    #[arg(long, default_value_t = 0)]
    frame_index_offset: i64,
    /// Match gate as a fraction of each labeled ball's radius.
    #[arg(long, default_value_t = DEFAULT_MATCHING_RADIUS_FRACTION)]
    matching_radius_fraction: f64,
    /// Required with schema-v2 physical labels; maximum table-space match distance in millimeters.
    #[arg(long)]
    physical_matching_radius_mm: Option<f64>,
    /// Maximum acceptable observed-state age in recorded-video seconds.
    #[arg(long, default_value_t = DEFAULT_FRESHNESS_BUDGET_SECONDS)]
    freshness_budget_seconds: f64,
    /// Maximum video bytes hashed for identity binding.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_VIDEO_BYTES)]
    maximum_video_bytes: u64,
    /// Maximum temporal annotation bytes read into memory.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_ANNOTATION_BYTES)]
    maximum_annotation_bytes: u64,
    /// Maximum total Rust scene JSONL bytes read.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SCENE_JSONL_BYTES)]
    maximum_scene_jsonl_bytes: u64,
    /// Maximum bytes in one Rust scene JSONL record.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SCENE_LINE_BYTES)]
    maximum_scene_line_bytes: usize,
    /// Maximum total cue geometry JSONL bytes read.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_CUE_GEOMETRY_JSONL_BYTES)]
    maximum_cue_geometry_jsonl_bytes: u64,
    /// Maximum bytes in one cue geometry JSONL record.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_CUE_GEOMETRY_LINE_BYTES)]
    maximum_cue_geometry_line_bytes: usize,
    /// Optional report path. The report is also printed to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let config = FileEvaluationConfig {
        frame_index_offset: arguments.frame_index_offset,
        evaluation: EvaluationConfig {
            matching_radius_fraction: arguments.matching_radius_fraction,
            freshness_budget_seconds: arguments.freshness_budget_seconds,
            physical_matching_radius_mm: arguments.physical_matching_radius_mm,
        },
        ffprobe_timeout: Duration::from_secs(arguments.ffprobe_timeout_seconds),
        maximum_video_bytes: arguments.maximum_video_bytes,
        maximum_annotation_bytes: arguments.maximum_annotation_bytes,
        maximum_scene_jsonl_bytes: arguments.maximum_scene_jsonl_bytes,
        maximum_scene_line_bytes: arguments.maximum_scene_line_bytes,
        cue_geometry_jsonl_path: arguments.cue_geometry_jsonl,
        maximum_cue_geometry_jsonl_bytes: arguments.maximum_cue_geometry_jsonl_bytes,
        maximum_cue_geometry_line_bytes: arguments.maximum_cue_geometry_line_bytes,
    };
    let report = evaluate_files(
        &arguments.video,
        &arguments.annotations,
        &arguments.scene_snapshots,
        &arguments.ffprobe_executable,
        config,
    )?;
    let serialized = serde_json::to_string_pretty(&report)
        .context("could not serialize temporal evaluation report")?;
    if let Some(path) = arguments.output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("could not create report directory {}", parent.display())
            })?;
        }
        std::fs::write(&path, format!("{serialized}\n"))
            .with_context(|| format!("could not write report {}", path.display()))?;
    }
    println!("{serialized}");
    Ok(())
}
