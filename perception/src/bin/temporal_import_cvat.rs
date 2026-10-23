use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use billiards_perception::temporal_annotation_import::{
    build_temporal_annotations, write_annotations_new, CvatImportConfig,
    DEFAULT_FFPROBE_TIMEOUT_SECONDS, DEFAULT_MAXIMUM_CVAT_XML_BYTES,
    DEFAULT_MAXIMUM_FFPROBE_OUTPUT_BYTES, DEFAULT_MAXIMUM_REVIEW_FILE_BYTES,
    DEFAULT_MAXIMUM_XML_DEPTH, DEFAULT_MAXIMUM_XML_NODES,
};
use billiards_perception::temporal_evaluation::DEFAULT_MAXIMUM_VIDEO_BYTES;
use clap::Parser;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-temporal-import-cvat",
    about = "Convert explicitly reviewed CVAT video keyframes to temporal localization labels"
)]
struct Arguments {
    /// Source video whose identity was recorded when creating the CVAT task.
    #[arg(long)]
    video: PathBuf,
    /// Expected SHA-256 for the exact video uploaded to CVAT.
    #[arg(long)]
    expected_video_sha256: String,
    /// CVAT 1.1 video interpolation-task XML export.
    #[arg(long)]
    cvat_xml: PathBuf,
    /// JSON array of fully reviewed zero-based CVAT frame ordinals.
    #[arg(long)]
    reviewed_frames: PathBuf,
    /// New versioned temporal annotation JSON file. Existing output is never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Optional JSON array of calibration/session revision boundary events.
    #[arg(long)]
    epoch_events: Option<PathBuf>,
    /// ffprobe executable used to read selected-stream PTS and geometry.
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
    /// Wall-clock timeout for ffprobe.
    #[arg(long, default_value_t = DEFAULT_FFPROBE_TIMEOUT_SECONDS)]
    ffprobe_timeout_seconds: u64,
    /// Maximum bytes retained from ffprobe's JSON output.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_FFPROBE_OUTPUT_BYTES)]
    maximum_ffprobe_output_bytes: usize,
    /// Add this offset to CVAT frame ordinals for source-frame identity.
    #[arg(long, default_value_t = 0)]
    frame_index_offset: u64,
    /// Zero-based index among source video streams selected by CVAT.
    #[arg(long, default_value_t = 0)]
    video_stream_index: u32,
    /// CVAT label used for pool balls.
    #[arg(long, default_value = "ball")]
    ball_label: String,
    /// CVAT label used for the cue.
    #[arg(long, default_value = "cue")]
    cue_label: String,
    /// Permit a fully reviewed empty-table sequence without ball tracks.
    #[arg(long)]
    allow_empty_ball_sequence: bool,
    /// Maximum source video size used for hashing.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_VIDEO_BYTES)]
    maximum_video_bytes: u64,
    /// Maximum CVAT XML export size.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_CVAT_XML_BYTES)]
    maximum_cvat_xml_bytes: u64,
    /// Maximum size of each reviewed-frame and epoch-event JSON file.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_REVIEW_FILE_BYTES)]
    maximum_review_file_bytes: u64,
    /// Maximum number of XML elements accepted in the CVAT export.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_XML_NODES)]
    maximum_xml_nodes: usize,
    /// Maximum CVAT XML nesting depth accepted.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_XML_DEPTH)]
    maximum_xml_depth: usize,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let mut config = CvatImportConfig::new(
        arguments.video,
        arguments.expected_video_sha256,
        arguments.cvat_xml,
        arguments.reviewed_frames,
        arguments.output.clone(),
    );
    config.epoch_events_path = arguments.epoch_events;
    config.ffprobe_executable = arguments.ffprobe;
    config.ffprobe_timeout = Duration::from_secs(arguments.ffprobe_timeout_seconds);
    config.maximum_ffprobe_output_bytes = arguments.maximum_ffprobe_output_bytes;
    config.frame_index_offset = arguments.frame_index_offset;
    config.video_stream_index = arguments.video_stream_index;
    config.ball_label = arguments.ball_label;
    config.cue_label = arguments.cue_label;
    config.allow_empty_ball_sequence = arguments.allow_empty_ball_sequence;
    config.maximum_video_bytes = arguments.maximum_video_bytes;
    config.maximum_cvat_xml_bytes = arguments.maximum_cvat_xml_bytes;
    config.maximum_review_file_bytes = arguments.maximum_review_file_bytes;
    config.maximum_xml_nodes = arguments.maximum_xml_nodes;
    config.maximum_xml_depth = arguments.maximum_xml_depth;

    let document = build_temporal_annotations(&config)?;
    write_annotations_new(&arguments.output, &document)?;
    let summary = json!({
        "output": arguments.output,
        "video_sha256": document.sequence.video_sha256,
        "video_geometry": {
            "width": document.sequence.width,
            "height": document.sequence.height
        },
        "source_video_stream_index": document.annotation_metadata.as_ref().map(|metadata| metadata.source_video_stream_index),
        "reviewed_frames": document.frames.len(),
        "labeled_ball_instances": document.frames.iter().map(|frame| frame.balls.len()).sum::<usize>(),
        "timestamp_basis": "ffprobe best_effort_timestamp_time",
        "interpolated_states_used_as_ground_truth": false
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
