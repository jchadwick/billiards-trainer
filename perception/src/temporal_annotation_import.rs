//! Import explicitly reviewed CVAT video keyframes as temporal labels.
//!
//! CVAT interpolation is useful while annotating, but only explicitly listed
//! reviewed frames with corresponding keyframes become ground truth here.
//! This module shares the Rust temporal evaluator's typed schema and semantic
//! validator so imported documents cannot silently diverge from evaluation.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use quick_xml::XmlVersion;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::temporal_evaluation::{
    validate_annotation_document, AnnotationBall, AnnotationBallStatus, AnnotationCue,
    AnnotationCueStatus, AnnotationDocument, AnnotationFrame, AnnotationMetadata,
    AnnotationSequence, Point2d, DEFAULT_MAXIMUM_VIDEO_BYTES,
};

pub const DEFAULT_FFPROBE_TIMEOUT_SECONDS: u64 = 120;
pub const DEFAULT_MAXIMUM_CVAT_XML_BYTES: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_REVIEW_FILE_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_FFPROBE_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_XML_NODES: usize = 2_000_000;
pub const DEFAULT_MAXIMUM_XML_DEPTH: usize = 128;
pub const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAXIMUM_FFPROBE_STDERR_BYTES: usize = 8 * 1024;
const SHA256_LENGTH: usize = 64;

/// Inputs, resource bounds, and domain choices for one import.
#[derive(Debug, Clone)]
pub struct CvatImportConfig {
    pub video_path: PathBuf,
    pub expected_video_sha256: String,
    pub cvat_xml_path: PathBuf,
    pub reviewed_frames_path: PathBuf,
    pub output_path: PathBuf,
    pub epoch_events_path: Option<PathBuf>,
    pub ffprobe_executable: PathBuf,
    pub ffprobe_timeout: Duration,
    pub frame_index_offset: u64,
    pub video_stream_index: u32,
    pub ball_label: String,
    pub cue_label: String,
    pub allow_empty_ball_sequence: bool,
    pub maximum_video_bytes: u64,
    pub maximum_cvat_xml_bytes: u64,
    pub maximum_review_file_bytes: u64,
    pub maximum_ffprobe_output_bytes: usize,
    pub maximum_xml_nodes: usize,
    pub maximum_xml_depth: usize,
}

impl CvatImportConfig {
    pub fn new(
        video_path: PathBuf,
        expected_video_sha256: String,
        cvat_xml_path: PathBuf,
        reviewed_frames_path: PathBuf,
        output_path: PathBuf,
    ) -> Self {
        Self {
            video_path,
            expected_video_sha256,
            cvat_xml_path,
            reviewed_frames_path,
            output_path,
            epoch_events_path: None,
            ffprobe_executable: PathBuf::from("ffprobe"),
            ffprobe_timeout: Duration::from_secs(DEFAULT_FFPROBE_TIMEOUT_SECONDS),
            frame_index_offset: 0,
            video_stream_index: 0,
            ball_label: "ball".to_owned(),
            cue_label: "cue".to_owned(),
            allow_empty_ball_sequence: false,
            maximum_video_bytes: DEFAULT_MAXIMUM_VIDEO_BYTES,
            maximum_cvat_xml_bytes: DEFAULT_MAXIMUM_CVAT_XML_BYTES,
            maximum_review_file_bytes: DEFAULT_MAXIMUM_REVIEW_FILE_BYTES,
            maximum_ffprobe_output_bytes: DEFAULT_MAXIMUM_FFPROBE_OUTPUT_BYTES,
            maximum_xml_nodes: DEFAULT_MAXIMUM_XML_NODES,
            maximum_xml_depth: DEFAULT_MAXIMUM_XML_DEPTH,
        }
    }
}

/// Build one validated annotation document without writing its destination.
pub fn build_temporal_annotations(config: &CvatImportConfig) -> Result<AnnotationDocument> {
    validate_config(config)?;
    validate_input_files(config)?;
    let actual_video_sha256 = hash_file(&config.video_path, config.maximum_video_bytes)?;
    if !actual_video_sha256.eq_ignore_ascii_case(&config.expected_video_sha256) {
        bail!("source video SHA-256 does not match the identity recorded for CVAT");
    }
    ensure_output_does_not_alias_inputs(config)?;

    let probe_output = execute_ffprobe(config)?;
    let timeline = parse_probe_output(&probe_output, config.video_stream_index)?;
    let reviewed_content = read_bounded(
        &config.reviewed_frames_path,
        config.maximum_review_file_bytes,
        "reviewed-frame list",
    )?;
    let reviewed_frames =
        parse_reviewed_frames(&reviewed_content, timeline.presentation_times.len())?;
    let (epoch_events, epoch_content) = match &config.epoch_events_path {
        Some(path) => {
            let content = read_bounded(path, config.maximum_review_file_bytes, "epoch-event file")?;
            (parse_epoch_events(&content)?, Some(content))
        }
        None => (Vec::new(), None),
    };
    let xml_content = read_bounded(
        &config.cvat_xml_path,
        config.maximum_cvat_xml_bytes,
        "CVAT XML",
    )?;
    let root = parse_xml_tree(
        &xml_content,
        config.maximum_xml_nodes,
        config.maximum_xml_depth,
    )?;
    let tracks = parse_cvat_tracks(
        config,
        &root,
        timeline.width,
        timeline.height,
        timeline.presentation_times.len(),
    )?;

    let event_frames: Vec<u64> = epoch_events
        .iter()
        .map(|event| event.source_frame_index)
        .collect();
    let mut frames = Vec::with_capacity(reviewed_frames.len());
    for cvat_frame in reviewed_frames {
        let source_frame = config
            .frame_index_offset
            .checked_add(cvat_frame)
            .filter(|index| *index <= MAX_SAFE_JSON_INTEGER)
            .context("source frame index exceeds the safe JSON integer range")?;
        let epoch_position = upper_bound(&event_frames, &source_frame).checked_sub(1);
        let epoch = epoch_position.map(|position| &epoch_events[position]);
        let mut balls = Vec::new();
        for track in &tracks.balls {
            if let Some(shape) = state_at_reviewed_frame(&track.shapes, cvat_frame, &track.id)? {
                let ball = if shape.outside {
                    AnnotationBall {
                        ball_id: track.id.clone(),
                        status: AnnotationBallStatus::Removed,
                        center: None,
                        radius_px: None,
                        center_table_mm: None,
                    }
                } else if shape.occluded {
                    AnnotationBall {
                        ball_id: track.id.clone(),
                        status: AnnotationBallStatus::Occluded,
                        center: None,
                        radius_px: None,
                        center_table_mm: None,
                    }
                } else {
                    let (x1, y1, x2, y2) = shape
                        .box_coordinates
                        .context("visible CVAT ball keyframe is missing box coordinates")?;
                    AnnotationBall {
                        ball_id: track.id.clone(),
                        status: AnnotationBallStatus::Visible,
                        center: Some(Point2d {
                            x: (x1 + x2) / 2.0,
                            y: (y1 + y2) / 2.0,
                        }),
                        radius_px: Some(((x2 - x1) + (y2 - y1)) / 4.0),
                        center_table_mm: None,
                    }
                };
                balls.push(ball);
            }
        }

        let cue = if let Some(track) = tracks.cues.first() {
            match state_at_reviewed_frame(&track.shapes, cvat_frame, &track.id)? {
                None => AnnotationCue {
                    status: AnnotationCueStatus::Absent,
                    tip: None,
                    butt: None,
                },
                Some(shape) if shape.outside => AnnotationCue {
                    status: AnnotationCueStatus::Absent,
                    tip: None,
                    butt: None,
                },
                Some(shape) if shape.occluded => AnnotationCue {
                    status: AnnotationCueStatus::Occluded,
                    tip: None,
                    butt: None,
                },
                Some(shape) => AnnotationCue {
                    status: AnnotationCueStatus::Visible,
                    tip: shape.cue_tip,
                    butt: shape.cue_butt,
                },
            }
        } else {
            AnnotationCue {
                status: AnnotationCueStatus::Absent,
                tip: None,
                butt: None,
            }
        };

        frames.push(AnnotationFrame {
            source_frame_index: source_frame,
            presentation_time_seconds: timeline.presentation_times[cvat_frame as usize],
            expected_calibration_revision: epoch.and_then(|event| event.calibration_revision),
            expected_source_session_epoch: epoch
                .and_then(|event| event.source_session_epoch.clone()),
            physical_ground_truth_status: None,
            balls,
            cue,
        });
    }

    let document = AnnotationDocument {
        schema_version: 1,
        sequence: AnnotationSequence {
            video_sha256: actual_video_sha256,
            width: timeline.width,
            height: timeline.height,
            coordinate_space: "source_pixels".to_owned(),
            physical_ground_truth: None,
        },
        frames,
        annotation_metadata: Some(AnnotationMetadata {
            source_annotation_format: "cvat_video_1.1".to_owned(),
            source_annotation_sha256: sha256_bytes(&xml_content),
            reviewed_frames_sha256: sha256_bytes(&reviewed_content),
            epoch_events_sha256: epoch_content.as_deref().map(sha256_bytes),
            frame_index_offset: i64::try_from(config.frame_index_offset)
                .context("frame index offset exceeds the annotation schema integer range")?,
            source_video_stream_index: u64::from(config.video_stream_index),
            presentation_time_source: "ffprobe_best_effort_timestamp_time".to_owned(),
            presentation_time_origin_seconds: timeline.origin_seconds,
            ball_radius_method: "mean_box_half_extent".to_owned(),
            cue_polyline_point_order: "first_tip_second_butt".to_owned(),
            tool_version: 1,
        }),
    };
    validate_annotation_document(&document)
        .context("constructed temporal annotation is invalid")?;
    Ok(document)
}

/// Atomically create a new JSON output. Existing paths are never overwritten.
pub fn write_annotations_new(path: &Path, document: &AnnotationDocument) -> Result<()> {
    validate_annotation_document(document)?;
    if path.exists() {
        bail!(
            "output already exists; refusing to clobber {}",
            path.display()
        );
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("could not create output directory {}", parent.display()))?;
    let serialized =
        serde_json::to_vec_pretty(document).context("could not serialize temporal annotations")?;
    let mut temporary = NamedTempFile::new_in(parent)
        .with_context(|| format!("could not create temporary output in {}", parent.display()))?;
    temporary
        .write_all(&serialized)
        .context("could not write temporary annotation output")?;
    temporary
        .write_all(b"\n")
        .context("could not finish temporary annotation output")?;
    temporary
        .as_file()
        .sync_all()
        .context("could not sync temporary annotation output")?;
    temporary
        .persist_noclobber(path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "could not publish annotation output {} without clobbering",
                path.display()
            )
        })?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("could not sync output directory {}", parent.display()))?;
    Ok(())
}

fn validate_config(config: &CvatImportConfig) -> Result<()> {
    if config.expected_video_sha256.len() != SHA256_LENGTH
        || !config
            .expected_video_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("expected_video_sha256 must be a 64-character SHA-256 hex digest recorded for the CVAT source video");
    }
    if config.frame_index_offset > MAX_SAFE_JSON_INTEGER
        || config.frame_index_offset > i64::MAX as u64
    {
        bail!("frame_index_offset exceeds the safe annotation integer range");
    }
    if config.ffprobe_timeout.is_zero() {
        bail!("ffprobe timeout must be positive");
    }
    if config.ball_label.is_empty()
        || config.cue_label.is_empty()
        || config.ball_label == config.cue_label
    {
        bail!("ball and cue labels must be distinct non-empty names");
    }
    for (name, value) in [
        ("maximum_video_bytes", config.maximum_video_bytes),
        ("maximum_cvat_xml_bytes", config.maximum_cvat_xml_bytes),
        (
            "maximum_review_file_bytes",
            config.maximum_review_file_bytes,
        ),
        (
            "maximum_ffprobe_output_bytes",
            config.maximum_ffprobe_output_bytes as u64,
        ),
    ] {
        if value == 0 {
            bail!("{name} must be positive");
        }
    }
    if config.maximum_xml_nodes == 0 || config.maximum_xml_depth == 0 {
        bail!("maximum XML nodes and depth must be positive");
    }
    Ok(())
}

fn validate_input_files(config: &CvatImportConfig) -> Result<()> {
    for (label, path, maximum) in [
        (
            "video",
            config.video_path.as_path(),
            config.maximum_video_bytes,
        ),
        (
            "CVAT XML",
            config.cvat_xml_path.as_path(),
            config.maximum_cvat_xml_bytes,
        ),
        (
            "reviewed-frame list",
            config.reviewed_frames_path.as_path(),
            config.maximum_review_file_bytes,
        ),
    ] {
        validate_file(label, path, maximum)?;
    }
    if let Some(path) = &config.epoch_events_path {
        validate_file("epoch-event file", path, config.maximum_review_file_bytes)?;
    }
    Ok(())
}

fn validate_file(label: &str, path: &Path, maximum: u64) -> Result<()> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("{label} does not exist: {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{label} is not a regular file: {}", path.display());
    }
    if metadata.len() > maximum {
        bail!("{label} exceeds configured size bound ({})", maximum);
    }
    Ok(())
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let file =
        File::open(path).with_context(|| format!("could not read {label}: {}", path.display()))?;
    let mut content =
        Vec::with_capacity(usize::try_from(fs::metadata(path)?.len().min(maximum)).unwrap_or(0));
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut content)
        .with_context(|| format!("could not read {label}: {}", path.display()))?;
    if content.len() as u64 > maximum {
        bail!("{label} exceeds configured size bound ({maximum})");
    }
    Ok(content)
}

fn hash_file(path: &Path, maximum: u64) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("could not hash video {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .context("could not read source video while hashing")?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("video byte count overflow")?;
        if total > maximum {
            bail!("video exceeds configured size bound ({maximum})");
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_digest(digest.finalize().as_slice()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes).as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn execute_ffprobe(config: &CvatImportConfig) -> Result<Vec<u8>> {
    let mut command = Command::new(&config.ffprobe_executable);
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg(format!("v:{}", config.video_stream_index))
        .arg("-show_frames")
        .arg("-show_streams")
        .arg("-show_entries")
        .arg("stream=codec_type,width,height:frame=best_effort_timestamp_time")
        .arg("-of")
        .arg("json")
        .arg(&config.video_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().with_context(|| {
        format!(
            "ffprobe executable was not found or could not start: {}",
            config.ffprobe_executable.display()
        )
    })?;
    let mut stdout = child
        .stdout
        .take()
        .context("ffprobe stdout pipe was unavailable")?;
    let mut stderr = child
        .stderr
        .take()
        .context("ffprobe stderr pipe was unavailable")?;
    let stdout_limit = config.maximum_ffprobe_output_bytes;
    let stdout_overflow = Arc::new(AtomicBool::new(false));
    let overflow_flag = Arc::clone(&stdout_overflow);
    let stdout_thread = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(stdout_limit.min(8 * 1024 * 1024));
        let mut chunk = [0u8; 64 * 1024];
        loop {
            let count = stdout.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let remaining = stdout_limit.saturating_sub(output.len());
            let retained = remaining.min(count);
            output.extend_from_slice(&chunk[..retained]);
            if retained != count {
                overflow_flag.store(true, Ordering::Relaxed);
            }
        }
        Ok(output)
    });
    let stderr_thread = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(MAXIMUM_FFPROBE_STDERR_BYTES);
        let mut chunk = [0u8; 4096];
        loop {
            let count = stderr.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let retained = MAXIMUM_FFPROBE_STDERR_BYTES
                .saturating_sub(output.len())
                .min(count);
            output.extend_from_slice(&chunk[..retained]);
        }
        Ok(output)
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().context("could not check ffprobe status")? {
            break status;
        }
        if start.elapsed() >= config.ffprobe_timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            bail!("ffprobe exceeded the {:?} timeout", config.ffprobe_timeout);
        }
        if stdout_overflow.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_thread.join();
            let _ = stderr_thread.join();
            bail!("ffprobe output exceeds configured size bound ({stdout_limit})");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_thread
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stdout reader panicked"))??;
    let stderr = stderr_thread
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stderr reader panicked"))??;
    if stdout_overflow.load(Ordering::Relaxed) {
        bail!("ffprobe output exceeds configured size bound ({stdout_limit})");
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr);
        let detail = detail.trim();
        if detail.is_empty() {
            bail!("could not inspect video: ffprobe returned a non-zero exit status");
        }
        bail!("could not inspect video: {detail}");
    }
    Ok(stdout)
}

#[derive(Debug)]
struct Timeline {
    width: u32,
    height: u32,
    presentation_times: Vec<f64>,
    origin_seconds: f64,
}

fn parse_probe_output(bytes: &[u8], selected_stream_index: u32) -> Result<Timeline> {
    let root: Value = serde_json::from_slice(bytes).context("ffprobe returned invalid JSON")?;
    let root = root
        .as_object()
        .context("ffprobe output root must be an object")?;
    let streams = root
        .get("streams")
        .and_then(Value::as_array)
        .context("ffprobe streams must be an array")?;
    let frames = root
        .get("frames")
        .and_then(Value::as_array)
        .context("ffprobe frames must be an array")?;
    if streams.len() != 1 {
        bail!("video must contain exactly one selected video stream");
    }
    if frames.is_empty() {
        bail!("video contains no decodable video frames");
    }
    let stream = streams[0]
        .as_object()
        .context("ffprobe selected stream must be an object")?;
    if stream.get("codec_type").and_then(Value::as_str) != Some("video") {
        bail!("ffprobe did not identify a video stream");
    }
    let width = positive_u32(
        stream.get("width"),
        "video stream has invalid width or height",
    )?;
    let height = positive_u32(
        stream.get("height"),
        "video stream has invalid width or height",
    )?;
    let mut presentation_times = Vec::with_capacity(frames.len());
    for (frame_index, frame) in frames.iter().enumerate() {
        let object = frame
            .as_object()
            .with_context(|| format!("ffprobe frame {frame_index} must be an object"))?;
        let raw = object.get("best_effort_timestamp_time").with_context(|| {
            format!("ffprobe frame {frame_index} has no presentation timestamp")
        })?;
        let timestamp = raw
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .or_else(|| raw.as_f64())
            .with_context(|| {
                format!("ffprobe frame {frame_index} has an invalid presentation timestamp")
            })?;
        if !timestamp.is_finite() {
            bail!("ffprobe frame {frame_index} presentation timestamp is not finite");
        }
        if presentation_times
            .last()
            .is_some_and(|previous| timestamp <= *previous)
        {
            bail!("video presentation timestamps must be strictly increasing");
        }
        presentation_times.push(timestamp);
    }
    let origin_seconds = presentation_times[0];
    for timestamp in &mut presentation_times {
        *timestamp -= origin_seconds;
    }
    if presentation_times.iter().any(|value| !value.is_finite()) {
        bail!("normalized video presentation timestamps must be finite");
    }
    let _ = selected_stream_index;
    Ok(Timeline {
        width,
        height,
        presentation_times,
        origin_seconds,
    })
}

fn positive_u32(value: Option<&Value>, message: &str) -> Result<u32> {
    let number = value
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0);
    number.with_context(|| message.to_owned())
}

fn parse_reviewed_frames(bytes: &[u8], frame_count: usize) -> Result<Vec<u64>> {
    let raw: Value =
        serde_json::from_slice(bytes).context("reviewed-frame list must be a JSON array")?;
    let values = raw
        .as_array()
        .context("reviewed-frame list must be a non-empty JSON array")?;
    if values.is_empty() {
        bail!("reviewed-frame list must be a non-empty JSON array");
    }
    let mut reviewed = Vec::with_capacity(values.len());
    let mut previous = None;
    for (position, value) in values.iter().enumerate() {
        let frame = value
            .as_u64()
            .with_context(|| format!("reviewed frame at position {position} must be an integer"))?;
        if previous.is_some_and(|previous| frame <= previous) {
            bail!("reviewed frame indices must be unique and strictly increasing");
        }
        if frame as u128 >= frame_count as u128 {
            bail!("reviewed frame index {frame} is outside the video");
        }
        reviewed.push(frame);
        previous = Some(frame);
    }
    Ok(reviewed)
}

#[derive(Debug, Clone)]
struct EpochEvent {
    source_frame_index: u64,
    calibration_revision: Option<u64>,
    source_session_epoch: Option<String>,
}

fn parse_epoch_events(bytes: &[u8]) -> Result<Vec<EpochEvent>> {
    let raw: Value =
        serde_json::from_slice(bytes).context("epoch-event file must be a JSON array")?;
    let values = raw
        .as_array()
        .context("epoch-event file must be a JSON array")?;
    let expected: HashSet<&str> = [
        "source_frame_index",
        "expected_calibration_revision",
        "expected_source_session_epoch",
    ]
    .into_iter()
    .collect();
    let mut events = Vec::with_capacity(values.len());
    let mut previous = None;
    for (position, value) in values.iter().enumerate() {
        let object = value
            .as_object()
            .with_context(|| format!("epoch event {position} must be an object"))?;
        if object.len() != expected.len()
            || object.keys().any(|key| !expected.contains(key.as_str()))
        {
            bail!("epoch event {position} must contain exactly [expected_calibration_revision, expected_source_session_epoch, source_frame_index]");
        }
        let frame = object
            .get("source_frame_index")
            .and_then(Value::as_u64)
            .with_context(|| {
                format!("epoch event {position} source_frame_index must be an integer")
            })?;
        if frame > MAX_SAFE_JSON_INTEGER {
            bail!("epoch event {position} source_frame_index must be a safe integer");
        }
        if previous.is_some_and(|previous| frame <= previous) {
            bail!("epoch event frame indices must be unique and strictly increasing");
        }
        let calibration_revision = match object.get("expected_calibration_revision").context("missing expected_calibration_revision")? {
            Value::Null => None,
            value => Some(value.as_u64().filter(|revision| *revision <= MAX_SAFE_JSON_INTEGER).with_context(|| format!("epoch event {position} calibration revision must be a safe integer or null"))?),
        };
        let source_session_epoch = match object
            .get("expected_source_session_epoch")
            .context("missing expected_source_session_epoch")?
        {
            Value::Null => None,
            Value::String(value) if !value.is_empty() => Some(value.clone()),
            _ => bail!(
                "epoch event {position} source session epoch must be a non-empty string or null"
            ),
        };
        events.push(EpochEvent {
            source_frame_index: frame,
            calibration_revision,
            source_session_epoch,
        });
        previous = Some(frame);
    }
    Ok(events)
}

#[derive(Debug, Default)]
struct XmlNode {
    name: String,
    attributes: BTreeMap<String, String>,
    children: Vec<XmlNode>,
    text: String,
}

impl XmlNode {
    fn child(&self, name: &str) -> Option<&XmlNode> {
        self.children.iter().find(|node| node.name == name)
    }
    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlNode> + 'a {
        self.children.iter().filter(move |node| node.name == name)
    }
    fn attr(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(String::as_str)
    }
    fn text_trimmed(&self) -> &str {
        self.text.trim()
    }
    fn child_text(&self, name: &str) -> Option<&str> {
        self.child(name).map(XmlNode::text_trimmed)
    }
}

fn parse_xml_tree(bytes: &[u8], maximum_nodes: usize, maximum_depth: usize) -> Result<XmlNode> {
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(false);
    reader.config_mut().check_end_names = true;
    let mut buffer = Vec::new();
    let mut stack = vec![XmlNode::default()];
    let mut node_count = 0usize;
    loop {
        match reader
            .read_event_into(&mut buffer)
            .context("could not parse CVAT XML")?
        {
            Event::Start(event) => {
                let node = xml_node_from_start(&event, &reader)?;
                node_count = node_count
                    .checked_add(1)
                    .context("XML node count overflow")?;
                if node_count > maximum_nodes {
                    bail!("CVAT XML exceeds configured node bound ({maximum_nodes})");
                }
                if stack.len() > maximum_depth {
                    bail!("CVAT XML exceeds configured nesting bound ({maximum_depth})");
                }
                stack.push(node);
            }
            Event::Empty(event) => {
                let node = xml_node_from_start(&event, &reader)?;
                node_count = node_count
                    .checked_add(1)
                    .context("XML node count overflow")?;
                if node_count > maximum_nodes {
                    bail!("CVAT XML exceeds configured node bound ({maximum_nodes})");
                }
                if stack.len() > maximum_depth {
                    bail!("CVAT XML exceeds configured nesting bound ({maximum_depth})");
                }
                stack
                    .last_mut()
                    .context("XML parser lost root node")?
                    .children
                    .push(node);
            }
            Event::End(_) => {
                if stack.len() <= 1 {
                    bail!("CVAT XML has an unexpected closing element");
                }
                let node = stack.pop().context("XML parser stack underflow")?;
                stack
                    .last_mut()
                    .context("XML parser lost parent node")?
                    .children
                    .push(node);
            }
            Event::Text(event) => {
                let decoded = event.xml_content(XmlVersion::Explicit1_1);
                let value = quick_xml::escape::unescape(&decoded)
                    .context("CVAT XML contains invalid escaped text")?;
                stack
                    .last_mut()
                    .context("XML parser lost current node")?
                    .text
                    .push_str(&value);
            }
            Event::GeneralRef(event) => {
                let value = if let Some(character) = event
                    .resolve_char_ref()
                    .context("CVAT XML contains an invalid character reference")?
                {
                    character.to_string()
                } else {
                    quick_xml::escape::resolve_predefined_entity(event.as_ref())
                        .context("CVAT XML contains an unsupported entity reference")?
                        .to_owned()
                };
                stack
                    .last_mut()
                    .context("XML parser lost current node")?
                    .text
                    .push_str(&value);
            }
            Event::CData(event) => {
                stack
                    .last_mut()
                    .context("XML parser lost current node")?
                    .text
                    .push_str(&event.xml11_content());
            }
            Event::DocType(_) => bail!("CVAT XML document type declarations are not supported"),
            Event::Eof => break,
            Event::Decl(_) | Event::Comment(_) | Event::PI(_) => {}
        }
        buffer.clear();
    }
    if stack.len() != 1 {
        bail!("CVAT XML ended before all elements were closed");
    }
    let root = stack.pop().context("XML parser lost document root")?;
    if root.children.len() != 1 {
        bail!("CVAT XML must contain exactly one document element");
    }
    root.children
        .into_iter()
        .next()
        .context("CVAT XML document element is missing")
}

fn xml_node_from_start(
    event: &quick_xml::events::BytesStart<'_>,
    _reader: &Reader<&[u8]>,
) -> Result<XmlNode> {
    let name = event.name().as_ref().to_owned();
    let mut attributes = BTreeMap::new();
    for attribute in event.attributes().with_checks(true) {
        let attribute = attribute.context("CVAT XML contains a malformed attribute")?;
        let key = attribute.key.as_ref().to_owned();
        let value = attribute
            .normalized_value(XmlVersion::Explicit1_1)
            .context("CVAT XML contains an invalid attribute value")?
            .into_owned();
        if attributes.insert(key.clone(), value).is_some() {
            bail!("CVAT XML contains duplicate attribute {key}");
        }
    }
    Ok(XmlNode {
        name,
        attributes,
        children: Vec::new(),
        text: String::new(),
    })
}

#[derive(Debug, Clone)]
struct TrackShape {
    frame: u64,
    outside: bool,
    occluded: bool,
    keyframe: bool,
    box_coordinates: Option<(f64, f64, f64, f64)>,
    cue_tip: Option<Point2d>,
    cue_butt: Option<Point2d>,
}

#[derive(Debug)]
struct Track {
    id: String,
    shapes: Vec<TrackShape>,
}

#[derive(Debug, Default)]
struct Tracks {
    balls: Vec<Track>,
    cues: Vec<Track>,
}

fn parse_cvat_tracks(
    config: &CvatImportConfig,
    root: &XmlNode,
    width: u32,
    height: u32,
    frame_count: usize,
) -> Result<Tracks> {
    if root.name != "annotations" || root.child_text("version") != Some("1.1") {
        bail!("CVAT XML must use the 1.1 annotations format");
    }
    let task = root
        .child("meta")
        .and_then(|meta| meta.child("task"))
        .context("CVAT video task metadata is missing")?;
    if task.child_text("mode") != Some("interpolation") {
        bail!("CVAT XML must be exported from a video interpolation task");
    }
    let original_size = task
        .child("original_size")
        .context("CVAT video task is missing original_size")?;
    let task_width = parse_non_negative(original_size.child_text("width"), "CVAT original width")?;
    let task_height =
        parse_non_negative(original_size.child_text("height"), "CVAT original height")?;
    if task_width == 0 || task_height == 0 {
        bail!("CVAT original_size dimensions must be positive");
    }
    if task_width != u64::from(width) || task_height != u64::from(height) {
        bail!("CVAT task original_size does not match source video");
    }
    let task_size = parse_non_negative(task.child_text("size"), "CVAT task size")?;
    if task_size != frame_count as u64 {
        bail!("CVAT task size does not match the selected source video stream ({task_size} != {frame_count}); create annotations for the full stream");
    }
    let mut declared_labels = HashSet::new();
    if let Some(labels) = task.child("labels") {
        for label in labels.children_named("label") {
            if let Some(name) = label.child_text("name").filter(|name| !name.is_empty()) {
                declared_labels.insert(name.to_owned());
            }
        }
    }
    let missing: Vec<&str> = [&config.ball_label, &config.cue_label]
        .into_iter()
        .filter(|name| !declared_labels.contains(name.as_str()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        bail!(
            "CVAT task does not declare the configured labels: {}",
            missing.join(", ")
        );
    }

    let mut tracks = Tracks::default();
    let mut seen_track_ids = HashSet::new();
    for track_node in root.children_named("track") {
        let label = track_node.attr("label");
        let is_ball = label == Some(config.ball_label.as_str());
        let is_cue = label == Some(config.cue_label.as_str());
        if !is_ball && !is_cue {
            continue;
        }
        let id = track_node
            .attr("id")
            .filter(|value| !value.is_empty())
            .context("CVAT track is missing a stable track id")?
            .to_owned();
        if !seen_track_ids.insert(id.clone()) {
            bail!("duplicate CVAT track id: {id}");
        }
        let expected_shape = if is_ball { "box" } else { "polyline" };
        let mut shapes = Vec::new();
        let mut previous_frame = None;
        for shape in &track_node.children {
            if shape.name != "box" && shape.name != "polyline" {
                bail!(
                    "unsupported CVAT {} track shape: {}",
                    if is_ball { "ball" } else { "cue" },
                    shape.name
                );
            }
            if shape.name != expected_shape {
                bail!(
                    "CVAT {} tracks must use {expected_shape} shapes",
                    if is_ball { "ball" } else { "cue" }
                );
            }
            let frame = parse_non_negative(shape.attr("frame"), "CVAT shape frame")?;
            if frame as u128 >= frame_count as u128 {
                bail!("CVAT shape frame {frame} is outside the source video");
            }
            if previous_frame.is_some_and(|previous| frame <= previous) {
                bail!("CVAT shapes in track {id} must be strictly ordered by frame");
            }
            previous_frame = Some(frame);
            let outside = parse_flag(shape, "outside")?;
            let occluded = parse_flag(shape, "occluded")?;
            let keyframe = parse_flag(shape, "keyframe")?;
            if outside && occluded {
                bail!("CVAT shape in track {id} cannot be both outside and occluded");
            }
            let (box_coordinates, cue_tip, cue_butt) = if is_ball {
                let coordinates = (
                    parse_finite(shape.attr("xtl"), "CVAT box xtl")?,
                    parse_finite(shape.attr("ytl"), "CVAT box ytl")?,
                    parse_finite(shape.attr("xbr"), "CVAT box xbr")?,
                    parse_finite(shape.attr("ybr"), "CVAT box ybr")?,
                );
                if !outside && (coordinates.2 <= coordinates.0 || coordinates.3 <= coordinates.1) {
                    bail!("visible CVAT ball box in track {id} has non-positive dimensions");
                }
                (Some(coordinates), None, None)
            } else if !outside && !occluded {
                let (tip, butt) = parse_point_pair(shape.attr("points"))?;
                (None, Some(tip), Some(butt))
            } else {
                (None, None, None)
            };
            shapes.push(TrackShape {
                frame,
                outside,
                occluded,
                keyframe,
                box_coordinates,
                cue_tip,
                cue_butt,
            });
        }
        if shapes.is_empty() {
            bail!("CVAT track {id} has no shapes");
        }
        let parsed = Track { id, shapes };
        if is_ball {
            tracks.balls.push(parsed);
        } else {
            tracks.cues.push(parsed);
        }
    }
    if tracks.balls.is_empty() && !config.allow_empty_ball_sequence {
        bail!("CVAT contains no ball tracks; pass --allow-empty-ball-sequence only for a fully reviewed empty-table sequence");
    }
    if tracks.cues.len() > 1 {
        bail!("CVAT video must contain at most one cue track");
    }
    Ok(tracks)
}

fn parse_non_negative(value: Option<&str>, label: &str) -> Result<u64> {
    let value = value.with_context(|| format!("{label} must be a non-negative integer"))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("{label} must be a non-negative integer");
    }
    value
        .parse()
        .with_context(|| format!("{label} must be a non-negative integer"))
}

fn parse_finite(value: Option<&str>, label: &str) -> Result<f64> {
    let value = value.with_context(|| format!("{label} is required"))?;
    let number = value
        .parse::<f64>()
        .with_context(|| format!("{label} must be numeric"))?;
    if !number.is_finite() {
        bail!("{label} must be finite");
    }
    Ok(number)
}

fn parse_flag(node: &XmlNode, name: &str) -> Result<bool> {
    match node.attr(name) {
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("CVAT {} {name} must be 0 or 1", node.name),
    }
}

fn parse_point_pair(points: Option<&str>) -> Result<(Point2d, Point2d)> {
    let points = points.context("CVAT cue polyline is missing points")?;
    let mut pairs = points.split(';');
    let first = pairs.next().unwrap_or_default();
    let second = pairs
        .next()
        .context("CVAT cue polyline must have exactly two points ordered tip then butt")?;
    if pairs.next().is_some() {
        bail!("CVAT cue polyline must have exactly two points ordered tip then butt");
    }
    let parse_point = |coordinate: &str, index: usize| -> Result<Point2d> {
        let mut components = coordinate.split(',');
        let x = components.next().unwrap_or_default();
        let y = components
            .next()
            .context(format!("CVAT cue point {index} must contain x,y"))?;
        if components.next().is_some() {
            bail!("CVAT cue point {index} must contain x,y");
        }
        Ok(Point2d {
            x: parse_finite(Some(x), "CVAT cue x")?,
            y: parse_finite(Some(y), "CVAT cue y")?,
        })
    };
    Ok((parse_point(first, 0)?, parse_point(second, 1)?))
}

fn state_at_reviewed_frame<'a>(
    shapes: &'a [TrackShape],
    frame: u64,
    track_id: &str,
) -> Result<Option<&'a TrackShape>> {
    let position = upper_bound_by(shapes, frame).checked_sub(1);
    let Some(position) = position else {
        return Ok(None);
    };
    let shape = &shapes[position];
    if shape.frame == frame {
        if !shape.keyframe {
            bail!("reviewed frame {frame} has a non-keyframe shape for track {track_id}");
        }
        return Ok(Some(shape));
    }
    if shape.outside {
        return Ok(Some(shape));
    }
    bail!("reviewed frame {frame} relies on interpolated state for active track {track_id}; add an explicit CVAT keyframe after reviewing this frame");
}

fn upper_bound(values: &[u64], target: &u64) -> usize {
    values.partition_point(|value| value <= target)
}
fn upper_bound_by(shapes: &[TrackShape], target: u64) -> usize {
    shapes.partition_point(|shape| shape.frame <= target)
}

fn ensure_output_does_not_alias_inputs(config: &CvatImportConfig) -> Result<()> {
    let output = resolve_future_path(&config.output_path)?;
    let inputs = [
        Some(&config.video_path),
        Some(&config.cvat_xml_path),
        Some(&config.reviewed_frames_path),
        config.epoch_events_path.as_ref(),
    ];
    for path in inputs.into_iter().flatten() {
        let input = fs::canonicalize(path)
            .with_context(|| format!("could not resolve input path {}", path.display()))?;
        if output == input {
            bail!("output path must not overwrite an input file");
        }
    }
    Ok(())
}

fn resolve_future_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let normalized = normalize_path(&absolute);
    if normalized.exists() {
        return fs::canonicalize(&normalized)
            .with_context(|| format!("could not resolve output path {}", path.display()));
    }
    let mut ancestor = normalized.as_path();
    while !ancestor.exists() {
        ancestor = ancestor
            .parent()
            .context("output path has no existing parent")?;
    }
    let resolved_ancestor = fs::canonicalize(ancestor)?;
    let suffix = normalized
        .strip_prefix(ancestor)
        .context("could not normalize output path")?;
    Ok(resolved_ancestor.join(suffix))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    const TEST_XML: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><size>3</size><mode>interpolation</mode><original_size>
    <width>100</width><height>80</height>
  </original_size><labels><label><name>ball</name></label>
    <label><name>cue</name></label></labels></task></meta>
  <track id="ball-a" label="ball">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
    <box frame="2" outside="1" occluded="0" keyframe="1" xtl="10" ytl="20" xbr="30" ybr="40"/>
  </track>
  <track id="ball-b" label="ball">
    <box frame="2" outside="0" occluded="0" keyframe="1" xtl="40" ytl="30" xbr="60" ybr="50"/>
  </track>
  <track id="cue-a" label="cue">
    <polyline frame="0" outside="0" occluded="0" keyframe="1" points="4,5;80,50"/>
    <polyline frame="2" outside="0" occluded="1" keyframe="1"/>
  </track>
</annotations>"#;

    fn test_config(directory: &Path) -> CvatImportConfig {
        let video_path = directory.join("capture.mp4");
        fs::write(&video_path, b"video bytes for importer test").unwrap();
        let cvat_xml_path = directory.join("cvat.xml");
        fs::write(&cvat_xml_path, TEST_XML).unwrap();
        let reviewed_frames_path = directory.join("reviewed.json");
        fs::write(&reviewed_frames_path, b"[0,2]").unwrap();
        let output_path = directory.join("labels.json");
        let expected_video_sha256 = hash_file(&video_path, 1024).unwrap();
        let ffprobe_executable = directory.join("ffprobe-test");
        let probe = json!({
            "streams": [{"codec_type": "video", "width": 100, "height": 80}],
            "frames": [
                {"best_effort_timestamp_time": "12.000"},
                {"best_effort_timestamp_time": "12.040"},
                {"best_effort_timestamp_time": "12.110"}
            ]
        });
        let script = format!("#!/bin/sh\nprintf '%s' '{}'\n", probe);
        fs::write(&ffprobe_executable, script).unwrap();
        let mut permissions = fs::metadata(&ffprobe_executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&ffprobe_executable, permissions).unwrap();
        let mut config = CvatImportConfig::new(
            video_path,
            expected_video_sha256,
            cvat_xml_path,
            reviewed_frames_path,
            output_path,
        );
        config.ffprobe_executable = ffprobe_executable;
        config
    }

    #[test]
    fn normalized_timestamps_are_relative_to_the_first_best_effort_pts() {
        let probe = json!({
            "streams": [{"codec_type": "video", "width": 100, "height": 80}],
            "frames": [
                {"best_effort_timestamp_time": "12.000"},
                {"best_effort_timestamp_time": "12.040"},
                {"best_effort_timestamp_time": "12.110"}
            ]
        });
        let timeline = parse_probe_output(probe.to_string().as_bytes(), 2).unwrap();
        assert_eq!(timeline.width, 100);
        assert_eq!(timeline.height, 80);
        assert_eq!(timeline.origin_seconds, 12.0);
        assert!((timeline.presentation_times[0] - 0.0).abs() < f64::EPSILON);
        assert!((timeline.presentation_times[1] - 0.04).abs() < 1e-12);
        assert!((timeline.presentation_times[2] - 0.11).abs() < 1e-12);
    }

    #[test]
    fn reviewed_frame_indices_reject_float_boolean_order_and_range_errors() {
        for value in ["[true]", "[1.0]", "[2,1]", "[3]"] {
            assert!(
                parse_reviewed_frames(value.as_bytes(), 3).is_err(),
                "{value}"
            );
        }
        assert_eq!(parse_reviewed_frames(b"[0,2]", 3).unwrap(), vec![0, 2]);
    }

    #[test]
    fn event_revision_and_session_contract_is_strict() {
        let valid = br#"[{"source_frame_index":10,"expected_calibration_revision":null,"expected_source_session_epoch":"session"}]"#;
        let events = parse_epoch_events(valid).unwrap();
        assert_eq!(events[0].source_frame_index, 10);
        assert_eq!(events[0].source_session_epoch.as_deref(), Some("session"));
        for invalid in [
            br#"[{"source_frame_index":1,"expected_calibration_revision":1,"expected_source_session_epoch":"a","extra":true}]"#.as_slice(),
            br#"[{"source_frame_index":1,"expected_calibration_revision":9007199254740992,"expected_source_session_epoch":"a"}]"#,
            br#"[{"source_frame_index":1,"expected_calibration_revision":1.5,"expected_source_session_epoch":"a"}]"#,
            br#"[{"source_frame_index":1,"expected_calibration_revision":null,"expected_source_session_epoch":""}]"#,
        ] { assert!(parse_epoch_events(invalid).is_err()); }
    }

    #[test]
    fn xml_parser_rejects_doctypes_and_enforces_bounds() {
        assert!(parse_xml_tree(b"<!DOCTYPE x><x/>", 10, 10).is_err());
        assert!(parse_xml_tree(b"<x><y/></x>", 1, 10)
            .unwrap_err()
            .to_string()
            .contains("node bound"));
        assert!(parse_xml_tree(b"<x><y/></x>", 10, 1)
            .unwrap_err()
            .to_string()
            .contains("nesting bound"));
    }

    #[test]
    fn only_explicit_reviewed_keyframes_are_used_and_outside_carries_removed_state() {
        let shapes = vec![
            TrackShape {
                frame: 0,
                outside: true,
                occluded: false,
                keyframe: true,
                box_coordinates: None,
                cue_tip: None,
                cue_butt: None,
            },
            TrackShape {
                frame: 2,
                outside: false,
                occluded: false,
                keyframe: true,
                box_coordinates: Some((0.0, 0.0, 10.0, 10.0)),
                cue_tip: None,
                cue_butt: None,
            },
        ];
        assert!(
            state_at_reviewed_frame(&shapes, 1, "ball")
                .unwrap()
                .unwrap()
                .outside
        );
        assert_eq!(
            state_at_reviewed_frame(&shapes, 2, "ball")
                .unwrap()
                .unwrap()
                .frame,
            2
        );
        assert!(state_at_reviewed_frame(
            &[TrackShape {
                keyframe: false,
                ..shapes[1].clone()
            }],
            2,
            "ball"
        )
        .is_err());
    }

    #[test]
    fn full_import_preserves_reviewed_lifecycle_geometry_pts_provenance_and_epochs() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.frame_index_offset = 100;
        let epoch_path = directory.path().join("epochs.json");
        let epoch_bytes = br#"[
            {"source_frame_index":0,"expected_calibration_revision":3,"expected_source_session_epoch":"before"},
            {"source_frame_index":100,"expected_calibration_revision":4,"expected_source_session_epoch":"session-a"},
            {"source_frame_index":102,"expected_calibration_revision":5,"expected_source_session_epoch":"session-b"}
        ]"#;
        fs::write(&epoch_path, epoch_bytes).unwrap();
        config.epoch_events_path = Some(epoch_path.clone());

        let document = build_temporal_annotations(&config).unwrap();
        assert_eq!(document.sequence.width, 100);
        assert_eq!(document.sequence.height, 80);
        assert_eq!(
            document
                .frames
                .iter()
                .map(|frame| frame.source_frame_index)
                .collect::<Vec<_>>(),
            vec![100, 102]
        );
        assert!((document.frames[0].presentation_time_seconds - 0.0).abs() < f64::EPSILON);
        assert!((document.frames[1].presentation_time_seconds - 0.11).abs() < 1e-12);
        assert_eq!(document.frames[0].expected_calibration_revision, Some(4));
        assert_eq!(
            document.frames[1].expected_source_session_epoch.as_deref(),
            Some("session-b")
        );
        assert_eq!(document.frames[0].balls.len(), 1);
        assert_eq!(document.frames[0].balls[0].ball_id, "ball-a");
        assert_eq!(document.frames[0].balls[0].center.unwrap().x, 20.0);
        assert_eq!(document.frames[0].balls[0].center.unwrap().y, 30.0);
        assert_eq!(document.frames[0].balls[0].radius_px, Some(10.0));
        assert_eq!(document.frames[0].cue.status, AnnotationCueStatus::Visible);
        assert_eq!(document.frames[0].cue.tip.unwrap().x, 4.0);
        assert_eq!(document.frames[0].cue.butt.unwrap().x, 80.0);
        assert_eq!(document.frames[1].balls.len(), 2);
        assert_eq!(
            document.frames[1].balls[0].status,
            AnnotationBallStatus::Removed
        );
        assert_eq!(
            document.frames[1].balls[1].status,
            AnnotationBallStatus::Visible
        );
        assert_eq!(document.frames[1].cue.status, AnnotationCueStatus::Occluded);
        let metadata = document.annotation_metadata.unwrap();
        assert_eq!(metadata.presentation_time_origin_seconds, 12.0);
        assert_eq!(metadata.source_video_stream_index, 0);
        assert_eq!(
            metadata.source_annotation_sha256,
            sha256_bytes(TEST_XML.as_bytes())
        );
        assert_eq!(
            metadata.epoch_events_sha256,
            Some(sha256_bytes(epoch_bytes))
        );
    }

    #[test]
    fn import_rejects_interpolation_non_keyframes_mismatched_geometry_and_hash() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        fs::write(&config.reviewed_frames_path, b"[1]").unwrap();
        assert!(build_temporal_annotations(&config)
            .unwrap_err()
            .to_string()
            .contains("interpolated state"));

        fs::write(&config.reviewed_frames_path, b"[0]").unwrap();
        let xml = fs::read_to_string(&config.cvat_xml_path).unwrap();
        fs::write(&config.cvat_xml_path, xml.replace("width>100", "width>101")).unwrap();
        assert!(build_temporal_annotations(&config)
            .unwrap_err()
            .to_string()
            .contains("does not match source video"));

        config.expected_video_sha256 = "f".repeat(64);
        assert!(build_temporal_annotations(&config)
            .unwrap_err()
            .to_string()
            .contains("SHA-256 does not match"));
    }

    #[test]
    fn empty_ball_sequence_requires_explicit_permission_and_cue_order_is_preserved() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        let xml = fs::read_to_string(&config.cvat_xml_path).unwrap();
        let balls_removed = xml
            .replace(
                "  <track id=\"ball-a\" label=\"ball\">\n    <box frame=\"0\" outside=\"0\" occluded=\"0\" keyframe=\"1\" xtl=\"10\" ytl=\"20\" xbr=\"30\" ybr=\"40\"/>\n    <box frame=\"2\" outside=\"1\" occluded=\"0\" keyframe=\"1\" xtl=\"10\" ytl=\"20\" xbr=\"30\" ybr=\"40\"/>\n  </track>\n  <track id=\"ball-b\" label=\"ball\">\n    <box frame=\"2\" outside=\"0\" occluded=\"0\" keyframe=\"1\" xtl=\"40\" ytl=\"30\" xbr=\"60\" ybr=\"50\"/>\n  </track>\n",
                "",
            );
        fs::write(&config.cvat_xml_path, balls_removed).unwrap();
        assert!(build_temporal_annotations(&config)
            .unwrap_err()
            .to_string()
            .contains("no ball tracks"));
        config.allow_empty_ball_sequence = true;
        let document = build_temporal_annotations(&config).unwrap();
        assert!(document.frames.iter().all(|frame| frame.balls.is_empty()));
        assert_eq!(document.frames[0].cue.tip.unwrap().x, 4.0);
        assert_eq!(document.frames[0].cue.butt.unwrap().x, 80.0);
    }

    #[test]
    fn writer_creates_atomically_and_refuses_to_clobber_existing_output() {
        let directory = tempfile::tempdir().unwrap();
        let config = test_config(directory.path());
        let document = build_temporal_annotations(&config).unwrap();
        write_annotations_new(&config.output_path, &document).unwrap();
        let parsed = crate::temporal_evaluation::parse_annotation_document(
            &fs::read(&config.output_path).unwrap(),
        )
        .unwrap();
        assert_eq!(parsed.frames.len(), 2);
        assert!(write_annotations_new(&config.output_path, &document)
            .unwrap_err()
            .to_string()
            .contains("refusing to clobber"));
        assert_eq!(
            fs::read_dir(directory.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains("tmp"))
                .count(),
            0
        );
    }

    #[test]
    fn ffprobe_timeout_and_output_limits_are_enforced() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = test_config(directory.path());
        config.ffprobe_timeout = Duration::from_millis(25);
        fs::write(&config.ffprobe_executable, "#!/bin/sh\nsleep 1\n").unwrap();
        let mut permissions = fs::metadata(&config.ffprobe_executable)
            .unwrap()
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&config.ffprobe_executable, permissions).unwrap();
        assert!(execute_ffprobe(&config)
            .unwrap_err()
            .to_string()
            .contains("timeout"));

        fs::write(
            &config.ffprobe_executable,
            "#!/bin/sh\nhead -c 4096 /dev/zero\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&config.ffprobe_executable)
            .unwrap()
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&config.ffprobe_executable, permissions).unwrap();
        config.ffprobe_timeout = Duration::from_secs(2);
        config.maximum_ffprobe_output_bytes = 32;
        assert!(execute_ffprobe(&config)
            .unwrap_err()
            .to_string()
            .contains("size bound"));
    }
}
